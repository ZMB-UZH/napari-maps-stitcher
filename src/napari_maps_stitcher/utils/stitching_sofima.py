"""SOFIMA-based stitching implementation."""

from __future__ import annotations

import functools as ft
import multiprocessing
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from ngio import Roi, open_ome_zarr_container
from sofima import flow_utils, mesh, stitch_elastic, stitch_rigid, warp

from .omezarr_utils import export_single_ROI

# The OpenBLAS bundled with numpy/scipy is compiled with MAX_THREADS=24. When
# more threads than that call into it, it falls back to a racy code path that
# segfaults (0xC0000005 on Windows). Rendering calls scipy from every worker
# thread, so the pool has to stay below that limit -- with headroom, since
# napari's own threads register with OpenBLAS too. OPENBLAS_NUM_THREADS does
# not help here: it caps OpenBLAS's internal parallelism, not the number of
# threads calling in.
MAX_RENDER_THREADS = 16


def get_tile_map(ome_zarr_container, rois: list[Roi]):
    tile_map = {}
    x_length = rois[0]["x"].length
    y_length = rois[0]["y"].length
    x_min = min(roi["x"].start for roi in rois)
    y_min = min(roi["y"].start for roi in rois)
    for roi in rois:
        ix = round((roi["x"].start - x_min) / x_length)
        iy = round((roi["y"].start - y_min) / y_length)
        data = ome_zarr_container.get_image().get_roi(roi, mode="numpy")
        data = np.squeeze(data)
        if len(data.shape) != 2:
            raise ValueError("Expected 2D tiles")
        tile_map[(ix, iy)] = data
    return tile_map


def stitch_sofima(
    tile_map: dict, tile_space: tuple, stride: int = 20
) -> np.ndarray:
    """Stitch tiles into a single image using SOFIMA.

    Args:
        tile_map (dict): A dictionary where keys are tile coordinates (x, y)
            and values are the corresponding image tiles (numpy arrays).
        stride (int): The stride (in pixels) for computing the flow fields.
            Default is 60.

    Returns:
        np.ndarray: The stitched image.
    """
    print("Computing coarse offsets...")
    cx, cy = stitch_rigid.compute_coarse_offsets(
        tile_space,
        tile_map,  # , overlaps_xy=((410,410), (410,410)),
    )
    cx_org = cx.copy()
    cy_org = cy.copy()

    # set inf to mean
    cx[0][cx[0] == np.inf] = np.round(np.ma.masked_invalid(cx[0]).mean())
    cx[1][cx[1] == np.inf] = np.round(np.ma.masked_invalid(cx[1]).mean())
    cy[0][cy[0] == np.inf] = np.round(np.ma.masked_invalid(cy[0]).mean())
    cy[1][cy[1] == np.inf] = np.round(np.ma.masked_invalid(cy[1]).mean())

    coarse_mesh = stitch_rigid.optimize_coarse_mesh(cx, cy)
    print("Computing fine flow maps...")
    # The stride (in pixels) specifies the resolution at which to compute the flow
    # fields between tile pairs. This is the same as the resolution at which the
    # mesh is later optimized. The more deformed the tiles initially are, the lower
    # the stride needs to be to get good stitching results.
    cx = np.squeeze(cx)
    cy = np.squeeze(cy)
    fine_x, offsets_x = stitch_elastic.compute_flow_map(
        tile_map,
        cx,
        0,
        stride=(stride, stride),
        batch_size=multiprocessing.cpu_count(),
    )  # (x,y) -> (x+1,y)
    fine_y, offsets_y = stitch_elastic.compute_flow_map(
        tile_map,
        cy,
        1,
        stride=(stride, stride),
        batch_size=multiprocessing.cpu_count(),
    )  # (x,y) -> (x,y+1)

    kwargs = {
        "min_peak_ratio": 1.1,
        "min_peak_sharpness": 1.1,
        "max_deviation": 10,
        "max_magnitude": 0,
    }
    fine_x = {
        k: flow_utils.clean_flow(v[:, np.newaxis, ...], **kwargs)[:, 0, :, :]
        for k, v in fine_x.items()
    }
    fine_y = {
        k: flow_utils.clean_flow(v[:, np.newaxis, ...], **kwargs)[:, 0, :, :]
        for k, v in fine_y.items()
    }

    kwargs = {
        "min_patch_size": 10,
        "max_gradient": -1,
        "max_deviation": -1,
    }
    fine_x = {
        k: flow_utils.reconcile_flows([v[:, np.newaxis, ...]], **kwargs)[
            :, 0, :, :
        ]
        for k, v in fine_x.items()
    }
    fine_y = {
        k: flow_utils.reconcile_flows([v[:, np.newaxis, ...]], **kwargs)[
            :, 0, :, :
        ]
        for k, v in fine_y.items()
    }

    data_x = (cx, fine_x, offsets_x)
    data_y = (cy, fine_y, offsets_y)
    print("Optimizing elastic mesh...")
    fx, fy, x, nbors, key_to_idx = stitch_elastic.aggregate_arrays(
        data_x,
        data_y,
        list(tile_map.keys()),
        coarse_mesh[:, 0, ...],
        stride=(stride, stride),
        tile_shape=next(iter(tile_map.values())).shape,
    )

    @jax.jit
    def prev_fn(x):
        target_fn = ft.partial(
            stitch_elastic.compute_target_mesh,
            x=x,
            fx=fx,
            fy=fy,
            stride=(stride, stride),
        )
        x = jax.vmap(target_fn)(nbors)
        return jnp.transpose(x, [1, 0, 2, 3])

    # These detault settings are expect to work well in most configurations. Perhaps
    # the most salient parameter is the elasticity ratio k0 / k. The larger it gets,
    # the more the tiles will be allowed to deform to match their neighbors (in which
    # case you might want use aggressive flow filtering to ensure that there are no
    # inaccurate flow vectors). Lower ratios will reduce deformation, which, depending
    # on the initial state of the tiles, might result in visible seams.
    config = mesh.IntegrationConfig(
        dt=0.001,
        gamma=0.0,
        k0=0.01,
        k=0.1,
        stride=(stride, stride),
        num_iters=1000,
        max_iters=20000,
        stop_v_max=0.001,
        dt_max=100,
        prefer_orig_order=True,
        start_cap=0.1,
        final_cap=10.0,
        remove_drift=True,
    )

    x, ekin, t = mesh.relax_mesh(x, None, config, prev_fn=prev_fn)
    # Unpack meshes into a dictionary.
    idx_to_key = {v: k for k, v in key_to_idx.items()}
    meshes = {
        idx_to_key[i]: np.array(x[:, i : i + 1 :, :])
        for i in range(x.shape[1])
    }
    print("Rendering tiles...")
    # Warp the tiles into a single image.
    stitched, mask = warp.render_tiles(
        tile_map,
        meshes,
        stride=(stride, stride),
        margin=4,
        parallelism=min(multiprocessing.cpu_count(), MAX_RENDER_THREADS),
    )
    print("Stitching complete.")
    return stitched, fine_x, fine_y, meshes, (cx_org, cy_org), (cx, cy)


def stitch_rois_sofima(
    input_zarr_url: str | Path,
    output_zarr_url: str | Path,
    rois: list[Roi],
    stride: int = 20,
) -> None:
    print("Stitching ROIs with SOFIMA...")
    if len(rois) == 1:
        export_single_ROI(input_zarr_url, output_zarr_url, rois[0])
        return
    ome_zarr_container = open_ome_zarr_container(input_zarr_url)
    print("Creating tile map...")
    tile_map = get_tile_map(ome_zarr_container, rois)
    tile_space = (
        max([k[1] for k in tile_map]) + 1,
        max([k[0] for k in tile_map]) + 1,
    )
    print("Stitching tiles...")
    stitched, *_ = stitch_sofima(tile_map, tile_space, stride=stride)

    # TODO: don't hardcode this
    stitched_reshaped = stitched.reshape((1, 1, *stitched.shape))
    print("Saving stitched image...")
    # create a new OME-Zarr container with the fused image
    new_ome_zarr_container = ome_zarr_container.derive_image(
        store=output_zarr_url,
        shape=stitched_reshaped.shape,
        overwrite=True,
    )

    # get (empty) image in the new OME-Zarr container
    image = new_ome_zarr_container.get_image(path="0")

    # write the fused data to the ome-zarr image
    image.set_array(patch=stitched_reshaped)

    # calculate and write image-pyramid
    image.consolidate()
