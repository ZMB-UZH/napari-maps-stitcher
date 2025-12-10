"""Stitching functions for image tiles using multiview-stitcher."""

from pathlib import Path

from dask.diagnostics import ProgressBar
from multiview_stitcher import fusion, msi_utils, registration
from multiview_stitcher.fusion import weighted_average_fusion
from ngio import Roi, open_ome_zarr_container
from ngio.tables import RoiTable

from napari_maps_stitcher.multiview_stitcher_utils.fusion import overlay_fusion

from .omezarr_utils import export_single_ROI, get_msims


def stitch_rois(
    input_zarr_url: str | Path,
    output_zarr_url: str | Path,
    rois: list[Roi],
    resolution_path: str = None,
    blend=False,
    z_project: bool = True,
) -> None:
    """Stitch ROIs using multiview-stitcher registration and fusion.

    Args:
        input_zarr_url: Path to the input OME-Zarr container.
        output_zarr_url: Path to the output OME-Zarr container.
        rois: List of ROIs to stitch.
        resolution_path: Resolution path to use for registration.
            If None, uses the lowest resolution.
        blend: If True, use weighted average blending for fusion.
            If False, use overlay fusion.
        z_project: If True, project z-axis for registration.
    """
    if len(rois) == 1:
        export_single_ROI(input_zarr_url, output_zarr_url, rois[0])
        return
    ome_zarr_container = open_ome_zarr_container(input_zarr_url)
    if resolution_path is None:
        resolution_path = ome_zarr_container.levels_paths[-1]
    FOV_ROI_table = RoiTable(rois)

    # load the FOVs as multiscale spatial images (msims)
    msims_reg = get_msims(
        ome_zarr_container.get_image(path=resolution_path),
        FOV_ROI_table,
        z_project=z_project,
    )

    # calculate the stitching transformations
    print("    Calculating registration transformations...")
    with ProgressBar():
        _ = registration.register(
            msims_reg,
            reg_channel="C00",
            transform_key="initial_affine",
            new_transform_key="affine_registered",
            pre_registration_pruning_method="keep_axis_aligned",  # works well for tiles on a grid
            # groupwise_resolution_method="shortest_paths",
        )

    # apply the stitching transformations to the full-resolution FOVs
    if resolution_path == "0" and not z_project:
        msims_fusion = msims_reg
    else:
        msims_fusion = get_msims(
            ome_zarr_container.get_image(path="0"),
            FOV_ROI_table,
            z_project=False,
        )
        for i in range(len(msims_fusion)):
            affine = msi_utils.get_transform_from_msim(
                msims_reg[i], "affine_registered"
            )
            if z_project:
                affine_3d = registration.param_utils.identity_transform(
                    ndim=3,
                    t_coords=affine.coords["t"]
                    if "t" in affine.dims
                    else None,
                )
                affine_3d.loc[
                    {pdim: affine.coords[pdim] for pdim in affine.dims}
                ] = affine
                affine = affine_3d

            msi_utils.set_affine_transform(
                msims_fusion[i], affine, "affine_registered"
            )

    # get fused image (lazy calculation)
    fused = fusion.fuse(
        [msi_utils.get_sim_from_msim(msim) for msim in msims_fusion],
        fusion_func=weighted_average_fusion if blend else overlay_fusion,
        transform_key="affine_registered",
        output_chunksize=1024,
    )
    # remove axes that are not in the original image
    image_obj = ome_zarr_container.get_image()
    axes_in = image_obj.axes
    fused = fused.squeeze([dim for dim in fused.dims if dim not in axes_in])

    # create a new OME-Zarr container with the fused image
    new_ome_zarr_container = ome_zarr_container.derive_image(
        store=output_zarr_url,
        shape=fused.shape,
        overwrite=True,
    )

    # get (empty) image in the new OME-Zarr container
    image = new_ome_zarr_container.get_image(path="0")

    # write the fused data to the ome-zarr image
    print("    Writing fused data...")
    with ProgressBar():
        image.set_array(patch=fused.data, axes_order=fused.dims)
        image.consolidate()
