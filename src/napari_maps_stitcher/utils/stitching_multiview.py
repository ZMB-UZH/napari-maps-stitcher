"""Multiview-stitcher implementation of stitching algorithm."""

from pathlib import Path

from dask.diagnostics import ProgressBar
from multiview_stitcher import fusion, msi_utils, registration
from multiview_stitcher.fusion import weighted_average_fusion
from ngio import Roi, open_ome_zarr_container
from ngio.tables import RoiTable

from ..multiview_stitcher_utils.fusion import overlay_fusion
from ..multiview_stitcher_utils.intensity_correction import (
    DEFAULT_DRIFT_LENGTH,
    apply_intensity_params,
    estimate_intensity_correction,
)
from .omezarr_utils import export_single_ROI, get_msims, zarrs_codec


def stitch_rois_multiview_stitcher(
    input_zarr_url: str | Path,
    output_zarr_url: str | Path,
    rois: list[Roi],
    resolution_path: str = None,
    blend: bool = False,
    z_project: bool = True,
    intensity_correction: str = "offset",
    intensity_estimator: str = "quantile",
    intensity_drift_length: float = DEFAULT_DRIFT_LENGTH,
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
        intensity_correction: Per-tile intensity correction to apply before
            fusion. ``"offset"`` (the default) fits a brightness offset per
            tile, ``"affine"`` fits brightness and contrast, ``"none"``
            disables it.
        intensity_estimator: How overlapping tiles are compared:
            ``"quantile"`` matches their intensity distributions and does not
            rely on the registration being pixel-accurate, ``"pixel"`` compares
            co-located pixels and is only valid when it is.
        intensity_drift_length: Distance in tiles over which the correction may
            drift before being pulled back toward identity. Lower values guard
            harder against one side of a large mosaic going dark.
    """
    if intensity_correction not in ("none", "offset", "affine"):
        raise ValueError(
            f"Unknown intensity_correction {intensity_correction!r}; "
            "expected 'none', 'offset' or 'affine'"
        )
    if len(rois) == 1:
        export_single_ROI(input_zarr_url, output_zarr_url, rois[0])
        return
    ome_zarr_container = open_ome_zarr_container(input_zarr_url)
    if resolution_path is None:
        resolution_path = ome_zarr_container.level_paths[-1]
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

    # estimate the per-tile intensity correction on the registration-
    # resolution views. The model is affine in intensity, so it commutes with
    # both the downsampling and the z-projection those views may carry: fitting
    # here and applying at full resolution is exact, and costs no extra I/O.
    corrections = None
    if intensity_correction != "none":
        print("    Estimating intensity correction...")
        reg_sim = msi_utils.get_sim_from_msim(msims_reg[0])
        channels = (
            list(reg_sim.coords["c"].values)
            if "c" in reg_sim.dims
            else [None]
        )
        with ProgressBar():
            corrections = [
                estimate_intensity_correction(
                    msims_reg,
                    transform_key="affine_registered",
                    estimator=intensity_estimator,
                    model=intensity_correction,
                    drift_length=intensity_drift_length,
                    channel=channel,
                )
                for channel in channels
            ]
        for channel, correction in zip(channels, corrections, strict=True):
            label = "" if channel is None else f" [{channel}]"
            print(
                f"      {correction.n_pairs} pair(s){label}: "
                f"gain {correction.gains.min():.3f}-"
                f"{correction.gains.max():.3f}, offset "
                f"{correction.offsets.min():.1f}-"
                f"{correction.offsets.max():.1f}"
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

    sims_fusion = [
        msi_utils.get_sim_from_msim(msim) for msim in msims_fusion
    ]
    if corrections is not None:
        # Applied per view rather than to the fused mosaic, so the zero padding
        # that fusion introduces around each tile stays zero.
        sims_fusion = apply_intensity_params(sims_fusion, corrections)

    # get fused image (lazy calculation)
    fused = fusion.fuse(
        sims_fusion,
        fusion_func=weighted_average_fusion if blend else overlay_fusion,
        transform_key="affine_registered",
        output_chunksize=1024,
    )
    # remove axes that are not in the original image
    image_obj = ome_zarr_container.get_image()
    axes_in = image_obj.axes
    fused = fused.squeeze([dim for dim in fused.dims if dim not in axes_in])

    # Align the dask blocks to the zarr chunks. ngio writes with
    # ``da.store(..., lock=False)``; if the dask blocks don't map 1:1 onto the
    # zarr chunks, parallel threads read-modify-write the *same* chunk and clobber
    # each other, dropping data ("missing chunks") on large mosaics. Deriving the
    # output with chunks equal to the (uniform) dask blocks makes each zarr chunk
    # written by exactly one task.
    write_chunks = fused.data.chunksize
    fused_data = fused.data.rechunk(write_chunks)

    # create a new OME-Zarr container with the fused image
    new_ome_zarr_container = ome_zarr_container.derive_image(
        store=output_zarr_url,
        shape=fused.shape,
        chunks=write_chunks,
        overwrite=True,
    )

    # get (empty) image in the new OME-Zarr container
    image = new_ome_zarr_container.get_image(path="0")

    # write the fused data to the ome-zarr image
    print("    Writing fused data...")
    with zarrs_codec(), ProgressBar():
        image.set_array(patch=fused_data, axes_order=fused.dims)
        image.consolidate()
