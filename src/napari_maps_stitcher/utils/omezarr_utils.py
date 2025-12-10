"""Utilities for OME-Zarr image operations."""

from pathlib import Path

from dask.diagnostics import ProgressBar
from multiview_stitcher import msi_utils
from multiview_stitcher import spatial_image_utils as si_utils
from ngio import Roi, open_ome_zarr_container


def get_levels_paths_dict(zarr_path: Path | str) -> dict[str, str]:
    """Get resolution paths from an OME-Zarr file with human-readable labels.

    Args:
        zarr_path: Path to the OME-Zarr container.

    Returns:
        Dictionary mapping human-readable resolution labels to path strings.
    """
    if not Path(zarr_path).exists():
        return {}
    ome_zarr_container = open_ome_zarr_container(Path(zarr_path))
    output = {}
    for path in ome_zarr_container.levels_paths:
        pixel_size = ome_zarr_container.get_image(path).pixel_size
        unit = ome_zarr_container.get_image(path).space_unit
        # TODO: handle units better
        if (unit == "micrometer") & (1 >= pixel_size.x > 0.0001):
            output[f"{pixel_size.x * 1000:.2f} nm"] = path
        elif (unit == "micrometer") & (1000 >= pixel_size.x > 1):
            output[f"{pixel_size.x:.2f} µm"] = path
        else:
            output[f"{pixel_size.x:.2e} {unit}"] = path
    return output


def _get_original_translation(roi, spatial_dims):
    """Get original stage positions from an ROI.

    Args:
        roi: The ROI object.
        spatial_dims: List of spatial dimension names (e.g., ['y', 'x']).

    Returns:
        Dictionary mapping dimension names to their original positions.
    """
    translation = {}
    for dim in spatial_dims:
        try:
            translation[dim] = getattr(roi, f"{dim}_micrometer_original")
        except AttributeError:
            translation[dim] = getattr(roi, dim)
    return translation


def get_msims(image, FOV_ROI_table, z_project=False):
    """Get multiscale spatial images (msims) from OME-Zarr image.

    Args:
        image: The OME-Zarr image object.
        FOV_ROI_table: RoiTable containing field of view ROIs.
        z_project: If True, project z-axis by taking maximum.

    Returns:
        List of multiscale spatial image objects.
    """
    msims = []
    for roi in FOV_ROI_table.rois():
        # load data of the FOV lazily as a dask-array
        data_da = image.get_roi(roi, mode="dask")
        # get the image axes-order
        axes = list(image.axes)
        # determine the spatial dimensions in the image
        spatial_dims = [dim for dim in axes if dim in ["z", "y", "x"]]
        # calculate z-projection if requested
        if z_project and "z" in axes:
            data_da = data_da.max(axis=axes.index("z"))
            # remove the z-axis from the axes
            axes.remove("z")
            spatial_dims.remove("z")
        # create a spatial image
        sim = si_utils.get_sim_from_array(
            data_da,
            dims=axes,
            scale={dim: getattr(image.pixel_size, dim) for dim in spatial_dims},
            c_coords=image.wavelength_ids,
            translation=_get_original_translation(roi, spatial_dims=spatial_dims),
            transform_key="initial_affine",
        )
        # create a multiscale spatial image (msim) from the spatial image
        msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])
        msims.append(msim)
    return msims


def export_single_ROI(
    input_zarr_url: str | Path,
    output_zarr_url: str | Path,
    roi: Roi,
) -> None:
    """Export a single ROI from an OME-Zarr container to a new container.

    Args:
        input_zarr_url: Path to the input OME-Zarr container.
        output_zarr_url: Path to the output OME-Zarr container.
        roi: The ROI to export.
    """
    ome_zarr_container = open_ome_zarr_container(input_zarr_url)
    image = ome_zarr_container.get_image()
    data_da = image.get_roi(roi, mode="dask")

    # create a new OME-Zarr container with the fused image
    new_ome_zarr_container = ome_zarr_container.derive_image(
        store=output_zarr_url,
        shape=data_da.shape,
        overwrite=True,
    )

    # get (empty) image in the new OME-Zarr container
    image = new_ome_zarr_container.get_image(path="0")

    # write the fused data to the ome-zarr image
    print("  Writing single tile data...")
    with ProgressBar():
        image.set_array(patch=data_da)
        image.consolidate()
