"""Utilities for stitching ROIs from OME-Zarr files."""

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .roi_utils import get_overlapping_tile_rois, get_tile_rois
from .stitching_core import stitch_rois

if TYPE_CHECKING:
    from napari.layers import Shapes


def generate_roi_output_paths(
    zarr_path: str | Path, num_rois: int, output_format: str = "zarr"
) -> list[Path]:
    """Generate output paths for stitched ROI files.

    Checks for existing fused files and generates unique names.

    Args:
        zarr_path: Path to the input OME-Zarr file.
        num_rois: Number of ROIs to generate paths for.
        output_format: Output format ('zarr' or 'tiff').

    Returns:
        List of Path objects for the output files.
    """
    zarr_path = Path(zarr_path)
    output_dir = zarr_path.parent
    base_name = zarr_path.stem  # Remove .zarr extension

    # Determine file extension based on format
    extension = ".zarr" if output_format == "zarr" else ".ome.tif"

    output_paths = []
    roi_index = 0

    for i in range(num_rois):
        # Check for existing files and already-assigned paths
        output_name = f"{base_name}_ROI_{roi_index}{extension}"
        output_path = output_dir / output_name

        # Increment index if file already exists or path already assigned
        while output_path.exists() or output_path in output_paths:
            roi_index += 1
            output_name = f"{base_name}_ROI_{roi_index}{extension}"
            output_path = output_dir / output_name

        output_paths.append(output_path)
        roi_index += 1  # Move to next index for next ROI

    return output_paths


def prepare_roi_stitching(
    zarr_path: str | Path,
    shapes_layer: "Shapes",
    get_shapes_func: Callable[["Shapes"], list[np.ndarray]],
    output_format: str = "zarr",
) -> tuple[list[np.ndarray], list[Path]]:
    """Prepare data for ROI stitching.

    Extracts shapes from the layer and generates output paths.

    Args:
        zarr_path: Path to the input OME-Zarr file.
        shapes_layer: Napari Shapes layer containing ROI definitions.
        get_shapes_func: Function to extract shapes from the layer.
        output_format: Output format ('zarr' or 'tiff').

    Returns:
        Tuple of (list of shape coordinates, list of output paths).
    """
    # Extract shapes from the layer
    shapes = get_shapes_func(shapes_layer)

    if not shapes:
        raise ValueError("No shapes found in the shapes layer")

    # Generate output paths
    output_paths = generate_roi_output_paths(
        zarr_path, len(shapes), output_format
    )

    return shapes, output_paths


def stitch_single_roi(
    input_zarr_path: str | Path,
    output_path: str | Path,
    roi_shape: np.ndarray,
    algorithm: str = "multiview-stitcher",
    resolution_path: str | None = None,
    blend: bool = False,
    stride: int | None = None,
    output_format: str = "zarr",
) -> Path:
    """Stitch a single ROI from the input zarr.

    Args:
        input_zarr_path: Path to the input OME-Zarr file.
        output_path: Path where the stitched ROI will be saved.
        roi_shape: Numpy array of shape coordinates defining the ROI.
        algorithm: Name of the stitching algorithm to use.
            Default: "multiview-stitcher".
        resolution_path: Resolution path to use for registration (algorithm-specific).
            If None, uses the lowest resolution.
        blend: If True, use blending for fusion (algorithm-specific).
        stride: Stride parameter for SOFIMA algorithm.
        output_format: Output format ('zarr' or 'tiff').

    Returns:
        Path to the final output file.
    """
    input_zarr_path = Path(input_zarr_path)
    output_path = Path(output_path)

    print(f"Stitching ROI to {output_path.name}...")

    # Get all tile ROIs from the FOV_ROI_table
    rois_all = get_tile_rois(input_zarr_path)

    # Find overlapping tile ROIs with the selected shape
    rois_sel = get_overlapping_tile_rois(roi_shape, rois_all)
    print(f"  Stitching {len(rois_sel)} tiles out of {len(rois_all)} total.")

    # For TIFF output, first stitch to a temporary zarr
    if output_format == "tiff":
        temp_zarr_path = output_path.with_suffix(".zarr")
    else:
        temp_zarr_path = output_path

    # Perform stitching using selected algorithm
    print(f"  Starting stitching with {algorithm}...")
    stitch_rois(
        input_zarr_url=input_zarr_path,
        output_zarr_url=temp_zarr_path,
        rois=rois_sel,
        algorithm=algorithm,
        resolution_path=resolution_path,
        blend=blend,
        stride=stride,
        z_project=True,  # Project z-axis for registration (multiview-stitcher)
    )

    # Convert to TIFF if needed
    if output_format == "tiff":
        print("  Converting to OME-TIFF...")
        _convert_zarr_to_ome_tiff(temp_zarr_path, output_path)
        # Remove temporary zarr
        import shutil

        shutil.rmtree(temp_zarr_path)
        print(f"  Stitching complete: {output_path.name}")
        return output_path
    else:
        print(f"  Stitching complete: {output_path.name}")
        return temp_zarr_path


def _convert_zarr_to_ome_tiff(zarr_path: Path, tiff_path: Path) -> None:
    """Convert a zarr file to OME-TIFF format.

    Reads the highest resolution level from the zarr and saves as OME-TIFF
    with proper metadata including physical pixel sizes.

    Args:
        zarr_path: Path to the input zarr file.
        tiff_path: Path where the OME-TIFF file will be saved.
    """
    from ngio import open_ome_zarr_container
    from tifffile import TiffWriter

    # Open the zarr container and get the highest resolution image
    ome_zarr_container = open_ome_zarr_container(zarr_path)
    image = ome_zarr_container.get_image()

    # Get image data and metadata
    data = image.get_as_dask()
    axes_order = "".join(image.axes).upper()  # e.g., 'CYX' or 'TCZYX'
    pixel_size = image.pixel_size
    dimensions = image.dimensions

    # Get pixel sizes
    pixel_size_x = pixel_size.x if pixel_size.x else 1.0
    pixel_size_y = pixel_size.y if pixel_size.y else 1.0
    pixel_size_z = pixel_size.z if pixel_size.z else 1.0
    space_unit = (
        pixel_size.space_unit if pixel_size.space_unit else "micrometer"
    )

    # Map space unit to abbreviation
    unit_map = {
        "micrometer": "µm",
        "nanometer": "nm",
        "millimeter": "mm",
        "meter": "m",
    }
    unit_abbr = unit_map.get(space_unit, "µm")

    # Build metadata dictionary
    metadata = {
        "axes": axes_order,
        "PhysicalSizeX": pixel_size_x,
        "PhysicalSizeXUnit": unit_abbr,
        "PhysicalSizeY": pixel_size_y,
        "PhysicalSizeYUnit": unit_abbr,
    }

    # Add Z spacing if present
    if "Z" in axes_order:
        metadata["PhysicalSizeZ"] = pixel_size_z
        metadata["PhysicalSizeZUnit"] = unit_abbr

    # Add channel names
    if hasattr(dimensions, "c") and dimensions.c > 0:
        channel_labels = image.channel_labels
        if channel_labels:
            metadata["Channel"] = {"Name": list(channel_labels)}
        else:
            metadata["Channel"] = {
                "Name": [f"Channel {i}" for i in range(dimensions.c)]
            }

    # Write OME-TIFF with metadata
    with TiffWriter(tiff_path, bigtiff=True) as tif:
        tif.write(
            data,
            compression="zlib",
            metadata=metadata,
        )


def stitch_all_rois(
    zarr_path: str | Path,
    shapes_layer: "Shapes",
    get_shapes_func: Callable[["Shapes"], list[np.ndarray]],
    algorithm: str = "multiview-stitcher",
    resolution_path: str | None = None,
    blend: bool = False,
    stride: int | None = None,
) -> list[Path]:
    """Stitch all ROIs from a shapes layer.

    Args:
        zarr_path: Path to the input OME-Zarr file.
        shapes_layer: Napari Shapes layer containing ROI definitions.
        get_shapes_func: Function to extract shapes from the layer.
        algorithm: Name of the stitching algorithm to use.
            Default: "multiview-stitcher".
        resolution_path: Resolution path to use for registration (algorithm-specific).
            If None, uses the lowest resolution.
        blend: If True, use blending for fusion (algorithm-specific).
        stride: Stride parameter for SOFIMA algorithm.

    Returns:
        List of paths to the generated stitched zarr files.
    """
    # Prepare stitching data
    shapes, output_paths = prepare_roi_stitching(
        zarr_path, shapes_layer, get_shapes_func
    )

    print(f"\nStarting stitching of {len(shapes)} ROI(s)...")

    # Stitch each ROI
    for i, (shape, output_path) in enumerate(
        zip(shapes, output_paths, strict=True)
    ):
        print(f"\nProcessing ROI {i + 1}/{len(shapes)}:")
        stitch_single_roi(
            zarr_path,
            output_path,
            shape,
            algorithm,
            resolution_path,
            blend,
            stride,
        )

    print(f"\nCompleted stitching {len(shapes)} ROI(s)")
    return output_paths
