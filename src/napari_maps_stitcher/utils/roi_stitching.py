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
    zarr_path: str | Path, num_rois: int
) -> list[Path]:
    """Generate output paths for stitched ROI zarr files.

    Checks for existing fused zarr files and generates unique names.

    Args:
        zarr_path: Path to the input OME-Zarr file.
        num_rois: Number of ROIs to generate paths for.

    Returns:
        List of Path objects for the output zarr files.
    """
    zarr_path = Path(zarr_path)
    output_dir = zarr_path.parent
    base_name = zarr_path.stem  # Remove .zarr extension

    output_paths = []
    roi_index = 0

    for i in range(num_rois):
        # Check for existing fused zarrs and already-assigned paths
        output_name = f"{base_name}_ROI_{roi_index}.zarr"
        output_path = output_dir / output_name

        # Increment index if file already exists or path already assigned
        while output_path.exists() or output_path in output_paths:
            roi_index += 1
            output_name = f"{base_name}_ROI_{roi_index}.zarr"
            output_path = output_dir / output_name

        output_paths.append(output_path)
        roi_index += 1  # Move to next index for next ROI

    return output_paths


def prepare_roi_stitching(
    zarr_path: str | Path,
    shapes_layer: "Shapes",
    get_shapes_func: Callable[["Shapes"], list[np.ndarray]],
) -> tuple[list[np.ndarray], list[Path]]:
    """Prepare data for ROI stitching.

    Extracts shapes from the layer and generates output paths.

    Args:
        zarr_path: Path to the input OME-Zarr file.
        shapes_layer: Napari Shapes layer containing ROI definitions.
        get_shapes_func: Function to extract shapes from the layer.

    Returns:
        Tuple of (list of shape coordinates, list of output paths).
    """
    # Extract shapes from the layer
    shapes = get_shapes_func(shapes_layer)

    if not shapes:
        raise ValueError("No shapes found in the shapes layer")

    # Generate output paths
    output_paths = generate_roi_output_paths(zarr_path, len(shapes))

    return shapes, output_paths


def stitch_single_roi(
    input_zarr_path: str | Path,
    output_zarr_path: str | Path,
    roi_shape: np.ndarray,
    algorithm: str = "multiview-stitcher",
    resolution_path: str | None = None,
    blend: bool = False,
    stride: int | None = None,
) -> None:
    """Stitch a single ROI from the input zarr.

    Args:
        input_zarr_path: Path to the input OME-Zarr file.
        output_zarr_path: Path where the stitched ROI zarr will be saved.
        roi_shape: Numpy array of shape coordinates defining the ROI.
        algorithm: Name of the stitching algorithm to use.
            Default: "multiview-stitcher".
        resolution_path: Resolution path to use for registration (algorithm-specific).
            If None, uses the lowest resolution.
        blend: If True, use blending for fusion (algorithm-specific).
        stride: Stride parameter for SOFIMA algorithm.
    """
    input_zarr_path = Path(input_zarr_path)
    output_zarr_path = Path(output_zarr_path)

    print(f"Stitching ROI to {output_zarr_path.name}...")

    # Get all tile ROIs from the FOV_ROI_table
    rois_all = get_tile_rois(input_zarr_path)

    # Find overlapping tile ROIs with the selected shape
    rois_sel = get_overlapping_tile_rois(roi_shape, rois_all)
    print(f"  Stitching {len(rois_sel)} tiles out of {len(rois_all)} total.")

    # Perform stitching using selected algorithm
    print(f"  Starting stitching with {algorithm}...")
    stitch_rois(
        input_zarr_url=input_zarr_path,
        output_zarr_url=output_zarr_path,
        rois=rois_sel,
        algorithm=algorithm,
        resolution_path=resolution_path,
        blend=blend,
        stride=stride,
        z_project=True,  # Project z-axis for registration (multiview-stitcher)
    )
    print(f"  Stitching complete: {output_zarr_path.name}")


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
