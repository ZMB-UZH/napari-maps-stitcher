"""Utilities for working with ROIs (Regions of Interest) from OME-Zarr containers."""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ngio import Roi, open_ome_zarr_container
from shapely.geometry import Polygon

if TYPE_CHECKING:
    pass


def get_tile_rois(ome_zarr_path: str | Path) -> list[Roi]:
    """Get all tile ROIs from the FOV_ROI_table in an OME-Zarr container.

    Args:
        ome_zarr_path: Path to the OME-Zarr container.

    Returns:
        List of Roi objects representing individual tiles.
    """
    omezarr = open_ome_zarr_container(ome_zarr_path)
    return omezarr.get_table("FOV_ROI_table").rois()


def _roi_to_polygon(roi: Roi) -> Polygon:
    """Convert an ROI to a Shapely Polygon.

    Args:
        roi: The ROI to convert.

    Returns:
        A Shapely Polygon representing the ROI bounds.
    """
    shape = [
        [roi.y, roi.x],
        [roi.y + roi.y_length, roi.x],
        [roi.y + roi.y_length, roi.x + roi.x_length],
        [roi.y, roi.x + roi.x_length],
    ]
    return Polygon(shape)


def get_overlapping_tile_rois(shape: np.ndarray, rois: list[Roi]) -> list[Roi]:
    """Get all tile ROIs that overlap with a given shape.

    Args:
        shape: A polygon defined as a numpy array of coordinates.
        rois: List of tile ROIs to check for overlap.

    Returns:
        List of tile ROIs that intersect with the given shape.
    """
    overlapping_rois = []
    shape_polygon = Polygon(shape)
    for roi in rois:
        roi_polygon = _roi_to_polygon(roi)
        if shape_polygon.intersects(roi_polygon):
            overlapping_rois.append(roi)
    return overlapping_rois
