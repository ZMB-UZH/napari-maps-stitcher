"""Tests for ROI geometry helpers (pure, no OME-Zarr/napari needed)."""

import numpy as np
from shapely.geometry import Polygon

from napari_maps_stitcher.utils.roi_utils import (
    _roi_to_polygon,
    get_overlapping_tile_rois,
)


def test_roi_to_polygon_bounds(roi_factory):
    """The polygon spans the ROI's (y, x) start/end corners."""
    roi = roi_factory("a", x=2.0, y=3.0, x_length=10.0, y_length=4.0)
    poly = _roi_to_polygon(roi)

    assert isinstance(poly, Polygon)
    # bounds are (minx, miny, maxx, maxy); polygon is built in (y, x) order
    minx, miny, maxx, maxy = poly.bounds
    assert (miny, maxy) == (2.0, 12.0)  # x: start..start+length
    assert (minx, maxx) == (3.0, 7.0)  # y: start..start+length
    assert poly.area == 10.0 * 4.0


def test_get_overlapping_tile_rois_selects_intersecting(roi_factory):
    """Only ROIs intersecting the query shape are returned."""
    rois = [
        roi_factory("inside", x=0.0, y=0.0),
        roi_factory("touching", x=10.0, y=0.0),
        roi_factory("far", x=100.0, y=100.0),
    ]
    # query rectangle covering (y, x) in [0, 5] x [0, 5]
    shape = np.array([[0.0, 0.0], [0.0, 5.0], [5.0, 5.0], [5.0, 0.0]])

    overlapping = get_overlapping_tile_rois(shape, rois)
    names = {r.get_name() for r in overlapping}

    assert "inside" in names
    assert "far" not in names


def test_get_overlapping_tile_rois_none(roi_factory):
    """A disjoint query shape returns no ROIs."""
    rois = [roi_factory("a", x=0.0, y=0.0)]
    shape = np.array(
        [[50.0, 50.0], [50.0, 60.0], [60.0, 60.0], [60.0, 50.0]]
    )
    assert get_overlapping_tile_rois(shape, rois) == []
