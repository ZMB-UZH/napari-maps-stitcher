"""Tests for the custom fusion functions."""

import numpy as np

from napari_maps_stitcher.multiview_stitcher_utils.fusion import overlay_fusion

NAN = np.nan


def test_uncovered_pixels_stay_zero():
    """Where no view has data, the output is background."""
    views = np.array(
        [
            [[1.0, NAN], [NAN, NAN]],
            [[NAN, 2.0], [NAN, NAN]],
        ]
    )
    result = overlay_fusion(views)
    assert result[0, 0] == 1.0
    assert result[0, 1] == 2.0
    assert result[1, 0] == 0.0
    assert result[1, 1] == 0.0


def test_dark_pixels_are_not_treated_as_missing():
    """A genuinely black pixel is data, not a hole.

    The mask keys on NaN rather than on brightness, so a view that legitimately
    reads 0 overwrites a brighter earlier view instead of being skipped.
    """
    views = np.array(
        [
            [[500.0, 500.0]],
            [[0.0, NAN]],
        ]
    )
    result = overlay_fusion(views)
    assert result[0, 0] == 0.0
    assert result[0, 1] == 500.0


def test_later_views_win_in_the_overlap():
    """Overlaying is last-view-wins where several views cover a pixel."""
    views = np.array([[[10.0, 10.0]], [[20.0, NAN]]])
    result = overlay_fusion(views)
    assert result[0, 0] == 20.0
    assert result[0, 1] == 10.0
