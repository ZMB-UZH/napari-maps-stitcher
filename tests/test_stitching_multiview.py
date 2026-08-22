"""Tests for the multiview-stitcher parameter plumbing."""

import pytest

from napari_maps_stitcher.multiview_stitcher_utils import (
    intensity_correction as ic,
)
from napari_maps_stitcher.utils import (
    stitching_algorithms,
    stitching_multiview,
)


def test_intensity_defaults_to_offset():
    """Brightness matching is on by default; contrast matching is opt-in."""
    algo = stitching_algorithms.MultiviewStitcherAlgorithm()
    params = algo.get_default_params()
    assert params["intensity_correction"] == "offset"
    assert params["intensity_estimator"] == "quantile"
    assert params["intensity_drift_length"] == ic.DEFAULT_DRIFT_LENGTH


def test_every_parameter_is_documented():
    """Each advertised parameter carries a description for the UI tooltip."""
    algo = stitching_algorithms.MultiviewStitcherAlgorithm()
    assert set(algo.get_default_params()) == set(
        algo.get_param_descriptions()
    )


def test_intensity_kwargs_reach_the_stitcher(monkeypatch):
    """The dispatcher forwards the intensity options, not drops them."""
    captured = {}

    def fake_stitch(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        stitching_algorithms,
        "stitch_rois_multiview_stitcher",
        fake_stitch,
    )
    stitching_algorithms.MultiviewStitcherAlgorithm().stitch(
        "in.zarr",
        "out.zarr",
        ["roi_a", "roi_b"],
        intensity_correction="affine",
        intensity_estimator="pixel",
        intensity_drift_length=4.0,
    )

    assert captured["intensity_correction"] == "affine"
    assert captured["intensity_estimator"] == "pixel"
    assert captured["intensity_drift_length"] == 4.0


def test_unknown_intensity_correction_is_rejected_early():
    """A bad mode fails before any data is touched, not halfway through."""
    with pytest.raises(ValueError, match="Unknown intensity_correction"):
        stitching_multiview.stitch_rois_multiview_stitcher(
            "does-not-exist.zarr",
            "out.zarr",
            ["roi_a", "roi_b"],
            intensity_correction="histogram",
        )
