"""Tests for the stitching-algorithm dispatcher."""

import pytest

from napari_maps_stitcher.utils import stitching_core
from napari_maps_stitcher.utils.stitching_base import StitchingAlgorithm


def test_available_algorithms():
    """Both shipped algorithms are advertised."""
    algorithms = stitching_core.get_available_algorithms()
    assert algorithms == ["multiview-stitcher", "sofima"]


@pytest.mark.parametrize("name", ["multiview-stitcher", "sofima"])
def test_get_algorithm_returns_instance(name):
    """Each known name resolves to a StitchingAlgorithm."""
    algo = stitching_core.get_algorithm(name)
    assert isinstance(algo, StitchingAlgorithm)


def test_get_algorithm_unknown_raises():
    """An unknown algorithm name raises a helpful ValueError."""
    with pytest.raises(ValueError, match="Unknown algorithm"):
        stitching_core.get_algorithm("does-not-exist")


def test_stitch_rois_dispatches(monkeypatch):
    """``stitch_rois`` routes to the selected algorithm's ``stitch``."""
    calls = {}

    def fake_stitch(input_zarr_url, output_zarr_url, rois, **kwargs):
        calls["args"] = (input_zarr_url, output_zarr_url, rois)
        calls["kwargs"] = kwargs

    algo = stitching_core.get_algorithm("sofima")
    monkeypatch.setattr(algo, "stitch", fake_stitch)

    stitching_core.stitch_rois(
        "in.zarr", "out.zarr", ["roi"], algorithm="sofima", stride=42
    )

    assert calls["args"] == ("in.zarr", "out.zarr", ["roi"])
    assert calls["kwargs"] == {"stride": 42}


def test_stitch_rois_unknown_algorithm_raises():
    """Dispatching to an unknown algorithm raises before any work."""
    with pytest.raises(ValueError, match="Unknown algorithm"):
        stitching_core.stitch_rois(
            "in.zarr", "out.zarr", [], algorithm="nope"
        )
