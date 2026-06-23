"""Shared pytest fixtures."""

import pytest
from ngio import Roi, RoiSlice


def make_roi(
    name: str,
    x: float,
    y: float,
    x_length: float = 10.0,
    y_length: float = 10.0,
    **extra: float,
) -> Roi:
    """Build a 2D (x, y) ngio ``Roi`` for tests.

    Args:
        name: ROI name.
        x: Start coordinate along x.
        y: Start coordinate along y.
        x_length: Length along x.
        y_length: Length along y.
        **extra: Extra columns stored on the ROI (e.g.
            ``x_micrometer_original``), kept in ``roi.model_extra``.
    """
    return Roi(
        name=name,
        slices=[
            RoiSlice(axis_name="x", start=x, length=x_length),
            RoiSlice(axis_name="y", start=y, length=y_length),
        ],
        **extra,
    )


@pytest.fixture
def roi_factory():
    """Return the :func:`make_roi` helper as a fixture."""
    return make_roi
