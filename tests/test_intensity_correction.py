"""Tests for the global intensity correction solver."""

import numpy as np
import pytest

from napari_maps_stitcher.multiview_stitcher_utils import (
    intensity_correction as ic,
)


def _grid_edges(n_rows, n_cols):
    """Yield the 4-connected edges of an ``n_rows`` x ``n_cols`` tile grid."""
    for row in range(n_rows):
        for col in range(n_cols):
            index = row * n_cols + col
            if col + 1 < n_cols:
                yield index, index + 1
            if row + 1 < n_rows:
                yield index, index + n_cols


def _mosaic(n_rows, n_cols, gains, offsets, seed=0, n_samples=200):
    """Simulate a mosaic whose tiles carry known intensity distortions.

    Each edge gets a batch of shared "true" intensities, observed by the two
    tiles through their own gain and offset. Returns the pairwise statistics,
    the per-view statistics, and the observed sample pairs per edge.
    """
    rng = np.random.default_rng(seed)
    n_views = n_rows * n_cols
    pair_stats = []
    view_stats = [ic.ViewStats() for _ in range(n_views)]
    observed = []
    for index_i, index_j in _grid_edges(n_rows, n_cols):
        truth = rng.uniform(0.1, 0.9, size=n_samples)
        u = gains[index_i] * truth + offsets[index_i]
        v = gains[index_j] * truth + offsets[index_j]
        view_stats[index_i].update(u)
        view_stats[index_j].update(v)
        pair_stats.append(
            ic.pair_stats_from_samples(
                index_i, index_j, u, v, weight=float(n_samples)
            )
        )
        observed.append((index_i, index_j, u, v))
    return pair_stats, view_stats, observed


def _seam_error(observed, params):
    """Root-mean-square intensity mismatch across all seams."""
    residuals = []
    for index_i, index_j, u, v in observed:
        gain_i, offset_i = params[index_i]
        gain_j, offset_j = params[index_j]
        residuals.append(
            (gain_i * u + offset_i) - (gain_j * v + offset_j)
        )
    return float(np.sqrt(np.mean(np.concatenate(residuals) ** 2)))


def test_quantile_samples_recovers_affine_relation():
    """Matched quantiles of affinely related samples lie on a line."""
    rng = np.random.default_rng(1)
    truth = rng.normal(size=5000)
    # Different lengths and orderings: no pixel correspondence at all.
    values_i = truth[:4000]
    values_j = 2.5 * rng.permutation(truth)[:3000] + 7.0
    q_i, q_j = ic.quantile_samples(values_i, values_j, n_quantiles=64)

    slope, intercept = np.polyfit(q_i, q_j, 1)
    assert slope == pytest.approx(2.5, rel=0.05)
    assert intercept == pytest.approx(7.0, abs=0.1)


def test_pixel_samples_requires_equal_shapes():
    """The pixel estimator rejects unpaired inputs instead of guessing."""
    with pytest.raises(ValueError, match="co-registered samples"):
        ic.pixel_samples(np.zeros(10), np.zeros(11))


def test_recovers_known_affine_distortions():
    """A known per-tile gain/offset is undone up to a global affine."""
    rng = np.random.default_rng(2)
    n_views = 16
    gains = rng.uniform(0.7, 1.4, size=n_views)
    offsets = rng.uniform(-0.1, 0.1, size=n_views)
    pair_stats, view_stats, observed = _mosaic(4, 4, gains, offsets)

    correction = ic.solve_intensity_params(
        n_views,
        pair_stats,
        view_stats,
        model="affine",
        drift_length=1e4,
        renormalize=False,
    )

    # The composition of distortion and correction must be the *same* affine
    # for every tile; which affine it is, is the free global gauge.
    composed_gain = correction.gains * gains
    composed_offset = correction.gains * offsets + correction.offsets
    assert np.std(composed_gain) / np.mean(composed_gain) < 1e-3
    assert np.std(composed_offset) < 1e-3 * np.mean(composed_gain)

    assert _seam_error(observed, correction.params) < 1e-4
    assert correction.failed_views == []


def test_offset_model_recovers_offsets():
    """With gains equal, the offset-only model matches the tiles exactly."""
    rng = np.random.default_rng(3)
    n_views = 9
    gains = np.ones(n_views)
    offsets = rng.uniform(-0.2, 0.2, size=n_views)
    pair_stats, view_stats, observed = _mosaic(3, 3, gains, offsets)

    correction = ic.solve_intensity_params(
        n_views,
        pair_stats,
        view_stats,
        model="offset",
        drift_length=1e4,
        renormalize=False,
    )

    assert np.allclose(correction.gains, 1.0)
    composed = offsets + correction.offsets
    assert np.std(composed) < 1e-6
    assert _seam_error(observed, correction.params) < 1e-6


def test_offset_model_cannot_undo_gain_differences():
    """Gain differences need the affine model; offset-only leaves a seam."""
    rng = np.random.default_rng(4)
    n_views = 9
    gains = rng.uniform(0.7, 1.4, size=n_views)
    offsets = np.zeros(n_views)
    pair_stats, view_stats, observed = _mosaic(3, 3, gains, offsets)

    offset_only = ic.solve_intensity_params(
        n_views,
        pair_stats,
        view_stats,
        model="offset",
        drift_length=1e4,
        renormalize=False,
    )
    affine = ic.solve_intensity_params(
        n_views,
        pair_stats,
        view_stats,
        model="affine",
        drift_length=1e4,
        renormalize=False,
    )

    assert _seam_error(observed, affine.params) < 1e-4
    assert _seam_error(observed, offset_only.params) > 1e-2


def test_renormalize_preserves_global_statistics():
    """Renormalisation keeps the mosaic's overall mean and spread."""
    rng = np.random.default_rng(5)
    n_views = 16
    gains = rng.uniform(0.7, 1.4, size=n_views)
    offsets = rng.uniform(-0.1, 0.1, size=n_views)
    pair_stats, view_stats, _ = _mosaic(4, 4, gains, offsets)

    correction = ic.solve_intensity_params(
        n_views, pair_stats, view_stats, drift_length=1e4, renormalize=True
    )

    counts = np.array([v.n_samples for v in view_stats], dtype=float)
    weights = counts / counts.sum()
    m1 = np.array([v.mean for v in view_stats])
    m2 = np.array([v.mean_sq for v in view_stats])
    gain, offset = correction.gains, correction.offsets

    orig_mean = weights @ m1
    orig_var = weights @ m2 - orig_mean**2
    corr_mean = weights @ (gain * m1 + offset)
    corr_var = (
        weights @ (gain**2 * m2 + 2 * gain * offset * m1 + offset**2)
        - corr_mean**2
    )
    assert corr_mean == pytest.approx(orig_mean, rel=1e-6)
    assert corr_var == pytest.approx(orig_var, rel=1e-6)


def test_anchor_bounds_drift_along_a_long_chain():
    """A tight anchor stops noisy pairwise fits random-walking down a chain."""
    rng = np.random.default_rng(6)
    n_views = 60
    n_samples = 200
    pair_stats = []
    view_stats = [ic.ViewStats() for _ in range(n_views)]
    # No real intensity differences at all: every apparent tile-to-tile step
    # is pure estimation noise, which is exactly what accumulates as drift.
    for index in range(n_views - 1):
        truth = rng.uniform(0.1, 0.9, size=n_samples)
        u = truth
        v = truth + rng.normal(scale=0.02)
        view_stats[index].update(u)
        view_stats[index + 1].update(v)
        pair_stats.append(
            ic.pair_stats_from_samples(
                index, index + 1, u, v, weight=float(n_samples)
            )
        )

    loose = ic.solve_intensity_params(
        n_views,
        pair_stats,
        view_stats,
        model="offset",
        drift_length=1e4,
        renormalize=False,
    )
    tight = ic.solve_intensity_params(
        n_views,
        pair_stats,
        view_stats,
        model="offset",
        drift_length=3.0,
        renormalize=False,
    )

    loose_span = np.ptp(loose.offsets)
    tight_span = np.ptp(tight.offsets)
    assert tight_span < 0.5 * loose_span
    # A correlation length of 3 tiles should hold the total excursion to a
    # small multiple of the per-step noise, not sqrt(60) times it.
    assert tight_span < 10 * 0.02


def test_view_without_overlap_falls_back_to_identity():
    """An isolated tile is left alone rather than driven to zero."""
    gains = np.array([1.0, 1.3, 0.8])
    offsets = np.array([0.0, 0.05, -0.05])
    pair_stats, view_stats, _ = _mosaic(1, 3, gains, offsets, seed=8)
    # Append a fourth view that overlaps nothing.
    view_stats.append(ic.ViewStats())

    correction = ic.solve_intensity_params(
        4, pair_stats, view_stats, drift_length=1e4, renormalize=False
    )

    assert correction.gains[3] == pytest.approx(1.0)
    assert correction.offsets[3] == pytest.approx(0.0)
    assert correction.failed_views == []


def test_solve_rejects_bad_arguments():
    """Unknown models and non-positive anchors are refused up front."""
    with pytest.raises(ValueError, match="Unknown model"):
        ic.solve_intensity_params(1, [], model="quadratic")
    with pytest.raises(ValueError, match="drift_length must be positive"):
        ic.solve_intensity_params(1, [], drift_length=0.0)


def test_solve_with_no_pairs_is_identity():
    """With nothing to match, every tile keeps its original intensities."""
    correction = ic.solve_intensity_params(5, [], renormalize=False)
    assert np.allclose(correction.gains, 1.0)
    assert np.allclose(correction.offsets, 0.0)
    assert correction.n_pairs == 0


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_transform_array_clips_and_keeps_dtype(dtype):
    """Unsigned output stays inside the dtype range."""
    info = np.iinfo(dtype)
    array = np.array([0, 10, info.max], dtype=dtype)

    result = ic.transform_array(array, gain=2.0, offset=0.0, dtype=dtype)

    assert result.dtype == np.dtype(dtype)
    assert result[0] == 0
    assert result[1] == 20
    assert result[2] == info.max


def test_transform_array_honours_explicit_min_value():
    """An explicit floor overrides the dtype minimum."""
    array = np.array([0, 10], dtype="uint8")
    result = ic.transform_array(
        array, gain=1.0, offset=0.0, dtype="uint8", min_value=5
    )
    assert result[0] == 5
    assert result[1] == 10


def test_transform_array_broadcasts_per_channel_params():
    """Per-channel gains apply along the channel axis, not across it."""
    array = np.ones((2, 3, 3), dtype="uint16") * 100
    gain = np.array([1.0, 2.0], dtype="float32").reshape(2, 1, 1)
    offset = np.array([0.0, 50.0], dtype="float32").reshape(2, 1, 1)

    result = ic.transform_array(array, gain, offset, dtype="uint16")

    assert np.all(result[0] == 100)
    assert np.all(result[1] == 250)


def test_transform_array_float_is_unclipped():
    """Float output keeps negative values instead of clamping them."""
    array = np.array([0.0, 1.0], dtype="float32")
    result = ic.transform_array(array, gain=1.0, offset=-5.0, dtype="float32")
    assert result.dtype == np.dtype("float32")
    assert result[0] == pytest.approx(-5.0)


def test_apply_intensity_params_checks_view_count():
    """A correction fitted on a different mosaic is rejected."""
    correction = ic.IntensityCorrection(params=np.ones((3, 2)))
    with pytest.raises(ValueError, match="fitted on 3 views"):
        ic.apply_intensity_params([object(), object()], correction)
