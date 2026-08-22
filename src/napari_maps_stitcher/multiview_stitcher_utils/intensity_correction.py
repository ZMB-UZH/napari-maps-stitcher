"""Global intensity correction for tiled mosaics.

Tiles acquired at different times often differ in brightness and contrast,
which leaves visible seams in the fused mosaic. This module estimates one
intensity transform per tile so that overlapping tiles agree, and applies it
before fusion.

The model per view ``i`` is affine in intensity::

    I_corrected = a_i * I + b_i

with ``a_i == 1`` fixed for the offset-only model. The parameters are found by
minimising, over all overlapping pairs ``(i, j)``::

    E = sum_ij w_ij * mean_k (a_i*u_k + b_i - a_j*v_k - b_j)^2
        + lam * sum_i mean_u ((a_i - 1)*u + b_i)^2

where ``(u_k, v_k)`` are paired intensity samples drawn from the overlap of
views ``i`` and ``j``.

The first term is a graph Laplacian: it pins *differences* between neighbours
but barely constrains the smooth, low-frequency modes, so noise in the pairwise
estimates accumulates as a random walk over graph distance -- the classic
"one side of the mosaic ends up dark" failure. The second term is the anchor:
it is the mean squared intensity change the correction inflicts on view ``i``,
and it turns that random walk into a mean-reverting one with a correlation
length of roughly ``sqrt(w_median / lam)`` tiles. It is exposed as
``drift_length`` (in tiles), with ``lam = 1 / drift_length**2`` after the pair
weights are normalised to a median of 1.

The anchor is not optional for the affine model: without it, ``a = b = 0`` is a
global minimum of the pairwise term.

Everything is parameterised by the *deviation* from identity rather than by the
absolute transform, which keeps the solve well conditioned and makes a view
with no usable overlap fall back to identity instead of to zero.
"""

import warnings
from dataclasses import dataclass, field

import dask
import numpy as np
from multiview_stitcher import msi_utils, mv_graph, registration
from multiview_stitcher import spatial_image_utils as si_utils

#: Number of quantile levels used by the ``"quantile"`` estimator.
DEFAULT_N_QUANTILES = 64

#: Quantile range used by the ``"quantile"`` estimator.
DEFAULT_TRIM = (0.05, 0.95)

#: Default drift correlation length, in tiles.
DEFAULT_DRIFT_LENGTH = 15.0

#: Number of overlapping pairs resampled per ``dask.compute`` call.
DEFAULT_BATCH_SIZE = 32

#: Views solving to a gain at or below this are treated as a failed fit and
#: reset to identity.
MIN_GAIN = 0.05


@dataclass
class PairStats:
    """Sufficient statistics for one overlapping pair of views.

    The pairwise energy of a pair depends on its samples only through these
    six moments, so the samples themselves are never retained. Moments are
    *means*, not sums: the per-pair sample count is divided out so that the
    ``"pixel"`` and ``"quantile"`` estimators produce comparable energies and
    ``drift_length`` means the same thing for both. The sample count enters
    only through ``weight``.

    Attributes:
        index_i: Index of the first view.
        index_j: Index of the second view.
        weight: Relative weight of the pair, typically the overlap area.
        n_samples: Number of samples the moments were computed from.
        mean_i: Mean of the samples from view ``i``.
        mean_j: Mean of the samples from view ``j``.
        mean_ii: Mean of the squared samples from view ``i``.
        mean_ij: Mean of the product of paired samples.
        mean_jj: Mean of the squared samples from view ``j``.
    """

    index_i: int
    index_j: int
    weight: float
    n_samples: int
    mean_i: float
    mean_j: float
    mean_ii: float
    mean_ij: float
    mean_jj: float


@dataclass
class ViewStats:
    """Running intensity moments of a single view.

    Used by the anchor term and by the final renormalisation. Accumulated from
    the overlap crops that were loaded for the pairwise fits, so estimating
    them costs no extra I/O.

    Attributes:
        n_samples: Number of samples accumulated so far.
        mean: Mean intensity.
        mean_sq: Mean squared intensity.
    """

    n_samples: int = 0
    mean: float = 0.0
    mean_sq: float = 0.0

    def update(self, values: np.ndarray) -> None:
        """Fold a new batch of intensity samples into the running moments.

        Args:
            values: 1-D array of intensity samples.
        """
        n_new = int(values.size)
        if n_new == 0:
            return
        total = self.n_samples + n_new
        w_old = self.n_samples / total
        w_new = n_new / total
        self.mean = w_old * self.mean + w_new * float(np.mean(values))
        self.mean_sq = w_old * self.mean_sq + w_new * float(
            np.mean(np.square(np.asarray(values, dtype=np.float64)))
        )
        self.n_samples = total


@dataclass
class IntensityCorrection:
    """Result of a global intensity fit.

    Attributes:
        params: ``(n_views, 2)`` array of ``(gain, offset)`` per view, in the
            intensity units of the input data.
        model: The model that was fitted, ``"affine"`` or ``"offset"``.
        n_pairs: Number of overlapping pairs that contributed.
        failed_views: Indices of views that were reset to identity.
    """

    params: np.ndarray
    model: str = "affine"
    n_pairs: int = 0
    failed_views: list = field(default_factory=list)

    @property
    def gains(self) -> np.ndarray:
        """Per-view gains ``a_i``."""
        return self.params[:, 0]

    @property
    def offsets(self) -> np.ndarray:
        """Per-view offsets ``b_i``."""
        return self.params[:, 1]


def quantile_samples(
    values_i: np.ndarray,
    values_j: np.ndarray,
    n_quantiles: int = DEFAULT_N_QUANTILES,
    trim: tuple[float, float] = DEFAULT_TRIM,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair up two intensity samples by matching their quantiles.

    This is the registration-independent estimator. It never assumes that the
    two arrays are pixel-aligned -- only that they sample approximately the
    same tissue, and therefore approximately the same intensity distribution.
    A residual alignment error of ``d`` over an overlap of width ``W`` biases
    the fit by ``O(d / W)``, which is unavoidable; keeping the overlap as large
    as the geometry allows is what minimises it.

    Args:
        values_i: 1-D intensity samples from view ``i``.
        values_j: 1-D intensity samples from view ``j``. Need not have the same
            length as ``values_i``.
        n_quantiles: Number of quantile levels to match.
        trim: Lower and upper quantile bounds; see :data:`DEFAULT_TRIM`.

    Returns:
        Tuple of two ``(n_quantiles,)`` arrays of matched quantiles.
    """
    levels = np.linspace(trim[0], trim[1], n_quantiles)
    return (
        np.quantile(values_i, levels),
        np.quantile(values_j, levels),
    )


def pixel_samples(
    values_i: np.ndarray,
    values_j: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair up two intensity samples pixel by pixel.

    Only valid when the two arrays are accurately registered and already
    resampled onto a common grid, so that ``values_i[k]`` and ``values_j[k]``
    image the same physical point. If the registration is off, this estimator
    degrades silently -- prefer :func:`quantile_samples` unless the alignment
    is known to be good.

    Args:
        values_i: 1-D intensity samples from view ``i``.
        values_j: 1-D intensity samples from view ``j``, same length and
            ordering as ``values_i``.

    Returns:
        Tuple of the two input arrays, unchanged.

    Raises:
        ValueError: If the two arrays have different lengths.
    """
    if values_i.shape != values_j.shape:
        raise ValueError(
            "pixel_samples needs co-registered samples of equal shape, got "
            f"{values_i.shape} and {values_j.shape}"
        )
    return values_i, values_j


def pair_stats_from_samples(
    index_i: int,
    index_j: int,
    samples_i: np.ndarray,
    samples_j: np.ndarray,
    weight: float = 1.0,
) -> PairStats:
    """Reduce paired intensity samples to their sufficient statistics.

    Args:
        index_i: Index of the first view.
        index_j: Index of the second view.
        samples_i: 1-D samples from view ``i``.
        samples_j: 1-D samples from view ``j``, paired with ``samples_i``.
        weight: Relative weight of this pair, typically the overlap area.

    Returns:
        The corresponding :class:`PairStats`.
    """
    u = np.asarray(samples_i, dtype=np.float64).ravel()
    v = np.asarray(samples_j, dtype=np.float64).ravel()
    return PairStats(
        index_i=int(index_i),
        index_j=int(index_j),
        weight=float(weight),
        n_samples=int(u.size),
        mean_i=float(np.mean(u)),
        mean_j=float(np.mean(v)),
        mean_ii=float(np.mean(u * u)),
        mean_ij=float(np.mean(u * v)),
        mean_jj=float(np.mean(v * v)),
    )


def _pair_blocks(stats: PairStats, affine: bool):
    """Build the local normal-equation blocks for one pair.

    The residual for sample ``k`` is ``r_k = c_k + z_k . y``, where ``y`` holds
    the *deviations from identity* of the two views, ``c_k = u_k - v_k`` is the
    residual of the identity transform, and ``z_k = (u_k, 1, -v_k, -1)`` for
    the affine model (``(1, -1)`` for offset-only). Minimising
    ``mean_k r_k**2`` therefore contributes ``mean(z z^T)`` to the normal
    matrix and ``-mean(c_k z_k)`` to the right-hand side. Both reduce to the
    stored moments.

    Args:
        stats: Sufficient statistics of the pair.
        affine: Whether the gain is a free parameter.

    Returns:
        Tuple of (local matrix, local right-hand side).
    """
    mu, mv = stats.mean_i, stats.mean_j
    muu, muv, mvv = stats.mean_ii, stats.mean_ij, stats.mean_jj
    if affine:
        zzt = np.array(
            [
                [muu, mu, -muv, -mu],
                [mu, 1.0, -mv, -1.0],
                [-muv, -mv, mvv, mv],
                [-mu, -1.0, mv, 1.0],
            ]
        )
        cz = np.array([muu - muv, mu - mv, mvv - muv, mv - mu])
    else:
        zzt = np.array([[1.0, -1.0], [-1.0, 1.0]])
        cz = np.array([mu - mv, mv - mu])
    return zzt, cz


def _anchor_block(view: ViewStats, affine: bool) -> np.ndarray:
    """Build the anchor block for one view.

    The anchor penalises the mean squared intensity change the correction
    inflicts on the view, ``mean_u ((a - 1)*u + b)**2``, which in terms of the
    deviation ``y = (a - 1, b)`` is the moment matrix
    ``[[mean_sq, mean], [mean, 1]]``. It is positive semi-definite, and
    singular exactly when the view has no intensity variance -- in which case
    gain genuinely is unidentifiable and only the combined effect matters.

    Args:
        view: Intensity moments of the view.
        affine: Whether the gain is a free parameter.

    Returns:
        The local anchor matrix.
    """
    if not affine:
        return np.array([[1.0]])
    return np.array(
        [
            [view.mean_sq, view.mean],
            [view.mean, 1.0],
        ]
    )


def _renormalize(
    params: np.ndarray, view_stats: list[ViewStats]
) -> np.ndarray:
    """Rescale all views by one global affine to preserve overall appearance.

    The energy is invariant under a global affine applied to every view, so the
    solve leaves that degree of freedom to the anchor. This restores the
    mosaic's original mean and standard deviation, so the correction only
    removes *relative* differences between tiles and does not shift the overall
    look. It cannot and does not address drift across the mosaic -- that is the
    anchor's job.

    Args:
        params: ``(n_views, 2)`` array of ``(gain, offset)``.
        view_stats: Per-view intensity moments.

    Returns:
        The rescaled ``(n_views, 2)`` parameter array.
    """
    counts = np.array([v.n_samples for v in view_stats], dtype=np.float64)
    if counts.sum() <= 0:
        return params
    weights = counts / counts.sum()
    m1 = np.array([v.mean for v in view_stats])
    m2 = np.array([v.mean_sq for v in view_stats])
    gain, offset = params[:, 0], params[:, 1]

    orig_mean = float(weights @ m1)
    orig_var = float(weights @ m2) - orig_mean**2
    corr_mean = float(weights @ (gain * m1 + offset))
    corr_var = (
        float(weights @ (gain**2 * m2 + 2 * gain * offset * m1 + offset**2))
        - corr_mean**2
    )
    if corr_var <= 0 or orig_var <= 0:
        return params

    alpha = np.sqrt(orig_var / corr_var)
    beta = orig_mean - alpha * corr_mean
    return np.column_stack([alpha * gain, alpha * offset + beta])


def solve_intensity_params(
    n_views: int,
    pair_stats: list[PairStats],
    view_stats: list[ViewStats] | None = None,
    model: str = "affine",
    drift_length: float = DEFAULT_DRIFT_LENGTH,
    renormalize: bool = True,
    eps: float = 1e-9,
) -> IntensityCorrection:
    """Solve the global least-squares problem for per-view intensity params.

    Args:
        n_views: Total number of views.
        pair_stats: Sufficient statistics for every overlapping pair.
        view_stats: Per-view intensity moments, used by the anchor term and the
            final renormalisation. Defaults to unit moments, which weights the
            anchor equally for every view.
        model: ``"affine"`` to fit gain and offset, ``"offset"`` to fit offset
            only. Offset-only halves the number of unknowns and is markedly
            more robust when the overlaps are low-contrast or the pairwise
            estimates are distribution-mismatched, since slope estimation is
            the part that suffers.
        drift_length: Anchor strength, expressed as the distance in tiles over
            which corrections may drift before being pulled back toward
            identity. Larger means a looser anchor and better local seam
            matching; smaller means stronger protection against one side of the
            mosaic going dark. Must be positive.
        renormalize: Whether to rescale all views by one global affine so the
            mosaic keeps its original mean and standard deviation.
        eps: Tiny ridge added to the diagonal, so that views with neither
            overlaps nor intensity variance still solve (to identity).

    Returns:
        The fitted :class:`IntensityCorrection`.

    Raises:
        ValueError: If ``model`` is unknown or ``drift_length`` is not
            positive.
    """
    if model not in ("affine", "offset"):
        raise ValueError(
            f"Unknown model {model!r}; expected 'affine' or 'offset'"
        )
    if not drift_length > 0:
        raise ValueError(
            f"drift_length must be positive, got {drift_length!r}"
        )
    affine = model == "affine"
    if view_stats is None:
        view_stats = [
            ViewStats(n_samples=1, mean=0.0, mean_sq=1.0)
            for _ in range(n_views)
        ]

    n_local = 2 if affine else 1
    size = n_views * n_local
    matrix = np.zeros((size, size))
    rhs = np.zeros(size)

    # Normalise the pair weights to a median of 1 so that ``drift_length`` has
    # the same meaning regardless of tile size, estimator, or the units the
    # overlap area happens to be measured in.
    weights = np.array([p.weight for p in pair_stats], dtype=np.float64)
    scale = float(np.median(weights)) if weights.size else 1.0
    if not scale > 0:
        scale = 1.0

    for stats, weight in zip(pair_stats, weights / scale, strict=True):
        zzt, cz = _pair_blocks(stats, affine)
        idx = [stats.index_i * n_local, stats.index_j * n_local]
        if affine:
            idx = [idx[0], idx[0] + 1, idx[1], idx[1] + 1]
        matrix[np.ix_(idx, idx)] += weight * zzt
        rhs[idx] -= weight * cz

    lam = 1.0 / drift_length**2
    for view_index, view in enumerate(view_stats):
        idx = [view_index * n_local]
        if affine:
            idx = [idx[0], idx[0] + 1]
        matrix[np.ix_(idx, idx)] += lam * _anchor_block(view, affine)
    matrix[np.diag_indices(size)] += eps

    deviations = np.linalg.solve(matrix, rhs).reshape(n_views, n_local)

    params = np.ones((n_views, 2))
    params[:, 1] = 0.0
    if affine:
        params[:, 0] += deviations[:, 0]
        params[:, 1] = deviations[:, 1]
    else:
        params[:, 1] = deviations[:, 0]

    failed = np.flatnonzero(
        ~np.isfinite(params).all(axis=1) | (params[:, 0] <= MIN_GAIN)
    )
    if failed.size:
        warnings.warn(
            f"Intensity correction failed for {failed.size} view(s) "
            f"{failed.tolist()}; falling back to identity for those views.",
            RuntimeWarning,
            stacklevel=2,
        )
        params[failed] = (1.0, 0.0)

    if renormalize:
        params = _renormalize(params, view_stats)

    return IntensityCorrection(
        params=params,
        model=model,
        n_pairs=len(pair_stats),
        failed_views=failed.tolist(),
    )


def _select_scalar_view(sim, channel=None):
    """Reduce a view to a single channel and timepoint.

    Args:
        sim: The spatial image.
        channel: Channel coordinate to use. If None, the first channel is used.

    Returns:
        The spatial image with all non-spatial dimensions selected away.
    """
    nonspatial_dims = si_utils.get_nonspatial_dims_from_sim(sim)
    if not nonspatial_dims:
        return sim
    selection = {}
    for dim in nonspatial_dims:
        if dim == "c" and channel is not None:
            selection[dim] = channel
        else:
            selection[dim] = sim.coords[dim][0]
    return si_utils.sim_sel_coords(sim, selection)


def _rescale_stats(pair_stats, view_stats, scale):
    """Divide all collected moments by an intensity scale.

    The solve is better conditioned when intensities are O(1), and the moments
    are homogeneous in intensity, so this is exact: first moments scale with
    ``scale`` and second moments with ``scale**2``. Gains come back out
    unchanged and offsets are multiplied by ``scale`` again.

    Args:
        pair_stats: Pairwise statistics, in raw intensity units.
        view_stats: Per-view moments, in raw intensity units.
        scale: The intensity scale to divide by.
    """
    for stats in pair_stats:
        stats.mean_i /= scale
        stats.mean_j /= scale
        stats.mean_ii /= scale**2
        stats.mean_ij /= scale**2
        stats.mean_jj /= scale**2
    for view in view_stats:
        view.mean /= scale
        view.mean_sq /= scale**2


def _iter_overlap_crops(sims, edges, transform_key, batch_size):
    """Yield the overlap crops of each edge, computing them in batches.

    ``registration.sims_to_intrinsic_coord_system`` returns *lazy* views: with
    dask-backed input it resamples through ``dask_image``, so nothing is read
    until the arrays are realised. Realising one edge at a time would let dask
    parallelise within an edge but leave cores idle between edges, so the crops
    are built lazily for a whole batch and computed in one call. The batch --
    rather than the whole graph -- keeps peak memory bounded.

    Args:
        sims: List of spatial images, one per view.
        edges: List of ``(index_i, index_j)`` view index pairs.
        transform_key: Coordinate system to compute overlaps in.
        batch_size: Number of edges to realise per ``dask.compute`` call.

    Yields:
        Tuples of ``(index_i, index_j, values_i, values_j)``, where the values
        are in-memory float64 arrays over the shared overlap grid.
    """
    for start in range(0, len(edges), batch_size):
        batch = edges[start : start + batch_size]
        lazy = []
        for index_i, index_j in batch:
            overlap_bboxes = registration.get_overlap_bboxes(
                sims[index_i],
                sims[index_j],
                input_transform_key=transform_key,
            )
            crop_i, crop_j = registration.sims_to_intrinsic_coord_system(
                sims[index_i],
                sims[index_j],
                transform_key,
                overlap_bboxes,
            )
            lazy += [crop_i.data, crop_j.data]
        # Passes numpy arrays through untouched, so this also covers views that
        # were built from in-memory data rather than from a dask-backed store.
        computed = dask.compute(*lazy)
        for offset, (index_i, index_j) in enumerate(batch):
            yield (
                index_i,
                index_j,
                np.asarray(computed[2 * offset], dtype=np.float64),
                np.asarray(computed[2 * offset + 1], dtype=np.float64),
            )


def collect_pair_stats(
    msims,
    transform_key,
    estimator: str = "quantile",
    channel=None,
    n_quantiles: int = DEFAULT_N_QUANTILES,
    trim: tuple[float, float] = DEFAULT_TRIM,
    min_samples: int = 100,
    overlap_tolerance=None,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    """Collect pairwise and per-view intensity statistics from a mosaic.

    Overlapping pairs come from the same adjacency graph that
    ``multiview_stitcher.registration.register`` uses. For each pair, both
    views are resampled onto one common grid over their overlap, with ``NaN``
    outside the source data; only the intersection of the two valid masks is
    used, which is what handles a misalignment that pushes one crop past the
    other tile's real extent.

    Args:
        msims: List of multiscale spatial images, one per view.
        transform_key: Coordinate system to compute overlaps in. Use
            ``"initial_affine"`` to run off the raw stage positions, with no
            registration involved at all.
        estimator: ``"quantile"`` (default) to pair samples by matching
            quantiles, or ``"pixel"`` to pair them pixel by pixel. ``"pixel"``
            requires an accurate registration and degrades silently without
            one; see :func:`pixel_samples`.
        channel: Channel coordinate to estimate on. Defaults to the first.
        n_quantiles: Number of quantile levels, for the quantile estimator.
        trim: Quantile range, for the quantile estimator.
        min_samples: Pairs with fewer valid samples than this are skipped.
        overlap_tolerance: Passed through to the adjacency graph, to let views
            that only just touch count as overlapping.
        batch_size: Number of overlapping pairs whose crops are resampled per
            ``dask.compute`` call. Larger keeps more cores busy; smaller keeps
            peak memory down.

    Returns:
        Tuple of (pair statistics, per-view statistics, intensity scale). The
        moments are normalised by the returned scale.

    Raises:
        ValueError: If ``estimator`` is unknown.
    """
    if estimator not in ("quantile", "pixel"):
        raise ValueError(
            f"Unknown estimator {estimator!r}; expected 'quantile' or 'pixel'"
        )

    sims = [
        _select_scalar_view(msi_utils.get_sim_from_msim(msim), channel)
        for msim in msims
    ]
    graph = mv_graph.build_view_adjacency_graph_from_msims(
        msims,
        transform_key=transform_key,
        overlap_tolerance=overlap_tolerance,
    )

    pair_stats = []
    view_stats = [ViewStats() for _ in sims]
    n_skipped = 0

    for index_i, index_j, values_i, values_j in _iter_overlap_crops(
        sims, list(graph.edges), transform_key, batch_size
    ):
        valid = np.isfinite(values_i) & np.isfinite(values_j)
        if int(valid.sum()) < min_samples:
            n_skipped += 1
            continue
        values_i = values_i[valid]
        values_j = values_j[valid]

        view_stats[index_i].update(values_i)
        view_stats[index_j].update(values_j)

        if estimator == "quantile":
            samples_i, samples_j = quantile_samples(
                values_i, values_j, n_quantiles=n_quantiles, trim=trim
            )
        else:
            samples_i, samples_j = pixel_samples(values_i, values_j)

        pair_stats.append(
            pair_stats_from_samples(
                index_i,
                index_j,
                samples_i,
                samples_j,
                weight=float(valid.sum()),
            )
        )

    if n_skipped:
        warnings.warn(
            f"Skipped {n_skipped} overlapping pair(s) with fewer than "
            f"{min_samples} valid samples.",
            RuntimeWarning,
            stacklevel=2,
        )

    rms = [np.sqrt(v.mean_sq) for v in view_stats if v.n_samples]
    scale = float(max(rms)) if rms else 1.0
    if not scale > 0:
        scale = 1.0
    _rescale_stats(pair_stats, view_stats, scale)
    return pair_stats, view_stats, scale


def estimate_intensity_correction(
    msims,
    transform_key: str = "initial_affine",
    estimator: str = "quantile",
    model: str = "affine",
    drift_length: float = DEFAULT_DRIFT_LENGTH,
    channel=None,
    n_quantiles: int = DEFAULT_N_QUANTILES,
    trim: tuple[float, float] = DEFAULT_TRIM,
    min_samples: int = 100,
    overlap_tolerance=None,
    renormalize: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> IntensityCorrection:
    """Estimate one intensity transform per view from the tile overlaps.

    Because the model is affine in intensity, it commutes with both
    downsampling and (for positive gain) maximum projection. Estimating on a
    low-resolution, z-projected copy of the mosaic and applying the result to
    the full-resolution data is therefore exact, not an approximation.

    Args:
        msims: List of multiscale spatial images, one per view.
        transform_key: Coordinate system to compute overlaps in.
        estimator: ``"quantile"`` or ``"pixel"``; see
            :func:`collect_pair_stats`.
        model: ``"affine"`` or ``"offset"``; see
            :func:`solve_intensity_params`.
        drift_length: Anchor strength in tiles; see
            :func:`solve_intensity_params`.
        channel: Channel coordinate to estimate on. Defaults to the first.
        n_quantiles: Number of quantile levels, for the quantile estimator.
        trim: Quantile range, for the quantile estimator.
        min_samples: Pairs with fewer valid samples than this are skipped.
        overlap_tolerance: Passed through to the adjacency graph.
        renormalize: Whether to preserve the mosaic's overall mean and
            standard deviation.
        batch_size: Overlapping pairs resampled per ``dask.compute`` call.

    Returns:
        The fitted :class:`IntensityCorrection`, with parameters in the
        intensity units of the input data.
    """
    pair_stats, view_stats, scale = collect_pair_stats(
        msims,
        transform_key=transform_key,
        estimator=estimator,
        channel=channel,
        n_quantiles=n_quantiles,
        trim=trim,
        min_samples=min_samples,
        overlap_tolerance=overlap_tolerance,
        batch_size=batch_size,
    )
    correction = solve_intensity_params(
        n_views=len(msims),
        pair_stats=pair_stats,
        view_stats=view_stats,
        model=model,
        drift_length=drift_length,
        renormalize=renormalize,
    )
    # Undo the conditioning rescale: gains are dimensionless, offsets are not.
    correction.params[:, 1] *= scale
    return correction


def transform_array(array, gain, offset, dtype, min_value=None):
    """Apply ``gain * array + offset`` and cast back to ``dtype``.

    Works on numpy and dask arrays alike. ``gain`` and ``offset`` may be
    scalars, or arrays shaped to broadcast against ``array`` -- which is how a
    per-channel correction is applied in a single expression.

    Args:
        array: Input intensity array.
        gain: Multiplicative factor, scalar or broadcastable array.
        offset: Additive term in the intensity units of ``array``, scalar or
            broadcastable array.
        dtype: Output dtype, normally the dtype of the input mosaic.
        min_value: Lower clipping bound. Defaults to the dtype minimum.

    Returns:
        The transformed array, with dtype ``dtype``.
    """
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        low, high = info.min, info.max
    else:
        low, high = -np.inf, np.inf
    if min_value is not None:
        low = min_value

    gain = np.asarray(gain, dtype=np.float32)
    offset = np.asarray(offset, dtype=np.float32)
    result = array.astype(np.float32) * gain + offset
    result = result.clip(low, high)
    if np.issubdtype(dtype, np.integer):
        result = (result + np.float32(0.5)).astype(dtype)
    else:
        result = result.astype(dtype)
    return result


def _broadcast_params(sim, params, channel_axis):
    """Shape per-channel ``(gain, offset)`` to broadcast against a view.

    Args:
        sim: The spatial image the parameters will be applied to.
        params: ``(n_channels, 2)`` array of per-channel parameters.
        channel_axis: Position of the channel axis in ``sim``, or None if the
            view has no channel axis.

    Returns:
        Tuple of (gain, offset), each a scalar or a broadcastable array.
    """
    if channel_axis is None or len(params) == 1:
        return float(params[0][0]), float(params[0][1])
    shape = [1] * sim.ndim
    shape[channel_axis] = len(params)
    return (
        params[:, 0].reshape(shape).astype(np.float32),
        params[:, 1].reshape(shape).astype(np.float32),
    )


def apply_intensity_params(
    sims,
    corrections,
    dtype=None,
    min_value=None,
):
    """Apply a fitted intensity correction to a list of views.

    Applying before fusion (rather than to the fused mosaic) keeps the
    background padding that fusion introduces at zero, so the correction never
    disturbs the region outside the tiles.

    Args:
        sims: List of spatial images, one per view, in the same order as the
            views the correction was fitted on.
        corrections: A single :class:`IntensityCorrection`, or a list of them
            with one entry per channel, ordered as the views' ``c``
            coordinate. Channels are fitted independently because a gain
            estimated on one channel does not describe another.
        dtype: Output dtype. Defaults to the dtype of each input view.
        min_value: Lower clipping bound; see :func:`transform_array`.

    Returns:
        A new list of spatial images with corrected intensities.

    Raises:
        ValueError: If the number of views does not match the corrections, or
            if the number of corrections does not match the number of
            channels.
    """
    if isinstance(corrections, IntensityCorrection):
        corrections = [corrections]
    n_views = len(corrections[0].params)
    if len(sims) != n_views:
        raise ValueError(
            f"Correction was fitted on {n_views} views but "
            f"{len(sims)} were given."
        )

    corrected = []
    for view_index, sim in enumerate(sims):
        channel_axis = (
            sim.dims.index("c") if "c" in sim.dims else None
        )
        if channel_axis is not None and len(corrections) not in (
            1,
            sim.shape[channel_axis],
        ):
            raise ValueError(
                f"Got {len(corrections)} channel correction(s) for a view "
                f"with {sim.shape[channel_axis]} channels."
            )
        params = np.array(
            [correction.params[view_index] for correction in corrections]
        )
        gain, offset = _broadcast_params(sim, params, channel_axis)
        out_dtype = sim.dtype if dtype is None else dtype
        corrected.append(
            sim.copy(
                data=transform_array(
                    sim.data, gain, offset, out_dtype, min_value=min_value
                )
            )
        )
    return corrected
