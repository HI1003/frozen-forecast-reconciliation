"""Paired forecast comparisons, rank agreement and moving-block uncertainty.

These helpers operate on supplied losses or scores. They do not load test
forecasts, fit models, or choose hyperparameters.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

# A degenerate Bartlett-HAC variance fails rather than changing estimators.
DM_HAC_RELATIVE_ZERO_TOL = 1e-12


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1) -> tuple[float, float]:
    """Harvey-adjusted two-sided Diebold-Mariano test on per-origin losses.

    The pre-registered long-run variance estimator is a Bartlett-kernel HAC
    estimate with maximum lag ``h - 1``. Autocovariances use denominator ``T``
    and Bartlett weight ``1 - k / h``. A non-positive/degenerate HAC estimate
    makes the DM statistic undefined and therefore fails explicitly; it is
    never replaced by an unregistered variance estimator.

    Returns (statistic, p_value). Negative statistic favours method a.
    """
    a = np.asarray(loss_a, dtype=np.float64)
    b = np.asarray(loss_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError(
            "dm_test needs two 1-D loss arrays of equal shape; "
            f"got {a.shape} and {b.shape}"
        )
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("dm_test loss arrays must contain only finite values")
    if isinstance(h, (bool, np.bool_)) or not isinstance(h, (int, np.integer)):
        raise ValueError(f"dm_test horizon h must be a positive integer; got {h!r}")
    h = int(h)
    if h < 1:
        raise ValueError(f"dm_test horizon h must be >= 1; got {h}")

    d = a - b
    T = d.size
    if T <= h:
        raise ValueError(
            f"dm_test requires more origins than the horizon (T > h); got "
            f"T={T}, h={h}"
        )
    dbar = d.mean()
    centered = d - dbar
    gamma0 = float(np.dot(centered, centered) / T)
    long_run_variance = gamma0
    for k in range(1, h):
        gamma_k = float(np.dot(centered[k:], centered[:-k]) / T)
        long_run_variance += 2.0 * (1.0 - k / h) * gamma_k
    if not np.isfinite(long_run_variance):
        raise ValueError(
            "dm_test Bartlett-HAC long-run variance is non-finite; "
            "check the loss scale"
        )
    zero_tol = max(
        np.finfo(np.float64).tiny,
        DM_HAC_RELATIVE_ZERO_TOL * gamma0,
    )
    if long_run_variance <= zero_tol:
        raise ValueError(
            "dm_test Bartlett-HAC long-run variance is non-positive or "
            f"numerically degenerate (estimate={long_run_variance:.17g}, "
            f"threshold={zero_tol:.17g}, T={T}, lag={h - 1}); the DM "
            "statistic is undefined and no fallback estimator is allowed"
        )
    var_d = long_run_variance / T
    dm = dbar / np.sqrt(var_d)
    harvey = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm *= harvey
    p = 2.0 * (1.0 - stats.t.cdf(abs(dm), df=T - 1))
    return float(dm), float(p)


def kendall_tau_b(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Kendall tau-b rank correlation between two score vectors.

    tau-b (not tau-a) because reconciliation methods can tie on a rounded WAPE
    and tau-b is the variant with the tie correction in both margins.  Inputs
    are scores in a consistent direction (here: WAPE, lower is better); the
    correlation is invariant to any monotone transform, so ranking first is
    unnecessary.  Identical orderings give +1, exactly reversed gives -1.

    The returned p-value is the asymptotic rank-correlation test, which treats
    the METHODS as the sample.  It says nothing about rolling-origin sampling
    uncertainty, so the analysis scripts report tau descriptively and take its
    interval from a paired moving-block bootstrap over origins instead.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError(f"kendall_tau_b needs two 1-D arrays of equal length; "
                         f"got {a.shape} and {b.shape}")
    if a.size < 2:
        return float("nan"), float("nan")
    # Tuple unpacking, not `.statistic`: SciPy returns KendalltauResult
    # (`.correlation`) before 1.9 and SignificanceResult (`.statistic`) after,
    # but both unpack as (tau, p) on every version.
    tau, p = stats.kendalltau(a, b, variant="b", nan_policy="raise")
    return float(tau), float(p)


def moving_block_indices(n_obs: int, block_len: int, n_boot: int,
                         seed: int = 42) -> np.ndarray:
    """(n_boot, n_obs) resample indices built from contiguous blocks.

    Shared by every bootstrap in the project so that a PAIRED bootstrap — one
    draw of origins reused across bases and methods — is the easy thing to
    write.  Comparing two rankings resampled independently would inflate their
    apparent disagreement.
    """
    block_len = max(1, int(block_len))
    if n_obs < block_len or n_obs < 2:
        raise ValueError(f"need at least {block_len} observations; got {n_obs}")
    n_blocks = int(np.ceil(n_obs / block_len))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n_obs - block_len + 1, size=(n_boot, n_blocks))
    offsets = np.arange(block_len)
    return (starts[:, :, None] + offsets[None, None, :]).reshape(n_boot, -1)[:, :n_obs]


def percentile_ci(draws: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided percentile interval of a bootstrap draw vector.

    The single place the interval convention lives, so the CIs published for
    WAPE and for Kendall tau cannot drift apart.  Non-finite draws are dropped
    (tau is undefined when a resample ties every method); an all-non-finite
    input yields (nan, nan) rather than a fabricated interval.

    Rolling origins overlap heavily: consecutive entries share context rows and
    (for h > 1) forecast horizons, so the draws fed here must come from
    `moving_block_indices`, never from an IID resample of origins.
    """
    draws = np.asarray(draws, dtype=np.float64)
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def seed_mean_per_origin_abs_loss(predictions, ytrue: np.ndarray) -> np.ndarray:
    """Mean training-seed loss at each origin, never an ensemble forecast.

    Absolute loss is computed for each seed first and only then averaged. This
    distinction matters: opposing seed errors can cancel in an average forecast
    even though neither training run achieved that ensemble performance.
    """
    preds = [np.asarray(p, dtype=np.float64) for p in predictions]
    if not preds:
        raise ValueError("at least one prediction array is required")
    ytrue = np.asarray(ytrue, dtype=np.float64)
    for p in preds:
        if p.shape != ytrue.shape:
            raise ValueError(f"prediction shape {p.shape} != ytrue shape {ytrue.shape}")
    losses = [np.abs(p - ytrue).sum(axis=1) for p in preds]
    return np.mean(np.stack(losses, axis=0), axis=0)


def holm_adjust(p_values) -> np.ndarray:
    """Holm step-down FWER-adjusted p-values in the original input order."""
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("p_values must be a finite 1-D array in [0, 1]")
    m = p.size
    if m == 0:
        return p.copy()
    order = np.argsort(p, kind="stable")
    scaled = (m - np.arange(m)) * p[order]
    adjusted_sorted = np.minimum(1.0, np.maximum.accumulate(scaled))
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def block_mean_ci(values: np.ndarray, indices: np.ndarray,
                  alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI for a mean using precomputed moving-block indices."""
    values = np.asarray(values, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    if values.ndim != 1 or indices.ndim != 2:
        raise ValueError("values must be 1-D and indices must be 2-D")
    if indices.size and (indices.min() < 0 or indices.max() >= values.size):
        raise ValueError("bootstrap index out of range")
    return percentile_ci(values[indices].mean(axis=1), alpha)
