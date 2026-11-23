"""Original-scale accuracy and coherence metrics, with level-wise reporting.
Statistical comparisons and uncertainty intervals live in statistics.py.

All metrics are computed in the original scale (thesis framework protocol).
`yhat`/`ytrue` arrays follow the frozen-forecast layout (N_origins, n, H) and a
step index selects one horizon step h (0-based position in the stored steps).
"""
from __future__ import annotations

import numpy as np

from src.data.hierarchy import LEVEL_NAMES, Hierarchy

EPS = 1e-8


def wape(err_abs: np.ndarray, y_abs: np.ndarray) -> float:
    return float(err_abs.sum() / max(y_abs.sum(), EPS)) * 100.0


# Frozen MASE seasonal period for each configured formal sampling frequency, in
# rows. Seasonal MASE uses the in-sample seasonal-naive error as its scale;
# m=1 is the non-seasonal special case. The two registered choices are protocol
# decisions supported by domain cadence and validation-only diagnostics, not
# model hyperparameters. Unknown frequencies fail rather than silently using a
# non-seasonal scale for a dataset whose MASE period has not been registered.
SEASONAL_PERIODS = {
    "D": 7,      # daily retail -> weekly cycle
    "h": 24,     # hourly load -> daily cycle
    "H": 24,     # pandas < 2.2 spelling of the hourly alias
}


def seasonal_period(freq: str) -> int:
    """Rows in one dominant seasonal cycle for a configured pandas frequency."""
    key = str(freq)
    if key not in SEASONAL_PERIODS:
        raise ValueError(
            f"no seasonal period registered for frequency {freq!r}; add it to "
            "SEASONAL_PERIODS rather than defaulting to a lag-1 scale"
        )
    return SEASONAL_PERIODS[key]


def mase_scale(Y: np.ndarray, train_end: int, season: int) -> np.ndarray:
    """Per-node in-sample seasonal-naive scale over the training rows.

    ``season`` rows back is the benchmark forecast, so the scale is
    mean |Y[t] - Y[t-season]| across the training segment.  With season=1 this
    reduces to the classical Hyndman-Koehler MASE scale.

    Rows imputed upstream are not excluded here: they are a handful of DST hours
    inside tens of thousands of training rows, and excluding them would make the
    denominator depend on the origin-exclusion policy, which is about forecast
    targets rather than about the benchmark's in-sample error.
    """
    season = int(season)
    if season < 1:
        raise ValueError(f"seasonal period must be >= 1; got {season}")
    if train_end <= season:
        raise ValueError(
            f"training segment of {train_end} rows cannot support a lag-{season} "
            "seasonal scale"
        )
    diffs = np.abs(Y[season:train_end] - Y[: train_end - season])
    return diffs.mean(axis=0)


def level_metrics(
    yhat: np.ndarray,          # (N, n) point forecasts at one step, original scale
    ytrue: np.ndarray,         # (N, n)
    hier: Hierarchy,
    scale: np.ndarray,         # (n,) seasonal-naive MASE scale per node
) -> dict:
    """Return {level: {"wape": ..., "mase": ...}} plus "all"."""
    err = np.abs(yhat - ytrue)
    out = {}
    for lvl in ["all"] + LEVEL_NAMES:
        idx = np.arange(hier.n) if lvl == "all" else hier.level_index[lvl]
        if idx.size == 0:
            continue
        e, y = err[:, idx], np.abs(ytrue[:, idx])
        node_mae = e.mean(axis=0)
        sc = scale[idx]
        ok = sc > EPS
        out[lvl] = {
            "wape": wape(e, y),
            "mase": float((node_mae[ok] / sc[ok]).mean()) if ok.any() else float("nan"),
            "n_nodes": int(idx.size),
            "n_mase_excluded": int((~ok).sum()),
        }
    return out


def coherence_scale(ytrue_train: np.ndarray, hier: Hierarchy) -> float:
    """Shared normalization scale for coherence: mean |y| over AGGREGATE nodes
    of the TRAIN targets.  Train loss, validation selection, and test metric
    all use this one denominator, so epsilon targets keep the same meaning
    across splits and datasets."""
    return max(float(np.abs(ytrue_train[:, hier.agg_idx]).mean()), EPS)


def coherence_nmae(ytilde: np.ndarray, hier: Hierarchy, coh_scale: float) -> float:
    """Aggregate-constraint violation in percent: mean |ytilde - S b_tilde| over
    AGGREGATE rows only, normalized by coh_scale (see coherence_scale).

    Bottom rows of the residual are identically zero by construction; averaging
    over all n nodes would dilute the violation by n/(n-m) — a factor that
    differs ~3x between our datasets — so they are excluded."""
    recon = ytilde[:, hier.bottom_idx] @ hier.S.T
    viol = np.abs(ytilde - recon)[:, hier.agg_idx].mean()
    return float(viol / max(coh_scale, EPS)) * 100.0


def per_origin_abs_loss(yhat: np.ndarray, ytrue: np.ndarray) -> np.ndarray:
    """Total absolute error per origin (summed over nodes) — DM input."""
    return np.abs(yhat - ytrue).sum(axis=1)
