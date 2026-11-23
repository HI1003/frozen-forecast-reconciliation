"""Classical post-hoc reconciliation baselines.

Methods: BU, TD-FP, OLS, WLS-var, WLS-struct, MinT-shrink.
All operate per horizon step on the cross-section of all-node base forecasts.

Protocol (thesis Chapter 3):
- Base forecasts are never truncated; non-negativity (for non-negative
  datasets) is imposed jointly with coherence at the reconciliation stage:
  BU clips bottom at zero, TD-FP uses positive-part proportions with a uniform
  fallback, and projection methods fall back to NNLS when the unconstrained
  solution has negative bottom values.
- Error statistics for WLS-var / MinT-shrink are estimated from base errors on
  the TRAIN origins only (never validation or test).

MinT-shrink uses node-column blocks to bound covariance temporaries. For repeated
prediction, call ``prepare_projection`` once and then ``model.predict(yhat)``;
``reconcile_classical`` remains the convenience entry point for a single call.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from src.data.hierarchy import Hierarchy

EPS = 1e-8
MAX_WEIGHT_CONDITION = 1e12
MAX_DESIGN_CONDITION = 1e10

CLASSICAL_METHODS = ["base", "bu", "td_fp", "ols", "wls_var", "wls_struct", "mint_shrink"]


def reconcile_classical(
    method: str,
    yhat: np.ndarray,            # (N, n) base forecasts at one step
    hier: Hierarchy,
    train_errors: np.ndarray | None = None,   # (N_train, n) base errors for weight estimation
    nonneg: bool = True,
) -> np.ndarray:
    yhat = np.asarray(yhat, dtype=np.float64)
    if yhat.ndim != 2 or yhat.shape[1] != hier.n:
        raise ValueError(f"yhat shape {yhat.shape} must be (N, {hier.n})")
    if not np.isfinite(yhat).all():
        raise ValueError("yhat contains non-finite values")
    if method == "base":
        return yhat.copy()
    if method == "bu":
        return _bu(yhat, hier, nonneg)
    if method == "td_fp":
        return _td_fp(yhat, hier, nonneg)
    if method in ("ols", "wls_var", "wls_struct", "mint_shrink"):
        W = _weight_matrix(method, hier, train_errors)
        return _projection(yhat, hier, W, nonneg)
    raise ValueError(f"unknown classical method: {method}")


def _finalize(b: np.ndarray, hier: Hierarchy) -> np.ndarray:
    return b @ hier.S.T


def _bu(yhat: np.ndarray, hier: Hierarchy, nonneg: bool) -> np.ndarray:
    b = yhat[:, hier.bottom_idx]
    if nonneg:
        b = np.maximum(b, 0.0)
    return _finalize(b, hier)


def _td_fp(yhat: np.ndarray, hier: Hierarchy, nonneg: bool) -> np.ndarray:
    """Top-down with forecast proportions from the base bottom forecasts."""
    top = yhat[:, hier.top_idx[0]]
    if nonneg:
        top = np.maximum(top, 0.0)
    b_base = np.maximum(yhat[:, hier.bottom_idx], 0.0)
    denom = b_base.sum(axis=1, keepdims=True)
    uniform = np.full((1, hier.m), 1.0 / hier.m)
    props = np.where(denom > EPS, b_base / np.maximum(denom, EPS), uniform)
    b = props * top[:, None]
    return _finalize(b, hier)


def _weight_matrix(method: str, hier: Hierarchy, train_errors: np.ndarray | None) -> np.ndarray:
    n = hier.n
    if method == "ols":
        return np.eye(n)
    if method == "wls_struct":
        return np.diag(hier.S.sum(axis=1))
    if train_errors is None:
        raise ValueError(f"{method} requires train-origin base errors")
    train_errors = np.asarray(train_errors, dtype=np.float64)
    if train_errors.ndim != 2 or train_errors.shape[1] != n:
        raise ValueError(f"{method} train_errors shape must be (N, {n})")
    if train_errors.shape[0] < 3:
        raise ValueError(f"{method} requires at least three train-origin base errors")
    if not np.isfinite(train_errors).all():
        raise ValueError(f"{method} train_errors contain non-finite values")
    if method == "wls_var":
        return np.diag(np.maximum(train_errors.var(axis=0, ddof=1), EPS))
    if method == "mint_shrink":
        return shrunk_covariance(train_errors)
    raise ValueError(method)


def shrunk_covariance(E: np.ndarray, block_columns: int = 4) -> np.ndarray:
    """Schafer-Strimmer shrinkage with bounded node-column temporaries.

    ``E`` contains training-origin errors only. Blocking changes allocation,
    not the two-pass formula, ddof, shrinkage clipping or diagonal ridge.
    The largest product block has shape (origins, block_columns, nodes).
    """
    E = np.asarray(E, dtype=np.float64)
    if E.ndim != 2 or E.shape[0] < 3 or E.shape[1] < 1 or not np.isfinite(E).all():
        raise ValueError("finite (N>=3, n>=1) train errors required")
    if isinstance(block_columns, bool) or not isinstance(block_columns, int) or block_columns < 1:
        raise ValueError("positive integer block size required")
    N, n = E.shape
    X = E - E.mean(axis=0, keepdims=True)
    C = (X.T @ X) / (N - 1)
    sd = np.sqrt(np.maximum(np.diag(C), EPS))
    Xs = X / sd
    R = (Xs.T @ Xs) / (N - 1)
    var_r = np.empty((n, n), dtype=np.float64)
    for lo in range(0, n, block_columns):
        hi = min(lo + block_columns, n)
        products = np.einsum("ki,kj->kij", Xs[:, lo:hi], Xs)
        mean = products.mean(axis=0)
        products -= mean
        np.square(products, out=products)
        var_r[lo:hi] = (N / ((N - 1) ** 3)) * products.sum(axis=0)
        del products
    off = ~np.eye(n, dtype=bool)
    denom = (R[off] ** 2).sum()
    lam = float(var_r[off].sum() / denom) if denom > EPS else 1.0
    lam = min(max(lam, 0.0), 1.0)
    shrunk = (1.0 - lam) * R
    np.fill_diagonal(shrunk, 1.0)
    W = shrunk * np.outer(sd, sd)
    W[np.diag_indices_from(W)] += 1e-6 * np.trace(W) / n
    return W


def _shrunk_covariance_reference(E: np.ndarray) -> np.ndarray:
    """Original dense formula, only for small preflight/parity fixtures.

    Production fitting uses ``shrunk_covariance``. Retaining this independent
    expression lets the final experiment driver verify blocked calculations.
    """
    N, n = E.shape
    X = E - E.mean(axis=0, keepdims=True)
    C = (X.T @ X) / (N - 1)
    sd = np.sqrt(np.maximum(np.diag(C), EPS))
    Xs = X / sd
    R = (Xs.T @ Xs) / (N - 1)
    # variance of the correlation estimates
    W_prod = np.einsum("ki,kj->kij", Xs, Xs)             # (N, n, n)
    wbar = W_prod.mean(axis=0)
    var_r = (N / ((N - 1) ** 3)) * ((W_prod - wbar) ** 2).sum(axis=0)
    off = ~np.eye(n, dtype=bool)
    denom = (R[off] ** 2).sum()
    lam = float(var_r[off].sum() / denom) if denom > EPS else 1.0
    lam = min(max(lam, 0.0), 1.0)
    R_shrunk = (1.0 - lam) * R
    np.fill_diagonal(R_shrunk, 1.0)
    W = R_shrunk * np.outer(sd, sd)
    # ridge for numerical safety
    W[np.diag_indices_from(W)] += 1e-6 * np.trace(W) / n
    return W


class PreparedProjection:
    """A fixed weighted reconciler; predict never estimates covariance again.

    With W = L L', unconstrained least squares and NNLS use the same whitened
    geometry (L^-1 S, L^-1 y). Preparation caches the whitening and bottom
    projection, following the deployment adapter used for the cost table.
    """

    def __init__(self, hier: Hierarchy, W: np.ndarray, nonneg: bool = True):
        self.n = hier.n
        self.nonneg = nonneg
        self.S = np.array(hier.S, dtype=np.float64, copy=True)
        W = np.asarray(W, dtype=np.float64)
        if W.shape != (hier.n, hier.n):
            raise ValueError(f"weight matrix shape {W.shape} != ({hier.n}, {hier.n})")
        if not np.isfinite(W).all():
            raise ValueError("weight matrix contains non-finite values")
        asymmetry = float(np.max(np.abs(W - W.T), initial=0.0))
        scale = max(float(np.max(np.abs(W), initial=0.0)), 1.0)
        if asymmetry > 1e-10 * scale:
            raise ValueError(f"weight matrix is not symmetric (max asymmetry={asymmetry:.3e})")
        W = 0.5 * (W + W.T)
        try:
            chol = np.linalg.cholesky(W)
        except np.linalg.LinAlgError as exc:
            raise ValueError("weight matrix is not symmetric positive definite") from exc
        eigenvalues = np.linalg.eigvalsh(W)
        condition = float(eigenvalues[-1] / eigenvalues[0])
        if not np.isfinite(condition) or condition > MAX_WEIGHT_CONDITION:
            raise ValueError(
                f"weight matrix is ill-conditioned (condition={condition:.3e}, "
                f"limit={MAX_WEIGHT_CONDITION:.1e})"
            )
        self.whitening = np.linalg.solve(chol, np.eye(hier.n))
        self.design = self.whitening @ self.S
        design_condition = float(np.linalg.cond(self.design))
        if not np.isfinite(design_condition) or design_condition > MAX_DESIGN_CONDITION:
            raise ValueError(
                f"weighted hierarchy design is ill-conditioned "
                f"(condition={design_condition:.3e}, limit={MAX_DESIGN_CONDITION:.1e})"
            )
        self.bottom_projection, _, rank, _ = np.linalg.lstsq(self.design, np.eye(hier.n), rcond=None)
        if rank != hier.m:
            raise ValueError(f"weighted hierarchy design is rank deficient ({rank} < {hier.m})")

    def predict(self, yhat: np.ndarray) -> np.ndarray:
        """Reconcile original-scale forecasts using the prepared mapping."""
        yhat = np.asarray(yhat, dtype=np.float64)
        if yhat.ndim != 2 or yhat.shape[1] != self.n:
            raise ValueError(f"yhat shape {yhat.shape} must be (N, {self.n})")
        if not np.isfinite(yhat).all():
            raise ValueError("yhat contains non-finite values")
        whitened_y = yhat @ self.whitening.T
        bottom = whitened_y @ self.bottom_projection.T
        if self.nonneg:
            for i in np.where((bottom < -EPS).any(axis=1))[0]:
                bottom[i], _ = nnls(self.design, whitened_y[i])
            bottom = np.maximum(bottom, 0.0)
        return bottom @ self.S.T


def prepare_projection(
    method: str,
    hier: Hierarchy,
    train_errors: np.ndarray | None = None,
    nonneg: bool = True,
) -> PreparedProjection:
    """Prepare OLS, WLS-var, WLS-struct or MinT-shrink once for repeated use.

    Supply only training-origin errors for WLS-var and MinT-shrink. For example::

        model = prepare_projection('mint_shrink', hierarchy, train_errors)
        predictions = model.predict(base_forecasts)
    """
    if method not in ("ols", "wls_var", "wls_struct", "mint_shrink"):
        raise ValueError(f"method does not use a weighted projection: {method}")
    return PreparedProjection(hier, _weight_matrix(method, hier, train_errors), nonneg)


def _projection(yhat: np.ndarray, hier: Hierarchy, W: np.ndarray, nonneg: bool) -> np.ndarray:
    """Single-call compatibility wrapper around the prepared projection."""
    return PreparedProjection(hier, W, nonneg).predict(yhat)
