"""Empirical column-group lasso, no intercept, with a primal-dual certificate.

Objective: ||Y - X G.T S.T||_F^2/(2 N) + lambda sum_j w_j ||G[:,j]||_2.
Only a common scalar rescales the entire objective for numerical conditioning;
there is no node-wise standardisation or change to original-scale weighting.
Training sufficient statistics avoid an (N*n) by (m*n) Kronecker design.
"""
from __future__ import annotations


import time
import numpy as np


def prepare(X, Y, S):
    X, Y, S = (np.asarray(v, dtype=np.float64) for v in (X, Y, S))
    if (X.ndim != 2 or Y.shape != X.shape or S.ndim != 2
            or X.shape[1] != S.shape[0] or len(X) < 2
            or not all(np.isfinite(v).all() for v in (X, Y, S))):
        raise ValueError('invalid or nonfinite training arrays')
    A = S.T @ S
    if np.linalg.matrix_rank(S) != S.shape[1]:
        raise ValueError('S must have full column rank')
    scale2 = max(float(np.mean(Y * Y)), 1.0)
    B = X.T @ X / (len(X) * scale2)
    C = S.T @ (Y.T @ X / (len(X) * scale2))
    yy = float(np.sum(Y * Y) / (len(X) * scale2))
    ols = np.linalg.solve(A, S.T)
    weights = 1.0 / np.linalg.norm(ols, axis=0)
    if not np.isfinite(weights).all():
        raise ValueError('undefined OLS group weight')
    lmax = float(np.max(np.linalg.norm(C, axis=0) / weights))
    lip = float(np.linalg.eigvalsh(A)[-1] * np.linalg.eigvalsh(B)[-1])
    if lip <= 0 or lmax <= 0:
        raise ValueError('degenerate all-zero training predictors/targets')
    # Direct least squares is used only for the lambda=0 endpoint. It avoids
    # the conditioning loss of solving normal equations or silently adding ridge.
    bottom_targets = np.linalg.solve(A, S.T @ Y.T).T
    zero = np.linalg.lstsq(X, bottom_targets, rcond=None)[0].T
    return dict(A=A, B=B, C=C, yy=yy, weights=weights, lmax=lmax,
                lip=lip, scale2=scale2, zero=zero)


def certificate(p, G, lam):
    linear = float(np.sum(G * p['C']))
    quad = float(np.sum(G * (p['A'] @ G @ p['B'])))
    residual2 = max(0.0, p['yy'] - 2 * linear + quad)
    primal = .5 * residual2 + lam * float(p['weights'] @ np.linalg.norm(G, axis=0))
    corr = p['C'] - p['A'] @ G @ p['B']
    norms = np.linalg.norm(corr, axis=0)
    ratio = np.divide(lam * p['weights'], norms, out=np.full_like(norms, np.inf), where=norms > 0)
    factor = min(1.0, float(np.min(ratio)))
    dual = factor * (p['yy'] - linear) - .5 * factor * factor * residual2
    gap = primal - dual
    if gap < -1e-7 * max(1.0, abs(primal)):
        raise ValueError('negative primal-dual gap beyond rounding tolerance')
    return primal, max(0.0, gap)


def solve(p, alpha, initial=None, *, max_iter=50000, gap_rtol=1e-6,
          gap_atol=1e-8, check_every=25):
    if not 0 <= alpha <= 1 or max_iter < 1 or check_every < 1:
        raise ValueError('invalid solver configuration')
    start = time.perf_counter()
    lam = float(alpha * p['lmax'])
    if alpha == 0:
        G = p['zero'].copy()
        grad = p['A'] @ G @ p['B'] - p['C']
        stationarity = float(np.linalg.norm(grad) / max(1., np.linalg.norm(p['C'])))
        if stationarity > 1e-7:
            raise RuntimeError('unregularized least-squares stationarity check failed')
        primal = certificate(p, G, 0)[0]
        return G, dict(iterations=0, objective=primal*p['scale2'], dual_gap=None,
                       stationarity=stationarity, seconds=time.perf_counter()-start,
                       lambda_original=0., converged=True)
    G = np.zeros_like(p['C']) if initial is None else np.array(initial, dtype=float, copy=True)
    Z, momentum = G.copy(), 1.0
    for iteration in range(1, max_iter + 1):
        step = Z - (p['A'] @ Z @ p['B'] - p['C']) / p['lip']
        norms = np.linalg.norm(step, axis=0)
        shrink = np.maximum(0., 1. - lam * p['weights'] / (p['lip'] * np.maximum(norms, 1e-300)))
        new = step * shrink
        next_momentum = .5 * (1 + np.sqrt(1 + 4 * momentum * momentum))
        if np.sum((Z - new) * (new - G)) > 0:  # adaptive gradient restart
            next_momentum = 1.0
        Z = new + (momentum - 1) / next_momentum * (new - G) if next_momentum > 1 else new.copy()
        G, momentum = new, next_momentum
        if iteration == 1 or iteration % check_every == 0 or iteration == max_iter:
            primal, gap = certificate(p, G, lam)
            if gap <= gap_atol + gap_rtol * abs(primal):
                return G, dict(iterations=iteration, objective=primal*p['scale2'],
                    dual_gap=gap*p['scale2'], relative_gap=gap/max(abs(primal), gap_atol),
                    seconds=time.perf_counter()-start, lambda_original=lam*p['scale2'], converged=True)
    raise RuntimeError(f'Elasso did not converge: alpha={alpha:g}, iterations={max_iter}, gap={gap:g}, objective={primal:g}')


def predict(X, G, S):
    bottom = np.asarray(X, dtype=np.float64) @ G.T
    if not np.isfinite(bottom).all():
        raise ValueError('nonfinite predictions')
    return np.maximum(bottom, 0) @ S.T, float(np.mean(bottom < 0))


def choose_alpha(rows, alphas, expected_cells):
    """Require complete, unique coverage before selecting any dataset winner."""
    keys = [(r['cell'], r['alpha_index']) for r in rows]
    expected = {(c, i) for c in expected_cells for i in range(len(alphas))}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('incomplete or duplicate validation grid')
    scores = []
    for i, alpha in enumerate(alphas):
        unit = [r for r in rows if r['alpha_index'] == i]
        ratios = [r['val_mae']/r['base_val_mae'] for r in unit if r['base_val_mae'] > 0]
        if len(ratios) != len(expected_cells) or not np.isfinite(ratios).all():
            raise ValueError('invalid validation denominator/score')
        scores.append(dict(alpha_index=i, alpha=float(alpha), score=float(np.mean(ratios))))
    return min(scores, key=lambda r: (r['score'], -r['alpha'])), scores
