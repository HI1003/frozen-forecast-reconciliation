"""Original-scale prediction/coherence losses and checkpoint L1 scoring.

HYBN supports a fixed coherence penalty or an epsilon constraint. The shared
checkpoint scorer is used only on the splits supplied by each caller.
"""
from __future__ import annotations

import numpy as np
import torch

from src.strategies.common import ReconciliationHead


def _pred_and_coh(head: ReconciliationHead, ytilde, ytrue):
    """L_pred over all nodes; L_coh over AGGREGATE rows only (bottom rows of
    the coherence residual are zero by construction and would dilute it)."""
    l_pred = (ytilde - ytrue).abs().mean()
    if head.coherent_by_construction:
        l_coh = torch.zeros((), device=ytilde.device)
    else:
        recon = ytilde[:, head.bottom_idx] @ head.S.T
        l_coh = (ytilde - recon)[:, head.agg_idx].abs().mean()
    return l_pred, l_coh


def _normalized_constraint(l_coh, y_scale: float, eps_target: float):
    """Dimensionless HYBN-epsilon constraint residual: L_coh / s - epsilon."""
    return l_coh / y_scale - eps_target


def _epsilon_lagrangian(l_pred, l_coh, lam: float,
                        y_scale: float, eps_target: float):
    """Raw-original-scale Lagrangian for L_coh / s <= epsilon.

    Multiplying the normalized constraint by s gives the equivalent raw-scale
    constraint L_coh - epsilon*s <= 0.  This keeps the complete optimization
    loss in the original scale while the dual update can use the normalized
    residual as a dataset-scale-independent preconditioner.
    """
    return l_pred + lam * (l_coh - eps_target * y_scale)


@torch.no_grad()
def _dataset_l1(
    head: ReconciliationHead,
    yhat: np.ndarray,
    ytrue: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> float:
    """Recompute mean L1 for one selected checkpoint without a large GPU copy."""
    total = 0.0
    count = 0
    step = max(int(batch_size), 1)
    head.eval()
    for i in range(0, len(yhat), step):
        xb = torch.as_tensor(yhat[i : i + step], dtype=torch.float32,
                             device=device)
        tb = torch.as_tensor(ytrue[i : i + step], dtype=torch.float32,
                             device=device)
        total += float((head(xb) - tb).abs().sum().item())
        count += int(tb.numel())
    if count == 0:
        raise ValueError("cannot compute selected-checkpoint L1 on an empty split")
    value = total / count
    if not np.isfinite(value):
        raise ValueError("selected-checkpoint L1 is non-finite")
    return float(value)
