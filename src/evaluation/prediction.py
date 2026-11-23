"""Apply a fitted reconciliation head to original-scale base forecasts.

The caller supplies a selected head in evaluation mode. This function neither
fits parameters nor selects a checkpoint; it preserves the supplied mode.
"""
from __future__ import annotations

import numpy as np
import torch

from src.strategies.common import ReconciliationHead


@torch.no_grad()
def apply_head(head: ReconciliationHead, yhat: np.ndarray, device: str = "cpu") -> np.ndarray:
    x = torch.tensor(yhat, dtype=torch.float32, device=device)
    return head.to(device)(x).cpu().numpy().astype(np.float64)
