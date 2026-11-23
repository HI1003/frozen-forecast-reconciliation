"""Shared hierarchy buffers, train-only scaling and zero-output MLP builder.

Strategy definitions are in bottom_up.py, top_down.py and hybrid.py.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn

def _mlp(d_in: int, d_out: int, hidden: list[int], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = d_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.GELU(), nn.LayerNorm(h), nn.Dropout(dropout)]
        prev = h
    last = nn.Linear(prev, d_out)
    # Canonical residual heads start with a zero correction. Direct-map controls
    # reuse this builder but can set a different output bias after construction.
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


class ReconciliationHead(nn.Module):
    """Base class holding hierarchy structure and input scaling."""

    def __init__(self, S: np.ndarray, bottom_idx: np.ndarray, top_idx: np.ndarray,
                 x_mean: np.ndarray, x_std: np.ndarray, nonneg: bool = True):
        super().__init__()
        self.register_buffer("S", torch.tensor(S, dtype=torch.float32))
        self.register_buffer("x_mean", torch.tensor(x_mean, dtype=torch.float32))
        self.register_buffer("x_std", torch.tensor(np.maximum(x_std, 1e-6), dtype=torch.float32))
        self.bottom_idx = torch.tensor(np.asarray(bottom_idx), dtype=torch.long)
        agg = np.setdiff1d(np.arange(S.shape[0]), np.asarray(bottom_idx))
        self.agg_idx = torch.tensor(agg, dtype=torch.long)
        self.top_idx = int(np.asarray(top_idx)[0])
        self.n, self.m = S.shape
        self.nonneg = nonneg
        self.coherent_by_construction = True

    def _norm(self, yhat: torch.Tensor) -> torch.Tensor:
        return (yhat - self.x_mean) / self.x_std

    def _clip(self, b: torch.Tensor) -> torch.Tensor:
        return torch.relu(b) if self.nonneg else b

    def _reconstruct(self, b: torch.Tensor) -> torch.Tensor:
        return b @ self.S.T

    def bottom(self, yhat: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def forward(self, yhat: torch.Tensor) -> torch.Tensor:
        return self._reconstruct(self.bottom(yhat))
