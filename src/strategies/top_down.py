"""Top-down strategies: TD (fixed), TDL (linear), TDN (MLP).

The nonnegative top total is retained. TDL/TDN correct allocation logits using
only (top total, bottom shares), with feature statistics fitted on train data.
AlignedTD is the shared implementation name retained for checkpoint audits;
TDL and TDN below expose the final paper taxonomy directly.
"""
import numpy as np
import torch
from torch import nn
from .common import ReconciliationHead, _mlp

__all__ = ['TD', 'TDL', 'TDN']

class TD(ReconciliationHead):
    """Direct top-down: forecast proportions from base bottom positive parts."""

    def bottom(self, yhat):
        top = yhat[:, self.top_idx]
        if self.nonneg:
            top = torch.relu(top)
        b_base = torch.relu(yhat[:, self.bottom_idx])
        denom = b_base.sum(dim=1, keepdim=True)
        uniform = torch.full_like(b_base, 1.0 / self.m)
        props = torch.where(denom > 1e-8, b_base / denom.clamp_min(1e-8), uniform)
        return props * top.unsqueeze(1)


class _TDLearned(ReconciliationHead):
    """Shared logic: trusted top value allocated by learned softmax proportions.
    The mapping form (linear vs nonlinear) lives in the logit network, which is
    a zero-initialised residual on top of the direct-TD allocation logits."""

    def _base_logits(self, yhat):
        """Log of the direct-TD shares (positive-part forecast proportions with
        uniform fallback). A zero logit network numerically approximates
        classical TD at initialisation: strictly positive shares are preserved,
        while exact-zero shares receive the documented 1e-8 floor required for
        finite logits."""
        b_base = torch.relu(yhat[:, self.bottom_idx])
        denom = b_base.sum(dim=1, keepdim=True)
        uniform = torch.full_like(b_base, 1.0 / self.m)
        shares = torch.where(denom > 1e-8, b_base / denom.clamp_min(1e-8), uniform)
        return torch.log(shares.clamp_min(1e-8))

    def _proportions(self, yhat):  # pragma: no cover - abstract
        raise NotImplementedError

    def bottom(self, yhat):
        top = yhat[:, self.top_idx]
        if self.nonneg:
            top = torch.relu(top)
        props = torch.softmax(self._proportions(yhat), dim=1)
        return props * top.unsqueeze(1)


def td_features_numpy(yhat, bottom_idx, top_idx):
    """Training-only feature statistics; clipping is reconciliation, not base edits."""
    values = np.asarray(yhat, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("TD feature forecasts must be finite origin-by-node arrays")
    positive = np.maximum(values[:, bottom_idx], 0)
    total = positive.sum(1, keepdims=True)
    shares = positive / np.maximum(total, 1e-8)
    shares[total[:, 0] <= 1e-8] = 1.0 / positive.shape[1]
    top = np.maximum(values[:, int(np.asarray(top_idx)[0])], 0)[:, None]
    return np.concatenate((top, shares), axis=1)


class AlignedTD(_TDLearned):
    def __init__(self, *args, nonlinear=False, hidden=(256, 128), dropout=.1,
                 feature_mean=None, feature_std=None, **kw):
        super().__init__(*args, **kw)
        if not self.nonneg:
            raise ValueError("aligned TD requires the registered nonnegative protocol")
        count = self.m + 1
        mean = np.zeros(count) if feature_mean is None else np.asarray(feature_mean)
        std = np.ones(count) if feature_std is None else np.asarray(feature_std)
        if mean.shape != (count,) or std.shape != (count,) or not (
                np.isfinite(mean).all() and np.isfinite(std).all() and (std >= 0).all()):
            raise ValueError("invalid train-only TD feature statistics")
        self.register_buffer("td_mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("td_std", torch.tensor(np.maximum(std, 1e-6), dtype=torch.float32))
        self.logits = (_mlp(count, self.m, list(hidden), dropout) if nonlinear
                       else nn.Linear(count, self.m))
        if not nonlinear:
            nn.init.zeros_(self.logits.weight)
            nn.init.zeros_(self.logits.bias)

    def features(self, yhat):
        bottom = torch.relu(yhat[:, self.bottom_idx])
        total = bottom.sum(1, keepdim=True)
        shares = torch.where(total > 1e-8, bottom / total.clamp_min(1e-8),
                             torch.full_like(bottom, 1.0 / self.m))
        top = torch.relu(yhat[:, self.top_idx:self.top_idx + 1])
        return (torch.cat((top, shares), dim=1) - self.td_mean) / self.td_std

    def _proportions(self, yhat):
        return self._base_logits(yhat) + self.logits(self.features(yhat))


class TDL(AlignedTD):
    """Linear correction of TD allocation logits from standardized (T, q)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, nonlinear=False, **kwargs)


class TDN(AlignedTD):
    """MLP correction of TD allocation logits from standardized (T, q)."""

    def __init__(self, *args, hidden=(256, 128), dropout=.1, **kwargs):
        super().__init__(*args, nonlinear=True, hidden=hidden, dropout=dropout, **kwargs)
