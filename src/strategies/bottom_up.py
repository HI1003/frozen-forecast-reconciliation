"""Bottom-up strategies: BU (fixed), BUL (linear), BUN (MLP).

Only bottom forecasts are corrected; all aggregate outputs are reconstructed
through the summing matrix. BUL/BUN start with a zero correction around BU.
"""
from torch import nn
from .common import ReconciliationHead, _mlp

__all__ = ['BU', 'BUL', 'BUN']

class BU(ReconciliationHead):
    def bottom(self, yhat):
        return self._clip(yhat[:, self.bottom_idx])


class BUL(ReconciliationHead):
    """Learned linear residual on the bottom slice, zero-initialised: the
    initial map is exactly classical BU and weight decay shrinks back to it."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.delta = nn.Linear(self.m, self.m)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def bottom(self, yhat):
        xb = self._norm(yhat)[:, self.bottom_idx]
        return self._clip(yhat[:, self.bottom_idx] + self.delta(xb) * self.x_std[self.bottom_idx])


class BUN(ReconciliationHead):
    """Nonlinear residual correction of the bottom slice."""

    def __init__(self, *args, hidden=(256, 128), dropout=0.1, **kw):
        super().__init__(*args, **kw)
        self.mlp = _mlp(self.m, self.m, list(hidden), dropout)

    def bottom(self, yhat):
        xb = self._norm(yhat)[:, self.bottom_idx]
        delta = self.mlp(xb) * self.x_std[self.bottom_idx]
        return self._clip(yhat[:, self.bottom_idx] + delta)
