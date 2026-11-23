"""Hybrid strategies: HYB (fixed), HYBL (linear), HYBN (MLP).

All three start from the equal-weight four-level HYB allocation. HYBL corrects
bottom outputs and reconstructs aggregates; HYBN corrects all nodes and is the
only relaxed canonical strategy. Its coherence penalties live in training.
AlignedHybrid retains the shared implementation name used by source audits.
"""
import numpy as np
import torch
from torch import nn
from .common import ReconciliationHead, _mlp
from .hybrid_allocation import MULTILEVEL_HYB, LEVELS, hierarchy_contract

__all__ = ['HYB', 'HYBL', 'HYBN']

class HYB(ReconciliationHead):
    """Equal-weight allocation from all four hierarchy levels to the bottom.

    The final HYB definition has no fitted parameters. The definition argument
    preserves its recorded identity and rejects historical two-level hybrids.
    """

    def __init__(self, *args, level_index=None, definition=MULTILEVEL_HYB, **kw):
        super().__init__(*args, **kw)
        self.definition = definition
        if definition != MULTILEVEL_HYB:
            raise ValueError('final HYB requires the four-level definition')
        if not self.nonneg or level_index is None:
            raise ValueError('four-level HYB requires nonneg=True and level_index')
        S, indices = hierarchy_contract(self.S.detach().cpu().numpy(), level_index)
        if (not np.array_equal(indices['bottom'], self.bottom_idx.cpu().numpy())
                or int(indices['top'][0]) != self.top_idx):
            raise ValueError('HYB layer indices differ from the head hierarchy')
        for level in LEVELS[:-1]:
            self.register_buffer('layer_nodes_' + level, torch.tensor(indices[level]))
            self.register_buffer('layer_membership_' + level,
                                 torch.tensor(S[indices[level]], dtype=torch.bool))

    def bottom(self, yhat):
        positive = torch.relu(yhat[:, self.bottom_idx])
        result = 0.25 * positive
        for level in LEVELS[:-1]:
            nodes = getattr(self, 'layer_nodes_' + level)
            members = getattr(self, 'layer_membership_' + level)
            allocation = torch.zeros_like(positive)
            for node, descendants in zip(nodes, members):
                values = positive[:, descendants]
                denominator = values.sum(1, keepdim=True)
                q = torch.where(denominator > 1e-8, values / denominator.clamp_min(1e-8),
                                torch.ones_like(values) / descendants.sum())
                allocation[:, descendants] = torch.relu(yhat[:, node]).unsqueeze(1) * q
            result = result + 0.25 * allocation
        return result


class AlignedHybrid(ReconciliationHead):
    def __init__(self, *args, level_index, nonlinear=False, relaxed=False,
                 hidden=(256, 128), dropout=.1, **kw):
        super().__init__(*args, **kw)
        if not self.nonneg or (relaxed and not nonlinear):
            raise ValueError("unregistered aligned hybrid output structure")
        self.anchor = HYB(*args, definition=MULTILEVEL_HYB,
                          level_index=level_index, **kw)
        self.relaxed = relaxed
        self.coherent_by_construction = not relaxed
        outputs = self.n if relaxed else self.m
        self.mapping = (_mlp(self.n, outputs, list(hidden), dropout) if nonlinear
                        else nn.Linear(self.n, outputs))
        if not nonlinear:
            nn.init.zeros_(self.mapping.weight)
            nn.init.zeros_(self.mapping.bias)

    def bottom(self, yhat):
        if self.relaxed:
            raise RuntimeError("relaxed HYBN produces all-node outputs directly")
        delta = self.mapping(self._norm(yhat)) * self.x_std[self.bottom_idx]
        return self._clip(self.anchor.bottom(yhat) + delta)

    def forward(self, yhat):
        if not self.relaxed:
            return self._reconstruct(self.bottom(yhat))
        delta = self.mapping(self._norm(yhat)) * self.x_std
        return self._clip(self.anchor(yhat) + delta)


class HYBL(AlignedHybrid):
    """Linear all-node correction of the HYB bottom anchor; hard coherence."""

    def __init__(self, *args, level_index, **kwargs):
        super().__init__(*args, level_index=level_index, nonlinear=False, relaxed=False, **kwargs)


class HYBN(AlignedHybrid):
    """MLP all-node correction of HYB; relaxed coherence controlled by loss."""

    def __init__(self, *args, level_index, hidden=(256, 128), dropout=.1, **kwargs):
        super().__init__(*args, level_index=level_index, nonlinear=True, relaxed=True,
                         hidden=hidden, dropout=dropout, **kwargs)
