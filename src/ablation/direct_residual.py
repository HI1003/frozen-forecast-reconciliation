"""Paper direct-versus-residual mapping comparisons and their initialization.

Use make() to build a registered search arm. Fitting and validation checkpoint
selection remain in src/training/mlp_search_training.py. These controls are not additional
canonical strategies.
"""
import numpy as np
import torch
from torch import nn
from src.strategies.common import ReconciliationHead, _mlp
from src.strategies.top_down import AlignedTD
from src.strategies.bottom_up import BUN
from src.strategies.hybrid import HYBN


class DirectBottom(ReconciliationHead):
    """Bottom-input direct MLP, initialized at training node means."""

    def __init__(self, *args, hidden, dropout, target_mean, **kw):
        super().__init__(*args, **kw)
        self.register_buffer('input_idx', self.bottom_idx.clone())
        # Kept in the state dictionary so original direct-fit weights still load.
        self.register_buffer('top_shares', torch.full((self.m,), 1 / self.m))
        self.mapping = _mlp(self.m, self.m, list(hidden), dropout)
        nn.init.zeros_(self.mapping[-1].weight)
        nn.init.zeros_(self.mapping[-1].bias)
        with torch.no_grad():
            mean = torch.tensor(target_mean, dtype=torch.float32)
            self.mapping[-1].bias.copy_((mean / self.x_std[self.bottom_idx]).clamp_min(.001))

    def bottom(self, yhat):
        mapped = self.mapping(self._norm(yhat)[:, self.input_idx]) * self.x_std[self.bottom_idx]
        return self._clip(mapped)

class DirectTD(AlignedTD):
    def _proportions(self,x): return self.logits(self.features(x))


class DirectAll(ReconciliationHead):
    def __init__(self,*args,hidden,dropout,target_mean,**kw):
        super().__init__(*args,**kw)
        self.coherent_by_construction=False
        self.mapping=_mlp(self.n,self.n,list(hidden),dropout)
        with torch.no_grad():
            self.mapping[-1].bias.copy_((torch.as_tensor(target_mean,dtype=torch.float32)/self.x_std).clamp_min(.001))
    def forward(self,x): return self._clip(self.mapping(self._norm(x))*self.x_std)


def make(arm,data,mean,std,hidden,training_x=None,dropout=.1):
    h=data.hierarchy;args=(h.S,h.bottom_idx,h.top_idx,mean,std)
    if arm=='bottom_mlp_plain':
        return DirectBottom(*args, nonneg=True, hidden=hidden, dropout=dropout,
            target_mean=data.Y[:data.train_end,h.bottom_idx].mean(0))
    if arm=='bottom_mlp_residual': return BUN(*args,nonneg=True,hidden=hidden,dropout=dropout)
    if arm in ('td_features_mlp_plain', 'td_features_mlp_residual'):
        from src.strategies.top_down import td_features_numpy
        v=td_features_numpy(training_x,h.bottom_idx,h.top_idx) if training_x is not None else None
        cls = DirectTD if arm == 'td_features_mlp_plain' else AlignedTD
        return cls(*args,nonneg=True,nonlinear=True,hidden=hidden,dropout=dropout,
            feature_mean=None if v is None else v.mean(0),feature_std=None if v is None else v.std(0))
    if arm=='allnode_mlp_plain': return DirectAll(*args,nonneg=True,hidden=hidden,dropout=dropout,target_mean=data.Y[:data.train_end].mean(0))
    if arm == 'allnode_hyb_residual':
        return HYBN(*args,nonneg=True,level_index=h.level_index,hidden=hidden,dropout=dropout)
    raise ValueError('unknown mapping')
