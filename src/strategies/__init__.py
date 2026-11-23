"""Final nine strategies: bottom_up (BU/BUL/BUN), top_down (TD/TDL/TDN),
hybrid (HYB/HYBL/HYBN). Use build_strategy / fit_strategy from this package.
Direct-mapping and structural comparisons live in src.ablation.
"""
from .registry import HEAD_NAMES, LEARNABLE_HEADS, build_strategy, fit_strategy

__all__ = ['HEAD_NAMES', 'LEARNABLE_HEADS', 'build_strategy', 'fit_strategy']
