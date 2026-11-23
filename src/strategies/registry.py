"""Public names and construction of the final paper's nine strategies.

bottom_up.py: BU / BUL / BUN; top_down.py: TD / TDL / TDN;
hybrid.py: HYB / HYBL / HYBN. Only HYBN has relaxed coherence.
Imports of numerical/model libraries are delayed until a model is requested.
"""
from __future__ import annotations

HEAD_NAMES = ["BU", "BUL", "BUN", "TD", "TDL", "TDN", "HYB", "HYBL", "HYBN"]
LEARNABLE_HEADS = ["BUL", "BUN", "TDL", "TDN", "HYBL", "HYBN"]


def build_strategy(name, hierarchy, training_forecasts, *, hidden=(256, 128), dropout=0.1):
    """Construct a strategy with normalization fitted on training forecasts only."""
    import numpy as np
    x = np.asarray(training_forecasts)
    if name not in HEAD_NAMES:
        raise ValueError(f'Unknown strategy {name!r}; choose from {HEAD_NAMES}')
    if x.ndim != 2 or x.shape[1] != hierarchy.n or not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected finite training forecasts with shape (origins, nodes)')
    return _construct(name, hierarchy, x.mean(0), x.std(0), x, hidden, dropout)


def _construct(name, hierarchy, mean, std, training_forecasts, hidden, dropout):
    from .bottom_up import BU, BUL, BUN
    from .top_down import TD, TDL, TDN, td_features_numpy
    from .hybrid import HYB, HYBL, HYBN

    classes = dict(BU=BU, BUL=BUL, BUN=BUN, TD=TD, TDL=TDL, TDN=TDN,
                   HYB=HYB, HYBL=HYBL, HYBN=HYBN)
    options = dict(nonneg=True)
    if name in ('BUN', 'TDL', 'TDN', 'HYBL', 'HYBN'):
        options.update(hidden=hidden, dropout=dropout)
    if name in ('TDL', 'TDN'):
        features = td_features_numpy(training_forecasts, hierarchy.bottom_idx, hierarchy.top_idx)
        options.update(feature_mean=features.mean(0), feature_std=features.std(0))
    if name in ('HYB', 'HYBL', 'HYBN'):
        options['level_index'] = hierarchy.level_index
    return classes[name](hierarchy.S, hierarchy.bottom_idx, hierarchy.top_idx, mean, std, **options)


def fit_strategy(name, hierarchy, x_train, y_train, x_val, y_val, config):
    """Fit one candidate and select its checkpoint using validation only.

    This primitive does not perform the paper's dataset-wide architecture
    selection and never accepts a test block. Callers must freeze selection
    across all registered cells before evaluating test forecasts.
    """
    from src.training.head_training import train_head

    if name not in HEAD_NAMES:
        raise ValueError(f'Unknown strategy {name!r}')
    factory = lambda mean, std: _construct(name, hierarchy, mean, std, x_train,
                                           config.hidden, config.dropout)
    return train_head(name, hierarchy, x_train, y_train, x_val, y_val,
                      True, config, head_factory=factory)
