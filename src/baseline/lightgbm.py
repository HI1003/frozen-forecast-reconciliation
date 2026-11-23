"""LightGBM bottom-output baseline from the frozen paper protocol.

Each bottom series has a separate regressor on all-node base forecasts.
The caller selects trees/leaves on validation across the registered cells.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=1)
def protocol():
    path = Path(__file__).resolve().parents[2] / 'configs/experiments/lightgbm_hybn.json'
    return json.loads(path.read_text(encoding='utf-8'))['f03']


def parameters(seed, leaves):
    return dict(protocol()['params'], num_leaves=leaves, seed=seed,
                feature_fraction_seed=seed, bagging_seed=seed, data_random_seed=seed)


def fit(x, y, trees, leaves, seed):
    import lightgbm as lgb
    return lgb.train(parameters(seed, leaves), lgb.Dataset(np.ascontiguousarray(x), label=y),
                     num_boost_round=trees)
