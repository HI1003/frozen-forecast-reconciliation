"""Four-level allocation operations shared by HYB and middle-out baselines.

This NumPy implementation validates the hierarchy and allocates each layer's
forecasts to bottom descendants. The learned strategy classes are in hybrid.py.
Recorded protocol identifiers below are retained for reading existing evidence;
no allocation weight is fitted or chosen from validation/test accuracy here.
"""
from __future__ import annotations

# Resolve the approved project layout while retaining sealed source-path keys.
from src.support.paths.project import project_file


import json
from pathlib import Path

import numpy as np

LEVELS = ("top", "middle1", "middle2", "bottom")
LEGACY_HYB = "bu_td_equal_v1"
MULTILEVEL_HYB = "four_level_equal_v1"
EPS = 1e-8
PROTOCOL = {
    "schema_version": 1, "experiment_id": "multilevel-hyb-v1",
    "source_experiment_id": "main", "datasets": ["walmart", "gefcom2017f"],
    "bases": ["chronos2", "timesfm25"],
    "horizons": {"walmart": [1, 7, 14], "gefcom2017f": [1, 6, 24]},
    "methods": ["middle_out_middle1", "middle_out_middle2", "HYB"],
    "levels": list(LEVELS), "hyb_definition": MULTILEVEL_HYB,
    "hyb_weights": [0.25] * 4,
    "allocation": "positive_bottom_descendant_proportions",
    "fallback": "uniform_bottom_descendants", "epsilon": EPS,
    "nonneg": True, "fit_parameters": False, "select_methods": False,
    "evidence_status": "post_completion_exploratory",
    "unchanged_learnable_heads": ["HYBL", "HYBN"],
    "test_requires_explicit_confirmation": True,
}


def load_protocol(path):
    protocol = json.loads(project_file(Path(path)).read_text(encoding="utf-8"))
    # Do not let changing a weight, grid or definition reuse the same namespace.
    if (protocol != PROTOCOL
            or json.dumps(protocol, sort_keys=True) != json.dumps(PROTOCOL, sort_keys=True)):
        raise ValueError("multilevel-hyb-v1 protocol drift; register a new version")
    return protocol


def task_plan(protocol):
    if protocol != PROTOCOL:
        raise ValueError("unregistered multilevel protocol")
    return [dict(dataset=d, base=b, step=h, method=method)
            for d in protocol["datasets"] for b in protocol["bases"]
            for h in protocol["horizons"][d] for method in protocol["methods"]]


def hierarchy_contract(S, level_index):
    """Require a binary, nested four-level tree, with each level a full cut.

    Row order may be arbitrary, but S's column order must match bottom_idx.
    This prevents silently treating an incomplete/overlapping group as a layer.
    """
    S = np.asarray(S, dtype=np.float64)
    if (S.ndim != 2 or min(S.shape) < 1 or not np.isfinite(S).all()
            or not np.isin(S, [0.0, 1.0]).all() or (S.sum(1) == 0).any()):
        raise ValueError("S must be a finite nonempty binary membership matrix")
    if set(level_index) != set(LEVELS):
        raise ValueError("exactly top, middle1, middle2 and bottom are required")
    indices = {}
    for level in LEVELS:
        raw = np.asarray(level_index[level])
        if (raw.ndim != 1 or raw.size == 0
                or not np.issubdtype(raw.dtype, np.integer)
                or (raw < 0).any() or (raw >= len(S)).any()):
            raise ValueError(f"invalid node indices for {level}")
        indices[level] = raw.astype(np.int64, copy=True)
        if not np.array_equal(S[raw].sum(0), np.ones(S.shape[1])):
            raise ValueError(f"{level} does not partition all bottom descendants")
    flat = np.concatenate(list(indices.values()))
    if len(flat) != len(S) or not np.array_equal(np.sort(flat), np.arange(len(S))):
        raise ValueError("level indices must cover every node exactly once")
    if len(indices["top"]) != 1 or not np.array_equal(S[indices["top"]], np.ones((1, S.shape[1]))):
        raise ValueError("one top node must cover all bottom nodes")
    if not np.array_equal(S[indices["bottom"]], np.eye(S.shape[1])):
        raise ValueError("bottom row order must match S columns")
    for upper, lower in zip(LEVELS[:-1], LEVELS[1:]):
        for child in S[indices[lower]]:
            parents = np.all(child <= S[indices[upper]], axis=1)
            if parents.sum() != 1:
                raise ValueError(f"{lower} must be nested within {upper}")
    return S.copy(), indices


def _input(x, S):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(S) or not np.isfinite(x).all():
        raise ValueError("forecasts must be finite (origins, all_nodes) values")
    return x


def _anchor_bottom(x, S, indices, level):
    bottom = np.maximum(x[:, indices["bottom"]], 0.0)
    if level == "bottom":
        return bottom, 0
    result = np.zeros_like(bottom)
    fallback = 0
    for node in indices[level]:
        descendants = np.flatnonzero(S[node])
        positive = bottom[:, descendants]
        denominator = positive.sum(axis=1, keepdims=True)
        if not np.isfinite(denominator).all():
            raise ValueError("nonfinite descendant total; input magnitude overflowed")
        q = positive / np.maximum(denominator, EPS)
        zero = denominator[:, 0] <= EPS
        q[zero] = 1.0 / len(descendants)
        result[:, descendants] = np.maximum(x[:, node:node+1], 0.0) * q
        fallback += int(zero.sum())
    if not np.isfinite(result).all():
        raise ValueError("nonfinite allocation; input magnitude overflowed")
    return result, fallback


def layer_bottoms(x, S, level_index):
    """Return four coherent-component bottom vectors and fallback counts.

    Intermediate components use anchor-layer AND bottom forecasts, not only
    anchor-layer forecasts. Top equals forecast-proportion TD; bottom equals BU.
    """
    S, indices = hierarchy_contract(S, level_index)
    x = _input(x, S)
    components, fallbacks = {}, {}
    for level in LEVELS:
        components[level], fallbacks[level] = _anchor_bottom(x, S, indices, level)
    return components, fallbacks


def reconcile(x, S, level_index, method):
    """Return original-scale all-node prediction and explicit design metadata."""
    S, indices = hierarchy_contract(S, level_index)
    x = _input(x, S)
    if method in ("middle_out_middle1", "middle_out_middle2"):
        level = method.removeprefix("middle_out_")
        bottom, count = _anchor_bottom(x, S, indices, level)
        meta = dict(definition="middle_out_bottom_proportions_v1", anchor_level=level,
                    fallback_groups=count, input_levels=[level, "bottom"])
    elif method == "HYB":
        components, fallbacks = layer_bottoms(x, S, indices)
        # Sum in bottom space, then aggregate. Do not average raw node forecasts.
        bottom = sum(0.25 * components[level] for level in LEVELS)
        meta = dict(definition=MULTILEVEL_HYB, input_levels=list(LEVELS),
                    weights=[0.25] * 4, fallback_groups=fallbacks)
    else:
        raise ValueError(f"unregistered multilevel method: {method}")
    prediction = bottom @ S.T
    if not np.isfinite(prediction).all():
        raise ValueError("nonfinite reconstructed prediction")
    return prediction, meta
