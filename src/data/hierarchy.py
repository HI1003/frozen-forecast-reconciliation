"""Hierarchy loading and validation.

Both datasets ship the same artifacts:
  hierarchy.csv       path table with columns Top,Middle1,Middle2,Bottom
  hierarchy_info.json canonical node order: top_nodes + mid_nodes + bottom_nodes
  sum_matrix.csv      S (n x m), row order == canonical node order, no header

The canonical all-node order is [top, middle1..., middle2..., bottom...] and the
summing matrix S maps bottom-level values to all nodes: y = S @ b.
"""
from __future__ import annotations

# Resolve the approved project layout while retaining sealed source-path keys.
from src.support.paths.project import project_file


import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

LEVEL_NAMES = ["top", "middle1", "middle2", "bottom"]


@dataclass
class Hierarchy:
    node_names: list[str]
    S: np.ndarray                      # (n, m)
    level_index: dict[str, np.ndarray] = field(default_factory=dict)  # level -> node idx array

    @property
    def n(self) -> int:
        return self.S.shape[0]

    @property
    def m(self) -> int:
        return self.S.shape[1]

    @property
    def bottom_idx(self) -> np.ndarray:
        return self.level_index["bottom"]

    @property
    def top_idx(self) -> np.ndarray:
        return self.level_index["top"]

    @property
    def agg_idx(self) -> np.ndarray:
        """Aggregate (non-bottom) node indices — the rows carrying independent
        aggregation constraints; bottom rows of |y - S b| are zero by construction."""
        return np.setdiff1d(np.arange(self.n), self.bottom_idx)


def load_hierarchy(data_dir: str | Path) -> Hierarchy:
    data_dir = Path(data_dir)
    info = json.loads((project_file(data_dir / "hierarchy_info.json")).read_text(encoding="utf-8"))
    paths = pd.read_csv(
        project_file(data_dir / "hierarchy.csv"), dtype=str, keep_default_na=False
    )
    required_columns = ["Top", "Middle1", "Middle2", "Bottom"]
    missing_columns = [c for c in required_columns if c not in paths.columns]
    if missing_columns:
        raise ValueError(f"hierarchy.csv missing columns {missing_columns}")
    if paths.empty:
        raise ValueError("hierarchy.csv contains no bottom paths")
    paths = paths[required_columns].copy()
    for col in required_columns:
        values = paths[col].astype(str)
        if (values.str.strip() != values).any() or (values.str.len() == 0).any():
            raise ValueError(
                f"hierarchy.csv column {col} contains empty or whitespace-padded names"
            )

    def _names(key: str) -> list[str]:
        value = info.get(key)
        if not isinstance(value, list) or not value:
            raise ValueError(f"hierarchy_info.json {key!r} must be a non-empty list")
        names = [str(x) for x in value]
        if any(not x or x.strip() != x for x in names):
            raise ValueError(f"hierarchy_info.json {key!r} has empty/padded names")
        if len(set(names)) != len(names):
            raise ValueError(f"hierarchy_info.json {key!r} contains duplicate names")
        return names

    top_nodes = _names("top_nodes")
    mid_nodes = _names("mid_nodes")
    bottom_nodes = _names("bottom_nodes")
    node_names = top_nodes + mid_nodes + bottom_nodes
    if len(set(node_names)) != len(node_names):
        raise ValueError("top/middle/bottom node names must be globally unique")
    if len(top_nodes) != 1:
        raise ValueError(
            f"this reconciliation implementation requires exactly one top node; "
            f"got {len(top_nodes)}"
        )
    if "node_order" in info:
        declared_order = [str(x) for x in info["node_order"]]
        if declared_order != node_names:
            raise ValueError(
                "hierarchy_info.json node_order does not equal the canonical "
                "top_nodes + mid_nodes + bottom_nodes order"
            )
    count_checks = {
        "num_total_nodes": len(node_names),
        "num_bottom_nodes": len(bottom_nodes),
        "num_mid_nodes": len(mid_nodes),
        "num_top_nodes": len(top_nodes),
        "bottom_start_idx": len(top_nodes) + len(mid_nodes),
    }
    for key, want in count_checks.items():
        if key in info and int(info[key]) != want:
            raise ValueError(f"hierarchy_info.json {key}={info[key]} != expected {want}")

    path_sets = {col: set(paths[col].tolist()) for col in required_columns}
    if path_sets["Top"] != set(top_nodes):
        raise ValueError("hierarchy.csv Top names do not match info.top_nodes")
    if path_sets["Bottom"] != set(bottom_nodes):
        raise ValueError("hierarchy.csv Bottom names do not match info.bottom_nodes")
    if paths["Bottom"].duplicated().any():
        duplicated = paths.loc[paths["Bottom"].duplicated(False), "Bottom"].unique()
        raise ValueError(
            f"each bottom node must have exactly one hierarchy path; duplicates "
            f"{duplicated[:3].tolist()}"
        )
    mid1_set, mid2_set = path_sets["Middle1"], path_sets["Middle2"]
    if mid1_set & mid2_set:
        raise ValueError(
            f"Middle1 and Middle2 names overlap: {sorted(mid1_set & mid2_set)[:3]}"
        )
    parent_counts = paths.groupby("Middle2", sort=False)["Middle1"].nunique()
    cross_parent = parent_counts[parent_counts != 1]
    if not cross_parent.empty:
        examples = [
            f"{name!r}->{sorted(paths.loc[paths['Middle2'] == name, 'Middle1'].unique())}"
            for name in cross_parent.index[:3]
        ]
        raise ValueError(
            "each Middle2 node must belong to exactly one Middle1 parent; "
            f"violations {examples}"
        )
    if mid1_set | mid2_set != set(mid_nodes):
        raise ValueError("hierarchy.csv middle names do not exactly cover info.mid_nodes")

    S = pd.read_csv(project_file(data_dir / "sum_matrix.csv"), header=None).to_numpy(dtype=np.float64)

    if S.shape != (len(node_names), len(bottom_nodes)):
        raise ValueError(
            f"sum_matrix shape {S.shape} does not match node counts "
            f"({len(node_names)} x {len(bottom_nodes)})"
        )
    if not np.isfinite(S).all():
        raise ValueError("sum_matrix contains non-finite entries")
    if not np.isin(S, (0.0, 1.0)).all():
        bad = np.argwhere(~np.isin(S, (0.0, 1.0)))[0]
        raise ValueError(
            f"sum_matrix must be non-negative binary; entry {tuple(bad)}={S[tuple(bad)]}"
        )

    # Rebuild S independently from the path table and canonical name lists.  The
    # older partition checks could not detect two middle rows being swapped.
    expected = np.zeros_like(S)
    node_position = {name: i for i, name in enumerate(node_names)}
    bottom_position = {name: i for i, name in enumerate(bottom_nodes)}
    for top, middle1, middle2, bottom in paths.itertuples(index=False, name=None):
        j = bottom_position[bottom]
        for node in (top, middle1, middle2, bottom):
            expected[node_position[node], j] = 1.0
    if not np.array_equal(S, expected):
        i, j = np.argwhere(S != expected)[0]
        raise ValueError(
            "sum_matrix disagrees with hierarchy.csv at "
            f"row {i} ({node_names[i]!r}), column {j} ({bottom_nodes[j]!r}): "
            f"stored={S[i, j]:g}, expected={expected[i, j]:g}"
        )

    level_of = {
        **{name: "top" for name in top_nodes},
        **{name: "middle1" for name in mid1_set},
        **{name: "middle2" for name in mid2_set},
        **{name: "bottom" for name in bottom_nodes},
    }

    level_index = {
        lvl: np.array([i for i, nm in enumerate(node_names) if level_of[str(nm)] == lvl], dtype=int)
        for lvl in LEVEL_NAMES
    }

    hier = Hierarchy(node_names=node_names, S=S, level_index=level_index)
    _validate(hier)
    return hier


def _validate(h: Hierarchy) -> None:
    if len(h.node_names) != h.n or len(set(h.node_names)) != h.n:
        raise ValueError("hierarchy node_names must be unique and match S rows")
    if not np.isfinite(h.S).all() or not np.isin(h.S, (0.0, 1.0)).all():
        raise ValueError("hierarchy S must be finite, non-negative, and binary")
    missing_levels = [lvl for lvl in LEVEL_NAMES if lvl not in h.level_index]
    if missing_levels:
        raise ValueError(f"hierarchy missing level indices {missing_levels}")
    level_arrays = []
    for lvl in LEVEL_NAMES:
        idx = np.asarray(h.level_index[lvl])
        if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer):
            raise ValueError(f"level {lvl} indices must be a one-dimensional integer array")
        if idx.size and (idx.min() < 0 or idx.max() >= h.n):
            raise ValueError(f"level {lvl} contains an out-of-range node index")
        if np.unique(idx).size != idx.size:
            raise ValueError(f"level {lvl} contains duplicate node indices")
        level_arrays.append(idx.astype(np.int64, copy=False))
    combined = np.concatenate(level_arrays)
    if combined.size != h.n or not np.array_equal(np.sort(combined), np.arange(h.n)):
        raise ValueError("hierarchy levels must be disjoint and cover every node exactly once")
    if h.top_idx.size != 1:
        raise ValueError("hierarchy must contain exactly one top node")
    if h.bottom_idx.size != h.m:
        raise ValueError("bottom level size must equal the number of S columns")
    # top row(s) of S must aggregate every bottom node exactly once
    top_rows = h.S[h.top_idx]
    if not np.array_equal(top_rows, np.ones_like(top_rows)):
        raise ValueError("top row of S is not all ones")
    # bottom block of S must be the identity
    bottom_block = h.S[h.bottom_idx]
    if not np.array_equal(bottom_block, np.eye(h.m)):
        raise ValueError("bottom block of S is not the identity matrix")
    # every bottom node belongs to exactly one parent per middle level
    for lvl in ("middle1", "middle2"):
        rows = h.S[h.level_index[lvl]]
        if rows.size and not np.array_equal(rows.sum(axis=0), np.ones(h.m)):
            raise ValueError(f"level {lvl} rows of S do not partition the bottom nodes")


def coherent_actuals(h: Hierarchy, bottom_values: np.ndarray) -> np.ndarray:
    """Rebuild the all-node matrix from bottom values so actuals are coherent
    by construction: Y = (S @ B^T)^T with B of shape (T, m)."""
    if bottom_values.shape[1] != h.m:
        raise ValueError("bottom_values second dim must equal number of bottom nodes")
    return bottom_values @ h.S.T
