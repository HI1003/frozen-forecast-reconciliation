"""Tests for the aggregate-row coherence metric (dilution fix)."""

# Resolve the approved project layout while retaining sealed source-path keys.
from src.support.paths.project import project_file

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_file(ROOT)))

from src.data.hierarchy import Hierarchy
from src.evaluation.metrics import coherence_nmae, coherence_scale


def _hier(m: int) -> Hierarchy:
    """1 top node + m bottom nodes."""
    S = np.vstack([np.ones((1, m)), np.eye(m)])
    names = ["T"] + [f"b{i}" for i in range(m)]
    level_index = {
        "top": np.array([0]),
        "middle1": np.array([], dtype=int),
        "middle2": np.array([], dtype=int),
        "bottom": np.arange(1, m + 1),
    }
    return Hierarchy(node_names=names, S=S, level_index=level_index)


class TestCoherenceNMAE(unittest.TestCase):
    def test_zero_for_coherent(self):
        h = _hier(2)
        ytilde = np.array([[3.0, 1.0, 2.0]])
        self.assertAlmostEqual(coherence_nmae(ytilde, h, 10.0), 0.0)

    def test_known_violation(self):
        h = _hier(2)
        ytilde = np.array([[4.0, 1.0, 2.0]])   # top says 4, bottom sums to 3
        self.assertAlmostEqual(coherence_nmae(ytilde, h, 10.0), 10.0)  # 1/10*100

    def test_no_dilution_by_bottom_count(self):
        """Same top-level violation must yield the same nMAE regardless of how
        many bottom nodes the hierarchy has (the old all-node mean shrank it
        by n/(n-m))."""
        vals = []
        for m in (2, 9, 141):
            h = _hier(m)
            b = np.ones((1, m))
            ytilde = np.concatenate([[[m + 5.0]], b], axis=1)  # violation = 5
            vals.append(coherence_nmae(ytilde, h, 10.0))
        self.assertAlmostEqual(vals[0], 50.0)
        self.assertAlmostEqual(vals[0], vals[1])
        self.assertAlmostEqual(vals[0], vals[2])

    def test_scale_uses_aggregate_nodes_only(self):
        h = _hier(2)
        ytrue_train = np.array([[8.0, 1.0, 2.0], [12.0, 3.0, 4.0]])
        self.assertAlmostEqual(coherence_scale(ytrue_train, h), 10.0)


if __name__ == "__main__":
    unittest.main()
