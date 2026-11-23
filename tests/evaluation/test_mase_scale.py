"""Tests for the seasonal-naive scale behind MASE."""

# Resolve the approved project layout while retaining sealed source-path keys.
from src.support.paths.project import project_file

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_file(ROOT)))

from src.data.hierarchy import Hierarchy
from src.evaluation.metrics import level_metrics, seasonal_period, mase_scale


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


class TestSeasonalPeriod(unittest.TestCase):
    def test_registered_frequencies(self):
        self.assertEqual(seasonal_period("D"), 7)
        self.assertEqual(seasonal_period("h"), 24)
        self.assertEqual(seasonal_period("H"), 24)

    def test_unregistered_frequency_fails_instead_of_defaulting(self):
        """A silent lag-1 fallback would publish an unregistered non-seasonal MASE."""
        with self.assertRaisesRegex(ValueError, "no seasonal period registered"):
            seasonal_period("W")


class TestMASEScale(unittest.TestCase):
    def test_seasonal_series_has_near_zero_scale(self):
        """A perfectly seasonal series is exactly predicted by seasonal naive."""
        Y = np.tile(np.array([[1.0], [5.0], [9.0]]), (4, 1))
        scale = mase_scale(Y, train_end=12, season=3)
        self.assertAlmostEqual(float(scale[0]), 0.0)

    def test_season_one_reproduces_the_lag_one_scale(self):
        Y = np.array([[0.0], [2.0], [5.0], [9.0]])
        lag1 = np.abs(np.diff(Y[:4], axis=0)).mean(axis=0)
        np.testing.assert_allclose(mase_scale(Y, 4, 1), lag1)

    def test_only_training_rows_enter_the_scale(self):
        """Test-segment values must not leak into the denominator."""
        Y = np.array([[0.0], [1.0], [2.0], [1000.0]])
        self.assertAlmostEqual(float(mase_scale(Y, train_end=3, season=1)[0]), 1.0)

    def test_known_seasonal_scale(self):
        Y = np.array([[0.0], [0.0], [3.0], [7.0]])   # lag-2 diffs: 3, 7
        self.assertAlmostEqual(float(mase_scale(Y, 4, 2)[0]), 5.0)

    def test_season_at_least_one(self):
        Y = np.zeros((5, 1))
        with self.assertRaisesRegex(ValueError, "seasonal period must be >= 1"):
            mase_scale(Y, 5, 0)

    def test_training_segment_shorter_than_season_fails(self):
        Y = np.zeros((30, 1))
        with self.assertRaisesRegex(ValueError, "cannot support a lag-24"):
            mase_scale(Y, train_end=24, season=24)


class TestLevelMetricsUseSeasonalScale(unittest.TestCase):
    def test_mase_divides_by_the_seasonal_scale(self):
        h = _hier(2)
        ytrue = np.array([[3.0, 1.0, 2.0]])
        yhat = np.array([[5.0, 3.0, 4.0]])      # absolute error 2 on every node
        out = level_metrics(yhat, ytrue, h, np.array([4.0, 1.0, 2.0]))
        self.assertAlmostEqual(out["top"]["mase"], 0.5)      # 2/4
        self.assertAlmostEqual(out["bottom"]["mase"], 1.5)   # mean(2/1, 2/2)
        self.assertEqual(out["all"]["n_mase_excluded"], 0)

    def test_zero_scale_nodes_are_excluded_and_counted(self):
        h = _hier(2)
        ytrue = np.array([[3.0, 1.0, 2.0]])
        yhat = np.array([[5.0, 3.0, 4.0]])
        out = level_metrics(yhat, ytrue, h, np.array([4.0, 0.0, 2.0]))
        self.assertEqual(out["bottom"]["n_mase_excluded"], 1)
        self.assertAlmostEqual(out["bottom"]["mase"], 1.0)   # only the 2/2 node


if __name__ == "__main__":
    unittest.main()
