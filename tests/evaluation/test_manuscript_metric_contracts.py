"""Execute the actual NumPy metric bodies on synthetic data, without SciPy/GPU.

AST extraction avoids importing the unrelated SciPy DM implementation. These
tests check metric arithmetic and edge cases, not model training, tensor
execution, saved-test reevaluation or whole-environment reproduction.
"""

# Resolve the approved project layout while retaining sealed source-path keys.
from src.support.paths.project import project_file

import ast
from pathlib import Path
import unittest

import numpy as np

from src.data.hierarchy import Hierarchy, LEVEL_NAMES


ROOT = Path(__file__).resolve().parents[2]
FUNCTIONS = {
    "wape", "seasonal_period", "mase_scale", "level_metrics",
    "coherence_scale", "coherence_nmae",
}
tree = ast.parse((project_file(ROOT / "src/metrics.py")).read_text(encoding="utf-8"))
nodes = [node for node in tree.body if (
    isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS
) or (
    isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id in {"EPS", "SEASONAL_PERIODS"}
        for target in node.targets
    )
)]
namespace = {"np": np, "Hierarchy": Hierarchy, "LEVEL_NAMES": LEVEL_NAMES}
exec(compile(ast.Module(body=nodes, type_ignores=[]), "src/metrics.py", "exec"), namespace)


class ManuscriptMetricContracts(unittest.TestCase):
    def test_registered_seasons_are_7_and_24(self):
        self.assertEqual(namespace["seasonal_period"]("D"), 7)
        self.assertEqual(namespace["seasonal_period"]("h"), 24)
        with self.assertRaises(ValueError):
            namespace["seasonal_period"]("unregistered")

    def test_seasonal_scale_uses_training_rows_only(self):
        values = np.arange(12, dtype=float)[:, None] ** 2
        expected = np.array([np.mean([9., 15., 21., 27., 33.])])
        actual = namespace["mase_scale"](values, 8, 3)
        np.testing.assert_array_equal(actual, expected)
        values[8:] = 1e9
        np.testing.assert_array_equal(namespace["mase_scale"](values, 8, 3), expected)

    def test_invalid_season_or_short_training_fails(self):
        for train_end, season in [(8, 0), (3, 3), (2, 3)]:
            with self.assertRaises(ValueError):
                namespace["mase_scale"](np.ones((12, 2)), train_end, season)

    def test_wape_is_percent_ratio_of_sums_not_mean_percentage(self):
        actual = namespace["wape"](np.array([[1., 3.]]), np.array([[2., 100.]]))
        self.assertAlmostEqual(actual, 100 * 4 / 102)

    def test_smase_excludes_scales_at_or_below_1e_minus_8_threshold(self):
        hierarchy = Hierarchy(
            node_names=["a", "b", "c", "d"], S=np.eye(4),
            level_index={name: np.array([i]) for i, name in enumerate(LEVEL_NAMES)},
        )
        truth = np.full((2, 4), 10.)
        actual = namespace["level_metrics"](
            truth + 2, truth, hierarchy, np.array([2., 1e-8, 5e-9, 4.]),
        )["all"]
        self.assertEqual(namespace["EPS"], 1e-8)
        self.assertEqual(actual["n_mase_excluded"], 2)
        self.assertAlmostEqual(actual["mase"], (1 + .5) / 2)

    def test_coherence_uses_aggregate_rows_and_training_scale(self):
        hierarchy = Hierarchy(
            node_names=["total", "a", "b"],
            S=np.array([[1., 1.], [1., 0.], [0., 1.]]),
            level_index={"top": np.array([0]), "middle1": np.array([], dtype=int),
                         "middle2": np.array([], dtype=int), "bottom": np.array([1, 2])},
        )
        train = np.array([[10., 4., 6.], [20., 8., 12.]])
        scale = namespace["coherence_scale"](train, hierarchy)
        self.assertEqual(scale, 15.)
        prediction = np.array([[12., 4., 6.], [22., 8., 12.]])
        self.assertAlmostEqual(namespace["coherence_nmae"](prediction, hierarchy, scale),
                               100 * 2 / 15)
        coherent = prediction[:, hierarchy.bottom_idx] @ hierarchy.S.T
        self.assertEqual(namespace["coherence_nmae"](coherent, hierarchy, scale), 0.)


if __name__ == "__main__":
    unittest.main()
