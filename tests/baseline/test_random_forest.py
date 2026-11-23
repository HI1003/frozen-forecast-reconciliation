"""RF contracts with deterministic doubles only; no sklearn fit or real data."""

# Resolve the approved project layout while retaining sealed source-path keys.
from src.support.paths.project import project_file

import ast
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.baseline import random_forest as rf

ROOT = Path(__file__).resolve().parents[2]


def scratch():
    # Leave task-owned fixtures for inspection; no recursive cleanup/deletion.
    return Path(tempfile.mkdtemp(prefix="topic3-rf-contract-"))


def rows(p):
    return [dict(dataset=d, base=b, horizon=h, seed=s, candidate_id=c,
                 split="val", val_l1=2.0, reference_val_l1=4.0)
            for d in p["datasets"] for b in p["bases"]
            for h in p["horizons"][d] for s in p["seeds"] for c in rf.candidates(p)]


class StubForest:
    """Pickleable fixed-output double, never an actual RF implementation."""
    def __init__(self, **params):
        self.params = params
        self.estimators_ = [SimpleNamespace(tree_=SimpleNamespace(node_count=1))]

    def fit(self, x, y):
        self.value = float(y[0])

    def predict(self, x):
        return np.repeat(self.value, len(x))

    def get_params(self):
        return self.params


class ProtocolAndSelection(unittest.TestCase):
    def setUp(self):
        self.p = rf.load_protocol()

    def test_exact_workload_and_feature_counts(self):
        plan = rf.task_plan(self.p)
        self.assertEqual(len(plan["jobs"]), 36)
        self.assertEqual(plan["search_units"], 324)
        self.assertEqual(plan["selected_units"], 36)
        self.assertEqual(plan["search_forests"], 27702)
        self.assertEqual(plan["selected_refit_forests"], 3078)
        self.assertEqual(plan["search_trees"], 4155300)
        self.assertEqual(plan["refit_trees"], 461700)
        self.assertFalse(plan["execution_authorized"])
        self.assertTrue(plan["test_stage_available"])
        self.assertEqual([rf.feature_count(k, 44) for k in self.p["feature_rules"]], [6, 14, 44])
        self.assertEqual([rf.feature_count(k, 158) for k in self.p["feature_rules"]], [12, 52, 158])

    def test_protocol_drift_rejected(self):
        directory = scratch()
        for key, value in (("test_policy", "evaluate"), ("seeds", [42]),
                           ("parameterization", "residual"), ("n_estimators", 500)):
            p = copy.deepcopy(self.p)
            p[key] = value
            path = directory / f"{key}.json"
            project_file(path).write_text(json.dumps(p), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "drift"):
                rf.load_protocol(path)

    def test_exact_tie_prefers_larger_leaf_and_fewer_features(self):
        result = rf.select_candidates(self.p, rows(self.p))
        self.assertEqual(result["selected"], {"walmart": "sqrt-leaf50", "gefcom2017f": "sqrt-leaf50"})
        self.assertEqual(result["rows_checked"], 324)
        self.assertFalse(result["test_used"])

    def test_selection_uses_equal_cell_ratios_not_pooled_losses(self):
        data = rows(self.p)
        for row in data:
            row["reference_val_l1"] = 1000.0 if row["base"] == "chronos2" else 1.0
            ratio = 2.0
            if row["candidate_id"] == "sqrt-leaf10":
                ratio = 0.1 if row["base"] == "chronos2" else 1.1
            elif row["candidate_id"] == "all-leaf25":
                ratio = 0.7
            row["val_l1"] = row["reference_val_l1"] * ratio
        result = rf.select_candidates(self.p, data)
        self.assertTrue(all(c == "sqrt-leaf10" for c in result["selected"].values()))
        self.assertAlmostEqual(result["scores"]["walmart"]["sqrt-leaf10"], 0.6)
        self.assertEqual(result, rf.select_candidates(self.p, list(reversed(data))))

    def test_missing_duplicate_and_unknown_cells_fail(self):
        full = rows(self.p)
        for bad in (full[:-1], full + [full[0]], [dict(full[0], base="other")] + full[1:]):
            with self.assertRaises(ValueError):
                rf.select_candidates(self.p, bad)

    def test_test_fields_test_split_nonfinite_and_reference_drift_fail(self):
        for change in ({"split": "test"}, {"test_wape": 0.1}, {"val_l1": float("nan")},
                       {"reference_val_l1": 0.0}, {"reference_val_l1": 5.0},
                       {"seed": 42.0}, {"val_l1": True}):
            data = rows(self.p)
            data[0].update(change)
            with self.assertRaises(ValueError):
                rf.select_candidates(self.p, data)

    def test_seeds_stable_distinct_and_cpu_bounds(self):
        a = rf.forest_parameters(self.p, "sqrt-leaf10", 44, 42, 0, 2)
        self.assertEqual(a, rf.forest_parameters(self.p, "sqrt-leaf10", 44, 42, 0, 2))
        self.assertNotEqual(a["random_state"], rf.forest_parameters(self.p, "sqrt-leaf10", 44, 42, 1, 2)["random_state"])
        self.assertEqual(a["criterion"], "squared_error")
        self.assertFalse(a["oob_score"])
        for cpus in (0, -1, True):
            with self.assertRaises(ValueError):
                rf.forest_parameters(self.p, "sqrt-leaf10", 44, 42, 0, cpus)


class MappingContracts(unittest.TestCase):
    def setUp(self):
        self.p = rf.load_protocol()
        self.S = np.array([[1., 1.], [1., 0.], [0., 1.]])
        self.idx = np.array([1, 2])
        self.y = np.array([[1., 3.], [2., 5.], [4., 6.]]) @ self.S.T
        self.x = self.y.copy()
        self.x[0, 0] = -9
        self.xv = np.array([[-100., 2., 9.], [400., -2., 3.]])
        self.created = []

    def factory(self, **kwargs):
        # A stub, not a fitted RandomForestRegressor. Its fixed arithmetic output
        # exists solely to test input routing, units, aggregation and guards.
        obj = SimpleNamespace(params=kwargs, estimators_=[SimpleNamespace(tree_=SimpleNamespace(node_count=3))])
        def fit(x, y):
            obj.x, obj.y = x.copy(), y.copy()
        def predict(x):
            obj.xv = x.copy()
            return np.repeat(obj.y[0], len(x))
        obj.fit, obj.predict = fit, predict
        self.created.append(obj)
        return obj

    def call(self, **kwargs):
        return rf.fit_validation(self.p, "all-leaf10", 42, self.S, self.idx,
                                 self.x, self.y, self.xv, factory=self.factory, **kwargs)

    def test_one_forest_per_bottom_original_scale_direct_and_unclipped_inputs(self):
        saved = []
        pred, stats = self.call(save_model=lambda j, model: saved.append(j))
        self.assertEqual(len(self.created), 2)
        self.assertEqual(saved, [0, 1])
        for j, model in enumerate(self.created):
            np.testing.assert_array_equal(model.x, self.x)
            np.testing.assert_array_equal(model.xv, self.xv)
            np.testing.assert_array_equal(model.y, self.y[:, self.idx[j]])
        np.testing.assert_array_equal(pred, np.tile(self.y[0], (2, 1)))
        np.testing.assert_array_equal(pred, pred[:, self.idx] @ self.S.T)
        self.assertEqual(stats["forests"], 2)
        self.assertEqual(stats["tree_nodes"], 6)

    def test_malformed_hierarchy_and_training_targets_fail_before_factory(self):
        for idx in (np.array([2, 1]), np.array([1., 2.]), np.array([1, 1])):
            with self.assertRaises(ValueError):
                rf.fit_validation(self.p, "all-leaf10", 42, self.S, idx, self.x, self.y, self.xv, factory=self.factory)
        self.y[0, 0] += 1
        with self.assertRaises(ValueError):
            self.call()
        self.assertEqual(self.created, [])

    def test_invalid_prediction_rejected_not_repaired_by_clipping(self):
        original = self.factory
        def bad_factory(**kwargs):
            obj = original(**kwargs)
            obj.predict = lambda x: np.full(len(x), -1.0)
            return obj
        with self.assertRaisesRegex(ValueError, "nonnegativity"):
            rf.fit_validation(self.p, "all-leaf10", 42, self.S, self.idx,
                              self.x, self.y, self.xv, factory=bad_factory)

    def test_validation_reference_uses_raw_base_and_rejects_shape_broadcast(self):
        truth = np.array([[2., 1., 1.]])
        base = np.array([[-4., -2., -2.]])
        row = rf.validation_row("walmart", "chronos2", 1, 42, "all-leaf10", truth, truth, base)
        self.assertEqual(row["reference_val_l1"], 4.0)
        self.assertEqual(row["val_l1"], 0)
        with self.assertRaises(ValueError):
            rf.validation_row("walmart", "chronos2", 1, 42, "all-leaf10", np.vstack([truth, truth]), truth, base)
