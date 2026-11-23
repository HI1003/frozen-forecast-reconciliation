"""Regression tests for timestamp, value, hierarchy, and S-matrix identity gates."""

# Resolve the approved project layout while retaining sealed source-path keys.
from src.support.paths.project import project_file

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_file(ROOT)))

from src.data.datasets import load_config, load_dataset
from src.data.datasets import semantic_values_fingerprint
from src.data.hierarchy import load_hierarchy


NODE_NAMES = ["T", "A", "A1", "A2", "b1", "b2"]
PATHS = pd.DataFrame({
    "Top": ["T", "T"],
    "Middle1": ["A", "A"],
    "Middle2": ["A1", "A2"],
    "Bottom": ["b1", "b2"],
})
S = np.array([
    [1.0, 1.0],
    [1.0, 1.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [0.0, 1.0],
])


class ToyDatasetFiles:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data_dir = self.root / "Data" / "Toy"
        project_file(self.data_dir).mkdir(parents=True)
        self.write_hierarchy()
        self.write_values()

    def cleanup(self):
        self.tmp.cleanup()

    def info(self):
        return {
            "num_total_nodes": 6,
            "num_bottom_nodes": 2,
            "bottom_start_idx": 4,
            "num_mid_nodes": 3,
            "num_top_nodes": 1,
            "top_nodes": ["T"],
            "mid_nodes": ["A", "A1", "A2"],
            "bottom_nodes": ["b1", "b2"],
            "node_order": NODE_NAMES,
        }

    def write_hierarchy(self, *, info=None, paths=None, matrix=None):
        (project_file(self.data_dir / "hierarchy_info.json")).write_text(
            json.dumps(self.info() if info is None else info), encoding="utf-8"
        )
        (PATHS if paths is None else paths).to_csv(
            project_file(self.data_dir / "hierarchy.csv"), index=False
        )
        np.savetxt(
            self.data_dir / "sum_matrix.csv", S if matrix is None else matrix,
            delimiter=",", fmt="%.17g",
        )

    def values(self):
        dates = pd.date_range("2020-01-01", periods=8, freq="D")
        b1 = np.arange(1.0, 9.0)
        b2 = np.arange(11.0, 19.0)
        return pd.DataFrame({
            "Date": dates,
            "T": b1 + b2,
            "A": b1 + b2,
            "A1": b1,
            "A2": b2,
            "b1": b1,
            "b2": b2,
        })

    def write_values(self, frame=None):
        (self.values() if frame is None else frame).to_csv(
            project_file(self.data_dir / "values.csv"), index=False
        )

    def config(self, **overrides):
        cfg = {
            "name": "toy",
            "data_dir": "Data/Toy",
            "values_csv": "values.csv",
            "timestamp_col": "Date",
            "freq": "D",
            "fill_gaps": True,
            "train_frac": 0.8,
            "val_frac": 0.1,
            "context_len": 2,
            "predict_len": 1,
            "eval_steps": [1],
            "nonneg": True,
        }
        cfg.update(overrides)
        return cfg


class TestHierarchyIdentity(unittest.TestCase):
    def setUp(self):
        self.files = ToyDatasetFiles()

    def tearDown(self):
        self.files.cleanup()

    def test_valid_toy_and_current_hierarchies_load(self):
        toy = load_hierarchy(self.files.data_dir)
        self.assertEqual((toy.n, toy.m), (6, 2))
        for relative, shape in (
            ("Data/Walmart_4level", (44, 30)),
            ("Data/Gefcom2017F_4level", (158, 141)),
        ):
            current = load_hierarchy(ROOT / relative)
            self.assertEqual((current.n, current.m), shape)

    def test_global_duplicate_name_is_rejected(self):
        info = self.files.info()
        info["bottom_nodes"] = ["A2", "b2"]
        with self.assertRaisesRegex(ValueError, "globally unique"):
            self.files.write_hierarchy(info=info)
            load_hierarchy(self.files.data_dir)

    def test_node_order_mismatch_is_rejected(self):
        info = self.files.info()
        info["node_order"] = list(reversed(info["node_order"]))
        self.files.write_hierarchy(info=info)
        with self.assertRaisesRegex(ValueError, "node_order"):
            load_hierarchy(self.files.data_dir)

    def test_overlapping_middle_levels_are_rejected(self):
        paths = PATHS.copy()
        paths.loc[0, "Middle2"] = "A"
        self.files.write_hierarchy(paths=paths)
        with self.assertRaisesRegex(ValueError, "overlap"):
            load_hierarchy(self.files.data_dir)

    def test_middle2_cannot_have_multiple_middle1_parents(self):
        paths = pd.DataFrame({
            "Top": ["T", "T"],
            "Middle1": ["A", "A_alt"],
            "Middle2": ["A1", "A1"],
            "Bottom": ["b1", "b2"],
        })
        info = self.files.info()
        info["mid_nodes"] = ["A", "A_alt", "A1"]
        info["node_order"] = ["T", "A", "A_alt", "A1", "b1", "b2"]
        matrix = np.array([
            [1.0, 1.0], [1.0, 0.0], [0.0, 1.0],
            [1.0, 1.0], [1.0, 0.0], [0.0, 1.0],
        ])
        self.files.write_hierarchy(info=info, paths=paths, matrix=matrix)
        with self.assertRaisesRegex(ValueError, "exactly one Middle1 parent"):
            load_hierarchy(self.files.data_dir)

    def test_duplicate_bottom_path_is_rejected(self):
        paths = pd.concat([PATHS, PATHS.iloc[[0]]], ignore_index=True)
        self.files.write_hierarchy(paths=paths)
        with self.assertRaisesRegex(ValueError, "exactly one hierarchy path"):
            load_hierarchy(self.files.data_dir)

    def test_nonbinary_and_nonfinite_S_are_rejected(self):
        for value, message in ((0.5, "binary"), (np.inf, "non-finite")):
            matrix = S.copy()
            matrix[2, 0] = value
            self.files.write_hierarchy(matrix=matrix)
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, message):
                load_hierarchy(self.files.data_dir)

    def test_binary_S_row_swap_is_detected_from_paths(self):
        matrix = S.copy()
        matrix[[2, 3]] = matrix[[3, 2]]
        self.files.write_hierarchy(matrix=matrix)
        with self.assertRaisesRegex(ValueError, "disagrees with hierarchy.csv"):
            load_hierarchy(self.files.data_dir)


class TestDatasetTimelineAndValues(unittest.TestCase):
    def setUp(self):
        self.files = ToyDatasetFiles()

    def tearDown(self):
        self.files.cleanup()

    def load(self, **cfg):
        return load_dataset(self.files.config(**cfg), self.files.root)

    def test_valid_regular_data_loads(self):
        data = self.load()
        self.assertEqual(data.T, 8)
        self.assertEqual(data.gaps_filled, 0)
        self.assertEqual(data.excluded_imputed_indices.size, 0)
        self.assertTrue(np.isfinite(data.Y).all())

    def test_semantic_values_fingerprint_normalizes_dtype_layout_and_endian(self):
        frame = self.files.values()
        timestamps = pd.DatetimeIndex(pd.to_datetime(frame["Date"]))
        values = frame[NODE_NAMES].to_numpy(dtype=np.float64)
        expected = semantic_values_fingerprint(timestamps, NODE_NAMES, values)
        normalized = np.asfortranarray(values.astype(">f8"))
        self.assertEqual(
            expected,
            semantic_values_fingerprint(timestamps, NODE_NAMES, normalized),
        )
        changed = values.copy()
        changed[0, 0] += 1.0
        self.assertNotEqual(
            expected, semantic_values_fingerprint(timestamps, NODE_NAMES, changed)
        )
        self.assertNotEqual(
            expected,
            semantic_values_fingerprint(
                timestamps + pd.Timedelta(days=1), NODE_NAMES, values
            ),
        )
        self.assertNotEqual(
            expected,
            semantic_values_fingerprint(
                timestamps, list(reversed(NODE_NAMES)), values
            ),
        )

    def test_configured_semantic_values_fingerprint_is_enforced(self):
        data = self.load()
        self.assertEqual(
            data.values_fingerprint,
            semantic_values_fingerprint(
                data.timestamps, data.hierarchy.node_names,
                self.files.values()[NODE_NAMES].to_numpy(dtype=np.float64),
            ),
        )
        self.load(values_fingerprint=data.values_fingerprint)
        with self.assertRaisesRegex(ValueError, "semantic values fingerprint"):
            self.load(values_fingerprint="0" * 32)

    def test_exact_source_schema_rejects_extra_or_reordered_columns(self):
        frame = self.files.values()
        variants = {
            "extra": frame.assign(unused=1.0),
            "reordered": frame[["Date", *reversed(NODE_NAMES)]],
        }
        for name, variant in variants.items():
            with self.subTest(name=name):
                self.files.write_values(variant)
                with self.assertRaisesRegex(ValueError, "exact source schema"):
                    self.load(require_exact_source_schema=True)

    def test_raw_aggregate_deviation_uses_aggregate_denominator(self):
        frame = self.files.values()
        frame.loc[0, "T"] += 10.0
        self.files.write_values(frame)
        data = self.load()
        raw = frame[NODE_NAMES].to_numpy(dtype=float)
        agg = data.hierarchy.agg_idx
        expected = (
            np.abs(data.Y[:, agg] - raw[:, agg]).sum()
            / np.abs(raw[:, agg]).sum()
        )
        self.assertAlmostEqual(data.raw_aggregate_dev, expected)

    def test_required_source_coherence_rejects_aggregate_drift(self):
        frame = self.files.values()
        frame.loc[0, "T"] += 1.0
        self.files.write_values(frame)
        with self.assertRaisesRegex(ValueError, "source coherence is required"):
            self.load(require_source_coherence=True)

    def test_valid_source_passes_exact_schema_and_coherence_gates(self):
        data = self.load(require_exact_source_schema=True,
                         require_source_coherence=True)
        self.assertEqual(data.raw_aggregate_dev, 0.0)

    def test_declared_imputation_excludes_origin_and_target_uses(self):
        data = self.load(excluded_imputed_timestamps=["2020-01-04"])
        np.testing.assert_array_equal(data.excluded_imputed_indices, [3])
        self.assertEqual(data.origin_exclusion_counts(),
                         {"train": 2, "val": 0, "test": 0})
        np.testing.assert_array_equal(data.origins().train, [1, 4])

    def test_declared_imputation_must_be_an_exact_unique_grid_row(self):
        for values, message in (
            (["not-a-date"], "NaT/invalid"),
            (["2020-02-01"], "not exact rows"),
            (["2020-01-04", "2020-01-04"], "duplicate"),
        ):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, message):
                self.load(excluded_imputed_timestamps=values)

    def test_declared_imputations_must_be_a_list(self):
        with self.assertRaisesRegex(ValueError, "must be a YAML list"):
            self.load(excluded_imputed_timestamps="2020-01-04")

    def test_invalid_nat_is_rejected(self):
        frame = self.files.values()
        frame["Date"] = frame["Date"].astype(str)
        frame.loc[2, "Date"] = "not-a-date"
        self.files.write_values(frame)
        with self.assertRaisesRegex(ValueError, "NaT/invalid"):
            self.load()

    def test_nonmonotonic_timestamps_are_rejected(self):
        frame = self.files.values()
        frame.loc[[2, 3], "Date"] = frame.loc[[3, 2], "Date"].to_numpy()
        self.files.write_values(frame)
        with self.assertRaisesRegex(ValueError, "not strictly increasing"):
            self.load()

    def test_duplicate_timestamps_are_rejected(self):
        frame = self.files.values()
        frame.loc[3, "Date"] = frame.loc[2, "Date"]
        self.files.write_values(frame)
        with self.assertRaisesRegex(ValueError, "duplicate timestamps"):
            self.load()

    def test_off_grid_timestamp_is_rejected_not_silently_dropped(self):
        frame = self.files.values()
        frame.loc[3, "Date"] = pd.Timestamp(frame.loc[3, "Date"]) + pd.Timedelta(minutes=30)
        self.files.write_values(frame)
        with self.assertRaisesRegex(ValueError, "off the configured"):
            self.load()

    def test_missing_grid_row_is_filled_and_grid_is_exact(self):
        frame = self.files.values().drop(index=3)
        self.files.write_values(frame)
        data = self.load(fill_gaps=True)
        self.assertEqual(data.T, 8)
        self.assertEqual(data.gaps_filled, 1)
        pd.testing.assert_index_equal(
            data.timestamps, pd.date_range("2020-01-01", periods=8, freq="D")
        )

    def test_missing_grid_row_hard_fails_when_fill_is_disabled(self):
        self.files.write_values(self.files.values().drop(index=3))
        with self.assertRaisesRegex(ValueError, "fill_gaps is disabled"):
            self.load(fill_gaps=False)

    def test_nan_and_inf_measurements_are_rejected(self):
        for value in (np.nan, np.inf):
            frame = self.files.values()
            frame.loc[2, "b1"] = value
            self.files.write_values(frame)
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "non-finite"):
                self.load()


class TestFormalGefcomSource(unittest.TestCase):
    def test_source_is_strictly_coherent_and_semantically_locked(self):
        cfg = load_config(ROOT / "configs" / "gefcom2017f.yaml")
        self.assertTrue(cfg["require_exact_source_schema"])
        self.assertTrue(cfg["require_source_coherence"])
        data = load_dataset(cfg, ROOT)
        self.assertEqual((data.T, data.hierarchy.n, data.hierarchy.m),
                         (61344, 158, 141))
        self.assertEqual(data.values_fingerprint,
                         "e933f3d0876ac5d5bf654a370d5745e3")
        self.assertEqual(data.values_fingerprint, cfg["values_fingerprint"])
        self.assertEqual(data.raw_aggregate_dev, 0.0)
        self.assertEqual(data.origin_exclusion_counts(),
                         {"train": 150, "val": 25, "test": 0})
        origins = data.origins()
        self.assertEqual(
            (origins.train.size, origins.val.size, origins.test.size),
            (48390, 6086, 6112),
        )



if __name__ == "__main__":
    unittest.main()
