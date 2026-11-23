"""Tests for the deterministic, atomic GEFCom aggregate repair utility."""
from __future__ import annotations

# Resolve the approved project layout while retaining sealed source-path keys.
from src.support.paths.project import project_file


import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_file(ROOT / "scripts")))

import scripts.data.rebuild_gefcom_aggregates as repair  # noqa: E402


NODE_NAMES = ["T", "A", "A1", "A2", "b1", "b2"]
S = np.array([
    [1.0, 1.0],
    [1.0, 1.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [0.0, 1.0],
])


class ToyRepairFiles:
    def __init__(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data_dir = self.root / "Data" / "Toy"
        project_file(self.data_dir).mkdir(parents=True)
        self.csv_path = self.data_dir / "values.csv"
        self.config_path = self.root / "toy.yaml"
        self.timestamps = pd.date_range("2020-01-01", periods=8, freq="h")
        self.dst_timestamp = str(self.timestamps[3])
        self._write_hierarchy()
        self._write_values()
        self._write_config()

    def cleanup(self):
        self.temp.cleanup()

    def _write_hierarchy(self):
        info = {
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
        (project_file(self.data_dir / "hierarchy_info.json")).write_text(
            json.dumps(info), encoding="utf-8"
        )
        pd.DataFrame({
            "Top": ["T", "T"],
            "Middle1": ["A", "A"],
            "Middle2": ["A1", "A2"],
            "Bottom": ["b1", "b2"],
        }).to_csv(project_file(self.data_dir / "hierarchy.csv"), index=False)
        np.savetxt(self.data_dir / "sum_matrix.csv", S, delimiter=",", fmt="%.1f")

    def _write_values(self):
        b1 = np.arange(1.0, 9.0)
        b2 = np.arange(11.0, 19.0)
        b1[3] = 4.5
        b2[3] = 14.5
        coherent = np.column_stack([b1, b2]) @ S.T
        frame = pd.DataFrame(coherent, columns=NODE_NAMES)
        # Every aggregate column is wrong, while the canonical bottoms remain valid.
        frame.loc[:, NODE_NAMES[:4]] += np.array([100.0, 50.0, 25.0, 10.0])
        frame.insert(0, "timestamp", self.timestamps.astype(str))
        frame.to_csv(project_file(self.csv_path), index=False)

    def _write_config(self):
        config = {
            "data_dir": str(self.data_dir),
            "values_csv": self.csv_path.name,
            "timestamp_col": "timestamp",
            "freq": "h",
            "excluded_imputed_timestamps": [self.dst_timestamp],
        }
        project_file(self.config_path).write_text(yaml.safe_dump(config), encoding="utf-8")

    def run(self, *, write=False):
        return repair.rebuild_aggregate_columns(
            self.csv_path,
            self.data_dir,
            timestamp_col="timestamp",
            freq="h",
            excluded_imputed_timestamps=[self.dst_timestamp],
            write=write,
        )


class TestRebuildGefcomAggregates(unittest.TestCase):
    def setUp(self):
        self.files = ToyRepairFiles()

    def tearDown(self):
        self.files.cleanup()

    def test_default_cli_is_non_mutating_and_emits_json(self):
        before = project_file(self.files.csv_path).read_bytes()
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            repair.main(["--config", str(self.files.config_path)])
        report = json.loads(stdout.getvalue())

        self.assertEqual(project_file(self.files.csv_path).read_bytes(), before)
        self.assertEqual(report["mode"], "check")
        self.assertFalse(report["replacement_performed"])
        self.assertTrue(report["content_changed"])
        self.assertGreater(report["aggregate_deviation"]["pre"], 0.0)
        self.assertEqual(report["aggregate_deviation"]["post"], 0.0)
        self.assertEqual(report["hierarchy"], {
            "total_nodes": 6, "bottom_nodes": 2, "aggregate_nodes": 4,
        })
        self.assertEqual(report["aggregate_names"], NODE_NAMES[:4])
        self.assertEqual(
            report["semantic_sha256"]["before"]["bottom"],
            report["semantic_sha256"]["after"]["bottom"],
        )
        self.assertEqual(
            report["file_sha256"]["before"],
            report["file_sha256"]["target_after_run"],
        )
        self.assertFalse(list(project_file(self.files.data_dir).glob("*.candidate")))

    def test_write_is_atomic_bottom_preserving_and_idempotent(self):
        original = pd.read_csv(project_file(self.files.csv_path))
        original_bottom = original[["b1", "b2"]].to_numpy(np.float64)
        original_timestamps = original["timestamp"].astype(str).to_numpy()
        calls = []
        real_replace = os.replace

        def recording_replace(source, destination):
            calls.append((Path(source), Path(destination)))
            return real_replace(source, destination)

        with mock.patch.object(repair.os, "replace", side_effect=recording_replace):
            first = self.files.run(write=True)

        self.assertTrue(first["replacement_performed"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0].parent, self.files.csv_path.parent)
        self.assertEqual(calls[0][1], self.files.csv_path)
        repaired = pd.read_csv(project_file(self.files.csv_path))
        np.testing.assert_array_equal(
            repaired["timestamp"].astype(str).to_numpy(), original_timestamps
        )
        np.testing.assert_array_equal(
            repaired[["b1", "b2"]].to_numpy(np.float64), original_bottom
        )
        all_nodes = repaired[NODE_NAMES].to_numpy(np.float64)
        expected = original_bottom @ S.T
        np.testing.assert_array_equal(all_nodes, expected)
        np.testing.assert_array_equal(
            repaired.loc[3, ["b1", "b2"]].to_numpy(np.float64),
            original.loc[3, ["b1", "b2"]].to_numpy(np.float64),
        )

        bytes_after_first = project_file(self.files.csv_path).read_bytes()
        second = self.files.run(write=True)
        self.assertFalse(second["content_changed"])
        self.assertFalse(second["replacement_performed"])
        self.assertTrue(second["validation"]["idempotent_bytes"])
        self.assertEqual(project_file(self.files.csv_path).read_bytes(), bytes_after_first)
        self.assertEqual(
            second["file_sha256"]["before"], second["file_sha256"]["after"]
        )

    def test_rejects_schema_drift_irregular_time_and_negative_values(self):
        valid = pd.read_csv(project_file(self.files.csv_path))
        cases = []

        reordered = valid[["timestamp", "A", "T", "A1", "A2", "b1", "b2"]]
        cases.append((reordered, "schema/order"))

        irregular = valid.drop(index=4).reset_index(drop=True)
        cases.append((irregular, "exact configured"))

        negative = valid.copy()
        negative.loc[0, "b1"] = -1.0
        cases.append((negative, "negative value"))

        for frame, message in cases:
            with self.subTest(message=message):
                frame.to_csv(project_file(self.files.csv_path), index=False)
                before = project_file(self.files.csv_path).read_bytes()
                with self.assertRaisesRegex(ValueError, message):
                    self.files.run(write=True)
                self.assertEqual(project_file(self.files.csv_path).read_bytes(), before)
                self.assertFalse(list(project_file(self.files.data_dir).glob("*.candidate")))


if __name__ == "__main__":
    unittest.main()
