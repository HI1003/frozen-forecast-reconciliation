"""OPT-001: one HYBN-epsilon rule at epoch and weight-decay levels."""

# Resolve the approved project layout while retaining sealed source-path keys.
from src.support.paths.project import project_file

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_file(ROOT)))

from src.training.model_selection import info_selection_key, selection_key


class TestSelectionKey(unittest.TestCase):
    def test_feasible_always_beats_infeasible(self):
        feasible = selection_key(True, val_l1=100.0, val_coh_nmae=0.5)
        infeasible = selection_key(False, val_l1=1.0, val_coh_nmae=0.5001)
        self.assertLess(feasible, infeasible)

    def test_feasible_candidates_prioritise_validation_l1(self):
        self.assertLess(selection_key(True, 2.0, 0.5),
                        selection_key(True, 3.0, 0.1))

    def test_all_infeasible_prioritise_coherence(self):
        self.assertLess(selection_key(False, 100.0, 0.6),
                        selection_key(False, 1.0, 0.7))

    def test_infeasible_coherence_tie_uses_validation_l1(self):
        self.assertLess(selection_key(False, 2.0, 0.7),
                        selection_key(False, 3.0, 0.7))

    def test_info_key_retains_infeasible_state(self):
        chosen = min([
            {"feasible": False, "best_val": 1.0, "val_coh_nmae": 2.0},
            {"feasible": False, "best_val": 9.0, "val_coh_nmae": 1.0},
        ], key=info_selection_key)
        self.assertFalse(chosen["feasible"])
        self.assertEqual(chosen["val_coh_nmae"], 1.0)

    def test_non_finite_metrics_are_rejected(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(field="val_l1", value=value):
                with self.assertRaisesRegex(ValueError, "must be finite"):
                    selection_key(True, value, 0.0)
            with self.subTest(field="val_coh_nmae", value=value):
                with self.assertRaisesRegex(ValueError, "must be finite"):
                    selection_key(True, 1.0, value)


if __name__ == "__main__":
    unittest.main()
