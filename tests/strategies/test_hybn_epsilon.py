"""Scale and unit checks for the HYBN-epsilon Lagrangian (torch required)."""

# Resolve the approved project layout while retaining sealed source-path keys.
from src.support.paths.project import project_file

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_file(ROOT)))

try:
    import torch
    TORCH = True
except ImportError:  # pragma: no cover
    TORCH = False


@unittest.skipUnless(TORCH, "torch not installed")
class TestHYBNEpsilonScale(unittest.TestCase):
    def setUp(self):
        from src.training.losses import _epsilon_lagrangian, _normalized_constraint
        self.lagrangian = _epsilon_lagrangian
        self.constraint = _normalized_constraint

    def test_penalty_is_zero_at_constraint_boundary(self):
        pred = torch.tensor(4.0)
        coh = torch.tensor(2.0)
        loss = self.lagrangian(
            pred, coh, lam=7.0, y_scale=100.0, eps_target=0.02
        )
        self.assertAlmostEqual(loss.item(), pred.item())

    def test_raw_lagrangian_scales_with_data_units(self):
        pred = torch.tensor(4.0)
        coh = torch.tensor(3.0)
        loss = self.lagrangian(
            pred, coh, lam=2.0, y_scale=100.0, eps_target=0.02
        )
        factor = 11.0
        scaled = self.lagrangian(
            pred * factor, coh * factor, lam=2.0,
            y_scale=100.0 * factor, eps_target=0.02,
        )
        self.assertAlmostEqual(scaled.item(), loss.item() * factor)

    def test_dual_residual_is_scale_invariant(self):
        residual = self.constraint(3.0, y_scale=100.0, eps_target=0.02)
        scaled = self.constraint(33.0, y_scale=1100.0, eps_target=0.02)
        self.assertAlmostEqual(residual, 0.01)
        self.assertAlmostEqual(residual, scaled)

    def test_epsilon_training_smoke_reports_consistent_metadata(self):
        from src.data.hierarchy import Hierarchy
        from src.training.head_training import HeadTrainConfig, train_head
        from src.evaluation.prediction import apply_head

        S = np.array([[1.,1.],[1.,1.],[1.,1.],[1.,0.],[0.,1.]])
        hier = Hierarchy(
            node_names=["T", "M1", "M2", "b0", "b1"],
            S=S,
            level_index={
                "top": np.array([0]),
                "middle1": np.array([1]),
                "middle2": np.array([2]),
                "bottom": np.array([3, 4]),
            },
        )
        rng = np.random.default_rng(3)
        bottom = rng.uniform(1.0, 5.0, size=(24, 2))
        ytrue = bottom @ S.T
        yhat = ytrue + rng.normal(0.0, 0.2, size=ytrue.shape)
        yhat[:, 0] += 1.0  # ensure the relaxed input starts incoherent
        cfg = HeadTrainConfig(
            hidden=(8, 4), dropout=0.0, batch_size=8, max_epochs=2,
            patience=2, eps_target=0.05, dual_rho=1.0, seed=7,
        )
        head, info = train_head(
            "HYBN", hier, yhat[:16], ytrue[:16], yhat[16:], ytrue[16:],
            nonneg=True, cfg=cfg,
        )
        self.assertIn("lambda", info)
        self.assertIn("selected_lambda", info)
        self.assertGreaterEqual(info["lambda"], 0.0)
        self.assertTrue(np.isfinite(info["val_coh_nmae"]))
        for key in ("selected_train_l1", "selected_val_l1",
                    "best_epoch", "epochs_run"):
            self.assertIn(key, info)
        self.assertTrue(np.isfinite(info["selected_train_l1"]))
        self.assertTrue(np.isfinite(info["selected_val_l1"]))
        self.assertEqual(info["epochs"], info["best_epoch"])
        self.assertLessEqual(info["best_epoch"], info["epochs_run"])
        selected_train = apply_head(head, yhat[:16])
        selected_val = apply_head(head, yhat[16:])
        self.assertAlmostEqual(
            info["selected_train_l1"],
            float(np.abs(selected_train - ytrue[:16]).mean()), places=5,
        )
        self.assertAlmostEqual(
            info["selected_val_l1"],
            float(np.abs(selected_val - ytrue[16:]).mean()), places=5,
        )
        self.assertAlmostEqual(info["best_val"], info["selected_val_l1"], places=5)

        _, direct_info = train_head(
            "BU", hier, yhat[:16], ytrue[:16], yhat[16:], ytrue[16:],
            nonneg=True, cfg=cfg,
        )
        self.assertEqual(direct_info["epochs"], 0)
        self.assertEqual(direct_info["best_epoch"], 0)
        self.assertEqual(direct_info["epochs_run"], 0)
        self.assertTrue(np.isfinite(direct_info["best_val"]))
        self.assertTrue(np.isfinite(direct_info["selected_train_l1"]))
        self.assertTrue(np.isfinite(direct_info["selected_val_l1"]))


if __name__ == "__main__":
    unittest.main()
