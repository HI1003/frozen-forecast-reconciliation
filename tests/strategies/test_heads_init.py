"""Check canonical anchors, with numerical tolerance for TD share floors (torch required)."""

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
class TestHeadInitialization(unittest.TestCase):
    def setUp(self):
        from src.strategies import build_strategy
        from src.data.hierarchy import Hierarchy
        self.build_head = build_strategy
        rng = np.random.default_rng(0)
        self.m, extra = 6, 4
        self.n = self.m + 1 + extra
        # 1 top + 3 mid (2 children each) + 6 bottom
        S = np.vstack([
            np.ones((1, self.m)),
            np.ones((1, self.m)),
            np.kron(np.eye(3), np.ones((1, 2))),
            np.eye(self.m),
        ])
        self.S = S
        self.bottom_idx = np.arange(self.n - self.m, self.n)
        self.top_idx = np.array([0])
        yhat = rng.uniform(1.0, 10.0, size=(8, self.n))
        self.x = torch.tensor(yhat, dtype=torch.float32)
        self.x_mean = yhat.mean(axis=0)
        self.x_std = yhat.std(axis=0)
        self.hierarchy = Hierarchy(node_names=[str(i) for i in range(self.n)],S=S,
            level_index=dict(top=np.array([0]),middle1=np.array([1]),
                             middle2=np.arange(2,5),bottom=self.bottom_idx))

    def _make(self, name):
        torch.manual_seed(0)
        return self.build_head(name,self.hierarchy,self.x.numpy()).eval()

    def _assert_matches_direct(self, learned_name, direct_name, atol=1e-4):
        with torch.no_grad():
            out_l = self._make(learned_name)(self.x)
            out_d = self._make(direct_name)(self.x)
        self.assertTrue(torch.allclose(out_l, out_d, atol=atol),
                        f"{learned_name} init != {direct_name}: "
                        f"max diff {(out_l - out_d).abs().max().item():.2e}")

    def test_bul_bun_start_at_bu(self):
        self._assert_matches_direct("BUL", "BU")
        self._assert_matches_direct("BUN", "BU")

    def test_tdl_tdn_start_at_td(self):
        self._assert_matches_direct("TDL", "TD")
        self._assert_matches_direct("TDN", "TD")

    def test_hybl_starts_at_hyb(self):
        self._assert_matches_direct("HYBL", "HYB")

    def test_hybn_starts_at_hyb(self):
        self._assert_matches_direct("HYBN", "HYB")

    def test_td_zero_denominator_fallback(self):
        """All-zero bottom base -> uniform allocation, for TD and learned TD alike."""
        x = self.x.clone()
        x[:, self.bottom_idx] = 0.0
        with torch.no_grad():
            td = self._make("TD")(x)
            tdl = self._make("TDL")(x)
        top = torch.relu(x[:, 0])
        expect = (top / self.m).unsqueeze(1).expand(-1, self.m)
        self.assertTrue(torch.allclose(td[:, self.bottom_idx], expect, atol=1e-5))
        self.assertTrue(torch.allclose(tdl[:, self.bottom_idx], expect, atol=1e-4))

    def test_learned_td_zero_share_floor_is_finite_and_negligible(self):
        """Partial zero shares get the documented finite-logit floor."""
        x = self.x.clone()
        x[:, self.bottom_idx] = torch.tensor(
            [5.0, 0.0, 2.0, 0.0, 1.0, 0.0], dtype=x.dtype
        )
        with torch.no_grad():
            td = self._make("TD")(x)
            for learned_name in ("TDL", "TDN"):
                learned = self._make(learned_name)(x)
                self.assertTrue(torch.isfinite(learned).all())
                self.assertTrue(torch.allclose(
                    learned[:, self.bottom_idx].sum(dim=1),
                    torch.relu(x[:, 0]), atol=1e-5,
                ))
                zero_columns = self.bottom_idx[[1, 3, 5]]
                self.assertTrue(torch.equal(
                    td[:, zero_columns], torch.zeros_like(td[:, zero_columns])
                ))
                self.assertTrue((learned[:, zero_columns] > 0.0).all())
                self.assertTrue(torch.allclose(learned, td, atol=1e-5))


    def test_cpu_state_snapshot_rehydrates_identical_predictions(self):
        from scripts.experiments.experiment_storage import cpu_state_dict as _cpu_state_dict

        source = self._make("BUL")
        with torch.no_grad():
            expected = source(self.x).clone()
        snapshot = _cpu_state_dict(source)
        self.assertTrue(all(t.device.type == "cpu" for t in snapshot.values()))
        with torch.no_grad():
            next(source.parameters()).add_(1.0)
        rebuilt = self._make("BUL")
        rebuilt.load_state_dict(snapshot)
        with torch.no_grad():
            actual = rebuilt(self.x)
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
