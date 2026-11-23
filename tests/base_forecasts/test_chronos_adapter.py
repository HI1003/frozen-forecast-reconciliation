"""Return-structure normalization tests for the Chronos adapter (torch required).

Fake pipelines emulate the pipeline generations the adapter must handle:
  - unified `inputs=` returning a batched (B, H, Q) tensor tuple (Chronos/Bolt)
  - unified `inputs=` returning a per-task list of (1, H, Q) tensors
    (official Chronos-2 univariate shape)
  - legacy `context=` keyword (older releases -> TypeError fallback)
"""

# Resolve the approved project layout while retaining sealed source-path keys.
from src.support.paths.project import project_file

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_file(ROOT)))

try:
    import torch
    TORCH = True
except ImportError:  # pragma: no cover
    TORCH = False


if TORCH:
    class _TuplePipe:
        def predict_quantiles(self, inputs, prediction_length, quantile_levels):
            B = len(inputs)
            q = torch.full((B, prediction_length, len(quantile_levels)), 7.0)
            return q, torch.zeros(B, prediction_length)

    class _ListPipe:
        def predict_quantiles(self, inputs, prediction_length, quantile_levels):
            return [
                torch.full(
                    (1, prediction_length, len(quantile_levels)),
                    float(i),
                )
                for i, _ in enumerate(inputs)
            ]

    class _LegacyContextPipe:
        def predict_quantiles(self, context, prediction_length, quantile_levels):
            B = len(context)
            q = torch.full((B, prediction_length, len(quantile_levels)), 5.0)
            return q, torch.zeros(B, prediction_length)

    class _BrokenInputsPipe:
        def predict_quantiles(self, inputs, prediction_length, quantile_levels):
            raise TypeError("internal pipeline bug")

    class _WrongShapePipe:
        def predict_quantiles(self, inputs, prediction_length, quantile_levels):
            return [torch.zeros(2, prediction_length, len(quantile_levels))
                    for _ in inputs]


@unittest.skipUnless(TORCH, "torch not installed")
class TestPredictQuantileBatch(unittest.TestCase):
    def setUp(self):
        from src.base_forecasts.chronos2 import _predict_quantile_batch
        self.f = _predict_quantile_batch
        self.contexts = [torch.arange(16, dtype=torch.float32) for _ in range(4)]

    def test_batched_tensor_tuple(self):
        q = self.f(_TuplePipe(), self.contexts, H=5)
        self.assertEqual(q.shape, (4, 5))
        self.assertTrue((q == 7.0).all())

    def test_per_series_list(self):
        q = self.f(_ListPipe(), self.contexts, H=5)
        self.assertEqual(q.shape, (4, 5))
        for i in range(4):
            self.assertTrue((q[i] == float(i)).all())

    def test_legacy_context_fallback(self):
        q = self.f(_LegacyContextPipe(), self.contexts, H=5)
        self.assertEqual(q.shape, (4, 5))
        self.assertTrue((q == 5.0).all())

    def test_single_item_official_shape(self):
        q = self.f(_ListPipe(), self.contexts[:1], H=5)
        self.assertEqual(q.shape, (1, 5))

    def test_internal_type_error_is_not_masked(self):
        with self.assertRaisesRegex(TypeError, "internal pipeline bug"):
            self.f(_BrokenInputsPipe(), self.contexts, H=5)

    def test_non_univariate_item_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected 5"):
            self.f(_WrongShapePipe(), self.contexts, H=5)


if __name__ == "__main__":
    unittest.main()
