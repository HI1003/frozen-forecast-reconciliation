"""Contract tests for the TimesFM adapter (numpy only — no weights needed).

The adapter's job is to guarantee that TimesFM and Chronos hand the pipeline
the SAME point statistic.  TimesFM returns `(point, quantiles)` and the point
forecast is internally `quantiles[..., 5]`, the 0.5 quantile.  These fakes
check that the adapter enforces that identity and the output geometry instead
of trusting an undocumented internal convention.
"""

# Resolve the approved project layout while retaining sealed source-path keys.
from src.support.paths.project import project_file

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_file(ROOT)))

from src.base_forecasts.timesfm25 import Q50_INDEX, _forecast_batch, compiled_max_context, context_padding_info


class _OfficialModel:
    """Mimics timesfm 2.0.2: point forecast IS the q50 channel."""

    def __init__(self, n_quantiles=10):
        self.n_quantiles = n_quantiles
        self.seen = None

    def forecast(self, horizon, inputs):
        # The real API pads its `inputs` argument in place up to the global
        # batch size and then slices the padding back off the output.
        self.seen = n = len(inputs)
        inputs += [np.zeros(3)]
        b = len(inputs)
        quant = np.arange(b * horizon * self.n_quantiles, dtype=np.float64)
        quant = quant.reshape(b, horizon, self.n_quantiles)[:n]
        return quant[..., Q50_INDEX], quant


class _MeanPointModel(_OfficialModel):
    """A hypothetical release that returns the mean channel as the point."""

    def forecast(self, horizon, inputs):
        point, quant = super().forecast(horizon, inputs)
        return quant[..., 0], quant


class _TooFewQuantiles(_OfficialModel):
    def __init__(self):
        super().__init__(n_quantiles=3)

    def forecast(self, horizon, inputs):
        b = len(inputs)
        quant = np.zeros((b, horizon, self.n_quantiles))
        return quant[..., 0], quant


class _WrongHorizon(_OfficialModel):
    def forecast(self, horizon, inputs):
        b = len(inputs)
        quant = np.zeros((b, horizon + 1, self.n_quantiles))
        return quant[..., Q50_INDEX], quant


class TestForecastBatch(unittest.TestCase):
    def setUp(self):
        self.contexts = [np.arange(16, dtype=np.float32) for _ in range(4)]

    def test_returns_point_and_quantiles_with_expected_shapes(self):
        point, quant = _forecast_batch(_OfficialModel(), self.contexts, H=5)
        self.assertEqual(point.shape, (4, 5))
        self.assertEqual(quant.shape, (4, 5, 10))

    def test_point_equals_q50_channel(self):
        point, quant = _forecast_batch(_OfficialModel(), self.contexts, H=5)
        np.testing.assert_array_equal(point, quant[:, :, Q50_INDEX])

    def test_caller_context_list_is_not_mutated(self):
        model = _OfficialModel()
        _forecast_batch(model, self.contexts, H=5)
        self.assertEqual(len(self.contexts), 4)   # the model padded its own copy
        self.assertEqual(model.seen, 4)

    def test_point_that_is_not_q50_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not the q50 channel"):
            _forecast_batch(_MeanPointModel(), self.contexts, H=5)

    def test_missing_q50_channel_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "0.5 quantile"):
            _forecast_batch(_TooFewQuantiles(), self.contexts, H=5)

    def test_wrong_horizon_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            _forecast_batch(_WrongHorizon(), self.contexts, H=5)

    def test_single_item_batch(self):
        point, quant = _forecast_batch(_OfficialModel(), self.contexts[:1], H=5)
        self.assertEqual(point.shape, (1, 5))


class TestCompiledMaxContext(unittest.TestCase):
    def test_rounds_up_to_the_input_patch_size(self):
        self.assertEqual(compiled_max_context(365), 384)   # Walmart
        self.assertEqual(compiled_max_context(512), 512)   # GEFCom, already exact
        self.assertEqual(compiled_max_context(1), 32)

    def test_never_truncates_the_requested_context(self):
        for L in (60, 168, 365, 512, 1000):
            self.assertGreaterEqual(compiled_max_context(L), L)

    def test_padding_metadata_records_pre_normalization_effect(self):
        walmart = context_padding_info(365)
        self.assertEqual(walmart["left_zero_padding"], 19)
        self.assertTrue(walmart["decode_masked"])
        self.assertTrue(walmart["normalization_includes_padding"])
        gefcom = context_padding_info(512)
        self.assertEqual(gefcom["left_zero_padding"], 0)
        self.assertFalse(gefcom["decode_masked"])
        self.assertFalse(gefcom["normalization_includes_padding"])


if __name__ == "__main__":
    unittest.main()
