"""Tests for the RQ-D statistics: Kendall tau-b and the moving-block bootstrap.

Kendall tau-b is checked against hand-built rankings, including the two cases
the acceptance protocol names explicitly (identical -> +1, exactly reversed ->
-1) and the tie handling that makes tau-b the right variant.
"""

# Resolve the approved project layout while retaining sealed source-path keys.
from src.support.paths.project import project_file

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_file(ROOT)))

from src.evaluation.statistics import kendall_tau_b, moving_block_indices, percentile_ci


class TestKendallTauB(unittest.TestCase):
    def test_identical_ranking_is_plus_one(self):
        wape_a = np.array([3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0])
        wape_b = np.array([9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8])
        tau, _ = kendall_tau_b(wape_a, wape_b)
        self.assertAlmostEqual(tau, 1.0, places=12)

    def test_reversed_ranking_is_minus_one(self):
        wape_a = np.arange(9, dtype=float)
        tau, _ = kendall_tau_b(wape_a, wape_a[::-1])
        self.assertAlmostEqual(tau, -1.0, places=12)

    def test_monotone_rescaling_does_not_change_tau(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([2.0, 4.0, 1.0, 5.0, 3.0])
        tau_raw, _ = kendall_tau_b(a, b)
        tau_scaled, _ = kendall_tau_b(a * 100.0 + 7.0, np.exp(b))
        self.assertAlmostEqual(tau_raw, tau_scaled, places=12)

    def test_ties_use_the_b_variant_correction(self):
        # One tied pair in a: tau-b divides by sqrt((n0-n1)(n0-n2)) rather than n0,
        # so a fully concordant-with-ties comparison stays below 1 but above tau-a.
        a = np.array([1.0, 1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0, 4.0])
        tau, _ = kendall_tau_b(a, b)
        n0 = 4 * 3 / 2
        self.assertAlmostEqual(tau, 5.0 / np.sqrt((n0 - 1) * n0), places=12)
        self.assertLess(tau, 1.0)
        self.assertGreater(tau, 5.0 / n0)

    def test_all_tied_is_undefined_not_a_crash(self):
        tau, _ = kendall_tau_b(np.ones(5), np.arange(5.0))
        self.assertTrue(np.isnan(tau))

    def test_single_method_returns_nan(self):
        tau, p = kendall_tau_b(np.array([1.0]), np.array([2.0]))
        self.assertTrue(np.isnan(tau) and np.isnan(p))

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            kendall_tau_b(np.arange(4.0), np.arange(5.0))


class TestPercentileCI(unittest.TestCase):
    """The single place the interval convention lives, so the WAPE CI and the
    Kendall tau CI cannot drift apart."""

    def test_brackets_the_centre_of_the_draws(self):
        rng = np.random.default_rng(0)
        draws = rng.normal(10.0, 1.0, size=4000)
        lo, hi = percentile_ci(draws)
        self.assertLess(lo, 10.0)
        self.assertGreater(hi, 10.0)

    def test_alpha_controls_the_width(self):
        draws = np.linspace(0.0, 1.0, 1001)
        narrow = percentile_ci(draws, alpha=0.5)
        wide = percentile_ci(draws, alpha=0.01)
        self.assertGreater(wide[1] - wide[0], narrow[1] - narrow[0])

    def test_constant_draws_give_a_degenerate_interval(self):
        lo, hi = percentile_ci(np.full(100, 25.0))
        self.assertAlmostEqual(lo, 25.0, places=10)
        self.assertAlmostEqual(hi, 25.0, places=10)

    def test_non_finite_draws_are_dropped(self):
        draws = np.concatenate([np.full(50, 2.0), np.full(50, np.nan)])
        lo, hi = percentile_ci(draws)
        self.assertAlmostEqual(lo, 2.0, places=10)
        self.assertAlmostEqual(hi, 2.0, places=10)

    def test_all_non_finite_yields_nan_not_a_fabricated_interval(self):
        lo, hi = percentile_ci(np.full(10, np.nan))
        self.assertTrue(np.isnan(lo) and np.isnan(hi))


class TestMovingBlockIndices(unittest.TestCase):
    """The shared sampler that makes a PAIRED bootstrap possible: one draw of
    origins is reused across bases and methods, so two rankings are never
    compared on independently resampled data."""

    def test_shape_and_range(self):
        idx = moving_block_indices(100, block_len=14, n_boot=50, seed=1)
        self.assertEqual(idx.shape, (50, 100))
        self.assertGreaterEqual(idx.min(), 0)
        self.assertLess(idx.max(), 100)

    def test_draws_are_contiguous_blocks(self):
        block = 7
        idx = moving_block_indices(70, block_len=block, n_boot=5, seed=2)
        for row in idx:
            for start in range(0, 70, block):
                chunk = row[start : start + block]
                self.assertTrue((np.diff(chunk) == 1).all())

    def test_same_seed_gives_the_same_draw_for_paired_use(self):
        a = moving_block_indices(200, block_len=24, n_boot=10, seed=5)
        b = moving_block_indices(200, block_len=24, n_boot=10, seed=5)
        np.testing.assert_array_equal(a, b)

    def test_too_few_observations_raises(self):
        with self.assertRaises(ValueError):
            moving_block_indices(10, block_len=24, n_boot=5)


if __name__ == "__main__":
    unittest.main()
