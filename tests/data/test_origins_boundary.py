"""Boundary tests for origin enumeration under the strict 80/10/10 protocol.

Semantics under test (src/datasets.py): origin t = last observed row index; the
context is rows t+1-L .. t and the targets are rows t+1 .. t+H, so the last
target index is t + H.  Invariants: every origin has a full context (t >= L-1),
no target crosses a split boundary, and the three segments stay ordered.
"""

# Resolve the approved project layout while retaining sealed source-path keys.
from src.support.paths.project import project_file

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_file(ROOT)))

from src.data.datasets import SplitOrigins, _enumerate_origins, _exclude_imputed_origins, assert_imputed_exclusion_invariants, assert_split_invariants


GEFCOM_IMPUTED_INDICES = np.array([
    2209, 10945, 19177, 27913, 36649, 45553, 54289,
])


def _origins(T, train_frac, val_frac, H, L, strides=(1, 1, 1), excluded=()):
    """Mirror HTSData.origins() without loading a dataset from disk."""
    train_end = int(T * train_frac)
    val_end = int(T * (train_frac + val_frac))
    candidate = SplitOrigins(
        train=_enumerate_origins(0, train_end, H, strides[0], min_origin=L - 1),
        val=_enumerate_origins(train_end, val_end, H, strides[1], min_origin=L - 1),
        test=_enumerate_origins(val_end, T, H, strides[2], min_origin=L - 1),
    )
    excluded = np.asarray(excluded, dtype=int)
    o = SplitOrigins(**{
        seg: _exclude_imputed_origins(getattr(candidate, seg), excluded, H)
        for seg in ("train", "val", "test")
    })
    assert_split_invariants(o, train_end, val_end, T, H, L)
    assert_imputed_exclusion_invariants(o, excluded, H)
    return o, train_end, val_end


class TestEnumerateOrigins(unittest.TestCase):
    def test_targets_stay_inside_segment(self):
        H = 4
        o = _enumerate_origins(10, 30, H, 1)
        self.assertEqual(o.min(), 9)            # first target = 10 = seg_start
        self.assertEqual(o.max(), 25)           # last target = 29 = seg_end - 1
        self.assertTrue(((o + 1) >= 10).all())
        self.assertTrue(((o + H) <= 29).all())

    def test_empty_when_segment_too_short(self):
        self.assertEqual(_enumerate_origins(10, 12, 4, 1).size, 0)

    def test_stride(self):
        o = _enumerate_origins(10, 40, 5, 3)
        self.assertTrue((np.diff(o) == 3).all())

    def test_min_origin_enforces_full_context(self):
        o = _enumerate_origins(0, 100, 5, 1, min_origin=29)
        self.assertEqual(o.min(), 29)           # context rows 0..29 available
        self.assertEqual(o.max(), 94)

    def test_min_origin_only_binds_when_it_is_later(self):
        # For val/test the segment start already exceeds the context requirement.
        o = _enumerate_origins(500, 600, 5, 1, min_origin=29)
        self.assertEqual(o.min(), 499)


class TestSplitInvariants(unittest.TestCase):
    def test_walmart_geometry_exact_counts(self):
        o, train_end, val_end = _origins(1941, 0.8, 0.1, H=14, L=365)
        self.assertEqual((train_end, val_end), (1552, 1746))
        self.assertEqual((o.train.size, o.val.size, o.test.size), (1174, 181, 182))
        self.assertEqual(o.train.min(), 364)          # context_len - 1
        self.assertEqual(o.train.max() + 14, 1551)    # last train target < train_end

    def test_gefcom_geometry_exact_counts(self):
        o, train_end, val_end = _origins(61344, 0.8, 0.1, H=24, L=512,
                                         excluded=GEFCOM_IMPUTED_INDICES)
        self.assertEqual((train_end, val_end), (49075, 55209))
        self.assertEqual((o.train.size, o.val.size, o.test.size), (48390, 6086, 6112))
        self.assertEqual(o.train.min(), 511)
        self.assertEqual(o.train.max(), 49050)

    def test_gefcom_dense_counts_before_imputation_exclusions(self):
        dense, _, _ = _origins(61344, 0.8, 0.1, H=24, L=512)
        self.assertEqual((dense.train.size, dense.val.size, dense.test.size),
                         (48540, 6111, 6112))
        self.assertTrue((np.diff(dense.train) == 1).all())

    def test_imputed_origin_and_target_window_are_excluded_exactly(self):
        origins = np.arange(80, 121)
        kept = _exclude_imputed_origins(origins, np.array([100]), predict_len=4)
        for t in range(96, 101):
            self.assertNotIn(t, kept)
        self.assertIn(95, kept)
        self.assertIn(101, kept)  # imputation is now older, fully observed context

    def test_every_origin_has_a_full_context(self):
        for L in (60, 365, 512):
            o, _, _ = _origins(3000, 0.8, 0.1, H=14, L=L)
            for arr in (o.train, o.val, o.test):
                self.assertGreaterEqual(int(arr.min()) - L + 1, 0)

    def test_targets_never_cross_a_boundary(self):
        H, L = 14, 365
        o, train_end, val_end = _origins(1941, 0.8, 0.1, H=H, L=L)
        self.assertLess(o.train.max() + H, train_end)
        self.assertGreaterEqual(o.val.min() + 1, train_end)
        self.assertLess(o.val.max() + H, val_end)
        self.assertGreaterEqual(o.test.min() + 1, val_end)
        self.assertLess(o.test.max() + H, 1941)

    def test_segments_are_ordered_and_disjoint(self):
        o, _, _ = _origins(1941, 0.8, 0.1, H=14, L=365)
        self.assertLess(o.train.max(), o.val.min())
        self.assertLess(o.val.max(), o.test.min())
        self.assertEqual(np.intersect1d(o.train, o.val).size, 0)
        self.assertEqual(np.intersect1d(o.val, o.test).size, 0)

    def test_context_may_reach_back_across_a_boundary(self):
        # The first validation origin legitimately reads training rows: those
        # observations have already happened when the forecast is made.
        H, L = 14, 365
        o, train_end, _ = _origins(1941, 0.8, 0.1, H=H, L=L)
        self.assertLess(o.val.min() - L + 1, train_end)
        self.assertEqual(o.val.min() + 1, train_end)

    def test_invariant_violation_is_detected(self):
        bad = SplitOrigins(train=np.array([100, 200]), val=np.array([150]),
                           test=np.array([300]))
        with self.assertRaises(AssertionError):
            assert_split_invariants(bad, 250, 280, 400, predict_len=14, context_len=10)

    def test_empty_segment_is_rejected(self):
        # A context longer than the whole training segment leaves no train origin.
        with self.assertRaises(AssertionError):
            _origins(600, 0.8, 0.1, H=14, L=600)


if __name__ == "__main__":
    unittest.main()
