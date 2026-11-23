"""Algebraic and numerical guards for the seven classical strategies."""

# Resolve the approved project layout while retaining sealed source-path keys.
from src.support.paths.project import project_file

import sys
import unittest
from contextlib import ExitStack
from unittest.mock import patch
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_file(ROOT)))

from src.data.hierarchy import Hierarchy
from src.baseline.classical import CLASSICAL_METHODS, _projection, reconcile_classical, prepare_projection


def _hierarchy():
    S = np.array([
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    return Hierarchy(
        node_names=["T", "A", "A1", "A2", "b1", "b2"],
        S=S,
        level_index={
            "top": np.array([0]),
            "middle1": np.array([1]),
            "middle2": np.array([2, 3]),
            "bottom": np.array([4, 5]),
        },
    )


class TestClassicalProjection(unittest.TestCase):
    def setUp(self):
        self.hier = _hierarchy()
        self.rng = np.random.default_rng(7)

    def test_all_seven_are_identity_on_nonnegative_coherent_input(self):
        bottom = self.rng.uniform(0.1, 20.0, size=(40, self.hier.m))
        coherent = bottom @ self.hier.S.T
        errors = self.rng.normal(size=(100, self.hier.n))
        for method in CLASSICAL_METHODS:
            train = errors if method in {"wls_var", "mint_shrink"} else None
            got = reconcile_classical(
                method, coherent, self.hier, train_errors=train, nonneg=True
            )
            with self.subTest(method=method):
                np.testing.assert_allclose(got, coherent, rtol=1e-11, atol=1e-11)

    def test_whitened_solution_matches_well_conditioned_reference(self):
        yhat = self.rng.normal(size=(12, self.hier.n))
        W = np.diag(np.linspace(0.5, 3.0, self.hier.n))
        Winv = np.linalg.solve(W, np.eye(self.hier.n))
        G = np.linalg.solve(
            self.hier.S.T @ Winv @ self.hier.S,
            self.hier.S.T @ Winv,
        )
        expected = yhat @ G.T
        expected = expected @ self.hier.S.T
        got = _projection(yhat, self.hier, W, nonneg=False)
        np.testing.assert_allclose(got, expected, rtol=1e-11, atol=1e-11)

    def test_nnls_uses_same_whitened_geometry_and_returns_coherent_nonnegative(self):
        yhat = -np.ones((4, self.hier.n))
        W = np.diag(np.linspace(1.0, 2.0, self.hier.n))
        unconstrained = _projection(yhat, self.hier, W, nonneg=False)
        self.assertLess(unconstrained[:, self.hier.bottom_idx].min(), 0.0)
        constrained = _projection(yhat, self.hier, W, nonneg=True)
        self.assertGreaterEqual(constrained.min(), 0.0)
        np.testing.assert_allclose(
            constrained,
            constrained[:, self.hier.bottom_idx] @ self.hier.S.T,
            rtol=0.0, atol=1e-12,
        )

    def test_non_spd_asymmetric_nonfinite_and_ill_conditioned_weights_fail(self):
        yhat = np.ones((2, self.hier.n))
        cases = []
        non_spd = np.eye(self.hier.n)
        non_spd[0, 0] = -1.0
        cases.append((non_spd, "positive definite"))
        asymmetric = np.eye(self.hier.n)
        asymmetric[0, 1] = 0.1
        cases.append((asymmetric, "not symmetric"))
        nonfinite = np.eye(self.hier.n)
        nonfinite[0, 0] = np.inf
        cases.append((nonfinite, "non-finite"))
        ill = np.eye(self.hier.n)
        ill[0, 0] = 1e-13
        cases.append((ill, "ill-conditioned"))
        for W, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                _projection(yhat, self.hier, W, nonneg=False)

    def test_bad_inputs_and_train_errors_fail_before_linear_algebra(self):
        with self.assertRaisesRegex(ValueError, "yhat shape"):
            reconcile_classical("ols", np.ones((3, 5)), self.hier)
        bad_yhat = np.ones((3, self.hier.n))
        bad_yhat[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            reconcile_classical("ols", bad_yhat, self.hier)
        with self.assertRaisesRegex(ValueError, "train_errors shape"):
            reconcile_classical(
                "wls_var", np.ones((3, self.hier.n)), self.hier,
                train_errors=np.ones((10, self.hier.n - 1)),
            )
        errors = np.ones((10, self.hier.n))
        errors[0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "non-finite"):
            reconcile_classical(
                "mint_shrink", np.ones((3, self.hier.n)), self.hier,
                train_errors=errors,
            )

    def test_prepared_projection_reuses_training_statistics_across_batches(self):
        errors = self.rng.normal(size=(100,self.hier.n))
        forecasts = self.rng.normal(size=(17,self.hier.n))
        for method in ('ols','wls_var','wls_struct','mint_shrink'):
            with self.subTest(method=method):
                model = prepare_projection(method,self.hier,errors)
                expected = reconcile_classical(method,forecasts,self.hier,errors)
                with ExitStack() as stack:
                    for name in ('_weight_matrix','shrunk_covariance','np.linalg.cholesky',
                                 'np.linalg.solve','np.linalg.lstsq'):
                        stack.enter_context(patch('src.baseline.classical.'+name,
                            side_effect=AssertionError('preparation repeated during prediction')))
                    actual = np.concatenate([model.predict(forecasts[:5]),model.predict(forecasts[5:])])
                np.testing.assert_allclose(actual,expected,rtol=1e-11,atol=1e-11)

    def test_prepared_projection_validates_new_batches(self):
        model = prepare_projection('ols',self.hier)
        with self.assertRaisesRegex(ValueError,'yhat shape'):
            model.predict(np.ones((3,self.hier.n-1)))
        with self.assertRaisesRegex(ValueError,'non-finite'):
            model.predict(np.full((3,self.hier.n),np.nan))
        self.assertEqual(model.predict(np.empty((0,self.hier.n))).shape,(0,self.hier.n))
        with self.assertRaisesRegex(ValueError,'does not use'):
            prepare_projection('unknown',self.hier)


if __name__ == "__main__":
    unittest.main()
