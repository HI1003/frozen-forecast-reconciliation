"""Blocked covariance equals the dense formula without full product allocation."""
import unittest
from unittest.mock import patch
import numpy as np
from src.baseline.classical import shrunk_covariance, _shrunk_covariance_reference, _projection


class CovarianceParity(unittest.TestCase):
    def test_prediction_parity_with_nonnegative_projection(self):
        from src.data.hierarchy import Hierarchy
        from src.baseline.classical import reconcile_classical
        S=np.array([[1,1],[1,1],[1,0],[0,1],[1,0],[0,1]],dtype=float)
        h=Hierarchy(node_names=['t','a','a1','a2','b1','b2'],S=S,
            level_index=dict(top=np.array([0]),middle1=np.array([1]),middle2=np.array([2,3]),bottom=np.array([4,5])))
        rng=np.random.default_rng(8);E=rng.normal(size=(300,6));x=rng.normal(size=(20,6))
        for nonneg in (False,True):
            expected = _projection(x,h,_shrunk_covariance_reference(E),nonneg)
            np.testing.assert_allclose(reconcile_classical('mint_shrink',x,h,E,nonneg),
                expected,rtol=1e-11,atol=1e-11)
    def test_random_and_correlated(self):
        for seed in range(5):
            rng=np.random.default_rng(seed)
            E=rng.normal(size=(503,17));E[:,2:]+=E[:,:1]*3
            for block in (1,4,8,17):
                with self.subTest(seed=seed,block=block):
                    np.testing.assert_allclose(shrunk_covariance(E,block),_shrunk_covariance_reference(E),rtol=2e-13,atol=2e-13)
    def test_scales_constant_columns(self):
        rng=np.random.default_rng(4);E=rng.normal(size=(149,19))*np.geomspace(.001,10000,19);E[:,3]=4
        np.testing.assert_allclose(shrunk_covariance(E),_shrunk_covariance_reference(E),rtol=2e-13,atol=2e-13)
    def test_zero(self):
        E=np.zeros((25,9));np.testing.assert_array_equal(shrunk_covariance(E),_shrunk_covariance_reference(E))
    def test_allocation_bounded(self):
        original=np.einsum;shapes=[]
        def inspect(spec,*args,**kw):
            result=original(spec,*args,**kw);shapes.append(result.shape);return result
        with patch('src.baseline.classical.np.einsum',side_effect=inspect):
            shrunk_covariance(np.random.default_rng(1).normal(size=(60,19)))
        self.assertEqual(len(shapes),5)
        self.assertTrue(all(s[1]<=4 for s in shapes))
    def test_invalid(self):
        for bad in (np.ones((2,9)),np.array([[np.nan]]*3),np.ones(5),np.empty((3,0))):
            with self.assertRaises(ValueError):shrunk_covariance(bad)
        for block in (0,-1,1.5,True):
            with self.assertRaises(ValueError):shrunk_covariance(np.ones((5,3)),block)

if __name__=='__main__':unittest.main()
