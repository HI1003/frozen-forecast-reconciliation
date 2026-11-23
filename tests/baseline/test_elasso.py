"""Small synthetic contracts only: no project data, caches or Slurm execution."""

# Resolve the approved project layout while retaining sealed source-path keys.
from src.support.paths.project import project_file

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from scipy.optimize import minimize
from src.baseline.elasso_adapted import prepare, solve, predict, choose_alpha


class ElassoTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(19)
        self.S=np.array([[1,1],[1,0],[0,1.]])
        self.X=rng.normal(size=(90,3))
        self.Y=(self.X@rng.normal(size=(3,2))+.1*rng.normal(size=(90,2)))@self.S.T
        self.p=prepare(self.X,self.Y,self.S)

    def test_zero_matches_direct_least_squares(self):
        G,stats=solve(self.p,0)
        expected=np.linalg.lstsq(self.X,self.Y,rcond=None)[0]
        np.testing.assert_allclose(self.X@G.T@self.S.T,self.X@expected,atol=1e-10)
        self.assertTrue(stats['converged'])

    def test_lambda_max_zero_solution(self):
        G,stats=solve(self.p,1.)
        np.testing.assert_allclose(G,0,atol=1e-12)
        self.assertLess(stats['dual_gap'],1e-8)

    def test_independent_epigraph_optimizer(self):
        alpha=.06; p=self.p; lam=alpha*p['lmax']; m,n=p['C'].shape
        G,stats=solve(p,alpha,gap_rtol=1e-9,gap_atol=1e-11)
        def objective(v):
            g=v[:m*n].reshape(m,n)
            residual=self.Y-self.X@g.T@self.S.T
            return .5*np.sum(residual**2)/(len(self.X)*p['scale2'])+lam*p['weights']@v[m*n:]
        def feasible(v):
            return v[m*n:]-np.linalg.norm(v[:m*n].reshape(m,n),axis=0)
        # Different algorithm, explicit data loss, no warm start from our solution.
        v0=np.r_[np.zeros(m*n),np.ones(n)]
        opt=minimize(objective,v0,method='SLSQP',constraints={'type':'ineq','fun':feasible},
                     options={'ftol':1e-11,'maxiter':2000})
        self.assertTrue(opt.success,opt.message)
        self.assertAlmostEqual(stats['objective']/p['scale2'],opt.fun,places=7)
        np.testing.assert_allclose(self.X@G.T@self.S.T,self.X@opt.x[:m*n].reshape(m,n).T@self.S.T,atol=2e-4)

    def test_scale_equivariance_and_coherence(self):
        G,_=solve(self.p,.1)
        scaled=prepare(self.X*100,self.Y*100,self.S)
        G2,_=solve(scaled,.1)
        np.testing.assert_allclose(G,G2,atol=1e-7)
        pred,_=predict(self.X,G,self.S)
        self.assertTrue((pred>=0).all())
        np.testing.assert_allclose(pred[:,0],pred[:,1]+pred[:,2],atol=1e-12)

    def test_nonconvergence_is_fatal(self):
        with self.assertRaises(RuntimeError):
            solve(self.p,.01,max_iter=1,gap_rtol=1e-14,gap_atol=1e-14)

    def test_validation_selection_and_coverage(self):
        rows=[dict(cell=c,alpha_index=i,val_mae=v,base_val_mae=2.)
              for c in ['a','b'] for i,v in enumerate([2.,1.])]
        winner,_=choose_alpha(rows,[1.,0.],['a','b'])
        self.assertEqual(winner['alpha'],0)
        with self.assertRaises(ValueError): choose_alpha(rows[:-1],[1.,0.],['a','b'])
        with self.assertRaises(ValueError): choose_alpha(rows+rows[:1],[1.,0.],['a','b'])
