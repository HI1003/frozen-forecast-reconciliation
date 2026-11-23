"""Numerical estimator parity and complete-family safeguards, no real-data fit."""

# Resolve the approved project layout while retaining sealed source-path keys.
from src.support.paths.project import project_file

import unittest
import numpy as np
from scripts.analysis.analyze_paper_results import hac_test, block_ci, holm_family, METHODS
from src.evaluation.statistics import dm_test, moving_block_indices
from src.evaluation.statistics import holm_adjust

class DependenceTests(unittest.TestCase):
    def test_registered_hac_matches_existing_dm_across_leads(self):
        rng=np.random.default_rng(42)
        white=rng.normal(size=300)
        d=np.convolve(white,[1,.5,.3],mode='same')+.08
        for h in (1,7,14,24):
            original=dm_test(d,np.zeros(len(d)),h)
            new=hac_test(d,h,h-1)
            np.testing.assert_allclose([new['dm_stat'],new['p_raw']],original,atol=1e-12)

    def test_zero_lag_matches_analytic_variance_and_scale_invariance(self):
        d=np.array([1.,-2.,3.,4.,-.5,2.2])
        result=hac_test(d,1,0)
        self.assertAlmostEqual(result['long_run_variance'],np.var(d))
        scaled=hac_test(100*d,1,0)
        np.testing.assert_allclose([result['dm_stat'],result['p_raw']],
                                   [scaled['dm_stat'],scaled['p_raw']],atol=1e-13)

    def test_block_sum_implementation_matches_existing_index_resampling(self):
        d=np.random.default_rng(7).normal(size=59)
        for L in (7,14,28):
            idx=moving_block_indices(len(d),L,2000,42)
            expected=np.quantile(d[idx].mean(axis=1),[.025,.975])
            np.testing.assert_allclose(block_ci(d,L),expected,atol=1e-13)

    def test_positive_serial_dependence_changes_the_long_run_variance(self):
        white=np.random.default_rng(26).normal(size=2000)
        d=np.zeros_like(white)
        for t in range(1,len(d)):d[t]=.9*d[t-1]+white[t]
        self.assertGreater(hac_test(d,1,24)['long_run_variance'],hac_test(d,1,0)['long_run_variance']*5)

    def test_holm_matches_existing_and_refuses_missing_duplicate_or_extra_methods(self):
        p=np.linspace(.001,.5,13)
        np.testing.assert_allclose(holm_family(METHODS,p),holm_adjust(p))
        for methods in (METHODS[:-1],METHODS[:-1]+('BU',),METHODS+('coherent_MLP_control',)):
            with self.assertRaises(ValueError):holm_family(methods,p)

    def test_invalid_or_degenerate_inputs_abort_without_fallback(self):
        for d,q in [(np.ones(10),0),(np.arange(10),10),(np.array([np.nan]*10),1)]:
            with self.assertRaises(ValueError):hac_test(d,1,q)
        with self.assertRaises(ValueError):block_ci(np.ones(10),11)

if __name__=='__main__':unittest.main()
