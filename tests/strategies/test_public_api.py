"""Public final-paper factory contracts, independent of cluster orchestration."""
import unittest
from pathlib import Path
import numpy as np
import torch

from src.data.hierarchy import load_hierarchy
from src.strategies import HEAD_NAMES, build_strategy, fit_strategy
from src.strategies.hybrid_allocation import reconcile
from src.training.head_training import HeadTrainConfig

ROOT = Path(__file__).resolve().parents[2]


class PublicStrategies(unittest.TestCase):
    def setUp(self):
        self.hierarchy = load_hierarchy(ROOT/'Data/Walmart_4level')
        self.rng = np.random.default_rng(51)
        self.train = self.rng.uniform(1,20,(16,self.hierarchy.n))
        self.x = self.rng.uniform(1,20,(7,self.hierarchy.n))

    def test_all_nine_are_nonnegative_and_only_hybn_is_relaxed(self):
        for name in HEAD_NAMES:
            with self.subTest(strategy=name):
                model=build_strategy(name,self.hierarchy,self.train,hidden=(8,),dropout=0).eval()
                family = 'bottom_up' if name in ('BU','BUL','BUN') else 'top_down' if name in ('TD','TDL','TDN') else 'hybrid'
                self.assertEqual(type(model).__name__,name)
                self.assertEqual(type(model).__module__,'src.strategies.'+family)
                y=model(torch.tensor(self.x,dtype=torch.float32)).detach().numpy()
                self.assertTrue(np.isfinite(y).all() and (y>=0).all())
                self.assertEqual(model.coherent_by_construction,name!='HYBN')
                if name!='HYBN':
                    np.testing.assert_allclose(y,y[:,self.hierarchy.bottom_idx]@self.hierarchy.S.T,rtol=1e-5,atol=1e-5)

    def test_hybrid_family_initializes_at_four_level_anchor(self):
        expected,_=reconcile(self.x,self.hierarchy.S,self.hierarchy.level_index,'HYB')
        for name in ('HYB','HYBL','HYBN'):
            model=build_strategy(name,self.hierarchy,self.train,hidden=(8,),dropout=0).eval()
            actual=model(torch.tensor(self.x,dtype=torch.float32)).detach().numpy()
            np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-5)

    def test_top_down_never_uses_intermediate_forecasts(self):
        changed=self.x.copy()
        changed[:,np.r_[self.hierarchy.level_index['middle1'],self.hierarchy.level_index['middle2']]]+=1000
        for name in ('TD','TDL','TDN'):
            model=build_strategy(name,self.hierarchy,self.train,hidden=(8,),dropout=0).eval()
            for p in model.parameters():p.data.add_(.01)
            torch.testing.assert_close(model(torch.tensor(self.x,dtype=torch.float32)),model(torch.tensor(changed,dtype=torch.float32)))

    def test_one_epoch_fit_restores_identical_checkpoint_predictions(self):
        targets=self.rng.uniform(1,20,(16,self.hierarchy.m))@self.hierarchy.S.T
        truth=self.rng.uniform(1,20,(7,self.hierarchy.m))@self.hierarchy.S.T
        cfg=HeadTrainConfig(hidden=(8,),dropout=0,max_epochs=1,patience=1,device='cpu')
        model,info=fit_strategy('HYBL',self.hierarchy,self.train,targets,self.x,truth,cfg)
        restored=build_strategy('HYBL',self.hierarchy,self.train,hidden=(8,),dropout=0)
        restored.load_state_dict(model.state_dict(),strict=True)
        model.eval();restored.eval()
        torch.testing.assert_close(model(torch.tensor(self.x,dtype=torch.float32)),restored(torch.tensor(self.x,dtype=torch.float32)))
        self.assertEqual(info['best_epoch'],1)

    def test_unknown_strategy_and_bad_training_shape_fail(self):
        with self.assertRaises(ValueError):build_strategy('unknown',self.hierarchy,self.train)
        with self.assertRaises(ValueError):build_strategy('BUN',self.hierarchy,self.train[:,:-1])


if __name__=='__main__':unittest.main()
