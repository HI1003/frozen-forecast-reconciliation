"""Protocol coverage, immutable IO and synthetic end-to-end paper experiments."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import TestCase, mock

import numpy as np
import torch
from scripts.experiments import paper_protocol as P, experiment_storage as A
from scripts.experiments import run_paper_experiments as R
from src.data.hierarchy import Hierarchy


def score(u,**kw):
    info=dict(best_val=1.,selected_val_l1=1.,best_epoch=2,parameters=100,
              val_coherence_fraction=.01,feasible=True,val_coh_nmae=1.)
    info.update(kw)
    return dict(unit=u,info=info,base_val_l1=2.,test_used=False)


class PaperProtocol(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid=P.core_grid()
        cls.rows=[score(u) for u in cls.grid]
        cls.core=P.select('core',cls.grid,cls.rows)

    def test_full_candidate_and_selected_counts(self):
        from collections import Counter
        self.assertEqual(Counter(u['kind'] for u in self.grid),
            dict(mlp=5832,head=324,rf=324,lightgbm=324,fixed=120,elasso=252))
        self.assertEqual(len({P.unit_id(u) for u in self.grid}),7176)
        self.assertEqual(len(self.core['units']),528)
        penalty=P.extra_grid('penalty',self.core);budget=P.extra_grid('budget',self.core)
        self.assertEqual(len(penalty),972);self.assertEqual(len(budget),324)
        self.assertEqual(len(P.select('penalty',penalty,[score(u) for u in penalty],self.core)['units']),360)
        self.assertEqual(len(P.select('budget',budget,[score(u) for u in budget],self.core)['units']),108)

    def test_all_architectures_share_eighteen_complete_validation_cells(self):
        ranks=[r for r in self.core['rankings'] if 'architecture' in r]
        self.assertEqual(len(ranks),2*6*9)
        self.assertTrue(all(len(r['units'])==18 for r in ranks))
        self.assertTrue(all(len({(u['base'],u['lead'],u['seed']) for u in r['units']})==18 for r in ranks))

    def test_incomplete_duplicate_unknown_and_denominator_drift_are_rejected(self):
        for rows in (self.rows[:-1],self.rows+self.rows[:1]):
            with self.assertRaises(ValueError):P.select('core',self.grid,rows)
        rows=deepcopy(self.rows);rows[0]['base_val_l1']=3.
        with self.assertRaisesRegex(ValueError,'denominator'):P.select('core',self.grid,rows)
        rows=deepcopy(self.rows);rows[0]['unit']['method']='unknown'
        with self.assertRaisesRegex(ValueError,'unknown'):P.select('core',self.grid,rows)
        rows=deepcopy(self.rows);rows[0]['test_used']=True
        with self.assertRaisesRegex(ValueError,'validation-only'):P.select('core',self.grid,rows)

    def test_mlp_tie_is_earlier_epoch_then_smaller_decay(self):
        u=next(u for u in self.grid if u['kind']=='mlp')
        a=score(dict(u,weight_decay=.01),best_epoch=1)
        b=score(dict(u,weight_decay=0),best_epoch=2)
        self.assertLess(P.inner_key(a),P.inner_key(b))
        b['info']['best_epoch']=1
        self.assertLess(P.inner_key(b),P.inner_key(a))

    def test_outer_tie_uses_parameter_count_before_architecture_id(self):
        rows=[deepcopy(r) for r in self.rows if r['unit']['dataset']=='walmart' and r['unit']['method']=='BUN']
        for r in rows:r['info']['parameters']=1 if r['unit']['architecture']=='w256-l3' else 100
        self.assertEqual(P.rank_architectures(rows)[0]['architecture'],'w256-l3')

    def test_penalties_hold_new_core_architecture_and_zero_is_reused(self):
        grid=P.extra_grid('penalty',self.core)
        for u in grid:self.assertEqual(u['architecture'],self.core['architectures'][u['dataset']+'/HYBN'])
        s=P.select('penalty',grid,[score(u) for u in grid],self.core)
        zeros=[u for u in s['units'] if u['method']=='HYBN']
        self.assertEqual(zeros,[u for u in self.core['units'] if u['method']=='HYBN'])
        self.assertEqual(set(s['deployable_lambda'].values()),{0.})

    def test_budgets_preserve_earlier_frozen_paper_architectures_and_percent_names(self):
        grid=P.extra_grid('budget',self.core)
        self.assertEqual({u['method'] for u in grid},{'HYBN-eps0.5','HYBN-eps1','HYBN-eps2'})
        self.assertEqual({u['architecture'] for u in grid if u['dataset']=='walmart'},{'w64-l3'})
        self.assertEqual({u['architecture'] for u in grid if u['dataset']=='gefcom2017f'},{'w256-l2'})

    def test_budget_feasibility_dominates_accuracy(self):
        u=P.extra_grid('budget',self.core)[0]
        feasible=score(u,best_val=99.,feasible=True,val_coh_nmae=.4)
        infeasible=score(u,best_val=.01,feasible=False,val_coh_nmae=.6)
        self.assertLess(P.inner_key(feasible),P.inner_key(infeasible))
        worse=score(u,best_val=.001,feasible=False,val_coh_nmae=.7)
        self.assertLess(P.inner_key(infeasible),P.inner_key(worse))

    def test_baseline_shared_selection_ties(self):
        rf=[u for u in self.core['units'] if u['method']=='RF']
        self.assertEqual({u['candidate'] for u in rf},{'sqrt-leaf50'})
        lgb=[u for u in self.core['units'] if u['method']=='LightGBM']
        self.assertEqual({(u['trees'],u['leaves']) for u in lgb},{(150,15)})
        self.assertEqual({u['alpha'] for u in self.core['units'] if u['kind']=='elasso'},{1.})


class PaperLifecycle(TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.run=Path(self.temp.name)
        S=np.array([[1.,1.],[1.,1.],[1.,1.],[1.,0.],[0.,1.]])
        h=Hierarchy(node_names=['top','mid1','mid2','a','b'],S=S,
            level_index=dict(top=np.array([0]),middle1=np.array([1]),middle2=np.array([2]),bottom=np.array([3,4])))
        rng=np.random.default_rng(8)
        self.data=SimpleNamespace(hierarchy=h,Y=rng.uniform(1,4,(40,2))@S.T,train_end=30,cfg={'freq':'D'})
        self.arrays={}
        for seg,n in [('train',16),('val',8),('test',6)]:
            self.arrays['ytrue_'+seg]=(rng.uniform(1,4,(n,2))@S.T)[:,:,None]
            self.arrays['yhat_'+seg]=self.arrays['ytrue_'+seg]+rng.normal(0,1,(n,5,1))
            self.arrays['origins_'+seg]=np.arange(n)
        self.u=P.unit('walmart','chronos2',1,42,'BUN','mlp',arm='bottom_mlp_residual',
            architecture='synthetic',hidden=[8],weight_decay=0.,**{'lambda':0.})

    def fit(self,u=None):
        from src.training import mlp_search_training
        arrays={k:v for k,v in self.arrays.items() if not k.endswith('_test')}
        # A synthetic two-epoch numerical run; public CLI exposes no budget override.
        with mock.patch.dict(mlp_search_training.P['training'],max_epochs=2,patience=2):
            return R.fit_candidate(self.run,u or self.u,self.data,arrays,'cpu')

    def freeze(self,u=None):
        for g in ('core','penalty','budget'):
            A.freeze(self.run,dict(group=g,units=[u or self.u],test_used=False))

    def test_synthetic_fit_reload_validation_matches_and_test_is_once(self):
        row=self.fit();self.assertEqual(row['info']['epochs_run'],2)
        predict=R.predictor(self.run,self.u,self.data,self.arrays,'cpu')
        pred=predict(self.arrays['yhat_val'][:,:,0])
        self.assertAlmostEqual(np.abs(pred-self.arrays['ytrue_val'][:,:,0]).mean(),row['info']['selected_val_l1'],places=5)
        np.testing.assert_array_equal(pred,predict(self.arrays['yhat_val'][:,:,0]))  # dropout off
        self.freeze();A.frozen_units(self.run,start=True,confirmed=True)
        with mock.patch.object(A,'load_pair',return_value=(self.data,self.arrays)) as loader:
            R.evaluate_candidate(self.run,self.u,{},'cpu')
            self.assertEqual(loader.call_args.kwargs['segments'],('train','test'))
            with self.assertRaises(FileExistsError):R.evaluate_candidate(self.run,self.u,{},'cpu')
            self.assertEqual(loader.call_count,1)
        with self.assertRaises(ValueError):A.require_pretest(self.run)

    def test_missing_freeze_and_missing_confirmation_block_test(self):
        with self.assertRaises(ValueError):A.frozen_units(self.run,start=True)
        with self.assertRaises(FileNotFoundError):A.frozen_units(self.run,start=True,confirmed=True)
        self.assertFalse((self.run/'test.started.json').exists())

    def test_invalid_evaluation_command_does_not_close_training(self):
        self.fit();self.freeze()
        with mock.patch.object(A,'verify_run',return_value={}),mock.patch('sys.argv',
            ['run_paper_experiments.py','evaluate','--run',str(self.run),'--task','99999','--confirm-selection-frozen']):
            with self.assertRaisesRegex(ValueError,'outside grid'):R.main()
        self.assertFalse((self.run/'test.started.json').exists())

    def test_partial_candidate_cannot_resume_and_completed_one_can(self):
        folder=A.candidate_dir(self.run,self.u);folder.mkdir(parents=True)
        with self.assertRaises(FileNotFoundError):self.fit()
        self.assertFalse((folder/'weights.pt').exists())

    def test_frozen_checkpoint_and_selection_tampering_fail(self):
        self.fit();self.freeze();A.frozen_units(self.run,start=True,confirmed=True)
        (self.run/'core.selection.json').write_text('{}')
        with self.assertRaises((ValueError,KeyError)):A.frozen_units(self.run)
        (A.candidate_dir(self.run,self.u)/'weights.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'changed candidate'):R.predictor(self.run,self.u,self.data,self.arrays,'cpu')

    def test_same_completed_candidate_does_not_train_twice(self):
        first=self.fit()
        with mock.patch('src.training.mlp_search_training.fit',side_effect=AssertionError('duplicate fit')):
            self.assertEqual(self.fit(),first)

    def test_elasso_continuation_requires_previous_path_and_preserves_solver(self):
        from src.baseline.elasso_adapted import prepare,solve
        grid=[u for u in P.core_grid() if u['kind']=='elasso' and (u['dataset'],u['base'],u['lead'])==('walmart','chronos2',1)]
        a,b=grid[:2];self.fit(a);row=self.fit(b)
        folder=A.candidate_dir(self.run,b)
        with np.load(A.candidate_dir(self.run,a)/'coefficients.npz') as z:previous=z['G']
        expected,_=solve(prepare(self.arrays['yhat_train'][:,:,0],self.arrays['ytrue_train'][:,:,0],self.data.hierarchy.S),b['alpha'],previous,
                         **P.config('baselines/elasso_adapted.json')['solver'])
        with np.load(folder/'coefficients.npz') as z:np.testing.assert_array_equal(z['G'],expected)
        self.assertTrue(row['info']['converged'])

    def test_immutable_json_and_complete_receipt(self):
        A.write(self.run/'one.json',{'x':1})
        with self.assertRaises(FileExistsError):A.write(self.run/'one.json',{'x':2})
        with self.assertRaises(ValueError):A.write(self.run/'nan.json',{'x':float('nan')})
        self.assertFalse((self.run/'nan.json').exists())

    def test_base_loader_requests_no_test_arrays(self):
        record={'inputs':{'walmart/chronos2':'reader.npz'}}
        with mock.patch('src.data.datasets.load_dataset',return_value=self.data),mock.patch('src.support.storage.forecasts.load_base_forecasts',return_value={}) as loader:
            A.load_pair(record,'walmart','chronos2')
        self.assertEqual(loader.call_args.kwargs['segments'],('train','val'))

    def test_rf_serialized_bottom_models_preserve_validation_predictions(self):
        from src.baseline import random_forest as rf
        from tests.baseline.test_random_forest import StubForest
        u=P.unit('walmart','chronos2',1,42,'RF','rf',candidate='sqrt-leaf50')
        original=rf.fit_validation
        with mock.patch.object(rf,'fit_validation',side_effect=lambda *a,**kw:original(*a,**kw,factory=StubForest)):
            row=self.fit(u)
        pred=R.predictor(self.run,u,self.data,self.arrays,'cpu')(self.arrays['yhat_val'][:,:,0])
        self.assertAlmostEqual(np.abs(pred-self.arrays['ytrue_val'][:,:,0]).mean(),row['info']['selected_val_l1'])
        np.testing.assert_allclose(pred,pred[:,self.data.hierarchy.bottom_idx]@self.data.hierarchy.S.T)

    def test_analysis_keeps_undefined_tests_separate_and_runs_pooled_blocks(self):
        from scripts.analysis import analyze_paper_results as analysis
        import pandas as pd
        rng=np.random.default_rng(41);packed={};origins={};rows=[]
        methods=list(P.METHODS)+['bottom_mlp_plain','td_features_mlp_plain','allnode_mlp_plain']
        for d in P.DATASETS:
            for b,h,_ in P.cells(d,(42,)):
                cell=(d,b,h);origins[cell]=np.arange(220)
                for m in methods:
                    values=rng.uniform(1,4,(2,220));packed[cell+(m,)]=values
                    rows.append(dict(dataset=d,base=b,lead=h,method=m,level='all',wape=values[0].mean(),mase=values[1].mean()))
                packed[cell+('BU',)]=packed[cell+('base',)].copy()
        tests,intervals=analysis.compare_losses(packed,origins)
        self.assertEqual(len(tests),12*4*(13+90))
        self.assertTrue(any(r['status']=='undefined_degenerate_variance' and r['reject'] is None for r in tests))
        with mock.patch.dict(analysis.RANK_PROTOCOL,bootstrap_replicates=19):
            ranks,pooled,groups=analysis.pooled_ranks(packed,origins,pd.DataFrame(rows))
        self.assertEqual(ranks.shape,(19,2));self.assertEqual(len(pooled),342*3)
        self.assertTrue(all(r['family_size']==342 for r in pooled))

    def test_paired_statistics_and_seed_loss_aggregation(self):
        from scripts.analysis.analyze_paper_results import bootstrap_dataset,RANK_METHODS,contiguous_groups
        rng=np.random.default_rng(20);x=rng.uniform(1,3,(31,6,2,len(RANK_METHODS)))
        result=bootstrap_dataset(x,7,50,np.random.default_rng(10))
        self.assertEqual(result.shape,(50,2,19))
        np.testing.assert_array_equal(result[:,:,RANK_METHODS.index('base')],1.)
        decisions={('a','b'):False,('a','c'):True,('b','c'):False}
        self.assertEqual(contiguous_groups(['a','b','c'],decisions),[(0,1),(1,2)])
