"""Run the final paper protocol: plan -> fit/select -> evaluate once -> timing.

All commands use a separate reader run directory and explicit generated inputs.
See scripts/README.md for the four-input mapping and complete command sequence.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.experiments import paper_protocol as P, experiment_storage as A


def blocks(arrays,lead,segment):
    return arrays['yhat_'+segment][:,:,lead-1],arrays['ytrue_'+segment][:,:,lead-1]


def save_pickle(path,model):
    with Path(path).open('xb') as stream: pickle.dump(model,stream,protocol=5)


def load_pickle(path):
    # Only locally fitted models whose complete.json hash has just been checked.
    with Path(path).open('rb') as stream: return pickle.load(stream)


def fit_candidate(run,u,data,arrays,device,n_jobs=1):
    """One train-only candidate. Validation targets are used solely for scores."""
    import numpy as np
    from src.evaluation.metrics import coherence_nmae,coherence_scale
    A.require_pretest(run)
    folder=A.candidate_dir(run,u)
    if folder.exists():
        return A.candidate_record(run,u)  # complete-only resume, no partial retry
    xt,yt=blocks(arrays,u['lead'],'train');xv,yv=blocks(arrays,u['lead'],'val')
    folder.mkdir(parents=True,exist_ok=False)
    kind=u['kind']; h=data.hierarchy; info={}; started=time.perf_counter()
    if kind in ('mlp','head'):
        import torch
        from src.evaluation.prediction import apply_head
        if str(device).startswith('cuda'): torch.cuda.reset_peak_memory_stats()
        if kind=='mlp':
            from src.training.mlp_search_training import fit
            model,info=fit(u,data,xt,yt,xv,yv,device=device)
        else:
            from src.training.head_training import HeadTrainConfig,train_head
            cfg=HeadTrainConfig(device=device,seed=u['seed'],weight_decay=u['weight_decay'],
                                hidden=tuple(u.get('hidden',(256,128))),eps_target=u.get('epsilon'))
            name='HYBN' if 'epsilon' in u else u['method']
            fit_started=time.perf_counter()
            model,info=train_head(name,h,xt,yt,xv,yv,True,cfg,record_history=True)
            if str(device).startswith('cuda'):torch.cuda.synchronize()
            info['train_s']=time.perf_counter()-fit_started
        torch.save(A.cpu_state_dict(model),folder/'weights.pt')
        prediction=apply_head(model.eval(),xv,device)
        info['parameters']=sum(p.numel() for p in model.parameters())
        info['peak_gpu_mib']=torch.cuda.max_memory_allocated()/2**20 if str(device).startswith('cuda') else None
    elif kind=='rf':
        from src.baseline.random_forest import fit_validation,load_protocol
        prediction,cost=fit_validation(load_protocol(),u['candidate'],u['seed'],h.S,h.bottom_idx,xt,yt,xv,
            n_jobs=n_jobs,save_model=lambda j,m:save_pickle(folder/f'bottom_{j}.pkl',m))
        info.update(cost)
    elif kind=='lightgbm':
        from importlib.metadata import version
        from src.baseline.lightgbm import fit,protocol
        if version('lightgbm')!=protocol()['lightgbm_version']: raise ValueError('LightGBM version differs from protocol')
        bottom=np.empty((len(xv),h.m));fitting_s=0.
        for j,col in enumerate(h.bottom_idx):
            fit_started=time.perf_counter()
            model=fit(xt,yt[:,col],u['trees'],u['leaves'],u['seed'])
            fitting_s+=time.perf_counter()-fit_started
            model.save_model(str(folder/f'bottom_{j}.txt'))
            bottom[:,j]=np.maximum(model.predict(xv),0)
        prediction=bottom@h.S.T
        info['fit_seconds']=fitting_s
    elif kind=='elasso':
        from src.baseline.elasso_adapted import prepare,solve,predict
        prepared=prepare(xt,yt,h.S); initial=None
        if u['alpha_index']:
            previous=next(c for c in P.core_grid() if c['kind']=='elasso'
                and (c['dataset'],c['base'],c['lead'],c['alpha_index'])==(u['dataset'],u['base'],u['lead'],u['alpha_index']-1))
            A.candidate_record(run,previous)
            with np.load(A.candidate_dir(run,previous)/'coefficients.npz') as z: initial=z['G']
        G,solver=solve(prepared,u['alpha'],initial,**P.config('baselines/elasso_adapted.json')['solver'])
        np.savez_compressed(folder/'coefficients.npz',G=G)
        prediction,negative=predict(xv,G,h.S)
        info.update(solver,train_s=solver['seconds'],negative_bottom_fraction=negative,parameters=int(G.size))
    else:
        method=u['method']
        if method in ('ols','wls_var','wls_struct','mint_shrink'):
            from src.baseline.classical import prepare_projection
            model=prepare_projection(method,h,yt-xt,True)
            save_pickle(folder/'projection.pkl',model)
            prediction=model.predict(xv)
        else: prediction=fixed_prediction(method,data,xt,xv)
    loss=float(np.abs(np.asarray(prediction,dtype=np.float64)-yv).mean())
    info.setdefault('best_val',loss);info.setdefault('selected_val_l1',loss)
    info.setdefault('best_epoch',0);info.setdefault('parameters',0)
    info.setdefault('val_coherence_fraction',info.get('val_coh_nmae',
        coherence_nmae(prediction,h,coherence_scale(yt,h)))/100)
    info['candidate_seconds']=time.perf_counter()-started
    row=dict(unit=u,info=info,base_val_l1=float(np.abs(xv-yv).mean()),
             device=device if kind in ('mlp','head') else 'cpu',test_used=False)
    A.complete_candidate(folder,row)
    return row


def fixed_prediction(method,data,xt,x):
    if method=='base': return x.copy()  # raw base is never clipped
    if method in ('MO-L1','MO-L2'):
        from src.strategies.hybrid_allocation import reconcile
        return reconcile(x,data.hierarchy.S,data.hierarchy.level_index,
                         'middle_out_middle'+('1' if method=='MO-L1' else '2'))[0]
    from src.strategies import build_strategy
    from src.evaluation.prediction import apply_head
    return apply_head(build_strategy(method,data.hierarchy,xt).eval(),x)


def predictor(run,u,data,arrays,device):
    """Load frozen weights; this function never calls a fitting routine."""
    import numpy as np
    A.candidate_record(run,u)
    folder=A.candidate_dir(run,u);xt,_=blocks(arrays,u['lead'],'train');kind=u['kind'];h=data.hierarchy
    if kind in ('mlp','head'):
        import torch
        if kind=='mlp':
            from src.ablation.direct_residual import make
            model=make(u['arm'],data,xt.mean(0),xt.std(0),u['hidden'],xt)
        else:
            from src.strategies import build_strategy
            model=build_strategy('HYBN' if 'epsilon' in u else u['method'],h,xt,hidden=u.get('hidden',(256,128)))
        model.load_state_dict(torch.load(folder/'weights.pt',map_location='cpu',weights_only=True),strict=True)
        model.eval().to(device)
        from src.evaluation.prediction import apply_head
        return lambda x:apply_head(model,x,device)
    if kind in ('rf','lightgbm'):
        # Sequential model loading bounds memory, matching bottom-wise fitting.
        def predict(x):
            bottom=np.empty((len(x),h.m))
            for j in range(h.m):
                if kind=='rf': model=load_pickle(folder/f'bottom_{j}.pkl')
                else:
                    import lightgbm as lgb
                    model=lgb.Booster(model_file=str(folder/f'bottom_{j}.txt'))
                bottom[:,j]=np.maximum(model.predict(x),0)
            return bottom@h.S.T
        return predict
    if kind=='elasso':
        from src.baseline.elasso_adapted import predict
        with np.load(folder/'coefficients.npz') as z:G=z['G']
        return lambda x:predict(x,G,h.S)[0]
    if u['method'] in ('ols','wls_var','wls_struct','mint_shrink'):
        return load_pickle(folder/'projection.pkl').predict
    if u['method'] in ('BU','TD','HYB'):
        from src.strategies import build_strategy
        from src.evaluation.prediction import apply_head
        model=build_strategy(u['method'],h,xt).eval().to(device)
        return lambda x:apply_head(model,x,device)
    return lambda x:fixed_prediction(u['method'],data,xt,x)


def evaluate_candidate(run,u,record,device):
    import numpy as np
    from src.evaluation.metrics import level_metrics,mase_scale,seasonal_period,coherence_nmae,coherence_scale
    if u not in A.frozen_units(run): raise ValueError('checkpoint was not frozen for test')
    A.candidate_record(run,u)
    folder=A.begin_test(run,u)  # exclusive claim precedes every test-array read
    data,arrays=A.load_pair(record,u['dataset'],u['base'],segments=('train','test'))
    x,y=blocks(arrays,u['lead'],'test');_,yt=blocks(arrays,u['lead'],'train')
    predict=predictor(run,u,data,arrays,device)
    started=time.perf_counter();pred=predict(x);seconds=time.perf_counter()-started
    if pred.shape!=y.shape or not np.isfinite(pred).all(): raise ValueError('invalid test prediction')
    scales=mase_scale(data.Y,data.train_end,seasonal_period(data.cfg['freq']))
    metrics=level_metrics(pred,y,data.hierarchy,scales)
    coh=coherence_nmae(pred,data.hierarchy,coherence_scale(yt,data.hierarchy))
    error=np.abs(pred-y);valid=scales>1e-8
    with (folder/'predictions.npz').open('xb') as stream:
        np.savez_compressed(stream,prediction=pred,origins=arrays['origins_test'],
            mae=error.mean(1),mase=(error[:,valid]/scales[valid]).mean(1))
    receipt=dict(unit=u,level_metrics=metrics,coherence_nmae_pct=coh,inference_seconds=seconds,
        n_origins=len(x),predictions_sha256=A.sha(folder/'predictions.npz'))
    if 'epsilon' in u: receipt['test_epsilon_feasible']=coh<=100*u['epsilon']
    A.write(folder/'metrics.json',receipt)


def timing_candidate(run,u,record,device,repeats):
    """Deployment timing is a separate observation, never checkpoint selection."""
    import numpy as np
    from threadpoolctl import threadpool_limits
    neural=u['kind'] in ('mlp','head') or u['method'] in ('BU','TD','HYB')
    if neural:
        import torch
    folder=Path(run)/'test'/P.unit_id(u)
    metrics=A.read(folder/'metrics.json')
    if A.sha(folder/'predictions.npz')!=metrics['predictions_sha256']: raise ValueError('test output changed')
    data,arrays=A.load_pair(record,u['dataset'],u['base'],segments=('train','test'))
    x,_=blocks(arrays,u['lead'],'test')
    predict=predictor(run,u,data,arrays,device)
    samples=[]
    if neural:torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        predict(x[:128])
        for _ in range(repeats):
            if neural and str(device).startswith('cuda'):torch.cuda.synchronize()
            t=time.perf_counter()
            for start in range(0,len(x),128):predict(x[start:start+128])
            if neural and str(device).startswith('cuda'):torch.cuda.synchronize()
            samples.append(time.perf_counter()-t)
    A.write(folder/'timing.json',dict(unit=u,device=device if neural else 'cpu',batch_size=128,repeats=repeats,
        seconds=samples,median_ms_per_origin=1000*float(np.median(samples))/len(x),threads=1,
        includes_model_loading=u['kind'] in ('rf','lightgbm'),
        scope='reader deployment wall time; archival hardware timing is not reproduced'))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('plan','fit','select','evaluate','timing'))
    p.add_argument('--run',type=Path,default=ROOT/'runs/paper')
    p.add_argument('--inputs',type=Path,help='JSON mapping dataset/base to generated .npz path; required for first fit')
    p.add_argument('--group',choices=('core','penalty','budget'),default='core')
    choice=p.add_mutually_exclusive_group()
    choice.add_argument('--task',type=int,help='Zero-based candidate or selected-test index from plan')
    choice.add_argument('--all',action='store_true',help='Run all candidates/selected checkpoints sequentially')
    p.add_argument('--device',default='cpu')
    p.add_argument('--jobs',type=int,default=1,help='Allocated RF threads')
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--confirm-selection-frozen',action='store_true')
    p.add_argument('--list-tasks',action='store_true')
    p.add_argument('--methods',nargs='+',help='Optional subset for fit/evaluate/timing; select always checks the entire group')
    return p


def main():
    args=parser().parse_args();run=args.run.resolve()
    core=None if args.group=='core' else A.selection(run,'core')
    grid=P.core_grid() if args.group=='core' else P.extra_grid(args.group,core)
    if args.stage=='plan':
        print(json.dumps(dict(group=args.group,candidates=len(grid),by_kind=dict(Counter(u['kind'] for u in grid)),
            tasks=[dict(task=i,id=P.unit_id(u),**u) for i,u in enumerate(grid)] if args.list_tasks else None),indent=2))
        return
    if args.jobs<1 or args.repeats<1: raise ValueError('jobs/repeats must be positive')
    with A.run_lock(run):
        if args.stage=='fit' and not (run/'run.json').exists():
            if args.inputs is None: raise ValueError('first fit requires --inputs')
            A.initialise(run,args.inputs)
        record=A.verify_run(run)
        if args.inputs is not None: A.initialise(run,args.inputs)
        if args.stage in ('fit','select'):
            A.require_pretest(run)
            if (run/(args.group+'.selection.json')).exists(): raise FileExistsError('group already frozen')
        if args.stage=='select':
            rows=[A.candidate_record(run,u) for u in grid]
            selected=P.select(args.group,grid,rows,core)
            A.freeze(run,selected)
            print(json.dumps(dict(group=args.group,frozen=len(selected['units']),architectures=selected['architectures'])))
            return
        if args.stage in ('evaluate','timing'):
            grid=A.frozen_units(run,preview=args.stage=='evaluate')
        if args.task is None and not args.all: raise ValueError('choose --task INDEX or --all explicitly')
        if args.task is not None and not 0<=args.task<len(grid): raise ValueError('task outside grid')
        tasks=list(enumerate(grid)) if args.all else [(args.task,grid[args.task])]
        if args.methods:
            unknown=set(args.methods)-{u['method'] for u in grid}
            if unknown: raise ValueError('methods outside grid: '+str(sorted(unknown)))
            tasks=[(i,u) for i,u in tasks if u['method'] in args.methods]
        if args.stage=='evaluate':
            A.frozen_units(run,start=True,confirmed=args.confirm_selection_frozen)
        cached_key=None;data=arrays=None
        for i,u in tasks:
            if args.stage=='fit':
                key=(u['dataset'],u['base'])
                if key!=cached_key:data,arrays=A.load_pair(record,*key);cached_key=key
                fit_candidate(run,u,data,arrays,args.device,args.jobs)
            elif args.stage=='evaluate':
                folder=run/'test'/P.unit_id(u)
                if (folder/'metrics.json').exists():
                    m=A.read(folder/'metrics.json')
                    if A.sha(folder/'predictions.npz')!=m['predictions_sha256']:raise ValueError('test output changed')
                    continue  # complete-only resume; never reread test for a completed unit
                evaluate_candidate(run,u,record,args.device)
            else:timing_candidate(run,u,record,args.device,args.repeats)
            print(f'{args.stage} {i+1}/{len(grid)} {P.unit_id(u)}',flush=True)


if __name__=='__main__':main()
