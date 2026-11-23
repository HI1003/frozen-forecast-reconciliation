"""Analyze a completed reader run; never train, select or change archived results.

The 13-comparison recipe, exploratory 90-comparison families and pooled rank
bootstrap are reported separately. Applying an old recipe to new forecasts does
not recreate the paper's historical pre-registration or its saved test results.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import stats
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.experiments import paper_protocol as P, experiment_storage as A
from src.evaluation.statistics import holm_adjust
METHODS=('base','BU','BUL','TD','TDL','TDN','HYB','HYBL','HYBN','ols','wls_var','wls_struct','mint_shrink')
NAMES={'OLS':'ols','WLS-var':'wls_var','WLS-struct':'wls_struct','MinT-shrink':'mint_shrink'}
RANK_PROTOCOL=P.config('statistics/rank_groups.json')
RANK_METHODS=[NAMES.get(m,m) for m in RANK_PROTOCOL['methods']]


def hac_test(diff, lead, max_lag):
    """Bartlett LRV, original lead-based Harvey factor and t(N-1) reference."""
    d=np.asarray(diff,dtype=np.float64)
    n=d.size
    if d.ndim!=1 or not np.isfinite(d).all() or not 0<=max_lag<n or not 1<=lead<n:
        raise ValueError('Invalid loss series, lag or lead')
    mean=float(d.mean()); centered=d-mean
    gamma0=float(centered@centered/n); lrv=gamma0
    for k in range(1,max_lag+1):
        lrv+=2*(1-k/(max_lag+1))*float(centered[k:]@centered[:-k]/n)
    if not np.isfinite(lrv) or lrv<=max(np.finfo(float).tiny,1e-12*gamma0):
        raise ValueError('Degenerate HAC variance; no fallback')
    harvey=float(np.sqrt((n+1-2*lead+lead*(lead-1)/n)/n))
    standard_error=float(np.sqrt(lrv/n))
    statistic=mean/standard_error*harvey
    p=float(2*stats.t.sf(abs(statistic),df=n-1))
    return dict(mean_difference=mean,standard_error=standard_error,long_run_variance=lrv,
                dm_stat=statistic,p_raw=p,harvey_factor=harvey,n_origins=n)

def holm_family(methods,p_values):
    if len(methods)!=13 or len(set(methods))!=13 or set(methods)!=set(METHODS):
        raise ValueError('Holm requires the complete thirteen-method family')
    p=np.asarray(p_values,dtype=float)
    if p.shape!=(13,) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Invalid family p-values')
    order=np.argsort(p,kind='stable')
    adj=np.minimum(1.,np.maximum.accumulate(p[order]*(13-np.arange(13))))
    out=np.empty_like(adj);out[order]=adj
    return out

def block_ci(diff,block_length,resamples=2000,seed=42):
    """Exactly the existing non-circular draw convention, using block sums."""
    d=np.asarray(diff,dtype=np.float64);n=len(d)
    if d.ndim!=1 or not np.isfinite(d).all() or not 1<=block_length<=n:
        raise ValueError('Invalid series or block length')
    blocks=int(np.ceil(n/block_length))
    starts=np.random.default_rng(seed).integers(0,n-block_length+1,size=(resamples,blocks))
    prefix=np.r_[0.,np.cumsum(d)]
    sums=(prefix[starts+block_length]-prefix[starts]).sum(axis=1)
    remainder=n-(blocks-1)*block_length
    last=starts[:,-1]
    sums-=prefix[last+block_length]-prefix[last+remainder]
    draws=sums/n
    lo,hi=np.quantile(draws,[.025,.975])
    return float(lo),float(hi)

def bootstrap_dataset(x, length, n_boot, rng):
    """Synchronized non-circular MBB; exact n observations, last block truncated.

    x: origin, setting, metric, method. Ratios use resampled base means.
    """
    n = len(x)
    assert 1 <= length < n
    q, remainder = divmod(n, length)
    flat = x.reshape(n, -1)
    cs = np.vstack([np.zeros((1, flat.shape[1])), np.cumsum(flat, axis=0)])
    sums = cs[length:]-cs[:-length]
    tail = cs[remainder:]-cs[:-remainder] if remainder else None
    answer = np.empty((n_boot, 2, len(RANK_METHODS)))
    base_idx = RANK_METHODS.index('base')
    for first in range(0, n_boot, 80):
        size = min(80, n_boot-first)
        starts = rng.integers(0, n-length+1, size=(size, q))
        sample = sums[starts].sum(axis=1)
        if remainder:
            # Draw a full-length valid block start, then truncate its tail.
            sample += tail[rng.integers(0, n-length+1, size=size)]
        means = (sample/n).reshape(size, *x.shape[1:])
        answer[first:first+size] = (means/means[..., base_idx, None]).mean(axis=1)
    return answer

def contiguous_groups(order, decisions):
    """Maximal rank-contiguous cliques, never use transitive closure."""
    candidates = []
    for start in range(len(order)):
        for end in range(start+1, len(order)):
            members = order[start:end+1]
            if all(not decisions[a,b] for i,a in enumerate(members) for b in members[i+1:]):
                candidates.append((start, end))
            else:
                break
    return [(a,b) for a,b in candidates if not any(c<=a and d>=b and (c,d)!=(a,b) for c,d in candidates)]


def collect(run):
    """Check frozen coverage and average losses AFTER computing each seed's error."""
    units=A.frozen_units(run); rows=[]; packed=defaultdict(list); origins={}
    for u in units:
        folder=run/'test'/P.unit_id(u);r=A.read(folder/'metrics.json')
        if r['unit']!=u or A.sha(folder/'predictions.npz')!=r['predictions_sha256']:
            raise ValueError('changed or mismatched test artifact')
        cell=(u['dataset'],u['base'],u['lead'])
        with np.load(folder/'predictions.npz') as z:
            if cell in origins:np.testing.assert_array_equal(origins[cell],z['origins'])
            origins[cell]=z['origins'].copy()
            packed[cell+(u['method'],)].append((z['mae'].copy(),z['mase'].copy()))
        for level,metric in r['level_metrics'].items():
            rows.append(dict(dataset=u['dataset'],base=u['base'],lead=u['lead'],method=u['method'],seed=u['seed'],
                level=level,**metric,coherence_nmae_pct=r['coherence_nmae_pct'],
                test_epsilon_feasible=r.get('test_epsilon_feasible'),inference_seconds=r['inference_seconds']))
    packed={k:np.mean(v,axis=0) for k,v in packed.items()}
    return pd.DataFrame(rows),packed,origins


def compare_losses(packed,origins):
    """Separate Holm families per setting and bandwidth, without an IID fallback."""
    rows=[];intervals=[]
    protocol=P.config('statistics/pairwise_significance.json')
    refs=[NAMES.get(m,m) for m in protocol['references']]
    for d,b,h in origins:
        def loss(m):return packed[d,b,h,m][0]
        families=[('original_13_recipe_on_reader_forecasts',[('BUN',m) for m in METHODS]),
                  ('exploratory_90',list(itertools.product(protocol['strategies'],refs)))]
        # Direct/residual effects get descriptive intervals, no new test family.
        direct=[('BUN','bottom_mlp_plain'),('TDN','td_features_mlp_plain'),('HYBN','allnode_mlp_plain')]
        lags=[('h-1',h-1)]+[(f'lag{q}',q) for q in protocol['sensitivity_hac'][d]]
        for family,pairs in families:
            for label,q in lags:
                group=[]
                for candidate,reference in pairs:
                    diff=loss(reference)-loss(candidate)
                    record=dict(dataset=d,base=b,lead=h,family=family,bandwidth=label,
                        candidate=candidate,reference=reference,positive_favours='candidate',hac_lag=q)
                    try:record.update(hac_test(diff,h,q),status='tested')
                    except ValueError as exc:
                        if family=='original_13_recipe_on_reader_forecasts':raise
                        if not str(exc).startswith('Degenerate HAC'):raise
                        # Undefined is recorded, never relabelled as a non-rejection.
                        record.update(status='undefined_degenerate_variance',p_raw=None,mean_difference=float(diff.mean()))
                    group.append(record)
                adjusted=holm_adjust([r['p_raw'] if r['p_raw'] is not None else 1. for r in group])
                for r,p in zip(group,adjusted):
                    r.update(family_size=len(pairs),p_holm=float(p) if r['p_raw'] is not None else None,
                             reject=bool(p<.05) if r['p_raw'] is not None else None)
                rows.extend(group)
            for candidate,reference in pairs:
                for length in protocol['sensitivity_hac'][d]:
                    lo,hi=block_ci(loss(reference)-loss(candidate),length)
                    intervals.append(dict(dataset=d,base=b,lead=h,family=family,candidate=candidate,
                        reference=reference,block_length=length,ci_lo=lo,ci_hi=hi,coverage='marginal'))
        for candidate,reference in direct:
            for length in protocol['sensitivity_hac'][d]:
                lo,hi=block_ci(loss(reference)-loss(candidate),length)
                intervals.append(dict(dataset=d,base=b,lead=h,family='descriptive_direct_residual',
                    candidate=candidate,reference=reference,block_length=length,ci_lo=lo,ci_hi=hi,coverage='marginal'))
    return rows,intervals


def pooled_ranks(packed,origins,metrics):
    p=RANK_PROTOCOL; arrays={};point_d={}
    for d in P.DATASETS:
        cells=[(d,b,h) for b in P.BASES for h in P.LEADS[d]]
        common=origins[cells[0]]
        for c in cells:
            np.testing.assert_array_equal(origins[c],common)
            if not np.all(np.diff(origins[c])==1): raise ValueError('pooled blocks require consecutive synchronized origins')
        x=np.stack([np.stack([packed[c+(m,)] for m in RANK_METHODS],axis=-1).transpose(1,0,2) for c in cells],axis=1)
        arrays[d]=x;means=x.mean(0)
        point_d[d]=(means/means[...,RANK_METHODS.index('base'),None]).mean(0)
    point=np.mean(list(point_d.values()),axis=0)
    all_metrics=metrics[(metrics.level=='all')&metrics.method.isin(RANK_METHODS)]
    values=all_metrics.groupby(['dataset','base','lead','method'])[['wape','mase']].mean().unstack('method')
    rank_means=pd.DataFrame({m:values[m].rank(axis=1,method='average').mean() for m in ('wape','mase')}).loc[RANK_METHODS]
    tests=[];groups=[]
    specs=[('primary',p['primary_block_lengths'])]+[(f'sensitivity_{i+1}',v) for i,v in enumerate(p['sensitivity_block_lengths'])]
    for mode,lengths in specs:
        rng=np.random.default_rng(p['seed'])
        draws=np.mean([bootstrap_dataset(arrays[d],lengths[d],p['bootstrap_replicates'],rng) for d in P.DATASETS],axis=0)
        rows=[]
        for k,metric in enumerate(('wape','mase')):
            for i,j in itertools.combinations(range(len(RANK_METHODS)),2):
                delta=point[k,i]-point[k,j];sim=draws[:,k,i]-draws[:,k,j];centered=sim-sim.mean()
                raw=(1+np.count_nonzero(np.abs(centered)>=abs(delta)))/(len(sim)+1)
                lo,hi=np.quantile(centered,[.025,.975])
                rows.append(dict(mode=mode,metric=metric,method_a=RANK_METHODS[i],method_b=RANK_METHODS[j],
                    difference=float(delta),p_raw=float(raw),ci_lo=float(delta-hi),ci_hi=float(delta-lo)))
        for r,adjusted in zip(rows,holm_adjust([r['p_raw'] for r in rows])):
            r.update(p_holm=float(adjusted),reject=bool(adjusted<p['alpha']),family_size=342)
        tests.extend(rows)
        for metric in ('wape','mase'):
            order=rank_means[metric].sort_values(kind='stable').index.tolist();decisions={}
            for r in rows:
                if r['metric']==metric:
                    decisions[r['method_a'],r['method_b']]=r['reject']
                    decisions[r['method_b'],r['method_a']]=r['reject']
            groups.extend(dict(mode=mode,metric=metric,members=order[a:b+1]) for a,b in contiguous_groups(order,decisions))
    return rank_means,tests,groups


def cost_rows(run):
    rows=[];selected={P.unit_id(u) for u in A.frozen_units(run)}
    for group in ('core','penalty','budget'):
        grid=P.core_grid() if group=='core' else P.extra_grid(group,A.selection(run,'core'))
        for u in grid:
            r=A.candidate_record(run,u);i=r['info'];folder=run/'test'/P.unit_id(u)
            timing=A.read(folder/'timing.json') if (folder/'timing.json').exists() else {}
            rows.append(dict(group=group,id=P.unit_id(u),dataset=u['dataset'],base=u['base'],lead=u['lead'],
                method=u['method'],seed=u['seed'],selected=P.unit_id(u) in selected,
                candidate_seconds=i['candidate_seconds'],parameters=i['parameters'],
                train_seconds=i.get('train_s',i.get('fit_seconds')),epochs_run=i.get('epochs_run'),
                peak_gpu_mib=i.get('peak_gpu_mib'),median_ms_per_origin=timing.get('median_ms_per_origin'),
                timing_includes_model_loading=timing.get('includes_model_loading')))
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,help='New output directory; default RUN/analysis')
    args=parser.parse_args();run=args.run.resolve();out=args.output or run/'analysis'
    A.verify_run(run)
    if out.exists():raise FileExistsError('analysis output exists; choose a new output directory')
    metrics,packed,origins=collect(run)
    tests,intervals=compare_losses(packed,origins)
    ranks,pooled,groups=pooled_ranks(packed,origins,metrics)
    costs=cost_rows(run)
    out.mkdir(parents=True,exist_ok=False)
    metrics.to_csv(out/'per_seed_metrics.csv',index=False)
    averaged=metrics.groupby(['dataset','base','lead','method','level'])[['wape','mase','coherence_nmae_pct']].mean().reset_index()
    averaged.to_csv(out/'metrics.csv',index=False)
    for name,methods in [('main',P.METHODS),('direct_residual',tuple(P.ARMS))]:
        averaged[averaged.method.isin(methods)].to_csv(out/(name+'.csv'),index=False)
    for prefix,name in [('HYBN-lam','penalty'),('HYBN-eps','budget')]:
        averaged[averaged.method.str.startswith(prefix)|(averaged.method=='HYBN')].to_csv(out/(name+'.csv'),index=False)
    metrics[metrics.method.str.startswith('HYBN-eps')].to_csv(out/'budget_feasibility.csv',index=False)
    pd.DataFrame(tests).to_csv(out/'paired_tests.csv',index=False)
    pd.DataFrame(intervals).to_csv(out/'paired_intervals.csv',index=False)
    ranks.to_csv(out/'mean_ranks.csv');pd.DataFrame(pooled).to_csv(out/'rank_pairs.csv',index=False)
    A.write(out/'rank_groups.json',groups)
    costs.to_csv(out/'cost_per_candidate.csv',index=False)
    costs.groupby(['group','dataset','method']).agg(candidates=('id','count'),selected=('selected','sum'),
        search_seconds=('candidate_seconds','sum')).to_csv(out/'search_costs.csv')
    A.write(out/'analysis.json',dict(scope='reader-run analysis; original archived statistics unchanged',
        inference='conditional on two fixed datasets and fitted models; exploratory comparisons are separate',
        run_sha256=A.sha(run/'run.json'),seeds='mean losses, never ensemble predictions',
        deployable_lambdas=A.selection(run,'penalty')['deployable_lambda'],
        outputs={p.name:A.sha(p) for p in out.iterdir() if p.is_file()}))
    print(out)


if __name__=='__main__':main()
