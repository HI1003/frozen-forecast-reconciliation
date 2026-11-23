"""Final-paper grids and validation-only selection; no datasets are opened here.

Historical config files remain immutable evidence. Only their scientific fields
are used: server job IDs, earlier superseded arms and dispatch plans are ignored.
"""
from __future__ import annotations
import hashlib
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def config(name):
    return json.loads((ROOT / 'configs' / name).read_text(encoding='utf-8'))


MLP = config('experiments/mlp.json')
DATASETS = tuple(MLP['fixed']['datasets'])
BASES = tuple(MLP['fixed']['bases'])
LEADS = MLP['fixed']['leads']
SEEDS = tuple(MLP['fixed']['seeds'])
ARCHITECTURES = MLP['architectures']
WEIGHT_DECAYS = MLP['training']['weight_decays']
LAMBDAS = config('experiments/hybn_lambda.json')['lambdas']
EPSILONS = tuple(v/100 for v in config('protocols/tdn_hybn_alignment.json')['epsilon_percent_grid'])
BUDGET_SELECTION = ROOT/'results/evidence/aligned/selected_architectures.json'
ARMS = {r['canonical_method'] or r['id']: r['id'] for r in MLP['arms']
        if r['id'] != 'coherent_hyb_residual'}
LINEAR = ('BUL', 'TDL', 'HYBL')
FIXED = ('base', 'BU', 'TD', 'HYB', 'MO-L1', 'MO-L2',
         'ols', 'wls_var', 'wls_struct', 'mint_shrink')
METHODS = ('base', 'BU', 'BUL', 'BUN', 'TD', 'TDL', 'TDN', 'HYB', 'HYBL',
           'HYBN', 'MO-L1', 'MO-L2', 'ols', 'wls_var', 'wls_struct',
           'mint_shrink', 'RF', 'LightGBM', 'Elasso_adapted')


def unit_id(unit):
    raw = json.dumps(unit, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def cells(dataset, seeds=SEEDS):
    return itertools.product(BASES, LEADS[dataset], seeds)


def unit(dataset, base, lead, seed, method, kind, **parameters):
    return dict(dataset=dataset, base=base, lead=lead, seed=seed,
                method=method, kind=kind, **parameters)


def core_grid():
    """All paper candidates, including every unselected search fit."""
    from src.baseline.random_forest import candidates, load_protocol
    rf = candidates(load_protocol())
    lgb = config('experiments/lightgbm_hybn.json')['f03']
    # Same descending path as numpy.logspace(0, -4, 20), then zero.
    import numpy as np
    elasso = config('baselines/elasso_adapted.json')
    alphas = list(np.logspace(0, math.log10(elasso['alpha_min']), elasso['positive_alphas'])) + [0.]
    out = []
    for d in DATASETS:
        for b, h, s in cells(d):
            for m, arm in ARMS.items():
                for arch, wd in itertools.product(ARCHITECTURES, WEIGHT_DECAYS):
                    out.append(unit(d,b,h,s,m,'mlp',arm=arm,architecture=arch,
                                    hidden=ARCHITECTURES[arch],weight_decay=wd,**{'lambda':0.}))
            for m, wd in itertools.product(LINEAR, WEIGHT_DECAYS):
                out.append(unit(d,b,h,s,m,'head',weight_decay=wd))
            for candidate in rf:
                out.append(unit(d,b,h,s,'RF','rf',candidate=candidate))
            for trees, leaves in itertools.product(lgb['n_estimators'],lgb['num_leaves']):
                out.append(unit(d,b,h,s,'LightGBM','lightgbm',trees=trees,leaves=leaves))
        for b, h, s in cells(d, (42,)):
            for m in FIXED:
                out.append(unit(d,b,h,s,m,'fixed'))
            for i, alpha in enumerate(alphas):
                out.append(unit(d,b,h,s,'Elasso_adapted','elasso',alpha_index=i,alpha=float(alpha)))
    return out


def extra_grid(group, core_selection):
    """Penalties inherit core HYBN; budgets retain the paper's earlier frozen architecture."""
    if group not in ('penalty', 'budget'):
        raise ValueError('unknown group')
    winners = core_selection['architectures']
    out = []
    for d in DATASETS:
        arch = (winners[d + '/HYBN'] if group=='penalty' else
                json.loads(BUDGET_SELECTION.read_text(encoding='utf-8'))['selected'][d]['HYBN']['architecture_id'])
        for b,h,s in cells(d):
            for value,wd in itertools.product(LAMBDAS[1:] if group=='penalty' else EPSILONS,WEIGHT_DECAYS):
                kw = dict(architecture=arch,hidden=ARCHITECTURES[arch],weight_decay=wd)
                if group == 'penalty':
                    kw.update(arm=ARMS['HYBN'], **{'lambda':float(value)})
                    name,kind = f'HYBN-lam{value:g}','mlp'
                else:
                    kw.update(epsilon=value)
                    name,kind = f'HYBN-eps{100*value:g}','head'
                out.append(unit(d,b,h,s,name,kind,**kw))
    return out


def inner_key(row):
    u, info = row['unit'], row['info']
    if u['kind']=='head':
        from src.training.model_selection import info_selection_key
        return (*info_selection_key(info),u['weight_decay'])
    return (info['best_val'],info['best_epoch'],u.get('weight_decay',0.))


def checked_rows(grid, rows):
    expected = {unit_id(u):u for u in grid}
    seen = {}
    references = {}
    for r in rows:
        if r.get('test_used') is not False:
            raise ValueError('selection accepts validation-only records')
        u=r['unit']; key=unit_id(u)
        if key not in expected or key in seen or u != expected[key]:
            raise ValueError('unknown or duplicate validation candidate')
        values=[r['info']['best_val'],r['info']['selected_val_l1'],r['base_val_l1']]
        if not all(math.isfinite(v) and v>=0 for v in values) or r['base_val_l1']<=0:
            raise ValueError('invalid validation loss/reference')
        cell=(u['dataset'],u['base'],u['lead'])
        if cell in references and references[cell]!=r['base_val_l1']:
            raise ValueError('base validation denominator changed between candidates')
        references[cell]=r['base_val_l1'];seen[key]=r
    if set(seen)!=set(expected):
        raise ValueError(f'incomplete validation grid: {len(seen)}/{len(expected)}')
    return list(seen.values())


def rank_architectures(rows):
    ranked=[]
    for arch in sorted({r['unit']['architecture'] for r in rows}):
        selected=[]
        d=rows[0]['unit']['dataset']
        for b,h,s in cells(d):
            pool=[r for r in rows if (r['unit']['architecture'],r['unit']['base'],r['unit']['lead'],r['unit']['seed'])==(arch,b,h,s)]
            if len(pool)!=3:
                raise ValueError('architecture ranking requires all three weight decays in 18 cells')
            selected.append(min(pool,key=inner_key))
        parameters={r['info']['parameters'] for r in selected}
        if len(parameters)!=1: raise ValueError('parameter count drift')
        ranked.append(dict(architecture=arch,parameters=next(iter(parameters)),
            mean_relative_val_l1=sum(r['info']['selected_val_l1']/r['base_val_l1'] for r in selected)/18,
            mean_val_coherence=sum(r['info'].get('val_coherence_fraction',0.) for r in selected)/18,
            units=[r['unit'] for r in selected]))
    return sorted(ranked,key=lambda r:(r['mean_relative_val_l1'],r['parameters'],r['architecture']))


def select(group, grid, rows, core_selection=None):
    """Validate complete coverage before freezing any winner; never refit."""
    rows=checked_rows(grid,rows)
    selected=[]; rankings=[]; architectures={}; deploy={}
    for d in DATASETS:
        for m in sorted({u['method'] for u in grid}):
            pool=[r for r in rows if (r['unit']['dataset'],r['unit']['method'])==(d,m)]
            kind=pool[0]['unit']['kind']
            if kind=='mlp':
                ranked=rank_architectures(pool)
                rankings.extend(dict(dataset=d,method=m,**r) for r in ranked)
                architectures[d+'/'+m]=ranked[0]['architecture']
                selected.extend(ranked[0]['units'])
            elif kind=='head':
                for b,h,s in cells(d):
                    selected.append(min([r for r in pool if (r['unit']['base'],r['unit']['lead'],r['unit']['seed'])==(b,h,s)],key=inner_key)['unit'])
            elif kind=='fixed': selected.extend(r['unit'] for r in pool)
            elif kind in ('rf','lightgbm','elasso'):
                def candidate(u):
                    return u['candidate'] if kind=='rf' else ((u['trees'],u['leaves']) if kind=='lightgbm' else u['alpha'])
                scores=[]
                for c in sorted({candidate(r['unit']) for r in pool}):
                    members=[r for r in pool if candidate(r['unit'])==c]
                    score=math.fsum(sorted(r['info']['selected_val_l1']/r['base_val_l1'] for r in members))/len(members)
                    if kind=='rf':
                        from src.baseline.random_forest import candidates,load_protocol,feature_count
                        p=load_protocol();v=candidates(p)[c]
                        tie=(-v['min_samples_leaf'],feature_count(v['feature_rule'],p['nodes'][d]),c)
                    else: tie=c if kind=='lightgbm' else (-c,)
                    scores.append((score,tie,members))
                    rankings.append(dict(dataset=d,method=m,candidate=c,mean_relative_val_l1=score))
                selected.extend(r['unit'] for r in min(scores,key=lambda v:v[:2])[2])
    if group=='penalty':
        zero=[u for u in core_selection['units'] if u['method']=='HYBN']
        # Zero is a reference to the already selected checkpoint, not a new fit.
        selected=zero+selected
        for d in DATASETS:
            zero_rank=next(r for r in core_selection['rankings'] if r.get('architecture')==core_selection['architectures'][d+'/HYBN'] and (r['dataset'],r['method'])==(d,'HYBN'))
            curve=[dict(zero_rank,penalty=0.)]+[dict(r,penalty=float(r['method'].split('lam')[1])) for r in rankings if r['dataset']==d]
            deploy[d]=min(curve,key=lambda r:(r['mean_relative_val_l1'],r['mean_val_coherence'],r['parameters'],r['penalty'],r['architecture']))['penalty']
    return dict(group=group,units=selected,architectures=architectures,rankings=rankings,
                deployable_lambda=deploy,test_used=False,refit=False)
