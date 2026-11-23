"""Fit/evaluate one final-paper strategy using previously generated base forecasts.

This reader example selects an epoch within one supplied configuration. It does
not replace the paper's shared architecture search across bases/leads/seeds.
Fit never reads test target or prediction arrays. Evaluate requires an explicit
selection-freeze acknowledgement and refuses to overwrite prior test output.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='stage', required=True)
    fit = sub.add_parser('fit', help='Fit on train and select an epoch on validation only')
    fit.add_argument('--config', required=True)
    fit.add_argument('--base', choices=('chronos2', 'timesfm25'), required=True)
    fit.add_argument('--forecasts', type=Path, required=True, help='Generated .npz and matching .meta.json')
    fit.add_argument('--strategy', choices=('BU','BUL','BUN','TD','TDL','TDN','HYB','HYBL','HYBN'), required=True)
    fit.add_argument('--lead', type=int, required=True)
    fit.add_argument('--hidden', type=int, nargs='+', default=[256,128])
    fit.add_argument('--weight-decay', type=float, choices=(0,0.0001,0.01), default=0)
    fit.add_argument('--seed', type=int, choices=(42,43,44), default=42)
    fit.add_argument('--device', default='cpu')
    controls = fit.add_mutually_exclusive_group()
    controls.add_argument('--coherence-lambda', type=float)
    controls.add_argument('--epsilon', type=float, choices=(0.005,0.01,0.02), help='HYBN budget as a fraction')
    fit.add_argument('--output', type=Path, required=True, help='New directory; existing fits are never overwritten')
    evaluate = sub.add_parser('evaluate', help='Evaluate the frozen selected checkpoint once')
    evaluate.add_argument('--checkpoint', type=Path, required=True, help='Directory written by fit')
    evaluate.add_argument('--device', default='cpu')
    evaluate.add_argument('--confirm-selection-frozen', action='store_true')
    return p


def fit_one(args):
    import torch
    from src.base_forecasts import check_base_dataset
    from src.data.datasets import load_config, load_dataset
    from src.strategies import fit_strategy
    from src.training.head_training import HeadTrainConfig
    from src.support.storage.forecasts import load_base_forecasts
    from src.support.paths.forecasts import resolve_base_path

    if args.output.exists():
        raise FileExistsError(f'Output exists: {args.output}; choose a new fit directory')
    config_hash = sha256(args.config)
    cfg = load_config(args.config)
    data = load_dataset(cfg, ROOT)
    check_base_dataset(args.base, data.name)
    if args.lead not in data.eval_steps:
        raise ValueError(f'Lead must be one of {data.eval_steps}')
    if args.strategy != 'HYBN' and (args.coherence_lambda is not None or args.epsilon is not None):
        raise ValueError('Coherence controls apply only to HYBN')
    if args.coherence_lambda is not None and args.coherence_lambda < 0:
        raise ValueError('Coherence penalty must be nonnegative')
    plan = json.loads((ROOT/'configs/experiments/mlp.json').read_text(encoding='utf-8'))
    if tuple(args.hidden) not in {tuple(x) for x in plan['architectures'].values()}:
        raise ValueError('Hidden widths must belong to the registered nine-architecture grid')
    source = Path(resolve_base_path(args.forecasts.resolve()))
    source_hash = sha256(source)
    metadata_hash = sha256(source.with_suffix('.meta.json'))
    arrays = load_base_forecasts(source, data, args.base, segments=('train','val'))
    k = args.lead - 1
    settings = HeadTrainConfig(hidden=tuple(args.hidden), weight_decay=args.weight_decay,
        seed=args.seed, device=args.device, lam=args.coherence_lambda, eps_target=args.epsilon)
    model, info = fit_strategy(args.strategy, data.hierarchy,
        arrays['yhat_train'][:,:,k], arrays['ytrue_train'][:,:,k],
        arrays['yhat_val'][:,:,k], arrays['ytrue_val'][:,:,k], settings)
    if (sha256(args.config) != config_hash or sha256(source) != source_hash
            or sha256(source.with_suffix('.meta.json')) != metadata_hash):
        raise ValueError('Forecast inputs changed while fitting')
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.output / 'weights.pt'
    torch.save({key: value.detach().cpu().clone() for key,value in model.state_dict().items()}, checkpoint)
    manifest = dict(scope='single_configuration_reader_example', strategy=args.strategy,
        config=str(Path(args.config).resolve()), config_sha256=config_hash,
        dataset=data.name, base=args.base, lead=args.lead, forecasts=str(source),
        forecast_sha256=source_hash, forecast_metadata_sha256=metadata_hash,
        settings=asdict(settings), selection=info, checkpoint_sha256=sha256(checkpoint),
        selection_data='validation_only', test_evaluated=False)
    (args.output/'fit.json').write_text(json.dumps(manifest,indent=2,default=str)+'\n',encoding='utf-8')
    print(json.dumps({'checkpoint':str(args.output.resolve()),'validation_selection':info,
                      'test_evaluated':False},indent=2,default=str))


def evaluate_one(args):
    if not args.confirm_selection_frozen:
        raise ValueError('Freeze all model choices before passing --confirm-selection-frozen')
    import numpy as np
    import torch
    from src.data.datasets import load_config, load_dataset
    from src.strategies import build_strategy
    from src.evaluation.prediction import apply_head
    from src.evaluation.metrics import level_metrics, mase_scale, seasonal_period, coherence_scale, coherence_nmae
    from src.support.storage.forecasts import load_base_forecasts

    output = args.checkpoint
    manifest_path = output/'fit.json'
    record = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (output/'test_evaluation.json').exists() or (output/'test_predictions.npz').exists():
        raise FileExistsError('This checkpoint already has test output; evaluation will not overwrite it')
    source = Path(record['forecasts'])
    for path, digest in ((Path(record['config']),record['config_sha256']),
                         (source,record['forecast_sha256']),
                         (source.with_suffix('.meta.json'),record['forecast_metadata_sha256']),
                         (output/'weights.pt',record['checkpoint_sha256'])):
        if sha256(path) != digest:
            raise ValueError(f'Frozen input/checkpoint changed: {path}')
    with (output/'test_evaluation.started.json').open('x',encoding='utf-8') as stream:
        json.dump({'fit_manifest_sha256':sha256(manifest_path)},stream)
    data = load_dataset(load_config(record['config']), ROOT)
    arrays = load_base_forecasts(source,data,record['base'],segments=('train','test'))
    k = record['lead'] - 1
    model = build_strategy(record['strategy'],data.hierarchy,arrays['yhat_train'][:,:,k],
        hidden=record['settings']['hidden'],dropout=record['settings']['dropout'])
    model.load_state_dict(torch.load(output/'weights.pt',map_location='cpu',weights_only=True),strict=True)
    model.eval()
    predicted = apply_head(model,arrays['yhat_test'][:,:,k],device=args.device)
    truth = arrays['ytrue_test'][:,:,k]
    metrics = level_metrics(predicted,truth,data.hierarchy,
        mase_scale(data.Y,data.train_end,seasonal_period(data.cfg['freq'])))
    coherence = coherence_nmae(predicted,data.hierarchy,
        coherence_scale(arrays['ytrue_train'][:,:,k],data.hierarchy))
    receipt = dict(scope=record['scope'],fit_manifest_sha256=sha256(manifest_path),
                   strategy=record['strategy'],base=record['base'],lead=record['lead'],
                   level_metrics=metrics,coherence_nmae=coherence)
    with (output/'test_predictions.npz').open('xb') as stream:
        np.savez_compressed(stream,predictions=predicted,origins=arrays['origins_test'])
    with (output/'test_evaluation.json').open('x',encoding='utf-8') as stream:
        json.dump(receipt,stream,indent=2,default=str)
    print(json.dumps(receipt,indent=2,default=str))


def main():
    args = parser().parse_args()
    (fit_one if args.stage == 'fit' else evaluate_one)(args)


if __name__ == '__main__':
    main()
