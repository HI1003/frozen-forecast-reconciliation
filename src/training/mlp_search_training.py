"""Fit one registered MLP search candidate using train/validation data only.

Unlike head_training.py's constrained HYBN-epsilon rule, this sealed paper search
selects the lowest validation prediction L1 and keeps the earliest epoch on a
tie. The candidate's architecture, weight decay and lambda come from its search
record; shared optimizer settings come from configs/experiments/mlp.json.
Candidate orchestration and artifact storage remain in scripts and support.
"""
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.ablation.direct_residual import make
from src.training.losses import _dataset_l1, _pred_and_coh
from src.support.runtime.environment import set_seed

P = json.loads((Path(__file__).resolve().parents[2] / 'configs/experiments/mlp.json').read_text(encoding='utf-8'))


def fit(c, data, x, y, xv, yv, *, epochs=None, device='cuda'):
    """Historical AdamW loop, with the registered prediction-only earliest-epoch tie."""
    set_seed(c['seed'])
    cfg = P['training']
    dev = torch.device(device)
    model = make(
        c['arm'], data, x.mean(0), x.std(0), c['hidden'], x, cfg['dropout']
    ).to(dev)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg['lr'], weight_decay=c['weight_decay'],
    )
    X = torch.tensor(x, dtype=torch.float32)
    Y = torch.tensor(y, dtype=torch.float32)
    XV = torch.tensor(xv, dtype=torch.float32, device=dev)
    YV = torch.tensor(yv, dtype=torch.float32, device=dev)
    scale = max(float(np.abs(y[:, data.hierarchy.agg_idx]).mean()), 1e-08)
    best = float('inf')
    state = None
    bad = 0
    history = []
    chosen = 0
    if dev.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for epoch in range(epochs or cfg['max_epochs']):
        t = time.perf_counter()
        model.train()
        perm = torch.randperm(len(X))
        running = 0.0
        for i in range(0, len(X), cfg['batch_size']):
            idx = perm[i:i + cfg['batch_size']]
            xb = X[idx].to(dev)
            yb = Y[idx].to(dev)
            pred = model(xb)
            l1, coh = _pred_and_coh(model, pred, yb)
            loss = l1 + c['lambda'] * coh
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(l1.detach()) * len(idx)
        model.eval()
        with torch.no_grad():
            p = model(XV)
            val = float((p - YV).abs().mean())
            coh = float(
                (p - p[:, model.bottom_idx] @ model.S.T)[:, model.agg_idx].abs().mean()
            ) / scale
        if not np.isfinite([val, coh, running]).all():
            raise ValueError('nonfinite training')
        history.append(dict(
            epoch=epoch + 1, online_train_l1=running / len(X), val_l1=val,
            val_coherence_fraction=coh, epoch_s=time.perf_counter() - t,
        ))
        if val < best:
            best = val
            state = copy.deepcopy(model.state_dict())
            chosen = epoch + 1
            bad = 0
        else:
            bad += 1
            if bad >= cfg['patience']:
                break
    model.load_state_dict(state)
    model.eval()
    train_l1 = _dataset_l1(model, x, y, dev, cfg['batch_size'])
    val_l1 = _dataset_l1(model, xv, yv, dev, cfg['batch_size'])
    if dev.type == 'cuda':
        torch.cuda.synchronize()
    info = dict(
        best_val=best, best_epoch=chosen, epochs_run=len(history),
        selected_train_l1=train_l1, selected_val_l1=val_l1,
        val_coherence_fraction=history[chosen - 1]['val_coherence_fraction'],
        train_s=time.perf_counter() - started,
        parameters=sum((p.numel() for p in model.parameters())), history=history,
    )
    return (model, info)
