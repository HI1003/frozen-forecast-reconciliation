"""Training loop for learnable reconciliation heads on FROZEN base forecasts.

Loss (original scale, L1 throughout):
  coherent heads:  L = L_pred
  HYBN fixed-lam:  L = L_pred + lam * L_coh              (penalty form)
  HYBN-eps:        min L_pred  subject to  L_coh / s <= eps, represented by the
                   raw-scale Lagrangian
                     L_eps = L_pred + lam_t * (L_coh - eps * s).
                   Here s is mean |y| over aggregate nodes of the TRAIN
                   targets.  The projected dual update uses the equivalent
                   normalized residual L_coh / s - eps as a scale-independent
                   preconditioner, while both terms differentiated through the
                   head remain in the original data scale.
                   Model selection is constrained: best validation L1 among
                   epochs whose validation coherence nMAE satisfies the target
                   (before any epoch is feasible, the least-violating epoch is
                   tracked instead).

L_pred = mean |ytilde - ytrue|       (accuracy)
L_coh  = mean |ytilde - S b_tilde|   (aggregate rows; HYBN only)

Regularization: learnable heads use zero-initialised corrections around their
registered anchors. AdamW penalizes trainable correction parameters; early
stopping uses validation only. Both hybrid corrections start at the four-level
HYB anchor. Budget variants use the feasible-first rule described above.

Heads are trained per (dataset, base, step, head, seed) on the train origins of
the 80/10/10 split, model-selected on validation origins, and evaluated once on
test.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from src.strategies.common import ReconciliationHead
from src.data.hierarchy import Hierarchy
from src.training.model_selection import selection_key
from src.training.losses import (
    _dataset_l1, _epsilon_lagrangian, _normalized_constraint, _pred_and_coh,
)
from src.support.runtime.environment import set_seed


@dataclass
class HeadTrainConfig:
    lr: float = 1e-3
    batch_size: int = 128
    max_epochs: int = 200
    patience: int = 30
    weight_decay: float = 0.0         # decoupled (AdamW); regularizes trainable parameters
    hidden: tuple = (256, 128)
    dropout: float = 0.1
    # HYBN coherence control
    lam: float | None = None          # fixed penalty weight; None = pure accuracy
    eps_target: float | None = None   # target coherence nMAE (fraction); enables dual ascent
    dual_rho: float = 10.0            # dual-ascent step per unit of nMAE violation
    lam_max: float = 100.0
    device: str = "cpu"
    seed: int = 42
    extra: dict = field(default_factory=dict)


def train_head(
    head_name: str,
    hier: Hierarchy,
    yhat_train: np.ndarray, ytrue_train: np.ndarray,   # (N, n) one step, original scale
    yhat_val: np.ndarray, ytrue_val: np.ndarray,
    nonneg: bool,
    cfg: HeadTrainConfig,
    *, head_factory=None, record_history: bool = False,
) -> tuple[ReconciliationHead, dict]:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    x_mean, x_std = yhat_train.mean(axis=0), yhat_train.std(axis=0)
    if head_factory is None:
        from src.strategies.registry import _construct
        if not nonneg:
            raise ValueError('Final paper strategies require nonnegative reconciled outputs')
        head = _construct(head_name, hier, x_mean, x_std, yhat_train,
                          cfg.hidden, cfg.dropout).to(device)
    else:
        # A supplied factory lets direct/residual controls share the same loop.
        head = head_factory(x_mean, x_std).to(device)

    params = [p for p in head.parameters() if p.requires_grad]
    if not params:  # direct heads: nothing to train
        selected_train_l1 = _dataset_l1(
            head, yhat_train, ytrue_train, device, cfg.batch_size
        )
        selected_val_l1 = _dataset_l1(
            head, yhat_val, ytrue_val, device, cfg.batch_size
        )
        return head.eval(), {
            # Compatibility aliases: direct heads have no selected epoch, while
            # best_val remains the validation L1 used for result reporting.
            "epochs": 0,
            "best_val": selected_val_l1,
            "best_epoch": 0,
            "epochs_run": 0,
            "selected_train_l1": selected_train_l1,
            "selected_val_l1": selected_val_l1,
            "feasible": True,
        }

    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    Xtr = torch.tensor(yhat_train, dtype=torch.float32)
    Ttr = torch.tensor(ytrue_train, dtype=torch.float32)
    Xva = torch.tensor(yhat_val, dtype=torch.float32, device=device)
    Tva = torch.tensor(ytrue_val, dtype=torch.float32, device=device)

    # coherence normalization: mean |y| over AGGREGATE nodes of the TRAIN
    # targets — the same denominator metrics.coherence_scale uses, so the
    # epsilon target means the same thing in training, selection, and testing
    y_scale = max(float(np.abs(ytrue_train[:, np.asarray(hier.agg_idx)]).mean()), 1e-8)
    eps_mode = head_name == "HYBN" and cfg.eps_target is not None
    lam = 0.0 if eps_mode else float(cfg.lam or 0.0)
    penalized = head_name == "HYBN" and (eps_mode or lam > 0.0)

    best = {"val": float("inf"), "coh": float("inf"), "state": None,
            "epoch": 0, "feasible": False, "lambda": lam}
    bad = 0
    N = Xtr.shape[0]
    epochs_run = 0
    history = []

    for epoch in range(cfg.max_epochs):
        epoch_started = time.perf_counter()
        pred_sum = 0.0
        epochs_run = epoch + 1
        head.train()
        perm = torch.randperm(N)
        coh_sum, n_seen = 0.0, 0
        epoch_lam = lam
        for i in range(0, N, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            xb, tb = Xtr[idx].to(device), Ttr[idx].to(device)
            ytilde = head(xb)
            l_pred, l_coh = _pred_and_coh(head, ytilde, tb)
            if eps_mode:
                loss = _epsilon_lagrangian(
                    l_pred, l_coh, epoch_lam, y_scale, cfg.eps_target
                )
            elif penalized:
                loss = l_pred + epoch_lam * l_coh
            else:
                loss = l_pred
            opt.zero_grad()
            loss.backward()
            opt.step()
            coh_sum += float(l_coh.detach()) * len(idx)
            if record_history:
                pred_sum += float(l_pred.detach()) * len(idx)
            n_seen += len(idx)

        if eps_mode:
            train_coh = coh_sum / max(n_seen, 1)
            viol = _normalized_constraint(train_coh, y_scale, cfg.eps_target)
            lam = float(min(max(lam + cfg.dual_rho * viol, 0.0), cfg.lam_max))

        head.eval()
        with torch.no_grad():
            ytilde_v = head(Xva)
            val_pred = (ytilde_v - Tva).abs().mean().item()
            if head_name == "HYBN":
                recon = ytilde_v[:, head.bottom_idx] @ head.S.T
                val_coh = (ytilde_v - recon)[:, head.agg_idx].abs().mean().item() / y_scale
            else:
                val_coh = 0.0

        feasible = (not eps_mode) or (val_coh <= cfg.eps_target)
        candidate_key = selection_key(feasible, val_pred, val_coh)
        if record_history:
            if not np.isfinite([pred_sum, val_pred, val_coh]).all():
                raise ValueError('non-finite supplementary training trajectory')
            history.append(dict(epoch=epoch + 1,
                online_train_l1=pred_sum / max(n_seen, 1), val_l1=val_pred,
                val_coherence_fraction=val_coh, epoch_s=time.perf_counter() - epoch_started))
        improved = (best["state"] is None or candidate_key < selection_key(
            best["feasible"], best["val"], best["coh"]
        ))

        if improved:
            best = {"val": val_pred, "coh": val_coh,
                    "state": copy.deepcopy(head.state_dict()),
                    "epoch": epoch, "feasible": feasible, "lambda": epoch_lam}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best["state"] is not None:
        head.load_state_dict(best["state"])
    head.eval()

    selected_train_l1 = _dataset_l1(
        head, yhat_train, ytrue_train, device, cfg.batch_size
    )
    selected_val_l1 = _dataset_l1(
        head, yhat_val, ytrue_val, device, cfg.batch_size
    )

    best_epoch = best["epoch"] + 1
    info = {
        # `epochs` and `best_val` are retained for existing consumers. `epochs`
        # is the one-based selected epoch (not the number of epochs executed),
        # and `best_val` is the validation L1 recorded when that epoch won.
        "epochs": best_epoch,
        "best_val": best["val"],
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "selected_train_l1": selected_train_l1,
        "selected_val_l1": selected_val_l1,
        "feasible": best["feasible"],
    }
    if head_name == "HYBN":
        info["lambda"] = lam
        info["selected_lambda"] = best["lambda"]
        info["val_coh_nmae"] = best["coh"] * 100.0
    if record_history:
        info['history'] = history
    return head, info
