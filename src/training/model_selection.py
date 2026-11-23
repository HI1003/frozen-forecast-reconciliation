"""Validation-only selection rules shared by epochs and weight-decay runs."""
from __future__ import annotations

import math


def selection_key(feasible: bool, val_l1: float,
                  val_coh_nmae: float) -> tuple[float, ...]:
    """Lower is better under the frozen HYBN-epsilon rule.

    Feasible candidates always win. Among feasible candidates validation L1 is
    primary; if all candidates are infeasible, coherence violation is primary
    and validation L1 breaks a tie.
    """
    val_l1 = float(val_l1)
    val_coh_nmae = float(val_coh_nmae)
    if not math.isfinite(val_l1) or not math.isfinite(val_coh_nmae):
        raise ValueError("selection metrics must be finite")
    return ((0.0, val_l1, val_coh_nmae) if feasible
            else (1.0, val_coh_nmae, val_l1))


def info_selection_key(info: dict) -> tuple[float, ...]:
    return selection_key(bool(info["feasible"]), float(info["best_val"]),
                         float(info.get("val_coh_nmae", 0.0)))
