"""Step 0: validate a dataset against the framework protocol.

Usage:  python scripts/data/validate_data.py --config configs/data/walmart.yaml
"""

# Resolve the approved project layout while retaining sealed source-path keys.
from pathlib import Path as _ProjectPath
import sys as _project_sys
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
_project_sys.path.insert(0, str(_PROJECT_ROOT))
from src.support.paths.project import project_file

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_file(ROOT)))

from src.data.datasets import SPLIT_PROTOCOL, assert_regular_time_grid, load_config, load_dataset
from src.data.hierarchy import LEVEL_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    data = load_dataset(cfg, ROOT)
    h = data.hierarchy
    # Defensive hard gate: step0 must never print OK for an irregular timeline,
    # even if a future loader refactor accidentally weakens its own check.
    assert_regular_time_grid(
        data.timestamps, cfg["freq"], where=f"step0:{data.name}"
    )

    print(f"dataset          : {data.name}")
    print(f"rows (T)         : {data.T}   {data.timestamps[0]} .. {data.timestamps[-1]}")
    diffs = data.timestamps.to_series().diff().dropna().unique()
    print(f"time step(s)     : {[str(d) for d in diffs[:3]]}{' (IRREGULAR!)' if len(diffs) > 1 else ''}")
    print(f"nodes (n/m)      : {h.n}/{h.m}")
    for lvl in LEVEL_NAMES:
        print(f"  level {lvl:<8}: {h.level_index[lvl].size} nodes")
    gap_policy = ("interpolate missing rows" if bool(cfg.get("fill_gaps", True))
                  else "fail on any missing row")
    print(f"runtime gaps     : {data.gaps_filled} rows added; policy = {gap_policy}")
    print(f"values fingerprint: {data.values_fingerprint} (semantic timestamp/names/float64)")
    print(
        "source gates     : "
        f"exact_schema={bool(cfg.get('require_exact_source_schema', False))}, "
        f"coherent={bool(cfg.get('require_source_coherence', False))}"
    )
    exclusion_counts = data.origin_exclusion_counts()
    print(
        f"imputed exclusions: {len(data.excluded_imputed_timestamps)} declared rows; "
        f"origins removed train/val/test = "
        f"{exclusion_counts['train']}/{exclusion_counts['val']}/{exclusion_counts['test']}"
    )
    print(
        f"source-vs-S dev  : {data.raw_aggregate_dev:.3e} "
        "(aggregate-relative; Y is defensively rebuilt from bottom via S)"
    )
    print(f"negative values  : {(data.Y < 0).sum()} entries; zero share {(data.Y == 0).mean():.4%}")
    print(f"split protocol   : {SPLIT_PROTOCOL}")
    print(f"splits (rows)    : train [0,{data.train_end}) val [{data.train_end},{data.val_end}) test [{data.val_end},{data.T})")
    print(f"                   {data.train_end} / {data.val_end - data.train_end} / {data.T - data.val_end} time points")

    o = data.origins()
    stride = data.cfg.get("origin_stride", {}) or {}
    print(f"origins          : train {o.train.size}, val {o.val.size}, test {o.test.size}  (stride {stride})")
    print(f"                   first train origin {o.train[0]} = context_len-1; last test origin {o.test[-1]}")
    print(f"eval steps       : {data.eval_steps} (predict_len {data.predict_len}, context {data.context_len})")
    print("OK")


if __name__ == "__main__":
    main()
