"""Smoke tests for the frozen base-model inference paths.  Writes NOTHING to results/.

Levels:
  api          a handful of short contexts — verifies the adapter's API contract
               (Chronos return-structure normalization / TimesFM point==q50) in
               seconds; CPU is fine.
  integration  8 origins x ALL nodes at the configured context length — verifies
               shape, finiteness, non-negativity, throughput and peak GPU memory
               end-to-end.  Expected shapes: Walmart (8, 44, 14),
               GEFCom (8, 158, 24).
  pair         runs BOTH main bases over the same 8 origins and asserts the
               origins and ground-truth targets are bit-identical, which is the
               precondition for the RQ-D cross-base ranking comparison.

Usage:
  python scripts/checks/check_base_forecasts.py --config configs/data/walmart.yaml --base chronos2  --level api
  python scripts/checks/check_base_forecasts.py --config configs/data/walmart.yaml --base timesfm25 --level integration --device cuda
  python scripts/checks/check_base_forecasts.py --config configs/data/gefcom2017f.yaml --level pair --device cuda
"""

# Resolve the approved project layout while retaining sealed source-path keys.
from pathlib import Path as _ProjectPath
import sys as _project_sys
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
_project_sys.path.insert(0, str(_PROJECT_ROOT))
from src.support.paths.project import project_file

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_file(ROOT)))

from src.base_forecasts.registry import MAIN_BASES, check_base_dataset
from src.evaluation.base_checks import assert_forecasts_distinct
from src.data.datasets import load_config, load_dataset
from src.support.runtime.environment import env_report, peak_gpu_mib, pick_device, reset_peak_gpu


def _run_full(data, base_key, origins, device, batch_size):
    """Full-node forecasts for `origins` under one registered base.

    Returns (yhat, ytrue, load_seconds, infer_seconds).  Model loading is timed
    SEPARATELY from inference: it is a fixed one-off cost, and amortising it
    over a handful of smoke origins would inflate the ms/context used to size
    the production run.
    """
    spec = check_base_dataset(base_key, data.name)
    if spec.family == "chronos":
        from src.base_forecasts.chronos2 import generate_chronos_forecasts, load_pipeline

        bs = batch_size or 256
        t0 = time.time()
        pipe = load_pipeline(spec.model_id, device, revision=spec.revision)
        load_s = time.time() - t0
        t0 = time.time()
        yhat, ytrue = generate_chronos_forecasts(
            data, origins, model_id=spec.model_id, device=device,
            batch_size=bs, pipe=pipe)
        return yhat, ytrue, load_s, time.time() - t0

    from src.base_forecasts.timesfm25 import generate_timesfm_forecasts, load_model

    bs = batch_size or 32
    t0 = time.time()
    model = load_model(spec.model_id, device=device, context_len=data.context_len,
                       predict_len=data.predict_len, batch_size=bs,
                       revision=spec.revision)
    load_s = time.time() - t0
    t0 = time.time()
    yhat, ytrue = generate_timesfm_forecasts(
        data, origins, model_id=spec.model_id, device=device,
        batch_size=bs, model=model)
    return yhat, ytrue, load_s, time.time() - t0


def _api_smoke(data, base_key, device):
    """Cheapest possible contract check: 2 origins x 3 nodes, short context."""
    spec = check_base_dataset(base_key, data.name)
    H = data.predict_len
    origins = data.origins().val[:2]
    L = min(64, data.context_len)
    ctx = [data.Y[t + 1 - L : t + 1, i] for t in origins for i in range(3)]

    t0 = time.time()
    if spec.family == "chronos":
        import torch

        from src.base_forecasts.chronos2 import _predict_quantile_batch, load_pipeline

        pipe = load_pipeline(spec.model_id, device, revision=spec.revision)
        q = _predict_quantile_batch(
            pipe, [torch.tensor(c, dtype=torch.float32) for c in ctx], H
        )
        assert q.shape == (len(ctx), H), f"shape {q.shape}"
        assert np.isfinite(q).all(), "non-finite forecasts"
        extra = ""
    else:
        from src.base_forecasts.timesfm25 import _forecast_batch, load_model

        model = load_model(spec.model_id, device=device, context_len=L,
                           predict_len=H, batch_size=len(ctx),
                           revision=spec.revision)
        point, quant = _forecast_batch(
            model, [np.asarray(c, dtype=np.float32) for c in ctx], H
        )
        assert point.shape == (len(ctx), H), f"shape {point.shape}"
        assert np.isfinite(point).all(), "non-finite forecasts"
        extra = f"; quantiles {quant.shape}; point == q50 verified"
    print(f"API smoke PASS [{base_key}]: {len(ctx)} contexts -> "
          f"({len(ctx)}, {H}){extra} in {time.time()-t0:.1f}s")


def _integration_smoke(data, base_key, device, batch_size, n_origins):
    n, H = data.hierarchy.n, data.predict_len
    origins = data.origins().val[:n_origins]
    # Fresh measurement window: in --level pair two models run in one process,
    # and CUDA's peak counter is cumulative, so without this the second base
    # would report the maximum over both. These figures feed the paper's
    # efficiency table.
    reset_peak_gpu(device)
    yhat, ytrue, load_s, infer_s = _run_full(data, base_key, origins, device, batch_size)

    assert yhat.shape == (len(origins), n, H), f"shape {yhat.shape}"
    assert ytrue.shape == yhat.shape
    assert np.isfinite(yhat).all(), "non-finite forecasts"
    per_ctx = infer_s / (len(origins) * n)
    full = data.origins()
    total_ctx = (full.train.size + full.val.size + full.test.size) * n
    print(f"integration smoke PASS [{base_key}]: {yhat.shape} in {infer_s:.1f}s "
          f"inference + {load_s:.1f}s load "
          f"({per_ctx*1000:.1f} ms/context; min {yhat.min():.4g}, max {yhat.max():.4g}); "
          f"peak GPU {peak_gpu_mib(device)} MiB; full run ~"
          f"{(per_ctx * total_ctx + load_s) / 60:.1f} min "
          f"(from {len(origins)} warm-up-dominated origins — an upper bound)")
    return origins, yhat, ytrue


def _pair_smoke(data, device, batch_size, n_origins):
    """Both main bases over identical origins — the RQ-D precondition."""
    out = {}
    for base_key in MAIN_BASES:
        out[base_key] = _integration_smoke(data, base_key, device, batch_size, n_origins)
    a, b = MAIN_BASES[0], MAIN_BASES[1]
    assert np.array_equal(out[a][0], out[b][0]), "origins differ between main bases"
    assert np.array_equal(out[a][2], out[b][2]), "ytrue differs between main bases"
    diff = assert_forecasts_distinct(out[a][1], out[b][1], where="pair smoke")
    print(f"pair smoke PASS: {a} and {b} share bit-identical origins and ytrue; "
          f"mean |yhat_{a} - yhat_{b}| = {diff:.4g} (they should differ)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base", default="chronos2")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--origins", type=int, default=8,
                    help="origins used by the integration/pair levels")
    ap.add_argument("--level", choices=["api", "integration", "pair"], default="api")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data = load_dataset(cfg, ROOT)
    device = pick_device(args.device)
    print(f"environment: {json.dumps(env_report(), default=str)}")
    print(f"dataset {data.name}: context {data.context_len}, H {data.predict_len}, "
          f"n {data.hierarchy.n}, device {device}")

    if args.level == "api":
        _api_smoke(data, args.base, device)
    elif args.level == "integration":
        _integration_smoke(data, args.base, device, args.batch_size, args.origins)
    else:
        _pair_smoke(data, device, args.batch_size, args.origins)


if __name__ == "__main__":
    main()
