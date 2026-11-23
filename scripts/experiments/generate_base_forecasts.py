"""Generate and freeze one base inside a new immutable experiment ID.

Reader example (the sealed ``main`` namespace is read-only):
  python scripts/experiments/generate_base_forecasts.py --config configs/data/walmart.yaml \
      --base chronos2 --experiment-id dev_walmart --device cuda

Sensitivity and appendix runs must use a different ID. A sensitivity-role base
also needs ``--allow-sensitivity``. Existing base artifacts are never
overwritten; a rerun uses a new experiment ID.
"""

# Resolve the approved project layout while retaining sealed source-path keys.
from pathlib import Path as _ProjectPath
import sys as _project_sys
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
_project_sys.path.insert(0, str(_PROJECT_ROOT))
from src.support.paths.project import project_file

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_file(ROOT)))

from src.support.storage.forecast_artifacts import MAIN_EXPERIMENT_ID
from src.support.storage.forecast_artifacts import base_forecast_path, claim_experiment_namespace, ensure_base_sidecar, load_base_artifact, save_base_artifact
from src.base_forecasts.registry import BASE_MODELS, MAIN_BASES, check_base_dataset
from src.data.datasets import SPLIT_PROTOCOL, load_config, load_dataset
from src.support.runtime.environment import checkpoint_revision, env_report, peak_gpu_mib, pick_device, reset_peak_gpu
from src.support.storage.forecasts import compute_base_fingerprint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base", choices=sorted(BASE_MODELS), required=True)
    ap.add_argument("--experiment-id", required=True,
                    help="new immutable output namespace; main is reserved")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="contexts per forward pass (default 256 chronos / 32 timesfm)")
    ap.add_argument("--origin-chunk", type=int, default=64)
    ap.add_argument("--torch-compile", action="store_true",
                    help="TimesFM only: opt-in torch.compile of the decode step")
    ap.add_argument("--allow-sensitivity", action="store_true",
                    help="required to run a sensitivity-role base")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data = load_dataset(cfg, ROOT)
    spec = check_base_dataset(args.base, data.name)
    if spec.role != "main" and not args.allow_sensitivity:
        raise SystemExit(
            f"'{spec.key}' is a {spec.role} base, not part of the main matrix "
            f"({', '.join(MAIN_BASES)}). Pass --allow-sensitivity deliberately."
        )
    if spec.role != "main" and args.experiment_id == MAIN_EXPERIMENT_ID:
        raise SystemExit(
            "a sensitivity base can never write into experiment_id='main'; "
            "choose a separate sensitivity namespace"
        )
    if args.experiment_id == MAIN_EXPERIMENT_ID:
        raise ValueError("The sealed main namespace is read-only; choose a new experiment ID")

    claim = claim_experiment_namespace(
        ROOT, args.experiment_id, cfg, data.name, create=True
    )
    out = base_forecast_path(ROOT, args.experiment_id, data.name, spec.key)
    sidecar = out.with_suffix(".meta.json")
    if project_file(out).exists():
        ensure_base_sidecar(out)
        load_base_artifact(ROOT, args.experiment_id, data, spec.key)
        print(f"resume verified existing immutable base -> {out}")
        return
    if project_file(sidecar).exists():
        raise SystemExit(
            f"orphaned base publication: npz={project_file(out).exists()} sidecar={project_file(sidecar).exists()}. "
            "Do not rerun inference or overwrite it; inspect the failed namespace."
        )

    device = pick_device(args.device)
    reset_peak_gpu(device)
    origins = data.origins()
    batch_size = args.batch_size or (32 if spec.family == "timesfm" else 256)

    meta = {
        "experiment_id": args.experiment_id,
        "config_fingerprint": claim["config_fingerprint"],
        "values_fingerprint": data.values_fingerprint,
        "dataset": data.name,
        "base": spec.key,
        "model_id": spec.model_id,
        "checkpoint_revision": None,
        "repo": claim["repo"],
        "family": spec.family,
        "role": spec.role,
        "split_protocol": SPLIT_PROTOCOL,
        "gradient_updates": 0,
        "point_statistic": "q50",
        "context_len": data.context_len,
        "predict_len": data.predict_len,
        "eval_steps": data.eval_steps,
        "n_nodes": data.hierarchy.n,
        "origin_stride": cfg.get("origin_stride"),
        "n_origins": {s: int(getattr(origins, s).size)
                      for s in ("train", "val", "test")},
        "device": device,
        "batch_size": batch_size,
        "origin_chunk": args.origin_chunk,
        "environment": env_report(),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if spec.family == "chronos":
        from src.base_forecasts.chronos2 import generate_chronos_forecasts, load_pipeline

        t0 = time.time()
        pipe = load_pipeline(spec.model_id, device, revision=spec.revision)
        meta["model_load_s"] = round(time.time() - t0, 1)
        meta["pipeline_class"] = type(pipe).__name__

        def run(segment):
            return generate_chronos_forecasts(
                data, segment, model_id=spec.model_id, device=device,
                batch_size=batch_size, origin_chunk=args.origin_chunk, pipe=pipe,
            )
    else:
        from src.base_forecasts.timesfm25 import FORECAST_FLAGS, compile_state, context_padding_info, generate_timesfm_forecasts, load_model

        t0 = time.time()
        model = load_model(spec.model_id, device=device, context_len=data.context_len,
                           predict_len=data.predict_len, batch_size=batch_size,
                           torch_compile=args.torch_compile,
                           revision=spec.revision)
        meta["model_load_s"] = round(time.time() - t0, 1)
        meta["compilation"] = compile_state(model)
        meta["context_padding"] = context_padding_info(data.context_len)
        # Retained as a top-level compatibility field for existing metadata
        # readers; `context_padding` is the complete auditable record.
        meta["compiled_max_context"] = meta["context_padding"]["compiled_max_context"]
        meta["forecast_config"] = dict(FORECAST_FLAGS)

        def run(segment):
            return generate_timesfm_forecasts(
                data, segment, model_id=spec.model_id, device=device,
                batch_size=batch_size, origin_chunk=args.origin_chunk, model=model,
            )

    meta["checkpoint_revision"] = spec.revision or checkpoint_revision(spec.model_id)
    if args.experiment_id == MAIN_EXPERIMENT_ID and not meta["checkpoint_revision"]:
        raise SystemExit(
            f"cannot resolve the immutable checkpoint SHA for {spec.model_id}; "
            "formal main artifacts require it"
        )

    payload, timings = {}, {}
    for seg in ("train", "val", "test"):
        segment = getattr(origins, seg)
        t0 = time.time()
        yhat, ytrue = run(segment)
        dt = time.time() - t0
        payload[f"origins_{seg}"] = segment
        payload[f"yhat_{seg}"] = yhat
        payload[f"ytrue_{seg}"] = ytrue
        n_ctx = int(segment.size) * data.hierarchy.n
        timings[seg] = {
            "seconds": round(dt, 1), "contexts": n_ctx,
            "ms_per_context": round(1000.0 * dt / max(n_ctx, 1), 3),
        }
        print(f"[{seg}] {segment.size} origins x {data.hierarchy.n} nodes "
              f"-> {yhat.shape} in {dt:.1f}s "
              f"({timings[seg]['ms_per_context']} ms/context)")

    meta["timings"] = timings
    meta["peak_gpu_mib"] = peak_gpu_mib(device)
    meta["nonfinite_check"] = "per-batch in adapter (check_forecast_block)"
    meta["fingerprint"] = compute_base_fingerprint(payload)
    save_base_artifact(out, payload, meta)
    print(f"frozen -> {out}  (peak GPU {meta['peak_gpu_mib']} MiB)")


if __name__ == "__main__":
    main()
