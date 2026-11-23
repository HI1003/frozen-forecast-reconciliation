"""Frozen-forecast IO with exact identity, split, target and fingerprint checks.

Requested segments alone are decompressed; all origin arrays are checked.
Failed identity or data checks abort rather than substitute another cache.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.support.paths.project import project_file
from src.support.paths.forecasts import base_forecast_path as _base_forecast_path, resolve_base_path
from src.support.runtime.environment import env_report
from src.base_forecasts.registry import BASE_MODELS
from src.data.datasets import SPLIT_PROTOCOL, HTSData, SplitOrigins, targets_for_origins

FINGERPRINT_ARRAYS = tuple(
    f"{kind}_{seg}"
    for seg in ("train", "val", "test")
    for kind in ("origins", "yhat", "ytrue")
)


def check_forecast_block(arr: np.ndarray, expected_shape: tuple[int, ...], *,
                         nonneg: bool, where: str, kind: str = "forecast") -> None:
    """Reject an array block before it can reach disk or be trusted on load.

    Non-finite output is a hard failure, not something to impute: a NaN batch
    means the model/runtime combination is broken (TimesFM fused attention under
    PyTorch 2.3 is the known case), and saving it would poison every downstream
    reconciliation result.  The environment report goes into the message.

    `kind` only shapes the negative-value diagnostic, which means different
    things for the two array families this guards: base forecasts are
    deliberately unclipped in this study, so a negative FORECAST is expected and
    only flagged when a caller explicitly demands otherwise, whereas a negative
    GROUND TRUTH value points at the data pipeline and never at a model adapter.
    """
    if arr.shape != tuple(expected_shape):
        raise ValueError(f"{where}: shape {arr.shape} != expected {tuple(expected_shape)}")
    if not np.isfinite(arr).all():
        bad = int((~np.isfinite(arr)).sum())
        raise ValueError(
            f"{where}: {bad}/{arr.size} non-finite {kind} values — refusing to "
            f"save or reuse contaminated arrays.\nenvironment: "
            f"{json.dumps(env_report(), indent=2, default=str)}"
        )
    if nonneg and float(arr.min()) < 0.0:
        n_neg, worst = int((arr < 0).sum()), float(arr.min())
        if kind == "actuals":
            raise ValueError(
                f"{where}: {n_neg} negative ground-truth values (min {worst:.6g}) "
                "on a dataset declared non-negative. This is a data-pipeline "
                "issue — check the values CSV, the node exclusions recorded in "
                "hierarchy_info.json, and the `nonneg` config flag."
            )
        raise ValueError(
            f"{where}: {n_neg} negative {kind} values (min {worst:.6g}) where the "
            "caller required non-negativity. Base forecasts are NOT clipped in "
            "this study, so requiring non-negativity of raw base output is "
            "probably the wrong expectation."
        )


def base_forecast_path(root: str | Path, dataset: str, base: str) -> Path:
    return _base_forecast_path(root, 'main', dataset, base)


def save_base_forecasts(path: str | Path, payload: dict, meta: dict) -> None:
    path = Path(path)
    project_file(path.parent).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(project_file(path), **payload)
    project_file(path.with_suffix(".meta.json")).write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )


def _check_base_identity(meta: dict, base: str, path: Path, dataset: str) -> None:
    """The file must be the forecasts of the base the caller asked for.

    Without this, `{dataset}_timesfm25.npz` can hold Chronos forecasts and every
    downstream table — including the RQ-D cross-base ranking, whose entire claim
    is that two DIFFERENT models were compared — silently reads as valid.
    """
    if meta.get("base") != base:
        raise ValueError(
            f"{path.name}: metadata base '{meta.get('base')}' != requested '{base}'"
        )
    allowed_names = {f"{dataset}_{base}.npz"}  # sealed historical filename
    experiment_id = meta.get('experiment_id')
    if experiment_id is not None:
        allowed_names.add(_base_forecast_path(Path('.'), experiment_id, dataset, base).name)
    if path.name not in allowed_names:
        raise ValueError(
            f"{path.name}: filename does not match dataset '{dataset}' and base '{base}'"
        )
    if meta.get("point_statistic") != "q50":
        raise ValueError(
            f"{path.name}: point_statistic '{meta.get('point_statistic')}' != 'q50'; "
            "the bases must share one point statistic to be comparable"
        )
    spec = BASE_MODELS.get(base)
    if spec is None:
        return
    if meta.get("model_id") != spec.model_id:
        raise ValueError(
            f"{path.name}: model_id '{meta.get('model_id')}' != registry entry "
            f"'{spec.model_id}' for base '{base}'"
        )
    if meta.get("role") != spec.role:
        raise ValueError(
            f"{path.name}: role '{meta.get('role')}' != registry role '{spec.role}'"
        )


def load_base_forecasts(path: str | Path, data: HTSData, base: str,
                        origins: SplitOrigins | None = None,
                        segments: tuple[str, ...] = ("train", "val", "test")) -> dict:
    """Load frozen forecasts and verify they belong to THIS dataset, base and split.

    Raises on any mismatch rather than warning: every check below has a failure
    mode that produces plausible-looking but meaningless reconciliation tables.

    `segments` limits which yhat/ytrue arrays are read and re-verified.  The
    origin arrays of ALL three segments are always checked — they are tiny and
    they are what proves the split geometry — but a read-only analysis that
    only touches the test block should not pay to decompress and re-derive the
    training targets (~2.9 GB across both GEFCom bases).
    """
    path = resolve_base_path(path)
    if origins is None:
        origins = data.origins()
    unknown = set(segments) - {"train", "val", "test"}
    if unknown:
        raise ValueError(f"unknown segments {sorted(unknown)}")

    meta_path = path.with_suffix(".meta.json")
    if not project_file(meta_path).exists():
        raise FileNotFoundError(f"{path}: metadata sidecar {meta_path.name} is missing")
    meta = json.loads(project_file(meta_path).read_text(encoding="utf-8"))
    _check_base_identity(meta, base, path, data.name)
    stored_protocol = meta.get("split_protocol")
    if stored_protocol != SPLIT_PROTOCOL:
        raise ValueError(
            f"{path.name}: split_protocol '{stored_protocol}' != '{SPLIT_PROTOCOL}'. "
            "This file predates the strict 80/10/10 protocol; regenerate it with "
            "scripts/experiments/generate_base_forecasts.py."
        )
    if meta.get("dataset") != data.name:
        raise ValueError(f"{path.name}: dataset '{meta.get('dataset')}' != '{data.name}'")
    if int(meta.get("context_len", -1)) != data.context_len:
        raise ValueError(
            f"{path.name}: context_len {meta.get('context_len')} != {data.context_len}"
        )
    if int(meta.get("predict_len", -1)) != data.predict_len:
        raise ValueError(
            f"{path.name}: predict_len {meta.get('predict_len')} != {data.predict_len}"
        )
    if int(meta.get("gradient_updates", -1)) != 0:
        raise ValueError(
            f"{path.name}: metadata reports {meta.get('gradient_updates')} gradient "
            "updates; this study only admits frozen base models"
        )

    n, H = data.hierarchy.n, data.predict_len
    nonneg = bool(data.cfg.get("nonneg", True))
    fz = {}

    def validated_origins(raw: np.ndarray, key: str) -> np.ndarray:
        """Validate storage type and range before any lossy int64 cast."""
        arr = np.asarray(raw)
        if np.issubdtype(arr.dtype, np.number) and not np.isfinite(arr).all():
            raise ValueError(f"{path.name}: {key} contains non-finite values")
        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(
                f"{path.name}: {key} must have an integer dtype; got {arr.dtype}"
            )
        if arr.size:
            lo, hi = int(arr.min()), int(arr.max())
            limits = np.iinfo(np.int64)
            if lo < limits.min or hi > limits.max:
                raise ValueError(f"{path.name}: {key} values exceed int64 range")
        return arr.astype(np.int64, copy=False)

    with np.load(project_file(path), allow_pickle=False) as z:
        for seg in ("train", "val", "test"):
            want = getattr(origins, seg)
            key = f"origins_{seg}"
            if key not in z.files:
                raise KeyError(f"{path.name}: missing array {key}")
            got = validated_origins(z[key], key)
            if got.shape != want.shape or not np.array_equal(got, want):
                raise ValueError(
                    f"{path.name}: origins_{seg} ({got.size} entries) do not match the "
                    f"current config ({want.size} entries) — the split, stride, or "
                    "context length changed; regenerate the frozen forecasts."
                )
            fz[key] = got
            if seg not in segments:
                continue
            expected = (want.size, n, H)
            yhat, ytrue = z[f"yhat_{seg}"], z[f"ytrue_{seg}"]
            check_forecast_block(yhat, expected, nonneg=False,
                                 where=f"{path.name}:yhat_{seg}", kind="forecast")
            check_forecast_block(ytrue, expected, nonneg=nonneg,
                                 where=f"{path.name}:ytrue_{seg}", kind="actuals")
            if not np.array_equal(ytrue, targets_for_origins(data.Y, want, H)):
                raise ValueError(
                    f"{path.name}: ytrue_{seg} does not match the targets rebuilt from "
                    "the current data; the values CSV or hierarchy changed."
                )
            fz[f"yhat_{seg}"], fz[f"ytrue_{seg}"] = yhat, ytrue

    # Whenever the whole payload is in hand, confirm the sidecar fingerprint
    # still describes it — otherwise the cheap `stored_base_fingerprint` path
    # used by later stages could be trusting a stale or hand-edited sidecar.
    if set(segments) == {"train", "val", "test"}:
        recomputed = compute_base_fingerprint(fz)
        declared = meta.get("fingerprint")
        if not declared:
            raise ValueError(
                f"{path.name}: no `fingerprint` in metadata — this base predates "
                "the lineage-bound format; regenerate it with step1."
            )
        if declared != recomputed:
            raise ValueError(
                f"{path.name}: metadata fingerprint {declared[:12]}… does not match "
                f"the stored arrays ({recomputed[:12]}…); the .npz and its sidecar "
                "have diverged."
            )
    return fz


def compute_base_fingerprint(arrays: dict) -> str:
    """Content hash of a complete frozen-forecast payload.

    Covers all three segments, so a product built under different training or
    selection conditions can never pass as current.  Shapes and dtypes are
    folded in, and values are canonicalised (origins int64, forecasts float64)
    so the hash does not depend on incidental storage types.
    """
    missing = [k for k in FINGERPRINT_ARRAYS if k not in arrays]
    if missing:
        raise KeyError(f"cannot fingerprint an incomplete payload; missing {missing}")
    h = hashlib.sha256()
    for key in FINGERPRINT_ARRAYS:
        dtype = np.int64 if key.startswith("origins") else np.float64
        arr = np.ascontiguousarray(arrays[key], dtype=dtype)
        h.update(key.encode("utf-8"))
        h.update(str(arr.shape).encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()[:32]


def stored_base_fingerprint(root: str | Path, dataset: str, base: str) -> str:
    """The fingerprint step1 recorded, read from the metadata sidecar alone.

    Lets step2/step3/step4 bind their outputs to a base without decompressing
    ~3 GB of train/val arrays just to recompute a hash.  `load_base_forecasts`
    re-derives and re-checks it whenever it reads all three segments, so the
    sidecar cannot drift away from the .npz unnoticed.
    """
    path = resolve_base_path(base_forecast_path(root, dataset, base).with_suffix(".meta.json"))
    if not project_file(path).exists():
        raise FileNotFoundError(f"missing metadata sidecar {path.name}")
    meta = json.loads(project_file(path).read_text(encoding="utf-8"))
    fp = meta.get("fingerprint")
    if not fp:
        raise ValueError(
            f"{path.name}: no `fingerprint` — this base predates the lineage-bound "
            "format; re-run scripts/experiments/generate_base_forecasts.py for it."
        )
    return str(fp)
