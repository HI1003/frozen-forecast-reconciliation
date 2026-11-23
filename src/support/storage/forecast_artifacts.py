"""Immutable frozen-forecast namespaces, fingerprints and atomic publication."""
from __future__ import annotations
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any
import numpy as np
from src.support.storage.locking import lock_file, unlock_file
from src.support.paths.project import project_file
from src.support.paths.forecasts import base_forecast_path as _base_forecast_path
from src.support.runtime.environment import repo_state
from src.support.storage.forecasts import load_base_forecasts
from src.data.datasets import SPLIT_PROTOCOL, HTSData
from src.base_forecasts.registry import get_base
MAIN_EXPERIMENT_ID = 'main'
EXPERIMENT_ID_RE = re.compile(r'^[a-z0-9][a-z0-9._-]{0,63}$')

def validate_experiment_id(value: str) -> str:
    """Return a safe experiment ID or fail before constructing any path."""
    value = str(value)
    if not EXPERIMENT_ID_RE.fullmatch(value) or value in {".", ".."}:
        raise ValueError(
            "experiment_id must match [a-z0-9][a-z0-9._-]{0,63}; "
            f"got {value!r}"
        )
    return value

def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items())
                if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    return value

def canonical_config(cfg: dict) -> dict:
    """Scientific config without runtime-only keys such as ``_config_path``."""
    return _canonical(cfg)

def config_fingerprint(cfg: dict) -> str:
    raw = json.dumps(canonical_config(cfg), sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]

def experiment_root(root: str | Path, experiment_id: str) -> Path:
    return Path(root) / "results" / "experiments" / validate_experiment_id(experiment_id)

def base_forecast_path(root: str | Path, experiment_id: str,
                       dataset: str, base: str) -> Path:
    return _base_forecast_path(root, validate_experiment_id(experiment_id), dataset, base)

def _write_json_atomic(path: Path, value: dict) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp",
                                    dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            json.dump(value, tmp, indent=2, sort_keys=True, ensure_ascii=False,
                      default=str)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)

def claim_experiment_namespace(root: str | Path, experiment_id: str, cfg: dict,
                               dataset: str, *, create: bool) -> dict:
    """Create or validate a commit/config-pinned namespace under a file lock."""
    experiment_id = validate_experiment_id(experiment_id)
    state = repo_state(root)
    if not state.get("commit"):
        raise RuntimeError("cannot claim an experiment outside an identified git commit")
    if state.get("dirty") is not False:
        raise RuntimeError(
            "production experiment namespaces require a clean worktree; commit "
            "the implementation or use an already clean worktree first"
        )

    exp_root = experiment_root(root, experiment_id)
    manifest_path = exp_root / "experiment.meta.json"
    if create:
        project_file(exp_root).mkdir(parents=True, exist_ok=True)
    elif not project_file(manifest_path).exists():
        raise FileNotFoundError(
            f"experiment namespace '{experiment_id}' has no manifest; run step1 first"
        )

    lock_path = exp_root / ".namespace.lock"
    with project_file(lock_path).open("a+", encoding="utf-8") as lock:
        lock_file(lock)
        if project_file(manifest_path).exists():
            manifest = json.loads(project_file(manifest_path).read_text(encoding="utf-8"))
        elif create:
            manifest = {
                "schema_version": 1,
                "experiment_id": experiment_id,
                "repo": state,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "datasets": {},
            }
        else:  # pragma: no cover - protected by the pre-lock check
            raise FileNotFoundError(manifest_path)

        if manifest.get("experiment_id") != experiment_id:
            raise ValueError(f"{manifest_path}: experiment_id mismatch")
        pinned = manifest.get("repo") or {}
        if (pinned.get("commit") != state.get("commit")
                or pinned.get("dirty") != state.get("dirty")):
            raise ValueError(
                f"experiment '{experiment_id}' is pinned to repo {pinned}, but the "
                f"current worktree is {state}; use a new experiment_id"
            )

        fp = config_fingerprint(cfg)
        entry = {"config_fingerprint": fp, "config": canonical_config(cfg)}
        old = (manifest.get("datasets") or {}).get(dataset)
        if old is not None and old != entry:
            raise ValueError(
                f"experiment '{experiment_id}' already pins dataset '{dataset}' to "
                f"config {old.get('config_fingerprint')}; current config is {fp}. "
                "Use a new experiment_id for changed context, stride, capacity, "
                "or any other appendix setting."
            )
        if old is None:
            if not create:
                raise ValueError(
                    f"experiment '{experiment_id}' has no config for '{dataset}'"
                )
            manifest.setdefault("datasets", {})[dataset] = entry
            _write_json_atomic(manifest_path, manifest)
        unlock_file(lock)

    return {
        "experiment_id": experiment_id,
        "config_fingerprint": fp,
        "repo": dict(state),
        "manifest": manifest_path,
    }

def save_base_artifact(path: str | Path, payload: dict, meta: dict) -> None:
    """Publish a base artifact once; reruns need a new experiment ID."""
    path = Path(path)
    sidecar = path.with_suffix(".meta.json")
    project_file(path.parent).mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(".publish.lock")
    with project_file(lock_path).open("a+", encoding="utf-8") as lock:
        lock_file(lock)
        if project_file(path).exists() or project_file(sidecar).exists():
            raise FileExistsError(
                f"{path}: immutable base artifact already exists; choose a new "
                "experiment_id instead of overwriting it"
            )
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npz",
                                        dir=path.parent)
        os.close(fd)
        try:
            np.savez_compressed(
                project_file(tmp_name), **payload,
                artifact_meta_json=np.array(json.dumps(meta, sort_keys=True,
                                                       default=str)),
            )
            os.replace(tmp_name, path)
            _write_json_atomic(sidecar, meta)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        unlock_file(lock)

def ensure_base_sidecar(path: str | Path) -> Path:
    """Recover a missing sidecar from the already-atomic NPZ metadata record."""
    path = Path(path)
    sidecar = path.with_suffix(".meta.json")
    if project_file(sidecar).exists():
        return sidecar
    if not project_file(path).exists():
        raise FileNotFoundError(path)
    lock_path = path.with_suffix(".publish.lock")
    with project_file(lock_path).open("a+", encoding="utf-8") as lock:
        lock_file(lock)
        if not project_file(sidecar).exists():
            with np.load(project_file(path), allow_pickle=False) as z:
                if "artifact_meta_json" not in z.files:
                    raise ValueError(
                        f"{path}: missing sidecar and embedded metadata; cannot "
                        "recover publication safely"
                    )
                meta = json.loads(str(z["artifact_meta_json"]))
            _write_json_atomic(sidecar, meta)
        unlock_file(lock)
    return sidecar

def _metadata(root: str | Path, experiment_id: str, data: HTSData,
              base: str) -> tuple[dict, Path, dict]:
    claim = claim_experiment_namespace(
        root, experiment_id, data.cfg, data.name, create=False
    )
    path = base_forecast_path(root, experiment_id, data.name, base)
    sidecar = ensure_base_sidecar(path)
    meta = json.loads(project_file(sidecar).read_text(encoding="utf-8"))
    checks = {
        "experiment_id": experiment_id,
        "dataset": data.name,
        "base": base,
        "config_fingerprint": claim["config_fingerprint"],
        "split_protocol": SPLIT_PROTOCOL,
        "values_fingerprint": data.values_fingerprint,
    }
    for key, want in checks.items():
        if meta.get(key) != want:
            raise ValueError(
                f"{sidecar.name}: {key} {meta.get(key)!r} != expected {want!r}"
            )
    if meta.get("repo") != claim["repo"]:
        raise ValueError(
            f"{sidecar.name}: repo lineage {meta.get('repo')} != namespace "
            f"lineage {claim['repo']}"
        )
    if experiment_id == MAIN_EXPERIMENT_ID and not meta.get("checkpoint_revision"):
        raise ValueError(f"{sidecar.name}: formal main artifact lacks checkpoint revision")
    expected_revision = get_base(base).revision
    if expected_revision and meta.get("checkpoint_revision") != expected_revision:
        raise ValueError(
            f"{sidecar.name}: checkpoint revision {meta.get('checkpoint_revision')} "
            f"!= registered immutable revision {expected_revision}"
        )
    fp = str(meta.get("fingerprint") or "")
    if len(fp) != 32 or any(c not in "0123456789abcdef" for c in fp):
        raise ValueError(f"{sidecar.name}: invalid base fingerprint {fp!r}")
    return meta, path, claim

def load_base_artifact(root: str | Path, experiment_id: str, data: HTSData,
                       base: str, *, segments=("train", "val", "test")) -> dict:
    _, path, _ = _metadata(root, experiment_id, data, base)
    return load_base_forecasts(path, data, base, segments=tuple(segments))
