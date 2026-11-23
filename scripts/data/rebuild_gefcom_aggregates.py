"""Deterministically rebuild GEFCom aggregate columns from bottom nodes.

The default mode is a non-mutating check: a candidate CSV is written beside
the source, reread and fully validated, then removed.  Passing ``--write`` is
the only way to atomically replace the source file.
"""
from __future__ import annotations

# Resolve the approved project layout while retaining sealed source-path keys.
from pathlib import Path as _ProjectPath
import sys as _project_sys
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
_project_sys.path.insert(0, str(_PROJECT_ROOT))
from src.support.paths.project import project_file


import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_file(ROOT)))

from src.data.hierarchy import Hierarchy, coherent_actuals, load_hierarchy


@dataclass
class ValidatedFrame:
    frame: pd.DataFrame
    timestamp_text: np.ndarray
    timestamps: pd.DatetimeIndex
    values: np.ndarray
    bottom: np.ndarray
    reconstructed: np.ndarray
    aggregate_abs_error: float
    aggregate_abs_denominator: float
    aggregate_relative_deviation: float | None
    semantic_hashes: dict[str, str]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with project_file(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_semantic_hash(values: np.ndarray, *, label: str) -> str:
    digest = hashlib.sha256()
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(tuple(values.shape)).encode("ascii"))
    digest.update(b"\0")
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _numeric_semantic_hash(values: np.ndarray, names: Sequence[str], *, label: str) -> str:
    canonical = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    digest = hashlib.sha256()
    header = {
        "dtype": "<f8",
        "label": label,
        "names": [str(name) for name in names],
        "shape": list(canonical.shape),
    }
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _parse_excluded_rows(
    values: Sequence[str], timestamps: pd.DatetimeIndex, *, where: str
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    raw = list(values or [])
    parsed = pd.DatetimeIndex(pd.to_datetime(raw, errors="coerce"))
    if parsed.hasnans:
        raise ValueError(f"{where}: excluded imputed timestamps contain invalid values")
    if not parsed.is_unique:
        raise ValueError(f"{where}: excluded imputed timestamps contain duplicates")
    if parsed.tz != timestamps.tz:
        raise ValueError(f"{where}: excluded timestamp timezone differs from data")
    positions = timestamps.get_indexer(parsed)
    if (positions < 0).any():
        missing = [str(x) for x in parsed[positions < 0][:3]]
        raise ValueError(f"{where}: excluded timestamps are absent from the grid: {missing}")
    return parsed, positions.astype(np.int64, copy=False)


def _validate_frame(
    path: Path,
    hierarchy: Hierarchy,
    *,
    timestamp_col: str,
    freq: str,
) -> ValidatedFrame:
    frame = pd.read_csv(project_file(path), dtype={timestamp_col: "string"})
    expected_columns = [timestamp_col] + hierarchy.node_names
    actual_columns = frame.columns.tolist()
    if actual_columns != expected_columns:
        raise ValueError(
            f"{path}: schema/order differs from canonical timestamp + node_order; "
            f"expected {len(expected_columns)} columns, got {len(actual_columns)}"
        )
    if frame.empty:
        raise ValueError(f"{path}: data file is empty")

    timestamp_text = frame[timestamp_col].astype(str).to_numpy(copy=True)
    timestamps = pd.DatetimeIndex(pd.to_datetime(timestamp_text, errors="coerce"))
    if timestamps.hasnans:
        raise ValueError(f"{path}: timestamp column contains invalid values")
    if not timestamps.is_unique:
        raise ValueError(f"{path}: timestamp column contains duplicates")
    if not timestamps.is_monotonic_increasing:
        raise ValueError(f"{path}: timestamps are not strictly increasing")
    try:
        expected_grid = pd.date_range(timestamps[0], timestamps[-1], freq=freq)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: invalid configured frequency {freq!r}") from exc
    if not timestamps.equals(expected_grid):
        raise ValueError(
            f"{path}: timestamps do not form the exact configured {freq!r} grid"
        )

    try:
        numeric = frame[hierarchy.node_names].astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: node columns are not all numeric") from exc
    values = numeric.to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: node columns contain NaN or infinite values")
    if (values < 0).any():
        where = np.argwhere(values < 0)[0]
        raise ValueError(
            f"{path}: negative value at row {int(where[0])}, "
            f"node {hierarchy.node_names[int(where[1])]!r}"
        )

    # Rebuild the in-memory frame rather than assigning float columns into
    # pandas-inferred integer blocks (pandas 2.x rejects that lossy dtype
    # mutation). Timestamp text and canonical column order stay unchanged.
    frame = pd.concat(
        [frame[[timestamp_col]].copy(), numeric.copy()], axis=1
    )
    bottom = values[:, hierarchy.bottom_idx].copy()
    reconstructed = coherent_actuals(hierarchy, bottom)
    aggregate = np.asarray(hierarchy.agg_idx, dtype=np.int64)
    delta = values[:, aggregate] - reconstructed[:, aggregate]
    numerator = float(np.abs(delta).sum())
    denominator = float(np.abs(values[:, aggregate]).sum())
    if denominator > 0:
        relative: float | None = numerator / denominator
    elif numerator == 0:
        relative = 0.0
    else:
        relative = None

    semantic_hashes = {
        "timestamps": _text_semantic_hash(timestamp_text, label="timestamps"),
        "bottom": _numeric_semantic_hash(
            bottom,
            [hierarchy.node_names[i] for i in hierarchy.bottom_idx],
            label="bottom",
        ),
        "all_nodes": _numeric_semantic_hash(
            values, hierarchy.node_names, label="all_nodes"
        ),
    }
    return ValidatedFrame(
        frame=frame,
        timestamp_text=timestamp_text,
        timestamps=timestamps,
        values=values,
        bottom=bottom,
        reconstructed=reconstructed,
        aggregate_abs_error=numerator,
        aggregate_abs_denominator=denominator,
        aggregate_relative_deviation=relative,
        semantic_hashes=semantic_hashes,
    )


def _write_candidate(frame: pd.DataFrame, path: Path, *, mode: int) -> None:
    with project_file(path).open("w", encoding="utf-8", newline="") as handle:
        frame.to_csv(
            project_file(handle),
            index=False,
            lineterminator="\n",
            float_format="%.17g",
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, mode)


def _fsync_directory(path: Path) -> None:
    # Windows cannot open directory descriptors through os.open. The staged
    # file is still flushed/fsynced before the atomic replacement above.
    if os.name == 'nt':
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = project_file(os).open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def rebuild_aggregate_columns(
    csv_path: str | Path,
    data_dir: str | Path,
    *,
    timestamp_col: str,
    freq: str,
    excluded_imputed_timestamps: Sequence[str] = (),
    write: bool = False,
) -> dict:
    """Build, reread and validate a canonical aggregate-column replacement.

    In check mode (the default), the validated candidate is removed and the
    source remains byte-for-byte unchanged.  In write mode, replacement occurs
    only after all semantic invariants pass and only through ``os.replace``.
    """
    target = Path(csv_path).resolve()
    hierarchy_dir = Path(data_dir).resolve()
    if not project_file(target).is_file():
        raise FileNotFoundError(target)
    hierarchy = load_hierarchy(hierarchy_dir)
    aggregate_idx = np.asarray(hierarchy.agg_idx, dtype=np.int64)
    aggregate_names = [hierarchy.node_names[i] for i in aggregate_idx]
    bottom_names = [hierarchy.node_names[i] for i in hierarchy.bottom_idx]

    before_file_sha = _file_sha256(target)
    before_mode = stat.S_IMODE(project_file(target).stat().st_mode)
    before = _validate_frame(
        target, hierarchy, timestamp_col=timestamp_col, freq=freq
    )
    excluded, excluded_positions = _parse_excluded_rows(
        excluded_imputed_timestamps,
        before.timestamps,
        where=str(project_file(target)),
    )
    excluded_bottom_before = before.bottom[excluded_positions].copy()

    candidate_frame = before.frame.copy()
    candidate_frame.loc[:, aggregate_names] = before.reconstructed[:, aggregate_idx]

    descriptor, raw_temp_path = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".candidate", dir=target.parent
    )
    os.close(descriptor)
    candidate_path = Path(raw_temp_path)
    replacement_performed = False
    try:
        _write_candidate(candidate_frame, candidate_path, mode=before_mode)
        candidate_file_sha = _file_sha256(candidate_path)
        after = _validate_frame(
            candidate_path, hierarchy, timestamp_col=timestamp_col, freq=freq
        )

        if not np.array_equal(before.timestamp_text, after.timestamp_text):
            raise AssertionError("candidate changed timestamp text or row order")
        if not np.array_equal(before.bottom, after.bottom):
            raise AssertionError("candidate changed one or more bottom-node values")
        if not np.array_equal(excluded_bottom_before, after.bottom[excluded_positions]):
            raise AssertionError("candidate changed a declared DST/imputed bottom row")
        if not np.array_equal(after.values, after.reconstructed):
            difference = float(np.abs(after.values - after.reconstructed).max())
            raise AssertionError(
                f"candidate is not exactly coherent after reread; max error={difference}"
            )
        if after.aggregate_abs_error != 0.0:
            raise AssertionError(
                f"candidate aggregate deviation is not zero: {after.aggregate_abs_error}"
            )

        # Detect a concurrent edit before an optional atomic replacement.
        current_file_sha = _file_sha256(target)
        if current_file_sha != before_file_sha:
            raise RuntimeError(
                f"{target}: source changed while the candidate was being built"
            )
        if write and candidate_file_sha != current_file_sha:
            os.replace(candidate_path, target)
            replacement_performed = True
            _fsync_directory(target.parent)

        target_file_sha_after_run = _file_sha256(target)
        if write:
            final = _validate_frame(
                target, hierarchy, timestamp_col=timestamp_col, freq=freq
            )
            if not np.array_equal(final.values, after.values):
                raise AssertionError("atomic target differs semantically from validated candidate")
            if target_file_sha_after_run != candidate_file_sha:
                raise AssertionError("atomic target bytes differ from validated candidate")
        else:
            final = before

        report = {
            "schema_version": 1,
            "mode": "write" if write else "check",
            "target": str(project_file(target)),
            "rows": int(before.values.shape[0]),
            "hierarchy": {
                "total_nodes": hierarchy.n,
                "bottom_nodes": hierarchy.m,
                "aggregate_nodes": int(aggregate_idx.size),
            },
            "aggregate_names": aggregate_names,
            "bottom_names": bottom_names,
            "excluded_imputed_timestamps": [str(value) for value in excluded],
            "file_sha256": {
                "before": before_file_sha,
                "after": candidate_file_sha,
                "target_after_run": target_file_sha_after_run,
            },
            "semantic_sha256": {
                "before": before.semantic_hashes,
                "after": after.semantic_hashes,
            },
            "aggregate_deviation": {
                "definition": "sum(abs(raw_aggregate - S_bottom)) / sum(abs(raw_aggregate))",
                "pre": before.aggregate_relative_deviation,
                "post": after.aggregate_relative_deviation,
                "pre_absolute_error": before.aggregate_abs_error,
                "post_absolute_error": after.aggregate_abs_error,
                "pre_denominator": before.aggregate_abs_denominator,
                "post_denominator": after.aggregate_abs_denominator,
            },
            "content_changed": before_file_sha != candidate_file_sha,
            "replacement_performed": replacement_performed,
            "validation": {
                "schema_and_order": True,
                "regular_time_grid": True,
                "finite": True,
                "nonnegative": True,
                "timestamps_preserved_exactly": True,
                "bottom_preserved_exactly": True,
                "excluded_bottom_rows_preserved_exactly": True,
                "all_nodes_exactly_coherent": True,
                "candidate_reread": True,
                "atomic_same_directory": True,
                "idempotent_bytes": (
                    candidate_file_sha == before_file_sha
                    if before.aggregate_abs_error == 0.0
                    else None
                ),
            },
        }
        # A strict JSON emitter must never leak NaN/Infinity.
        json.dumps(report, allow_nan=False)
        return report
    finally:
        if project_file(candidate_path).exists():
            candidate_path.unlink()


def _resolve_from_config(config_path: Path) -> tuple[dict, Path, Path]:
    config = yaml.safe_load(project_file(config_path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{config_path}: configuration must be a mapping")
    data_dir = Path(config["data_dir"])
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    csv_path = Path(config["values_csv"])
    if not csv_path.is_absolute():
        csv_path = data_dir / csv_path
    return config, data_dir, csv_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check or atomically rebuild GEFCom aggregate columns from the "
            "canonical bottom nodes and sum matrix."
        )
    )
    parser.add_argument(
        "--config",
        default=str(project_file(ROOT / "configs" / "gefcom2017f.yaml")),
        help="dataset YAML (default: configs/data/gefcom2017f.yaml)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="atomically replace the CSV; without this flag the source is unchanged",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    config, data_dir, csv_path = _resolve_from_config(config_path)
    report = rebuild_aggregate_columns(
        csv_path,
        data_dir,
        timestamp_col=str(config["timestamp_col"]),
        freq=str(config["freq"]),
        excluded_imputed_timestamps=config.get("excluded_imputed_timestamps") or [],
        write=bool(args.write),
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
