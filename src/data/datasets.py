"""Dataset loading, chronological splits, and forecast-origin enumeration.

Protocol (thesis Chapter 3):
- Actual values are reconstructed from the bottom level through S, so ground
  truth is coherent by construction. Formal configs may additionally require
  the persisted source itself to be exactly coherent and semantically
  fingerprinted; reconstruction then remains a defensive invariant.
- SPLIT_PROTOCOL: one strict chronological split by rows,
  train / val / test = 80% / 10% / 10%.  There is no base-train/base-val
  sub-split: every base model is a FROZEN foundation model that receives no
  dataset-specific gradient update, so the whole training segment is available
  for fitting reconciliation heads and classical error statistics.
- A forecast origin t uses rows [t+1-context_len, t] as context and rows
  [t+1, t+predict_len] as targets.  Targets must not cross a split boundary
  (boundary-clean windows); the context MAY reach back across a boundary,
  because those rows are already observed when the forecast is made.
- Every origin — in every segment — requires a full context, so the first
  train origin is context_len - 1.
- Upstream-imputed timestamps declared by the dataset config are never allowed
  to be the forecast origin or part of an H-step target.  For an imputed row g,
  origins t in [g-H, g] are removed after boundary-clean enumeration.
- Heads train on train origins, are selected on validation origins, and are
  evaluated ONCE on test origins.
"""
from __future__ import annotations

# Resolve the approved project layout while retaining sealed source-path keys.
from src.support.paths.project import project_file


import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.hierarchy import Hierarchy, coherent_actuals, load_hierarchy

# Stamped into every frozen-forecast metadata file and checked on load, so a
# .npz produced under an older split geometry can never be silently reused.
SPLIT_PROTOCOL = "chronological_80_10_10"


def semantic_values_fingerprint(timestamps: pd.DatetimeIndex,
                                node_names: list[str],
                                values: np.ndarray) -> str:
    """Cross-platform identity of one canonical all-node values table.

    CSV byte hashes are sensitive to line endings and float formatting. This
    fingerprint instead hashes timestamp microseconds (the original pandas 3 run unit), the ordered canonical
    node names, and a C-contiguous little-endian float64 value matrix. It is
    stable across CSV serialization and host byte order while still changing
    for any scientific input value, timestamp, or node-order change.
    """
    index = pd.DatetimeIndex(timestamps)
    names = [str(name) for name in node_names]
    matrix = np.asarray(values)
    if index.hasnans:
        raise ValueError("cannot fingerprint values with NaT timestamps")
    if matrix.ndim != 2 or matrix.shape != (len(index), len(names)):
        raise ValueError(
            "cannot fingerprint values: matrix shape "
            f"{matrix.shape} != ({len(index)}, {len(names)})"
        )
    matrix64 = np.ascontiguousarray(matrix, dtype="<f8")
    if not np.isfinite(matrix64).all():
        raise ValueError("cannot fingerprint non-finite values")
    time64 = np.ascontiguousarray(index.to_numpy(dtype="datetime64[us]").view("i8"), dtype="<i8")

    digest = hashlib.blake2b(digest_size=16)
    digest.update(b"topic3-semantic-values-v1\0")
    digest.update(json.dumps(names, ensure_ascii=False, separators=(",", ":"))
                  .encode("utf-8"))
    digest.update(b"\0")
    digest.update(np.asarray(matrix64.shape, dtype="<i8").tobytes())
    digest.update(time64.tobytes())
    digest.update(matrix64.tobytes())
    return digest.hexdigest()


def _validate_timestamp_index(index: pd.DatetimeIndex, *, where: str) -> None:
    """Reject timestamp defects before sorting/reindexing can hide them."""
    if len(index) == 0:
        raise ValueError(f"{where}: timestamp column is empty")
    if index.hasnans:
        raise ValueError(f"{where}: timestamp column contains NaT/invalid values")
    if not index.is_monotonic_increasing:
        raise ValueError(
            f"{where}: timestamps are not strictly increasing; do not sort them "
            "silently because row order defines the chronological split"
        )
    if not index.is_unique:
        duplicates = index[index.duplicated()].unique()
        sample = [str(x) for x in duplicates[:3]]
        raise ValueError(f"{where}: duplicate timestamps {sample}")


def expected_time_grid(index: pd.DatetimeIndex, freq: str, *, where: str) \
        -> pd.DatetimeIndex:
    """Return the exact configured grid after validating timestamp identity."""
    _validate_timestamp_index(index, where=where)
    try:
        return pd.date_range(index[0], index[-1], freq=freq)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: invalid configured frequency {freq!r}") from exc


def assert_regular_time_grid(index: pd.DatetimeIndex, freq: str, *, where: str) \
        -> None:
    """Require every timestamp, in order, to equal the configured regular grid."""
    expected = expected_time_grid(index, freq, where=where)
    if not index.equals(expected):
        missing = expected.difference(index)
        off_grid = index.difference(expected)
        raise ValueError(
            f"{where}: timestamps do not form the exact {freq!r} grid "
            f"(missing={len(missing)}, off_grid={len(off_grid)}); enable the "
            "declared gap-filling path or repair the source data"
        )


def _parse_excluded_imputed_timestamps(raw_values, index: pd.DatetimeIndex,
                                       *, where: str) \
        -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Resolve declared upstream imputations to exact rows on the loaded grid."""
    if raw_values is None:
        raw_values = []
    if isinstance(raw_values, (str, bytes)) or not isinstance(raw_values, (list, tuple)):
        raise ValueError(
            f"{where}: excluded_imputed_timestamps must be a YAML list, not "
            f"{type(raw_values).__name__}"
        )
    if not raw_values:
        return pd.DatetimeIndex([]), np.array([], dtype=int)

    try:
        parsed = pd.DatetimeIndex(pd.to_datetime(raw_values, errors="coerce"))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{where}: could not parse excluded_imputed_timestamps"
        ) from exc
    if parsed.hasnans:
        raise ValueError(f"{where}: excluded_imputed_timestamps contains NaT/invalid values")
    if not parsed.is_unique:
        duplicates = parsed[parsed.duplicated()].unique()
        raise ValueError(
            f"{where}: duplicate excluded_imputed_timestamps "
            f"{[str(x) for x in duplicates[:3]]}"
        )
    if parsed.tz != index.tz:
        raise ValueError(
            f"{where}: excluded_imputed_timestamps timezone does not match the data"
        )

    positions = index.get_indexer(parsed)
    if (positions < 0).any():
        missing = parsed[positions < 0]
        raise ValueError(
            f"{where}: excluded_imputed_timestamps are not exact rows on the "
            f"configured grid: {[str(x) for x in missing[:3]]}"
        )
    return parsed, positions.astype(int, copy=False)


@dataclass
class SplitOrigins:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


@dataclass
class HTSData:
    name: str
    hierarchy: Hierarchy
    timestamps: pd.DatetimeIndex
    Y: np.ndarray                 # (T, n) original scale, coherent by construction
    raw_aggregate_dev: float      # relative deviation of CSV aggregates vs reconstruction
    gaps_filled: int              # rows added to complete the regular time grid
    excluded_imputed_timestamps: pd.DatetimeIndex
    excluded_imputed_indices: np.ndarray
    values_fingerprint: str      # semantic identity of timestamps/names/raw values
    cfg: dict

    @property
    def T(self) -> int:
        return self.Y.shape[0]

    @property
    def train_end(self) -> int:
        return int(self.T * float(self.cfg["train_frac"]))

    @property
    def val_end(self) -> int:
        return int(self.T * (float(self.cfg["train_frac"]) + float(self.cfg["val_frac"])))

    @property
    def eval_steps(self) -> list[int]:
        return [int(s) for s in self.cfg["eval_steps"]]

    @property
    def predict_len(self) -> int:
        return int(self.cfg["predict_len"])

    @property
    def context_len(self) -> int:
        return int(self.cfg["context_len"])

    def _candidate_origins(self) -> SplitOrigins:
        """Boundary-clean, full-context origins before imputation exclusions."""
        H, L = self.predict_len, self.context_len
        stride = self.cfg.get("origin_stride", {}) or {}
        return SplitOrigins(
            train=_enumerate_origins(0, self.train_end, H,
                                     int(stride.get("train", 1)), min_origin=L - 1),
            val=_enumerate_origins(self.train_end, self.val_end, H,
                                   int(stride.get("val", 1)), min_origin=L - 1),
            test=_enumerate_origins(self.val_end, self.T, H,
                                    int(stride.get("test", 1)), min_origin=L - 1),
        )

    def origins(self) -> SplitOrigins:
        """Boundary-clean origins with future-informed imputed uses removed."""
        H, L = self.predict_len, self.context_len
        candidate = self._candidate_origins()
        o = SplitOrigins(**{
            seg: _exclude_imputed_origins(
                getattr(candidate, seg), self.excluded_imputed_indices, H
            )
            for seg in ("train", "val", "test")
        })
        assert_split_invariants(o, self.train_end, self.val_end, self.T, H, L)
        assert_imputed_exclusion_invariants(o, self.excluded_imputed_indices, H)
        return o

    def origin_exclusion_counts(self) -> dict[str, int]:
        """Number of otherwise-valid origins removed in each split."""
        candidate = self._candidate_origins()
        final = self.origins()
        return {
            seg: int(getattr(candidate, seg).size - getattr(final, seg).size)
            for seg in ("train", "val", "test")
        }


def _enumerate_origins(seg_start: int, seg_end: int, predict_len: int, stride: int,
                       min_origin: int = 0) -> np.ndarray:
    """Origins t whose targets t+1..t+predict_len all lie inside [seg_start, seg_end).

    `min_origin` additionally requires a full context: with min_origin =
    context_len - 1 no returned origin reads rows before row 0.  It only ever
    binds on the train segment, whose nominal first origin is -1.
    """
    first = max(seg_start - 1, min_origin)  # earliest origin whose first target is seg_start
    last = seg_end - 1 - predict_len        # latest origin whose final target is seg_end-1
    if last < first:
        return np.array([], dtype=int)
    return np.arange(first, last + 1, stride, dtype=int)


def _imputed_origin_mask(origins: np.ndarray, imputed_indices: np.ndarray,
                         predict_len: int) -> np.ndarray:
    """Mark origins whose target includes an imputation, or which end on one.

    If row ``g`` was produced by two-sided interpolation, an origin at ``g``
    contains future information in its last context value, while any origin
    ``t`` with ``g`` in ``[t+1, t+H]`` has an interpolated target.  Their union
    is exactly ``t in [g-H, g]``.  An imputed row may remain as older context for
    ``t >= g+1`` because both observations used to construct it are then known.
    """
    arr = np.asarray(origins, dtype=int)
    excluded = np.asarray(imputed_indices, dtype=int)
    bad = np.zeros(arr.shape, dtype=bool)
    for g in excluded:
        bad |= (arr >= int(g) - int(predict_len)) & (arr <= int(g))
    return bad


def _exclude_imputed_origins(origins: np.ndarray, imputed_indices: np.ndarray,
                             predict_len: int) -> np.ndarray:
    """Remove exactly the future-informed origins induced by imputed rows."""
    arr = np.asarray(origins, dtype=int)
    return arr[~_imputed_origin_mask(arr, imputed_indices, predict_len)]


def assert_imputed_exclusion_invariants(o: SplitOrigins, imputed_indices: np.ndarray,
                                        predict_len: int) -> None:
    """Fail if a configured imputation survives as origin or forecast target."""
    for seg in ("train", "val", "test"):
        arr = getattr(o, seg)
        if _imputed_origin_mask(arr, imputed_indices, predict_len).any():
            raise AssertionError(
                f"{seg}: an excluded imputed timestamp remains at the forecast "
                "origin or inside its target window"
            )


def require_full_context(origins: np.ndarray, context_len: int, where: str) -> None:
    """Every origin must be able to read a complete context window.

    `assert_split_invariants` guarantees this for `HTSData.origins()`, but the
    adapters are public entry points that also take caller-supplied slices (the
    smoke levels do exactly that).  Without the guard, an origin below
    context_len - 1 turns `Y[t + 1 - L : t + 1]` into a negative-start slice,
    which silently yields an empty or short context instead of raising.
    """
    arr = np.asarray(origins)
    if arr.size == 0:
        return
    lo = int(arr.min())
    if lo < context_len - 1:
        raise ValueError(
            f"{where}: origin {lo} cannot supply a full context of {context_len} "
            f"rows (needs origin >= {context_len - 1}); a negative-start slice "
            "would silently produce a short context."
        )


def assert_split_invariants(o: SplitOrigins, train_end: int, val_end: int, T: int,
                            predict_len: int, context_len: int) -> None:
    """Fail loudly if any origin violates the 80/10/10 protocol.

    Checked per segment: the context start is non-negative (full context), the
    first target lands on or after the segment start, and the last target stays
    strictly inside the segment.  Also checked: the three origin sets are
    strictly ordered and disjoint.
    """
    H, L = predict_len, context_len
    bounds = (("train", o.train, 0, train_end),
              ("val", o.val, train_end, val_end),
              ("test", o.test, val_end, T))
    for seg, arr, lo, hi in bounds:
        if arr.size == 0:
            raise AssertionError(f"{seg} segment has no valid origins "
                                 f"(context_len={L}, predict_len={H})")
        if int(arr.min()) - L + 1 < 0:
            raise AssertionError(f"{seg}: context start {int(arr.min()) - L + 1} < 0")
        if int(arr.min()) + 1 < lo:
            raise AssertionError(f"{seg}: first target {int(arr.min()) + 1} < segment start {lo}")
        if int(arr.max()) + H > hi - 1:
            raise AssertionError(f"{seg}: last target {int(arr.max()) + H} >= segment end {hi}")
    if int(o.train.max()) >= int(o.val.min()) or int(o.val.max()) >= int(o.test.min()):
        raise AssertionError("origin segments are not strictly ordered in time")


def load_config(path: str | Path) -> dict:
    with open(project_file(path), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(project_file(path))
    return cfg


def load_dataset(cfg: dict, repo_root: str | Path = ".") -> HTSData:
    root = Path(repo_root)
    data_dir = root / cfg["data_dir"]
    hier = load_hierarchy(data_dir)

    values_path = data_dir / cfg["values_csv"]
    df = pd.read_csv(project_file(values_path))
    timestamp_col = cfg["timestamp_col"]
    if timestamp_col not in df.columns:
        raise ValueError(f"{values_path}: missing timestamp column {timestamp_col!r}")
    expected_columns = [timestamp_col, *hier.node_names]
    if (bool(cfg.get("require_exact_source_schema", False))
            and list(df.columns) != expected_columns):
        missing_schema = [c for c in expected_columns if c not in df.columns]
        extra_schema = [c for c in df.columns if c not in expected_columns]
        raise ValueError(
            f"{values_path}: exact source schema required; expected timestamp "
            f"followed by the canonical {len(hier.node_names)}-node order "
            f"(missing={missing_schema[:5]}, extra={extra_schema[:5]}, "
            f"same_columns={not missing_schema and not extra_schema})"
        )
    ts = pd.DatetimeIndex(pd.to_datetime(df[timestamp_col], errors="coerce"))
    _validate_timestamp_index(ts, where=f"{values_path}:{timestamp_col}")
    missing = [c for c in hier.node_names if c not in df.columns]
    if missing:
        raise ValueError(f"values csv missing hierarchy columns: {missing[:5]} ...")

    frame = df[hier.node_names].astype(np.float64)
    frame.index = ts
    source_values = frame.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(source_values).all():
        bad = int((~np.isfinite(source_values)).sum())
        raise ValueError(
            f"{values_path}: {bad} non-finite source values; interpolation is only "
            "for missing timestamps, not NaN/Inf measurements"
        )

    # Complete the regular time grid; small gaps (e.g. DST hours) are filled by
    # time interpolation on the raw columns BEFORE the coherent reconstruction,
    # so filled rows are coherent by construction like every other row.
    where = f"{values_path}:{timestamp_col}"
    full = expected_time_grid(frame.index, cfg["freq"], where=where)
    off_grid = frame.index.difference(full)
    if len(off_grid):
        raise ValueError(
            f"{where}: {len(off_grid)} timestamps are off the configured "
            f"{cfg['freq']!r} grid; sample={[str(x) for x in off_grid[:3]]}"
        )
    gaps_filled = int(len(full) - len(frame))
    if bool(cfg.get("fill_gaps", True)):
        if gaps_filled:
            frame = frame.reindex(full).interpolate(method="time").ffill().bfill()
    elif gaps_filled:
        raise ValueError(
            f"{where}: irregular series has {gaps_filled} missing grid rows and "
            "fill_gaps is disabled"
        )
    assert_regular_time_grid(frame.index, cfg["freq"], where=f"{where} after gap handling")
    excluded_imputed_timestamps, excluded_imputed_indices = \
        _parse_excluded_imputed_timestamps(
            cfg.get("excluded_imputed_timestamps"),
            pd.DatetimeIndex(frame.index),
            where=where,
        )

    raw = frame.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(raw).all():
        bad = int((~np.isfinite(raw)).sum())
        raise ValueError(f"{values_path}: {bad} non-finite values remain after gap handling")
    values_fingerprint = semantic_values_fingerprint(
        pd.DatetimeIndex(frame.index), hier.node_names, raw
    )
    expected_fingerprint = cfg.get("values_fingerprint")
    if (expected_fingerprint is not None
            and str(expected_fingerprint) != values_fingerprint):
        raise ValueError(
            f"{values_path}: semantic values fingerprint {values_fingerprint} "
            f"!= configured {expected_fingerprint}; timestamps, node order, or "
            "scientific values changed"
        )
    bottom = raw[:, hier.bottom_idx]
    Y = coherent_actuals(hier, bottom)

    agg = np.asarray(hier.agg_idx, dtype=int)
    denom = np.abs(raw[:, agg]).sum()
    raw_dev = (float(np.abs(Y[:, agg] - raw[:, agg]).sum() / denom)
               if denom > 0 else 0.0)

    if (bool(cfg.get("require_source_coherence", False))
            and not np.array_equal(raw, Y)):
        i, j = np.argwhere(raw != Y)[0]
        raise ValueError(
            f"{values_path}: source coherence is required but node "
            f"{hier.node_names[int(j)]!r} at {frame.index[int(i)]} is "
            f"{raw[int(i), int(j)]:g}, not bottom reconstruction "
            f"{Y[int(i), int(j)]:g} (relative aggregate deviation "
            f"{raw_dev:.6g})"
        )

    if not np.isfinite(Y).all():
        raise ValueError(
            "non-finite values in reconstructed series; fix the data before running"
        )

    return HTSData(
        name=str(cfg["name"]),
        hierarchy=hier,
        timestamps=pd.DatetimeIndex(frame.index),
        Y=Y,
        raw_aggregate_dev=raw_dev,
        gaps_filled=gaps_filled,
        excluded_imputed_timestamps=excluded_imputed_timestamps,
        excluded_imputed_indices=excluded_imputed_indices,
        values_fingerprint=values_fingerprint,
        cfg=cfg,
    )


def targets_for_origins(Y: np.ndarray, origins: np.ndarray, predict_len: int) -> np.ndarray:
    """Return (num_origins, n, predict_len) ground-truth targets."""
    out = np.stack([Y[t + 1 : t + 1 + predict_len].T for t in origins], axis=0)
    return out
