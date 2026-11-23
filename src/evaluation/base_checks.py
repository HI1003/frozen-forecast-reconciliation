"""Check distinct base forecasts and canonical-nine cross-base result tables.

These checks cover the final nine strategies and their shared frozen inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies import HEAD_NAMES, LEARNABLE_HEADS
CANONICAL_HEADS = tuple(HEAD_NAMES)


def method_seeds(method, seeds):
    return tuple(seeds) if method in LEARNABLE_HEADS else (seeds[0],)


def assert_forecasts_distinct(a: np.ndarray, b: np.ndarray,
                              *, where: str) -> float:
    """Reject copied or numerically indistinguishable main-base forecasts."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"{where}: forecast shapes differ: {a.shape} vs {b.shape}")
    diff = float(np.abs(a - b).mean())
    scale = max(1.0, float(np.abs(a).mean()), float(np.abs(b).mean()))
    if not np.isfinite(diff) or diff <= 1e-10 * scale:
        raise ValueError(
            f"{where}: main base forecasts are copied or numerically "
            f"indistinguishable (mean difference={diff}, scale={scale})"
        )
    return diff


def assert_complete(df: pd.DataFrame, base: str, steps: list[int],
                    levels: list[str], expect_seeds: list[int]) -> None:
    heads = df[df.method.isin(CANONICAL_HEADS)]
    seen = (heads.groupby(["method", "step", "level"])["seed"]
                 .agg(lambda s: frozenset(int(x) for x in s)).to_dict())
    for step in steps:
        for level in levels:
            missing = [m for m in CANONICAL_HEADS if (m, step, level) not in seen]
            if missing:
                raise SystemExit(
                    f"base '{base}', step {step}, level '{level}': missing {missing}"
                )
            for method in CANONICAL_HEADS:
                want = set(method_seeds(method, expect_seeds))
                got = set(seen[(method, step, level)])
                if got != want:
                    raise SystemExit(
                        f"base '{base}', head '{method}' at step {step}, level "
                        f"'{level}': seeds {sorted(got)} != expected {sorted(want)}"
                    )
    if "base" not in set(df.method):
        raise SystemExit(f"base '{base}': no unreconciled 'base' rows; step2 must run first")


def assert_csv_lineage(df: pd.DataFrame, base: str, fingerprint: str,
                       source: str) -> None:
    """Legacy fingerprint-only contract; formal code calls the full guard."""
    if "base_fingerprint" not in df.columns:
        raise SystemExit(f"{source}: no `base_fingerprint` column")
    found = set(df["base_fingerprint"].astype(str))
    if found != {fingerprint}:
        raise SystemExit(
            f"{source}: rows reference stale base fingerprint(s); re-run step2/step3"
        )
