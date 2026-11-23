"""Exact numerical runtime validation for queued formal jobs."""
from __future__ import annotations

# Resolve the approved project layout while retaining sealed source-path keys.
from src.support.paths.project import project_file


import importlib.metadata
import json
import sys
from pathlib import Path


PACKAGES = (
    "torch", "numpy", "pandas", "scipy", "PyYAML", "transformers",
    "chronos-forecasting", "timesfm",
)


def installed_runtime() -> dict:
    return {
        "python": ".".join(str(x) for x in sys.version_info[:3]),
        "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
    }


def assert_runtime_lock(root: str | Path) -> dict:
    lock_path = Path(root) / "configs" / "formal_run_lock.json"
    expected = json.loads(project_file(lock_path).read_text(encoding="utf-8")).get("runtime")
    actual = installed_runtime()
    if expected != actual:
        raise RuntimeError(
            f"formal runtime drift: installed {actual} != locked {expected}; "
            "do not mix queued stages across environment changes"
        )
    return actual
