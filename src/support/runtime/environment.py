"""Random seeds, device selection, runtime metadata and resource measurements."""
from __future__ import annotations

import platform
import random
import subprocess
from importlib import metadata
from pathlib import Path

import numpy as np

from src.support.paths.project import project_file

_PKGS = ("numpy", "scipy", "pandas", "torch", "transformers",
         "chronos-forecasting", "timesfm")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def pick_device(arg: str | None = None) -> str:
    if arg:
        return arg
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def env_report() -> dict:
    """Versions and accelerator identity, recorded in every frozen-forecast
    metadata file and printed with any numerical failure — a NaN that depends
    on the PyTorch build is unreadable without it."""
    rep = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in _PKGS:
        try:
            rep[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            rep[pkg] = None
    try:
        import torch

        rep["torch_cuda"] = torch.version.cuda
        rep["cudnn"] = torch.backends.cudnn.version()
        rep["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            rep["gpu"] = torch.cuda.get_device_name(0)
            rep["gpu_count"] = torch.cuda.device_count()
    except ImportError:
        pass
    return rep


def repo_state(root: str | Path) -> dict:
    """Commit and dirtiness of the working tree that produced an artifact.

    A model ID alone is not enough to reproduce a run years later: the pipeline
    code around it moves too.  A dirty tree is recorded rather than rejected, so
    exploratory runs stay possible but are never mistaken for reproducible ones.
    """
    def _git(*args):
        return subprocess.run(("git", "-C", str(project_file(root))) + args, capture_output=True,
                              text=True, check=True, timeout=15).stdout.strip()

    try:
        return {"commit": _git("rev-parse", "HEAD"),
                "dirty": bool(_git("status", "--porcelain"))}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def checkpoint_revision(model_id: str) -> str | None:
    """Hugging Face commit SHA of the locally cached snapshot, if resolvable.

    Tags like `main` move; the SHA is what makes a checkpoint citable.  Resolved
    from the local cache only, so it never adds a network dependency to a run.
    """
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(model_id, local_files_only=True)).name
    except Exception:
        pass
    try:
        from huggingface_hub import constants
        from huggingface_hub.file_download import repo_folder_name

        ref = (Path(constants.HF_HUB_CACHE)
               / repo_folder_name(repo_id=model_id, repo_type="model")
               / "refs" / "main")
        return project_file(ref).read_text(encoding="utf-8").strip() if project_file(ref).exists() else None
    except Exception:
        return None


def peak_gpu_mib(device: str) -> float | None:
    """Peak CUDA allocation since the last reset, in MiB (None on CPU).

    Pair with `reset_peak_gpu` so a reported figure describes ONE model instead
    of the running maximum over everything the process has loaded — the number
    goes into the paper's efficiency table.
    """
    if not str(device).startswith("cuda"):
        return None
    import torch

    return round(torch.cuda.max_memory_allocated() / 2**20, 1)


def reset_peak_gpu(device: str) -> None:
    """Start a fresh peak-memory measurement window."""
    if not str(device).startswith("cuda"):
        return
    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
