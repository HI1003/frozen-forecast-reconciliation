"""Frozen base forecasts from TimesFM 2.5 (google/timesfm-2.5-200m-pytorch).

Second MAIN base of the study.  Used exactly like Chronos-2: FROZEN inference,
one independent univariate context per node, `context_len` real observations
in, `predict_len` steps out, median (0.5 quantile) as the point forecast.  No
gradient update ever touches this model.

Official API (timesfm 2.0.2):
  TimesFM_2p5_200M_torch.from_pretrained(model_id)
  model.compile(ForecastConfig(...))
  point, quantiles = model.forecast(horizon=H, inputs=[1D arrays])
  -> point (B, H); quantiles (B, H, 10) where channel 5 is the 0.5 quantile
     and `forecast` returns `quantiles[..., 5]` as the point forecast.

`_forecast_batch` re-verifies that identity on every batch rather than trusting
it, because it is an undocumented internal convention: if a future release
returns the mean instead, the two main bases would silently stop sharing a
point statistic and the RQ-D ranking comparison would be meaningless.

ForecastConfig: normalize_inputs, use_continuous_quantile_head,
force_flip_invariance and fix_quantile_crossing are on, but
`infer_is_positive` is deliberately OFF.  That flag clamps forecasts at zero
whenever the whole context is non-negative, which would violate the workspace
rule that base forecasts are never truncated (AGENTS.md) and would put a
non-negativity treatment on one main base and not the other — an avoidable
confound in exactly the ranking comparison RQ-D asks about.  Non-negativity
enters only at the reconciliation stage, as it does for Chronos.

`torch_compile` is forwarded to the official wrapper, which is the ONLY place
compilation may happen: `TimesFM_2p5_200M_torch.__init__` defaults it to True
and `load_checkpoint` then wraps `model.forward` with `torch.compile`.
Compiling anything here on top of that would stack a second, undeclared layer,
so the adapter never calls `torch.compile` itself and the metadata records the
resolved state of the loaded instance rather than the CLI flag.

Context geometry: the model rounds `max_context` up to a multiple of the input
patch size (32), so a 365-day window compiles as 384 and TimesFM left-pads 19
MASKED zeros.  They contain no future observations and the real input budget is
still exactly `context_len`, but the official 2.0.2 wrapper computes input
mean/std over the padded tensor before passing the mask into `decode`. Padding
therefore affects preprocessing and is recorded explicitly in metadata.

PyTorch requirement: TimesFM 2.5 produces NaN under the fused attention path of
PyTorch 2.3.x with these weights.  PyTorch 2.7.1 is verified good.  Every batch
is checked for finiteness here, and a failure aborts the run with the full
environment report instead of writing contaminated forecasts to disk.
"""
from __future__ import annotations



import numpy as np

from src.data.datasets import HTSData, require_full_context, targets_for_origins
from src.support.storage.forecasts import check_forecast_block

# Channel of the 10-wide TimesFM quantile axis holding the 0.5 quantile.
Q50_INDEX = 5
# Input patch size of TimesFM 2.5; max_context must be a multiple of it.
INPUT_PATCH_LEN = 32


def compiled_max_context(context_len: int) -> int:
    """`context_len` rounded up to the input patch size, as the model would."""
    p = INPUT_PATCH_LEN
    return -(-int(context_len) // p) * p


def context_padding_info(context_len: int) -> dict:
    """Auditable geometry of the official TimesFM left-padding path.

    In timesfm 2.0.2 the zero padding is masked for `decode` but remains present
    when the wrapper computes input mean/std. This is not leakage; it is an
    adapter-specific preprocessing detail that must not be described as inert.
    """
    real = int(context_len)
    compiled = compiled_max_context(real)
    left = compiled - real
    return {
        "real_context": real,
        "compiled_max_context": compiled,
        "left_zero_padding": left,
        "decode_masked": bool(left),
        "normalization_includes_padding": bool(
            left and FORECAST_FLAGS["normalize_inputs"]
        ),
        "timesfm_version_contract": "2.0.2",
    }


FORECAST_FLAGS = {
    "normalize_inputs": True,
    "use_continuous_quantile_head": True,
    "force_flip_invariance": True,
    # OFF on purpose — see the module docstring. Base forecasts are never
    # truncated; non-negativity belongs to the reconciliation stage.
    "infer_is_positive": False,
    "fix_quantile_crossing": True,
}


def load_model(model_id: str = "google/timesfm-2.5-200m-pytorch",
               device: str = "cpu",
               context_len: int = 512,
               predict_len: int = 24,
               batch_size: int = 32,
               torch_compile: bool = False,
               revision: str | None = None):
    """Load and compile TimesFM 2.5 for a fixed (context, horizon) geometry.

    `torch_compile` is passed straight to the official wrapper, whose own
    default is True.  Default OFF here: compilation must be validated by a GPU
    smoke run before a production matrix depends on it.
    """
    try:
        import timesfm
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "pip install timesfm==2.0.2 (see requirements.txt); then run "
            "scripts/checks/check_base_forecasts.py --config configs/data/walmart.yaml "
            "--base timesfm25 --level api"
        ) from e
    import torch

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        model_id, torch_compile=bool(torch_compile), revision=revision
    )

    # The wrapper picks cuda:0 whenever CUDA is visible AND sets device_count to
    # every visible GPU, but this adapter runs strictly single-device: all
    # parameters live on one device and `forecast` batches sequentially. Pin the
    # canonical indexed device and force device_count = 1 so the wrapper's
    # per_core_batch_size accounting matches reality. `torch.device("cuda")` has
    # index None and never compares equal to `cuda:0`, so normalise before use.
    target = torch.device(device)
    if target.type == "cuda" and target.index is None:
        target = torch.device("cuda", torch.cuda.current_device())
    if model.model.device != target:
        model.model.to(target)
    model.model.device = target
    model.model.device_count = 1

    model.compile(
        timesfm.ForecastConfig(
            max_context=compiled_max_context(context_len),
            max_horizon=int(predict_len),
            per_core_batch_size=int(batch_size),
            **FORECAST_FLAGS,
        )
    )
    return model


def compile_state(model) -> dict:
    """Resolved compilation state of a loaded instance, for the metadata.

    Reports what the object actually is, not what was requested: the official
    wrapper compiles `model.forward` by default, so a run that never passed
    `--torch-compile` can still be running compiled code.
    """
    fwd = getattr(model.model, "forward", None)
    return {
        "torch_compile_requested": bool(getattr(model, "torch_compile", False)),
        "forward_compiled": bool(
            hasattr(fwd, "_torchdynamo_orig_callable")
            or type(fwd).__module__.startswith("torch._dynamo")
        ),
    }


def _forecast_batch(model, contexts, H: int) -> tuple[np.ndarray, np.ndarray]:
    """Point and quantile forecasts for a batch of univariate contexts.

    Returns (point (B, H), quantiles (B, H, Q)) and enforces the contract that
    the point forecast IS the q50 channel — see the module docstring.
    `model.forecast` pads its `inputs` list in place to a multiple of the global
    batch size, so it is handed a private copy.
    """
    B = len(contexts)
    point, quant = model.forecast(horizon=H, inputs=list(contexts))
    point = np.asarray(point, dtype=np.float64)
    quant = np.asarray(quant, dtype=np.float64)

    # Finiteness FIRST. NaN != NaN under allclose, so a NaN batch — the
    # documented torch-2.3 fused-attention failure — would otherwise be
    # misreported below as "the release changed its point statistic", with no
    # runtime versions attached and an all-NaN warning from np.nanmax.
    if not np.isfinite(point).all():
        check_forecast_block(point, point.shape, nonneg=False,
                             where="timesfm:_forecast_batch:point")
    if point.shape != (B, H):
        raise ValueError(f"TimesFM point forecast shape {point.shape} != {(B, H)}")
    if quant.ndim != 3 or quant.shape[:2] != (B, H):
        raise ValueError(f"TimesFM quantile forecast shape {quant.shape} != {(B, H, 'Q')}")
    if quant.shape[2] <= Q50_INDEX:
        raise ValueError(
            f"TimesFM quantile axis has {quant.shape[2]} channels; channel "
            f"{Q50_INDEX} (the 0.5 quantile) does not exist"
        )

    q50 = quant[:, :, Q50_INDEX]
    if not np.isfinite(q50).all():
        check_forecast_block(q50, q50.shape, nonneg=False,
                             where="timesfm:_forecast_batch:q50")
    if not np.allclose(point, q50, rtol=1e-6, atol=0.0, equal_nan=False):
        gap = float(np.nanmax(np.abs(point - q50)))
        raise ValueError(
            f"TimesFM point forecast is not the q50 channel (max |diff| {gap:.6g}). "
            "The point statistic must match Chronos-2's median for the cross-base "
            "comparison to be valid; check the timesfm release notes."
        )
    return point, quant


def generate_timesfm_forecasts(
    data: HTSData,
    origins: np.ndarray,
    model_id: str = "google/timesfm-2.5-200m-pytorch",
    device: str = "cpu",
    batch_size: int = 32,
    origin_chunk: int = 64,
    torch_compile: bool = False,
    model=None,
    revision: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (yhat, ytrue) with shape (num_origins, n, predict_len), original scale."""
    H = data.predict_len
    L = data.context_len
    n = data.hierarchy.n
    Y = data.Y
    require_full_context(origins, L, where=f"timesfm:{model_id}")

    if model is None:
        model = load_model(model_id, device=device, context_len=L, predict_len=H,
                           batch_size=batch_size, torch_compile=torch_compile,
                           revision=revision)

    yhat = np.empty((len(origins), n, H), dtype=np.float64)
    for c0 in range(0, len(origins), origin_chunk):
        chunk = origins[c0 : c0 + origin_chunk]
        contexts = [
            np.ascontiguousarray(Y[t + 1 - L : t + 1, i], dtype=np.float32)
            for t in chunk for i in range(n)
        ]
        preds = np.empty((len(contexts), H), dtype=np.float64)
        for s in range(0, len(contexts), batch_size):
            batch = contexts[s : s + batch_size]
            preds[s : s + len(batch)], _ = _forecast_batch(model, batch, H)
        block = preds.reshape(len(chunk), n, H)
        # nonneg=False: base forecasts are never truncated, on either main base.
        check_forecast_block(
            block, (len(chunk), n, H), nonneg=False,
            where=f"timesfm:{model_id}:origins[{c0}:{c0+len(chunk)}]",
        )
        yhat[c0 : c0 + len(chunk)] = block
        del contexts

    ytrue = targets_for_origins(Y, origins, H)
    return yhat, ytrue
