"""Frozen base forecasts from the Chronos-2 family.

Registered checkpoints (src/base_forecasts/registry.py):
  amazon/chronos-2          main base, ~120M params, context limit 8192
  autogluon/chronos-2-synth sensitivity base, synthetic-only pretraining

The model is used in FROZEN inference: no gradient update, no fine-tuning.
Each node's history is an independent univariate context (channel-independent
inference), which makes the base forecasts generally incoherent across the
hierarchy — exactly the condition the reconciliation study needs.  Point
forecasts are the median (0.5 quantile), matching the TimesFM adapter.

There is deliberately NO silent checkpoint fallback: if the requested weights
fail to load, the run aborts.  Substituting a different model would produce a
.npz labelled with the wrong base and silently break the RQ-D comparison.

API contract (verified 2026-07-24 against amazon-science/chronos-forecasting):
`predict_quantiles` takes `inputs=` (a list of 1D tensors or a batch tensor)
across the pipeline family; return structures differ (batched tensor tuple
vs a per-task tensor list whose univariate entries are shaped `(1, H, Q)`), so
`_predict_quantile_batch` normalizes both to a `(B, H)` float array.  Older
releases with `context=` are handled by fallback.
After installing, verify the contract before a full run:
`python scripts/checks/check_base_forecasts.py --config configs/data/walmart.yaml
--base chronos2 --level api`.

Memory: contexts are built and released per origin chunk (streaming), so peak
memory is O(origin_chunk x n x context_len) instead of the full
origins x nodes context list.

Out of scope by design (docs/EXPERIMENT_DESIGN.md §3.5): fine-tuning, LoRA,
hierarchy-aware adapters, and Chronos-2 grouped multivariate inference.  The
contribution is the reconciliation axis; the base predictor is an instrument.
"""
from __future__ import annotations



import numpy as np
import torch

from src.data.datasets import HTSData, require_full_context, targets_for_origins
from src.support.storage.forecasts import check_forecast_block


def load_pipeline(model_id: str, device: str, revision: str | None = None):
    try:
        from chronos import BaseChronosPipeline
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "pip install chronos-forecasting (see requirements.txt); then run "
            "scripts/checks/check_base_forecasts.py --config configs/data/walmart.yaml "
            "--base chronos2 --level api"
        ) from e
    return BaseChronosPipeline.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        revision=revision,
    )


def _predict_quantile_batch(pipe, contexts: list[torch.Tensor], H: int) -> np.ndarray:
    """Median forecasts for a batch of univariate contexts, normalized to (B, H).

    Handles both current pipeline generations: unified `inputs=` kwarg (with a
    `context=` fallback for older releases) and either return structure —
    a batched `(B, H, Q)` tensor tuple or a per-task list of `(1, H, Q)`
    tensors.  Only one quantile is requested, so every batch item must contain
    exactly H forecast values after singleton series/quantile axes are removed.
    """
    try:
        out = pipe.predict_quantiles(
            inputs=contexts, prediction_length=H, quantile_levels=[0.5]
        )
    except TypeError as e:  # older releases use `context=`
        # Do not mask an unrelated TypeError raised inside a current pipeline.
        if "unexpected keyword argument 'inputs'" not in str(e):
            raise
        out = pipe.predict_quantiles(
            context=contexts, prediction_length=H, quantile_levels=[0.5]
        )
    quantiles = out[0] if isinstance(out, tuple) else out
    B = len(contexts)

    if isinstance(quantiles, (list, tuple)):
        if len(quantiles) != B:
            raise ValueError(
                f"per-task quantile list length {len(quantiles)} != batch size {B}"
            )
        rows = []
        for i, item in enumerate(quantiles):
            q_i = torch.as_tensor(item)
            if q_i.numel() != H:
                raise ValueError(
                    f"quantile item {i} shape {tuple(q_i.shape)} contains "
                    f"{q_i.numel()} values; expected {H} for one univariate "
                    "series and one requested quantile"
                )
            rows.append(q_i.reshape(H))
        q = torch.stack(rows)
    else:
        q = torch.as_tensor(quantiles)
        if q.numel() != B * H:
            raise ValueError(
                f"batched quantile tensor shape {tuple(q.shape)} contains "
                f"{q.numel()} values; expected {B * H} for batch={B}, H={H}, Q=1"
            )
        q = q.reshape(B, H)

    return q.to(torch.float32).cpu().numpy().astype(np.float64)


def generate_chronos_forecasts(
    data: HTSData,
    origins: np.ndarray,
    model_id: str = "amazon/chronos-2",
    device: str = "cpu",
    batch_size: int = 256,
    origin_chunk: int = 64,
    pipe=None,
    revision: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (yhat, ytrue) with shape (num_origins, n, predict_len), original scale.

    Every origin carries a full `context_len` window (guaranteed by the origin
    enumeration), so no context is left-padded or truncated here — the two main
    bases see the identical information budget.
    """
    if pipe is None:
        pipe = load_pipeline(model_id, device, revision=revision)
    H = data.predict_len
    L = data.context_len
    n = data.hierarchy.n
    Y = data.Y
    require_full_context(origins, L, where=f"chronos:{model_id}")

    yhat = np.empty((len(origins), n, H), dtype=np.float64)
    for c0 in range(0, len(origins), origin_chunk):
        chunk = origins[c0 : c0 + origin_chunk]
        contexts = [
            torch.tensor(Y[t + 1 - L : t + 1, i], dtype=torch.float32)
            for t in chunk for i in range(n)
        ]
        preds = np.empty((len(contexts), H), dtype=np.float64)
        for s in range(0, len(contexts), batch_size):
            batch = contexts[s : s + batch_size]
            preds[s : s + len(batch)] = _predict_quantile_batch(pipe, batch, H)
        block = preds.reshape(len(chunk), n, H)
        check_forecast_block(block, (len(chunk), n, H), nonneg=False,
                             where=f"chronos:{model_id}:origins[{c0}:{c0+len(chunk)}]")
        yhat[c0 : c0 + len(chunk)] = block
        del contexts

    ytrue = targets_for_origins(Y, origins, H)
    return yhat, ytrue
