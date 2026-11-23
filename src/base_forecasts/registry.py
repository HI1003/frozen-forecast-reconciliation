"""Registry of the frozen base forecasters.

Every base is a pretrained time series foundation model used in FROZEN
inference: no dataset-specific gradient update, no fine-tuning, no adapter.
Each key owns a file under `results/raw_local/base_forecasts/{dataset}/`.
Non-main experiments have an explicit experiment suffix in the filename, so
swapping checkpoints can never overwrite forecasts produced by another model — the
reason the generic key `tsfm` was retired.

  chronos2        MAIN     amazon/chronos-2                 (~120M, ctx<=8192)
  timesfm25       MAIN     google/timesfm-2.5-200m-pytorch  (~200M, ctx<=16k)
  chronos2_synth  SENS     autogluon/chronos-2-synth        (synthetic-only pretraining)

The two MAIN bases carry RQ-D: do the nine reconciliation heads rank the same
way under two independently pretrained frozen TSFMs?  They must therefore be
run on byte-identical origins, contexts, and point statistic (q50).

`chronos2_synth` is a SENSITIVITY base only.  Its pretraining corpus is
documented as synthetic-only, so comparing it against `chronos2` bounds how
much of the Chronos result could be explained by real public data in the
Chronos pretraining mix.  It is scoped to Walmart and never enters the main
tables; see docs/EXPERIMENT_DESIGN.md for what it does and does not establish.
"""
from __future__ import annotations



from dataclasses import dataclass


@dataclass(frozen=True)
class BaseSpec:
    key: str
    model_id: str
    family: str                          # "chronos" | "timesfm" — selects the adapter
    role: str                            # "main" | "sensitivity"
    note: str
    revision: str | None = None          # immutable HF commit for formal main
    allowed_datasets: tuple[str, ...] | None = None   # None = every dataset


BASE_MODELS: dict[str, BaseSpec] = {
    "chronos2": BaseSpec(
        key="chronos2",
        model_id="amazon/chronos-2",
        family="chronos",
        role="main",
        note="frozen zero-gradient inference; q50 point forecast",
        revision="29ec3766d36d6f73f0696f85560a422f50e8498c",
    ),
    "timesfm25": BaseSpec(
        key="timesfm25",
        model_id="google/timesfm-2.5-200m-pytorch",
        family="timesfm",
        role="main",
        note="frozen zero-gradient inference; point forecast verified == q50 channel",
        revision="1d952420fba87f3c6dee4f240de0f1a0fbc790e3",
    ),
    "chronos2_synth": BaseSpec(
        key="chronos2_synth",
        model_id="autogluon/chronos-2-synth",
        family="chronos",
        role="sensitivity",
        note="synthetic-only pretraining; Walmart contamination-sensitivity arm only",
        revision="3607918a9fd027d5c465d8213e46b98e2c041cea",
        # Scoped in the registry, not just in prose: the synthetic-only arm
        # bounds contamination for the Chronos pretraining mix on Walmart and
        # supports no claim about GEFCom, so it must not be runnable there.
        allowed_datasets=("walmart",),
    ),
}

MAIN_BASES = tuple(k for k, s in BASE_MODELS.items() if s.role == "main")
SENSITIVITY_BASES = tuple(k for k, s in BASE_MODELS.items() if s.role == "sensitivity")


def get_base(key: str) -> BaseSpec:
    """Resolve a registry key.  Unknown keys fail loudly — never fall back to a
    different checkpoint, which would silently mislabel the frozen forecasts."""
    if key not in BASE_MODELS:
        raise KeyError(
            f"unknown base '{key}'; registered bases are {sorted(BASE_MODELS)}"
        )
    return BASE_MODELS[key]


def check_base_dataset(key: str, dataset: str) -> BaseSpec:
    """Resolve a base and reject dataset/base pairs the design does not admit.

    Called from every entry point (step1..step4) so the scope of a sensitivity
    base cannot be widened just by running a later stage with a different
    config.
    """
    spec = get_base(key)
    if spec.allowed_datasets is not None and dataset not in spec.allowed_datasets:
        raise SystemExit(
            f"base '{key}' is a {spec.role} arm restricted to "
            f"{list(spec.allowed_datasets)}; it supports no claim about "
            f"'{dataset}'. Refusing to run."
        )
    return spec
