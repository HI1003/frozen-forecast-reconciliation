# Frozen forecast reconciliation

Reproduction code for the hierarchical forecasting study comparing nine reconciliation heads (BU/BUL/BUN, TD/TDL/TDN, HYB/HYBL/HYBN), classical and learned baselines, and soft coherence control. Within each dataset and foundation model, the methods share one set of frozen base forecasts. The tagged paper snapshot is **v1.0.0**.

## Rebuild the reported results on CPU

From the repository root, install NumPy, pandas and Matplotlib in a reporting environment (the study environment is listed in `requirements-reporting.txt`). These commands use the retained test results and validation choices, without model weights or competition downloads.

```shell
python scripts/checks/check_paper_artifacts.py
python scripts/analysis/render_accuracy_tables.py --check
python scripts/analysis/render_accuracy_tables.py --output runs/table2
python scripts/analysis/figures/render_rank_comparison.py --output runs/figure2
python scripts/analysis/figures/render_paper_figures.py --output runs/figures3-6
```

Table 2 contains 19 methods in 12 settings. Its source includes WAPE, sMASE and coherence nMAE. Figure 2 uses the archived mean ranks and non-rejection groups. Figures 3–6 cover strategy anchors, direct and residual mappings, fixed penalties and coherence budgets. Table 1 is supplied in `results/tables/data_en.tex`. Table 3 and the original machine-specific timing observations are in `results/cost/native_device_timing/`. Figure 1 is a conceptual diagram. Rebuilding these displays uses saved observations. Independent fitting and evaluation follow the workflow below.

## Prepare the competition inputs

Obtain the original inputs from their distributors. The code license covers the software. Each dataset and model has its own terms. The repository includes the hierarchy definitions and summing matrices. Install pandas, NumPy, PyYAML, torch and openpyxl for the preparation and validation commands.

| Dataset | Input | Source |
| --- | --- | --- |
| Walmart | `sales_train_evaluation.csv` | [M5 Forecasting Accuracy competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) |
| GEFCom2017F | `load.xlsx` from the Final Match archive | [Competition paper and supplementary data](https://doi.org/10.1016/j.ijforecast.2019.02.006) · [IEEE PES dataset record](https://ieee-pes-data-sharing.org/datasets/detail/ca195e5b-c607-4897-94b1-8f45b7baa7b6) |

```shell
python scripts/data/prepare_competition_data.py walmart /path/to/sales_train_evaluation.csv
python scripts/data/prepare_competition_data.py gefcom2017f /path/to/load.xlsx
python scripts/data/validate_data.py --config configs/data/walmart.yaml
python scripts/data/validate_data.py --config configs/data/gefcom2017f.yaml
```

Preparation verifies the timestamps, ordered node names and all-node values against the paper fingerprints before writing the CSVs. The Walmart hierarchy aggregates item sales by store and category. The GEFCom hierarchy aggregates 141 bottom meters after time interpolation of the seven missing hours. The forecasting protocol excludes origins affected by these imputed hours.

## Rerun the experiments

Follow [the experiment instructions](scripts/README.md) for the locked base-model revisions, forecast generation, train-only fitting, validation selection, held-out evaluation and analysis. Frozen inference uses Chronos-2 and TimesFM 2.5 with no dataset-specific gradient updates. Within each base-model/dataset pair, every reconciliation strategy uses the same base forecasts. Generating the forecasts requires access to the model checkpoints. The full experiment uses separate model and classical-baseline environments.

The `plan` command lists all candidate jobs without starting inference or fitting:

```shell
python scripts/experiments/run_paper_experiments.py plan
```

Source and tests are in `src/` and `tests/`. Experiment definitions are in `configs/`. Retained numerical outputs are in `results/`. Code is distributed under the [MIT License](LICENSE).
