# Running the paper experiments

Run commands from the repository root. `src/` implements the methods; this directory contains entry points for data checks, experiment execution and analysis. No scheduler, private archive or historical job IDs are needed.

## 1. Validate data and generate forecasts

The model run used Python 3.11.15. Use the pinned model environment in `requirements.txt`. Keep reporting and CPU baseline environments separate when reproducing their original runtimes. Install `requirements-baseline.txt` in the CPU environment. The unified runner supports `--methods` so the same run can be filled from both environments; implementations and inputs must remain unchanged.

```text
python scripts/data/validate_data.py --config configs/data/walmart.yaml
python scripts/data/validate_data.py --config configs/data/gefcom2017f.yaml
python scripts/checks/check_strategies.py
python scripts/experiments/generate_base_forecasts.py --config configs/data/walmart.yaml --base chronos2 --experiment-id reader_walmart --device cuda
python scripts/experiments/generate_base_forecasts.py --config configs/data/walmart.yaml --base timesfm25 --experiment-id reader_walmart --device cuda
python scripts/experiments/generate_base_forecasts.py --config configs/data/gefcom2017f.yaml --base chronos2 --experiment-id reader_gefcom --device cuda
python scripts/experiments/generate_base_forecasts.py --config configs/data/gefcom2017f.yaml --base timesfm25 --experiment-id reader_gefcom --device cuda
```

Generation requires a clean Git checkout, records exact checkpoint revisions, and refuses existing outputs. No gradient updates are made to base models. The sealed `main` namespace is read-only. See `generate_base_forecasts.py --help` and `check_base_forecasts.py --help` for adapter checks.

Create `runs/inputs.json` with these four entries. Paths are relative to the JSON file, not the shell's working directory:

```json
{
  "walmart/chronos2": "../results/raw_local/base_forecasts/walmart/walmart_chronos2__experiment-reader_walmart.npz",
  "walmart/timesfm25": "../results/raw_local/base_forecasts/walmart/walmart_timesfm25__experiment-reader_walmart.npz",
  "gefcom2017f/chronos2": "../results/raw_local/base_forecasts/gefcom2017f/gefcom2017f_chronos2__experiment-reader_gefcom.npz",
  "gefcom2017f/timesfm25": "../results/raw_local/base_forecasts/gefcom2017f/gefcom2017f_timesfm25__experiment-reader_gefcom.npz"
}
```

## 2. Fit and freeze the complete validation grid

`run_paper_experiments.py` has five stages: `plan`, `fit`, `select`, `evaluate`, `timing`. `paper_protocol.py` defines grids and selection; `experiment_storage.py` handles input/checkpoint hashes and immutable outputs. All computation is opt-in. `plan` only lists work.

```text
python scripts/experiments/run_paper_experiments.py plan
python scripts/experiments/run_paper_experiments.py fit --run runs/paper --inputs runs/inputs.json --group core --all --device cuda --methods BUN TDN HYBN bottom_mlp_plain td_features_mlp_plain allnode_mlp_plain BUL TDL HYBL BU TD HYB
```

In the CPU baseline environment, complete the same core grid:

```text
python scripts/experiments/run_paper_experiments.py fit --run runs/paper --group core --all --methods base MO-L1 MO-L2 ols wls_var wls_struct mint_shrink RF LightGBM Elasso_adapted --jobs 1
python scripts/experiments/run_paper_experiments.py select --run runs/paper --group core
```

Then, in the model environment:

```text
python scripts/experiments/run_paper_experiments.py fit --run runs/paper --group penalty --all --device cuda
python scripts/experiments/run_paper_experiments.py select --run runs/paper --group penalty
python scripts/experiments/run_paper_experiments.py fit --run runs/paper --group budget --all --device cuda
python scripts/experiments/run_paper_experiments.py select --run runs/paper --group budget
```

| Group | Candidate fits | Validation choice |
|---|---:|---|
| Core MLP | 5,832 | Six branches, nine architectures, three weight decays, 36 base/lead/seed cells |
| Core linear heads | 324 | BUL/TDL/HYBL; weight decay within each cell |
| RF / LightGBM | 324 each | One configuration per dataset across 18 cells |
| Adapted Elasso | 252 | 21 descending penalties per base/lead cell; warm starts; one penalty per dataset |
| Deterministic / classical | 120 preparations | Ten methods across 12 settings |
| HYBN penalty | 972 | Nine nonzero penalties at the core-selected architecture; zero reuses core weights |
| HYBN budget | 324 | Three budgets, feasible-first weight-decay choice; 108 selected fits |

MLP selection minimizes validation prediction L1, then earlier epoch, then smaller weight decay. Architectures minimize mean relative validation L1 over 18 cells; ties use parameter count then architecture ID. Every candidate remains available in `candidates/`. No train+validation refit occurs.

The budget experiment is a distinct historical lineage: it retains the validation-frozen architecture in `results/evidence/aligned/selected_architectures.json` (Walmart `[64,32,32]`, GEFCom `[256,128]`), rather than inheriting the later core-search winner. Budgets are fractions `.005/.01/.02`; output names use percentages `HYBN-eps0.5/1/2`. This reruns the budget fits conditional on the published architecture selection, not the superseded alignment search.

Use `plan --group core --list-tasks` to obtain stable zero-based indices and `fit --task INDEX` to run one candidate. Elasso tasks must follow ascending `alpha_index` within each cell to preserve warm starts. One process holds the run lock at a time. Completed candidates can resume; partial candidates fail explicitly. A new input/config/code version needs a new run directory.

## 3. Evaluate once, after all three selections are frozen

Review the three `*.selection.json` files before acknowledging the freeze. Run the neural/deterministic subset in the model environment, then the baselines in the CPU environment:

```text
python scripts/experiments/run_paper_experiments.py evaluate --run runs/paper --all --device cuda --confirm-selection-frozen --methods BU BUL BUN TD TDL TDN HYB HYBL HYBN bottom_mlp_plain td_features_mlp_plain allnode_mlp_plain HYBN-lam0.001 HYBN-lam0.003 HYBN-lam0.01 HYBN-lam0.03 HYBN-lam0.1 HYBN-lam0.3 HYBN-lam1 HYBN-lam3 HYBN-lam10 HYBN-eps0.5 HYBN-eps1 HYBN-eps2
python scripts/experiments/run_paper_experiments.py evaluate --run runs/paper --all --confirm-selection-frozen --methods base MO-L1 MO-L2 ols wls_var wls_struct mint_shrink RF LightGBM Elasso_adapted
```

The first test access closes fitting/selection. Each checkpoint receives an exclusive start marker before reading test arrays. Complete tests are skipped on resume; interrupted tests cannot be silently retried. Never tune after inspecting test results. Outputs include predictions, paired per-origin losses, level metrics and budget violations. Raw base predictions remain unclipped.

For a small single-configuration example, use `run_strategy.py fit --help` and `evaluate --help`. That example does not replace the complete search.

## 4. Analyze and inspect

In the reporting environment:

```text
python scripts/analysis/analyze_paper_results.py --run runs/paper
python scripts/checks/check_paper_artifacts.py
python scripts/analysis/render_accuracy_tables.py --check
python scripts/analysis/render_accuracy_tables.py --output runs/accuracy_tables
python scripts/analysis/figures/render_paper_figures.py --output runs/section5_figures
python scripts/analysis/figures/render_rank_comparison.py --output runs/rank_figures
```

Analysis writes `runs/paper/analysis/`: main/direct-residual/penalty/budget metrics, fit-level feasibility, candidate/search costs, paired tests, block intervals and pooled rank groups. It averages seed losses, not forecasts. The original 13-comparison recipe, exploratory 90-comparison families and pooled 342-hypothesis bootstrap remain separate. Applying a historical recipe to new forecasts is not a recreation of its original preregistration. The three renderers above read the **retained paper results**, while the new analysis runner reads your independent run.

Optional `timing --all --methods ...` repeats deployment prediction after evaluation, with batch size 128 and one CPU thread. Use the appropriate model/CPU environment as above. Timings record the current hardware; RF/LightGBM wall times include sequential model loading. They do not reproduce the archived Table 3 observations. No timing outcome changes selection.

See the root README for competition file preparation. Run the test suite with `python -m pytest tests` after preparing both datasets. Adapter tests use test doubles.
