"""External direct RF reconciliation; not a new canonical head.

NumPy-only protocol, mapping and selection contracts. scikit-learn is imported
only by an explicitly invoked fit. No data, checkpoint, or test is read here.
"""
from __future__ import annotations

# Resolve the approved project layout while retaining sealed source-path keys.
from src.support.paths.project import project_file


import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs/baselines/random_forest.json"
ROW_FIELDS = {"dataset", "base", "horizon", "seed", "candidate_id", "split",
              "val_l1", "reference_val_l1"}


def json_digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_protocol(path=PROTOCOL_PATH):
    p = json.loads(project_file(Path(path)).read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1, "experiment_id": "external-rf-v1",
        "reference_id": "spiliotis_rf_adapted", "source_experiment_id": "main",
        "role": "external_reference_not_canonical_head",
        "evidence_status": "post_test_exposure_exploratory_owner_authorized",
        "datasets": ["walmart", "gefcom2017f"],
        "bases": ["chronos2", "timesfm25"],
        "horizons": {"walmart": [1, 7, 14], "gefcom2017f": [1, 6, 24]},
        "nodes": {"walmart": 44, "gefcom2017f": 158},
        "bottom_nodes": {"walmart": 30, "gefcom2017f": 141},
        "seeds": [42, 43, 44], "n_estimators": 150,
        "feature_rules": ["sqrt", "one_third", "all"],
        "min_samples_leaf": [10, 25, 50], "criterion": "squared_error",
        "bootstrap": True, "max_depth": None,
        "input": "all_node_untruncated_base_forecasts_at_same_lead",
        "target": "one_original_scale_bottom_series_per_forest",
        "parameterization": "direct_not_residual",
        "fit_scope": ["dataset", "base", "horizon", "seed", "bottom_node"],
        "selection_scope": ["dataset"],
        "selection_metric": "equal_cell_mean_validation_all_node_l1_over_unreconciled_base_l1",
        "tie_break": ["larger_min_samples_leaf", "smaller_feature_count", "candidate_id"],
        "node_seed_rule": "SeedSequence_seed_bottom_index_uint32",
        "refit": "train_only_same_seed_verify_saved_validation_predictions",
        "refit_prediction_rtol": 1e-12, "refit_prediction_atol": 1e-10,
        "sklearn_version": "1.9.1",
        "training_policy": "separate_owner_authorization_clean_commit_and_runtime_receipt_required",
        "test_policy": "owner_authorized_only_after_selected_checkpoint_manifest_freeze",
        "reporting_policy": "retain_all_cells_and_report_regardless_of_direction_no_main_overwrite",
    }
    if p != expected:
        raise ValueError("External RF protocol drift: register an amendment before changing it")
    return p


def candidates(p):
    return {f"{rule}-leaf{leaf}": {"feature_rule": rule, "min_samples_leaf": leaf}
            for rule in p["feature_rules"] for leaf in p["min_samples_leaf"]}


def feature_count(rule, n):
    if type(n) is not int or n < 1:
        raise ValueError("feature dimension must be a positive integer")
    if rule == "sqrt":
        return max(1, math.isqrt(n))
    if rule == "one_third":
        return max(1, n // 3)
    if rule == "all":
        return n
    raise ValueError("unknown feature rule")


def task_plan(p):
    grid = candidates(p)
    jobs = [dict(dataset=d, base=b, candidate_id=c)
            for d in p["datasets"] for b in p["bases"] for c in grid]
    cells_per_dataset = len(p["bases"]) * len(p["seeds"])
    selected_forests = sum(cells_per_dataset * len(p["horizons"][d]) *
                           p["bottom_nodes"][d] for d in p["datasets"])
    return dict(experiment_id=p["experiment_id"], protocol_sha256=json_digest(p),
        execution_authorized=False, test_stage_available=True, jobs=jobs,
        search_units=sum(len(p["horizons"][j["dataset"]]) * len(p["seeds"]) for j in jobs),
        selected_units=sum(cells_per_dataset * len(p["horizons"][d]) for d in p["datasets"]),
        search_forests=len(grid) * selected_forests,
        selected_refit_forests=selected_forests,
        search_trees=len(grid) * selected_forests * p["n_estimators"],
        refit_trees=selected_forests * p["n_estimators"],
        candidates={d: {c: {**v, "max_features": feature_count(v["feature_rule"], p["nodes"][d])}
                       for c, v in grid.items()} for d in p["datasets"]})


def forest_parameters(p, candidate_id, n_features, seed, bottom_index, n_jobs):
    if type(seed) is not int or seed not in p["seeds"]:
        raise ValueError("unregistered seed")
    if type(bottom_index) is not int or bottom_index < 0:
        raise ValueError("invalid bottom index")
    if type(n_jobs) is not int or n_jobs < 1:
        raise ValueError("n_jobs must be a positive allocated CPU count, never -1")
    c = candidates(p)[candidate_id]
    state = int(np.random.SeedSequence([seed, bottom_index]).generate_state(1)[0])
    return dict(n_estimators=p["n_estimators"], criterion=p["criterion"],
        max_depth=None, min_samples_split=2, min_samples_leaf=c["min_samples_leaf"],
        min_weight_fraction_leaf=0.0,
        max_features=feature_count(c["feature_rule"], n_features), max_leaf_nodes=None,
        min_impurity_decrease=0.0, bootstrap=True, oob_score=False,
        n_jobs=n_jobs, random_state=state, verbose=0, warm_start=False,
        ccp_alpha=0.0, max_samples=None, monotonic_cst=None)


def matrix(value, columns, name, *, nonnegative=False):
    value = np.asarray(value, dtype=np.float64)
    if (value.ndim != 2 or not len(value) or value.shape[1] != columns
            or not np.isfinite(value).all()):
        raise ValueError(f"invalid {name} shape or nonfinite values")
    if nonnegative and (value < 0).any():
        raise ValueError(f"negative {name}")
    return value


def hierarchy_contract(S, bottom_idx):
    S = np.asarray(S, dtype=np.float64)
    raw_idx = np.asarray(bottom_idx)
    if (S.ndim != 2 or not all(S.shape) or not np.isfinite(S).all()
            or not np.isin(S, [0, 1]).all()):
        raise ValueError("invalid summing matrix")
    if (raw_idx.ndim != 1 or not np.issubdtype(raw_idx.dtype, np.integer)
            or len(raw_idx) != S.shape[1] or len(set(raw_idx)) != len(raw_idx)
            or (raw_idx < 0).any() or (raw_idx >= S.shape[0]).any()):
        raise ValueError("invalid bottom indices")
    idx = raw_idx.astype(np.int64)
    if not np.array_equal(S[idx], np.eye(S.shape[1])):
        raise ValueError("bottom rows must be the ordered identity")
    return S, idx


def fit_validation(p, candidate_id, seed, S, bottom_idx, x_train, y_train,
                   x_val, *, n_jobs=1, save_model=None, factory=None):
    """Fit bottom forests serially; never accept validation targets for fitting.

    save_model(index, estimator) can persist each trusted, locally fitted forest
    before it is released. No collection of all bottom forests is kept in RAM.
    A factory hook permits contract tests without importing or fitting sklearn.
    """
    S, idx = hierarchy_contract(S, bottom_idx)
    xt = matrix(x_train, len(S), "training forecasts")
    yt = matrix(y_train, len(S), "training targets", nonnegative=True)
    xv = matrix(x_val, len(S), "validation forecasts")
    if len(xt) != len(yt) or not np.allclose(yt, yt[:, idx] @ S.T, rtol=1e-12, atol=1e-10):
        raise ValueError("training target alignment/coherence failed")
    params = forest_parameters(p, candidate_id, len(S), seed, 0, n_jobs)
    if factory is None:
        from importlib.metadata import version
        if version("scikit-learn") != p["sklearn_version"]:
            raise ValueError("scikit-learn version differs from the frozen protocol")
        from sklearn.ensemble import RandomForestRegressor
        factory = RandomForestRegressor
    bottom = np.empty((len(xv), len(idx)), dtype=np.float64)
    fitting_s = predicting_s = 0.0
    tree_nodes = 0
    for j, col in enumerate(idx):
        params = forest_parameters(p, candidate_id, len(S), seed, j, n_jobs)
        model = factory(**params)
        started = time.perf_counter()
        model.fit(xt, yt[:, col])
        fitting_s += time.perf_counter() - started
        started = time.perf_counter()
        pred = np.asarray(model.predict(xv), dtype=np.float64)
        predicting_s += time.perf_counter() - started
        if pred.shape != (len(xv),) or not np.isfinite(pred).all() or (pred < 0).any():
            raise ValueError("RF bottom predictions violate shape/finiteness/nonnegativity")
        bottom[:, j] = pred
        tree_nodes += sum(int(t.tree_.node_count) for t in model.estimators_)
        if save_model is not None:
            save_model(j, model)
        del model
    prediction = bottom @ S.T
    if not np.isfinite(prediction).all():
        raise ValueError("nonfinite aggregated prediction")
    return prediction, dict(fit_seconds=fitting_s, validation_predict_seconds=predicting_s,
                           tree_nodes=tree_nodes, forests=len(idx), n_jobs=n_jobs)


def validation_row(dataset, base, horizon, seed, candidate_id, prediction, truth, raw_base):
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.ndim != 2:
        raise ValueError("validation prediction must be a matrix")
    prediction = matrix(prediction, prediction.shape[1], "validation prediction", nonnegative=True)
    truth = matrix(truth, prediction.shape[1], "validation truth", nonnegative=True)
    raw_base = matrix(raw_base, prediction.shape[1], "raw validation base")
    if prediction.shape != truth.shape or prediction.shape != raw_base.shape:
        raise ValueError("validation arrays are not aligned")
    val_l1 = float(np.abs(prediction - truth).mean())
    reference = float(np.abs(raw_base - truth).mean())
    if not np.isfinite([val_l1, reference]).all() or reference <= 0:
        raise ValueError("invalid validation loss or nonpositive raw-base reference")
    return dict(dataset=dataset, base=base, horizon=horizon, seed=seed,
                candidate_id=candidate_id, split="val", val_l1=val_l1,
                reference_val_l1=reference)


def select_candidates(p, rows):
    """One candidate per dataset; require the complete prespecified val grid."""
    grid = candidates(p)
    expected = {(d, b, h, s, c) for d in p["datasets"] for b in p["bases"]
                for h in p["horizons"][d] for s in p["seeds"] for c in grid}
    seen, references, ratios = set(), {}, {}
    for row in rows:
        if set(row) != ROW_FIELDS or row["split"] != "val":
            raise ValueError("selection accepts only the exact validation-row schema")
        if type(row["horizon"]) is not int or type(row["seed"]) is not int:
            raise ValueError("selection horizon/seed must be integers")
        key = (row["dataset"], row["base"], row["horizon"], row["seed"], row["candidate_id"])
        if key not in expected or key in seen:
            raise ValueError("unknown or duplicate validation cell")
        seen.add(key)
        loss, reference = row["val_l1"], row["reference_val_l1"]
        if (isinstance(loss, bool) or isinstance(reference, bool)
                or not isinstance(loss, (int, float)) or not isinstance(reference, (int, float))
                or not math.isfinite(loss) or not math.isfinite(reference)
                or loss < 0 or reference <= 0):
            raise ValueError("invalid validation loss/reference")
        ref_key = key[:3]
        if ref_key in references and references[ref_key] != reference:
            raise ValueError("raw-base reference must agree across candidates and seeds")
        references[ref_key] = reference
        ratios.setdefault((key[0], key[-1]), []).append(loss / reference)
    if seen != expected:
        raise ValueError(f"incomplete validation grid: {len(seen)}/{len(expected)}")
    scores, selected = {}, {}
    for d in p["datasets"]:
        scores[d] = {c: math.fsum(sorted(ratios[d, c])) / len(ratios[d, c]) for c in grid}
        selected[d] = min(grid, key=lambda c: (scores[d][c], -grid[c]["min_samples_leaf"],
                             feature_count(grid[c]["feature_rule"], p["nodes"][d]), c))
    return dict(protocol_sha256=json_digest(p), selected=selected, scores=scores,
                rows_checked=len(rows), test_used=False)
