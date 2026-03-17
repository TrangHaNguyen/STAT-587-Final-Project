#!/usr/bin/env python3
from typing import Any, cast
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import time
import numpy as np

from H_prep import clean_data, import_data, to_binary_class
from H_eval import (
    TEST_SELECTION_CRITERIA,
    get_or_compute_final_metrics,
    _metrics_stage_name,
    rank_models_by_metrics,
    select_non_degenerate_plot_model,
    save_best_model_plots_from_gridsearch_all_params,
    comparison_row_from_metrics,
    build_base_style_comparison_df,
    register_global_model_candidates,
    write_base_style_latex_table,
)
from H_helpers import get_cwd
from H_search_history import (
    append_search_history,
    append_search_run,
    get_checkpoint_dir,
    get_git_commit,
    history_has_entry,
    load_search_checkpoint,
    now_iso,
    save_search_checkpoint,
    search_checkpoint_exists,
)
from model_grids import (
    SVM_LINEAR_C_GRID_OPTIONS,
    SVM_GAMMA_GRID_OPTIONS,
    SVM_DEGREE_GRID_OPTIONS,
    RANDOM_SEED,
    SVM_CLASS_WEIGHT,
    SVM_TOL,
    TEST_SIZE,
    TIME_SERIES_CV_SPLITS,
    TRAIN_TEST_SHUFFLE,
)

cwd = get_cwd("STAT-587-Final-Project")
MODEL_N_JOBS = int(os.getenv("MODEL_N_JOBS", "-1"))
GRID_VERSION = os.getenv("GRID_VERSION", "v1")
SEARCH_NOTES = os.getenv("SEARCH_NOTES", "")
GRIDSEARCH_VERBOSE = int(os.getenv("GRIDSEARCH_VERBOSE", "0"))


def _keep_raw_stock_ohlcv(X: pd.DataFrame) -> pd.DataFrame:
    idx = pd.IndexSlice
    metrics = ['Open', 'Close', 'High', 'Low', 'Volume']
    return X.loc[:, idx[metrics, 'Stocks', :]].copy()


def _assert_no_lag_features(X: pd.DataFrame) -> None:
    lag_like_cols = [col for col in X.columns if "_lag" in str(col).lower() or " lag " in str(col).lower()]
    if lag_like_cols:
        sample = lag_like_cols[:5]
        raise ValueError(f"Raw SVM should not include lag features. Found lag-like columns: {sample}")


def _as_sortable_numeric(value):
    try:
        return float(value)
    except Exception:
        return float("inf")


def _effective_svm_gamma(value, n_features: int) -> float:
    n_features = max(1, int(n_features))
    if value in {"scale", "auto"}:
        return 1.0 / float(n_features)
    try:
        return float(value)
    except Exception:
        return float("inf")


class _OneSERefit:
    """Picklable callable implementing the 1-SE rule for GridSearchCV refit."""

    def __init__(self, complexity_cols: list[str], sort_value_map: dict[str, callable] | None = None):
        self.complexity_cols = complexity_cols
        self.sort_value_map = sort_value_map

    def __call__(self, cv_results):
        mean = np.asarray(cv_results["mean_test_score"], dtype=float)
        std = np.asarray(cv_results["std_test_score"], dtype=float)
        se = std / np.sqrt(TIME_SERIES_CV_SPLITS)
        best_idx = int(np.argmax(mean))
        threshold = float(mean[best_idx] - se[best_idx])
        candidate_idx = np.where(mean >= threshold)[0]
        if len(candidate_idx) == 0:
            return best_idx

        def key_fn(i: int):
            complexity = [
                (self.sort_value_map or {}).get(col, _as_sortable_numeric)(cv_results[f"param_{col}"][i])
                for col in self.complexity_cols
            ]
            return tuple(complexity + [-float(mean[i])])

        return int(min(candidate_idx, key=key_fn))


def make_one_se_refit(
    complexity_cols: list[str], sort_value_map: dict[str, callable] | None = None
) -> "_OneSERefit":
    """Return a GridSearchCV refit callable implementing the 1-SE rule."""
    return _OneSERefit(complexity_cols, sort_value_map)


def _fit_or_load_search(
    checkpoint_dir,
    stage_name: str,
    search_obj,
    X_train,
    y_train,
    history_path,
    run_time: str,
    model_name: str,
    grid_label: str,
):
    if search_checkpoint_exists(checkpoint_dir, stage_name):
        print(f"Loading checkpoint for {model_name} from {checkpoint_dir / stage_name}")
        loaded = load_search_checkpoint(checkpoint_dir, stage_name)
        if not history_has_entry(history_path, model_name, grid_label):
            append_search_history(
                history_path=history_path,
                cv_results=loaded.cv_results_,
                run_time=run_time,
                model_name=model_name,
                search_type="grid",
                grid_version=grid_label,
                notes=SEARCH_NOTES,
                best_params=loaded.best_params_,
            )
        return loaded
    search_obj.fit(X_train, y_train)
    append_search_history(
        history_path=history_path,
        cv_results=search_obj.cv_results_,
        run_time=run_time,
        model_name=model_name,
        search_type="grid",
        grid_version=grid_label,
        notes=SEARCH_NOTES,
        best_params=search_obj.best_params_,
    )
    save_search_checkpoint(checkpoint_dir, stage_name, search_obj)
    return search_obj


if __name__ == "__main__":
    run_start = time.time()
    run_time = now_iso()
    print(f"MODEL_N_JOBS={MODEL_N_JOBS} (set env MODEL_N_JOBS to override)")
    print(f"GRID_VERSION={GRID_VERSION}")
    grid_label = GRID_VERSION
    output_prefix = "8yrs"
    history_path = cwd / "output" / "8yrs_search_history_base_svm.csv"
    runs_path = cwd / "output" / "8yrs_search_runs.csv"
    # Same checkpoint_dir as base_SVM.py so raw-variant checkpoints are shared/reused.
    checkpoint_dir = get_checkpoint_dir(cwd / "output", "base_SVM", f"{output_prefix}_{grid_label}")
    dataset_version = "testing=False,extra_features=False,cluster=False,corr_threshold=0.95,corr_level=0"

    DATA = import_data(extra_features=False, testing=False, cluster=False, n_clusters=100, corr_threshold=0.95, corr_level=0)
    parameters_ = {
        "raw": True,
        "extra_features": False,
        "lag_period": [1],
        "lookback_period": 0,
    }
    X, y_regression = cast(Any, clean_data(*DATA, **parameters_))
    X = _keep_raw_stock_ohlcv(X)
    X.columns = [f"{metric}_{ticker}" for metric, _, ticker in X.columns]
    _assert_no_lag_features(X)
    print(f"Feature matrix shape: {X.shape[0]} rows, {X.shape[1]} columns.")

    y_classification = to_binary_class(y_regression)
    print(f"Final shape - X: {X.shape}, y: {y_classification.shape}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_classification, test_size=TEST_SIZE, random_state=RANDOM_SEED, shuffle=TRAIN_TEST_SHUFFLE
    )
    gamma_sort = lambda value: _effective_svm_gamma(value, X_train.shape[1])
    tscv = TimeSeriesSplit(n_splits=TIME_SERIES_CV_SPLITS)

    # ===================================================================
    # RAW OHLCV — 3 kernel variants (reuse checkpoints from base_SVM.py)
    # ===================================================================
    print("\n\n------- Raw Linear SVM Model -------")
    svm_linear = SVC(kernel="linear", cache_size=1000, class_weight=SVM_CLASS_WEIGHT, gamma="scale", random_state=RANDOM_SEED, tol=SVM_TOL)
    svm_linear_pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", svm_linear)])
    grid_search_linear = GridSearchCV(
        svm_linear_pipeline, {"classifier__C": SVM_LINEAR_C_GRID_OPTIONS}, cv=tscv,
        scoring="balanced_accuracy", n_jobs=MODEL_N_JOBS, verbose=GRIDSEARCH_VERBOSE,
        return_train_score=True, refit=make_one_se_refit(["classifier__C"])
    )
    grid_search_linear = _fit_or_load_search(
        checkpoint_dir, "base_svm_linear", grid_search_linear, X_train, y_train,
        history_path, run_time, "Raw_SVM_linear", grid_label,
    )
    linear_results = get_or_compute_final_metrics(
        checkpoint_dir, _metrics_stage_name("Raw Linear Ker. SVM"),
        grid_search_linear.best_estimator_, X_train, y_train, X_test, y_test,
        label="Raw Linear Ker. SVM"
    ).copy()

    print("\n\n------- Raw RBF SVM Model -------")
    svm_rbf = SVC(kernel="rbf", cache_size=1000, class_weight=SVM_CLASS_WEIGHT, gamma="scale", random_state=RANDOM_SEED, tol=SVM_TOL)
    svm_rbf_pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", svm_rbf)])
    grid_search_rbf = GridSearchCV(
        svm_rbf_pipeline,
        {"classifier__C": SVM_LINEAR_C_GRID_OPTIONS, "classifier__gamma": SVM_GAMMA_GRID_OPTIONS},
        cv=tscv, scoring="balanced_accuracy", n_jobs=MODEL_N_JOBS, verbose=GRIDSEARCH_VERBOSE,
        return_train_score=True,
        refit=make_one_se_refit(
            ["classifier__C", "classifier__gamma"],
            sort_value_map={"classifier__gamma": gamma_sort},
        )
    )
    grid_search_rbf = _fit_or_load_search(
        checkpoint_dir, "base_svm_rbf", grid_search_rbf, X_train, y_train,
        history_path, run_time, "Raw_SVM_rbf", grid_label,
    )
    rbf_results = get_or_compute_final_metrics(
        checkpoint_dir, _metrics_stage_name("Raw RBF Ker. SVM"),
        grid_search_rbf.best_estimator_, X_train, y_train, X_test, y_test,
        label="Raw RBF Ker. SVM"
    ).copy()

    print("\n\n------- Raw Polynomial SVM Model -------")
    svm_poly = SVC(kernel="poly", cache_size=1000, class_weight=SVM_CLASS_WEIGHT, gamma="scale", random_state=RANDOM_SEED, tol=SVM_TOL)
    svm_poly_pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", svm_poly)])
    grid_search_poly = GridSearchCV(
        svm_poly_pipeline,
        {
            "classifier__C": SVM_LINEAR_C_GRID_OPTIONS,
            "classifier__gamma": SVM_GAMMA_GRID_OPTIONS,
            "classifier__degree": SVM_DEGREE_GRID_OPTIONS,
        },
        cv=tscv, scoring="balanced_accuracy", n_jobs=MODEL_N_JOBS, verbose=GRIDSEARCH_VERBOSE,
        return_train_score=True,
        refit=make_one_se_refit(
            ["classifier__C", "classifier__degree", "classifier__gamma"],
            sort_value_map={"classifier__gamma": gamma_sort},
        )
    )
    grid_search_poly = _fit_or_load_search(
        checkpoint_dir, "base_svm_poly", grid_search_poly, X_train, y_train,
        history_path, run_time, "Raw_SVM_poly", grid_label,
    )
    poly_results = get_or_compute_final_metrics(
        checkpoint_dir, _metrics_stage_name("Raw Poly. Ker. SVM"),
        grid_search_poly.best_estimator_, X_train, y_train, X_test, y_test,
        label="Raw Poly. Ker. SVM"
    ).copy()

    # ===================================================================
    # DAY-OF-WEEK EXTENSION
    # Add one-hot encoded day-of-week (Mon–Thu, drop Fri) to raw OHLCV
    # then re-run the same 3 SVM kernel variants.
    # ===================================================================
    print("\n========== Adding Day-of-Week Features ==========")
    dow_dummies = pd.get_dummies(X.index.dayofweek, prefix='DOW').astype(float)
    dow_dummies.index = X.index
    dow_dummies = dow_dummies.iloc[:, :-1]   # drop last column (Fri) to avoid dummy trap
    print(f"Day-of-week columns added: {list(dow_dummies.columns)}")

    X_dow = pd.concat([X, dow_dummies], axis=1)
    X_train_dow, X_test_dow, y_train_dow, y_test_dow = train_test_split(
        X_dow, y_classification, test_size=TEST_SIZE, random_state=RANDOM_SEED, shuffle=TRAIN_TEST_SHUFFLE
    )
    print(f"Feature matrix with DOW: {X_dow.shape[1]} columns")
    gamma_sort_dow = lambda value: _effective_svm_gamma(value, X_train_dow.shape[1])

    print("\n\n------- Linear SVM + DOW Model -------")
    svm_linear_dow = SVC(kernel="linear", cache_size=1000, class_weight=SVM_CLASS_WEIGHT, gamma="scale", random_state=RANDOM_SEED, tol=SVM_TOL)
    svm_linear_dow_pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", svm_linear_dow)])
    grid_search_linear_dow = GridSearchCV(
        svm_linear_dow_pipeline, {"classifier__C": SVM_LINEAR_C_GRID_OPTIONS}, cv=tscv,
        scoring="balanced_accuracy", n_jobs=MODEL_N_JOBS, verbose=GRIDSEARCH_VERBOSE,
        return_train_score=True, refit=make_one_se_refit(["classifier__C"])
    )
    grid_search_linear_dow = _fit_or_load_search(
        checkpoint_dir, "base_svm_linear_dow", grid_search_linear_dow, X_train_dow, y_train_dow,
        history_path, run_time, "Raw_SVM_linear_dow", grid_label,
    )
    linear_dow_results = get_or_compute_final_metrics(
        checkpoint_dir, _metrics_stage_name("Linear Ker. SVM+DOW"),
        grid_search_linear_dow.best_estimator_, X_train_dow, y_train_dow, X_test_dow, y_test_dow,
        label="Linear Ker. SVM+DOW"
    ).copy()

    print("\n\n------- RBF SVM + DOW Model -------")
    svm_rbf_dow = SVC(kernel="rbf", cache_size=1000, class_weight=SVM_CLASS_WEIGHT, gamma="scale", random_state=RANDOM_SEED, tol=SVM_TOL)
    svm_rbf_dow_pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", svm_rbf_dow)])
    grid_search_rbf_dow = GridSearchCV(
        svm_rbf_dow_pipeline,
        {"classifier__C": SVM_LINEAR_C_GRID_OPTIONS, "classifier__gamma": SVM_GAMMA_GRID_OPTIONS},
        cv=tscv, scoring="balanced_accuracy", n_jobs=MODEL_N_JOBS, verbose=GRIDSEARCH_VERBOSE,
        return_train_score=True,
        refit=make_one_se_refit(
            ["classifier__C", "classifier__gamma"],
            sort_value_map={"classifier__gamma": gamma_sort_dow},
        )
    )
    grid_search_rbf_dow = _fit_or_load_search(
        checkpoint_dir, "base_svm_rbf_dow", grid_search_rbf_dow, X_train_dow, y_train_dow,
        history_path, run_time, "Raw_SVM_rbf_dow", grid_label,
    )
    rbf_dow_results = get_or_compute_final_metrics(
        checkpoint_dir, _metrics_stage_name("RBF Ker. SVM+DOW"),
        grid_search_rbf_dow.best_estimator_, X_train_dow, y_train_dow, X_test_dow, y_test_dow,
        label="RBF Ker. SVM+DOW"
    ).copy()

    print("\n\n------- Polynomial SVM + DOW Model -------")
    svm_poly_dow = SVC(kernel="poly", cache_size=1000, class_weight=SVM_CLASS_WEIGHT, gamma="scale", random_state=RANDOM_SEED, tol=SVM_TOL)
    svm_poly_dow_pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", svm_poly_dow)])
    grid_search_poly_dow = GridSearchCV(
        svm_poly_dow_pipeline,
        {
            "classifier__C": SVM_LINEAR_C_GRID_OPTIONS,
            "classifier__gamma": SVM_GAMMA_GRID_OPTIONS,
            "classifier__degree": SVM_DEGREE_GRID_OPTIONS,
        },
        cv=tscv, scoring="balanced_accuracy", n_jobs=MODEL_N_JOBS, verbose=GRIDSEARCH_VERBOSE,
        return_train_score=True,
        refit=make_one_se_refit(
            ["classifier__C", "classifier__degree", "classifier__gamma"],
            sort_value_map={"classifier__gamma": gamma_sort_dow},
        )
    )
    grid_search_poly_dow = _fit_or_load_search(
        checkpoint_dir, "base_svm_poly_dow", grid_search_poly_dow, X_train_dow, y_train_dow,
        history_path, run_time, "Raw_SVM_poly_dow", grid_label,
    )
    poly_dow_results = get_or_compute_final_metrics(
        checkpoint_dir, _metrics_stage_name("Poly. Ker. SVM+DOW"),
        grid_search_poly_dow.best_estimator_, X_train_dow, y_train_dow, X_test_dow, y_test_dow,
        label="Poly. Ker. SVM+DOW"
    ).copy()

    # ===================================================================
    # RANKING AND BEST MODEL PLOT (across all 6 models)
    # ===================================================================
    ranking_df = pd.DataFrame([
        {"Model": "Raw Linear SVM", **linear_results},
        {"Model": "Raw RBF SVM", **rbf_results},
        {"Model": "Raw Poly SVM", **poly_results},
        {"Model": "Linear SVM+DOW", **linear_dow_results},
        {"Model": "RBF SVM+DOW", **rbf_dow_results},
        {"Model": "Poly SVM+DOW", **poly_dow_results},
    ])
    ranked_df = rank_models_by_metrics(ranking_df, criteria=TEST_SELECTION_CRITERIA)
    best_model_name = str(ranked_df.iloc[0]["Model"])

    plot_candidates = {
        "Raw Linear SVM": {
            "search": grid_search_linear,
            "param_specs": [("classifier__C", "C")],
            "X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test,
        },
        "Raw RBF SVM": {
            "search": grid_search_rbf,
            "param_specs": [("classifier__C", "C"), ("classifier__gamma", "gamma")],
            "X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test,
        },
        "Raw Poly SVM": {
            "search": grid_search_poly,
            "param_specs": [("classifier__degree", "degree"), ("classifier__C", "C"), ("classifier__gamma", "gamma")],
            "X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test,
        },
        "Linear SVM+DOW": {
            "search": grid_search_linear_dow,
            "param_specs": [("classifier__C", "C")],
            "X_train": X_train_dow, "y_train": y_train_dow, "X_test": X_test_dow, "y_test": y_test_dow,
        },
        "RBF SVM+DOW": {
            "search": grid_search_rbf_dow,
            "param_specs": [("classifier__C", "C"), ("classifier__gamma", "gamma")],
            "X_train": X_train_dow, "y_train": y_train_dow, "X_test": X_test_dow, "y_test": y_test_dow,
        },
        "Poly SVM+DOW": {
            "search": grid_search_poly_dow,
            "param_specs": [("classifier__degree", "degree"), ("classifier__C", "C"), ("classifier__gamma", "gamma")],
            "X_train": X_train_dow, "y_train": y_train_dow, "X_test": X_test_dow, "y_test": y_test_dow,
        },
    }

    plot_model_name = select_non_degenerate_plot_model(ranked_df, available_models=plot_candidates)
    best_candidate = plot_candidates[plot_model_name]

    output_dir = cwd / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_best_model_plots_from_gridsearch_all_params(
        best_candidate["search"],
        best_candidate["param_specs"],
        plot_model_name,
        output_dir / f"{output_prefix}_base_svm_best_bias_variance.png",
        output_dir / f"{output_prefix}_base_svm_best_train_test.png",
        best_candidate["X_train"],
        best_candidate["y_train"],
        best_candidate["X_test"],
        best_candidate["y_test"],
    )
    print(f"\nBest raw/DOW SVM model by average rank: {best_model_name}")
    if plot_model_name != best_model_name:
        print(f"Plotting fallback (non-degenerate): {plot_model_name}")

    # ===================================================================
    # COMPARISON TABLE — same output filenames as base_SVM.py (overwrite)
    # ===================================================================
    comparison_df = build_base_style_comparison_df([
        comparison_row_from_metrics("Raw Linear SVM", linear_results),
        comparison_row_from_metrics("Raw RBF SVM", rbf_results),
        comparison_row_from_metrics("Raw Poly SVM", poly_results),
        comparison_row_from_metrics("Linear SVM+DOW", linear_dow_results),
        comparison_row_from_metrics("RBF SVM+DOW", rbf_dow_results),
        comparison_row_from_metrics("Poly SVM+DOW", poly_dow_results),
    ])
    comparison_export_df = comparison_df.copy()
    print("\n===== Raw + DOW SVM Comparison Table =====")
    print(comparison_df.to_string())

    comparison_csv = cwd / "output" / f"{output_prefix}_base_svm_comparison.csv"
    comparison_tex = cwd / "output" / f"{output_prefix}_base_svm_comparison.tex"
    comparison_export_df.to_csv(comparison_csv, float_format='%.3f')
    write_base_style_latex_table(
        comparison_export_df,
        comparison_tex,
        'Raw SVM Model Comparison (Raw OHLCV vs +Day-of-Week)',
        'tab:base_svm_comparison',
        'Test Acc = plain hold-out accuracy on the final 20% test split. All reported CV/train/test accuracy columns in this table use plain accuracy after hyperparameters were selected by CV balanced accuracy. Recall = positive-class sensitivity, TP / (TP + FN). Specificity = TN / (TN + FP).'
    )
    print(f"Local ranked/exported winner in base_SVM_withDOW.py: {best_model_name}")
    print(f"Local plot winner in base_SVM_withDOW.py: {plot_model_name}")

    global_ranked_df = register_global_model_candidates(
        ranked_df,
        cwd / "output" / f"{output_prefix}_global_model_leaderboard.csv",
        source_script="base_SVM.py",
        dataset_label=output_prefix,
        comparison_scope="tuned_candidates",
    )
    print(f"Current global best model across registered scripts (informational only): {global_ranked_df.iloc[0]['Model']}")

    append_search_run(
        runs_path=runs_path,
        model_name="base_SVM_withDOW",
        run_time=run_time,
        run_duration_sec=(time.time() - run_start),
        grid_version=grid_label,
        n_jobs=MODEL_N_JOBS,
        dataset_version=dataset_version,
        code_commit=get_git_commit(cwd),
        notes=SEARCH_NOTES,
    )
