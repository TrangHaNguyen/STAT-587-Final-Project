from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.svm import SVC
from sklearn.base import clone
from typing import Any, cast
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import os
import time
import pandas as pd
import numpy as np

MPLCONFIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.mplconfig')
os.makedirs(MPLCONFIGDIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(MPLCONFIGDIR))

from H_prep import clean_data, import_data, data_clean_param_selection, to_binary_class
from H_modeling import fit_or_load_search, load_input_data, make_one_se_refit
from H_eval import (
    TEST_SELECTION_CRITERIA,
    get_final_metrics,
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
    append_search_run,
    get_checkpoint_dir,
    get_git_commit,
    now_iso,
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
USE_SAMPLE_PARQUET = os.getenv("USE_SAMPLE_PARQUET", "0") == "1"
GRIDSEARCH_VERBOSE = int(os.getenv("GRIDSEARCH_VERBOSE", "0"))
# RollingWindowBacktest controls kept here as comments for possible later reuse.
# RUN_BACKTEST = os.getenv("RUN_BACKTEST", "0") == "1"
# BACKTEST_VERBOSE = int(os.getenv("BACKTEST_VERBOSE", "0"))
# SHOW_BACKTEST_PLOT = os.getenv("SHOW_BACKTEST_PLOT", "0") == "1"
SAMPLE_PARQUET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'Data', 'sample.parquet'
)

FINAL_METHOD_GROUPS = {
    "Logistic Regression": ["base.py", "logistic_regression.py"],
    "Random Forest": ["base_random_forest.py", "random_forest.py"],
    "SVM": ["base_SVM.py", "SVM.py"],
    "Neural Network": ["base_NN.py", "NN.py"],
}


def _effective_svm_gamma(value, n_features: int) -> float:
    n_features = max(1, int(n_features))
    if value in {"scale", "auto"}:
        return 1.0 / float(n_features)
    try:
        return float(value)
    except Exception:
        return float("inf")


def write_final_method_comparison_from_leaderboard(output_dir, output_prefix: str) -> None:
    leaderboard_path = output_dir / f"{output_prefix}_global_model_leaderboard.csv"
    if not leaderboard_path.exists():
        print(f"Skipping final methods table: missing leaderboard at {leaderboard_path}")
        return

    leaderboard_df = pd.read_csv(leaderboard_path)
    finalists_df = leaderboard_df.loc[
        (leaderboard_df["dataset_label"] == output_prefix)
        & (leaderboard_df["comparison_scope"] == "tuned_candidates")
        & (leaderboard_df["source_script"].isin({
            script_name
            for script_names in FINAL_METHOD_GROUPS.values()
            for script_name in script_names
        }))
    ].copy()
    if finalists_df.empty:
        print("Skipping final methods table: no tuned-candidate rows found for the target scripts.")
        return

    finalist_rows = []
    for method_label, source_scripts in FINAL_METHOD_GROUPS.items():
        script_df = finalists_df.loc[finalists_df["source_script"].isin(source_scripts)].copy()
        if script_df.empty:
            print(f"Skipping {method_label} in final methods table: no registered candidates found.")
            continue
        ranked_script_df = rank_models_by_metrics(script_df, criteria=TEST_SELECTION_CRITERIA)
        winner = ranked_script_df.iloc[0]
        finalist_rows.append(
            comparison_row_from_metrics(
                f"{method_label}: {winner['candidate_model']}",
                winner,
            )
        )

    if not finalist_rows:
        print("Skipping final methods table: no finalists were available after per-script filtering.")
        return

    final_comparison_df = build_base_style_comparison_df(finalist_rows)
    final_comparison_csv = output_dir / f"{output_prefix}_4methods_comparison.csv"
    final_comparison_tex = output_dir / f"{output_prefix}_4methods_comparison.tex"
    final_comparison_df.to_csv(final_comparison_csv, float_format='%.3f')
    write_base_style_latex_table(
        final_comparison_df,
        final_comparison_tex,
        f'{output_prefix} 4-method comparison',
        f'tab:{output_prefix}_4methods_comparison',
        'Each row is the single winner for one method family across its feature-set variants. For Logistic Regression, Random Forest, and SVM, winners are selected using training-only time-series CV metrics. For Neural Network, winners are selected using hold-out test metrics (no CV tuning). All rows report final hold-out test metrics.'
    )
    print("\n===== Final Cross-Family Comparison (CV-selected finalists, test-reported) =====")
    print(final_comparison_df.to_string())


def load_svm_input_data():
    return load_input_data(
        use_sample_parquet=USE_SAMPLE_PARQUET,
        sample_parquet_path=SAMPLE_PARQUET_PATH,
        import_data_fn=import_data,
        import_data_kwargs={
            "extra_features": True,
            "testing": False,
            "cluster": False,
            "n_clusters": 100,
            "corr_threshold": 0.95,
            "corr_level": 0,
        },
    )


if __name__ == "__main__":
    run_start = time.time()
    run_time = now_iso()
    WINDOW_SIZE=200
    HORIZON=40
    # testing: bool =False, extra_features: bool =True, cluster: bool =False, n_clusters: int =100, corr_threshold: float =0.95, corr_level: int =0
    DATA=load_svm_input_data()
    # Keep feature-engineering configuration fixed for consistency across models.
    FIND_OPTIMAL=False
    print(f"MODEL_N_JOBS={MODEL_N_JOBS} (set env MODEL_N_JOBS to override)")
    print(f"GRID_VERSION={GRID_VERSION}")
    grid_label = GRID_VERSION
    output_prefix = "sample" if USE_SAMPLE_PARQUET else "8yrs"
    history_path = cwd / "output" / f"{output_prefix}_search_history_svm.csv"
    runs_path = cwd / "output" / f"{output_prefix}_search_runs.csv"
    checkpoint_dir = get_checkpoint_dir(cwd / "output", "SVM", f"{output_prefix}_{grid_label}")
    dataset_version = (
        "sample_parquet=PyScripts/Data/sample.parquet,extra_features=True,cluster=False,corr_threshold=0.95,corr_level=0"
        if USE_SAMPLE_PARQUET
        else "testing=False,extra_features=True,cluster=False,corr_threshold=0.95,corr_level=0"
    )
    
    parameters_={
        "raw": False,
        "extra_features": True,
        "lag_period": [1, 2, 3, 4, 5, 6, 7],
        "lookback_period": 30,
        "sector": False,
        "corr_threshold": 0.95,
        "corr_level": 0,
    }

    if (FIND_OPTIMAL):
        # ------- Selection of Remaining data_clean() Parameters -------
        base_SVM_rbf_model=SVC(kernel="rbf", cache_size=1000, class_weight=SVM_CLASS_WEIGHT, gamma='scale', random_state=RANDOM_SEED, tol=SVM_TOL)
        base_SVM_rbf_model_pipeline=Pipeline([('scaler', StandardScaler()), ('classifier', base_SVM_rbf_model)])

        # ------- Selection of Optimal data_clean() Parameters -------
        print("------- Finding Optimal data_clean() Parameters")
        param_grid={
            'raw': [True, False],
            'extra_features': [True, False],
            'lag_period': [[1, 2, 3, 4, 5, 6, 7]],
            'lookback_period': [30],
            'sector': [False],
            'corr_level': [0],
        }

        _, parameters_, best_score=data_clean_param_selection(*DATA, clone(base_SVM_rbf_model_pipeline), TEST_SIZE, WINDOW_SIZE, HORIZON, **param_grid)
        print(f"Best Utility Score {best_score}")
        print(f"Optimal parameter {parameters_}")

    X, y_regression=cast(Any, clean_data(*DATA, **parameters_))
    y_classification=to_binary_class(y_regression)
    X_train, X_test, y_train, y_test=train_test_split(X, y_classification, test_size=TEST_SIZE, random_state=RANDOM_SEED, shuffle=TRAIN_TEST_SHUFFLE)
    gamma_sort = lambda value: _effective_svm_gamma(value, X_train.shape[1])

    # This script does not define a PCA-based SVM branch, so there is no
    # shared PCA object to reuse from logistic_regression.py here.
    # Previous temporary change used `KFold(n_splits=5, shuffle=False)`.
    tscv = TimeSeriesSplit(n_splits=TIME_SERIES_CV_SPLITS)
    # ------- Linear SVM -------
    print("\n\n------- Linear SVM Model -------")
    SVM_linear=SVC(kernel="linear", cache_size=1000, class_weight=SVM_CLASS_WEIGHT, gamma='scale', random_state=RANDOM_SEED, tol=SVM_TOL)

    SVM_linear_pipeline = Pipeline([('scaler', StandardScaler()),
                                    ('classifier', SVM_linear)])

    param_grid={
        'classifier__C': SVM_LINEAR_C_GRID_OPTIONS
    }
    
    grid_search_linear = GridSearchCV(
        SVM_linear_pipeline, param_grid, cv=tscv, scoring='balanced_accuracy',
        n_jobs=MODEL_N_JOBS, verbose=GRIDSEARCH_VERBOSE, return_train_score=True,
        refit=make_one_se_refit(['classifier__C'], n_splits=TIME_SERIES_CV_SPLITS)
    )
    grid_search_linear = fit_or_load_search(
        checkpoint_dir=checkpoint_dir,
        stage_name="svm_linear",
        search_obj=grid_search_linear,
        X_train=X_train,
        y_train=y_train,
        history_path=history_path,
        run_time=run_time,
        model_name="SVM_linear",
        grid_label=grid_label,
        notes=SEARCH_NOTES,
    )

    optimized_linear_ = grid_search_linear.best_estimator_

    results=get_or_compute_final_metrics(checkpoint_dir, _metrics_stage_name("Linear Ker. SVM"), optimized_linear_, X_train, y_train, X_test, y_test, label="Linear Ker. SVM")
    linear_results = results.copy()
    # RollingWindowBacktest disabled to save runtime. Restore this block if needed later.
    # rwb_obj=RollingWindowBacktest(clone(grid_search_linear.best_estimator_), X, y_classification, X_train, WINDOW_SIZE, HORIZON)
    # rwb_obj.rolling_window_backtest(verbose=BACKTEST_VERBOSE)
    # if SHOW_BACKTEST_PLOT:
    #     rwb_obj.display_wfv_results()
    # util_score=utility_score(results, rwb_obj)
    # print(f"Utility Score {util_score:.4}")
    # ------- RBF SVM -------
    print("\n\n------- RBF SVM Model -------")
    SVM_rbf=SVC(kernel="rbf", cache_size=1000, class_weight=SVM_CLASS_WEIGHT, gamma='scale', random_state=RANDOM_SEED, tol=SVM_TOL)

    SVM_rbf_pipeline = Pipeline([('scaler', StandardScaler()),
                                 ('classifier', SVM_rbf)])

    param_grid={
        'classifier__C': SVM_LINEAR_C_GRID_OPTIONS,
        'classifier__gamma': SVM_GAMMA_GRID_OPTIONS
    }
    
    grid_search_rbf = GridSearchCV(
        SVM_rbf_pipeline, param_grid, cv=tscv, scoring='balanced_accuracy',
        n_jobs=MODEL_N_JOBS, verbose=GRIDSEARCH_VERBOSE, return_train_score=True,
        refit=make_one_se_refit(
            ['classifier__C', 'classifier__gamma'],
            n_splits=TIME_SERIES_CV_SPLITS,
            sort_value_map={'classifier__gamma': gamma_sort},
        )
    )
    grid_search_rbf = fit_or_load_search(
        checkpoint_dir=checkpoint_dir,
        stage_name="svm_rbf",
        search_obj=grid_search_rbf,
        X_train=X_train,
        y_train=y_train,
        history_path=history_path,
        run_time=run_time,
        model_name="SVM_rbf",
        grid_label=grid_label,
        notes=SEARCH_NOTES,
    )

    optimized_rbf_ = grid_search_rbf.best_estimator_

    results=get_or_compute_final_metrics(checkpoint_dir, _metrics_stage_name("RBF Ker. SVM"), optimized_rbf_, X_train, y_train, X_test, y_test, label="RBF Ker. SVM")
    rbf_results = results.copy()
    # RollingWindowBacktest disabled to save runtime. Restore this block if needed later.
    # rwb_obj=RollingWindowBacktest(clone(grid_search_rbf.best_estimator_), X, y_classification, X_train, WINDOW_SIZE, HORIZON)
    # rwb_obj.rolling_window_backtest(verbose=BACKTEST_VERBOSE)
    # if SHOW_BACKTEST_PLOT:
    #     rwb_obj.display_wfv_results()
    # util_score=utility_score(results, rwb_obj)
    # print(f"Utility Score {util_score:.4}")
    # ------- Polynomial SVM -------
    print("\n\n------- Polynomial SVM Model -------")
    SVM_poly=SVC(kernel="poly", cache_size=1000, class_weight=SVM_CLASS_WEIGHT, gamma='scale', random_state=RANDOM_SEED, tol=SVM_TOL)

    SVM_poly_pipeline = Pipeline([('scaler', StandardScaler()),
                                  ('classifier', SVM_poly)])

    param_grid={
        'classifier__C': SVM_LINEAR_C_GRID_OPTIONS,
        'classifier__gamma': SVM_GAMMA_GRID_OPTIONS,
        'classifier__degree': SVM_DEGREE_GRID_OPTIONS
    }
    
    grid_search_poly = GridSearchCV(
        SVM_poly_pipeline, param_grid, cv=tscv, scoring='balanced_accuracy',
        n_jobs=MODEL_N_JOBS, verbose=GRIDSEARCH_VERBOSE, return_train_score=True,
        refit=make_one_se_refit(
            ['classifier__C', 'classifier__degree', 'classifier__gamma'],
            n_splits=TIME_SERIES_CV_SPLITS,
            sort_value_map={'classifier__gamma': gamma_sort},
        )
    )
    grid_search_poly = fit_or_load_search(
        checkpoint_dir=checkpoint_dir,
        stage_name="svm_poly",
        search_obj=grid_search_poly,
        X_train=X_train,
        y_train=y_train,
        history_path=history_path,
        run_time=run_time,
        model_name="SVM_poly",
        grid_label=grid_label,
        notes=SEARCH_NOTES,
    )

    optimized_poly_ = grid_search_poly.best_estimator_

    results=get_or_compute_final_metrics(checkpoint_dir, _metrics_stage_name("Poly. Ker. SVM"), optimized_poly_, X_train, y_train, X_test, y_test, label="Poly. Ker. SVM")
    poly_results = results.copy()
    # RollingWindowBacktest disabled to save runtime. Restore this block if needed later.
    # rwb_obj=RollingWindowBacktest(clone(grid_search_poly.best_estimator_), X, y_classification, X_train, WINDOW_SIZE, HORIZON)
    # rwb_obj.rolling_window_backtest(verbose=BACKTEST_VERBOSE)
    # if SHOW_BACKTEST_PLOT:
    #     rwb_obj.display_wfv_results()
    # util_score=utility_score(results, rwb_obj)
    # print(f"Utility Score {util_score:.4}")
    ranking_df = pd.DataFrame([
        {"Model": "Linear SVM", **linear_results},
        {"Model": "RBF SVM", **rbf_results},
        {"Model": "Poly SVM", **poly_results},
    ])
    ranked_df = rank_models_by_metrics(ranking_df, criteria=TEST_SELECTION_CRITERIA)
    best_model_name = str(ranked_df.iloc[0]["Model"])
    plot_model_name = select_non_degenerate_plot_model(ranked_df)
    best_plot_config = {
        "Linear SVM": (grid_search_linear, [("classifier__C", "C")]),
        "RBF SVM": (grid_search_rbf, [("classifier__C", "C"), ("classifier__gamma", "gamma")]),
        "Poly SVM": (grid_search_poly, [("classifier__degree", "degree"), ("classifier__C", "C"), ("classifier__gamma", "gamma")]),
    }[plot_model_name]
    output_dir = cwd / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_best_model_plots_from_gridsearch_all_params(
        best_plot_config[0],
        best_plot_config[1],
        plot_model_name,
        output_dir / f"{output_prefix}_svm_best_bias_variance.png",
        output_dir / f"{output_prefix}_svm_best_train_test.png",
        X_train,
        y_train,
        X_test,
        y_test,
    )
    print(f"\nBest SVM model by average rank: {best_model_name}")
    if plot_model_name != best_model_name:
        print(f"Plotting fallback (non-degenerate): {plot_model_name}")
    comparison_df = build_base_style_comparison_df([
        comparison_row_from_metrics("Linear SVM", linear_results),
        comparison_row_from_metrics("RBF SVM", rbf_results),
        comparison_row_from_metrics("Poly SVM", poly_results),
    ])
    print("\n===== SVM Comparison Table =====")
    print(comparison_df.to_string())
    comparison_csv = cwd / "output" / f"{output_prefix}_svm_comparison.csv"
    comparison_tex = cwd / "output" / f"{output_prefix}_svm_comparison.tex"
    comparison_df.to_csv(comparison_csv, float_format='%.3f')
    write_base_style_latex_table(
        comparison_df,
        comparison_tex,
        'SVM Model Comparison',
        'tab:svm_comparison',
        'Test Acc = plain hold-out accuracy on the final 20% test split. All reported CV/train/test accuracy columns in this table use plain accuracy after hyperparameters were selected by CV balanced accuracy. Recall = positive-class sensitivity, TP / (TP + FN). Specificity = TN / (TN + FP).'
    )
    print(f"Local ranked/exported winner in SVM.py: {best_model_name}")
    print(f"Local plot winner in SVM.py: {plot_model_name}")
    global_ranked_df = register_global_model_candidates(
        ranked_df,
        cwd / "output" / f"{output_prefix}_global_model_leaderboard.csv",
        source_script="SVM.py",
        dataset_label=output_prefix,
        comparison_scope="tuned_candidates",
    )
    write_final_method_comparison_from_leaderboard(output_dir, output_prefix)
    print(f"Current global best model across registered scripts (informational only): {global_ranked_df.iloc[0]['Model']}")

    append_search_run(
        runs_path=runs_path,
        model_name="SVM",
        run_time=run_time,
        run_duration_sec=(time.time() - run_start),
        grid_version=grid_label,
        n_jobs=MODEL_N_JOBS,
        dataset_version=dataset_version,
        code_commit=get_git_commit(cwd),
        notes=SEARCH_NOTES
    )
