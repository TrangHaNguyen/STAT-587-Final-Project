from typing import Any, cast
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import clone
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
import os
import warnings
import pandas as pd
import numpy as np
import time

MPLCONFIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.mplconfig')
os.makedirs(MPLCONFIGDIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(MPLCONFIGDIR))

import matplotlib
matplotlib.use('Agg')
# import matplotlib.pyplot as plt  # Unused in the active code path.

from H_prep import clean_data, data_clean_param_selection, import_data
from H_modeling import (
    fit_or_load_baseline_logistic_pca_search,
    fit_or_load_fixed_classifier_pca_search,
    fit_or_load_search,
    load_input_data,
    make_one_se_refit,
    transform_with_fitted_scaler_pca,
)
from H_eval import (
    get_final_metrics,
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
    BASELINE_PCA_GRID,
    LOGISTIC_BASELINE_SOLVER,
    LOGISTIC_LASSO_SOLVER,
    LOGISTIC_MAX_ITER,
    LOGISTIC_RIDGE_SOLVER,
    LASSO_GRID,
    LOGISTIC_TOL,
    RIDGE_GRID,
    TEST_SIZE,
    TIME_SERIES_CV_SPLITS,
    TRAIN_TEST_SHUFFLE,
)

VERBOSE = 0
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

cwd = get_cwd("STAT-587-Final-Project")

def _build_logistic_kwargs(*, solver: str, l1_ratio=None, c_value=None, verbose: int | None = None):
    kwargs = {
        'solver': solver,
        'class_weight': 'balanced',
        'random_state': 1,
        'max_iter': LOGISTIC_MAX_ITER,
        'tol': LOGISTIC_TOL,
    }
    if c_value is not None:
        kwargs['C'] = c_value
    if l1_ratio is not None:
        kwargs['l1_ratio'] = l1_ratio
    if verbose is not None:
        kwargs['verbose'] = verbose
    return kwargs


def _fit_logistic_search(search_obj, X_train, y_train):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Setting penalty=None will ignore the C and l1_ratio parameters",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*'penalty' was deprecated.*use 'l1_ratio' or 'C' instead.*",
            category=FutureWarning,
        )
        search_obj.fit(X_train, y_train)


def load_logreg_input_data():
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

if __name__=="__main__":
    run_start = time.time()
    run_time = now_iso()
    WINDOW_SIZE=200
    HORIZON=40
    # Previous temporary change used `KFold(n_splits=5, shuffle=False)`.
    tscv = TimeSeriesSplit(n_splits=TIME_SERIES_CV_SPLITS)
    print(f"MODEL_N_JOBS={MODEL_N_JOBS} (set env MODEL_N_JOBS to override)")
    print(f"GRID_VERSION={GRID_VERSION}")
    grid_label = GRID_VERSION
    output_prefix = "sample" if USE_SAMPLE_PARQUET else "8yrs"
    history_path = cwd / "output" / f"{output_prefix}_search_history_logreg.csv"
    runs_path = cwd / "output" / f"{output_prefix}_search_runs.csv"
    checkpoint_dir = get_checkpoint_dir(cwd / "output", "logistic_regression", f"{output_prefix}_{grid_label}")
    dataset_version = (
        "sample_parquet=PyScripts/Data/sample.parquet,extra_features=True,cluster=False,corr_threshold=0.95,corr_level=0"
        if USE_SAMPLE_PARQUET
        else "testing=False,extra_features=True,cluster=False,corr_threshold=0.95,corr_level=0"
    )
    # testing: bool =False, extra_features: bool =True, cluster: bool =False, n_clusters: int =100, corr_threshold: float =0.95, corr_level: int =0
    DATA=load_logreg_input_data()

    FIND_OPTIMAL=False
    
    parameters_={  # These are optimal as of 3/8/2026 4:00 PM w=4
        "raw": False,
        "extra_features": True,
        "lag_period": [1, 2, 3, 4, 5, 6, 7],
        "lookback_period": 30,
        "sector": False,
        "corr_threshold": 0.95,
        "corr_level": 0
    }

    if (FIND_OPTIMAL):
        # ------- Selection of Remaining data_clean() Parameters -------
        base_Log_Reg_model=LogisticRegression(**_build_logistic_kwargs(
            solver=LOGISTIC_RIDGE_SOLVER, l1_ratio=0, c_value=1.0, verbose=VERBOSE
        ))
        base_Log_Reg_model_pipeline=Pipeline([('scaler', StandardScaler()), ('classifier', base_Log_Reg_model)])

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

        _, parameters_, best_score=data_clean_param_selection(*DATA, clone(base_Log_Reg_model_pipeline), TEST_SIZE, WINDOW_SIZE, HORIZON, **param_grid)
        print(f"Best Utility Score {best_score}")
        print(f"Optimal parameter {parameters_}")

    X, y_regression=cast(Any, clean_data(*DATA, **parameters_))
    def to_binary_class(y):
        return (y>=0).astype(int)
    y_classification=to_binary_class(y_regression)
    X_train, X_test, y_train, y_test=train_test_split(X, y_classification, test_size=TEST_SIZE, random_state=1, shuffle=TRAIN_TEST_SHUFFLE)

    # ------- PCA Base (No Logistic Regularization) -------
    print("\n\n------- PCA Base Logistic Model -------")
    grid_search_PCA_base = fit_or_load_baseline_logistic_pca_search(
        checkpoint_dir=checkpoint_dir,
        stage_name="logreg_pca_base",
        X_train=X_train,
        y_train=y_train,
        history_path=history_path,
        run_time=run_time,
        model_name="LogReg_PCA_Base",
        grid_label=grid_label,
        n_splits=TIME_SERIES_CV_SPLITS,
        notes=SEARCH_NOTES,
    )
    optimized_Log_Reg_PCA_base_ = grid_search_PCA_base.best_estimator_
    X_train_pca_base, X_test_pca_base, pca_base_scaler, pca_base_reducer = transform_with_fitted_scaler_pca(
        grid_search_PCA_base,
        X_train,
        X_test,
    )
    results=get_final_metrics(optimized_Log_Reg_PCA_base_, X_train, y_train, X_test, y_test, n_splits=tscv.n_splits, label="PCA Base")
    pca_base_results = results.copy()
    # RollingWindowBacktest disabled to save runtime. Restore this block if needed later.
    # rwb_obj=RollingWindowBacktest(clone(grid_search_PCA_base.best_estimator_), X, y_classification, X_train, WINDOW_SIZE, HORIZON)
    # rwb_obj.rolling_window_backtest(verbose=BACKTEST_VERBOSE)
    # if SHOW_BACKTEST_PLOT:
    #     rwb_obj.display_wfv_results()
    # util_score=utility_score(results, rwb_obj)
    # print(f"Utility Score {util_score:.4}")
    # ------- Ridge (No PCA) -------
    print("\n\n------- Logistic Ridge Model -------")
    Log_Reg_Ridge = LogisticRegression(**_build_logistic_kwargs(
        solver=LOGISTIC_RIDGE_SOLVER, l1_ratio=0
    ))
    Log_Reg_model_pipeline_Ridge = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', Log_Reg_Ridge)
    ])
    param_grid = {
        'classifier__C': RIDGE_GRID
    }
    grid_search_ridge = GridSearchCV(
        Log_Reg_model_pipeline_Ridge, param_grid, cv=tscv, return_train_score=True, verbose=GRIDSEARCH_VERBOSE,
        scoring='balanced_accuracy',
        refit=make_one_se_refit(['classifier__C'], n_splits=TIME_SERIES_CV_SPLITS)
    )
    grid_search_ridge = fit_or_load_search(
        checkpoint_dir=checkpoint_dir,
        stage_name="logreg_ridge",
        search_obj=grid_search_ridge,
        X_train=X_train,
        y_train=y_train,
        history_path=history_path,
        run_time=run_time,
        model_name="LogReg_Ridge",
        grid_label=grid_label,
        notes=SEARCH_NOTES,
        fit_search=_fit_logistic_search,
    )
    optimized_Log_Reg_ridge_ = grid_search_ridge.best_estimator_
    results=get_final_metrics(optimized_Log_Reg_ridge_, X_train, y_train, X_test, y_test, n_splits=tscv.n_splits, label="Ridge Log. Reg.")
    ridge_results = results.copy()
    # RollingWindowBacktest disabled to save runtime. Restore this block if needed later.
    # rwb_obj=RollingWindowBacktest(clone(grid_search_ridge.best_estimator_), X, y_classification, X_train, WINDOW_SIZE, HORIZON)
    # rwb_obj.rolling_window_backtest(verbose=BACKTEST_VERBOSE)
    # if SHOW_BACKTEST_PLOT:
    #     rwb_obj.display_wfv_results()
    # util_score=utility_score(results, rwb_obj)
    # print(f"Utility Score {util_score:.4}")
    # ------- LASSO (No PCA) -------
    print("\n\n------- Logistic LASSO Model -------")
    Log_Reg_Lasso = LogisticRegression(**_build_logistic_kwargs(
        solver=LOGISTIC_LASSO_SOLVER, l1_ratio=1
    ))
    Log_Reg_model_pipeline_Lasso = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', Log_Reg_Lasso)
    ])
    param_grid = {
        'classifier__C': LASSO_GRID
    }
    grid_search_lasso = GridSearchCV(
        Log_Reg_model_pipeline_Lasso, param_grid, cv=tscv, return_train_score=True, verbose=GRIDSEARCH_VERBOSE,
        scoring='balanced_accuracy',
        refit=make_one_se_refit(['classifier__C'], n_splits=TIME_SERIES_CV_SPLITS)
    )
    grid_search_lasso = fit_or_load_search(
        checkpoint_dir=checkpoint_dir,
        stage_name="logreg_lasso",
        search_obj=grid_search_lasso,
        X_train=X_train,
        y_train=y_train,
        history_path=history_path,
        run_time=run_time,
        model_name="LogReg_LASSO",
        grid_label=grid_label,
        notes=SEARCH_NOTES,
        fit_search=_fit_logistic_search,
    )
    optimized_Log_Reg_lasso_ = grid_search_lasso.best_estimator_
    results=get_final_metrics(optimized_Log_Reg_lasso_, X_train, y_train, X_test, y_test, n_splits=tscv.n_splits, label="LASSO Log. Reg.")
    lasso_results = results.copy()
    # RollingWindowBacktest disabled to save runtime. Restore this block if needed later.
    # rwb_obj=RollingWindowBacktest(clone(grid_search_lasso.best_estimator_), X, y_classification, X_train, WINDOW_SIZE, HORIZON)
    # rwb_obj.rolling_window_backtest(verbose=BACKTEST_VERBOSE)
    # if SHOW_BACKTEST_PLOT:
    #     rwb_obj.display_wfv_results()
    # util_score=utility_score(results, rwb_obj)
    # print(f"Utility Score {util_score:.4}")
    # ------- Elastic Net (No PCA) -------
    # print("\n\n------- Logistic Elastic Net Model -------")
    # Log_Reg_Elastic = LogisticRegression(...)
    # Log_Reg_model_pipeline_Elastic = Pipeline([
    #     ('scaler', StandardScaler()),
    #     ('classifier', Log_Reg_Elastic)
    # ])
    # param_grid = {
    #     'classifier__C': ...,
    #     'classifier__l1_ratio': ...,
    # }
    # grid_search_elastic = GridSearchCV(
    #     Log_Reg_model_pipeline_Elastic, param_grid, cv=tscv, return_train_score=True, verbose=VERBOSE,
    #     scoring='balanced_accuracy',
    #     refit=make_one_se_refit(['classifier__C', 'classifier__l1_ratio'])
    # )
    # grid_search_elastic.fit(X_train, y_train)
    # append_search_history(
    #     history_path=history_path,
    #     cv_results=grid_search_elastic.cv_results_,
    #     run_time=run_time,
    #     model_name="LogReg_ElasticNet",
    #     search_type="grid",
    #     grid_version=grid_label,
    #     notes=SEARCH_NOTES,
    #     best_params=grid_search_elastic.best_params_
    # )
    # rwb_obj=RollingWindowBacktest(clone(grid_search_elastic.best_estimator_), X, y_classification, X_train, WINDOW_SIZE, HORIZON)
    # rwb_obj.rolling_window_backtest(verbose=1)
    # rwb_obj.display_wfv_results()
    # optimized_Log_Reg_elastic_ = grid_search_elastic.best_estimator_
    # results=get_final_metrics(optimized_Log_Reg_elastic_, X_train, y_train, X_test, y_test, n_splits=10, label="Elastic Net Log. Reg.")
    # elastic_results = results.copy()
    # util_score=utility_score(results, rwb_obj)
    # print(f"Utility Score {util_score:.4}")
    # ------- PCA to Ridge(Internal) APPLICATION -------
    Log_Reg_PCA_ridge = LogisticRegression(**_build_logistic_kwargs(
        solver=LOGISTIC_RIDGE_SOLVER, l1_ratio=0
    ))

    param_grid={
        'C': RIDGE_GRID
    }
    grid_search_PCA_ridge=GridSearchCV(
        Log_Reg_PCA_ridge, param_grid, cv=tscv, return_train_score=True, verbose=GRIDSEARCH_VERBOSE,
        scoring='balanced_accuracy',
        refit=make_one_se_refit(['classifier__C'], n_splits=TIME_SERIES_CV_SPLITS)
    )
    grid_search_PCA_ridge = fit_or_load_search(
        checkpoint_dir=checkpoint_dir,
        stage_name="logreg_pca_ridge",
        search_obj=grid_search_PCA_ridge,
        X_train=X_train_pca_base,
        y_train=y_train,
        history_path=history_path,
        run_time=run_time,
        model_name="LogReg_PCA_Ridge",
        grid_label=grid_label,
        notes=SEARCH_NOTES,
        fit_search=_fit_logistic_search,
    )
    fixed_ridge_classifier = LogisticRegression(**_build_logistic_kwargs(
        solver=LOGISTIC_RIDGE_SOLVER,
        l1_ratio=0,
        c_value=grid_search_PCA_ridge.best_params_['classifier__C'] if 'classifier__C' in grid_search_PCA_ridge.best_params_ else grid_search_PCA_ridge.best_params_['C'],
    ))
    pca_ridge_search = fit_or_load_fixed_classifier_pca_search(
        checkpoint_dir=checkpoint_dir,
        stage_name="logreg_pca_ridge_retuned_n_components",
        X_train=X_train,
        y_train=y_train,
        history_path=history_path,
        run_time=run_time,
        model_name="LogReg_PCA_Ridge_retuned_n_components",
        grid_label=grid_label,
        n_splits=TIME_SERIES_CV_SPLITS,
        classifier=fixed_ridge_classifier,
        notes=SEARCH_NOTES,
        fit_search=_fit_logistic_search,
    )
    X_train_pca_ridge, X_test_pca_ridge, _, _ = transform_with_fitted_scaler_pca(
        pca_ridge_search,
        X_train,
        X_test,
    )
    selected_pca_ridge_n_components = pca_ridge_search.best_params_['pca__n_components']
    print(
        "Retuned PCA for PCA Ridge(int.) after Ridge model selection: "
        f"n_components={selected_pca_ridge_n_components} ({X_train_pca_ridge.shape[1]} components)."
    )
    grid_search_PCA_ridge = fit_or_load_search(
        checkpoint_dir=checkpoint_dir,
        stage_name="logreg_pca_ridge_refit",
        search_obj=GridSearchCV(
            Log_Reg_PCA_ridge, {'C': RIDGE_GRID}, cv=tscv, return_train_score=True, verbose=GRIDSEARCH_VERBOSE,
            scoring='balanced_accuracy',
            refit=make_one_se_refit(['C'], n_splits=TIME_SERIES_CV_SPLITS)
        ),
        X_train=X_train_pca_ridge,
        y_train=y_train,
        history_path=history_path,
        run_time=run_time,
        model_name="LogReg_PCA_Ridge_refit",
        grid_label=grid_label,
        notes=SEARCH_NOTES,
        fit_search=_fit_logistic_search,
    )

    optimized_Log_Reg_PCA_ridge_ = grid_search_PCA_ridge.best_estimator_

    results=get_final_metrics(optimized_Log_Reg_PCA_ridge_, X_train_pca_ridge, y_train, X_test_pca_ridge, y_test, n_splits=tscv.n_splits, label="PCA Ridge(int.) Log. Reg.")
    pca_ridge_results = results.copy()
    # RollingWindowBacktest disabled to save runtime. Restore this block if needed later.
    # rwb_obj=RollingWindowBacktest(clone(grid_search_PCA_ridge.best_estimator_), X, y_classification, X_train, WINDOW_SIZE, HORIZON)
    # rwb_obj.rolling_window_backtest(verbose=BACKTEST_VERBOSE)
    # if SHOW_BACKTEST_PLOT:
    #     rwb_obj.display_wfv_results()
    # util_score=utility_score(results, rwb_obj)
    # print(f"Utility Score {util_score:.4}")
    # ------- PCA to LASSO(Internal) APPLICATION -------
    Log_Reg_PCA_lasso = LogisticRegression(**_build_logistic_kwargs(
        solver=LOGISTIC_LASSO_SOLVER, l1_ratio=1
    ))

    param_grid={
        'C': LASSO_GRID
    }
    grid_search_PCA_lasso=GridSearchCV(
        Log_Reg_PCA_lasso, param_grid, cv=tscv, return_train_score=True, verbose=GRIDSEARCH_VERBOSE,
        scoring='balanced_accuracy',
        refit=make_one_se_refit(['classifier__C'], n_splits=TIME_SERIES_CV_SPLITS)
    )
    grid_search_PCA_lasso = fit_or_load_search(
        checkpoint_dir=checkpoint_dir,
        stage_name="logreg_pca_lasso",
        search_obj=grid_search_PCA_lasso,
        X_train=X_train_pca_base,
        y_train=y_train,
        history_path=history_path,
        run_time=run_time,
        model_name="LogReg_PCA_LASSO",
        grid_label=grid_label,
        notes=SEARCH_NOTES,
        fit_search=_fit_logistic_search,
    )
    fixed_lasso_classifier = LogisticRegression(**_build_logistic_kwargs(
        solver=LOGISTIC_LASSO_SOLVER,
        l1_ratio=1,
        c_value=grid_search_PCA_lasso.best_params_['classifier__C'] if 'classifier__C' in grid_search_PCA_lasso.best_params_ else grid_search_PCA_lasso.best_params_['C'],
    ))
    pca_lasso_search = fit_or_load_fixed_classifier_pca_search(
        checkpoint_dir=checkpoint_dir,
        stage_name="logreg_pca_lasso_retuned_n_components",
        X_train=X_train,
        y_train=y_train,
        history_path=history_path,
        run_time=run_time,
        model_name="LogReg_PCA_LASSO_retuned_n_components",
        grid_label=grid_label,
        n_splits=TIME_SERIES_CV_SPLITS,
        classifier=fixed_lasso_classifier,
        notes=SEARCH_NOTES,
        fit_search=_fit_logistic_search,
    )
    X_train_pca_lasso, X_test_pca_lasso, _, _ = transform_with_fitted_scaler_pca(
        pca_lasso_search,
        X_train,
        X_test,
    )
    selected_pca_lasso_n_components = pca_lasso_search.best_params_['pca__n_components']
    print(
        "Retuned PCA for PCA LASSO(int.) after LASSO model selection: "
        f"n_components={selected_pca_lasso_n_components} ({X_train_pca_lasso.shape[1]} components)."
    )
    grid_search_PCA_lasso = fit_or_load_search(
        checkpoint_dir=checkpoint_dir,
        stage_name="logreg_pca_lasso_refit",
        search_obj=GridSearchCV(
            Log_Reg_PCA_lasso, {'C': LASSO_GRID}, cv=tscv, return_train_score=True, verbose=GRIDSEARCH_VERBOSE,
            scoring='balanced_accuracy',
            refit=make_one_se_refit(['C'], n_splits=TIME_SERIES_CV_SPLITS)
        ),
        X_train=X_train_pca_lasso,
        y_train=y_train,
        history_path=history_path,
        run_time=run_time,
        model_name="LogReg_PCA_LASSO_refit",
        grid_label=grid_label,
        notes=SEARCH_NOTES,
        fit_search=_fit_logistic_search,
    )

    optimized_Log_Reg_PCA_lasso_ = grid_search_PCA_lasso.best_estimator_

    results=get_final_metrics(optimized_Log_Reg_PCA_lasso_, X_train_pca_lasso, y_train, X_test_pca_lasso, y_test, n_splits=tscv.n_splits, label="PCA LASSO(int.) Log. Reg.")
    pca_lasso_results = results.copy()
    # RollingWindowBacktest disabled to save runtime. Restore this block if needed later.
    # rwb_obj=RollingWindowBacktest(clone(grid_search_PCA_lasso.best_estimator_), X, y_classification, X_train, WINDOW_SIZE, HORIZON)
    # rwb_obj.rolling_window_backtest(verbose=BACKTEST_VERBOSE)
    # if SHOW_BACKTEST_PLOT:
    #     rwb_obj.display_wfv_results()
    # util_score=utility_score(results, rwb_obj)
    # print(f"Utility Score {util_score:.4}")
    ranking_df = pd.DataFrame([
        {"Model": "PCA Base", **pca_base_results},
        {"Model": "Ridge Log. Reg.", **ridge_results},
        {"Model": "LASSO Log. Reg.", **lasso_results},
        # {"Model": "Elastic Net Log. Reg.", **elastic_results},
        {"Model": "PCA Ridge(int.) Log. Reg.", **pca_ridge_results},
        {"Model": "PCA LASSO(int.) Log. Reg.", **pca_lasso_results},
    ])
    ranked_df = rank_models_by_metrics(ranking_df)
    best_model_name = str(ranked_df.iloc[0]["Model"])
    plot_model_name = select_non_degenerate_plot_model(ranked_df)
    output_dir = cwd / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    if plot_model_name == "PCA Base":
        best_plot_config = (grid_search_PCA_base, [("pca__n_components", "PCA n_components")])
    elif plot_model_name == "Ridge Log. Reg.":
        best_plot_config = (grid_search_ridge, [("classifier__C", "C")])
    elif plot_model_name == "LASSO Log. Reg.":
        best_plot_config = (grid_search_lasso, [("classifier__C", "C")])
    # elif plot_model_name == "Elastic Net Log. Reg.":
    #     best_plot_config = (grid_search_elastic, [("classifier__C", "C"), ("classifier__l1_ratio", "l1_ratio")])
    elif plot_model_name == "PCA Ridge(int.) Log. Reg.":
        best_plot_config = (
            grid_search_PCA_ridge,
            [("C", "C")],
        )
    else:
        best_plot_config = (
            grid_search_PCA_lasso,
            [("C", "C")],
        )
    plot_X_train = X_train
    plot_X_test = X_test
    if plot_model_name == "PCA Ridge(int.) Log. Reg.":
        plot_X_train = X_train_pca_ridge
        plot_X_test = X_test_pca_ridge
    elif plot_model_name == "PCA LASSO(int.) Log. Reg.":
        plot_X_train = X_train_pca_lasso
        plot_X_test = X_test_pca_lasso
    save_best_model_plots_from_gridsearch_all_params(
        best_plot_config[0],
        best_plot_config[1],
        plot_model_name,
        output_dir / f"{output_prefix}_logreg_best_bias_variance.png",
        output_dir / f"{output_prefix}_logreg_best_train_test.png",
        plot_X_train,
        y_train,
        plot_X_test,
        y_test,
    )
    print(f"\nBest logistic model by average rank: {best_model_name}")
    if plot_model_name != best_model_name:
        print(f"Plotting fallback (non-degenerate): {plot_model_name}")

    comparison_df = build_base_style_comparison_df([
        comparison_row_from_metrics("PCA Base", pca_base_results),
        comparison_row_from_metrics("Ridge Log. Reg.", ridge_results),
        comparison_row_from_metrics("LASSO Log. Reg.", lasso_results),
        # comparison_row_from_metrics("Elastic Net Log. Reg.", elastic_results),
        comparison_row_from_metrics("PCA Ridge(int.) Log. Reg.", pca_ridge_results),
        comparison_row_from_metrics("PCA LASSO(int.) Log. Reg.", pca_lasso_results),
    ])
    comparison_export_df = comparison_df.rename(
        index={
            "Ridge Log. Reg.": "Ridge",
            "LASSO Log. Reg.": "LASSO",
            # "Elastic Net Log. Reg.": "Elastic Net",
            "PCA Ridge(int.) Log. Reg.": "PCA Ridge(int.)",
            "PCA LASSO(int.) Log. Reg.": "PCA LASSO(int.)",
        }
    )
    print("\n===== Logistic Regression Comparison Table =====")
    print(comparison_df.to_string())
    comparison_csv = cwd / "output" / f"{output_prefix}_logistic_regression_comparison.csv"
    comparison_tex = cwd / "output" / f"{output_prefix}_logistic_regression_comparison.tex"
    comparison_export_df.to_csv(comparison_csv, float_format='%.3f')
    write_base_style_latex_table(
        comparison_export_df,
        comparison_tex,
        'Logistic Regression Model Comparison',
        'tab:logistic_regression_comparison',
        'Test Acc = plain hold-out accuracy on the final 20% test split. All reported CV/train/test accuracy columns in this table use plain accuracy after hyperparameters were selected by CV balanced accuracy. Recall = positive-class sensitivity.'
    )
    print(f"Local ranked/exported winner in logistic_regression.py: {best_model_name}")
    print(f"Local plot winner in logistic_regression.py: {plot_model_name}")
    global_ranked_df = register_global_model_candidates(
        ranked_df,
        cwd / "output" / f"{output_prefix}_global_model_leaderboard.csv",
        source_script="logistic_regression.py",
        dataset_label=output_prefix,
        comparison_scope="tuned_candidates",
    )
    print(f"Current global best model across registered scripts (informational only): {global_ranked_df.iloc[0]['Model']}")

    append_search_run(
        runs_path=runs_path,
        model_name="LogisticRegression",
        run_time=run_time,
        run_duration_sec=(time.time() - run_start),
        grid_version=grid_label,
        n_jobs=MODEL_N_JOBS,
        dataset_version=dataset_version,
        code_commit=get_git_commit(cwd),
        notes=SEARCH_NOTES
    )
