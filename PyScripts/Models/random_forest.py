#!/usr/bin/env python3
from types import SimpleNamespace
from typing import Any, cast
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
import os
import numpy as np

MPLCONFIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.mplconfig')
os.makedirs(MPLCONFIGDIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(MPLCONFIGDIR))

import matplotlib
matplotlib.use('Agg')
# import matplotlib.pyplot as plt  # Unused in the active code path.
import time

from H_reduce import step_wise_reg_wfv
from H_prep import clean_data, data_clean_param_selection, import_data, to_binary_class
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
    BASE_RF_PARAM_GRID,
    LOGISTIC_MAX_ITER,
    LOGISTIC_TOL,
    PCA_RF_PARAM_GRID,
    SEL_RF_PARAM_GRID,
    TEST_SIZE,
    TIME_SERIES_CV_SPLITS,
    TRAIN_TEST_SHUFFLE,
)

VERBOSE = 0
WINDOW_SIZE = 220
HORIZON = 40
MODEL_N_JOBS = int(os.getenv("MODEL_N_JOBS", "-1"))
# Keep GridSearchCV parallel, but make each RF fit single-threaded to
# avoid nested parallel oversubscription.
RF_FIT_N_JOBS = 1
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

def _rf_effective_max_features(value, n_features: int) -> float:
    n_features = max(1, int(n_features))
    if value == 'sqrt':
        return float(max(1, int(np.sqrt(n_features))))
    if value == 'log2':
        return float(max(1, int(np.log2(n_features))))
    try:
        numeric_value = float(value)
    except Exception:
        return float("inf")
    if 0.0 < numeric_value <= 1.0:
        return float(max(1, int(numeric_value * n_features)))
    return numeric_value


def _make_rf_max_features_sort_value(n_features: int):
    def _sort_value(value):
        return _rf_effective_max_features(value, n_features)

    return _sort_value


def _make_rf_selector_one_se_refit(selector_pipeline, X_train, y_train):
    selector_feature_count_cache: dict[str, int] = {}

    def _selected_feature_count(selector_c) -> int:
        cache_key = str(selector_c)
        if cache_key not in selector_feature_count_cache:
            fitted_pipeline = clone(selector_pipeline)
            fitted_pipeline.set_params(feature_selector__estimator__C=float(selector_c))
            fitted_pipeline.fit(X_train, y_train)
            support = fitted_pipeline.named_steps["feature_selector"].get_support()
            selector_feature_count_cache[cache_key] = int(np.sum(support))
        return selector_feature_count_cache[cache_key]

    def _pick_index(cv_results):
        mean = np.asarray(cv_results["mean_test_score"], dtype=float)
        std = np.asarray(cv_results["std_test_score"], dtype=float)
        se = std / np.sqrt(TIME_SERIES_CV_SPLITS)
        best_idx = int(np.argmax(mean))
        threshold = float(mean[best_idx] - se[best_idx])
        candidate_idx = np.where(mean >= threshold)[0]
        if len(candidate_idx) == 0:
            return best_idx

        def key_fn(i: int):
            selector_c = cv_results["param_feature_selector__estimator__C"][i]
            selected_feature_count = _selected_feature_count(selector_c)
            return (
                selected_feature_count,
                float(cv_results["param_classifier__max_depth"][i]),
                float(cv_results["param_classifier__n_estimators"][i]),
                _rf_effective_max_features(
                    cv_results["param_classifier__max_features"][i],
                    selected_feature_count,
                ),
                float(selector_c),
                -float(mean[i]),
            )

        return int(min(candidate_idx, key=key_fn))

    return _pick_index


def load_rf_input_data():
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
    print(f"MODEL_N_JOBS={MODEL_N_JOBS} (set env MODEL_N_JOBS to override)")
    print(f"GRID_VERSION={GRID_VERSION}")
    grid_label = GRID_VERSION
    output_prefix = "sample" if USE_SAMPLE_PARQUET else "8yrs"
    history_path = cwd / "output" / f"{output_prefix}_search_history_rf.csv"
    runs_path = cwd / "output" / f"{output_prefix}_search_runs.csv"
    checkpoint_dir = get_checkpoint_dir(cwd / "output", "random_forest", f"{output_prefix}_{grid_label}")
    dataset_version = (
        "sample_parquet=PyScripts/Data/sample.parquet,extra_features=True,cluster=False,corr_threshold=0.95,corr_level=0"
        if USE_SAMPLE_PARQUET
        else "testing=False,extra_features=True,cluster=False,corr_threshold=0.95,corr_level=0"
    )
    # testing: bool =False, extra_features: bool =True, cluster: bool =False, n_clusters: int =100, corr_threshold: float =0.95, corr_level: int =0
    DATA=load_rf_input_data()

    # Keep feature-engineering configuration fixed for consistency across models.
    FIND_OPTIMAL=False
    
    parameters_={ # These are optimal as of 3/8/2026 4:00 PM w=4
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
        base_RF_model=RandomForestClassifier(max_depth=10, n_estimators=250, random_state=1, n_jobs=RF_FIT_N_JOBS, class_weight='balanced')
        base_RF_model_pipeline=Pipeline([('scaler', StandardScaler()), ('classifier', base_RF_model)])

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

        _, parameters_, best_score=data_clean_param_selection(*DATA, clone(base_RF_model_pipeline), TEST_SIZE, WINDOW_SIZE, HORIZON, **param_grid)
        print(f"Best Utility Score {best_score}")
        print(f"Optimal parameter {parameters_}")

    X, y_regression=cast(Any, clean_data(*DATA, **parameters_))
    y_classification=to_binary_class(y_regression)
    X_train, X_test, y_train, y_test=train_test_split(X, y_classification, test_size=TEST_SIZE, random_state=1, shuffle=TRAIN_TEST_SHUFFLE)

    # Keep the train/validation fold count centralized in `model_grids.TIME_SERIES_CV_SPLITS`.
    tscv = TimeSeriesSplit(n_splits=TIME_SERIES_CV_SPLITS)
    
    # ------- BASE APPLICATION -------
    print("\n\n------- Base RF Model -------")
    RFClassifier_base=RandomForestClassifier(random_state=1, n_jobs=RF_FIT_N_JOBS, class_weight='balanced')

    RF_pipeline_base=Pipeline([('scaler', StandardScaler()), 
                               ('classifier', RFClassifier_base)])

    param_grid={
        'classifier__max_depth': BASE_RF_PARAM_GRID['classifier__max_depth'],
        'classifier__n_estimators': BASE_RF_PARAM_GRID['classifier__n_estimators'],
        'classifier__max_features': BASE_RF_PARAM_GRID['classifier__max_features'],
    }
    raw_max_features_sort = _make_rf_max_features_sort_value(X_train.shape[1])
    grid_search_base=GridSearchCV(
        RF_pipeline_base, param_grid, cv=tscv, n_jobs=MODEL_N_JOBS, return_train_score=True, verbose=GRIDSEARCH_VERBOSE,
        scoring='balanced_accuracy',
        refit=make_one_se_refit(
            ['classifier__max_depth', 'classifier__n_estimators', 'classifier__max_features'],
            n_splits=TIME_SERIES_CV_SPLITS,
            sort_value_map={'classifier__max_features': raw_max_features_sort},
        )
    )
    grid_search_base = fit_or_load_search(
        checkpoint_dir=checkpoint_dir,
        stage_name="rf_base",
        search_obj=grid_search_base,
        X_train=X_train,
        y_train=y_train,
        history_path=history_path,
        run_time=run_time,
        model_name="RF_base",
        grid_label=grid_label,
        notes=SEARCH_NOTES,
    )

    optimized_base_ = grid_search_base.best_estimator_

    results=get_final_metrics(optimized_base_, X_train, y_train, X_test, y_test, label="Base RF")
    base_results = results.copy()
    # RollingWindowBacktest disabled to save runtime. Restore this block if needed later.
    # rwb_obj=RollingWindowBacktest(clone(grid_search_base.best_estimator_), X, y_classification, X_train, WINDOW_SIZE, HORIZON)
    # rwb_obj.rolling_window_backtest(verbose=BACKTEST_VERBOSE)
    # if SHOW_BACKTEST_PLOT:
    #     rwb_obj.display_wfv_results()
    # util_score=utility_score(results, rwb_obj)
    # print(f"Utility Score {util_score:.4}")
    # ------- PCA APPLICATION -------
    print("\n\n------- PCA RF Model -------")
    pca_base_search = fit_or_load_baseline_logistic_pca_search(
        checkpoint_dir=checkpoint_dir,
        stage_name="rf_shared_logreg_pca_base",
        X_train=X_train,
        y_train=y_train,
        history_path=history_path,
        run_time=run_time,
        model_name="LogReg_PCA_Base_for_RF",
        grid_label=grid_label,
        n_splits=TIME_SERIES_CV_SPLITS,
        notes=SEARCH_NOTES,
    )
    X_train_pca, X_test_pca, _, _ = transform_with_fitted_scaler_pca(
        pca_base_search,
        X_train,
        X_test,
    )
    initial_pca_n_components = pca_base_search.best_params_['pca__n_components']
    print(
        "Initial PCA for PCA RF from 1SE no-regularization logistic regression in logistic_regression.py: "
        f"n_components={initial_pca_n_components} ({X_train_pca.shape[1]} components)."
    )

    RFClassifier_PCA=RandomForestClassifier(random_state=1, n_jobs=RF_FIT_N_JOBS, class_weight='balanced')
    RF_pipeline_PCA=Pipeline([('classifier', RFClassifier_PCA)])

    param_grid={
        'classifier__max_depth': PCA_RF_PARAM_GRID['classifier__max_depth'],
        'classifier__n_estimators': PCA_RF_PARAM_GRID['classifier__n_estimators'],
        'classifier__max_features': PCA_RF_PARAM_GRID['classifier__max_features'],
    }
    pca_max_features_sort = _make_rf_max_features_sort_value(X_train_pca.shape[1])
    grid_search_PCA=GridSearchCV(
        RF_pipeline_PCA, param_grid, cv=tscv, n_jobs=MODEL_N_JOBS, return_train_score=True, verbose=GRIDSEARCH_VERBOSE,
        scoring='balanced_accuracy',
        refit=make_one_se_refit(
            ['classifier__max_depth', 'classifier__n_estimators', 'classifier__max_features'],
            n_splits=TIME_SERIES_CV_SPLITS,
            sort_value_map={'classifier__max_features': pca_max_features_sort},
        )
    )
    grid_search_PCA = fit_or_load_search(
        checkpoint_dir=checkpoint_dir,
        stage_name="rf_pca_sequential",
        search_obj=grid_search_PCA,
        X_train=X_train_pca,
        y_train=y_train,
        history_path=history_path,
        run_time=run_time,
        model_name="RF_pca_sequential",
        grid_label=grid_label,
        notes=SEARCH_NOTES,
    )
    fixed_pca_rf = RandomForestClassifier(
        random_state=1,
        n_jobs=RF_FIT_N_JOBS,
        class_weight='balanced',
        max_depth=grid_search_PCA.best_params_['classifier__max_depth'],
        n_estimators=grid_search_PCA.best_params_['classifier__n_estimators'],
        max_features=grid_search_PCA.best_params_['classifier__max_features'],
    )
    pca_rf_search = fit_or_load_fixed_classifier_pca_search(
        checkpoint_dir=checkpoint_dir,
        stage_name="rf_pca_retuned_n_components",
        X_train=X_train,
        y_train=y_train,
        history_path=history_path,
        run_time=run_time,
        model_name="RF_PCA_retuned_n_components",
        grid_label=grid_label,
        n_splits=TIME_SERIES_CV_SPLITS,
        classifier=fixed_pca_rf,
        notes=SEARCH_NOTES,
    )
    X_train_pca, X_test_pca, _, _ = transform_with_fitted_scaler_pca(
        pca_rf_search,
        X_train,
        X_test,
    )
    selected_pca_n_components = pca_rf_search.best_params_['pca__n_components']
    print(
        "Retuned PCA for PCA RF after RF model selection: "
        f"n_components={selected_pca_n_components} ({X_train_pca.shape[1]} components)."
    )
    print("\n========== PCA + RF Refit (retuned PCA, fixed RF params) ==========")
    fixed_pca_rf_refit = clone(fixed_pca_rf)
    fixed_pca_rf_refit.fit(X_train_pca, y_train)
    pca_refit_result = SimpleNamespace(
        best_estimator_=fixed_pca_rf_refit,
        best_params_=grid_search_PCA.best_params_,
    )
    print(f"Fixed params (PCA RF retuned): {pca_refit_result.best_params_}")

    optimized_PCA_ = fixed_pca_rf_refit

    results=get_final_metrics(optimized_PCA_, X_train_pca, y_train, X_test_pca, y_test, label="PCA RF")
    pca_results = results.copy()
    # RollingWindowBacktest disabled to save runtime. Restore this block if needed later.
    # rwb_obj=RollingWindowBacktest(clone(grid_search_PCA.best_estimator_), X, y_classification, X_train, WINDOW_SIZE, HORIZON)
    # rwb_obj.rolling_window_backtest(verbose=BACKTEST_VERBOSE)
    # if SHOW_BACKTEST_PLOT:
    #     rwb_obj.display_wfv_results()
    # util_score=utility_score(results, rwb_obj)
    # print(f"Utility Score {util_score:.4}")
    # ------- LASSO APPLICATION -------
    print("\n\n------- LASSO RF Model -------")
    lasso_selector=SelectFromModel(
        LogisticRegression(
            penalty='l1', solver='saga', random_state=1,
            class_weight='balanced', max_iter=LOGISTIC_MAX_ITER, tol=LOGISTIC_TOL
        ),
        threshold='mean'
    )
    RFClassifier_red_lasso=RandomForestClassifier(random_state=1, n_jobs=RF_FIT_N_JOBS, class_weight='balanced')

    RF_pipeline_lasso=Pipeline([('scaler', StandardScaler()), 
                              ('feature_selector', lasso_selector),
                              ('classifier', RFClassifier_red_lasso)])

    param_grid={
        'feature_selector__estimator__C': SEL_RF_PARAM_GRID['feature_selector__estimator__C'],
        'classifier__max_depth': SEL_RF_PARAM_GRID['classifier__max_depth'],
        'classifier__n_estimators': SEL_RF_PARAM_GRID['classifier__n_estimators'],
        'classifier__max_features': SEL_RF_PARAM_GRID['classifier__max_features'],
    }
    lasso_selector_refit = _make_rf_selector_one_se_refit(RF_pipeline_lasso, X_train, y_train)
    grid_search_LASSO=GridSearchCV(
        RF_pipeline_lasso, param_grid, cv=tscv, n_jobs=MODEL_N_JOBS, return_train_score=True, verbose=GRIDSEARCH_VERBOSE,
        scoring='balanced_accuracy',
        refit=lasso_selector_refit
    )
    grid_search_LASSO = fit_or_load_search(
        checkpoint_dir=checkpoint_dir,
        stage_name="rf_lasso",
        search_obj=grid_search_LASSO,
        X_train=X_train,
        y_train=y_train,
        history_path=history_path,
        run_time=run_time,
        model_name="RF_lasso",
        grid_label=grid_label,
        notes=SEARCH_NOTES,
    )

    optimized_LASSO_ = grid_search_LASSO.best_estimator_

    results=get_final_metrics(optimized_LASSO_, X_train, y_train, X_test, y_test, label="LASSO RF")
    lasso_results = results.copy()
    # RollingWindowBacktest disabled to save runtime. Restore this block if needed later.
    # rwb_obj=RollingWindowBacktest(clone(grid_search_LASSO.best_estimator_), X, y_classification, X_train, WINDOW_SIZE, HORIZON)
    # rwb_obj.rolling_window_backtest(verbose=BACKTEST_VERBOSE)
    # if SHOW_BACKTEST_PLOT:
    #     rwb_obj.display_wfv_results()
    # util_score=utility_score(results, rwb_obj)
    # print(f"Utility Score {util_score:.4}")
    # ------- RIDGE APPLICATION -------
    print("\n\n------- RIDGE RF Model -------")
    ridge_selector=SelectFromModel(
        LogisticRegression(
            penalty='l2', solver="saga", random_state=1,
            class_weight='balanced', max_iter=LOGISTIC_MAX_ITER, tol=LOGISTIC_TOL
        ),
        threshold='mean'
    )
    RFClassifier_red_ridge=RandomForestClassifier(random_state=1, n_jobs=RF_FIT_N_JOBS, class_weight='balanced')

    RF_pipeline_ridge=Pipeline([('scaler', StandardScaler()), 
                              ('feature_selector', ridge_selector),
                              ('classifier', RFClassifier_red_ridge)])

    param_grid={
        'feature_selector__estimator__C': SEL_RF_PARAM_GRID['feature_selector__estimator__C'],
        'classifier__max_depth': SEL_RF_PARAM_GRID['classifier__max_depth'],
        'classifier__n_estimators': SEL_RF_PARAM_GRID['classifier__n_estimators'],
        'classifier__max_features': SEL_RF_PARAM_GRID['classifier__max_features'],
    }
    grid_search_ridge=GridSearchCV(
        RF_pipeline_ridge, param_grid, cv=tscv, n_jobs=MODEL_N_JOBS, return_train_score=True, verbose=GRIDSEARCH_VERBOSE,
        scoring='balanced_accuracy',
        refit=_make_rf_selector_one_se_refit(RF_pipeline_ridge, X_train, y_train)
    )
    grid_search_ridge = fit_or_load_search(
        checkpoint_dir=checkpoint_dir,
        stage_name="rf_ridge",
        search_obj=grid_search_ridge,
        X_train=X_train,
        y_train=y_train,
        history_path=history_path,
        run_time=run_time,
        model_name="RF_ridge",
        grid_label=grid_label,
        notes=SEARCH_NOTES,
    )

    optimized_ridge_ = grid_search_ridge.best_estimator_

    results=get_final_metrics(optimized_ridge_, X_train, y_train, X_test, y_test, label="Ridge RF")
    ridge_results = results.copy()
    # RollingWindowBacktest disabled to save runtime. Restore this block if needed later.
    # rwb_obj=RollingWindowBacktest(clone(grid_search_ridge.best_estimator_), X, y_classification, X_train, WINDOW_SIZE, HORIZON)
    # rwb_obj.rolling_window_backtest(verbose=BACKTEST_VERBOSE)
    # if SHOW_BACKTEST_PLOT:
    #     rwb_obj.display_wfv_results()
    # util_score=utility_score(results, rwb_obj)
    # print(f"Utility Score {util_score:.4}")
    ranking_df = pd.DataFrame([
        {"Model": "Base RF", **base_results},
        {"Model": "PCA RF", **pca_results},
        {"Model": "LASSO RF", **lasso_results},
        {"Model": "Ridge RF", **ridge_results},
    ])
    ranked_df = rank_models_by_metrics(ranking_df)
    best_model_name = str(ranked_df.iloc[0]["Model"])
    plot_model_name = select_non_degenerate_plot_model(ranked_df)
    print(f"\nBest RF model by average rank: {best_model_name}")
    best_plot_config = {
        "Base RF": (
            grid_search_base,
            [("classifier__max_depth", "max_depth"), ("classifier__n_estimators", "n_estimators"), ("classifier__max_features", "max_features")],
        ),
        "PCA RF": (
            grid_search_PCA,
            [("classifier__max_depth", "max_depth"), ("classifier__n_estimators", "n_estimators"), ("classifier__max_features", "max_features")],
        ),
        "LASSO RF": (
            grid_search_LASSO,
            [("feature_selector__estimator__C", "Selector C"), ("classifier__max_depth", "max_depth"), ("classifier__n_estimators", "n_estimators"), ("classifier__max_features", "max_features")],
        ),
        "Ridge RF": (
            grid_search_ridge,
            [("feature_selector__estimator__C", "Selector C"), ("classifier__max_depth", "max_depth"), ("classifier__n_estimators", "n_estimators"), ("classifier__max_features", "max_features")],
        ),
    }[plot_model_name]
    output_dir = cwd / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_X_train = X_train_pca if plot_model_name == "PCA RF" else X_train
    plot_X_test = X_test_pca if plot_model_name == "PCA RF" else X_test
    save_best_model_plots_from_gridsearch_all_params(
        best_plot_config[0],
        best_plot_config[1],
        plot_model_name,
        output_dir / f"{output_prefix}_rf_best_bias_variance.png",
        output_dir / f"{output_prefix}_rf_best_train_test.png",
        plot_X_train,
        y_train,
        plot_X_test,
        y_test,
    )
    if plot_model_name != best_model_name:
        print(f"Plotting fallback (non-degenerate): {plot_model_name}")
    comparison_df = build_base_style_comparison_df([
        comparison_row_from_metrics("Base RF", base_results),
        comparison_row_from_metrics("PCA RF", pca_results),
        comparison_row_from_metrics("LASSO RF", lasso_results),
        comparison_row_from_metrics("Ridge RF", ridge_results),
    ])
    comparison_export_df = comparison_df.rename(index={"Base RF": "Raw RF"})
    print("\n===== Random Forest Comparison Table =====")
    print(comparison_df.to_string())
    comparison_csv = cwd / "output" / f"{output_prefix}_random_forest_comparison.csv"
    comparison_tex = cwd / "output" / f"{output_prefix}_random_forest_comparison.tex"
    comparison_export_df.to_csv(comparison_csv, float_format='%.3f')
    write_base_style_latex_table(
        comparison_export_df,
        comparison_tex,
        'Random Forest Model Comparison',
        'tab:random_forest_comparison',
        'Test Acc = plain hold-out accuracy on the final 20% test split. All reported CV/train/test accuracy columns in this table use plain accuracy after hyperparameters were selected by CV balanced accuracy. Recall = positive-class sensitivity.'
    )
    print(f"Local ranked/exported winner in random_forest.py: {best_model_name}")
    print(f"Local plot winner in random_forest.py: {plot_model_name}")
    global_ranked_df = register_global_model_candidates(
        ranked_df,
        cwd / "output" / f"{output_prefix}_global_model_leaderboard.csv",
        source_script="random_forest.py",
        dataset_label=output_prefix,
        comparison_scope="tuned_candidates",
    )
    print(f"Current global best model across registered scripts (informational only): {global_ranked_df.iloc[0]['Model']}")

    # ------- STEP-WISE REGRESSION APPLICATION (DISABLED) -------
    if False:
        print("\n\n------- LASSO(internal) -> STEP-WISE REGRESSION RF Model -------")
        lasso_selector=SelectFromModel(
            LogisticRegression(
                penalty='l1', solver='saga', random_state=1,
                class_weight='balanced', max_iter=LOGISTIC_MAX_ITER, tol=LOGISTIC_TOL
            ),
            max_features=100,
            threshold='mean'
        )
        RFClassifier_red_lasso=RandomForestClassifier(random_state=1, n_jobs=RF_FIT_N_JOBS, class_weight='balanced')

        RF_pipeline_lasso=Pipeline([('scaler', StandardScaler()), 
                                  ('feature_selector', lasso_selector),
                                  ('classifier', RFClassifier_red_lasso)])

        param_grid={
            'feature_selector__estimator__C': SEL_RF_PARAM_GRID['feature_selector__estimator__C'],
            'classifier__max_depth': SEL_RF_PARAM_GRID['classifier__max_depth'],
            'classifier__n_estimators': SEL_RF_PARAM_GRID['classifier__n_estimators'],
            'classifier__max_features': SEL_RF_PARAM_GRID['classifier__max_features'],
        }
        grid_search_LASSO=GridSearchCV(
            RF_pipeline_lasso, param_grid, cv=tscv, n_jobs=MODEL_N_JOBS, return_train_score=True, verbose=GRIDSEARCH_VERBOSE,
            scoring='balanced_accuracy',
            refit=_make_rf_selector_one_se_refit(RF_pipeline_lasso, X_train, y_train)
        )
        grid_search_LASSO.fit(X_train, y_train) 
        append_search_history(
            history_path=history_path,
            cv_results=grid_search_LASSO.cv_results_,
            run_time=run_time,
            model_name="RF_stepwise_prefilter",
            search_type="grid",
            grid_version=grid_label,
            notes=SEARCH_NOTES,
            best_params=grid_search_LASSO.best_params_
        )

        best_params_from_grid = grid_search_LASSO.best_params_

        RF_params = {k.replace('classifier__', ''): v 
                 for k, v in best_params_from_grid.items() 
                 if k.startswith('classifier__')}

        lasso_support = grid_search_LASSO.best_estimator_.named_steps['feature_selector'].get_support()

        lasso_coefficient_names = X_train.columns[lasso_support].tolist()

        X_train_red=X_train[lasso_coefficient_names]
        X_test_red=X_test[lasso_coefficient_names]

        RFClassifier_red_sw_wfv_pipeline=Pipeline([('scaler', StandardScaler()),
                                                   ('classifier', RandomForestClassifier(**RF_params, random_state=1, n_jobs=1, class_weight='balanced'))])

        X_train_final, X_test_final=step_wise_reg_wfv(RFClassifier_red_sw_wfv_pipeline, X_train_red, y_train, X_test_red) 

        RFClassifier_red_sw_wfv_pipeline.fit(X_train_final, y_train)

        copy_RFClassifier_red_sw_wfv_pipeline=clone(RFClassifier_red_sw_wfv_pipeline)
        copy_RFClassifier_red_sw_wfv_pipeline.fit(X_train_final, y_train)

        results=get_final_metrics(copy_RFClassifier_red_sw_wfv_pipeline, X_train_final, y_train, X_test_final, y_test, label="Stepwise RF")
        # RollingWindowBacktest disabled to save runtime. Restore this block if needed later.
        # rwb_obj=RollingWindowBacktest(clone(RFClassifier_red_sw_wfv_pipeline), X, y_classification, X_train, WINDOW_SIZE, HORIZON)
        # rwb_obj.rolling_window_backtest(verbose=BACKTEST_VERBOSE)
        # if SHOW_BACKTEST_PLOT:
        #     rwb_obj.display_wfv_results()
        # util_score=utility_score(results, rwb_obj)
        # print(f"Utility Score {util_score:.4}")
    else:
        print("\n\n------- STEP-WISE RF Model skipped (disabled) -------")
    append_search_run(
        runs_path=runs_path,
        model_name="RandomForest",
        run_time=run_time,
        run_duration_sec=(time.time() - run_start),
        grid_version=grid_label,
        n_jobs=MODEL_N_JOBS,
        dataset_version=dataset_version,
        code_commit=get_git_commit(cwd),
        notes=SEARCH_NOTES
    )
