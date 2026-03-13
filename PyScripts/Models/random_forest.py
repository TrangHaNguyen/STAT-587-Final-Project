#!/usr/bin/env python3
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq
import time

from H_reduce import step_wise_reg_wfv
from H_prep import clean_data, data_clean_param_selection, import_data
from H_eval import (
    RollingWindowBacktest,
    get_final_metrics,
    utility_score,
    rank_models_by_metrics,
    save_best_model_plots_from_gridsearch,
)
from H_helpers import log_result, get_cwd, append_params_to_dict
from H_search_history import append_search_history, append_search_run, get_git_commit, now_iso
from model_grids import (
    BASE_RF_PARAM_GRID,
    PCA_RF_PARAM_GRID,
    SEL_RF_PARAM_GRID,
)

VERBOSE=0
WINDOW_SIZE=220
HORIZON=40
EXPORT=True
MODEL_N_JOBS=int(os.getenv("MODEL_N_JOBS", "-1"))
# Keep GridSearchCV parallel, but make each RF fit single-threaded to
# avoid nested parallel oversubscription.
RF_FIT_N_JOBS = 1
GRID_VERSION=os.getenv("GRID_VERSION", "v1")
SEARCH_NOTES=os.getenv("SEARCH_NOTES", "")
USE_SAMPLE_PARQUET = os.getenv("USE_SAMPLE_PARQUET", "0") == "1"
SAMPLE_PARQUET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'Data', 'sample.parquet'
)

cwd=get_cwd("STAT-587-Final-Project")


def _as_sortable_numeric(value):
    try:
        return float(value)
    except Exception:
        return float("inf")


def _rf_max_features_sort_value(value):
    if value == 'log2':
        return 0.0
    if value == 'sqrt':
        return 1.0
    try:
        return 2.0 + float(value)
    except Exception:
        return float("inf")


def make_one_se_refit(complexity_cols: list[str], fixed_cols: list[str] | None = None):
    """Return a GridSearchCV refit callable implementing the 1-SE rule."""
    def _pick_index(cv_results):
        import numpy as np
        mean = np.asarray(cv_results["mean_test_score"], dtype=float)
        std = np.asarray(cv_results["std_test_score"], dtype=float)
        se = std / np.sqrt(5)
        best_idx = int(np.argmax(mean))
        threshold = float(mean[best_idx] - se[best_idx])
        candidate_idx = np.where(mean >= threshold)[0]
        if len(candidate_idx) == 0:
            return best_idx
        if fixed_cols:
            for col in fixed_cols:
                param_key = f"param_{col}"
                best_val = cv_results[param_key][best_idx]
                candidate_idx = np.array([i for i in candidate_idx if cv_results[param_key][i] == best_val], dtype=int)
                if len(candidate_idx) == 0:
                    return best_idx

        def key_fn(i: int):
            complexity = []
            for col in complexity_cols:
                val = cv_results[f"param_{col}"][i]
                if col == 'classifier__max_features':
                    complexity.append(_rf_max_features_sort_value(val))
                else:
                    complexity.append(_as_sortable_numeric(val))
            # Prefer simplest model; if tie, prefer higher score.
            return tuple(complexity + [-float(mean[i])])

        return int(min(candidate_idx, key=key_fn))

    return _pick_index


def load_rf_input_data():
    if not USE_SAMPLE_PARQUET:
        return import_data(extra_features=True, testing=False, cluster=False, n_clusters=100, corr_threshold=0.95, corr_level=0)

    print(f"USE_SAMPLE_PARQUET=1 -> loading sample parquet from {os.path.abspath(SAMPLE_PARQUET_PATH)}")
    idx = pd.IndexSlice
    table = pq.read_table(SAMPLE_PARQUET_PATH)
    data = table.to_pandas()
    print("Finished Downloading Data -------")
    print("Initial shape:", data.shape[0], "rows,", data.shape[1], "columns.")

    print("------- Cleaning data")
    for data_type in ['Stocks']:
        temp_data = data.loc[:, idx[:, data_type, :]].dropna(how="all", axis=0)
        missing_one = (temp_data.isna().sum() == 1)
        cols = missing_one[missing_one == 1].index
        temp_data[cols] = temp_data[cols].ffill()
        temp_data = temp_data.dropna(how="any", axis=1)
        data = data.drop(columns=data_type, level=1).join(temp_data)

    stocks = data.loc[:, idx[:, 'Stocks', :]]
    to_drop = stocks.index[stocks.isna().all(axis=1)]
    data = data.drop(index=to_drop)
    print("Finished Cleaning Data -------")
    print("Current shape:", data.shape[0], "rows,", data.shape[1], "columns.")

    data = pd.concat([
        data,
        data.loc[:, idx[['Close', 'Open', 'High', 'Low'], 'Stocks', :]]
        .copy()
        .pct_change()
        .rename(columns={metric: f"{metric} PC" for metric in ['Close', 'Open', 'High', 'Low']}, level=0)
    ], axis=1)
    print("Created Percent Changes.")

    y_regression = (
        (data.loc[:, idx['Close', 'Index', '^SPX']] - data.loc[:, idx['Open', 'Index', '^SPX']])
        / data.loc[:, idx['Open', 'Index', '^SPX']]
    ).rename("Target Regression").shift(-1)
    print("Created Target (Regression).")

    data[("Day of Week", "Calendar", "All")] = data.index.dayofweek
    print("---EXTRA---: Created Day of Week.")

    high_ = data.loc[:, idx['High', :, :]]
    low_ = data.loc[:, idx['Low', :, :]]
    data = pd.concat([
        data,
        pd.DataFrame(high_.values - low_.values, index=high_.index, columns=high_.columns)
        .rename(columns={'High': 'Daily Range'}, level=0)
    ], axis=1)
    print("---EXTRA---: Created Daily Range.")

    return data, y_regression

if __name__=="__main__":
    run_start = time.time()
    run_time = now_iso()
    WINDOW_SIZE=200
    HORIZON=40
    EXPORT=True
    TEST_SIZE=0.2
    print(f"MODEL_N_JOBS={MODEL_N_JOBS} (set env MODEL_N_JOBS to override)")
    print(f"GRID_VERSION={GRID_VERSION}")
    grid_label = GRID_VERSION
    output_prefix = "sample" if USE_SAMPLE_PARQUET else "8yrs"
    history_path = cwd / "output" / f"{output_prefix}_search_history_rf.csv"
    runs_path = cwd / "output" / f"{output_prefix}_search_runs.csv"
    results_file = f"{output_prefix}_results.csv"
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
        "sector": True,
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
            'sector': [True],
            'corr_level': [0],
        }

        _, parameters_, best_score=data_clean_param_selection(*DATA, clone(base_RF_model_pipeline), TEST_SIZE, WINDOW_SIZE, HORIZON, **param_grid)
        print(f"Best Utility Score {best_score}")
        print(f"Optimal parameter {parameters_}")

    download_params = {f"clean_data__{k}=": v for k, v in parameters_.items()}

    X, y_regression=cast(Any, clean_data(*DATA, **parameters_))
    def to_binary_class(y):
        return (y>=0).astype(int)
    y_classification=to_binary_class(y_regression)
    X_train, X_test, y_train, y_test=train_test_split(X, y_classification, test_size=TEST_SIZE, random_state=1, shuffle=False)

    # Previous temporary change used `KFold(n_splits=5, shuffle=False)`.
    tscv = TimeSeriesSplit(n_splits=5)  # CHANGEABLE (OPTIONAL)
    
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
    grid_search_base=GridSearchCV(
        RF_pipeline_base, param_grid, cv=tscv, n_jobs=MODEL_N_JOBS, return_train_score=True, verbose=VERBOSE,
        scoring='balanced_accuracy',
        refit=make_one_se_refit(['classifier__max_depth', 'classifier__n_estimators', 'classifier__max_features'])
    )
    grid_search_base.fit(X_train, y_train)
    append_search_history(
        history_path=history_path,
        cv_results=grid_search_base.cv_results_,
        run_time=run_time,
        model_name="RF_base",
        search_type="grid",
        grid_version=grid_label,
        notes=SEARCH_NOTES,
        best_params=grid_search_base.best_params_
    )

    rwb_obj=RollingWindowBacktest(clone(grid_search_base.best_estimator_), X, y_classification, X_train, WINDOW_SIZE, HORIZON)
    rwb_obj.rolling_window_backtest(verbose=1)
    rwb_obj.display_wfv_results()

    optimized_base_=clone(grid_search_base.best_estimator_)
    optimized_base_.fit(X_train, y_train)

    results=get_final_metrics(optimized_base_, X_train, y_train, X_test, y_test, label="Base RF")
    base_results = results.copy()
    util_score=utility_score(results, rwb_obj)
    print(f"Utility Score {util_score:.4}")
    if (EXPORT):
        results.update({'utility_score': round(util_score, 3)})
        results=append_params_to_dict(results, optimized_base_)
        results.update(rwb_obj.results[2])
        results.update(download_params)
        log_result(results, cwd / 'output', results_file)
    
    # ------- PCA APPLICATION -------
    print("\n\n------- PCA RF Model -------")
    RFClassifier_PCA=RandomForestClassifier(random_state=1, n_jobs=RF_FIT_N_JOBS, class_weight='balanced')

    RF_pipeline_PCA=Pipeline([('scaler', StandardScaler()),
                              ('reducer', PCA()),
                              ('classifier', RFClassifier_PCA)])
    
    param_grid={
        'reducer__n_components': PCA_RF_PARAM_GRID['reducer__n_components'],
        'classifier__max_depth': PCA_RF_PARAM_GRID['classifier__max_depth'],
        'classifier__n_estimators': PCA_RF_PARAM_GRID['classifier__n_estimators'],
        'classifier__max_features': PCA_RF_PARAM_GRID['classifier__max_features'],
    }
    grid_search_PCA=GridSearchCV(
        RF_pipeline_PCA, param_grid, cv=tscv, n_jobs=MODEL_N_JOBS, return_train_score=True, verbose=VERBOSE,
        scoring='balanced_accuracy',
        refit=make_one_se_refit(['classifier__max_depth', 'classifier__n_estimators', 'classifier__max_features'], fixed_cols=['reducer__n_components'])
    )
    grid_search_PCA.fit(X_train, y_train)
    append_search_history(
        history_path=history_path,
        cv_results=grid_search_PCA.cv_results_,
        run_time=run_time,
        model_name="RF_pca",
        search_type="grid",
        grid_version=grid_label,
        notes=SEARCH_NOTES,
        best_params=grid_search_PCA.best_params_
    )

    rwb_obj=RollingWindowBacktest(clone(grid_search_PCA.best_estimator_), X, y_classification, X_train, WINDOW_SIZE, HORIZON)
    rwb_obj.rolling_window_backtest(verbose=1)
    rwb_obj.display_wfv_results()

    optimized_PCA_=clone(grid_search_PCA.best_estimator_)
    optimized_PCA_.fit(X_train, y_train)

    results=get_final_metrics(optimized_PCA_, X_train, y_train, X_test, y_test, label="PCA RF")
    pca_results = results.copy()
    util_score=utility_score(results, rwb_obj)
    print(f"Utility Score {util_score:.4}")
    if (EXPORT):
        results.update({'utility_score': round(util_score, 3)})
        results=append_params_to_dict(results, optimized_PCA_)
        results.update(rwb_obj.results[2])
        results.update(download_params)
        log_result(results, cwd / 'output', results_file)

    # ------- LASSO APPLICATION -------
    print("\n\n------- LASSO RF Model -------")
    lasso_selector=SelectFromModel(LogisticRegression(l1_ratio=1, solver='saga', random_state=1, class_weight='balanced', max_iter=500, tol=5e-2), threshold='mean')
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
        RF_pipeline_lasso, param_grid, cv=tscv, n_jobs=MODEL_N_JOBS, return_train_score=True, verbose=VERBOSE,
        scoring='balanced_accuracy',
        refit=make_one_se_refit(['feature_selector__estimator__C', 'classifier__max_depth', 'classifier__n_estimators', 'classifier__max_features'])
    )
    grid_search_LASSO.fit(X_train, y_train)
    append_search_history(
        history_path=history_path,
        cv_results=grid_search_LASSO.cv_results_,
        run_time=run_time,
        model_name="RF_lasso",
        search_type="grid",
        grid_version=grid_label,
        notes=SEARCH_NOTES,
        best_params=grid_search_LASSO.best_params_
    )

    rwb_obj=RollingWindowBacktest(clone(grid_search_LASSO.best_estimator_), X, y_classification, X_train, WINDOW_SIZE, HORIZON)
    rwb_obj.rolling_window_backtest(verbose=1)
    rwb_obj.display_wfv_results()

    optimized_LASSO_=clone(grid_search_LASSO.best_estimator_)
    optimized_LASSO_.fit(X_train, y_train)

    results=get_final_metrics(optimized_LASSO_, X_train, y_train, X_test, y_test, label="LASSO RF")
    lasso_results = results.copy()
    util_score=utility_score(results, rwb_obj)
    print(f"Utility Score {util_score:.4}")
    if (EXPORT):
        results.update({'utility_score': round(util_score, 3)})
        results=append_params_to_dict(results, optimized_LASSO_)
        results.update(rwb_obj.results[2])
        results.update(download_params)
        log_result(results, cwd / 'output', results_file)
        
    # ------- RIDGE APPLICATION -------
    print("\n\n------- RIDGE RF Model -------")
    ridge_selector=SelectFromModel(LogisticRegression(l1_ratio=0, solver="saga", random_state=1, class_weight='balanced', max_iter=500, tol=5e-2), threshold='mean')
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
        RF_pipeline_ridge, param_grid, cv=tscv, n_jobs=MODEL_N_JOBS, return_train_score=True, verbose=VERBOSE,
        scoring='balanced_accuracy',
        refit=make_one_se_refit(['feature_selector__estimator__C', 'classifier__max_depth', 'classifier__n_estimators', 'classifier__max_features'])
    )
    grid_search_ridge.fit(X_train, y_train)
    append_search_history(
        history_path=history_path,
        cv_results=grid_search_ridge.cv_results_,
        run_time=run_time,
        model_name="RF_ridge",
        search_type="grid",
        grid_version=grid_label,
        notes=SEARCH_NOTES,
        best_params=grid_search_ridge.best_params_
    )

    rwb_obj=RollingWindowBacktest(clone(grid_search_ridge.best_estimator_), X, y_classification, X_train, WINDOW_SIZE, HORIZON)
    rwb_obj.rolling_window_backtest(verbose=1)
    rwb_obj.display_wfv_results()

    optimized_ridge_=clone(grid_search_ridge.best_estimator_)
    optimized_ridge_.fit(X_train, y_train)

    results=get_final_metrics(optimized_ridge_, X_train, y_train, X_test, y_test, label="Ridge RF")
    ridge_results = results.copy()
    util_score=utility_score(results, rwb_obj)
    print(f"Utility Score {util_score:.4}")
    if (EXPORT):
        results.update({'utility_score': round(util_score, 3)})
        results=append_params_to_dict(results, optimized_ridge_)
        results.update(rwb_obj.results[2])
        results.update(download_params)
        log_result(results, cwd / 'output', results_file)
        
    ranking_df = pd.DataFrame([
        {"Model": "Base RF", **base_results},
        {"Model": "PCA RF", **pca_results},
        {"Model": "LASSO RF", **lasso_results},
        {"Model": "Ridge RF", **ridge_results},
    ])
    ranked_df = rank_models_by_metrics(ranking_df)
    best_model_name = str(ranked_df.iloc[0]["Model"])
    print(f"\nBest RF model by average rank: {best_model_name}")
    best_plot_config = {
        "Base RF": (grid_search_base, "classifier__max_depth", "max_depth"),
        "PCA RF": (grid_search_PCA, "reducer__n_components", "PCA n_components"),
        "LASSO RF": (grid_search_LASSO, "feature_selector__estimator__C", "Selector C"),
        "Ridge RF": (grid_search_ridge, "feature_selector__estimator__C", "Selector C"),
    }[best_model_name]
    output_dir = cwd / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_best_model_plots_from_gridsearch(
        best_plot_config[0],
        best_plot_config[1],
        best_plot_config[2],
        best_model_name,
        output_dir / f"{output_prefix}_rf_best_bias_variance.png",
        output_dir / f"{output_prefix}_rf_best_train_test.png",
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # ------- STEP-WISE REGRESSION APPLICATION (DISABLED) -------
    if False:
        print("\n\n------- LASSO(internal) -> STEP-WISE REGRESSION RF Model -------")
        lasso_selector=SelectFromModel(LogisticRegression(l1_ratio=1, solver='saga', random_state=1, class_weight='balanced', max_iter=500, tol=5e-2), max_features=100, threshold='mean')
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
            RF_pipeline_lasso, param_grid, cv=tscv, n_jobs=MODEL_N_JOBS, return_train_score=True, verbose=VERBOSE,
            scoring='balanced_accuracy',
            refit=make_one_se_refit(['feature_selector__estimator__C', 'classifier__max_depth', 'classifier__n_estimators', 'classifier__max_features'])
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

        rwb_obj=RollingWindowBacktest(clone(RFClassifier_red_sw_wfv_pipeline), X, y_classification, X_train, WINDOW_SIZE, HORIZON)
        rwb_obj.rolling_window_backtest(verbose=1)
        rwb_obj.display_wfv_results()

        copy_RFClassifier_red_sw_wfv_pipeline=clone(RFClassifier_red_sw_wfv_pipeline)
        copy_RFClassifier_red_sw_wfv_pipeline.fit(X_train_final, y_train)

        results=get_final_metrics(copy_RFClassifier_red_sw_wfv_pipeline, X_train_final, y_train, X_test_final, y_test, label="Stepwise RF")
        util_score=utility_score(results, rwb_obj)
        print(f"Utility Score {util_score:.4}")
        if (EXPORT):
            results.update({'utility_score': round(util_score, 3)})
            results=append_params_to_dict(results, RFClassifier_red_sw_wfv_pipeline)
            results.update(rwb_obj.results[2])
            results.update(download_params)
            log_result(results, cwd / 'output', results_file)
            
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
