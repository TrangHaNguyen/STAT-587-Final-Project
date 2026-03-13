from typing import Any, cast
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import clone
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
import os
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from H_prep import clean_data, data_clean_param_selection, import_data
from H_eval import (
    get_final_metrics,
    RollingWindowBacktest,
    utility_score,
    rank_models_by_metrics,
    save_best_model_plots_from_gridsearch,
    comparison_row_from_metrics,
    build_base_style_comparison_df,
    write_base_style_latex_table,
)
from H_helpers import log_result, append_params_to_dict, get_cwd
from H_search_history import append_search_history, append_search_run, get_git_commit, now_iso
from model_grids import (
    BASELINE_PCA_GRID,
    ELASTIC_NET_GRID,
    ELASTIC_NET_L1_RATIO_GRID,
    LASSO_GRID,
    LOGISTIC_TOL,
    RIDGE_GRID,
)

VERBOSE=0
MODEL_N_JOBS=int(os.getenv("MODEL_N_JOBS", "-1"))
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


def make_one_se_refit(complexity_cols: list[str], fixed_cols: list[str] | None = None):
    """Return a GridSearchCV refit callable implementing the 1-SE rule."""
    def _pick_index(cv_results):
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
                complexity.append(_as_sortable_numeric(val))
            # Prefer simplest model; if tie, prefer higher score.
            return tuple(complexity + [-float(mean[i])])

        return int(min(candidate_idx, key=key_fn))

    return _pick_index


def load_logreg_input_data():
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
    # Previous temporary change used `KFold(n_splits=5, shuffle=False)`.
    tscv = TimeSeriesSplit(n_splits=5)
    print(f"MODEL_N_JOBS={MODEL_N_JOBS} (set env MODEL_N_JOBS to override)")
    print(f"GRID_VERSION={GRID_VERSION}")
    grid_label = GRID_VERSION
    output_prefix = "sample" if USE_SAMPLE_PARQUET else "8yrs"
    history_path = cwd / "output" / f"{output_prefix}_search_history_logreg.csv"
    runs_path = cwd / "output" / f"{output_prefix}_search_runs.csv"
    results_file = f"{output_prefix}_results.csv"
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
        base_Log_Reg_model=LogisticRegression(C=1.0, l1_ratio=0, solver='saga', class_weight='balanced', random_state=1, max_iter=1000, tol=LOGISTIC_TOL, verbose=VERBOSE)
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

    download_params = {f"clean_data__{k}=": v for k, v in parameters_.items()}

    X, y_regression=cast(Any, clean_data(*DATA, **parameters_))
    def to_binary_class(y):
        return (y>=0).astype(int)
    y_classification=to_binary_class(y_regression)
    X_train, X_test, y_train, y_test=train_test_split(X, y_classification, test_size=TEST_SIZE, random_state=1, shuffle=False)

    # ------- PCA Base (No Logistic Regularization) -------
    print("\n\n------- PCA Base Logistic Model -------")
    Log_Reg_PCA_Base = LogisticRegression(
        C=np.inf, solver='lbfgs', class_weight='balanced',
        random_state=1, max_iter=500, tol=LOGISTIC_TOL
    )
    Log_Reg_model_pipeline_PCA_Base = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA()),
        ('classifier', Log_Reg_PCA_Base)
    ])
    param_grid = {
        'pca__n_components': BASELINE_PCA_GRID
    }
    grid_search_PCA_base = GridSearchCV(
        Log_Reg_model_pipeline_PCA_Base, param_grid, cv=tscv,
        return_train_score=True, verbose=VERBOSE,
        scoring='balanced_accuracy', refit=True
    )
    grid_search_PCA_base.fit(X_train, y_train)
    append_search_history(
        history_path=history_path,
        cv_results=grid_search_PCA_base.cv_results_,
        run_time=run_time,
        model_name="LogReg_PCA_Base",
        search_type="grid",
        grid_version=grid_label,
        notes=SEARCH_NOTES,
        best_params=grid_search_PCA_base.best_params_
    )
    rwb_obj=RollingWindowBacktest(clone(grid_search_PCA_base.best_estimator_), X, y_classification, X_train, WINDOW_SIZE, HORIZON)
    rwb_obj.rolling_window_backtest(verbose=1)
    rwb_obj.display_wfv_results()
    optimized_Log_Reg_PCA_base_ = grid_search_PCA_base.best_estimator_
    results=get_final_metrics(optimized_Log_Reg_PCA_base_, X_train, y_train, X_test, y_test, n_splits=10, label="PCA Base")
    pca_base_results = results.copy()
    util_score=utility_score(results, rwb_obj)
    print(f"Utility Score {util_score:.4}")
    if (EXPORT):
        results.update({'utility_score': round(util_score, 3)})
        results=append_params_to_dict(results, optimized_Log_Reg_PCA_base_)
        results.update(rwb_obj.results[2])
        results.update(download_params)
        log_result(results, cwd / 'output', results_file)

    # ------- Ridge (No PCA) -------
    print("\n\n------- Logistic Ridge Model -------")
    Log_Reg_Ridge = LogisticRegression(
        l1_ratio=0, solver='saga', class_weight='balanced',
        random_state=1, max_iter=500, tol=LOGISTIC_TOL
    )
    Log_Reg_model_pipeline_Ridge = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', Log_Reg_Ridge)
    ])
    param_grid = {
        'classifier__C': RIDGE_GRID
    }
    grid_search_ridge = GridSearchCV(
        Log_Reg_model_pipeline_Ridge, param_grid, cv=tscv, return_train_score=True, verbose=VERBOSE,
        scoring='balanced_accuracy',
        refit=make_one_se_refit(['classifier__C'])
    )
    grid_search_ridge.fit(X_train, y_train)
    append_search_history(
        history_path=history_path,
        cv_results=grid_search_ridge.cv_results_,
        run_time=run_time,
        model_name="LogReg_Ridge",
        search_type="grid",
        grid_version=grid_label,
        notes=SEARCH_NOTES,
        best_params=grid_search_ridge.best_params_
    )
    rwb_obj=RollingWindowBacktest(clone(grid_search_ridge.best_estimator_), X, y_classification, X_train, WINDOW_SIZE, HORIZON)
    rwb_obj.rolling_window_backtest(verbose=1)
    rwb_obj.display_wfv_results()
    optimized_Log_Reg_ridge_ = grid_search_ridge.best_estimator_
    results=get_final_metrics(optimized_Log_Reg_ridge_, X_train, y_train, X_test, y_test, n_splits=10, label="Ridge Log. Reg.")
    ridge_results = results.copy()
    util_score=utility_score(results, rwb_obj)
    print(f"Utility Score {util_score:.4}")
    if (EXPORT):
        results.update({'utility_score': round(util_score, 3)})
        results=append_params_to_dict(results, optimized_Log_Reg_ridge_)
        results.update(rwb_obj.results[2])
        results.update(download_params)
        log_result(results, cwd / 'output', results_file)

    # ------- LASSO (No PCA) -------
    print("\n\n------- Logistic LASSO Model -------")
    Log_Reg_Lasso = LogisticRegression(
        l1_ratio=1, solver='saga', class_weight='balanced',
        random_state=1, max_iter=500, tol=LOGISTIC_TOL
    )
    Log_Reg_model_pipeline_Lasso = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', Log_Reg_Lasso)
    ])
    param_grid = {
        'classifier__C': LASSO_GRID
    }
    grid_search_lasso = GridSearchCV(
        Log_Reg_model_pipeline_Lasso, param_grid, cv=tscv, return_train_score=True, verbose=VERBOSE,
        scoring='balanced_accuracy',
        refit=make_one_se_refit(['classifier__C'])
    )
    grid_search_lasso.fit(X_train, y_train)
    append_search_history(
        history_path=history_path,
        cv_results=grid_search_lasso.cv_results_,
        run_time=run_time,
        model_name="LogReg_LASSO",
        search_type="grid",
        grid_version=grid_label,
        notes=SEARCH_NOTES,
        best_params=grid_search_lasso.best_params_
    )
    rwb_obj=RollingWindowBacktest(clone(grid_search_lasso.best_estimator_), X, y_classification, X_train, WINDOW_SIZE, HORIZON)
    rwb_obj.rolling_window_backtest(verbose=1)
    rwb_obj.display_wfv_results()
    optimized_Log_Reg_lasso_ = grid_search_lasso.best_estimator_
    results=get_final_metrics(optimized_Log_Reg_lasso_, X_train, y_train, X_test, y_test, n_splits=10, label="LASSO Log. Reg.")
    lasso_results = results.copy()
    util_score=utility_score(results, rwb_obj)
    print(f"Utility Score {util_score:.4}")
    if (EXPORT):
        results.update({'utility_score': round(util_score, 3)})
        results=append_params_to_dict(results, optimized_Log_Reg_lasso_)
        results.update(rwb_obj.results[2])
        results.update(download_params)
        log_result(results, cwd / 'output', results_file)

    # ------- Elastic Net (No PCA) -------
    print("\n\n------- Logistic Elastic Net Model -------")
    Log_Reg_Elastic = LogisticRegression(
        solver='saga', class_weight='balanced',
        random_state=1, max_iter=500, tol=LOGISTIC_TOL
    )
    Log_Reg_model_pipeline_Elastic = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', Log_Reg_Elastic)
    ])
    param_grid = {
        'classifier__C': ELASTIC_NET_GRID,
        'classifier__l1_ratio': ELASTIC_NET_L1_RATIO_GRID,
    }
    grid_search_elastic = GridSearchCV(
        Log_Reg_model_pipeline_Elastic, param_grid, cv=tscv, return_train_score=True, verbose=VERBOSE,
        scoring='balanced_accuracy',
        refit=make_one_se_refit(['classifier__C', 'classifier__l1_ratio'])
    )
    grid_search_elastic.fit(X_train, y_train)
    append_search_history(
        history_path=history_path,
        cv_results=grid_search_elastic.cv_results_,
        run_time=run_time,
        model_name="LogReg_ElasticNet",
        search_type="grid",
        grid_version=grid_label,
        notes=SEARCH_NOTES,
        best_params=grid_search_elastic.best_params_
    )
    rwb_obj=RollingWindowBacktest(clone(grid_search_elastic.best_estimator_), X, y_classification, X_train, WINDOW_SIZE, HORIZON)
    rwb_obj.rolling_window_backtest(verbose=1)
    rwb_obj.display_wfv_results()
    optimized_Log_Reg_elastic_ = grid_search_elastic.best_estimator_
    results=get_final_metrics(optimized_Log_Reg_elastic_, X_train, y_train, X_test, y_test, n_splits=10, label="Elastic Net Log. Reg.")
    elastic_results = results.copy()
    util_score=utility_score(results, rwb_obj)
    print(f"Utility Score {util_score:.4}")
    if (EXPORT):
        results.update({'utility_score': round(util_score, 3)})
        results=append_params_to_dict(results, optimized_Log_Reg_elastic_)
        results.update(rwb_obj.results[2])
        results.update(download_params)
        log_result(results, cwd / 'output', results_file)

    # ------- PCA to Ridge(Internal) APPLICATION -------
    Log_Reg_PCA_L=LogisticRegression(l1_ratio=0, solver='liblinear', class_weight='balanced', random_state=1)
    
    Log_Reg_model_pipeline_PCA_L=Pipeline([('scaler', StandardScaler()),
                                           ('pca', PCA()), 
                                           ('classifier', Log_Reg_PCA_L)])

    param_grid={
        'pca__n_components': BASELINE_PCA_GRID,
        'classifier__C': RIDGE_GRID
    }
    grid_search_PCA_ridge=GridSearchCV(
        Log_Reg_model_pipeline_PCA_L, param_grid, cv=tscv, return_train_score=True, verbose=VERBOSE,
        scoring='balanced_accuracy',
        refit=make_one_se_refit(['classifier__C'], fixed_cols=['pca__n_components'])
    )

    grid_search_PCA_ridge.fit(X_train, y_train)
    append_search_history(
        history_path=history_path,
        cv_results=grid_search_PCA_ridge.cv_results_,
        run_time=run_time,
        model_name="LogReg_PCA_Ridge",
        search_type="grid",
        grid_version=grid_label,
        notes=SEARCH_NOTES,
        best_params=grid_search_PCA_ridge.best_params_
    )

    rwb_obj=RollingWindowBacktest(clone(grid_search_PCA_ridge.best_estimator_), X, y_classification, X_train, WINDOW_SIZE, HORIZON)
    rwb_obj.rolling_window_backtest(verbose=1)
    rwb_obj.display_wfv_results()

    optimized_Log_Reg_PCA_ridge_ = grid_search_PCA_ridge.best_estimator_

    results=get_final_metrics(optimized_Log_Reg_PCA_ridge_, X_train, y_train, X_test, y_test, n_splits=10, label="PCA Ridge(int.) Log. Reg.")
    pca_ridge_results = results.copy()
    util_score=utility_score(results, rwb_obj)
    print(f"Utility Score {util_score:.4}")
    if (EXPORT):
        results.update({'utility_score': round(util_score, 3)})
        results=append_params_to_dict(results, optimized_Log_Reg_PCA_ridge_)
        results.update(rwb_obj.results[2])
        results.update(download_params)
        log_result(results, cwd / 'output', results_file)

    # ------- PCA to LASSO(Internal) APPLICATION -------
    Log_Reg_PCA_R=LogisticRegression(l1_ratio=1, solver='saga', class_weight='balanced', random_state=1, max_iter=500, tol=LOGISTIC_TOL)

    Log_Reg_model_pipeline_PCA_R=Pipeline([('scaler', StandardScaler()),
                                           ('pca', PCA()),
                                           ('classifier', Log_Reg_PCA_R)])

    param_grid={
        'pca__n_components': BASELINE_PCA_GRID,
        'classifier__C': LASSO_GRID
    }
    grid_search_PCA_lasso=GridSearchCV(
        Log_Reg_model_pipeline_PCA_R, param_grid, cv=tscv, return_train_score=True, verbose=VERBOSE,
        scoring='balanced_accuracy',
        refit=make_one_se_refit(['classifier__C'], fixed_cols=['pca__n_components'])
    )

    grid_search_PCA_lasso.fit(X_train, y_train)
    append_search_history(
        history_path=history_path,
        cv_results=grid_search_PCA_lasso.cv_results_,
        run_time=run_time,
        model_name="LogReg_PCA_LASSO",
        search_type="grid",
        grid_version=grid_label,
        notes=SEARCH_NOTES,
        best_params=grid_search_PCA_lasso.best_params_
    )

    rwb_obj=RollingWindowBacktest(clone(grid_search_PCA_lasso.best_estimator_), X, y_classification, X_train, WINDOW_SIZE, HORIZON)
    rwb_obj.rolling_window_backtest(verbose=1)
    rwb_obj.display_wfv_results()

    optimized_Log_Reg_PCA_lasso_ = grid_search_PCA_lasso.best_estimator_

    results=get_final_metrics(optimized_Log_Reg_PCA_lasso_, X_train, y_train, X_test, y_test, n_splits=10, label="PCA LASSO(int.) Log. Reg.")
    pca_lasso_results = results.copy()
    util_score=utility_score(results, rwb_obj)
    print(f"Utility Score {util_score:.4}")
    if (EXPORT):
        results.update({'utility_score': round(util_score, 3)})
        results=append_params_to_dict(results, optimized_Log_Reg_PCA_lasso_)
        results.update(rwb_obj.results[2])
        results.update(download_params)
        log_result(results, cwd / 'output', results_file)

    ranking_df = pd.DataFrame([
        {"Model": "PCA Base", **pca_base_results},
        {"Model": "Ridge Log. Reg.", **ridge_results},
        {"Model": "LASSO Log. Reg.", **lasso_results},
        {"Model": "Elastic Net Log. Reg.", **elastic_results},
        {"Model": "PCA Ridge(int.) Log. Reg.", **pca_ridge_results},
        {"Model": "PCA LASSO(int.) Log. Reg.", **pca_lasso_results},
    ])
    ranked_df = rank_models_by_metrics(ranking_df)
    best_model_name = str(ranked_df.iloc[0]["Model"])
    output_dir = cwd / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    if best_model_name == "PCA Base":
        save_best_model_plots_from_gridsearch(
            grid_search_PCA_base,
            "pca__n_components",
            "PCA n_components",
            best_model_name,
            output_dir / f"{output_prefix}_logreg_best_bias_variance.png",
            output_dir / f"{output_prefix}_logreg_best_train_test.png",
            X_train,
            y_train,
            X_test,
            y_test,
        )
    elif best_model_name == "Ridge Log. Reg.":
        save_best_model_plots_from_gridsearch(
            grid_search_ridge,
            "classifier__C",
            "C",
            best_model_name,
            output_dir / f"{output_prefix}_logreg_best_bias_variance.png",
            output_dir / f"{output_prefix}_logreg_best_train_test.png",
            X_train,
            y_train,
            X_test,
            y_test,
        )
    elif best_model_name == "LASSO Log. Reg.":
        save_best_model_plots_from_gridsearch(
            grid_search_lasso,
            "classifier__C",
            "C",
            best_model_name,
            output_dir / f"{output_prefix}_logreg_best_bias_variance.png",
            output_dir / f"{output_prefix}_logreg_best_train_test.png",
            X_train,
            y_train,
            X_test,
            y_test,
        )
    elif best_model_name == "Elastic Net Log. Reg.":
        save_best_model_plots_from_gridsearch(
            grid_search_elastic,
            "classifier__C",
            "C",
            best_model_name,
            output_dir / f"{output_prefix}_logreg_best_bias_variance.png",
            output_dir / f"{output_prefix}_logreg_best_train_test.png",
            X_train,
            y_train,
            X_test,
            y_test,
        )
    elif best_model_name == "PCA Ridge(int.) Log. Reg.":
        save_best_model_plots_from_gridsearch(
            grid_search_PCA_ridge,
            "classifier__C",
            "C",
            best_model_name,
            output_dir / f"{output_prefix}_logreg_best_bias_variance.png",
            output_dir / f"{output_prefix}_logreg_best_train_test.png",
            X_train,
            y_train,
            X_test,
            y_test,
        )
    else:
        save_best_model_plots_from_gridsearch(
            grid_search_PCA_lasso,
            "classifier__C",
            "C",
            best_model_name,
            output_dir / f"{output_prefix}_logreg_best_bias_variance.png",
            output_dir / f"{output_prefix}_logreg_best_train_test.png",
            X_train,
            y_train,
            X_test,
            y_test,
        )
    print(f"\nBest logistic model by average rank: {best_model_name}")

    comparison_df = build_base_style_comparison_df([
        comparison_row_from_metrics("PCA Base", pca_base_results),
        comparison_row_from_metrics("Ridge Log. Reg.", ridge_results),
        comparison_row_from_metrics("LASSO Log. Reg.", lasso_results),
        comparison_row_from_metrics("Elastic Net Log. Reg.", elastic_results),
        comparison_row_from_metrics("PCA Ridge(int.) Log. Reg.", pca_ridge_results),
        comparison_row_from_metrics("PCA LASSO(int.) Log. Reg.", pca_lasso_results),
    ])
    comparison_export_df = comparison_df.rename(
        index={
            "Ridge Log. Reg.": "Ridge",
            "LASSO Log. Reg.": "LASSO",
            "Elastic Net Log. Reg.": "Elastic Net",
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
