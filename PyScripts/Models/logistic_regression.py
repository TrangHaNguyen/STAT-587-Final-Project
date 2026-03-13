from typing import Any, cast
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import clone
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score
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
)
from H_helpers import log_result, append_params_to_dict, get_cwd
from H_search_history import append_search_history, append_search_run, get_git_commit, now_iso
from model_grids import (
    BASELINE_PCA_GRID,
    LASSO_GRID,
    RIDGE_GRID,
)

'''No need for hyperparameter tuning for Logistic Regression via GridSearchCV since LogisticRegressionCV performs internal CV to select the best C value. We will just use the default 10 values of C that LogisticRegressionCV tests.'''

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
def logregcv_to_rows(cv_obj: LogisticRegressionCV, param_name: str = "param_classifier__C") -> dict:
    scores_dict = cv_obj.scores_
    class_key = list(scores_dict.keys())[0]
    scores = np.array(scores_dict[class_key])
    if scores.ndim == 3:
        # Defensive handling for configurations where sklearn stores an l1_ratio axis.
        scores = scores.mean(axis=2)
    elif scores.ndim != 2:
        raise ValueError(f"Unexpected LogisticRegressionCV scores_ shape: {scores.shape}")
    mean_test = scores.mean(axis=0)
    std_test = scores.std(axis=0)
    rank_test = (-mean_test).argsort().argsort() + 1
    return {
        param_name: list(cv_obj.Cs_),
        "mean_train_score": [np.nan] * len(cv_obj.Cs_),
        "mean_test_score": list(mean_test),
        "std_test_score": list(std_test),
        "rank_test_score": list(rank_test)
    }


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


def select_logregcv_c_1se(cv_obj: LogisticRegressionCV) -> float:
    """Select C by 1-SE rule from LogisticRegressionCV scores_."""
    scores_dict = cv_obj.scores_
    class_key = list(scores_dict.keys())[0]
    scores = np.array(scores_dict[class_key])
    if scores.ndim == 3:
        scores = scores.mean(axis=2)
    elif scores.ndim != 2:
        raise ValueError(f"Unexpected LogisticRegressionCV scores_ shape: {scores.shape}")
    mean_test = scores.mean(axis=0)
    std_test = scores.std(axis=0)
    se_test = std_test / np.sqrt(scores.shape[0])
    cs = np.array(cv_obj.Cs_, dtype=float)
    best_idx = int(np.argmax(mean_test))
    threshold = float(mean_test[best_idx] - se_test[best_idx])
    candidate_idx = np.where(mean_test >= threshold)[0]
    if len(candidate_idx) == 0:
        return float(cs[best_idx])
    # Simpler model => smaller C.
    chosen_idx = int(candidate_idx[np.argmin(cs[candidate_idx])])
    return float(cs[chosen_idx])


def _compute_logregcv_bv_curves(X_train, y_train, c_grid, tscv_splitter, l1_ratio, solver):
    cs = np.asarray(c_grid, dtype=float)
    n_splits = tscv_splitter.get_n_splits()
    train_bal_errors = np.zeros((n_splits, len(cs)))
    cv_bal_errors = np.zeros((n_splits, len(cs)))
    for fold_idx, (tr, val) in enumerate(tscv_splitter.split(X_train, y_train)):
        X_fold = X_train.iloc[tr] if hasattr(X_train, 'iloc') else X_train[tr]
        y_fold = y_train.iloc[tr] if hasattr(y_train, 'iloc') else y_train[tr]
        X_val = X_train.iloc[val] if hasattr(X_train, 'iloc') else X_train[val]
        y_val = y_train.iloc[val] if hasattr(y_train, 'iloc') else y_train[val]
        for c_idx, c_val in enumerate(cs):
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(
                    C=float(c_val), l1_ratio=l1_ratio, solver=solver,
                    class_weight='balanced', random_state=1,
                    max_iter=500, tol=1e-2
                ))
            ])
            model.fit(X_fold, y_fold)
            train_preds = model.predict(X_fold)
            cv_preds = model.predict(X_val)
            train_bal_errors[fold_idx, c_idx] = 1 - balanced_accuracy_score(y_fold, train_preds)
            cv_bal_errors[fold_idx, c_idx] = 1 - balanced_accuracy_score(y_val, cv_preds)
    return {
        'cs': cs,
        'train_bal_err_mean': train_bal_errors.mean(axis=0),
        'train_bal_err_std': train_bal_errors.std(axis=0),
        'cv_bal_err_mean': cv_bal_errors.mean(axis=0),
        'cv_bal_err_std': cv_bal_errors.std(axis=0),
    }


def _compute_logreg_direct_errors(X_train, y_train, X_test, y_test, c_grid, l1_ratio, solver):
    train_errors, test_errors = [], []
    for c_val in c_grid:
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                C=float(c_val), l1_ratio=l1_ratio, solver=solver,
                class_weight='balanced', random_state=1,
                max_iter=500, tol=1e-2
            ))
        ])
        model.fit(X_train, y_train)
        train_errors.append(1 - model.score(X_train, y_train))
        test_errors.append(1 - model.score(X_test, y_test))
    return {
        'cs': np.asarray(c_grid, dtype=float),
        'train_errors': np.asarray(train_errors, dtype=float),
        'test_errors': np.asarray(test_errors, dtype=float),
    }


def _compute_single_logreg_direct_error(X_train, y_train, X_test, y_test, c_val, l1_ratio, solver):
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            C=float(c_val), l1_ratio=l1_ratio, solver=solver,
            class_weight='balanced', random_state=1,
            max_iter=500, tol=1e-2
        ))
    ])
    model.fit(X_train, y_train)
    return 1 - model.score(X_train, y_train), 1 - model.score(X_test, y_test)


def _save_best_logregcv_plots(
    X_train,
    y_train,
    X_test,
    y_test,
    c_grid,
    one_se_c,
    model_title,
    l1_ratio,
    output_bv,
    output_direct,
):
    solver = 'saga'
    curves = _compute_logregcv_bv_curves(X_train, y_train, c_grid, TimeSeriesSplit(n_splits=5), l1_ratio, solver)
    direct = _compute_logreg_direct_errors(X_train, y_train, X_test, y_test, c_grid, l1_ratio, solver)
    best_idx = int(np.argmin(curves['cv_bal_err_mean']))
    selected_c = float(curves['cs'][best_idx])
    _, best_test_error = _compute_single_logreg_direct_error(X_train, y_train, X_test, y_test, selected_c, l1_ratio, solver)
    selected_idx = int(np.argmin(np.abs(curves['cs'] - float(one_se_c))))

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(f'Bias-Variance Tradeoff — {model_title}', fontsize=13, fontweight='bold')
    ax.semilogx(curves['cs'], curves['train_bal_err_mean'], marker='o', color='steelblue', linewidth=1.8, label='CV Train balanced error')
    ax.semilogx(curves['cs'], curves['cv_bal_err_mean'], marker='s', color='darkorange', linewidth=1.8, label='CV Test balanced error')
    ax.fill_between(
        curves['cs'],
        np.clip(curves['train_bal_err_mean'] - curves['train_bal_err_std'], 0.0, 1.0),
        np.clip(curves['train_bal_err_mean'] + curves['train_bal_err_std'], 0.0, 1.0),
        alpha=0.15, color='steelblue', label='CV Train balanced error ±1 SD'
    )
    ax.fill_between(
        curves['cs'],
        np.clip(curves['cv_bal_err_mean'] - curves['cv_bal_err_std'], 0.0, 1.0),
        np.clip(curves['cv_bal_err_mean'] + curves['cv_bal_err_std'], 0.0, 1.0),
        alpha=0.15, color='darkorange', label='CV Test balanced error ±1 SD'
    )
    ax.scatter([curves['cs'][best_idx]], [curves['cv_bal_err_mean'][best_idx]], color='gold', edgecolor='black', s=90, zorder=6, label='Value at best CV balanced error')
    ax.axvline(float(curves['cs'][selected_idx]), color='red', linestyle='--', linewidth=1.5, label=f'1SE-selected C = {float(one_se_c):.4g}')
    ax.set_title(f'{model_title} — Bias-Variance Tradeoff')
    ax.set_xlabel('C')
    ax.set_ylabel('Balanced Error (1 - balanced accuracy)')
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_bv, dpi=150, bbox_inches='tight')
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    fig2.suptitle(f'Over/Underfitting Analysis — {model_title}', fontsize=13, fontweight='bold')
    ax2.semilogx(direct['cs'], direct['train_errors'], marker='o', color='steelblue', linewidth=2, label='Train error')
    ax2.semilogx(direct['cs'], direct['test_errors'], marker='s', color='darkorange', linewidth=2, label='Test error')
    ax2.scatter([selected_c], [best_test_error], color='gold', edgecolor='black', s=90, zorder=6, label='Value at best CV balanced error')
    ax2.axvline(float(curves['cs'][selected_idx]), color='red', linestyle='--', linewidth=1.5, label=f'1SE-selected C = {float(one_se_c):.4g}')
    ax2.set_title(f'{model_title} — Train vs Test Error')
    ax2.set_xlabel('C')
    ax2.set_ylabel('Plain Error (1 - accuracy)')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_direct, dpi=150, bbox_inches='tight')
    plt.close()


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
        "sector": True,
        "corr_threshold": 0.95,
        "corr_level": 0
    }

    if (FIND_OPTIMAL):
        # ------- Selection of Remaining data_clean() Parameters -------
        base_Log_Reg_model=LogisticRegression(C=1.0, l1_ratio=0, solver='saga', class_weight='balanced', random_state=1, max_iter=1000, tol=1e-3, verbose=VERBOSE)
        base_Log_Reg_model_pipeline=Pipeline([('scaler', StandardScaler()), ('classifier', base_Log_Reg_model)])

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

        _, parameters_, best_score=data_clean_param_selection(*DATA, clone(base_Log_Reg_model_pipeline), TEST_SIZE, WINDOW_SIZE, HORIZON, **param_grid)
        print(f"Best Utility Score {best_score}")
        print(f"Optimal parameter {parameters_}")

    download_params = {f"clean_data__{k}=": v for k, v in parameters_.items()}

    X, y_regression=cast(Any, clean_data(*DATA, **parameters_))
    def to_binary_class(y):
        return (y>=0).astype(int)
    y_classification=to_binary_class(y_regression)
    X_train, X_test, y_train, y_test=train_test_split(X, y_classification, test_size=TEST_SIZE, random_state=1, shuffle=False)


    # ------- LASSO(Internal) APPLICATION -------
    Log_Reg_R=LogisticRegressionCV(Cs=LASSO_GRID, cv=tscv, l1_ratios=[1], solver='saga', class_weight='balanced', random_state=1, n_jobs=MODEL_N_JOBS, max_iter=500, tol=1e-2, verbose=VERBOSE, scoring='balanced_accuracy')
    
    Log_Reg_model_pipeline_R=Pipeline([('scaler', StandardScaler()), ('classifier', Log_Reg_R)])

    Log_Reg_model_pipeline_R.fit(X_train, y_train)
    append_search_history(
        history_path=history_path,
        cv_results=logregcv_to_rows(Log_Reg_model_pipeline_R.named_steps['classifier']),
        run_time=run_time,
        model_name="LogReg_LASSO_internal",
        search_type="grid",
        grid_version=grid_label,
        notes=SEARCH_NOTES,
        best_params={"classifier__C": select_logregcv_c_1se(Log_Reg_model_pipeline_R.named_steps['classifier'])}
    )

    best_c = select_logregcv_c_1se(Log_Reg_model_pipeline_R.named_steps['classifier'])
    Opt_Log_Reg_R=LogisticRegression(
        C=best_c, l1_ratio=1, solver='saga', class_weight='balanced',
        random_state=1, max_iter=500, tol=1e-2
    )

    Opt_Log_Reg_model_pipeline_R=Pipeline([('scaler', StandardScaler()), ('classifier', Opt_Log_Reg_R)])

    rwb_obj=RollingWindowBacktest(clone(Opt_Log_Reg_model_pipeline_R), X, y_classification, X_train, WINDOW_SIZE, HORIZON)
    rwb_obj.rolling_window_backtest(verbose=1)
    rwb_obj.display_wfv_results()

    optimized_Log_Reg_R_=clone(Opt_Log_Reg_model_pipeline_R)
    optimized_Log_Reg_R_.fit(X_train, y_train)

    results=get_final_metrics(optimized_Log_Reg_R_, X_train, y_train, X_test, y_test, n_splits=10, label="LASSO(int.) Log. Reg.")
    lasso_results = results.copy()
    util_score=utility_score(results, rwb_obj)
    print(f"Utility Score {util_score:.4}")
    if (EXPORT):
        results.update({'utility_score': round(util_score, 3)})
        results=append_params_to_dict(results, optimized_Log_Reg_R_)
        results.update(rwb_obj.results[2])
        results.update(download_params)
        log_result(results, cwd / 'output', results_file)

    # ------- RIDGE(Internal) APPLICATION -------
    Log_Reg_L=LogisticRegressionCV(Cs=RIDGE_GRID, cv=tscv, l1_ratios=[0], solver='saga', class_weight='balanced', random_state=1, n_jobs=MODEL_N_JOBS, max_iter=500, tol=1e-2, verbose=VERBOSE, scoring='balanced_accuracy')
    
    Log_Reg_model_pipeline_L=Pipeline([('scaler', StandardScaler()), ('classifier', Log_Reg_L)])

    Log_Reg_model_pipeline_L.fit(X_train, y_train)
    append_search_history(
        history_path=history_path,
        cv_results=logregcv_to_rows(Log_Reg_model_pipeline_L.named_steps['classifier']),
        run_time=run_time,
        model_name="LogReg_Ridge_internal",
        search_type="grid",
        grid_version=grid_label,
        notes=SEARCH_NOTES,
        best_params={"classifier__C": select_logregcv_c_1se(Log_Reg_model_pipeline_L.named_steps['classifier'])}
    )

    best_c = select_logregcv_c_1se(Log_Reg_model_pipeline_L.named_steps['classifier'])
    Opt_Log_Reg_L=LogisticRegression(
        C=best_c, l1_ratio=0, solver='saga', class_weight='balanced',
        random_state=1, max_iter=500, tol=1e-2
    )

    Opt_Log_Reg_model_pipeline_L=Pipeline([('scaler', StandardScaler()), ('classifier', Opt_Log_Reg_L)])

    rwb_obj=RollingWindowBacktest(clone(Opt_Log_Reg_model_pipeline_L), X, y_classification, X_train, WINDOW_SIZE, HORIZON)
    rwb_obj.rolling_window_backtest(verbose=1)
    rwb_obj.display_wfv_results()

    optimized_Log_Reg_L_=clone(Opt_Log_Reg_model_pipeline_L)
    optimized_Log_Reg_L_.fit(X_train, y_train)

    results=get_final_metrics(optimized_Log_Reg_L_, X_train, y_train, X_test, y_test, n_splits=10, label="Ridge(int.) Log. Reg.")
    ridge_results = results.copy()
    util_score=utility_score(results, rwb_obj)
    print(f"Utility Score {util_score:.4}")
    if (EXPORT):
        results.update({'utility_score': round(util_score, 3)})
        results=append_params_to_dict(results, optimized_Log_Reg_L_)
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

    optimized_Log_Reg_PCA_ridge_=clone(grid_search_PCA_ridge.best_estimator_)
    optimized_Log_Reg_PCA_ridge_.fit(X_train, y_train)

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
    Log_Reg_PCA_R=LogisticRegression(l1_ratio=1, solver='saga', class_weight='balanced', random_state=1, max_iter=500, tol=1e-2)

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

    optimized_Log_Reg_PCA_lasso_=clone(grid_search_PCA_lasso.best_estimator_)
    optimized_Log_Reg_PCA_lasso_.fit(X_train, y_train)

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
        {"Model": "LASSO(int.) Log. Reg.", **lasso_results},
        {"Model": "Ridge(int.) Log. Reg.", **ridge_results},
        {"Model": "PCA Ridge(int.) Log. Reg.", **pca_ridge_results},
        {"Model": "PCA LASSO(int.) Log. Reg.", **pca_lasso_results},
    ])
    ranked_df = rank_models_by_metrics(ranking_df)
    best_model_name = str(ranked_df.iloc[0]["Model"])
    output_dir = cwd / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    if best_model_name == "LASSO(int.) Log. Reg.":
        _save_best_logregcv_plots(
            X_train, y_train, X_test, y_test,
            LASSO_GRID,
            select_logregcv_c_1se(Log_Reg_model_pipeline_R.named_steps['classifier']),
            best_model_name,
            1,
            output_dir / f"{output_prefix}_logreg_best_bias_variance.png",
            output_dir / f"{output_prefix}_logreg_best_train_test.png",
        )
    elif best_model_name == "Ridge(int.) Log. Reg.":
        _save_best_logregcv_plots(
            X_train, y_train, X_test, y_test,
            RIDGE_GRID,
            select_logregcv_c_1se(Log_Reg_model_pipeline_L.named_steps['classifier']),
            best_model_name,
            0,
            output_dir / f"{output_prefix}_logreg_best_bias_variance.png",
            output_dir / f"{output_prefix}_logreg_best_train_test.png",
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
