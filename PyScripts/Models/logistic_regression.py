from typing import Any, cast
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import clone
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
import os
import pandas as pd
import numpy as np
import time

from H_prep import clean_data, data_clean_param_selection, import_data
from H_eval import get_final_metrics, RollingWindowBacktest, utility_score
from H_helpers import log_result, append_params_to_dict, get_cwd
from H_search_history import append_search_history, append_search_run, get_git_commit, now_iso
from model_grids import LOGREG_PCA_GRID_OPTIONS

'''No need for hyperparameter tuning for Logistic Regression via GridSearchCV since LogisticRegressionCV performs internal CV to select the best C value. We will just use the default 10 values of C that LogisticRegressionCV tests.'''

VERBOSE=0
MODEL_N_JOBS=int(os.getenv("MODEL_N_JOBS", "-1"))
PAUSE_BETWEEN_MODELS=(os.getenv("PAUSE_BETWEEN_MODELS", "0") == "1")
GRID_VARIANT=os.getenv("GRID_VARIANT", "center").lower()
GRID_VERSION=os.getenv("GRID_VERSION", "v1")
SEARCH_NOTES=os.getenv("SEARCH_NOTES", "")

cwd=get_cwd("STAT-587-Final-Project")


def choose_grid(left_values, center_values, right_values):
    options={"left": left_values, "center": center_values, "right": right_values}
    if (GRID_VARIANT not in options):
        raise ValueError("GRID_VARIANT must be one of: left, center, right.")
    return options[GRID_VARIANT]


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


def make_one_se_refit(complexity_cols: list[str]):
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

if __name__=="__main__":
    run_start = time.time()
    run_time = now_iso()
    WINDOW_SIZE=200
    HORIZON=40
    EXPORT=True
    TEST_SIZE=0.2
    # Previous temporary change used `KFold(n_splits=5, shuffle=False)`.
    tscv = TimeSeriesSplit(n_splits=5)
    custom_Cs=choose_grid(
        [0.005, 0.01, 0.1, 1.0],
        [0.05, 0.1, 1.0, 10.0],
        [0.1, 1.0, 10.0, 100.0]
    )
    print(f"MODEL_N_JOBS={MODEL_N_JOBS} (set env MODEL_N_JOBS to override)")
    print(f"GRID_VARIANT={GRID_VARIANT} (left/center/right)")
    print(f"GRID_VERSION={GRID_VERSION}")
    grid_label = f"{GRID_VERSION}_{GRID_VARIANT}"
    history_path = cwd / "output" / "8yrs_search_history_logreg.csv"
    runs_path = cwd / "output" / "8yrs_search_runs.csv"
    dataset_version = "testing=False,extra_features=True,cluster=False,corr_threshold=0.95,corr_level=0"
    # testing: bool =False, extra_features: bool =True, cluster: bool =False, n_clusters: int =100, corr_threshold: float =0.95, corr_level: int =0
    DATA=import_data(extra_features=True, testing=False, cluster=False, n_clusters=100, corr_threshold=0.95, corr_level=0)

    FIND_OPTIMAL=False
    
    parameters_={  # These are optimal as of 3/8/2026 4:00 PM w=4
        "raw": False,
        "extra_features": True,
        "lag_period": 2,
        "lookback_period": 7,
        "sector": True,
        "corr_threshold": 0.8,
        "corr_level": 2
    }

    if (FIND_OPTIMAL):
        # ------- Selection of Optimal lag_period and lookback_period Parameters -------
        base_Log_Reg_model=LogisticRegression(C=1.0, l1_ratio=0, solver='saga', class_weight='balanced', random_state=1, max_iter=1000, tol=1e-3, verbose=VERBOSE)
        base_Log_Reg_model_pipeline=Pipeline([('scaler', StandardScaler()), ('classifier', base_Log_Reg_model)])
        
        print("------- Finding Optimal lag_period Value")
        param_grid={
            'lag_period': choose_grid(
                [1, 2, [1, 2], [1, 2, 3]],
                [1, 2, 3, 4, 5, [1, 2], [1, 2, 3], [2, 3], [1, 3]],
                [3, 4, 5, 6, 7, [2, 3], [3, 4], [2, 3, 4], [3, 5]]
            ),
            'sector': [True],
            'corr_level': [2]
        }

        _, best_parameters, best_score=data_clean_param_selection(*DATA, clone(base_Log_Reg_model_pipeline), TEST_SIZE, WINDOW_SIZE, HORIZON, eff_support=True, **param_grid)
        best_lag=best_parameters['lag_period']
        print(f"Best Utility Score (lag_period): {best_score}")
        print(f"Best lag_period: {best_lag}")

        print("------- Finding Optimal lookback_period Value")
        param_grid={
            'lookback_period': choose_grid(
                [5, 7, 10, 12, 14, 17, 21],
                [7, 10, 14, 17, 21, 24, 28],
                [14, 17, 21, 24, 28, 32, 36]
            ),
            'sector': [True],
            'corr_level': [2]
        }
        
        _, best_parameters, best_score=data_clean_param_selection(*DATA, clone(base_Log_Reg_model_pipeline), TEST_SIZE, WINDOW_SIZE, HORIZON, eff_support=True, **param_grid)
        best_lookback=best_parameters['lookback_period']
        print(f"Best Utility Score (lookback_period): {best_score}")
        print(f"Best lookback_period: {best_lookback}")

        # ------- Selection of Optimal data_clean() Parameters -------
        print("------- Finding Optimal data_clean() Parameters")
        param_grid={
            'raw': [True, False],
            'extra_features': [True, False],
            'lag_period': [best_lag],
            'lookback_period': [best_lookback],
            'sector': [True],
            'corr_level': [0, 1, 2, 3],
            'corr_threshold': choose_grid(
                [0.7, 0.8, 0.85],
                [0.8, 0.9, 0.95],
                [0.9, 0.95, 0.98]
            )
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
    Log_Reg_R=LogisticRegressionCV(Cs=custom_Cs, cv=tscv, l1_ratios=[1], solver='saga', class_weight='balanced', random_state=1, n_jobs=MODEL_N_JOBS, max_iter=500, tol=1e-2, verbose=VERBOSE, scoring='balanced_accuracy')
    
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
    Opt_Log_Reg_R=LogisticRegression(C=best_c, l1_ratio=1, solver='saga', random_state=1, max_iter=500, tol=1e-2)

    Opt_Log_Reg_model_pipeline_R=Pipeline([('scaler', StandardScaler()), ('classifier', Opt_Log_Reg_R)])

    rwb_obj=RollingWindowBacktest(clone(Opt_Log_Reg_model_pipeline_R), X, y_classification, X_train, WINDOW_SIZE, HORIZON)
    rwb_obj.rolling_window_backtest(verbose=1)
    rwb_obj.display_wfv_results()

    optimized_Log_Reg_R_=clone(Opt_Log_Reg_model_pipeline_R)
    optimized_Log_Reg_R_.fit(X_train, y_train)

    results=get_final_metrics(optimized_Log_Reg_R_, X_train, y_train, X_test, y_test, n_splits=10, label="LASSO(int.) Log. Reg.")
    util_score=utility_score(results, rwb_obj)
    print(f"Utility Score {util_score:.4}")
    if (EXPORT):
        results.update({'utility_score': round(util_score, 3)})
        results=append_params_to_dict(results, optimized_Log_Reg_R_)
        results.update(rwb_obj.results[2])
        results.update(download_params)
        log_result(results, cwd / 'output', "8yrs_results.csv")

    if (PAUSE_BETWEEN_MODELS):
        input("Press Enter to continue...")

    # ------- RIDGE(Internal) APPLICATION -------
    Log_Reg_L=LogisticRegressionCV(Cs=custom_Cs, cv=tscv, l1_ratios=[0], solver='saga', class_weight='balanced', random_state=1, n_jobs=MODEL_N_JOBS, max_iter=500, tol=1e-2, verbose=VERBOSE, scoring='balanced_accuracy')
    
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
    Opt_Log_Reg_L=LogisticRegression(C=best_c, l1_ratio=0, solver='saga', random_state=1, max_iter=500, tol=1e-2)

    Opt_Log_Reg_model_pipeline_L=Pipeline([('scaler', StandardScaler()), ('classifier', Opt_Log_Reg_L)])

    rwb_obj=RollingWindowBacktest(clone(Opt_Log_Reg_model_pipeline_L), X, y_classification, X_train, WINDOW_SIZE, HORIZON)
    rwb_obj.rolling_window_backtest(verbose=1)
    rwb_obj.display_wfv_results()

    optimized_Log_Reg_L_=clone(Opt_Log_Reg_model_pipeline_L)
    optimized_Log_Reg_L_.fit(X_train, y_train)

    results=get_final_metrics(optimized_Log_Reg_L_, X_train, y_train, X_test, y_test, n_splits=10, label="Ridge(int.) Log. Reg.")
    util_score=utility_score(results, rwb_obj)
    print(f"Utility Score {util_score:.4}")
    if (EXPORT):
        results.update({'utility_score': round(util_score, 3)})
        results=append_params_to_dict(results, optimized_Log_Reg_L_)
        results.update(rwb_obj.results[2])
        results.update(download_params)
        log_result(results, cwd / 'output', "8yrs_results.csv")

    if (PAUSE_BETWEEN_MODELS):
        input("Press Enter to continue...")

    # ------- PCA to Ridge(Internal) APPLICATION -------
    Log_Reg_PCA_L=LogisticRegression(l1_ratio=0, solver='liblinear', class_weight='balanced', random_state=1)
    
    Log_Reg_model_pipeline_PCA_L=Pipeline([('scaler', StandardScaler()),
                                           ('pca', PCA()), 
                                           ('classifier', Log_Reg_PCA_L)])

    param_grid={
        'pca__n_components': choose_grid(
            *LOGREG_PCA_GRID_OPTIONS
        ),
        'classifier__C': choose_grid(
            [0.001, 0.01, 0.1, 1.0],
            [0.01, 0.1, 1.0, 10.0],
            [0.1, 1.0, 10.0, 100.0]
        )
    }
    grid_search_PCA_ridge=GridSearchCV(
        Log_Reg_model_pipeline_PCA_L, param_grid, cv=tscv, return_train_score=True, verbose=VERBOSE,
        scoring='balanced_accuracy',
        refit=make_one_se_refit(['pca__n_components', 'classifier__C'])
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
    util_score=utility_score(results, rwb_obj)
    print(f"Utility Score {util_score:.4}")
    if (EXPORT):
        results.update({'utility_score': round(util_score, 3)})
        results=append_params_to_dict(results, optimized_Log_Reg_PCA_ridge_)
        results.update(rwb_obj.results[2])
        results.update(download_params)
        log_result(results, cwd / 'output', "8yrs_results.csv")

    if (PAUSE_BETWEEN_MODELS):
        input("Press Enter to Finish...")
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
