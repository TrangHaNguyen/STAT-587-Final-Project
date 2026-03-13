from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.svm import SVC
from sklearn.base import clone
from typing import Any, cast
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import os
import time

from H_prep import clean_data, import_data, data_clean_param_selection
from H_eval import RollingWindowBacktest, get_final_metrics, utility_score
from H_helpers import log_result, get_cwd, append_params_to_dict
from H_search_history import append_search_history, append_search_run, get_git_commit, now_iso
from model_grids import (
    SVM_LINEAR_C_GRID_OPTIONS,
    SVM_GAMMA_GRID_OPTIONS,
    SVM_DEGREE_GRID_OPTIONS,
    DATA_CLEAN_LAG_GRID_OPTIONS,
    DATA_CLEAN_LOOKBACK_GRID_OPTIONS,
    DATA_CLEAN_CORR_THRESHOLD_OPTIONS,
    choose_grid_variant,
)

cwd=get_cwd("STAT-587-Final-Project")
MODEL_N_JOBS=int(os.getenv("MODEL_N_JOBS", "-1"))
PAUSE_BETWEEN_MODELS=(os.getenv("PAUSE_BETWEEN_MODELS", "0") == "1")
GRID_VARIANT=os.getenv("GRID_VARIANT", "center").lower()
GRID_VERSION=os.getenv("GRID_VERSION", "v1")
SEARCH_NOTES=os.getenv("SEARCH_NOTES", "")


def choose_grid(left_values, center_values, right_values):
    return choose_grid_variant(GRID_VARIANT, left_values, center_values, right_values)


def _as_sortable_numeric(value):
    try:
        return float(value)
    except Exception:
        return float("inf")


def make_one_se_refit(complexity_cols: list[str]):
    """Return a GridSearchCV refit callable implementing the 1-SE rule."""
    import numpy as np
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


if __name__ == "__main__":
    run_start = time.time()
    run_time = now_iso()
    WINDOW_SIZE=200
    HORIZON=40
    EXPORT=True
    TEST_SIZE=0.2
    # testing: bool =False, extra_features: bool =True, cluster: bool =False, n_clusters: int =100, corr_threshold: float =0.95, corr_level: int =0
    DATA=import_data(extra_features=True, testing=False, cluster=False, n_clusters=100, corr_threshold=0.95, corr_level=0)
    # Keep feature-engineering configuration fixed for consistency across models.
    FIND_OPTIMAL=False
    print(f"MODEL_N_JOBS={MODEL_N_JOBS} (set env MODEL_N_JOBS to override)")
    print(f"GRID_VARIANT={GRID_VARIANT} (left/center/right)")
    print(f"GRID_VERSION={GRID_VERSION}")
    grid_label = f"{GRID_VERSION}_{GRID_VARIANT}"
    history_path = cwd / "output" / "8yrs_search_history_svm.csv"
    runs_path = cwd / "output" / "8yrs_search_runs.csv"
    dataset_version = "testing=False,extra_features=True,cluster=False,corr_threshold=0.95,corr_level=0"
    
    parameters_={
        "raw": False,
        "extra_features": True,
        "lag_period": 2,
        "lookback_period": 7,
        "sector": True,
        "corr_threshold": 0.8,
        "corr_level": 2,
    }

    if (FIND_OPTIMAL):
        # ------- Selection of Optimal lag_period and lookback_period Parameters -------
        base_SVM_rbf_model=SVC(kernel="rbf", cache_size=1000, class_weight='balanced', gamma='scale', random_state=1, tol=5e-2)
        base_SVM_rbf_model_pipeline=Pipeline([('scaler', StandardScaler()), ('classifier', base_SVM_rbf_model)])
        
        print("------- Finding Optimal lag_period Value")
        param_grid={
            'lag_period': choose_grid(*DATA_CLEAN_LAG_GRID_OPTIONS),
            'sector': [True],
            'corr_level': [2]
        }

        _, best_parameters, best_score=data_clean_param_selection(*DATA, clone(base_SVM_rbf_model_pipeline), TEST_SIZE, WINDOW_SIZE, HORIZON, eff_support=True, **param_grid)
        best_lag=best_parameters['lag_period']
        print(f"Best Utility Score (lag_period): {best_score}")
        print(f"Best lag_period: {best_lag}")

        print("------- Finding Optimal lookback_period Value")
        param_grid={
            'lookback_period': choose_grid(*DATA_CLEAN_LOOKBACK_GRID_OPTIONS),
            'sector': [True],
            'corr_level': [2]
        }
        
        _, best_parameters, best_score=data_clean_param_selection(*DATA, clone(base_SVM_rbf_model_pipeline), TEST_SIZE, WINDOW_SIZE, HORIZON, eff_support=True, **param_grid)
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
            'corr_threshold': choose_grid(*DATA_CLEAN_CORR_THRESHOLD_OPTIONS)
        }

        _, parameters_, best_score=data_clean_param_selection(*DATA, clone(base_SVM_rbf_model_pipeline), TEST_SIZE, WINDOW_SIZE, HORIZON, **param_grid)
        print(f"Best Utility Score {best_score}")
        print(f"Optimal parameter {parameters_}")

    download_params = {f"clean_data__{k}=": v for k, v in parameters_.items()}

    X, y_regression=cast(Any, clean_data(*DATA, **parameters_))
    def to_binary_class(y):
        return (y>=0).astype(int)
    y_classification=to_binary_class(y_regression)
    X_train, X_test, y_train, y_test=train_test_split(X, y_classification, test_size=TEST_SIZE, random_state=1, shuffle=False)

    # Previous temporary change used `KFold(n_splits=5, shuffle=False)`.
    tscv = TimeSeriesSplit(n_splits=5)
    # ------- Linear SVM -------
    print("\n\n------- Linear SVM Model -------")
    SVM_linear=SVC(kernel="linear", cache_size=1000, class_weight='balanced', gamma='scale', random_state=1, tol=5e-2)

    SVM_linear_pipeline = Pipeline([('scaler', StandardScaler()),
                                    ('classifier', SVM_linear)])

    param_grid={
        'classifier__C': choose_grid(*SVM_LINEAR_C_GRID_OPTIONS)
    }
    
    grid_search_linear = GridSearchCV(
        SVM_linear_pipeline, param_grid, cv=tscv, scoring='balanced_accuracy',
        n_jobs=MODEL_N_JOBS, verbose=1, return_train_score=True,
        refit=make_one_se_refit(['classifier__C'])
    )
    grid_search_linear.fit(X_train, y_train)
    append_search_history(
        history_path=history_path,
        cv_results=grid_search_linear.cv_results_,
        run_time=run_time,
        model_name="SVM_linear",
        search_type="grid",
        grid_version=grid_label,
        notes=SEARCH_NOTES,
        best_params=grid_search_linear.best_params_
    )

    rwb_obj=RollingWindowBacktest(clone(grid_search_linear.best_estimator_), X, y_classification, X_train, WINDOW_SIZE, HORIZON)
    rwb_obj.rolling_window_backtest(verbose=1)
    rwb_obj.display_wfv_results()

    optimized_linear_=clone(grid_search_linear.best_estimator_)
    optimized_linear_.fit(X_train, y_train)

    results=get_final_metrics(optimized_linear_, X_train, y_train, X_test, y_test, label="Linear Ker. SVM")
    util_score=utility_score(results, rwb_obj)
    print(f"Utility Score {util_score:.4}")
    if (EXPORT):
        results.update({'utility_score': round(util_score, 3)})
        results=append_params_to_dict(results, grid_search_linear.best_estimator_)
        results.update(rwb_obj.results[2])
        results.update(download_params)
        log_result(results, cwd / 'output', "8yrs_results.csv")

    if (PAUSE_BETWEEN_MODELS):
        input("Press Enter to continue...")

    # ------- RBF SVM -------
    print("\n\n------- RBF SVM Model -------")
    SVM_rbf=SVC(kernel="rbf", cache_size=1000, class_weight='balanced', gamma='scale', random_state=1, tol=5e-2)

    SVM_rbf_pipeline = Pipeline([('scaler', StandardScaler()),
                                 ('classifier', SVM_rbf)])

    param_grid={
        'classifier__C': choose_grid(*SVM_LINEAR_C_GRID_OPTIONS),
        'classifier__gamma': choose_grid(*SVM_GAMMA_GRID_OPTIONS)
    }
    
    grid_search_rbf = GridSearchCV(
        SVM_rbf_pipeline, param_grid, cv=tscv, scoring='balanced_accuracy',
        n_jobs=MODEL_N_JOBS, verbose=1, return_train_score=True,
        refit=make_one_se_refit(['classifier__C', 'classifier__gamma'])
    )
    grid_search_rbf.fit(X_train, y_train)
    append_search_history(
        history_path=history_path,
        cv_results=grid_search_rbf.cv_results_,
        run_time=run_time,
        model_name="SVM_rbf",
        search_type="grid",
        grid_version=grid_label,
        notes=SEARCH_NOTES,
        best_params=grid_search_rbf.best_params_
    )

    rwb_obj=RollingWindowBacktest(clone(grid_search_rbf.best_estimator_), X, y_classification, X_train, WINDOW_SIZE, HORIZON)
    rwb_obj.rolling_window_backtest(verbose=1)
    rwb_obj.display_wfv_results()

    optimized_rbf_=clone(grid_search_rbf.best_estimator_)
    optimized_rbf_.fit(X_train, y_train)

    results=get_final_metrics(optimized_rbf_, X_train, y_train, X_test, y_test, label="RBF Ker. SVM")
    util_score=utility_score(results, rwb_obj)
    print(f"Utility Score {util_score:.4}")
    if (EXPORT):
        results.update({'utility_score': round(util_score, 3)})
        results=append_params_to_dict(results, grid_search_rbf.best_estimator_)
        results.update(rwb_obj.results[2])
        results.update(download_params)
        log_result(results, cwd / 'output', "8yrs_results.csv")

    if (PAUSE_BETWEEN_MODELS):
        input("Press Enter to continue...")

    # ------- Polynomial SVM -------
    print("\n\n------- Polynomial SVM Model -------")
    SVM_poly=SVC(kernel="poly", cache_size=1000, class_weight='balanced', gamma='scale', random_state=1, tol=5e-2)

    SVM_poly_pipeline = Pipeline([('scaler', StandardScaler()),
                                  ('classifier', SVM_poly)])

    param_grid={
        'classifier__C': choose_grid(*SVM_LINEAR_C_GRID_OPTIONS),
        'classifier__gamma': choose_grid(*SVM_GAMMA_GRID_OPTIONS),
        'classifier__degree': choose_grid(*SVM_DEGREE_GRID_OPTIONS)
    }
    
    grid_search_poly = GridSearchCV(
        SVM_poly_pipeline, param_grid, cv=tscv, scoring='balanced_accuracy',
        n_jobs=MODEL_N_JOBS, verbose=1, return_train_score=True,
        refit=make_one_se_refit(['classifier__C', 'classifier__degree', 'classifier__gamma'])
    )
    grid_search_poly.fit(X_train, y_train)
    append_search_history(
        history_path=history_path,
        cv_results=grid_search_poly.cv_results_,
        run_time=run_time,
        model_name="SVM_poly",
        search_type="grid",
        grid_version=grid_label,
        notes=SEARCH_NOTES,
        best_params=grid_search_poly.best_params_
    )

    rwb_obj=RollingWindowBacktest(clone(grid_search_poly.best_estimator_), X, y_classification, X_train, WINDOW_SIZE, HORIZON)
    rwb_obj.rolling_window_backtest(verbose=1)
    rwb_obj.display_wfv_results()

    optimized_poly_=clone(grid_search_poly.best_estimator_)
    optimized_poly_.fit(X_train, y_train)

    results=get_final_metrics(optimized_poly_, X_train, y_train, X_test, y_test, label="Poly. Ker. SVM")
    util_score=utility_score(results, rwb_obj)
    print(f"Utility Score {util_score:.4}")
    if (EXPORT):
        results.update({'utility_score': round(util_score, 3)})
        results=append_params_to_dict(results, grid_search_poly.best_estimator_)
        results.update(rwb_obj.results[2])
        results.update(download_params)
        log_result(results, cwd / 'output', "8yrs_results.csv")

    if (PAUSE_BETWEEN_MODELS):
        input("Press Enter to finish...")
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
