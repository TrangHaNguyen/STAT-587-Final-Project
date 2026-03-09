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
import time

from H_reduce import step_wise_reg_wfv
from H_prep import clean_data, data_clean_param_selection, import_data
from H_eval import RollingWindowBacktest, get_final_metrics, utility_score
from H_helpers import log_result, get_cwd, append_params_to_dict
from H_search_history import append_search_history, append_search_run, get_git_commit, now_iso

VERBOSE=0
WINDOW_SIZE=220
HORIZON=40
EXPORT=True
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


if __name__=="__main__":
    run_start = time.time()
    run_time = now_iso()
    WINDOW_SIZE=200
    HORIZON=40
    EXPORT=True
    TEST_SIZE=0.2
    print(f"MODEL_N_JOBS={MODEL_N_JOBS} (set env MODEL_N_JOBS to override)")
    print(f"GRID_VARIANT={GRID_VARIANT} (left/center/right)")
    print(f"GRID_VERSION={GRID_VERSION}")
    grid_label = f"{GRID_VERSION}_{GRID_VARIANT}"
    history_path = cwd / "output" / "results" / "search_history_rf.csv"
    runs_path = cwd / "output" / "results" / "search_runs.csv"
    dataset_version = "testing=False,extra_features=True,cluster=False,corr_threshold=0.95,corr_level=0"
    # testing: bool =False, extra_features: bool =True, cluster: bool =False, n_clusters: int =100, corr_threshold: float =0.95, corr_level: int =0
    DATA=import_data(extra_features=True, testing=False, cluster=False, n_clusters=100, corr_threshold=0.95, corr_level=0)

    FIND_OPTIMAL=True
    
    parameters_={ # These are optimal as of 3/8/2026 4:00 PM w=4
        "raw": False,
        "extra_features": True,
        "lag_period": 2,
        "lookback_period": 7,
        "sector": True,
        "corr_threshold": 0.8,
        "corr_level": 0,
    }

    if (FIND_OPTIMAL):
        # ------- Selection of Optimal lag_period and lookback_period Parameters -------
        base_RF_model=RandomForestClassifier(max_depth=10, n_estimators=250, random_state=1, n_jobs=MODEL_N_JOBS, class_weight='balanced')
        base_RF_model_pipeline=Pipeline([('scaler', StandardScaler()), ('classifier', base_RF_model)])
        
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

        _, best_parameters, best_score=data_clean_param_selection(*DATA, clone(base_RF_model_pipeline), TEST_SIZE, WINDOW_SIZE, HORIZON, eff_support=True, **param_grid)
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
        
        _, best_parameters, best_score=data_clean_param_selection(*DATA, clone(base_RF_model_pipeline), TEST_SIZE, WINDOW_SIZE, HORIZON, eff_support=True, **param_grid)
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

        _, parameters_, best_score=data_clean_param_selection(*DATA, clone(base_RF_model_pipeline), TEST_SIZE, WINDOW_SIZE, HORIZON, **param_grid)
        print(f"Best Utility Score {best_score}")
        print(f"Optimal parameter {parameters_}")

    download_params = {f"clean_data__{k}=": v for k, v in parameters_.items()}

    X, y_regression=cast(Any, clean_data(*DATA, **parameters_))
    def to_binary_class(y):
        return (y>=0).astype(int)
    y_classification=to_binary_class(y_regression)
    X_train, X_test, y_train, y_test=train_test_split(X, y_classification, test_size=TEST_SIZE, random_state=1, shuffle=False)

    tscv=TimeSeriesSplit(n_splits=5) # CHANGEABLE (OPTIONAL)
    
    # ------- BASE APPLICATION -------
    print("\n\n------- Base RF Model -------")
    RFClassifier_base=RandomForestClassifier(random_state=1, n_jobs=MODEL_N_JOBS, class_weight='balanced')

    RF_pipeline_base=Pipeline([('scaler', StandardScaler()), 
                               ('classifier', RFClassifier_base)])

    param_grid={
        'classifier__max_depth': choose_grid(
            [1, 2, 3, 5],
            [2, 3, 5, 10],
            [3, 5, 10, 20]
        ),
        'classifier__n_estimators': choose_grid(
            [100, 250],
            [250, 500],
            [500, 750]
        )
    }
    grid_search_base=GridSearchCV(RF_pipeline_base, param_grid, cv=tscv, n_jobs=MODEL_N_JOBS, return_train_score=True, verbose=VERBOSE)
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
    util_score=utility_score(results, rwb_obj)
    print(f"Utility Score {util_score:.4}")
    if (EXPORT):
        results.update({'utility_score': round(util_score, 3)})
        results=append_params_to_dict(results, optimized_base_)
        results.update(rwb_obj.results[2])
        results.update(download_params)
        log_result(results, cwd / 'output' / 'results', "results.csv")
    
    if (PAUSE_BETWEEN_MODELS):
        input("Press Enter to continue...")

    # ------- PCA APPLICATION -------
    print("\n\n------- PCA RF Model -------")
    RFClassifier_PCA=RandomForestClassifier(random_state=1, n_jobs=MODEL_N_JOBS, class_weight='balanced')

    RF_pipeline_PCA=Pipeline([('scaler', StandardScaler()),
                              ('reducer', PCA()),
                              ('classifier', RFClassifier_PCA)])
    
    param_grid={
        'reducer__n_components': choose_grid(
            [0.7, 0.85],
            [0.8, 0.95],
            [0.9, 0.99]
        ),
        'classifier__max_depth': choose_grid(
            [1, 2, 3, 5],
            [2, 3, 5, 10],
            [3, 5, 10, 20]
        ),
        'classifier__n_estimators': choose_grid(
            [100, 250],
            [250, 500],
            [500, 750]
        )
    }
    grid_search_PCA=GridSearchCV(RF_pipeline_PCA, param_grid, cv=tscv, n_jobs=MODEL_N_JOBS, return_train_score=True, verbose=VERBOSE)
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
    util_score=utility_score(results, rwb_obj)
    print(f"Utility Score {util_score:.4}")
    if (EXPORT):
        results.update({'utility_score': round(util_score, 3)})
        results=append_params_to_dict(results, optimized_PCA_)
        results.update(rwb_obj.results[2])
        results.update(download_params)
        log_result(results, cwd / 'output' / 'results', "results.csv")

    if (PAUSE_BETWEEN_MODELS):
        input("Press Enter to continue...")

    # ------- LASSO APPLICATION -------
    print("\n\n------- LASSO RF Model -------")
    lasso_selector=SelectFromModel(LogisticRegression(l1_ratio=1, solver='saga', random_state=1, class_weight='balanced', max_iter=500, tol=5e-2), threshold='mean')
    RFClassifier_red_lasso=RandomForestClassifier(random_state=1, n_jobs=MODEL_N_JOBS, class_weight='balanced')

    RF_pipeline_lasso=Pipeline([('scaler', StandardScaler()), 
                              ('feature_selector', lasso_selector),
                              ('classifier', RFClassifier_red_lasso)])

    param_grid={
        'feature_selector__estimator__C': choose_grid(
            [0.0001, 0.001, 0.01, 0.1],
            [0.001, 0.01, 0.1, 1],
            [0.01, 0.1, 1, 10]
        ),
        'classifier__max_depth': choose_grid(
            [1, 2, 3, 5],
            [2, 3, 5, 10],
            [3, 5, 10, 20]
        ),              
        'classifier__n_estimators': choose_grid(
            [250],
            [500],
            [750]
        )
    }
    grid_search_LASSO=GridSearchCV(RF_pipeline_lasso, param_grid, cv=tscv, n_jobs=MODEL_N_JOBS, return_train_score=True, verbose=VERBOSE)
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
    util_score=utility_score(results, rwb_obj)
    print(f"Utility Score {util_score:.4}")
    if (EXPORT):
        results.update({'utility_score': round(util_score, 3)})
        results=append_params_to_dict(results, optimized_LASSO_)
        results.update(rwb_obj.results[2])
        results.update(download_params)
        log_result(results, cwd / 'output' / 'results', "results.csv")
        
    if (PAUSE_BETWEEN_MODELS):
        input("Press Enter to continue...")

    # ------- RIDGE APPLICATION -------
    print("\n\n------- RIDGE RF Model -------")
    ridge_selector=SelectFromModel(LogisticRegression(l1_ratio=0, solver="saga", random_state=1, class_weight='balanced', max_iter=500, tol=5e-2), threshold='mean')
    RFClassifier_red_ridge=RandomForestClassifier(random_state=1, n_jobs=MODEL_N_JOBS, class_weight='balanced')

    RF_pipeline_ridge=Pipeline([('scaler', StandardScaler()), 
                              ('feature_selector', ridge_selector),
                              ('classifier', RFClassifier_red_ridge)])

    param_grid={
        'feature_selector__estimator__C': choose_grid(
            [0.0001, 0.001, 0.01, 0.1],
            [0.001, 0.01, 0.1, 1],
            [0.01, 0.1, 1, 10]
        ),
        'classifier__max_depth': choose_grid(
            [1, 2, 3, 5],
            [2, 3, 5, 10],
            [3, 5, 10, 20]
        ),              
        'classifier__n_estimators': choose_grid(
            [250],
            [500],
            [750]
        )
    }
    grid_search_ridge=GridSearchCV(RF_pipeline_ridge, param_grid, cv=tscv, n_jobs=MODEL_N_JOBS, return_train_score=True, verbose=VERBOSE)
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
    util_score=utility_score(results, rwb_obj)
    print(f"Utility Score {util_score:.4}")
    if (EXPORT):
        results.update({'utility_score': round(util_score, 3)})
        results=append_params_to_dict(results, optimized_ridge_)
        results.update(rwb_obj.results[2])
        results.update(download_params)
        log_result(results, cwd / 'output' / 'results', "results.csv")
        
    if (PAUSE_BETWEEN_MODELS):
        input("Press Enter to continue...")

    # ------- STEP-WISE REGRESSION APPLICATION -------
    print("\n\n------- LASSO(internal) -> STEP-WISE REGRESSION RF Model -------")
    lasso_selector=SelectFromModel(LogisticRegression(l1_ratio=1, solver='saga', random_state=1, class_weight='balanced', max_iter=500, tol=5e-2), max_features=100, threshold='mean')
    RFClassifier_red_lasso=RandomForestClassifier(random_state=1, n_jobs=MODEL_N_JOBS, class_weight='balanced')

    RF_pipeline_lasso=Pipeline([('scaler', StandardScaler()), 
                              ('feature_selector', lasso_selector),
                              ('classifier', RFClassifier_red_lasso)])

    param_grid={
        'feature_selector__estimator__C': choose_grid(
            [0.0001, 0.001, 0.01, 0.1],
            [0.001, 0.01, 0.1, 1],
            [0.01, 0.1, 1, 10]
        ), 
        'classifier__max_depth': choose_grid(
            [1, 2, 3, 5],
            [2, 3, 5, 10],
            [3, 5, 10, 20]
        ),              
        'classifier__n_estimators': choose_grid(
            [250],
            [500],
            [750]
        )
    }
    grid_search_LASSO=GridSearchCV(RF_pipeline_lasso, param_grid, cv=tscv, n_jobs=MODEL_N_JOBS, return_train_score=True, verbose=VERBOSE)
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
        log_result(results, cwd / 'output' / 'results', "results.csv")
        
    if (PAUSE_BETWEEN_MODELS):
        input("Press Enter to Finish...")
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
