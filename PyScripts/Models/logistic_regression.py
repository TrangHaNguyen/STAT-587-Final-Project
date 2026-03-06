from typing import Any, cast
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import clone
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np

from H_prep import clean_data
from H_eval import rolling_window_backtest, get_final_metrics, display_wfv_results
from H_helpers import log_result, append_params_to_dict, get_cwd

'''No need for hyperparameter tuning for Logistic Regression via GridSearchCV since LogisticRegressionCV performs internal CV to select the best C value. We will just use the default 10 values of C that LogisticRegressionCV tests.'''

VERBOSE=0

cwd=get_cwd("STAT-587-Final-Project")

if __name__=="__main__":
    WINDOW_SIZE=200
    HORIZON=40
    EXPORT=True
    data_clean_params={
        "lag_period": [1, 2, 3],
        "lookback_period": 5,
        "cluster": False,
        "sector": True,
        "corr": True,
        "corr_level": 3,
        "testing": False
    }
    download_params = {f"clean_data__{k}=": v for k, v in data_clean_params.items()}
    TEST_SIZE=0.2

    X, y_regression=cast(Any, clean_data(**data_clean_params))
    X_train, X_test, y_train, y_test=train_test_split(X, y_regression, test_size=TEST_SIZE, random_state=1, shuffle=False)
    data_params={

    }
    def to_binary_class(y):
        return (y>=0).astype(int)

    y_classification=to_binary_class(y_regression)
    y_train=to_binary_class(y_train)
    y_test=to_binary_class(y_test)
    tscv=TimeSeriesSplit(n_splits=5)
    custom_Cs=[0.05, 0.1, 1.0, 10.0]

    # ------- LASSO(Internal) APPLICATION -------
    Log_Reg_R=LogisticRegressionCV(Cs=custom_Cs, cv=tscv, l1_ratios=[1], solver='saga', random_state=1, n_jobs=-1, max_iter=500, tol=1e-2, verbose=VERBOSE)
    
    Log_Reg_model_pipeline_R=Pipeline([('scaler', StandardScaler()), ('classifier', Log_Reg_R)])

    Log_Reg_model_pipeline_R.fit(X_train, y_train)

    best_c = Log_Reg_model_pipeline_R.named_steps['classifier'].C_[0]
    Opt_Log_Reg_R=LogisticRegression(C=best_c, l1_ratio=1, solver='saga', random_state=1, max_iter=500, tol=1e-2)

    Opt_Log_Reg_model_pipeline_R=Pipeline([('scaler', StandardScaler()), ('classifier', Opt_Log_Reg_R)])

    optimized_Log_Reg_R_=clone(Opt_Log_Reg_model_pipeline_R)

    wfv_results=rolling_window_backtest(optimized_Log_Reg_R_, X, y_classification, verbose=1, window_size=WINDOW_SIZE, horizon=HORIZON)

    display_wfv_results(wfv_results, X_train, X_test, window_size=WINDOW_SIZE, horizon=HORIZON)

    optimized_Log_Reg_R_=clone(Opt_Log_Reg_model_pipeline_R)
    optimized_Log_Reg_R_.fit(X_train, y_train)

    results=get_final_metrics(optimized_Log_Reg_R_, X_train, y_train, X_test, y_test, n_splits=10, label="LASSO(int.) Log. Reg.")
    if (EXPORT):
        results=append_params_to_dict(results, clone(optimized_Log_Reg_R_))
        results.update(wfv_results[2])
        results.update(download_params)
        log_result(results, cwd / 'output' / 'results', "results.csv")

    input("Press Enter to continue...")

    # ------- RIDGE(Internal) APPLICATION -------
    Log_Reg_L=LogisticRegressionCV(Cs=custom_Cs, cv=tscv, l1_ratios=[0], solver='saga', random_state=1, n_jobs=-1, max_iter=500, tol=1e-2, verbose=VERBOSE)
    
    Log_Reg_model_pipeline_L=Pipeline([('scaler', StandardScaler()), ('classifier', Log_Reg_L)])

    Log_Reg_model_pipeline_L.fit(X_train, y_train)

    best_c = Log_Reg_model_pipeline_L.named_steps['classifier'].C_[0]
    Opt_Log_Reg_L=LogisticRegression(C=best_c, l1_ratio=0, solver='saga', random_state=1, max_iter=500, tol=1e-2)

    Opt_Log_Reg_model_pipeline_L=Pipeline([('scaler', StandardScaler()), ('classifier', Opt_Log_Reg_L)])

    optimized_Log_Reg_L_=clone(Opt_Log_Reg_model_pipeline_L)

    wfv_results=rolling_window_backtest(optimized_Log_Reg_L_, X, y_classification, verbose=1, window_size=WINDOW_SIZE, horizon=HORIZON)

    display_wfv_results(wfv_results, X_train, X_test, window_size=WINDOW_SIZE, horizon=HORIZON)

    optimized_Log_Reg_L_=clone(Opt_Log_Reg_model_pipeline_L)
    optimized_Log_Reg_L_.fit(X_train, y_train)

    results=get_final_metrics(optimized_Log_Reg_L_, X_train, y_train, X_test, y_test, n_splits=10, label="Ridge(int.) Log. Reg.")
    if (EXPORT):
        results=append_params_to_dict(results, clone(optimized_Log_Reg_L_))
        results.update(wfv_results[2])
        results.update(download_params)
        log_result(results, cwd / 'output' / 'results', "results.csv")

    input("Press Enter to continue...")

    # ------- PCA to Ridge(Internal) APPLICATION -------
    Log_Reg_PCA_L=LogisticRegression(l1_ratio=0, solver='liblinear', random_state=1)
    
    Log_Reg_model_pipeline_PCA_L=Pipeline([('scaler', StandardScaler()),
                                           ('pca', PCA()), 
                                           ('classifier', Log_Reg_PCA_L)])

    param_grid={
        'pca__n_components': [0.7, 0.8, 0.9, 0.95],
        'classifier__C': [0.01, 0.1, 1.0, 10.0]
    }
    grid_search_PCA_ridge=GridSearchCV(Log_Reg_model_pipeline_PCA_L, param_grid, cv=tscv, return_train_score=True, verbose=VERBOSE)

    grid_search_PCA_ridge.fit(X_train, y_train)

    optimized_PCA_ridge_=grid_search_PCA_ridge.best_estimator_

    optimized_Log_Reg_PCA_ridge_=clone(grid_search_PCA_ridge.best_estimator_)
    
    wfv_results=rolling_window_backtest(optimized_Log_Reg_PCA_ridge_, X, y_classification, verbose=1, window_size=WINDOW_SIZE, horizon=HORIZON)

    display_wfv_results(wfv_results, X_train, X_test, window_size=WINDOW_SIZE, horizon=HORIZON)

    optimized_Log_Reg_PCA_ridge_=clone(grid_search_PCA_ridge.best_estimator_)
    optimized_Log_Reg_PCA_ridge_.fit(X_train, y_train)

    results=get_final_metrics(optimized_Log_Reg_PCA_ridge_, X_train, y_train, X_test, y_test, n_splits=10, label="PCA Ridge(int.) Log. Reg.")
    if (EXPORT):
        results=append_params_to_dict(results, clone(optimized_Log_Reg_PCA_ridge_))
        results.update(wfv_results[2])
        results.update(download_params)
        log_result(results, cwd / 'output' / 'results', "results.csv")

    input("Press Enter to Finish...")