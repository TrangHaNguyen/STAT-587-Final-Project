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

from H_reduce import step_wise_reg_wfv
from H_prep import clean_data
from H_eval import rolling_window_backtest, get_final_metrics, display_wfv_results
from H_helpers import append_params_to_dict, log_result, get_cwd

VERBOSE=0
WINDOW_SIZE=220
HORIZON=40
EXPORT=True

cwd=get_cwd("STAT-587-Final-Project")

if __name__=="__main__":
    WINDOW_SIZE=200
    HORIZON=40
    EXPORT=True
    data_clean_params={
        "lag_period": [1, 2],
        "lookback_period": 5,
        "cluster": True,
        "sector": True,
        "corr": True,
        "corr_level": 3,
        "testing": False
    }
    download_params = {f"clean_data__{k}=": v for k, v in data_clean_params.items()}
    TEST_SIZE=0.2

    X, y_regression=cast(Any, clean_data(**data_clean_params))
    X_train, X_test, y_train, y_test=train_test_split(X, y_regression, test_size=0.2, random_state=1, shuffle=False)
    def to_binary_class(y):
        return (y>=0).astype(int)
    y_regression=to_binary_class(y_regression)
    y_train=to_binary_class(y_train)
    y_test=to_binary_class(y_test)
    tscv=TimeSeriesSplit(n_splits=5) # CHANGEABLE (OPTIONAL)
    
    # ------- BASE APPLICATION -------
    print("\n\n------- Base RF Model -------")
    RFClassifier_base=RandomForestClassifier(random_state=1, n_jobs=-1, class_weight='balanced')

    RF_pipeline_base=Pipeline([('scaler', StandardScaler()), 
                               ('classifier', RFClassifier_base)])

    param_grid={
        'classifier__max_depth': [2, 3, 5, 10],
        'classifier__n_estimators': [250, 500]
    }
    grid_search_base=GridSearchCV(RF_pipeline_base, param_grid, cv=tscv, n_jobs=-1, return_train_score=True, verbose=VERBOSE)
    grid_search_base.fit(X_train, y_train)

    optimized_base_=clone(grid_search_base.best_estimator_)

    wfv_results=rolling_window_backtest(optimized_base_, X, y_regression, verbose=1, window_size=WINDOW_SIZE, horizon=HORIZON)
    display_wfv_results(wfv_results, X_train, X_test, window_size=WINDOW_SIZE, horizon=HORIZON)

    optimized_base_=clone(grid_search_base.best_estimator_)
    optimized_base_.fit(X_train, y_train)

    results=get_final_metrics(optimized_base_, X_train, y_train, X_test, y_test, label="Base RF")
    if (EXPORT):
        results=append_params_to_dict(results, clone(optimized_base_))
        results.update(wfv_results[2])
        results.update(download_params)
        log_result(results, cwd / 'output' / 'results', "results.csv")
    
    input("Press Enter to continue...")

    # ------- PCA APPLICATION -------
    print("\n\n------- PCA RF Model -------")
    RFClassifier_PCA=RandomForestClassifier(random_state=1, n_jobs=-1, class_weight='balanced')

    RF_pipeline_PCA=Pipeline([('scaler', StandardScaler()),
                              ('reducer', PCA()),
                              ('classifier', RFClassifier_PCA)])
    
    param_grid={
        'reducer__n_components': [0.8, 0.95],
        'classifier__max_depth': [2, 3, 5, 10],
        'classifier__n_estimators': [250, 500]
    }
    grid_search_PCA=GridSearchCV(RF_pipeline_PCA, param_grid, cv=tscv, n_jobs=-1, return_train_score=True, verbose=VERBOSE)
    grid_search_PCA.fit(X_train, y_train)

    optimized_PCA_=clone(grid_search_PCA.best_estimator_)

    wfv_results=rolling_window_backtest(optimized_PCA_, X, y_regression, verbose=1, window_size=WINDOW_SIZE, horizon=HORIZON)
    display_wfv_results(wfv_results, X_train, X_test, window_size=WINDOW_SIZE, horizon=HORIZON)

    optimized_PCA_=clone(grid_search_PCA.best_estimator_)
    optimized_PCA_.fit(X_train, y_train)

    results=get_final_metrics(optimized_PCA_, X_train, y_train, X_test, y_test, label="PCA RF")
    if (EXPORT):
        results=append_params_to_dict(results, clone(optimized_PCA_))
        results.update(wfv_results[2])
        results.update(download_params)
        log_result(results, cwd / 'output' / 'results', "results.csv")

    input("Press Enter to continue...")

    # ------- LASSO APPLICATION -------
    print("\n\n------- LASSO RF Model -------")
    lasso_selector=SelectFromModel(LogisticRegression(l1_ratio=1, solver='saga', random_state=1, max_iter=500, tol=5e-2), threshold='mean')
    RFClassifier_red_lasso=RandomForestClassifier(random_state=1, n_jobs=-1, class_weight='balanced')

    RF_pipeline_lasso=Pipeline([('scaler', StandardScaler()), 
                              ('feature_selector', lasso_selector),
                              ('classifier', RFClassifier_red_lasso)])

    param_grid={
        'feature_selector__estimator__C': [0.001, 0.01, 0.1], 
        'classifier__max_depth': [2, 3, 5, 10],              
        'classifier__n_estimators': [500]
    }
    grid_search_LASSO=GridSearchCV(RF_pipeline_lasso, param_grid, cv=tscv, n_jobs=-1, return_train_score=True, verbose=VERBOSE)
    grid_search_LASSO.fit(X_train, y_train)

    optimized_LASSO_=clone(grid_search_LASSO.best_estimator_)

    wfv_results=rolling_window_backtest(optimized_LASSO_, X, y_regression, verbose=1, window_size=WINDOW_SIZE, horizon=HORIZON)
    display_wfv_results(wfv_results, X_train, X_test, window_size=WINDOW_SIZE, horizon=HORIZON)

    optimized_LASSO_=clone(grid_search_LASSO.best_estimator_)
    optimized_LASSO_.fit(X_train, y_train)

    results=get_final_metrics(optimized_LASSO_, X_train, y_train, X_test, y_test, label="LASSO RF")
    if (EXPORT):
        results=append_params_to_dict(results, clone(optimized_LASSO_))
        results.update(wfv_results[2])
        results.update(download_params)
        log_result(results, cwd / 'output' / 'results', "results.csv")
        
    input("Press Enter to continue...")

    # ------- RIDGE APPLICATION -------
    print("\n\n------- RIDGE RF Model -------")
    ridge_selector=SelectFromModel(LogisticRegression(l1_ratio=0, solver="saga", random_state=1, max_iter=500, tol=5e-2), threshold='mean')
    RFClassifier_red_ridge=RandomForestClassifier(random_state=1, n_jobs=-1, class_weight='balanced')

    RF_pipeline_ridge=Pipeline([('scaler', StandardScaler()), 
                              ('feature_selector', ridge_selector),
                              ('classifier', RFClassifier_red_ridge)])

    param_grid={
        'feature_selector__estimator__C': [0.001, 0.01, 0.1],
        'classifier__max_depth': [2, 3, 5, 10],              
        'classifier__n_estimators': [500]
    }
    grid_search_ridge=GridSearchCV(RF_pipeline_ridge, param_grid, cv=tscv, n_jobs=-1, return_train_score=True, verbose=VERBOSE)
    grid_search_ridge.fit(X_train, y_train)

    optimized_ridge_=clone(grid_search_ridge.best_estimator_)

    wfv_results=rolling_window_backtest(optimized_ridge_, X, y_regression, verbose=1, window_size=WINDOW_SIZE, horizon=HORIZON)
    display_wfv_results(wfv_results, X_train, X_test, window_size=WINDOW_SIZE, horizon=HORIZON)

    optimized_ridge_=clone(grid_search_ridge.best_estimator_)
    optimized_ridge_.fit(X_train, y_train)

    results=get_final_metrics(optimized_ridge_, X_train, y_train, X_test, y_test, label="Ridge RF")
    if (EXPORT):
        results=append_params_to_dict(results, clone(optimized_ridge_))
        results.update(wfv_results[2])
        results.update(download_params)
        log_result(results, cwd / 'output' / 'results', "results.csv")
        
    input("Press Enter to continue...")

    # ------- STEP-WISE REGRESSION APPLICATION -------
    print("\n\n------- LASSO(internal) -> STEP-WISE REGRESSION RF Model -------")
    lasso_selector=SelectFromModel(LogisticRegression(l1_ratio=1, solver='saga', random_state=1, max_iter=500, tol=5e-2), max_features=100, threshold='mean')
    RFClassifier_red_lasso=RandomForestClassifier(random_state=1, n_jobs=-1, class_weight='balanced')

    RF_pipeline_lasso=Pipeline([('scaler', StandardScaler()), 
                              ('feature_selector', lasso_selector),
                              ('classifier', RFClassifier_red_lasso)])

    param_grid={
        'feature_selector__estimator__C': [0.001, 0.01, 0.1], 
        'classifier__max_depth': [2, 3, 5, 10],              
        'classifier__n_estimators': [500]
    }
    grid_search_LASSO=GridSearchCV(RF_pipeline_lasso, param_grid, cv=tscv, n_jobs=-1, return_train_score=True, verbose=VERBOSE)
    grid_search_LASSO.fit(X_train, y_train) 

    best_params_from_grid = grid_search_LASSO.best_params_

    RF_params = {k.replace('classifier__', ''): v 
             for k, v in best_params_from_grid.items() 
             if k.startswith('classifier__')}

    lasso_support = grid_search_LASSO.best_estimator_.named_steps['feature_selector'].get_support()

    lasso_coefficient_names = X_train.columns[lasso_support].tolist()

    X_train_red=X_train[lasso_coefficient_names]
    X_test_red=X_test[lasso_coefficient_names]

    RFClassifier_red_sw_wfv=RandomForestClassifier(**RF_params, random_state=1, n_jobs=1, class_weight='balanced')

    X_train_final, X_test_final=step_wise_reg_wfv(RFClassifier_red_sw_wfv, X_train_red, y_train, X_test_red) 

    RFClassifier_red_sw_wfv.fit(X_train_final, y_train)

    wfv_results=rolling_window_backtest(RFClassifier_red_sw_wfv, X[X_train_final.columns], y_regression, verbose=1, window_size=WINDOW_SIZE, horizon=21)
    display_wfv_results(wfv_results, X_train, X_test, window_size=WINDOW_SIZE, horizon=HORIZON)

    RFClassifier_red_sw_wfv.fit(X_train_final, y_train)

    results=get_final_metrics(RFClassifier_red_sw_wfv, X_train_final, y_train, X_test_final, y_test, label="Stepwise RF")
    if (EXPORT):
        results=append_params_to_dict(results, clone(RFClassifier_red_sw_wfv))
        results.update(wfv_results[2])
        results.update(download_params)
        log_result(results, cwd / 'output' / 'results', "results.csv")
        
    input("Press Enter to Finish...")
