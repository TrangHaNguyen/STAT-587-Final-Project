from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.svm import SVC
from sklearn.base import clone
from typing import Any, cast
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from H_prep import clean_data, import_data, data_clean_param_selection
from H_eval import RollingWindowBacktest, get_final_metrics, utility_score
from H_helpers import log_result, get_cwd, append_params_to_dict

cwd=get_cwd("STAT-587-Final-Project")

if __name__ == "__main__":
    WINDOW_SIZE=200
    HORIZON=40
    EXPORT=True
    TEST_SIZE=0.2
    # testing: bool =False, extra_features: bool =True, cluster: bool =False, n_clusters: int =100, corr_threshold: float =0.95, corr_level: int =0
    DATA=import_data(extra_features=True, testing=False, cluster=False, n_clusters=100, corr_threshold=0.95, corr_level=0)
    FIND_OPTIMAL=True
    
    parameters_={
        "raw": False,
        "extra_features": False,
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
            'lag_period': [1, 2, 3, 4, 5, [1, 2], [1, 2, 3], [2, 3], [1, 3]],
            'sector': [True],
            'corr_level': [2]
        }

        _, best_parameters, best_score=data_clean_param_selection(*DATA, clone(base_SVM_rbf_model_pipeline), TEST_SIZE, WINDOW_SIZE, HORIZON, eff_support=True, **param_grid)
        best_lag=best_parameters['lag_period']
        print(f"Best Utility Score (lag_period): {best_score}")
        print(f"Best lag_period: {best_lag}")

        print("------- Finding Optimal lookback_period Value")
        param_grid={
            'lookback_period': [7, 10, 14, 17, 21, 24, 28],
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
            'corr_threshold': [0.8, 0.9, 0.95]
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

    tscv=TimeSeriesSplit(n_splits=3)

    # ------- Linear SVM -------
    print("\n\n------- Linear SVM Model -------")
    SVM_linear=SVC(kernel="linear", cache_size=1000, class_weight='balanced', gamma='scale', random_state=1, tol=5e-2)

    SVM_linear_pipeline = Pipeline([('scaler', StandardScaler()),
                                    ('classifier', SVM_linear)])

    param_grid={
        'classifier__C': [0.1, 1, 10]
    }
    
    grid_search_linear = GridSearchCV(SVM_linear_pipeline, param_grid, cv=tscv, scoring='balanced_accuracy', n_jobs=-1, verbose=1, return_train_score=True)
    grid_search_linear.fit(X_train, y_train)

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
        log_result(results, cwd / 'output' / 'results', "results.csv")

    input("Press Enter to continue...")

    # ------- RBF SVM -------
    print("\n\n------- RBF SVM Model -------")
    SVM_rbf=SVC(kernel="rbf", cache_size=1000, class_weight='balanced', gamma='scale', random_state=1, tol=5e-2)

    SVM_rbf_pipeline = Pipeline([('scaler', StandardScaler()),
                                 ('classifier', SVM_rbf)])

    param_grid={
        'classifier__C': [0.1, 1, 10],
        'classifier__gamma': ['scale', 'auto', 0.01, 0.1, 1]
    }
    
    grid_search_rbf = GridSearchCV(SVM_rbf_pipeline, param_grid, cv=tscv, scoring='balanced_accuracy', n_jobs=-1, verbose=1, return_train_score=True)
    grid_search_rbf.fit(X_train, y_train)

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
        log_result(results, cwd / 'output' / 'results', "results.csv")

    input("Press Enter to continue...")

    # ------- Polynomial SVM -------
    print("\n\n------- Polynomial SVM Model -------")
    SVM_poly=SVC(kernel="poly", cache_size=1000, class_weight='balanced', gamma='scale', random_state=1, tol=5e-2)

    SVM_poly_pipeline = Pipeline([('scaler', StandardScaler()),
                                  ('classifier', SVM_poly)])

    param_grid={
        'classifier__C': [0.1, 1, 10],
        'classifier__gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'classifier__degree': [2, 3, 4, 5]
    }
    
    grid_search_poly = GridSearchCV(SVM_poly_pipeline, param_grid, cv=tscv, scoring='balanced_accuracy', n_jobs=-1, verbose=1, return_train_score=True)
    grid_search_poly.fit(X_train, y_train)

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
        log_result(results, cwd / 'output' / 'results', "results.csv")

    input("Press Enter to finish...")
