from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.svm import SVC
from sklearn.base import clone
from typing import Any, cast
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from H_prep import clean_data
from H_eval import RollingWindowBacktest, get_final_metrics
from H_helpers import log_result, get_cwd, append_params_to_dict

cwd=get_cwd("STAT-587-Final-Project")

if __name__ == "__main__":
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
    
    y_classification=to_binary_class(y_regression)
    y_train=to_binary_class(y_train)
    y_test=to_binary_class(y_test)
    tscv=TimeSeriesSplit(n_splits=3)

    # ------- Linear SVM -------
    print("\n\n------- Linear SVM Model -------")
    SVM_linear=SVC(kernel="linear", cache_size=1000, class_weight='balanced', gamma='scale', random_state=1, tol=5e-2)

    SVM_linear_pipeline = Pipeline([('scaler', StandardScaler()),
                                    ('classifier', SVM_linear)])

    param_grid={
        'classifier__C': [0.05, 0.1, 1, 10]
    }
    
    grid_search_linear = GridSearchCV(SVM_linear_pipeline, param_grid, cv=tscv, scoring='balanced_accuracy', n_jobs=-1, verbose=1, return_train_score=True)
    grid_search_linear.fit(X_train, y_train)

    rwb_obj=RollingWindowBacktest(clone(grid_search_linear.best_estimator_), X, y_classification, WINDOW_SIZE, HORIZON)
    rwb_obj.rolling_window_backtest(verbose=1)
    rwb_obj.display_wfv_results(X_train)

    optimized_linear_=clone(grid_search_linear.best_estimator_)
    optimized_linear_.fit(X_train, y_train)

    results=get_final_metrics(optimized_linear_, X_train, y_train, X_test, y_test, label="Linear Ker. SVM")
    if (EXPORT):
        results=append_params_to_dict(results, clone(grid_search_linear.best_estimator_))
        results.update(rwb_obj.results[2])
        results.update(download_params)
        log_result(results, cwd / 'output' / 'results', "results.csv")

    # input("Press Enter to continue...")

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

    rwb_obj=RollingWindowBacktest(clone(grid_search_rbf.best_estimator_), X, y_classification, WINDOW_SIZE, HORIZON)
    rwb_obj.rolling_window_backtest(verbose=1)
    rwb_obj.display_wfv_results(X_train)

    optimized_rbf_=clone(grid_search_rbf.best_estimator_)
    optimized_rbf_.fit(X_train, y_train)

    results=get_final_metrics(optimized_rbf_, X_train, y_train, X_test, y_test, label="RBF Ker. SVM")
    if (EXPORT):
        results=append_params_to_dict(results, clone(grid_search_rbf.best_estimator_))
        results.update(rwb_obj.results[2])
        results.update(download_params)
        log_result(results, cwd / 'output' / 'results', "results.csv")

    # input("Press Enter to continue...")

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

    rwb_obj=RollingWindowBacktest(clone(grid_search_poly.best_estimator_), X, y_classification, WINDOW_SIZE, HORIZON)
    rwb_obj.rolling_window_backtest(verbose=1)
    rwb_obj.display_wfv_results(X_train)

    optimized_poly_=clone(grid_search_poly.best_estimator_)
    optimized_poly_.fit(X_train, y_train)

    results=get_final_metrics(optimized_poly_, X_train, y_train, X_test, y_test, label="Poly. Ker. SVM")
    if (EXPORT):
        results=append_params_to_dict(results, clone(grid_search_poly.best_estimator_))
        results.update(rwb_obj.results[2])
        results.update(download_params)
        log_result(results, cwd / 'output' / 'results', "results.csv")

    # input("Press Enter to finish...")
