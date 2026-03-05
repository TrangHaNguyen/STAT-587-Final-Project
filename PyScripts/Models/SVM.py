import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.svm import SVC
from sklearn.base import clone
from typing import Any, cast
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from H_prep import clean_data
from H_eval import rolling_window_backtest, get_final_metrics

WINDOW_SIZE=121
HORIZON=21

if __name__ == "__main__":
    X, y_regression=cast(Any, clean_data(lag_period=[1, 2, 3], lookback_period=0, sector=True, corr=True, corr_level=2, testing=True))
    X_train, X_test, y_train, y_test=train_test_split(X, y_regression, test_size=0.2, random_state=1, shuffle=False)
    def to_binary_class(y):
        return (y>=0).astype(int)
    y_classification=to_binary_class(y_regression)
    y_train=to_binary_class(y_train)
    y_test=to_binary_class(y_test)
    tscv=TimeSeriesSplit(n_splits=3)

    # ------- BASE SVM -------
    print("\n\n------- Linear SVM Model -------")
    SVM_linear=SVC(kernel="linear", cache_size=1000, class_weight='balanced', gamma='scale', random_state=1, tol=5e-2)

    SVM_linear_pipeline = Pipeline([('scaler', StandardScaler()),
                                    ('svc', SVM_linear)])

    param_grid={
        'svc__C': [0.05, 0.1, 1, 10]
    }
    
    grid_search_linear = GridSearchCV(SVM_linear_pipeline, param_grid, cv=tscv, scoring='balanced_accuracy', n_jobs=-1, verbose=1, return_train_score=True)
    grid_search_linear.fit(X_train, y_train)

    optimized_linear_=clone(grid_search_linear.best_estimator_)
    optimized_linear_.fit(X_train, y_train)

    rolling_window_backtest(optimized_linear_, X, y_classification, verbose=1, window_size=WINDOW_SIZE, horizon=HORIZON)

    optimized_linear_=clone(grid_search_linear.best_estimator_)
    optimized_linear_.fit(X_train, y_train)

    get_final_metrics(optimized_linear_, X_train, y_train, X_test, y_test)

    input("Press Enter to continue...")

    # ------- RBF SVM -------
    print("\n\n------- RBF SVM Model -------")
    SVM_rbf=SVC(kernel="rbf", cache_size=1000, class_weight='balanced', gamma='scale', random_state=1, tol=5e-2)

    SVM_rbf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVM_rbf)
    ])

    param_grid={
        'svc__C': [0.1, 1, 10],
        'svc__gamma': ['scale', 'auto', 0.01, 0.1, 1]
    }
    
    grid_search_rbf = GridSearchCV(SVM_rbf_pipeline, param_grid, cv=tscv, scoring='balanced_accuracy', n_jobs=-1, verbose=1, return_train_score=True)
    grid_search_rbf.fit(X_train, y_train)

    optimized_rbf_=clone(grid_search_rbf.best_estimator_)
    optimized_rbf_.fit(X_train, y_train)

    rolling_window_backtest(optimized_rbf_, X, y_classification, verbose=1, window_size=WINDOW_SIZE, horizon=HORIZON)

    optimized_rbf_=clone(grid_search_rbf.best_estimator_)
    optimized_rbf_.fit(X_train, y_train)

    get_final_metrics(optimized_rbf_, X_train, y_train, X_test, y_test)

    input("Press Enter to continue...")

    # ------- Polynomial SVM -------
    print("\n\n------- Polynomial SVM Model -------")
    SVM_poly=SVC(kernel="poly", cache_size=1000, class_weight='balanced', gamma='scale', random_state=1, tol=5e-2)

    SVM_poly_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVM_poly)
    ])

    param_grid={
        'svc__C': [0.1, 1, 10],
        'svc__gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'svc__degree': [2, 3, 4, 5]
    }
    
    grid_search_poly = GridSearchCV(SVM_poly_pipeline, param_grid, cv=tscv, scoring='balanced_accuracy', n_jobs=-1, verbose=1, return_train_score=True)
    grid_search_poly.fit(X_train, y_train)

    optimized_poly_=clone(grid_search_poly.best_estimator_)
    optimized_poly_.fit(X_train, y_train)

    rolling_window_backtest(optimized_poly_, X, y_classification, verbose=1, window_size=WINDOW_SIZE, horizon=HORIZON)

    optimized_poly_=clone(grid_search_poly.best_estimator_)
    optimized_poly_.fit(X_train, y_train)

    get_final_metrics(optimized_poly_, X_train, y_train, X_test, y_test)

    input("Press Enter to finish...")
