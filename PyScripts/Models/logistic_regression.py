from typing import Any, cast
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline

from data_preprocessing_and_cleaning import clean_data
from model_evaluation import get_final_metrics_grid, rolling_window_backtest, classification_accuracy, get_final_metrics

'''No need for hyperparameter tuning for Logistic Regression via GridSearchCV since LogisticRegressionCV performs internal CV to select the best C value. We will just use the default 10 values of C that LogisticRegressionCV tests.'''

VERBOSE=0

if __name__=="__main__":
    X, y_regression=cast(Any, clean_data(sector=True, corr=True, corr_level=2, testing=True))
    X_train, X_test, y_train, y_test=train_test_split(X, y_regression, test_size=0.2, random_state=1)
    def to_binary_class(y):
        return (y>=0).astype(int)
    y_classification=to_binary_class(y_regression)
    y_train=to_binary_class(y_train)
    y_test=to_binary_class(y_test)
    tscv=TimeSeriesSplit(n_splits=3)

    # ------- LASSO(Internal) APPLICATION -------
    Log_Reg_R=LogisticRegressionCV(Cs=5, cv=tscv, l1_ratios=[1], solver='saga', random_state=1, n_jobs=-1, max_iter=500, tol=1e-2, verbose=VERBOSE)
    
    Log_Reg_model_pipeline_R=Pipeline([('scaler', StandardScaler()), ('classifier', Log_Reg_R)])

    Log_Reg_model_pipeline_R.fit(X_train, y_train)

    best_c = Log_Reg_model_pipeline_R.named_steps['classifier'].C_[0]
    Opt_Log_Reg_R=LogisticRegression(C=best_c, l1_ratio=1, solver='saga', random_state=1, max_iter=500, tol=1e-2)

    Opt_Log_Reg_model_pipeline_R=Pipeline([('scaler', StandardScaler()), ('classifier', Opt_Log_Reg_R)])

    Opt_Log_Reg_model_pipeline_R.fit(X_train, y_train)

    rolling_window_backtest(Opt_Log_Reg_model_pipeline_R, X, y_classification, verbose=1)

    get_final_metrics(Opt_Log_Reg_model_pipeline_R, X_train, y_train, X_test, y_test)

    input("Press Enter to continue...")

    # ------- RIDGE(Internal) APPLICATION -------
    Log_Reg_L=LogisticRegressionCV(Cs=5, cv=tscv, l1_ratios=[0], solver='saga', random_state=1, n_jobs=-1, max_iter=500, tol=1e-2, verbose=VERBOSE)
    
    Log_Reg_model_pipeline_L=Pipeline([('scaler', StandardScaler()), ('classifier', Log_Reg_L)])

    Log_Reg_model_pipeline_L.fit(X_train, y_train)

    best_c = Log_Reg_model_pipeline_L.named_steps['classifier'].C_[0]
    Opt_Log_Reg_L=LogisticRegression(C=best_c, l1_ratio=1, solver='saga', random_state=1, max_iter=500, tol=1e-2)

    Opt_Log_Reg_model_pipeline_L=Pipeline([('scaler', StandardScaler()), ('classifier', Opt_Log_Reg_L)])

    Opt_Log_Reg_model_pipeline_L.fit(X_train, y_train)

    rolling_window_backtest(Opt_Log_Reg_model_pipeline_L, X, y_classification, verbose=1)

    get_final_metrics(Opt_Log_Reg_model_pipeline_L, X_train, y_train, X_test, y_test)

    input("Press Enter to continue...")

    # ------- PCA to Ridge(Internal) APPLICATION -------
    Log_Reg_PCA_L=LogisticRegression(l1_ratio=0, solver='liblinear', random_state=1)
    
    Log_Reg_model_pipeline_PCA_L=Pipeline([('scaler', StandardScaler()),
                                           ('pca', PCA(n_components=0.9)), 
                                           ('classifier', Log_Reg_PCA_L)])

    param_grid={
        'pca__n_components': [0.7, 0.8, 0.9, 0.95],
        'classifier__C': [0.01, 0.1, 1.0, 10.0]     
    }
    grid_search_PCA_ridge=GridSearchCV(Log_Reg_model_pipeline_PCA_L, param_grid, cv=tscv, return_train_score=True, verbose=VERBOSE)

    grid_search_PCA_ridge.fit(X_train, y_train)

    optimized_PCA_ridge_=grid_search_PCA_ridge.best_estimator_

    rolling_window_backtest(optimized_PCA_ridge_, X, y_classification, verbose=1)

    get_final_metrics_grid(grid_search_PCA_ridge, X_test, y_test)

    input("Press Enter to Finish...")
