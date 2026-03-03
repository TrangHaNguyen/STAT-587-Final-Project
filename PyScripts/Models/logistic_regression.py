from typing import Any, cast
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline

from data_preprocessing_and_cleaning import clean_data

'''No need for hyperparameter tuning for Logistic Regression via GridSearchCV since LogisticRegressionCV performs internal CV to select the best C value. We will just use the default 10 values of C that LogisticRegressionCV tests.'''

VERBOSE=2

if __name__=="__main__":
    X, y_regression=cast(Any, clean_data())
    X_train, X_test, y_train, y_test=train_test_split(X, y_regression, test_size=0.2, random_state=1)
    def to_binary_class(y):
        return (y>=0).astype(int).to_numpy()
    y_regression=to_binary_class(y_regression)
    y_train=to_binary_class(y_train)
    y_test=to_binary_class(y_test)
    tscv=TimeSeriesSplit(n_splits=3)

    # ------- LASSO(Internal) APPLICATION -------
    Log_Reg_R=LogisticRegressionCV(Cs=10, cv=tscv, penalty='l1', solver='saga', random_state=1, n_jobs=-1, max_iter=500, tol=1e-2, verbose=VERBOSE)
    
    Log_Reg_model_pipeline_R = Pipeline([('scaler', StandardScaler()), ('classifier', Log_Reg_R)])

    Log_Reg_model_pipeline_R.fit(X_train, y_train)

    input("Press Enter to continue...")

    # ------- RIDGE(Internal) APPLICATION -------
    Log_Reg_L=LogisticRegressionCV(Cs=10, cv=tscv, penalty='l2', solver='saga', random_state=1, n_jobs=-1, max_iter=500, tol=1e-2, verbose=VERBOSE)
    
    Log_Reg_model_pipeline_L=Pipeline([('scaler', StandardScaler()), ('classifier', Log_Reg_L)])

    Log_Reg_model_pipeline_L.fit(X_train, y_train)

    input("Press Enter to continue...")

    # ------- PCA to Ridge(Internal) APPLICATION -------
    Log_Reg_PCA_L=LogisticRegression(penalty='l2', solver='liblinear', random_state=1)
    
    Log_Reg_model_pipeline_PCA_L=Pipeline([('scaler', StandardScaler()),
                                           ('pca', PCA(n_components=0.9)), 
                                           ('classifier', Log_Reg_PCA_L)])

    param_grid={
        'pca__n_components': [0.7, 0.8, 0.9, 0.95],
        'classifier__C': [0.01, 0.1, 1.0, 10.0]     
    }
    grid_search_PCA_ridge=GridSearchCV(Log_Reg_model_pipeline_PCA_L, param_grid, cv=tscv, return_train_score=True, verbose=VERBOSE)

    grid_search_PCA_ridge.fit(X_train, y_train)

    input("Press Enter to Finish...")
