from typing import Any, cast
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from data_preprocessing_and_cleaning import clean_data
from model_evaluation import display_feat_importances_logistic, classification_wfv_eval, classification_accuracy
from dimension_reduction import RIDGE, apply_PCA, LASSO, step_wise_reg_wfv  
from sklearn.model_selection import KFold

from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error

if __name__=="__main__":
    X, y_regression=cast(Any, clean_data())
    X_train, X_test, y_train_reg, y_test_reg=train_test_split(X, y_regression, test_size=0.2, random_state=1)
    X_train, X_test, y_train, y_test=train_test_split(X, y_regression, test_size=0.2, random_state=1)
    def to_binary_class(y):
        import numpy as _np
        # accept pandas Series or numpy arrays
        if hasattr(y, "to_numpy"):
            arr = y.to_numpy()
        else:
            arr = _np.asarray(y)
        return (arr >= 0).astype(int)
    y_train=to_binary_class(y_train)
    y_test=to_binary_class(y_test)

# ------- Original Linear Regression Model -------
    X_train, X_test=RIDGE(X_train, X_test, y_train_reg, n_features=200)
    #tscv=TimeSeriesSplit(n_splits=3)
    kf=KFold(n_splits=5, shuffle=True, random_state=1)
    # Multiple Linear Regression (OLS)
    Lin_Reg = LinearRegression()
    Lin_Reg_model_pipeline_R_L = Pipeline([('scaler', StandardScaler()), ('regressor', Lin_Reg)])
    Lin_Reg_model_pipeline_R_L.fit(X_train, y_train_reg)

    # Regression evaluation on hold-out test set
    preds = Lin_Reg_model_pipeline_R_L.predict(X_test)
    print("R2 (Test):", r2_score(y_test_reg, preds))
    print("MSE (Test):", mean_squared_error(y_test_reg, preds))
    preds_binary=to_binary_class(preds)
    Accuracy, avg_dir=classification_accuracy(preds_binary, y_test)
    print("Accuracy (Test):", Accuracy)
    print("Average Direction (Test):", avg_dir)
 

    # ------- RIDGE(External)->Ridge(Internal) APPLICATION -------
    X_train, X_test=RIDGE(X_train, X_test, y_train_reg, n_features=200)
    kf=KFold(n_splits=5, shuffle=True, random_state=1)
    Log_Reg_R_L=LogisticRegressionCV(Cs=10, cv=kf, penalty='l2', solver='saga', random_state=1, n_jobs=-1, max_iter=500, tol=1e-2)
    Log_Reg_model_pipeline_R_L = Pipeline([('scaler', StandardScaler()), ('classifier', Log_Reg_R_L)])
    Log_Reg_model_pipeline_R_L.fit(X_train, y_train)
    display_feat_importances_logistic(Log_Reg_model_pipeline_R_L.named_steps['classifier'], X_train)
    classification_wfv_eval(Log_Reg_model_pipeline_R_L, X_train, y_train)
    preds=Log_Reg_model_pipeline_R_L.predict(X_test)
    Accuracy, avg_dir=classification_accuracy(preds, y_test)
    print("Accuracy (Test):", Accuracy)
    print("Average Direction (Test):", avg_dir)



    

    # ------- PCA APPLICATION and Ridge-------
    X_train, X_test, _, _=train_test_split(X, y_regression, test_size=0.2, random_state=1)
    X_train, X_test=apply_PCA(X_train, X_test, n_comp=0.95) 
    # Log_Reg_PCA=LogisticRegression(solver='saga', max_iter=5000, random_state=1, n_jobs=-1, tol=1e-2)
    Log_Reg_PCA_R=LogisticRegressionCV(Cs=10, cv=kf, penalty='l2', solver='saga', random_state=1, n_jobs=-1, max_iter=500, tol=1e-2)
    Log_Reg_PCA_R = Pipeline([('scaler', StandardScaler()), ('classifier', Log_Reg_PCA_R)])
    Log_Reg_PCA_R.fit(X_train, y_train)
    acc, avg_dir=classification_accuracy(Log_Reg_PCA_R.predict(X_test), y_test)
    print("Accuracy (Test):", acc)
    print("Average Direction (Test):", avg_dir)
    display_feat_importances_logistic(Log_Reg_PCA_R.named_steps['classifier'], X_train)
    classification_wfv_eval(Log_Reg_PCA_R, X_train, y_train)
    preds=Log_Reg_PCA_R.predict(X_test)
    Accuracy, avg_dir=classification_accuracy(preds, y_test)
    print("Accuracy (Test):", Accuracy)
    print("Average Direction (Test):", avg_dir)



    # ------- LASSO(External)->RIDGE(Internal) APPLICATION -------
    # X_train, X_test, _, _=train_test_split(X, y_regression, test_size=0.2, random_state=1)
    # X_train=LASSO(X_train, y_train_reg)
    # X_test=X_test[X_train.columns]
    # kf=KFold(n_splits=5, shuffle=True, random_state=1)
    # Log_Reg_L_R=LogisticRegressionCV(Cs=10, cv=kf, penalty='l2', solver='saga', random_state=1, n_jobs=-1, max_iter=500, tol=1e-2)
    # Log_Reg_model_pipeline_L_R = Pipeline([('scaler', StandardScaler()), ('classifier', Log_Reg_L_R)])
    # Log_Reg_model_pipeline_L_R.fit(X_train, y_train)
    # classification_wfv_eval(Log_Reg_model_pipeline_L_R, X_train, y_train)
    # preds=Log_Reg_model_pipeline_L_R.predict(X_test)
    # Accuracy, avg_dir=classification_accuracy(preds, y_test)
    # print("Accuracy (Test):", Accuracy)
    # print("Average Direction (Test):", avg_dir)