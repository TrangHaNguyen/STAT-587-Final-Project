from typing import Any, cast
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from data_preprocessing_and_cleaning import clean_data
from model_evaluation import display_feat_importances_logistic, classification_wfv_eval, classification_accuracy
from dimension_reduction import RIDGE, apply_PCA, LASSO, step_wise_reg_wfv

from sklearn.pipeline import Pipeline

if __name__=="__main__":
    X, y_regression=cast(Any, clean_data())
    X_train, X_test, y_train, y_test=train_test_split(X, y_regression, test_size=0.2, random_state=1)
    def to_binary_class(y):
        return (y>=0).astype(int).to_numpy()
    y_train=to_binary_class(y_train)
    y_test=to_binary_class(y_test)

    # ------- RIDGE(External)->LASSO(Internal) APPLICATION -------
    # X_train, X_test=RIDGE(X_train, X_test, y_train, n_features=100)
    # tscv=TimeSeriesSplit(n_splits=3)
    # Log_Reg_R_L=LogisticRegressionCV(Cs=10, cv=tscv, penalty='l1', solver='saga', random_state=1, n_jobs=-1, max_iter=500, tol=1e-2)
    
    # Log_Reg_model_pipeline_R_L = Pipeline([('scaler', StandardScaler()), ('classifier', Log_Reg_R_L)])

    # Log_Reg_model_pipeline_R_L.fit(X_train, y_train)

    # acc, avg_dir=classification_accuracy(Log_Reg_model_pipeline_R_L.predict(X_test), y_test)
    # print("Accuracy (Test):", acc)
    # print("Average Direction (Test):", avg_dir)

    # display_feat_importances_logistic(Log_Reg_model_pipeline_R_L.named_steps['classifier'], X_train)

    # classification_wfv_eval(Log_Reg_model_pipeline_R_L, X_train, y_train)
    # input("Press Enter to continue...")
    # # ------- LASSO(External)->RIDGE(Internal) APPLICATION -------
    # X_train, X_test, _, _=train_test_split(X, y_regression, test_size=0.2, random_state=1)
    # X_train=LASSO(X_train, y_train)
    # X_test=X_test[X_train.columns]
    # tscv=TimeSeriesSplit(n_splits=3)
    # Log_Reg_L_R=LogisticRegressionCV(Cs=10, cv=tscv, penalty='l2', solver='saga', random_state=1, n_jobs=-1, max_iter=500, tol=1e-2)
    
    # Log_Reg_model_pipeline_L_R = Pipeline([('scaler', StandardScaler()), ('classifier', Log_Reg_L_R)])

    # Log_Reg_model_pipeline_L_R.fit(X_train, y_train)

    # acc, avg_dir=classification_accuracy(Log_Reg_model_pipeline_L_R.predict(X_test), y_test)
    # print("Accuracy (Test):", acc)
    # print("Average Direction (Test):", avg_dir)

    # display_feat_importances_logistic(Log_Reg_model_pipeline_L_R.named_steps['classifier'], X_train)

    # classification_wfv_eval(Log_Reg_model_pipeline_L_R, X_train, y_train)
    # input("Press Enter to continue...")
    # ------- PCA APPLICATION -------
    X_train, X_test, _, _=train_test_split(X, y_regression, test_size=0.2, random_state=1)
    X_train, X_test=apply_PCA(X_train, X_test, n_comp=0.9)
    tscv=TimeSeriesSplit(n_splits=3)
    
    Log_Reg_PCA=LogisticRegression(solver='saga', max_iter=5000, random_state=1, n_jobs=-1, tol=1e-2)

    Log_Reg_PCA.fit(X_train, y_train)

    acc, avg_dir=classification_accuracy(Log_Reg_PCA.predict(X_test), y_test)
    print("Accuracy (Test):", acc)
    print("Average Direction (Test):", avg_dir)

    display_feat_importances_logistic(Log_Reg_PCA, X_train)

    classification_wfv_eval(Log_Reg_PCA, X_train, y_train)
    input("Press Enter to Finish...")