from typing import Any, cast
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from data_preprocessing_and_cleaning import clean_data
# from model_evaluation import display_feat_importances_logistic, classification_wfv_eval, classification_accuracy
# from dimension_reduction import RIDGE, apply_PCA, LASSO, step_wise_reg_wfv

from sklearn.pipeline import Pipeline

if __name__=="__main__":
    X, y_regression=cast(Any, clean_data())
    X_train, X_test, y_train, y_test=train_test_split(X, y_regression, test_size=0.2, random_state=1)
    def to_binary_class(y):
        return (y>=0).astype(int).to_numpy()
    y_train=to_binary_class(y_train)
    y_test=to_binary_class(y_test)

    # ------- LASSO(Internal) APPLICATION -------
    tscv=TimeSeriesSplit(n_splits=3)
    Log_Reg_R=LogisticRegressionCV(Cs=10, cv=tscv, penalty='l1', solver='saga', random_state=1, n_jobs=-1, max_iter=500, tol=1e-2)
    
    Log_Reg_model_pipeline_R = Pipeline([('scaler', StandardScaler()), ('classifier', Log_Reg_R)])

    Log_Reg_model_pipeline_R.fit(X_train, y_train)

    # acc, avg_dir=classification_accuracy(Log_Reg_model_pipeline_R.predict(X_test), y_test)
    # print("Accuracy (Test):", acc)
    # print("Average Direction (Test):", avg_dir)

    # display_feat_importances_logistic(Log_Reg_model_pipeline_R.named_steps['classifier'], X_train)

    # classification_wfv_eval(Log_Reg_model_pipeline_R, X_train, y_train)
    input("Press Enter to continue...")
    # ------- RIDGE(Internal) APPLICATION -------
    X_train, X_test, _, _=train_test_split(X, y_regression, test_size=0.2, random_state=1)
    tscv=TimeSeriesSplit(n_splits=3)
    Log_Reg_L=LogisticRegressionCV(Cs=10, cv=tscv, penalty='l2', solver='saga', random_state=1, n_jobs=-1, max_iter=500, tol=1e-2)
    
    Log_Reg_model_pipeline_L=Pipeline([('scaler', StandardScaler()), ('classifier', Log_Reg_L)])

    Log_Reg_model_pipeline_L.fit(X_train, y_train)

    # acc, avg_dir=classification_accuracy(Log_Reg_model_pipeline_L.predict(X_test), y_test)
    # print("Accuracy (Test):", acc)
    # print("Average Direction (Test):", avg_dir)

    # display_feat_importances_logistic(Log_Reg_model_pipeline_L.named_steps['classifier'], X_train)

    # classification_wfv_eval(Log_Reg_model_pipeline_L, X_train, y_train)
    input("Press Enter to continue...")
    # ------- PCA to Ridge(Internal) APPLICATION -------
    X_train, X_test, _, _=train_test_split(X, y_regression, test_size=0.2, random_state=1)
    tscv=TimeSeriesSplit(n_splits=3)
    Log_Reg_PCA_L=LogisticRegressionCV(Cs=10, cv=tscv, penalty='l1', solver='saga', random_state=1, n_jobs=-1, max_iter=500, tol=1e-2)

    Log_Reg_model_pipeline_PCA_L=Pipeline([('scaler', StandardScaler()),
                                           ('pca', PCA(n_components=0.9)), 
                                           ('classifier', Log_Reg_PCA_L)])

    Log_Reg_model_pipeline_PCA_L.fit(X_train, y_train)

    # acc, avg_dir=classification_accuracy(Log_Reg_model_pipeline_PCA_L.predict(X_test), y_test)
    # print("Accuracy (Test):", acc)
    # print("Average Direction (Test):", avg_dir)

    # display_feat_importances_logistic(Log_Reg_model_pipeline_PCA_L.named_steps['classifier'], X_train)

    # classification_wfv_eval(Log_Reg_model_pipeline_PCA_L, X_train, y_train)
    input("Press Enter to Finish...")