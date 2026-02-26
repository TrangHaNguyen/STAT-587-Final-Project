#!/usr/bin/env python3
from typing import Any, cast
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from dimension_reduction import apply_PCA, LASSO, RIDGE, step_wise_reg_wfv
from model_evaluation import classification_accuracy, classification_cv_eval, display_feat_importances, classification_wfv_eval
from data_preprocessing_and_cleaning import clean_data

if __name__=="__main__":
    X, y_regression=cast(Any, clean_data())
    X_train, X_test, y_train, y_test=train_test_split(X, y_regression, test_size=0.2, random_state=1)
    def to_binary_class(y):
        return (y>=0).astype(int)
    y_train=to_binary_class(y_train)
    y_test=to_binary_class(y_test)
    
    # ------- BASE APPLICATION -------
    RFClassifier_base=RandomForestClassifier(random_state=1, n_jobs=-1)
    RFClassifier_base.fit(X_train, y_train)

    display_feat_importances(RFClassifier_base, X_train)

    acc, avg_dir=classification_accuracy(RFClassifier_base.predict(X_test), y_test)
    print("Accuracy (Test):", acc)
    print("Average Direction (Test):", avg_dir)

    classification_wfv_eval(RFClassifier_base, X_train, y_train)

    input("Press Enter to continue...")
    # ------- PCA APPLICATION -------

    X_train, X_test=apply_PCA(X_train, X_test, n_comp=0.9) 

    RFClassifier_red_PCA=RandomForestClassifier(random_state=1, n_jobs=-1)
    RFClassifier_red_PCA.fit(X_train, y_train)

    display_feat_importances(RFClassifier_red_PCA, X_train)

    acc, avg_dir=classification_accuracy(RFClassifier_red_PCA.predict(X_test), y_test)
    print("Accuracy (Test):", acc)
    print("Average Direction (Test):", avg_dir)

    # classification_cv_eval(RFClassifier_red_PCA, X_train, y_train)
    classification_wfv_eval(RFClassifier_red_PCA, X_train, y_train)

    input("Press Enter to continue...")
    # ------- LASSO APPLICATION -------

    X_train, X_test, _, _=train_test_split(X, y_regression, test_size=0.2, random_state=1)

    X_train=LASSO(X_train, y_train)
    X_test=X_test[X_train.columns]
    print(X_train.shape)
    print(X_test.shape)
    
    RFClassifier_red_LASSO=RandomForestClassifier(random_state=1, n_jobs=-1)
    RFClassifier_red_LASSO.fit(X_train, y_train)

    display_feat_importances(RFClassifier_red_LASSO, X_train)

    acc, avg_dir=classification_accuracy(RFClassifier_red_LASSO.predict(X_test), y_test)
    print("Accuracy (Test):", acc)
    print("Average Direction (Test):", avg_dir)

    classification_wfv_eval(RFClassifier_red_LASSO, X_train, y_train)

    input("Press Enter to continue...")
    # ------- RIDGE APPLICATION -------

    X_train, X_test, _, _=train_test_split(X, y_regression, test_size=0.2, random_state=1)

    X_train, X_test=RIDGE(X_train, X_test, y_train, n_features=50)

    RFClassifier_red_RIDGE=RandomForestClassifier(random_state=1, n_jobs=-1)
    RFClassifier_red_RIDGE.fit(X_train, y_train)

    display_feat_importances(RFClassifier_red_RIDGE, X_train)

    acc, avg_dir=classification_accuracy(RFClassifier_red_RIDGE.predict(X_test), y_test)
    print("Accuracy (Test):", acc)
    print("Average Direction (Test):", avg_dir)

    classification_wfv_eval(RFClassifier_red_RIDGE, X_train, y_train)

    input("Press Enter to continue...")
    # ------- STEP-WISE REGRESSION APPLICATION -------

    X_train, X_test, _, _=train_test_split(X, y_regression, test_size=0.2, random_state=1)

    RFClassifier_red_sw_wfv=RandomForestClassifier(random_state=1, n_jobs=-1)

    X_train, X_test=RIDGE(X_train, X_test, y_train, n_features=50)
    X_train, X_test=step_wise_reg_wfv(RFClassifier_red_sw_wfv, X_train, y_train, X_test) 

    RFClassifier_red_sw_wfv.fit(X_train, y_train)

    display_feat_importances(RFClassifier_red_sw_wfv, X_train)

    acc, avg_dir=classification_accuracy(RFClassifier_red_sw_wfv.predict(X_test), y_test)
    print("Accuracy (Test):", acc)
    print("Average Direction (Test):", avg_dir)

    classification_wfv_eval(RFClassifier_red_sw_wfv, X_train, y_train)
    # classification_cv_eval(RFClassifier_red_sw_wfv, X_train, y_train)

    input("Press Enter to Finish...")