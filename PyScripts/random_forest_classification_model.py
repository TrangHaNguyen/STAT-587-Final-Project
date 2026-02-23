#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, cast
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score, KFold, cross_validate
from dimension_reduction import apply_PCA, LASSO, RIDGE, step_wise_reg_wfv
from model_evaluation import classification_accuracy, classification_cv_eval, display_feat_importances, classification_wfv_eval
from data_preprocessing_and_cleaning import clean_data, pull_features

if __name__=="__main__":
    
    X, y_regression=cast(Any, clean_data())
    X_train, X_test, y_train, y_test=train_test_split(X, y_regression, test_size=0.2, random_state=1)
    def to_binary_class(y):
        return (y>=0).astype(int).to_numpy()
    y_train=to_binary_class(y_train)
    y_test=to_binary_class(y_test)
    X_train, X_test=apply_PCA(X_train, X_test, n_comp=0.9) 
    
    
    # ------- PCA APPLICATION -------

    # RFClassifier_red_PCA=RandomForestClassifier(random_state=1, n_jobs=-1)
    # RFClassifier_red_PCA.fit(X_train, y_train)

    # display_feat_importances(RFClassifier_red_PCA, X_train)

    # acc, avg_dir=classification_accuracy(RFClassifier_red_PCA.predict(X_test), y_test)
    # print("Accuracy (Test):", acc)
    # print("Average Direction (Test):", avg_dir)

    # # classification_cv_eval(RFClassifier_red_PCA, X_train, y_train)
    # classification_wfv_eval(RFClassifier_red_PCA, X_train, y_train)

    # ------- LASSO APPLICATION -------

    # X_train, X_test, _, _=train_test_split(X, y_regression, test_size=0.2, random_state=1)

    # X_train=LASSO(X_train, y_train)
    # X_test=X_test[X_train.columns]

    # RFClassifier_red_LASSO=RandomForestClassifier(random_state=1, n_jobs=-1)
    # RFClassifier_red_LASSO.fit(X_train, y_train)

    # display_feat_importances(RFClassifier_red_LASSO, X_train)

    # acc, avg_dir=classification_accuracy(RFClassifier_red_PCA.predict(X_test), y_test)
    # print("Accuracy (Test):", acc)
    # print("Average Direction (Test):", avg_dir)

    # classification_wfv_eval(RFClassifier_red_LASSO, X_train, y_train)

    # ------- RIDGE APPLICATION -------

    # X_train, X_test, _, _=train_test_split(X, y_regression, test_size=0.2, random_state=1)

    # X_train, X_test=RIDGE(X_train, X_test, y_train, n_features=50)

    # RFClassifier_red_RIDGE=RandomForestClassifier(random_state=1, n_jobs=-1)
    # RFClassifier_red_RIDGE.fit(X_train, y_train)

    # display_feat_importances(RFClassifier_red_RIDGE, X_train)

    # acc, avg_dir=classification_accuracy(RFClassifier_red_RIDGE.predict(X_test), y_test)
    # print("Accuracy (Test):", acc)
    # print("Average Direction (Test):", avg_dir)

    # classification_wfv_eval(RFClassifier_red_RIDGE, X_train, y_train)

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

    # classification_wfv_eval(RFClassifier_red_sw_wfv, X_train, y_train)
    classification_cv_eval(RFClassifier_red_sw_wfv, X_train, y_train)

