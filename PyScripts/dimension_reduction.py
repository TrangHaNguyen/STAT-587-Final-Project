#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mlxtend.feature_selection import SequentialFeatureSelector as MFS
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import TimeSeriesSplit

def apply_PCA(X_train: pd.DataFrame, X_test: pd.DataFrame, n_comp: float|int =0.9) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("------- Applying PCA")
    X_train_index=X_train.index
    X_test_index=X_test.index
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    print("Finished Scaling.")
    pca=PCA(n_components=n_comp, random_state=1)
    X_train=pca.fit_transform(X_train)
    X_test=pca.transform(X_test)
    print("Finished Applying PCA -------")

    pca_columns=[f'PC {i+1}' for i in range(X_train.shape[1])]
    X_train_pca=pd.DataFrame(X_train, columns=pca_columns, index=X_train_index)
    X_test_pca=pd.DataFrame(X_test, columns=pca_columns, index=X_test_index)
    return X_train_pca, X_test_pca

def step_wise_reg_cv(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, model) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("------- Applying Step-Wise Regression (CV)")
    X_train_index=X_train.index
    X_train_columns=X_train.columns
    X_test_index=X_test.index
    X_test_columns=X_test.columns
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    X_train = pd.DataFrame(X_train, index=X_train_index, columns=X_train_columns)
    X_test = pd.DataFrame(X_test, index=X_test_index, columns=X_test_columns)
    print("Finished Scaling.")
    mfs=MFS(model, int(np.sqrt(X_train.shape[0])), forward=True, floating=True, cv=3, n_jobs=-1)
    mfs.fit(X_train, y_train)
    selected_features=list(mfs.k_feature_names_)
    print("Finished Applying Step-Wise Regression (CV) -------")
    return X_train[selected_features], X_test[selected_features]

def step_wise_reg_wfv(model, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, n_splits: int =3) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("------- Applying Step-Wise Regression (WFV)")
    tscv=TimeSeriesSplit(n_splits=3)
    mfs=MFS(model, (1, int(np.sqrt(X_train.shape[0]))), forward=True, floating=True, cv=tscv, n_jobs=-1, verbose=2)
    mfs.fit(X_train, y_train)
    selected_features=list(mfs.k_feature_names_)
    print("Finished Applying Step-Wise Regression (WFV) -------")
    return X_train[selected_features], X_test[selected_features]


def LASSO(X_train: pd.DataFrame, y_train: pd.DataFrame) -> pd.DataFrame:
    print("------- Applying LASSO")
    alphas=np.logspace(-7, 2, 100)
    lasso_cv=LassoCV(alphas=alphas, cv=4, random_state=42, n_jobs=-1, tol=1e-2)
    lasso_cv.fit(X_train, y_train)
    print("Finished Applying LASSO -------")
    return X_train[X_train.columns[lasso_cv.coef_!=0]]

def RIDGE(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, n_features: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("------- Applying RIDGE")
    scaler=StandardScaler()
    X_train_index=X_train.index
    X_train_columns=X_train.columns
    X_test_index=X_test.index
    X_test_columns=X_test.columns
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    X_train = pd.DataFrame(X_train, index=X_train_index, columns=X_train_columns)
    X_test = pd.DataFrame(X_test, index=X_test_index, columns=X_test_columns)
    alphas=[1e-3, 1e-2, 0.1, 1, 10, 100, 1000]
    ridge_cv=RidgeCV(alphas=alphas)
    ridge_cv.fit(X_train, y_train)
    importance=np.abs(ridge_cv.coef_)
    top_importance=np.argsort(importance)[-n_features:]
    top_stocks=X_train.columns[top_importance]
    print("Finished Applying RIDGE -------")
    return X_train[top_stocks], X_test[top_stocks]