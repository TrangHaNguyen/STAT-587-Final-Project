#!/usr/bin/env python3
import pandas as pd
import numpy as np
from mlxtend.feature_selection import SequentialFeatureSelector as MFS
from sklearn.model_selection import TimeSeriesSplit

def step_wise_reg_wfv(model, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, n_splits: int =3, verbose: int =2) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("------- Applying Step-Wise Regression (WFV)")
    tscv=TimeSeriesSplit(n_splits=n_splits)
    mfs=MFS(model, (1, min(int(np.sqrt(X_train.shape[0])), X_train.shape[1])), forward=True, floating=True, cv=tscv, n_jobs=-1, verbose=verbose)
    mfs.fit(X_train, y_train)
    selected_features=list(mfs.k_feature_names_)
    print("Finished Applying Step-Wise Regression (WFV) -------")
    return X_train[selected_features], X_test[selected_features]