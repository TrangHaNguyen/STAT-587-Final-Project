#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mlxtend.feature_selection import SequentialFeatureSelector as MFS
from sklearn.linear_model import Lasso, Ridge

def apply_PCA(X_train, X_test, n_comp=0.9):
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.fit(X_test)
    pca=PCA(n_components=n_comp)
    X_train=pca.fit_transform(X_train)
    X_test=pca.fit(X_test)
    return X_train, X_test