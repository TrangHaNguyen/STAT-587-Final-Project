#!/usr/bin/env python3
# Run from project root: .venv/bin/python PyScripts/Models/inspect_features.py
import os
import sys
from pathlib import Path
from typing import Any, cast
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from H_prep import clean_data, import_data, to_binary_class
from H_modeling import load_input_data
from H_helpers import get_cwd
from model_grids import RANDOM_SEED, TEST_SIZE, TRAIN_TEST_SHUFFLE

cwd = get_cwd("STAT-587-Final-Project")

DATA = load_input_data(
    use_sample_parquet=False,
    sample_parquet_path="",
    import_data_fn=import_data,
    import_data_kwargs={
        "extra_features": True,
        "testing": False,
        "cluster": False,
        "n_clusters": 100,
        "corr_threshold": 0.95,
        "corr_level": 0,
    },
)

parameters_ = {
    "raw": False,
    "extra_features": True,
    "lag_period": [1, 2, 3, 4, 5, 6, 7],
    "lookback_period": 30,
    "sector": False,
    "corr_threshold": 0.95,
    "corr_level": 0,
}

X, y_regression = cast(Any, clean_data(*DATA, **parameters_))
y_classification = to_binary_class(y_regression)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_classification, test_size=TEST_SIZE,
    random_state=RANDOM_SEED, shuffle=TRAIN_TEST_SHUFFLE
)

print(f"X_train shape:    {X_train.shape}")
print(f"X_test shape:     {X_test.shape}")
print(f"n_features:       {X_train.shape[1]}")
print(f"n_train_samples:  {X_train.shape[0]}")
print(f"n_test_samples:   {X_test.shape[0]}")
