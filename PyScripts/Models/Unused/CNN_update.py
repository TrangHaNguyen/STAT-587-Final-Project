#!/usr/bin/env python3
"""CNN-style classifier aligned with the raw baseline data-loading pipeline."""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from model_grids import NN_LAG_PERIOD, RANDOM_SEED, TEST_SIZE, TRAIN_TEST_SHUFFLE

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)

cwd = Path.cwd()
for _ in range(5):
    if cwd.name != "STAT-587-Final-Project":
        cwd = cwd.parent
    else:
        break
else:
    raise FileNotFoundError("Could not find correct workspace folder.")

sys.path.append(os.path.abspath(cwd / "PyScripts" / "Models"))
from H_prep import clean_data, import_data, to_binary_class


def reshape_for_cnn(X, sequence_length=5):
    """Create flattened rolling windows for sequence modeling."""
    n_samples, n_features = X.shape
    n_sequences = n_samples - sequence_length + 1
    X_reshaped = np.zeros((n_sequences, sequence_length * n_features))
    indices = []

    for i in range(n_sequences):
        X_reshaped[i] = X[i : i + sequence_length].flatten()
        indices.append(i + sequence_length - 1)

    return X_reshaped, np.array(indices)


def _keep_raw_stock_ohlcv(X: pd.DataFrame) -> pd.DataFrame:
    """Match the raw OHLCV feature subset used in the baseline models."""
    idx = pd.IndexSlice
    metrics = ["Open", "Close", "High", "Low", "Volume"]
    return X.loc[:, idx[metrics, "Stocks", :]].copy()


def main():
    print("=" * 70)
    print("CNN UPDATE: RAW BASELINE-STYLE FEATURES")
    print("=" * 70)

    testing = False
    print("Loading and cleaning 8-year data via H_prep...")
    DATA = import_data(
        testing=testing,
        extra_features=False,
        cluster=False,
        n_clusters=100,
        corr_threshold=0.95,
        corr_level=0,
    )
    X, y_regression = clean_data(
        *DATA,
        lookback_period=0,
        lag_period=NN_LAG_PERIOD,
        extra_features=False,
        raw=True,
        corr_threshold=0.95,
        corr_level=0,
    )

    X = _keep_raw_stock_ohlcv(X)
    y = to_binary_class(y_regression).to_numpy()
    X_raw = X.to_numpy()

    print(f"\nFeature data shape: X={X_raw.shape}, y={y.shape}")
    print(f"Binary target distribution: Down={sum(y == 0)}, Up={sum(y == 1)}")
    print(f"Class balance: {sum(y == 0)/len(y)*100:.1f}% Down, {sum(y == 1)/len(y)*100:.1f}% Up\n")

    print("Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    sequence_length = len(NN_LAG_PERIOD)
    print(f"Reshaping data for CNN with sequence_length={sequence_length}...")
    X_seq, seq_indices = reshape_for_cnn(X_scaled, sequence_length=sequence_length)
    y_seq = y[seq_indices]

    print(f"Reshaped data shape: X={X_seq.shape}")
    print(f"Target shape: y={y_seq.shape}")
    print(f"Sequence distribution: Down={sum(y_seq == 0)}, Up={sum(y_seq == 1)}\n")

    print("Splitting data into train/validation/test sets...")
    holdout_size = TEST_SIZE + 0.1
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_seq,
        y_seq,
        test_size=holdout_size,
        random_state=RANDOM_SEED,
        shuffle=TRAIN_TEST_SHUFFLE,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=TEST_SIZE / holdout_size,
        random_state=RANDOM_SEED,
        shuffle=TRAIN_TEST_SHUFFLE,
    )

    print("=" * 70)
    print("MODEL TRAINING")
    print("=" * 70 + "\n")

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        batch_size=32,
        learning_rate_init=0.001,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=15,
        random_state=RANDOM_SEED,
        verbose=0,
        alpha=0.0001,
    )

    print("Fitting model to training data...")
    model.fit(X_train, y_train)

    print("\n" + "=" * 70)
    print("MODEL EVALUATION ON TEST SET")
    print("=" * 70 + "\n")

    y_pred = model.predict(X_test)
    y_val_pred = model.predict(X_val)
    y_train_pred = model.predict(X_train)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Train Accuracy:       {train_acc:.4f}")
    print(f"Validation Accuracy:  {val_acc:.4f}")
    print(f"Test Accuracy:        {test_acc:.4f}\n")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall:    {recall:.4f}")
    print(f"Test F1-Score:  {f1:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Down", "Up"]))

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    output_dir = cwd / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "8yrs_cnn_update_raw_result.txt"
    with open(output_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("CNN UPDATE RESULTS - RAW BASELINE-STYLE FEATURES\n")
        f.write("=" * 70 + "\n\n")
        f.write("DATA SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write(f"Feature data shape: {X_raw.shape}\n")
        f.write(f"Sequence length: {sequence_length} trading days\n")
        f.write(f"Reshaped data shape: {X_seq.shape}\n")
        f.write(f"Train/Val/Test split: {X_train.shape[0]}/{X_val.shape[0]}/{X_test.shape[0]}\n\n")
        f.write("TEST SET PERFORMANCE\n")
        f.write("-" * 70 + "\n")
        f.write(f"Train Accuracy:       {train_acc:.4f}\n")
        f.write(f"Validation Accuracy:  {val_acc:.4f}\n")
        f.write(f"Test Accuracy:        {test_acc:.4f}\n\n")
        f.write(f"Test Precision: {precision:.4f}\n")
        f.write(f"Test Recall:    {recall:.4f}\n")
        f.write(f"Test F1-Score:  {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred, target_names=["Down", "Up"]))

    print(f"\nSaved results to: {output_path}")

    print("Run complete.")


if __name__ == "__main__":
    main()
