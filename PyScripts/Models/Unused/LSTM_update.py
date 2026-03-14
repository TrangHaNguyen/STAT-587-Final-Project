#!/usr/bin/env python3
"""LSTM update aligned with the raw baseline data-loading pipeline."""

import os
import sys
import time
from datetime import datetime
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
from sklearn.preprocessing import StandardScaler

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
from model_grids import NN_LAG_PERIOD, RANDOM_SEED, TEST_SIZE, TRAIN_TEST_SHUFFLE


def reshape_for_lstm(X, sequence_length=7):
    """Create overlapping sequences for LSTM: (samples, timesteps, features)."""
    n_samples, n_features = X.shape
    n_sequences = n_samples - sequence_length + 1
    X_seq = np.zeros((n_sequences, sequence_length, n_features), dtype=np.float32)
    indices = []
    for i in range(n_sequences):
        X_seq[i] = X[i : i + sequence_length]
        indices.append(i + sequence_length - 1)
    return X_seq, np.array(indices)


def _keep_raw_stock_ohlcv(X: pd.DataFrame) -> pd.DataFrame:
    """Match the raw OHLCV feature subset used in the baseline models."""
    idx = pd.IndexSlice
    metrics = ["Open", "Close", "High", "Low", "Volume"]
    return X.loc[:, idx[metrics, "Stocks", :]].copy()


def main():
    print("=" * 70)
    print("LSTM UPDATE: RAW BASELINE-STYLE FEATURES")
    print("=" * 70)
    start_time = time.perf_counter()
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except Exception as exc:
        print("ERROR: TensorFlow is required to run this script.")
        print("Install with: pip install tensorflow")
        print(f"Import error: {exc}")
        sys.exit(1)

    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    print("Loading and cleaning 8-year data via H_prep...")
    DATA = import_data(
        testing=False,
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

    print(f"Feature data shape: X={X_raw.shape}, y={y.shape}")
    print(f"Binary target distribution: Down={sum(y==0)}, Up={sum(y==1)}\n")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw).astype(np.float32)

    sequence_length = len(NN_LAG_PERIOD)
    X_seq, seq_indices = reshape_for_lstm(X_scaled, sequence_length=sequence_length)
    y_seq = y[seq_indices]
    print(f"Reshaped data shape: X={X_seq.shape}, y={y_seq.shape}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq,
        y_seq,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        shuffle=TRAIN_TEST_SHUFFLE,
    )

    n_features = X_seq.shape[2]
    model = keras.Sequential(
        [
            layers.LSTM(64, activation="relu", input_shape=(sequence_length, n_features), return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True,
        verbose=1,
    )

    print("Training LSTM model...")
    train_start = time.perf_counter()
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=60,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1,
    )
    train_elapsed = time.perf_counter() - train_start
    print(f"Training time: {train_elapsed:.1f}s")

    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_proba >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=["Down", "Up"]))
    print(cm)

    output_dir = cwd / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "8yrs_lstm_update_raw_result.txt"
    with open(output_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("LSTM UPDATE RESULTS - RAW BASELINE-STYLE FEATURES\n")
        f.write("=" * 70 + "\n\n")
        f.write("DATA SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write(f"Feature data shape: {X_raw.shape}\n")
        f.write(f"Sequence length: {sequence_length}\n")
        f.write(f"Reshaped data shape: {X_seq.shape}\n")
        f.write(f"Train/Test sizes: {X_train.shape[0]}/{X_test.shape[0]}\n")
        f.write(f"Training epochs: {len(history.history['loss'])}\n")
        f.write(f"Training time: {train_elapsed:.1f}s\n\n")
        f.write("TEST SET PERFORMANCE\n")
        f.write("-" * 70 + "\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred, target_names=["Down", "Up"]))

    print(f"\nSaved results to: {output_path}")

    total_elapsed = time.perf_counter() - start_time
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total elapsed: {total_elapsed/60:.2f} minutes")
    print("Run complete.")


if __name__ == "__main__":
    main()
