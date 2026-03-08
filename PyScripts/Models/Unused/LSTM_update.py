#!/usr/bin/env python3
"""
LSTM update using engineered features from H_prep.clean_data
with lag features removed.
"""

import os
import sys
import argparse
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
from H_prep import clean_data, import_data


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


def drop_lag_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Remove any columns whose feature name contains 'Lag'."""
    if not isinstance(X.columns, pd.MultiIndex):
        return X.loc[:, [c for c in X.columns if "Lag" not in str(c)]]
    keep_cols = [col for col in X.columns if "Lag" not in str(col[0])]
    return X.loc[:, keep_cols]


def main():
    parser = argparse.ArgumentParser(description="LSTM update with optional engineered features.")
    parser.add_argument(
        "--no_engineered_features",
        action="store_true",
        help="Use raw feature set from clean_data (no engineered features).",
    )
    args = parser.parse_args()
    use_engineered = not args.no_engineered_features
    mode_label = "ENGINEERED FEATURES (NO LAG VALUES)" if use_engineered else "RAW FEATURES (NO ENGINEERING)"
    mode_tag = "engineered" if use_engineered else "raw"

    print("=" * 70)
    print(f"LSTM UPDATE: {mode_label}")
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

    np.random.seed(42)
    tf.random.set_seed(42)

    print("Loading and cleaning 8-year data via H_prep...")
    DATA = import_data(testing=False)
    X, y_regression = clean_data(
        DATA=DATA,
        lookback_period=7,
        lag_period=0,
        extra_features=use_engineered,
        raw=not use_engineered,
        cluster=False,
        corr=False,
        corr_level=1,
    )
    X = drop_lag_columns(X)
    y = (y_regression >= 0).astype(int).to_numpy()
    X_raw = X.to_numpy()

    print(f"Feature data shape: X={X_raw.shape}, y={y.shape}")
    print(f"Binary target distribution: Down={sum(y==0)}, Up={sum(y==1)}\n")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw).astype(np.float32)

    sequence_length = 7
    X_seq, seq_indices = reshape_for_lstm(X_scaled, sequence_length=sequence_length)
    y_seq = y[seq_indices]
    print(f"Reshaped data shape: X={X_seq.shape}, y={y_seq.shape}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
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

    output_dir = cwd / "output" / "NotUsed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"lstm_update_{mode_tag}_result.txt"
    with open(output_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(f"LSTM UPDATE RESULTS - {mode_label}\n")
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
