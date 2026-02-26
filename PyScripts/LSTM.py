#!/usr/bin/env python3
"""
Long Short-Term Memory (LSTM) for time series classification.
Uses a single LSTM layer with minimal parameters for memory efficiency.
Predicts SPX binary direction (Up/Down) using y_binary.
Optimized for constrained compute environments.
"""

import numpy as np
from pathlib import Path
import sys
import time
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from basic_model_original_data import load_original_data


def reshape_for_lstm(X, sequence_length=5):
    """
    Create overlapping sequences for LSTM: (samples, timesteps, features).

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        sequence_length: Number of timesteps per sequence

    Returns:
        X_seq: 3D array (n_sequences, sequence_length, n_features)
        indices: Indices mapping sequences to targets
    """
    n_samples, n_features = X.shape
    n_sequences = n_samples - sequence_length + 1

    X_seq = np.zeros((n_sequences, sequence_length, n_features), dtype=np.float32)
    indices = []

    for i in range(n_sequences):
        X_seq[i] = X[i:i + sequence_length]
        indices.append(i + sequence_length - 1)

    return X_seq, np.array(indices)


def main():
    print("=" * 70)
    print("LSTM (LONG SHORT-TERM MEMORY) FOR TIME SERIES CLASSIFICATION")
    print("=" * 70)
    print("Task: Predict SPX binary direction (Up/Down) from stock time series")
    print("Architecture: Stacked LSTM layers with 7-day time window")
    print("=" * 70 + "\n")

    start_time = time.perf_counter()
    start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Start time: {start_ts}\n")

    # Attempt to import TensorFlow
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

    # Load data
    print("Loading data...")
    X_raw, targets = load_original_data()
    y = targets['binary']

    print(f"Original data shape: X={X_raw.shape}, y={y.shape}")
    print(f"Binary target distribution: Down={sum(y==0)}, Up={sum(y==1)}\n")

    # Standardize features
    print("Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw).astype(np.float32)

    # Reshape for LSTM
    print("Reshaping data for LSTM sequences...")
    sequence_length = 7
    X_seq, seq_indices = reshape_for_lstm(X_scaled, sequence_length=sequence_length)
    y_seq = y[seq_indices]

    print(f"Reshaped data shape: X={X_seq.shape} (samples, timesteps, features)")
    print(f"Target shape: y={y_seq.shape}\n")

    # Train/test split (80/20)
    # Validation comes from Keras validation_split during training
    print("Splitting into train/test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
    )

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Validation split: 20% of training data (automatic during training)\n")

    # Build minimal LSTM model
    print("Building LSTM model with moderate complexity...")
    n_features = X_seq.shape[2]
    
    model = keras.Sequential([
        layers.LSTM(64, activation='relu', input_shape=(sequence_length, n_features), return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("\nModel summary:")
    model.summary()

    # Early stopping with patience
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=12,
        restore_best_weights=True,
        verbose=1
    )

    # Train
    print("\n" + "=" * 70)
    print("Training LSTM model...")
    print("=" * 70 + "\n")
    
    train_start = time.perf_counter()
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=60,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    train_elapsed = time.perf_counter() - train_start
    print(f"\n✓ Training completed in {train_elapsed:.1f} seconds ({train_elapsed/60:.2f} minutes)")

    # Evaluate on test set
    print("\n" + "=" * 70)
    print("Evaluating on test set...")
    print("=" * 70 + "\n")
    
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_proba >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Test set performance:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"               Predicted Down  Predicted Up")
    print(f"Actual Down:   {cm[0,0]:>15} {cm[0,1]:>12}")
    print(f"Actual Up:     {cm[1,0]:>15} {cm[1,1]:>12}\n")

    # Save results
    output_dir = Path.cwd() / "output"
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "lstm_results.txt", 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("LSTM MODEL RESULTS - TIME SERIES CLASSIFICATION\n")
        f.write("=" * 70 + "\n\n")
        f.write("MODEL CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Architecture: Stacked LSTM(64, return_seq=True) + LSTM(32) + Dense(16) + Dense(1, sigmoid)\n")
        f.write(f"Dropout: 0.2 after each LSTM layer\n")
        f.write(f"Sequence length: {sequence_length} timesteps (10-day window)\n")
        f.write(f"Input features: {n_features}\n")
        f.write(f"Total model parameters: {model.count_params()}\n")
        f.write(f"Training epochs: {len(history.history['loss'])}\n")
        f.write(f"Training time: {train_elapsed:.1f} seconds\n\n")
        f.write(f"Train/Test split: 80% training / 20% test\n")
        f.write(f"Validation: 20% split from training data (for early stopping)\n")
        f.write(f"Train/Test sizes: {X_train.shape[0]}/{X_test.shape[0]}\n\n")
        f.write("TEST SET PERFORMANCE\n")
        f.write("-" * 70 + "\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write("               Predicted Down  Predicted Up\n")
        f.write(f"Actual Down:   {cm[0,0]:>15} {cm[0,1]:>12}\n")
        f.write(f"Actual Up:     {cm[1,0]:>15} {cm[1,1]:>12}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred, target_names=['Down', 'Up']))

    print(f"✓ Saved results to: {output_dir / 'lstm_results.txt'}\n")

    # Total elapsed time
    total_elapsed = time.perf_counter() - start_time
    end_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print(f"End time: {end_ts}")
    print(f"Total elapsed: {total_elapsed/60:.2f} minutes")
    print("=" * 70)


if __name__ == "__main__":
    main()
