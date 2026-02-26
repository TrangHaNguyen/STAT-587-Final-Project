#!/usr/bin/env python3
"""
Recurrent Neural Network (RNN) for time series classification.
Uses a simple RNN layer to learn temporal patterns in stock data.
Predicts SPX binary direction (Up/Down) using y_binary.
"""

import numpy as np
from pathlib import Path
import sys

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


def reshape_for_rnn(X, sequence_length=5):
    """
    Create overlapping sequences for RNN: (samples, timesteps, features).

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
    print("RECURRENT NEURAL NETWORK (RNN) FOR TIME SERIES CLASSIFICATION")
    print("=" * 70)
    print("Task: Predict SPX binary direction (Up/Down) from stock time series")
    print("Architecture: Simple RNN layer + 7-day time window")
    print("=" * 70 + "\n")

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
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Reshape for RNN
    sequence_length = 7
    X_seq, seq_indices = reshape_for_rnn(X_scaled, sequence_length=sequence_length)
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

    # Build a simple RNN model
    model = keras.Sequential([
        layers.SimpleRNN(64, activation='tanh', input_shape=(sequence_length, X_seq.shape[2])),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("Model summary:")
    model.summary()

    # Train
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_proba >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\nTest set performance:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Save results
    output_dir = Path.cwd() / "output"
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "rnn_results.txt", 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("RNN MODEL RESULTS - TIME SERIES CLASSIFICATION\n")
        f.write("=" * 70 + "\n\n")
        f.write("MODEL CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Architecture: SimpleRNN(64) + Dropout(0.3) + Dense(1, sigmoid)\n")
        f.write(f"Sequence length: {sequence_length} timesteps (10-day window)\n")
        f.write(f"Input features: {X_seq.shape[2]}\n")
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

    print(f"\n✓ Saved results to: {output_dir / 'rnn_results.txt'}")


if __name__ == "__main__":
    main()
