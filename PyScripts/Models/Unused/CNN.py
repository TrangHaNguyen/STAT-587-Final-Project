#!/usr/bin/env python3
"""
Convolutional Neural Network (CNN) for time series classification.
Lightweight implementation using scikit-learn's MLPClassifier.
Predicts SPX binary direction (Up/Down) based on stock features and temporal patterns.

Note: This uses a simplified CNN approach by:
1. Creating time-windowed sequences from 1D stock data
2. Flattening the sequences
3. Training a neural network with conv-like behavior via hidden layers
"""

import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import data loading function
from basic_model_original_data import load_original_data

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, f1_score)

import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)

def reshape_for_cnn(X, sequence_length=5):
    """
    Reshape 2D feature matrix into sequences for time series analysis.
    Creates overlapping time windows: (samples, timesteps * features)
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        sequence_length: Number of timesteps per sequence (default: 5 trading days)
    
    Returns:
        X_reshaped: 2D array of shape (n_sequences, sequence_length * n_features)
        indices: Start indices of each sequence (for tracking alignment with targets)
    """
    n_samples, n_features = X.shape
    n_sequences = n_samples - sequence_length + 1
    
    # Flatten the sequences into 1D feature vectors
    X_reshaped = np.zeros((n_sequences, sequence_length * n_features))
    indices = []
    
    for i in range(n_sequences):
        # Flatten each window: take timesteps [i, i+sequence_length) and flatten
        X_reshaped[i] = X[i:i + sequence_length].flatten()
        indices.append(i + sequence_length - 1)  # Index of the last (most recent) sample
    
    return X_reshaped, np.array(indices)

def main():
    print("="*70)
    print("CONVOLUTIONAL NEURAL NETWORK FOR TIME SERIES CLASSIFICATION")
    print("="*70)
    print("Task: Predict SPX binary direction (Up/Down) from stock time series")
    print("Method: Neural Network with temporal sequence features")
    print("="*70 + "\n")
    
    # Load data
    print("Loading data...")
    X_raw, targets = load_original_data()
    y = targets['binary']
    
    print(f"\nOriginal data shape: X={X_raw.shape}, y={y.shape}")
    print(f"Binary target distribution: Down={sum(y==0)}, Up={sum(y==1)}")
    print(f"Class balance: {sum(y==0)/len(y)*100:.1f}% Down, {sum(y==1)/len(y)*100:.1f}% Up\n")
    
    # Standardize features
    print("Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # Reshape for CNN (create time series sequences)
    sequence_length = 7  # Use 7 trading days as context
    print(f"Reshaping data for CNN with sequence_length={sequence_length}...")
    X_seq, seq_indices = reshape_for_cnn(X_scaled, sequence_length=sequence_length)
    y_seq = y[seq_indices]
    
    print(f"Reshaped data shape: X={X_seq.shape} (samples, flattened sequences)")
    print(f"  * Each sample: {sequence_length} trading days × {X_raw.shape[1]} features")
    print(f"  * Total features per sample: {X_seq.shape[1]}")
    print(f"Target shape: y={y_seq.shape}")
    print(f"Sequence distribution: Down={sum(y_seq==0)}, Up={sum(y_seq==1)}\n")
    
    # Train-test split
    print("Splitting data into train/validation/test sets...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_seq, y_seq, test_size=0.3, random_state=42, stratify=y_seq
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Train set size: {X_train.shape[0]} samples ({X_train.shape[0]/len(y_seq)*100:.1f}%)")
    print(f"Validation set size: {X_val.shape[0]} samples ({X_val.shape[0]/len(y_seq)*100:.1f}%)")
    print(f"Test set size: {X_test.shape[0]} samples ({X_test.shape[0]/len(y_seq)*100:.1f}%)\n")
    
    # Create and train model
    print("="*70)
    print("MODEL TRAINING")
    print("="*70 + "\n")
    
    print("Creating Multi-Layer Perceptron (MLP) model...")
    print("Architecture:")
    print("  - Input layer: {} features".format(X_seq.shape[1]))
    print("  - Hidden layer 1: 128 units (ReLU activation)")
    print("  - Hidden layer 2: 64 units (ReLU activation)")
    print("  - Output layer: 1 unit (Sigmoid activation, binary classification)")
    print("  - Regularization: L2, early stopping\n")
    
    # MLP model with architecture similar to simple CNN
    # Using 'lbfgs' optimizer for small data, 'adam' for larger data
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        batch_size=32,
        learning_rate_init=0.001,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.2,  # Use 20% for internal validation
        n_iter_no_change=15,  # Stop if no improvement for 15 iterations
        random_state=42,
        verbose=1,
        alpha=0.0001  # L2 regularization
    )
    
    print("Fitting model to training data...")
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    print("\n" + "="*70)
    print("MODEL EVALUATION ON TEST SET")
    print("="*70 + "\n")
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Also evaluate on validation set for history
    y_val_pred = model.predict(X_val)
    y_train_pred = model.predict(X_train)
    
    # Metrics for all sets
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
    print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Save results to file
    output_dir = Path.cwd() / "output"
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "cnn_result.txt", 'w') as f:
        f.write("="*70 + "\n")
        f.write("NEURAL NETWORK MODEL RESULTS - TIME SERIES CLASSIFICATION\n")
        f.write("="*70 + "\n\n")
        
        f.write("DATA SUMMARY\n")
        f.write("-"*70 + "\n")
        f.write(f"Original data shape: {X_raw.shape}\n")
        f.write(f"Sequence length: {sequence_length} trading days\n")
        f.write(f"Reshaped data shape: {X_seq.shape}\n")
        f.write(f"Train/Val/Test split: {X_train.shape[0]}/{X_val.shape[0]}/{X_test.shape[0]}\n\n")
        
        f.write("MODEL ARCHITECTURE\n")
        f.write("-"*70 + "\n")
        f.write("Multi-Layer Perceptron (MLP) with temporal sequence features:\n")
        f.write(f"Input Layer: {X_seq.shape[1]} features\n")
        f.write("  ({}d sequences × {} features, flattened)\n".format(sequence_length, X_raw.shape[1]))
        f.write("Hidden Layer 1: 128 units (ReLU activation)\n")
        f.write("Hidden Layer 2: 64 units (ReLU activation)\n")
        f.write("Output Layer: 1 unit (Sigmoid activation, binary classification)\n")
        f.write("Regularization: L2 (alpha=0.0001), Early Stopping (patience=15)\n")
        f.write("Optimizer: Adam (lr=0.001)\n\n")
        
        f.write("TEST SET PERFORMANCE\n")
        f.write("-"*70 + "\n")
        f.write(f"Train Accuracy:       {train_acc:.4f}\n")
        f.write(f"Validation Accuracy:  {val_acc:.4f}\n")
        f.write(f"Test Accuracy:        {test_acc:.4f}\n\n")
        f.write(f"Test Precision: {precision:.4f}\n")
        f.write(f"Test Recall:    {recall:.4f}\n")
        f.write(f"Test F1-Score:  {f1:.4f}\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write(f"               Predicted Down  Predicted Up\n")
        f.write(f"Actual Down:   {cm[0,0]:>15} {cm[0,1]:>12}\n")
        f.write(f"Actual Up:     {cm[1,0]:>15} {cm[1,1]:>12}\n\n")
        
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred, target_names=['Down', 'Up']))
    
    print(f"\n✓ Saved results to: {output_dir / 'cnn_result.txt'}")
    
    # Plot training loss history
    print("\nGenerating training history plots...")
    
    # Get loss history from model
    train_losses = model.loss_curve_[:len(model.loss_curve_)]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(train_losses, label='Training Loss', linewidth=2, color='steelblue')
    axes[0].set_title('Model Loss Over Iterations', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss (Binary Crossentropy)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy progression
    axes[1].axhline(y=train_acc, label='Train Accuracy', linewidth=2, linestyle='--', alpha=0.7)
    axes[1].axhline(y=val_acc, label='Validation Accuracy', linewidth=2, linestyle='--', alpha=0.7)
    axes[1].axhline(y=test_acc, label='Test Accuracy', linewidth=2, color='green', alpha=0.7)
    axes[1].set_title('Model Accuracy (Final)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Dataset')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0.4, 0.75])
    axes[1].set_xticks([])
    
    plt.tight_layout()
    plt.savefig(output_dir / "cnn_training_history.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved training history plot to: {output_dir / 'cnn_training_history.png'}")
    
    # Confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'],
                cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='gray')
    plt.title('Confusion Matrix - Neural Network Test Set Performance', fontsize=12, fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(output_dir / "cnn_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved confusion matrix to: {output_dir / 'cnn_confusion_matrix.png'}")
    
    # Save model coefficients and metadata
    import pickle
    model_metadata = {
        'model': model,
        'scaler': scaler,
        'sequence_length': sequence_length,
        'feature_count': X_raw.shape[1]
    }
    
    with open(output_dir / "cnn_model.pkl", 'wb') as f:
        pickle.dump(model_metadata, f)
    print(f"✓ Saved trained model to: {output_dir / 'cnn_model.pkl'}")
    
    print("\n" + "="*70)
    print("NEURAL NETWORK MODEL TRAINING COMPLETE")
    print("="*70)
    print("\nModel Details:")
    print(f"  - Trained for {model.n_iter_} iterations")
    print(f"  - Final training loss: {model.loss_:.6f}")
    print(f"  - Input features: {X_seq.shape[1]} (5-day sequences)")
    print(f"  - Classes: 2 (Market Down / Market Up)")

if __name__ == "__main__":
    main()
