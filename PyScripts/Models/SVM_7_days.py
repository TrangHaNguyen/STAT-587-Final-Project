#!/usr/bin/env python3
"""
Support Vector Machine (SVM) Classification with 7-day Lagged Features.
Creates lag features from t-1 to t-7 for stock predictors only.
Performs manual 5-fold cross-validation to select C from: (0.01, 0.1, 10)
"""

import warnings
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from basic_model_original_data import load_original_data

warnings.filterwarnings('ignore')
np.random.seed(42)


def create_lagged_features(X, y, n_lags=7, temporal_count=17):
    """
    Create lagged features from t-1 to t-n_lags for stock predictors, including SPX Open.
    Exclude only temporal dummy variables (day-of-week and month) from lagging.

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Target vector of shape (n_samples,)
        n_lags: Number of lags to create (default: 7)
        temporal_count: Number of temporal dummy columns at the end (default: 17)

    Returns:
        X_lagged: Feature matrix with lagged stock features (including SPX Open) + current temporal dummies
        y_lagged: Aligned target vector (with first n_lags rows removed)
    """
    if not isinstance(X, pd.DataFrame):
        X_df = pd.DataFrame(X)
    else:
        X_df = X.copy()

    # Identify temporal features (only day-of-week and month, NOT SPX Open)
    # SPX Open should be lagged like other stock features
    temporal_cols = [col for col in X_df.columns
                     if 'day_of_week' in str(col) or 'month' in str(col)]

    # Fallback: assume last temporal_count columns are temporal dummies
    if len(temporal_cols) == 0 and X_df.shape[1] >= temporal_count:
        temporal_cols = list(X_df.columns[-temporal_count:])

    stock_cols = [col for col in X_df.columns if col not in temporal_cols]

    print("\nFeature Separation:")
    print(f"  - Stock features (to be lagged): {len(stock_cols)}")
    print(f"  - Temporal features (NOT lagged): {len(temporal_cols)}")

    X_stock = X_df[stock_cols].copy()
    X_temporal = X_df[temporal_cols].copy()

    X_lagged = X_stock.copy()
    for lag in range(1, n_lags + 1):
        X_lag = X_stock.shift(lag)
        X_lag.columns = [f"{col}_lag{lag}" for col in X_stock.columns]
        X_lagged = pd.concat([X_lagged, X_lag], axis=1)

    # Add temporal features without lagging
    X_lagged = pd.concat([X_lagged, X_temporal], axis=1)

    X_lagged = X_lagged.dropna()

    y_lagged = y[n_lags:len(y)]
    y_lagged = y_lagged[:len(X_lagged)]

    print("\nLagged Features Summary:")
    print(f"  - Stock features: {len(stock_cols)}")
    print(f"  - Lags created: {n_lags} (t-1 to t-{n_lags})")
    print(f"  - Lagged stock features: {len(stock_cols) * n_lags}")
    print(f"  - Temporal features (current day/month): {len(temporal_cols)}")
    print(f"  - Total features after lagging: {X_lagged.shape[1]}")
    print(f"    * {len(stock_cols)} (current day) + {len(stock_cols) * n_lags} (lags) + {len(temporal_cols)} (temporal)")
    print(f"  - Samples retained (after removing NaN): {X_lagged.shape[0]}")
    print(f"  - Target distribution: Down={sum(y_lagged==0)}, Up={sum(y_lagged==1)}")

    return X_lagged.values.astype(np.float32, copy=False), y_lagged


def main():
    start_time = time.perf_counter()
    start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print("SUPPORT VECTOR MACHINE WITH 7-DAY LAGGED FEATURES")
    print("=" * 70)
    print("Task: Binary classification of SPX direction with lagged predictors")
    print("Grid Search: C in (0.01, 0.1, 10)")
    print("Cross-Validation: 5-fold stratified")
    print("=" * 70 + "\n")
    print(f"Start time: {start_ts}\n")

    print("Loading original data...")
    X_raw, targets = load_original_data()
    y = targets['binary']

    print(f"Original data shape: X={X_raw.shape}, y={y.shape}")
    print(f"Target distribution: Down={sum(y==0)}, Up={sum(y==1)}")
    print(f"Class balance: {sum(y==0)/len(y)*100:.1f}% Down, {sum(y==1)/len(y)*100:.1f}% Up")

    print("\n" + "=" * 70)
    print("CREATING 7-DAY LAGGED FEATURES")
    print("=" * 70)

    X_lagged, y_lagged = create_lagged_features(X_raw, y, n_lags=7, temporal_count=17)

    print("\nSplitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_lagged, y_lagged, test_size=0.2, random_state=42, stratify=y_lagged
    )

    print(f"Train set size: {X_train.shape[0]} samples ({X_train.shape[0]/len(y_lagged)*100:.1f}%)")
    print(f"Test set size: {X_test.shape[0]} samples ({X_test.shape[0]/len(y_lagged)*100:.1f}%)")
    print(f"  - Train: Down={sum(y_train==0)}, Up={sum(y_train==1)}")
    print(f"  - Test: Down={sum(y_test==0)}, Up={sum(y_test==1)}")

    print("\n" + "=" * 70)
    print("GRID SEARCH WITH 5-FOLD CROSS-VALIDATION")
    print("=" * 70 + "\n")

    C_values = [0.01, 0.1, 10]
    print(f"Parameter Grid: C = {C_values}")
    print("Kernel: RBF (Radial Basis Function - Nonlinear)")
    print("WARNING: RBF kernel is more memory-intensive than linear kernel")
    print("Cross-Validation: 5-fold Stratified\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_mean_scores = []
    cv_std_scores = []

    print("Fitting SVM with manual CV (memory-safe)...")
    for c_idx, C in enumerate(C_values, start=1):
        c_start = time.perf_counter()
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] C={C} ({c_idx}/{len(C_values)})")
        fold_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), start=1):
            fold_start = time.perf_counter()
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_val_scaled = scaler.transform(X_val)

            model = SVC(C=C, kernel='rbf', gamma='scale', max_iter=5000, random_state=42)
            model.fit(X_tr_scaled, y_tr)

            y_val_pred = model.predict(X_val_scaled)
            fold_scores.append(f1_score(y_val, y_val_pred))

            fold_elapsed = time.perf_counter() - fold_start
            print(f"  Fold {fold_idx}/5 done in {fold_elapsed:.1f}s")

        cv_mean_scores.append(np.mean(fold_scores))
        cv_std_scores.append(np.std(fold_scores))
        c_elapsed = time.perf_counter() - c_start
        print(f"  C={C} summary: mean F1={cv_mean_scores[-1]:.4f}, time={c_elapsed:.1f}s")

    print("\n" + "=" * 70)
    print("GRID SEARCH RESULTS")
    print("=" * 70 + "\n")

    cv_results = pd.DataFrame({
        'C': C_values,
        'mean_test_score': cv_mean_scores,
        'std_test_score': cv_std_scores,
    })

    print("Cross-Validation Scores for each C value:")
    print("-" * 70)
    for i, C in enumerate(C_values):
        mean_score = cv_results.loc[i, 'mean_test_score']
        std_score = cv_results.loc[i, 'std_test_score']
        print(f"  C={C:>5} | Mean F1-Score: {mean_score:.4f} (+/- {std_score:.4f})")

    best_idx = int(np.argmax(cv_mean_scores))
    best_C = C_values[best_idx]
    best_cv_score = cv_mean_scores[best_idx]

    print(f"\nOptimal C: {best_C}")
    print(f"Best CV F1-Score: {best_cv_score:.4f}\n")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    best_model = SVC(C=best_C, kernel='rbf', gamma='scale', max_iter=5000, random_state=42)
    best_model.fit(X_train_scaled, y_train)

    print("=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70 + "\n")

    y_pred = best_model.predict(X_test_scaled)
    y_decision = best_model.decision_function(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    try:
        roc_auc = roc_auc_score(y_test, y_decision)
    except Exception:
        roc_auc = np.nan

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    output_dir = Path.cwd() / "output"
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "SVM_results_original_data.txt", 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("SVM WITH 7-DAY LAGGED FEATURES - RESULTS\n")
        f.write("=" * 70 + "\n\n")

        f.write("DATA SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write(f"Original data shape: {X_raw.shape}\n")
        f.write(f"Number of lags: 7 (t-1 to t-7)\n")
        f.write(f"Features after lagging: {X_lagged.shape[1]}\n")
        f.write(f"Samples after removing NaN: {X_lagged.shape[0]}\n")
        f.write(f"Train/Test split: {X_train.shape[0]}/{X_test.shape[0]}\n")
        f.write(f"  - Training samples: {X_train.shape[0]}\n")
        f.write(f"    * Down: {sum(y_train==0)} ({sum(y_train==0)/len(y_train)*100:.1f}%)\n")
        f.write(f"    * Up: {sum(y_train==1)} ({sum(y_train==1)/len(y_train)*100:.1f}%)\n")
        f.write(f"  - Test samples: {X_test.shape[0]}\n")
        f.write(f"    * Down: {sum(y_test==0)} ({sum(y_test==0)/len(y_test)*100:.1f}%)\n")
        f.write(f"    * Up: {sum(y_test==1)} ({sum(y_test==1)/len(y_test)*100:.1f}%)\n\n")

        f.write("MODEL CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write("Algorithm: Support Vector Machine (SVM)\n")
        f.write("Kernel: RBF (Radial Basis Function - Nonlinear)\n")
        f.write("Probability: False (using decision function for ROC-AUC)\n")
        f.write("Parameter Grid:\n")
        f.write("  - C: [0.01, 0.1, 10]\n")
        f.write("Cross-Validation: 5-fold Stratified\n")
        f.write("Scoring Metric: F1-Score\n\n")

        f.write("CROSS-VALIDATION RESULTS\n")
        f.write("-" * 70 + "\n")
        for i, C in enumerate(C_values):
            mean_score = cv_results.loc[i, 'mean_test_score']
            std_score = cv_results.loc[i, 'std_test_score']
            f.write(f"C={C:>5} | Mean F1-Score: {mean_score:.4f} (+/- {std_score:.4f})\n")

        f.write(f"\nOptimal C: {best_C}\n")
        f.write(f"Best CV F1-Score: {best_cv_score:.4f}\n\n")

        f.write("TEST SET PERFORMANCE\n")
        f.write("-" * 70 + "\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n")
        f.write(f"ROC-AUC:   {roc_auc:.4f}\n\n")

        f.write("Confusion Matrix:\n")
        f.write("               Predicted Down  Predicted Up\n")
        f.write(f"Actual Down:   {cm[0,0]:>15} {cm[0,1]:>12}\n")
        f.write(f"Actual Up:     {cm[1,0]:>15} {cm[1,1]:>12}\n\n")

        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred, target_names=['Down', 'Up']))

    print(f"\n✓ Saved results to: {output_dir / 'SVM_results_original_data.txt'}")

    print("\nGenerating GridSearchCV visualization...")
    plt.figure(figsize=(10, 6))
    mean_scores = [cv_results.loc[i, 'mean_test_score'] for i in range(len(C_values))]
    std_scores = [cv_results.loc[i, 'std_test_score'] for i in range(len(C_values))]

    plt.errorbar(range(len(C_values)), mean_scores, yerr=std_scores,
                 marker='o', markersize=10, capsize=5, capthick=2, linewidth=2)
    plt.xticks(range(len(C_values)), [f"C={c}" for c in C_values], fontsize=11)
    plt.ylabel('F1-Score', fontsize=12)
    plt.title('GridSearchCV Results: F1-Score vs C Parameter\n5-Fold Cross-Validation',
              fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.axhline(y=best_cv_score, color='green', linestyle='--',
                linewidth=2, label=f"Best Score: {best_cv_score:.4f}")
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "svm7days_gridsearch.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved GridSearchCV plot to: {output_dir / 'svm7days_gridsearch.png'}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'],
                cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='gray')
    plt.title('Confusion Matrix - SVM Test Set Performance\n(7-Day Lagged Features)',
              fontsize=12, fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(output_dir / "svm7days_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved confusion matrix to: {output_dir / 'svm7days_confusion_matrix.png'}")

    cv_results_export = cv_results[['C', 'mean_test_score', 'std_test_score']].copy()
    cv_results_export.columns = ['C', 'Mean_F1_Score', 'Std_F1_Score']
    cv_results_export.to_csv(output_dir / "svm7days_cv_results.csv", index=False)
    print(f"✓ Saved CV results to: {output_dir / 'svm7days_cv_results.csv'}")

    print("\n" + "=" * 70)
    print("SVM WITH 7-DAY LAGGED FEATURES - COMPLETE")
    print("=" * 70)
    print("\nSummary:")
    print(f"  - Features: {X_lagged.shape[1]} (original + 7-day lags)")
    print(f"  - Training samples: {X_train.shape[0]}")
    print(f"  - Optimal C: {best_C}")
    print(f"  - Best CV F1-Score: {best_cv_score:.4f}")
    print(f"  - Test Accuracy: {accuracy:.4f}")
    print(f"  - Test F1-Score: {f1:.4f}")
    total_elapsed = time.perf_counter() - start_time
    end_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nEnd time: {end_ts}")
    print(f"Total elapsed: {total_elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
