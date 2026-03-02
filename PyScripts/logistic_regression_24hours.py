#!/usr/bin/env python3
"""
Logistic Regression with 24-hour Lagged Features.
Creates lag features from t-1 to t-24 for stock predictors only.
Performs 5-fold cross-validation to select optimal regularization parameter C.
"""

import warnings
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')
np.random.seed(42)


def load_hourly_data(parquet_file="hourly_data.parquet"):
    """
    Load hourly parquet data and prepare features/targets for classification.

    Returns:
        X: numpy array of predictors
        targets: dict with binary/SPX shifted targets
    """
    cwd = Path.cwd()
    for _ in range(5):
        if cwd.name != "STAT-587-Final-Project":
            cwd = cwd.parent
        else:
            break
    else:
        raise FileNotFoundError("Could not find correct workspace folder.")

    data_path = cwd / "PyScripts" / parquet_file
    if not data_path.exists():
        raise FileNotFoundError(f"Hourly parquet not found: {data_path}")

    print("------- Loading Hourly Data")
    print(f"Source file: {data_path.name}")
    table = pq.read_table(data_path)
    DATA = table.to_pandas()
    print(f"Loaded data shape: {DATA.shape}")

    idx = pd.IndexSlice

    stocks = DATA.loc[:, idx[:, 'Stocks', :]]
    to_drop = stocks.index[stocks.isna().all(axis=1)]
    DATA = DATA.drop(index=to_drop)

    X = DATA.loc[:, idx[['Close', 'High', 'Low', 'Open', 'Volume'], 'Stocks', :]].copy()

    spx_close = DATA.loc[:, idx['Close', 'Index', '^SPX']]
    spx_high = DATA.loc[:, idx['High', 'Index', '^SPX']]
    spx_low = DATA.loc[:, idx['Low', 'Index', '^SPX']]
    spx_open = DATA.loc[:, idx['Open', 'Index', '^SPX']]
    spx_volume = DATA.loc[:, idx['Volume', 'Index', '^SPX']]

    spx_open_current = spx_open.copy().reset_index(drop=True)

    y_binary = ((spx_close - spx_open) / spx_open).shift(-1)
    y_binary = (y_binary >= 0).astype(int).to_numpy()

    spx_close_shifted = spx_close.shift(-1).to_numpy()
    spx_high_shifted = spx_high.shift(-1).to_numpy()
    spx_low_shifted = spx_low.shift(-1).to_numpy()
    spx_open_shifted = spx_open.shift(-1).to_numpy()
    spx_volume_shifted = spx_volume.shift(-1).to_numpy()

    X = X[:-1]
    y_binary = y_binary[:-1]
    spx_close_shifted = spx_close_shifted[:-1]
    spx_high_shifted = spx_high_shifted[:-1]
    spx_low_shifted = spx_low_shifted[:-1]
    spx_open_shifted = spx_open_shifted[:-1]
    spx_volume_shifted = spx_volume_shifted[:-1]
    spx_open_current = spx_open_current[:-1].values.reshape(-1, 1)

    X = X.dropna(how='any', axis=1)
    X = X.ffill().bfill()

    X_reset = X.reset_index(drop=True)
    spx_open_df = pd.DataFrame(spx_open_current[:X_reset.shape[0]], columns=['spx_open_current'])

    day_of_week_full = pd.get_dummies(X.index.dayofweek, prefix='day_of_week', dtype=int)
    day_of_week_dummies = day_of_week_full.iloc[:, 1:].reset_index(drop=True)

    month_full = pd.get_dummies(X.index.month, prefix='month', dtype=int)
    month_dummies = month_full.iloc[:, 1:].reset_index(drop=True)

    X_flat = X_reset.copy()
    X_flat.columns = ['_'.join(map(str, col)) for col in X_flat.columns]
    X_combined = pd.concat([X_flat, spx_open_df, day_of_week_dummies, month_dummies], axis=1)

    targets = {
        'binary': y_binary,
        'spx_close': spx_close_shifted,
        'spx_high': spx_high_shifted,
        'spx_low': spx_low_shifted,
        'spx_open': spx_open_shifted,
        'spx_volume': spx_volume_shifted,
    }

    print("Finished Preparing Hourly Data -------\n")
    return X_combined.values, targets


def create_lagged_features(X, y, n_lags=24, temporal_count=17):
    """
    Create lagged features from t-1 to t-n_lags for stock predictors, including SPX Open.
    Exclude only temporal dummy variables (day-of-week and month) from lagging.

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Target vector of shape (n_samples,)
        n_lags: Number of lags to create (default: 24)
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

    # Convert to numpy arrays (handle both Series and array inputs)
    X_result = X_lagged.values.astype(np.float32, copy=False)
    y_result = y_lagged.values if hasattr(y_lagged, 'values') else np.asarray(y_lagged)
    
    return X_result, y_result


def main():
    print("=" * 70)
    print("LOGISTIC REGRESSION WITH 24-HOUR LAGGED FEATURES")
    print("=" * 70)
    print("Task: Predict SPX binary direction (Up/Down)")
    print("Features: Stock predictors lagged 24 hours + current temporal dummies")
    print("=" * 70 + "\n")

    start_time = time.perf_counter()
    start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Start time: {start_ts}\n")

    # Load hourly data from local parquet (no re-download)
    print("Loading data...")
    X_raw, targets = load_hourly_data("hourly_data.parquet")
    y = targets['binary']

    print(f"Original data shape: X={X_raw.shape}, y={y.shape}\n")

    # Create lagged features
    print("Creating 24-hour lagged features...")
    X_lagged, y_lagged = create_lagged_features(X_raw, y, n_lags=24, temporal_count=17)

    print(f"\nFinal lagged data shape: X={X_lagged.shape}, y={y_lagged.shape}")

    # Train/test split (80/20)
    print("\nSplitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_lagged, y_lagged, test_size=0.2, random_state=42, stratify=y_lagged
    )

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Train target distribution: Down={sum(y_train==0)}, Up={sum(y_train==1)}")
    print(f"Test target distribution: Down={sum(y_test==0)}, Up={sum(y_test==1)}\n")

    # Create 5-fold cross-validation
    print("=" * 70)
    print("Building Logistic Regression with 5-fold CV")
    print("=" * 70 + "\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # LogisticRegressionCV with pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegressionCV(
            Cs=10,
            cv=cv,
            penalty='l2',
            solver='lbfgs',
            random_state=42,
            max_iter=1000,
            verbose=1,
            n_jobs=-1
        ))
    ])

    print("Training Logistic Regression with cross-validation...\n")
    train_start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_elapsed = time.perf_counter() - train_start

    print(f"\n✓ Training completed in {train_elapsed:.1f} seconds ({train_elapsed/60:.2f} minutes)")

    # Get the trained classifier
    log_reg = pipeline.named_steps['classifier']
    print(f"\nOptimal C parameter selected: {log_reg.C_[0]:.6f}")
    print(f"Mean CV score (accuracy): {log_reg.scores_[1].mean(axis=0).max():.4f}")

    # Evaluate on test set
    print("\n" + "=" * 70)
    print("Evaluating on test set...")
    print("=" * 70 + "\n")

    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print("Test set performance:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"               Predicted Down  Predicted Up")
    print(f"Actual Down:   {cm[0,0]:>15} {cm[0,1]:>12}")
    print(f"Actual Up:     {cm[1,0]:>15} {cm[1,1]:>12}\n")

    # Plot confusion matrix
    output_dir = Path.cwd() / "output"
    output_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'],
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Logistic Regression (24-hour Lags)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(output_dir / 'logistic_regression_24hours_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save results to text file
    with open(output_dir / "logistic_regression_24hours_results.txt", 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("LOGISTIC REGRESSION WITH 24-HOUR LAGGED FEATURES\n")
        f.write("=" * 70 + "\n\n")
        f.write("MODEL CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Algorithm: Logistic Regression (L2 regularization)\n")
        f.write(f"Solver: lbfgs\n")
        f.write(f"Max iterations: 1000\n")
        f.write(f"Cross-validation: 5-fold stratified\n")
        f.write(f"Optimal C parameter: {log_reg.C_[0]:.6f}\n\n")
        f.write(f"FEATURES\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total features: {X_train.shape[1]}\n")
        f.write(f"  - Current stock features: {X_train.shape[1] // 25}  (approximately)\n")
        f.write(f"  - Lagged stock features (24 lags): {(X_train.shape[1] // 25) * 24}  (approximately)\n")
        f.write(f"  - Temporal features (day/month): 17\n\n")
        f.write(f"DATA SPLIT\n")
        f.write("-" * 70 + "\n")
        f.write(f"Train set: {X_train.shape[0]} samples (80%)\n")
        f.write(f"Test set: {X_test.shape[0]} samples (20%)\n")
        f.write(f"Train target: Down={sum(y_train==0)}, Up={sum(y_train==1)}\n")
        f.write(f"Test target: Down={sum(y_test==0)}, Up={sum(y_test==1)}\n\n")
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

    print(f"✓ Saved confusion matrix plot: {output_dir / 'logistic_regression_24hours_confusion_matrix.png'}")
    print(f"✓ Saved results to: {output_dir / 'logistic_regression_24hours_results.txt'}\n")

    # ========================================================================
    # L1 PENALTY (LASSO) LOGISTIC REGRESSION WITH 5-FOLD CV
    # ========================================================================
    print("\n" + "=" * 70)
    print("LOGISTIC REGRESSION WITH L1 PENALTY (LASSO) - 5-fold CV")
    print("=" * 70 + "\n")

    # LogisticRegressionCV with L1 penalty (uses liblinear solver)
    pipeline_l1 = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegressionCV(
            Cs=10,
            cv=cv,
            penalty='l1',
            solver='liblinear',  # liblinear is required for L1
            random_state=42,
            max_iter=1000,
            verbose=1,
            n_jobs=1  # Use single-threaded to avoid memory issues with L1
        ))
    ])

    print("Training Logistic Regression (L1) with cross-validation...\n")
    train_start_l1 = time.perf_counter()
    pipeline_l1.fit(X_train, y_train)
    train_elapsed_l1 = time.perf_counter() - train_start_l1

    print(f"\n✓ Training completed in {train_elapsed_l1:.1f} seconds ({train_elapsed_l1/60:.2f} minutes)")

    # Get the trained classifier (L1)
    log_reg_l1 = pipeline_l1.named_steps['classifier']
    print(f"\nOptimal C parameter selected (L1): {log_reg_l1.C_[0]:.6f}")
    print(f"Mean CV score (accuracy): {log_reg_l1.scores_[1].mean(axis=0).max():.4f}")

    # Evaluate on test set
    print("\n" + "=" * 70)
    print("Evaluating L1 model on test set...")
    print("=" * 70 + "\n")

    y_pred_l1 = pipeline_l1.predict(X_test)
    y_pred_proba_l1 = pipeline_l1.predict_proba(X_test)[:, 1]

    accuracy_l1 = accuracy_score(y_test, y_pred_l1)
    precision_l1 = precision_score(y_test, y_pred_l1)
    recall_l1 = recall_score(y_test, y_pred_l1)
    f1_l1 = f1_score(y_test, y_pred_l1)
    roc_auc_l1 = roc_auc_score(y_test, y_pred_proba_l1)

    print("Test set performance (L1 penalty):")
    print(f"Accuracy:  {accuracy_l1:.4f}")
    print(f"Precision: {precision_l1:.4f}")
    print(f"Recall:    {recall_l1:.4f}")
    print(f"F1-Score:  {f1_l1:.4f}")
    print(f"ROC-AUC:   {roc_auc_l1:.4f}\n")

    print("Classification Report (L1):")
    print(classification_report(y_test, y_pred_l1, target_names=['Down', 'Up']))

    cm_l1 = confusion_matrix(y_test, y_pred_l1)
    print("\nConfusion Matrix (L1):")
    print(f"               Predicted Down  Predicted Up")
    print(f"Actual Down:   {cm_l1[0,0]:>15} {cm_l1[0,1]:>12}")
    print(f"Actual Up:     {cm_l1[1,0]:>15} {cm_l1[1,1]:>12}\n")

    # Plot confusion matrix (L1)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_l1, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'],
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Logistic Regression L1 (24-hour Lags)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(output_dir / 'logistic_regression_24hours_l1_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save L1 results
    with open(output_dir / "logistic_regression_24hours_l1_results.txt", 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("LOGISTIC REGRESSION (L1 PENALTY) WITH 24-HOUR LAGGED FEATURES\n")
        f.write("=" * 70 + "\n\n")
        f.write("MODEL CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Algorithm: Logistic Regression (L1 regularization / LASSO)\n")
        f.write(f"Solver: liblinear\n")
        f.write(f"Max iterations: 1000\n")
        f.write(f"Cross-validation: 5-fold stratified\n")
        f.write(f"Optimal C parameter: {log_reg_l1.C_[0]:.6f}\n\n")
        f.write(f"FEATURES\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total features: {X_train.shape[1]}\n")
        f.write(f"  - Current stock features: {X_train.shape[1] // 25}  (approximately)\n")
        f.write(f"  - Lagged stock features (24 lags): {(X_train.shape[1] // 25) * 24}  (approximately)\n")
        f.write(f"  - Temporal features (day/month): 17\n")
        f.write(f"Note: L1 penalty performs feature selection (shrinks coefficients to 0)\n\n")
        f.write(f"DATA SPLIT\n")
        f.write("-" * 70 + "\n")
        f.write(f"Train set: {X_train.shape[0]} samples (80%)\n")
        f.write(f"Test set: {X_test.shape[0]} samples (20%)\n")
        f.write(f"Train target: Down={sum(y_train==0)}, Up={sum(y_train==1)}\n")
        f.write(f"Test target: Down={sum(y_test==0)}, Up={sum(y_test==1)}\n\n")
        f.write("TEST SET PERFORMANCE\n")
        f.write("-" * 70 + "\n")
        f.write(f"Accuracy:  {accuracy_l1:.4f}\n")
        f.write(f"Precision: {precision_l1:.4f}\n")
        f.write(f"Recall:    {recall_l1:.4f}\n")
        f.write(f"F1-Score:  {f1_l1:.4f}\n")
        f.write(f"ROC-AUC:   {roc_auc_l1:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write("               Predicted Down  Predicted Up\n")
        f.write(f"Actual Down:   {cm_l1[0,0]:>15} {cm_l1[0,1]:>12}\n")
        f.write(f"Actual Up:     {cm_l1[1,0]:>15} {cm_l1[1,1]:>12}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred_l1, target_names=['Down', 'Up']))

    print(f"✓ Saved L1 confusion matrix plot: {output_dir / 'logistic_regression_24hours_l1_confusion_matrix.png'}")
    print(f"✓ Saved L1 results to: {output_dir / 'logistic_regression_24hours_l1_results.txt'}\n")

    # ========================================================================
    # MODEL COMPARISON: L2 vs L1
    # ========================================================================
    print("\n" + "=" * 70)
    print("MODEL COMPARISON: L2 (Ridge) vs L1 (LASSO)")
    print("=" * 70 + "\n")

    comparison_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Training Time (sec)'],
        'L2 (Ridge)': [
            f"{accuracy:.4f}",
            f"{precision:.4f}",
            f"{recall:.4f}",
            f"{f1:.4f}",
            f"{roc_auc:.4f}",
            f"{train_elapsed:.2f}"
        ],
        'L1 (LASSO)': [
            f"{accuracy_l1:.4f}",
            f"{precision_l1:.4f}",
            f"{recall_l1:.4f}",
            f"{f1_l1:.4f}",
            f"{roc_auc_l1:.4f}",
            f"{train_elapsed_l1:.2f}"
        ]
    }

    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))

    # Save comparison
    with open(output_dir / "logistic_regression_24hours_comparison.txt", 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("LOGISTIC REGRESSION COMPARISON: L2 vs L1 PENALTIES\n")
        f.write("=" * 70 + "\n\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\nKEY DIFFERENCES\n")
        f.write("-" * 70 + "\n")
        f.write("L2 (Ridge) Regularization:\n")
        f.write("  - Shrinks large coefficients toward zero\n")
        f.write("  - Keeps all features (no exact zero coefficients)\n")
        f.write("  - Better for correlated features\n")
        f.write("  - Solver: lbfgs\n\n")
        f.write("L1 (LASSO) Regularization:\n")
        f.write("  - Performs automatic feature selection\n")
        f.write("  - Sets some coefficients exactly to zero\n")
        f.write("  - More interpretable (fewer features)\n")
        f.write("  - Solver: liblinear\n\n")
        f.write(f"Optimal C (L2): {log_reg.C_[0]:.6f}\n")
        f.write(f"Optimal C (L1): {log_reg_l1.C_[0]:.6f}\n")

    print(f"\n✓ Saved comparison to: {output_dir / 'logistic_regression_24hours_comparison.txt'}\n")

    # Total elapsed time
    total_elapsed = time.perf_counter() - start_time
    end_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print(f"End time: {end_ts}")
    print(f"Total elapsed: {total_elapsed/60:.2f} minutes")
    print("=" * 70)


if __name__ == "__main__":
    main()
