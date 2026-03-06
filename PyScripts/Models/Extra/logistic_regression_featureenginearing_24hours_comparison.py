#!/usr/bin/env python3
"""
Logistic Regression (L2 vs L1) with feature engineering and 24-hour lagged predictors.
Feature engineering is inspired by data_preprocessing_and_cleaning.py and adapted for hourly data.
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

LAG_STEPS = [1, 3, 6, 12, 24]
EMA_WINDOWS = [6, 12, 24]
VOL_WINDOWS = [6, 12, 24]
MAX_MIN_WINDOWS = [24, 72]
ROLLING_VWAP_WINDOWS = [6, 12, 24]
MAX_STOCK_TICKERS = 40


def _flatten_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    result.columns = [
        "_".join([str(part) for part in col if str(part) != ""]).replace(" ", "_")
        if isinstance(col, tuple)
        else str(col)
        for col in result.columns
    ]
    return result


def load_engineered_hourly_data(parquet_file: str = "hourly_data.parquet"):
    """
    Load hourly parquet data and build engineered features for classification.

    Returns:
        X: DataFrame of predictors
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
    data = table.to_pandas()
    print(f"Loaded data shape: {data.shape}")

    idx = pd.IndexSlice

    stocks = data.loc[:, idx[:, 'Stocks', :]]
    to_drop = stocks.index[stocks.isna().all(axis=1)]
    data = data.drop(index=to_drop)

    # Target from SPX next-bar direction
    spx_close = data.loc[:, idx['Close', 'Index', '^SPX']]
    spx_high = data.loc[:, idx['High', 'Index', '^SPX']]
    spx_low = data.loc[:, idx['Low', 'Index', '^SPX']]
    spx_open = data.loc[:, idx['Open', 'Index', '^SPX']]
    spx_volume = data.loc[:, idx['Volume', 'Index', '^SPX']]

    target_return = ((spx_close - spx_open) / spx_open).shift(-1).rename("target_return")

    available_stock_tickers = sorted(data.loc[:, idx[:, 'Stocks', :]].columns.get_level_values(2).unique())
    selected_stock_tickers = available_stock_tickers[:MAX_STOCK_TICKERS]
    print(f"Selected stock tickers for engineered features: {len(selected_stock_tickers)} / {len(available_stock_tickers)}")

    # Base stock features
    features = data.loc[:, idx[['Close', 'Open', 'High', 'Low', 'Volume'], 'Stocks', selected_stock_tickers]].copy()
    features = features.astype(np.float32)

    # Percent change features
    pct_features = data.loc[:, idx[['Close', 'Open', 'High', 'Low'], 'Stocks', selected_stock_tickers]].pct_change()
    pct_features = pct_features.rename(
        columns={metric: f"{metric} PC" for metric in ['Close', 'Open', 'High', 'Low']},
        level=0
    )
    features = pd.concat([features, pct_features], axis=1)
    print("Created percent change features.")

    # Daily range
    high_ = data.loc[:, idx['High', 'Stocks', selected_stock_tickers]]
    low_ = data.loc[:, idx['Low', 'Stocks', selected_stock_tickers]]
    range_df = pd.DataFrame(high_.values - low_.values, index=high_.index, columns=high_.columns)
    range_df = range_df.rename(columns={'High': 'Daily Range'}, level=0)
    features = pd.concat([features, range_df], axis=1)
    print("Created daily range features.")

    # Lagged percent-change features
    for metric in ['Close PC', 'Open PC', 'High PC', 'Low PC']:
        metric_df = features.loc[:, idx[metric, :, :]]
        for lag_period in LAG_STEPS:
            lagged = metric_df.shift(lag_period).rename(
                columns={metric: f"{metric} Lag {lag_period}"}, level=0
            )
            features = pd.concat([features, lagged], axis=1)
    print("Created lagged percent-change features.")

    # EMA + normalized volatility
    for metric in ['Close', 'Open', 'High', 'Low']:
        metric_df = features.loc[:, idx[metric, :, :]]
        for ema_window in EMA_WINDOWS:
            ema = metric_df.ewm(span=ema_window, adjust=False).mean().rename(
                columns={metric: f"{metric} EMA {ema_window}"}, level=0
            )
            features = pd.concat([features, ema], axis=1)

        for vol_window in VOL_WINDOWS:
            vol = metric_df.rolling(window=vol_window).std()
            ema = features.loc[:, idx[f"{metric} EMA {vol_window}", :, :]]
            scaled_vol = vol / ema.values
            scaled_vol = scaled_vol.rename(columns={metric: f"{metric} VOL {vol_window}"}, level=0)
            features = pd.concat([features, scaled_vol], axis=1)
    print("Created EMA and normalized volatility features.")

    # Max/Min channel features
    for max_min_window in MAX_MIN_WINDOWS:
        max_df = features.loc[:, idx['High', :, :]].rolling(window=max_min_window).max()
        min_df = features.loc[:, idx['Low', :, :]].rolling(window=max_min_window).min()
        max_df = max_df.rename(columns={'High': f"MAX {max_min_window}"}, level=0)
        min_df = min_df.rename(columns={'Low': f"MIN {max_min_window}"}, level=0)
        features = pd.concat([features, max_df, min_df], axis=1)

    for metric in ['Close', 'Open', 'High', 'Low']:
        for max_min_window in MAX_MIN_WINDOWS:
            max_df = features.loc[:, idx[f'MAX {max_min_window}', :, :]]
            min_df = features.loc[:, idx[f'MIN {max_min_window}', :, :]]
            metric_df = features.loc[:, idx[metric, :, :]]
            denominator = (max_df.values - min_df.values)
            channel_pos = np.divide(
                (metric_df.values - min_df.values),
                denominator,
                out=np.full_like(metric_df.values, np.nan, dtype=float),
                where=denominator != 0
            )
            channel_df = pd.DataFrame(channel_pos, index=features.index, columns=metric_df.columns)
            channel_df = channel_df.rename(
                columns={metric: f'Channel Position {metric} {max_min_window}'},
                level=0
            ).ffill().fillna(0.5)
            features = pd.concat([features, channel_df], axis=1)

    for max_min_window in MAX_MIN_WINDOWS:
        features.drop(columns=[f"MAX {max_min_window}", f"MIN {max_min_window}"], level=0, inplace=True)
    print("Created max/min channel position features.")

    # Rolling VWAP
    for vwap_window in ROLLING_VWAP_WINDOWS:
        high_vals = features.loc[:, idx['High', :, :]].values
        low_vals = features.loc[:, idx['Low', :, :]].values
        close_vals = features.loc[:, idx['Close', :, :]].values
        typical_price = (high_vals + low_vals + close_vals) / 3.0

        volume = features.loc[:, idx['Volume', :, :]]
        price_volume = typical_price * volume.values

        pv_roll = pd.DataFrame(price_volume, index=features.index, columns=volume.columns).rolling(vwap_window).sum()
        vol_roll = volume.rolling(vwap_window).sum()

        vwap = (pv_roll / vol_roll).rename(columns={'Volume': f'Rolling VWAP {vwap_window}'}, level=0)
        features = pd.concat([features, vwap], axis=1)
    print("Created rolling VWAP features.")

    # Rolling z-score
    for z_window in EMA_WINDOWS:
        for metric in ['Close', 'Open', 'High', 'Low']:
            price = features.loc[:, idx[metric, :, :]]
            ema = features.loc[:, idx[f"{metric} EMA {z_window}", :, :]]
            vol = features.loc[:, idx[f"{metric} VOL {z_window}", :, :]]
            z_score = (price.values - ema.values) / vol.values
            z_df = pd.DataFrame(z_score, index=features.index, columns=price.columns)
            z_df = z_df.rename(columns={metric: f"{metric} Z-Score {z_window}"}, level=0)
            features = pd.concat([features, z_df], axis=1)
    print("Created rolling z-score features.")

    # Temporal dummies
    day_of_week_full = pd.get_dummies(features.index.dayofweek, prefix='day_of_week', dtype=int)
    day_of_week_dummies = day_of_week_full.iloc[:, 1:].set_index(features.index)

    month_full = pd.get_dummies(features.index.month, prefix='month', dtype=int)
    month_dummies = month_full.iloc[:, 1:].set_index(features.index)

    # Current SPX open as predictor
    spx_open_current = spx_open.rename('spx_open_current')

    # Flatten multi-index feature columns and combine all predictors/target
    features_flat = _flatten_columns(features)
    combined = pd.concat(
        [features_flat, spx_open_current, day_of_week_dummies, month_dummies, target_return],
        axis=1
    )

    combined = combined.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    y_binary = (combined.pop('target_return') >= 0).astype(int).to_numpy()

    # Additional shifted SPX targets for compatibility with existing output structure
    targets = {
        'binary': y_binary,
        'spx_close': spx_close.shift(-1).reindex(combined.index).to_numpy(),
        'spx_high': spx_high.shift(-1).reindex(combined.index).to_numpy(),
        'spx_low': spx_low.shift(-1).reindex(combined.index).to_numpy(),
        'spx_open': spx_open.shift(-1).reindex(combined.index).to_numpy(),
        'spx_volume': spx_volume.shift(-1).reindex(combined.index).to_numpy(),
    }

    print(f"Engineered feature matrix shape (pre-lag): {combined.shape}")
    print("Finished Preparing Engineered Hourly Data -------\n")
    return combined, targets


def create_lagged_features(X, y, n_lags=24, temporal_count=17):
    """
    Create lagged features from t-1 to t-n_lags for non-temporal predictors.
    Temporal dummy variables are kept at current time (not lagged).
    """
    if not isinstance(X, pd.DataFrame):
        X_df = pd.DataFrame(X)
    else:
        X_df = X.copy()

    temporal_cols = [
        col for col in X_df.columns
        if 'day_of_week' in str(col) or 'month_' in str(col)
    ]

    if len(temporal_cols) == 0 and temporal_count > 0 and X_df.shape[1] >= temporal_count:
        temporal_cols = list(X_df.columns[-temporal_count:])

    base_prefixes = (
        'Close_Stocks_',
        'Open_Stocks_',
        'High_Stocks_',
        'Low_Stocks_',
        'Volume_Stocks_'
    )

    laggable_cols = [
        col for col in X_df.columns
        if str(col).startswith(base_prefixes) or str(col) == 'spx_open_current'
    ]

    static_cols = [
        col for col in X_df.columns
        if col not in temporal_cols and col not in laggable_cols
    ]

    print("\nFeature Separation:")
    print(f"  - Core lagged features: {len(laggable_cols)}")
    print(f"  - Engineered static features: {len(static_cols)}")
    print(f"  - Temporal features (NOT lagged): {len(temporal_cols)}")

    X_laggable = X_df[laggable_cols].copy()
    X_static = X_df[static_cols].copy() if static_cols else pd.DataFrame(index=X_df.index)
    X_temporal = X_df[temporal_cols].copy() if temporal_cols else pd.DataFrame(index=X_df.index)

    X_lagged = pd.concat([X_laggable, X_static], axis=1)
    for lag in range(1, n_lags + 1):
        X_lag = X_laggable.shift(lag)
        X_lag.columns = [f"{col}_lag{lag}" for col in X_laggable.columns]
        X_lagged = pd.concat([X_lagged, X_lag], axis=1)

    X_lagged = pd.concat([X_lagged, X_temporal], axis=1)

    valid_mask = X_lagged.notna().all(axis=1)
    X_lagged = X_lagged.loc[valid_mask]
    y_lagged = np.asarray(y)[valid_mask.to_numpy()]

    print("\nLagged Features Summary:")
    print(f"  - Core lagged features: {len(laggable_cols)}")
    print(f"  - Engineered static features: {len(static_cols)}")
    print(f"  - Lags created: {n_lags} (t-1 to t-{n_lags})")
    print(f"  - Added lagged features: {len(laggable_cols) * n_lags}")
    print(f"  - Temporal features: {len(temporal_cols)}")
    print(f"  - Total features after lagging: {X_lagged.shape[1]}")
    print(f"  - Samples retained: {X_lagged.shape[0]}")
    print(f"  - Target distribution: Down={int(np.sum(y_lagged==0))}, Up={int(np.sum(y_lagged==1))}")

    return X_lagged.values.astype(np.float32, copy=False), y_lagged


def main():
    print("=" * 80)
    print("LOGISTIC REGRESSION FEATUREENGINEARING 24H COMPARISON (L2 vs L1)")
    print("=" * 80)
    print("Task: Predict SPX binary direction (Up/Down)")
    print("Features: Engineered hourly predictors + 24-hour lagged features")
    print("=" * 80 + "\n")

    start_time = time.perf_counter()
    start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Start time: {start_ts}\n")

    print("Loading and engineering hourly data...")
    X_raw, targets = load_engineered_hourly_data("hourly_data.parquet")
    y = targets['binary']

    print(f"Original engineered shape: X={X_raw.shape}, y={y.shape}\n")

    print("Creating 24-hour lagged features...")
    X_lagged, y_lagged = create_lagged_features(X_raw, y, n_lags=24, temporal_count=17)

    print(f"\nFinal lagged data shape: X={X_lagged.shape}, y={y_lagged.shape}")

    print("\nSplitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_lagged, y_lagged, test_size=0.2, random_state=42, stratify=y_lagged
    )

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Train target distribution: Down={np.sum(y_train==0)}, Up={np.sum(y_train==1)}")
    print(f"Test target distribution: Down={np.sum(y_test==0)}, Up={np.sum(y_test==1)}\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("=" * 80)
    print("Training L2 LogisticRegressionCV")
    print("=" * 80 + "\n")

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

    train_start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_elapsed = time.perf_counter() - train_start

    log_reg = pipeline.named_steps['classifier']

    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\nL2 training time: {train_elapsed:.2f} sec")
    print(f"L2 optimal C: {log_reg.C_[0]:.6f}")
    print(f"L2 accuracy: {accuracy:.4f}")

    cm = confusion_matrix(y_test, y_pred)

    output_dir = Path.cwd() / "output"
    output_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'],
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Logistic Regression L2 (Featureenginearing 24h)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(output_dir / 'logistic_regression_featureenginearing_24hours_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    with open(output_dir / "logistic_regression_featureenginearing_24hours_results.txt", 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("LOGISTIC REGRESSION FEATUREENGINEARING 24H - L2\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total features: {X_train.shape[1]}\n")
        f.write(f"Train samples: {X_train.shape[0]}\n")
        f.write(f"Test samples: {X_test.shape[0]}\n")
        f.write(f"Optimal C: {log_reg.C_[0]:.6f}\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n")
        f.write(f"ROC-AUC:   {roc_auc:.4f}\n\n")
        f.write(classification_report(y_test, y_pred, target_names=['Down', 'Up']))

    print("\n" + "=" * 80)
    print("Training L1 LogisticRegressionCV")
    print("=" * 80 + "\n")

    pipeline_l1 = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegressionCV(
            Cs=10,
            cv=cv,
            penalty='l1',
            solver='liblinear',
            random_state=42,
            max_iter=1000,
            verbose=1,
            n_jobs=1
        ))
    ])

    train_start_l1 = time.perf_counter()
    pipeline_l1.fit(X_train, y_train)
    train_elapsed_l1 = time.perf_counter() - train_start_l1

    log_reg_l1 = pipeline_l1.named_steps['classifier']

    y_pred_l1 = pipeline_l1.predict(X_test)
    y_pred_proba_l1 = pipeline_l1.predict_proba(X_test)[:, 1]

    accuracy_l1 = accuracy_score(y_test, y_pred_l1)
    precision_l1 = precision_score(y_test, y_pred_l1)
    recall_l1 = recall_score(y_test, y_pred_l1)
    f1_l1 = f1_score(y_test, y_pred_l1)
    roc_auc_l1 = roc_auc_score(y_test, y_pred_proba_l1)

    print(f"\nL1 training time: {train_elapsed_l1:.2f} sec")
    print(f"L1 optimal C: {log_reg_l1.C_[0]:.6f}")
    print(f"L1 accuracy: {accuracy_l1:.4f}")

    cm_l1 = confusion_matrix(y_test, y_pred_l1)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_l1, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'],
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Logistic Regression L1 (Featureenginearing 24h)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(output_dir / 'logistic_regression_featureenginearing_24hours_l1_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    with open(output_dir / "logistic_regression_featureenginearing_24hours_l1_results.txt", 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("LOGISTIC REGRESSION FEATUREENGINEARING 24H - L1\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total features: {X_train.shape[1]}\n")
        f.write(f"Train samples: {X_train.shape[0]}\n")
        f.write(f"Test samples: {X_test.shape[0]}\n")
        f.write(f"Optimal C: {log_reg_l1.C_[0]:.6f}\n")
        f.write(f"Accuracy:  {accuracy_l1:.4f}\n")
        f.write(f"Precision: {precision_l1:.4f}\n")
        f.write(f"Recall:    {recall_l1:.4f}\n")
        f.write(f"F1-Score:  {f1_l1:.4f}\n")
        f.write(f"ROC-AUC:   {roc_auc_l1:.4f}\n\n")
        f.write(classification_report(y_test, y_pred_l1, target_names=['Down', 'Up']))

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
    print("\n" + "=" * 80)
    print("MODEL COMPARISON: L2 vs L1")
    print("=" * 80)
    print(comparison_df.to_string(index=False))

    with open(output_dir / "logistic_regression_featureenginearing_24hours_comparison.txt", 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("LOGISTIC REGRESSION FEATUREENGINEARING 24H COMPARISON: L2 vs L1\n")
        f.write("=" * 80 + "\n\n")
        f.write(comparison_df.to_string(index=False))

    total_elapsed = time.perf_counter() - start_time
    end_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "=" * 80)
    print(f"End time: {end_ts}")
    print(f"Total elapsed: {total_elapsed/60:.2f} minutes")
    print("=" * 80)


if __name__ == "__main__":
    main()
