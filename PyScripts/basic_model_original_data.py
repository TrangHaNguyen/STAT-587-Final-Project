#!/usr/bin/env python3
"""
Script to load original raw data without extensive preprocessing.
Useful for baseline models and quick testing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq
from PyScripts.helper_functions import get_cwd

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 8)

def load_original_data():
    """
    Load original raw data from parquet file with minimal preprocessing.
    
    Returns:
        X: Feature matrix (numpy array) with:
            - Stock features (5 metrics × ~476 stocks): Close, High, Low, Open, Volume
            - SPX Open (current day) as a predictor
            - Temporal features: day-of-week and month one-hot dummies (k-1 encoding)
        targets: Dictionary containing SPX target variables (next day's values):
            - 'binary': Binary classification (1=Up, 0=Down)
            - 'spx_close': Next day's SPX close price
            - 'spx_high': Next day's SPX high price
            - 'spx_low': Next day's SPX low price
            - 'spx_open': Next day's SPX open price (target, not same as feature input)
            - 'spx_volume': Next day's SPX volume
    """
    
    cwd = get_cwd("STAT-587-Final-Project")
    
    print("------- Loading Original Data")
    # Load raw parquet file
    table = pq.read_table(cwd / "PyScripts" / "raw_data.parquet")
    DATA = table.to_pandas()
    print(f"Loaded data shape: {DATA.shape}")
    print("\nFirst observation of original data:")
    print(DATA.iloc[0])
    
    # Print features information
    print("\n\nAvailable Features (Metrics):")
    idx = pd.IndexSlice
    print(DATA.columns.get_level_values(0).unique().tolist())
    
    print("\nAvailable Data Types:")
    print(DATA.columns.get_level_values(1).unique().tolist())
    
    print("\nNumber of Stocks:")
    stocks = DATA.columns.get_level_values(2)[DATA.columns.get_level_values(1) == 'Stocks'].unique()
    print(f"Total stocks: {len(stocks)}")
    print(f"Sample stocks: {stocks[:10].tolist()}")
    
    print("\nFinished Loading Original Data -------\n")
    
    print("------- Preparing Data for Training")
    idx = pd.IndexSlice
    
    # Remove rows with all NaN values (holidays)
    stocks = DATA.loc[:, idx[:, 'Stocks', :]]
    to_drop = stocks.index[stocks.isna().all(axis=1)]
    DATA = DATA.drop(index=to_drop)
    
    # Use all 5 features (Close, High, Low, Open, Volume) for all stocks only
    X = DATA.loc[:, idx[['Close', 'High', 'Low', 'Open', 'Volume'], 'Stocks', :]].copy()
    
    # Extract SPX features
    spx_close = DATA.loc[:, idx['Close', 'Index', '^SPX']]
    spx_high = DATA.loc[:, idx['High', 'Index', '^SPX']]
    spx_low = DATA.loc[:, idx['Low', 'Index', '^SPX']]
    spx_open = DATA.loc[:, idx['Open', 'Index', '^SPX']]
    spx_volume = DATA.loc[:, idx['Volume', 'Index', '^SPX']]
    
    # Add SPX Open (current day) as a predictor feature
    spx_open_current = spx_open.copy().reset_index(drop=True)
    
    # Create binary target: 1 if next day's SPX closes up from its open, 0 otherwise
    y_binary = ((spx_close - spx_open) / spx_open).shift(-1)
    y_binary = (y_binary >= 0).astype(int).to_numpy()
    
    # Shift all SPX target features to align with prediction (predicting next day's values)
    spx_close_shifted = spx_close.shift(-1).to_numpy()
    spx_high_shifted = spx_high.shift(-1).to_numpy()
    spx_low_shifted = spx_low.shift(-1).to_numpy()
    spx_open_shifted = spx_open.shift(-1).to_numpy()
    spx_volume_shifted = spx_volume.shift(-1).to_numpy()
    
    # Remove the last row (NaN target due to shift)
    X = X[:-1]
    y_binary = y_binary[:-1]
    spx_close_shifted = spx_close_shifted[:-1]
    spx_high_shifted = spx_high_shifted[:-1]
    spx_low_shifted = spx_low_shifted[:-1]
    spx_open_shifted = spx_open_shifted[:-1]
    spx_volume_shifted = spx_volume_shifted[:-1]
    spx_open_current = spx_open_current[:-1].values.reshape(-1, 1)
    
    # Handle any remaining NaN values in stocks (SPX should have no NaN)
    X = X.dropna(how='any', axis=1)  # Drop columns (stocks) with any NaN
    X = X.ffill().bfill()  # Fill any remaining rows with NaN
    
    # Add Day of Week and Month features as one-hot encoded dummy variables
    # Create k-1 dummies for k categories to avoid perfect multicollinearity
    # Reset index to ensure proper alignment with concat
    X_reset = X.reset_index(drop=True)
    
    # Add SPX Open as a feature
    spx_open_df = pd.DataFrame(spx_open_current[:X_reset.shape[0]], columns=['spx_open_current'])
    
    day_of_week_full = pd.get_dummies(X.index.dayofweek, prefix='day_of_week', dtype=int)
    day_of_week_dummies = day_of_week_full.iloc[:, 1:].reset_index(drop=True)  # Drop first column (Monday as reference)
    
    month_full = pd.get_dummies(X.index.month, prefix='month', dtype=int)
    month_dummies = month_full.iloc[:, 1:].reset_index(drop=True)  # Drop first column (January as reference)
    
    print(f"Added Day of Week one-hot encoding ({day_of_week_dummies.shape[1]} columns: Tuesday-Sunday, with Monday as reference)")
    print(f"Added Month one-hot encoding ({month_dummies.shape[1]} columns: February-December, with January as reference)")
    print(f"Added SPX Open (current day) as a predictor feature")
    
    # Flatten multi-level columns and concatenate with temporal features and SPX Open
    X_flat = X_reset.copy()
    X_flat.columns = ['_'.join(map(str, col)) for col in X_flat.columns]
    X_combined = pd.concat([X_flat, spx_open_df, day_of_week_dummies, month_dummies], axis=1)
    
    print(f"\nFinal X shape: {X_combined.shape}")
    print(f"Number of features per stock: 5 (Close, High, Low, Open, Volume)")
    print(f"Number of stocks with complete data: {X.shape[1] // 5}")
    print(f"Total features: {X_combined.shape[1]} ({X.shape[1]} stock features + 1 SPX Open + 6 day-of-week dummies + 11 month dummies)")
    print(f"\nFeature Components:")
    print(f"  - Stock features: {X.shape[1]} (Close, High, Low, Open, Volume from {X.shape[1]//5} stocks)")
    print(f"  - SPX Open (current day): 1 (predictor, not target)")
    print(f"  - Temporal features: 17 (6 day-of-week dummies + 11 month dummies)")
    print(f"\nTarget Variables (SPX Index - Next Day):")
    print(f"  - y_binary shape: {y_binary.shape} - Binary (1=Up, 0=Down)")
    print(f"  - spx_close shape: {spx_close_shifted.shape} - SPX Close price")
    print(f"  - spx_high shape: {spx_high_shifted.shape} - SPX High price")
    print(f"  - spx_low shape: {spx_low_shifted.shape} - SPX Low price")
    print(f"  - spx_open shape: {spx_open_shifted.shape} - SPX Open price (next day)")
    print(f"  - spx_volume shape: {spx_volume_shifted.shape} - SPX Volume")
    print(f"\nNote: Current day's SPX Open is now used as a PREDICTOR feature (spx_open_current)")
    print(f"      Next day's SPX Open is still available as a TARGET variable (spx_open in targets dict)")
    print(f"  - spx_volume shape: {spx_volume_shifted.shape} - SPX Volume")
    print(f"\nBinary target distribution - Class 0 (Down): {(y_binary == 0).sum()}, Class 1 (Up): {(y_binary == 1).sum()}")
    print("Finished Preparing Data -------\n")
    
    # Return X and dictionary of target variables
    targets = {
        'binary': y_binary,
        'spx_close': spx_close_shifted,
        'spx_high': spx_high_shifted,
        'spx_low': spx_low_shifted,
        'spx_open': spx_open_shifted,
        'spx_volume': spx_volume_shifted
    }
    
    return X_combined.values, targets


if __name__ == "__main__":
    X, targets = load_original_data()
    
    # Feature breakdown: stock features (÷5 for 5 metrics per stock) + 1 spx_open + 17 temporal
    num_stocks = (X.shape[1] - 17 - 1) // 5  # Subtract temporal features (17) and spx_open (1), then divide by 5
    
    print("="*60)
    print("FINAL DATA SHAPE REPORT")
    print("="*60)
    print(f"Feature Matrix (X):")
    print(f"  - Shape: {X.shape}")
    print(f"  - Observations: {X.shape[0]} trading days")
    print(f"  - Features: {X.shape[1]} total")
    print(f"    * Stock features: {num_stocks * 5} ({num_stocks} stocks × 5 metrics)")
    print(f"    * SPX Open (current day): 1 (PREDICTOR)")
    print(f"    * Temporal features: 17 (6 day-of-week dummies + 11 month dummies)")
    
    print(f"\nTarget Variables (Next Day SPX):")
    print(f"  - 'binary': Binary classification (shape: {targets['binary'].shape})")
    print(f"    * Class 0 (Down): {(targets['binary'] == 0).sum()} ({(targets['binary'] == 0).sum()/len(targets['binary'])*100:.1f}%)")
    print(f"    * Class 1 (Up): {(targets['binary'] == 1).sum()} ({(targets['binary'] == 1).sum()/len(targets['binary'])*100:.1f}%)")
    print(f"\n  - 'spx_close': SPX Close price (shape: {targets['spx_close'].shape})")
    print(f"  - 'spx_high': SPX High price (shape: {targets['spx_high'].shape})")
    print(f"  - 'spx_low': SPX Low price (shape: {targets['spx_low'].shape})")
    print(f"  - 'spx_open': SPX Open price - next day (shape: {targets['spx_open'].shape})")
    print(f"  - 'spx_volume': SPX Volume (shape: {targets['spx_volume'].shape})")
    print(f"\nNote: Current day's SPX Open is included in X as a PREDICTOR feature.")
    print(f"      Next day's SPX Open is available in targets['spx_open'].")
    
    print("\n" + "="*60)
    print("Usage Example:")
    print("  X, targets = load_original_data()")
    print("  y = targets['binary']  # For binary classification")
    print("  y = targets['spx_close']  # To predict SPX close price")
    print("="*60)
