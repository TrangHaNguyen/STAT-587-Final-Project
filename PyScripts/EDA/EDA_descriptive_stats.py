#!/usr/bin/env python3
"""
Descriptive statistics EDA script using the same clean_data pipeline as EDA_simple.py.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 8)

cwd = Path.cwd()
for _ in range(5):
    if cwd.name != "STAT-587-Final-Project":
        cwd = cwd.parent
    else:
        break
else:
    raise FileNotFoundError("Could not find correct workspace folder.")

# Save text/csv outputs under output/Used/txt
output_dir = cwd / "output" / "Used" / "txt"
output_dir.mkdir(parents=True, exist_ok=True)

sys.path.append(os.path.abspath(cwd / "PyScripts" / "Models"))
from H_prep import import_data, clean_data


def aggregate_features_by_sector(X: pd.DataFrame, sector_map: dict) -> pd.DataFrame:
    """Aggregate stock-level feature columns into sector-level feature means."""
    sector_feature_dict = {}
    available_tickers = set(X.columns.get_level_values(2))
    sectors = sorted({sector_map[t] for t in available_tickers if t in sector_map and pd.notna(sector_map[t])})

    for feature in X.columns.get_level_values(0).unique():
        feature_df = X.xs(key=feature, axis=1, level=0)
        if isinstance(feature_df.columns, pd.MultiIndex):
            feature_df = feature_df.droplevel(0, axis=1)
        feature_df = feature_df.loc[:, ~feature_df.columns.duplicated()]

        for sector in sectors:
            sector_tickers = [ticker for ticker in feature_df.columns if sector_map.get(ticker) == sector]
            if sector_tickers:
                sector_feature_dict[(feature, sector)] = feature_df[sector_tickers].mean(axis=1)

    sector_X = pd.DataFrame(sector_feature_dict)
    sector_X.columns = pd.MultiIndex.from_tuples(sector_X.columns, names=["Feature", "Sector"])
    return sector_X


def ensure_datetime_index(obj):
    """Ensure index is DatetimeIndex so yearly grouping is valid."""
    if not isinstance(obj.index, pd.DatetimeIndex):
        obj = obj.copy()
        obj.index = pd.to_datetime(obj.index)
    return obj


def to_binary_class(y: pd.Series) -> pd.Series:
    """Match target construction used by logistic_regression/random_forest."""
    return (y >= 0).astype(int)


# Command-line parameters (mirrors clean_data signature)
parser = argparse.ArgumentParser(description="Descriptive statistics with configurable clean_data options.")
parser.add_argument("--lookback_period", type=int, default=5, help="Lookback window for EMA/VOL/VWAP features (default: 5)")
parser.add_argument("--lag_period", type=int, nargs="+", default=[1], help="Lag periods, e.g. --lag_period 1 2 (default: [1])")
parser.add_argument("--no_extra_features", action="store_true", help="Disable extra features (day of week, daily range, forward lags)")
parser.add_argument("--raw", action="store_true", help="Return raw OHLCV data without engineered features")
parser.add_argument("--cluster", action="store_true", help="Apply KMeans clustering to reduce stocks")
parser.add_argument("--n_clusters", type=int, default=100, help="Number of KMeans clusters (default: 100)")
parser.add_argument("--corr", action="store_true", help="Drop highly correlated stocks/features")
parser.add_argument("--corr_threshold", type=float, default=0.95, help="Correlation threshold for dropping (default: 0.95)")
parser.add_argument("--corr_level", type=int, default=1, choices=[1, 2, 3], help="Correlation drop level: 1=before features, 2=after, 3=both (default: 1)")
parser.add_argument("--testing", action="store_true", help="Use 2-year dataset instead of 8-year dataset")
parser.add_argument("--prefix", type=str, default="", help="Optional extra prefix appended after the dataset prefix")
args = parser.parse_args()
args.extra_features = not args.no_extra_features
dataset_prefix = "2yrs_" if args.testing else "8yrs_"
prefix = f"{dataset_prefix}{args.prefix}"
lookup_df = pd.read_csv(cwd / "PyScripts" / "Data" / "stock_lookup_table.csv")
sector_map = lookup_df.set_index("Ticker")["Sector"].to_dict()

print("Loading data via clean_data...")
results = {}
for testing_mode, label in [(True, "2 years"), (False, "8 years")]:
    DATA = import_data(testing=testing_mode)
    X_mode, y_mode = clean_data(
        DATA=DATA,
        lookback_period=args.lookback_period,
        lag_period=args.lag_period,
        extra_features=args.extra_features,
        raw=True,
        cluster=args.cluster,
        n_clusters=args.n_clusters,
        corr=args.corr,
        corr_threshold=args.corr_threshold,
        corr_level=args.corr_level,
    )
    y_mode = to_binary_class(y_mode)
    sector_X_mode = aggregate_features_by_sector(X_mode, sector_map)
    # Keep both Feature and Sector levels so output columns identify variable + sector.
    sector_returns_mode = sector_X_mode.xs(
        key="Close PC", axis=1, level="Feature", drop_level=False
    ).dropna()
    results[testing_mode] = (sector_X_mode, y_mode, sector_returns_mode)

    print(
        f"{label} | DATA shape: {DATA.shape} | X shape: {X_mode.shape} | "
        f"Sector X shape: {sector_X_mode.shape} | y shape: {y_mode.shape} | "
        f"Sector Returns shape: {sector_returns_mode.shape}"
    )

sector_X, y, sector_returns = results[args.testing]
sector_X = ensure_datetime_index(sector_X)
sector_returns = ensure_datetime_index(sector_returns)
y = ensure_datetime_index(y)

# Export descriptive statistics by year
X_stats_by_year = sector_X.groupby(sector_X.index.year).describe()
y_stats_by_year = y.groupby(y.index.year).describe()
returns_stats_by_year = sector_returns.groupby(sector_returns.index.year).describe()

X_stats_year_path = output_dir / f"{prefix}X_sector_descriptive_stats_by_year.csv"
y_stats_year_path = output_dir / f"{prefix}y_descriptive_stats_by_year.csv"
returns_stats_year_path = output_dir / f"{prefix}returns_sector_descriptive_stats_by_year.csv"

X_stats_by_year.to_csv(X_stats_year_path)
y_stats_by_year.to_csv(y_stats_year_path)
returns_stats_by_year.to_csv(returns_stats_year_path)

print(f"Saved: {X_stats_year_path}")
print(f"Saved: {y_stats_year_path}")
print(f"Saved: {returns_stats_year_path}")
print(f"Output directory: {output_dir}")
