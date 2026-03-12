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

mpl_config_dir = cwd / ".mplconfig"
mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

# Save the final LaTeX table directly under output/.
output_dir = cwd / "output"
output_dir.mkdir(parents=True, exist_ok=True)

sys.path.append(os.path.abspath(cwd / "PyScripts" / "Models"))
from H_prep import import_data, clean_data


def aggregate_features_by_sector(X: pd.DataFrame, sector_map: dict) -> pd.DataFrame:
    """Aggregate stock-level feature columns into sector-level feature means."""
    valid_mask = X.columns.get_level_values(2).map(
        lambda ticker: ticker in sector_map and pd.notna(sector_map[ticker])
    )
    sector_source = X.loc[:, valid_mask]

    feature_labels = sector_source.columns.get_level_values(0)
    sector_labels = sector_source.columns.get_level_values(2).map(sector_map)
    sector_X = sector_source.T.groupby([feature_labels, sector_labels]).mean().T
    sector_X.columns = sector_X.columns.set_names(["Feature", "Sector"])
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


def build_descriptive_stats_table(
    sector_returns: pd.DataFrame,
    y: pd.Series,
    selected_sectors: list[str],
) -> pd.DataFrame:
    """Create the final yearly summary table used in the report."""
    sector_returns_wide = sector_returns.xs("Close PC", axis=1, level="Feature")
    sector_returns_wide = sector_returns_wide.loc[:, sector_returns_wide.columns.intersection(selected_sectors)]

    count_by_year = y.groupby(y.index.year).size().rename("Count")
    prop_up_by_year = y.groupby(y.index.year).mean().rename("S&P 500 Prop. Up")
    sector_summary = sector_returns_wide.groupby(sector_returns_wide.index.year).agg(["mean", "std"])

    table = pd.concat([count_by_year, prop_up_by_year, sector_summary], axis=1)
    return table


def write_descriptive_stats_latex(table: pd.DataFrame, output_path: Path, selected_sectors: list[str]) -> None:
    """Write the final descriptive statistics table as LaTeX."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{tabular}{lrr|rr|rr|rr}",
        r"\hline",
        (
            r"Year & Count & \multicolumn{1}{c|}{S \&P 500 Prop. Up} "
            rf"& \multicolumn{{2}}{{c|}}{{{selected_sectors[0]}}} "
            rf"& \multicolumn{{2}}{{c|}}{{{selected_sectors[1]}}} "
            rf"& \multicolumn{{2}}{{c}}{{{selected_sectors[2]}}} \\"
        ),
        r" &  & Value & Mean & SD & Mean & SD & Mean & SD \\",
        r"\hline",
    ]

    for year, row in table.iterrows():
        lines.append(
            f"{year} & {int(row['Count'])} & {row['S&P 500 Prop. Up']:.4f} "
            f"& {row[(selected_sectors[0], 'mean')]:.4f} & {row[(selected_sectors[0], 'std')]:.4f} "
            f"& {row[(selected_sectors[1], 'mean')]:.4f} & {row[(selected_sectors[1], 'std')]:.4f} "
            f"& {row[(selected_sectors[2], 'mean')]:.4f} & {row[(selected_sectors[2], 'std')]:.4f} \\\\"
        )

    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\caption{Yearly selected descriptive statistics: trading-day count, S\&P 500 up-day proportion, and sector return moments}",
            r"\label{tab:descriptive_stats}",
            r"\end{table}",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n")


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
DATA, y_regression = import_data(
    testing=args.testing,
    extra_features=args.extra_features,
    cluster=args.cluster,
    n_clusters=args.n_clusters,
    corr_threshold=args.corr_threshold,
    corr_level=args.corr_level,
)
X, y = clean_data(
    DATA=DATA,
    y_regression=y_regression,
    lookback_period=args.lookback_period,
    lag_period=args.lag_period,
    extra_features=args.extra_features,
    raw=args.raw,
    corr_threshold=args.corr_threshold,
    corr_level=args.corr_level,
)
y = to_binary_class(y)
sector_X = aggregate_features_by_sector(X, sector_map)
# Keep both Feature and Sector levels so output columns identify variable + sector.
sector_returns = sector_X.xs(
    key="Close PC", axis=1, level="Feature", drop_level=False
).dropna()

label = "2 years" if args.testing else "8 years"
print(
    f"{label} | DATA shape: {DATA.shape} | X shape: {X.shape} | "
    f"Sector X shape: {sector_X.shape} | y shape: {y.shape} | "
    f"Sector Returns shape: {sector_returns.shape}"
)

sector_X = ensure_datetime_index(sector_X)
sector_returns = ensure_datetime_index(sector_returns)
y = ensure_datetime_index(y)

selected_sectors = ["Technology", "Healthcare", "Consumer Defensive"]
missing_sectors = [sector for sector in selected_sectors if sector not in sector_returns.columns.get_level_values("Sector")]
if missing_sectors:
    raise KeyError(f"Missing required sectors for descriptive_stats.tex: {missing_sectors}")

descriptive_stats_table = build_descriptive_stats_table(sector_returns, y, selected_sectors)
output_path = output_dir / "descriptive_stats.tex"
write_descriptive_stats_latex(descriptive_stats_table, output_path, selected_sectors)

print(f"Saved: {output_path}")
print(f"Output directory: {output_dir}")
