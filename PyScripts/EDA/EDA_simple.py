#!/usr/bin/env python3
"""
Simple EDA script generating correlation and volatility plots.
"""

import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from pathlib import Path
import sys
import os

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 8)

cwd = Path.cwd()
for _ in range(5):
    if cwd.name != "STAT-587-Final-Project":
        cwd = cwd.parent
    else:
        break
else:
    raise FileNotFoundError("Could not find correct workspace folder.")

# Create output/Used/png directory if it doesn't exist
output_dir = cwd / "output" 
output_dir.mkdir(parents=True, exist_ok=True)

sys.path.append(os.path.abspath(cwd / "PyScripts" / "Models"))
from H_prep import import_data, clean_data

# ── Command-line parameters (raw-only EDA, prefix retained for saved plots) ─
parser = argparse.ArgumentParser(description="EDA plots using raw data only.")
# parser.add_argument("--lookback_period",  type=int,   default=5,    help="Lookback window for EMA/VOL/VWAP features (default: 5)")
# parser.add_argument("--lag_period",       type=int,   nargs="+",    default=[1], help="Lag periods, e.g. --lag_period 1 2 (default: [1])")
# parser.add_argument("--no_extra_features",action="store_true",      help="Disable extra features (day of week, daily range, forward lags)")
# parser.add_argument("--raw",              action="store_true",       help="Return raw OHLCV data without engineered features")
# parser.add_argument("--corr_threshold",   type=float, default=0.95, help="Correlation threshold for dropping (default: 0.95)")
# parser.add_argument("--corr_level",       type=int,   default=1,    choices=[0, 1, 2, 3], help="Correlation drop level: 0=none, 1=before features, 2=after, 3=both (default: 1)")
parser.add_argument("--prefix",           type=str,   default="8yrs_",   help="Prefix for saved plot filenames (default: '8yrs_')")
args = parser.parse_args()
# args.extra_features = not args.no_extra_features
prefix = args.prefix

# Force raw-data-only behavior for this script while keeping the old options
# commented above for review.
raw_mode = True
extra_features = False
corr_threshold = 0.95
corr_level = 0
# lookback_period = args.lookback_period
# lag_period = args.lag_period
# extra_features = args.extra_features
# raw_mode = args.raw
# corr_threshold = args.corr_threshold
# corr_level = args.corr_level
# ────────────────────────────────────────────────────────────────────────────

lookup_df = pd.read_csv(cwd / "PyScripts" / "Data" / "stock_lookup_table.csv")

print("Loading data via clean_data...")
DATA, y_regression = import_data(
    testing=False,
    extra_features=extra_features,
    corr_level=corr_level,
)
X, y = clean_data(
    DATA=DATA,
    y_regression=y_regression,
    lookback_period=5,
    lag_period=[1],
    extra_features=extra_features,
    raw=raw_mode,
    corr_threshold=corr_threshold,
    corr_level=corr_level,
)

# # Extract daily returns (Close PC = percent change of close price)
# # In the current pipeline, import_data() already creates Close PC and
# # clean_data(..., raw=True) keeps it. This fallback keeps the EDA script
# # robust if that upstream behavior changes later.
# if 'Close PC' not in X.columns.get_level_values(0):
#     close_pc = (
#         X.loc[:, pd.IndexSlice['Close', :, :]]
#         .copy()
#         .pct_change()
#         .rename(columns={'Close': 'Close PC'}, level=0)
#     )
#     X = pd.concat([X, close_pc], axis=1).sort_index(axis=1)

returns_df = X.xs(key='Close PC', axis=1, level=0)
returns_df.columns = returns_df.columns.droplevel(0)
returns = returns_df.dropna()

print(f"Returns shape: {returns.shape}")

# Create sector mapping
sector_map = lookup_df.set_index('Ticker')['Sector'].to_dict()

# Get sector information for available stocks
available_tickers = returns.columns.tolist()
stock_sectors = {}
for ticker in available_tickers:
    if ticker in sector_map:
        stock_sectors[ticker] = sector_map[ticker]

print(f"\nSectors found: {set(stock_sectors.values())}")

# Plot 1: Sector correlation
print("\nGenerating sector correlation heat map...")
sector_dict = {}
for sector in set(stock_sectors.values()):
    sector_stocks = [t for t in available_tickers if stock_sectors.get(t) == sector]
    if len(sector_stocks) > 0:
        sector_dict[sector] = returns[sector_stocks].mean(axis=1)

sector_returns = pd.DataFrame(sector_dict)
sector_corr = sector_returns.corr()

# Create clustermap for sector correlation with hierarchical clustering
g = sns.clustermap(sector_corr,
                   cmap='RdYlGn',
                   center=0,
                   figsize=(10, 10),
                   cbar_kws={'label': 'Correlation'},
                   method='ward',
                   metric='euclidean',
                   xticklabels=True,
                   yticklabels=True,
                   linewidths=0.5,
                   linecolor='gray',
                   dendrogram_ratio=0.15)

g.ax_heatmap.set_title("Correlation of Sector Mean Returns\n(Hierarchical Clustering)", 
                       fontsize=14, pad=20, fontweight='bold')
g.ax_heatmap.set_xlabel("")
g.ax_heatmap.set_ylabel("")

plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=10)
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / f"{prefix}sector_correlation.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {prefix}sector_correlation.png")

# Plot 2-4: Individual sector correlations sorted by market cap (initially my idea, but I changed my mind to group by correlation for better presentation)
print("\nGenerating individual sector correlation heat maps...")

def plot_sector_with_marketcap(sector_name, stocks_in_sector, returns_df, output_dir):
    """
    Plot correlation matrix for a sector using hierarchical clustering.
    Stocks are automatically sorted by correlation patterns for better interpretability.
    """
    if len(stocks_in_sector) < 2:
        return
    
    # Get market cap ordering
    mc_data = lookup_df[lookup_df['Ticker'].isin(stocks_in_sector)]
    sorted_stocks = mc_data.sort_values('Market Cap (in tens of millions)', ascending=False)['Ticker'].tolist()
    
    # Filter to available stocks
    sorted_stocks = [s for s in sorted_stocks if s in returns_df.columns]
    
    if len(sorted_stocks) < 2:
        return
    
    # Calculate correlation
    sector_returns = returns_df[sorted_stocks]
    corr_matrix = sector_returns.corr()
    
    # Convert correlation to distance (for clustering)
    # Distance = 1 - correlation (so perfect correlation = 0 distance)
    distance_matrix = 1 - corr_matrix
    
    # Create clustermap with hierarchical clustering
    figsize = (min(16, max(10, len(sorted_stocks)*0.15)), 
               min(14, max(10, len(sorted_stocks)*0.15)))
    
    # Create the clustermap
    g = sns.clustermap(corr_matrix, 
                       cmap='coolwarm', 
                       center=0,
                       figsize=figsize,
                       cbar_kws={'label': 'Correlation'},
                       method='ward',  # Hierarchical clustering method
                       metric='euclidean',  # Distance metric
                       xticklabels=True,
                       yticklabels=True,
                       linewidths=0.5,
                       linecolor='gray',
                       dendrogram_ratio=0.15)
    
    # Customize appearance
    g.ax_heatmap.set_title(f"Intra-Sector Correlation: {sector_name}\n(Hierarchical Clustering - Similar Stocks Together)", 
                           fontsize=14, pad=20, fontweight='bold')
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")
    
    # Rotate labels for better readability
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=9)
    
    plt.tight_layout()
    
    filename = f"{prefix}sector_{sector_name.replace(' ', '_')}_clustered.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filename}")
    
    # Print top stocks
    print(f"\n  {sector_name} - Top 10 stocks by Market Cap:")
    mc_dict = dict(zip(lookup_df['Ticker'], lookup_df['Market Cap (in tens of millions)']))
    for i, stock in enumerate(sorted_stocks[:10], 1):
        print(f"    {i:2d}. {stock:6s} - ${mc_dict.get(stock, 0):10.2f}M")

# # THIS IS THE PART THAT THE OTHER CONTRIBUTOR USED BUT I DONT, WILL DELETE LATER WHEN DOUBLE CHECK AGAIN
# #Process top 3 sectors by number of stocks
# sectors_list = sorted(set(stock_sectors.values()), 
#                      key=lambda s: sum(1 for t in available_tickers if stock_sectors.get(t) == s), 
#                      reverse=True)

# for sector in sectors_list[:3]:
#     sector_stocks = [t for t in available_tickers if stock_sectors.get(t) == sector]
#     print(f"\nProcessing {sector} ({len(sector_stocks)} stocks)...")
#     plot_sector_with_marketcap(sector, sector_stocks, returns, output_dir)

# Plot 5: Sector volatility over time (the other contributor's idea. I used it in my report)
print("\nGenerating volatility plot...")
volatility_window = 21  # 1 month
sector_volatility = {}
for sector in sector_dict.keys():
    sector_stocks = [t for t in available_tickers if stock_sectors.get(t) == sector]
    sector_vol = returns[sector_stocks].std(axis=1).rolling(window=volatility_window).mean()
    sector_volatility[sector] = sector_vol

vol_df = pd.DataFrame(sector_volatility)

plt.figure(figsize=(14, 8))
for sector in vol_df.columns:
    plt.plot(vol_df.index, vol_df[sector], label=sector, linewidth=2, alpha=0.8)
plt.title(f"Sector Return Dispersion Over Time ({volatility_window}-day Rolling Average)")
plt.xlabel("Date")
plt.ylabel("Return Dispersion")
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / f"{prefix}sector_volatility_over_time.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {prefix}sector_volatility_over_time.png")

# Plot 5b: Smoothed Sector Volatility Over Time
print("\nGenerating smoothed sector volatility plot...")
smoothed_sector_vol = vol_df.rolling(window=40).mean()

plt.figure(figsize=(14, 8))
for sector in smoothed_sector_vol.columns:
    plt.plot(smoothed_sector_vol.index, smoothed_sector_vol[sector], label=sector)
plt.title("Smoothed Sector Volatility Over Time")
plt.xlabel("Date")
plt.ylabel("Return Dispersion")
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / f"{prefix}smoothed_sector_volatility.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {prefix}smoothed_sector_volatility.png")

# Plot 6: Sector volatility by day of week (seasonal impact) DONT USE FINALLY, TOO MUCH INFOR FOR A SHORT REPORT
# print("\nGenerating day-of-week volatility plot...")
# # Add day of week information
# returns.index = pd.to_datetime(returns.index)
# returns['day_of_week'] = returns.index.dayofweek
# returns['day_name'] = returns.index.day_name()

# day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
# day_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# # Calculate volatility by day of week for each sector
# plt.figure(figsize=(14, 8))
# for sector_idx, sector in enumerate(sector_dict.keys()):
#     sector_stocks = [t for t in available_tickers if stock_sectors.get(t) == sector]
#     dayofweek_volatility = []
    
#     for day in range(5):  # 0=Monday, 4=Friday
#         day_returns = returns[returns['day_of_week'] == day][sector_stocks]
#         if len(day_returns) > 0:
#             vol = day_returns.std().mean()
#         else:
#             vol = 0
#         dayofweek_volatility.append(vol)
    
#     plt.plot(range(5), dayofweek_volatility, marker='o', label=sector, linewidth=2, markersize=8, alpha=0.8)

# plt.xticks(range(5), day_names)
# plt.title("Sector Volatility by Day of Week (Seasonal Impact)")
# plt.xlabel("Day of Week")
# plt.ylabel("Average Volatility (Std Dev of Returns)")
# plt.legend(loc='best', ncol=2)
# plt.grid(True, alpha=0.3, axis='y')
# plt.tight_layout()
# plt.savefig(output_dir / f"{prefix}sector_volatility_by_day_of_week.png", dpi=300, bbox_inches='tight')
# plt.close()
# print(f"✓ Saved: {prefix}sector_volatility_by_day_of_week.png")

# # Remove day_of_week columns before final processing
# if 'day_of_week' in returns.columns:
#     returns_clean = returns.drop(columns=['day_of_week', 'day_name'])
# else:
#     returns_clean = returns.copy()

# # Plot 8: Sector volatility by month (seasonal impact) DONT USE FINALLY, TOO MUCH INFO FOR A SHORT REPORT
# print("\nGenerating monthly volatility plot...")
# returns_clean.index = pd.to_datetime(returns_clean.index)
# returns_clean['month'] = returns_clean.index.month
# returns_clean['month_name'] = returns_clean.index.strftime('%B')

# month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
#                'July', 'August', 'September', 'October', 'November', 'December']

# plt.figure(figsize=(14, 8))
# for sector in sector_dict.keys():
#     sector_stocks = [t for t in available_tickers if stock_sectors.get(t) == sector]
#     monthly_volatility = []
    
#     for month in range(1, 13):  # 1=January, 12=December
#         month_returns = returns_clean[returns_clean['month'] == month][sector_stocks]
#         if len(month_returns) > 0:
#             vol = month_returns.std().mean()
#         else:
#             vol = 0
#         monthly_volatility.append(vol)
    
#     plt.plot(range(1, 13), monthly_volatility, marker='o', label=sector, linewidth=2, markersize=8, alpha=0.8)

# plt.xticks(range(1, 13), [m[:3] for m in month_names])
# plt.title("Sector Volatility by Month (Seasonal/Monthly Impact)")
# plt.xlabel("Month")
# plt.ylabel("Average Volatility (Std Dev of Returns)")
# plt.legend(loc='best', ncol=2)
# plt.grid(True, alpha=0.3, axis='y')
# plt.tight_layout()
# plt.savefig(output_dir / f"{prefix}sector_volatility_by_month.png", dpi=300, bbox_inches='tight')
# plt.close()
# print(f"✓ Saved: {prefix}sector_volatility_by_month.png")

# # Plot 9: Heatmap showing volatility pattern by sector and month TOO MUCH INFOR FOR A SHORT REPORT, DONT USE FINALLY
# print("\nGenerating monthly volatility heatmap...")
# heatmap_monthly = []
# for sector in sector_dict.keys():
#     sector_stocks = [t for t in available_tickers if stock_sectors.get(t) == sector]
#     monthly_volatility = []
    
#     for month in range(1, 13):
#         month_returns = returns_clean[returns_clean['month'] == month][sector_stocks]
#         if len(month_returns) > 0:
#             vol = month_returns.std().mean()
#         else:
#             vol = 0
#         monthly_volatility.append(vol)
    
#     heatmap_monthly.append(monthly_volatility)

# heatmap_monthly_df = pd.DataFrame(heatmap_monthly, 
#                                  index=sector_dict.keys(), 
#                                  columns=[m[:3] for m in month_names])

# plt.figure(figsize=(14, 8))
# sns.heatmap(heatmap_monthly_df, annot=True, fmt='.4f', cmap='YlOrRd', 
#             cbar_kws={'label': 'Volatility'}, linewidths=0.5, linecolor='gray')
# plt.title("Sector Volatility Heatmap by Month\n(Showing Seasonal/Monthly Patterns)")
# plt.xlabel("Month")
# plt.ylabel("Sector")
# plt.tight_layout()
# plt.savefig(output_dir / f"{prefix}sector_volatility_heatmap_monthly.png", dpi=300, bbox_inches='tight')
# plt.close()
# print(f"✓ Saved: {prefix}sector_volatility_heatmap_monthly.png")

# Plot 10: Top 3 sectors by correlation with SP500
print("\nGenerating SP500 sector correlation ranking...")

# Align SPX returns with the returns index
spx_returns = y.reindex(returns.index).dropna()
common_index = returns.index.intersection(spx_returns.index)

sector_spx_corr = {}
for sector in sector_dict.keys():
    sector_stocks = [t for t in available_tickers if stock_sectors.get(t) == sector]
    sector_ret = returns.loc[common_index, sector_stocks].mean(axis=1)
    sector_spx_corr[sector] = sector_ret.corr(spx_returns.loc[common_index])

corr_series = pd.Series(sector_spx_corr).sort_values(key=lambda x: x.abs(), ascending=False)
top3_sectors = corr_series.head(3)

print("\nTop 3 sectors by absolute correlation with SP500:")
for i, (sector, corr_val) in enumerate(top3_sectors.items(), 1):
    print(f"  {i}. {sector}: {corr_val:.4f} (|{abs(corr_val):.4f}|)")

colors = ['#2ecc71' if s in top3_sectors.index else '#95a5a6' for s in corr_series.index]
plt.figure(figsize=(12, 6))
plt.bar(corr_series.index, corr_series.values, color=colors, edgecolor='white', linewidth=0.8)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.xticks(rotation=45, ha='right')
plt.title("Sector Correlation with SP500 Returns\n(Top 3 by Absolute Correlation Highlighted in Green)")
plt.xlabel("Sector")
plt.ylabel("Pearson Correlation")
plt.tight_layout()
plt.savefig(output_dir / f"{prefix}sector_sp500_correlation.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {prefix}sector_sp500_correlation.png")

# Plot 11: Top 3 sectors by correlation with SP500 (market-cap weighted sector returns)
print("\nGenerating SP500 sector correlation ranking (market-cap weighted sector returns)...")

market_cap_col = "Market Cap (in tens of millions)"
if market_cap_col not in lookup_df.columns:
    raise KeyError(f"Expected '{market_cap_col}' in stock lookup table.")

sector_spx_corr_weighted = {}
for sector in sector_dict.keys():
    sector_stocks = [t for t in available_tickers if stock_sectors.get(t) == sector]
    if len(sector_stocks) == 0:
        continue

    mc_data = lookup_df[lookup_df["Ticker"].isin(sector_stocks)][["Ticker", market_cap_col]].copy()
    mc_data = mc_data.dropna(subset=[market_cap_col])
    if mc_data.empty:
        continue

    weights = mc_data.set_index("Ticker")[market_cap_col]
    weights = weights[weights > 0]
    if weights.empty:
        continue

    valid_stocks = [s for s in weights.index if s in returns.columns]
    if len(valid_stocks) == 0:
        continue

    weights = weights.loc[valid_stocks]
    weights = weights / weights.sum()

    weighted_sector_ret = returns.loc[common_index, valid_stocks].mul(weights, axis=1).sum(axis=1)
    sector_spx_corr_weighted[sector] = weighted_sector_ret.corr(spx_returns.loc[common_index])

corr_series_weighted = pd.Series(sector_spx_corr_weighted).sort_values(key=lambda x: x.abs(), ascending=False)
top3_sectors_weighted = corr_series_weighted.head(3)

print("\nTop 3 sectors by absolute correlation with SP500 (market-cap weighted):")
for i, (sector, corr_val) in enumerate(top3_sectors_weighted.items(), 1):
    print(f"  {i}. {sector}: {corr_val:.4f} (|{abs(corr_val):.4f}|)")

colors_weighted = ['#2ecc71' if s in top3_sectors_weighted.index else '#95a5a6' for s in corr_series_weighted.index]
plt.figure(figsize=(12, 6))
plt.bar(corr_series_weighted.index, corr_series_weighted.values, color=colors_weighted, edgecolor='white', linewidth=0.8)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.xticks(rotation=45, ha='right')
plt.title("Sector Correlation with SP500 Returns\n(Market-Cap Weighted Sector Returns; Top 3 by Absolute Correlation in Green)")
plt.xlabel("Sector")
plt.ylabel("Pearson Correlation")
plt.tight_layout()
plt.savefig(output_dir / f"{prefix}sector_sp500_correlation_mcap_weighted.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {prefix}sector_sp500_correlation_mcap_weighted.png")

# Generate individual heatmaps for top 3 sectors by absolute SP500 correlation
print("\nGenerating individual heatmaps for top 3 SP500-correlated sectors...")
for sector in top3_sectors.index:
    sector_stocks = [t for t in available_tickers if stock_sectors.get(t) == sector]
    print(f"\nProcessing {sector} ({len(sector_stocks)} stocks, SP500 corr={sector_spx_corr[sector]:.4f})...")
    plot_sector_with_marketcap(sector, sector_stocks, returns, output_dir)

print("\n" + "="*60)
print("✓ All graphs successfully exported to output folder!")
print(f"Output directory: {output_dir}")
print("="*60)
