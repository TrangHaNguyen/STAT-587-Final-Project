#!/usr/bin/env python3
"""
EDA script using original data with market cap sorting - lighter version.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from basic_model_original_data import load_original_data
from pathlib import Path

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

lookup_df = pd.read_csv(cwd / "PyScripts" / "stock_lookup_table.csv")

# Create output directory if it doesn't exist
output_dir = cwd / "output"
output_dir.mkdir(exist_ok=True)

print("Loading original data...")
X, targets = load_original_data()

# Convert to DataFrame for easier manipulation
X_df = pd.DataFrame(X)
print(f"Data shape: {X_df.shape}")

# Extract stock close prices (columns 0 to 2375 are stock features, 5 per stock)
num_stock_features = 476 * 5
stock_cols = list(range(num_stock_features))
X_stocks = X_df.iloc[:, stock_cols]

# Reorganize into (date, metric_ticker) format
# The multi-level columns from basic_model_original_data are flattened
# We need to extract individual stock prices

print("\nGenerating correlation heat maps...")

# Create simple correlation matrix of stock close prices
# Extract close prices for each stock (every 5th column starting from 0)
stock_tickers = lookup_df['Ticker'].tolist()
close_prices = {}
for i, ticker in enumerate(stock_tickers[:476]):
    col_idx = i * 5  # Close is the first feature for each stock
    if col_idx < X_stocks.shape[1]:
        close_prices[ticker] = X_stocks.iloc[:, col_idx]

close_df = pd.DataFrame(close_prices)

# Calculate returns
returns_df = close_df.pct_change().dropna()

# Calculate daily returns
daily_returns = returns_df.copy()

# Create sector mapping
sector_map = lookup_df.set_index('Ticker')['Sector'].to_dict()

# Get unique sectors
sectors = lookup_df['Sector'].unique()

print(f"Found {len(sectors)} sectors: {list(sectors)}")

# Plot sector-level correlation
print("\nCreating sector correlation heat map...")
sector_returns_dict = {}
for sector in sectors:
    sector_tickers = lookup_df[lookup_df['Sector'] == sector]['Ticker'].tolist()
    available_tickers = [t for t in sector_tickers if t in daily_returns.columns]
    if available_tickers:
        sector_returns_dict[sector] = daily_returns[available_tickers].mean(axis=1)

sector_returns = pd.DataFrame(sector_returns_dict)
sector_correlation = sector_returns.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(sector_correlation, annot=True, cmap='RdYlGn', center=0, fmt='.2f')
plt.title("Correlation of Sector Returns")
plt.tight_layout()
plt.savefig(output_dir / "sector_correlation.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: sector_correlation.png")


def plot_sector_correlation_by_marketcap(sector_name, returns_df, lookup_df, output_dir):
    """
    Plot intra-sector correlation heat map with stocks sorted by market capitalization.
    
    Args:
        sector_name: Name of the sector to analyze
        returns_df: DataFrame containing stock returns
        lookup_df: DataFrame containing stock metadata including market cap
        output_dir: Path to save the output files
    """
    # Get tickers for the specified sector
    sector_data = lookup_df[lookup_df['Sector'] == sector_name]
    sector_tickers = sector_data['Ticker'].tolist()
    
    # Filter returns for this sector
    available_tickers = [t for t in sector_tickers if t in returns_df.columns]
    if len(available_tickers) < 2:
        print(f"Sector {sector_name} has fewer than 2 available tickers, skipping...")
        return
    
    sector_returns = returns_df[available_tickers]
    
    # Create a mapping of ticker to market cap
    marketcap_map = lookup_df.set_index('Ticker')['Market Cap (in tens of millions)'].to_dict()
    
    # Sort columns by market cap (descending - largest first)
    sorted_tickers = sorted(available_tickers, key=lambda x: marketcap_map.get(x, 0), reverse=True)
    sector_returns_sorted = sector_returns[sorted_tickers]
    
    # Calculate correlation matrix
    sector_corr = sector_returns_sorted.corr()
    
    # Create the heat map
    plt.figure(figsize=(14, 12))
    sns.heatmap(sector_corr, annot=False, cmap='coolwarm', center=0, 
                xticklabels=True, yticklabels=True)
    plt.title(f"Intra-Sector Correlation (Sorted by Market Cap): {sector_name}")
    plt.xlabel("Stocks (Largest to Smallest Market Cap)")
    plt.ylabel("Stocks (Largest to Smallest Market Cap)")
    plt.tight_layout()
    
    filename = f"sector_{sector_name.replace(' ', '_')}_marketcap.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")
    
    # Print market cap info
    print(f"\n{sector_name} - Stocks sorted by Market Cap:")
    print("="*60)
    for i, ticker in enumerate(sorted_tickers[:10], 1):
        market_cap = marketcap_map.get(ticker, 0)
        print(f"{i:2d}. {ticker:6s} - ${market_cap:10.2f} (tens of millions)")
    if len(sorted_tickers) > 10:
        print(f"... and {len(sorted_tickers) - 10} more stocks")
    print("="*60 + "\n")


# Plot sector correlations for selected sectors (lighter version - only a few sectors)
sectors_to_plot = ['Technology', 'Financial Services', 'Industrials']
for sector in sectors_to_plot:
    if sector in sector_returns.columns:
        sector_tickers = lookup_df[lookup_df['Sector'] == sector]['Ticker'].tolist()
        try:
            plot_sector_correlation_by_marketcap(sector, daily_returns, lookup_df, output_dir)
        except Exception as e:
            print(f"Error processing {sector}: {str(e)}")

print("\n✓ All graphs exported to output folder!")
print(f"Output directory: {output_dir}")
