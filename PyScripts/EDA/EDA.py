import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data_preprocessing_and_cleaning import clean_data
from pathlib import Path

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 8)

cwd=Path.cwd()
for _ in range(5): 
    if cwd.name!="STAT-587-Final-Project":
        cwd=cwd.parent
    else:
        break
else:
    raise FileNotFoundError("Could not find correct workspace folder.")

lookup_df = pd.read_csv(cwd / "PyScripts" / "stock_lookup_table.csv")

X, y=clean_data()

returns_df=X.xs(key='Close PC', axis=1, level=0)
returns_df.columns=returns_df.columns.droplevel(0)
sector_map=lookup_df.set_index('Ticker')['Sector'].to_dict()
sector_returns = returns_df.groupby(sector_map, axis=1).mean()
sector_correlation=sector_returns.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(sector_correlation, annot=True, cmap='RdYlGn', center=0)
plt.title("Correlation of Sector Returns")
plt.show()

def plot_sector_correlation(sector_name, returns_df, lookup_df):
    sector_tickers=lookup_df[lookup_df['Sector'] == sector_name]['Ticker']
    sector_returns=returns_df[returns_df.columns.intersection(sector_tickers)]
    sector_corr=sector_returns.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(sector_corr, annot=False, cmap='coolwarm', center=0)
    plt.title(f"Intra-Sector Correlation: {sector_name}")
    plt.show()

plot_sector_correlation('Technology', returns_df, lookup_df)
plot_sector_correlation('Financial Services', returns_df, lookup_df)
plot_sector_correlation('Real Estate', returns_df, lookup_df)

vol_df=X.xs(key='Close VOL 28', axis=1, level=0)
vol_df.columns=vol_df.columns.droplevel(0)
sector_vol_map=lookup_df.set_index('Ticker')['Sector'].to_dict()
sector_vol_series=vol_df.groupby(sector_vol_map, axis=1).mean()
for sector in sector_vol_series.columns:
    plt.plot(sector_vol_series.index, sector_vol_series[sector], label=sector)
plt.title("Sector Volatility Over Time")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.legend()
plt.show()

smoothed_sector_vol = sector_vol_series.rolling(window=40).mean()
for sector in smoothed_sector_vol.columns:
    plt.plot(smoothed_sector_vol.index, smoothed_sector_vol[sector], label=sector)
plt.title("Smoothed Sector Volatility Over Time")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.legend()
plt.tight_layout()
plt.show()

total_average_volatility=vol_df.mean(axis=1)
plt.title("Total Average Sector Volatility Over Time against Sector Specific")
plt.plot(total_average_volatility.index, total_average_volatility, color="black", linewidth=2, label="Total Market")
sector_vol_series=vol_df.groupby(sector_vol_map, axis=1).mean()
for sector in sector_vol_series.columns:
    plt.plot(sector_vol_series.index, sector_vol_series[sector], label=sector, alpha=0.35, linewidth=1)
plt.title("Smoothed Sector Volatility Over Time")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.legend()
plt.tight_layout()
plt.show()