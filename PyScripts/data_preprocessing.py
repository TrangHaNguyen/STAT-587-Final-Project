#!/usr/bin/env python3
import yfinance as yf
import pandas as pd
from pathlib import Path

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 8)

cwd=Path.cwd()
while (cwd.name!="STAT-587-Final-Project"): cwd=cwd.parent

# Hyperparameters
lag=[1, 3, 7, 14]
ema_windows=[7, 14, 21]
vol_windows=[14, 28]
max_min_windows=[7, 21]
rol_VWAP_windows=[7, 14, 21]
rol_zscore_windows=[7, 14, 21]

# Day of the week lists for Day of the Week analysis
Monday=[]
Tuesday=[]
Wednesday=[]
Thursday=[]
Friday=[]

idx = pd.IndexSlice

DATA=pd.read_csv(cwd / "bin" / "total_data.csv", header=[0, 1, 2], index_col=0, parse_dates=True)

for type in ['Stocks', 'Commodities']:
    # Retrieve the specific data and drop rows that are all NA's (accounts for Holidays)
    TEMP_DATA=DATA.loc[:, idx[:, type, :]].dropna(how="all", axis=0)
    # Front fill for all tickers that have one NA value (accounts for ticker name changes or for holidays not being observed (specifically for commodities))
    TEMP_DATA[(TEMP_DATA.isna().sum().sort_values(ascending=False)==1).index]=TEMP_DATA[(TEMP_DATA.isna().sum().sort_values(ascending=False)==1).index].ffill()
    # Remove any columns that still contain NA's (usually tickers that were listed on any exchange after Jan 1st, 2024)
    TEMP_DATA=TEMP_DATA.dropna(how="any", axis=1)
    # For debugging -
    # print(TEMP_DATA.isna().sum().sum())
    # print((TEMP_DATA.isna().sum().sort_values(ascending=False).head(100))) 
    # print("\n\n\n\n")
    DATA = DATA.drop(columns=type, level=1).join(TEMP_DATA)

for index in ["^SPX", "^RUT", "^IXIC", "^DJI", "^VIX", "^N225", "^GDAXI"]:
    # Front Fill for all NA values.
    TEMP_DATA=DATA.loc[:, idx[:, "Indexes", index]].ffill().bfill()
    DATA = DATA.drop(columns=index, level=2).join(TEMP_DATA)

# Dropping all rows where the Stocks observe a holiday in alignment with predicting if SPX will go up or down.
stocks=DATA.loc[:, idx[:, 'Stocks', :]]
to_drop=stocks.index[stocks.isna().all(axis=1)]
DATA=DATA.drop(index=to_drop)

# Shifting ^GDAXI Close prices back by one to avoid data leakage.
DATA.loc[:, idx['Close', 'Indexes', '^GDAXI']] = DATA.loc[:, idx['Close', 'Indexes', '^GDAXI']].shift(-1)

# Dropping last row that contains the ^GDAXI NA
DATA=DATA.iloc[:-1]

MODIFIED_DATA=DATA.loc[:, idx[['Close', 'Open', 'High', 'Low'], :, :]].copy().pct_change().rename(columns={metric: f"{metric} PC" for metric in ['Close', 'Open', 'High', 'Low']}, level=0).iloc[1:]
MODIFIED_DATA=pd.concat([MODIFIED_DATA, DATA.loc[:, idx[['Close', 'Open', 'High', 'Low', 'Volume'], :, :]].copy().iloc[1:]], axis=1)

for metric in ['Close PC', 'Open PC', 'High PC', 'Low PC']:
    for lag_period in lag:
        MODIFIED_DATA=pd.concat([MODIFIED_DATA, MODIFIED_DATA.loc[:, idx[metric, :, :]].shift(lag_period).rename(columns={metric: f"{metric} Lag {lag_period}"}, level=0)], axis=1)

# # TODO Utilize Volatility to get Z-Score rows and potentially look into implications of running these feature creation loops while having NA's in the mix due to difference in timezones.
for metric in ['Close', 'Open', 'High', 'Low']:
    for ema_window in ema_windows: 
        MODIFIED_DATA=pd.concat([MODIFIED_DATA, MODIFIED_DATA.loc[:, idx[metric, :, :]].ewm(span=ema_window, adjust=False).mean().rename(columns={metric: f"{metric} EMA {ema_window}"}, level=0)], axis=1)
    for vol_window in vol_windows:
        MODIFIED_DATA=pd.concat([MODIFIED_DATA, MODIFIED_DATA.loc[:, idx[metric, :, :]].rolling(window=vol_window).std().rename(columns={metric: f"{metric} VOL {vol_window}"}, level=0)], axis=1)

for max_min_window in max_min_windows:
    MODIFIED_DATA=pd.concat([MODIFIED_DATA, MODIFIED_DATA.loc[:, idx['High', :, :]].rolling(window=max_min_window).max().rename(columns={'High': f"MAX {max_min_window}"}, level=0)], axis=1)
    MODIFIED_DATA=pd.concat([MODIFIED_DATA, MODIFIED_DATA.loc[:, idx['Low', :, :]].rolling(window=max_min_window).min().rename(columns={'Low': f"MIN {max_min_window}"}, level=0)], axis=1)

for metric in ['Close', 'Open', 'High', 'Low']:
    for max_min_window in max_min_windows:
        max_=MODIFIED_DATA.loc[:, idx[f'MAX {max_min_window}', :, :]]
        min_=MODIFIED_DATA.loc[:, idx[f'MIN {max_min_window}', :, :]]
        metric_=MODIFIED_DATA.loc[:, idx[metric, :, :]]
        max_min_channel_pos = (metric_.values - min_.values) / (max_.values - min_.values)
        MODIFIED_DATA=pd.concat([MODIFIED_DATA, pd.DataFrame(max_min_channel_pos, index=MODIFIED_DATA.index, columns=metric_.columns).rename(columns={metric: f'Channel Position {metric} {max_min_window}'}, level=0)], axis=1)

# print(MODIFIED_DATA.columns.levels[0])
# print(MODIFIED_DATA["Channel Position Close 21"]["Stocks"].iloc[0:100])

print(MODIFIED_DATA["Close"]["Indexes"]["^VIX"].head(100))
print(MODIFIED_DATA.loc[:, idx["Close", ["Stocks", "Indexes"], ["AAPL", "^SPX", "^N225", "^GDAXI"]]].head(100))