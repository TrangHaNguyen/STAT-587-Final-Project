#!/usr/bin/env python3
import pandas as pd
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

def clean_data():
    # Hyperparameters
    lag=[1, 3, 7, 14]
    ema_windows=[7, 14, 28]
    vol_windows=[7, 14, 28]
    max_min_windows=[7, 21]
    rol_VWAP_windows=[7, 14, 21]
    idx = pd.IndexSlice

    print("------- Downloading Data")
    DATA=pd.read_csv(cwd / "PyScripts" / "raw_data.csv", header=[0, 1, 2], index_col=0, parse_dates=True)
    print("Finished Downloading Data -------")
    print("Initial shape:", DATA.shape[0], "rows,", DATA.shape[1], "columns.")

    print("------- Cleaning data")
    for type in ['Stocks']: # This list contains 'Commodities' if you include Commodities
        # Retrieve the specific data and drop rows that are all NA's (accounts for Holidays)
        TEMP_DATA=DATA.loc[:, idx[:, type, :]].dropna(how="all", axis=0)
        # Front fill for all tickers that have one NA value (accounts for ticker name changes or for holidays not being observed (specifically for commodities))
        TEMP_DATA[(TEMP_DATA.isna().sum().sort_values(ascending=False)==1).index]=TEMP_DATA[(TEMP_DATA.isna().sum().sort_values(ascending=False)==1).index].ffill()
        # Remove any columns that still contain NA's (usually tickers that were listed on any exchange after Jan 1st, 2024)
        TEMP_DATA=TEMP_DATA.dropna(how="any", axis=1)
        DATA = DATA.drop(columns=type, level=1).join(TEMP_DATA)

    # Dropping all rows where the Stocks observe a holiday in alignment with predicting if SPX will go up or down.
    stocks=DATA.loc[:, idx[:, 'Stocks', :]]
    to_drop=stocks.index[stocks.isna().all(axis=1)]
    DATA=DATA.drop(index=to_drop)
    print("Finished Cleaning Data -------")

    print("------- Generating Necessary Features")
    # Generating percent change from day before to current day. 
    features=DATA.loc[:, idx[['Close', 'Open', 'High', 'Low'], 'Stocks', :]].copy().pct_change().rename(columns={metric: f"{metric} PC" for metric in ['Close', 'Open', 'High', 'Low']}, level=0)
    features=pd.concat([features, DATA.loc[:, idx[['Close', 'Open', 'High', 'Low', 'Volume'], :, :]].copy()], axis=1)
    print("Created Percent Changes.")

    y_regression=((DATA.loc[:, idx['Close', 'Index', '^SPX']] - DATA.loc[:, idx['Open', 'Index', '^SPX']]) / DATA.loc[:, idx['Open', 'Index', '^SPX']]).rename("Target Regression").shift(-1)
    print("Created Target (Regression).")

    High_=DATA.loc[:, idx['High', :, :]]
    Low_=DATA.loc[:, idx['Low', :, :]]
    features=pd.concat([features, pd.DataFrame(High_.values-Low_.values, index=High_.index, columns=High_.columns).rename(columns={'High': f"Daily Range"}, level=0)], axis=1)
    print("Created Daily Range.")

    for metric in ['Close PC', 'Open PC', 'High PC', 'Low PC']:
        for lag_period in lag:
            features=pd.concat([features, features.loc[:, idx[metric, :, :]].shift(lag_period).rename(columns={metric: f"{metric} Lag {lag_period}"}, level=0)], axis=1)
    print("Created Lag.")

    for metric in ['Close', 'Open', 'High', 'Low']:
        for ema_window in ema_windows: 
            features=pd.concat([features, features.loc[:, idx[metric, :, :]].ewm(span=ema_window, adjust=False).mean().rename(columns={metric: f"{metric} EMA {ema_window}"}, level=0)], axis=1)
        for vol_window in vol_windows:
            features=pd.concat([features, features.loc[:, idx[metric, :, :]].rolling(window=vol_window).std().rename(columns={metric: f"{metric} VOL {vol_window}"}, level=0)], axis=1)
    print("Created EMA and Rolling Volatility.")

    for max_min_window in max_min_windows:
        features=pd.concat([features, features.loc[:, idx['High', :, :]].rolling(window=max_min_window).max().rename(columns={'High': f"MAX {max_min_window}"}, level=0)], axis=1)
        features=pd.concat([features, features.loc[:, idx['Low', :, :]].rolling(window=max_min_window).min().rename(columns={'Low': f"MIN {max_min_window}"}, level=0)], axis=1)

    for metric in ['Close', 'Open', 'High', 'Low']:
        for max_min_window in max_min_windows:
            max_=features.loc[:, idx[f'MAX {max_min_window}', :, :]]
            min_=features.loc[:, idx[f'MIN {max_min_window}', :, :]]
            metric_=features.loc[:, idx[metric, :, :]]
            # A case was noted when the max_ and min_ values are equal to each other. We can simply drop the relative stock to remove this.
            is_zero = (max_.values - min_.values == 0)
            if is_zero.any():
                problem_tickers = max_.columns[is_zero.any(axis=0)].get_level_values(2).unique()
                features.drop(columns=problem_tickers, level=2, inplace=True)
                continue
            max_min_channel_pos =(metric_.values-min_.values)/(max_.values-min_.values)
            features=pd.concat([features, pd.DataFrame(max_min_channel_pos, index=features.index, columns=metric_.columns).rename(columns={metric: f'Channel Position {metric} {max_min_window}'}, level=0).ffill().fillna(0.5)], axis=1)
    print("Created Max/Min Channel Positions/")

    for max_min_window in max_min_windows:
        features.drop(columns=[f"MAX {max_min_window}", f"MIN {max_min_window}"], level=0, inplace=True)
    print("Cleaned up Unnecessary Columns.")

    for rol_VWAP_window in rol_VWAP_windows:
        typical_price=(features.loc[:, idx['High', :, :]].values + features.loc[:, idx['Low', :, :]].values + features.loc[:, idx['Close', :, :]].values)/3
        volume=(features.loc[:, idx['Volume', :, :]])
        price_volume=typical_price*volume.values
        price_volume_rol_sum=pd.DataFrame(price_volume, index=features.index, columns=volume.columns).rolling(rol_VWAP_window).sum()
        volume_rol_sum=volume.rolling(rol_VWAP_window).sum()
        features=pd.concat([features, (price_volume_rol_sum / volume_rol_sum).rename(columns={'Volume': f'Rolling VWAP {rol_VWAP_window}'}, level=0)], axis=1)
    print("Created Rolling Volume Weighted Average Price.")

    for rol_zscore_window in ema_windows:
        for metric in ['Close', 'Open', 'High', 'Low']:
            price=features.loc[:, idx[metric, :, :]]
            EMA=features.loc[:, idx[f"{metric} EMA {rol_zscore_window}", :, :]]
            Vol=features.loc[:, idx[f"{metric} VOL {rol_zscore_window}", :, :]]
            z_score=(price.values-EMA.values)/Vol.values 
            features=pd.concat([features, pd.DataFrame(z_score, index=features.index, columns=price.columns).rename(columns={metric: f"{metric} Z-Score {rol_zscore_window}"}, level=0)], axis=1)
    print("Created Rolling Z-Score.")

    for metric in features.columns.get_level_values(0).unique():
        if metric[:4] == "Open":
            features=pd.concat([features, features.loc[:, idx[metric, :, :]].shift(-1).rename(columns={metric: f"{metric} Forward Lag"})], axis=1)
    print("Created Open Metrics Forward Lag.")

    features.drop(columns=["Close", "Open", "High", "Low"], inplace=True)
    print("Cleaned up Unnecessary Columns.")

    y_regression = y_regression.to_frame()
    y_regression.columns = pd.MultiIndex.from_tuples([('Target', 'Index', 'Regression')])
    print("Created Target (Classification)")

    X=pd.concat([features, y_regression], axis=1)
    X.dropna(how="any", axis=0, inplace=True)
    print("Cleaned up remaining/resulting NA values.")

    y_regression=X[('Target', 'Index', 'Regression')].rename("Target Regression")
    X=X.drop(columns=['Target'], level=0)
    print("Predictors, and Target (Regression) successfully split.")

    print("Finished Generating Features -------")
    print("Final shape (X):", X.shape[0], "rows,", X.shape[1], "columns.")
    print("Final shape (y_regression):", y_regression.shape[0], "rows, 1 columns.")
    return X, y_regression

def pull_features(dataframe, feature_name, include=False):
    idx = pd.IndexSlice
    if not include:
        return dataframe.loc[:, idx[feature_name, :, :]]
    else:
        new_dataframe = pd.DataFrame()
        for metric in dataframe.columns.get_level_values(0).unique():
            if feature_name in metric:
                new_dataframe = pd.concat([new_dataframe, dataframe.loc[:, idx[metric, :, :]]], axis=1)
        return new_dataframe



