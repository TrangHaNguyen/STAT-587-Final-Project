#!/usr/bin/env python3
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from typing import Dict, Any, List, Type
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os, sys
from contextlib import contextmanager
import itertools

from H_eval import RollingWindowBacktest, get_final_metrics, utility_score
from H_helpers import get_cwd
from model_grids import RANDOM_SEED, TRAIN_TEST_SHUFFLE

@contextmanager
def silence_stdout():
    new_target = open(os.devnull, "w")
    old_target = sys.stdout
    sys.stdout = new_target
    try:
        yield
    finally:
        sys.stdout = old_target
        new_target.close()

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 8)

cwd = get_cwd("STAT-587-Final-Project")


def to_binary_class(y: pd.Series) -> pd.Series:
    """Shared up/down target: 1 when next-day SPX open-to-close return is non-negative."""
    return (y >= 0).astype(int)

def import_data(
    testing: bool = False,
    extra_features: bool = True,
    cluster: bool = False,
    n_clusters: int = 100,
    corr_threshold: float = 0.95,
    corr_level: int = 0,
) -> pd.DataFrame:
    idx = pd.IndexSlice
    table = None
    print("------- Downloading Data")
    if testing:
        table = pq.read_table(cwd / "PyScripts" / "Data" / "raw_data_2_years.parquet")
    else:
        table = pq.read_table(cwd / "PyScripts" / "Data" / "raw_data_8_years.parquet")
    DATA = table.to_pandas()
    print("Finished Downloading Data -------")
    print("Initial shape:", DATA.shape[0], "rows,", DATA.shape[1], "columns.")

    print("------- Cleaning data")
    for data_type in ['Stocks']: # This list contains 'Commodities' if you include Commodities
        # Retrieve the specific data and drop rows that are all NA's (accounts for Holidays)
        TEMP_DATA = DATA.loc[:, idx[:, data_type, :]].dropna(how="all", axis=0)
        # Front fill for all tickers that have one NA value (accounts for ticker name changes or for holidays not being observed (specifically for commodities))
        missing_one = (TEMP_DATA.isna().sum() == 1)
        cols = missing_one[missing_one == 1].index
        TEMP_DATA[cols] = TEMP_DATA[cols].ffill()
        # Remove any columns that still contain NA's (usually tickers that were listed on any exchange after Jan 1st, 2024)
        TEMP_DATA = TEMP_DATA.dropna(how="any", axis=1)
        DATA = DATA.drop(columns=data_type, level=1).join(TEMP_DATA)

    # Dropping all rows where the Stocks observe a holiday in alignment with predicting if SPX will go up or down.
    stocks = DATA.loc[:, idx[:, 'Stocks', :]]
    to_drop = stocks.index[stocks.isna().all(axis=1)]
    DATA = DATA.drop(index=to_drop)
    print("Finished Cleaning Data -------")
    print("Current shape:", DATA.shape[0], "rows,", DATA.shape[1], "columns.")

    DATA = pd.concat([DATA, DATA.loc[:, idx[['Close', 'Open', 'High', 'Low'], 'Stocks', :]].copy().pct_change().rename(columns={metric: f"{metric} PC" for metric in ['Close', 'Open', 'High', 'Low']}, level=0)], axis=1)
    print("Created Percent Changes.")

    y_regression = ((DATA.loc[:, idx['Close', 'Index', '^SPX']] - DATA.loc[:, idx['Open', 'Index', '^SPX']]) / DATA.loc[:, idx['Open', 'Index', '^SPX']]).rename("Target Regression").shift(-1)
    print("Created Target (Regression).")

    if cluster:
        X_stocks = DATA.xs('Close PC', level=0, axis=1).droplevel('Type', axis=1).dropna(axis=0, how='all').T

        kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
        clusters = kmeans.fit_predict(X_stocks)

        representative_stocks = []
        centers = kmeans.cluster_centers_

        for i in range(n_clusters):
            indices = np.where(clusters == i)[0] # Get all stocks in this cluster
            distances = np.linalg.norm(X_stocks.iloc[indices].values - centers[i], axis=1) # Find the one stock closest to the cluster center
            representative_stocks.append(X_stocks.index[indices[np.argmin(distances)]])

        DATA = DATA.loc[:, idx[:, 'Stocks', representative_stocks]]
        print("---EXTRA---: Applied clustering and selected representative stocks.")
        print("Current shape:", DATA.shape[0], "rows,", DATA.shape[1], "columns.")

    
    if corr_level not in [0, 1, 2, 3]:
        raise ValueError("corr_level must be 0, 1, 2 or 3 (default 0).")
    elif corr_level == 1 or corr_level == 3:
        X_stocks = DATA.xs('Close PC', level=0, axis=1).droplevel('Type', axis=1).dropna(axis=0, how='all')
        corr_matrix = X_stocks.corr().abs()

        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)) # Upper diagonal matrix where main diagonal is zero.

        to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]

        DATA = DATA.drop(columns=to_drop, level='Ticker')
        print(f"---EXTRA---: (corr_level={corr_level}) Dropped {len(to_drop)} highly correlated stocks by closing percent change.")
        print("Current shape:", DATA.shape[0], "rows,", DATA.shape[1], "columns.")

    if extra_features:
        DATA[("Day of Week", "Calendar", "All")] = DATA.index.dayofweek
        print("---EXTRA---: Created Day of Week.")
    
        High_ = DATA.loc[:, idx['High', :, :]]
        Low_ = DATA.loc[:, idx['Low', :, :]]
        DATA = pd.concat([DATA, pd.DataFrame(High_.values - Low_.values, index=High_.index, columns=High_.columns).rename(columns={'High' : 'Daily Range'}, level=0)], axis=1)
        print("---EXTRA---: Created Daily Range.")

    return DATA, y_regression

def clean_data(DATA: pd.DataFrame, y_regression: pd.DataFrame, lookback_period: int =7, lag_period: list =[1], extra_features: bool =True, raw: bool =False, sector: bool =False, corr_threshold: float =0.95, corr_level: int =0) -> tuple[pd.DataFrame, pd.Series]:    
    if (lookback_period < 5 and lookback_period!=0): 
        raise ValueError("lookback_period must be greater than or equal to 6.")
    if isinstance(lag_period, int):
        lag_period=[lag_period]
    for lag in lag_period:
        if (lag <= 0):
            raise ValueError("lag_period must be greater than or equal to 0.")

    print("------- Generating Necessary Features")
    new_columns=[]
    idx=pd.IndexSlice

    DATA=DATA.copy()
    if (not raw):
        for lag in lag_period:
            for metric in ['Close PC', 'Open PC']:
                new_columns.append(DATA.loc[:, idx[metric, :, :]].shift(lag).rename(columns={metric: f"{metric} Lag {lag}"}, level=0))
        print("Created Lag.")

        if (lookback_period != 0):
            for metric in ['Close', 'Open']:
                price=DATA.loc[:, idx[metric, :, :]]
                vol=price.rolling(window=lookback_period).std()
                ema=price.ewm(span=lookback_period, adjust=False).mean()
                norm_vol=vol/ema.values
                z_score=np.full_like(price.values, 0.0)
                np.divide((price.values - ema.values), vol.values, out=z_score, where=(vol.values > 0))
                new_columns.append(norm_vol.rename(columns={metric: f"{metric} VOL {lookback_period}"}, level=0))
                new_columns.append(pd.DataFrame(z_score, index=DATA.index, columns=price.columns).rename(columns={metric: f"{metric} Z-Score {lookback_period}"}, level=0))
            print("Created EMA, Rolling Volatility (Scaled) and Rolling Z-Score.")

            rolling_High_=DATA.loc[:, idx['High', :, :]].rolling(window=lookback_period).max().rename(columns={'High': f"MAX {lookback_period}"}, level=0)
            rolling_Low_=DATA.loc[:, idx['Low', :, :]].rolling(window=lookback_period).min().rename(columns={'Low': f"MIN {lookback_period}"}, level=0)

            for metric in ['Close', 'Open']:
                price=DATA.loc[:, idx[metric, :, :]]
                max_min_channel_pos=np.full_like(price.values, 0.5)
                diff=rolling_High_.values - rolling_Low_.values
                np.divide((price.values-rolling_Low_.values), diff, out=max_min_channel_pos, where=(diff != 0))
                new_columns.append(pd.DataFrame(max_min_channel_pos, index=DATA.index, columns=price.columns).rename(columns={metric: f'Channel Position {metric} {lookback_period}'}, level=0))
            print("Created Max/Min Channel Positions/")

            typical_price=(DATA.loc[:, idx['High', :, :]].values + DATA.loc[:, idx['Low', :, :]].values + DATA.loc[:, idx['Close', :, :]].values)/3
            volume=(DATA.loc[:, idx['Volume', :, :]])
            price_volume=typical_price*volume.values
            price_volume_rol_sum=pd.DataFrame(price_volume, index=DATA.index, columns=volume.columns).rolling(lookback_period).sum()
            volume_rol_sum=volume.rolling(lookback_period).sum()
            new_columns.append((price_volume_rol_sum / volume_rol_sum).rename(columns={'Volume': f'Rolling VWAP {lookback_period}'}, level=0))
            print("Created Rolling Volume Weighted Average Price.")

        if (extra_features):
            for metric in DATA.columns.get_level_values(0).unique():
                if metric[:4] == "Open":
                    new_columns.append(DATA.loc[:, idx[metric, :, :]].shift(-1).rename(columns={metric: f"{metric} Forward Lag"}, level=0))
            print("---EXTRA---: Created Open Metrics Forward Lag.")

        DATA.drop(columns=["Close", "Open", "High", "Low"], level=0, inplace=True)
        DATA=DATA.sort_index(axis=1)
        print("Cleaned up Unnecessary Columns.")

    if (new_columns):
        DATA=pd.concat([DATA] + new_columns, axis=1)
    DATA=DATA.sort_index(axis=1)

    y_regression = y_regression.to_frame()
    y_regression.columns = pd.MultiIndex.from_tuples([('Target', 'Index', 'Regression')])
    print("Created Target (Classification)")

    X=pd.concat([DATA, y_regression], axis=1)
    X.dropna(how="any", axis=0, inplace=True)
    print("Cleaned up remaining/resulting NA values.")

    y_regression=X[('Target', 'Index', 'Regression')].rename("Target Regression")
    X=X.drop(columns=['Target'], level=0)

    if (sector):
        if ((corr_level not in [0, 2])): print("!!!WARNING!!!: Since dimensionality was reduced before grouping by sector, the sector feature averages may no longer reflect the sector itself.")
        lookup_df = pd.read_csv(cwd / "PyScripts" / "Data" / "stock_lookup_table.csv")
        sector_map = lookup_df.set_index('Ticker')['Sector'].to_dict()

        metrics = X.columns.get_level_values(0)
        sectors = X.columns.get_level_values(2).map(sector_map)
        
        X = X.T.groupby([metrics, sectors]).mean().T
        print("---EXTRA---: Grouped Features by Sector Averages.")
        print("Current shape:", X.shape[0], "rows,", X.shape[1], "columns.")

    if (corr_level in [2, 3]):
        cols_to_remove=[]
        if (not sector):
            for feature in X.columns.get_level_values(0).unique():
                X_stocks=X.xs(feature, level=0, axis=1).droplevel('Type', axis=1)
                corr_matrix=X_stocks.corr().abs()

                upper=corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)) # Upper diagonal matrix where main diagonal is zero.

                to_drop=[column for column in upper.columns if any(upper[column]>corr_threshold)]

                for ticker in to_drop:
                    cols_to_remove.append((feature, 'Stocks', ticker))
        else:
            for feature in X.columns.get_level_values(0).unique():
                X_stocks=X.xs(feature, level=0, axis=1)
                corr_matrix=X_stocks.corr().abs()

                upper=corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)) # Upper diagonal matrix where main diagonal is zero.

                to_drop=[column for column in upper.columns if any(upper[column]>corr_threshold)]

                for ticker in to_drop:
                    cols_to_remove.append((feature, ticker))
        X=X.drop(columns=cols_to_remove)
        print(f"---EXTRA---: (corr_level={corr_level}) Dropped {len(cols_to_remove)} highly correlated feature columns.")

    print("Predictors, and Target (Regression) successfully split.")

    print("Finished Generating Features -------")
    print("Final shape (X):", X.shape[0], "rows,", X.shape[1], "columns.")
    print("Final shape (y_regression):", y_regression.shape[0], "rows, 1 columns.")
    return X, y_regression

# clean_data(*import_data(), sector=True, corr=True, corr_level=3)

# Legacy helper retained here only as commented reference. The active model
# scripts use `clean_data`, and `data_clean_param_selection` no longer supports
# the old efficient single-parameter branch.
#
# def efficient_clean_data(DATA: pd.DataFrame, y_regression: pd.DataFrame, sector: bool =False, corr_threshold: float =0.95, corr_level: int =0, **kwargs) -> tuple[pd.DataFrame, pd.Series]:
#     ...
#
# efficient_clean_data(*import_data(), sector=True, corr_level=2, lookback_period=7)

def data_clean_param_selection(
    DATA: pd.DataFrame,
    y_regression: pd.DataFrame,
    model: BaseEstimator,
    test_size: float,
    window_size: int,
    horizon: int,
    w: float = 4.0,
    eff_support: bool = False,
    **kwargs: List[Any],
) -> tuple[pd.DataFrame, dict, float]:
    SCHEMA: Dict[str, Type] = {
        'raw': bool,
        'extra_features': bool,
        'lag_period': (int, list),
        'lookback_period': int,
        'sector': bool,
        'corr_threshold': float,
        'corr_level': int
    }
    
    cleaned_kwargs = {}

    current_keys = set(kwargs.keys())
    invalid_keys = current_keys - set(SCHEMA.keys())
    if invalid_keys:
        raise ValueError(f"Invalid parameter(s): {invalid_keys}. "
                         f"Valid parameters are: {', '.join(SCHEMA.keys())}")
    for key in kwargs:
        if not isinstance(kwargs[key], list):
            raise TypeError(f"Parameter {key} must be a list, got {type(kwargs[key]).__name__}.")
        expected_type = SCHEMA[key]
        for i, item in enumerate(kwargs[key]):
            if not isinstance(item, expected_type):
                raise TypeError(
                    f"Item {i} in '{key}' is {type(item).__name__}, "
                    f"but expected {expected_type.__name__}.")
            
        seen = []
        unique_list = []
        
        for item in kwargs[key]:
            comparison_item = tuple(item) if isinstance(item, list) else item
            
            if comparison_item not in seen:
                seen.append(comparison_item)
                unique_list.append(item)
            else:
                print(f"--- Duplicate found in '{key}': {item} removed.")

        cleaned_kwargs[key] = unique_list

    keys = cleaned_kwargs.keys()
    values = cleaned_kwargs.values()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Total Unique Cleaning Combinations to test: {len(combinations)}")

    scores = []
    for config in combinations:
        with silence_stdout():
            X = None
            y = None
            if eff_support:
                raise NotImplementedError(
                    "eff_support=True is retired. Active model scripts use clean_data()."
                )
            X, y = clean_data(DATA, y_regression, **config)
            y_classification = to_binary_class(y)
            X_train, X_test, y_train, y_test=train_test_split(X, y_classification, test_size=test_size, shuffle=TRAIN_TEST_SHUFFLE, random_state=RANDOM_SEED)

            pipeline_base_=Pipeline([('scaler', StandardScaler()), 
                                    ('classifier', clone(model))])
            
            rwb_obj=RollingWindowBacktest(clone(pipeline_base_), X, y_classification, X_train, window_size, horizon)
            rwb_obj.rolling_window_backtest(verbose=1)

            cloned_model=clone(pipeline_base_)
            cloned_model.fit(X_train, y_train)
            results=get_final_metrics(cloned_model, X_train, y_train, X_test, y_test)

            row=config.copy()

            # Higher w means favors accuracy, lower w means favors stability/robustness.
            row['score']=utility_score(results, rwb_obj, w)
            scores.append(row)

    results_df=pd.DataFrame(scores)
    best_score=results_df.loc[results_df['score'].idxmax()]['score']
    optimal_parameters=results_df.loc[results_df['score'].idxmax()].drop('score').to_dict()

    print(f"Optimal data_clean() parameters: {optimal_parameters}")
    return results_df, optimal_parameters, best_score

def pull_features(dataframe, feature_name, include=False) -> dict:
    idx = pd.IndexSlice
    if not include:
        return dataframe.loc[:, idx[feature_name, :, :]]
    else:
        new_dataframe = pd.DataFrame()
        for metric in dataframe.columns.get_level_values(0).unique():
            if feature_name in metric:
                new_dataframe = pd.concat([new_dataframe, dataframe.loc[:, idx[metric, :, :]]], axis=1)
        return new_dataframe

# if __name__ == "__main__":
#     clean_data(sector=True, cluster=True, corr=True, corr_level=3)
