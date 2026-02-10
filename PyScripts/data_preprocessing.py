#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mlxtend.feature_selection import SequentialFeatureSelector as MFS
from sklearn.model_selection import TimeSeriesSplit

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

# Hyperparameters
lag=[1, 3, 7, 14]
ema_windows=[7, 14, 28]
vol_windows=[7, 14, 28]
max_min_windows=[7, 21]
rol_VWAP_windows=[7, 14, 21]

idx = pd.IndexSlice

DATA=pd.read_csv(cwd / "PyScripts" / "raw_data.csv", header=[0, 1, 2], index_col=0, parse_dates=True)

print("Initial shape:", DATA.shape[0], "rows,", DATA.shape[1], "columns.")

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

# Generating percent change from day before to current day. 
features=DATA.loc[:, idx[['Close', 'Open', 'High', 'Low'], 'Stocks', :]].copy().pct_change().rename(columns={metric: f"{metric} PC" for metric in ['Close', 'Open', 'High', 'Low']}, level=0)
features=pd.concat([features, DATA.loc[:, idx[['Close', 'Open', 'High', 'Low', 'Volume'], :, :]].copy()], axis=1)
y_regression=((DATA.loc[:, idx['Close', 'Index', '^SPX']] - DATA.loc[:, idx['Open', 'Index', '^SPX']]) / DATA.loc[:, idx['Open', 'Index', '^SPX']]).rename("Target Regression").shift(-1)

High_=DATA.loc[:, idx['High', :, :]]
Low_=DATA.loc[:, idx['Low', :, :]]
features=pd.concat([features, pd.DataFrame(High_.values-Low_.values, index=High_.index, columns=High_.columns).rename(columns={'High': f"Daily Range"}, level=0)], axis=1)

for metric in ['Close PC', 'Open PC', 'High PC', 'Low PC']:
    for lag_period in lag:
        features=pd.concat([features, features.loc[:, idx[metric, :, :]].shift(lag_period).rename(columns={metric: f"{metric} Lag {lag_period}"}, level=0)], axis=1)

for metric in ['Close', 'Open', 'High', 'Low']:
    for ema_window in ema_windows: 
        features=pd.concat([features, features.loc[:, idx[metric, :, :]].ewm(span=ema_window, adjust=False).mean().rename(columns={metric: f"{metric} EMA {ema_window}"}, level=0)], axis=1)
    for vol_window in vol_windows:
        features=pd.concat([features, features.loc[:, idx[metric, :, :]].rolling(window=vol_window).std().rename(columns={metric: f"{metric} VOL {vol_window}"}, level=0)], axis=1)

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

for max_min_window in max_min_windows:
    features.drop(columns=[f"MAX {max_min_window}", f"MIN {max_min_window}"], level=0, inplace=True)

for rol_VWAP_window in rol_VWAP_windows:
    typical_price=(features.loc[:, idx['High', :, :]].values + features.loc[:, idx['Low', :, :]].values + features.loc[:, idx['Close', :, :]].values)/3
    volume=(features.loc[:, idx['Volume', :, :]])
    price_volume=typical_price*volume.values
    price_volume_rol_sum=pd.DataFrame(price_volume, index=features.index, columns=volume.columns).rolling(rol_VWAP_window).sum()
    volume_rol_sum=volume.rolling(rol_VWAP_window).sum()
    features=pd.concat([features, (price_volume_rol_sum / volume_rol_sum).rename(columns={'Volume': f'Rolling VWAP {rol_VWAP_window}'}, level=0)], axis=1)

for rol_zscore_window in ema_windows:
    for metric in ['Close', 'Open', 'High', 'Low']:
        price=features.loc[:, idx[metric, :, :]]
        EMA=features.loc[:, idx[f"{metric} EMA {rol_zscore_window}", :, :]]
        Vol=features.loc[:, idx[f"{metric} VOL {rol_zscore_window}", :, :]]
        z_score=(price.values-EMA.values)/Vol.values 
        features=pd.concat([features, pd.DataFrame(z_score, index=features.index, columns=price.columns).rename(columns={metric: f"{metric} Z-Score {rol_zscore_window}"}, level=0)], axis=1)

for metric in features.columns.get_level_values(0).unique():
    if metric[:4] == "Open":
        features=pd.concat([features, features.loc[:, idx[metric, :, :]].shift(-1).rename(columns={metric: f"{metric} Forward Lag"})], axis=1)

features.drop(columns=["Close", "Open", "High", "Low"], inplace=True)
y_classification=(y_regression > 0).astype(int).rename("Target Classification").to_frame()
y_classification.columns = pd.MultiIndex.from_tuples([('Target', 'Index', 'Classification')])
y_regression = y_regression.to_frame()
y_regression.columns = pd.MultiIndex.from_tuples([('Target', 'Index', 'Regression')])

X=pd.concat([features, y_classification, y_regression], axis=1)
X.dropna(how="any", axis=0, inplace=True)

y_classification=X[('Target', 'Index', 'Classification')].rename("Target Classification")
y_regression=X[('Target', 'Index', 'Regression')].rename("Target Regression")
X=X.drop(columns=['Target'], level=0)

print("Final shape:", X.shape[0], "rows,", X.shape[1], "columns.")

X_train, X_test, yr_train, yr_test=train_test_split(X, y_regression, test_size=0.2)
Xc_train, Xc_test, yc_train, yc_test=train_test_split(X, y_classification, test_size=0.2)

# ------- Generic Random Forest Regression Model ------------------
# print("------- Random Forest Regression -------")
# RFRegressor = RandomForestRegressor(max_depth=10, max_features=1000, n_jobs=-1)
# RFRegressor.fit(X_train, yr_train)

# RF_predictions = RFRegressor.predict(X_test)
# RF_prediction_direction = (pd.Series(RF_predictions) >= 0).astype(int).to_numpy()
# print("Average predicted direction:", np.mean(RF_prediction_direction))
# yr_test = (yr_test >= 0).astype(int).to_numpy()

# accuracy = np.mean(RF_prediction_direction == yr_test)
# print("Accuracy (*100%):", accuracy * 100)

# RFRegressor_feature_df=pd.DataFrame({
#     'Feature': X_train.columns,
#     'Importance': RFRegressor.feature_importances_
# }).sort_values(by='Importance', ascending=False)
# RFRegressor_feature_df.head(50).plot(kind='barh', x="Feature", y="Importance")
# plt.xlabel("Feature Importance")
# plt.xticks(rotation=45)
# plt.ylabel("Feature Name")
# plt.show()
# print("------- ------- -------")
#-------------------------

# ------- Principal Component Analysis --------
print("------- Principal Component Analysis -------")
scaler = StandardScaler()
X_train_std=scaler.fit_transform(X_train)
X_test_std=scaler.transform(X_test)
Xc_train_std=scaler.fit_transform(Xc_train)
Xc_test_std=scaler.transform(Xc_test)

pca=PCA(n_components=0.90)
X_train_pca=pca.fit_transform(X_train_std)
X_test_pca=pca.transform(X_test_std)
Xc_train_pca=pca.fit_transform(Xc_train_std)
Xc_test_pca=pca.transform(Xc_test_std)

pca_columns=[f'PC {i}' for i in range(X_train_pca.shape[1])]
X_train_pca=pd.DataFrame(X_train_pca, columns=pca_columns, index=X_train.index)
X_test_pca=pd.DataFrame(X_test_pca, columns=pca_columns, index=X_test.index)

pca_columns=[f'PC {i}' for i in range(Xc_train_pca.shape[1])]
Xc_train_pca=pd.DataFrame(Xc_train_pca, columns=pca_columns, index=Xc_train.index)
Xc_test_pca=pd.DataFrame(Xc_test_pca, columns=pca_columns, index=Xc_test.index)

print("Reduced shape (Regression):", X_train_pca.shape[0] + X_test_pca.shape[0], "rows,", X_train_pca.shape[1], "columns.")
print("Reduced shape (Classification):", Xc_train_pca.shape[0] + Xc_test_pca.shape[0], "rows,", Xc_train_pca.shape[1], "columns.")
print("------- ------- -------")
# ------------------------

# ------- Random Forest Regression (reduced) -------
# print("------- Random Forest Regression (Reduced) -------")
RFRegressor_red=RandomForestRegressor(n_jobs=-1)
# RFRegressor_red.fit(X_train_pca, yr_train)

# RF_red_predictions=RFRegressor_red.predict(X_test_pca)
# RF_red_prediction_direction = (pd.Series(RF_red_predictions) >= 0).astype(int).to_numpy()
# print("Average predicted direction:", np.mean(RF_red_prediction_direction))
# yr_test=(yr_test >= 0).astype(int).to_numpy()

# accuracy=np.mean(RF_red_prediction_direction == yr_test)
# print("Accuracy (*100%):", accuracy * 100)

# RFRegressor_red_feature_df=pd.DataFrame({
#     'Feature': X_train_pca.columns,
#     'Importance': RFRegressor_red.feature_importances_
# }).sort_values(by='Importance', ascending=False)
# RFRegressor_red_feature_df.head(50).plot(kind='barh', x="Feature", y="Importance")
# plt.xlabel("Feature Importance")
# plt.xticks(rotation=45)
# plt.ylabel("Feature Name")
# plt.show()
# print("------- ------- -------")
# ------------------------

# ------- Random Forest Classification (Reduced)
# print("------- Random Forest Classifier (Reduced) -------")
RFClassifier_red=RandomForestClassifier(n_jobs=-1)
# RFClassifier_red.fit(Xc_train_pca, yc_train)

# RFC_red_predictions=RFClassifier_red.predict(Xc_test_pca)
# print("Average predicted direction:", np.mean(RFC_red_predictions))

# accuracy=np.mean(RFC_red_predictions == yc_test)
# print("Accuracy (*100%):", accuracy * 100)

# RFClassifier_red_feature_df=pd.DataFrame({
#     'Feature': Xc_train_pca.columns,
#     'Importance': RFClassifier_red.feature_importances_
# }).sort_values(by='Importance', ascending=False)
# RFClassifier_red_feature_df.head(50).plot(kind='barh', x="Feature", y="Importance")
# plt.xlabel("Feature Importance")
# plt.xticks(rotation=45)
# plt.ylabel("Feature Name")
# plt.show()
# print("------- ------- -------")
# ------------------------

# tscv=TimeSeriesSplit(n_splits=3)
# sfs=SequentialFeatureSelector(RFRegression, n_features_to_select=10, direction='forward', cv=tscv)

# ------- Stepwise Regression -------
mfs=MFS(RFRegressor_red, k_features=int(np.sqrt(X.shape[0])), forward=True, floating=True, cv=3, n_jobs=-1)
mfs.fit(X_train_pca, yr_train)
X_train_stepwise=mfs.transform(X_train_pca)
X_test_stepwise=mfs.transform(X_test_pca)

mfs=MFS(RFClassifier_red, k_features=int(np.sqrt(X.shape[0])), forward=True, floating=True, cv=3, n_jobs=-1)
mfs.fit(Xc_train_pca, yc_train)
Xc_train_stepwise=mfs.transform(Xc_train_pca)
Xc_test_stepwise=mfs.transform(Xc_test_pca)
# ------------------------


# print(X.columns.get_level_values(0).unique())

