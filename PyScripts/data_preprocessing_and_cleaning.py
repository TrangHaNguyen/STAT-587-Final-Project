#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mlxtend.feature_selection import SequentialFeatureSelector as MFS
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score, KFold, cross_validate

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

X, y_regression=clean_data()

print("Setting up for fitting models...")
kf = KFold(n_splits=12, shuffle=True, random_state=1)

X_train, X_test, yr_train, yr_test=train_test_split(X, y_regression, test_size=0.2)
yc_train=(yr_train>=0).astype(int).to_numpy()
y_test=(yr_test>=0).astype(int).to_numpy()


# ------- Generic Random Forest Regression Model ------------------
# print("------- Random Forest Regression -------")
# RFRegressor = RandomForestRegressor(max_depth=10, max_features=1000, n_jobs=-1)
# RFRegressor.fit(X_train, yr_train)

# RF_predictions = RFRegressor.predict(X_test)
# RF_prediction_direction = (pd.Series(RF_predictions) >= 0).astype(int).to_numpy()
# print("Average predicted direction:", np.mean(RF_prediction_direction))

# accuracy = np.mean(RF_prediction_direction == y_test)
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
#-------------------------

# ------- Principal Component Analysis --------
print("------- Principal Component Analysis -------")
scaler = StandardScaler()
X_train_std=scaler.fit_transform(X_train)
X_test_std=scaler.transform(X_test)
print("Scaled X_train, X_test, Xc_train, and Xc_test.")

pca=PCA(n_components=0.90)
X_train_pca=pca.fit_transform(X_train_std)
X_test_pca=pca.transform(X_test_std)
print("Finished PCA on X_train_std, X_test_std, Xc_train_std, and Xc_test_std.")

pca_columns=[f'PC {i}' for i in range(X_train_pca.shape[1])]
X_train_pca=pd.DataFrame(X_train_pca, columns=pca_columns, index=X_train.index)
X_test_pca=pd.DataFrame(X_test_pca, columns=pca_columns, index=X_test.index)
print("Converted Final Principal Components to Data Frame for Ease of Use.")

print("Reduced shape (Regression):", X_train_pca.shape[0] + X_test_pca.shape[0], "rows,", X_train_pca.shape[1], "columns.")
# ------------------------

# ------- Random Forest Regression (reduced) -------
# print("------- Random Forest Regression (Reduced) -------")
RFRegressor_red=RandomForestRegressor(n_jobs=-1)
# RFRegressor_red.fit(X_train_pca, yr_train)

# RFR_red_predictions=RFRegressor_red.predict(X_test_pca)
# RFR_red_prediction_direction = (pd.Series(RFR_red_predictions) >= 0).astype(int).to_numpy()
# print("Average predicted direction:", np.mean(RFR_red_prediction_direction))

# accuracy=np.mean(RFR_red_prediction_direction == yr_test)
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
# ------------------------

# ------- Random Forest Classification (Reduced)
print("------- Random Forest Classification (Reduced) -------")
RFClassifier_red=RandomForestClassifier(min_samples_leaf=10, n_jobs=-1)
RFClassifier_red.fit(X_train_pca, yc_train)

scoring_metrics = ['accuracy', 'precision', 'recall']
cv_scores = cross_val_score(RFClassifier_red, X_train_pca, yc_train, cv=kf, n_jobs=-1)
results = cross_validate(RFClassifier_red, X_train_pca, yc_train, cv=kf, scoring=scoring_metrics)

print(f"Average Accuracy:   {results['test_accuracy'].mean():.2%}")
print(f"Standard Deviation: {cv_scores.std() * 100:.2f}%")
print(f"Average Precision:  {results['test_precision'].mean():.2%}")
print(f"Average Recall:     {results['test_recall'].mean():.2%}")

RFC_red_predictions=RFClassifier_red.predict(X_test_pca)
print("Average predicted direction:", np.mean(RFC_red_predictions))

accuracy=np.mean(RFC_red_predictions == y_test)
print("Accuracy (*100%):", accuracy * 100)

RFClassifier_red_feature_df=pd.DataFrame({
    'Feature': X_train_pca.columns,
    'Importance': RFClassifier_red.feature_importances_
}).sort_values(by='Importance', ascending=False)
RFClassifier_red_feature_df.head(50).plot(kind='barh', x="Feature", y="Importance")
plt.xlabel("Feature Importance")
plt.xticks(rotation=45)
plt.ylabel("Feature Name")
plt.show()
# ------------------------

# tscv=TimeSeriesSplit(n_splits=3)
# sfs=SequentialFeatureSelector(RFRegression, n_features_to_select=10, direction='forward', cv=tscv)

# ------- Step-wise Regression -------
print("------- Step-wise Regression -------")
mfs=MFS(RFRegressor_red, k_features=(1, int(np.sqrt(X.shape[0]))), forward=True, floating=True, cv=3, n_jobs=-1)
mfs.fit(X_train_pca, yr_train)
X_train_stepwise=mfs.transform(X_train_pca)
X_test_stepwise=mfs.transform(X_test_pca)
print("Performed Step-wise Regression on X_train_pca and X_test_pca.")

mfs=MFS(RFClassifier_red, k_features=(1, int(np.sqrt(X.shape[0]))), forward=True, floating=True, cv=3, n_jobs=-1)
mfs.fit(Xc_train_pca, yc_train)
Xc_train_stepwise=mfs.transform(Xc_train_pca)
Xc_test_stepwise=mfs.transform(Xc_test_pca)
print("Performed Step-wise Regression on Xc_train_pca and Xc_test_pca.")
# ------------------------

# ------- Random Forest Regression (Step-wise; Reduced)
print("------- Random Forest Regression (Step-wise; Reduced) -------")
RFRegressor_red_stepwise = RandomForestRegressor(n_jobs=-1)
RFRegressor_red_stepwise.fit(X_train_stepwise, yr_train)

RFR_red_stepwise_predictions=RFRegressor_red_stepwise.predict(X_test_stepwise)
RFR_red_stepwise_prediction_direction = (pd.Series(RFR_red_stepwise_predictions) >= 0).astype(int).to_numpy()
print("Average predicted direction:", np.mean(RFR_red_stepwise_prediction_direction))

accuracy=np.mean(RFR_red_stepwise_prediction_direction == yr_test)
print("Accuracy (*100%):", accuracy * 100)
# ------------------------

# ------- Random Forest Classification (Step-wise; Reduced)
print("------- Random Forest Classification (Step-wise; Reduced) -------")
RFClassifier_red_stepwise = RandomForestClassifier(n_jobs=-1)
RFClassifier_red_stepwise.fit(Xc_train_stepwise, yc_train)

RFC_red_stepwise_predictions=RFClassifier_red_stepwise.predict(Xc_test_stepwise)
print("Average predicted direction:", np.mean(RFC_red_stepwise_predictions))

accuracy=np.mean(RFC_red_stepwise_predictions == yc_test)
print("Accuracy (*100%):", accuracy * 100)
# ------------------------


# print(X.columns.get_level_values(0).unique())

