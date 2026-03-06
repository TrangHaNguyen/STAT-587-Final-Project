#!/usr/bin/env python3
import os
import joblib
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import (train_test_split, TimeSeriesSplit,
                                     GridSearchCV, cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score

from helper_functions import get_cwd
from model_evaluation import rolling_window_backtest, get_final_metrics_grid, get_final_metrics

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

def to_binary_class(y):
    return (y >= 0).astype(int)

if __name__ == "__main__":
    # ------- Load raw data (no feature engineering) -------
    TESTING = True

    cwd = get_cwd("STAT-587-Final-Project")
    parquet_file = "raw_data_2_years.parquet" if TESTING else "raw_data_8_years.parquet"

    print("------- Loading Raw Data")
    table = pq.read_table(cwd / "PyScripts" / "Data" / parquet_file)
    DATA = table.to_pandas()
    print(f"Loaded: {DATA.shape[0]} rows, {DATA.shape[1]} columns.")

    idx = pd.IndexSlice

    # ------- Clean: drop holiday rows and tickers with missing data -------
    STOCKS = DATA.loc[:, idx[:, 'Stocks', :]].dropna(how="all", axis=0)
    STOCKS = STOCKS.dropna(how="any", axis=1)
    to_drop = STOCKS.index[STOCKS.isna().all(axis=1)]
    STOCKS = STOCKS.drop(index=to_drop)

    # ------- Extract only Close, Open, High, Low, Volume for Stocks -------
    RAW_METRICS = ['Close', 'Open', 'High', 'Low', 'Volume']
    X = STOCKS.loc[:, idx[RAW_METRICS, :, :]].copy()
    X.columns = [f"{metric}_{ticker}" for metric, _, ticker in X.columns]
    print(f"Feature matrix shape: {X.shape[0]} rows, {X.shape[1]} columns.")

    # ------- Build target: next-day SPX return (binary: up=1, down=0) -------
    spx_return = (
        (DATA.loc[:, idx['Close', 'Index', '^SPX']] - DATA.loc[:, idx['Open', 'Index', '^SPX']])
        / DATA.loc[:, idx['Open', 'Index', '^SPX']]
    ).shift(-1).rename("Target")

    combined = X.join(spx_return).dropna()
    X = combined.drop(columns=["Target"])
    y_regression = combined["Target"]
    y_classification = to_binary_class(y_regression)
    print(f"Final shape — X: {X.shape}, y: {y_classification.shape}")

    # ------- Train/test split (80/20) -------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_classification, test_size=0.2, random_state=1
    )
    tscv = TimeSeriesSplit(n_splits=3)

    # ------- GridSearchCV: 5 depths x 4 n_estimators = 20 combinations -------
    print("\n========== Random Forest GridSearch (20 combinations) ==========")
    _rf_cache = os.path.join(CACHE_DIR, 'base_rf_gridsearch.pkl')
    param_grid = {
        'classifier__max_depth':    [2, 3, 5, 8, 15],
        'classifier__n_estimators': [50, 100, 200, 500],
    }
    pipeline_rf = Pipeline([
        ('scaler',     StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=1, n_jobs=-1))
    ])
    if os.path.exists(_rf_cache):
        print("Loading RF GridSearch from cache...")
        grid_search_rf = joblib.load(_rf_cache)
    else:
        grid_search_rf = GridSearchCV(
            pipeline_rf, param_grid, cv=tscv,
            n_jobs=-1, return_train_score=True, verbose=1
        )
        grid_search_rf.fit(X_train, y_train)
        joblib.dump(grid_search_rf, _rf_cache)

    best_depth  = grid_search_rf.best_params_['classifier__max_depth']
    best_n_est  = grid_search_rf.best_params_['classifier__n_estimators']
    print(f"Best params: max_depth={best_depth}, n_estimators={best_n_est}")
    print("\n--- Model Report ---")
    get_final_metrics_grid(grid_search_rf, X_test, y_test)
    print("\n--- Rolling Window Backtest ---")
    rolling_window_backtest(grid_search_rf.best_estimator_, X, y_classification, verbose=1)

    # ------- PCA RF — GridSearch over n_components, max_depth, n_estimators -------
    print("\n========== PCA + Random Forest GridSearch ==========")
    _rf_pca_cache = os.path.join(CACHE_DIR, 'base_rf_pca_gridsearch.pkl')
    param_grid_pca = {
        'reducer__n_components':    [0.8, 0.9, 0.95],
        'classifier__max_depth':    [2, 3, 5],
        'classifier__n_estimators': [250, 500],
    }
    pipeline_rf_pca = Pipeline([
        ('scaler',     StandardScaler()),
        ('reducer',    PCA()),
        ('classifier', RandomForestClassifier(random_state=1, n_jobs=-1))
    ])
    if os.path.exists(_rf_pca_cache):
        print("Loading PCA RF GridSearch from cache...")
        grid_search_pca = joblib.load(_rf_pca_cache)
    else:
        grid_search_pca = GridSearchCV(
            pipeline_rf_pca, param_grid_pca, cv=tscv,
            n_jobs=-1, return_train_score=True, verbose=1
        )
        grid_search_pca.fit(X_train, y_train)
        joblib.dump(grid_search_pca, _rf_pca_cache)
    print(f"Best params (PCA RF): {grid_search_pca.best_params_}")
    print("\n--- Model Report ---")
    get_final_metrics_grid(grid_search_pca, X_test, y_test)
    print("\n--- Rolling Window Backtest ---")
    rolling_window_backtest(grid_search_pca.best_estimator_, X, y_classification, verbose=1)

    # ------- LASSO feature selection + RF -------
    print("\n========== LASSO Feature Selection + Random Forest GridSearch ==========")
    _rf_lasso_cache = os.path.join(CACHE_DIR, 'base_rf_lasso_gridsearch.pkl')
    param_grid_lasso = {
        'feature_selector__estimator__C': [0.001, 0.01, 0.1],
        'classifier__max_depth':          [2, 3, 5],
        'classifier__n_estimators':       [500],
    }
    pipeline_rf_lasso = Pipeline([
        ('scaler',           StandardScaler()),
        ('feature_selector', SelectFromModel(
            LogisticRegression(l1_ratio=1, solver='saga', random_state=1,
                               max_iter=500, tol=5e-2), threshold='mean')),
        ('classifier',       RandomForestClassifier(random_state=1, n_jobs=-1))
    ])
    if os.path.exists(_rf_lasso_cache):
        print("Loading LASSO RF GridSearch from cache...")
        grid_search_lasso = joblib.load(_rf_lasso_cache)
    else:
        grid_search_lasso = GridSearchCV(
            pipeline_rf_lasso, param_grid_lasso, cv=tscv,
            n_jobs=-1, return_train_score=True, verbose=1
        )
        grid_search_lasso.fit(X_train, y_train)
        joblib.dump(grid_search_lasso, _rf_lasso_cache)
    print(f"Best params (LASSO RF): {grid_search_lasso.best_params_}")
    print("\n--- Model Report ---")
    get_final_metrics_grid(grid_search_lasso, X_test, y_test)
    print("\n--- Rolling Window Backtest ---")
    rolling_window_backtest(grid_search_lasso.best_estimator_, X, y_classification, verbose=1)

    # ------- Ridge feature selection + RF -------
    print("\n========== Ridge Feature Selection + Random Forest GridSearch ==========")
    _rf_ridge_cache = os.path.join(CACHE_DIR, 'base_rf_ridge_gridsearch.pkl')
    param_grid_ridge = {
        'feature_selector__estimator__C': [0.001, 0.01, 0.1],
        'classifier__max_depth':          [2, 3, 5],
        'classifier__n_estimators':       [500],
    }
    pipeline_rf_ridge = Pipeline([
        ('scaler',           StandardScaler()),
        ('feature_selector', SelectFromModel(
            LogisticRegression(l1_ratio=0, solver='saga', random_state=1,
                               max_iter=500, tol=5e-2), threshold='mean')),
        ('classifier',       RandomForestClassifier(random_state=1, n_jobs=-1))
    ])
    if os.path.exists(_rf_ridge_cache):
        print("Loading Ridge RF GridSearch from cache...")
        grid_search_ridge = joblib.load(_rf_ridge_cache)
    else:
        grid_search_ridge = GridSearchCV(
            pipeline_rf_ridge, param_grid_ridge, cv=tscv,
            n_jobs=-1, return_train_score=True, verbose=1
        )
        grid_search_ridge.fit(X_train, y_train)
        joblib.dump(grid_search_ridge, _rf_ridge_cache)
    print(f"Best params (Ridge RF): {grid_search_ridge.best_params_}")
    print("\n--- Model Report ---")
    get_final_metrics_grid(grid_search_ridge, X_test, y_test)
    print("\n--- Rolling Window Backtest ---")
    rolling_window_backtest(grid_search_ridge.best_estimator_, X, y_classification, verbose=1)

    # ===================================================================
    # PLOT 1: Bias-Variance Tradeoff (CV train + CV test error ± std)
    # Sweep max_depth 1..20 with fixed best n_estimators
    # ===================================================================
    print("\n========== Generating Bias-Variance Tradeoff Plot (CV) ==========")

    depth_grid = list(range(1, 21))          # 20 values: 1, 2, ..., 20
    tr_means, tr_stds = [], []
    cv_means, cv_stds = [], []

    for depth in depth_grid:
        clf = Pipeline([
            ('scaler',     StandardScaler()),
            ('classifier', RandomForestClassifier(
                max_depth=depth, n_estimators=best_n_est,
                random_state=1, n_jobs=-1))
        ])
        cv_res = cross_validate(clf, X_train, y_train, cv=tscv,
                                return_train_score=True, n_jobs=-1)
        tr_means.append(1 - cv_res['train_score'].mean())
        tr_stds.append(cv_res['train_score'].std())
        cv_means.append(1 - cv_res['test_score'].mean())
        cv_stds.append(cv_res['test_score'].std())

    tr_means = np.array(tr_means)
    tr_stds  = np.array(tr_stds)
    cv_means = np.array(cv_means)
    cv_stds  = np.array(cv_stds)

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    fig1.suptitle(
        'Bias-Variance Tradeoff — Random Forest, Raw OHLCV Features\n'
        f'(Train vs CV Test Error, n_estimators={best_n_est})',
        fontsize=13, fontweight='bold'
    )
    ax1.plot(depth_grid, tr_means, marker='o', color='steelblue',
             linewidth=2, label='Train error')
    ax1.fill_between(depth_grid, tr_means - tr_stds, tr_means + tr_stds,
                     alpha=0.15, color='steelblue')
    ax1.plot(depth_grid, cv_means, marker='s', color='darkorange',
             linewidth=2, label='CV Test error')
    ax1.fill_between(depth_grid, cv_means - cv_stds, cv_means + cv_stds,
                     alpha=0.15, color='darkorange')
    ax1.axvline(best_depth, color='red', linestyle='--',
                label=f'Best max_depth = {best_depth}')
    ax1.set_title('Random Forest — Bias-Variance Tradeoff')
    ax1.set_xlabel('max_depth\n'
                   '← Low Depth, High Regularization, Simpler Model      '
                   'High Depth, Low Regularization, More Complex →')
    ax1.set_ylabel('Prediction Error')
    ax1.set_xticks(depth_grid)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    out_path1 = os.path.join(output_dir, 'base_rf_bias_variance.png')
    plt.savefig(out_path1, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {os.path.abspath(out_path1)}")
    plt.close()

    # ===================================================================
    # PLOT 2: Direct Train vs Test Error (no CV averaging)
    # Sweep max_depth 1..20, fit on X_train, score on X_train and X_test
    # ===================================================================
    print("\n========== Generating Train vs Test Error Plot (Direct Split) ==========")

    scaler_direct = StandardScaler()
    X_tr_sc = scaler_direct.fit_transform(X_train)
    X_te_sc = scaler_direct.transform(X_test)

    train_errors, test_errors = [], []
    for depth in depth_grid:
        clf = RandomForestClassifier(
            max_depth=depth, n_estimators=best_n_est,
            random_state=1, n_jobs=-1)
        clf.fit(X_tr_sc, y_train)
        train_errors.append(1 - clf.score(X_tr_sc, y_train))
        test_errors.append(1 - clf.score(X_te_sc, y_test))

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    fig2.suptitle(
        'Train vs Test Error — Random Forest, Raw OHLCV Features\n'
        f'(Direct Train/Test Split, No CV, n_estimators={best_n_est})',
        fontsize=13, fontweight='bold'
    )
    ax2.plot(depth_grid, train_errors, marker='o', color='steelblue',
             linewidth=2, label='Train error')
    ax2.plot(depth_grid, test_errors, marker='s', color='darkorange',
             linewidth=2, label='Test error')
    ax2.axvline(best_depth, color='red', linestyle='--',
                label=f'Best max_depth = {best_depth}')
    ax2.set_title('Random Forest — Train vs Test Error')
    ax2.set_xlabel('max_depth\n'
                   '← Low Depth, High Regularization, Simpler Model      '
                   'High Depth, Low Regularization, More Complex →')
    ax2.set_ylabel('Prediction Error')
    ax2.set_xticks(depth_grid)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path2 = os.path.join(output_dir, 'base_rf_train_test.png')
    plt.savefig(out_path2, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {os.path.abspath(out_path2)}")
    plt.close()

    # ===================================================================
    # PLOT 3: Draw one representative tree from the optimal RF model
    # Picks the tree with the most leaves (most "complete" tree)
    # ===================================================================
    print("\n========== Drawing Representative Tree from Optimal RF ==========")
    from sklearn.tree import plot_tree

    rf_best = grid_search_rf.best_estimator_.named_steps['classifier']
    leaf_counts    = [est.get_n_leaves() for est in rf_best.estimators_]
    best_tree_idx  = int(np.argmax(leaf_counts))
    tree_to_draw   = rf_best.estimators_[best_tree_idx]
    print(f"Drawing tree #{best_tree_idx} "
          f"({leaf_counts[best_tree_idx]} leaves, max_depth={best_depth})")

    fig3, ax3 = plt.subplots(figsize=(20, 8))
    plot_tree(
        tree_to_draw,
        max_depth=best_depth,
        feature_names=list(X_train.columns),
        class_names=['Down (0)', 'Up (1)'],
        filled=True, rounded=True,
        fontsize=9, ax=ax3,
        impurity=True, proportion=False,
    )
    ax3.set_title(
        f'Random Forest — Representative Tree #{best_tree_idx}\n'
        f'(max_depth={best_depth}, n_estimators={best_n_est}, '
        f'{leaf_counts[best_tree_idx]} leaves)',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    out_path3 = os.path.join(output_dir, 'base_rf_best_tree.png')
    plt.savefig(out_path3, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {os.path.abspath(out_path3)}")
    plt.close()

    # ===================================================================
    # MODEL COMPARISON TABLE
    # ===================================================================
    print("\n========== Model Comparison Table ==========")

    def _rf_metrics(name, grid_obj, X_te, y_te, tscv_splitter):
        idx_best = grid_obj.best_index_
        res      = grid_obj.cv_results_
        preds    = grid_obj.predict(X_te)
        return {
            'Model':             name,
            'Avg Train Acc':     round(res['mean_train_score'][idx_best], 4),
            'Std Train Acc':     round(res['std_train_score'][idx_best],  4),
            'Avg CV Test Acc':   round(res['mean_test_score'][idx_best],  4),
            'Std CV Test Acc':   round(res['std_test_score'][idx_best],   4),
            'Hold-out Test Acc': round(grid_obj.score(X_te, y_te),        4),
            'Precision':         round(precision_score(y_te, preds, zero_division=0), 4),
            'Recall':            round(recall_score(y_te, preds,    zero_division=0), 4),
            'F1':                round(f1_score(y_te, preds,         zero_division=0), 4),
        }

    rows = [
        _rf_metrics('Base RF',         grid_search_rf,    X_test, y_test, tscv),
        _rf_metrics('PCA RF',          grid_search_pca,   X_test, y_test, tscv),
        _rf_metrics('LASSO-sel RF',    grid_search_lasso, X_test, y_test, tscv),
        _rf_metrics('Ridge-sel RF',    grid_search_ridge, X_test, y_test, tscv),
    ]

    comparison_df = pd.DataFrame(rows).set_index('Model')
    print(comparison_df.to_string())

    csv_path = os.path.join(output_dir, 'base_rf_comparison.csv')
    comparison_df.to_csv(csv_path)
    print(f"\nComparison table (raw) saved to: {os.path.abspath(csv_path)}")

    # ===================================================================
    # DAY-OF-WEEK EXTENSION
    # Add one-hot encoded day-of-week (Mon–Thu, drop Fri) to raw OHLCV
    # then re-run all 4 RF model variants
    # ===================================================================
    print("\n========== Adding Day-of-Week Features ==========")

    dow_dummies = pd.get_dummies(X.index.dayofweek, prefix='DOW').astype(float)
    dow_dummies.index = X.index
    dow_dummies = dow_dummies.iloc[:, :-1]   # drop last column (Fri) to avoid dummy trap
    print(f"Day-of-week columns added: {list(dow_dummies.columns)}")

    X_dow = pd.concat([X, dow_dummies], axis=1)
    X_train_dow, X_test_dow, _, _ = train_test_split(
        X_dow, y_classification, test_size=0.2, random_state=1
    )
    print(f"Feature matrix with DOW: {X_dow.shape[1]} columns")

    # --- Base RF + DOW ---
    print("\n========== Base RF + DOW GridSearch ==========")
    _rf_dow_cache = os.path.join(CACHE_DIR, 'base_rf_dow_gridsearch.pkl')
    param_grid_base = {
        'classifier__max_depth':    [2, 3, 5, 8, 15],
        'classifier__n_estimators': [50, 100, 200, 500],
    }
    if os.path.exists(_rf_dow_cache):
        print("Loading Base RF+DOW from cache...")
        grid_search_rf_dow = joblib.load(_rf_dow_cache)
    else:
        grid_search_rf_dow = GridSearchCV(
            Pipeline([('scaler', StandardScaler()),
                      ('classifier', RandomForestClassifier(random_state=1, n_jobs=-1))]),
            param_grid_base, cv=tscv, n_jobs=-1, return_train_score=True, verbose=1
        )
        grid_search_rf_dow.fit(X_train_dow, y_train)
        joblib.dump(grid_search_rf_dow, _rf_dow_cache)
    print(f"Best params (Base RF+DOW): {grid_search_rf_dow.best_params_}")
    get_final_metrics_grid(grid_search_rf_dow, X_test_dow, y_test)

    # --- PCA RF + DOW ---
    print("\n========== PCA RF + DOW GridSearch ==========")
    _rf_pca_dow_cache = os.path.join(CACHE_DIR, 'base_rf_pca_dow_gridsearch.pkl')
    param_grid_pca_dow = {
        'reducer__n_components':    [0.8, 0.9, 0.95],
        'classifier__max_depth':    [2, 3, 5],
        'classifier__n_estimators': [250, 500],
    }
    if os.path.exists(_rf_pca_dow_cache):
        print("Loading PCA RF+DOW from cache...")
        grid_search_pca_dow = joblib.load(_rf_pca_dow_cache)
    else:
        grid_search_pca_dow = GridSearchCV(
            Pipeline([('scaler', StandardScaler()),
                      ('reducer', PCA()),
                      ('classifier', RandomForestClassifier(random_state=1, n_jobs=-1))]),
            param_grid_pca_dow, cv=tscv, n_jobs=-1, return_train_score=True, verbose=1
        )
        grid_search_pca_dow.fit(X_train_dow, y_train)
        joblib.dump(grid_search_pca_dow, _rf_pca_dow_cache)
    print(f"Best params (PCA RF+DOW): {grid_search_pca_dow.best_params_}")
    get_final_metrics_grid(grid_search_pca_dow, X_test_dow, y_test)

    # --- LASSO-sel RF + DOW ---
    print("\n========== LASSO-sel RF + DOW GridSearch ==========")
    _rf_lasso_dow_cache = os.path.join(CACHE_DIR, 'base_rf_lasso_dow_gridsearch.pkl')
    param_grid_lasso_dow = {
        'feature_selector__estimator__C': [0.001, 0.01, 0.1],
        'classifier__max_depth':          [2, 3, 5],
        'classifier__n_estimators':       [500],
    }
    if os.path.exists(_rf_lasso_dow_cache):
        print("Loading LASSO-sel RF+DOW from cache...")
        grid_search_lasso_dow = joblib.load(_rf_lasso_dow_cache)
    else:
        grid_search_lasso_dow = GridSearchCV(
            Pipeline([('scaler', StandardScaler()),
                      ('feature_selector', SelectFromModel(
                          LogisticRegression(l1_ratio=1, solver='saga', random_state=1,
                                             max_iter=500, tol=5e-2), threshold='mean')),
                      ('classifier', RandomForestClassifier(random_state=1, n_jobs=-1))]),
            param_grid_lasso_dow, cv=tscv, n_jobs=-1, return_train_score=True, verbose=1
        )
        grid_search_lasso_dow.fit(X_train_dow, y_train)
        joblib.dump(grid_search_lasso_dow, _rf_lasso_dow_cache)
    print(f"Best params (LASSO-sel RF+DOW): {grid_search_lasso_dow.best_params_}")
    get_final_metrics_grid(grid_search_lasso_dow, X_test_dow, y_test)

    # --- Ridge-sel RF + DOW ---
    print("\n========== Ridge-sel RF + DOW GridSearch ==========")
    _rf_ridge_dow_cache = os.path.join(CACHE_DIR, 'base_rf_ridge_dow_gridsearch.pkl')
    param_grid_ridge_dow = {
        'feature_selector__estimator__C': [0.001, 0.01, 0.1],
        'classifier__max_depth':          [2, 3, 5],
        'classifier__n_estimators':       [500],
    }
    if os.path.exists(_rf_ridge_dow_cache):
        print("Loading Ridge-sel RF+DOW from cache...")
        grid_search_ridge_dow = joblib.load(_rf_ridge_dow_cache)
    else:
        grid_search_ridge_dow = GridSearchCV(
            Pipeline([('scaler', StandardScaler()),
                      ('feature_selector', SelectFromModel(
                          LogisticRegression(l1_ratio=0, solver='saga', random_state=1,
                                             max_iter=500, tol=5e-2), threshold='mean')),
                      ('classifier', RandomForestClassifier(random_state=1, n_jobs=-1))]),
            param_grid_ridge_dow, cv=tscv, n_jobs=-1, return_train_score=True, verbose=1
        )
        grid_search_ridge_dow.fit(X_train_dow, y_train)
        joblib.dump(grid_search_ridge_dow, _rf_ridge_dow_cache)
    print(f"Best params (Ridge-sel RF+DOW): {grid_search_ridge_dow.best_params_}")
    get_final_metrics_grid(grid_search_ridge_dow, X_test_dow, y_test)

    # ===================================================================
    # COMBINED COMPARISON TABLE (raw OHLCV vs raw OHLCV + DOW)
    # ===================================================================
    rows_dow = [
        _rf_metrics('Base RF+DOW',      grid_search_rf_dow,    X_test_dow, y_test, tscv),
        _rf_metrics('PCA RF+DOW',       grid_search_pca_dow,   X_test_dow, y_test, tscv),
        _rf_metrics('LASSO-sel RF+DOW', grid_search_lasso_dow, X_test_dow, y_test, tscv),
        _rf_metrics('Ridge-sel RF+DOW', grid_search_ridge_dow, X_test_dow, y_test, tscv),
    ]
    dow_df = pd.DataFrame(rows_dow).set_index('Model')

    combined_df = pd.concat([comparison_df, dow_df])
    print("\n===== Combined Comparison Table =====")
    print(combined_df.to_string())

    combined_csv = os.path.join(output_dir, 'base_rf_comparison.csv')
    combined_df.to_csv(combined_csv)
    print(f"\nCombined comparison table saved to: {os.path.abspath(combined_csv)}")

    tex_path = os.path.join(output_dir, 'base_rf_comparison.tex')
    combined_df.to_latex(tex_path, float_format='%.4f',
                         caption='Random Forest Model Comparison: '
                                 'Raw OHLCV vs Raw OHLCV + Day-of-Week',
                         label='tab:base_rf_comparison')
    print(f"LaTeX table saved to:               {os.path.abspath(tex_path)}")

    # ===================================================================
    # LAG1–LAG7 EXTENSION
    # Generate lag1..lag7 of raw OHLCV features (no DOW), then re-run
    # all 4 RF model variants
    # ===================================================================
    print("\n========== Adding Lag1–Lag7 of Raw OHLCV Features ==========")

    lag_frames = [X]
    for lag in range(1, 8):
        lagged = X.shift(lag).add_suffix(f'_lag{lag}')
        lag_frames.append(lagged)
    X_lag = pd.concat(lag_frames, axis=1).dropna()

    # Re-align target to the rows that survived the lag dropna
    y_lag = y_classification.loc[X_lag.index]

    X_train_lag, X_test_lag, y_train_lag, y_test_lag = train_test_split(
        X_lag, y_lag, test_size=0.2, random_state=1
    )
    print(f"Feature matrix with lags: {X_lag.shape[1]} columns, {X_lag.shape[0]} rows")

    # --- Base RF + Lags ---
    print("\n========== Base RF + Lags GridSearch ==========")
    _rf_lag_cache = os.path.join(CACHE_DIR, 'base_rf_lag_gridsearch.pkl')
    if os.path.exists(_rf_lag_cache):
        print("Loading Base RF+Lags from cache...")
        grid_search_rf_lag = joblib.load(_rf_lag_cache)
    else:
        grid_search_rf_lag = GridSearchCV(
            Pipeline([('scaler', StandardScaler()),
                      ('classifier', RandomForestClassifier(random_state=1, n_jobs=-1))]),
            {'classifier__max_depth': [2, 3, 5, 8, 15],
             'classifier__n_estimators': [50, 100, 200, 500]},
            cv=tscv, n_jobs=-1, return_train_score=True, verbose=1
        )
        grid_search_rf_lag.fit(X_train_lag, y_train_lag)
        joblib.dump(grid_search_rf_lag, _rf_lag_cache)
    print(f"Best params (Base RF+Lags): {grid_search_rf_lag.best_params_}")
    get_final_metrics_grid(grid_search_rf_lag, X_test_lag, y_test_lag)

    # --- PCA RF + Lags ---
    print("\n========== PCA RF + Lags GridSearch ==========")
    _rf_pca_lag_cache = os.path.join(CACHE_DIR, 'base_rf_pca_lag_gridsearch.pkl')
    if os.path.exists(_rf_pca_lag_cache):
        print("Loading PCA RF+Lags from cache...")
        grid_search_pca_lag = joblib.load(_rf_pca_lag_cache)
    else:
        grid_search_pca_lag = GridSearchCV(
            Pipeline([('scaler', StandardScaler()),
                      ('reducer', PCA()),
                      ('classifier', RandomForestClassifier(random_state=1, n_jobs=-1))]),
            {'reducer__n_components': [0.8, 0.9, 0.95],
             'classifier__max_depth': [2, 3, 5],
             'classifier__n_estimators': [250, 500]},
            cv=tscv, n_jobs=-1, return_train_score=True, verbose=1
        )
        grid_search_pca_lag.fit(X_train_lag, y_train_lag)
        joblib.dump(grid_search_pca_lag, _rf_pca_lag_cache)
    print(f"Best params (PCA RF+Lags): {grid_search_pca_lag.best_params_}")
    get_final_metrics_grid(grid_search_pca_lag, X_test_lag, y_test_lag)

    # --- LASSO-sel RF + Lags ---
    print("\n========== LASSO-sel RF + Lags GridSearch ==========")
    _rf_lasso_lag_cache = os.path.join(CACHE_DIR, 'base_rf_lasso_lag_gridsearch.pkl')
    if os.path.exists(_rf_lasso_lag_cache):
        print("Loading LASSO-sel RF+Lags from cache...")
        grid_search_lasso_lag = joblib.load(_rf_lasso_lag_cache)
    else:
        grid_search_lasso_lag = GridSearchCV(
            Pipeline([('scaler', StandardScaler()),
                      ('feature_selector', SelectFromModel(
                          LogisticRegression(l1_ratio=1, solver='saga', random_state=1,
                                             max_iter=500, tol=5e-2), threshold='mean')),
                      ('classifier', RandomForestClassifier(random_state=1, n_jobs=-1))]),
            {'feature_selector__estimator__C': [0.001, 0.01, 0.1],
             'classifier__max_depth': [2, 3, 5],
             'classifier__n_estimators': [500]},
            cv=tscv, n_jobs=-1, return_train_score=True, verbose=1
        )
        grid_search_lasso_lag.fit(X_train_lag, y_train_lag)
        joblib.dump(grid_search_lasso_lag, _rf_lasso_lag_cache)
    print(f"Best params (LASSO-sel RF+Lags): {grid_search_lasso_lag.best_params_}")
    get_final_metrics_grid(grid_search_lasso_lag, X_test_lag, y_test_lag)

    # --- Ridge-sel RF + Lags ---
    print("\n========== Ridge-sel RF + Lags GridSearch ==========")
    _rf_ridge_lag_cache = os.path.join(CACHE_DIR, 'base_rf_ridge_lag_gridsearch.pkl')
    if os.path.exists(_rf_ridge_lag_cache):
        print("Loading Ridge-sel RF+Lags from cache...")
        grid_search_ridge_lag = joblib.load(_rf_ridge_lag_cache)
    else:
        grid_search_ridge_lag = GridSearchCV(
            Pipeline([('scaler', StandardScaler()),
                      ('feature_selector', SelectFromModel(
                          LogisticRegression(l1_ratio=0, solver='saga', random_state=1,
                                             max_iter=500, tol=5e-2), threshold='mean')),
                      ('classifier', RandomForestClassifier(random_state=1, n_jobs=-1))]),
            {'feature_selector__estimator__C': [0.001, 0.01, 0.1],
             'classifier__max_depth': [2, 3, 5],
             'classifier__n_estimators': [500]},
            cv=tscv, n_jobs=-1, return_train_score=True, verbose=1
        )
        grid_search_ridge_lag.fit(X_train_lag, y_train_lag)
        joblib.dump(grid_search_ridge_lag, _rf_ridge_lag_cache)
    print(f"Best params (Ridge-sel RF+Lags): {grid_search_ridge_lag.best_params_}")
    get_final_metrics_grid(grid_search_ridge_lag, X_test_lag, y_test_lag)

    # ===================================================================
    # FULL COMPARISON TABLE (raw + DOW + Lags)
    # ===================================================================
    rows_lag = [
        _rf_metrics('Base RF+Lags',      grid_search_rf_lag,    X_test_lag, y_test_lag, tscv),
        _rf_metrics('PCA RF+Lags',       grid_search_pca_lag,   X_test_lag, y_test_lag, tscv),
        _rf_metrics('LASSO-sel RF+Lags', grid_search_lasso_lag, X_test_lag, y_test_lag, tscv),
        _rf_metrics('Ridge-sel RF+Lags', grid_search_ridge_lag, X_test_lag, y_test_lag, tscv),
    ]
    lag_df = pd.DataFrame(rows_lag).set_index('Model')

    full_df = pd.concat([comparison_df, dow_df, lag_df])
    print("\n===== Full Comparison Table =====")
    print(full_df.to_string())

    full_csv = os.path.join(output_dir, 'base_rf_comparison.csv')
    full_df.to_csv(full_csv)
    print(f"\nFull comparison table saved to: {os.path.abspath(full_csv)}")

    tex_path = os.path.join(output_dir, 'base_rf_comparison.tex')
    full_df.to_latex(tex_path, float_format='%.4f',
                     caption='Random Forest Model Comparison: '
                             'Raw OHLCV vs +Day-of-Week vs +Lag1--7',
                     label='tab:base_rf_comparison')
    print(f"LaTeX table saved to:           {os.path.abspath(tex_path)}")
