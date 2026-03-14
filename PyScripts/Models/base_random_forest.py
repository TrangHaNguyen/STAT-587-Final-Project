#!/usr/bin/env python3
import os
import time
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

MPLCONFIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.mplconfig')
os.makedirs(MPLCONFIGDIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(MPLCONFIGDIR))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import (train_test_split, TimeSeriesSplit,
                                     GridSearchCV, cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from H_prep import clean_data, import_data
from H_modeling import (
    fit_or_load_baseline_logistic_pca_search,
    fit_or_load_fixed_classifier_pca_search,
    transform_with_fitted_scaler_pca,
)
from H_eval import (
    get_final_metrics,
    rank_models_by_metrics,
    select_non_degenerate_plot_model,
    save_best_model_plots_from_gridsearch_all_params,
    register_global_model_candidates,
    build_compact_export_table,
    write_grouped_latex_table,
)
from H_search_history import (
    append_search_run,
    get_checkpoint_dir,
    get_git_commit,
    load_search_checkpoint,
    now_iso,
    save_search_checkpoint,
    search_checkpoint_exists,
)
from model_grids import BASE_RF_PARAM_GRID, PCA_RF_PARAM_GRID, TEST_SIZE, TIME_SERIES_CV_SPLITS, TRAIN_TEST_SHUFFLE

MODEL_N_JOBS = int(os.getenv("MODEL_N_JOBS", "-1"))
GRID_VERSION = os.getenv("GRID_VERSION", "v1")
SEARCH_NOTES = os.getenv("SEARCH_NOTES", "")
# Keep the outer GridSearchCV/cross-validation parallel, but make each
# RandomForest fit single-threaded to avoid nested parallel oversubscription.
RF_FIT_N_JOBS = 1
GRIDSEARCH_VERBOSE = int(os.getenv("GRIDSEARCH_VERBOSE", "0"))
# RollingWindowBacktest controls kept here as comments for possible later reuse.
# RUN_BACKTEST = os.getenv("RUN_BACKTEST", "0") == "1"
# BACKTEST_VERBOSE = int(os.getenv("BACKTEST_VERBOSE", "0"))
BACKTEST_WINDOW_SIZE = 100
BACKTEST_HORIZON = 30
USE_SAMPLE_PARQUET = os.getenv("USE_SAMPLE_PARQUET", "0") == "1"
SAMPLE_PARQUET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'Data', 'sample.parquet'
)

def to_binary_class(y):
    return (y >= 0).astype(int)


def _keep_raw_stock_ohlcv(X: pd.DataFrame) -> pd.DataFrame:
    idx = pd.IndexSlice
    metrics = ['Open', 'Close', 'High', 'Low', 'Volume']
    return X.loc[:, idx[metrics, 'Stocks', :]].copy()


def _assert_no_lag_features(X: pd.DataFrame) -> None:
    lag_like_cols = [col for col in X.columns if "_lag" in str(col).lower() or " lag " in str(col).lower()]
    if lag_like_cols:
        sample = lag_like_cols[:5]
        raise ValueError(f"Raw RF should not include lag features. Found lag-like columns: {sample}")

def _as_sortable_numeric(value):
    try:
        return float(value)
    except Exception:
        return float("inf")


def _rf_max_features_sort_value(value):
    if value == 'log2':
        return 0.0
    if value == 'sqrt':
        return 1.0
    try:
        return 2.0 + float(value)
    except Exception:
        return float("inf")

def make_one_se_refit(complexity_cols: list[str], fixed_cols: list[str] | None = None):
    """Return a GridSearchCV refit callable implementing the 1-SE rule."""
    def _pick_index(cv_results):
        mean = np.asarray(cv_results["mean_test_score"], dtype=float)
        std = np.asarray(cv_results["std_test_score"], dtype=float)
        se = std / np.sqrt(TIME_SERIES_CV_SPLITS)
        best_idx = int(np.argmax(mean))
        threshold = float(mean[best_idx] - se[best_idx])
        candidate_idx = np.where(mean >= threshold)[0]
        if len(candidate_idx) == 0:
            return best_idx
        if fixed_cols:
            for col in fixed_cols:
                param_key = f"param_{col}"
                best_val = cv_results[param_key][best_idx]
                candidate_idx = np.array([i for i in candidate_idx if cv_results[param_key][i] == best_val], dtype=int)
                if len(candidate_idx) == 0:
                    return best_idx

        def key_fn(i: int):
            complexity = []
            for col in complexity_cols:
                param_key = f"param_{col}"
                val = cv_results[param_key][i]
                if col == 'classifier__max_features':
                    complexity.append(_rf_max_features_sort_value(val))
                else:
                    complexity.append(_as_sortable_numeric(val))
            # Prefer simplest model; if tie, prefer higher score.
            return tuple(complexity + [-float(mean[i])])

        return int(min(candidate_idx, key=key_fn))

    return _pick_index

RECALL_NOTE = "Recall = positive-class sensitivity."

def _highlight_selected_value(
    ax,
    x_vals,
    curve,
    selected_idx,
    label_prefix="Value at best CV balanced error"
):
    ax.scatter(
        [x_vals[selected_idx]],
        [curve[selected_idx]],
        color='gold',
        edgecolor='black',
        s=90,
        zorder=6,
        label=f'{label_prefix} point'
    )

def _select_index_for_value(x_vals, selected_value):
    x_vals = np.asarray(x_vals, dtype=float)
    return int(np.argmin(np.abs(x_vals - float(selected_value))))


def _compute_cv_metric_curves(model_factory, X_train, y_train, cv):
    """Return plain/balanced train and CV error curves with fold SDs."""
    train_plain_errors, cv_plain_errors = [], []
    train_bal_errors, cv_bal_errors = [], []

    for train_idx, test_idx in cv.split(X_train):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_test = X_train.iloc[test_idx]
        y_fold_test = y_train.iloc[test_idx]

        fold_train_plain = []
        fold_cv_plain = []
        fold_train_bal = []
        fold_cv_bal = []

        for model in model_factory():
            model.fit(X_fold_train, y_fold_train)
            y_pred_train = model.predict(X_fold_train)
            y_pred_test = model.predict(X_fold_test)
            fold_train_plain.append(1 - accuracy_score(y_fold_train, y_pred_train))
            fold_cv_plain.append(1 - accuracy_score(y_fold_test, y_pred_test))
            fold_train_bal.append(1 - balanced_accuracy_score(y_fold_train, y_pred_train))
            fold_cv_bal.append(1 - balanced_accuracy_score(y_fold_test, y_pred_test))

        train_plain_errors.append(fold_train_plain)
        cv_plain_errors.append(fold_cv_plain)
        train_bal_errors.append(fold_train_bal)
        cv_bal_errors.append(fold_cv_bal)

    train_plain_errors = np.asarray(train_plain_errors, dtype=float)
    cv_plain_errors = np.asarray(cv_plain_errors, dtype=float)
    train_bal_errors = np.asarray(train_bal_errors, dtype=float)
    cv_bal_errors = np.asarray(cv_bal_errors, dtype=float)

    return {
        'train_plain_err_mean': train_plain_errors.mean(axis=0),
        'train_plain_err_std': train_plain_errors.std(axis=0),
        'cv_plain_err_mean': cv_plain_errors.mean(axis=0),
        'cv_plain_err_std': cv_plain_errors.std(axis=0),
        'train_bal_err_mean': train_bal_errors.mean(axis=0),
        'train_bal_err_std': train_bal_errors.std(axis=0),
        'cv_bal_err_mean': cv_bal_errors.mean(axis=0),
        'cv_bal_err_std': cv_bal_errors.std(axis=0),
        'cv_bal_err_se': cv_bal_errors.std(axis=0) / np.sqrt(cv.get_n_splits()),
    }

def _base_rf_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=1, n_jobs=RF_FIT_N_JOBS, class_weight='balanced'))
    ])

def _pca_rf_pipeline():
    return Pipeline([
        ('classifier', RandomForestClassifier(random_state=1, n_jobs=RF_FIT_N_JOBS, class_weight='balanced'))
    ])

def _run_grid_search(checkpoint_dir, stage_name, pipeline, param_grid, X_train, y_train, tscv, refit, heading):
    print(f"\n========== {heading} ==========")
    if search_checkpoint_exists(checkpoint_dir, stage_name):
        print(f"Loading checkpoint from {checkpoint_dir / stage_name}")
        return load_search_checkpoint(checkpoint_dir, stage_name)
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=tscv,
        n_jobs=MODEL_N_JOBS, return_train_score=True, verbose=GRIDSEARCH_VERBOSE,
        scoring='balanced_accuracy', refit=refit
    )
    grid_search.fit(X_train, y_train)
    save_search_checkpoint(checkpoint_dir, stage_name, grid_search)
    return grid_search

def _run_model_report_and_backtest(search_obj, X_full, y_full, X_train, y_train, X_test, y_test, label, n_splits):
    print("\n--- Model Report ---")
    shared = get_final_metrics(search_obj.best_estimator_, X_train, y_train, X_test, y_test, n_splits=n_splits, label=label)
    # RollingWindowBacktest disabled to save runtime. Restore this block if needed later.
    # print("\n--- Rolling Window Backtest ---")
    # rwb_obj = RollingWindowBacktest(
    #     clone(search_obj.best_estimator_),
    #     X_full,
    #     y_full,
    #     X_train,
    #     window_size=BACKTEST_WINDOW_SIZE,
    #     horizon=BACKTEST_HORIZON,
    # )
    # rwb_obj.rolling_window_backtest(verbose=BACKTEST_VERBOSE)
    return shared

def _run_rf_suite(
    checkpoint_dir,
    X_full,
    y_full,
    X_train,
    y_train,
    X_test,
    y_test,
    tscv,
    *,
    history_path,
    run_time,
    grid_label,
    dataset_suffix="",
    variant_label="",
    search_notes="",
):
    heading_suffix = f" {variant_label}" if variant_label else ""
    pca_stage_name = f"{dataset_suffix.strip('_') or 'raw'}_shared_logreg_pca_base"
    pca_model_name = f"LogReg_PCA_Base_for_base_rf{dataset_suffix or '_raw'}"
    pca_search = fit_or_load_baseline_logistic_pca_search(
        checkpoint_dir=checkpoint_dir,
        stage_name=pca_stage_name,
        X_train=X_train,
        y_train=y_train,
        history_path=history_path,
        run_time=run_time,
        model_name=pca_model_name,
        grid_label=grid_label,
        n_splits=TIME_SERIES_CV_SPLITS,
        notes=search_notes,
    )
    X_train_pca, X_test_pca, _, _ = transform_with_fitted_scaler_pca(
        pca_search,
        X_train,
        X_test,
    )
    initial_pca_n_components = pca_search.best_params_['pca__n_components']
    print(
        f"Initial PCA from 1SE no-regularization logistic regression in base.py{heading_suffix}: "
        f"n_components={initial_pca_n_components} ({X_train_pca.shape[1]} components)."
    )
    specs = [
        {
            'name': 'Raw RF',
            'heading': 'RF GridSearch (20 combinations)' if not variant_label else f'Raw RF{heading_suffix} GridSearch',
            'pipeline': _base_rf_pipeline(),
            'param_grid': BASE_RF_PARAM_GRID,
            'refit': make_one_se_refit(['classifier__max_depth', 'classifier__n_estimators', 'classifier__max_features']),
        },
        {
            'name': 'PCA RF',
            'heading': 'PCA + RF GridSearch' if not variant_label else f'PCA RF{heading_suffix} GridSearch',
            'pipeline': _pca_rf_pipeline(),
            'param_grid': {
                'classifier__max_depth': PCA_RF_PARAM_GRID['classifier__max_depth'],
                'classifier__n_estimators': PCA_RF_PARAM_GRID['classifier__n_estimators'],
                'classifier__max_features': PCA_RF_PARAM_GRID['classifier__max_features'],
            },
            'refit': make_one_se_refit(['classifier__max_depth', 'classifier__n_estimators', 'classifier__max_features']),
            'X_train_fit': X_train_pca,
            'X_test_eval': X_test_pca,
        },
    ]

    searches = {}
    metric_rows = {}
    for spec in specs:
        stage_name = f"{dataset_suffix.strip('_') or 'raw'}_{spec['name'].lower().replace(' ', '_')}"
        search_obj = _run_grid_search(
            checkpoint_dir,
            stage_name,
            spec['pipeline'],
            spec['param_grid'],
            spec.get('X_train_fit', X_train),
            y_train,
            tscv,
            spec['refit'],
            spec['heading']
        )
        best_prefix = "Best params" if spec['name'] == 'Raw RF' and not variant_label else f"Best params ({spec['name']}{heading_suffix})"
        print(f"{best_prefix}: {search_obj.best_params_}")
        report_label = f"{spec['name']}{heading_suffix}"
        X_test_eval = spec.get('X_test_eval', X_test)
        X_train_eval = spec.get('X_train_fit', X_train)
        metric_rows[spec['name']] = _run_model_report_and_backtest(
            search_obj, X_full, y_full, X_train_eval, y_train, X_test_eval, y_test, report_label, tscv.n_splits
        )
        searches[spec['name']] = search_obj
    pca_rf_search = searches['PCA RF']
    fixed_pca_rf = RandomForestClassifier(
        random_state=1,
        n_jobs=RF_FIT_N_JOBS,
        class_weight='balanced',
        max_depth=pca_rf_search.best_params_['classifier__max_depth'],
        n_estimators=pca_rf_search.best_params_['classifier__n_estimators'],
        max_features=pca_rf_search.best_params_['classifier__max_features'],
    )
    pca_retune_stage = f"{dataset_suffix.strip('_') or 'raw'}_pca_rf_retuned_n_components"
    pca_retune_model_name = f"PCA_RF_retuned_n_components{dataset_suffix or '_raw'}"
    pca_search = fit_or_load_fixed_classifier_pca_search(
        checkpoint_dir=checkpoint_dir,
        stage_name=pca_retune_stage,
        X_train=X_train,
        y_train=y_train,
        history_path=history_path,
        run_time=run_time,
        model_name=pca_retune_model_name,
        grid_label=grid_label,
        n_splits=TIME_SERIES_CV_SPLITS,
        classifier=fixed_pca_rf,
        notes=search_notes,
    )
    X_train_pca, X_test_pca, _, _ = transform_with_fitted_scaler_pca(
        pca_search,
        X_train,
        X_test,
    )
    selected_pca_n_components = pca_search.best_params_['pca__n_components']
    print(
        f"Retuned PCA for PCA RF after RF model selection{heading_suffix}: "
        f"n_components={selected_pca_n_components} ({X_train_pca.shape[1]} components)."
    )
    stage_name = f"{dataset_suffix.strip('_') or 'raw'}_pca_rf_refit"
    pca_search_obj = _run_grid_search(
        checkpoint_dir,
        stage_name,
        _pca_rf_pipeline(),
        {
            'classifier__max_depth': PCA_RF_PARAM_GRID['classifier__max_depth'],
            'classifier__n_estimators': PCA_RF_PARAM_GRID['classifier__n_estimators'],
            'classifier__max_features': PCA_RF_PARAM_GRID['classifier__max_features'],
        },
        X_train_pca,
        y_train,
        tscv,
        make_one_se_refit(['classifier__max_depth', 'classifier__n_estimators', 'classifier__max_features']),
        'PCA + RF GridSearch (retuned PCA)' if not variant_label else f'PCA RF{heading_suffix} GridSearch (retuned PCA)',
    )
    print(f"Best params (PCA RF retuned{heading_suffix}): {pca_search_obj.best_params_}")
    metric_rows['PCA RF'] = _run_model_report_and_backtest(
        pca_search_obj,
        X_full,
        y_full,
        X_train_pca,
        y_train,
        X_test_pca,
        y_test,
        f"PCA RF{heading_suffix}",
        tscv.n_splits,
    )
    searches['PCA RF'] = pca_search_obj
    return searches, metric_rows, {
        'pca_search': pca_search,
        'X_train_pca': X_train_pca,
        'X_test_pca': X_test_pca,
        'selected_pca_n_components': selected_pca_n_components,
    }

def _rf_metrics_payload(name, shared):
    display_row = {
        'Model':             name,
        'Avg CV Train Plain Acc':      shared['train_avg_accuracy'],
        'CV Train Plain Acc SD':       shared['train_std_accuracy'],
        'Avg CV Validation Plain Acc': shared['validation_avg_accuracy'],
        'CV Acc SD':                   shared['validation_std_accuracy'],
        'Test Acc':                    shared['test_split_accuracy'],
        'MCC':               shared['test_matthew_corr_coef'],
        'Precision':         shared['test_precision'],
        'Recall':            shared['test_recall'],
        'Specificity':       shared['test_specificity'],
        'F1':                shared['test_f1'],
        'ROC-AUC':           shared['test_roc_auc_macro'],
    }
    ranking_row = {'Model': name, **shared}
    return display_row, ranking_row

def _build_comparison_df(metric_rows, suffix=""):
    display_rows = []
    ranking_rows = []
    for model_name in ['Raw RF', 'PCA RF']:
        display_name = f'{model_name}{suffix}'
        if suffix == "+DOW" and model_name == 'Raw RF':
            display_name = 'RF+DOW'
        display_row, ranking_row = _rf_metrics_payload(
            display_name,
            metric_rows[model_name],
        )
        display_rows.append(display_row)
        ranking_rows.append(ranking_row)
    return pd.DataFrame(display_rows).set_index('Model'), ranking_rows

if __name__ == "__main__":
    run_start = time.time()
    run_time = now_iso()
    # ------- Load raw data (no extra clean_data feature engineering) -------
    TESTING = False
    output_prefix = "sample" if USE_SAMPLE_PARQUET else "8yrs"
    print(f"MODEL_N_JOBS={MODEL_N_JOBS} (set env MODEL_N_JOBS to override)")
    print(f"GRID_VERSION={GRID_VERSION}")
    if USE_SAMPLE_PARQUET:
        print(f"USE_SAMPLE_PARQUET=1 -> loading sample parquet from {os.path.abspath(SAMPLE_PARQUET_PATH)}")
        sample_table = pq.read_table(SAMPLE_PARQUET_PATH)
        DATA = sample_table.to_pandas(), None
    else:
        DATA = import_data(testing=TESTING, extra_features=False, cluster=False, n_clusters=100, corr_threshold=0.95, corr_level=0)
    if USE_SAMPLE_PARQUET:
        idx = pd.IndexSlice
        raw_data = DATA[0]
        print("Finished Downloading Data -------")
        print("Initial shape:", raw_data.shape[0], "rows,", raw_data.shape[1], "columns.")
        print("------- Cleaning data")
        for type in ['Stocks']:
            temp_data = raw_data.loc[:, idx[:, type, :]].dropna(how="all", axis=0)
            missing_one = (temp_data.isna().sum() == 1)
            cols = missing_one[missing_one == 1].index
            temp_data[cols] = temp_data[cols].ffill()
            temp_data = temp_data.dropna(how="any", axis=1)
            raw_data = raw_data.drop(columns=type, level=1).join(temp_data)
        stocks = raw_data.loc[:, idx[:, 'Stocks', :]]
        to_drop = stocks.index[stocks.isna().all(axis=1)]
        raw_data = raw_data.drop(index=to_drop)
        print("Finished Cleaning Data -------")
        print("Current shape:", raw_data.shape[0], "rows,", raw_data.shape[1], "columns.")
        raw_data = pd.concat([
            raw_data,
            raw_data.loc[:, idx[['Close', 'Open', 'High', 'Low'], 'Stocks', :]]
            .copy()
            .pct_change()
            .rename(columns={metric: f"{metric} PC" for metric in ['Close', 'Open', 'High', 'Low']}, level=0)
        ], axis=1)
        y_regression = (
            (raw_data.loc[:, idx['Close', 'Index', '^SPX']] - raw_data.loc[:, idx['Open', 'Index', '^SPX']])
            / raw_data.loc[:, idx['Open', 'Index', '^SPX']]
        ).rename("Target Regression").shift(-1)
        DATA = raw_data, y_regression
    X, y_regression = clean_data(
        *DATA,
        raw=True,
        extra_features=False,
        lag_period=[1],
        lookback_period=0,
    )
    X = _keep_raw_stock_ohlcv(X)
    X.columns = [f"{metric}_{ticker}" for metric, _, ticker in X.columns]
    _assert_no_lag_features(X)
    print(f"Feature matrix shape: {X.shape[0]} rows, {X.shape[1]} columns.")

    y_classification = to_binary_class(y_regression)
    print(f"Final shape — X: {X.shape}, y: {y_classification.shape}")

    # ------- Train/test split (80/20, no shuffle to preserve time order) -------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_classification, test_size=TEST_SIZE, random_state=1, shuffle=TRAIN_TEST_SHUFFLE
    )
    # Previous temporary change used `KFold(n_splits=5, shuffle=False)`.
    tscv = TimeSeriesSplit(n_splits=TIME_SERIES_CV_SPLITS)
    checkpoint_dir = get_checkpoint_dir(
        Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'output')),
        "base_random_forest",
        f"{output_prefix}_{GRID_VERSION}",
    )

    raw_searches, raw_metric_rows, raw_pca_context = _run_rf_suite(
        checkpoint_dir,
        X,
        y_classification,
        X_train,
        y_train,
        X_test,
        y_test,
        tscv,
        history_path=Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'output', f"{output_prefix}_search_history_rf.csv")),
        run_time=run_time,
        grid_label=GRID_VERSION,
        search_notes=SEARCH_NOTES,
    )
    grid_search_rf = raw_searches['Raw RF']
    grid_search_pca = raw_searches['PCA RF']

    param_grid = BASE_RF_PARAM_GRID

    best_depth  = grid_search_rf.best_params_['classifier__max_depth']
    best_n_est  = grid_search_rf.best_params_['classifier__n_estimators']
    best_max_features = grid_search_rf.best_params_['classifier__max_features']

    # ===================================================================
    # PLOT 1: Bias-Variance Tradeoff (CV train + CV test error ± std)
    # Sweep the same max_depth grid used in GridSearchCV.
    # ===================================================================
    print("\n========== Generating Bias-Variance Tradeoff Plot (CV) ==========")

    depth_grid = list(param_grid['classifier__max_depth'])
    def _depth_curve_models():
        models = []
        for depth in depth_grid:
            models.append(Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(
                    max_depth=depth, n_estimators=best_n_est,
                    max_features=best_max_features,
                    random_state=1, n_jobs=RF_FIT_N_JOBS, class_weight='balanced'))
            ]))
        return models

    depth_curves = _compute_cv_metric_curves(_depth_curve_models, X_train, y_train, tscv)

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    fig1.suptitle(
        'Bias-Variance Tradeoff — RF, Raw OHLCV Features\n'
        f'(Train vs CV Balanced Error, n_estimators={best_n_est}, max_features={best_max_features})',
        fontsize=13, fontweight='bold'
    )
    ax1.plot(depth_grid, depth_curves['train_bal_err_mean'], marker='o', color='lightsteelblue',
             linewidth=1.8, label='CV Train balanced error')
    ax1.plot(depth_grid, depth_curves['cv_bal_err_mean'], marker='s', color='navajowhite',
             linewidth=1.8, label='CV Test balanced error')
    ax1.fill_between(
        depth_grid,
        np.clip(depth_curves['train_bal_err_mean'] - depth_curves['train_bal_err_std'], 0.0, 1.0),
        np.clip(depth_curves['train_bal_err_mean'] + depth_curves['train_bal_err_std'], 0.0, 1.0),
        alpha=0.15,
        color='lightsteelblue',
        label='CV Train balanced error ±1 SD'
    )
    ax1.fill_between(
        depth_grid,
        np.clip(depth_curves['cv_bal_err_mean'] - depth_curves['cv_bal_err_std'], 0.0, 1.0),
        np.clip(depth_curves['cv_bal_err_mean'] + depth_curves['cv_bal_err_std'], 0.0, 1.0),
        alpha=0.15,
        color='navajowhite',
        label='CV Test balanced error ±1 SD'
    )
    best_depth_idx = int(np.argmin(depth_curves['cv_bal_err_mean']))
    _highlight_selected_value(
        ax1, depth_grid, depth_curves['cv_bal_err_mean'], best_depth_idx,
        label_prefix='Value at best CV balanced error'
    )
    ax1.axvline(best_depth, color='red', linestyle='--', linewidth=1.5,
                label=f'1SE-selected max_depth = {best_depth}')
    ax1.set_title('RF — Bias-Variance Tradeoff (Balanced Error)')
    ax1.set_xlabel('max_depth\n'
                   '← Low Depth, High Regularization, Simpler Model      '
                   'High Depth, Low Regularization, More Complex →')
    ax1.set_ylabel('Balanced Error (1 - balanced accuracy)')
    ax1.set_ylim(0, 1.02)
    ax1.set_xticks(depth_grid)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    tex_output_dir = output_dir
    os.makedirs(tex_output_dir, exist_ok=True)
    out_path1 = os.path.join(output_dir, f'{output_prefix}_base_rf_bias_variance.png')
    plt.savefig(out_path1, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {os.path.abspath(out_path1)}")
    plt.close()

    # ===================================================================
    # PLOT 2: Direct Train vs Test Error (no CV averaging)
    # Sweep the same max_depth grid used in GridSearchCV.
    # ===================================================================
    print("\n========== Generating Train vs Test Error Plot (Direct Split) ==========")

    scaler_direct = StandardScaler()
    X_tr_sc = scaler_direct.fit_transform(X_train)
    X_te_sc = scaler_direct.transform(X_test)

    train_errors, test_errors = [], []
    for depth in depth_grid:
        clf = RandomForestClassifier(
            max_depth=depth, n_estimators=best_n_est,
            max_features=best_max_features,
            random_state=1, n_jobs=RF_FIT_N_JOBS, class_weight='balanced')
        clf.fit(X_tr_sc, y_train)
        train_errors.append(1 - clf.score(X_tr_sc, y_train))
        test_errors.append(1 - clf.score(X_te_sc, y_test))

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    fig2.suptitle(
        'Over/Underfitting Analysis — RF, Raw OHLCV Features\n'
        f'(Direct Train/Test Split, No CV, n_estimators={best_n_est}, max_features={best_max_features})',
        fontsize=13, fontweight='bold'
    )
    ax2.plot(depth_grid, train_errors, marker='o', color='steelblue',
             linewidth=2, label='Train error')
    ax2.plot(depth_grid, test_errors, marker='s', color='darkorange',
             linewidth=2, label='Test error')
    best_depth_idx = int(np.argmin(depth_curves['cv_bal_err_mean']))
    ax2.scatter(
        [depth_grid[best_depth_idx]], [test_errors[best_depth_idx]],
        color='gold', edgecolor='black', s=90, zorder=6,
        label='Value at best CV balanced error'
    )
    ax2.axvline(best_depth, color='red', linestyle='--', linewidth=1.5,
                label=f'1SE-selected max_depth = {best_depth}')
    ax2.set_title('RF — Train vs Test Error (Plain Error)')
    ax2.set_xlabel('max_depth\n'
                   '← Low Depth, High Regularization, Simpler Model      '
                   'High Depth, Low Regularization, More Complex →')
    ax2.set_ylabel('Plain Error (1 - accuracy)')
    ax2.set_xticks(depth_grid)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path2 = os.path.join(output_dir, f'{output_prefix}_base_rf_train_test.png')
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
        f'RF — Representative Tree #{best_tree_idx}\n'
        f'(max_depth={best_depth}, n_estimators={best_n_est}, max_features={best_max_features}, '
        f'{leaf_counts[best_tree_idx]} leaves)',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    out_path3 = os.path.join(output_dir, f'{output_prefix}_base_rf_best_tree.png')
    plt.savefig(out_path3, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {os.path.abspath(out_path3)}")
    plt.close()

    # ===================================================================
    # MODEL COMPARISON TABLE
    # ===================================================================
    print("\n========== Model Comparison Table ==========")

    comparison_df, raw_ranking_rows = _build_comparison_df(raw_metric_rows)
    print(comparison_df.to_string())

    csv_path = os.path.join(output_dir, f'{output_prefix}_base_rf_comparison.csv')
    comparison_df.to_csv(csv_path, float_format='%.3f')
    print(f"\nComparison table (raw) saved to: {os.path.abspath(csv_path)}")

    # ===================================================================
    # DAY-OF-WEEK EXTENSION
    # Add one-hot encoded day-of-week (Mon–Thu, drop Fri) to raw OHLCV
    # then re-run the remaining RF model variants
    # ===================================================================
    print("\n========== Adding Day-of-Week Features ==========")

    dow_dummies = pd.get_dummies(X.index.dayofweek, prefix='DOW').astype(float)
    dow_dummies.index = X.index
    dow_dummies = dow_dummies.iloc[:, :-1]   # drop last column (Fri) to avoid dummy trap
    print(f"Day-of-week columns added: {list(dow_dummies.columns)}")

    X_dow = pd.concat([X, dow_dummies], axis=1)
    X_train_dow, X_test_dow, y_train_dow, y_test_dow = train_test_split(
        X_dow, y_classification, test_size=TEST_SIZE, random_state=1, shuffle=TRAIN_TEST_SHUFFLE
    )
    print(f"Feature matrix with DOW: {X_dow.shape[1]} columns")

    dow_searches, dow_metric_rows, dow_pca_context = _run_rf_suite(
        checkpoint_dir,
        X_dow,
        y_classification,
        X_train_dow,
        y_train_dow,
        X_test_dow,
        y_test_dow,
        tscv,
        history_path=Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'output', f"{output_prefix}_search_history_rf.csv")),
        run_time=run_time,
        grid_label=GRID_VERSION,
        dataset_suffix="_dow",
        variant_label="+ DOW",
        search_notes=SEARCH_NOTES,
    )

    # ===================================================================
    # COMBINED COMPARISON TABLE (raw OHLCV vs raw OHLCV + DOW)
    # ===================================================================
    dow_df, dow_ranking_rows = _build_comparison_df(dow_metric_rows, suffix="+DOW")

    combined_df = pd.concat([comparison_df, dow_df])

    # ===================================================================
    # LAG1–LAG7 EXTENSION
    # Disabled to reduce runtime for the baseline RF script.
    # Restore this block if lag-feature comparisons are needed again.
    # ===================================================================
    print("\n========== Lag1–Lag7 RF extension skipped to save runtime ==========")

    full_df = combined_df
    print("\n===== Full Comparison Table =====")
    print(full_df.to_string())

    full_export_df = build_compact_export_table(
        full_df,
        index_renames={
            'Raw RF': 'Raw RF',
            'RF+DOW': 'RF+DOW',
            'RF+Lags': 'RF+Lags',
        },
    )
    full_csv = os.path.join(output_dir, f'{output_prefix}_base_rf_comparison.csv')
    full_export_df.to_csv(full_csv, float_format='%.3f')
    print(f"\nFull comparison table saved to: {os.path.abspath(full_csv)}")

    tex_path = os.path.join(tex_output_dir, f'{output_prefix}_1SE_base_random_forest.tex')
    write_grouped_latex_table(
        full_export_df,
        tex_path,
        'Random Forest Model Comparison: Raw OHLCV vs +Day-of-Week',
        'tab:base_rf_comparison',
        note=f'Test Acc = plain hold-out accuracy on the final 20% test split. All reported CV/train/test accuracy columns in this table use plain accuracy after hyperparameters were selected by CV balanced accuracy. {RECALL_NOTE}',
        groups=[
            ['Raw RF', 'PCA RF'],
            ['RF+DOW', 'PCA RF+DOW'],
        ],
    )
    print(f"LaTeX table saved to:           {os.path.abspath(tex_path)}")

    plot_candidates = {
        'Raw RF': {
            'search': raw_searches['Raw RF'],
            'param_specs': [('classifier__max_depth', 'max_depth'), ('classifier__n_estimators', 'n_estimators'), ('classifier__max_features', 'max_features')],
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
        },
        'PCA RF': {
            'search': raw_searches['PCA RF'],
            'param_specs': [('classifier__max_depth', 'max_depth'), ('classifier__n_estimators', 'n_estimators'), ('classifier__max_features', 'max_features')],
            'X_train': raw_pca_context['X_train_pca'],
            'y_train': y_train,
            'X_test': raw_pca_context['X_test_pca'],
            'y_test': y_test,
        },
        'RF+DOW': {
            'search': dow_searches['Raw RF'],
            'param_specs': [('classifier__max_depth', 'max_depth'), ('classifier__n_estimators', 'n_estimators'), ('classifier__max_features', 'max_features')],
            'X_train': X_train_dow,
            'y_train': y_train_dow,
            'X_test': X_test_dow,
            'y_test': y_test_dow,
        },
        'PCA RF+DOW': {
            'search': dow_searches['PCA RF'],
            'param_specs': [('classifier__max_depth', 'max_depth'), ('classifier__n_estimators', 'n_estimators'), ('classifier__max_features', 'max_features')],
            'X_train': dow_pca_context['X_train_pca'],
            'y_train': y_train_dow,
            'X_test': dow_pca_context['X_test_pca'],
            'y_test': y_test_dow,
        },
    }
    ranking_rows = raw_ranking_rows + dow_ranking_rows
    ranked_df = rank_models_by_metrics(pd.DataFrame(ranking_rows))
    best_model_name = str(ranked_df.iloc[0]["Model"])
    plot_model_name = select_non_degenerate_plot_model(ranked_df, available_models=plot_candidates)
    best_candidate = plot_candidates[plot_model_name]
    global_ranked_df = register_global_model_candidates(
        ranked_df,
        Path(output_dir) / f"{output_prefix}_global_model_leaderboard.csv",
        source_script="base_random_forest.py",
        dataset_label=output_prefix,
        comparison_scope="raw_vs_dow",
    )
    save_best_model_plots_from_gridsearch_all_params(
        best_candidate['search'],
        best_candidate['param_specs'],
        plot_model_name,
        os.path.join(output_dir, f'{output_prefix}_base_rf_best_model_bias_variance.png'),
        os.path.join(output_dir, f'{output_prefix}_base_rf_best_model_train_test.png'),
        best_candidate['X_train'],
        best_candidate['y_train'],
        best_candidate['X_test'],
        best_candidate['y_test'],
    )
    print(f"\nBest raw/DOW RF-family model by average rank: {best_model_name}")
    if plot_model_name != best_model_name:
        print(f"Plotting fallback (non-degenerate): {plot_model_name}")
    print(f"Local ranked/exported winner in base_random_forest.py: {best_model_name}")
    print(f"Local plot winner in base_random_forest.py: {plot_model_name}")
    print(f"Current global best model across registered scripts (informational only): {global_ranked_df.iloc[0]['Model']}")
    append_search_run(
        runs_path=Path(output_dir) / f"{output_prefix}_search_runs.csv",
        model_name="base_rf",
        run_time=run_time,
        run_duration_sec=(time.time() - run_start),
        grid_version=GRID_VERSION,
        n_jobs=MODEL_N_JOBS,
        dataset_version="testing=False,extra_features=False,cluster=False,corr_threshold=0.95,corr_level=0",
        code_commit=get_git_commit(Path(output_dir).resolve().parents[0]),
        notes=SEARCH_NOTES,
    )
