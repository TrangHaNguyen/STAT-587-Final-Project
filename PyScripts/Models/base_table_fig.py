#!/usr/bin/env python3
"""
base_table_fig.py — Fast re-rank, re-plot, and LaTeX table for base.py logistic models.

Requires base.py to have been run at least once (stage checkpoints must exist).
Plot curve data is cached in output/base_plot_curves/ — a separate directory that
base.py does NOT clear — so changing TEST_SELECTION_CRITERIA weights is fast after
the first run.

Run order:
  1. python base.py          (trains models, writes stage checkpoints)
  2. python base_table_fig.py  (caches all plot curves, writes plots + LaTeX table)
  3. Edit TEST_SELECTION_CRITERIA weights in H_eval.py
  4. python base_table_fig.py  (< 2 min: loads all caches, re-ranks, re-plots)

NOTE: base.py calls clear_output_checkpoints() at startup, which deletes the
      checkpoint directory. After any base.py re-run, re-run base_table_fig.py
      once to rebuild the plot curve cache. To avoid this, comment out the
      clear_output_checkpoints() call in base.py's __main__ block.
"""

import os
import time
import warnings
from pathlib import Path

import matplotlib
MPLCONFIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.mplconfig')
os.makedirs(MPLCONFIGDIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(MPLCONFIGDIR))
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: F401  (imported for side effects via base helpers)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit

# ---------------------------------------------------------------------------
# Import module-level helpers from base.py.
# Safe: the __main__ block (which calls clear_output_checkpoints) does NOT run
# when base is imported as a module.
# ---------------------------------------------------------------------------
from base import (
    _fit_logistic_model,
    _keep_raw_stock_ohlcv,
    _load_or_compute_stage,
    _format_pca_grid_value,
    _compute_bv_curves,
    _compute_direct_split_errors,
    _compute_pca_curve_diagnostics,
    _plot_single_model_diagnostics,
    _append_plot_suffix,
    _save_curve_pair,
    _augment_c_grid_with_selected_values,
)
from H_prep import clean_data, import_data, to_binary_class
from H_eval import (
    TEST_SELECTION_CRITERIA,
    get_or_compute_final_metrics,
    _metrics_stage_name,
    rank_models_by_metrics,
    select_non_degenerate_plot_model,
    build_compact_export_table,
    write_grouped_latex_table,
    register_global_model_candidates,
)
from H_search_history import (
    get_checkpoint_dir,
    get_git_commit,
    stage_checkpoint_exists,
    load_stage_checkpoint,
    save_stage_checkpoint,
    append_search_run,
    now_iso,
)
from H_helpers import get_cwd
from model_grids import (
    BASELINE_PCA_GRID,
    RANDOM_SEED,
    TEST_SIZE,
    TIME_SERIES_CV_SPLITS,
    TRAIN_TEST_SHUFFLE,
)

MODEL_N_JOBS = int(os.getenv("MODEL_N_JOBS", "-1"))
GRID_VERSION = os.getenv("GRID_VERSION", "v1")

cwd = get_cwd("STAT-587-Final-Project")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_or_compute_plot_curves(plot_cache_dir: Path, stage_name: str, compute_fn):
    """Cache plot curve data in a directory NOT cleared by base.py."""
    if stage_checkpoint_exists(plot_cache_dir, stage_name):
        print(f"  Loading cached plot curves: {stage_name}")
        return load_stage_checkpoint(plot_cache_dir, stage_name)
    print(f"  Computing plot curves (first run): {stage_name}")
    result = compute_fn()
    save_stage_checkpoint(plot_cache_dir, stage_name, result)
    return result


def _require_stage(checkpoint_dir: Path, stage_name: str):
    """Load a base.py stage checkpoint or raise a clear error if missing."""
    if not stage_checkpoint_exists(checkpoint_dir, stage_name):
        raise FileNotFoundError(
            f"\nStage checkpoint '{stage_name}' not found in:\n  {checkpoint_dir}\n"
            f"Run base.py first, then run base_table_fig.py before the next base.py run.\n"
            f"TIP: Comment out clear_output_checkpoints() in base.py's __main__ block "
            f"so checkpoints persist across base.py re-runs."
        )
    return load_stage_checkpoint(checkpoint_dir, stage_name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_start = time.time()
    run_time = now_iso()
    output_prefix = "8yrs"
    output_dir = cwd / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Same checkpoint_dir as base.py (stage checkpoints for fitted models)
    checkpoint_dir = get_checkpoint_dir(output_dir, "base", f"{output_prefix}_{GRID_VERSION}")

    # Separate plot curve cache — NOT cleared by base.py's clear_output_checkpoints()
    plot_cache_dir = output_dir / "base_plot_curves" / f"{output_prefix}_{GRID_VERSION}"
    plot_cache_dir.mkdir(parents=True, exist_ok=True)

    # ===================================================================
    # 1. LOAD DATA
    #    Needed to reconstruct X_train / X_test (for raw scaler),
    #    X_train_dow / X_test_dow (for PCA curve plots with DOW features),
    #    and y_train / y_test (passed to all curve compute functions).
    # ===================================================================
    print("\n===== Loading Data =====")
    DATA = import_data(
        extra_features=False, testing=False, cluster=False,
        n_clusters=100, corr_threshold=0.95, corr_level=0,
    )
    X, y_regression = clean_data(
        *DATA,
        raw=True,
        extra_features=False,
        lag_period=[1],
        lookback_period=0,
    )
    X = _keep_raw_stock_ohlcv(X)
    X.columns = [f"{metric}_{ticker}" for metric, _, ticker in X.columns]
    y_classification = to_binary_class(y_regression)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_classification, test_size=TEST_SIZE, random_state=RANDOM_SEED,
        shuffle=TRAIN_TEST_SHUFFLE,
    )
    tscv = TimeSeriesSplit(n_splits=TIME_SERIES_CV_SPLITS)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Pre-scaled raw features (Ridge / LASSO 'c'-type plot curves expect scaled input)
    raw_scaler = StandardScaler()
    X_train_raw_sc = raw_scaler.fit_transform(X_train)
    X_test_raw_sc  = raw_scaler.transform(X_test)

    # DOW features (needed for PCA curve plots involving DOW models)
    dow_dummies_train = pd.get_dummies(X_train.index.dayofweek, prefix='DOW').astype(float)
    dow_dummies_train.index = X_train.index
    if 'DOW_4' in dow_dummies_train.columns:
        dow_dummies_train = dow_dummies_train.drop(columns=['DOW_4'])
    X_train_dow = pd.concat([X_train, dow_dummies_train], axis=1)

    dow_dummies_test = pd.get_dummies(X_test.index.dayofweek, prefix='DOW').astype(float)
    dow_dummies_test.index = X_test.index
    dow_dummies_test = dow_dummies_test.reindex(columns=dow_dummies_train.columns, fill_value=0.0)
    X_test_dow = pd.concat([X_test, dow_dummies_test], axis=1)

    # ===================================================================
    # 2. LOAD STAGE CHECKPOINTS FROM base.py
    # ===================================================================
    print("\n===== Loading base.py Stage Checkpoints =====")

    # --- Raw models ---
    raw_pca_stage      = _require_stage(checkpoint_dir, "raw_pca_selection")
    best_n_comp        = raw_pca_stage['best_n_comp']
    n_components_raw   = raw_pca_stage['n_components_raw']
    X_train_pca        = raw_pca_stage['X_train_pca']
    X_test_pca         = raw_pca_stage['X_test_pca']

    baseline_clf       = _require_stage(checkpoint_dir, "raw_baseline_pca_model")['baseline_clf']

    ridge_raw_stage    = _require_stage(checkpoint_dir, "raw_ridge_model")
    ridge_cv           = ridge_raw_stage['ridge_cv']
    ridge_c_1se        = ridge_raw_stage['ridge_c_1se']
    pipeline_ridge_1se = ridge_raw_stage['pipeline_ridge_1se']

    lasso_raw_stage    = _require_stage(checkpoint_dir, "raw_lasso_model")
    lasso_cv           = lasso_raw_stage['lasso_cv']
    lasso_c_1se        = lasso_raw_stage['lasso_c_1se']
    pipeline_lasso_1se = lasso_raw_stage['pipeline_lasso_1se']

    ridge_pca_stage      = _require_stage(checkpoint_dir, "raw_ridge_pca_model")
    clf_ridge_pca        = ridge_pca_stage['clf_ridge_pca']
    ridge_pca_c_1se      = ridge_pca_stage['ridge_pca_c_1se']
    clf_ridge_pca_1se    = ridge_pca_stage['clf_ridge_pca_1se']
    best_n_comp_ridge    = ridge_pca_stage['best_n_comp']
    n_components_ridge   = ridge_pca_stage['n_components_raw']
    X_train_pca_ridge    = ridge_pca_stage['X_train_pca']
    X_test_pca_ridge     = ridge_pca_stage['X_test_pca']

    lasso_pca_stage      = _require_stage(checkpoint_dir, "raw_lasso_pca_model")
    clf_lasso_pca        = lasso_pca_stage['clf_lasso_pca']
    lasso_pca_c_1se      = lasso_pca_stage['lasso_pca_c_1se']
    clf_lasso_pca_1se    = lasso_pca_stage['clf_lasso_pca_1se']
    best_n_comp_lasso    = lasso_pca_stage['best_n_comp']
    n_components_lasso   = lasso_pca_stage['n_components_raw']
    X_train_pca_lasso    = lasso_pca_stage['X_train_pca']
    X_test_pca_lasso     = lasso_pca_stage['X_test_pca']

    # --- DOW models ---
    dow_pca_stage        = _require_stage(checkpoint_dir, "dow_pca_selection")
    best_n_comp_dow      = dow_pca_stage['best_n_comp_dow']
    n_components_dow     = dow_pca_stage['n_components_dow']
    X_train_dow_sc       = dow_pca_stage['X_train_dow_sc']
    X_test_dow_sc        = dow_pca_stage['X_test_dow_sc']
    X_train_dow_pca      = dow_pca_stage['X_train_dow_pca']
    X_test_dow_pca       = dow_pca_stage['X_test_dow_pca']

    baseline_dow_clf     = _require_stage(checkpoint_dir, "dow_baseline_pca_model")['baseline_dow_clf']

    ridge_dow_stage      = _require_stage(checkpoint_dir, "dow_ridge_model")
    ridge_dow_cv         = ridge_dow_stage['ridge_dow_cv']
    ridge_dow_c_1se      = ridge_dow_stage['ridge_dow_c_1se']
    pipeline_ridge_dow_1se = ridge_dow_stage['pipeline_ridge_dow_1se']

    lasso_dow_stage      = _require_stage(checkpoint_dir, "dow_lasso_model")
    lasso_dow_cv         = lasso_dow_stage['lasso_dow_cv']
    lasso_dow_c_1se      = lasso_dow_stage['lasso_dow_c_1se']
    pipeline_lasso_dow_1se = lasso_dow_stage['pipeline_lasso_dow_1se']

    ridge_pca_dow_stage      = _require_stage(checkpoint_dir, "dow_ridge_pca_model")
    clf_ridge_pca_dow        = ridge_pca_dow_stage['clf_ridge_pca_dow']
    ridge_pca_dow_c_1se      = ridge_pca_dow_stage['ridge_pca_dow_c_1se']
    clf_ridge_pca_dow_1se    = ridge_pca_dow_stage['clf_ridge_pca_dow_1se']
    best_n_comp_ridge_dow    = ridge_pca_dow_stage['best_n_comp']
    n_components_ridge_dow   = ridge_pca_dow_stage['n_components_raw']
    X_train_pca_ridge_dow    = ridge_pca_dow_stage['X_train_pca']
    X_test_pca_ridge_dow     = ridge_pca_dow_stage['X_test_pca']

    lasso_pca_dow_stage      = _require_stage(checkpoint_dir, "dow_lasso_pca_model")
    clf_lasso_pca_dow        = lasso_pca_dow_stage['clf_lasso_pca_dow']
    lasso_pca_dow_c_1se      = lasso_pca_dow_stage['lasso_pca_dow_c_1se']
    clf_lasso_pca_dow_1se    = lasso_pca_dow_stage['clf_lasso_pca_dow_1se']
    best_n_comp_lasso_dow    = lasso_pca_dow_stage['best_n_comp']
    n_components_lasso_dow   = lasso_pca_dow_stage['n_components_raw']
    X_train_pca_lasso_dow    = lasso_pca_dow_stage['X_train_pca']
    X_test_pca_lasso_dow     = lasso_pca_dow_stage['X_test_pca']

    print("All stage checkpoints loaded.")

    # ===================================================================
    # 3. METRICS TABLE  (loads from get_or_compute_final_metrics cache;
    #    uses same checkpoint_dir and stage names as base.py's _eval_row)
    # ===================================================================
    print("\n===== Building Metrics Table =====")
    ranking_rows = []

    def _eval_row(name, model, X_tr, y_tr, X_te, y_te, n_splits, best_c=None, best_c_label=None):
        shared = get_or_compute_final_metrics(
            checkpoint_dir, _metrics_stage_name(name), model,
            X_tr, y_tr, X_te, y_te, n_splits=n_splits, label=name,
        )
        ranking_rows.append({'Model': name, **shared})
        preds = model.predict(X_te)
        is_degenerate = (
            shared['is_degenerate_classifier']
            or (best_c is not None and np.isclose(best_c, 1e-6) and np.unique(preds).size == 1)
        )
        return {
            'Model':                       name,
            'Best C':                      best_c_label if best_c_label is not None else (f'{best_c:.6f}' if best_c is not None else 'N/A'),
            'Avg CV Train Plain Acc':      shared['train_avg_accuracy'],
            'CV Train Plain Acc SD':       shared['train_std_accuracy'],
            'Avg CV Validation Plain Acc': shared['validation_avg_accuracy'],
            'CV Acc SD':                   shared['validation_std_accuracy'],
            'Test Acc':                    shared['test_split_accuracy'],
            'MCC':                         shared['test_matthew_corr_coef'],
            'Precision':                   shared['test_precision'],
            'Recall':                      shared['test_sensitivity'],
            'Specificity':                 shared['test_specificity'],
            'F1':                          shared['test_f1'],
            'ROC-AUC':                     shared['test_roc_auc_macro'],
            'Degenerate':                  is_degenerate,
        }

    # Raw models — same names as base.py so metrics checkpoints are shared
    rows = [
        _eval_row(f'Base+PCA ({n_components_raw}, {best_n_comp*100:.0f}%)',   baseline_clf,      X_train_pca,       y_train, X_test_pca,       y_test, tscv.n_splits),
        _eval_row('Raw Ridge',                                                  pipeline_ridge_1se, X_train,           y_train, X_test,           y_test, tscv.n_splits, best_c=ridge_c_1se),
        _eval_row('Raw LASSO',                                                  pipeline_lasso_1se, X_train,           y_train, X_test,           y_test, tscv.n_splits, best_c=lasso_c_1se),
        _eval_row(f'Raw Ridge+PCA ({n_components_ridge})',                     clf_ridge_pca_1se,  X_train_pca_ridge, y_train, X_test_pca_ridge,  y_test, tscv.n_splits, best_c=ridge_pca_c_1se),
        _eval_row(f'Raw LASSO+PCA ({n_components_lasso})',                     clf_lasso_pca_1se,  X_train_pca_lasso, y_train, X_test_pca_lasso,  y_test, tscv.n_splits, best_c=lasso_pca_c_1se),
    ]
    comparison_df = pd.DataFrame(rows).set_index('Model')

    # DOW models
    rows_dow = [
        _eval_row(f'Base+DOW+PCA ({n_components_dow}, {best_n_comp_dow*100:.0f}%)', baseline_dow_clf,       X_train_dow_pca,       y_train, X_test_dow_pca,       y_test, tscv.n_splits),
        _eval_row('Ridge+DOW',                                                        pipeline_ridge_dow_1se, X_train_dow,           y_train, X_test_dow,           y_test, tscv.n_splits, best_c=ridge_dow_c_1se),
        _eval_row('LASSO+DOW',                                                        pipeline_lasso_dow_1se, X_train_dow,           y_train, X_test_dow,           y_test, tscv.n_splits, best_c=lasso_dow_c_1se),
        _eval_row(f'Ridge+PCA+DOW ({n_components_ridge_dow})',                       clf_ridge_pca_dow_1se,  X_train_pca_ridge_dow, y_train, X_test_pca_ridge_dow, y_test, tscv.n_splits, best_c=ridge_pca_dow_c_1se),
        _eval_row(f'LASSO+PCA+DOW ({n_components_lasso_dow})',                       clf_lasso_pca_dow_1se,  X_train_pca_lasso_dow, y_train, X_test_pca_lasso_dow, y_test, tscv.n_splits, best_c=lasso_pca_dow_c_1se),
    ]
    dow_df = pd.DataFrame(rows_dow).set_index('Model')

    combined_df = pd.concat([comparison_df, dow_df])
    combined_df.index.name = 'Model'

    # Rank using TEST_SELECTION_CRITERIA (only this section changes when you reweight)
    ranked_df = rank_models_by_metrics(pd.DataFrame(ranking_rows), criteria=TEST_SELECTION_CRITERIA)
    rank_cols = ['Model'] + [f'rank_{k}' for k in TEST_SELECTION_CRITERIA] + ['average_rank']
    print("\n===== Ranked Models =====")
    print(ranked_df[[c for c in rank_cols if c in ranked_df.columns]].to_string(index=False))

    # ===================================================================
    # 4. CACHE PLOT CURVES FOR ALL 10 MODELS
    #    Stored in plot_cache_dir — survives base.py re-runs.
    #    On first run: computes all curves (slow).
    #    On subsequent runs: loads from cache (fast).
    # ===================================================================
    PCA_X_LABEL = 'PCA n_components from tuning grid\n← Lower retained variance, Simpler Model      Higher retained variance, More Complex →'

    all_plot_configs = {
        f'Base+PCA ({n_components_raw}, {best_n_comp*100:.0f}%)': {
            'plot_specs': [{
                'type': 'pca', 'suffix': 'pca_n_components',
                'selected_n_comp': best_n_comp,
                'selected_label': f'{_format_pca_grid_value(best_n_comp)} ({n_components_raw} comps)',
                'model_title': 'Baseline LR + PCA', 'feature_title': 'Raw OHLCV Features',
                'X_train_plot': X_train, 'X_test_plot': X_test,
                'c_value': np.inf, 'solver': 'lbfgs', 'l1_ratio': None,
                'grid': BASELINE_PCA_GRID, 'x_label': PCA_X_LABEL, 'direct_color': 'darkorange',
            }],
        },
        'Raw Ridge': {
            'plot_specs': [{
                'type': 'c', 'suffix': 'classifier_C',
                'logregcv': ridge_cv, 'one_se_c': ridge_c_1se,
                'model_title': 'Ridge (L2) - LR', 'feature_title': 'Raw OHLCV Features',
                'X_train_plot': X_train_raw_sc, 'X_test_plot': X_test_raw_sc,
                'l1_ratio': 0, 'direct_color': 'darkorange',
            }],
        },
        'Raw LASSO': {
            'plot_specs': [{
                'type': 'c', 'suffix': 'classifier_C',
                'logregcv': lasso_cv, 'one_se_c': lasso_c_1se,
                'model_title': 'LASSO (L1) - LR', 'feature_title': 'Raw OHLCV Features',
                'X_train_plot': X_train_raw_sc, 'X_test_plot': X_test_raw_sc,
                'l1_ratio': 1, 'direct_color': 'seagreen',
            }],
        },
        f'Raw Ridge+PCA ({n_components_ridge})': {
            'plot_specs': [
                {
                    'type': 'c', 'suffix': 'classifier_C',
                    'logregcv': clf_ridge_pca, 'one_se_c': ridge_pca_c_1se,
                    'model_title': 'Ridge (L2) - LR',
                    'feature_title': f'PCA Features ({n_components_ridge} comps, {best_n_comp_ridge*100:.0f}% variance)',
                    'X_train_plot': X_train_pca_ridge, 'X_test_plot': X_test_pca_ridge,
                    'l1_ratio': 0, 'direct_color': 'darkorange',
                },
                {
                    'type': 'pca', 'suffix': 'pca_n_components',
                    'selected_n_comp': best_n_comp_ridge,
                    'selected_label': f'{_format_pca_grid_value(best_n_comp_ridge)} ({n_components_ridge} comps)',
                    'model_title': 'Ridge (L2) - LR', 'feature_title': 'Raw OHLCV to PCA preprocessing',
                    'X_train_plot': X_train, 'X_test_plot': X_test,
                    'c_value': ridge_pca_c_1se, 'solver': 'saga', 'l1_ratio': 0,
                    'grid': BASELINE_PCA_GRID, 'x_label': PCA_X_LABEL, 'direct_color': 'darkorange',
                },
            ],
        },
        f'Raw LASSO+PCA ({n_components_lasso})': {
            'plot_specs': [
                {
                    'type': 'c', 'suffix': 'classifier_C',
                    'logregcv': clf_lasso_pca, 'one_se_c': lasso_pca_c_1se,
                    'model_title': 'LASSO (L1) - LR',
                    'feature_title': f'PCA Features ({n_components_lasso} comps, {best_n_comp_lasso*100:.0f}% variance)',
                    'X_train_plot': X_train_pca_lasso, 'X_test_plot': X_test_pca_lasso,
                    'l1_ratio': 1, 'direct_color': 'seagreen',
                },
                {
                    'type': 'pca', 'suffix': 'pca_n_components',
                    'selected_n_comp': best_n_comp_lasso,
                    'selected_label': f'{_format_pca_grid_value(best_n_comp_lasso)} ({n_components_lasso} comps)',
                    'model_title': 'LASSO (L1) - LR', 'feature_title': 'Raw OHLCV to PCA preprocessing',
                    'X_train_plot': X_train, 'X_test_plot': X_test,
                    'c_value': lasso_pca_c_1se, 'solver': 'saga', 'l1_ratio': 1,
                    'grid': BASELINE_PCA_GRID, 'x_label': PCA_X_LABEL, 'direct_color': 'seagreen',
                },
            ],
        },
        f'Base+DOW+PCA ({n_components_dow}, {best_n_comp_dow*100:.0f}%)': {
            'plot_specs': [{
                'type': 'pca', 'suffix': 'dow_pca_n_components',
                'selected_n_comp': best_n_comp_dow,
                'selected_label': f'{_format_pca_grid_value(best_n_comp_dow)} ({n_components_dow} comps)',
                'model_title': 'Baseline LR + PCA + DOW',
                'feature_title': 'Raw OHLCV + Day-of-Week Features',
                'X_train_plot': X_train_dow, 'X_test_plot': X_test_dow,
                'c_value': np.inf, 'solver': 'lbfgs', 'l1_ratio': None,
                'grid': BASELINE_PCA_GRID, 'x_label': PCA_X_LABEL, 'direct_color': 'darkorange',
            }],
        },
        'Ridge+DOW': {
            'plot_specs': [{
                'type': 'c', 'suffix': 'dow_classifier_C',
                'logregcv': ridge_dow_cv, 'one_se_c': ridge_dow_c_1se,
                'model_title': 'Ridge (L2) - LR + DOW',
                'feature_title': 'Raw OHLCV + Day-of-Week Features',
                'X_train_plot': X_train_dow_sc, 'X_test_plot': X_test_dow_sc,
                'l1_ratio': 0, 'direct_color': 'darkorange',
            }],
        },
        'LASSO+DOW': {
            'plot_specs': [{
                'type': 'c', 'suffix': 'dow_classifier_C',
                'logregcv': lasso_dow_cv, 'one_se_c': lasso_dow_c_1se,
                'model_title': 'LASSO (L1) - LR + DOW',
                'feature_title': 'Raw OHLCV + Day-of-Week Features',
                'X_train_plot': X_train_dow_sc, 'X_test_plot': X_test_dow_sc,
                'l1_ratio': 1, 'direct_color': 'seagreen',
            }],
        },
        f'Ridge+PCA+DOW ({n_components_ridge_dow})': {
            'plot_specs': [
                {
                    'type': 'c', 'suffix': 'dow_classifier_C',
                    'logregcv': clf_ridge_pca_dow, 'one_se_c': ridge_pca_dow_c_1se,
                    'model_title': 'Ridge (L2) - LR + DOW',
                    'feature_title': f'PCA + DOW Features ({n_components_ridge_dow} comps, {best_n_comp_ridge_dow*100:.0f}% variance)',
                    'X_train_plot': X_train_pca_ridge_dow, 'X_test_plot': X_test_pca_ridge_dow,
                    'l1_ratio': 0, 'direct_color': 'darkorange',
                },
                {
                    'type': 'pca', 'suffix': 'dow_pca_n_components',
                    'selected_n_comp': best_n_comp_ridge_dow,
                    'selected_label': f'{_format_pca_grid_value(best_n_comp_ridge_dow)} ({n_components_ridge_dow} comps)',
                    'model_title': 'Ridge (L2) - LR + DOW',
                    'feature_title': 'Raw OHLCV + Day-of-Week to PCA preprocessing',
                    'X_train_plot': X_train_dow, 'X_test_plot': X_test_dow,
                    'c_value': ridge_pca_dow_c_1se, 'solver': 'saga', 'l1_ratio': 0,
                    'grid': BASELINE_PCA_GRID, 'x_label': PCA_X_LABEL, 'direct_color': 'darkorange',
                },
            ],
        },
        f'LASSO+PCA+DOW ({n_components_lasso_dow})': {
            'plot_specs': [
                {
                    'type': 'c', 'suffix': 'dow_classifier_C',
                    'logregcv': clf_lasso_pca_dow, 'one_se_c': lasso_pca_dow_c_1se,
                    'model_title': 'LASSO (L1) - LR + DOW',
                    'feature_title': f'PCA + DOW Features ({n_components_lasso_dow} comps, {best_n_comp_lasso_dow*100:.0f}% variance)',
                    'X_train_plot': X_train_pca_lasso_dow, 'X_test_plot': X_test_pca_lasso_dow,
                    'l1_ratio': 1, 'direct_color': 'seagreen',
                },
                {
                    'type': 'pca', 'suffix': 'dow_pca_n_components',
                    'selected_n_comp': best_n_comp_lasso_dow,
                    'selected_label': f'{_format_pca_grid_value(best_n_comp_lasso_dow)} ({n_components_lasso_dow} comps)',
                    'model_title': 'LASSO (L1) - LR + DOW',
                    'feature_title': 'Raw OHLCV + Day-of-Week to PCA preprocessing',
                    'X_train_plot': X_train_dow, 'X_test_plot': X_test_dow,
                    'c_value': lasso_pca_dow_c_1se, 'solver': 'saga', 'l1_ratio': 1,
                    'grid': BASELINE_PCA_GRID, 'x_label': PCA_X_LABEL, 'direct_color': 'seagreen',
                },
            ],
        },
    }

    # Cache plot curves for every model
    print("\n===== Caching Plot Curves for All Models =====")
    all_curves = {}
    for model_name, cfg in all_plot_configs.items():
        # Derive a stable cache key from the model name slug
        model_key = _metrics_stage_name(model_name)
        print(f"\n  Model: {model_name}")
        spec_curves = []
        for spec in cfg['plot_specs']:
            cache_stage = f"{model_key}__{spec['suffix']}"
            if spec['type'] == 'c':
                def _make_c_compute(spec=spec):
                    bv = _compute_bv_curves(
                        spec['logregcv'],
                        spec['X_train_plot'],
                        y_train,
                        tscv,
                        spec['l1_ratio'],
                        'saga',
                    )
                    c_grid = _augment_c_grid_with_selected_values(
                        spec['logregcv'].Cs_,
                        spec['one_se_c'],
                    )
                    direct = _compute_direct_split_errors(
                        spec['X_train_plot'],
                        y_train,
                        spec['X_test_plot'],
                        y_test,
                        c_grid,
                        spec['l1_ratio'],
                        'saga',
                    )
                    return {'bv': bv, 'direct': direct}
                spec_curves.append(_load_or_compute_plot_curves(plot_cache_dir, cache_stage, _make_c_compute))
            else:  # 'pca'
                def _make_pca_compute(spec=spec):
                    return _compute_pca_curve_diagnostics(
                        spec['X_train_plot'],
                        y_train,
                        spec['X_test_plot'],
                        y_test,
                        spec['grid'],
                        tscv,
                        c_value=spec['c_value'],
                        solver=spec['solver'],
                        l1_ratio=spec['l1_ratio'],
                        selected_n_comp=spec.get('selected_n_comp'),
                        selected_label=spec.get('selected_label'),
                    )
                spec_curves.append(_load_or_compute_plot_curves(plot_cache_dir, cache_stage, _make_pca_compute))
        all_curves[model_name] = spec_curves

    # ===================================================================
    # 5. SELECT BEST MODEL AND GENERATE PLOTS FROM CACHED CURVES
    # ===================================================================
    best_model_name = str(ranked_df.iloc[0]['Model'])
    plot_model_name = select_non_degenerate_plot_model(ranked_df, available_models=all_plot_configs)

    print(f"\nBest model by average rank: {best_model_name}")
    if plot_model_name != best_model_name:
        print(f"Plotting fallback (non-degenerate): {plot_model_name}")
    print(f"Plotting diagnostics for: {plot_model_name}")

    base_bv_path     = output_dir / f'{output_prefix}_1SE_base_logistic_best_bias_variance.png'
    base_direct_path = output_dir / f'{output_prefix}_1SE_base_logistic_best_train_test.png'

    best_cfg    = all_plot_configs[plot_model_name]
    best_curves = all_curves[plot_model_name]

    for idx, (spec, curves) in enumerate(zip(best_cfg['plot_specs'], best_curves)):
        output_bv_path     = base_bv_path     if idx == 0 else _append_plot_suffix(base_bv_path,     spec['suffix'])
        output_direct_path = base_direct_path if idx == 0 else _append_plot_suffix(base_direct_path, spec['suffix'])

        if spec['type'] == 'c':
            _plot_single_model_diagnostics(
                diagnostics={'bv': curves['bv'], 'direct': curves['direct']},
                bv_key='bv',
                direct_key='direct',
                one_se_c=spec['one_se_c'],
                model_title=spec['model_title'],
                feature_title=spec['feature_title'],
                X_train_plot=spec['X_train_plot'],
                y_train=y_train,
                X_test_plot=spec['X_test_plot'],
                y_test=y_test,
                l1_ratio=spec['l1_ratio'],
                output_bv=str(output_bv_path),
                output_direct=str(output_direct_path),
                direct_color=spec.get('direct_color', 'darkorange'),
            )
        else:  # 'pca'
            _save_curve_pair(
                curves['curve'],
                curves['direct'],
                curves['selected_idx'],
                curves['selected_label'],
                spec['model_title'],
                spec['feature_title'],
                spec['x_label'],
                str(output_bv_path),
                str(output_direct_path),
                x_scale='linear',
                direct_color=spec.get('direct_color', 'darkorange'),
            )

    # ===================================================================
    # 6. LATEX TABLE
    # ===================================================================
    print("\n===== Generating LaTeX Table =====")
    full_df = build_compact_export_table(combined_df)
    full_df.index.name = 'Model'
    print(full_df.to_string())

    tex_path = output_dir / f'{output_prefix}_1SE_base_logistic_comparison.tex'
    baseline_note = (
        r'Base = baseline logistic regression without regularization. '
        r'Test Acc = plain hold-out accuracy on the final 20\% test split. '
        r'All reported CV/train/test accuracy columns in this table use plain accuracy after hyperparameters were selected by CV balanced accuracy. '
        r'Recall = positive-class sensitivity, TP / (TP + FN). Specificity = TN / (TN + FP).'
    )
    lasso_note = (
        r'$^\dagger$ Degenerate classifier: optimal $C = 10^{-6}$ shrinks all '
        r'coefficients to zero; model predicts majority class for every observation '
        r'(Recall $\approx 0$ or 1 depending on majority class; Precision $\approx$ base rate).'
    )
    degenerate_models = set(combined_df.index[combined_df['Degenerate']])
    write_grouped_latex_table(
        full_df,
        str(tex_path),
        'Logistic Regression Model Comparison: Raw OHLCV vs Raw OHLCV + Day-of-Week',
        'tab:base_logistic_comparison',
        baseline_note,
        groups=[comparison_df.index, dow_df.index],
        degenerate_models=degenerate_models,
        degenerate_note=lasso_note,
        escape_note=False,
        escape_degenerate_note=False,
    )
    print(f"LaTeX table saved to: {tex_path}")

    # ===================================================================
    # 7. GLOBAL LEADERBOARD + RUN LOG
    # ===================================================================
    global_ranked_df = register_global_model_candidates(
        ranked_df,
        output_dir / f"{output_prefix}_global_model_leaderboard.csv",
        source_script="base_table_fig.py",
        dataset_label=output_prefix,
        comparison_scope="raw_vs_dow",
        reset_leaderboard=True,
    )
    append_search_run(
        runs_path=output_dir / f"{output_prefix}_search_runs.csv",
        model_name="base_logreg_table_fig",
        run_time=run_time,
        run_duration_sec=(time.time() - run_start),
        grid_version=GRID_VERSION,
        n_jobs=MODEL_N_JOBS,
        dataset_version="testing=False,extra_features=False,cluster=False,corr_threshold=0.95,corr_level=0",
        code_commit=get_git_commit(cwd),
        notes="base_table_fig.py run",
    )

    print(f"\nBest model (ranked):  {best_model_name}")
    print(f"Plot model:           {plot_model_name}")
    print(f"Global leaderboard:   {global_ranked_df.iloc[0]['Model']}")
    print(f"\nTotal time: {time.time() - run_start:.1f}s")
