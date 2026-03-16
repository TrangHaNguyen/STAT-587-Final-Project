"""
Baseline Logistic Regression Models with Regularization Comparison

This script trains and evaluates baseline logistic regression models:
  - Baseline (No Regularization) with optimal PCA components (grid search)
  - Ridge (L2) regularization with cross-validation
  - LASSO (L1) regularization with cross-validation

Tested on:
  - Raw OHLCV features
  - Raw OHLCV + Day-of-Week features

Best-model diagnostics are computed only for the final selected plot winner.

SCRIPT STRUCTURE:
  1. Model Training (lines 50-600): Run models and prepare diagnostics
  2. Comparison Tables & LaTeX Export (lines 600-680)
  3. Helper Functions (lines 680+): Optional helpers for manual cache inspection

USAGE:
  - The script trains/evaluates all candidate models.
  - Bias-variance and train/test diagnostics are computed only for the final
    selected best model.
"""

import os
import shutil
import time
import warnings
from pathlib import Path
import pandas as pd
from H_prep import clean_data, import_data, to_binary_class
import numpy as np

MPLCONFIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.mplconfig')
os.makedirs(MPLCONFIGDIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(MPLCONFIGDIR))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
from model_grids import (
    BASELINE_PCA_GRID,
    LOGISTIC_BASELINE_SOLVER,
    LOGISTIC_CLASS_WEIGHT,
    LOGISTIC_LASSO_SOLVER,
    LOGISTIC_MAX_ITER,
    LOGISTIC_RIDGE_SOLVER,
    LOGISTIC_TOL,
    LASSO_GRID,
    RANDOM_SEED,
    RIDGE_GRID,
    TEST_SIZE,
    TIME_SERIES_CV_SPLITS,
    TRAIN_TEST_SHUFFLE,
)
from H_eval import (
    CV_SELECTION_CRITERIA,
    get_final_metrics,
    get_or_compute_final_metrics,
    _metrics_stage_name,
    rank_models_by_metrics,
    register_global_model_candidates,
    select_non_degenerate_plot_model,
    build_compact_export_table,
    write_grouped_latex_table,
)
from H_search_history import (
    append_search_run,
    get_git_commit,
    get_checkpoint_dir,
    load_stage_checkpoint,
    now_iso,
    save_stage_checkpoint,
    stage_checkpoint_exists,
)

MODEL_N_JOBS = int(os.getenv("MODEL_N_JOBS", "-1"))
GRID_VERSION = os.getenv("GRID_VERSION", "v1")
SEARCH_NOTES = os.getenv("SEARCH_NOTES", "")

def clear_base_caches():
    # No active .pkl caches are needed for the current best-model-only
    # plotting workflow.
    print("No active base caches to clear.")


def clear_output_checkpoints() -> None:
    checkpoints_dir = Path(__file__).resolve().parents[2] / "output" / "checkpoints"
    if checkpoints_dir.exists():
        shutil.rmtree(checkpoints_dir)
        print(f"Deleted checkpoint directory: {checkpoints_dir}")
    else:
        print(f"No checkpoint directory to delete: {checkpoints_dir}")


def _build_logistic_kwargs(*, solver: str, l1_ratio=None, c_value=None):
    kwargs = {
        'solver': solver,
        'class_weight': LOGISTIC_CLASS_WEIGHT,
        'random_state': 1,
        'max_iter': LOGISTIC_MAX_ITER,
        'tol': LOGISTIC_TOL,
    }
    if c_value is not None:
        kwargs['C'] = c_value
    if l1_ratio is not None:
        kwargs['l1_ratio'] = l1_ratio
    return kwargs


def _build_logistic_cv_kwargs(*, solver: str, cs, cv, l1_ratio, scoring: str):
    kwargs = {
        'Cs': cs,
        'cv': cv,
        'solver': solver,
        'class_weight': LOGISTIC_CLASS_WEIGHT,
        'random_state': 1,
        'n_jobs': MODEL_N_JOBS,
        'max_iter': LOGISTIC_MAX_ITER,
        'tol': LOGISTIC_TOL,
        'scoring': scoring,
        'use_legacy_attributes': False,
    }
    if l1_ratio is not None:
        kwargs['l1_ratios'] = [l1_ratio]
    return kwargs


def _suppress_expected_no_penalty_warning():
    warnings.filterwarnings(
        "ignore",
        message=r"Setting penalty=None will ignore the C and l1_ratio parameters",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Inconsistent values: penalty=l1 with l1_ratio=0\.0\. penalty is deprecated\. Please use l1_ratio only",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*'penalty' was deprecated.*use 'l1_ratios' instead.*",
        category=FutureWarning,
    )


def _fit_logistic_model(clf, X, y):
    with warnings.catch_warnings():
        _suppress_expected_no_penalty_warning()
        clf.fit(X, y)
    return clf


def _keep_raw_stock_ohlcv(X: pd.DataFrame) -> pd.DataFrame:
    idx = pd.IndexSlice
    metrics = ['Open', 'Close', 'High', 'Low', 'Volume']
    return X.loc[:, idx[metrics, 'Stocks', :]].copy()


def _load_or_compute_stage(checkpoint_dir, stage_name: str, compute_fn, heading: str):
    if stage_checkpoint_exists(checkpoint_dir, stage_name):
        print(f"\n========== {heading} ==========")
        print(f"Loading checkpoint from {checkpoint_dir / stage_name}")
        return load_stage_checkpoint(checkpoint_dir, stage_name)
    payload = compute_fn()
    save_stage_checkpoint(checkpoint_dir, stage_name, payload)
    return payload

def _select_pca_n_components_1se(grid_results):
    """Choose the smallest PCA n_components within one SE of the best mean CV score."""
    mean_scores = np.array([row['mean_cv_score'] for row in grid_results], dtype=float)
    std_scores = np.array([row['std_cv_score'] for row in grid_results], dtype=float)
    split_counts = np.array([len(row['cv_scores']) for row in grid_results], dtype=float)
    se_scores = std_scores / np.sqrt(split_counts)
    best_idx = int(np.argmax(mean_scores))
    threshold = float(mean_scores[best_idx] - se_scores[best_idx])
    candidate_idx = np.where(mean_scores >= threshold)[0]
    chosen_idx = int(min(candidate_idx, key=lambda i: (float(grid_results[i]['n_components']), -float(mean_scores[i]))))
    return grid_results[chosen_idx], grid_results[best_idx], threshold

def _select_c_1se_from_logregcv(cv_clf):
    raw_scores = cv_clf.scores_
    if isinstance(raw_scores, dict):
        scores = np.array(list(raw_scores.values())[0])
    else:
        scores = np.array(raw_scores)
    if scores.ndim == 3:
        if scores.shape[1] == len(cv_clf.Cs_):
            scores = scores[:, :, 0]
        elif scores.shape[2] == len(cv_clf.Cs_):
            scores = scores[:, 0, :]
        else:
            scores = np.squeeze(scores)
    mean_scores = scores.mean(axis=0)
    std_scores = scores.std(axis=0)
    se_scores = std_scores / np.sqrt(scores.shape[0])
    cs = np.array(cv_clf.Cs_)
    best_idx = int(np.argmax(mean_scores))
    # Classic 1SE rule on accuracy: choose the smallest C whose mean CV score
    # is within one standard error of the best-performing C.
    threshold = mean_scores[best_idx] - se_scores[best_idx]
    candidate_idx = np.where(mean_scores >= threshold)[0]
    # Simpler model for logistic regularization is smaller C.
    chosen_idx = int(candidate_idx[np.argmin(cs[candidate_idx])])
    return float(cs[chosen_idx]), float(cs[best_idx]), float(threshold)


def _format_pca_grid_value(n_comp):
    """Format PCA n_components from the tuning grid for display."""
    if isinstance(n_comp, (float, np.floating)):
        return f"{float(n_comp):.2f}"
    return str(n_comp)

def _compute_bv_curves(cv_clf, X_tr, y_tr, tscv_splitter, l1_ratio, solver):
    """Compute train/CV plain and balanced error curves for each C."""
    cs = np.array(cv_clf.Cs_)
    n_splits = tscv_splitter.get_n_splits()
    train_plain_errors = np.zeros((n_splits, len(cs)))
    cv_plain_errors = np.zeros((n_splits, len(cs)))
    train_bal_errors = np.zeros((n_splits, len(cs)))
    cv_bal_errors = np.zeros((n_splits, len(cs)))
    for fold_idx, (tr, val) in enumerate(tscv_splitter.split(X_tr, y_tr)):
        X_fold = X_tr.iloc[tr] if hasattr(X_tr, 'iloc') else X_tr[tr]
        y_fold = y_tr.iloc[tr] if hasattr(y_tr, 'iloc') else y_tr[tr]
        X_val = X_tr.iloc[val] if hasattr(X_tr, 'iloc') else X_tr[val]
        y_val = y_tr.iloc[val] if hasattr(y_tr, 'iloc') else y_tr[val]
        # Recreate each model per fold/C so all four metrics come from the same fit.
        for c_idx, c_val in enumerate(cs):
            clf = LogisticRegression(**_build_logistic_kwargs(
                solver=solver,
                l1_ratio=l1_ratio,
                c_value=c_val,
            ))
            _fit_logistic_model(clf, X_fold, y_fold)
            train_preds = clf.predict(X_fold)
            val_preds = clf.predict(X_val)
            train_plain_errors[fold_idx, c_idx] = 1 - accuracy_score(y_fold, train_preds)
            cv_plain_errors[fold_idx, c_idx] = 1 - accuracy_score(y_val, val_preds)
            train_bal_errors[fold_idx, c_idx] = 1 - balanced_accuracy_score(y_fold, train_preds)
            cv_bal_errors[fold_idx, c_idx] = 1 - balanced_accuracy_score(y_val, val_preds)

    return {
        'cs': cs,
        'train_plain_err_mean': train_plain_errors.mean(axis=0),
        'train_plain_err_std': train_plain_errors.std(axis=0),
        'cv_plain_err_mean': cv_plain_errors.mean(axis=0),
        'cv_plain_err_std': cv_plain_errors.std(axis=0),
        'train_bal_err_mean': train_bal_errors.mean(axis=0),
        'train_bal_err_std': train_bal_errors.std(axis=0),
        'cv_bal_err_mean': cv_bal_errors.mean(axis=0),
        'cv_bal_err_std': cv_bal_errors.std(axis=0),
        'cv_bal_err_se': cv_bal_errors.std(axis=0) / np.sqrt(n_splits),
    }

def _compute_direct_split_errors(X_train, y_train, X_test, y_test, c_grid, l1_ratio, solver):
    train_errors, test_errors = [], []
    for c_val in c_grid:
        clf = LogisticRegression(**_build_logistic_kwargs(
            solver=solver,
            l1_ratio=l1_ratio,
            c_value=c_val,
        ))
        _fit_logistic_model(clf, X_train, y_train)
        train_errors.append(1 - clf.score(X_train, y_train))
        test_errors.append(1 - clf.score(X_test, y_test))
    return {
        'cs': np.array(c_grid),
        'train_errors': np.array(train_errors),
        'test_errors': np.array(test_errors),
    }

def _augment_c_grid_with_selected_values(c_grid, *selected_values):
    combined = np.asarray(list(c_grid) + [float(v) for v in selected_values], dtype=float)
    return np.unique(combined)

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

def _compute_single_direct_split_error(X_train, y_train, X_test, y_test, c_val, l1_ratio, solver):
    clf = LogisticRegression(**_build_logistic_kwargs(
        solver=solver,
        l1_ratio=l1_ratio,
        c_value=c_val,
    ))
    _fit_logistic_model(clf, X_train, y_train)
    return 1 - clf.score(X_train, y_train), 1 - clf.score(X_test, y_test)


def _as_sortable_numeric(value):
    try:
        return float(value)
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
                val = cv_results[f"param_{col}"][i]
                complexity.append(_as_sortable_numeric(val))
            return tuple(complexity + [-float(mean[i])])

        return int(min(candidate_idx, key=key_fn))

    return _pick_index


def _plot_single_model_diagnostics(
    diagnostics,
    bv_key,
    direct_key,
    one_se_c,
    model_title,
    feature_title,
    X_train_plot,
    y_train,
    X_test_plot,
    y_test,
    l1_ratio,
    output_bv,
    output_direct,
    direct_color='darkorange',
):
    diag = diagnostics[bv_key]
    direct_diag = diagnostics[direct_key]
    cs = diag['cs']
    best_idx = int(np.argmin(diag['cv_bal_err_mean']))
    selected_c = float(cs[best_idx])

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        f'Bias-Variance Tradeoff - {model_title}\n{feature_title}',
        fontsize=13, fontweight='bold'
    )
    ax.semilogx(cs, diag['train_bal_err_mean'], marker='o', color='steelblue', linewidth=1.8, label='CV Train balanced error')
    ax.semilogx(cs, diag['cv_bal_err_mean'], marker='s', color='darkorange', linewidth=1.8, label='CV Test balanced error')
    ax.fill_between(
        cs,
        np.clip(diag['train_bal_err_mean'] - diag['train_bal_err_std'], 0.0, 1.0),
        np.clip(diag['train_bal_err_mean'] + diag['train_bal_err_std'], 0.0, 1.0),
        alpha=0.15,
        color='steelblue',
        label='CV Train balanced error ±1 SD'
    )
    ax.fill_between(
        cs,
        np.clip(diag['cv_bal_err_mean'] - diag['cv_bal_err_std'], 0.0, 1.0),
        np.clip(diag['cv_bal_err_mean'] + diag['cv_bal_err_std'], 0.0, 1.0),
        alpha=0.15,
        color='darkorange',
        label='CV Test balanced error ±1 SD'
    )
    _highlight_selected_value(ax, cs, diag['cv_bal_err_mean'], best_idx, label_prefix='Value at best CV balanced error')
    ax.axvline(one_se_c, color='red', linestyle='--', linewidth=1.5, label=f'1SE-selected C = {one_se_c:.4f}')
    ax.set_title(f'{model_title} - Bias-Variance Tradeoff (Balanced Error)')
    ax.set_xlabel('C  (Inverse Regularization Strength)\n← High Regularization, Simpler Model      Low Regularization, More Complex →')
    ax.set_ylabel('Balanced Error (1 - balanced accuracy)')
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_bv, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {os.path.abspath(output_bv)}")
    plt.close()

    _, best_test_error = _compute_single_direct_split_error(
        X_train_plot, y_train, X_test_plot, y_test, selected_c, l1_ratio, 'saga'
    )
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    fig2.suptitle(
        f'Over/Underfitting Analysis - {model_title}\n{feature_title}',
        fontsize=13, fontweight='bold'
    )
    ax2.semilogx(direct_diag['cs'], direct_diag['train_errors'], marker='o', color='steelblue', linewidth=2, label='Train error')
    ax2.semilogx(direct_diag['cs'], direct_diag['test_errors'], marker='s', color=direct_color, linewidth=2, label='Test error')
    ax2.scatter([selected_c], [best_test_error], color='gold', edgecolor='black', s=90, zorder=6, label='Value at best CV balanced error')
    ax2.axvline(one_se_c, color='red', linestyle='--', linewidth=1.5, label=f'1SE-selected C = {one_se_c:.4f}')
    ax2.set_title(f'{model_title} - Train vs Test Error (Plain Error)')
    ax2.set_xlabel('C  (Inverse Regularization Strength)\n← High Regularization, Simpler Model      Low Regularization, More Complex →')
    ax2.set_ylabel('Plain Error (1 - accuracy)')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_direct, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {os.path.abspath(output_direct)}")
    plt.close()


def _append_plot_suffix(path_like, suffix: str):
    path = Path(path_like)
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def _save_curve_pair(
    curve,
    direct_curve,
    selected_idx,
    selected_label,
    model_title,
    feature_title,
    x_label,
    output_bv,
    output_direct,
    *,
    x_scale: str = 'linear',
    direct_color: str = 'darkorange',
):
    best_idx = int(np.argmin(curve['cv_bal_err_mean']))
    best_point_label = curve.get(
        'best_point_label',
        'Value at best CV balanced error'
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        f'Bias-Variance Tradeoff - {model_title}\n{feature_title}',
        fontsize=13, fontweight='bold'
    )
    if x_scale == 'log':
        ax.semilogx(curve['x_numeric'], curve['train_bal_err_mean'], marker='o', color='steelblue', linewidth=1.8, label='CV Train balanced error')
        ax.semilogx(curve['x_numeric'], curve['cv_bal_err_mean'], marker='s', color='darkorange', linewidth=1.8, label='CV Test balanced error')
    else:
        ax.plot(curve['x_numeric'], curve['train_bal_err_mean'], marker='o', color='steelblue', linewidth=1.8, label='CV Train balanced error')
        ax.plot(curve['x_numeric'], curve['cv_bal_err_mean'], marker='s', color='darkorange', linewidth=1.8, label='CV Test balanced error')
    ax.fill_between(
        curve['x_numeric'],
        np.clip(curve['train_bal_err_mean'] - curve['train_bal_err_std'], 0.0, 1.0),
        np.clip(curve['train_bal_err_mean'] + curve['train_bal_err_std'], 0.0, 1.0),
        alpha=0.15,
        color='steelblue',
        label='CV Train balanced error ±1 SD'
    )
    ax.fill_between(
        curve['x_numeric'],
        np.clip(curve['cv_bal_err_mean'] - curve['cv_bal_err_std'], 0.0, 1.0),
        np.clip(curve['cv_bal_err_mean'] + curve['cv_bal_err_std'], 0.0, 1.0),
        alpha=0.15,
        color='darkorange',
        label='CV Test balanced error ±1 SD'
    )
    _highlight_selected_value(
        ax,
        curve['x_numeric'],
        curve['cv_bal_err_mean'],
        best_idx,
        label_prefix=best_point_label,
    )
    ax.axvline(curve['x_numeric'][selected_idx], color='red', linestyle='--', linewidth=1.5, label=f'1SE-selected value = {selected_label}')
    ax.set_title(f'{model_title} - Bias-Variance Tradeoff (Balanced Error)')
    ax.set_xlabel(x_label)
    ax.set_ylabel('Balanced Error (1 - balanced accuracy)')
    ax.set_ylim(0, 1.02)
    ax.set_xticks(curve['x_numeric'])
    ax.set_xticklabels(curve['x_labels'], rotation=45, ha='right')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_bv, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {os.path.abspath(output_bv)}")
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    fig2.suptitle(
        f'Over/Underfitting Analysis - {model_title}\n{feature_title}',
        fontsize=13, fontweight='bold'
    )
    if x_scale == 'log':
        ax2.semilogx(direct_curve['x_numeric'], direct_curve['train_errors'], marker='o', color='steelblue', linewidth=2, label='Train error')
        ax2.semilogx(direct_curve['x_numeric'], direct_curve['test_errors'], marker='s', color=direct_color, linewidth=2, label='Test error')
    else:
        ax2.plot(direct_curve['x_numeric'], direct_curve['train_errors'], marker='o', color='steelblue', linewidth=2, label='Train error')
        ax2.plot(direct_curve['x_numeric'], direct_curve['test_errors'], marker='s', color=direct_color, linewidth=2, label='Test error')
    ax2.scatter(
        [direct_curve['x_numeric'][best_idx]], [direct_curve['test_errors'][best_idx]],
        color='gold', edgecolor='black', s=90, zorder=6, label=best_point_label
    )
    ax2.axvline(direct_curve['x_numeric'][selected_idx], color='red', linestyle='--', linewidth=1.5, label=f'1SE-selected value = {selected_label}')
    ax2.set_title(f'{model_title} - Train vs Test Error (Plain Error)')
    ax2.set_xlabel(x_label)
    ax2.set_ylabel('Plain Error (1 - accuracy)')
    ax2.set_xticks(direct_curve['x_numeric'])
    ax2.set_xticklabels(direct_curve['x_labels'], rotation=45, ha='right')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_direct, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {os.path.abspath(output_direct)}")
    plt.close()


def _compute_pca_curve_diagnostics(
    X_train,
    y_train,
    X_test,
    y_test,
    n_components_grid,
    tscv_splitter,
    *,
    c_value,
    solver,
    l1_ratio=None,
    selected_n_comp=None,
    selected_label=None,
):
    curve_rows = []

    for n_comp in n_components_grid:
        fold_train_bal = []
        fold_cv_bal = []
        fold_train_plain = []
        fold_cv_plain = []

        for tr, val in tscv_splitter.split(X_train, y_train):
            X_fold = X_train.iloc[tr] if hasattr(X_train, 'iloc') else X_train[tr]
            y_fold = y_train.iloc[tr] if hasattr(y_train, 'iloc') else y_train[tr]
            X_val = X_train.iloc[val] if hasattr(X_train, 'iloc') else X_train[val]
            y_val = y_train.iloc[val] if hasattr(y_train, 'iloc') else y_train[val]

            scaler = StandardScaler()
            X_fold_sc = scaler.fit_transform(X_fold)
            X_val_sc = scaler.transform(X_val)

            pca = PCA(n_components=n_comp)
            X_fold_pca = pca.fit_transform(X_fold_sc)
            X_val_pca = pca.transform(X_val_sc)

            clf = LogisticRegression(
                **_build_logistic_kwargs(
                    solver=solver,
                    l1_ratio=l1_ratio,
                    c_value=c_value,
                ),
            )
            _fit_logistic_model(clf, X_fold_pca, y_fold)
            fold_train_pred = clf.predict(X_fold_pca)
            fold_val_pred = clf.predict(X_val_pca)
            fold_train_plain.append(1 - accuracy_score(y_fold, fold_train_pred))
            fold_cv_plain.append(1 - accuracy_score(y_val, fold_val_pred))
            fold_train_bal.append(1 - balanced_accuracy_score(y_fold, fold_train_pred))
            fold_cv_bal.append(1 - balanced_accuracy_score(y_val, fold_val_pred))

        scaler_full = StandardScaler()
        X_train_sc = scaler_full.fit_transform(X_train)
        X_test_sc = scaler_full.transform(X_test)
        pca_full = PCA(n_components=n_comp)
        X_train_pca = pca_full.fit_transform(X_train_sc)
        X_test_pca = pca_full.transform(X_test_sc)
        clf_full = LogisticRegression(
            **_build_logistic_kwargs(
                solver=solver,
                l1_ratio=l1_ratio,
                c_value=c_value,
            ),
        )
        _fit_logistic_model(clf_full, X_train_pca, y_train)
        component_count = X_train_pca.shape[1]
        n_comp_label = _format_pca_grid_value(n_comp)
        curve_rows.append({
            'n_components': n_comp,
            'component_count': component_count,
            'x_label': n_comp_label,
            'point_label': f'{n_comp_label} ({component_count} comps)',
            'train_bal_mean': float(np.mean(fold_train_bal)),
            'train_bal_std': float(np.std(fold_train_bal)),
            'cv_bal_mean': float(np.mean(fold_cv_bal)),
            'cv_bal_std': float(np.std(fold_cv_bal)),
            'train_error': float(1 - clf_full.score(X_train_pca, y_train)),
            'test_error': float(1 - clf_full.score(X_test_pca, y_test)),
        })

    curve_rows = sorted(curve_rows, key=lambda row: float(row['n_components']))
    cv_bal_mean = np.asarray([row['cv_bal_mean'] for row in curve_rows], dtype=float)
    cv_bal_std = np.asarray([row['cv_bal_std'] for row in curve_rows], dtype=float)
    n_splits = float(tscv_splitter.get_n_splits())
    cv_bal_se = cv_bal_std / np.sqrt(n_splits)
    best_idx = int(np.argmin(cv_bal_mean))
    threshold = float(cv_bal_mean[best_idx] + cv_bal_se[best_idx])
    candidate_idx = np.where(cv_bal_mean <= threshold)[0]
    selected_idx = int(min(candidate_idx, key=lambda i: (float(curve_rows[i]['n_components']), float(cv_bal_mean[i]))))
    if selected_n_comp is not None:
        selected_idx = int(min(
            range(len(curve_rows)),
            key=lambda i: abs(float(curve_rows[i]['n_components']) - float(selected_n_comp)),
        ))
    x_positions = np.arange(len(curve_rows), dtype=float)
    return {
        'curve': {
            'x_numeric': x_positions,
            'x_labels': [row['x_label'] for row in curve_rows],
            'cv_bal_err_mean': cv_bal_mean,
            'cv_bal_err_std': cv_bal_std,
            'train_bal_err_mean': np.asarray([row['train_bal_mean'] for row in curve_rows], dtype=float),
            'train_bal_err_std': np.asarray([row['train_bal_std'] for row in curve_rows], dtype=float),
            'best_point_label': f"Value at best CV balanced error = {curve_rows[best_idx]['point_label']}",
        },
        'direct': {
            'x_numeric': x_positions,
            'x_labels': [row['x_label'] for row in curve_rows],
            'train_errors': np.asarray([row['train_error'] for row in curve_rows], dtype=float),
            'test_errors': np.asarray([row['test_error'] for row in curve_rows], dtype=float),
        },
        'selected_idx': selected_idx,
        'selected_label': selected_label if selected_label is not None else curve_rows[selected_idx]['point_label'],
    }


def _retune_pca_with_fixed_logistic(
    X_train,
    y_train,
    X_test,
    n_components_grid,
    tscv_splitter,
    *,
    c_value,
    solver,
    l1_ratio=None,
):
    grid_search_results = []
    print(f"Testing n_components: {n_components_grid}")
    for n_comp in n_components_grid:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_comp)),
            ('classifier', LogisticRegression(**_build_logistic_kwargs(
                solver=solver,
                l1_ratio=l1_ratio,
                c_value=c_value,
            ))),
        ])
        with warnings.catch_warnings():
            _suppress_expected_no_penalty_warning()
            scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=tscv_splitter,
                n_jobs=MODEL_N_JOBS,
                scoring='balanced_accuracy',
            )

        scaler_full = StandardScaler()
        X_train_sc = scaler_full.fit_transform(X_train)
        pca_full = PCA(n_components=n_comp)
        X_train_pca = pca_full.fit_transform(X_train_sc)
        grid_search_results.append({
            'n_components': n_comp,
            'n_components_value': X_train_pca.shape[1],
            'cv_scores': scores,
            'mean_cv_score': float(scores.mean()),
            'std_cv_score': float(scores.std()),
        })
        print(
            f"  n_components={n_comp} ({X_train_pca.shape[1]} components): "
            f"CV Balanced Accuracy = {scores.mean():.4f} ± {scores.std():.4f}"
        )

    selected_pca, best_pca, pca_threshold = _select_pca_n_components_1se(grid_search_results)
    best_n_comp = selected_pca['n_components']

    scaler_pca = StandardScaler()
    X_train_sc = scaler_pca.fit_transform(X_train)
    X_test_sc = scaler_pca.transform(X_test)
    pca = PCA(n_components=best_n_comp)
    X_train_pca = pca.fit_transform(X_train_sc)
    X_test_pca = pca.transform(X_test_sc)

    return {
        'grid_search_results': grid_search_results,
        'selected_pca': selected_pca,
        'best_pca': best_pca,
        'pca_threshold': pca_threshold,
        'best_n_comp': best_n_comp,
        'best_n_comp_value': selected_pca['n_components_value'],
        'best_score': selected_pca['mean_cv_score'],
        'scaler_pca': scaler_pca,
        'X_train_sc': X_train_sc,
        'X_test_sc': X_test_sc,
        'pca': pca,
        'X_train_pca': X_train_pca,
        'X_test_pca': X_test_pca,
        'n_components_raw': X_train_pca.shape[1],
    }

if __name__ == "__main__":
    run_start = time.time()
    run_time = now_iso()
    #clear_output_checkpoints()
    # Retained only as a compatibility switch. The active workflow no longer
    # writes diagnostic caches.
    RETRAIN_ALL = True
    output_prefix = "8yrs"
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = get_checkpoint_dir(Path(output_dir), "base", f"{output_prefix}_{GRID_VERSION}")
    print(f"MODEL_N_JOBS={MODEL_N_JOBS} (set env MODEL_N_JOBS to override)")
    print(f"GRID_VERSION={GRID_VERSION}")

    # ------- Load and preprocess data -------
    # Set testing=True for the 2-year dataset; False for the full 8-year dataset.
    TESTING = False
    DATA = import_data(testing=TESTING, extra_features=False, cluster=False, n_clusters=100, corr_threshold=0.95, corr_level=0)
    X, y_regression = clean_data(*DATA, raw=True, extra_features=False)
    X = _keep_raw_stock_ohlcv(X)

    # Flatten multi-level columns to single strings: "Close_AAPL", "Volume_MSFT", etc.
    X.columns = [f"{metric}_{ticker}" for metric, _, ticker in X.columns]
    print(f"Feature matrix shape: {X.shape[0]} rows, {X.shape[1]} columns.")

    y_classification = to_binary_class(y_regression)
    print(f"Final shape — X: {X.shape}, y: {y_classification.shape}")

    # ------- Train/test split (80/20, no shuffle — time series order must be preserved) -------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_classification, test_size=TEST_SIZE, random_state=RANDOM_SEED, shuffle=TRAIN_TEST_SHUFFLE
    )

    # Previous temporary change used `KFold(n_splits=5, shuffle=False)`.
    tscv = TimeSeriesSplit(n_splits=TIME_SERIES_CV_SPLITS)

    def _compute_raw_pca_stage():
        print("\n========== Grid Search for Optimal PCA Components (Baseline) ===========")
        scaler_pca = StandardScaler()
        X_train_sc = scaler_pca.fit_transform(X_train)
        X_test_sc = scaler_pca.transform(X_test)

        n_components_grid = BASELINE_PCA_GRID
        grid_search_results = []
        print(f"Testing n_components: {n_components_grid}")
        for n_comp in n_components_grid:
            pca_temp = PCA(n_components=n_comp)
            X_pca_temp = pca_temp.fit_transform(X_train_sc)
            baseline_temp = LogisticRegression(**_build_logistic_kwargs(
                solver=LOGISTIC_BASELINE_SOLVER,
                l1_ratio=None,
                c_value=np.inf,
            ))
            with warnings.catch_warnings():
                _suppress_expected_no_penalty_warning()
                scores = cross_val_score(
                    baseline_temp, X_pca_temp, y_train, cv=tscv, n_jobs=MODEL_N_JOBS, scoring='balanced_accuracy'
                )
            mean_score = scores.mean()
            std_score = scores.std()
            grid_search_results.append({
                'n_components': n_comp,
                'n_components_value': X_pca_temp.shape[1],
                'cv_scores': scores,
                'mean_cv_score': mean_score,
                'std_cv_score': std_score
            })
            print(f"  n_components={n_comp} ({X_pca_temp.shape[1]} components): CV Balanced Accuracy = {mean_score:.4f} ± {std_score:.4f}")

        selected_pca, best_pca, pca_threshold = _select_pca_n_components_1se(grid_search_results)
        best_n_comp = selected_pca['n_components']
        pca = PCA(n_components=best_n_comp)
        X_train_pca = pca.fit_transform(X_train_sc)
        X_test_pca = pca.transform(X_test_sc)
        return {
            'scaler_pca': scaler_pca,
            'X_train_sc': X_train_sc,
            'X_test_sc': X_test_sc,
            'grid_search_results': grid_search_results,
            'selected_pca': selected_pca,
            'best_pca': best_pca,
            'pca_threshold': pca_threshold,
            'best_n_comp': best_n_comp,
            'best_n_comp_value': selected_pca['n_components_value'],
            'best_score': selected_pca['mean_cv_score'],
            'pca': pca,
            'X_train_pca': X_train_pca,
            'X_test_pca': X_test_pca,
            'n_components_raw': X_train_pca.shape[1],
        }

    raw_pca_stage = _load_or_compute_stage(
        checkpoint_dir,
        "raw_pca_selection",
        _compute_raw_pca_stage,
        "Grid Search for Optimal PCA Components (Baseline)",
    )
    scaler_pca = raw_pca_stage['scaler_pca']
    X_train_sc = raw_pca_stage['X_train_sc']
    X_test_sc = raw_pca_stage['X_test_sc']
    grid_search_results = raw_pca_stage['grid_search_results']
    selected_pca = raw_pca_stage['selected_pca']
    best_pca = raw_pca_stage['best_pca']
    pca_threshold = raw_pca_stage['pca_threshold']
    best_n_comp = raw_pca_stage['best_n_comp']
    best_n_comp_value = raw_pca_stage['best_n_comp_value']
    best_score = raw_pca_stage['best_score']
    pca = raw_pca_stage['pca']
    X_train_pca = raw_pca_stage['X_train_pca']
    X_test_pca = raw_pca_stage['X_test_pca']
    n_components_raw = raw_pca_stage['n_components_raw']
    print(
        f"\nBest-mean CV n_components: {best_pca['n_components']} ({best_pca['n_components_value']} components), "
        f"CV={best_pca['mean_cv_score']:.4f}±{best_pca['std_cv_score']:.4f}"
    )
    print(
        f"1SE-selected n_components: {best_n_comp} ({best_n_comp_value} components), "
        f"CV={best_score:.4f}±{selected_pca['std_cv_score']:.4f}, threshold={pca_threshold:.4f}"
    )
    print(f"Final PCA: {X_train.shape[1]} features → {n_components_raw} components ({best_n_comp*100:.0f}% variance)")

    # ------- Baseline: plain Logistic Regression (no regularization) after PCA -------
    def _compute_baseline_pca_stage():
        print("\n========== BASELINE: Plain Logistic Regression (No Regularization) + PCA ===========")
        baseline_clf = LogisticRegression(**_build_logistic_kwargs(
            solver=LOGISTIC_BASELINE_SOLVER,
            l1_ratio=None,
            c_value=np.inf,
        ))
        _fit_logistic_model(baseline_clf, X_train_pca, y_train)
        return {'baseline_clf': baseline_clf}

    baseline_clf = _load_or_compute_stage(
        checkpoint_dir,
        "raw_baseline_pca_model",
        _compute_baseline_pca_stage,
        "BASELINE: Plain Logistic Regression (No Regularization) + PCA",
    )['baseline_clf']
    
    # Cache the model and grid search results
    # Exporting the fitted baseline PCA object to .pkl is not needed for any
    # later figures/tables in the active workflow.
    # cache_data = {
    #     'model': baseline_clf,
    #     'pca': pca,
    #     'scaler': scaler_pca,
    #     'best_n_comp': best_n_comp,
    #     'best_cv_score': best_score,
    #     'grid_search_results': grid_search_results,
    #     'n_components_value': n_components_raw
    # }

    # ------- Ridge (L2): LogisticRegressionCV — stores per-fold scores for bias-variance plot -------
    def _compute_ridge_raw_stage():
        print("\n========== RIDGE (L2) Logistic Regression CV ==========")
        pipeline_ridge = Pipeline([
            ('scaler',     StandardScaler()),
            ('classifier', LogisticRegressionCV(**_build_logistic_cv_kwargs(
                cs=RIDGE_GRID, cv=tscv, solver=LOGISTIC_RIDGE_SOLVER, l1_ratio=0, scoring='balanced_accuracy'
            )))
        ])
        _fit_logistic_model(pipeline_ridge, X_train, y_train)
        ridge_cv = pipeline_ridge.named_steps['classifier']
        ridge_c_1se, ridge_c_best, _ = _select_c_1se_from_logregcv(ridge_cv)
        pipeline_ridge_1se = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(**_build_logistic_kwargs(
                solver=LOGISTIC_RIDGE_SOLVER,
                l1_ratio=0,
                c_value=ridge_c_1se,
            )))
        ])
        _fit_logistic_model(pipeline_ridge_1se, X_train, y_train)
        return {
            'pipeline_ridge': pipeline_ridge,
            'ridge_cv': ridge_cv,
            'ridge_c_1se': ridge_c_1se,
            'ridge_c_best': ridge_c_best,
            'pipeline_ridge_1se': pipeline_ridge_1se,
        }

    ridge_raw_stage = _load_or_compute_stage(
        checkpoint_dir,
        "raw_ridge_model",
        _compute_ridge_raw_stage,
        "RIDGE (L2) Logistic Regression CV",
    )
    pipeline_ridge = ridge_raw_stage['pipeline_ridge']
    ridge_cv = ridge_raw_stage['ridge_cv']
    ridge_c_1se = ridge_raw_stage['ridge_c_1se']
    ridge_c_best = ridge_raw_stage['ridge_c_best']
    pipeline_ridge_1se = ridge_raw_stage['pipeline_ridge_1se']
    print(f"Best C by mean CV (Ridge): {ridge_c_best:.6f}")
    print(f"1SE-selected C (Ridge):    {ridge_c_1se:.6f}")

    # ------- LASSO (L1): LogisticRegressionCV — stores per-fold scores for bias-variance plot -------
    def _compute_lasso_raw_stage():
        print("\n========== LASSO (L1) Logistic Regression CV ==========")
        pipeline_lasso = Pipeline([
            ('scaler',     StandardScaler()),
            ('classifier', LogisticRegressionCV(**_build_logistic_cv_kwargs(
                cs=LASSO_GRID, cv=tscv, solver=LOGISTIC_LASSO_SOLVER, l1_ratio=1, scoring='balanced_accuracy'
            )))
        ])
        _fit_logistic_model(pipeline_lasso, X_train, y_train)
        lasso_cv = pipeline_lasso.named_steps['classifier']
        lasso_c_1se, lasso_c_best, _ = _select_c_1se_from_logregcv(lasso_cv)
        pipeline_lasso_1se = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(**_build_logistic_kwargs(
                solver=LOGISTIC_LASSO_SOLVER,
                l1_ratio=1,
                c_value=lasso_c_1se,
            )))
        ])
        _fit_logistic_model(pipeline_lasso_1se, X_train, y_train)
        return {
            'pipeline_lasso': pipeline_lasso,
            'lasso_cv': lasso_cv,
            'lasso_c_1se': lasso_c_1se,
            'lasso_c_best': lasso_c_best,
            'pipeline_lasso_1se': pipeline_lasso_1se,
        }

    lasso_raw_stage = _load_or_compute_stage(
        checkpoint_dir,
        "raw_lasso_model",
        _compute_lasso_raw_stage,
        "LASSO (L1) Logistic Regression CV",
    )
    pipeline_lasso = lasso_raw_stage['pipeline_lasso']
    lasso_cv = lasso_raw_stage['lasso_cv']
    lasso_c_1se = lasso_raw_stage['lasso_c_1se']
    lasso_c_best = lasso_raw_stage['lasso_c_best']
    pipeline_lasso_1se = lasso_raw_stage['pipeline_lasso_1se']
    print(f"Best C by mean CV (LASSO): {lasso_c_best:.6f}")
    print(f"1SE-selected C (LASSO):    {lasso_c_1se:.6f}")

    # ------- Elastic Net (raw): GridSearchCV with 1SE refit -------
    # print("\n========== ELASTIC NET Logistic Regression Grid Search ==========")
    # pipeline_elastic = Pipeline([...])
    # elastic_param_grid = {
    #     'classifier__C': ...,
    #     'classifier__l1_ratio': ...,
    # }
    # grid_search_elastic = GridSearchCV(
    #     pipeline_elastic, elastic_param_grid, cv=tscv, return_train_score=True,
    #     n_jobs=MODEL_N_JOBS, scoring='balanced_accuracy',
    #     refit=make_one_se_refit(['classifier__C', 'classifier__l1_ratio'])
    # )
    # grid_search_elastic.fit(X_train, y_train)
    # elastic_best_c = float(grid_search_elastic.best_params_['classifier__C'])
    # elastic_best_l1 = float(grid_search_elastic.best_params_['classifier__l1_ratio'])
    # print(f"1SE-selected Elastic Net params: C={elastic_best_c:.6f}, l1_ratio={elastic_best_l1:.3f}")
    # pipeline_elastic_1se = clone(grid_search_elastic.best_estimator_)
    # pipeline_elastic_1se.fit(X_train, y_train)

    raw_scaler = StandardScaler()
    X_train_raw_sc = raw_scaler.fit_transform(X_train)
    X_test_raw_sc = raw_scaler.transform(X_test)


    # ------- Ridge CV after PCA -------
    def _compute_ridge_pca_stage():
        print("\n========== RIDGE (L2) + PCA — LogisticRegressionCV ==========")
        clf_ridge_pca_initial = LogisticRegressionCV(**_build_logistic_cv_kwargs(
            cs=RIDGE_GRID, cv=tscv, solver=LOGISTIC_RIDGE_SOLVER, l1_ratio=0, scoring='balanced_accuracy'
        ))
        _fit_logistic_model(clf_ridge_pca_initial, X_train_pca, y_train)
        ridge_pca_c_initial, _, _ = _select_c_1se_from_logregcv(clf_ridge_pca_initial)
        pca_selection = _retune_pca_with_fixed_logistic(
            X_train,
            y_train,
            X_test,
            BASELINE_PCA_GRID,
            tscv,
            c_value=ridge_pca_c_initial,
            solver=LOGISTIC_RIDGE_SOLVER,
            l1_ratio=0,
        )
        clf_ridge_pca = LogisticRegressionCV(**_build_logistic_cv_kwargs(
            cs=RIDGE_GRID, cv=tscv, solver=LOGISTIC_RIDGE_SOLVER, l1_ratio=0, scoring='balanced_accuracy'
        ))
        _fit_logistic_model(clf_ridge_pca, pca_selection['X_train_pca'], y_train)
        ridge_pca_c_1se, ridge_pca_c_best, _ = _select_c_1se_from_logregcv(clf_ridge_pca)
        clf_ridge_pca_1se = LogisticRegression(**_build_logistic_kwargs(
            solver=LOGISTIC_RIDGE_SOLVER,
            l1_ratio=0,
            c_value=ridge_pca_c_1se,
        ))
        _fit_logistic_model(clf_ridge_pca_1se, pca_selection['X_train_pca'], y_train)
        return {
            'clf_ridge_pca': clf_ridge_pca,
            'ridge_pca_c_1se': ridge_pca_c_1se,
            'ridge_pca_c_best': ridge_pca_c_best,
            'clf_ridge_pca_1se': clf_ridge_pca_1se,
            **pca_selection,
        }

    ridge_pca_stage = _load_or_compute_stage(
        checkpoint_dir,
        "raw_ridge_pca_model",
        _compute_ridge_pca_stage,
        "RIDGE (L2) + PCA — LogisticRegressionCV",
    )
    clf_ridge_pca = ridge_pca_stage['clf_ridge_pca']
    ridge_pca_c_1se = ridge_pca_stage['ridge_pca_c_1se']
    ridge_pca_c_best = ridge_pca_stage['ridge_pca_c_best']
    clf_ridge_pca_1se = ridge_pca_stage['clf_ridge_pca_1se']
    selected_pca_ridge = ridge_pca_stage['selected_pca']
    best_pca_ridge = ridge_pca_stage['best_pca']
    pca_threshold_ridge = ridge_pca_stage['pca_threshold']
    best_n_comp_ridge = ridge_pca_stage['best_n_comp']
    best_n_comp_value_ridge = ridge_pca_stage['best_n_comp_value']
    X_train_pca_ridge = ridge_pca_stage['X_train_pca']
    X_test_pca_ridge = ridge_pca_stage['X_test_pca']
    n_components_ridge = ridge_pca_stage['n_components_raw']
    print(f"Best C by mean CV (Ridge+PCA): {ridge_pca_c_best:.6f}")
    print(f"1SE-selected C (Ridge+PCA):    {ridge_pca_c_1se:.6f}")
    print(
        f"1SE-selected n_components (Ridge+PCA): {best_n_comp_ridge} "
        f"({best_n_comp_value_ridge} components), "
        f"CV={selected_pca_ridge['mean_cv_score']:.4f}±{selected_pca_ridge['std_cv_score']:.4f}, "
        f"threshold={pca_threshold_ridge:.4f}"
    )

    # ------- LASSO CV after PCA -------
    def _compute_lasso_pca_stage():
        print("\n========== LASSO (L1) + PCA — LogisticRegressionCV ==========")
        clf_lasso_pca_initial = LogisticRegressionCV(**_build_logistic_cv_kwargs(
            cs=LASSO_GRID, cv=tscv, solver=LOGISTIC_LASSO_SOLVER, l1_ratio=1, scoring='balanced_accuracy'
        ))
        _fit_logistic_model(clf_lasso_pca_initial, X_train_pca, y_train)
        lasso_pca_c_initial, _, _ = _select_c_1se_from_logregcv(clf_lasso_pca_initial)
        pca_selection = _retune_pca_with_fixed_logistic(
            X_train,
            y_train,
            X_test,
            BASELINE_PCA_GRID,
            tscv,
            c_value=lasso_pca_c_initial,
            solver=LOGISTIC_LASSO_SOLVER,
            l1_ratio=1,
        )
        clf_lasso_pca = LogisticRegressionCV(**_build_logistic_cv_kwargs(
            cs=LASSO_GRID, cv=tscv, solver=LOGISTIC_LASSO_SOLVER, l1_ratio=1, scoring='balanced_accuracy'
        ))
        _fit_logistic_model(clf_lasso_pca, pca_selection['X_train_pca'], y_train)
        lasso_pca_c_1se, lasso_pca_c_best, _ = _select_c_1se_from_logregcv(clf_lasso_pca)
        clf_lasso_pca_1se = LogisticRegression(**_build_logistic_kwargs(
            solver=LOGISTIC_LASSO_SOLVER,
            l1_ratio=1,
            c_value=lasso_pca_c_1se,
        ))
        _fit_logistic_model(clf_lasso_pca_1se, pca_selection['X_train_pca'], y_train)
        return {
            'clf_lasso_pca': clf_lasso_pca,
            'lasso_pca_c_1se': lasso_pca_c_1se,
            'lasso_pca_c_best': lasso_pca_c_best,
            'clf_lasso_pca_1se': clf_lasso_pca_1se,
            **pca_selection,
        }

    lasso_pca_stage = _load_or_compute_stage(
        checkpoint_dir,
        "raw_lasso_pca_model",
        _compute_lasso_pca_stage,
        "LASSO (L1) + PCA — LogisticRegressionCV",
    )
    clf_lasso_pca = lasso_pca_stage['clf_lasso_pca']
    lasso_pca_c_1se = lasso_pca_stage['lasso_pca_c_1se']
    lasso_pca_c_best = lasso_pca_stage['lasso_pca_c_best']
    clf_lasso_pca_1se = lasso_pca_stage['clf_lasso_pca_1se']
    selected_pca_lasso = lasso_pca_stage['selected_pca']
    best_pca_lasso = lasso_pca_stage['best_pca']
    pca_threshold_lasso = lasso_pca_stage['pca_threshold']
    best_n_comp_lasso = lasso_pca_stage['best_n_comp']
    best_n_comp_value_lasso = lasso_pca_stage['best_n_comp_value']
    X_train_pca_lasso = lasso_pca_stage['X_train_pca']
    X_test_pca_lasso = lasso_pca_stage['X_test_pca']
    n_components_lasso = lasso_pca_stage['n_components_raw']
    print(f"Best C by mean CV (LASSO+PCA): {lasso_pca_c_best:.6f}")
    print(f"1SE-selected C (LASSO+PCA):    {lasso_pca_c_1se:.6f}")
    print(
        f"1SE-selected n_components (LASSO+PCA): {best_n_comp_lasso} "
        f"({best_n_comp_value_lasso} components), "
        f"CV={selected_pca_lasso['mean_cv_score']:.4f}±{selected_pca_lasso['std_cv_score']:.4f}, "
        f"threshold={pca_threshold_lasso:.4f}"
    )

    # ===================================================================
    # MODEL COMPARISON TABLE
    # ===================================================================
    print("\n========== Model Comparison Table ==========")

    ranking_rows = []

    def _eval_row(name, model, X_tr, y_tr, X_te, y_te, n_splits, best_c=None, best_c_label=None):
        shared = get_or_compute_final_metrics(
            checkpoint_dir, _metrics_stage_name(name), model, X_tr, y_tr, X_te, y_te, n_splits=n_splits, label=name
        )
        ranking_rows.append({'Model': name, **shared})
        preds = model.predict(X_te)
        is_degenerate = (
            shared['is_degenerate_classifier']
            or (
                best_c is not None
                and np.isclose(best_c, 1e-6)
                and np.unique(preds).size == 1
            )
        )
        return {
            'Model':           name,
            'Best C':          best_c_label if best_c_label is not None else (f'{best_c:.6f}' if best_c is not None else 'N/A'),
            'Avg CV Train Plain Acc':   shared['train_avg_accuracy'],
            'CV Train Plain Acc SD':    shared['train_std_accuracy'],
            'Avg CV Validation Plain Acc': shared['validation_avg_accuracy'],
            'CV Acc SD':                   shared['validation_std_accuracy'],
            'Test Acc':                    shared['test_split_accuracy'],
            'MCC':             shared['test_matthew_corr_coef'],
            'Precision':       shared['test_precision'],
            'Recall':          shared['test_sensitivity'],
            'Specificity':     shared['test_specificity'],
            'F1':              shared['test_f1'],
            'ROC-AUC':         shared['test_roc_auc_macro'],
            'Degenerate':      is_degenerate,
        }

    rows = [
        _eval_row(f'Base+PCA ({n_components_raw}, {best_n_comp*100:.0f}%)', baseline_clf, X_train_pca, y_train, X_test_pca, y_test, tscv.n_splits, best_c=None),
        _eval_row('Raw Ridge', pipeline_ridge_1se, X_train, y_train, X_test, y_test, tscv.n_splits, best_c=ridge_c_1se),
        _eval_row('Raw LASSO', pipeline_lasso_1se, X_train, y_train, X_test, y_test, tscv.n_splits, best_c=lasso_c_1se),
        # _eval_row('Elastic Net (raw)', pipeline_elastic_1se, X_train, y_train, X_test, y_test, tscv.n_splits, best_c=elastic_best_c, best_c_label=f'{elastic_best_c:.6f} (l1={elastic_best_l1:.2f})'),
        _eval_row(f'Raw Ridge+PCA ({n_components_ridge})', clf_ridge_pca_1se, X_train_pca_ridge, y_train, X_test_pca_ridge, y_test, tscv.n_splits, best_c=ridge_pca_c_1se),
        _eval_row(f'Raw LASSO+PCA ({n_components_lasso})', clf_lasso_pca_1se, X_train_pca_lasso, y_train, X_test_pca_lasso, y_test, tscv.n_splits, best_c=lasso_pca_c_1se),
    ]

    comparison_df = pd.DataFrame(rows).set_index('Model')
    print(comparison_df.to_string())

    ranked_df = rank_models_by_metrics(pd.DataFrame(ranking_rows), criteria=CV_SELECTION_CRITERIA)
    print("\n===== Ranked Baseline Models =====")
    print(ranked_df[['Model', 'rank_validation_avg_roc_auc', 'rank_validation_avg_sensitivity',
                     'rank_validation_avg_specificity', 'rank_validation_std_accuracy', 'average_rank']].to_string(index=False))

    plot_configs = {
        f'Base+PCA ({n_components_raw}, {best_n_comp*100:.0f}%)': {
            'plot_specs': [
                {
                    'type': 'pca',
                    'suffix': 'pca_n_components',
                    'selected_n_comp': best_n_comp,
                    'selected_label': f'{_format_pca_grid_value(best_n_comp)} ({n_components_raw} comps)',
                    'model_title': 'Baseline LR + PCA',
                    'feature_title': 'Raw OHLCV Features',
                    'X_train_plot': X_train,
                    'X_test_plot': X_test,
                    'c_value': np.inf,
                    'solver': 'lbfgs',
                    'l1_ratio': None,
                    'grid': BASELINE_PCA_GRID,
                    'x_label': 'PCA n_components from tuning grid\n← Lower retained variance, Simpler Model      Higher retained variance, More Complex →',
                    'direct_color': 'darkorange',
                },
            ],
        },
        'Raw Ridge': {
            'plot_specs': [
                {
                    'type': 'c',
                    'suffix': 'classifier_C',
                    'logregcv': ridge_cv,
                    'one_se_c': ridge_c_1se,
                    'model_title': 'Ridge (L2) - LR',
                    'feature_title': 'Raw OHLCV Features',
                    'X_train_plot': X_train_raw_sc,
                    'X_test_plot': X_test_raw_sc,
                    'l1_ratio': 0,
                    'direct_color': 'darkorange',
                },
            ],
        },
        'Raw LASSO': {
            'plot_specs': [
                {
                    'type': 'c',
                    'suffix': 'classifier_C',
                    'logregcv': lasso_cv,
                    'one_se_c': lasso_c_1se,
                    'model_title': 'LASSO (L1) - LR',
                    'feature_title': 'Raw OHLCV Features',
                    'X_train_plot': X_train_raw_sc,
                    'X_test_plot': X_test_raw_sc,
                    'l1_ratio': 1,
                    'direct_color': 'seagreen',
                },
            ],
        },
        f'Raw Ridge+PCA ({n_components_ridge})': {
            'plot_specs': [
                {
                    'type': 'c',
                    'suffix': 'classifier_C',
                    'logregcv': clf_ridge_pca,
                    'one_se_c': ridge_pca_c_1se,
                    'model_title': 'Ridge (L2) - LR',
                    'feature_title': f'PCA Features ({n_components_ridge} comps, {best_n_comp_ridge*100:.0f}% variance)',
                    'X_train_plot': X_train_pca_ridge,
                    'X_test_plot': X_test_pca_ridge,
                    'l1_ratio': 0,
                    'direct_color': 'darkorange',
                },
                {
                    'type': 'pca',
                    'suffix': 'pca_n_components',
                    'selected_n_comp': best_n_comp_ridge,
                    'selected_label': f'{_format_pca_grid_value(best_n_comp_ridge)} ({n_components_ridge} comps)',
                    'model_title': 'Ridge (L2) - LR',
                    'feature_title': 'Raw OHLCV to PCA preprocessing',
                    'X_train_plot': X_train,
                    'X_test_plot': X_test,
                    'c_value': ridge_pca_c_1se,
                    'solver': 'saga',
                    'l1_ratio': 0,
                    'grid': BASELINE_PCA_GRID,
                    'x_label': 'PCA n_components from tuning grid\n← Lower retained variance, Simpler Model      Higher retained variance, More Complex →',
                    'direct_color': 'darkorange',
                },
            ],
        },
        f'Raw LASSO+PCA ({n_components_lasso})': {
            'plot_specs': [
                {
                    'type': 'c',
                    'suffix': 'classifier_C',
                    'logregcv': clf_lasso_pca,
                    'one_se_c': lasso_pca_c_1se,
                    'model_title': 'LASSO (L1) - LR',
                    'feature_title': f'PCA Features ({n_components_lasso} comps, {best_n_comp_lasso*100:.0f}% variance)',
                    'X_train_plot': X_train_pca_lasso,
                    'X_test_plot': X_test_pca_lasso,
                    'l1_ratio': 1,
                    'direct_color': 'seagreen',
                },
                {
                    'type': 'pca',
                    'suffix': 'pca_n_components',
                    'selected_n_comp': best_n_comp_lasso,
                    'selected_label': f'{_format_pca_grid_value(best_n_comp_lasso)} ({n_components_lasso} comps)',
                    'model_title': 'LASSO (L1) - LR',
                    'feature_title': 'Raw OHLCV to PCA preprocessing',
                    'X_train_plot': X_train,
                    'X_test_plot': X_test,
                    'c_value': lasso_pca_c_1se,
                    'solver': 'saga',
                    'l1_ratio': 1,
                    'grid': BASELINE_PCA_GRID,
                    'x_label': 'PCA n_components from tuning grid\n← Lower retained variance, Simpler Model      Higher retained variance, More Complex →',
                    'direct_color': 'seagreen',
                },
            ],
        },
        # 'Elastic Net (raw)': {
        #     'grid_search': grid_search_elastic,
        #     'x_param': 'classifier__C',
        #     'x_label': 'C',
        #     'model_title': 'Elastic Net - LR',
        #     'X_train_plot': X_train,
        #     'X_test_plot': X_test,
        # },
    }

    print(
        "\nDeferring best-model diagnostics until the combined raw + DOW ranking is available."
    )

    # ===================================================================
    # DAY-OF-WEEK EXTENSION
    # Add one-hot encoded day-of-week (Mon–Fri) to raw OHLCV features
    # then re-run the same 5 models
    # ===================================================================
    print("\n========== Adding Day-of-Week Features ==========")

    dow_dummies = pd.get_dummies(X.index.dayofweek, prefix='DOW').astype(float)
    dow_dummies.index = X.index
    # Drop Friday explicitly to avoid the dummy variable trap.
    if 'DOW_4' in dow_dummies.columns:
        dow_dummies = dow_dummies.drop(columns=['DOW_4'])
    print(f"Day-of-week columns added: {list(dow_dummies.columns)}")

    X_dow = pd.concat([X, dow_dummies], axis=1)
    X_train_dow, X_test_dow, _, _ = train_test_split(
        X_dow, y_classification, test_size=TEST_SIZE, random_state=RANDOM_SEED, shuffle=TRAIN_TEST_SHUFFLE
    )
    print(f"Feature matrix with DOW: {X_dow.shape[1]} columns")

    def _compute_dow_pca_stage():
        print("\n========== Grid Search for Optimal PCA Components (DOW) ===========")
        n_components_grid = BASELINE_PCA_GRID
        grid_search_results_dow = []

        scaler_pca_dow = StandardScaler()
        X_train_dow_sc = scaler_pca_dow.fit_transform(X_train_dow)
        X_test_dow_sc = scaler_pca_dow.transform(X_test_dow)

        print(f"Testing n_components: {n_components_grid}")
        for n_comp in n_components_grid:
            pca_temp = PCA(n_components=n_comp)
            X_pca_temp = pca_temp.fit_transform(X_train_dow_sc)
            baseline_temp = LogisticRegression(**_build_logistic_kwargs(
                solver=LOGISTIC_BASELINE_SOLVER,
                l1_ratio=None,
                c_value=np.inf,
            ))
            with warnings.catch_warnings():
                _suppress_expected_no_penalty_warning()
                scores = cross_val_score(
                    baseline_temp, X_pca_temp, y_train, cv=tscv, n_jobs=MODEL_N_JOBS, scoring='balanced_accuracy'
                )
            mean_score = scores.mean()
            std_score = scores.std()
            grid_search_results_dow.append({
                'n_components': n_comp,
                'n_components_value': X_pca_temp.shape[1],
                'cv_scores': scores,
                'mean_cv_score': mean_score,
                'std_cv_score': std_score
            })
            print(f"  n_components={n_comp} ({X_pca_temp.shape[1]} components): CV Balanced Accuracy = {mean_score:.4f} ± {std_score:.4f}")

        selected_pca_dow, best_pca_dow, pca_threshold_dow = _select_pca_n_components_1se(grid_search_results_dow)
        best_n_comp_dow = selected_pca_dow['n_components']
        pca_dow = PCA(n_components=best_n_comp_dow)
        X_train_dow_pca = pca_dow.fit_transform(X_train_dow_sc)
        X_test_dow_pca = pca_dow.transform(X_test_dow_sc)
        return {
            'grid_search_results_dow': grid_search_results_dow,
            'scaler_pca_dow': scaler_pca_dow,
            'X_train_dow_sc': X_train_dow_sc,
            'X_test_dow_sc': X_test_dow_sc,
            'selected_pca_dow': selected_pca_dow,
            'best_pca_dow': best_pca_dow,
            'pca_threshold_dow': pca_threshold_dow,
            'best_n_comp_dow': best_n_comp_dow,
            'best_n_comp_value_dow': selected_pca_dow['n_components_value'],
            'best_score_dow': selected_pca_dow['mean_cv_score'],
            'pca_dow': pca_dow,
            'X_train_dow_pca': X_train_dow_pca,
            'X_test_dow_pca': X_test_dow_pca,
            'n_components_dow': X_train_dow_pca.shape[1],
        }

    dow_pca_stage = _load_or_compute_stage(
        checkpoint_dir,
        "dow_pca_selection",
        _compute_dow_pca_stage,
        "Grid Search for Optimal PCA Components (DOW)",
    )
    grid_search_results_dow = dow_pca_stage['grid_search_results_dow']
    scaler_pca_dow = dow_pca_stage['scaler_pca_dow']
    X_train_dow_sc = dow_pca_stage['X_train_dow_sc']
    X_test_dow_sc = dow_pca_stage['X_test_dow_sc']
    selected_pca_dow = dow_pca_stage['selected_pca_dow']
    best_pca_dow = dow_pca_stage['best_pca_dow']
    pca_threshold_dow = dow_pca_stage['pca_threshold_dow']
    best_n_comp_dow = dow_pca_stage['best_n_comp_dow']
    best_n_comp_value_dow = dow_pca_stage['best_n_comp_value_dow']
    best_score_dow = dow_pca_stage['best_score_dow']
    pca_dow = dow_pca_stage['pca_dow']
    X_train_dow_pca = dow_pca_stage['X_train_dow_pca']
    X_test_dow_pca = dow_pca_stage['X_test_dow_pca']
    n_components_dow = dow_pca_stage['n_components_dow']
    print(
        f"\nBest-mean CV n_components (DOW): {best_pca_dow['n_components']} ({best_pca_dow['n_components_value']} components), "
        f"CV={best_pca_dow['mean_cv_score']:.4f}±{best_pca_dow['std_cv_score']:.4f}"
    )
    print(
        f"1SE-selected n_components (DOW): {best_n_comp_dow} ({best_n_comp_value_dow} components), "
        f"CV={best_score_dow:.4f}±{selected_pca_dow['std_cv_score']:.4f}, threshold={pca_threshold_dow:.4f}"
    )
    print(f"Final PCA: {X_train_dow.shape[1]} features → {n_components_dow} components ({best_n_comp_dow*100:.0f}% variance)")

    # --- Baseline + DOW + PCA ---
    def _compute_baseline_dow_pca_stage():
        print("\n========== BASELINE (No Reg) + DOW + PCA ===========")
        baseline_dow_clf = LogisticRegression(**_build_logistic_kwargs(
            solver=LOGISTIC_BASELINE_SOLVER,
            l1_ratio=None,
            c_value=np.inf,
        ))
        _fit_logistic_model(baseline_dow_clf, X_train_dow_pca, y_train)
        return {'baseline_dow_clf': baseline_dow_clf}

    baseline_dow_clf = _load_or_compute_stage(
        checkpoint_dir,
        "dow_baseline_pca_model",
        _compute_baseline_dow_pca_stage,
        "BASELINE (No Reg) + DOW + PCA",
    )['baseline_dow_clf']
    
    # Cache the model and grid search results
    # Exporting the fitted DOW baseline PCA object to .pkl is not needed for
    # any later figures/tables in the active workflow.
    # cache_data = {
    #     'model': baseline_dow_clf,
    #     'pca': pca_dow,
    #     'scaler': scaler_pca_dow,
    #     'best_n_comp': best_n_comp_dow,
    #     'best_cv_score': best_score_dow,
    #     'grid_search_results': grid_search_results_dow,
    #     'n_components_value': n_components_dow
    # }

    # --- Ridge CV + DOW ---
    def _compute_ridge_dow_stage():
        print("\n========== RIDGE CV + DOW ==========")
        pipeline_ridge_dow = Pipeline([
            ('scaler',     StandardScaler()),
            ('classifier', LogisticRegressionCV(**_build_logistic_cv_kwargs(
                cs=RIDGE_GRID, cv=tscv, solver=LOGISTIC_RIDGE_SOLVER, l1_ratio=0, scoring='balanced_accuracy'
            )))
        ])
        _fit_logistic_model(pipeline_ridge_dow, X_train_dow, y_train)
        ridge_dow_cv = pipeline_ridge_dow.named_steps['classifier']
        ridge_dow_c_1se, ridge_dow_c_best, _ = _select_c_1se_from_logregcv(ridge_dow_cv)
        pipeline_ridge_dow_1se = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(**_build_logistic_kwargs(
                solver=LOGISTIC_RIDGE_SOLVER,
                l1_ratio=0,
                c_value=ridge_dow_c_1se,
            )))
        ])
        _fit_logistic_model(pipeline_ridge_dow_1se, X_train_dow, y_train)
        return {
            'pipeline_ridge_dow': pipeline_ridge_dow,
            'ridge_dow_cv': ridge_dow_cv,
            'ridge_dow_c_1se': ridge_dow_c_1se,
            'ridge_dow_c_best': ridge_dow_c_best,
            'pipeline_ridge_dow_1se': pipeline_ridge_dow_1se,
        }

    ridge_dow_stage = _load_or_compute_stage(
        checkpoint_dir,
        "dow_ridge_model",
        _compute_ridge_dow_stage,
        "RIDGE CV + DOW",
    )
    pipeline_ridge_dow = ridge_dow_stage['pipeline_ridge_dow']
    ridge_dow_cv = ridge_dow_stage['ridge_dow_cv']
    ridge_dow_c_1se = ridge_dow_stage['ridge_dow_c_1se']
    ridge_dow_c_best = ridge_dow_stage['ridge_dow_c_best']
    pipeline_ridge_dow_1se = ridge_dow_stage['pipeline_ridge_dow_1se']
    print(f"Best C by mean CV (Ridge+DOW): {ridge_dow_c_best:.6f}")
    print(f"1SE-selected C (Ridge+DOW):    {ridge_dow_c_1se:.6f}")

    # --- LASSO CV + DOW ---
    def _compute_lasso_dow_stage():
        print("\n========== LASSO CV + DOW ==========")
        pipeline_lasso_dow = Pipeline([
            ('scaler',     StandardScaler()),
            ('classifier', LogisticRegressionCV(**_build_logistic_cv_kwargs(
                cs=LASSO_GRID, cv=tscv, solver=LOGISTIC_LASSO_SOLVER, l1_ratio=1, scoring='balanced_accuracy'
            )))
        ])
        _fit_logistic_model(pipeline_lasso_dow, X_train_dow, y_train)
        lasso_dow_cv = pipeline_lasso_dow.named_steps['classifier']
        lasso_dow_c_1se, lasso_dow_c_best, _ = _select_c_1se_from_logregcv(lasso_dow_cv)
        pipeline_lasso_dow_1se = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(**_build_logistic_kwargs(
                solver=LOGISTIC_LASSO_SOLVER,
                l1_ratio=1,
                c_value=lasso_dow_c_1se,
            )))
        ])
        _fit_logistic_model(pipeline_lasso_dow_1se, X_train_dow, y_train)
        return {
            'pipeline_lasso_dow': pipeline_lasso_dow,
            'lasso_dow_cv': lasso_dow_cv,
            'lasso_dow_c_1se': lasso_dow_c_1se,
            'lasso_dow_c_best': lasso_dow_c_best,
            'pipeline_lasso_dow_1se': pipeline_lasso_dow_1se,
        }

    lasso_dow_stage = _load_or_compute_stage(
        checkpoint_dir,
        "dow_lasso_model",
        _compute_lasso_dow_stage,
        "LASSO CV + DOW",
    )
    pipeline_lasso_dow = lasso_dow_stage['pipeline_lasso_dow']
    lasso_dow_cv = lasso_dow_stage['lasso_dow_cv']
    lasso_dow_c_1se = lasso_dow_stage['lasso_dow_c_1se']
    lasso_dow_c_best = lasso_dow_stage['lasso_dow_c_best']
    pipeline_lasso_dow_1se = lasso_dow_stage['pipeline_lasso_dow_1se']
    print(f"Best C by mean CV (LASSO+DOW): {lasso_dow_c_best:.6f}")
    print(f"1SE-selected C (LASSO+DOW):    {lasso_dow_c_1se:.6f}")


    # --- Ridge CV + PCA + DOW ---
    def _compute_ridge_pca_dow_stage():
        print("\n========== RIDGE CV + PCA + DOW ==========")
        clf_ridge_pca_dow_initial = LogisticRegressionCV(**_build_logistic_cv_kwargs(
            cs=RIDGE_GRID, cv=tscv, solver=LOGISTIC_RIDGE_SOLVER, l1_ratio=0, scoring='balanced_accuracy'
        ))
        _fit_logistic_model(clf_ridge_pca_dow_initial, X_train_dow_pca, y_train)
        ridge_pca_dow_c_initial, _, _ = _select_c_1se_from_logregcv(clf_ridge_pca_dow_initial)
        pca_selection = _retune_pca_with_fixed_logistic(
            X_train_dow,
            y_train,
            X_test_dow,
            BASELINE_PCA_GRID,
            tscv,
            c_value=ridge_pca_dow_c_initial,
            solver=LOGISTIC_RIDGE_SOLVER,
            l1_ratio=0,
        )
        clf_ridge_pca_dow = LogisticRegressionCV(**_build_logistic_cv_kwargs(
            cs=RIDGE_GRID, cv=tscv, solver=LOGISTIC_RIDGE_SOLVER, l1_ratio=0, scoring='balanced_accuracy'
        ))
        _fit_logistic_model(clf_ridge_pca_dow, pca_selection['X_train_pca'], y_train)
        ridge_pca_dow_c_1se, ridge_pca_dow_c_best, _ = _select_c_1se_from_logregcv(clf_ridge_pca_dow)
        clf_ridge_pca_dow_1se = LogisticRegression(**_build_logistic_kwargs(
            solver=LOGISTIC_RIDGE_SOLVER,
            l1_ratio=0,
            c_value=ridge_pca_dow_c_1se,
        ))
        _fit_logistic_model(clf_ridge_pca_dow_1se, pca_selection['X_train_pca'], y_train)
        return {
            'clf_ridge_pca_dow': clf_ridge_pca_dow,
            'ridge_pca_dow_c_1se': ridge_pca_dow_c_1se,
            'ridge_pca_dow_c_best': ridge_pca_dow_c_best,
            'clf_ridge_pca_dow_1se': clf_ridge_pca_dow_1se,
            **pca_selection,
        }

    ridge_pca_dow_stage = _load_or_compute_stage(
        checkpoint_dir,
        "dow_ridge_pca_model",
        _compute_ridge_pca_dow_stage,
        "RIDGE CV + PCA + DOW",
    )
    clf_ridge_pca_dow = ridge_pca_dow_stage['clf_ridge_pca_dow']
    ridge_pca_dow_c_1se = ridge_pca_dow_stage['ridge_pca_dow_c_1se']
    ridge_pca_dow_c_best = ridge_pca_dow_stage['ridge_pca_dow_c_best']
    clf_ridge_pca_dow_1se = ridge_pca_dow_stage['clf_ridge_pca_dow_1se']
    selected_pca_ridge_dow = ridge_pca_dow_stage['selected_pca']
    best_pca_ridge_dow = ridge_pca_dow_stage['best_pca']
    pca_threshold_ridge_dow = ridge_pca_dow_stage['pca_threshold']
    best_n_comp_ridge_dow = ridge_pca_dow_stage['best_n_comp']
    best_n_comp_value_ridge_dow = ridge_pca_dow_stage['best_n_comp_value']
    X_train_pca_ridge_dow = ridge_pca_dow_stage['X_train_pca']
    X_test_pca_ridge_dow = ridge_pca_dow_stage['X_test_pca']
    n_components_ridge_dow = ridge_pca_dow_stage['n_components_raw']
    print(f"Best C by mean CV (Ridge+PCA+DOW): {ridge_pca_dow_c_best:.6f}")
    print(f"1SE-selected C (Ridge+PCA+DOW):    {ridge_pca_dow_c_1se:.6f}")
    print(
        f"1SE-selected n_components (Ridge+PCA+DOW): {best_n_comp_ridge_dow} "
        f"({best_n_comp_value_ridge_dow} components), "
        f"CV={selected_pca_ridge_dow['mean_cv_score']:.4f}±{selected_pca_ridge_dow['std_cv_score']:.4f}, "
        f"threshold={pca_threshold_ridge_dow:.4f}"
    )

    # --- LASSO CV + PCA + DOW ---
    def _compute_lasso_pca_dow_stage():
        print("\n========== LASSO CV + PCA + DOW ==========")
        clf_lasso_pca_dow_initial = LogisticRegressionCV(**_build_logistic_cv_kwargs(
            cs=LASSO_GRID, cv=tscv, solver=LOGISTIC_LASSO_SOLVER, l1_ratio=1, scoring='balanced_accuracy'
        ))
        _fit_logistic_model(clf_lasso_pca_dow_initial, X_train_dow_pca, y_train)
        lasso_pca_dow_c_initial, _, _ = _select_c_1se_from_logregcv(clf_lasso_pca_dow_initial)
        pca_selection = _retune_pca_with_fixed_logistic(
            X_train_dow,
            y_train,
            X_test_dow,
            BASELINE_PCA_GRID,
            tscv,
            c_value=lasso_pca_dow_c_initial,
            solver=LOGISTIC_LASSO_SOLVER,
            l1_ratio=1,
        )
        clf_lasso_pca_dow = LogisticRegressionCV(**_build_logistic_cv_kwargs(
            cs=LASSO_GRID, cv=tscv, solver=LOGISTIC_LASSO_SOLVER, l1_ratio=1, scoring='balanced_accuracy'
        ))
        _fit_logistic_model(clf_lasso_pca_dow, pca_selection['X_train_pca'], y_train)
        lasso_pca_dow_c_1se, lasso_pca_dow_c_best, _ = _select_c_1se_from_logregcv(clf_lasso_pca_dow)
        clf_lasso_pca_dow_1se = LogisticRegression(**_build_logistic_kwargs(
            solver=LOGISTIC_LASSO_SOLVER,
            l1_ratio=1,
            c_value=lasso_pca_dow_c_1se,
        ))
        _fit_logistic_model(clf_lasso_pca_dow_1se, pca_selection['X_train_pca'], y_train)
        return {
            'clf_lasso_pca_dow': clf_lasso_pca_dow,
            'lasso_pca_dow_c_1se': lasso_pca_dow_c_1se,
            'lasso_pca_dow_c_best': lasso_pca_dow_c_best,
            'clf_lasso_pca_dow_1se': clf_lasso_pca_dow_1se,
            **pca_selection,
        }

    lasso_pca_dow_stage = _load_or_compute_stage(
        checkpoint_dir,
        "dow_lasso_pca_model",
        _compute_lasso_pca_dow_stage,
        "LASSO CV + PCA + DOW",
    )
    clf_lasso_pca_dow = lasso_pca_dow_stage['clf_lasso_pca_dow']
    lasso_pca_dow_c_1se = lasso_pca_dow_stage['lasso_pca_dow_c_1se']
    lasso_pca_dow_c_best = lasso_pca_dow_stage['lasso_pca_dow_c_best']
    clf_lasso_pca_dow_1se = lasso_pca_dow_stage['clf_lasso_pca_dow_1se']
    selected_pca_lasso_dow = lasso_pca_dow_stage['selected_pca']
    best_pca_lasso_dow = lasso_pca_dow_stage['best_pca']
    pca_threshold_lasso_dow = lasso_pca_dow_stage['pca_threshold']
    best_n_comp_lasso_dow = lasso_pca_dow_stage['best_n_comp']
    best_n_comp_value_lasso_dow = lasso_pca_dow_stage['best_n_comp_value']
    X_train_pca_lasso_dow = lasso_pca_dow_stage['X_train_pca']
    X_test_pca_lasso_dow = lasso_pca_dow_stage['X_test_pca']
    n_components_lasso_dow = lasso_pca_dow_stage['n_components_raw']
    print(f"Best C by mean CV (LASSO+PCA+DOW): {lasso_pca_dow_c_best:.6f}")
    print(f"1SE-selected C (LASSO+PCA+DOW):    {lasso_pca_dow_c_1se:.6f}")
    print(
        f"1SE-selected n_components (LASSO+PCA+DOW): {best_n_comp_lasso_dow} "
        f"({best_n_comp_value_lasso_dow} components), "
        f"CV={selected_pca_lasso_dow['mean_cv_score']:.4f}±{selected_pca_lasso_dow['std_cv_score']:.4f}, "
        f"threshold={pca_threshold_lasso_dow:.4f}"
    )

    # --- DOW rows using same _metrics helper ---
    rows_dow = [
        _eval_row(f'Base+DOW+PCA ({n_components_dow}, {best_n_comp_dow*100:.0f}%)', baseline_dow_clf, X_train_dow_pca, y_train, X_test_dow_pca, y_test, tscv.n_splits, best_c=None),
        _eval_row('Ridge+DOW', pipeline_ridge_dow_1se, X_train_dow, y_train, X_test_dow, y_test, tscv.n_splits, best_c=ridge_dow_c_1se),
        _eval_row('LASSO+DOW', pipeline_lasso_dow_1se, X_train_dow, y_train, X_test_dow, y_test, tscv.n_splits, best_c=lasso_dow_c_1se),
        _eval_row(f'Ridge+PCA+DOW ({n_components_ridge_dow})', clf_ridge_pca_dow_1se, X_train_pca_ridge_dow, y_train, X_test_pca_ridge_dow, y_test, tscv.n_splits, best_c=ridge_pca_dow_c_1se),
        _eval_row(f'LASSO+PCA+DOW ({n_components_lasso_dow})', clf_lasso_pca_dow_1se, X_train_pca_lasso_dow, y_train, X_test_pca_lasso_dow, y_test, tscv.n_splits, best_c=lasso_pca_dow_c_1se),
    ]
    dow_df = pd.DataFrame(rows_dow).set_index('Model')
    print("\nDay-of-Week Models:")
    print(dow_df.to_string())

    # ===================================================================
    # COMBINED COMPARISON TABLE (raw OHLCV vs raw OHLCV + DOW)
    # ===================================================================
    combined_df = pd.concat([comparison_df, dow_df])
    combined_df.index.name = 'Model'

    print("\n===== Combined Comparison Table (raw + DOW) =====")
    print(combined_df.to_string())

    ranked_df = rank_models_by_metrics(pd.DataFrame(ranking_rows), criteria=CV_SELECTION_CRITERIA)
    print("\n===== Ranked Models Across Raw + DOW =====")
    print(ranked_df[['Model', 'rank_validation_avg_roc_auc', 'rank_validation_avg_sensitivity',
                     'rank_validation_avg_specificity', 'rank_validation_std_accuracy', 'average_rank']].to_string(index=False))

    final_plot_configs = dict(plot_configs)
    final_plot_configs.update({
        f'Base+DOW+PCA ({n_components_dow}, {best_n_comp_dow*100:.0f}%)': {
            'plot_specs': [
                {
                    'type': 'pca',
                    'suffix': 'dow_pca_n_components',
                    'selected_n_comp': best_n_comp_dow,
                    'selected_label': f'{_format_pca_grid_value(best_n_comp_dow)} ({n_components_dow} comps)',
                    'model_title': 'Baseline LR + PCA + DOW',
                    'feature_title': 'Raw OHLCV + Day-of-Week Features',
                    'X_train_plot': X_train_dow,
                    'X_test_plot': X_test_dow,
                    'c_value': np.inf,
                    'solver': 'lbfgs',
                    'l1_ratio': None,
                    'grid': BASELINE_PCA_GRID,
                    'x_label': 'PCA n_components from tuning grid\n← Lower retained variance, Simpler Model      Higher retained variance, More Complex →',
                    'direct_color': 'darkorange',
                },
            ],
        },
        'Ridge+DOW': {
            'plot_specs': [
                {
                    'type': 'c',
                    'suffix': 'dow_classifier_C',
                    'logregcv': ridge_dow_cv,
                    'one_se_c': ridge_dow_c_1se,
                    'model_title': 'Ridge (L2) - LR + DOW',
                    'feature_title': 'Raw OHLCV + Day-of-Week Features',
                    'X_train_plot': X_train_dow_sc,
                    'X_test_plot': X_test_dow_sc,
                    'l1_ratio': 0,
                    'direct_color': 'darkorange',
                },
            ],
        },
        'LASSO+DOW': {
            'plot_specs': [
                {
                    'type': 'c',
                    'suffix': 'dow_classifier_C',
                    'logregcv': lasso_dow_cv,
                    'one_se_c': lasso_dow_c_1se,
                    'model_title': 'LASSO (L1) - LR + DOW',
                    'feature_title': 'Raw OHLCV + Day-of-Week Features',
                    'X_train_plot': X_train_dow_sc,
                    'X_test_plot': X_test_dow_sc,
                    'l1_ratio': 1,
                    'direct_color': 'seagreen',
                },
            ],
        },
        f'Ridge+PCA+DOW ({n_components_ridge_dow})': {
            'plot_specs': [
                {
                    'type': 'c',
                    'suffix': 'dow_classifier_C',
                    'logregcv': clf_ridge_pca_dow,
                    'one_se_c': ridge_pca_dow_c_1se,
                    'model_title': 'Ridge (L2) - LR + DOW',
                    'feature_title': f'PCA + DOW Features ({n_components_ridge_dow} comps, {best_n_comp_ridge_dow*100:.0f}% variance)',
                    'X_train_plot': X_train_pca_ridge_dow,
                    'X_test_plot': X_test_pca_ridge_dow,
                    'l1_ratio': 0,
                    'direct_color': 'darkorange',
                },
                {
                    'type': 'pca',
                    'suffix': 'dow_pca_n_components',
                    'selected_n_comp': best_n_comp_ridge_dow,
                    'selected_label': f'{_format_pca_grid_value(best_n_comp_ridge_dow)} ({n_components_ridge_dow} comps)',
                    'model_title': 'Ridge (L2) - LR + DOW',
                    'feature_title': 'Raw OHLCV + Day-of-Week to PCA preprocessing',
                    'X_train_plot': X_train_dow,
                    'X_test_plot': X_test_dow,
                    'c_value': ridge_pca_dow_c_1se,
                    'solver': 'saga',
                    'l1_ratio': 0,
                    'grid': BASELINE_PCA_GRID,
                    'x_label': 'PCA n_components from tuning grid\n← Lower retained variance, Simpler Model      Higher retained variance, More Complex →',
                    'direct_color': 'darkorange',
                },
            ],
        },
        f'LASSO+PCA+DOW ({n_components_lasso_dow})': {
            'plot_specs': [
                {
                    'type': 'c',
                    'suffix': 'dow_classifier_C',
                    'logregcv': clf_lasso_pca_dow,
                    'one_se_c': lasso_pca_dow_c_1se,
                    'model_title': 'LASSO (L1) - LR + DOW',
                    'feature_title': f'PCA + DOW Features ({n_components_lasso_dow} comps, {best_n_comp_lasso_dow*100:.0f}% variance)',
                    'X_train_plot': X_train_pca_lasso_dow,
                    'X_test_plot': X_test_pca_lasso_dow,
                    'l1_ratio': 1,
                    'direct_color': 'seagreen',
                },
                {
                    'type': 'pca',
                    'suffix': 'dow_pca_n_components',
                    'selected_n_comp': best_n_comp_lasso_dow,
                    'selected_label': f'{_format_pca_grid_value(best_n_comp_lasso_dow)} ({n_components_lasso_dow} comps)',
                    'model_title': 'LASSO (L1) - LR + DOW',
                    'feature_title': 'Raw OHLCV + Day-of-Week to PCA preprocessing',
                    'X_train_plot': X_train_dow,
                    'X_test_plot': X_test_dow,
                    'c_value': lasso_pca_dow_c_1se,
                    'solver': 'saga',
                    'l1_ratio': 1,
                    'grid': BASELINE_PCA_GRID,
                    'x_label': 'PCA n_components from tuning grid\n← Lower retained variance, Simpler Model      Higher retained variance, More Complex →',
                    'direct_color': 'seagreen',
                },
            ],
        },
    })

    best_model_name = str(ranked_df.iloc[0]['Model'])
    plot_model_name = select_non_degenerate_plot_model(
        ranked_df,
        available_models=final_plot_configs,
    )

    print(f"\nBest model across raw + DOW by average rank: {best_model_name}")
    if plot_model_name != best_model_name:
        print(f"Plotting fallback (non-degenerate): {plot_model_name}")
    print(f"Plotting diagnostics for: {plot_model_name}")
    best_plot_cfg = final_plot_configs[plot_model_name]
    base_bv_path = Path(output_dir) / '8yrs_1SE_base_logistic_best_bias_variance.png'
    base_direct_path = Path(output_dir) / '8yrs_1SE_base_logistic_best_train_test.png'
    for idx, spec in enumerate(best_plot_cfg['plot_specs']):
        output_bv_path = base_bv_path if idx == 0 else _append_plot_suffix(base_bv_path, spec['suffix'])
        output_direct_path = base_direct_path if idx == 0 else _append_plot_suffix(base_direct_path, spec['suffix'])
        if spec['type'] == 'c':
            diagnostics = {
                'single_bv': _compute_bv_curves(
                    spec['logregcv'],
                    spec['X_train_plot'],
                    y_train,
                    tscv,
                    spec['l1_ratio'],
                    'saga',
                ),
                'single_direct': _compute_direct_split_errors(
                    spec['X_train_plot'],
                    y_train,
                    spec['X_test_plot'],
                    y_test,
                    _augment_c_grid_with_selected_values(
                        spec['logregcv'].Cs_,
                        spec['one_se_c'],
                    ),
                    spec['l1_ratio'],
                    'saga',
                ),
            }
            _plot_single_model_diagnostics(
                diagnostics=diagnostics,
                bv_key='single_bv',
                direct_key='single_direct',
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
                direct_color=spec['direct_color'],
            )
        else:
            pca_diag = _compute_pca_curve_diagnostics(
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
            _save_curve_pair(
                pca_diag['curve'],
                pca_diag['direct'],
                pca_diag['selected_idx'],
                pca_diag['selected_label'],
                spec['model_title'],
                spec['feature_title'],
                spec['x_label'],
                str(output_bv_path),
                str(output_direct_path),
                x_scale='linear',
                direct_color=spec['direct_color'],
            )

    # ===================================================================
    # FINAL COMBINED TABLE: raw | raw+DOW
    # ===================================================================
    full_df = build_compact_export_table(combined_df)
    full_df.index.name = 'Model'

    print("\n===== Full Comparison Table =====")
    print(full_df.to_string())

    tex_path = os.path.join(output_dir, '8yrs_1SE_base_logistic_comparison.tex')
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
        tex_path,
        'Logistic Regression Model Comparison: Raw OHLCV vs Raw OHLCV + Day-of-Week',
        'tab:base_logistic_comparison',
        baseline_note,
        groups=[comparison_df.index, dow_df.index],
        degenerate_models=degenerate_models,
        degenerate_note=lasso_note,
        escape_note=False,
        escape_degenerate_note=False,
    )

    global_ranked_df = register_global_model_candidates(
        ranked_df,
        Path(output_dir) / f"{output_prefix}_global_model_leaderboard.csv",
        source_script="base.py",
        dataset_label=output_prefix,
        comparison_scope="raw_vs_dow",
        reset_leaderboard=True,
    )
    print(f"Local ranked/exported winner in base.py: {best_model_name}")
    print(f"Local plot winner in base.py: {plot_model_name}")
    print(f"Current global best model across registered scripts (informational only): {global_ranked_df.iloc[0]['Model']}")
    append_search_run(
        runs_path=Path(output_dir) / f"{output_prefix}_search_runs.csv",
        model_name="base_logreg",
        run_time=run_time,
        run_duration_sec=(time.time() - run_start),
        grid_version=GRID_VERSION,
        n_jobs=MODEL_N_JOBS,
        dataset_version="testing=False,extra_features=False,cluster=False,corr_threshold=0.95,corr_level=0",
        code_commit=get_git_commit(Path(output_dir).resolve().parents[0]),
        notes=SEARCH_NOTES,
    )
    print(f"LaTeX table saved to:           {os.path.abspath(tex_path)}")

    print("\n" + "="*70)
    print("All models trained successfully!")
    print("="*70)
