"""
Baseline Logistic Regression Models with Regularization Comparison

This script trains and evaluates baseline logistic regression models:
  - Baseline (No Regularization) with optimal PCA components (grid search)
  - Ridge (L2) regularization with cross-validation
  - LASSO (L1) regularization with cross-validation
  
Tested on:
  - Raw OHLCV features
  - Raw OHLCV + Day-of-Week features
  
All models are cached for later retrieval and analysis.

SCRIPT STRUCTURE:
  1. Model Training (lines 50-600): Run models and save to cache
  2. Comparison Tables & LaTeX Export (lines 600-680)
  3. Helper Functions (lines 680+): Functions to load cached results for plotting

USAGE:
  - First run / no cache files yet: keep RETRAIN_ALL = True
  - To refresh all cached outputs: Set RETRAIN_ALL = True
  - To reuse only the cached diagnostics curves: Set RETRAIN_ALL = False
  - To load cached results: Use helper functions at the end of this file
    Example: cached_data = load_baseline_pca_results()
"""

import os
import joblib
import pandas as pd
from H_prep import clean_data, import_data
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
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
from model_grids import BASELINE_PCA_GRID
from H_eval import get_final_metrics

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)
MODEL_N_JOBS = int(os.getenv("MODEL_N_JOBS", "-1"))

# Cache files produced/consumed by this script.
BASE_CACHE_FILES = [
    'base_baseline_pca_cv.pkl',
    'base_baseline_pca_dow_cv.pkl',
    'base_ridge_cv.pkl',
    'base_lasso_cv.pkl',
    'base_ridge_pca_cv.pkl',
    'base_lasso_pca_cv.pkl',
    'base_ridge_dow_cv.pkl',
    'base_lasso_dow_cv.pkl',
    'base_ridge_pca_dow_cv.pkl',
    'base_lasso_pca_dow_cv.pkl',
    'base_raw_diagnostics.pkl',
    'base_pca_diagnostics.pkl',
]

def clear_base_caches():
    removed = 0
    for filename in BASE_CACHE_FILES:
        path = os.path.join(CACHE_DIR, filename)
        if os.path.exists(path):
            os.remove(path)
            removed += 1
    print(f"Cleared {removed} base cache file(s) from {CACHE_DIR}.")

def to_binary_class(y):
    return (y >= 0).astype(int)

def _select_pca_n_components_best_cv(grid_results):
    """Choose PCA n_components by best mean CV score."""
    return max(grid_results, key=lambda r: r['mean_cv_score'])

def _select_c_1se_from_logregcv(cv_clf):
    scores = np.array(list(cv_clf.scores_.values())[0])
    if scores.ndim == 3:
        scores = scores[:, :, 0]
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
            clf = LogisticRegression(
                # Use sklearn's current single-stage regularization API:
                # l1_ratio=0 for ridge, l1_ratio=1 for lasso. This avoids the
                # deprecated penalty=... path that triggered warnings because
                # penalty='l1'/'l2' was being combined with the default l1_ratio=0.
                C=c_val, l1_ratio=l1_ratio, solver=solver,
                class_weight='balanced',
                random_state=1, max_iter=500, tol=1e-2
            )
            clf.fit(X_fold, y_fold)
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
        clf = LogisticRegression(
            C=c_val, l1_ratio=l1_ratio, solver=solver,
            class_weight='balanced',
            random_state=1, max_iter=500, tol=1e-2
        )
        clf.fit(X_train, y_train)
        train_errors.append(1 - clf.score(X_train, y_train))
        test_errors.append(1 - clf.score(X_test, y_test))
    return {
        'cs': np.array(c_grid),
        'train_errors': np.array(train_errors),
        'test_errors': np.array(test_errors),
    }

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
    clf = LogisticRegression(
        C=c_val, l1_ratio=l1_ratio, solver=solver,
        class_weight='balanced',
        random_state=1, max_iter=500, tol=1e-2
    )
    clf.fit(X_train, y_train)
    return 1 - clf.score(X_train, y_train), 1 - clf.score(X_test, y_test)

if __name__ == "__main__":
    # First run: keep True so the script generates the model and diagnostics caches.
    # False reuses only the diagnostics .pkl files for the curve plots; the fitted
    # models in this script are still retrained below.
    RETRAIN_ALL = True
    
    if RETRAIN_ALL:
        clear_base_caches()

    # ------- Load and preprocess data -------
    # Set testing=True for the 2-year dataset; False for the full 8-year dataset.
    TESTING = False
    DATA = import_data(testing=TESTING, extra_features=False, cluster=False, n_clusters=100, corr_threshold=0.95, corr_level=0)
    X, y_regression = clean_data(*DATA, raw=True, extra_features=False)

    # Flatten multi-level columns to single strings: "Close_AAPL", "Volume_MSFT", etc.
    X.columns = [f"{metric}_{ticker}" for metric, _, ticker in X.columns]
    print(f"Feature matrix shape: {X.shape[0]} rows, {X.shape[1]} columns.")

    y_classification = to_binary_class(y_regression)
    print(f"Final shape — X: {X.shape}, y: {y_classification.shape}")

    # ------- Train/test split (80/20, no shuffle — time series order must be preserved) -------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_classification, test_size=0.2, shuffle=False
    )

    # Previous temporary change used `KFold(n_splits=5, shuffle=False)`.
    tscv = TimeSeriesSplit(n_splits=5)

    # ===================================================================
    # PCA PRE-PROCESSING WITH GRID SEARCH FOR BASELINE
    # ===================================================================
    print("\n========== Grid Search for Optimal PCA Components (Baseline) ===========")
    
    # Fit scaler
    scaler_pca = StandardScaler()
    X_train_sc  = scaler_pca.fit_transform(X_train)
    X_test_sc   = scaler_pca.transform(X_test)
    
    # Grid search over different n_components
    n_components_grid = BASELINE_PCA_GRID
    grid_search_results = []
    
    print(f"Testing n_components: {n_components_grid}")
    for n_comp in n_components_grid:
        pca_temp = PCA(n_components=n_comp)
        X_pca_temp = pca_temp.fit_transform(X_train_sc)
        
        # Cross-validate baseline model
        # Use C=np.inf for the unpenalized baseline under sklearn's new API.
        baseline_temp = LogisticRegression(C=np.inf, solver="lbfgs", 
                                          class_weight='balanced',
                                          random_state=1, max_iter=500, tol=1e-2)
        scores = cross_val_score(
            baseline_temp, X_pca_temp, y_train, cv=tscv, n_jobs=MODEL_N_JOBS, scoring='balanced_accuracy'
        )
        mean_score = scores.mean()
        std_score = scores.std()
        
        # Store results
        grid_search_results.append({
            'n_components': n_comp,
            'n_components_value': X_pca_temp.shape[1],
            'cv_scores': scores,
            'mean_cv_score': mean_score,
            'std_cv_score': std_score
        })
        
        print(f"  n_components={n_comp} ({X_pca_temp.shape[1]} components): CV Balanced Accuracy = {mean_score:.4f} ± {std_score:.4f}")
        
    best_pca = _select_pca_n_components_best_cv(grid_search_results)
    best_n_comp = best_pca['n_components']
    best_n_comp_value = best_pca['n_components_value']
    best_score = best_pca['mean_cv_score']
    print(
        f"\nBest CV n_components: {best_n_comp} ({best_n_comp_value} components), "
        f"CV={best_score:.4f}±{best_pca['std_cv_score']:.4f}"
    )
    
    # Fit final PCA with best n_components
    pca = PCA(n_components=best_n_comp)
    X_train_pca = pca.fit_transform(X_train_sc)
    X_test_pca  = pca.transform(X_test_sc)
    n_components_raw = X_train_pca.shape[1]
    print(f"Final PCA: {X_train.shape[1]} features → {n_components_raw} components ({best_n_comp*100:.0f}% variance)")

    # ------- Baseline: plain Logistic Regression (no regularization) after PCA -------
    print("\n========== BASELINE: Plain Logistic Regression (No Regularization) + PCA ===========")
    baseline_clf = LogisticRegression(C=np.inf, solver="lbfgs", random_state=1,
                                      class_weight='balanced',
                                      max_iter=500, tol=1e-2)
    baseline_clf.fit(X_train_pca, y_train)
    
    # Cache the model and grid search results
    _baseline_pca_cache = os.path.join(CACHE_DIR, 'base_baseline_pca_cv.pkl')
    cache_data = {
        'model': baseline_clf,
        'pca': pca,
        'scaler': scaler_pca,
        'best_n_comp': best_n_comp,
        'best_cv_score': best_score,
        'grid_search_results': grid_search_results,
        'n_components_value': n_components_raw
    }
    joblib.dump(cache_data, _baseline_pca_cache)
    print(f"Baseline+PCA model cached to: {_baseline_pca_cache}")

    # ------- Ridge (L2): LogisticRegressionCV — stores per-fold scores for bias-variance plot -------
    print("\n========== RIDGE (L2) Logistic Regression CV ==========")
    pipeline_ridge = Pipeline([
        ('scaler',     StandardScaler()),
        ('classifier', LogisticRegressionCV(
            # Single-stage ridge logistic regression: l1_ratio=0 replaces penalty='l2'.
            Cs=20, cv=tscv, l1_ratios=[0], solver='saga',
            class_weight='balanced',
            random_state=1, n_jobs=MODEL_N_JOBS, max_iter=500, tol=1e-2, scoring='balanced_accuracy'
        ))
    ])
    pipeline_ridge.fit(X_train, y_train)
    _ridge_cache = os.path.join(CACHE_DIR, 'base_ridge_cv.pkl')
    joblib.dump(pipeline_ridge, _ridge_cache)
    
    ridge_cv = pipeline_ridge.named_steps['classifier']
    ridge_c_1se, ridge_c_best, _ = _select_c_1se_from_logregcv(ridge_cv)
    print(f"Best C by mean CV (Ridge): {ridge_c_best:.6f}")
    print(f"1SE-selected C (Ridge):    {ridge_c_1se:.6f}")
    pipeline_ridge_1se = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            C=ridge_c_1se, l1_ratio=0, solver='saga',
            class_weight='balanced',
            random_state=1, max_iter=500, tol=1e-2
        ))
    ])
    pipeline_ridge_1se.fit(X_train, y_train)

    # ------- LASSO (L1): LogisticRegressionCV — stores per-fold scores for bias-variance plot -------
    print("\n========== LASSO (L1) Logistic Regression CV ==========")
    pipeline_lasso = Pipeline([
        ('scaler',     StandardScaler()),
        ('classifier', LogisticRegressionCV(
            # Single-stage lasso logistic regression: l1_ratio=1 replaces penalty='l1'.
            Cs=np.logspace(-6, 4, 20), cv=tscv, l1_ratios=[1], solver='saga',
            class_weight='balanced',
            random_state=1, n_jobs=MODEL_N_JOBS, max_iter=500, tol=1e-2, scoring='balanced_accuracy'
        ))
    ])
    pipeline_lasso.fit(X_train, y_train)
    _lasso_cache = os.path.join(CACHE_DIR, 'base_lasso_cv.pkl')
    joblib.dump(pipeline_lasso, _lasso_cache)
    
    lasso_cv = pipeline_lasso.named_steps['classifier']
    lasso_c_1se, lasso_c_best, _ = _select_c_1se_from_logregcv(lasso_cv)
    print(f"Best C by mean CV (LASSO): {lasso_c_best:.6f}")
    print(f"1SE-selected C (LASSO):    {lasso_c_1se:.6f}")
    pipeline_lasso_1se = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            C=lasso_c_1se, l1_ratio=1, solver='saga',
            class_weight='balanced',
            random_state=1, max_iter=500, tol=1e-2
        ))
    ])
    pipeline_lasso_1se.fit(X_train, y_train)

    # ------- Bias-Variance Tradeoff Plot (Ridge + LASSO) -------
    print("\n========== Generating Bias-Variance Tradeoff Plot ==========")
    raw_diag_cache = os.path.join(CACHE_DIR, 'base_raw_diagnostics.pkl')
    if (not RETRAIN_ALL) and os.path.exists(raw_diag_cache):
        raw_diagnostics = joblib.load(raw_diag_cache)
    else:
        raw_scaler = StandardScaler()
        X_train_raw_sc = raw_scaler.fit_transform(X_train)
        X_test_raw_sc = raw_scaler.transform(X_test)
        raw_diagnostics = {
            'ridge_bv': _compute_bv_curves(ridge_cv, X_train_raw_sc, y_train, tscv, 0, 'saga'),
            'lasso_bv': _compute_bv_curves(lasso_cv, X_train_raw_sc, y_train, tscv, 1, 'saga'),
            'ridge_direct': _compute_direct_split_errors(
                X_train_raw_sc, y_train, X_test_raw_sc, y_test,
                ridge_cv.Cs_,
                0, 'saga'
            ),
            'lasso_direct': _compute_direct_split_errors(
                X_train_raw_sc, y_train, X_test_raw_sc, y_test,
                lasso_cv.Cs_,
                1, 'saga'
            ),
        }
        joblib.dump(raw_diagnostics, raw_diag_cache)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Bias-Variance Tradeoff - LR — Raw OHLCV Features\n(Train/CV Balanced Error per Regularization Strength)',
                 fontsize=13, fontweight='bold')

    for ax, (label, diag_key, c_1se) in zip(axes, [
        ('Ridge (L2)', 'ridge_bv', ridge_c_1se),
        ('LASSO (L1)', 'lasso_bv', lasso_c_1se),
    ]):
        diag = raw_diagnostics[diag_key]
        cs = diag['cs']
        train_bal_mean = diag['train_bal_err_mean']
        train_bal_std = diag['train_bal_err_std']
        cv_bal_mean = diag['cv_bal_err_mean']
        cv_bal_std = diag['cv_bal_err_std']
        best_idx = int(np.argmin(cv_bal_mean))
        ax.semilogx(cs, train_bal_mean, marker='o', color='steelblue', linewidth=1.8, label='CV Train balanced error')
        ax.semilogx(cs, cv_bal_mean, marker='s', color='darkorange', linewidth=1.8, label='CV Test balanced error')
        ax.fill_between(
            cs,
            np.clip(train_bal_mean - train_bal_std, 0.0, 1.0),
            np.clip(train_bal_mean + train_bal_std, 0.0, 1.0),
            alpha=0.15,
            color='steelblue',
            label='CV Train balanced error ±1 SD'
        )
        ax.fill_between(
            cs,
            np.clip(cv_bal_mean - cv_bal_std, 0.0, 1.0),
            np.clip(cv_bal_mean + cv_bal_std, 0.0, 1.0),
            alpha=0.15,
            color='darkorange',
            label='CV Test balanced error ±1 SD'
        )
        _highlight_selected_value(
            ax, cs, cv_bal_mean, best_idx,
            label_prefix='Value at best CV balanced error'
        )
        ax.axvline(c_1se, color='red', linestyle='--', linewidth=1.5,
                   label=f'1SE-selected C = {c_1se:.4f}')
        ax.set_title(f'{label} - LR — Bias-Variance Tradeoff (Balanced Error)')
        ax.set_xlabel('C  (Inverse Regularization Strength)\n'
                      '← High Regularization, Simpler Model      '
                      'Low Regularization, More Complex →')
        ax.set_ylabel('Balanced Error (1 - balanced accuracy)')
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, '8yrs_1SE_base_logistic_bias_variance.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {os.path.abspath(out_path)}")
    plt.close()

    # ------- Train vs Test Error Plot (direct, no CV averaging) -------
    print("\n========== Generating Train vs Test Error Plot (Direct Split) ==========")

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle('Over/Underfitting Analysis - LR — Raw OHLCV Features\n(Direct Train/Test Split, No CV)',
                  fontsize=13, fontweight='bold')

    for ax, (label, diag_key, guide_key, color) in zip(axes2, [
        ('Ridge (L2)', 'ridge_direct', 'ridge_bv', 'darkorange'),
        ('LASSO (L1)', 'lasso_direct', 'lasso_bv', 'seagreen'),
    ]):
        diag = raw_diagnostics[diag_key]
        guide_diag = raw_diagnostics[guide_key]
        best_idx = int(np.argmin(guide_diag['cv_bal_err_mean']))
        selected_c = float(guide_diag['cs'][best_idx])
        _, best_test_error = _compute_single_direct_split_error(
            X_train_raw_sc, y_train, X_test_raw_sc, y_test,
            selected_c, 0 if 'ridge' in diag_key else 1, 'saga'
        )
        ax.semilogx(diag['cs'], diag['train_errors'], marker='o', color='steelblue',
                    linewidth=2, label='Train error')
        ax.semilogx(diag['cs'], diag['test_errors'], marker='s', color=color,
                    linewidth=2, label='Test error')
        ax.scatter(
            [selected_c], [best_test_error],
            color='gold', edgecolor='black', s=90, zorder=6,
            label='Value at best CV balanced error'
        )
        one_se_c = ridge_c_1se if 'ridge' in diag_key else lasso_c_1se
        ax.axvline(one_se_c, color='red', linestyle='--', linewidth=1.5,
                   label=f'1SE-selected C = {one_se_c:.4f}')
        ax.set_title(f'{label} - LR — Train vs Test Error (Plain Error)')
        ax.set_xlabel('C  (Inverse Regularization Strength)\n'
                      '← High Regularization, Simpler Model      '
                      'Low Regularization, More Complex →')
        ax.set_ylabel('Plain Error (1 - accuracy)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path2 = os.path.join(output_dir, '8yrs_1SE_base_logistic_train_test.png')
    plt.savefig(out_path2, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {os.path.abspath(out_path2)}")
    plt.close()


    # ------- Ridge CV after PCA -------
    print("\n========== RIDGE (L2) + PCA — LogisticRegressionCV ==========")
    clf_ridge_pca = LogisticRegressionCV(
        Cs=20, cv=tscv, l1_ratios=[0], solver='saga',
        class_weight='balanced',
        random_state=1, n_jobs=MODEL_N_JOBS, max_iter=500, tol=1e-2, scoring='balanced_accuracy')
    clf_ridge_pca.fit(X_train_pca, y_train)
    _ridge_pca_cache = os.path.join(CACHE_DIR, 'base_ridge_pca_cv.pkl')
    joblib.dump(clf_ridge_pca, _ridge_pca_cache)
    ridge_pca_c_1se, ridge_pca_c_best, _ = _select_c_1se_from_logregcv(clf_ridge_pca)
    print(f"Best C by mean CV (Ridge+PCA): {ridge_pca_c_best:.6f}")
    print(f"1SE-selected C (Ridge+PCA):    {ridge_pca_c_1se:.6f}")
    clf_ridge_pca_1se = LogisticRegression(
        C=ridge_pca_c_1se, l1_ratio=0, solver='saga',
        class_weight='balanced',
        random_state=1, max_iter=500, tol=1e-2
    )
    clf_ridge_pca_1se.fit(X_train_pca, y_train)

    # ------- LASSO CV after PCA -------
    print("\n========== LASSO (L1) + PCA — LogisticRegressionCV ==========")
    clf_lasso_pca = LogisticRegressionCV(
        Cs=np.logspace(-6, 4, 20), cv=tscv, l1_ratios=[1], solver='saga',
        class_weight='balanced',
        random_state=1, n_jobs=MODEL_N_JOBS, max_iter=500, tol=1e-2, scoring='balanced_accuracy')
    clf_lasso_pca.fit(X_train_pca, y_train)
    _lasso_pca_cache = os.path.join(CACHE_DIR, 'base_lasso_pca_cv.pkl')
    joblib.dump(clf_lasso_pca, _lasso_pca_cache)
    lasso_pca_c_1se, lasso_pca_c_best, _ = _select_c_1se_from_logregcv(clf_lasso_pca)
    print(f"Best C by mean CV (LASSO+PCA): {lasso_pca_c_best:.6f}")
    print(f"1SE-selected C (LASSO+PCA):    {lasso_pca_c_1se:.6f}")
    clf_lasso_pca_1se = LogisticRegression(
        C=lasso_pca_c_1se, l1_ratio=1, solver='saga',
        class_weight='balanced',
        random_state=1, max_iter=500, tol=1e-2
    )
    clf_lasso_pca_1se.fit(X_train_pca, y_train)

    # ------- Bias-Variance Tradeoff Plot (PCA + Ridge/LASSO) -------
    print("\n========== Generating PCA Bias-Variance Tradeoff Plot ==========")
    pca_diag_cache = os.path.join(CACHE_DIR, 'base_pca_diagnostics.pkl')
    if (not RETRAIN_ALL) and os.path.exists(pca_diag_cache):
        pca_diagnostics = joblib.load(pca_diag_cache)
    else:
        pca_diagnostics = {
            'ridge_bv': _compute_bv_curves(clf_ridge_pca, X_train_pca, y_train, tscv, 0, 'saga'),
            'lasso_bv': _compute_bv_curves(clf_lasso_pca, X_train_pca, y_train, tscv, 1, 'saga'),
            'ridge_direct': _compute_direct_split_errors(
                X_train_pca, y_train, X_test_pca, y_test,
                clf_ridge_pca.Cs_,
                0, 'saga'
            ),
            'lasso_direct': _compute_direct_split_errors(
                X_train_pca, y_train, X_test_pca, y_test,
                clf_lasso_pca.Cs_,
                1, 'saga'
            ),
        }
        joblib.dump(pca_diagnostics, pca_diag_cache)

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    fig3.suptitle(f'Bias-Variance Tradeoff - LR — PCA Features ({n_components_raw} comps, {best_n_comp*100:.0f}% variance)\n'
                  '(Train/CV Balanced Error per Regularization Strength)',
                  fontsize=13, fontweight='bold')

    for ax, (label, diag_key, c_1se) in zip(axes3, [
        ('Ridge (L2)', 'ridge_bv', ridge_pca_c_1se),
        ('LASSO (L1)', 'lasso_bv', lasso_pca_c_1se),
    ]):
        diag = pca_diagnostics[diag_key]
        cs = diag['cs']
        train_bal_mean = diag['train_bal_err_mean']
        train_bal_std = diag['train_bal_err_std']
        cv_bal_mean = diag['cv_bal_err_mean']
        cv_bal_std = diag['cv_bal_err_std']
        best_idx = int(np.argmin(cv_bal_mean))
        ax.semilogx(cs, train_bal_mean, marker='o', color='steelblue', linewidth=1.8, label='CV Train balanced error')
        ax.semilogx(cs, cv_bal_mean, marker='s', color='darkorange', linewidth=1.8, label='CV Test balanced error')
        ax.fill_between(
            cs,
            np.clip(train_bal_mean - train_bal_std, 0.0, 1.0),
            np.clip(train_bal_mean + train_bal_std, 0.0, 1.0),
            alpha=0.15,
            color='steelblue',
            label='CV Train balanced error ±1 SD'
        )
        ax.fill_between(
            cs,
            np.clip(cv_bal_mean - cv_bal_std, 0.0, 1.0),
            np.clip(cv_bal_mean + cv_bal_std, 0.0, 1.0),
            alpha=0.15,
            color='darkorange',
            label='CV Test balanced error ±1 SD'
        )
        _highlight_selected_value(
            ax, cs, cv_bal_mean, best_idx,
            label_prefix='Value at best CV balanced error'
        )
        ax.axvline(c_1se, color='red', linestyle='--', linewidth=1.5,
                   label=f'1SE-selected C = {c_1se:.4f}')
        ax.set_title(f'{label} - LR — Bias-Variance Tradeoff (PCA, Balanced Error)')
        ax.set_xlabel('C  (Inverse Regularization Strength)\n'
                      '← High Regularization, Simpler Model      '
                      'Low Regularization, More Complex →')
        ax.set_ylabel('Balanced Error (1 - balanced accuracy)')
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path3 = os.path.join(output_dir, '8yrs_1SE_base_logistic_pca_bias_variance.png')
    plt.savefig(out_path3, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {os.path.abspath(out_path3)}")
    plt.close()

    # ------- Train vs Test Error Plot after PCA (direct split) -------
    print("\n========== Generating PCA Train vs Test Error Plot (Direct Split) ==========")

    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))
    fig4.suptitle(f'Over/Underfitting Analysis - LR — PCA Features ({n_components_raw} comps, {best_n_comp*100:.0f}% variance)\n'
                  '(Direct Train/Test Split, No CV)',
                  fontsize=13, fontweight='bold')

    for ax, (label, diag_key, guide_key, color) in zip(axes4, [
        ('Ridge (L2)', 'ridge_direct', 'ridge_bv', 'darkorange'),
        ('LASSO (L1)', 'lasso_direct', 'lasso_bv', 'seagreen'),
    ]):
        diag = pca_diagnostics[diag_key]
        guide_diag = pca_diagnostics[guide_key]
        best_idx = int(np.argmin(guide_diag['cv_bal_err_mean']))
        selected_c = float(guide_diag['cs'][best_idx])
        _, best_test_error = _compute_single_direct_split_error(
            X_train_pca, y_train, X_test_pca, y_test,
            selected_c, 0 if 'ridge' in diag_key else 1, 'saga'
        )
        ax.semilogx(diag['cs'], diag['train_errors'], marker='o', color='steelblue',
                    linewidth=2, label='Train error')
        ax.semilogx(diag['cs'], diag['test_errors'], marker='s', color=color,
                    linewidth=2, label='Test error')
        ax.scatter(
            [selected_c], [best_test_error],
            color='gold', edgecolor='black', s=90, zorder=6,
            label='Value at best CV balanced error'
        )
        one_se_c = ridge_pca_c_1se if 'ridge' in diag_key else lasso_pca_c_1se
        ax.axvline(one_se_c, color='red', linestyle='--', linewidth=1.5,
                   label=f'1SE-selected C = {one_se_c:.4f}')
        ax.set_title(f'{label} - LR — Train vs Test Error (PCA, Plain Error)')
        ax.set_xlabel('C  (Inverse Regularization Strength)\n'
                      '← High Regularization, Simpler Model      '
                      'Low Regularization, More Complex →')
        ax.set_ylabel('Plain Error (1 - accuracy)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path4 = os.path.join(output_dir, '8yrs_1SE_base_logistic_pca_train_test.png')
    plt.savefig(out_path4, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {os.path.abspath(out_path4)}")
    plt.close()

    # ===================================================================
    # MODEL COMPARISON TABLE
    # ===================================================================
    print("\n========== Model Comparison Table ==========")

    def _eval_row(name, model, X_tr, y_tr, X_te, y_te, n_splits, best_c=None):
        shared = get_final_metrics(
            model, X_tr, y_tr, X_te, y_te, n_splits=n_splits, label=name
        )
        preds = model.predict(X_te)
        is_degenerate = (
            best_c is not None
            and np.isclose(best_c, 1e-6)
            and np.unique(preds).size == 1
        )
        return {
            'Model':           name,
            'Best C':          f'{best_c:.6f}' if best_c is not None else 'N/A',
            'Avg CV Train Plain Acc':   shared['train_avg_accuracy'],
            'CV Train Plain Acc SD':    shared['train_std_accuracy'],
            'Avg CV Validation Plain Acc': shared['validation_avg_accuracy'],
            'CV Acc SD':                   shared['cv_test_sd_error'],
            'Test Acc':                    shared['test_split_accuracy'],
            'Precision':       shared['test_precision'],
            'Recall':          shared['test_recall'],
            'Specificity':     shared['test_specificity'],
            'F1':              shared['test_f1'],
            'ROC-AUC':         shared['test_roc_auc_macro'],
            'Degenerate':      is_degenerate,
        }

    rows = [
        _eval_row(f'Base+PCA ({n_components_raw}, {best_n_comp*100:.0f}%)', baseline_clf, X_train_pca, y_train, X_test_pca, y_test, tscv.n_splits, best_c=None),
        _eval_row('Ridge (raw)', pipeline_ridge_1se, X_train, y_train, X_test, y_test, tscv.n_splits, best_c=ridge_c_1se),
        _eval_row('LASSO (raw)', pipeline_lasso_1se, X_train, y_train, X_test, y_test, tscv.n_splits, best_c=lasso_c_1se),
        _eval_row(f'Ridge+PCA ({n_components_raw})', clf_ridge_pca_1se, X_train_pca, y_train, X_test_pca, y_test, tscv.n_splits, best_c=ridge_pca_c_1se),
        _eval_row(f'LASSO+PCA ({n_components_raw})', clf_lasso_pca_1se, X_train_pca, y_train, X_test_pca, y_test, tscv.n_splits, best_c=lasso_pca_c_1se),
    ]

    comparison_df = pd.DataFrame(rows).set_index('Model')
    print(comparison_df.to_string())

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
        X_dow, y_classification, test_size=0.2, shuffle=False
    )
    print(f"Feature matrix with DOW: {X_dow.shape[1]} columns")

    # --- PCA + DOW ---
    print("\n========== Grid Search for Optimal PCA Components (DOW) ===========")
    
    # Grid search over different n_components for DOW features
    n_components_grid = BASELINE_PCA_GRID
    grid_search_results_dow = []
    
    scaler_pca_dow = StandardScaler()
    X_train_dow_sc  = scaler_pca_dow.fit_transform(X_train_dow)
    X_test_dow_sc   = scaler_pca_dow.transform(X_test_dow)
    
    print(f"Testing n_components: {n_components_grid}")
    for n_comp in n_components_grid:
        pca_temp = PCA(n_components=n_comp)
        X_pca_temp = pca_temp.fit_transform(X_train_dow_sc)
        
        # Cross-validate baseline model
        baseline_temp = LogisticRegression(C=np.inf, solver="lbfgs", 
                                          class_weight='balanced',
                                          random_state=1, max_iter=500, tol=1e-2)
        scores = cross_val_score(
            baseline_temp, X_pca_temp, y_train, cv=tscv, n_jobs=MODEL_N_JOBS, scoring='balanced_accuracy'
        )
        mean_score = scores.mean()
        std_score = scores.std()
        
        # Store results
        grid_search_results_dow.append({
            'n_components': n_comp,
            'n_components_value': X_pca_temp.shape[1],
            'cv_scores': scores,
            'mean_cv_score': mean_score,
            'std_cv_score': std_score
        })
        
        print(f"  n_components={n_comp} ({X_pca_temp.shape[1]} components): CV Balanced Accuracy = {mean_score:.4f} ± {std_score:.4f}")
        
    best_pca_dow = _select_pca_n_components_best_cv(grid_search_results_dow)
    best_n_comp_dow = best_pca_dow['n_components']
    best_n_comp_value_dow = best_pca_dow['n_components_value']
    best_score_dow = best_pca_dow['mean_cv_score']
    print(
        f"\nBest CV n_components (DOW): {best_n_comp_dow} ({best_n_comp_value_dow} components), "
        f"CV={best_score_dow:.4f}±{best_pca_dow['std_cv_score']:.4f}"
    )
    
    # Fit final PCA with best n_components
    pca_dow = PCA(n_components=best_n_comp_dow)
    X_train_dow_pca = pca_dow.fit_transform(X_train_dow_sc)
    X_test_dow_pca  = pca_dow.transform(X_test_dow_sc)
    n_components_dow = X_train_dow_pca.shape[1]
    print(f"Final PCA: {X_train_dow.shape[1]} features → {n_components_dow} components ({best_n_comp_dow*100:.0f}% variance)")

    # --- Baseline + DOW + PCA ---
    print("\n========== BASELINE (No Reg) + DOW + PCA ===========")
    baseline_dow_clf = LogisticRegression(C=np.inf, solver='lbfgs', random_state=1,
                                          class_weight='balanced',
                                          max_iter=500, tol=1e-2)
    baseline_dow_clf.fit(X_train_dow_pca, y_train)
    
    # Cache the model and grid search results
    _baseline_pca_dow_cache = os.path.join(CACHE_DIR, 'base_baseline_pca_dow_cv.pkl')
    cache_data = {
        'model': baseline_dow_clf,
        'pca': pca_dow,
        'scaler': scaler_pca_dow,
        'best_n_comp': best_n_comp_dow,
        'best_cv_score': best_score_dow,
        'grid_search_results': grid_search_results_dow,
        'n_components_value': n_components_dow
    }
    joblib.dump(cache_data, _baseline_pca_dow_cache)
    print(f"Baseline+PCA+DOW model cached to: {_baseline_pca_dow_cache}")

    # --- Ridge CV + DOW ---
    print("\n========== RIDGE CV + DOW ==========")
    pipeline_ridge_dow = Pipeline([
        ('scaler',     StandardScaler()),
        ('classifier', LogisticRegressionCV(
            Cs=20, cv=tscv, l1_ratios=[0], solver='saga',
            class_weight='balanced',
            random_state=1, n_jobs=MODEL_N_JOBS, max_iter=500, tol=1e-2, scoring='balanced_accuracy'
        ))
    ])
    pipeline_ridge_dow.fit(X_train_dow, y_train)
    _ridge_dow_cache = os.path.join(CACHE_DIR, 'base_ridge_dow_cv.pkl')
    joblib.dump(pipeline_ridge_dow, _ridge_dow_cache)
    ridge_dow_cv = pipeline_ridge_dow.named_steps['classifier']
    ridge_dow_c_1se, ridge_dow_c_best, _ = _select_c_1se_from_logregcv(ridge_dow_cv)
    print(f"Best C by mean CV (Ridge+DOW): {ridge_dow_c_best:.6f}")
    print(f"1SE-selected C (Ridge+DOW):    {ridge_dow_c_1se:.6f}")
    pipeline_ridge_dow_1se = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            C=ridge_dow_c_1se, l1_ratio=0, solver='saga',
            class_weight='balanced',
            random_state=1, max_iter=500, tol=1e-2
        ))
    ])
    pipeline_ridge_dow_1se.fit(X_train_dow, y_train)

    # --- LASSO CV + DOW ---
    print("\n========== LASSO CV + DOW ==========")
    pipeline_lasso_dow = Pipeline([
        ('scaler',     StandardScaler()),
        ('classifier', LogisticRegressionCV(
            Cs=np.logspace(-6, 4, 20), cv=tscv, l1_ratios=[1], solver='saga',
            class_weight='balanced',
            random_state=1, n_jobs=MODEL_N_JOBS, max_iter=500, tol=1e-2, scoring='balanced_accuracy'
        ))
    ])
    pipeline_lasso_dow.fit(X_train_dow, y_train)
    _lasso_dow_cache = os.path.join(CACHE_DIR, 'base_lasso_dow_cv.pkl')
    joblib.dump(pipeline_lasso_dow, _lasso_dow_cache)
    lasso_dow_cv = pipeline_lasso_dow.named_steps['classifier']
    lasso_dow_c_1se, lasso_dow_c_best, _ = _select_c_1se_from_logregcv(lasso_dow_cv)
    print(f"Best C by mean CV (LASSO+DOW): {lasso_dow_c_best:.6f}")
    print(f"1SE-selected C (LASSO+DOW):    {lasso_dow_c_1se:.6f}")
    pipeline_lasso_dow_1se = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            C=lasso_dow_c_1se, l1_ratio=1, solver='saga',
            class_weight='balanced',
            random_state=1, max_iter=500, tol=1e-2
        ))
    ])
    pipeline_lasso_dow_1se.fit(X_train_dow, y_train)


    # --- Ridge CV + PCA + DOW ---
    print("\n========== RIDGE CV + PCA + DOW ==========")
    clf_ridge_pca_dow = LogisticRegressionCV(
        Cs=20, cv=tscv, l1_ratios=[0], solver='saga',
        class_weight='balanced',
        random_state=1, n_jobs=MODEL_N_JOBS, max_iter=500, tol=1e-2, scoring='balanced_accuracy')
    clf_ridge_pca_dow.fit(X_train_dow_pca, y_train)
    _ridge_pca_dow_cache = os.path.join(CACHE_DIR, 'base_ridge_pca_dow_cv.pkl')
    joblib.dump(clf_ridge_pca_dow, _ridge_pca_dow_cache)
    ridge_pca_dow_c_1se, ridge_pca_dow_c_best, _ = _select_c_1se_from_logregcv(clf_ridge_pca_dow)
    print(f"Best C by mean CV (Ridge+PCA+DOW): {ridge_pca_dow_c_best:.6f}")
    print(f"1SE-selected C (Ridge+PCA+DOW):    {ridge_pca_dow_c_1se:.6f}")
    clf_ridge_pca_dow_1se = LogisticRegression(
        C=ridge_pca_dow_c_1se, l1_ratio=0, solver='saga',
        class_weight='balanced',
        random_state=1, max_iter=500, tol=1e-2
    )
    clf_ridge_pca_dow_1se.fit(X_train_dow_pca, y_train)

    # --- LASSO CV + PCA + DOW ---
    print("\n========== LASSO CV + PCA + DOW ==========")
    clf_lasso_pca_dow = LogisticRegressionCV(
        Cs=np.logspace(-6, 4, 20), cv=tscv, l1_ratios=[1], solver='saga',
        class_weight='balanced',
        random_state=1, n_jobs=MODEL_N_JOBS, max_iter=500, tol=1e-2, scoring='balanced_accuracy')
    clf_lasso_pca_dow.fit(X_train_dow_pca, y_train)
    _lasso_pca_dow_cache = os.path.join(CACHE_DIR, 'base_lasso_pca_dow_cv.pkl')
    joblib.dump(clf_lasso_pca_dow, _lasso_pca_dow_cache)
    lasso_pca_dow_c_1se, lasso_pca_dow_c_best, _ = _select_c_1se_from_logregcv(clf_lasso_pca_dow)
    print(f"Best C by mean CV (LASSO+PCA+DOW): {lasso_pca_dow_c_best:.6f}")
    print(f"1SE-selected C (LASSO+PCA+DOW):    {lasso_pca_dow_c_1se:.6f}")
    clf_lasso_pca_dow_1se = LogisticRegression(
        C=lasso_pca_dow_c_1se, l1_ratio=1, solver='saga',
        class_weight='balanced',
        random_state=1, max_iter=500, tol=1e-2
    )
    clf_lasso_pca_dow_1se.fit(X_train_dow_pca, y_train)

    # --- DOW rows using same _metrics helper ---
    rows_dow = [
        _eval_row(f'Base+DOW+PCA ({n_components_dow}, {best_n_comp_dow*100:.0f}%)', baseline_dow_clf, X_train_dow_pca, y_train, X_test_dow_pca, y_test, tscv.n_splits, best_c=None),
        _eval_row('Ridge+DOW', pipeline_ridge_dow_1se, X_train_dow, y_train, X_test_dow, y_test, tscv.n_splits, best_c=ridge_dow_c_1se),
        _eval_row('LASSO+DOW', pipeline_lasso_dow_1se, X_train_dow, y_train, X_test_dow, y_test, tscv.n_splits, best_c=lasso_dow_c_1se),
        _eval_row(f'Ridge+PCA+DOW ({n_components_dow})', clf_ridge_pca_dow_1se, X_train_dow_pca, y_train, X_test_dow_pca, y_test, tscv.n_splits, best_c=ridge_pca_dow_c_1se),
        _eval_row(f'LASSO+PCA+DOW ({n_components_dow})', clf_lasso_pca_dow_1se, X_train_dow_pca, y_train, X_test_dow_pca, y_test, tscv.n_splits, best_c=lasso_pca_dow_c_1se),
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

    # ===================================================================
    # FINAL COMBINED TABLE: raw | raw+DOW
    # ===================================================================
    full_df = combined_df.copy()
    full_df.index.name = 'Model'
    # Keep only the reporting columns used in slides/tables.
    keep_cols = ['Test Acc', 'Precision', 'Recall', 'Specificity', 'F1', 'ROC-AUC', 'CV Acc SD']
    full_df = full_df[keep_cols]

    print("\n===== Full Comparison Table =====")
    print(full_df.to_string())

    tex_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'output')
    os.makedirs(tex_output_dir, exist_ok=True)

    # Build LaTeX manually with \midrule separating the three groups
    tex_path = os.path.join(tex_output_dir, '8yrs_1SE_base_logistic_comparison.tex')
    col_fmt  = 'l' + 'r' * len(full_df.columns)
    col_header = ' & '.join(['Model'] + list(full_df.columns)) + r' \\'
    baseline_note = (
        r'Base = baseline logistic regression without regularization. '
        r'Test Acc = plain hold-out accuracy on the final 20\% test split. '
        r'All reported CV/train/test accuracy columns in this table use plain accuracy after hyperparameters were selected by CV balanced accuracy. '
        r'Recall = positive-class sensitivity.'
    )
    lasso_note = (
        r'$^\dagger$ Degenerate classifier: optimal $C = 10^{-6}$ shrinks all '
        r'coefficients to zero; model predicts majority class for every observation '
        r'(Recall = 1.0, Precision $\approx$ base rate).'
    )

    def _latex_escape(text):
        return (str(text)
                .replace('\\', r'\textbackslash{}')
                .replace('&', r'\&')
                .replace('%', r'\%')
                .replace('_', r'\_')
                .replace('#', r'\#')
                .replace('$', r'\$')
                .replace('{', r'\{')
                .replace('}', r'\}'))

    degenerate_models = set(combined_df.index[combined_df['Degenerate']])

    def _row(name, vals):
        dagger = r'$^\dagger$' if name in degenerate_models else ''
        formatted_vals = [f'{v:.3f}' for v in vals]
        model_name = _latex_escape(name)
        return model_name + dagger + ' & ' + ' & '.join(formatted_vals) + r' \\'

    with open(tex_path, 'w') as f:
        f.write(r'\begin{table}[htbp]' + '\n')
        f.write(r'\centering' + '\n')
        f.write(r'\caption{Logistic Regression Model Comparison: '
                r'Raw OHLCV vs Raw OHLCV + Day-of-Week}' + '\n')
        f.write(r'\label{tab:base_logistic_comparison}' + '\n')
        f.write(r'\begin{tabular}{' + col_fmt + '}\n')
        f.write(r'\toprule' + '\n')
        f.write(col_header + '\n')
        f.write(r'\midrule' + '\n')
        # Group 1: raw
        for name, row in full_df.loc[comparison_df.index].iterrows():
            f.write(_row(name, row.values) + '\n')
        f.write(r'\midrule' + '\n')
        # Group 2: DOW
        for name, row in full_df.loc[dow_df.index].iterrows():
            f.write(_row(name, row.values) + '\n')
        f.write(r'\bottomrule' + '\n')
        f.write(r'\end{tabular}' + '\n')
        f.write(r'\par\smallskip' + '\n')
        f.write(r'\footnotesize ' + baseline_note + '\n')
        if degenerate_models:
            f.write(r'\\' + '\n')
            f.write(r'\footnotesize ' + lasso_note + '\n')
        f.write(r'\end{table}' + '\n')

    print(f"LaTeX table saved to:           {os.path.abspath(tex_path)}")

    print("\n" + "="*70)
    print("All models trained and cached successfully!")
    print("="*70)


# ===============================================================================
# HELPER FUNCTIONS FOR READING CACHED RESULTS (FOR PLOTTING & ANALYSIS)
# ===============================================================================

def load_baseline_pca_results(cache_dir=None):
    """
    Load the cached baseline PCA grid search results for analysis and plotting.
    
    Returns:
        dict: Contains model, PCA transformer, scaler, best n_components, 
              CV scores, and full grid search results
              
    Example usage:
        >>> cached_data = load_baseline_pca_results()
        >>> print(f"Best variance threshold: {cached_data['best_n_comp']}")
        >>> print(f"Best CV score: {cached_data['best_cv_score']:.4f}")
        >>> 
        >>> # Plot grid search results
        >>> import matplotlib.pyplot as plt
        >>> results = cached_data['grid_search_results']
        >>> n_comps = [r['n_components'] for r in results]
        >>> means = [r['mean_cv_score'] for r in results]
        >>> stds = [r['std_cv_score'] for r in results]
        >>> plt.errorbar(n_comps, means, yerr=stds, marker='o')
        >>> plt.xlabel('Variance Threshold')
        >>> plt.ylabel('CV Accuracy')
        >>> plt.show()
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cache')
    
    cache_path = os.path.join(cache_dir, 'base_baseline_pca_cv.pkl')
    
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache file not found: {cache_path}. Run base.py first to generate it.")
    
    return joblib.load(cache_path)


def load_baseline_pca_dow_results(cache_dir=None):
    """
    Load the cached baseline PCA+DOW grid search results for analysis and plotting.
    
    Returns:
        dict: Contains model, PCA transformer, scaler, best n_components, 
              CV scores, and full grid search results
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cache')
    
    cache_path = os.path.join(cache_dir, 'base_baseline_pca_dow_cv.pkl')
    
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache file not found: {cache_path}. Run base.py first to generate it.")
    
    return joblib.load(cache_path)


# Commented helper snippets for reading cached model artifacts without retraining.
# These are not used by the main pipeline; they are here as a reference for manual
# inspection if someone wants to recover the training path from the saved .pkl files.
#
# Example 1: inspect the baseline PCA cache
# ------------------------------------------------------------------------------
# cached_data = load_baseline_pca_results()
# print("Best PCA threshold:", cached_data['best_n_comp'])
# print("Best CV score:", round(cached_data['best_cv_score'], 4))
# print("PCA grid search summary:")
# for row in cached_data['grid_search_results']:
#     print(
#         row['n_components'],
#         round(row['mean_cv_score'], 4),
#         round(row['std_cv_score'], 4),
#         row['n_components_value'],
#     )
#
# Example 2: inspect a fitted model cache directly
# ------------------------------------------------------------------------------
# ridge_cache = os.path.join(CACHE_DIR, 'base_ridge_cv.pkl')
# ridge_model = joblib.load(ridge_cache)
# ridge_cv = ridge_model.named_steps['classifier']
# print("Ridge best C:", ridge_cv.C_[0])
# ridge_scores = np.array(list(ridge_cv.scores_.values())[0])
# print("Ridge CV mean accuracy by C:", ridge_scores.mean(axis=0))
#
# Example 3: inspect the cached diagnostics used for the figures
# ------------------------------------------------------------------------------
# raw_diag_cache = os.path.join(CACHE_DIR, 'base_raw_diagnostics.pkl')
# raw_diag = joblib.load(raw_diag_cache)
# print(raw_diag.keys())  # 'ridge_bv', 'lasso_bv', 'ridge_direct', 'lasso_direct'
# ridge_diag = raw_diag['ridge_bv']
# print("Stored C grid:", ridge_diag['cs'])
# print("CV error mean:", ridge_diag['cv_err_mean'])
# print("Train error mean:", ridge_diag['tr_err_mean'])
# print("Direct test error:", raw_diag['ridge_direct']['test_errors'])
#
# pca_diag_cache = os.path.join(CACHE_DIR, 'base_pca_diagnostics.pkl')
# pca_diag = joblib.load(pca_diag_cache)
# print("PCA diagnostic keys:", pca_diag.keys())
#
# Removed unused load_all_cached_models helper and dead example block.
# The main training, figure generation, and table export paths load only the
# specific cache files they need above.
