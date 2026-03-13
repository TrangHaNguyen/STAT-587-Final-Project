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
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
from model_grids import BASELINE_PCA_GRID, LASSO_GRID, RIDGE_GRID, ELASTIC_NET_GRID, ELASTIC_NET_L1_RATIO_GRID, LOGISTIC_TOL
from H_eval import get_final_metrics, rank_models_by_metrics, save_best_model_plots_from_gridsearch

MODEL_N_JOBS = int(os.getenv("MODEL_N_JOBS", "-1"))

def clear_base_caches():
    # No active .pkl caches are needed for the current best-model-only
    # plotting workflow.
    print("No active base caches to clear.")

def to_binary_class(y):
    return (y >= 0).astype(int)


def _keep_raw_stock_ohlcv(X: pd.DataFrame) -> pd.DataFrame:
    idx = pd.IndexSlice
    metrics = ['Open', 'Close', 'High', 'Low', 'Volume']
    return X.loc[:, idx[metrics, 'Stocks', :]].copy()

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
                random_state=1, max_iter=500, tol=LOGISTIC_TOL
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
            random_state=1, max_iter=500, tol=LOGISTIC_TOL
        )
        clf.fit(X_train, y_train)
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
    clf = LogisticRegression(
        C=c_val, l1_ratio=l1_ratio, solver=solver,
        class_weight='balanced',
        random_state=1, max_iter=500, tol=LOGISTIC_TOL
    )
    clf.fit(X_train, y_train)
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
        se = std / np.sqrt(5)
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

if __name__ == "__main__":
    # Retained only as a compatibility switch. The active workflow no longer
    # writes diagnostic caches.
    RETRAIN_ALL = True

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
                                          random_state=1, max_iter=500, tol=LOGISTIC_TOL)
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
                                      max_iter=500, tol=LOGISTIC_TOL)
    baseline_clf.fit(X_train_pca, y_train)
    
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
    print("\n========== RIDGE (L2) Logistic Regression CV ==========")
    pipeline_ridge = Pipeline([
        ('scaler',     StandardScaler()),
        ('classifier', LogisticRegressionCV(
            # Single-stage ridge logistic regression: l1_ratio=0 replaces penalty='l2'.
            Cs=RIDGE_GRID, cv=tscv, l1_ratios=[0], solver='saga',
            class_weight='balanced',
            random_state=1, n_jobs=MODEL_N_JOBS, max_iter=500, tol=LOGISTIC_TOL, scoring='balanced_accuracy'
        ))
    ])
    pipeline_ridge.fit(X_train, y_train)
    
    ridge_cv = pipeline_ridge.named_steps['classifier']
    ridge_c_1se, ridge_c_best, _ = _select_c_1se_from_logregcv(ridge_cv)
    print(f"Best C by mean CV (Ridge): {ridge_c_best:.6f}")
    print(f"1SE-selected C (Ridge):    {ridge_c_1se:.6f}")
    pipeline_ridge_1se = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            C=ridge_c_1se, l1_ratio=0, solver='saga',
            class_weight='balanced',
            random_state=1, max_iter=500, tol=LOGISTIC_TOL
        ))
    ])
    pipeline_ridge_1se.fit(X_train, y_train)

    # ------- LASSO (L1): LogisticRegressionCV — stores per-fold scores for bias-variance plot -------
    print("\n========== LASSO (L1) Logistic Regression CV ==========")
    pipeline_lasso = Pipeline([
        ('scaler',     StandardScaler()),
        ('classifier', LogisticRegressionCV(
            # Single-stage lasso logistic regression: l1_ratio=1 replaces penalty='l1'.
            Cs=LASSO_GRID, cv=tscv, l1_ratios=[1], solver='saga',
            class_weight='balanced',
            random_state=1, n_jobs=MODEL_N_JOBS, max_iter=500, tol=LOGISTIC_TOL, scoring='balanced_accuracy'
        ))
    ])
    pipeline_lasso.fit(X_train, y_train)
    
    lasso_cv = pipeline_lasso.named_steps['classifier']
    lasso_c_1se, lasso_c_best, _ = _select_c_1se_from_logregcv(lasso_cv)
    print(f"Best C by mean CV (LASSO): {lasso_c_best:.6f}")
    print(f"1SE-selected C (LASSO):    {lasso_c_1se:.6f}")
    pipeline_lasso_1se = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            C=lasso_c_1se, l1_ratio=1, solver='saga',
            class_weight='balanced',
            random_state=1, max_iter=500, tol=LOGISTIC_TOL
        ))
    ])
    pipeline_lasso_1se.fit(X_train, y_train)

    # ------- Elastic Net (raw): GridSearchCV with 1SE refit -------
    print("\n========== ELASTIC NET Logistic Regression Grid Search ==========")
    pipeline_elastic = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            solver='saga', class_weight='balanced',
            random_state=1, max_iter=500, tol=LOGISTIC_TOL
        ))
    ])
    elastic_param_grid = {
        'classifier__C': ELASTIC_NET_GRID,
        'classifier__l1_ratio': ELASTIC_NET_L1_RATIO_GRID,
    }
    grid_search_elastic = GridSearchCV(
        pipeline_elastic, elastic_param_grid, cv=tscv, return_train_score=True,
        n_jobs=MODEL_N_JOBS, scoring='balanced_accuracy',
        refit=make_one_se_refit(['classifier__C', 'classifier__l1_ratio'])
    )
    grid_search_elastic.fit(X_train, y_train)
    elastic_best_c = float(grid_search_elastic.best_params_['classifier__C'])
    elastic_best_l1 = float(grid_search_elastic.best_params_['classifier__l1_ratio'])
    print(f"1SE-selected Elastic Net params: C={elastic_best_c:.6f}, l1_ratio={elastic_best_l1:.3f}")
    pipeline_elastic_1se = clone(grid_search_elastic.best_estimator_)
    pipeline_elastic_1se.fit(X_train, y_train)

    raw_scaler = StandardScaler()
    X_train_raw_sc = raw_scaler.fit_transform(X_train)
    X_test_raw_sc = raw_scaler.transform(X_test)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'output')
    os.makedirs(output_dir, exist_ok=True)


    # ------- Ridge CV after PCA -------
    print("\n========== RIDGE (L2) + PCA — LogisticRegressionCV ==========")
    clf_ridge_pca = LogisticRegressionCV(
        Cs=RIDGE_GRID, cv=tscv, l1_ratios=[0], solver='saga',
        class_weight='balanced',
        random_state=1, n_jobs=MODEL_N_JOBS, max_iter=500, tol=LOGISTIC_TOL, scoring='balanced_accuracy')
    clf_ridge_pca.fit(X_train_pca, y_train)
    ridge_pca_c_1se, ridge_pca_c_best, _ = _select_c_1se_from_logregcv(clf_ridge_pca)
    print(f"Best C by mean CV (Ridge+PCA): {ridge_pca_c_best:.6f}")
    print(f"1SE-selected C (Ridge+PCA):    {ridge_pca_c_1se:.6f}")
    clf_ridge_pca_1se = LogisticRegression(
        C=ridge_pca_c_1se, l1_ratio=0, solver='saga',
        class_weight='balanced',
        random_state=1, max_iter=500, tol=LOGISTIC_TOL
    )
    clf_ridge_pca_1se.fit(X_train_pca, y_train)

    # ------- LASSO CV after PCA -------
    print("\n========== LASSO (L1) + PCA — LogisticRegressionCV ==========")
    clf_lasso_pca = LogisticRegressionCV(
        Cs=LASSO_GRID, cv=tscv, l1_ratios=[1], solver='saga',
        class_weight='balanced',
        random_state=1, n_jobs=MODEL_N_JOBS, max_iter=500, tol=LOGISTIC_TOL, scoring='balanced_accuracy')
    clf_lasso_pca.fit(X_train_pca, y_train)
    lasso_pca_c_1se, lasso_pca_c_best, _ = _select_c_1se_from_logregcv(clf_lasso_pca)
    print(f"Best C by mean CV (LASSO+PCA): {lasso_pca_c_best:.6f}")
    print(f"1SE-selected C (LASSO+PCA):    {lasso_pca_c_1se:.6f}")
    clf_lasso_pca_1se = LogisticRegression(
        C=lasso_pca_c_1se, l1_ratio=1, solver='saga',
        class_weight='balanced',
        random_state=1, max_iter=500, tol=LOGISTIC_TOL
    )
    clf_lasso_pca_1se.fit(X_train_pca, y_train)

    # ===================================================================
    # MODEL COMPARISON TABLE
    # ===================================================================
    print("\n========== Model Comparison Table ==========")

    ranking_rows = []

    def _eval_row(name, model, X_tr, y_tr, X_te, y_te, n_splits, best_c=None, best_c_label=None):
        shared = get_final_metrics(
            model, X_tr, y_tr, X_te, y_te, n_splits=n_splits, label=name
        )
        ranking_rows.append({'Model': name, **shared})
        preds = model.predict(X_te)
        is_degenerate = (
            best_c is not None
            and np.isclose(best_c, 1e-6)
            and np.unique(preds).size == 1
        )
        return {
            'Model':           name,
            'Best C':          best_c_label if best_c_label is not None else (f'{best_c:.6f}' if best_c is not None else 'N/A'),
            'Avg CV Train Plain Acc':   shared['train_avg_accuracy'],
            'CV Train Plain Acc SD':    shared['train_std_accuracy'],
            'Avg CV Validation Plain Acc': shared['validation_avg_accuracy'],
            'CV Acc SD':                   shared['validation_std_accuracy'],
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
        _eval_row('Elastic Net (raw)', pipeline_elastic_1se, X_train, y_train, X_test, y_test, tscv.n_splits, best_c=elastic_best_c, best_c_label=f'{elastic_best_c:.6f} (l1={elastic_best_l1:.2f})'),
        _eval_row(f'Ridge+PCA ({n_components_raw})', clf_ridge_pca_1se, X_train_pca, y_train, X_test_pca, y_test, tscv.n_splits, best_c=ridge_pca_c_1se),
        _eval_row(f'LASSO+PCA ({n_components_raw})', clf_lasso_pca_1se, X_train_pca, y_train, X_test_pca, y_test, tscv.n_splits, best_c=lasso_pca_c_1se),
    ]

    comparison_df = pd.DataFrame(rows).set_index('Model')
    print(comparison_df.to_string())

    ranked_df = rank_models_by_metrics(pd.DataFrame(ranking_rows))
    print("\n===== Ranked Baseline Models =====")
    print(ranked_df[['Model', 'rank_test_split_accuracy', 'rank_test_sensitivity_macro',
                     'rank_test_specificity_macro', 'rank_test_roc_auc_macro', 'average_rank']].to_string(index=False))

    plot_configs = {
        'Ridge (raw)': {
            'logregcv': ridge_cv,
            'one_se_c': ridge_c_1se,
            'model_title': 'Ridge (L2) - LR',
            'feature_title': 'Raw OHLCV Features',
            'X_train_plot': X_train_raw_sc,
            'X_test_plot': X_test_raw_sc,
            'l1_ratio': 0,
            'direct_color': 'darkorange',
        },
        'LASSO (raw)': {
            'logregcv': lasso_cv,
            'one_se_c': lasso_c_1se,
            'model_title': 'LASSO (L1) - LR',
            'feature_title': 'Raw OHLCV Features',
            'X_train_plot': X_train_raw_sc,
            'X_test_plot': X_test_raw_sc,
            'l1_ratio': 1,
            'direct_color': 'seagreen',
        },
        f'Ridge+PCA ({n_components_raw})': {
            'logregcv': clf_ridge_pca,
            'one_se_c': ridge_pca_c_1se,
            'model_title': 'Ridge (L2) - LR',
            'feature_title': f'PCA Features ({n_components_raw} comps, {best_n_comp*100:.0f}% variance)',
            'X_train_plot': X_train_pca,
            'X_test_plot': X_test_pca,
            'l1_ratio': 0,
            'direct_color': 'darkorange',
        },
        f'LASSO+PCA ({n_components_raw})': {
            'logregcv': clf_lasso_pca,
            'one_se_c': lasso_pca_c_1se,
            'model_title': 'LASSO (L1) - LR',
            'feature_title': f'PCA Features ({n_components_raw} comps, {best_n_comp*100:.0f}% variance)',
            'X_train_plot': X_train_pca,
            'X_test_plot': X_test_pca,
            'l1_ratio': 1,
            'direct_color': 'seagreen',
        },
        'Elastic Net (raw)': {
            'grid_search': grid_search_elastic,
            'x_param': 'classifier__C',
            'x_label': 'C',
            'model_title': 'Elastic Net - LR',
            'X_train_plot': X_train,
            'X_test_plot': X_test,
        },
    }

    best_model_name = str(ranked_df.iloc[0]['Model'])
    if best_model_name not in plot_configs:
        plottable_ranked = ranked_df[ranked_df['Model'].isin(plot_configs)]
        if plottable_ranked.empty:
            raise ValueError("No ranked model has available diagnostics for plotting.")
        plot_model_name = str(plottable_ranked.iloc[0]['Model'])
        print(
            f"\nBest overall model by average rank is '{best_model_name}', "
            f"but available bias-variance/train-test plots exist only for regularized models. "
            f"Plotting '{plot_model_name}' instead."
        )
    else:
        plot_model_name = best_model_name
        print(f"\nBest model by average rank: {plot_model_name}")

    best_plot_cfg = plot_configs[plot_model_name]
    if 'grid_search' in best_plot_cfg:
        save_best_model_plots_from_gridsearch(
            best_plot_cfg['grid_search'],
            best_plot_cfg['x_param'],
            best_plot_cfg['x_label'],
            best_plot_cfg['model_title'],
            os.path.join(output_dir, '8yrs_1SE_base_logistic_best_bias_variance.png'),
            os.path.join(output_dir, '8yrs_1SE_base_logistic_best_train_test.png'),
            best_plot_cfg['X_train_plot'],
            y_train,
            best_plot_cfg['X_test_plot'],
            y_test,
        )
    else:
        print(f"\n========== Preparing Diagnostics for {plot_model_name} ==========")
        diagnostics = {
            'single_bv': _compute_bv_curves(
                best_plot_cfg['logregcv'],
                best_plot_cfg['X_train_plot'],
                y_train,
                tscv,
                best_plot_cfg['l1_ratio'],
                'saga',
            ),
            'single_direct': _compute_direct_split_errors(
                best_plot_cfg['X_train_plot'],
                y_train,
                best_plot_cfg['X_test_plot'],
                y_test,
                _augment_c_grid_with_selected_values(
                    best_plot_cfg['logregcv'].Cs_,
                    best_plot_cfg['one_se_c'],
                ),
                best_plot_cfg['l1_ratio'],
                'saga',
            ),
        }
        _plot_single_model_diagnostics(
            diagnostics=diagnostics,
            bv_key='single_bv',
            direct_key='single_direct',
            one_se_c=best_plot_cfg['one_se_c'],
            model_title=best_plot_cfg['model_title'],
            feature_title=best_plot_cfg['feature_title'],
            X_train_plot=best_plot_cfg['X_train_plot'],
            y_train=y_train,
            X_test_plot=best_plot_cfg['X_test_plot'],
            y_test=y_test,
            l1_ratio=best_plot_cfg['l1_ratio'],
            output_bv=os.path.join(output_dir, '8yrs_1SE_base_logistic_best_bias_variance.png'),
            output_direct=os.path.join(output_dir, '8yrs_1SE_base_logistic_best_train_test.png'),
            direct_color=best_plot_cfg['direct_color'],
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
                                          random_state=1, max_iter=500, tol=LOGISTIC_TOL)
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
                                          max_iter=500, tol=LOGISTIC_TOL)
    baseline_dow_clf.fit(X_train_dow_pca, y_train)
    
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
    print("\n========== RIDGE CV + DOW ==========")
    pipeline_ridge_dow = Pipeline([
        ('scaler',     StandardScaler()),
        ('classifier', LogisticRegressionCV(
            Cs=RIDGE_GRID, cv=tscv, l1_ratios=[0], solver='saga',
            class_weight='balanced',
            random_state=1, n_jobs=MODEL_N_JOBS, max_iter=500, tol=LOGISTIC_TOL, scoring='balanced_accuracy'
        ))
    ])
    pipeline_ridge_dow.fit(X_train_dow, y_train)
    ridge_dow_cv = pipeline_ridge_dow.named_steps['classifier']
    ridge_dow_c_1se, ridge_dow_c_best, _ = _select_c_1se_from_logregcv(ridge_dow_cv)
    print(f"Best C by mean CV (Ridge+DOW): {ridge_dow_c_best:.6f}")
    print(f"1SE-selected C (Ridge+DOW):    {ridge_dow_c_1se:.6f}")
    pipeline_ridge_dow_1se = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            C=ridge_dow_c_1se, l1_ratio=0, solver='saga',
            class_weight='balanced',
            random_state=1, max_iter=500, tol=LOGISTIC_TOL
        ))
    ])
    pipeline_ridge_dow_1se.fit(X_train_dow, y_train)

    # --- LASSO CV + DOW ---
    print("\n========== LASSO CV + DOW ==========")
    pipeline_lasso_dow = Pipeline([
        ('scaler',     StandardScaler()),
        ('classifier', LogisticRegressionCV(
            Cs=LASSO_GRID, cv=tscv, l1_ratios=[1], solver='saga',
            class_weight='balanced',
            random_state=1, n_jobs=MODEL_N_JOBS, max_iter=500, tol=LOGISTIC_TOL, scoring='balanced_accuracy'
        ))
    ])
    pipeline_lasso_dow.fit(X_train_dow, y_train)
    lasso_dow_cv = pipeline_lasso_dow.named_steps['classifier']
    lasso_dow_c_1se, lasso_dow_c_best, _ = _select_c_1se_from_logregcv(lasso_dow_cv)
    print(f"Best C by mean CV (LASSO+DOW): {lasso_dow_c_best:.6f}")
    print(f"1SE-selected C (LASSO+DOW):    {lasso_dow_c_1se:.6f}")
    pipeline_lasso_dow_1se = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            C=lasso_dow_c_1se, l1_ratio=1, solver='saga',
            class_weight='balanced',
            random_state=1, max_iter=500, tol=LOGISTIC_TOL
        ))
    ])
    pipeline_lasso_dow_1se.fit(X_train_dow, y_train)


    # --- Ridge CV + PCA + DOW ---
    print("\n========== RIDGE CV + PCA + DOW ==========")
    clf_ridge_pca_dow = LogisticRegressionCV(
        Cs=RIDGE_GRID, cv=tscv, l1_ratios=[0], solver='saga',
        class_weight='balanced',
        random_state=1, n_jobs=MODEL_N_JOBS, max_iter=500, tol=LOGISTIC_TOL, scoring='balanced_accuracy')
    clf_ridge_pca_dow.fit(X_train_dow_pca, y_train)
    ridge_pca_dow_c_1se, ridge_pca_dow_c_best, _ = _select_c_1se_from_logregcv(clf_ridge_pca_dow)
    print(f"Best C by mean CV (Ridge+PCA+DOW): {ridge_pca_dow_c_best:.6f}")
    print(f"1SE-selected C (Ridge+PCA+DOW):    {ridge_pca_dow_c_1se:.6f}")
    clf_ridge_pca_dow_1se = LogisticRegression(
        C=ridge_pca_dow_c_1se, l1_ratio=0, solver='saga',
        class_weight='balanced',
        random_state=1, max_iter=500, tol=LOGISTIC_TOL
    )
    clf_ridge_pca_dow_1se.fit(X_train_dow_pca, y_train)

    # --- LASSO CV + PCA + DOW ---
    print("\n========== LASSO CV + PCA + DOW ==========")
    clf_lasso_pca_dow = LogisticRegressionCV(
        Cs=LASSO_GRID, cv=tscv, l1_ratios=[1], solver='saga',
        class_weight='balanced',
        random_state=1, n_jobs=MODEL_N_JOBS, max_iter=500, tol=LOGISTIC_TOL, scoring='balanced_accuracy')
    clf_lasso_pca_dow.fit(X_train_dow_pca, y_train)
    lasso_pca_dow_c_1se, lasso_pca_dow_c_best, _ = _select_c_1se_from_logregcv(clf_lasso_pca_dow)
    print(f"Best C by mean CV (LASSO+PCA+DOW): {lasso_pca_dow_c_best:.6f}")
    print(f"1SE-selected C (LASSO+PCA+DOW):    {lasso_pca_dow_c_1se:.6f}")
    clf_lasso_pca_dow_1se = LogisticRegression(
        C=lasso_pca_dow_c_1se, l1_ratio=1, solver='saga',
        class_weight='balanced',
        random_state=1, max_iter=500, tol=LOGISTIC_TOL
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
    print("All models trained successfully!")
    print("="*70)
