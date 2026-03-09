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
  - To retrain all models: Set RETRAIN_ALL = True (line 49)
  - To load cached results: Use helper functions at the end of this file
    Example: cached_data = load_baseline_pca_results()
"""

import os
import joblib
import pandas as pd
from H_prep import clean_data, import_data
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

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

def _select_pca_n_components_1se(grid_results):
    best = max(grid_results, key=lambda r: r['mean_cv_score'])
    threshold = best['mean_cv_score'] - best['std_cv_score']
    candidates = [r for r in grid_results if r['mean_cv_score'] >= threshold]
    # Simpler PCA model means fewer retained components.
    chosen = min(candidates, key=lambda r: r['n_components_value'])
    return chosen, best, threshold

def _select_c_1se_from_logregcv(cv_clf):
    scores = np.array(list(cv_clf.scores_.values())[0])
    if scores.ndim == 3:
        scores = scores[:, :, 0]
    mean_scores = scores.mean(axis=0)
    std_scores = scores.std(axis=0)
    cs = np.array(cv_clf.Cs_)
    best_idx = int(np.argmax(mean_scores))
    threshold = mean_scores[best_idx] - std_scores[best_idx]
    candidate_idx = np.where(mean_scores >= threshold)[0]
    # Simpler model for logistic regularization is smaller C.
    chosen_idx = int(candidate_idx[np.argmin(cs[candidate_idx])])
    return float(cs[chosen_idx]), float(cs[best_idx]), float(threshold)

if __name__ == "__main__":
    # Set to False to use cached models; True to retrain all models
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

    tscv = TimeSeriesSplit(n_splits=3)

    # ===================================================================
    # PCA PRE-PROCESSING WITH GRID SEARCH FOR BASELINE
    # ===================================================================
    print("\n========== Grid Search for Optimal PCA Components (Baseline) ===========")
    
    # Fit scaler
    scaler_pca = StandardScaler()
    X_train_sc  = scaler_pca.fit_transform(X_train)
    X_test_sc   = scaler_pca.transform(X_test)
    
    # Grid search over different n_components
    n_components_grid = [0.85, 0.90, 0.95, 0.99]
    grid_search_results = []
    
    print(f"Testing n_components: {n_components_grid}")
    for n_comp in n_components_grid:
        pca_temp = PCA(n_components=n_comp)
        X_pca_temp = pca_temp.fit_transform(X_train_sc)
        
        # Cross-validate baseline model
        baseline_temp = LogisticRegression(penalty=None, solver="lbfgs", 
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
        
        print(f"  n_components={n_comp} ({X_pca_temp.shape[1]} components): CV Score = {mean_score:.4f} ± {std_score:.4f}")
        
    chosen_pca, best_pca, pca_threshold = _select_pca_n_components_1se(grid_search_results)
    best_n_comp = chosen_pca['n_components']
    best_n_comp_value = chosen_pca['n_components_value']
    best_score = chosen_pca['mean_cv_score']
    print(
        f"\nBest-by-mean n_components: {best_pca['n_components']} ({best_pca['n_components_value']} components), "
        f"CV={best_pca['mean_cv_score']:.4f}±{best_pca['std_cv_score']:.4f}"
    )
    print(
        f"1SE-selected n_components: {best_n_comp} ({best_n_comp_value} components), "
        f"threshold={pca_threshold:.4f}, CV={best_score:.4f}"
    )
    
    # Fit final PCA with best n_components
    pca = PCA(n_components=best_n_comp)
    X_train_pca = pca.fit_transform(X_train_sc)
    X_test_pca  = pca.transform(X_test_sc)
    n_components_raw = X_train_pca.shape[1]
    print(f"Final PCA: {X_train.shape[1]} features → {n_components_raw} components ({best_n_comp*100:.0f}% variance)")

    # ------- Baseline: plain Logistic Regression (no regularization) after PCA -------
    print("\n========== BASELINE: Plain Logistic Regression (No Regularization) + PCA ===========")
    baseline_clf = LogisticRegression(penalty=None, solver="lbfgs", random_state=1,
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
            Cs=20, cv=tscv, penalty='l2', solver='saga',
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
            C=ridge_c_1se, penalty='l2', solver='saga',
            random_state=1, max_iter=500, tol=1e-2
        ))
    ])
    pipeline_ridge_1se.fit(X_train, y_train)

    # ------- LASSO (L1): LogisticRegressionCV — stores per-fold scores for bias-variance plot -------
    print("\n========== LASSO (L1) Logistic Regression CV ==========")
    pipeline_lasso = Pipeline([
        ('scaler',     StandardScaler()),
        ('classifier', LogisticRegressionCV(
            Cs=np.logspace(-6, 4, 20), cv=tscv, penalty='l1', solver='saga',
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
            C=lasso_c_1se, penalty='l1', solver='saga',
            random_state=1, max_iter=500, tol=1e-2
        ))
    ])
    pipeline_lasso_1se.fit(X_train, y_train)

    # ------- Bias-Variance Tradeoff Plot (Ridge + LASSO) -------
    print("\n========== Generating Bias-Variance Tradeoff Plot ==========")

    def _bv_curves(cv_clf, pipeline, X_tr, y_tr, tscv_splitter, penalty, solver):
        """Extract CV test error from stored scores_ and compute train error per fold."""
        cs    = cv_clf.Cs_
        raw   = np.array(list(cv_clf.scores_.values())[0])
        if raw.ndim == 3:
            raw = raw[:, :, 0]                          # (n_folds, n_Cs)
        cv_err_mean = 1 - raw.mean(axis=0)
        cv_err_std  = raw.std(axis=0)

        scaler = pipeline.named_steps['scaler']
        train_scores = np.zeros_like(raw)
        for fold_idx, (tr, _) in enumerate(tscv_splitter.split(X_tr, y_tr)):
            X_fold = scaler.fit_transform(
                X_tr.iloc[tr] if hasattr(X_tr, 'iloc') else X_tr[tr])
            y_fold = y_tr.iloc[tr] if hasattr(y_tr, 'iloc') else y_tr[tr]
            for c_idx, c_val in enumerate(cs):
                clf = LogisticRegression(
                    C=c_val, penalty=penalty, solver=solver,
                    random_state=1, max_iter=500, tol=1e-2)
                clf.fit(X_fold, y_fold)
                train_scores[fold_idx, c_idx] = clf.score(X_fold, y_fold)
        tr_err_mean = 1 - train_scores.mean(axis=0)
        tr_err_std  = train_scores.std(axis=0)
        return cs, tr_err_mean, tr_err_std, cv_err_mean, cv_err_std

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Bias-Variance Tradeoff — Raw OHLCV Features\n(Train vs CV Test Error per Regularization Strength)',
                 fontsize=13, fontweight='bold')

    for ax, (label, cv_clf, pipe, pen, solv, color, c_1se) in zip(axes, [
        ('Ridge (L2)', ridge_cv, pipeline_ridge, 'l2', 'saga', 'darkorange', ridge_c_1se),
        ('LASSO (L1)', lasso_cv, pipeline_lasso, 'l1', 'saga', 'seagreen', lasso_c_1se),
    ]):
        cs, tr_mean, tr_std, cv_mean, cv_std = _bv_curves(
            cv_clf, pipe, X_train, y_train, tscv, pen, solv)
        ax.semilogx(cs, tr_mean, marker='o', color='steelblue', linewidth=2, label='Train error')
        ax.fill_between(cs, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15, color='steelblue')
        ax.semilogx(cs, cv_mean, marker='s', color=color,      linewidth=2, label='CV Test error')
        ax.fill_between(cs, cv_mean - cv_std, cv_mean + cv_std, alpha=0.15, color=color)
        ax.axvline(c_1se, color='red', linestyle='--',
                   label=f'1SE C = {c_1se:.4f}')
        ax.set_title(f'{label} — Bias-Variance Tradeoff')
        ax.set_xlabel('C  (Inverse Regularization Strength)\n'
                      '← High Regularization, Simpler Model      '
                      'Low Regularization, More Complex →')
        ax.set_ylabel('Prediction Error')
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

    C_grid = np.logspace(-10, 2, 30)

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle('Train vs Test Error — Raw OHLCV Features\n(Direct Train/Test Split, No CV)',
                  fontsize=13, fontweight='bold')

    for ax, (label, pen, solv, color) in zip(axes2, [
        ('Ridge (L2)', 'l2', 'saga', 'darkorange'),
        ('LASSO (L1)', 'l1', 'saga', 'seagreen'),
    ]):
        train_errors, test_errors = [], []
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train)
        X_te_sc = scaler.transform(X_test)
        for c_val in C_grid:
            clf = LogisticRegression(
                C=c_val, penalty=pen, solver=solv,
                random_state=1, max_iter=500, tol=1e-2)
            clf.fit(X_tr_sc, y_train)
            train_errors.append(1 - clf.score(X_tr_sc, y_train))
            test_errors.append(1 - clf.score(X_te_sc, y_test))

        ax.semilogx(C_grid, train_errors, marker='o', color='steelblue',
                    linewidth=2, label='Train error')
        ax.semilogx(C_grid, test_errors, marker='s', color=color,
                    linewidth=2, label='Test error')
        ax.set_title(f'{label} — Train vs Test Error')
        ax.set_xlabel('C  (Inverse Regularization Strength)\n'
                      '← High Regularization, Simpler Model      '
                      'Low Regularization, More Complex →')
        ax.set_ylabel('Prediction Error')
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
        Cs=20, cv=tscv, penalty='l2', solver='saga',
        random_state=1, n_jobs=MODEL_N_JOBS, max_iter=500, tol=1e-2, scoring='balanced_accuracy')
    clf_ridge_pca.fit(X_train_pca, y_train)
    _ridge_pca_cache = os.path.join(CACHE_DIR, 'base_ridge_pca_cv.pkl')
    joblib.dump(clf_ridge_pca, _ridge_pca_cache)
    ridge_pca_c_1se, ridge_pca_c_best, _ = _select_c_1se_from_logregcv(clf_ridge_pca)
    print(f"Best C by mean CV (Ridge+PCA): {ridge_pca_c_best:.6f}")
    print(f"1SE-selected C (Ridge+PCA):    {ridge_pca_c_1se:.6f}")
    clf_ridge_pca_1se = LogisticRegression(
        C=ridge_pca_c_1se, penalty='l2', solver='saga',
        random_state=1, max_iter=500, tol=1e-2
    )
    clf_ridge_pca_1se.fit(X_train_pca, y_train)

    # ------- LASSO CV after PCA -------
    print("\n========== LASSO (L1) + PCA — LogisticRegressionCV ==========")
    clf_lasso_pca = LogisticRegressionCV(
        Cs=np.logspace(-6, 4, 20), cv=tscv, penalty='l1', solver='saga',
        random_state=1, n_jobs=MODEL_N_JOBS, max_iter=500, tol=1e-2, scoring='balanced_accuracy')
    clf_lasso_pca.fit(X_train_pca, y_train)
    _lasso_pca_cache = os.path.join(CACHE_DIR, 'base_lasso_pca_cv.pkl')
    joblib.dump(clf_lasso_pca, _lasso_pca_cache)
    lasso_pca_c_1se, lasso_pca_c_best, _ = _select_c_1se_from_logregcv(clf_lasso_pca)
    print(f"Best C by mean CV (LASSO+PCA): {lasso_pca_c_best:.6f}")
    print(f"1SE-selected C (LASSO+PCA):    {lasso_pca_c_1se:.6f}")
    clf_lasso_pca_1se = LogisticRegression(
        C=lasso_pca_c_1se, penalty='l1', solver='saga',
        random_state=1, max_iter=500, tol=1e-2
    )
    clf_lasso_pca_1se.fit(X_train_pca, y_train)

    # ------- Bias-Variance Tradeoff Plot (PCA + Ridge/LASSO) -------
    print("\n========== Generating PCA Bias-Variance Tradeoff Plot ==========")

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    fig3.suptitle(f'Bias-Variance Tradeoff — PCA Features ({n_components_raw} comps, {best_n_comp*100:.0f}% variance)\n'
                  '(Train vs CV Test Error per Regularization Strength)',
                  fontsize=13, fontweight='bold')

    for ax, (label, cv_clf, pen, solv, color, c_1se) in zip(axes3, [
        ('Ridge (L2)', clf_ridge_pca, 'l2', 'saga', 'darkorange', ridge_pca_c_1se),
        ('LASSO (L1)', clf_lasso_pca, 'l1', 'saga', 'seagreen', lasso_pca_c_1se),
    ]):
        cs   = cv_clf.Cs_
        raw  = np.array(list(cv_clf.scores_.values())[0])
        if raw.ndim == 3:
            raw = raw[:, :, 0]
        cv_err_mean = 1 - raw.mean(axis=0)
        cv_err_std  = raw.std(axis=0)

        train_scores = np.zeros_like(raw)
        for fold_idx, (tr, _) in enumerate(tscv.split(X_train_pca, y_train)):
            X_fold = X_train_pca[tr]
            y_fold = y_train.iloc[tr] if hasattr(y_train, 'iloc') else y_train[tr]
            for c_idx, c_val in enumerate(cs):
                clf = LogisticRegression(
                    C=c_val, penalty=pen, solver=solv,
                    random_state=1, max_iter=500, tol=1e-2)
                clf.fit(X_fold, y_fold)
                train_scores[fold_idx, c_idx] = clf.score(X_fold, y_fold)
        tr_err_mean = 1 - train_scores.mean(axis=0)
        tr_err_std  = train_scores.std(axis=0)

        ax.semilogx(cs, tr_err_mean, marker='o', color='steelblue', linewidth=2, label='Train error')
        ax.fill_between(cs, tr_err_mean - tr_err_std, tr_err_mean + tr_err_std, alpha=0.15, color='steelblue')
        ax.semilogx(cs, cv_err_mean, marker='s', color=color, linewidth=2, label='CV Test error')
        ax.fill_between(cs, cv_err_mean - cv_err_std, cv_err_mean + cv_err_std, alpha=0.15, color=color)
        ax.axvline(c_1se, color='red', linestyle='--',
                   label=f'1SE C = {c_1se:.4f}')
        ax.set_title(f'{label} — Bias-Variance Tradeoff (PCA)')
        ax.set_xlabel('C  (Inverse Regularization Strength)\n'
                      '← High Regularization, Simpler Model      '
                      'Low Regularization, More Complex →')
        ax.set_ylabel('Prediction Error')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path3 = os.path.join(output_dir, '8yrs_1SE_base_logistic_pca_bias_variance.png')
    plt.savefig(out_path3, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {os.path.abspath(out_path3)}")
    plt.close()

    # ------- Train vs Test Error Plot after PCA (direct split) -------
    print("\n========== Generating PCA Train vs Test Error Plot (Direct Split) ==========")

    C_grid_pca = np.logspace(-5, 2, 30)

    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))
    fig4.suptitle(f'Train vs Test Error — PCA Features ({n_components_raw} comps, {best_n_comp*100:.0f}% variance)\n'
                  '(Direct Train/Test Split, No CV)',
                  fontsize=13, fontweight='bold')

    for ax, (label, pen, solv, color) in zip(axes4, [
        ('Ridge (L2)', 'l2', 'saga', 'darkorange'),
        ('LASSO (L1)', 'l1', 'saga', 'seagreen'),
    ]):
        train_errors, test_errors = [], []
        for c_val in C_grid_pca:
            clf = LogisticRegression(
                C=c_val, penalty=pen, solver=solv,
                random_state=1, max_iter=500, tol=1e-2)
            clf.fit(X_train_pca, y_train)
            train_errors.append(1 - clf.score(X_train_pca, y_train))
            test_errors.append(1 - clf.score(X_test_pca, y_test))

        ax.semilogx(C_grid_pca, train_errors, marker='o', color='steelblue',
                    linewidth=2, label='Train error')
        ax.semilogx(C_grid_pca, test_errors, marker='s', color=color,
                    linewidth=2, label='Test error')
        ax.set_title(f'{label} — Train vs Test Error (PCA)')
        ax.set_xlabel('C  (Inverse Regularization Strength)\n'
                      '← High Regularization, Simpler Model      '
                      'Low Regularization, More Complex →')
        ax.set_ylabel('Prediction Error')
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
    from sklearn.model_selection import cross_validate as _cv
    from sklearn.metrics import precision_score, recall_score, f1_score

    def _metrics(name, model, X_tr, y_tr, X_te, y_te, tscv_splitter, best_c=None):
        cv_res = _cv(model, X_tr, y_tr, cv=tscv_splitter,
                     return_train_score=True, n_jobs=MODEL_N_JOBS, scoring='balanced_accuracy')
        preds = model.predict(X_te)
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_te)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_te)
        else:
            y_score = preds
        return {
            'Model':              name,
            'Best C':             f'{best_c:.6f}' if best_c is not None else 'N/A',
            'Avg Train Acc':      round(cv_res['train_score'].mean(), 4),
            'Std Train Acc':      round(cv_res['train_score'].std(),  4),
            'Avg CV Test Acc':    round(cv_res['test_score'].mean(),  4),
            'Std CV Test Acc':    round(cv_res['test_score'].std(),   4),
            'Hold-out Test Acc':  round(model.score(X_te, y_te),      4),
            'Precision':          round(precision_score(y_te, preds, zero_division=0), 4),
            'Recall':             round(recall_score(y_te, preds,    zero_division=0), 4),
            'F1':                 round(f1_score(y_te, preds,         zero_division=0), 4),
            'ROC-AUC':            round(roc_auc_score(y_te, y_score), 4),
        }

    rows = [
        _metrics(f'Baseline (No Reg) + PCA ({n_components_raw}, {best_n_comp*100:.0f}%)',    baseline_clf,  X_train_pca, y_train, X_test_pca, y_test, tscv, best_c=None),
        _metrics('Ridge CV (raw)',    pipeline_ridge_1se, X_train, y_train, X_test, y_test, tscv, best_c=ridge_c_1se),
        _metrics('LASSO CV (raw)',    pipeline_lasso_1se, X_train, y_train, X_test, y_test, tscv, best_c=lasso_c_1se),
        _metrics(f'Ridge CV (PCA-{n_components_raw})',    clf_ridge_pca_1se,  X_train_pca, y_train, X_test_pca, y_test, tscv, best_c=ridge_pca_c_1se),
        _metrics(f'LASSO CV (PCA-{n_components_raw})',    clf_lasso_pca_1se,  X_train_pca, y_train, X_test_pca, y_test, tscv, best_c=lasso_pca_c_1se),
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
    # Keep only Mon–Thu (drop Fri to avoid dummy variable trap)
    dow_dummies = dow_dummies.iloc[:, :-1]
    print(f"Day-of-week columns added: {list(dow_dummies.columns)}")

    X_dow = pd.concat([X, dow_dummies], axis=1)
    X_train_dow, X_test_dow, _, _ = train_test_split(
        X_dow, y_classification, test_size=0.2, shuffle=False
    )
    print(f"Feature matrix with DOW: {X_dow.shape[1]} columns")

    # --- PCA + DOW ---
    print("\n========== Grid Search for Optimal PCA Components (DOW) ===========")
    
    # Grid search over different n_components for DOW features
    n_components_grid = [0.85, 0.90, 0.95, 0.99]
    grid_search_results_dow = []
    
    scaler_pca_dow = StandardScaler()
    X_train_dow_sc  = scaler_pca_dow.fit_transform(X_train_dow)
    X_test_dow_sc   = scaler_pca_dow.transform(X_test_dow)
    
    print(f"Testing n_components: {n_components_grid}")
    for n_comp in n_components_grid:
        pca_temp = PCA(n_components=n_comp)
        X_pca_temp = pca_temp.fit_transform(X_train_dow_sc)
        
        # Cross-validate baseline model
        baseline_temp = LogisticRegression(penalty=None, solver="lbfgs", 
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
        
        print(f"  n_components={n_comp} ({X_pca_temp.shape[1]} components): CV Score = {mean_score:.4f} ± {std_score:.4f}")
        
    chosen_pca_dow, best_pca_dow, pca_threshold_dow = _select_pca_n_components_1se(grid_search_results_dow)
    best_n_comp_dow = chosen_pca_dow['n_components']
    best_n_comp_value_dow = chosen_pca_dow['n_components_value']
    best_score_dow = chosen_pca_dow['mean_cv_score']
    print(
        f"\nBest-by-mean n_components (DOW): {best_pca_dow['n_components']} ({best_pca_dow['n_components_value']} components), "
        f"CV={best_pca_dow['mean_cv_score']:.4f}±{best_pca_dow['std_cv_score']:.4f}"
    )
    print(
        f"1SE-selected n_components (DOW): {best_n_comp_dow} ({best_n_comp_value_dow} components), "
        f"threshold={pca_threshold_dow:.4f}, CV={best_score_dow:.4f}"
    )
    
    # Fit final PCA with best n_components
    pca_dow = PCA(n_components=best_n_comp_dow)
    X_train_dow_pca = pca_dow.fit_transform(X_train_dow_sc)
    X_test_dow_pca  = pca_dow.transform(X_test_dow_sc)
    n_components_dow = X_train_dow_pca.shape[1]
    print(f"Final PCA: {X_train_dow.shape[1]} features → {n_components_dow} components ({best_n_comp_dow*100:.0f}% variance)")

    # --- Baseline + DOW + PCA ---
    print("\n========== BASELINE (No Reg) + DOW + PCA ===========")
    baseline_dow_clf = LogisticRegression(penalty=None, solver='lbfgs', random_state=1,
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
            Cs=20, cv=tscv, penalty='l2', solver='saga',
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
            C=ridge_dow_c_1se, penalty='l2', solver='saga',
            random_state=1, max_iter=500, tol=1e-2
        ))
    ])
    pipeline_ridge_dow_1se.fit(X_train_dow, y_train)

    # --- LASSO CV + DOW ---
    print("\n========== LASSO CV + DOW ==========")
    pipeline_lasso_dow = Pipeline([
        ('scaler',     StandardScaler()),
        ('classifier', LogisticRegressionCV(
            Cs=np.logspace(-6, 4, 20), cv=tscv, penalty='l1', solver='saga',
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
            C=lasso_dow_c_1se, penalty='l1', solver='saga',
            random_state=1, max_iter=500, tol=1e-2
        ))
    ])
    pipeline_lasso_dow_1se.fit(X_train_dow, y_train)


    # --- Ridge CV + PCA + DOW ---
    print("\n========== RIDGE CV + PCA + DOW ==========")
    clf_ridge_pca_dow = LogisticRegressionCV(
        Cs=20, cv=tscv, penalty='l2', solver='saga',
        random_state=1, n_jobs=MODEL_N_JOBS, max_iter=500, tol=1e-2, scoring='balanced_accuracy')
    clf_ridge_pca_dow.fit(X_train_dow_pca, y_train)
    _ridge_pca_dow_cache = os.path.join(CACHE_DIR, 'base_ridge_pca_dow_cv.pkl')
    joblib.dump(clf_ridge_pca_dow, _ridge_pca_dow_cache)
    ridge_pca_dow_c_1se, ridge_pca_dow_c_best, _ = _select_c_1se_from_logregcv(clf_ridge_pca_dow)
    print(f"Best C by mean CV (Ridge+PCA+DOW): {ridge_pca_dow_c_best:.6f}")
    print(f"1SE-selected C (Ridge+PCA+DOW):    {ridge_pca_dow_c_1se:.6f}")
    clf_ridge_pca_dow_1se = LogisticRegression(
        C=ridge_pca_dow_c_1se, penalty='l2', solver='saga',
        random_state=1, max_iter=500, tol=1e-2
    )
    clf_ridge_pca_dow_1se.fit(X_train_dow_pca, y_train)

    # --- LASSO CV + PCA + DOW ---
    print("\n========== LASSO CV + PCA + DOW ==========")
    clf_lasso_pca_dow = LogisticRegressionCV(
        Cs=np.logspace(-6, 4, 20), cv=tscv, penalty='l1', solver='saga',
        random_state=1, n_jobs=MODEL_N_JOBS, max_iter=500, tol=1e-2, scoring='balanced_accuracy')
    clf_lasso_pca_dow.fit(X_train_dow_pca, y_train)
    _lasso_pca_dow_cache = os.path.join(CACHE_DIR, 'base_lasso_pca_dow_cv.pkl')
    joblib.dump(clf_lasso_pca_dow, _lasso_pca_dow_cache)
    lasso_pca_dow_c_1se, lasso_pca_dow_c_best, _ = _select_c_1se_from_logregcv(clf_lasso_pca_dow)
    print(f"Best C by mean CV (LASSO+PCA+DOW): {lasso_pca_dow_c_best:.6f}")
    print(f"1SE-selected C (LASSO+PCA+DOW):    {lasso_pca_dow_c_1se:.6f}")
    clf_lasso_pca_dow_1se = LogisticRegression(
        C=lasso_pca_dow_c_1se, penalty='l1', solver='saga',
        random_state=1, max_iter=500, tol=1e-2
    )
    clf_lasso_pca_dow_1se.fit(X_train_dow_pca, y_train)

    # --- DOW rows using same _metrics helper ---
    rows_dow = [
        _metrics(f'Baseline (No Reg)+DOW+PCA ({n_components_dow}, {best_n_comp_dow*100:.0f}%)',       baseline_dow_clf,  X_train_dow_pca, y_train, X_test_dow_pca, y_test, tscv, best_c=None),
        _metrics('Ridge CV+DOW',       pipeline_ridge_dow_1se, X_train_dow, y_train, X_test_dow, y_test, tscv, best_c=ridge_dow_c_1se),
        _metrics('LASSO CV+DOW',       pipeline_lasso_dow_1se, X_train_dow, y_train, X_test_dow, y_test, tscv, best_c=lasso_dow_c_1se),
        _metrics(f'Ridge CV+PCA ({n_components_dow})+DOW',   clf_ridge_pca_dow_1se,  X_train_dow_pca, y_train, X_test_dow_pca, y_test, tscv, best_c=ridge_pca_dow_c_1se),
        _metrics(f'LASSO CV+PCA ({n_components_dow})+DOW',   clf_lasso_pca_dow_1se,  X_train_dow_pca, y_train, X_test_dow_pca, y_test, tscv, best_c=lasso_pca_dow_c_1se),
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
    keep_cols = ['Hold-out Test Acc', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    full_df = full_df[keep_cols]

    print("\n===== Full Comparison Table =====")
    print(full_df.to_string())

    # Save CSV
    combined_csv = os.path.join(output_dir, '8yrs_1SE_base_logistic_comparison.csv')
    full_df.to_csv(combined_csv)
    print(f"\nFull comparison table saved to: {os.path.abspath(combined_csv)}")

    tex_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'output', 'Used', 'tex')
    os.makedirs(tex_output_dir, exist_ok=True)

    # Build LaTeX manually with \midrule separating the three groups
    tex_path = os.path.join(tex_output_dir, '8yrs_1SE_base_logistic_comparison.tex')
    col_fmt  = 'l' + 'r' * len(full_df.columns)
    col_header = ' & '.join(['Model'] + list(full_df.columns)) + r' \\'
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

    def _row(name, vals):
        dagger = r'$^\dagger$' if 'LASSO' in name else ''
        # First value is Best C (string), rest are numeric
        best_c = _latex_escape(vals[0])
        formatted_vals = [best_c] + [f'{v:.4f}' for v in vals[1:]]
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
        for name, row in comparison_df.iterrows():
            f.write(_row(name, row.values) + '\n')
        f.write(r'\midrule' + '\n')
        # Group 2: DOW
        for name, row in dow_df.iterrows():
            f.write(_row(name, row.values) + '\n')
        f.write(r'\bottomrule' + '\n')
        f.write(r'\end{tabular}' + '\n')
        f.write(r'\par\smallskip' + '\n')
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


def load_all_cached_models(cache_dir=None):
    """
    Load all cached logistic regression models.
    
    Returns:
        dict: Dictionary with model names as keys and cached models as values
        
    Example usage:
        >>> models = load_all_cached_models()
        >>> print(models.keys())
        >>> ridge_model = models['ridge_cv']
        >>> print(f"Best C for Ridge: {ridge_model.named_steps['classifier'].C_[0]}")
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cache')
    
    cache_files = {
        'baseline_pca': 'base_baseline_pca_cv.pkl',
        'baseline_pca_dow': 'base_baseline_pca_dow_cv.pkl',
        'ridge_cv': 'base_ridge_cv.pkl',
        'lasso_cv': 'base_lasso_cv.pkl',
        'ridge_pca_cv': 'base_ridge_pca_cv.pkl',
        'lasso_pca_cv': 'base_lasso_pca_cv.pkl',
        'ridge_dow_cv': 'base_ridge_dow_cv.pkl',
        'lasso_dow_cv': 'base_lasso_dow_cv.pkl',
        'ridge_pca_dow_cv': 'base_ridge_pca_dow_cv.pkl',
        'lasso_pca_dow_cv': 'base_lasso_pca_dow_cv.pkl',
    }
    
    models = {}
    for name, filename in cache_files.items():
        cache_path = os.path.join(cache_dir, filename)
        if os.path.exists(cache_path):
            models[name] = joblib.load(cache_path)
        else:
            print(f"Warning: {filename} not found")
    
    return models


if __name__ == "__main__" and False:  # Set to True to run this example
    # Example: Load and analyze baseline PCA grid search results
    print("\n" + "="*70)
    print("EXAMPLE: Reading Cached Results for Analysis")
    print("="*70)
    
    # Load baseline PCA results
    baseline_results = load_baseline_pca_results()
    print(f"\nBaseline PCA - Best variance threshold: {baseline_results['best_n_comp']}")
    print(f"Best CV score: {baseline_results['best_cv_score']:.4f}")
    print(f"Number of components: {baseline_results['n_components_value']}")
    
    print("\nGrid search details:")
    for result in baseline_results['grid_search_results']:
        print(f"  {result['n_components']}: {result['mean_cv_score']:.4f} ± {result['std_cv_score']:.4f} "
              f"({result['n_components_value']} components)")
    
    # Load all models
    all_models = load_all_cached_models()
    print(f"\nLoaded {len(all_models)} cached models")
    print(f"Available models: {list(all_models.keys())}")
    print(f"MODEL_N_JOBS={MODEL_N_JOBS} (set env MODEL_N_JOBS to override)")
