from typing import Any, cast
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import clone
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np

from H_prep import clean_data
from H_eval import rolling_window_backtest, get_final_metrics

'''No need for hyperparameter tuning for Logistic Regression via GridSearchCV since LogisticRegressionCV performs internal CV to select the best C value. We will just use the default 10 values of C that LogisticRegressionCV tests.'''

VERBOSE=0
WINDOW_SIZE=121
HORIZON=21

if __name__=="__main__":
    X, y_regression=cast(Any, clean_data(lag_period=[1, 2, 3], lookback_period=0, sector=True, corr=True, corr_level=2, testing=False))
    X_train, X_test, y_train, y_test=train_test_split(X, y_regression, test_size=0.2, random_state=1, shuffle=False)
    def to_binary_class(y):
        return (y>=0).astype(int)
        
    y_classification=to_binary_class(y_regression)
    y_train=to_binary_class(y_train)
    y_test=to_binary_class(y_test)
    tscv=TimeSeriesSplit(n_splits=10)
    custom_Cs=[0.05, 0.1, 1.0, 10.0]

    # ------- LASSO(Internal) APPLICATION -------
    Log_Reg_R=LogisticRegressionCV(Cs=custom_Cs, cv=tscv, l1_ratios=[1], solver='saga', random_state=1, n_jobs=-1, max_iter=500, tol=1e-2, verbose=VERBOSE)
    
    Log_Reg_model_pipeline_R=Pipeline([('scaler', StandardScaler()), ('classifier', Log_Reg_R)])

    Log_Reg_model_pipeline_R.fit(X_train, y_train)

    best_c = Log_Reg_model_pipeline_R.named_steps['classifier'].C_[0]
    Opt_Log_Reg_R=LogisticRegression(C=best_c, l1_ratio=1, solver='saga', random_state=1, max_iter=500, tol=1e-2)

    Opt_Log_Reg_model_pipeline_R=Pipeline([('scaler', StandardScaler()), ('classifier', Opt_Log_Reg_R)])

    optimized_Log_Reg_R_=clone(Opt_Log_Reg_model_pipeline_R)
    optimized_Log_Reg_R_.fit(X_train, y_train)

    coefs = optimized_Log_Reg_R_.named_steps['classifier'].coef_
    print(f"Non-zero coefficients: {np.count_nonzero(coefs)}")

    rolling_window_backtest(optimized_Log_Reg_R_, X, y_classification, verbose=1, window_size=WINDOW_SIZE, horizon=HORIZON)

    optimized_Log_Reg_R_=clone(optimized_Log_Reg_R_)
    optimized_Log_Reg_R_.fit(X_train, y_train)

    get_final_metrics(optimized_Log_Reg_R_, X_train, y_train, X_test, y_test, n_splits=10)

    input("Press Enter to continue...")

    # ------- RIDGE(Internal) APPLICATION -------
    Log_Reg_L=LogisticRegressionCV(Cs=custom_Cs, cv=tscv, l1_ratios=[0], solver='saga', random_state=1, n_jobs=-1, max_iter=500, tol=1e-2, verbose=VERBOSE)
    
    Log_Reg_model_pipeline_L=Pipeline([('scaler', StandardScaler()), ('classifier', Log_Reg_L)])

    Log_Reg_model_pipeline_L.fit(X_train, y_train)

    best_c = Log_Reg_model_pipeline_L.named_steps['classifier'].C_[0]
    Opt_Log_Reg_L=LogisticRegression(C=best_c, l1_ratio=1, solver='saga', random_state=1, max_iter=500, tol=1e-2)

    Opt_Log_Reg_model_pipeline_L=Pipeline([('scaler', StandardScaler()), ('classifier', Opt_Log_Reg_L)])

    optimized_Log_Reg_L_=clone(Opt_Log_Reg_model_pipeline_L)
    optimized_Log_Reg_L_.fit(X_train, y_train)

    coefs = optimized_Log_Reg_L_.named_steps['classifier'].coef_
    print(f"Non-zero coefficients: {np.count_nonzero(coefs)}")

    rolling_window_backtest(optimized_Log_Reg_L_, X, y_classification, verbose=1, window_size=WINDOW_SIZE, horizon=HORIZON)

    optimized_Log_Reg_L_=clone(Opt_Log_Reg_model_pipeline_L)
    optimized_Log_Reg_L_.fit(X_train, y_train)

    get_final_metrics(optimized_Log_Reg_L_, X_train, y_train, X_test, y_test, n_splits=10)

    input("Press Enter to continue...")

    # ------- PCA to Ridge(Internal) APPLICATION -------
    Log_Reg_PCA_L=LogisticRegression(l1_ratio=0, solver='liblinear', random_state=1)
    
    Log_Reg_model_pipeline_PCA_L=Pipeline([('scaler', StandardScaler()),
                                           ('pca', PCA(n_components=0.9)), 
                                           ('classifier', Log_Reg_PCA_L)])

    param_grid={
        'pca__n_components': [0.7, 0.8, 0.9, 0.95],
        'classifier__C': [0.01, 0.1, 1.0, 10.0]
    }
    grid_search_PCA_ridge=GridSearchCV(Log_Reg_model_pipeline_PCA_L, param_grid, cv=tscv, return_train_score=True, verbose=VERBOSE)

    grid_search_PCA_ridge.fit(X_train, y_train)

    optimized_PCA_ridge_=grid_search_PCA_ridge.best_estimator_

    optimized_Log_Reg_PCA_ridge_=clone(grid_search_PCA_ridge.best_estimator_)
    optimized_Log_Reg_PCA_ridge_.fit(X_train, y_train)
    
    rolling_window_backtest(optimized_Log_Reg_PCA_ridge_, X, y_classification, verbose=1, window_size=WINDOW_SIZE, horizon=HORIZON)

    optimized_Log_Reg_PCA_ridge_=clone(grid_search_PCA_ridge.best_estimator_)
    optimized_Log_Reg_PCA_ridge_.fit(X_train, y_train)

    get_final_metrics(optimized_Log_Reg_PCA_ridge_, X_train, y_train, X_test, y_test, n_splits=10)

    input("Press Enter to Finish...")



    # ------- EXPORT: CV Regularization Selection Figure -------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    from sklearn.model_selection import StratifiedKFold
    from sklearn.base import clone

    def _cv_train_test_errors(pipeline, X, y, cs, cv_splitter, penalty, solver, l1_ratio=None):
        """Return per-fold train scores shape (n_folds, n_Cs) by refitting plain LogisticRegression."""
        scaler = clone(pipeline.named_steps['scaler'])
        train_scores = np.zeros((cv_splitter.get_n_splits(), len(cs)))
        for fold_idx, (tr, _) in enumerate(cv_splitter.split(X, y)):
            X_tr = scaler.fit_transform(X.iloc[tr] if hasattr(X, 'iloc') else X[tr])
            y_tr = y.iloc[tr] if hasattr(y, 'iloc') else y[tr]
            for c_idx, c_val in enumerate(cs):
                kwargs = dict(C=c_val, solver=solver, random_state=1, max_iter=500, tol=1e-2)
                if penalty == 'elasticnet':
                    kwargs.update(penalty='elasticnet', l1_ratio=l1_ratio)
                else:
                    kwargs['penalty'] = penalty
                clf = LogisticRegression(**kwargs)
                clf.fit(X_tr, y_tr)
                train_scores[fold_idx, c_idx] = clf.score(X_tr, y_tr)
        return train_scores

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Bias-Variance Tradeoff: Train vs CV Test Error across Regularization Strength',
                 fontsize=14, fontweight='bold')

    # --- LASSO (L1) ---
    lasso_cv = Log_Reg_model_pipeline_R.named_steps['classifier']
    cs_lasso = lasso_cv.Cs_
    raw_scores_lasso = np.array(list(lasso_cv.scores_.values())[0])
    if raw_scores_lasso.ndim == 3:
        raw_scores_lasso = raw_scores_lasso[:, :, 0]
    mean_cv_err_lasso = 1 - raw_scores_lasso.mean(axis=0)
    std_cv_err_lasso  = raw_scores_lasso.std(axis=0)
    train_scores_lasso = _cv_train_test_errors(
        Log_Reg_model_pipeline_R, X_train, y_train, cs_lasso, tscv, 'l1', 'saga', l1_ratio=1)
    mean_tr_err_lasso = 1 - train_scores_lasso.mean(axis=0)
    std_tr_err_lasso  = train_scores_lasso.std(axis=0)
    axes[0].semilogx(cs_lasso, mean_tr_err_lasso, marker='o', color='steelblue', linewidth=2, label='Train error')
    axes[0].fill_between(cs_lasso, mean_tr_err_lasso - std_tr_err_lasso, mean_tr_err_lasso + std_tr_err_lasso,
                         alpha=0.15, color='steelblue')
    axes[0].semilogx(cs_lasso, mean_cv_err_lasso, marker='s', color='darkorange', linewidth=2, label='CV Test error')
    axes[0].fill_between(cs_lasso, mean_cv_err_lasso - std_cv_err_lasso, mean_cv_err_lasso + std_cv_err_lasso,
                         alpha=0.15, color='darkorange')
    axes[0].axvline(lasso_cv.C_[0], color='red', linestyle='--', label=f'Best C = {lasso_cv.C_[0]:.4f}')
    axes[0].set_title('LASSO (L1) — Bias-Variance Tradeoff')
    axes[0].set_xlabel('C  (Inverse Regularization Strength)\n← High Regularization, Simpler Model      Low Regularization, More Complex →')
    axes[0].set_ylabel('Prediction Error')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # --- Ridge (L2) ---
    ridge_cv = Log_Reg_model_pipeline_L.named_steps['classifier']
    cs_ridge = ridge_cv.Cs_
    raw_scores_ridge = np.array(list(ridge_cv.scores_.values())[0])
    if raw_scores_ridge.ndim == 3:
        raw_scores_ridge = raw_scores_ridge[:, :, 0]
    mean_cv_err_ridge = 1 - raw_scores_ridge.mean(axis=0)
    std_cv_err_ridge  = raw_scores_ridge.std(axis=0)
    train_scores_ridge = _cv_train_test_errors(
        Log_Reg_model_pipeline_L, X_train, y_train, cs_ridge, tscv, 'l2', 'saga')
    mean_tr_err_ridge = 1 - train_scores_ridge.mean(axis=0)
    std_tr_err_ridge  = train_scores_ridge.std(axis=0)
    axes[1].semilogx(cs_ridge, mean_tr_err_ridge, marker='o', color='steelblue', linewidth=2, label='Train error')
    axes[1].fill_between(cs_ridge, mean_tr_err_ridge - std_tr_err_ridge, mean_tr_err_ridge + std_tr_err_ridge,
                         alpha=0.15, color='steelblue')
    axes[1].semilogx(cs_ridge, mean_cv_err_ridge, marker='s', color='darkorange', linewidth=2, label='CV Test error')
    axes[1].fill_between(cs_ridge, mean_cv_err_ridge - std_cv_err_ridge, mean_cv_err_ridge + std_cv_err_ridge,
                         alpha=0.15, color='darkorange')
    axes[1].axvline(ridge_cv.C_[0], color='red', linestyle='--', label=f'Best C = {ridge_cv.C_[0]:.4f}')
    axes[1].set_title('Ridge (L2) — Bias-Variance Tradeoff')
    axes[1].set_xlabel('C  (Inverse Regularization Strength)\n← High Regularization, Simpler Model      Low Regularization, More Complex →')
    axes[1].set_ylabel('Prediction Error')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # --- PCA + Ridge (GridSearchCV) ---
    cv_results = grid_search_PCA_ridge.cv_results_
    c_vals = [p['classifier__C'] for p in cv_results['params']]
    unique_cs = sorted(set(c_vals))
    n_splits = tscv.get_n_splits()
    avg_cv_err_per_c = []
    avg_tr_err_per_c = []
    std_cv_err_per_c = []
    std_tr_err_per_c = []
    for uc in unique_cs:
        idxs = [i for i, c in enumerate(c_vals) if c == uc]
        fold_test  = np.array([cv_results[f'split{s}_test_score'][i]  for s in range(n_splits) for i in idxs])
        fold_train = np.array([cv_results[f'split{s}_train_score'][i] for s in range(n_splits) for i in idxs])
        avg_cv_err_per_c.append(1 - fold_test.mean())
        avg_tr_err_per_c.append(1 - fold_train.mean())
        std_cv_err_per_c.append(fold_test.std())
        std_tr_err_per_c.append(fold_train.std())
    avg_cv_err_per_c = np.array(avg_cv_err_per_c)
    avg_tr_err_per_c = np.array(avg_tr_err_per_c)
    std_cv_err_per_c = np.array(std_cv_err_per_c)
    std_tr_err_per_c = np.array(std_tr_err_per_c)
    best_c_pca = grid_search_PCA_ridge.best_params_['classifier__C']
    axes[2].semilogx(unique_cs, avg_tr_err_per_c, marker='o', color='steelblue', linewidth=2, label='Train error')
    axes[2].fill_between(unique_cs, avg_tr_err_per_c - std_tr_err_per_c, avg_tr_err_per_c + std_tr_err_per_c,
                         alpha=0.15, color='steelblue')
    axes[2].semilogx(unique_cs, avg_cv_err_per_c, marker='s', color='darkorange', linewidth=2, label='CV Test error')
    axes[2].fill_between(unique_cs, avg_cv_err_per_c - std_cv_err_per_c, avg_cv_err_per_c + std_cv_err_per_c,
                         alpha=0.15, color='darkorange')
    axes[2].axvline(best_c_pca, color='red', linestyle='--', label=f'Best C = {best_c_pca:.4f}')
    axes[2].set_title('PCA + Ridge — Bias-Variance Tradeoff\n(avg over PCA components)')
    axes[2].set_xlabel('C  (Inverse Regularization Strength)\n← High Regularization, Simpler Model      Low Regularization, More Complex →')
    axes[2].set_ylabel('Prediction Error')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'logistic_regression_cv_regularization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {os.path.abspath(output_path)}")
    plt.close()

    # ------- EXPORT: GridSearch Train vs Test Accuracy for all parameter values -------
    cv_results = grid_search_PCA_ridge.cv_results_
    params       = cv_results['params']
    mean_train   = cv_results['mean_train_score']
    mean_test    = cv_results['mean_test_score']

    unique_cs_grid  = sorted(set(p['classifier__C']         for p in params))
    unique_pca_grid = sorted(set(p['pca__n_components']     for p in params))

    n_pca = len(unique_pca_grid)
    fig2, axes2 = plt.subplots(1, n_pca, figsize=(5 * n_pca, 5), sharey=True)
    fig2.suptitle('GridSearch – Train vs Test Prediction Error\n(PCA + Ridge Logistic Regression)',
                  fontsize=13, fontweight='bold')

    colors = plt.cm.tab10.colors
    for ax, n_comp in zip(axes2, unique_pca_grid):
        mask   = [i for i, p in enumerate(params) if p['pca__n_components'] == n_comp]
        cs_sub = [params[i]['classifier__C']   for i in mask]
        tr_sub = [mean_train[i]                for i in mask]
        te_sub = [mean_test[i]                 for i in mask]

        tr_err = [1 - v for v in tr_sub]
        te_err = [1 - v for v in te_sub]
        ax.semilogx(cs_sub, tr_err, marker='o', color='steelblue',   label='Train error', linewidth=1.8)
        ax.semilogx(cs_sub, te_err, marker='s', color='darkorange',  label='Test error',  linewidth=1.8)

        best_idx = int(np.argmin(te_err))
        ax.axvline(cs_sub[best_idx], color='red', linestyle='--', alpha=0.7,
                   label=f'Best C = {cs_sub[best_idx]:.4f}')

        ax.set_title(f'PCA n_components = {n_comp}')
        ax.set_xlabel('C  (Inverse Regularization Strength)\n← High Regularization, Simpler Model      Low Regularization, More Complex →')
        ax.set_ylabel('Mean CV Prediction Error')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path2 = os.path.join(output_dir, 'logistic_regression_gridsearch_train_test.png')
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {os.path.abspath(output_path2)}")
    plt.close()

    # ------- EXPORT: Single Train vs Test Error over a fine C grid (no CV) -------
    best_n_components = grid_search_PCA_ridge.best_params_['pca__n_components']
    fine_c_grid = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    fine_train_scores = []
    fine_test_scores  = []
    for c_val in fine_c_grid:
        pipe_fine = Pipeline([
            ('scaler',     StandardScaler()),
            ('pca',        PCA(n_components=best_n_components)),
            ('classifier', LogisticRegression(C=c_val, penalty='l2', solver='saga',
                                              random_state=1, max_iter=500, tol=1e-2))
        ])
        pipe_fine.fit(X_train, y_train)
        fine_train_scores.append(1 - pipe_fine.score(X_train, y_train))
        fine_test_scores.append(1 - pipe_fine.score(X_test, y_test))

    best_fine_idx = int(np.argmin(fine_test_scores))

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.semilogx(fine_c_grid, fine_train_scores, marker='o', color='steelblue',
                 linewidth=2, label='Train error')
    ax3.semilogx(fine_c_grid, fine_test_scores,  marker='s', color='darkorange',
                 linewidth=2, label='Test error')
    ax3.axvline(fine_c_grid[best_fine_idx], color='red', linestyle='--', alpha=0.8,
                label=f'Best C = {fine_c_grid[best_fine_idx]:.4f}')
    ax3.set_title(f'Train vs Test Prediction Error over C Grid\n(PCA n_components = {best_n_components})',
                  fontsize=12, fontweight='bold')
    ax3.set_xlabel('C  (Inverse Regularization Strength)\n← High Regularization, Simpler Model      Low Regularization, More Complex →')
    ax3.set_ylabel('Prediction Error')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path3 = os.path.join(output_dir, 'logistic_regression_gridsearch_aggregated.png')
    plt.savefig(output_path3, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {os.path.abspath(output_path3)}")
    plt.close()
