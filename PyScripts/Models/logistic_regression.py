from typing import Any, cast
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline

from data_preprocessing_and_cleaning import clean_data
from model_evaluation import get_final_metrics_grid, rolling_window_backtest, classification_accuracy, get_final_metrics

'''No need for hyperparameter tuning for Logistic Regression via GridSearchCV since LogisticRegressionCV performs internal CV to select the best C value. We will just use the default 10 values of C that LogisticRegressionCV tests.'''

VERBOSE=0

if __name__=="__main__":
    X, y_regression=cast(Any, clean_data(sector=True, corr=True, corr_level=2, testing=True))
    X_train, X_test, y_train, y_test=train_test_split(X, y_regression, test_size=0.2, random_state=1)
    def to_binary_class(y):
        return (y>=0).astype(int)
    y_classification=to_binary_class(y_regression)
    y_train=to_binary_class(y_train)
    y_test=to_binary_class(y_test)
    tscv=TimeSeriesSplit(n_splits=3)

    # ------- LASSO(Internal) APPLICATION -------
    Log_Reg_R=LogisticRegressionCV(Cs=5, cv=tscv, l1_ratios=[1], solver='saga', random_state=1, n_jobs=-1, max_iter=500, tol=1e-2, verbose=VERBOSE)
    
    Log_Reg_model_pipeline_R=Pipeline([('scaler', StandardScaler()), ('classifier', Log_Reg_R)])

    Log_Reg_model_pipeline_R.fit(X_train, y_train)

    best_c = Log_Reg_model_pipeline_R.named_steps['classifier'].C_[0]
    Opt_Log_Reg_R=LogisticRegression(C=best_c, l1_ratio=1, solver='saga', random_state=1, max_iter=500, tol=1e-2)

    Opt_Log_Reg_model_pipeline_R=Pipeline([('scaler', StandardScaler()), ('classifier', Opt_Log_Reg_R)])

    Opt_Log_Reg_model_pipeline_R.fit(X_train, y_train)

    rolling_window_backtest(Opt_Log_Reg_model_pipeline_R, X, y_classification, verbose=1)

    get_final_metrics(Opt_Log_Reg_model_pipeline_R, X_train, y_train, X_test, y_test)

    input("Press Enter to continue...")

    # ------- RIDGE(Internal) APPLICATION -------
    Log_Reg_L=LogisticRegressionCV(Cs=5, cv=tscv, l1_ratios=[0], solver='saga', random_state=1, n_jobs=-1, max_iter=500, tol=1e-2, verbose=VERBOSE)
    
    Log_Reg_model_pipeline_L=Pipeline([('scaler', StandardScaler()), ('classifier', Log_Reg_L)])

    Log_Reg_model_pipeline_L.fit(X_train, y_train)

    best_c = Log_Reg_model_pipeline_L.named_steps['classifier'].C_[0]
    Opt_Log_Reg_L=LogisticRegression(C=best_c, l1_ratio=1, solver='saga', random_state=1, max_iter=500, tol=1e-2)

    Opt_Log_Reg_model_pipeline_L=Pipeline([('scaler', StandardScaler()), ('classifier', Opt_Log_Reg_L)])

    Opt_Log_Reg_model_pipeline_L.fit(X_train, y_train)

    rolling_window_backtest(Opt_Log_Reg_model_pipeline_L, X, y_classification, verbose=1)

    get_final_metrics(Opt_Log_Reg_model_pipeline_L, X_train, y_train, X_test, y_test)

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

    rolling_window_backtest(optimized_PCA_ridge_, X, y_classification, verbose=1)

    get_final_metrics_grid(grid_search_PCA_ridge, X_test, y_test)

    input("Press Enter to Finish...")

    # ------- EXPORT: CV Regularization Selection Figure -------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Cross-Validation: Optimal Regularization Selection (Prediction Error)', fontsize=14, fontweight='bold')

    # --- LASSO (L1) ---
    lasso_cv = Log_Reg_model_pipeline_R.named_steps['classifier']
    cs_lasso = lasso_cv.Cs_
    # scores_ shape: (n_folds, n_Cs) or (n_folds, n_Cs, n_l1_ratios) when l1_ratios is set
    raw_scores_lasso = np.array(list(lasso_cv.scores_.values())[0])
    if raw_scores_lasso.ndim == 3:
        raw_scores_lasso = raw_scores_lasso[:, :, 0]
    mean_error_lasso = 1 - raw_scores_lasso.mean(axis=0)
    axes[0].semilogx(cs_lasso, mean_error_lasso, marker='o', color='steelblue')
    axes[0].axvline(lasso_cv.C_[0], color='red', linestyle='--', label=f'Best C = {lasso_cv.C_[0]:.4f}')
    axes[0].set_title('LASSO (L1) Regularization')
    axes[0].set_xlabel('C  (Inverse Regularization Strength)\n← High Regularization, Simpler Model      Low Regularization, More Complex →')
    axes[0].set_ylabel('Mean CV Prediction Error')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Ridge (L2) ---
    ridge_cv = Log_Reg_model_pipeline_L.named_steps['classifier']
    cs_ridge = ridge_cv.Cs_
    raw_scores_ridge = np.array(list(ridge_cv.scores_.values())[0])
    if raw_scores_ridge.ndim == 3:
        raw_scores_ridge = raw_scores_ridge[:, :, 0]
    mean_error_ridge = 1 - raw_scores_ridge.mean(axis=0)
    axes[1].semilogx(cs_ridge, mean_error_ridge, marker='o', color='darkorange')
    axes[1].axvline(ridge_cv.C_[0], color='red', linestyle='--', label=f'Best C = {ridge_cv.C_[0]:.4f}')
    axes[1].set_title('Ridge (L2) Regularization')
    axes[1].set_xlabel('C  (Inverse Regularization Strength)\n← High Regularization, Simpler Model      Low Regularization, More Complex →')
    axes[1].set_ylabel('Mean CV Prediction Error')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # --- PCA + Ridge (GridSearchCV) ---
    cv_results = grid_search_PCA_ridge.cv_results_
    c_vals = [p['classifier__C'] for p in cv_results['params']]
    mean_scores_grid = cv_results['mean_test_score']
    unique_cs = sorted(set(c_vals))
    avg_errors_per_c = [
        1 - np.mean([mean_scores_grid[i] for i, c in enumerate(c_vals) if c == uc])
        for uc in unique_cs
    ]
    best_c_pca = grid_search_PCA_ridge.best_params_['classifier__C']
    axes[2].semilogx(unique_cs, avg_errors_per_c, marker='o', color='seagreen')
    axes[2].axvline(best_c_pca, color='red', linestyle='--', label=f'Best C = {best_c_pca:.4f}')
    axes[2].set_title('PCA + Ridge (GridSearchCV, avg over PCA components)')
    axes[2].set_xlabel('C  (Inverse Regularization Strength)\n← High Regularization, Simpler Model      Low Regularization, More Complex →')
    axes[2].set_ylabel('Mean CV Prediction Error')
    axes[2].legend()
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

    # ------- EXPORT: Single Train vs Test Accuracy over a fine C grid -------
    from sklearn.model_selection import cross_validate
    best_n_components = grid_search_PCA_ridge.best_params_['pca__n_components']
    fine_c_grid = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    fine_train_scores = []
    fine_test_scores  = []
    for c_val in fine_c_grid:
        pipe_fine = Pipeline([
            ('scaler',     StandardScaler()),
            ('pca',        PCA(n_components=best_n_components)),
            ('classifier', LogisticRegression(C=c_val, penalty='l2', solver='saga',
                                              random_state=1, max_iter=500, tol=1e-2))
        ])
        cv_res = cross_validate(pipe_fine, X_train, y_train, cv=tscv,
                                scoring='accuracy', return_train_score=True)
        fine_train_scores.append(1 - cv_res['train_score'].mean())
        fine_test_scores.append(1 - cv_res['test_score'].mean())

    best_fine_idx = int(np.argmin(fine_test_scores))

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.semilogx(fine_c_grid, fine_train_scores, marker='o', color='steelblue',
                 linewidth=2, label='Train error')
    ax3.semilogx(fine_c_grid, fine_test_scores,  marker='s', color='darkorange',
                 linewidth=2, label='Test error')
    ax3.axvline(fine_c_grid[best_fine_idx], color='red', linestyle='--', alpha=0.8,
                label=f'Best C = {fine_c_grid[best_fine_idx]:.4f}')
    ax3.set_title(f'Train vs Test Prediction Error over Fine C Grid\n(PCA n_components = {best_n_components})',
                  fontsize=12, fontweight='bold')
    ax3.set_xlabel('C  (Inverse Regularization Strength)\n← High Regularization, Simpler Model      Low Regularization, More Complex →')
    ax3.set_ylabel('Mean CV Prediction Error')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path3 = os.path.join(output_dir, 'logistic_regression_gridsearch_aggregated.png')
    plt.savefig(output_path3, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {os.path.abspath(output_path3)}")
    plt.close()
