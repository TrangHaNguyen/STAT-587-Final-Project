#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import (train_test_split, TimeSeriesSplit,
                                     GridSearchCV, cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from H_prep import clean_data, import_data
from H_eval import (
    RollingWindowBacktest,
    get_final_metrics,
    rank_models_by_metrics,
    save_best_model_plots_from_gridsearch,
)
from model_grids import BASE_RF_PARAM_GRID, PCA_RF_PARAM_GRID

MODEL_N_JOBS = int(os.getenv("MODEL_N_JOBS", "-1"))
# Keep the outer GridSearchCV/cross-validation parallel, but make each
# RandomForest fit single-threaded to avoid nested parallel oversubscription.
RF_FIT_N_JOBS = 1
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

def _latex_escape(text: str) -> str:
    """Escape minimal LaTeX special characters for table text."""
    return (str(text)
            .replace("\\", r"\textbackslash{}")
            .replace("&", r"\&")
            .replace("%", r"\%")
            .replace("_", r"\_")
            .replace("#", r"\#")
            .replace("$", r"\$")
            .replace("{", r"\{")
            .replace("}", r"\}"))

def write_latex_table(df: pd.DataFrame, path: str, caption: str, label: str, note: str | None =None) -> None:
    """Write DataFrame to LaTeX, with a no-jinja fallback."""
    try:
        latex = df.to_latex(float_format='%.3f', caption=caption, label=label)
        if note:
            latex = latex.replace(
                "\\end{table}\n",
                "\\par\\smallskip\n" + f"\\footnotesize {_latex_escape(note)}\n" + "\\end{table}\n"
            )
        with open(path, "w", encoding="utf-8") as f:
            f.write(latex)
        return
    except ImportError:
        pass

    col_count = len(df.columns) + 1
    align = "l" + "c" * (col_count - 1)
    idx_name = df.index.name if df.index.name else "Model"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write(f"\\caption{{{_latex_escape(caption)}}}\n")
        f.write(f"\\label{{{_latex_escape(label)}}}\n")
        f.write(f"\\begin{{tabular}}{{{align}}}\n\\hline\n")
        header = [_latex_escape(idx_name)] + [_latex_escape(c) for c in df.columns]
        f.write(" & ".join(header) + " \\\\\n\\hline\n")
        for idx, row in df.iterrows():
            values = [_latex_escape(idx)]
            for val in row.values:
                if isinstance(val, (float, np.floating)):
                    values.append(f"{float(val):.3f}")
                else:
                    values.append(_latex_escape(val))
            f.write(" & ".join(values) + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
        if note:
            f.write("\\par\\smallskip\n")
            f.write(f"\\footnotesize {_latex_escape(note)}\n")
        f.write("\\end{table}\n")

def build_export_table(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only compact metrics for reporting/export."""
    out = df.copy()
    keep_cols = ["Test Acc", "Precision", "Recall", "Specificity", "F1", "ROC-AUC", "CV Acc SD"]
    return out[keep_cols]


def build_latex_export_table(df: pd.DataFrame) -> pd.DataFrame:
    """Apply export-only row labels without changing internal model keys."""
    out = build_export_table(df).copy()
    out = out.rename(
        index={
            'Base RF': 'Raw RF',
            'Base RF+DOW': 'Raw RF+DOW',
            'Base RF+Lags': 'Raw RF+Lags',
        }
    )
    return out

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
        ('scaler', StandardScaler()),
        ('reducer', PCA()),
        ('classifier', RandomForestClassifier(random_state=1, n_jobs=RF_FIT_N_JOBS, class_weight='balanced'))
    ])

def _run_grid_search(pipeline, param_grid, X_train, y_train, tscv, refit, heading):
    print(f"\n========== {heading} ==========")
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=tscv,
        n_jobs=MODEL_N_JOBS, return_train_score=True, verbose=1,
        scoring='balanced_accuracy', refit=refit
    )
    grid_search.fit(X_train, y_train)
    return grid_search

def _run_model_report_and_backtest(search_obj, X_full, y_full, X_train, y_train, X_test, y_test, label, n_splits):
    print("\n--- Model Report ---")
    shared = get_final_metrics(search_obj.best_estimator_, X_train, y_train, X_test, y_test, n_splits=n_splits, label=label)
    print("\n--- Rolling Window Backtest ---")
    rwb_obj = RollingWindowBacktest(
        clone(search_obj.best_estimator_),
        X_full,
        y_full,
        X_train,
        window_size=BACKTEST_WINDOW_SIZE,
        horizon=BACKTEST_HORIZON,
    )
    rwb_obj.rolling_window_backtest(verbose=1)
    return shared

def _run_rf_suite(X_full, y_full, X_train, y_train, X_test, y_test, tscv, dataset_suffix="", variant_label=""):
    heading_suffix = f" {variant_label}" if variant_label else ""
    specs = [
        {
            'name': 'Base RF',
            'heading': 'RF GridSearch (20 combinations)' if not variant_label else f'Base RF{heading_suffix} GridSearch',
            'pipeline': _base_rf_pipeline(),
            'param_grid': BASE_RF_PARAM_GRID,
            'refit': make_one_se_refit(['classifier__max_depth', 'classifier__n_estimators', 'classifier__max_features']),
        },
        {
            'name': 'PCA RF',
            'heading': 'PCA + RF GridSearch' if not variant_label else f'PCA RF{heading_suffix} GridSearch',
            'pipeline': _pca_rf_pipeline(),
            'param_grid': PCA_RF_PARAM_GRID,
            'refit': make_one_se_refit(['classifier__max_depth', 'classifier__n_estimators', 'classifier__max_features'], fixed_cols=['reducer__n_components']),
        },
    ]

    searches = {}
    metric_rows = {}
    for spec in specs:
        search_obj = _run_grid_search(
            spec['pipeline'],
            spec['param_grid'],
            X_train,
            y_train,
            tscv,
            spec['refit'],
            spec['heading']
        )
        best_prefix = "Best params" if spec['name'] == 'Base RF' and not variant_label else f"Best params ({spec['name']}{heading_suffix})"
        print(f"{best_prefix}: {search_obj.best_params_}")
        report_label = f"{spec['name']}{heading_suffix}"
        metric_rows[spec['name']] = _run_model_report_and_backtest(
            search_obj, X_full, y_full, X_train, y_train, X_test, y_test, report_label, tscv.n_splits
        )
        searches[spec['name']] = search_obj
    return searches, metric_rows

def _rf_metrics_payload(name, shared):
    display_row = {
        'Model':             name,
        'Avg CV Train Plain Acc':      shared['train_avg_accuracy'],
        'CV Train Plain Acc SD':       shared['train_std_accuracy'],
        'Avg CV Validation Plain Acc': shared['validation_avg_accuracy'],
        'CV Acc SD':                   shared['validation_std_accuracy'],
        'Test Acc':                    shared['test_split_accuracy'],
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
    for model_name in ['Base RF', 'PCA RF']:
        display_row, ranking_row = _rf_metrics_payload(
            f'{model_name}{suffix}',
            metric_rows[model_name],
        )
        display_rows.append(display_row)
        ranking_rows.append(ranking_row)
    return pd.DataFrame(display_rows).set_index('Model'), ranking_rows

if __name__ == "__main__":
    # ------- Load raw data (no extra clean_data feature engineering) -------
    TESTING = False
    output_prefix = "sample" if USE_SAMPLE_PARQUET else "8yrs"
    print(f"MODEL_N_JOBS={MODEL_N_JOBS} (set env MODEL_N_JOBS to override)")
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
    X, y_regression = clean_data(*DATA, raw=True, extra_features=False)
    X = _keep_raw_stock_ohlcv(X)
    X.columns = [f"{metric}_{ticker}" for metric, _, ticker in X.columns]
    print(f"Feature matrix shape: {X.shape[0]} rows, {X.shape[1]} columns.")

    y_classification = to_binary_class(y_regression)
    print(f"Final shape — X: {X.shape}, y: {y_classification.shape}")

    # ------- Train/test split (80/20, no shuffle to preserve time order) -------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_classification, test_size=0.2, shuffle=False
    )
    # Previous temporary change used `KFold(n_splits=5, shuffle=False)`.
    tscv = TimeSeriesSplit(n_splits=5)

    raw_searches, raw_metric_rows = _run_rf_suite(X, y_classification, X_train, y_train, X_test, y_test, tscv)
    grid_search_rf = raw_searches['Base RF']
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
        X_dow, y_classification, test_size=0.2, shuffle=False
    )
    print(f"Feature matrix with DOW: {X_dow.shape[1]} columns")

    dow_searches, dow_metric_rows = _run_rf_suite(X_dow, y_classification, X_train_dow, y_train_dow, X_test_dow, y_test_dow, tscv, dataset_suffix="_dow", variant_label="+ DOW")

    # ===================================================================
    # COMBINED COMPARISON TABLE (raw OHLCV vs raw OHLCV + DOW)
    # ===================================================================
    dow_df, dow_ranking_rows = _build_comparison_df(dow_metric_rows, suffix="+DOW")

    combined_df = pd.concat([comparison_df, dow_df])
    print("\n===== Combined Comparison Table =====")
    print(combined_df.to_string())

    combined_export_df = build_latex_export_table(combined_df)
    combined_csv = os.path.join(output_dir, f'{output_prefix}_base_rf_comparison.csv')
    combined_export_df.to_csv(combined_csv, float_format='%.3f')
    print(f"\nCombined comparison table saved to: {os.path.abspath(combined_csv)}")

    tex_path = os.path.join(tex_output_dir, f'{output_prefix}_1SE_base_random_forest.tex')
    write_latex_table(
        combined_export_df,
        tex_path,
        'Random Forest Model Comparison: Raw OHLCV vs Raw OHLCV + Day-of-Week',
        'tab:base_rf_comparison',
        note=f'Test Acc = plain hold-out accuracy on the final 20% test split. All reported CV/train/test accuracy columns in this table use plain accuracy after hyperparameters were selected by CV balanced accuracy. {RECALL_NOTE}'
    )
    print(f"LaTeX table saved to:               {os.path.abspath(tex_path)}")

    # ===================================================================
    # LAG1–LAG7 EXTENSION
    # Generate lag1..lag7 of raw OHLCV features (no DOW), then re-run
    # the remaining RF model variants
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
        X_lag, y_lag, test_size=0.2, shuffle=False
    )
    print(f"Feature matrix with lags: {X_lag.shape[1]} columns, {X_lag.shape[0]} rows")

    lag_searches, lag_metric_rows = _run_rf_suite(X_lag, y_lag, X_train_lag, y_train_lag, X_test_lag, y_test_lag, tscv, dataset_suffix="_lag", variant_label="+ Lags")

    # ===================================================================
    # FULL COMPARISON TABLE (raw + DOW + Lags)
    # ===================================================================
    lag_df, lag_ranking_rows = _build_comparison_df(lag_metric_rows, suffix="+Lags")

    full_df = pd.concat([comparison_df, dow_df, lag_df])
    print("\n===== Full Comparison Table =====")
    print(full_df.to_string())

    full_export_df = build_latex_export_table(full_df)
    full_csv = os.path.join(output_dir, f'{output_prefix}_base_rf_comparison.csv')
    full_export_df.to_csv(full_csv, float_format='%.3f')
    print(f"\nFull comparison table saved to: {os.path.abspath(full_csv)}")

    tex_path = os.path.join(tex_output_dir, f'{output_prefix}_1SE_base_random_forest.tex')
    write_latex_table(
        full_export_df,
        tex_path,
        'Random Forest Model Comparison: Raw OHLCV vs +Day-of-Week vs +Lag1--7',
        'tab:base_rf_comparison',
        note=f'Test Acc = plain hold-out accuracy on the final 20% test split. All reported CV/train/test accuracy columns in this table use plain accuracy after hyperparameters were selected by CV balanced accuracy. {RECALL_NOTE}'
    )
    print(f"LaTeX table saved to:           {os.path.abspath(tex_path)}")

    plot_candidates = {
        'Base RF': {
            'search': raw_searches['Base RF'],
            'x_param': 'classifier__max_depth',
            'x_label': 'max_depth',
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
        },
        'PCA RF': {
            'search': raw_searches['PCA RF'],
            'x_param': 'reducer__n_components',
            'x_label': 'PCA n_components',
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
        },
        'Base RF+DOW': {
            'search': dow_searches['Base RF'],
            'x_param': 'classifier__max_depth',
            'x_label': 'max_depth',
            'X_train': X_train_dow,
            'y_train': y_train_dow,
            'X_test': X_test_dow,
            'y_test': y_test_dow,
        },
        'PCA RF+DOW': {
            'search': dow_searches['PCA RF'],
            'x_param': 'reducer__n_components',
            'x_label': 'PCA n_components',
            'X_train': X_train_dow,
            'y_train': y_train_dow,
            'X_test': X_test_dow,
            'y_test': y_test_dow,
        },
        'Base RF+Lags': {
            'search': lag_searches['Base RF'],
            'x_param': 'classifier__max_depth',
            'x_label': 'max_depth',
            'X_train': X_train_lag,
            'y_train': y_train_lag,
            'X_test': X_test_lag,
            'y_test': y_test_lag,
        },
        'PCA RF+Lags': {
            'search': lag_searches['PCA RF'],
            'x_param': 'reducer__n_components',
            'x_label': 'PCA n_components',
            'X_train': X_train_lag,
            'y_train': y_train_lag,
            'X_test': X_test_lag,
            'y_test': y_test_lag,
        },
    }
    ranking_rows = raw_ranking_rows + dow_ranking_rows + lag_ranking_rows
    ranked_df = rank_models_by_metrics(pd.DataFrame(ranking_rows))
    best_model_name = str(ranked_df.iloc[0]["Model"])
    best_candidate = plot_candidates[best_model_name]
    save_best_model_plots_from_gridsearch(
        best_candidate['search'],
        best_candidate['x_param'],
        best_candidate['x_label'],
        best_model_name,
        os.path.join(output_dir, f'{output_prefix}_base_rf_best_model_bias_variance.png'),
        os.path.join(output_dir, f'{output_prefix}_base_rf_best_model_train_test.png'),
        best_candidate['X_train'],
        best_candidate['y_train'],
        best_candidate['X_test'],
        best_candidate['y_test'],
    )
    print(f"\nBest baseline RF-family model by average rank: {best_model_name}")
