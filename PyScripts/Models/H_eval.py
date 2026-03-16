import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from pathlib import Path
from sklearn.model_selection import cross_validate
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score, f1_score, make_scorer
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import matthews_corrcoef
import datetime

from H_helpers import safe_div
from model_grids import TIME_SERIES_CV_SPLITS

MODEL_N_JOBS=int(os.getenv("MODEL_N_JOBS", "-1"))

# Good note, standard deviation of any accuracies is 0.5 achieved by having a perfectly split accuracy set of all correct and all correct instances.

def classification_accuracy(predictions, actuals) -> tuple[float, float]:
    predictions = np.asarray(predictions)
    if predictions.ndim != 1:
        predictions = predictions.ravel()
    unique_predictions = pd.unique(predictions)
    if not set(unique_predictions).issubset({0, 1, False, True}):
        predictions = (predictions >= 0).astype(int)
    else:
        predictions = predictions.astype(int, copy=False)
    avg_pred_direction=np.mean(predictions)
    accuracy=np.mean(predictions==actuals)
    return accuracy, avg_pred_direction

def _get_score_values(model_obj, X_test):
    if hasattr(model_obj, "predict_proba"):
        proba = model_obj.predict_proba(X_test)
        if np.ndim(proba) == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return np.ravel(proba)
    if hasattr(model_obj, "decision_function"):
        return model_obj.decision_function(X_test)
    return model_obj.predict(X_test)

def _macro_specificity_from_confusion(conf_mat: np.ndarray) -> float:
    specificities = []
    total = conf_mat.sum()
    for class_idx in range(conf_mat.shape[0]):
        tp = conf_mat[class_idx, class_idx]
        fp = conf_mat[:, class_idx].sum() - tp
        fn = conf_mat[class_idx, :].sum() - tp
        tn = total - tp - fp - fn
        specificities.append(safe_div(tn, tn + fp))
    return float(np.mean(specificities))

def _specificity_score(y_true, y_pred) -> float:
    conf_mat = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = conf_mat[0, 0], conf_mat[0, 1]
    return safe_div(tn, tn + fp)


def _safe_roc_auc_score(y_true, y_score) -> float:
    y_true = np.asarray(y_true)
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))

def rank_models_by_metrics(results, criteria=None) -> pd.DataFrame:
    """Rank models by multiple criteria and return average-rank ordering.

    By default, models are ranked using the final holdout test metrics for
    every tuned candidate model passed into this function. The default weights
    are:
    - test ROC-AUC: weight 1.5
    - test sensitivity: weight 1
    - test specificity: weight 1
    - test accuracy: weight 0.5

    Rank 1 is best for each metric, and larger ranks are worse. The reported
    `average_rank` is a weighted average of those metric-specific ranks, so the
    final "best model" selection compares all optimized models in the caller's
    comparison table while prioritizing threshold-free ranking performance
    without letting ROC-AUC dominate the thresholded class-balance metrics.
    """
    if criteria is None:
        # All default ranking metrics are "higher is better", so
        # ascending=False gives rank 1 to the best model on each metric.
        criteria = {
            'test_split_accuracy': {'ascending': False, 'weight': 0.5},
            'test_roc_auc_macro': {'ascending': False, 'weight': 1.5},
            'test_sensitivity_macro': {'ascending': False, 'weight': 1.0},
            'test_specificity_macro': {'ascending': False, 'weight': 1.0},
        }

    if isinstance(results, pd.DataFrame):
        ranked_df = results.copy()
    else:
        ranked_df = pd.DataFrame(results)

    normalized_criteria = {}
    for metric_name, config in criteria.items():
        if isinstance(config, dict):
            ascending = config.get('ascending', False)
            weight = float(config.get('weight', 1.0))
        else:
            ascending = config
            weight = 1.0
        normalized_criteria[metric_name] = {'ascending': ascending, 'weight': weight}

    for metric_name, config in normalized_criteria.items():
        if metric_name not in ranked_df.columns:
            raise KeyError(f"Missing metric for ranking: {metric_name}")
        rank_col = f"rank_{metric_name}"
        ranked_df[rank_col] = ranked_df[metric_name].rank(
            method='average', ascending=config['ascending']
        )

    weighted_rank_sum = 0.0
    total_weight = 0.0
    for metric_name, config in normalized_criteria.items():
        weighted_rank_sum += ranked_df[f"rank_{metric_name}"] * config['weight']
        total_weight += config['weight']
    ranked_df['average_rank'] = weighted_rank_sum / total_weight
    if 'validation_std_accuracy' in ranked_df.columns:
        # Tie-break lower average rank by preferring the model with the
        # lowest cross-validation accuracy SD.
        return ranked_df.sort_values(
            ['average_rank', 'validation_std_accuracy'],
            ascending=[True, True]
        ).reset_index(drop=True)
    return ranked_df.sort_values('average_rank').reset_index(drop=True)


def select_non_degenerate_plot_model(
    ranked_df: pd.DataFrame,
    available_models=None,
    degenerate_col: str = 'is_degenerate_classifier',
) -> str:
    """Return the highest-ranked plottable model that is not degenerate."""
    candidate_df = ranked_df.copy()
    if available_models is not None:
        candidate_df = candidate_df[candidate_df['Model'].isin(available_models)].copy()
    if candidate_df.empty:
        raise ValueError("No ranked models are available for plotting.")

    effective_deg_col = degenerate_col
    if effective_deg_col not in candidate_df.columns and 'Degenerate' in candidate_df.columns:
        effective_deg_col = 'Degenerate'

    if effective_deg_col in candidate_df.columns:
        non_degenerate_df = candidate_df.loc[~candidate_df[effective_deg_col].fillna(False).astype(bool)].copy()
        if non_degenerate_df.empty:
            raise ValueError("No non-degenerate ranked model is available for plotting.")
        candidate_df = non_degenerate_df

    return str(candidate_df.iloc[0]['Model'])


def _uses_discrete_hyperparameter_axis(x_param: str) -> bool:
    """Treat selected tuned params as ordered categories, not continuous distances."""
    return "n_components" in x_param and any(token in x_param for token in ("pca", "reducer"))


def _has_crowded_numeric_spacing(x_vals: np.ndarray) -> bool:
    """Detect grids where linear spacing bunches points into a small region."""
    x_vals = np.asarray(x_vals, dtype=float)
    if x_vals.size < 4:
        return False
    diffs = np.diff(np.sort(x_vals))
    positive_diffs = diffs[diffs > 0]
    if positive_diffs.size < 2:
        return False
    return float(np.max(positive_diffs) / np.min(positive_diffs)) >= 8.0


def _looks_log_spaced(x_vals: np.ndarray) -> bool:
    """Detect positive grids that are better shown on a log axis."""
    x_vals = np.asarray(x_vals, dtype=float)
    if x_vals.size < 4 or np.any(x_vals <= 0):
        return False
    log_diffs = np.diff(np.log10(np.sort(x_vals)))
    positive_log_diffs = log_diffs[log_diffs > 0]
    if positive_log_diffs.size < 2:
        return False
    return float(np.max(positive_log_diffs) / np.min(positive_log_diffs)) <= 2.5


def _choose_numeric_axis_mode(x_param: str, x_vals: np.ndarray) -> str:
    """Pick a readable axis representation for tuned numeric parameters."""
    if _uses_discrete_hyperparameter_axis(x_param):
        return "categorical"
    if _looks_log_spaced(x_vals):
        return "log"
    if _has_crowded_numeric_spacing(x_vals):
        return "categorical"
    return "linear"


def _effective_rf_max_features(value, n_features: int) -> float:
    n_features = max(1, int(n_features))
    if value == "sqrt":
        return float(max(1, int(np.sqrt(n_features))))
    if value == "log2":
        return float(max(1, int(np.log2(n_features))))
    try:
        numeric_value = float(value)
    except Exception:
        return float("inf")
    if 0.0 < numeric_value <= 1.0:
        return float(max(1, int(numeric_value * n_features)))
    return numeric_value


def _format_rf_max_features_label(value, n_features: int) -> str:
    effective = int(_effective_rf_max_features(value, n_features))
    return f"{value} ({effective})"


def _effective_svm_gamma(value, n_features: int) -> float:
    n_features = max(1, int(n_features))
    if value in {"scale", "auto"}:
        # With StandardScaler immediately before SVC, `scale` is approximately
        # 1 / n_features, which matches `auto` for ordering purposes.
        return 1.0 / float(n_features)
    try:
        return float(value)
    except Exception:
        return float("inf")


def _complexity_sort_value(search, x_param: str, value) -> float:
    n_features = int(getattr(search, "n_features_in_", 1))
    if x_param.endswith("max_features"):
        return _effective_rf_max_features(value, n_features)
    if x_param.endswith("gamma"):
        return _effective_svm_gamma(value, n_features)
    try:
        return float(value)
    except Exception:
        return float("inf")


def _complexity_axis_label(x_param: str, fallback_label: str) -> str:
    if x_param.endswith("__C") or x_param == "C":
        return (
            "C  (Inverse Regularization Strength)\n"
            "<- High Regularization, Simpler Model      Low Regularization, More Complex ->"
        )
    if "n_components" in x_param:
        return (
            "PCA n_components from tuning grid\n"
            "<- Lower retained variance, Simpler Model      Higher retained variance, More Complex ->"
        )
    if x_param.endswith("max_depth"):
        return (
            "max_depth\n"
            "<- Low Depth, High Regularization, Simpler Model      High Depth, Low Regularization, More Complex ->"
        )
    if x_param.endswith("max_features"):
        return (
            "max_features  (Features considered per split)\n"
            "<- Fewer candidate features, Simpler Model      More candidate features, More Complex ->"
        )
    if x_param.endswith("n_estimators"):
        return (
            "n_estimators\n"
            "<- Smaller forest, Simpler Model      Larger forest, More Complex ->"
        )
    if x_param.endswith("gamma"):
        return (
            "gamma  (Kernel locality)\n"
            "<- Smoother decision boundary, Simpler Model      More local boundary, More Complex ->"
        )
    if x_param.endswith("degree"):
        return (
            "Polynomial degree\n"
            "<- Lower-order boundary, Simpler Model      Higher-order boundary, More Complex ->"
        )
    return fallback_label


def _param_values_match(series: pd.Series, target_value) -> pd.Series:
    """Match GridSearchCV parameter values robustly across numeric/string dtypes."""
    target_numeric = pd.to_numeric(pd.Series([target_value]), errors="coerce").iloc[0]
    series_numeric = pd.to_numeric(series, errors="coerce")
    if pd.notna(target_numeric) and series_numeric.notna().all():
        return np.isclose(series_numeric.to_numpy(dtype=float), float(target_numeric))
    return series.astype(str) == str(target_value)


def _filter_curve_rows_by_best_params(df: pd.DataFrame, best_params: dict, x_param: str) -> pd.DataFrame:
    """Filter GridSearchCV rows to the slice matching the selected fixed params.

    Keep the historical column-wise filter first so existing successful model
    code keeps the same behavior. If mixed/masked parameter columns remove
    every row, fall back to the canonical `params` dict stored in cv_results_.
    """
    filtered_df = df.copy()
    for param_name, param_value in best_params.items():
        if param_name == x_param:
            continue
        col = f"param_{param_name}"
        if col not in filtered_df.columns:
            continue
        filtered_df = filtered_df[_param_values_match(filtered_df[col], param_value)]

    if not filtered_df.empty:
        return filtered_df

    if "params" not in df.columns:
        return filtered_df

    fallback_mask = df["params"].apply(
        lambda params: isinstance(params, dict)
        and all(params.get(param_name) == param_value for param_name, param_value in best_params.items() if param_name != x_param)
    )
    return df[fallback_mask].copy()


def gridsearch_curve_data(search, x_param: str) -> dict:
    """Extract a one-parameter CV curve from GridSearchCV at best fixed settings."""
    df = pd.DataFrame(search.cv_results_).copy()
    best_params = search.best_params_
    n_features = int(getattr(search, "n_features_in_", 1))
    df = _filter_curve_rows_by_best_params(df, best_params, x_param)

    if df.empty:
        raise ValueError(f"No GridSearchCV rows left for x_param={x_param} after filtering best fixed params.")

    x_raw = df[f"param_{x_param}"].to_list()
    x_complexity = pd.Series([_complexity_sort_value(search, x_param, value) for value in x_raw], dtype=float)
    if x_param.endswith("max_features"):
        x_labels = [_format_rf_max_features_label(value, n_features) for value in x_raw]
        axis_mode = "categorical"
        df = df.assign(
            _x_order=x_complexity.to_numpy(dtype=float),
            _x_label=pd.Series(x_labels, dtype=str).to_numpy(),
        )
        df = df.sort_values(["_x_order", "_x_label"]).reset_index(drop=True)
        df = df.assign(_x_numeric=np.arange(len(df), dtype=float))
    else:
        x_num = pd.to_numeric(pd.Series(x_raw), errors="coerce")
        if x_num.notna().all():
            order_vals = x_complexity.to_numpy(dtype=float)
            axis_mode = _choose_numeric_axis_mode(x_param, order_vals)
            if axis_mode == "categorical":
                df = df.assign(
                    _x_order=order_vals,
                    _x_label=pd.Series(x_raw).astype(str).to_numpy(),
                )
                df = df.sort_values("_x_order").reset_index(drop=True)
                df = df.assign(_x_numeric=np.arange(len(df), dtype=float))
            else:
                df = df.assign(_x_numeric=order_vals, _x_label=pd.Series(x_raw).astype(str).to_numpy())
                df = df.sort_values("_x_numeric")
        else:
            axis_mode = "categorical"
            if np.isfinite(x_complexity).any():
                df = df.assign(_x_order=x_complexity.to_numpy(dtype=float), _x_label=pd.Series(x_raw).astype(str).to_numpy())
                df = df.sort_values(["_x_order", "_x_label"]).reset_index(drop=True)
            else:
                df = df.assign(_x_label=pd.Series(x_raw).astype(str).to_numpy())
            df = df.assign(_x_numeric=np.arange(len(df), dtype=float))

    return {
        "x_raw": df[f"param_{x_param}"].to_list(),
        "x_numeric": df["_x_numeric"].to_numpy(dtype=float),
        "x_labels": df["_x_label"].to_list(),
        "x_axis_mode": axis_mode,
        "train_bal_err_mean": 1 - df["mean_train_score"].to_numpy(dtype=float),
        "train_bal_err_std": df["std_train_score"].to_numpy(dtype=float) if "std_train_score" in df.columns else np.zeros(len(df), dtype=float),
        "cv_bal_err_mean": 1 - df["mean_test_score"].to_numpy(dtype=float),
        "cv_bal_err_std": df["std_test_score"].to_numpy(dtype=float),
    }


def direct_errors_for_grid_param(search, x_param: str, X_train, y_train, X_test, y_test) -> dict:
    """Compute direct train/test plain error while varying one GridSearchCV parameter."""
    curve = gridsearch_curve_data(search, x_param)
    train_errors = []
    test_errors = []
    for x_val in curve["x_raw"]:
        model = clone(search.best_estimator_)
        model.set_params(**{x_param: x_val})
        model.fit(X_train, y_train)
        train_errors.append(1 - model.score(X_train, y_train))
        test_errors.append(1 - model.score(X_test, y_test))
    curve["train_errors"] = np.asarray(train_errors, dtype=float)
    curve["test_errors"] = np.asarray(test_errors, dtype=float)
    return curve


def save_best_model_plots_from_gridsearch(
    search,
    x_param: str,
    x_label: str,
    model_title: str,
    output_bv,
    output_direct,
    X_train,
    y_train,
    X_test,
    y_test,
):
    """Save best-model plots for one tuned parameter from a GridSearchCV object."""
    curve = gridsearch_curve_data(search, x_param)
    direct = direct_errors_for_grid_param(search, x_param, X_train, y_train, X_test, y_test)
    x_label = _complexity_axis_label(x_param, x_label)
    best_idx = int(np.argmin(curve["cv_bal_err_mean"]))
    one_se_value = search.best_params_[x_param]
    selected_idx = next(
        (i for i, value in enumerate(curve["x_raw"]) if str(value) == str(one_se_value)),
        best_idx,
    )
    rotate_ticks = curve["x_axis_mode"] == "categorical" or any(lbl != str(val) for lbl, val in zip(curve["x_labels"], curve["x_numeric"]))

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(f'Bias-Variance Tradeoff — {model_title}', fontsize=13, fontweight='bold')
    if curve["x_axis_mode"] == "log":
        ax.semilogx(curve["x_numeric"], curve["train_bal_err_mean"], marker='o', color='steelblue', linewidth=1.8, label='CV Train balanced error')
        ax.semilogx(curve["x_numeric"], curve["cv_bal_err_mean"], marker='s', color='darkorange', linewidth=1.8, label='CV Test balanced error')
    else:
        ax.plot(curve["x_numeric"], curve["train_bal_err_mean"], marker='o', color='steelblue', linewidth=1.8, label='CV Train balanced error')
        ax.plot(curve["x_numeric"], curve["cv_bal_err_mean"], marker='s', color='darkorange', linewidth=1.8, label='CV Test balanced error')
    ax.fill_between(
        curve["x_numeric"],
        np.clip(curve["train_bal_err_mean"] - curve["train_bal_err_std"], 0.0, 1.0),
        np.clip(curve["train_bal_err_mean"] + curve["train_bal_err_std"], 0.0, 1.0),
        alpha=0.15, color='steelblue', label='CV Train balanced error ±1 SD'
    )
    ax.fill_between(
        curve["x_numeric"],
        np.clip(curve["cv_bal_err_mean"] - curve["cv_bal_err_std"], 0.0, 1.0),
        np.clip(curve["cv_bal_err_mean"] + curve["cv_bal_err_std"], 0.0, 1.0),
        alpha=0.15, color='darkorange', label='CV Test balanced error ±1 SD'
    )
    ax.scatter(
        [curve["x_numeric"][best_idx]], [curve["cv_bal_err_mean"][best_idx]],
        color='gold', edgecolor='black', s=90, zorder=6,
        label='Value at best CV balanced error'
    )
    ax.axvline(
        float(curve["x_numeric"][selected_idx]), color='red', linestyle='--', linewidth=1.5,
        label=f'1SE-selected value = {one_se_value}'
    )
    ax.set_title(f'{model_title} — Bias-Variance Tradeoff')
    ax.set_xlabel(x_label)
    ax.set_ylabel('Balanced Error (1 - balanced accuracy)')
    ax.set_ylim(0, 1.02)
    if curve["x_axis_mode"] != "log" or rotate_ticks:
        ax.set_xticks(curve["x_numeric"])
        ax.set_xticklabels(curve["x_labels"], rotation=45 if rotate_ticks else 0, ha='right' if rotate_ticks else 'center')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_bv, dpi=150, bbox_inches='tight')
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    fig2.suptitle(f'Over/Underfitting Analysis — {model_title}', fontsize=13, fontweight='bold')
    if direct["x_axis_mode"] == "log":
        ax2.semilogx(direct["x_numeric"], direct["train_errors"], marker='o', color='steelblue', linewidth=2, label='Train error')
        ax2.semilogx(direct["x_numeric"], direct["test_errors"], marker='s', color='darkorange', linewidth=2, label='Test error')
    else:
        ax2.plot(direct["x_numeric"], direct["train_errors"], marker='o', color='steelblue', linewidth=2, label='Train error')
        ax2.plot(direct["x_numeric"], direct["test_errors"], marker='s', color='darkorange', linewidth=2, label='Test error')
    ax2.scatter(
        [direct["x_numeric"][best_idx]], [direct["test_errors"][best_idx]],
        color='gold', edgecolor='black', s=90, zorder=6,
        label='Value at best CV balanced error'
    )
    ax2.axvline(
        float(direct["x_numeric"][selected_idx]), color='red', linestyle='--', linewidth=1.5,
        label=f'1SE-selected value = {one_se_value}'
    )
    ax2.set_title(f'{model_title} — Train vs Test Error')
    ax2.set_xlabel(x_label)
    ax2.set_ylabel('Plain Error (1 - accuracy)')
    if direct["x_axis_mode"] != "log" or rotate_ticks:
        ax2.set_xticks(direct["x_numeric"])
        ax2.set_xticklabels(direct["x_labels"], rotation=45 if rotate_ticks else 0, ha='right' if rotate_ticks else 'center')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_direct, dpi=150, bbox_inches='tight')
    plt.close()


def _param_plot_suffix(x_param: str) -> str:
    return x_param.replace("__", "_").replace("param_", "").replace(" ", "_")


def _append_plot_suffix(path_like, suffix: str):
    path = Path(path_like)
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def save_best_model_plots_from_gridsearch_all_params(
    search,
    param_specs: list[tuple[str, str]],
    model_title: str,
    output_bv,
    output_direct,
    X_train,
    y_train,
    X_test,
    y_test,
):
    """Save best-model plots for every tuned parameter of the selected search."""
    if not param_specs:
        raise ValueError("param_specs must contain at least one (x_param, x_label) pair.")

    saved_paths = []
    for idx, (x_param, x_label) in enumerate(param_specs):
        if idx == 0:
            bv_path = output_bv
            direct_path = output_direct
        else:
            suffix = _param_plot_suffix(x_param)
            bv_path = _append_plot_suffix(output_bv, suffix)
            direct_path = _append_plot_suffix(output_direct, suffix)
        save_best_model_plots_from_gridsearch(
            search,
            x_param,
            x_label,
            model_title,
            bv_path,
            direct_path,
            X_train,
            y_train,
            X_test,
            y_test,
        )
        saved_paths.append((str(bv_path), str(direct_path)))
    return saved_paths


def comparison_row_from_metrics(model_name: str, metrics: dict) -> dict:
    """Format one model's metrics into the report-table columns used in base.py."""
    return {
        'Model': model_name,
        'Test Acc': metrics['test_split_accuracy'],
        'MCC': metrics['test_matthew_corr_coef'],
        'Precision': metrics['test_precision'],
        'Sensitivity (Macro)': metrics['test_sensitivity_macro'],
        'Specificity': metrics['test_specificity'],
        'F1': metrics['test_f1'],
        'ROC-AUC': metrics['test_roc_auc_macro'],
        'CV Acc SD': metrics['validation_std_accuracy'],
    }


def build_base_style_comparison_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows).set_index('Model')
    # Hidden for report export for now: Precision, F1, CV Acc SD
    keep_cols = ['ROC-AUC', 'MCC', 'Test Acc', 'Sensitivity (Macro)', 'Specificity']
    return df[keep_cols]


def build_compact_export_table(
    df: pd.DataFrame,
    keep_cols: list[str] | None = None,
    index_renames: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Keep only compact reporting columns and optionally rename rows for export."""
    if keep_cols is None:
        # Hidden for report export for now: Precision, F1, CV Acc SD
        keep_cols = ['ROC-AUC', 'MCC', 'Test Acc', 'Sensitivity (Macro)', 'Specificity']
    export_df = df.copy()[keep_cols]
    if index_renames:
        export_df = export_df.rename(index=index_renames)
    return export_df


def register_global_model_candidates(
    ranking_df: pd.DataFrame,
    leaderboard_path,
    source_script: str,
    dataset_label: str,
    comparison_scope: str,
    reset_leaderboard: bool = False,
) -> pd.DataFrame:
    """Persist tuned candidate metrics and return the current global ranking.

    This helper is informational only. It does not choose the plotted/exported
    winner for a script; each caller should decide its own local winner from its
    own `ranking_df` before or independently of writing to the global
    leaderboard.
    """
    leaderboard_path = Path(leaderboard_path)
    required_cols = [
        'Model',
        'test_split_accuracy',
        'test_roc_auc_macro',
        'test_sensitivity_macro',
        'test_specificity_macro',
        'validation_std_accuracy',
    ]
    missing = [col for col in required_cols if col not in ranking_df.columns]
    if missing:
        raise KeyError(f"Missing required leaderboard columns: {missing}")

    export_df = ranking_df.copy()
    export_df.insert(0, 'source_script', source_script)
    export_df.insert(1, 'dataset_label', dataset_label)
    export_df.insert(2, 'comparison_scope', comparison_scope)
    export_df.insert(3, 'candidate_model', export_df['Model'])
    export_df['Model'] = export_df['source_script'] + " :: " + export_df['candidate_model']

    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    ranked_path = leaderboard_path.with_name(f"{leaderboard_path.stem}_ranked{leaderboard_path.suffix}")

    if reset_leaderboard:
        combined_df = export_df
    elif leaderboard_path.exists():
        existing_df = pd.read_csv(leaderboard_path)
        required_metadata_cols = {'source_script', 'dataset_label', 'comparison_scope'}
        if not required_metadata_cols.issubset(existing_df.columns):
            combined_df = export_df
        else:
            replace_mask = (
                (existing_df['source_script'] == source_script)
                & (existing_df['dataset_label'] == dataset_label)
                & (existing_df['comparison_scope'] == comparison_scope)
            )
            existing_df = existing_df.loc[~replace_mask].copy()
            combined_df = pd.concat([existing_df, export_df], ignore_index=True, sort=False)
    else:
        combined_df = export_df

    combined_df.to_csv(leaderboard_path, index=False, float_format='%.3f')
    ranked_df = rank_models_by_metrics(combined_df)
    ranked_df.to_csv(ranked_path, index=False, float_format='%.3f')
    return ranked_df


def _latex_escape(text: str) -> str:
    return (str(text)
            .replace("\\", r"\textbackslash{}")
            .replace("&", r"\&")
            .replace("%", r"\%")
            .replace("_", r"\_")
            .replace("#", r"\#")
            .replace("$", r"\$")
            .replace("{", r"\{")
            .replace("}", r"\}"))


def _latex_label(text: str) -> str:
    """Keep LaTeX labels machine-safe without escaping underscores."""
    return str(text).strip().replace(" ", "_")


def write_base_style_latex_table(df: pd.DataFrame, path, caption: str, label: str, note: str) -> None:
    """Write a simple LaTeX comparison table using the base.py reporting columns."""
    col_fmt = 'l' + 'r' * len(df.columns)
    col_header = ' & '.join(['Model'] + list(df.columns)) + r' \\'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(r'\begin{table}[htbp]' + '\n')
        f.write(r'\centering' + '\n')
        f.write(r'\caption{' + _latex_escape(caption) + '}\n')
        f.write(r'\label{' + _latex_label(label) + '}\n')
        f.write(r'\begin{tabular}{' + col_fmt + '}\n')
        f.write(r'\toprule' + '\n')
        f.write(col_header + '\n')
        f.write(r'\midrule' + '\n')
        for name, row in df.iterrows():
            formatted_vals = [f'{v:.3f}' for v in row.values]
            f.write(_latex_escape(name) + ' & ' + ' & '.join(formatted_vals) + r' \\' + '\n')
        f.write(r'\bottomrule' + '\n')
        f.write(r'\end{tabular}' + '\n')
        f.write(r'\par\smallskip' + '\n')
        f.write(r'\footnotesize ' + _latex_escape(note) + '\n')
        f.write(r'\end{table}' + '\n')


def write_grouped_latex_table(
    df: pd.DataFrame,
    path,
    caption: str,
    label: str,
    note: str,
    groups: list[pd.Index | list[str] | tuple[str, ...]] | None = None,
    degenerate_models: set[str] | None = None,
    degenerate_note: str | None = None,
    escape_note: bool = True,
    escape_degenerate_note: bool = True,
) -> None:
    """Write a LaTeX comparison table with optional row groups and dagger marks."""
    col_fmt = 'l' + 'r' * len(df.columns)
    col_header = ' & '.join(['Model'] + list(df.columns)) + r' \\'
    group_list = groups if groups is not None else [list(df.index)]
    normalized_groups = []
    for group in group_list:
        ordered_names = [name for name in list(group) if name in df.index]
        if ordered_names:
            normalized_groups.append(ordered_names)
    if not normalized_groups:
        normalized_groups = [list(df.index)]

    degenerate_models = degenerate_models or set()

    with open(path, 'w', encoding='utf-8') as f:
        f.write(r'\begin{table}[htbp]' + '\n')
        f.write(r'\centering' + '\n')
        f.write(r'\caption{' + _latex_escape(caption) + '}\n')
        f.write(r'\label{' + _latex_label(label) + '}\n')
        f.write(r'\begin{tabular}{' + col_fmt + '}\n')
        f.write(r'\toprule' + '\n')
        f.write(col_header + '\n')
        f.write(r'\midrule' + '\n')
        for group_idx, row_names in enumerate(normalized_groups):
            for name in row_names:
                row = df.loc[name]
                formatted_vals = [f'{v:.3f}' if isinstance(v, (float, int, np.floating, np.integer)) else _latex_escape(v) for v in row.values]
                dagger = r'$^\dagger$' if name in degenerate_models else ''
                f.write(_latex_escape(name) + dagger + ' & ' + ' & '.join(formatted_vals) + r' \\' + '\n')
            if group_idx < len(normalized_groups) - 1:
                f.write(r'\midrule' + '\n')
        f.write(r'\bottomrule' + '\n')
        f.write(r'\end{tabular}' + '\n')
        f.write(r'\par\smallskip' + '\n')
        rendered_note = _latex_escape(note) if escape_note else note
        f.write(r'\footnotesize ' + rendered_note + '\n')
        if degenerate_models and degenerate_note:
            rendered_deg_note = _latex_escape(degenerate_note) if escape_degenerate_note else degenerate_note
            f.write(r'\\' + '\n')
            f.write(r'\footnotesize ' + rendered_deg_note + '\n')
        f.write(r'\end{table}' + '\n')

def get_final_metrics(model_obj, X_train, y_train, X_test, y_test, n_splits: int =TIME_SERIES_CV_SPLITS, label: str | None =None) -> dict:
    # Previous temporary change used `KFold(n_splits=5, shuffle=False)`.
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scoring_metrics = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, zero_division=0),
        'sensitivity': make_scorer(recall_score, zero_division=0),
        'specificity': make_scorer(_specificity_score),
    }
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Setting penalty=None will ignore the C and l1_ratio parameters",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*'penalty' was deprecated.*use 'l1_ratio' or 'C' instead.*",
            category=FutureWarning,
        )
        cv_results=cross_validate(
            model_obj,
            X_train,
            y_train,
            cv=tscv,
            scoring=scoring_metrics,
            return_train_score=True,
            n_jobs=MODEL_N_JOBS
        )

    mean_train=np.mean(cv_results['train_accuracy'])
    std_train=np.std(cv_results['train_accuracy'])

    mean_cv_test=np.mean(cv_results['test_accuracy'])
    std_cv_test=np.std(cv_results['test_accuracy'])

    mean_test_recall=np.mean(cv_results['test_sensitivity'])
    mean_test_precision=np.mean(cv_results['test_precision'])
    mean_test_specificity=np.mean(cv_results['test_specificity'])

    final_score=model_obj.score(X_test, y_test)

    preds=model_obj.predict(X_test)
    y_score=_get_score_values(model_obj, X_test)
    is_degenerate_classifier = bool(np.unique(preds).size == 1)

    conf_mat=confusion_matrix(y_test, preds, labels=[0, 1])
    up_precision=round(safe_div(conf_mat[1][1], conf_mat[1][1] + conf_mat[0][1]), 3)
    down_precision=round(safe_div(conf_mat[0][0], conf_mat[0][0] + conf_mat[1][0]), 3)
    misclassification_error=1 - final_score
    recall_macro=recall_score(y_test, preds, average='macro', zero_division=0)
    specificity_macro=_macro_specificity_from_confusion(conf_mat)
    roc_auc_macro=_safe_roc_auc_score(y_test, y_score)
    test_precision=precision_score(y_test, preds, zero_division=0)
    test_recall=recall_score(y_test, preds, zero_division=0)
    test_specificity=safe_div(conf_mat[0][0], conf_mat[0][0] + conf_mat[0][1])
    test_f1=f1_score(y_test, preds, zero_division=0)

    true_up_rate=round(np.mean(y_test), 3)
    true_down_rate=1 - true_up_rate 

    print(f"--- Model Report ---")
    print(f"Avg CV Train Plain Accuracy:      {mean_train:.4f} (±{std_train:.4f})")
    print(f"Avg CV Validation Plain Accuracy: {mean_cv_test:.4f} (±{std_cv_test:.4f})") # This validation score is computed from plain accuracy on time-series CV splits after balanced-accuracy tuning.
    print(f"Avg CV Validation Precision: {mean_test_precision:.4}")
    print(f"Avg CV Validation Recall:      {mean_test_recall:.4}")
    print(f"Avg CV Validation Specificity: {mean_test_specificity:.4}")
    print(f"Final Test Accuracy:    {final_score:.4f}")
    print(f"Test Plain Misclassification Error:    {misclassification_error:.4f}")
    roc_auc_display = f"{roc_auc_macro:.4f}" if not np.isnan(roc_auc_macro) else "N/A (single-class holdout)"
    print(f"Test ROC-AUC (macro):                 {roc_auc_display}")
    print(f"Test Recall:                          {test_recall:.4f}")
    print(f"Test Specificity:                     {test_specificity:.4f}")
    print(f"Test Recall (macro):                  {recall_macro:.4f}")
    print(f"Test Specificity (macro):             {specificity_macro:.4f}")
    print(f"Positive Prediction Rate (Test):        {np.mean(preds):.4f}")
    print(f"True Up Rate (Test):   {true_up_rate:.4f}")
    print(f"Up Precision (Test):   {up_precision:.4f}") # How correct is it when it guesses up
    print(f"Up Edge Rate (Test):   {up_precision - true_up_rate:.4f}") # How much more correct is it than guessing the average up rate
    print(f"True Down Rate (Test): {true_down_rate:.4f}") 
    print(f"Down Precision (Test): {down_precision:.4f}") # How correct is it when it guesses down
    print(f"Down Edge Rate (Test): {down_precision - true_down_rate:.4f}") # How much more correct is it than guessing the average down rate (1 - up rate)
    print(pd.DataFrame(conf_mat, index=["Actual Down", "Actual Up"], columns=["Predicted Down", "Predicted Up"]))  

    model_name=model_obj.__class__.__name__
    if (model_name=="Pipeline"):
        model_name=model_obj.named_steps['classifier'].__class__.__name__
    return {
        "model_name": model_name,
        "label": label,
        "time_ran": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_avg_accuracy": round(mean_train, 3),
        "train_std_accuracy": round(std_train, 3),
        "validation_avg_accuracy": round(mean_cv_test, 3),
        "validation_std_accuracy": round(std_cv_test, 3),
        "cv_test_sd_error": round(std_cv_test, 3),
        "validation_avg_precision": round(mean_test_precision, 3),
        "validation_avg_recall": round(mean_test_recall, 3),
        "validation_avg_sensitivity": round(mean_test_recall, 3),
        "validation_avg_specificity": round(mean_test_specificity, 3),
        "test_split_accuracy": round(final_score, 3),
        "test_misclassification_error": round(misclassification_error, 3),
        "test_roc_auc_macro": round(roc_auc_macro, 3),
        "test_recall_macro": round(recall_macro, 3),
        "test_sensitivity_macro": round(recall_macro, 3),
        "test_specificity_macro": round(specificity_macro, 3),
        "test_precision": round(test_precision, 3),
        "test_recall": round(test_recall, 3),
        "test_sensitivity": round(test_recall, 3),
        "test_specificity": round(test_specificity, 3),
        "test_f1": round(test_f1, 3),
        "is_degenerate_classifier": is_degenerate_classifier,
        "test_split_positive_prediction_rate": round(np.mean(preds), 3),
        "test_split_true_up_rate": true_up_rate,
        "test_true_down": conf_mat[0][0],
        "test_true_up": conf_mat[1][1],
        "test_negative_down": conf_mat[1][0],
        "test_negative_up": conf_mat[0][1],
        "test_up_precision": up_precision,
        "test_up_recall": round(safe_div(conf_mat[1][1], conf_mat[1][1] + conf_mat[1][0]), 3),
        "test_down_precision": down_precision,
        "test_down_recall": round(safe_div(conf_mat[0][0], conf_mat[0][0] + conf_mat[0][1]), 3),
        "test_matthew_corr_coef": round(matthews_corrcoef(preds, y_test), 3),
        "train_rows": X_train.shape[0],
        "train_cols": X_train.shape[1],
        "test_rows": X_test.shape[0],
        "test_cols": X_test.shape[1],
    }

class RollingWindowBacktest:
    def __init__(self, model: BaseEstimator, X: pd.DataFrame | None =None, y: pd.DataFrame | None =None, X_train: pd.DataFrame =None, window_size: int | None =None, horizon: int | None = None):
        self.model=model
        self.X=X
        self.y=y
        self.X_train=X_train
        self.window_size=window_size
        self.horizon=horizon
        self.results=None
    
    def rolling_window_backtest(self, verbose: int =1):
        train_accuracy=[]
        train_avg_direction=[]
        test_accuracy=[]
        test_avg_direction=[]

        if (self.window_size == None):
            self.window_size=min(self.X.shape[1] * 2, self.X.shape[0] // 3)
            if (self.window_size == (self.X.shape[0] // 3)):
                print("!!!WARNING!!! Overfitting will most likely occur.")
        if (self.horizon == None):
            self.horizon=self.window_size // 4
        elif (self.horizon > self.window_size): raise ValueError("horizon must be less than window_size")

        n=len(self.X)
        n_train=len(self.X_train)

        start_idx=max(self.window_size, n_train)
        total_iterations = max(0, ((n - self.horizon) - start_idx) // self.horizon + 1)
        if (verbose != 0): print(f"Rolling Window Backtest over {total_iterations} iterations.")
        current_step=0
        for i in range(start_idx, n - self.horizon + 1, self.horizon):
            current_step += 1
            X_train_roll=self.X.iloc[i-self.window_size : i]
            y_train_roll=self.y.iloc[i-self.window_size : i]
            
            X_test_roll=self.X.iloc[i : i+self.horizon]
            y_test_roll=self.y.iloc[i : i+self.horizon]
            
            self.model.fit(X_train_roll, y_train_roll)
            preds=self.model.predict(X_test_roll)
            
            acc, avg =classification_accuracy(preds, y_test_roll)
            if (i >= n_train):
                test_accuracy.append(acc)
                test_avg_direction.append(avg)
            else:
                train_accuracy.append(acc)
                train_avg_direction.append(avg)
            if (verbose>0 and ((current_step%10) == 0)): print(f"{current_step * 100 / total_iterations:.2f}% complete. Current iteration: {current_step}, True iteration: {i + 1 - self.window_size}")
            
        train_avg_accuracy=np.mean(train_accuracy) if train_accuracy else np.nan
        train_std_accuracy=np.std(train_accuracy) if train_accuracy else np.nan
        test_avg_accuracy=np.mean(test_accuracy) if test_accuracy else np.nan
        test_std_accuracy=np.std(test_accuracy) if test_accuracy else np.nan
        if (verbose > 0):
            if train_accuracy:
                print(f"Average Rolling Plain Accuracy (train) (rwb): {train_avg_accuracy:.4f} (±{train_std_accuracy:.4f})")
            else:
                print("Average Rolling Plain Accuracy (train) (rwb): n/a")
            if test_accuracy:
                print(f"Average Rolling Plain Accuracy (test)  (rwb): {test_avg_accuracy:.4f} (±{test_std_accuracy:.4f})")
            else:
                print("Average Rolling Plain Accuracy (test)  (rwb): n/a")
        self.results=[train_accuracy + test_accuracy, train_avg_direction + test_avg_direction, {
            "mwfv_train_avg_accuracy": round(train_avg_accuracy, 3), # Modified Walk Forward Validation is mwfv
            "mwfv_train_std_accuracy": round(train_std_accuracy, 3),
            "mwfv_test_avg_accuracy": round(test_avg_accuracy, 3),
            "mwfv_test_std_accuracy": round(test_std_accuracy, 3)
        }]
    
    def display_wfv_results(self, extra_metrics: bool =True, comparison_metric: list =None) -> None:
        plt.figure(figsize=(12, 6))
        n_train=len(self.X_train)
        n_total=len(self.X)
        start_idx=max(self.window_size, n_train)
        start_of_each_test=list(range(start_idx, n_total - self.horizon + 1, self.horizon))
        
        plt.plot(start_of_each_test, self.results[0], marker='o', linestyle='-', label='Segment Accuracy')
        plt.plot(start_of_each_test, self.results[1], color='gray',marker='o', linestyle='-', alpha=0.4, label='Prediction Direction')
        plt.plot(start_of_each_test, [0.5 for _ in range(len(start_of_each_test))], linestyle="--", label="Base Line")
        if (extra_metrics):
            in_X_train=[x for x in start_of_each_test if x < n_train]
            in_X_test=[x for x in start_of_each_test if x >= n_train]
            if (in_X_train):
                m_train=self.results[2]["mwfv_train_avg_accuracy"]
                s_train=self.results[2]["mwfv_train_std_accuracy"]
                plt.plot(in_X_train + [n_train], [m_train] * (len(in_X_train) + 1), color="#8EFF32", alpha=0.8, linestyle="--", label="Train Mean")
                plt.fill_between(in_X_train + [n_train], m_train - s_train, m_train + s_train, color="#8EFF32", alpha=0.15)
            if (in_X_test):
                m_test=self.results[2]["mwfv_test_avg_accuracy"]
                s_test=self.results[2]["mwfv_test_std_accuracy"]
                plt.plot([n_train] + in_X_test, [m_test] * (len(in_X_test) + 1), color="#2C8FFF", alpha=0.8, linestyle="--", label="Test Mean")
                plt.fill_between([n_train] + in_X_test, m_test - s_test, m_test + s_test, color="#65ADFF", alpha=0.15)
        
        plt.axvspan(self.window_size, n_train, color='lightblue', alpha=0.3, label='In-Sample Rolling')
        plt.axvspan(n_train, n_total, color='#FFFACD', alpha=0.5, label='Out-of-Sample Test')
        plt.axvline(x=n_train, color='r', linestyle='--', label='Train/Test Split Boundary')

        plt.title("Rolling Window Backtest Plain Accuracy")
        plt.xlabel("Sample Index (Start of Test Horizon)")
        plt.ylabel("Plain Accuracy")
        plt.ylim(0, 1.05)
        plt.grid(alpha=0.3)
        plt.legend()
        
        plt.text(n_train * 0.5, 0.05, 'In-Sample Rolling', horizontalalignment='center', color='gray')
        plt.text(((n_total + n_train) * 0.5), 0.05, 'Out-of-Sample Rolling', horizontalalignment='center', color='gray')
        
        plt.show()
        plt.close('all')

    def set(self, X: pd.DataFrame, y: pd.DataFrame, X_train: pd.DataFrame):
        self.X=X
        self.y=y
        self.X_train=X_train

def utility_score(results: dict, rwb: dict, w: float =4.0):
    return results['test_split_accuracy']

class ModelResults:
    """Class to store and compare classification model results"""
    def __init__(self):
        self.results_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1-Score'])
    
    def add_result(self, model_name: str, accuracy: float, precision: float, sensitivity: float, specificity: float, f1_score: float):
        """Add model results to the comparison dataframe"""
        new_result = pd.DataFrame({
            'Model': [model_name],
            'Accuracy': [accuracy],
            'Precision': [precision],
            'Sensitivity': [sensitivity],
            'Specificity': [specificity],
            'F1-Score': [f1_score]
        })
        self.results_df = pd.concat([self.results_df, new_result], ignore_index=True)
    
    def display_results(self):
        """Display all model results in a formatted table"""
        print("\n" + "="*80)
        print("MODEL COMPARISON RESULTS")
        print("="*80)
        print(self.results_df.to_string(index=False))
        print("="*80 + "\n")
    
    def save_results(self, filepath: str):
        """Save results to CSV file"""
        self.results_df.to_csv(filepath, index=False, float_format='%.3f')
        print(f"Results saved to {filepath}")
    
    def get_best_model(self, metric: str = 'F1-Score'):
        """Get the best model based on specified metric"""
        if metric in self.results_df.columns:
            metric_lower = metric.lower()
            lower_is_better = any(token in metric_lower for token in ("error", "loss", "misclassification"))
            best_idx = self.results_df[metric].idxmin() if lower_is_better else self.results_df[metric].idxmax()
            return self.results_df.loc[best_idx]
        else:
            print(f"Metric '{metric}' not found in results")
