import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import cross_validate
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score, f1_score, make_scorer
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import matthews_corrcoef
import datetime

from H_helpers import safe_div

MODEL_N_JOBS=int(os.getenv("MODEL_N_JOBS", "-1"))

# Good note, standard deviation of any accuracies is 0.5 achieved by having a perfectly split accuracy set of all correct and all correct instances.

def classification_accuracy(predictions, actuals) -> tuple[float, float]:
    if (type(predictions)!='numpy.array'):
        predictions=(pd.Series(predictions)>0).astype(int).to_numpy()
    avg_pred_direction=np.mean(predictions)
    accuracy=np.mean(predictions==actuals)
    return accuracy, avg_pred_direction

def display_feat_importances_tree(model, X: pd.DataFrame, n: int =50) -> pd.DataFrame:
    model_feature_df=pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    model_feature_df.head(n=n).plot(kind='barh', x="Feature", y="Importance")
    plt.gca().invert_yaxis()
    plt.show()
    plt.close()
    return model_feature_df

def display_coef_importances_regression(model, X: pd.DataFrame, n: int =50) -> pd.DataFrame:
    importances=np.abs(model.coef_[0]) 
    model_coef_df=pd.DataFrame({
        'Feature': X.columns,
        'Coef': importances
    }).sort_values(by='Importance', ascending=False)
    model_coef_df.head(n=n).plot(kind='barh', x="Feature", y="Coefficient")
    plt.gca().invert_yaxis()
    plt.show()
    plt.close()
    return model_coef_df

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

def rank_models_by_metrics(results, criteria=None) -> pd.DataFrame:
    """Rank models by multiple criteria and return average-rank ordering.

    By default, models are ranked using:
    - test accuracy
    - test sensitivity
    - test specificity
    - test ROC-AUC

    Rank 1 is best for each metric, and larger ranks are worse.
    """
    if criteria is None:
        # All default ranking metrics are "higher is better", so
        # ascending=False gives rank 1 to the best model on each metric.
        criteria = {
            'test_split_accuracy': False,
            'test_roc_auc_macro': False,
            'test_sensitivity_macro': False,
            'test_specificity_macro': False,
        }

    if isinstance(results, pd.DataFrame):
        ranked_df = results.copy()
    else:
        ranked_df = pd.DataFrame(results)

    for metric_name, ascending in criteria.items():
        if metric_name not in ranked_df.columns:
            raise KeyError(f"Missing metric for ranking: {metric_name}")
        rank_col = f"rank_{metric_name}"
        ranked_df[rank_col] = ranked_df[metric_name].rank(
            method='average', ascending=ascending
        )

    rank_cols = [f"rank_{metric_name}" for metric_name in criteria]
    ranked_df['average_rank'] = ranked_df[rank_cols].mean(axis=1)
    if 'validation_std_accuracy' in ranked_df.columns:
        # Tie-break lower average rank by preferring the model with the
        # lowest cross-validation accuracy SD.
        return ranked_df.sort_values(
            ['average_rank', 'validation_std_accuracy'],
            ascending=[True, True]
        ).reset_index(drop=True)
    return ranked_df.sort_values('average_rank').reset_index(drop=True)


def gridsearch_curve_data(search, x_param: str) -> dict:
    """Extract a one-parameter CV curve from GridSearchCV at best fixed settings."""
    df = pd.DataFrame(search.cv_results_).copy()
    best_params = search.best_params_

    for param_name, param_value in best_params.items():
        if param_name == x_param:
            continue
        col = f"param_{param_name}"
        if col not in df.columns:
            continue
        df = df[df[col].astype(str) == str(param_value)]

    if df.empty:
        raise ValueError(f"No GridSearchCV rows left for x_param={x_param} after filtering best fixed params.")

    x_raw = df[f"param_{x_param}"].to_list()
    x_num = pd.to_numeric(pd.Series(x_raw), errors="coerce")
    if x_num.notna().all():
        df = df.assign(_x_numeric=x_num.to_numpy(), _x_label=x_num.astype(str).to_numpy())
    else:
        df = df.assign(_x_numeric=np.arange(len(df)), _x_label=pd.Series(x_raw).astype(str).to_numpy())
    df = df.sort_values("_x_numeric")

    return {
        "x_raw": df[f"param_{x_param}"].to_list(),
        "x_numeric": df["_x_numeric"].to_numpy(dtype=float),
        "x_labels": df["_x_label"].to_list(),
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
    """Save best-model bias-variance and direct train/test plots from a GridSearchCV object."""
    curve = gridsearch_curve_data(search, x_param)
    direct = direct_errors_for_grid_param(search, x_param, X_train, y_train, X_test, y_test)
    best_idx = int(np.argmin(curve["cv_bal_err_mean"]))
    one_se_value = search.best_params_[x_param]
    selected_idx = next(
        (i for i, value in enumerate(curve["x_raw"]) if str(value) == str(one_se_value)),
        best_idx,
    )
    rotate_ticks = any(lbl != str(val) for lbl, val in zip(curve["x_labels"], curve["x_numeric"]))

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(f'Bias-Variance Tradeoff — {model_title}', fontsize=13, fontweight='bold')
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
    ax.set_xticks(curve["x_numeric"])
    ax.set_xticklabels(curve["x_labels"], rotation=45 if rotate_ticks else 0, ha='right' if rotate_ticks else 'center')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_bv, dpi=150, bbox_inches='tight')
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    fig2.suptitle(f'Over/Underfitting Analysis — {model_title}', fontsize=13, fontweight='bold')
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
    ax2.set_xticks(direct["x_numeric"])
    ax2.set_xticklabels(direct["x_labels"], rotation=45 if rotate_ticks else 0, ha='right' if rotate_ticks else 'center')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_direct, dpi=150, bbox_inches='tight')
    plt.close()


def comparison_row_from_metrics(model_name: str, metrics: dict) -> dict:
    """Format one model's metrics into the report-table columns used in base.py."""
    return {
        'Model': model_name,
        'Test Acc': metrics['test_split_accuracy'],
        'Precision': metrics['test_precision'],
        'Recall': metrics['test_recall'],
        'Specificity': metrics['test_specificity'],
        'F1': metrics['test_f1'],
        'ROC-AUC': metrics['test_roc_auc_macro'],
        'CV Acc SD': metrics['validation_std_accuracy'],
    }


def build_base_style_comparison_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows).set_index('Model')
    keep_cols = ['Test Acc', 'Precision', 'Recall', 'Specificity', 'F1', 'ROC-AUC', 'CV Acc SD']
    return df[keep_cols]


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


def write_base_style_latex_table(df: pd.DataFrame, path, caption: str, label: str, note: str) -> None:
    """Write a simple LaTeX comparison table using the base.py reporting columns."""
    col_fmt = 'l' + 'r' * len(df.columns)
    col_header = ' & '.join(['Model'] + list(df.columns)) + r' \\'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(r'\begin{table}[htbp]' + '\n')
        f.write(r'\centering' + '\n')
        f.write(r'\caption{' + _latex_escape(caption) + '}\n')
        f.write(r'\label{' + _latex_escape(label) + '}\n')
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

def get_final_metrics(model_obj, X_train, y_train, X_test, y_test, n_splits: int =5, label: str | None =None) -> dict:
    # Previous temporary change used `KFold(n_splits=5, shuffle=False)`.
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scoring_metrics = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'sensitivity': make_scorer(recall_score, zero_division=0),
        'specificity': make_scorer(_specificity_score),
    }
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

    conf_mat=confusion_matrix(y_test, preds)
    up_precision=round(safe_div(conf_mat[1][1], conf_mat[1][1] + conf_mat[0][1]), 3)
    down_precision=round(safe_div(conf_mat[0][0], conf_mat[0][0] + conf_mat[1][0]), 3)
    misclassification_error=1 - final_score
    recall_macro=recall_score(y_test, preds, average='macro', zero_division=0)
    specificity_macro=_macro_specificity_from_confusion(conf_mat)
    roc_auc_macro=roc_auc_score(y_test, y_score)
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
    print(f"Test ROC-AUC (macro):                 {roc_auc_macro:.4f}")
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
