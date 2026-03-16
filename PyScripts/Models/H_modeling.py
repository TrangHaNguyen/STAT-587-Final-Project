#!/usr/bin/env python3
import os
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

from H_search_history import (
    append_search_history,
    history_has_entry,
    load_search_checkpoint,
    save_search_checkpoint,
    search_checkpoint_exists,
)
from model_grids import (
    BASELINE_PCA_GRID,
    LOGISTIC_BASELINE_SOLVER,
    LOGISTIC_CLASS_WEIGHT,
    LOGISTIC_MAX_ITER,
    LOGISTIC_TOL,
    RANDOM_SEED,
)


def as_sortable_numeric(value):
    try:
        return float(value)
    except Exception:
        return float("inf")


def make_one_se_refit(
    complexity_cols: list[str],
    *,
    n_splits: int,
    fixed_cols: list[str] | None = None,
    sort_value_map: dict[str, Callable[[Any], float]] | None = None,
):
    """Return a GridSearchCV refit callable implementing the 1-SE rule."""

    def _pick_index(cv_results):
        mean = np.asarray(cv_results["mean_test_score"], dtype=float)
        std = np.asarray(cv_results["std_test_score"], dtype=float)
        se = std / np.sqrt(n_splits)
        best_idx = int(np.argmax(mean))
        threshold = float(mean[best_idx] - se[best_idx])
        candidate_idx = np.where(mean >= threshold)[0]
        if len(candidate_idx) == 0:
            return best_idx

        if fixed_cols:
            for col in fixed_cols:
                param_key = f"param_{col}"
                best_val = cv_results[param_key][best_idx]
                candidate_idx = np.array(
                    [i for i in candidate_idx if cv_results[param_key][i] == best_val],
                    dtype=int,
                )
                if len(candidate_idx) == 0:
                    return best_idx

        def key_fn(i: int):
            complexity = []
            for col in complexity_cols:
                val = cv_results[f"param_{col}"][i]
                sort_fn = (sort_value_map or {}).get(col, as_sortable_numeric)
                complexity.append(sort_fn(val))
            return tuple(complexity + [-float(mean[i])])

        return int(min(candidate_idx, key=key_fn))

    return _pick_index


def fit_or_load_search(
    *,
    checkpoint_dir,
    stage_name: str,
    search_obj,
    X_train,
    y_train,
    history_path,
    run_time: str,
    model_name: str,
    grid_label: str,
    notes: str = "",
    fit_search: Callable[[Any, Any, Any], None] | None = None,
):
    if search_checkpoint_exists(checkpoint_dir, stage_name):
        print(f"Loading checkpoint for {model_name} from {checkpoint_dir / stage_name}")
        loaded = load_search_checkpoint(checkpoint_dir, stage_name)
        if not history_has_entry(history_path, model_name, grid_label):
            append_search_history(
                history_path=history_path,
                cv_results=loaded.cv_results_,
                run_time=run_time,
                model_name=model_name,
                search_type="grid",
                grid_version=grid_label,
                notes=notes,
                best_params=loaded.best_params_,
            )
        return loaded

    if fit_search is None:
        search_obj.fit(X_train, y_train)
    else:
        fit_search(search_obj, X_train, y_train)

    append_search_history(
        history_path=history_path,
        cv_results=search_obj.cv_results_,
        run_time=run_time,
        model_name=model_name,
        search_type="grid",
        grid_version=grid_label,
        notes=notes,
        best_params=search_obj.best_params_,
    )
    save_search_checkpoint(checkpoint_dir, stage_name, search_obj)
    return search_obj


def _fit_logistic_baseline_pca_search(search_obj, X_train, y_train):
    import warnings

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
        search_obj.fit(X_train, y_train)


def fit_or_load_baseline_logistic_pca_search(
    *,
    checkpoint_dir,
    stage_name: str,
    X_train,
    y_train,
    history_path,
    run_time: str,
    model_name: str,
    grid_label: str,
    n_splits: int,
    notes: str = "",
):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    baseline_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA()),
        ("classifier", LogisticRegression(
            solver=LOGISTIC_BASELINE_SOLVER,
            class_weight=LOGISTIC_CLASS_WEIGHT,
            random_state=RANDOM_SEED,
            max_iter=LOGISTIC_MAX_ITER,
            tol=LOGISTIC_TOL,
            C=np.inf,
        )),
    ])
    param_grid = {"pca__n_components": BASELINE_PCA_GRID}
    search_obj = GridSearchCV(
        baseline_pipeline,
        param_grid,
        cv=tscv,
        return_train_score=True,
        scoring="balanced_accuracy",
        refit=make_one_se_refit(["pca__n_components"], n_splits=n_splits),
    )
    return fit_or_load_search(
        checkpoint_dir=checkpoint_dir,
        stage_name=stage_name,
        search_obj=search_obj,
        X_train=X_train,
        y_train=y_train,
        history_path=history_path,
        run_time=run_time,
        model_name=model_name,
        grid_label=grid_label,
        notes=notes,
        fit_search=_fit_logistic_baseline_pca_search,
    )


def fit_or_load_fixed_classifier_pca_search(
    *,
    checkpoint_dir,
    stage_name: str,
    X_train,
    y_train,
    history_path,
    run_time: str,
    model_name: str,
    grid_label: str,
    n_splits: int,
    classifier,
    notes: str = "",
    fit_search: Callable[[Any, Any, Any], None] | None = None,
):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA()),
        ("classifier", clone(classifier)),
    ])
    search_obj = GridSearchCV(
        pipeline,
        {"pca__n_components": BASELINE_PCA_GRID},
        cv=tscv,
        return_train_score=True,
        scoring="balanced_accuracy",
        refit=make_one_se_refit(["pca__n_components"], n_splits=n_splits),
    )
    return fit_or_load_search(
        checkpoint_dir=checkpoint_dir,
        stage_name=stage_name,
        search_obj=search_obj,
        X_train=X_train,
        y_train=y_train,
        history_path=history_path,
        run_time=run_time,
        model_name=model_name,
        grid_label=grid_label,
        notes=notes,
        fit_search=fit_search,
    )


def transform_with_fitted_scaler_pca(search_obj, X_train, X_test):
    fitted_pipeline = search_obj.best_estimator_
    scaler = fitted_pipeline.named_steps["scaler"]
    reducer = fitted_pipeline.named_steps["pca"]
    X_train_pca = reducer.transform(scaler.transform(X_train))
    X_test_pca = reducer.transform(scaler.transform(X_test))
    return X_train_pca, X_test_pca, scaler, reducer


def load_input_data(
    *,
    use_sample_parquet: bool,
    sample_parquet_path: str,
    import_data_fn: Callable[..., tuple[pd.DataFrame, pd.Series]],
    import_data_kwargs: dict[str, Any],
):
    if not use_sample_parquet:
        return import_data_fn(**import_data_kwargs)

    print(f"USE_SAMPLE_PARQUET=1 -> loading sample parquet from {os.path.abspath(sample_parquet_path)}")
    idx = pd.IndexSlice
    table = pq.read_table(sample_parquet_path)
    data = table.to_pandas()
    print("Finished Downloading Data -------")
    print("Initial shape:", data.shape[0], "rows,", data.shape[1], "columns.")

    print("------- Cleaning data")
    for data_type in ['Stocks']:
        temp_data = data.loc[:, idx[:, data_type, :]].dropna(how="all", axis=0)
        missing_one = (temp_data.isna().sum() == 1)
        cols = missing_one[missing_one == 1].index
        temp_data[cols] = temp_data[cols].ffill()
        temp_data = temp_data.dropna(how="any", axis=1)
        data = data.drop(columns=data_type, level=1).join(temp_data)

    stocks = data.loc[:, idx[:, 'Stocks', :]]
    to_drop = stocks.index[stocks.isna().all(axis=1)]
    data = data.drop(index=to_drop)
    print("Finished Cleaning Data -------")
    print("Current shape:", data.shape[0], "rows,", data.shape[1], "columns.")

    data = pd.concat([
        data,
        data.loc[:, idx[['Close', 'Open', 'High', 'Low'], 'Stocks', :]]
        .copy()
        .pct_change()
        .rename(columns={metric: f"{metric} PC" for metric in ['Close', 'Open', 'High', 'Low']}, level=0)
    ], axis=1)
    print("Created Percent Changes.")

    y_regression = (
        (data.loc[:, idx['Close', 'Index', '^SPX']] - data.loc[:, idx['Open', 'Index', '^SPX']])
        / data.loc[:, idx['Open', 'Index', '^SPX']]
    ).rename("Target Regression").shift(-1)
    print("Created Target (Regression).")

    data[("Day of Week", "Calendar", "All")] = data.index.dayofweek
    print("---EXTRA---: Created Day of Week.")

    high_ = data.loc[:, idx['High', :, :]]
    low_ = data.loc[:, idx['Low', :, :]]
    data = pd.concat([
        data,
        pd.DataFrame(high_.values - low_.values, index=high_.index, columns=high_.columns)
        .rename(columns={'High': 'Daily Range'}, level=0)
    ], axis=1)
    print("---EXTRA---: Created Daily Range.")

    return data, y_regression
