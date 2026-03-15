#!/usr/bin/env python3
"""Prediction visualizations for the best available leaderboard model.

Current scope:
- resolve the best candidate from the global leaderboard in auto mode
- generate three plots for supported baseline-logistic candidates from `base.py`
  1. prediction probability over time
  2. predicted-probability distribution by true class
  3. 2D PCA scatter colored by predicted probability

If the current global winner comes from an unsupported training script, the
script fails with a clear message rather than guessing the wrong model/data.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

MPLCONFIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".mplconfig")
os.makedirs(MPLCONFIGDIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from H_eval import rank_models_by_metrics
from H_prep import clean_data, import_data, to_binary_class
from H_search_history import load_search_checkpoint, load_stage_checkpoint
from model_grids import RANDOM_SEED, TEST_SIZE, TRAIN_TEST_SHUFFLE


EXPECTED_SOURCE_SCRIPTS = {
    "base.py",
    "base_random_forest.py",
    "base_SVM.py",
    "logistic_regression.py",
    "random_forest.py",
    "SVM.py",
}


def _default_output_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "output"


def _keep_raw_stock_ohlcv(X: pd.DataFrame) -> pd.DataFrame:
    idx = pd.IndexSlice
    metrics = ["Open", "Close", "High", "Low", "Volume"]
    return X.loc[:, idx[metrics, "Stocks", :]].copy()


def _load_leaderboard(dataset_label: str, output_dir: Path) -> pd.DataFrame:
    ranked_path = output_dir / f"{dataset_label}_global_model_leaderboard_ranked.csv"
    raw_path = output_dir / f"{dataset_label}_global_model_leaderboard.csv"

    if ranked_path.exists():
        return pd.read_csv(ranked_path)
    if raw_path.exists():
        return rank_models_by_metrics(pd.read_csv(raw_path))

    raise FileNotFoundError(
        f"No leaderboard found for dataset '{dataset_label}'. "
        f"Expected one of: {ranked_path} or {raw_path}"
    )


def _validate_expected_sources(
    leaderboard_df: pd.DataFrame,
    *,
    dataset_label: str,
    require_complete: bool,
) -> pd.DataFrame:
    dataset_df = leaderboard_df.loc[leaderboard_df["dataset_label"] == dataset_label].copy()
    if dataset_df.empty:
        raise ValueError(
            f"Leaderboard exists, but it has no rows for dataset_label='{dataset_label}'."
        )

    if require_complete:
        present_sources = set(dataset_df["source_script"].dropna().astype(str))
        missing_sources = sorted(EXPECTED_SOURCE_SCRIPTS - present_sources)
        if missing_sources:
            raise ValueError(
                "Auto mode requires a complete leaderboard before choosing the global best model. "
                f"Missing source scripts for dataset '{dataset_label}': {', '.join(missing_sources)}"
            )

    return dataset_df


def resolve_auto_candidate(
    *,
    dataset_label: str,
    output_dir: Path,
    require_complete: bool = True,
) -> pd.Series:
    leaderboard_df = _load_leaderboard(dataset_label, output_dir)
    dataset_df = _validate_expected_sources(
        leaderboard_df,
        dataset_label=dataset_label,
        require_complete=require_complete,
    )

    if "average_rank" in dataset_df.columns:
        dataset_df = dataset_df.sort_values(
            ["average_rank", "validation_std_accuracy"],
            ascending=[True, True],
        ).reset_index(drop=True)
    else:
        dataset_df = rank_models_by_metrics(dataset_df).reset_index(drop=True)

    return dataset_df.iloc[0]


def _load_base_raw_dataset() -> dict[str, Any]:
    data = import_data(
        testing=False,
        extra_features=False,
        cluster=False,
        n_clusters=100,
        corr_threshold=0.95,
        corr_level=0,
    )
    X, y_regression = clean_data(*data, raw=True, extra_features=False)
    X = _keep_raw_stock_ohlcv(X)
    X.columns = [f"{metric}_{ticker}" for metric, _, ticker in X.columns]
    y_classification = to_binary_class(y_regression)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_classification,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        shuffle=TRAIN_TEST_SHUFFLE,
    )

    dow_dummies = pd.get_dummies(X.index.dayofweek, prefix="DOW").astype(float)
    dow_dummies.index = X.index
    if "DOW_4" in dow_dummies.columns:
        dow_dummies = dow_dummies.drop(columns=["DOW_4"])
    X_dow = pd.concat([X, dow_dummies], axis=1)
    X_train_dow, X_test_dow, y_train_dow, y_test_dow = train_test_split(
        X_dow,
        y_classification,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        shuffle=TRAIN_TEST_SHUFFLE,
    )

    return {
        "X": X,
        "y": y_classification,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_dow": X_train_dow,
        "X_test_dow": X_test_dow,
        "y_train_dow": y_train_dow,
        "y_test_dow": y_test_dow,
    }


def _load_engineered_dataset() -> dict[str, Any]:
    data = import_data(
        testing=False,
        extra_features=True,
        cluster=False,
        n_clusters=100,
        corr_threshold=0.95,
        corr_level=0,
    )
    X, y_regression = clean_data(
        *data,
        raw=False,
        extra_features=True,
        lag_period=[1, 2, 3, 4, 5, 6, 7],
        lookback_period=30,
        sector=False,
        corr_threshold=0.95,
        corr_level=0,
    )
    y_classification = to_binary_class(y_regression)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_classification,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        shuffle=TRAIN_TEST_SHUFFLE,
    )
    return {
        "X": X,
        "y": y_classification,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def _extract_positive_scores(model, X_eval) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_eval)
        if np.ndim(proba) == 2 and proba.shape[1] >= 2:
            return np.asarray(proba[:, 1], dtype=float)
        return np.asarray(np.ravel(proba), dtype=float)
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X_eval), dtype=float)
        return 1.0 / (1.0 + np.exp(-scores))
    preds = np.asarray(model.predict(X_eval), dtype=float)
    return preds


def _make_safe_stem(text: str) -> str:
    out = []
    for char in str(text).lower():
        if char.isalnum():
            out.append(char)
        else:
            out.append("_")
    return "".join(out).strip("_")


def _resolve_base_candidate(best_row: pd.Series, output_dir: Path) -> dict[str, Any]:
    dataset = _load_base_raw_dataset()
    checkpoint_dir = output_dir / "checkpoints" / "base" / f"{best_row['dataset_label']}_v1"
    candidate_model = str(best_row["candidate_model"])

    def _payload(stage_name: str) -> dict[str, Any]:
        payload = load_stage_checkpoint(checkpoint_dir, stage_name)
        if not isinstance(payload, dict):
            raise TypeError(f"Expected dict payload for stage '{stage_name}', got {type(payload)}")
        return payload

    if candidate_model == "Raw Ridge":
        stage = _payload("raw_ridge_model")
        return {
            "model": stage["pipeline_ridge_1se"],
            "X_test_plot": dataset["X_test"],
            "y_test": dataset["y_test"],
            "plot_frame": dataset["X_test"],
            "title": candidate_model,
        }
    if candidate_model == "Raw LASSO":
        stage = _payload("raw_lasso_model")
        return {
            "model": stage["pipeline_lasso_1se"],
            "X_test_plot": dataset["X_test"],
            "y_test": dataset["y_test"],
            "plot_frame": dataset["X_test"],
            "title": candidate_model,
        }
    if candidate_model.startswith("Base+PCA"):
        stage = _payload("raw_baseline_pca_model")
        pca_stage = _payload("raw_pca_selection")
        return {
            "model": stage["baseline_clf"],
            "X_test_plot": pca_stage["X_test_pca"],
            "y_test": dataset["y_test"],
            "plot_frame": pd.DataFrame(
                pca_stage["X_test_pca"],
                index=dataset["X_test"].index,
                columns=[f"PC{i+1}" for i in range(pca_stage["X_test_pca"].shape[1])],
            ),
            "title": candidate_model,
        }
    if candidate_model.startswith("Raw Ridge+PCA"):
        stage = _payload("raw_ridge_pca_model")
        return {
            "model": stage["clf_ridge_pca_1se"],
            "X_test_plot": stage["X_test_pca"],
            "y_test": dataset["y_test"],
            "plot_frame": pd.DataFrame(
                stage["X_test_pca"],
                index=dataset["X_test"].index,
                columns=[f"PC{i+1}" for i in range(stage["X_test_pca"].shape[1])],
            ),
            "title": candidate_model,
        }
    if candidate_model.startswith("Raw LASSO+PCA"):
        stage = _payload("raw_lasso_pca_model")
        return {
            "model": stage["clf_lasso_pca_1se"],
            "X_test_plot": stage["X_test_pca"],
            "y_test": dataset["y_test"],
            "plot_frame": pd.DataFrame(
                stage["X_test_pca"],
                index=dataset["X_test"].index,
                columns=[f"PC{i+1}" for i in range(stage["X_test_pca"].shape[1])],
            ),
            "title": candidate_model,
        }
    if candidate_model == "Ridge+DOW":
        stage = _payload("dow_ridge_model")
        return {
            "model": stage["pipeline_ridge_1se"],
            "X_test_plot": dataset["X_test_dow"],
            "y_test": dataset["y_test_dow"],
            "plot_frame": dataset["X_test_dow"],
            "title": candidate_model,
        }
    if candidate_model == "LASSO+DOW":
        stage = _payload("dow_lasso_model")
        return {
            "model": stage["pipeline_lasso_1se"],
            "X_test_plot": dataset["X_test_dow"],
            "y_test": dataset["y_test_dow"],
            "plot_frame": dataset["X_test_dow"],
            "title": candidate_model,
        }
    if candidate_model.startswith("Base+DOW+PCA"):
        stage = _payload("dow_baseline_pca_model")
        return {
            "model": stage["baseline_clf"],
            "X_test_plot": stage["X_test_pca"],
            "y_test": dataset["y_test_dow"],
            "plot_frame": pd.DataFrame(
                stage["X_test_pca"],
                index=dataset["X_test_dow"].index,
                columns=[f"PC{i+1}" for i in range(stage["X_test_pca"].shape[1])],
            ),
            "title": candidate_model,
        }
    if candidate_model.startswith("Ridge+PCA+DOW"):
        stage = _payload("dow_ridge_pca_model")
        return {
            "model": stage["clf_ridge_pca_1se"],
            "X_test_plot": stage["X_test_pca"],
            "y_test": dataset["y_test_dow"],
            "plot_frame": pd.DataFrame(
                stage["X_test_pca"],
                index=dataset["X_test_dow"].index,
                columns=[f"PC{i+1}" for i in range(stage["X_test_pca"].shape[1])],
            ),
            "title": candidate_model,
        }
    if candidate_model.startswith("LASSO+PCA+DOW"):
        stage = _payload("dow_lasso_pca_model")
        return {
            "model": stage["clf_lasso_pca_1se"],
            "X_test_plot": stage["X_test_pca"],
            "y_test": dataset["y_test_dow"],
            "plot_frame": pd.DataFrame(
                stage["X_test_pca"],
                index=dataset["X_test_dow"].index,
                columns=[f"PC{i+1}" for i in range(stage["X_test_pca"].shape[1])],
            ),
            "title": candidate_model,
        }

    raise ValueError(f"Unsupported base.py candidate model: {candidate_model}")


def _resolve_logistic_regression_candidate(best_row: pd.Series, output_dir: Path) -> dict[str, Any]:
    dataset = _load_engineered_dataset()
    checkpoint_dir = output_dir / "checkpoints" / "logistic_regression" / f"{best_row['dataset_label']}_v1"
    candidate_model = str(best_row["candidate_model"])

    def _search(stage_name: str):
        return load_search_checkpoint(checkpoint_dir, stage_name)

    if candidate_model == "PCA Base":
        search = _search("logreg_pca_base")
        return {
            "model": search.best_estimator_,
            "X_test_plot": dataset["X_test"],
            "y_test": dataset["y_test"],
            "plot_frame": dataset["X_test"],
            "title": candidate_model,
        }
    if candidate_model == "Ridge Log. Reg.":
        search = _search("logreg_ridge")
        return {
            "model": search.best_estimator_,
            "X_test_plot": dataset["X_test"],
            "y_test": dataset["y_test"],
            "plot_frame": dataset["X_test"],
            "title": candidate_model,
        }
    if candidate_model == "LASSO Log. Reg.":
        search = _search("logreg_lasso")
        return {
            "model": search.best_estimator_,
            "X_test_plot": dataset["X_test"],
            "y_test": dataset["y_test"],
            "plot_frame": dataset["X_test"],
            "title": candidate_model,
        }
    if candidate_model == "PCA Ridge(int.) Log. Reg.":
        model_search = _search("logreg_pca_ridge_refit")
        pca_search = _search("logreg_pca_ridge_retuned_n_components")
        transformed = pca_search.best_estimator_.named_steps["pca"].transform(
            pca_search.best_estimator_.named_steps["scaler"].transform(dataset["X_test"])
        )
        return {
            "model": model_search.best_estimator_,
            "X_test_plot": transformed,
            "y_test": dataset["y_test"],
            "plot_frame": pd.DataFrame(
                transformed,
                index=dataset["X_test"].index,
                columns=[f"PC{i+1}" for i in range(transformed.shape[1])],
            ),
            "title": candidate_model,
        }
    if candidate_model == "PCA LASSO(int.) Log. Reg.":
        model_search = _search("logreg_pca_lasso_refit")
        pca_search = _search("logreg_pca_lasso_retuned_n_components")
        transformed = pca_search.best_estimator_.named_steps["pca"].transform(
            pca_search.best_estimator_.named_steps["scaler"].transform(dataset["X_test"])
        )
        return {
            "model": model_search.best_estimator_,
            "X_test_plot": transformed,
            "y_test": dataset["y_test"],
            "plot_frame": pd.DataFrame(
                transformed,
                index=dataset["X_test"].index,
                columns=[f"PC{i+1}" for i in range(transformed.shape[1])],
            ),
            "title": candidate_model,
        }

    raise ValueError(f"Unsupported logistic_regression.py candidate model: {candidate_model}")


def _resolve_base_svm_candidate(best_row: pd.Series, output_dir: Path) -> dict[str, Any]:
    dataset = _load_base_raw_dataset()
    checkpoint_dir = output_dir / "checkpoints" / "base_SVM" / f"{best_row['dataset_label']}_v1"
    candidate_model = str(best_row["candidate_model"])
    stage_map = {
        "Raw Linear SVM": "base_svm_linear",
        "Raw RBF SVM": "base_svm_rbf",
        "Raw Poly SVM": "base_svm_poly",
    }
    if candidate_model not in stage_map:
        raise ValueError(f"Unsupported base_SVM.py candidate model: {candidate_model}")
    search = load_search_checkpoint(checkpoint_dir, stage_map[candidate_model])
    return {
        "model": search.best_estimator_,
        "X_test_plot": dataset["X_test"],
        "y_test": dataset["y_test"],
        "plot_frame": dataset["X_test"],
        "title": candidate_model,
    }


def _resolve_svm_candidate(best_row: pd.Series, output_dir: Path) -> dict[str, Any]:
    dataset = _load_engineered_dataset()
    checkpoint_dir = output_dir / "checkpoints" / "SVM" / f"{best_row['dataset_label']}_v1"
    candidate_model = str(best_row["candidate_model"])
    stage_map = {
        "Linear SVM": "svm_linear",
        "RBF SVM": "svm_rbf",
        "Poly SVM": "svm_poly",
    }
    if candidate_model not in stage_map:
        raise ValueError(f"Unsupported SVM.py candidate model: {candidate_model}")
    search = load_search_checkpoint(checkpoint_dir, stage_map[candidate_model])
    return {
        "model": search.best_estimator_,
        "X_test_plot": dataset["X_test"],
        "y_test": dataset["y_test"],
        "plot_frame": dataset["X_test"],
        "title": candidate_model,
    }


def _resolve_base_random_forest_candidate(best_row: pd.Series, output_dir: Path) -> dict[str, Any]:
    dataset = _load_base_raw_dataset()
    checkpoint_dir = output_dir / "checkpoints" / "base_random_forest" / f"{best_row['dataset_label']}_v1"
    candidate_model = str(best_row["candidate_model"])

    if candidate_model == "Raw RF":
        search = load_search_checkpoint(checkpoint_dir, "raw_raw_rf")
        return {
            "model": search.best_estimator_,
            "X_test_plot": dataset["X_test"],
            "y_test": dataset["y_test"],
            "plot_frame": dataset["X_test"],
            "title": candidate_model,
        }
    if candidate_model == "RF+DOW":
        search = load_search_checkpoint(checkpoint_dir, "dow_raw_rf")
        return {
            "model": search.best_estimator_,
            "X_test_plot": dataset["X_test_dow"],
            "y_test": dataset["y_test_dow"],
            "plot_frame": dataset["X_test_dow"],
            "title": candidate_model,
        }
    if candidate_model == "PCA RF":
        search = load_search_checkpoint(checkpoint_dir, "raw_pca_rf_retuned_n_components")
        return {
            "model": search.best_estimator_,
            "X_test_plot": dataset["X_test"],
            "y_test": dataset["y_test"],
            "plot_frame": dataset["X_test"],
            "title": candidate_model,
        }
    if candidate_model == "PCA RF+DOW":
        search = load_search_checkpoint(checkpoint_dir, "dow_pca_rf_retuned_n_components")
        return {
            "model": search.best_estimator_,
            "X_test_plot": dataset["X_test_dow"],
            "y_test": dataset["y_test_dow"],
            "plot_frame": dataset["X_test_dow"],
            "title": candidate_model,
        }

    raise ValueError(f"Unsupported base_random_forest.py candidate model: {candidate_model}")


def _resolve_random_forest_candidate(best_row: pd.Series, output_dir: Path) -> dict[str, Any]:
    dataset = _load_engineered_dataset()
    checkpoint_dir = output_dir / "checkpoints" / "random_forest" / f"{best_row['dataset_label']}_v1"
    candidate_model = str(best_row["candidate_model"])

    if candidate_model == "Base RF":
        search = load_search_checkpoint(checkpoint_dir, "rf_base")
        return {
            "model": search.best_estimator_,
            "X_test_plot": dataset["X_test"],
            "y_test": dataset["y_test"],
            "plot_frame": dataset["X_test"],
            "title": candidate_model,
        }
    if candidate_model == "PCA RF":
        search = load_search_checkpoint(checkpoint_dir, "rf_pca_retuned_n_components")
        return {
            "model": search.best_estimator_,
            "X_test_plot": dataset["X_test"],
            "y_test": dataset["y_test"],
            "plot_frame": dataset["X_test"],
            "title": candidate_model,
        }
    if candidate_model == "LASSO RF":
        search = load_search_checkpoint(checkpoint_dir, "rf_lasso")
        return {
            "model": search.best_estimator_,
            "X_test_plot": dataset["X_test"],
            "y_test": dataset["y_test"],
            "plot_frame": dataset["X_test"],
            "title": candidate_model,
        }
    if candidate_model == "Ridge RF":
        search = load_search_checkpoint(checkpoint_dir, "rf_ridge")
        return {
            "model": search.best_estimator_,
            "X_test_plot": dataset["X_test"],
            "y_test": dataset["y_test"],
            "plot_frame": dataset["X_test"],
            "title": candidate_model,
        }

    raise ValueError(f"Unsupported random_forest.py candidate model: {candidate_model}")


def resolve_candidate_artifacts(best_row: pd.Series, output_dir: Path) -> dict[str, Any]:
    source_script = str(best_row["source_script"])
    if source_script == "base.py":
        return _resolve_base_candidate(best_row, output_dir)
    if source_script == "logistic_regression.py":
        return _resolve_logistic_regression_candidate(best_row, output_dir)
    if source_script == "base_SVM.py":
        return _resolve_base_svm_candidate(best_row, output_dir)
    if source_script == "SVM.py":
        return _resolve_svm_candidate(best_row, output_dir)
    if source_script == "base_random_forest.py":
        return _resolve_base_random_forest_candidate(best_row, output_dir)
    if source_script == "random_forest.py":
        return _resolve_random_forest_candidate(best_row, output_dir)

    raise NotImplementedError(
        f"Prediction visualization does not yet support source_script='{source_script}', "
        f"candidate_model='{best_row['candidate_model']}'."
    )


def _plot_probability_timeline(
    dates: pd.Index,
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, y_score, color="steelblue", linewidth=1.7, label="Predicted P(Up)")
    ax.scatter(
        dates,
        y_true,
        c=np.where(y_true == 1, "#d95f02", "#1b9e77"),
        s=18,
        alpha=0.8,
        label="True class (0/1)",
    )
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, label="0.5 threshold")
    ax.set_title(f"Prediction Probability Over Time\n{title}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Probability / True class")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_probability_distribution(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    score_min = float(np.nanmin(y_score))
    score_max = float(np.nanmax(y_score))

    if np.isclose(score_min, score_max):
        pad = max(0.05, abs(score_min) * 0.1 if score_min != 0 else 0.05)
        plot_min = score_min - pad
        plot_max = score_max + pad
    else:
        pad = max((score_max - score_min) * 0.05, 0.01)
        plot_min = score_min - pad
        plot_max = score_max + pad

    # Use a coarser adaptive bin count so the density curves read more clearly.
    sample_size = max(1, int(np.asarray(y_score).size))
    n_bins = int(np.clip(np.sqrt(sample_size) / 2.0, 18, 32))
    bins = np.linspace(plot_min, plot_max, n_bins + 1)
    ax.hist(
        y_score[y_true == 0],
        bins=bins,
        alpha=0.6,
        color="#1b9e77",
        label="True Down (0)",
        density=True,
    )
    ax.hist(
        y_score[y_true == 1],
        bins=bins,
        alpha=0.6,
        color="#d95f02",
        label="True Up (1)",
        density=True,
    )
    if plot_min <= 0.5 <= plot_max:
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xlim(plot_min, plot_max)
    ax.set_title(f"Predicted Probability Distribution by True Class\n{title}")
    ax.set_xlabel("Predicted P(Up)")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_pca_scatter(
    plot_frame: pd.DataFrame,
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    if plot_frame.shape[1] >= 2:
        scatter_x = np.asarray(plot_frame.iloc[:, 0], dtype=float)
        scatter_y = np.asarray(plot_frame.iloc[:, 1], dtype=float)
        x_label = str(plot_frame.columns[0])
        y_label = str(plot_frame.columns[1])
    else:
        reducer = PCA(n_components=2, random_state=RANDOM_SEED)
        coords = reducer.fit_transform(np.asarray(plot_frame, dtype=float))
        scatter_x = coords[:, 0]
        scatter_y = coords[:, 1]
        x_label = "PC1"
        y_label = "PC2"

    fig, ax = plt.subplots(figsize=(8, 6))
    down_mask = y_true == 0
    up_mask = y_true == 1
    sc = ax.scatter(
        scatter_x[down_mask],
        scatter_y[down_mask],
        c=y_score[down_mask],
        cmap="coolwarm",
        vmin=0.0,
        vmax=1.0,
        marker="o",
        s=40,
        alpha=0.9,
        edgecolors="black",
        linewidths=0.5,
        label="True Down (0)",
    )
    ax.scatter(
        scatter_x[up_mask],
        scatter_y[up_mask],
        c=y_score[up_mask],
        cmap="coolwarm",
        vmin=0.0,
        vmax=1.0,
        marker="s",
        s=40,
        alpha=0.9,
        edgecolors="black",
        linewidths=0.5,
        label="True Up (1)",
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=plt.cm.coolwarm(0.0),
            markeredgecolor="black",
            markeredgewidth=0.6,
            markersize=7,
            label="Ideal Down: round + cold color (P(Up)=0)",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markerfacecolor=plt.cm.coolwarm(1.0),
            markeredgecolor="black",
            markeredgewidth=0.6,
            markersize=7,
            label="Ideal Up: square + hot color (P(Up)=1)",
        ),
    ]
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Predicted P(Up)")
    ax.set_title(f"2D Feature Projection Colored by Predicted Probability\n{title}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.2)
    ax.legend(handles=legend_handles, fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_prediction_visualizations(
    *,
    best_row: pd.Series,
    resolved: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    y_test = np.asarray(resolved["y_test"]).astype(int).ravel()
    X_test_plot = resolved["X_test_plot"]
    plot_frame = resolved["plot_frame"]
    title = f"{best_row['source_script']} :: {resolved['title']}"
    dates = plot_frame.index
    y_score = _extract_positive_scores(resolved["model"], X_test_plot)

    stem = _make_safe_stem(f"{best_row['source_script']}_{resolved['title']}")
    paths = [
        output_dir / f"{best_row['dataset_label']}_prediction_timeline_{stem}.png",
        output_dir / f"{best_row['dataset_label']}_prediction_distribution_{stem}.png",
        output_dir / f"{best_row['dataset_label']}_prediction_pca_scatter_{stem}.png",
    ]

    _plot_probability_timeline(dates, y_test, y_score, title, paths[0])
    _plot_probability_distribution(y_test, y_score, title, paths[1])
    _plot_pca_scatter(plot_frame, y_test, y_score, title, paths[2])
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate prediction visualizations for the best available leaderboard model."
    )
    parser.add_argument(
        "--mode",
        choices=["auto"],
        default="auto",
        help="Selection mode. Only auto mode is implemented right now.",
    )
    parser.add_argument(
        "--dataset-label",
        default="8yrs",
        help="Dataset label used in the leaderboard filenames.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(),
        help="Directory containing the global leaderboard CSV files and plot outputs.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow auto mode to resolve the current top leaderboard row even if not all model scripts have been run yet.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    best_row = resolve_auto_candidate(
        dataset_label=args.dataset_label,
        output_dir=args.output_dir,
        require_complete=not args.allow_partial,
    )
    resolved = resolve_candidate_artifacts(best_row, args.output_dir)
    output_paths = save_prediction_visualizations(
        best_row=best_row,
        resolved=resolved,
        output_dir=args.output_dir,
    )

    print("Resolved auto-mode candidate for prediction visualization:")
    print(f"  dataset_label: {best_row['dataset_label']}")
    print(f"  source_script: {best_row['source_script']}")
    print(f"  comparison_scope: {best_row['comparison_scope']}")
    print(f"  candidate_model: {best_row['candidate_model']}")
    print(f"  leaderboard_model: {best_row['Model']}")
    if "average_rank" in best_row.index:
        print(f"  average_rank: {best_row['average_rank']}")
    print("Generated files:")
    for path in output_paths:
        print(f"  {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
