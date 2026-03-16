#!/usr/bin/env python3
"""Neural-network comparison table using the shared base-style reporting helpers."""

import os
import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from typing import NamedTuple
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cwd = Path.cwd()
for _ in range(5):
    if cwd.name != "STAT-587-Final-Project":
        cwd = cwd.parent
    else:
        break
else:
    raise FileNotFoundError("Could not find correct workspace folder.")

sys.path.append(os.path.abspath(cwd / "PyScripts" / "Models"))

from H_eval import (
    build_base_style_comparison_df,
    build_compact_export_table,
    comparison_row_from_metrics,
    get_final_metrics,
    register_global_model_candidates,
    rank_models_by_metrics,
    write_base_style_latex_table,
)
from H_prep import clean_data, import_data, to_binary_class
from H_search_history import (
    get_checkpoint_dir,
    load_stage_checkpoint,
    save_stage_checkpoint,
    stage_checkpoint_exists,
)
from model_grids import (
    NN_CNN_PARAM_GRID,
    NN_LAG_PERIOD,
    NN_LSTM_PARAM_GRID,
    NN_RNN_PARAM_GRID,
    RANDOM_SEED,
    TEST_SIZE,
    TIME_SERIES_CV_SPLITS,
    TRAIN_TEST_SHUFFLE,
)

GRID_VERSION = os.getenv("GRID_VERSION", "v1")
DATASET_LABEL = "8yrs_engineered"


class _KerasSequenceClassifier(BaseEstimator, ClassifierMixin):
    """Minimal sklearn-style wrapper so shared evaluation helpers can score Keras models."""

    def __init__(
        self,
        architecture: str,
        sequence_length: int,
        n_features: int,
        units_1: int,
        units_2: int | None = None,
        dense_units: int | None = None,
        dropout_1: float = 0.0,
        dropout_2: float = 0.0,
        activation: str = "relu",
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        patience: int = 10,
        validation_split: float = 0.2,
        verbose: int = 0,
        random_seed: int = RANDOM_SEED,
    ):
        self.architecture = architecture
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.units_1 = units_1
        self.units_2 = units_2
        self.dense_units = dense_units
        self.dropout_1 = dropout_1
        self.dropout_2 = dropout_2
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.validation_split = validation_split
        self.verbose = verbose
        self.random_seed = random_seed

    def _build_model(self):
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        tf.random.set_seed(self.random_seed)

        model = keras.Sequential()
        if self.architecture == "lstm":
            model.add(
                layers.LSTM(
                    self.units_1,
                    activation=self.activation,
                    input_shape=(self.sequence_length, self.n_features),
                    return_sequences=self.units_2 is not None,
                )
            )
        elif self.architecture == "rnn":
            model.add(
                layers.SimpleRNN(
                    self.units_1,
                    activation=self.activation,
                    input_shape=(self.sequence_length, self.n_features),
                )
            )
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")

        if self.dropout_1 > 0:
            model.add(layers.Dropout(self.dropout_1))

        if self.architecture == "lstm" and self.units_2 is not None:
            model.add(layers.LSTM(self.units_2, activation=self.activation))
            if self.dropout_2 > 0:
                model.add(layers.Dropout(self.dropout_2))

        if self.dense_units is not None:
            model.add(layers.Dense(self.dense_units, activation="relu"))

        model.add(layers.Dense(1, activation="sigmoid"))
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model, keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.patience,
            restore_best_weights=True,
            verbose=self.verbose,
        )

    def fit(self, X, y):
        import tensorflow as tf

        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.scaler_ = StandardScaler()
        X_2d = X.reshape(-1, self.n_features)
        X_scaled = self.scaler_.fit_transform(X_2d).reshape(-1, self.sequence_length, self.n_features)
        self.model_, early_stopping = self._build_model()
        callbacks = [early_stopping]
        fit_kwargs = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "callbacks": callbacks,
            "verbose": self.verbose,
            "shuffle": False,
        }

        if self.validation_split > 0 and len(X_scaled) > 1:
            val_size = max(1, int(np.ceil(len(X_scaled) * self.validation_split)))
            val_size = min(val_size, len(X_scaled) - 1)
            split_idx = len(X_scaled) - val_size
            X_fit, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
            y_fit, y_val = y[:split_idx], y[split_idx:]
            fit_kwargs["validation_data"] = (X_val, y_val)
        else:
            X_fit, y_fit = X_scaled, y

        self.history_ = self.model_.fit(
            X_fit,
            y_fit,
            **fit_kwargs,
        )
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        X_2d = X.reshape(-1, self.n_features)
        X_scaled = self.scaler_.transform(X_2d).reshape(-1, self.sequence_length, self.n_features)
        proba_up = self.model_.predict(X_scaled, verbose=0).reshape(-1)
        proba_down = 1.0 - proba_up
        return np.column_stack([proba_down, proba_up])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score(self, X, y):
        preds = self.predict(X)
        return float(np.mean(preds == y))


class _SequenceData(NamedTuple):
    """Container for sequence arrays and their alignment to original rows."""

    X_seq_3d: np.ndarray
    X_seq_flat: np.ndarray
    y_seq: np.ndarray
    seq_indices: np.ndarray
    seq_dates: np.ndarray


def _first_grid_params(param_grid: dict[str, list]) -> dict[str, object]:
    """Select the current default configuration from a CV-ready parameter grid."""
    return {key: values[0] for key, values in param_grid.items()}


def _first_pipeline_params(param_grid: dict[str, list], prefix: str) -> dict[str, object]:
    """Strip a pipeline prefix while selecting the current default configuration."""
    selected = {}
    for key, values in param_grid.items():
        selected_key = key[len(prefix):] if key.startswith(prefix) else key
        selected[selected_key] = values[0]
    return selected


def _resolve_default_nn_sequence_length() -> int:
    """Use the first configured sequence length across the active NN grids."""
    sequence_lengths = []
    for param_grid in (NN_LSTM_PARAM_GRID, NN_RNN_PARAM_GRID, NN_CNN_PARAM_GRID):
        if "sequence_length" in param_grid:
            sequence_lengths.append(int(param_grid["sequence_length"][0]))

    if not sequence_lengths:
        return len(NN_LAG_PERIOD)

    first_length = sequence_lengths[0]
    if any(length != first_length for length in sequence_lengths):
        raise ValueError(
            "NN grids disagree on the active default sequence_length. "
            "Keep the first value aligned across LSTM, RNN, and CNN grids."
        )
    return first_length


def _append_plot_suffix(path_like, suffix: str) -> Path:
    path = Path(path_like)
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def _reshape_sequences(X: np.ndarray, sequence_length: int, *, flatten: bool) -> tuple[np.ndarray, np.ndarray]:
    n_samples, n_features = X.shape
    n_sequences = n_samples - sequence_length + 1
    if flatten:
        X_seq = np.zeros((n_sequences, sequence_length * n_features), dtype=np.float32)
    else:
        X_seq = np.zeros((n_sequences, sequence_length, n_features), dtype=np.float32)
    indices = []

    for i in range(n_sequences):
        window = X[i : i + sequence_length]
        X_seq[i] = window.flatten() if flatten else window
        indices.append(i + sequence_length - 1)

    return X_seq, np.array(indices)


def _load_sequence_data(sequence_length: int) -> _SequenceData:
    print("Loading and cleaning 8-year data via H_prep...")
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
        lookback_period=30,
        lag_period=list(range(1, sequence_length + 1)),
        extra_features=True,
        raw=False,
        sector=False,
        corr_threshold=0.95,
        corr_level=0,
    )
    y = to_binary_class(y_regression).to_numpy()
    X_engineered = X.to_numpy(dtype=np.float32)

    X_seq_3d, seq_indices = _reshape_sequences(X_engineered, sequence_length, flatten=False)
    X_seq_flat, _ = _reshape_sequences(X_engineered, sequence_length, flatten=True)
    y_seq = y[seq_indices]
    seq_dates = np.asarray(X.index[seq_indices])

    print(f"Engineered feature shape: {X.shape}")
    print(f"Sequence length: {sequence_length}")
    print(f"3D sequence shape: {X_seq_3d.shape}")
    print(f"Flattened sequence shape: {X_seq_flat.shape}")
    print(f"Target shape: {y_seq.shape}\n")
    return _SequenceData(
        X_seq_3d=X_seq_3d,
        X_seq_flat=X_seq_flat,
        y_seq=y_seq,
        seq_indices=np.asarray(seq_indices),
        seq_dates=seq_dates,
    )


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


def _evaluate_model(model_name: str, estimator, X, y, sample_dates=None) -> dict:
    if sample_dates is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            shuffle=TRAIN_TEST_SHUFFLE,
        )
        test_dates = None
    else:
        X_train, X_test, y_train, y_test, _, test_dates = train_test_split(
            X,
            y,
            sample_dates,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            shuffle=TRAIN_TEST_SHUFFLE,
        )
    estimator.fit(X_train, y_train)
    metrics = get_final_metrics(
        estimator,
        X_train,
        y_train,
        X_test,
        y_test,
        n_splits=TIME_SERIES_CV_SPLITS,
        label=model_name,
    )
    plot_payload = {
        "y_test": np.asarray(y_test, dtype=int),
        "y_score": _extract_positive_scores(estimator, X_test),
    }
    if test_dates is not None:
        plot_payload["test_dates"] = np.asarray(test_dates)
    return {
        "metrics": metrics,
        "plot_payload": plot_payload,
    }


def _load_or_compute_model_payload(checkpoint_dir: Path, stage_name: str, compute_fn):
    if stage_checkpoint_exists(checkpoint_dir, stage_name):
        print(f"Loading checkpoint for {stage_name} from {checkpoint_dir / stage_name}")
        payload = load_stage_checkpoint(checkpoint_dir, stage_name)
        if isinstance(payload, dict) and "metrics" in payload and "plot_payload" in payload:
            return payload
        print(f"Checkpoint for {stage_name} is legacy and missing plot payload. Recomputing stage.")
    payload = compute_fn()
    save_stage_checkpoint(checkpoint_dir, stage_name, payload)
    return payload


def _train_test_split_with_optional_dates(X, y, sample_dates=None):
    if sample_dates is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            shuffle=TRAIN_TEST_SHUFFLE,
        )
        return X_train, X_test, y_train, y_test, None
    X_train, X_test, y_train, y_test, _, test_dates = train_test_split(
        X,
        y,
        sample_dates,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        shuffle=TRAIN_TEST_SHUFFLE,
    )
    return X_train, X_test, y_train, y_test, test_dates


def _compute_epoch_curve(estimator, X_train, y_train, X_test, y_test, epoch_values, epoch_param_name):
    tscv = TimeSeriesSplit(n_splits=TIME_SERIES_CV_SPLITS)
    rows = []

    for epoch_value in epoch_values:
        fold_train_bal = []
        fold_cv_bal = []
        fold_train_plain = []
        fold_cv_plain = []

        for tr_idx, val_idx in tscv.split(X_train, y_train):
            X_fold_train = X_train[tr_idx]
            y_fold_train = y_train[tr_idx]
            X_fold_val = X_train[val_idx]
            y_fold_val = y_train[val_idx]

            model = clone(estimator)
            model.set_params(**{epoch_param_name: int(epoch_value)})
            model.fit(X_fold_train, y_fold_train)

            train_pred = model.predict(X_fold_train)
            val_pred = model.predict(X_fold_val)
            fold_train_bal.append(balanced_accuracy_score(y_fold_train, train_pred))
            fold_cv_bal.append(balanced_accuracy_score(y_fold_val, val_pred))
            fold_train_plain.append(accuracy_score(y_fold_train, train_pred))
            fold_cv_plain.append(accuracy_score(y_fold_val, val_pred))

        direct_model = clone(estimator)
        direct_model.set_params(**{epoch_param_name: int(epoch_value)})
        direct_model.fit(X_train, y_train)
        train_pred_direct = direct_model.predict(X_train)
        test_pred_direct = direct_model.predict(X_test)

        rows.append({
            "epoch_value": int(epoch_value),
            "train_bal_err_mean": 1.0 - float(np.mean(fold_train_bal)),
            "train_bal_err_std": float(np.std(fold_train_bal)),
            "cv_bal_err_mean": 1.0 - float(np.mean(fold_cv_bal)),
            "cv_bal_err_std": float(np.std(fold_cv_bal)),
            "train_error": 1.0 - float(accuracy_score(y_train, train_pred_direct)),
            "test_error": 1.0 - float(accuracy_score(y_test, test_pred_direct)),
        })

    curve_df = pd.DataFrame(rows).sort_values("epoch_value").reset_index(drop=True)
    cv_bal_err_mean = curve_df["cv_bal_err_mean"].to_numpy(dtype=float)
    cv_bal_err_std = curve_df["cv_bal_err_std"].to_numpy(dtype=float)
    cv_bal_err_se = cv_bal_err_std / np.sqrt(float(tscv.get_n_splits()))
    best_idx = int(np.argmin(cv_bal_err_mean))
    threshold = float(cv_bal_err_mean[best_idx] + cv_bal_err_se[best_idx])
    candidate_idx = np.where(cv_bal_err_mean <= threshold)[0]
    selected_idx = int(min(candidate_idx, key=lambda i: (float(curve_df.loc[i, "epoch_value"]), float(cv_bal_err_mean[i]))))
    return {
        "x_numeric": curve_df["epoch_value"].to_numpy(dtype=float),
        "x_labels": [str(int(v)) for v in curve_df["epoch_value"].tolist()],
        "train_bal_err_mean": curve_df["train_bal_err_mean"].to_numpy(dtype=float),
        "train_bal_err_std": curve_df["train_bal_err_std"].to_numpy(dtype=float),
        "cv_bal_err_mean": cv_bal_err_mean,
        "cv_bal_err_std": cv_bal_err_std,
        "cv_bal_err_se": cv_bal_err_se,
        "train_errors": curve_df["train_error"].to_numpy(dtype=float),
        "test_errors": curve_df["test_error"].to_numpy(dtype=float),
        "best_idx": best_idx,
        "selected_idx": selected_idx,
        "selected_label": str(int(curve_df.loc[selected_idx, "epoch_value"])),
        "best_label": str(int(curve_df.loc[best_idx, "epoch_value"])),
        "threshold": threshold,
    }


def _save_nn_curve_pair(curve, model_title: str, feature_title: str, x_label: str, output_bv: Path, output_direct: Path):
    best_idx = int(curve["best_idx"])
    selected_idx = int(curve["selected_idx"])
    best_epoch = curve["best_label"]
    selected_epoch = curve["selected_label"]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(f"Bias-Variance Tradeoff - {model_title}\n{feature_title}", fontsize=13, fontweight="bold")
    ax.plot(curve["x_numeric"], curve["train_bal_err_mean"], marker="o", color="steelblue", linewidth=1.8, label="CV Train balanced error")
    ax.plot(curve["x_numeric"], curve["cv_bal_err_mean"], marker="s", color="darkorange", linewidth=1.8, label="CV Test balanced error")
    ax.fill_between(
        curve["x_numeric"],
        np.clip(curve["train_bal_err_mean"] - curve["train_bal_err_std"], 0.0, 1.0),
        np.clip(curve["train_bal_err_mean"] + curve["train_bal_err_std"], 0.0, 1.0),
        alpha=0.15,
        color="steelblue",
        label="CV Train balanced error ±1 SD",
    )
    ax.fill_between(
        curve["x_numeric"],
        np.clip(curve["cv_bal_err_mean"] - curve["cv_bal_err_std"], 0.0, 1.0),
        np.clip(curve["cv_bal_err_mean"] + curve["cv_bal_err_std"], 0.0, 1.0),
        alpha=0.15,
        color="darkorange",
        label="CV Test balanced error ±1 SD",
    )
    ax.scatter(
        [curve["x_numeric"][best_idx]],
        [curve["cv_bal_err_mean"][best_idx]],
        color="gold",
        edgecolor="black",
        s=90,
        zorder=6,
        label=f"Best CV balanced error @ {best_epoch}",
    )
    ax.axvline(
        curve["x_numeric"][selected_idx],
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"1SE-selected value = {selected_epoch}",
    )
    ax.set_title(f"{model_title} - Bias-Variance Tradeoff (Balanced Error)")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Balanced Error (1 - balanced accuracy)")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(curve["x_numeric"])
    ax.set_xticklabels(curve["x_labels"])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_bv, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    fig2.suptitle(f"Over/Underfitting Analysis - {model_title}\n{feature_title}", fontsize=13, fontweight="bold")
    ax2.plot(curve["x_numeric"], curve["train_errors"], marker="o", color="steelblue", linewidth=2, label="Train error")
    ax2.plot(curve["x_numeric"], curve["test_errors"], marker="s", color="darkorange", linewidth=2, label="Test error")
    ax2.scatter(
        [curve["x_numeric"][best_idx]],
        [curve["test_errors"][best_idx]],
        color="gold",
        edgecolor="black",
        s=90,
        zorder=6,
        label=f"Best CV balanced error @ {best_epoch}",
    )
    ax2.axvline(
        curve["x_numeric"][selected_idx],
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"1SE-selected value = {selected_epoch}",
    )
    ax2.set_title(f"{model_title} - Train vs Test Error (Plain Error)")
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Plain Error (1 - accuracy)")
    ax2.set_xticks(curve["x_numeric"])
    ax2.set_xticklabels(curve["x_labels"])
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_direct, dpi=150, bbox_inches="tight")
    plt.close(fig2)


def generate_nn_epoch_diagnostics(
    *,
    ranking_df: pd.DataFrame,
    sequence_length: int,
    n_features: int,
    X_seq_3d,
    X_seq_flat,
    y_seq,
    sample_dates,
    output_dir: Path,
    output_prefix: str,
    feature_title: str,
) -> tuple[str, list[Path]]:
    ranked_df = rank_models_by_metrics(ranking_df)
    plot_model_name = str(ranked_df.iloc[0]["Model"])
    output_bv = output_dir / f"{output_prefix}_nn_best_bias_variance.png"
    output_direct = output_dir / f"{output_prefix}_nn_best_train_test.png"

    if plot_model_name.endswith("LSTM"):
        params = _first_grid_params(NN_LSTM_PARAM_GRID)
        params.pop("sequence_length", None)
        epoch_values = list(NN_LSTM_PARAM_GRID["epochs"])
        estimator = _KerasSequenceClassifier(
            architecture="lstm",
            sequence_length=sequence_length,
            n_features=n_features,
            verbose=0,
            **params,
        )
        epoch_param_name = "epochs"
        model_title = "LSTM"
    elif plot_model_name.endswith("RNN"):
        params = _first_grid_params(NN_RNN_PARAM_GRID)
        params.pop("sequence_length", None)
        epoch_values = list(NN_RNN_PARAM_GRID["epochs"])
        estimator = _KerasSequenceClassifier(
            architecture="rnn",
            sequence_length=sequence_length,
            n_features=n_features,
            verbose=0,
            **params,
        )
        epoch_param_name = "epochs"
        model_title = "RNN"
    else:
        params = _first_pipeline_params(NN_CNN_PARAM_GRID, "classifier__")
        params.pop("sequence_length", None)
        epoch_values = list(NN_CNN_PARAM_GRID["classifier__max_iter"])
        estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", MLPClassifier(
                random_state=RANDOM_SEED,
                verbose=0,
                **params,
            )),
        ])
        epoch_param_name = "classifier__max_iter"
        model_title = "CNN"

    X_plot = X_seq_flat if plot_model_name.endswith("CNN") else X_seq_3d
    X_train, X_test, y_train, y_test, _ = _train_test_split_with_optional_dates(X_plot, y_seq, sample_dates)
    curve = _compute_epoch_curve(estimator, X_train, y_train, X_test, y_test, epoch_values, epoch_param_name)
    _save_nn_curve_pair(
        curve,
        model_title=plot_model_name,
        feature_title=feature_title,
        x_label="Epochs / Max Iterations\n← Simpler training budget                    More training budget →",
        output_bv=output_bv,
        output_direct=output_direct,
    )
    return plot_model_name, [output_bv, output_direct]


def main() -> None:
    print("=" * 70)
    print("NEURAL NETWORK COMPARISON")
    print("=" * 70)

    sequence_length = _resolve_default_nn_sequence_length()
    sequence_data = _load_sequence_data(sequence_length)
    X_seq_3d = sequence_data.X_seq_3d
    X_seq_flat = sequence_data.X_seq_flat
    y_seq = sequence_data.y_seq
    seq_dates = sequence_data.seq_dates
    n_features = X_seq_3d.shape[2]

    rows = []
    output_dir = cwd / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = get_checkpoint_dir(output_dir, "NN", f"{DATASET_LABEL}_{GRID_VERSION}")

    def _compute_lstm_metrics():
        lstm_params = _first_grid_params(NN_LSTM_PARAM_GRID)
        lstm_sequence_length = int(lstm_params.pop("sequence_length", sequence_length))
        lstm_estimator = _KerasSequenceClassifier(
            architecture="lstm",
            sequence_length=lstm_sequence_length,
            n_features=n_features,
            verbose=0,
            **lstm_params,
        )
        return _evaluate_model("LSTM", lstm_estimator, X_seq_3d, y_seq, sample_dates=seq_dates)

    lstm_payload = _load_or_compute_model_payload(checkpoint_dir, "lstm", _compute_lstm_metrics)
    lstm_metrics = lstm_payload["metrics"]
    rows.append(comparison_row_from_metrics("LSTM", lstm_metrics))

    def _compute_rnn_metrics():
        rnn_params = _first_grid_params(NN_RNN_PARAM_GRID)
        rnn_sequence_length = int(rnn_params.pop("sequence_length", sequence_length))
        rnn_estimator = _KerasSequenceClassifier(
            architecture="rnn",
            sequence_length=rnn_sequence_length,
            n_features=n_features,
            verbose=0,
            **rnn_params,
        )
        return _evaluate_model("RNN", rnn_estimator, X_seq_3d, y_seq, sample_dates=seq_dates)

    rnn_payload = _load_or_compute_model_payload(checkpoint_dir, "rnn", _compute_rnn_metrics)
    rnn_metrics = rnn_payload["metrics"]
    rows.append(comparison_row_from_metrics("RNN", rnn_metrics))

    def _compute_cnn_metrics():
        cnn_params = _first_pipeline_params(NN_CNN_PARAM_GRID, "classifier__")
        cnn_sequence_length = int(cnn_params.pop("sequence_length", sequence_length))
        if cnn_sequence_length != sequence_length:
            raise ValueError(
                "CNN grid default sequence_length does not match the loaded sequence data. "
                "Keep the first sequence_length value aligned across NN grids."
            )
        cnn_estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", MLPClassifier(
                random_state=RANDOM_SEED,
                verbose=0,
                **cnn_params,
            )),
        ])
        return _evaluate_model("CNN", cnn_estimator, X_seq_flat, y_seq, sample_dates=seq_dates)

    cnn_payload = _load_or_compute_model_payload(checkpoint_dir, "cnn", _compute_cnn_metrics)
    cnn_metrics = cnn_payload["metrics"]
    rows.append(comparison_row_from_metrics("CNN", cnn_metrics))

    ranking_df = pd.DataFrame([
        {"Model": "LSTM", **lstm_metrics},
        {"Model": "RNN", **rnn_metrics},
        {"Model": "CNN", **cnn_metrics},
    ])

    comparison_df = build_base_style_comparison_df(rows)
    export_df = build_compact_export_table(
        comparison_df,
        keep_cols=["ROC-AUC", "MCC", "Test Acc", "Sensitivity (Macro)", "Specificity"],
    )
    plot_model_name, plot_paths = generate_nn_epoch_diagnostics(
        ranking_df=ranking_df,
        sequence_length=sequence_length,
        n_features=n_features,
        X_seq_3d=X_seq_3d,
        X_seq_flat=X_seq_flat,
        y_seq=y_seq,
        sample_dates=seq_dates,
        output_dir=output_dir,
        output_prefix="8yrs",
        feature_title="Engineered sequence features",
    )
    tex_path = output_dir / "8yrs_nn_comparison.tex"

    write_base_style_latex_table(
        export_df,
        tex_path,
        caption="Neural-network model comparison on engineered sequence features.",
        label="tab:nn_comparison",
        note="Columns follow the base-model comparison format, excluding Precision, F1, and cross-validation columns because neural-network hyperparameters have not yet been tuned with cross-validation.",
    )

    print("\nNeural-network comparison table:")
    print(export_df)
    print(f"\nSaved LaTeX table to: {tex_path}")
    print(f"Local plot winner in NN.py: {plot_model_name}")
    for path in plot_paths:
        print(f"Saved figure to: {path}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    global_ranked_df = register_global_model_candidates(
        ranking_df,
        output_dir / "8yrs_global_model_leaderboard.csv",
        source_script="NN.py",
        dataset_label="8yrs",
        comparison_scope="tuned_candidates",
    )
    print(f"Current global best model across registered scripts (informational only): {global_ranked_df.iloc[0]['Model']}")


if __name__ == "__main__":
    main()
