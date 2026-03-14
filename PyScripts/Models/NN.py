#!/usr/bin/env python3
"""Neural-network comparison table using the shared base-style reporting helpers."""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from typing import NamedTuple
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

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


def _keep_raw_stock_ohlcv(X: pd.DataFrame) -> pd.DataFrame:
    idx = pd.IndexSlice
    metrics = ["Open", "Close", "High", "Low", "Volume"]
    return X.loc[:, idx[metrics, "Stocks", :]].copy()


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


def _load_sequence_data() -> _SequenceData:
    print("Loading and cleaning 8-year data via H_prep...")
    data = import_data(
        testing=False,
        extra_features=False,
        cluster=False,
        n_clusters=100,
        corr_threshold=0.95,
        corr_level=0,
    )
    X, y_regression = clean_data(
        *data,
        lookback_period=0,
        lag_period=NN_LAG_PERIOD,
        extra_features=False,
        raw=True,
        corr_threshold=0.95,
        corr_level=0,
    )
    X = _keep_raw_stock_ohlcv(X)
    y = to_binary_class(y_regression).to_numpy()
    X_raw = X.to_numpy(dtype=np.float32)

    sequence_length = len(NN_LAG_PERIOD)
    X_seq_3d, seq_indices = _reshape_sequences(X_raw, sequence_length, flatten=False)
    X_seq_flat, _ = _reshape_sequences(X_raw, sequence_length, flatten=True)
    y_seq = y[seq_indices]

    print(f"Raw feature shape: {X.shape}")
    print(f"Sequence length: {sequence_length}")
    print(f"3D sequence shape: {X_seq_3d.shape}")
    print(f"Flattened sequence shape: {X_seq_flat.shape}")
    print(f"Target shape: {y_seq.shape}\n")
    return _SequenceData(
        X_seq_3d=X_seq_3d,
        X_seq_flat=X_seq_flat,
        y_seq=y_seq,
        seq_indices=np.asarray(seq_indices),
    )


def _evaluate_model(model_name: str, estimator, X, y) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
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
    return metrics


def _load_or_compute_model_metrics(checkpoint_dir: Path, stage_name: str, compute_fn):
    if stage_checkpoint_exists(checkpoint_dir, stage_name):
        print(f"Loading checkpoint for {stage_name} from {checkpoint_dir / stage_name}")
        return load_stage_checkpoint(checkpoint_dir, stage_name)
    payload = compute_fn()
    save_stage_checkpoint(checkpoint_dir, stage_name, payload)
    return payload


def main() -> None:
    print("=" * 70)
    print("NEURAL NETWORK COMPARISON")
    print("=" * 70)

    sequence_data = _load_sequence_data()
    X_seq_3d = sequence_data.X_seq_3d
    X_seq_flat = sequence_data.X_seq_flat
    y_seq = sequence_data.y_seq
    sequence_length = len(NN_LAG_PERIOD)
    n_features = X_seq_3d.shape[2]

    rows = []
    output_dir = cwd / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = get_checkpoint_dir(output_dir, "NN", f"8yrs_{GRID_VERSION}")

    def _compute_lstm_metrics():
        lstm_estimator = _KerasSequenceClassifier(
            architecture="lstm",
            sequence_length=sequence_length,
            n_features=n_features,
            verbose=0,
            **_first_grid_params(NN_LSTM_PARAM_GRID),
        )
        return _evaluate_model("LSTM", lstm_estimator, X_seq_3d, y_seq)

    lstm_metrics = _load_or_compute_model_metrics(checkpoint_dir, "lstm", _compute_lstm_metrics)
    rows.append(comparison_row_from_metrics("LSTM", lstm_metrics))

    def _compute_rnn_metrics():
        rnn_estimator = _KerasSequenceClassifier(
            architecture="rnn",
            sequence_length=sequence_length,
            n_features=n_features,
            verbose=0,
            **_first_grid_params(NN_RNN_PARAM_GRID),
        )
        return _evaluate_model("RNN", rnn_estimator, X_seq_3d, y_seq)

    rnn_metrics = _load_or_compute_model_metrics(checkpoint_dir, "rnn", _compute_rnn_metrics)
    rows.append(comparison_row_from_metrics("RNN", rnn_metrics))

    def _compute_cnn_metrics():
        cnn_estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", MLPClassifier(
                random_state=RANDOM_SEED,
                verbose=0,
                **_first_pipeline_params(NN_CNN_PARAM_GRID, "classifier__"),
            )),
        ])
        return _evaluate_model("CNN", cnn_estimator, X_seq_flat, y_seq)

    cnn_metrics = _load_or_compute_model_metrics(checkpoint_dir, "cnn", _compute_cnn_metrics)
    rows.append(comparison_row_from_metrics("CNN", cnn_metrics))

    comparison_df = build_base_style_comparison_df(rows)
    export_df = build_compact_export_table(
        comparison_df,
        keep_cols=["Test Acc", "MCC", "Precision", "Recall", "Specificity", "F1", "ROC-AUC"],
    )
    tex_path = output_dir / "8yrs_nn_comparison.tex"

    write_base_style_latex_table(
        export_df,
        tex_path,
        caption="Neural-network model comparison on raw baseline-style sequence features.",
        label="tab:nn_comparison",
        note="Columns follow the base-model comparison format, excluding cross-validation columns because neural-network hyperparameters have not yet been tuned with cross-validation.",
    )

    print("\nNeural-network comparison table:")
    print(export_df)
    print(f"\nSaved LaTeX table to: {tex_path}")
    print(f"Checkpoint directory: {checkpoint_dir}")


if __name__ == "__main__":
    main()
