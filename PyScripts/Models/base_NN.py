#!/usr/bin/env python3
"""Baseline neural-network comparison on the raw base.py data branch."""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from H_eval import (
    build_base_style_comparison_df,
    build_compact_export_table,
    comparison_row_from_metrics,
    register_global_model_candidates,
    write_base_style_latex_table,
)
from H_prep import clean_data, import_data, to_binary_class
from H_search_history import (
    get_checkpoint_dir,
)
from NN import (
    _KerasSequenceClassifier,
    _evaluate_model,
    _first_grid_params,
    _load_or_compute_model_payload,
    _first_pipeline_params,
    generate_nn_epoch_diagnostics,
    _resolve_default_nn_sequence_length,
    _reshape_sequences,
)
from model_grids import (
    NN_CNN_PARAM_GRID,
    NN_LAG_PERIOD,
    NN_LSTM_PARAM_GRID,
    NN_RNN_PARAM_GRID,
    RANDOM_SEED,
)

GRID_VERSION = os.getenv("GRID_VERSION", "v1")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output"
DATASET_LABEL = "8yrs_raw"


def _keep_raw_stock_ohlcv(X: pd.DataFrame) -> pd.DataFrame:
    idx = pd.IndexSlice
    metrics = ["Open", "Close", "High", "Low", "Volume"]
    return X.loc[:, idx[metrics, "Stocks", :]].copy()


def _load_base_sequence_data(sequence_length: int):
    print("Loading and cleaning 8-year raw baseline data...")
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

    y = to_binary_class(y_regression).to_numpy()
    X_raw = X.to_numpy(dtype=np.float32)

    X_seq_3d, seq_indices = _reshape_sequences(X_raw, sequence_length, flatten=False)
    X_seq_flat, _ = _reshape_sequences(X_raw, sequence_length, flatten=True)
    y_seq = y[seq_indices]
    seq_dates = np.asarray(X.index[seq_indices])

    print(f"Raw feature shape: {X.shape}")
    print(f"Sequence length: {sequence_length}")
    print(f"3D sequence shape: {X_seq_3d.shape}")
    print(f"Flattened sequence shape: {X_seq_flat.shape}")
    print(f"Target shape: {y_seq.shape}\n")
    return SimpleNamespace(
        X_seq_3d=X_seq_3d,
        X_seq_flat=X_seq_flat,
        y_seq=y_seq,
        seq_indices=np.asarray(seq_indices),
        seq_dates=seq_dates,
    )


def main() -> None:
    print("=" * 70)
    print("BASELINE NEURAL NETWORK COMPARISON")
    print("=" * 70)

    sequence_length = _resolve_default_nn_sequence_length()
    sequence_data = _load_base_sequence_data(sequence_length)
    X_seq_3d = sequence_data.X_seq_3d
    X_seq_flat = sequence_data.X_seq_flat
    y_seq = sequence_data.y_seq
    seq_dates = sequence_data.seq_dates
    n_features = X_seq_3d.shape[2]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = get_checkpoint_dir(OUTPUT_DIR, "base_NN", f"{DATASET_LABEL}_{GRID_VERSION}")
    rows = []

    def _compute_lstm_metrics():
        lstm_params = _first_grid_params(NN_LSTM_PARAM_GRID)
        lstm_sequence_length = int(lstm_params.pop("sequence_length", sequence_length))
        estimator = _KerasSequenceClassifier(
            architecture="lstm",
            sequence_length=lstm_sequence_length,
            n_features=n_features,
            verbose=0,
            **lstm_params,
        )
        return _evaluate_model("Raw LSTM", estimator, X_seq_3d, y_seq, sample_dates=seq_dates)

    lstm_payload = _load_or_compute_model_payload(checkpoint_dir, "lstm", _compute_lstm_metrics)
    lstm_metrics = lstm_payload["metrics"]
    rows.append(comparison_row_from_metrics("Raw LSTM", lstm_metrics))

    def _compute_rnn_metrics():
        rnn_params = _first_grid_params(NN_RNN_PARAM_GRID)
        rnn_sequence_length = int(rnn_params.pop("sequence_length", sequence_length))
        estimator = _KerasSequenceClassifier(
            architecture="rnn",
            sequence_length=rnn_sequence_length,
            n_features=n_features,
            verbose=0,
            **rnn_params,
        )
        return _evaluate_model("Raw RNN", estimator, X_seq_3d, y_seq, sample_dates=seq_dates)

    rnn_payload = _load_or_compute_model_payload(checkpoint_dir, "rnn", _compute_rnn_metrics)
    rnn_metrics = rnn_payload["metrics"]
    rows.append(comparison_row_from_metrics("Raw RNN", rnn_metrics))

    def _compute_cnn_metrics():
        cnn_params = _first_pipeline_params(NN_CNN_PARAM_GRID, "classifier__")
        cnn_sequence_length = int(cnn_params.pop("sequence_length", sequence_length))
        if cnn_sequence_length != sequence_length:
            raise ValueError(
                "CNN grid default sequence_length does not match the loaded sequence data. "
                "Keep the first sequence_length value aligned across NN grids."
            )
        estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", MLPClassifier(
                random_state=RANDOM_SEED,
                verbose=0,
                **cnn_params,
            )),
        ])
        return _evaluate_model("Raw CNN", estimator, X_seq_flat, y_seq, sample_dates=seq_dates)

    cnn_payload = _load_or_compute_model_payload(checkpoint_dir, "cnn", _compute_cnn_metrics)
    cnn_metrics = cnn_payload["metrics"]
    rows.append(comparison_row_from_metrics("Raw CNN", cnn_metrics))

    ranking_df = pd.DataFrame([
        {"Model": "Raw LSTM", **lstm_metrics},
        {"Model": "Raw RNN", **rnn_metrics},
        {"Model": "Raw CNN", **cnn_metrics},
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
        output_dir=OUTPUT_DIR,
        output_prefix="8yrs_base",
        feature_title="Raw OHLCV sequence features",
    )

    tex_path = OUTPUT_DIR / "8yrs_base_nn_comparison.tex"
    write_base_style_latex_table(
        export_df,
        tex_path,
        caption="Baseline neural-network model comparison on raw OHLCV sequence features.",
        label="tab:base_nn_comparison",
        note="Columns follow the base-model comparison format, excluding Precision, F1, and cross-validation columns because neural-network hyperparameters have not yet been tuned with cross-validation.",
    )

    print("\nBaseline neural-network comparison table:")
    print(export_df)
    print(f"\nSaved LaTeX table to: {tex_path}")
    print(f"Local plot winner in base_NN.py: {plot_model_name}")
    for path in plot_paths:
        print(f"Saved figure to: {path}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    global_ranked_df = register_global_model_candidates(
        ranking_df,
        OUTPUT_DIR / "8yrs_global_model_leaderboard.csv",
        source_script="base_NN.py",
        dataset_label="8yrs",
        comparison_scope="tuned_candidates",
    )
    print(f"Current global best model across registered scripts (informational only): {global_ranked_df.iloc[0]['Model']}")


if __name__ == "__main__":
    main()
