#!/usr/bin/env python3
"""Baseline neural-network comparison on the raw base.py data branch."""

import os
from pathlib import Path

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from H_eval import (
    build_base_style_comparison_df,
    build_compact_export_table,
    comparison_row_from_metrics,
    write_base_style_latex_table,
)
from H_search_history import (
    get_checkpoint_dir,
    load_stage_checkpoint,
    save_stage_checkpoint,
    stage_checkpoint_exists,
)
from NN import (
    _KerasSequenceClassifier,
    _evaluate_model,
    _first_grid_params,
    _first_pipeline_params,
    _load_sequence_data,
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


def _load_or_compute_model_metrics(checkpoint_dir: Path, stage_name: str, compute_fn):
    if stage_checkpoint_exists(checkpoint_dir, stage_name):
        print(f"Loading checkpoint for {stage_name} from {checkpoint_dir / stage_name}")
        return load_stage_checkpoint(checkpoint_dir, stage_name)
    payload = compute_fn()
    save_stage_checkpoint(checkpoint_dir, stage_name, payload)
    return payload


def main() -> None:
    print("=" * 70)
    print("BASELINE NEURAL NETWORK COMPARISON")
    print("=" * 70)

    sequence_data = _load_sequence_data()
    X_seq_3d = sequence_data.X_seq_3d
    X_seq_flat = sequence_data.X_seq_flat
    y_seq = sequence_data.y_seq
    sequence_length = len(NN_LAG_PERIOD)
    n_features = X_seq_3d.shape[2]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = get_checkpoint_dir(OUTPUT_DIR, "base_NN", f"8yrs_{GRID_VERSION}")
    rows = []

    def _compute_lstm_metrics():
        estimator = _KerasSequenceClassifier(
            architecture="lstm",
            sequence_length=sequence_length,
            n_features=n_features,
            verbose=0,
            **_first_grid_params(NN_LSTM_PARAM_GRID),
        )
        return _evaluate_model("Raw LSTM", estimator, X_seq_3d, y_seq)

    rows.append(
        comparison_row_from_metrics(
            "Raw LSTM",
            _load_or_compute_model_metrics(checkpoint_dir, "lstm", _compute_lstm_metrics),
        )
    )

    def _compute_rnn_metrics():
        estimator = _KerasSequenceClassifier(
            architecture="rnn",
            sequence_length=sequence_length,
            n_features=n_features,
            verbose=0,
            **_first_grid_params(NN_RNN_PARAM_GRID),
        )
        return _evaluate_model("Raw RNN", estimator, X_seq_3d, y_seq)

    rows.append(
        comparison_row_from_metrics(
            "Raw RNN",
            _load_or_compute_model_metrics(checkpoint_dir, "rnn", _compute_rnn_metrics),
        )
    )

    def _compute_cnn_metrics():
        estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", MLPClassifier(
                random_state=RANDOM_SEED,
                verbose=0,
                **_first_pipeline_params(NN_CNN_PARAM_GRID, "classifier__"),
            )),
        ])
        return _evaluate_model("Raw CNN", estimator, X_seq_flat, y_seq)

    rows.append(
        comparison_row_from_metrics(
            "Raw CNN",
            _load_or_compute_model_metrics(checkpoint_dir, "cnn", _compute_cnn_metrics),
        )
    )

    comparison_df = build_base_style_comparison_df(rows)
    export_df = build_compact_export_table(
        comparison_df,
        keep_cols=["ROC-AUC", "MCC", "Test Acc", "Recall", "Specificity"],
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
    print(f"Checkpoint directory: {checkpoint_dir}")


if __name__ == "__main__":
    main()
