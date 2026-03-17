#!/usr/bin/env python3
"""Baseline neural-network comparison on raw OHLCV and raw OHLCV + DOW features."""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from H_eval import (
    TEST_SELECTION_CRITERIA,
    build_base_style_comparison_df,
    build_compact_export_table,
    comparison_row_from_metrics,
    rank_models_by_metrics,
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
    _load_or_compute_epoch_curve,
    _first_pipeline_params,
    _compute_epoch_curve,
    _train_test_split_with_optional_dates,
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


def _load_raw_and_dow_sequence_data(sequence_length: int):
    """Load data once and return sequence data for both raw and +DOW variants."""
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

    # --- Raw sequences ---
    y = to_binary_class(y_regression).to_numpy()
    X_raw_np = X.to_numpy(dtype=np.float32)
    X_seq_3d, seq_indices = _reshape_sequences(X_raw_np, sequence_length, flatten=False)
    X_seq_flat, _ = _reshape_sequences(X_raw_np, sequence_length, flatten=True)
    y_seq = y[seq_indices]
    seq_dates = np.asarray(X.index[seq_indices])

    print(f"Raw feature shape: {X.shape}")
    print(f"Sequence length: {sequence_length}")
    print(f"3D sequence shape (raw): {X_seq_3d.shape}")
    print(f"Flattened sequence shape (raw): {X_seq_flat.shape}")

    raw_data = SimpleNamespace(
        X_seq_3d=X_seq_3d,
        X_seq_flat=X_seq_flat,
        y_seq=y_seq,
        seq_indices=np.asarray(seq_indices),
        seq_dates=seq_dates,
    )

    # --- DOW sequences: add Mon–Thu dummies (drop Fri to avoid dummy trap) ---
    dow_dummies = pd.get_dummies(X.index.dayofweek, prefix='DOW').astype(float)
    dow_dummies.index = X.index
    dow_dummies = dow_dummies.iloc[:, :-1]
    print(f"Day-of-week columns added: {list(dow_dummies.columns)}")
    X_dow = pd.concat([X, dow_dummies], axis=1)

    X_dow_np = X_dow.to_numpy(dtype=np.float32)
    X_dow_seq_3d, seq_indices_dow = _reshape_sequences(X_dow_np, sequence_length, flatten=False)
    X_dow_seq_flat, _ = _reshape_sequences(X_dow_np, sequence_length, flatten=True)
    y_seq_dow = y[seq_indices_dow]
    seq_dates_dow = np.asarray(X_dow.index[seq_indices_dow])

    print(f"DOW feature shape: {X_dow.shape}")
    print(f"3D sequence shape (DOW): {X_dow_seq_3d.shape}")
    print(f"Flattened sequence shape (DOW): {X_dow_seq_flat.shape}")
    print(f"Target shape: {y_seq.shape}\n")

    dow_data = SimpleNamespace(
        X_seq_3d=X_dow_seq_3d,
        X_seq_flat=X_dow_seq_flat,
        y_seq=y_seq_dow,
        seq_indices=np.asarray(seq_indices_dow),
        seq_dates=seq_dates_dow,
    )

    return raw_data, dow_data


def main() -> None:
    print("=" * 70)
    print("BASELINE NEURAL NETWORK COMPARISON (Raw OHLCV vs +Day-of-Week)")
    print("=" * 70)

    sequence_length = _resolve_default_nn_sequence_length()
    raw_data, dow_data = _load_raw_and_dow_sequence_data(sequence_length)

    # Raw
    X_seq_3d = raw_data.X_seq_3d
    X_seq_flat = raw_data.X_seq_flat
    y_seq = raw_data.y_seq
    seq_dates = raw_data.seq_dates
    n_features = X_seq_3d.shape[2]

    # DOW
    X_dow_seq_3d = dow_data.X_seq_3d
    X_dow_seq_flat = dow_data.X_seq_flat
    y_seq_dow = dow_data.y_seq
    seq_dates_dow = dow_data.seq_dates
    n_features_dow = X_dow_seq_3d.shape[2]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Same checkpoint_dir as base_NN.py so raw-variant checkpoints are shared/reused.
    checkpoint_dir = get_checkpoint_dir(OUTPUT_DIR, "base_NN", f"{DATASET_LABEL}_{GRID_VERSION}")

    X_train_3d, X_test_3d, y_train, y_test, _ = _train_test_split_with_optional_dates(X_seq_3d, y_seq, seq_dates)
    X_train_flat, X_test_flat, _, _, _ = _train_test_split_with_optional_dates(X_seq_flat, y_seq, seq_dates)

    X_dow_train_3d, X_dow_test_3d, y_dow_train, y_dow_test, _ = _train_test_split_with_optional_dates(X_dow_seq_3d, y_seq_dow, seq_dates_dow)
    X_dow_train_flat, X_dow_test_flat, _, _, _ = _train_test_split_with_optional_dates(X_dow_seq_flat, y_seq_dow, seq_dates_dow)

    rows = []

    # ===================================================================
    # RAW OHLCV — Step 1: CV epoch selection (reuse base_NN.py checkpoints)
    # ===================================================================
    def _compute_lstm_curve():
        params = _first_grid_params(NN_LSTM_PARAM_GRID)
        params.pop("sequence_length", None)
        params.pop("epochs", None)
        estimator = _KerasSequenceClassifier(
            architecture="lstm", sequence_length=sequence_length, n_features=n_features, verbose=0, **params
        )
        return _compute_epoch_curve(estimator, X_train_3d, y_train, X_test_3d, y_test, list(NN_LSTM_PARAM_GRID["epochs"]), "epochs")

    lstm_curve = _load_or_compute_epoch_curve(checkpoint_dir, "lstm_epoch_curve", _compute_lstm_curve)
    selected_lstm_epochs = int(lstm_curve["selected_label"])
    print(f"CV-selected LSTM epochs (raw): {selected_lstm_epochs}")

    def _compute_rnn_curve():
        params = _first_grid_params(NN_RNN_PARAM_GRID)
        params.pop("sequence_length", None)
        params.pop("epochs", None)
        estimator = _KerasSequenceClassifier(
            architecture="rnn", sequence_length=sequence_length, n_features=n_features, verbose=0, **params
        )
        return _compute_epoch_curve(estimator, X_train_3d, y_train, X_test_3d, y_test, list(NN_RNN_PARAM_GRID["epochs"]), "epochs")

    rnn_curve = _load_or_compute_epoch_curve(checkpoint_dir, "rnn_epoch_curve", _compute_rnn_curve)
    selected_rnn_epochs = int(rnn_curve["selected_label"])
    print(f"CV-selected RNN epochs (raw): {selected_rnn_epochs}")

    def _compute_cnn_curve():
        params = _first_pipeline_params(NN_CNN_PARAM_GRID, "classifier__")
        params.pop("sequence_length", None)
        params.pop("max_iter", None)
        estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", MLPClassifier(random_state=RANDOM_SEED, verbose=0, **params)),
        ])
        return _compute_epoch_curve(estimator, X_train_flat, y_train, X_test_flat, y_test, list(NN_CNN_PARAM_GRID["classifier__max_iter"]), "classifier__max_iter")

    cnn_curve = _load_or_compute_epoch_curve(checkpoint_dir, "cnn_epoch_curve", _compute_cnn_curve)
    selected_cnn_max_iter = int(cnn_curve["selected_label"])
    print(f"CV-selected CNN max_iter (raw): {selected_cnn_max_iter}")

    raw_curves = {"LSTM": lstm_curve, "RNN": rnn_curve, "CNN": cnn_curve}

    # ===================================================================
    # DOW — Step 1: CV epoch selection (new checkpoints)
    # ===================================================================
    def _compute_lstm_curve_dow():
        params = _first_grid_params(NN_LSTM_PARAM_GRID)
        params.pop("sequence_length", None)
        params.pop("epochs", None)
        estimator = _KerasSequenceClassifier(
            architecture="lstm", sequence_length=sequence_length, n_features=n_features_dow, verbose=0, **params
        )
        return _compute_epoch_curve(estimator, X_dow_train_3d, y_dow_train, X_dow_test_3d, y_dow_test, list(NN_LSTM_PARAM_GRID["epochs"]), "epochs")

    lstm_curve_dow = _load_or_compute_epoch_curve(checkpoint_dir, "lstm_epoch_curve_dow", _compute_lstm_curve_dow)
    selected_lstm_epochs_dow = int(lstm_curve_dow["selected_label"])
    print(f"CV-selected LSTM epochs (DOW): {selected_lstm_epochs_dow}")

    def _compute_rnn_curve_dow():
        params = _first_grid_params(NN_RNN_PARAM_GRID)
        params.pop("sequence_length", None)
        params.pop("epochs", None)
        estimator = _KerasSequenceClassifier(
            architecture="rnn", sequence_length=sequence_length, n_features=n_features_dow, verbose=0, **params
        )
        return _compute_epoch_curve(estimator, X_dow_train_3d, y_dow_train, X_dow_test_3d, y_dow_test, list(NN_RNN_PARAM_GRID["epochs"]), "epochs")

    rnn_curve_dow = _load_or_compute_epoch_curve(checkpoint_dir, "rnn_epoch_curve_dow", _compute_rnn_curve_dow)
    selected_rnn_epochs_dow = int(rnn_curve_dow["selected_label"])
    print(f"CV-selected RNN epochs (DOW): {selected_rnn_epochs_dow}")

    def _compute_cnn_curve_dow():
        params = _first_pipeline_params(NN_CNN_PARAM_GRID, "classifier__")
        params.pop("sequence_length", None)
        params.pop("max_iter", None)
        estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", MLPClassifier(random_state=RANDOM_SEED, verbose=0, **params)),
        ])
        return _compute_epoch_curve(estimator, X_dow_train_flat, y_dow_train, X_dow_test_flat, y_dow_test, list(NN_CNN_PARAM_GRID["classifier__max_iter"]), "classifier__max_iter")

    cnn_curve_dow = _load_or_compute_epoch_curve(checkpoint_dir, "cnn_epoch_curve_dow", _compute_cnn_curve_dow)
    selected_cnn_max_iter_dow = int(cnn_curve_dow["selected_label"])
    print(f"CV-selected CNN max_iter (DOW): {selected_cnn_max_iter_dow}")

    dow_curves = {"LSTM": lstm_curve_dow, "RNN": rnn_curve_dow, "CNN": cnn_curve_dow}

    # ===================================================================
    # RAW OHLCV — Step 2: model evaluation (reuse base_NN.py checkpoints)
    # ===================================================================
    def _compute_lstm_metrics():
        params = _first_grid_params(NN_LSTM_PARAM_GRID)
        lstm_sequence_length = int(params.pop("sequence_length", sequence_length))
        params["epochs"] = selected_lstm_epochs
        estimator = _KerasSequenceClassifier(
            architecture="lstm", sequence_length=lstm_sequence_length, n_features=n_features, verbose=0, **params
        )
        return _evaluate_model("Raw LSTM", estimator, X_seq_3d, y_seq, sample_dates=seq_dates)

    lstm_payload = _load_or_compute_model_payload(checkpoint_dir, "lstm", _compute_lstm_metrics)
    lstm_metrics = lstm_payload["metrics"]
    rows.append(comparison_row_from_metrics("Raw LSTM", lstm_metrics))

    def _compute_rnn_metrics():
        params = _first_grid_params(NN_RNN_PARAM_GRID)
        rnn_sequence_length = int(params.pop("sequence_length", sequence_length))
        params["epochs"] = selected_rnn_epochs
        estimator = _KerasSequenceClassifier(
            architecture="rnn", sequence_length=rnn_sequence_length, n_features=n_features, verbose=0, **params
        )
        return _evaluate_model("Raw RNN", estimator, X_seq_3d, y_seq, sample_dates=seq_dates)

    rnn_payload = _load_or_compute_model_payload(checkpoint_dir, "rnn", _compute_rnn_metrics)
    rnn_metrics = rnn_payload["metrics"]
    rows.append(comparison_row_from_metrics("Raw RNN", rnn_metrics))

    def _compute_cnn_metrics():
        params = _first_pipeline_params(NN_CNN_PARAM_GRID, "classifier__")
        cnn_sequence_length = int(params.pop("sequence_length", sequence_length))
        if cnn_sequence_length != sequence_length:
            raise ValueError(
                "CNN grid default sequence_length does not match the loaded sequence data. "
                "Keep the first sequence_length value aligned across NN grids."
            )
        params["max_iter"] = selected_cnn_max_iter
        estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", MLPClassifier(random_state=RANDOM_SEED, verbose=0, **params)),
        ])
        return _evaluate_model("Raw CNN", estimator, X_seq_flat, y_seq, sample_dates=seq_dates)

    cnn_payload = _load_or_compute_model_payload(checkpoint_dir, "cnn", _compute_cnn_metrics)
    cnn_metrics = cnn_payload["metrics"]
    rows.append(comparison_row_from_metrics("Raw CNN", cnn_metrics))

    raw_ranking_df = pd.DataFrame([
        {"Model": "Raw LSTM", **lstm_metrics},
        {"Model": "Raw RNN", **rnn_metrics},
        {"Model": "Raw CNN", **cnn_metrics},
    ])

    # ===================================================================
    # DOW — Step 2: model evaluation (new checkpoints)
    # DOW model names end with the arch key so generate_nn_epoch_diagnostics
    # can detect the architecture correctly.
    # ===================================================================
    def _compute_lstm_metrics_dow():
        params = _first_grid_params(NN_LSTM_PARAM_GRID)
        lstm_sequence_length = int(params.pop("sequence_length", sequence_length))
        params["epochs"] = selected_lstm_epochs_dow
        estimator = _KerasSequenceClassifier(
            architecture="lstm", sequence_length=lstm_sequence_length, n_features=n_features_dow, verbose=0, **params
        )
        return _evaluate_model("DOW LSTM", estimator, X_dow_seq_3d, y_seq_dow, sample_dates=seq_dates_dow)

    lstm_dow_payload = _load_or_compute_model_payload(checkpoint_dir, "lstm_dow", _compute_lstm_metrics_dow)
    lstm_dow_metrics = lstm_dow_payload["metrics"]
    rows.append(comparison_row_from_metrics("DOW LSTM", lstm_dow_metrics))

    def _compute_rnn_metrics_dow():
        params = _first_grid_params(NN_RNN_PARAM_GRID)
        rnn_sequence_length = int(params.pop("sequence_length", sequence_length))
        params["epochs"] = selected_rnn_epochs_dow
        estimator = _KerasSequenceClassifier(
            architecture="rnn", sequence_length=rnn_sequence_length, n_features=n_features_dow, verbose=0, **params
        )
        return _evaluate_model("DOW RNN", estimator, X_dow_seq_3d, y_seq_dow, sample_dates=seq_dates_dow)

    rnn_dow_payload = _load_or_compute_model_payload(checkpoint_dir, "rnn_dow", _compute_rnn_metrics_dow)
    rnn_dow_metrics = rnn_dow_payload["metrics"]
    rows.append(comparison_row_from_metrics("DOW RNN", rnn_dow_metrics))

    def _compute_cnn_metrics_dow():
        params = _first_pipeline_params(NN_CNN_PARAM_GRID, "classifier__")
        cnn_sequence_length = int(params.pop("sequence_length", sequence_length))
        if cnn_sequence_length != sequence_length:
            raise ValueError(
                "CNN grid default sequence_length does not match the loaded sequence data. "
                "Keep the first sequence_length value aligned across NN grids."
            )
        params["max_iter"] = selected_cnn_max_iter_dow
        estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", MLPClassifier(random_state=RANDOM_SEED, verbose=0, **params)),
        ])
        return _evaluate_model("DOW CNN", estimator, X_dow_seq_flat, y_seq_dow, sample_dates=seq_dates_dow)

    cnn_dow_payload = _load_or_compute_model_payload(checkpoint_dir, "cnn_dow", _compute_cnn_metrics_dow)
    cnn_dow_metrics = cnn_dow_payload["metrics"]
    rows.append(comparison_row_from_metrics("DOW CNN", cnn_dow_metrics))

    dow_ranking_df = pd.DataFrame([
        {"Model": "DOW LSTM", **lstm_dow_metrics},
        {"Model": "DOW RNN", **rnn_dow_metrics},
        {"Model": "DOW CNN", **cnn_dow_metrics},
    ])

    # ===================================================================
    # COMBINED RANKING — determine overall best for plotting
    # ===================================================================
    combined_ranking_df = pd.concat([raw_ranking_df, dow_ranking_df], ignore_index=True)
    combined_ranked = rank_models_by_metrics(combined_ranking_df, criteria=TEST_SELECTION_CRITERIA)
    global_best = str(combined_ranked.iloc[0]["Model"])
    print(f"\nGlobal best model (raw + DOW): {global_best}")

    # Plot using the appropriate group's curves so the epoch curve shown
    # matches the data the best model was trained on.
    if global_best.startswith("DOW"):
        plot_ranking_df = dow_ranking_df
        plot_curves = dow_curves
        feature_title = "Raw OHLCV + Day-of-Week sequence features"
    else:
        plot_ranking_df = raw_ranking_df
        plot_curves = raw_curves
        feature_title = "Raw OHLCV sequence features"

    # ===================================================================
    # Step 3: Plot diagnostics — same output filenames as base_NN.py
    # ===================================================================
    plot_model_name, plot_paths = generate_nn_epoch_diagnostics(
        ranking_df=plot_ranking_df,
        curves=plot_curves,
        output_dir=OUTPUT_DIR,
        output_prefix="8yrs_base",
        feature_title=feature_title,
    )

    # ===================================================================
    # COMPARISON TABLE — same output filename as base_NN.py (overwrite)
    # ===================================================================
    comparison_df = build_base_style_comparison_df(rows)
    export_df = build_compact_export_table(
        comparison_df,
        keep_cols=["ROC-AUC", "MCC", "Test Acc", "Recall", "Specificity"],
    )

    tex_path = OUTPUT_DIR / "8yrs_base_nn_comparison.tex"
    write_base_style_latex_table(
        export_df,
        tex_path,
        caption="Baseline neural-network model comparison on raw OHLCV and raw OHLCV + Day-of-Week sequence features.",
        label="tab:base_nn_comparison",
        note="Columns follow the base-model comparison format, excluding Precision, F1, and cross-validation columns because neural-network hyperparameters have not yet been tuned with cross-validation.",
    )

    print("\nBaseline neural-network comparison table (raw + DOW):")
    print(export_df)
    print(f"\nSaved LaTeX table to: {tex_path}")
    print(f"Local plot winner in base_NN_withDOW.py: {plot_model_name}")
    for path in plot_paths:
        print(f"Saved figure to: {path}")
    print(f"Checkpoint directory: {checkpoint_dir}")

    global_ranked_df = register_global_model_candidates(
        combined_ranking_df,
        OUTPUT_DIR / "8yrs_global_model_leaderboard.csv",
        source_script="base_NN.py",
        dataset_label="8yrs",
        comparison_scope="tuned_candidates",
    )
    print(f"Current global best model across registered scripts (informational only): {global_ranked_df.iloc[0]['Model']}")


if __name__ == "__main__":
    main()
