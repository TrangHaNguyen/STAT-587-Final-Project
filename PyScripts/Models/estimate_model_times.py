#!/usr/bin/env python3
"""Summarize historical model runtimes from the project's run logs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output"

RUN_FILES = [
    OUTPUT_DIR / "8yrs_search_runs.csv",
    OUTPUT_DIR / "sample_search_runs.csv",
    OUTPUT_DIR / "OLD" / "results" / "search_runs.csv",
]

MODEL_LABELS = {
    "LogisticRegression": "logreg",
    "RandomForest": "rf",
    "SVM": "svm",
    "base_logreg": "base_logreg",
    "base_rf": "base_rf",
    "base_SVM": "base_svm",
}


def _format_seconds(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rem_seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {rem_seconds:.1f}s"
    hours, rem_minutes = divmod(int(minutes), 60)
    return f"{hours}h {rem_minutes}m {rem_seconds:.1f}s"


def _load_runs(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df = df.copy()
        df["source_file"] = str(path.relative_to(PROJECT_ROOT))
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _normalize_runs(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized["model_key"] = normalized["model_name"].map(MODEL_LABELS).fillna(normalized["model_name"])
    normalized["dataset_label"] = normalized["dataset_version"].fillna("unknown").apply(
        lambda value: "sample" if "sample.parquet" in str(value) else "8yrs"
    )
    return normalized


def _build_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["model_key", "dataset_label"], dropna=False)["run_duration_sec"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
        .sort_values(["dataset_label", "mean", "model_key"])
    )
    grouped["mean_pretty"] = grouped["mean"].map(_format_seconds)
    grouped["median_pretty"] = grouped["median"].map(_format_seconds)
    grouped["range_pretty"] = grouped.apply(
        lambda row: f"{_format_seconds(row['min'])} to {_format_seconds(row['max'])}",
        axis=1,
    )
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate per-model runtime from saved search run logs.")
    parser.add_argument(
        "--dataset",
        choices=["8yrs", "sample", "all"],
        default="all",
        help="Restrict the report to one dataset group.",
    )
    args = parser.parse_args()

    runs = _load_runs(RUN_FILES)
    if runs.empty:
        raise SystemExit("No run history found in output/*_search_runs.csv or output/OLD/results/search_runs.csv")

    runs = _normalize_runs(runs)
    if args.dataset != "all":
        runs = runs[runs["dataset_label"] == args.dataset].copy()

    if runs.empty:
        raise SystemExit(f"No runs found for dataset={args.dataset}")

    summary = _build_summary(runs)

    print("Per-model runtime estimate")
    print(summary[["model_key", "dataset_label", "count", "mean_pretty", "median_pretty", "range_pretty"]].to_string(index=False))
    print("\nMost recent runs")
    latest = (
        runs.sort_values("run_time")
        .groupby(["model_key", "dataset_label"], as_index=False)
        .tail(1)
        .sort_values(["dataset_label", "model_key"])
    )
    latest["run_duration_pretty"] = latest["run_duration_sec"].map(_format_seconds)
    print(
        latest[["model_key", "dataset_label", "grid_version", "n_jobs", "run_duration_pretty", "source_file"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
