#!/usr/bin/env python3
"""Scatter plots of family-leading models across accuracy and stability metrics."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

MPLCONFIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".mplconfig")
os.makedirs(MPLCONFIGDIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from H_eval import rank_models_by_metrics


def _default_output_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "output"


def _load_leaderboard(output_dir: Path, dataset_label: str) -> pd.DataFrame:
    leaderboard_path = output_dir / f"{dataset_label}_global_model_leaderboard.csv"
    if not leaderboard_path.exists():
        raise FileNotFoundError(f"Leaderboard not found: {leaderboard_path}")

    df = pd.read_csv(leaderboard_path)
    df = df.loc[df["dataset_label"] == dataset_label].copy()
    if df.empty:
        raise ValueError(f"No leaderboard rows found for dataset_label='{dataset_label}'.")
    return df


def _load_metrics_subset(output_dir: Path, dataset_label: str) -> pd.DataFrame | None:
    candidate_paths = [
        output_dir / f"{dataset_label}_global_model_leaderboard_metrics_subset.csv",
        output_dir / f"{dataset_label}_global_model_leaderboard_tuned_metrics_subset.csv",
    ]
    for path in candidate_paths:
        if path.exists():
            return pd.read_csv(path)
    return None


def _family_leaders(leaderboard_df: pd.DataFrame) -> pd.DataFrame:
    leaders = []
    for source_script, group in leaderboard_df.groupby("source_script", sort=True):
        ranked = rank_models_by_metrics(group).reset_index(drop=True)
        leaders.append(ranked.iloc[0])
    leader_df = pd.DataFrame(leaders).reset_index(drop=True)
    return leader_df.sort_values(["validation_std_accuracy", "test_split_accuracy"], ascending=[True, False])


def _short_source_label(source_script: str) -> str:
    return source_script.replace(".py", "")


def _default_point_label(row: pd.Series) -> str:
    return f"{row['candidate_model']}"


def _balanced_point_label(row: pd.Series) -> str:
    return (
        f"{row['candidate_model']}\n"
        f"Test Acc={float(row['test_split_accuracy']):.3f}"
    )


def _annotate_points(ax, plot_df: pd.DataFrame, x_col: str, y_col: str, label_fn) -> None:
    for _, row in plot_df.iterrows():
        label = label_fn(row)
        ax.annotate(
            label,
            (row[x_col], row[y_col]),
            textcoords="offset points",
            xytext=(7, 7),
            fontsize=6.3,
            ha="left",
            va="bottom",
        )


def _set_plot_limits(ax, plot_df: pd.DataFrame, x_col: str, y_col: str) -> None:
    x_vals = plot_df[x_col].astype(float)
    y_vals = plot_df[y_col].astype(float)
    x_pad = max(0.003, (x_vals.max() - x_vals.min()) * 0.12 if len(x_vals) > 1 else 0.01)
    y_pad = max(0.01, (y_vals.max() - y_vals.min()) * 0.12 if len(y_vals) > 1 else 0.02)
    ax.set_xlim(x_vals.min() - x_pad, x_vals.max() + x_pad)
    ax.set_ylim(y_vals.min() - y_pad, y_vals.max() + y_pad)


def plot_family_leaders_scatter(
    leader_df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
    out_path: Path,
    label_fn=_default_point_label,
) -> None:
    plot_df = leader_df.copy()
    required_cols = [x_col, y_col, "candidate_model", "source_script"]
    plot_df = plot_df.dropna(subset=required_cols).copy()
    if plot_df.empty:
        raise ValueError(f"No rows available for plot: {title}")

    fig, ax = plt.subplots(figsize=(11, 7))

    scatter = ax.scatter(
        plot_df[x_col],
        plot_df[y_col],
        s=90,
        c=range(len(plot_df)),
        cmap="tab10",
        alpha=0.85,
        edgecolors="black",
        linewidths=0.7,
    )
    scatter.set_array(None)

    _annotate_points(ax, plot_df, x_col, y_col, label_fn)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    _set_plot_limits(ax, plot_df, x_col, y_col)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot the leaderboard leader from each model code on accuracy vs CV SD axes."
    )
    parser.add_argument(
        "--dataset-label",
        default="8yrs",
        help="Dataset label used by the leaderboard file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(),
        help="Directory containing leaderboard CSVs and where the plot should be saved.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    leaderboard_df = _load_leaderboard(args.output_dir, args.dataset_label)
    leader_df = _family_leaders(leaderboard_df)
    metrics_subset_df = _load_metrics_subset(args.output_dir, args.dataset_label)

    cv_accuracy_path = args.output_dir / f"{args.dataset_label}_family_leaders_cv_accuracy_vs_cv_sd.png"
    plot_family_leaders_scatter(
        leader_df,
        x_col="validation_std_accuracy",
        y_col="validation_avg_accuracy",
        title=f"Leaderboard Family Leaders\nCV Accuracy vs CV Accuracy SD ({args.dataset_label})",
        x_label="CV Accuracy SD",
        y_label="CV Accuracy",
        out_path=cv_accuracy_path,
    )

    out_path = args.output_dir / f"{args.dataset_label}_family_leaders_accuracy_vs_cv_sd.png"
    plot_family_leaders_scatter(
        leader_df,
        x_col="validation_std_accuracy",
        y_col="test_split_accuracy",
        title=f"Leaderboard Family Leaders\nTest Accuracy vs CV Accuracy SD ({args.dataset_label})",
        x_label="CV Accuracy SD",
        y_label="Test Plain Accuracy",
        out_path=out_path,
    )

    balanced_path = None
    if metrics_subset_df is not None:
        balanced_leaders = leader_df.merge(
            metrics_subset_df[
                [
                    "source_script",
                    "candidate_model",
                    "cv_balanced_accuracy",
                    "cv_balanced_accuracy_sd",
                ]
            ].drop_duplicates(),
            on=["source_script", "candidate_model"],
            how="left",
        )
        if balanced_leaders[["cv_balanced_accuracy", "cv_balanced_accuracy_sd"]].notna().any().any():
            balanced_path = args.output_dir / (
                f"{args.dataset_label}_family_leaders_cv_balanced_accuracy_vs_cv_balanced_sd.png"
            )
            plot_family_leaders_scatter(
                balanced_leaders,
                x_col="cv_balanced_accuracy_sd",
                y_col="cv_balanced_accuracy",
                title=(
                    f"Leaderboard Family Leaders\n"
                    f"CV Balanced Accuracy vs CV Balanced Accuracy SD ({args.dataset_label})"
                ),
                x_label="CV Balanced Accuracy SD",
                y_label="CV Balanced Accuracy",
                out_path=balanced_path,
                label_fn=_balanced_point_label,
            )

    print("Plotted family leaders:")
    print(leader_df[[
        "source_script",
        "candidate_model",
        "validation_avg_accuracy",
        "validation_std_accuracy",
        "test_split_accuracy",
        "average_rank",
    ]].to_string(index=False))
    print(f"\nSaved plot to: {out_path}")
    print(f"Saved plot to: {cv_accuracy_path}")
    if balanced_path is not None:
        print(f"Saved plot to: {balanced_path}")
    else:
        print("Skipped CV balanced accuracy plot: no metrics subset file with balanced-accuracy columns was available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
