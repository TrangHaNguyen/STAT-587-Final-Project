#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def default_history_path(project_root: Path, model: str) -> Path:
    mapping = {
        "svm": "8yrs_search_history_svm.csv",
        "logreg": "8yrs_search_history_logreg.csv",
        "rf": "8yrs_search_history_rf.csv"
    }
    return project_root / "output" / mapping[model]


def sort_by_complexity(df: pd.DataFrame, x_col: str, ordered: list[str] | None =None) -> pd.DataFrame:
    out = df.copy()
    if ordered:
        out[x_col] = out[x_col].astype(str)
        out["_x"] = pd.Categorical(out[x_col], categories=ordered, ordered=True)
        out = out.dropna(subset=["_x"]).sort_values("_x")
        out["_x_numeric"] = np.arange(len(out))
        out["_x_tick"] = out[x_col].astype(str)
        return out
    x_num = pd.to_numeric(out[x_col], errors="coerce")
    if x_num.notna().all():
        out["_x_numeric"] = x_num
        out = out.sort_values("_x_numeric")
        out["_x_tick"] = out[x_col].astype(str)
    else:
        out[x_col] = out[x_col].astype(str)
        codes = pd.Categorical(out[x_col]).codes
        out["_x_numeric"] = codes
        out = out.sort_values("_x_numeric")
        out["_x_tick"] = out[x_col]
    return out

def metric_prefers_lower(metric_label: str) -> bool:
    label = metric_label.lower()
    return any(token in label for token in ("error", "loss", "misclassification"))

def select_1se_row(df: pd.DataFrame, lower_is_better: bool =False) -> pd.Series | None:
    if "std_test_score" not in df.columns:
        return None
    tmp = df.dropna(subset=["mean_test_score", "std_test_score", "_x_numeric"]).copy()
    if tmp.empty:
        return None
    best_idx = tmp["mean_test_score"].idxmin() if lower_is_better else tmp["mean_test_score"].idxmax()
    best_mean = float(tmp.loc[best_idx, "mean_test_score"])
    best_std = float(tmp.loc[best_idx, "std_test_score"])
    threshold = best_mean + best_std if lower_is_better else best_mean - best_std
    if lower_is_better:
        candidates = tmp[tmp["mean_test_score"] <= threshold].copy()
    else:
        candidates = tmp[tmp["mean_test_score"] >= threshold].copy()
    if candidates.empty:
        return None
    # Simpler model = lower complexity score on x-axis.
    candidates = candidates.sort_values(
        ["_x_numeric", "mean_test_score"],
        ascending=[True, lower_is_better]
    )
    chosen = candidates.iloc[0].copy()
    chosen["one_se_threshold"] = threshold
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot over/under-fit trends from accumulated search history.")
    parser.add_argument("--model", choices=["svm", "logreg", "rf"], required=True)
    parser.add_argument("--x-param", required=True, help="Parameter column name. Example: param_classifier__C")
    parser.add_argument("--history-file", default="", help="Optional custom history CSV path.")
    parser.add_argument("--grid-version", default="", help="Optional grid version filter.")
    parser.add_argument("--model-name", default="", help="Optional model_name filter.")
    parser.add_argument("--search-type", default="", help="Optional search_type filter.")
    parser.add_argument("--ordered-x", default="", help="Comma-separated explicit order for x values.")
    parser.add_argument("--trend-window", type=int, default=8, help="Rolling window for trend lines.")
    parser.add_argument("--score-label", default="CV Balanced Accuracy", help="Explicit label for plotted score metric.")
    parser.add_argument("--out", default="", help="Output PNG path.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    history_path = Path(args.history_file) if args.history_file else default_history_path(project_root, args.model)
    if not history_path.exists():
        raise FileNotFoundError(f"History file not found: {history_path}")

    # Some history rows may contain unescaped commas in free-text fields.
    # Read defensively and skip malformed lines so plotting can proceed.
    df = pd.read_csv(history_path, engine="python", on_bad_lines="skip")
    if args.x_param not in df.columns:
        raise KeyError(f"{args.x_param} not found in history columns.")

    if args.grid_version:
        df = df[df["grid_version"] == args.grid_version]
    if args.model_name:
        df = df[df["model_name"] == args.model_name]
    if args.search_type:
        df = df[df["search_type"] == args.search_type]

    df = df.dropna(subset=[args.x_param, "mean_test_score"])
    ordered = [x.strip() for x in args.ordered_x.split(",") if x.strip()] if args.ordered_x else None
    df = sort_by_complexity(df, args.x_param, ordered=ordered)
    if df.empty:
        raise ValueError("No rows left after filters.")
    lower_is_better = metric_prefers_lower(args.score_label)

    x = df["_x_numeric"].to_numpy()
    y_test = df["mean_test_score"].to_numpy()
    y_train = df["mean_train_score"].to_numpy() if "mean_train_score" in df.columns else np.full_like(y_test, np.nan, dtype=float)
    y_test_std = df["std_test_score"].to_numpy() if "std_test_score" in df.columns else np.full_like(y_test, np.nan, dtype=float)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y_test, linewidth=1.6, alpha=0.9, label=f"CV Test {args.score_label}")
    plt.scatter(x, y_test, s=16, alpha=0.55)
    if not np.isnan(y_test_std).all():
        lower = np.clip(y_test - y_test_std, 0.0, 1.0)
        upper = np.clip(y_test + y_test_std, 0.0, 1.0)
        plt.fill_between(x, lower, upper, alpha=0.18, label=f"Test {args.score_label} ±1 Std")
    if not np.isnan(y_train).all():
        plt.scatter(x, y_train, s=18, alpha=0.5, label=f"CV Train {args.score_label}")

    win = max(2, args.trend_window)
    if len(df) >= win:
        trend_test = pd.Series(y_test).rolling(win, min_periods=2).mean()
        plt.plot(x, trend_test, linewidth=2, label=f"Test Trend (w={win})")
        if not np.isnan(y_train).all():
            trend_train = pd.Series(y_train).rolling(win, min_periods=2).mean()
            plt.plot(x, trend_train, linewidth=2, label=f"Train Trend (w={win})")

    one_se_row = select_1se_row(df, lower_is_better=lower_is_better)
    best_idx = int(np.argmin(y_test)) if lower_is_better else int(np.argmax(y_test))
    plt.scatter(
        [x[best_idx]],
        [y_test[best_idx]],
        color="gold",
        edgecolor="black",
        s=42,
        zorder=6,
        label=f"Best {args.score_label} point"
    )
    if one_se_row is not None:
        x_1se = float(one_se_row["_x_numeric"])
        y_1se = float(one_se_row["mean_test_score"])
        x_label = str(one_se_row[args.x_param])
        plt.axvline(x_1se, color="crimson", linestyle="--", linewidth=1.4, label=f"1SE Selected ({x_label})")
        plt.scatter([x_1se], [y_1se], color="crimson", s=36, zorder=5)

    plt.xlabel(args.x_param)
    plt.ylabel(args.score_label)
    plt.title(f"{args.model.upper()} Search History: Train/Test {args.score_label} vs {args.x_param}")
    plt.grid(alpha=0.3)
    plt.legend()

    if ordered or (not pd.to_numeric(df[args.x_param], errors="coerce").notna().all()):
        ticks = df["_x_tick"].to_list()
        plt.xticks(x, ticks, rotation=45, ha="right")

    ax = plt.gca()
    arrow_y = -0.17
    label_y = -0.28
    ax.annotate(
        "Lower model complexity",
        xy=(0.02, label_y),
        xycoords="axes fraction",
        ha="left",
        va="center",
        annotation_clip=False
    )
    ax.annotate(
        "Higher model complexity",
        xy=(0.98, label_y),
        xycoords="axes fraction",
        ha="right",
        va="center",
        annotation_clip=False
    )
    ax.annotate(
        "",
        xy=(0.96, arrow_y),
        xytext=(0.04, arrow_y),
        xycoords="axes fraction",
        annotation_clip=False,
        arrowprops=dict(arrowstyle="->", lw=1.2)
    )

    out_path = Path(args.out) if args.out else (
        project_root / "output" / f"8yrs_over_under_fit_{args.model}_{args.x_param.replace('param_', '')}.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.30)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
