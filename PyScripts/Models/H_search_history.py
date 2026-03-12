#!/usr/bin/env python3
import datetime
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


def _safe_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def append_search_history(
    history_path: Path,
    cv_results: dict | pd.DataFrame,
    run_time: str,
    model_name: str,
    search_type: str,
    grid_version: str,
    notes: str = "",
    best_params: dict | None =None
) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(cv_results).copy()

    keep_cols = []
    for col in df.columns:
        if col.startswith("param_"):
            keep_cols.append(col)
    for col in ["mean_train_score", "mean_test_score", "std_test_score", "rank_test_score"]:
        if col in df.columns:
            keep_cols.append(col)
    if not keep_cols:
        keep_cols = list(df.columns)

    out = df[keep_cols].copy()
    out.insert(0, "run_time", run_time)
    out.insert(1, "model_name", model_name)
    out.insert(2, "search_type", search_type)
    out.insert(3, "grid_version", grid_version)
    out["best_params"] = _safe_value(best_params if best_params is not None else {})
    out["notes"] = notes

    out.to_csv(history_path, mode="a", header=not history_path.exists(), index=False, float_format='%.3f')


def append_search_run(
    runs_path: Path,
    model_name: str,
    run_time: str,
    run_duration_sec: float,
    grid_version: str,
    n_jobs: int,
    dataset_version: str,
    code_commit: str,
    notes: str = ""
) -> None:
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{
        "run_time": run_time,
        "model_name": model_name,
        "run_duration_sec": round(run_duration_sec, 2),
        "grid_version": grid_version,
        "n_jobs": n_jobs,
        "dataset_version": dataset_version,
        "code_commit": code_commit,
        "notes": notes
    }])
    row.to_csv(runs_path, mode="a", header=not runs_path.exists(), index=False, float_format='%.3f')


def get_git_commit(project_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def now_iso() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
