#!/usr/bin/env python3
import csv
import datetime
import json
import subprocess
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import numpy as np
import pandas as pd
from joblib import dump, load


def _safe_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _read_csv_header(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return []


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
    for col in ["mean_train_score", "std_train_score", "mean_test_score", "std_test_score", "rank_test_score"]:
        if col in df.columns:
            keep_cols.append(col)
    if not keep_cols:
        keep_cols = list(df.columns)

    out = df[keep_cols].copy()
    out.insert(0, "run_time", run_time)
    out.insert(1, "model_name", model_name)
    out.insert(2, "search_type", search_type)
    out.insert(3, "grid_version", grid_version)
    out.insert(4, "selection_metric", "cv_balanced_accuracy")
    if "mean_train_score" in out.columns:
        out["mean_cv_train_balanced_accuracy"] = out["mean_train_score"]
    if "std_train_score" in out.columns:
        out["std_cv_train_balanced_accuracy"] = out["std_train_score"]
    if "mean_test_score" in out.columns:
        out["mean_cv_validation_balanced_accuracy"] = out["mean_test_score"]
    if "std_test_score" in out.columns:
        out["std_cv_validation_balanced_accuracy"] = out["std_test_score"]
    out["best_params"] = _safe_value(best_params if best_params is not None else {})
    out["notes"] = notes

    existing_cols = _read_csv_header(history_path)
    if existing_cols:
        union_cols = list(dict.fromkeys(existing_cols + out.columns.tolist()))
        if union_cols != existing_cols:
            existing = pd.read_csv(history_path, engine="python", on_bad_lines="skip")
            existing = existing.reindex(columns=union_cols)
            out = out.reindex(columns=union_cols)
            combined = pd.concat([existing, out], ignore_index=True)
            combined.to_csv(history_path, index=False, float_format='%.3f')
            return
        out = out.reindex(columns=existing_cols)

    out.to_csv(history_path, mode="a", header=not history_path.exists(), index=False, float_format='%.3f')


def history_has_entry(history_path: Path, model_name: str, grid_version: str) -> bool:
    """Return True if the history CSV already has an entry for this model + grid version."""
    if not history_path.exists():
        return False
    try:
        df = pd.read_csv(history_path, engine="python", on_bad_lines="skip", usecols=["model_name", "grid_version"])
        return bool(((df["model_name"] == model_name) & (df["grid_version"] == grid_version)).any())
    except Exception:
        return False


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


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _restore_param_column_types(values: list[Any]) -> list[Any]:
    cleaned = [None if pd.isna(v) else v for v in values]
    non_missing = [v for v in cleaned if v is not None]
    if not non_missing:
        return cleaned

    lowered = {str(v).strip().lower() for v in non_missing}
    if lowered <= {"true", "false"}:
        return [None if v is None else str(v).strip().lower() == "true" for v in cleaned]

    numeric = pd.to_numeric(pd.Series(non_missing), errors="coerce")
    if numeric.notna().all():
        numeric_values = numeric.to_numpy(dtype=float)
        if np.allclose(numeric_values, np.round(numeric_values)):
            numeric_iter = iter(int(round(v)) for v in numeric_values)
        else:
            numeric_iter = iter(float(v) for v in numeric_values)
        restored = []
        for value in cleaned:
            restored.append(None if value is None else next(numeric_iter))
        return restored

    return cleaned


def get_checkpoint_dir(output_dir: Path, script_name: str, run_label: str) -> Path:
    checkpoint_dir = output_dir / "checkpoints" / script_name / run_label
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def _search_stage_dir(checkpoint_dir: Path, stage_name: str) -> Path:
    stage_dir = checkpoint_dir / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    return stage_dir


def search_checkpoint_exists(checkpoint_dir: Path, stage_name: str) -> bool:
    stage_dir = checkpoint_dir / stage_name
    return (
        (stage_dir / "best_estimator.joblib").exists()
        and ((stage_dir / "cv_results.joblib").exists() or (stage_dir / "cv_results.csv").exists())
        and (stage_dir / "metadata.json").exists()
    )


def save_search_checkpoint(
    checkpoint_dir: Path,
    stage_name: str,
    search_obj,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    stage_dir = _search_stage_dir(checkpoint_dir, stage_name)
    dump(search_obj.best_estimator_, stage_dir / "best_estimator.joblib")
    dump(search_obj.cv_results_, stage_dir / "cv_results.joblib")
    pd.DataFrame(search_obj.cv_results_).to_csv(stage_dir / "cv_results.csv", index=False)
    metadata = {
        "best_params": _json_ready(getattr(search_obj, "best_params_", {})),
        "best_index": _json_ready(getattr(search_obj, "best_index_", None)),
        "best_score": _json_ready(getattr(search_obj, "best_score_", None)),
    }
    if extra_metadata:
        metadata.update(_json_ready(extra_metadata))
    with (stage_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def load_search_checkpoint(checkpoint_dir: Path, stage_name: str):
    stage_dir = checkpoint_dir / stage_name
    estimator = load(stage_dir / "best_estimator.joblib")
    if (stage_dir / "cv_results.joblib").exists():
        cv_results = load(stage_dir / "cv_results.joblib")
    else:
        cv_results_df = pd.read_csv(stage_dir / "cv_results.csv")
        cv_results = cv_results_df.to_dict(orient="list")
        for key, values in list(cv_results.items()):
            if key.startswith("param_"):
                cv_results[key] = _restore_param_column_types(values)
    with (stage_dir / "metadata.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    return SimpleNamespace(
        best_estimator_=estimator,
        cv_results_=cv_results,
        best_params_=metadata.get("best_params", {}),
        best_index_=metadata.get("best_index"),
        best_score_=metadata.get("best_score"),
        metadata=metadata,
    )


def stage_checkpoint_exists(checkpoint_dir: Path, stage_name: str) -> bool:
    return (checkpoint_dir / stage_name / "payload.joblib").exists()


def save_stage_checkpoint(
    checkpoint_dir: Path,
    stage_name: str,
    payload: Any,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    stage_dir = _search_stage_dir(checkpoint_dir, stage_name)
    dump(payload, stage_dir / "payload.joblib")
    if extra_metadata is not None:
        with (stage_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(_json_ready(extra_metadata), f, indent=2, sort_keys=True)


def load_stage_checkpoint(checkpoint_dir: Path, stage_name: str) -> Any:
    return load(checkpoint_dir / stage_name / "payload.joblib")
