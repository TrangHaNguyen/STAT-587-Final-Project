#!/usr/bin/env python3
import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "PyScripts" / "Models"
RESULTS_DIR = PROJECT_ROOT / "output" / "results"
PLOTS_DIR = PROJECT_ROOT / "output" / "Used" / "png"
STATE_FILE = RESULTS_DIR / "all_pipeline_state.json"
LOG_FILE = RESULTS_DIR / "all_pipeline.log"


def now_iso() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"stages": {}, "updated_at": now_iso()}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"stages": {}, "updated_at": now_iso()}


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = now_iso()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(path)


def append_log(message: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{now_iso()}] {message}"
    print(line, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def stream_run(cmd: list[str], env: dict[str, str], stage_log: Path) -> int:
    stage_log.parent.mkdir(parents=True, exist_ok=True)
    with stage_log.open("a", encoding="utf-8") as f:
        f.write(f"\n[{now_iso()}] CMD: {' '.join(cmd)}\n")
        f.write(f"[{now_iso()}] ENV: MODEL_N_JOBS={env.get('MODEL_N_JOBS')} GRID_VARIANT={env.get('GRID_VARIANT')} GRID_VERSION={env.get('GRID_VERSION')}\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            f.write(line)
        rc = proc.wait()
        f.write(f"[{now_iso()}] EXIT_CODE={rc}\n")
    return rc


def run_stage(
    state: dict[str, Any],
    stage_id: str,
    cmd: list[str],
    env: dict[str, str],
    resume: bool,
    stop_on_error: bool
) -> bool:
    stages = state.setdefault("stages", {})
    stage = stages.get(stage_id, {})

    if resume and stage.get("status") == "completed":
        append_log(f"SKIP completed stage: {stage_id}")
        return True

    stage_log = RESULTS_DIR / f"{stage_id.replace(':', '_')}.log"
    append_log(f"START stage: {stage_id}")
    t0 = time.time()
    rc = stream_run(cmd, env, stage_log)
    dt = time.time() - t0

    stages[stage_id] = {
        "status": "completed" if rc == 0 else "failed",
        "return_code": rc,
        "duration_sec": round(dt, 2),
        "finished_at": now_iso(),
        "command": cmd,
        "log_file": str(stage_log)
    }
    save_state(STATE_FILE, state)

    if rc == 0:
        append_log(f"DONE stage: {stage_id} ({dt:.1f}s)")
        return True

    append_log(f"FAIL stage: {stage_id} (rc={rc}, {dt:.1f}s)")
    if stop_on_error:
        raise RuntimeError(f"Stage failed: {stage_id}")
    return False


def build_model_stages(models: list[str], variants: list[str], grid_version: str) -> list[tuple[str, list[str], str]]:
    model_to_script = {
        "svm": "SVM.py",
        "logreg": "logistic_regression.py",
        "rf": "random_forest.py"
    }
    stages: list[tuple[str, list[str], str]] = []
    for variant in variants:
        grid_label = f"{grid_version}_{variant}"
        for model in models:
            script = model_to_script[model]
            stage_id = f"model:{model}:{grid_label}"
            cmd = [sys.executable, str(MODELS_DIR / script)]
            stages.append((stage_id, cmd, variant))
    return stages


def parse_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def history_path_for_model(model: str) -> Path:
    mapping = {
        "svm": RESULTS_DIR / "search_history_svm.csv",
        "logreg": RESULTS_DIR / "search_history_logreg.csv",
        "rf": RESULTS_DIR / "search_history_rf.csv",
    }
    return mapping[model]


def plot_input_available(model: str, model_name: str, grid_version: str, x_param: str) -> tuple[bool, str]:
    history_path = history_path_for_model(model)
    if not history_path.exists():
        return False, f"history file missing: {history_path.name}"
    try:
        # Defensive read: skip malformed lines in accumulated CSV history files.
        df = pd.read_csv(history_path, engine="python", on_bad_lines="skip")
    except Exception as exc:
        return False, f"failed reading {history_path.name}: {exc}"

    required = {"model_name", "grid_version", "mean_test_score", x_param}
    missing = [c for c in required if c not in df.columns]
    if missing:
        return False, f"missing column(s): {missing}"

    view = df[(df["model_name"] == model_name) & (df["grid_version"] == grid_version)]
    view = view.dropna(subset=[x_param, "mean_test_score"])
    if view.empty:
        return False, "no matching rows for model_name/grid_version/x_param"
    return True, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all model searches safely with checkpoint/resume and optional plotting.")
    parser.add_argument("--models", default="svm,logreg,rf", help="Comma list: svm,logreg,rf")
    parser.add_argument("--grid-variants", default="left,center,right", help="Comma list: left,center,right")
    parser.add_argument("--grid-version", default="v1", help="Grid version label (e.g., v3).")
    parser.add_argument("--n-jobs", type=int, default=4, help="MODEL_N_JOBS passed to model scripts.")
    parser.add_argument("--resume", action="store_true", help="Skip completed stages from checkpoint.")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Ignore checkpoint and rerun all stages.")
    parser.add_argument("--stop-on-error", action="store_true", help="Abort on first failed stage.")
    parser.add_argument("--skip-models", action="store_true", help="Skip model runs; useful for plot-only refresh.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip over_under_fit plotting stages.")
    parser.add_argument("--notes", default="", help="SEARCH_NOTES passed to model scripts.")
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    models = parse_list(args.models)
    variants = parse_list(args.grid_variants)
    allowed_models = {"svm", "logreg", "rf"}
    allowed_variants = {"left", "center", "right"}

    unknown_models = [m for m in models if m not in allowed_models]
    unknown_variants = [v for v in variants if v not in allowed_variants]
    if unknown_models:
        raise ValueError(f"Unknown models: {unknown_models}. Allowed: {sorted(allowed_models)}")
    if unknown_variants:
        raise ValueError(f"Unknown grid variants: {unknown_variants}. Allowed: {sorted(allowed_variants)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state(STATE_FILE)

    base_env = os.environ.copy()
    base_env["MPLBACKEND"] = "Agg"
    base_env["MPLCONFIGDIR"] = "/tmp/mpl"
    base_env["OMP_NUM_THREADS"] = "1"
    base_env["OPENBLAS_NUM_THREADS"] = "1"
    base_env["MKL_NUM_THREADS"] = "1"
    base_env["VECLIB_MAXIMUM_THREADS"] = "1"
    base_env["NUMEXPR_NUM_THREADS"] = "1"
    base_env["MODEL_N_JOBS"] = str(args.n_jobs)
    base_env["PAUSE_BETWEEN_MODELS"] = "0"
    base_env["GRID_VERSION"] = args.grid_version
    base_env["SEARCH_NOTES"] = args.notes

    append_log(
        f"Pipeline start | models={models} variants={variants} grid_version={args.grid_version} "
        f"n_jobs={args.n_jobs} resume={args.resume} skip_models={args.skip_models} skip_plots={args.skip_plots}"
    )

    if not args.skip_models:
        for stage_id, cmd, variant in build_model_stages(models, variants, args.grid_version):
            env = base_env.copy()
            env["GRID_VARIANT"] = variant
            run_stage(state, stage_id, cmd, env, resume=args.resume, stop_on_error=args.stop_on_error)

    if not args.skip_plots:
        # Run plots after models, stage-by-stage.
        for variant in variants:
            gv = f"{args.grid_version}_{variant}"
            plot_specs = []
            if "svm" in models:
                plot_specs.extend([
                    ("svm_linear_c", ["--model", "svm", "--x-param", "param_classifier__C", "--model-name", "SVM_linear"]),
                    ("svm_rbf_c", ["--model", "svm", "--x-param", "param_classifier__C", "--model-name", "SVM_rbf"]),
                    ("svm_rbf_gamma", ["--model", "svm", "--x-param", "param_classifier__gamma", "--model-name", "SVM_rbf",
                                       "--ordered-x", "0.001,0.01,0.1,1,10,auto,scale"]),
                    ("svm_poly_degree", ["--model", "svm", "--x-param", "param_classifier__degree", "--model-name", "SVM_poly"]),
                ])
            if "logreg" in models:
                plot_specs.extend([
                    ("logreg_lasso_c", ["--model", "logreg", "--x-param", "param_classifier__C", "--model-name", "LogReg_LASSO_internal"]),
                    ("logreg_ridge_c", ["--model", "logreg", "--x-param", "param_classifier__C", "--model-name", "LogReg_Ridge_internal"]),
                    ("logreg_pca_c", ["--model", "logreg", "--x-param", "param_classifier__C", "--model-name", "LogReg_PCA_Ridge"]),
                    ("logreg_pca_ncomp", ["--model", "logreg", "--x-param", "param_pca__n_components", "--model-name", "LogReg_PCA_Ridge"]),
                ])
            if "rf" in models:
                plot_specs.extend([
                    ("rf_base_depth", ["--model", "rf", "--x-param", "param_classifier__max_depth", "--model-name", "RF_base"]),
                    ("rf_base_nest", ["--model", "rf", "--x-param", "param_classifier__n_estimators", "--model-name", "RF_base"]),
                    ("rf_pca_ncomp", ["--model", "rf", "--x-param", "param_reducer__n_components", "--model-name", "RF_pca"]),
                    ("rf_lasso_c", ["--model", "rf", "--x-param", "param_feature_selector__estimator__C", "--model-name", "RF_lasso"]),
                    ("rf_ridge_c", ["--model", "rf", "--x-param", "param_feature_selector__estimator__C", "--model-name", "RF_ridge"]),
                ])

            for name, core_args in plot_specs:
                model = core_args[1]
                x_param = core_args[3]
                model_name = core_args[5]
                ok, reason = plot_input_available(model=model, model_name=model_name, grid_version=gv, x_param=x_param)
                if not ok:
                    append_log(f"SKIP plot:{name}:{gv} ({reason})")
                    continue
                out_path = PLOTS_DIR / f"over_under_fit_{name}_{gv}.png"
                cmd = [
                    sys.executable,
                    str(MODELS_DIR / "over_under_fit.py"),
                    *core_args,
                    "--grid-version", gv,
                    "--trend-window", "8",
                    "--out", str(out_path)
                ]
                stage_id = f"plot:{name}:{gv}"
                run_stage(state, stage_id, cmd, base_env, resume=args.resume, stop_on_error=args.stop_on_error)

    append_log("Pipeline finished.")


if __name__ == "__main__":
    main()
