#!/bin/bash
# ORCA profile for NN scripts. Runs base_NN.py then NN.py by default.
# Submit with a dependency if you want it to wait for another job, e.g.:
#   sbatch --dependency=afterok:81704 run_nn.slurm
# Or regardless of success/failure:
#   sbatch --dependency=afterany:81704 run_nn.slurm
# Optional overrides:
#   USE_UV=1 sbatch run_nn.slurm
#   RUN_BOTH=0 NN_SCRIPT=PyScripts/Models/NN.py sbatch run_nn.slurm
#SBATCH --job-name=nn_only
#SBATCH --output=nn_only-%j.out
#SBATCH --error=nn_only-%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --hint=nomultithread
#SBATCH --requeue

USE_UV="${USE_UV:-0}"
RUN_BOTH="${RUN_BOTH:-1}"
NN_SCRIPT="${NN_SCRIPT:-PyScripts/Models/NN.py}"

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"

mkdir -p output

export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export MPLCONFIGDIR="$TMPDIR/mpl"
mkdir -p "$MPLCONFIGDIR"

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export VECLIB_MAXIMUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export TF_NUM_INTEROP_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=""

if command -v module >/dev/null 2>&1; then
    module purge
    if [ "$USE_UV" = "1" ]; then
        module load uv
    else
        module load python/3.11 || module load python/3.12 || true
    fi
fi

# NN scripts require a separate venv built with Python 3.11 + requirements_nn.txt
# (numpy==1.26.4, TensorFlow 2.17.x) which conflicts with the numpy==2.4.2 in
# the main .venv used by other models. Prefer .venv_nn, fall back to .venv311
# or .venv if that is what was created on this ORCA instance.
VENV_PATH=""
if [ -f "$PROJECT_ROOT/.venv_nn/bin/activate" ]; then
    VENV_PATH="$PROJECT_ROOT/.venv_nn"
elif [ -f "$PROJECT_ROOT/.venv311/bin/activate" ]; then
    VENV_PATH="$PROJECT_ROOT/.venv311"
elif [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    VENV_PATH="$PROJECT_ROOT/.venv"
fi

if [ -z "$VENV_PATH" ]; then
    echo "Missing virtual environment for NN scripts."
    echo "Expected one of: .venv_nn  .venv311  .venv"
    echo "Create it on ORCA with:"
    echo "  module load uv"
    echo "  uv python install 3.11"
    echo "  uv venv --python 3.11 .venv_nn"
    echo "  source .venv_nn/bin/activate"
    echo "  pip install -r requirements_nn.txt"
    exit 1
fi

if [ ! -f "$PROJECT_ROOT/$NN_SCRIPT" ]; then
    echo "NN script not found: $PROJECT_ROOT/$NN_SCRIPT"
    exit 1
fi

if [ "$RUN_BOTH" = "1" ] && [ ! -f "$PROJECT_ROOT/PyScripts/Models/base_NN.py" ]; then
    echo "base_NN.py not found: $PROJECT_ROOT/PyScripts/Models/base_NN.py"
    exit 1
fi

source "$VENV_PATH/bin/activate"

echo "Running from: $PROJECT_ROOT"
echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "USE_UV=$USE_UV"
echo "RUN_BOTH=$RUN_BOTH"
echo "VENV_PATH=$VENV_PATH"
echo "NN_SCRIPT=$NN_SCRIPT"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-unset}"
python --version

python - <<'PY'
import sys
try:
    import tensorflow as tf
except Exception as exc:
    print("TensorFlow import failed:", exc)
    print("Install requirements in the active ORCA environment before submitting this job.")
    sys.exit(1)
print("TensorFlow version:", tf.__version__)
PY

if [ "$RUN_BOTH" = "1" ]; then
    echo "=== Running base_NN.py ==="
    python PyScripts/Models/base_NN.py
    echo "=== Running NN.py ==="
    python PyScripts/Models/NN.py
else
    python "$NN_SCRIPT"
fi
