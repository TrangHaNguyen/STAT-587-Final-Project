#!/bin/bash
# Balanced ORCA profile for All.py: tuned for good throughput without
# over-requesting cluster resources or oversubscribing local threads.
#SBATCH --job-name=all_models
#SBATCH --output=all_models-%j.out
#SBATCH --error=all_models-%j.err
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --partition=compute
#SBATCH --hint=nomultithread
#SBATCH --requeue

# If ORCA requires an allocation/account, uncomment and edit:
#SBATCH --account=your_account

# Toggle this to 1 if your ORCA setup needs uv to build/manage the local
# Python 3.11 environment instead of using a directly available python module.
USE_UV="${USE_UV:-0}"

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

mkdir -p output

export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export MPLCONFIGDIR="$TMPDIR/mpl"
mkdir -p "$MPLCONFIGDIR"

# Stable CPU-only runtime defaults for sklearn/BLAS/TensorFlow.
# Keep these aligned with cpus-per-task to avoid nested oversubscription.
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

# ORCA uses environment modules. Load a Python module if available, then
# activate the project-local virtual environment created on ORCA.
if command -v module >/dev/null 2>&1; then
    module purge
    if [ "$USE_UV" = "1" ]; then
        module load uv
    else
        module load python/3.11 || module load python/3.12 || true
    fi
fi

if [ ! -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    echo "Missing $PROJECT_ROOT/.venv"
    echo "Create it on ORCA first."
    if [ "$USE_UV" = "1" ]; then
        echo "Suggested setup:"
        echo "  module load uv"
        echo "  uv python install 3.11"
        echo "  uv venv --python 3.11 .venv"
        echo "  source .venv/bin/activate"
        echo "  pip install -r requirements.txt"
    else
        echo "Suggested setup:"
        echo "  module load python/3.11"
        echo "  python -m venv .venv"
        echo "  source .venv/bin/activate"
        echo "  pip install -r requirements.txt"
    fi
    exit 1
fi

source "$PROJECT_ROOT/.venv/bin/activate"

echo "Running from: $PROJECT_ROOT"
echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "USE_UV=$USE_UV"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-unset}"
python --version

python PyScripts/Models/All.py \
    --resume \
    --n-jobs "${SLURM_CPUS_PER_TASK:-6}" \
    --grid-version v1
