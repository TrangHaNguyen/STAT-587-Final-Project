#!/bin/bash
# ORCA profile for NN.py only. Submit with a dependency if you want it to wait
# for another job, e.g.:
#   sbatch --dependency=afterok:81704 run_nn.slurm
# Or regardless of success/failure:
#   sbatch --dependency=afterany:81704 run_nn.slurm
#SBATCH --job-name=nn_only
#SBATCH --output=nn_only-%j.out
#SBATCH --error=nn_only-%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --hint=nomultithread
#SBATCH --requeue

USE_UV="${USE_UV:-0}"

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

if [ ! -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    echo "Missing $PROJECT_ROOT/.venv"
    echo "Create it on ORCA first before submitting this job."
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

python PyScripts/Models/NN.py
