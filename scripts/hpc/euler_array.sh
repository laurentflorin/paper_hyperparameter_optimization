#!/usr/bin/env bash
# Euler (LSF/Slurm) job-array template for the scope study.
#
# Usage — submit the whole matrix:
#   NJOBS=$(python scripts/run_scope_study.py --config configs/paper_experiment.json \
#              --dry-run 2>&1 | grep "^[0-9]" | wc -l)
#   sbatch --array=0-$((NJOBS-1)) scripts/hpc/euler_array.sh
#
# Or build the array range after running --plan:
#   sbatch --array=0-19 scripts/hpc/euler_array.sh
#
# Environment variables (override on sbatch command line or in your profile):
#   OUTPUT_ROOT        : output root dir  (default: outputs/scope_study)
#   LOG_DIR            : per-job log dir  (default: outputs/scope_study/logs)
#   GLP_PANEL          : path to GLP realtime panel
#   MFVAR_PANEL        : path to MFVAR realtime panel
#   RIDGE_PANEL        : path to ridge panel
#   GLP_WORKERS        : parallel workers per GLP job (default: 1, set to SLURM_CPUS_PER_TASK)
#   RIDGE_WORKERS      : parallel workers per ridge job (default: 1)
#   MAX_JOBS           : concurrent jobs from orchestrator (irrelevant in array mode; keep 1)
#   MAX_NESTED         : per-job nested parallelism limit (set to SLURM_CPUS_PER_TASK)
#   SEED_BASE          : global random seed base (default: 20150101)

#SBATCH --job-name=scope_study
#SBATCH --output=outputs/scope_study/logs/array_%A_%a.out
#SBATCH --error=outputs/scope_study/logs/array_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1

set -euo pipefail

# ---- environment -----------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# Activate virtual environment if present
if [[ -f .python-version ]]; then
    PY_VERSION=$(cat .python-version | tr -d '[:space:]')
fi
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
elif command -v conda &>/dev/null && conda info --envs | grep -q paper_hpo; then
    source activate paper_hpo
    PYTHON=python
else
    PYTHON=python
fi

# Propagate CPU count to nested workers
export GLP_WORKERS="${GLP_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
export RIDGE_WORKERS="${GLP_WORKERS}"
export MAX_NESTED="${MAX_NESTED:-${SLURM_CPUS_PER_TASK:-1}}"

# Prevent BLAS/MKL oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ---- validate environment ---------------------------------------------------
echo "Job array index: ${SLURM_ARRAY_TASK_ID}"
echo "Running from: $REPO_ROOT"
echo "Python: $($PYTHON --version)"
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Host: $(hostname)"
echo "CPUS: ${SLURM_CPUS_PER_TASK:-1}"
echo "GLP_WORKERS: $GLP_WORKERS"

# ---- run the job at this array index ----------------------------------------
exec "$PYTHON" scripts/run_scope_study.py \
    --config configs/paper_experiment.json \
    --job-index "${SLURM_ARRAY_TASK_ID}" \
    --resume \
    ${OUTPUT_ROOT:+--output-root "$OUTPUT_ROOT"}
