#!/bin/bash
# Mixed-frequency (Schorfheide-Song) prior-selection experiment on Euler.
# One task per forecast origin (151 monthly origins, 1997-07 to 2010-01), four
# priors per origin (published values, marginal likelihood, GDP RMSE over 12
# quarters, joint RMSE over 60 quarters). Resumes from per-origin checkpoints,
# so a resubmission after the time limit continues where it stopped.
# Needs the MBFVAR version pinned in requirements.txt (>= 0.9.1):
#   pip install -r requirements.txt

#SBATCH --nodes=1
#SBATCH -n 48
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=2500
#SBATCH --job-name=mf_ladder
#SBATCH --output=mf_ladder_%j.log

set -euo pipefail

module load stack/2025-06
module load gcc/12.2.0
module load python/3.13.0

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/mf_ladder}"
cd "$REPO_ROOT"

python scripts/run_mf_ladder.py \
  --output-dir "$OUTPUT_DIR" \
  --criteria "ml,rmse:h4:w12,rmsej:h4:w60" \
  --fit-nsim 4000 \
  --keep-draws 50 \
  --n-starts 4 \
  --maxfev 400 \
  --n-workers "${SLURM_NTASKS:-48}"

python scripts/compare_mf_ladder.py --run-dir "$OUTPUT_DIR"
