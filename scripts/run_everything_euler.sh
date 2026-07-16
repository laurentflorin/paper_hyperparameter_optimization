#!/bin/bash

#SBATCH --nodes=1
#SBATCH -n 48
#SBATCH --cpus-per-task=1
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=2500

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
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/euler}"

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_ROOT"

python download_data.py

python run_paper_hyperparameters.py \
  --output-dir "$OUTPUT_ROOT/paper_hyperparameters"

python run_mango_mdd.py \
  --output-dir "$OUTPUT_ROOT/mango_mdd"

python run_mango_rmse.py \
  --output-dir "$OUTPUT_ROOT/mango_rmse"

python scripts/run_mango_rmse_random.py \
  --output-dir "$OUTPUT_ROOT/mango_rmse_random"

python scripts/compare_forecasts.py \
  --paper-dir "$OUTPUT_ROOT/paper_hyperparameters" \
  --mango-mdd-dir "$OUTPUT_ROOT/mango_mdd" \
  --mango-rmse-dir "$OUTPUT_ROOT/mango_rmse" \
  --mango-rmse-random-dir "$OUTPUT_ROOT/mango_rmse_random" \
  --output-dir "$OUTPUT_ROOT/comparison"
