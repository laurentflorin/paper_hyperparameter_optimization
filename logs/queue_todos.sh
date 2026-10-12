#!/usr/bin/env bash
# Runs for the remaining paper TODOs: (1) the EM check of the MF-VAR
# marginal-likelihood selection at 16 origins; (2) the full GLP large list with
# reserves and the monetary base (GLP, its mode, and the two headline criteria
# at 80 quarters). Both resume or are cheap to rerun.
cd /run/media/system/Data/Repositories/paper_hyperparameter_optimization
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] EM check" >> logs/mf_em_check.log
.venv/bin/python -u scripts/mf_em_check.py --n-origins 16 --iterations 4 --fit-nsim 4000 --keep-draws 200 --n-workers 6 >> logs/mf_em_check.log 2>&1
GLP_COMMON="--variables GDP --mcmc-draws 20000 --mcmc-discard 5000 --seed-base 2015 --optimization-seed 123 --per-origin-selection --ladder-draws 200"
for attempt in 1 2 3; do
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] attempt $attempt" >> logs/canonical_large_full.log
  .venv/bin/python -u scripts/glp/run_glp_all.py --stages paper,paper_plugin,ladder,compare $GLP_COMMON \
    --output-root outputs/glp/canonical_large --model-size large --reselect-every 4 --ladder-starts 2 --n-workers 8 \
    --ladder-rungs "lsj1:w80,rmsej:h4:w80" >> logs/canonical_large_full.log 2>&1 && exit 0
  sleep 60
done
