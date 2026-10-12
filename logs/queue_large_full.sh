#!/usr/bin/env bash
# Resume the full GLP large run (reserves and monetary base) after the
# out-of-memory kill of 2026-10-02: the paper stage is complete, so start at the
# plug-in mode, with 6 workers instead of 8.
cd /run/media/system/Data/Repositories/paper_hyperparameter_optimization
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
GLP_COMMON="--variables GDP --mcmc-draws 20000 --mcmc-discard 5000 --seed-base 2015 --optimization-seed 123 --per-origin-selection --ladder-draws 200"
for attempt in 1 2 3; do
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] attempt $attempt (resume, 6 workers)" >> logs/canonical_large_full.log
  .venv/bin/python -u scripts/glp/run_glp_all.py --stages paper_plugin,ladder,compare $GLP_COMMON \
    --output-root outputs/glp/canonical_large --model-size large --reselect-every 4 --ladder-starts 2 --n-workers 6 \
    --ladder-rungs "lsj1:w80,rmsej:h4:w80" >> logs/canonical_large_full.log 2>&1 && exit 0
  sleep 60
done
