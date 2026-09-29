#!/usr/bin/env bash
# 18-variable ladder at an 80-quarter window; starts after the main 18-variable batch,
# resumes from per-origin checkpoints, retried up to 3 times.
cd /run/media/system/Data/Repositories/paper_hyperparameter_optimization
while kill -0 5315 2>/dev/null; do sleep 60; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for attempt in 1 2 3; do
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] attempt $attempt" >> logs/canonical_large_nr_w80.log
  .venv/bin/python -u scripts/glp/run_glp_all.py --stages ladder,compare \
    --output-root outputs/glp/canonical_large_nr --model-size large_nr --variables GDP \
    --mcmc-draws 20000 --mcmc-discard 5000 --seed-base 2015 --optimization-seed 123 --per-origin-selection \
    --reselect-every 4 --ladder-starts 2 --ladder-draws 200 --n-workers 8 \
    --ladder-rungs "rmsej:h4:w80,lsj1:w80,lsj:h4:w80,rmse:h4:w80" \
    >> logs/canonical_large_nr_w80.log 2>&1 && exit 0
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] attempt $attempt failed (exit $?); resuming from checkpoints" >> logs/canonical_large_nr_w80.log
  sleep 60
done
exit 1
