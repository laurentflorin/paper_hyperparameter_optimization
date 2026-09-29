#!/usr/bin/env bash
# Point criteria (and the 80-quarter joint density) at target horizons 1, 2, 8
# for the 7-variable and then the 18-variable model. Resumes from per-origin
# checkpoints; each batch is retried up to 3 times.
cd /run/media/system/Data/Repositories/paper_hyperparameter_optimization
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
COMMON="--variables GDP --mcmc-draws 20000 --mcmc-discard 5000 --seed-base 2015 --optimization-seed 123 --per-origin-selection --ladder-draws 200"
run_batch() {  # log, then run_glp_all arguments
  local log=$1; shift
  for attempt in 1 2 3; do
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] attempt $attempt" >> "$log"
    .venv/bin/python -u scripts/glp/run_glp_all.py --stages ladder,compare $COMMON "$@" >> "$log" 2>&1 && return 0
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] attempt $attempt failed; resuming from checkpoints" >> "$log"
    sleep 60
  done
  return 1
}
MED=""; for h in 1 2 8; do for w in 12 60 80; do MED="$MED,rmsej:h$h:w$w,rmse:h$h:w$w"; done; done
run_batch logs/canonical_medium_horizons.log --output-root outputs/glp/canonical_medium --model-size medium \
  --ladder-starts 4 --n-workers 15 --ladder-rungs "${MED#,}"
LRG=""; for h in 1 2 8; do for w in 60 80; do LRG="$LRG,rmsej:h$h:w$w,rmse:h$h:w$w"; done; [ $h -gt 1 ] && LRG="$LRG,lsj:h$h:w80"; done
run_batch logs/canonical_large_nr_horizons.log --output-root outputs/glp/canonical_large_nr --model-size large_nr \
  --reselect-every 4 --ladder-starts 2 --n-workers 8 --ladder-rungs "${LRG#,}"
