#!/usr/bin/env bash
# Random evaluation origins versus the last W origins: does spreading a short
# evaluation set over the last 20 years (p80) stop the short-window overfitting?
# Compares rN:p80 with wN (same count) and w80 (same span); rN without a pool
# draws from the whole history (compare wall); s1/s2 redraw the subset.
# Starts once the other-horizons queue has finished. Resumes from checkpoints.
cd /run/media/system/Data/Repositories/paper_hyperparameter_optimization
WAIT_PID=$(cat logs/queue_other_horizons.pid)
while grep -q queue_other_horizons "/proc/$WAIT_PID/cmdline" 2>/dev/null; do sleep 300; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
COMMON="--variables GDP --mcmc-draws 20000 --mcmc-discard 5000 --seed-base 2015 --optimization-seed 123 --per-origin-selection --ladder-draws 200 --ladder-starts 4"
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
SMALL="rmse:h4:r12:p80,rmsej:h4:r12:p80,rmse:h4:r40:p80,rmsej:h4:r40:p80,rmse:h4:r20:p80,rmsej:h4:r20:p80"
SMALL="$SMALL,rmse:h4:r12:p80:s1,rmsej:h4:r12:p80:s1,rmse:h4:r12:p80:s2,rmsej:h4:r12:p80:s2"
SMALL="$SMALL,rmse:h4:r12,rmsej:h4:r12,lsj1:r12:p80,rmsej:h4:w20"
run_batch logs/canonical_small_random.log --output-root outputs/glp/canonical_small --model-size small \
  --n-workers 15 --ladder-rungs "$SMALL"
MEDIUM="rmse:h4:r12:p80,rmsej:h4:r12:p80,rmse:h4:r40:p80,rmsej:h4:r40:p80,lsj1:r12:p80"
run_batch logs/canonical_medium_random.log --output-root outputs/glp/canonical_medium --model-size medium \
  --n-workers 15 --ladder-rungs "$MEDIUM"
