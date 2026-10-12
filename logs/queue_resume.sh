#!/usr/bin/env bash
# Resume after the out-of-memory kill of 2026-09-30: the 18-variable other-horizon
# batch (from its checkpoints), then the random-origin runs, then the
# mixed-frequency run. Same rungs as queue_other_horizons.sh,
# queue_random_origins.sh and queue_mf_ladder.sh, with fewer workers to stay
# within memory. Every batch resumes from per-origin checkpoints and is retried.
cd /run/media/system/Data/Repositories/paper_hyperparameter_optimization
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
GLP_COMMON="--variables GDP --mcmc-draws 20000 --mcmc-discard 5000 --seed-base 2015 --optimization-seed 123 --per-origin-selection --ladder-draws 200"
retry() {  # log, then command
  local log=$1; shift
  for attempt in 1 2 3; do
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] attempt $attempt (resume queue)" >> "$log"
    "$@" >> "$log" 2>&1 && return 0
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] attempt $attempt failed; resuming from checkpoints" >> "$log"
    sleep 60
  done
  return 1
}
glp() { .venv/bin/python -u scripts/glp/run_glp_all.py --stages ladder,compare $GLP_COMMON "$@"; }

# 1. 18 variables, target horizons 1, 2 and 8 (10 workers: two rounds of the 20 re-selections).
LRG=""; for h in 1 2 8; do for w in 60 80; do LRG="$LRG,rmsej:h$h:w$w,rmse:h$h:w$w"; done; [ $h -gt 1 ] && LRG="$LRG,lsj:h$h:w80"; done
retry logs/canonical_large_nr_horizons.log glp --output-root outputs/glp/canonical_large_nr --model-size large_nr \
  --reselect-every 4 --ladder-starts 2 --n-workers 10 --ladder-rungs "${LRG#,}"

# 2. Random evaluation origins (7-variable model with 12 workers instead of 15).
SMALL="rmse:h4:r12:p80,rmsej:h4:r12:p80,rmse:h4:r40:p80,rmsej:h4:r40:p80,rmse:h4:r20:p80,rmsej:h4:r20:p80"
SMALL="$SMALL,rmse:h4:r12:p80:s1,rmsej:h4:r12:p80:s1,rmse:h4:r12:p80:s2,rmsej:h4:r12:p80:s2"
SMALL="$SMALL,rmse:h4:r12,rmsej:h4:r12,lsj1:r12:p80,rmsej:h4:w20"
retry logs/canonical_small_random.log glp --output-root outputs/glp/canonical_small --model-size small \
  --ladder-starts 4 --n-workers 15 --ladder-rungs "$SMALL"
MEDIUM="rmse:h4:r12:p80,rmsej:h4:r12:p80,rmse:h4:r40:p80,rmsej:h4:r40:p80,lsj1:r12:p80"
retry logs/canonical_medium_random.log glp --output-root outputs/glp/canonical_medium --model-size medium \
  --ladder-starts 4 --n-workers 12 --ladder-rungs "$MEDIUM"

# 3. Mixed-frequency VAR (10 workers instead of 12).
mf() {
  .venv/bin/python -u scripts/run_mf_ladder.py --output-dir outputs/mf_ladder \
    --criteria "ml,rmse:h4:w12,rmsej:h4:w60" --fit-nsim 4000 --keep-draws 50 \
    --n-starts 4 --maxfev 400 --n-workers 10 \
  && .venv/bin/python scripts/compare_mf_ladder.py --run-dir outputs/mf_ladder
}
retry logs/mf_ladder.log mf
