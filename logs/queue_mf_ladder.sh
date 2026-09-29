#!/usr/bin/env bash
# Mixed-frequency (Schorfheide-Song) prior selection: published values, marginal
# likelihood, GDP RMSE over 12 quarters, joint RMSE over 60 quarters, at all 151
# monthly origins. Starts once the random-origin queue has finished; resumes from
# per-origin checkpoints and is retried up to 3 times.
cd /run/media/system/Data/Repositories/paper_hyperparameter_optimization
WAIT_PID=$(cat logs/queue_random_origins.pid)
while grep -q queue_random_origins "/proc/$WAIT_PID/cmdline" 2>/dev/null; do sleep 300; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
LOG=logs/mf_ladder.log
for attempt in 1 2 3; do
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] attempt $attempt" >> "$LOG"
  .venv/bin/python -u scripts/run_mf_ladder.py --output-dir outputs/mf_ladder \
    --criteria "ml,rmse:h4:w12,rmsej:h4:w60" --fit-nsim 4000 --keep-draws 50 \
    --n-starts 4 --maxfev 400 --n-workers 12 >> "$LOG" 2>&1 \
    && .venv/bin/python scripts/compare_mf_ladder.py --run-dir outputs/mf_ladder >> "$LOG" 2>&1 \
    && exit 0
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] attempt $attempt failed; resuming from checkpoints" >> "$LOG"
  sleep 60
done
exit 1
