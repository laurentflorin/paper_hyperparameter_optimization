#!/usr/bin/env bash
cd /run/media/system/Data/Repositories/paper_hyperparameter_optimization
while kill -0 53288 2>/dev/null; do sleep 30; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
COMMON="--variables GDP --mcmc-draws 20000 --mcmc-discard 5000 --seed-base 2015 --optimization-seed 123 --per-origin-selection --ladder-draws 200 --ladder-starts 4 --n-workers 15"
.venv/bin/python -u scripts/glp/run_glp_all.py --stages ladder --output-root outputs/glp/canonical_small --model-size small $COMMON \
  --ladder-rungs "lsj1:w60,lsm:h1:w60,lsj:h2:w12,lsj:h2:w60,lsm:h2:w12,lsm:h2:w60,lsm:h4:w60,lsj:h8:w12,lsj:h8:w60,lsm:h8:w12,lsm:h8:w60" \
  > logs/canonical_small_pyramid.log 2>&1
.venv/bin/python -u scripts/glp/run_glp_all.py --stages paper,paper_plugin,ladder,compare --output-root outputs/glp/canonical_medium --model-size medium $COMMON \
  --ladder-rungs "mdd,lsj1:w12,lsj1:w60,lsj1:wall,rmsej:h4:w12,rmsej:h4:w60,rmse:h4:w12,rmse:h4:w60,lsm:h4:w12,lsm:h4:w60,lsj:h4:w12,lsj:h4:w60" \
  > logs/canonical_medium_ladder.log 2>&1
