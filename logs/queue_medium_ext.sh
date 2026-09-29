#!/usr/bin/env bash
cd /run/media/system/Data/Repositories/paper_hyperparameter_optimization
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
COMMON="--output-root outputs/glp/canonical_medium --model-size medium --variables GDP --mcmc-draws 20000 --mcmc-discard 5000 --seed-base 2015 --optimization-seed 123 --per-origin-selection --ladder-draws 200 --ladder-starts 4 --n-workers 15"
.venv/bin/python -u scripts/glp/run_glp_all.py --stages ladder,compare $COMMON \
  --ladder-rungs "lsj1:w40,lsj1:w80,rmsej:h4:w40,rmsej:h4:w80,rmse:h4:w40,rmse:h4:w80,lsj:h4:w40,lsj:h4:w80,lsm:h4:w40,lsm:h4:w80" \
  > logs/canonical_medium_windows.log 2>&1
.venv/bin/python -u scripts/glp/run_glp_all.py --stages ladder,compare $COMMON \
  --ladder-rungs "lsm:h1:w12,lsm:h1:w60,lsj:h2:w12,lsj:h2:w60,lsm:h2:w12,lsm:h2:w60,lsj:h8:w12,lsj:h8:w60,lsm:h8:w12,lsm:h8:w60" \
  > logs/canonical_medium_pyramid.log 2>&1
