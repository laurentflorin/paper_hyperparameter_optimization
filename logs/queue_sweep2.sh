#!/usr/bin/env bash
cd /run/media/system/Data/Repositories/paper_hyperparameter_optimization
while kill -0 47414 2>/dev/null; do sleep 30; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -u scripts/glp/run_glp_all.py --stages ladder --output-root outputs/glp/canonical_small --model-size small --variables GDP --mcmc-draws 20000 --mcmc-discard 5000 --seed-base 2015 --optimization-seed 123 --per-origin-selection --ladder-rungs "rmse:h4:w20,rmse:h4:w60,rmse:h4:w80,rmsej:h4:w12,rmsej:h4:w40" --ladder-draws 200 --ladder-starts 4 --n-workers 15
