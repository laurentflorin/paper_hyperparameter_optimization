#!/usr/bin/env bash
cd /run/media/system/Data/Repositories/paper_hyperparameter_optimization
while kill -0 52826 2>/dev/null; do sleep 30; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec .venv/bin/python -u scripts/glp/run_glp_all.py --stages ladder --output-root outputs/glp/canonical_small --model-size small --variables GDP --mcmc-draws 20000 --mcmc-discard 5000 --seed-base 2015 --optimization-seed 123 --per-origin-selection --ladder-rungs "rmsej:h1:w40,rmsej:h1:w60,rmsej:h1:w80,rmsej:h1:w12,rmse:h1:w20,rmse:h1:w40,rmse:h1:w60,rmse:h1:w80,rmsej:h2:w40,rmsej:h2:w60,rmsej:h2:w80,rmsej:h2:w12,rmse:h2:w12,rmse:h2:w20,rmse:h2:w40,rmse:h2:w60,rmse:h2:w80,rmsej:h8:w40,rmsej:h8:w60,rmsej:h8:w80,rmsej:h8:w12,rmse:h8:w12,rmse:h8:w20,rmse:h8:w40,rmse:h8:w60,rmse:h8:w80,rmse:h1:wall,rmse:h2:wall,rmse:h8:wall" --ladder-draws 200 --ladder-starts 4 --n-workers 15
