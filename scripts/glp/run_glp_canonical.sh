#!/usr/bin/env bash
# Harmonized GLP design: one set of settings shared by every model size and
# strategy, so differences across runs reflect the strategy, not the run.
#
#   * predictive densities: 20,000 retained draws (5,000 burn-in for the paper
#     Metropolis chain); fixed-hyperparameter strategies use the same 20,000
#   * hyperparameters re-selected at every origin, optimizers seeded
#   * Mango: 10 random + 20 BO iterations; mango_mdd polished with Nelder-Mead
#   * RMSE strategies: GDP, horizons 1/2/4/8, 12 evaluation origins,
#     predictive-mean objective with 500 beta draws
#   * ladder: marginal likelihood -> targeted RMSE, one departure at a time
#
# Usage: scripts/glp/run_glp_canonical.sh [small|medium|large] [stages]
#   stages defaults to every stage; e.g. "ladder,compare" reruns the ladder only.
# Outputs: outputs/glp/canonical_<size>/{paper,paper_plugin,mango_*,ladder/*,comparison}
set -euo pipefail

SIZE="${1:-small}"
STAGES="${2:-paper,paper_plugin,mango_mdd,mango_rmse,mango_rmse_random,ladder,compare}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
N_WORKERS="${N_WORKERS:-15}"

# One BLAS thread per worker process (parallelism is across forecast origins).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

"$PYTHON" -u "$ROOT/scripts/glp/run_glp_all.py" \
	--stages "$STAGES" \
	--output-root "outputs/glp/canonical_${SIZE}" \
	--model-size "$SIZE" \
	--variables GDP \
	--mcmc-draws 20000 \
	--mcmc-discard 5000 \
	--seed-base 2015 \
	--optimization-seed 123 \
	--optimization-init-points 10 \
	--optimization-iterations 20 \
	--local-polish \
	--per-origin-selection \
	--optimization-eval-horizons-quarters 1,2,4,8 \
	--optimization-n-eval 12 \
	--optimization-n-obj-draws 500 \
	--optimization-random-seed 123 \
	--ladder-rungs "mdd,lsj1:w12,lsj1:wall,lsj:h4:w12,lsm:h1:w12,lsm:h4:w12,rmse:h1:w12,rmse:h4:w12" \
	--ladder-draws 200 \
	--ladder-starts 4 \
	--n-workers "$N_WORKERS"
