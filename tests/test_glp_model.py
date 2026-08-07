import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

pytest.importorskip("covbayesvar")

from glp_hyperparameter_optimization import glp_model as gm


def _synthetic_data(T: int = 90, n: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(size=(T, n)), axis=0) + 100.0


def _bounds(ctx) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([ctx.MIN["lambda"], ctx.MIN["theta"], ctx.MIN["miu"]])
    hi = np.array([ctx.MAX["lambda"], ctx.MAX["theta"], ctx.MAX["miu"]])
    return lo, hi


def test_context_shapes():
    y = _synthetic_data()
    ctx = gm.prepare_glp_context(y, lags=4)
    assert ctx.n == 3
    assert ctx.k == 3 * 4 + 1
    assert ctx.T == y.shape[0] - 4
    assert ctx.b.shape == (ctx.k, ctx.n)
    assert ctx.SS.shape == (ctx.n, 1)


def test_transform_round_trip():
    ctx = gm.prepare_glp_context(_synthetic_data(), lags=4)
    natural = [0.35, 2.5, 0.8]
    recovered = gm.to_natural(gm.to_transformed(natural, ctx), ctx)
    np.testing.assert_allclose(recovered, natural, rtol=1e-6)


def test_mode_objective_is_consistent_between_workhorses():
    # logMLVAR_formin and logMLVAR_formcmc must agree on the log posterior at the mode.
    ctx = gm.prepare_glp_context(_synthetic_data(seed=1), lags=5)
    mode = gm.glp_find_mode(ctx)
    from_formcmc = gm.glp_logposterior(ctx, mode["lambda"], mode["theta"], mode["miu"])
    assert mode["log_posterior"] == pytest.approx(from_formcmc, rel=1e-6, abs=1e-6)
    lo, hi = _bounds(ctx)
    natural = np.array([mode["lambda"], mode["theta"], mode["miu"]])
    assert np.all(natural > lo) and np.all(natural < hi)


def test_forecast_shapes():
    ctx = gm.prepare_glp_context(_synthetic_data(), lags=4)
    beta, sigma = gm.glp_mode_estimate(ctx, 0.2, 1.0, 1.0)
    assert beta.shape == (ctx.k, ctx.n)
    assert sigma.shape == (ctx.n, ctx.n)
    point = gm.point_forecast(ctx.y, beta, [1, 2, 4, 8])
    assert point.shape == (4, ctx.n)
    rng = np.random.default_rng(0)
    path = gm.simulate_forecast_path(ctx.y, beta, sigma, ctx.lags, 8, rng)
    assert path.shape == (8, ctx.n)


def test_mango_mdd_returns_in_bounds():
    y = _synthetic_data(seed=2)
    best = gm.update_hyperparameters_mango(y, lags=4, init_points=2, n_iter=2, njobs=1)
    assert set(best) == {"lambda", "theta", "miu"}
    for key, (lower, upper) in gm.GLP_PARAM_SPACE_BOUNDS.items():
        assert lower <= best[key] <= upper


def test_mango_rmse_variants_return_in_bounds():
    y = _synthetic_data(seed=3)
    codes = ["GDP", "DEFL", "FFR"]
    rolling = gm.update_hyperparameters_mango_rmse(
        y, lags=4, model_codes=codes, var_of_interest=["GDP"], H=4, h_eval=4, n_eval=2, init_points=2, n_iter=2
    )
    random = gm.update_hyperparameters_mango_rmse_random(
        y, lags=4, model_codes=codes, var_of_interest=["GDP"], H=4, h_eval=2, n_eval=2, min_t=40, random_seed=7,
        init_points=2, n_iter=2,
    )
    for best in (rolling, random):
        for key, (lower, upper) in gm.GLP_PARAM_SPACE_BOUNDS.items():
            assert lower <= best[key] <= upper


def test_rmse_eval_origins_rolling_and_random():
    rolling = gm._rmse_eval_origins(100, H=4, n_eval=3, random=False, min_t=40, random_seed=None)
    assert rolling == [0, 1, 2]
    random = gm._rmse_eval_origins(100, H=4, n_eval=3, random=True, min_t=40, random_seed=7)
    assert len(random) == 3 and len(set(random)) == 3
    assert all(0 <= k <= (100 - 4 - 40) for k in random)


def test_rmse_eval_origins_raises_when_infeasible():
    with pytest.raises(ValueError):
        gm._rmse_eval_origins(30, H=8, n_eval=1, random=False, min_t=40, random_seed=None)


def test_resolve_var_indices():
    assert gm._resolve_var_indices(["GDP", "DEFL", "FFR"], ["FFR", "GDP"]) == [2, 0]
    with pytest.raises(ValueError):
        gm._resolve_var_indices(["GDP", "DEFL"], ["CPI"])
