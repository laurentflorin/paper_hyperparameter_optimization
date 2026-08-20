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
    return gm._full_bounds(ctx)


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
    lo, hi = gm._full_bounds(ctx)
    natural = np.sqrt(lo * hi)  # geometric midpoint, guaranteed strictly in-bounds
    recovered = gm.to_natural(gm.to_transformed(natural, ctx), ctx)
    np.testing.assert_allclose(recovered, natural, rtol=1e-6)


def test_mode_objective_is_consistent_between_workhorses():
    # logMLVAR_formin and logMLVAR_formcmc must agree on the log posterior at the mode.
    ctx = gm.prepare_glp_context(_synthetic_data(seed=1), lags=5)
    mode = gm.glp_find_mode(ctx)
    mode_vec = gm.hyper_to_natural_vector(mode, ctx)
    from_formcmc = gm.glp_logposterior(ctx, mode_vec)
    assert mode["log_posterior"] == pytest.approx(from_formcmc, rel=1e-6, abs=1e-6)
    lo, hi = _bounds(ctx)
    assert np.all(mode_vec > lo) and np.all(mode_vec < hi)


def test_forecast_shapes():
    ctx = gm.prepare_glp_context(_synthetic_data(), lags=4)
    vec = gm.hyper_to_natural_vector(
        {"lambda": 0.2, "theta": 1.0, "miu": 1.0, "psi": np.ravel(ctx.SS).tolist()}, ctx
    )
    beta, sigma = gm.glp_mode_estimate(ctx, vec)
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
    assert {"lambda", "theta", "miu"} <= set(best)
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


def _rmse_setup(seed: int, *, lags: int = 4, H: int = 4, n_eval: int = 2):
    y = _synthetic_data(seed=seed)
    codes = ["GDP", "DEFL", "FFR"]
    prior_kwargs = {"hyperpriors": gm.GLP_HYPERPRIORS}
    var_indices = gm._resolve_var_indices(codes, ["GDP"])
    ks = gm._rmse_eval_origins(
        y.shape[0], H, n_eval=n_eval, lags=lags, random=False, min_t=None, random_seed=None
    )
    origins = gm._build_rmse_origins(y, lags, ks, H, prior_kwargs)
    ctx_ref = gm.prepare_glp_context(y, lags, **prior_kwargs)
    params = {
        "lam": 0.2,
        "theta": 1.0,
        "miu": 1.0,
        **{f"psi_{i}": float(v) for i, v in enumerate(np.ravel(ctx_ref.SS))},
    }
    return origins, ctx_ref, var_indices, params


def test_rmse_objective_single_draw_matches_posterior_mode():
    # With n_obj_draws <= 1 the objective must reproduce the deterministic
    # posterior-mode RMSE exactly (the pre-change behaviour).
    H, h_eval = 4, 4
    origins, ctx_ref, var_indices, params = _rmse_setup(seed=5, H=H)

    vec = gm._params_to_natural(params, ctx_ref)
    horizons = list(range(1, H + 1))
    squared = []
    for ctx, actual in origins:
        betahat, _ = gm.glp_mode_estimate(ctx, vec)
        forecast = gm.point_forecast(ctx.y, betahat, horizons)
        for vi in var_indices:
            squared.append(float(forecast[h_eval - 1, vi] - actual[h_eval - 1, vi]) ** 2)
    reference_rmse = float(np.sqrt(np.mean(squared)))

    mode_objective = gm._rmse_objective(origins, var_indices, H, h_eval, ctx_ref, n_obj_draws=1)
    assert mode_objective(**params) == pytest.approx(reference_rmse, rel=1e-9, abs=1e-9)


def test_rmse_objective_predictive_mean_is_finite_and_deterministic():
    # With n_obj_draws > 1 the objective averages over posterior beta draws; it
    # must return a finite score and be deterministic across repeated evaluations
    # so the Mango surrogate is not fed noise.
    H, h_eval = 4, 4
    origins, ctx_ref, var_indices, params = _rmse_setup(seed=6, H=H)

    draws_objective = gm._rmse_objective(origins, var_indices, H, h_eval, ctx_ref, n_obj_draws=8, seed_base=123)
    first = draws_objective(**params)
    assert np.isfinite(first) and 0.0 < first < 1.0e10
    assert draws_objective(**params) == first


def test_rmse_objective_draws_do_not_disturb_global_rng():
    # The seeded draw block saves/restores the global RNG so an intervening
    # objective evaluation does not change the caller's random stream.
    H, h_eval = 4, 4
    origins, ctx_ref, var_indices, params = _rmse_setup(seed=8, H=H)
    draws_objective = gm._rmse_objective(origins, var_indices, H, h_eval, ctx_ref, n_obj_draws=4, seed_base=1)

    np.random.seed(2024)
    expected = np.random.random(5)
    np.random.seed(2024)
    draws_objective(**params)
    after = np.random.random(5)
    np.testing.assert_array_equal(expected, after)


def test_mango_rmse_predictive_mean_returns_in_bounds():
    y = _synthetic_data(seed=7)
    codes = ["GDP", "DEFL", "FFR"]
    best = gm.update_hyperparameters_mango_rmse(
        y, lags=4, model_codes=codes, var_of_interest=["GDP"], H=4, h_eval=4, n_eval=2,
        n_obj_draws=4, init_points=2, n_iter=2,
    )
    for key, (lower, upper) in gm.GLP_PARAM_SPACE_BOUNDS.items():
        assert lower <= best[key] <= upper


def test_rmse_eval_origins_rolling_and_random():
    rolling = gm._rmse_eval_origins(100, 4, 3, lags=4, random=False, min_t=40, random_seed=None)
    assert rolling == [0, 1, 2]
    random = gm._rmse_eval_origins(100, 4, 3, lags=4, random=True, min_t=40, random_seed=7)
    assert len(random) == 3 and len(set(random)) == 3
    assert all(0 <= k <= (100 - 4 - 40) for k in random)


def test_rmse_eval_origins_raises_when_infeasible():
    with pytest.raises(ValueError):
        gm._rmse_eval_origins(30, H=8, n_eval=1, random=False, min_t=40, random_seed=None)


def test_resolve_var_indices():
    assert gm._resolve_var_indices(["GDP", "DEFL", "FFR"], ["FFR", "GDP"]) == [2, 0]
    with pytest.raises(ValueError):
        gm._resolve_var_indices(["GDP", "DEFL"], ["CPI"])
