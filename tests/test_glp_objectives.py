import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from glp_hyperparameter_optimization import glp_model as G
from glp_hyperparameter_optimization import objectives as O


@pytest.fixture(scope="module")
def sample():
    rng = np.random.default_rng(3)
    T, n = 90, 3
    y = np.zeros((T, n))
    y[0] = [100.0, 50.0, 5.0]
    for t in range(1, T):
        y[t] = y[t - 1] + np.array([0.5, 0.3, 0.0]) + rng.normal(0, [0.8, 0.3, 0.4])
    ctx = G.prepare_glp_context(y, 2, hyperpriors=0)
    mode = G.glp_find_mode(G.prepare_glp_context(y, 2))
    return y, ctx, G.hyper_to_natural_vector(mode, ctx)


def test_fast_log_ml_and_mode_match_covbayesvar(sample):
    y, ctx, vec = sample
    fast = O.FastGLP.from_context(y, ctx)
    post = fast.posterior(vec, y.shape[0])
    assert post.log_ml == pytest.approx(G.glp_logposterior(ctx, vec), abs=1e-5)
    betahat, _ = G.glp_mode_estimate(ctx, vec)
    assert np.allclose(post.betahat, betahat, atol=1e-7)
    # A prefix of the sample equals re-estimating covbayesvar on y[:cut].
    cut = y.shape[0] - 6
    sub = G.prepare_glp_context(y[:cut], 2, hyperpriors=0)
    assert fast.log_ml(vec, cut) == pytest.approx(G.glp_logposterior(sub, vec), abs=1e-5)


def test_unbounded_log_posterior_matches_inside_box(sample):
    y, _, vec = sample
    ctx1 = G.prepare_glp_context(y, 2)
    assert O.glp_log_posterior_unbounded(y, ctx1, vec) == pytest.approx(G.glp_logposterior(ctx1, vec), abs=1e-5)


def test_draw_moments_match_analytic_posterior(sample):
    y, ctx, vec = sample
    fast = O.FastGLP.from_context(y, ctx)
    post = fast.posterior(vec, y.shape[0])
    noise = O.DrawNoise.generate(20000, fast.k, fast.n, post.dof, 0)
    beta, sigma = O.draw_posterior(post, noise)
    expected_sigma = post.s_matrix / (post.dof - fast.n - 1)
    assert np.allclose(sigma.mean(0), expected_sigma, rtol=0.03, atol=1e-3)
    assert np.allclose(beta.mean(0), post.betahat, atol=0.05 * np.abs(post.betahat).max())


def test_one_step_mixture_converges_to_exact_ml_ratio(sample):
    y, ctx, vec = sample
    fast = O.FastGLP.from_context(y, ctx)
    T = y.shape[0]
    exact = fast.log_ml(vec, T) - fast.log_ml(vec, T - 1)
    post = fast.posterior(vec, T - 1)
    beta, sigma = O.draw_posterior(post, O.DrawNoise.generate(8000, fast.k, fast.n, post.dof, 1))
    mean, cov = O.predictive_moments(y[: T - 1], beta, sigma, fast.lags, 1)
    assert O.mixture_log_density(y[T - 1], mean, cov) == pytest.approx(exact, abs=0.03)


def test_ladder_losses_are_deterministic_and_finite(sample):
    y, ctx, vec = sample
    fast = O.FastGLP.from_context(y, ctx)
    for objective in ("logscore_joint_1", "logscore_joint", "logscore_marginal", "rmse", "rmse_joint"):
        loss = O.build_loss(fast, O.LadderSpec(objective, horizon=2, window=5, n_draws=50, seed=4))
        first, second = loss(vec), loss(vec)
        assert np.isfinite(first) and first == second


def test_logscore_joint_1_telescopes(sample):
    y, ctx, vec = sample
    fast = O.FastGLP.from_context(y, ctx)
    T = y.shape[0]
    loss = O.build_loss(fast, O.LadderSpec("logscore_joint_1", window=4))
    direct = -sum(fast.log_ml(vec, c + 1) - fast.log_ml(vec, c) for c in range(T - 4, T))
    assert loss(vec) == pytest.approx(direct, abs=1e-8)


def test_select_ladder_hyperparameters_improves_on_mdd_mode(sample):
    y, _, _ = sample
    spec = O.LadderSpec("rmse", horizon=2, window=5, n_draws=30, seed=0)
    result = O.select_ladder_hyperparameters(y, 2, spec, n_starts=1, maxfev=60)
    assert result["objective_value"] <= result["objective_at_mdd_mode"] + 1e-12
    assert O.select_ladder_hyperparameters(y, 2, O.LadderSpec("mdd"))["objective_at_mdd_mode"] is None


def test_fast_forecast_draws_shape_and_seed(sample):
    y, ctx, vec = sample
    a = O.fast_forecast_draws(y, ctx, vec, max_horizon=4, n_draws=50, seed=9)
    b = O.fast_forecast_draws(y, ctx, vec, max_horizon=4, n_draws=50, seed=9)
    assert a.shape == (50, 4, 3) and np.array_equal(a, b)


def test_evaluation_cuts_window_and_all():
    cuts = O.evaluation_cuts(100, 5, 4, 3)
    assert cuts == [96, 95, 94]
    assert O.evaluation_cuts(100, 5, 4, None)[-1] == max(20, 9) + 5
    with pytest.raises(ValueError):
        O.evaluation_cuts(20, 5, 8, 3)


def test_evaluation_cuts_random_subset():
    last12 = O.evaluation_cuts(200, 5, 4, 12)
    pool = O.evaluation_cuts(200, 5, 4, 80)
    cuts = O.evaluation_cuts(200, 5, 4, 12, random_origins=True, pool=80)
    assert len(cuts) == 12 and set(cuts) <= set(pool) and cuts == sorted(cuts, reverse=True)
    assert cuts != last12 and max(cuts) - min(cuts) > 12
    assert cuts == O.evaluation_cuts(200, 5, 4, 12, random_origins=True, pool=80)
    assert cuts != O.evaluation_cuts(200, 5, 4, 12, random_origins=True, pool=80, subset_seed=1)
    # One more quarter of data changes the subset by at most one origin in, one out.
    later = O.evaluation_cuts(201, 5, 4, 12, random_origins=True, pool=80)
    assert len(set(cuts) & set(later)) >= 10
    # Without a pool the draw ranges over the whole history.
    full = O.evaluation_cuts(200, 5, 4, 12, random_origins=True)
    assert min(full) < min(pool)
    with pytest.raises(ValueError):
        O.evaluation_cuts(200, 5, 4, None, random_origins=True)


def test_logscore_joint_1_on_random_origins_sums_one_step_ratios(sample):
    y, ctx, vec = sample
    fast = O.FastGLP.from_context(y, ctx)
    spec = O.LadderSpec("logscore_joint_1", window=6, random_origins=True, pool=30, subset_seed=2)
    cuts = spec.cuts(y.shape[0], fast.lags, 1)
    assert len(cuts) == 6 and max(cuts) - min(cuts) + 1 > 6
    expected = sum(fast.log_ml(vec, c + 1) - fast.log_ml(vec, c) for c in cuts)
    assert O.build_loss(fast, spec)(vec) == pytest.approx(-expected, abs=1e-8)
    assert spec.label() == "lsj1_r6p30s2"
    assert O.LadderSpec("rmse", 4, 12, random_origins=True).label() == "rmse_h4_r12"


def test_rmse_joint_weights_every_variable_by_its_ar1_scale(sample):
    y, ctx, vec = sample
    fast = O.FastGLP.from_context(y, ctx)
    spec = dict(horizon=2, window=5, n_draws=40, seed=1)
    joint = O.build_loss(fast, O.LadderSpec("rmse_joint", **spec))(vec)
    per_var = [O.build_loss(fast, O.LadderSpec("rmse", target_indices=(i,), **spec))(vec) for i in range(fast.n)]
    expected = np.sqrt(np.mean([r**2 / s for r, s in zip(per_var, fast.ss)]))
    assert joint == pytest.approx(expected, rel=1e-10)
    assert O.LadderSpec("rmse_joint", 4, 12).label() == "rmsej_h4_w12"
