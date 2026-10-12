import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from glp_hyperparameter_optimization import scoring as S


def test_adjusted_actual_removes_vintage_level_shift_for_log_series_only():
    adj = S.adjusted_actual([120.0, 5.0], [118.0, 4.0], [100.0, 4.5], [True, False])
    assert np.allclose(adj, [102.0, 5.0])  # log: 120 - 118 + 100; rate: unchanged


def test_crps_sample_matches_gaussian_closed_form():
    rng = np.random.default_rng(0)
    draws = rng.normal(1.0, 2.0, size=(200000, 1))
    assert S.crps_sample(draws, np.array([0.3]))[0] == pytest.approx(S.crps_gaussian(1.0, 2.0, 0.3), rel=0.01)


def test_diebold_mariano_sign_and_degenerate_cases():
    rng = np.random.default_rng(1)
    base = rng.normal(size=200) ** 2
    better = base * 0.5
    stat, p = S.diebold_mariano(better, base, 1)
    assert stat < 0 and p < 0.01
    assert np.isnan(S.diebold_mariano(base, base, 1)[0])
    assert np.isnan(S.diebold_mariano(base[:5], base[:5] * 2, 1)[0])


def test_holm_adjust_is_monotone_and_capped():
    adj = S.holm_adjust([0.01, 0.04, 0.03, np.nan])
    assert np.allclose(adj[:3], [0.03, 0.06, 0.06]) and np.isnan(adj[3])


def test_mcs_excludes_clearly_worse_model():
    rng = np.random.default_rng(2)
    good = rng.normal(1.0, 0.2, 300)
    losses = pd.DataFrame({"a": good, "b": good + rng.normal(0, 0.05, 300), "bad": good + 1.0})
    mcs = S.model_confidence_set(losses, alpha=0.1, n_boot=500)
    assert not mcs.loc["bad", "in_mcs"]
    assert mcs.loc[["a", "b"], "in_mcs"].all()


def test_pooled_diebold_mariano_detects_uniform_gain_and_is_antisymmetric():
    rng = np.random.default_rng(1)
    T, H = 80, 8
    base = rng.chisquare(1, (T, H)) * np.arange(1, H + 1)  # losses grow with the horizon
    better = base * np.exp(rng.normal(-0.2, 0.3, (T, H)))
    stat, p = S.pooled_diebold_mariano(better, base)
    assert stat < 0 and p < 0.05
    # Swapping the models flips the sign; with the baseline's scaling the
    # statistic is not exactly symmetric, but the conclusion is.
    stat_rev, p_rev = S.pooled_diebold_mariano(base, better)
    assert stat_rev > 0 and p_rev < 0.05


def test_fixed_smoothing_dm_has_reasonable_size_and_detects_a_gain():
    rng = np.random.default_rng(3)
    for method in ("ewc", "wpe"):
        p = [S.fixed_smoothing_dm(rng.normal(size=80), rng.normal(size=80), method)[1] for _ in range(400)]
        assert 0.02 <= np.mean(np.array(p) < 0.05) <= 0.09
        stat, pval = S.fixed_smoothing_dm(rng.normal(size=80) - 1.0, rng.normal(size=80), method)
        assert stat < 0 and pval < 0.01
    assert np.isnan(S.fixed_smoothing_dm(np.ones(80), np.ones(80))[1])


def test_benchmark_tests_find_the_better_criterion_and_adjust_upwards():
    rng = np.random.default_rng(5)
    T = 80
    diff = rng.normal(size=(T, 6))
    diff[:, 0] += 1.0  # criterion 0 beats the benchmark
    diff[:, 5] = 0.0  # criterion 5 is the benchmark itself
    res = S.benchmark_tests(diff, block=4, n_boot=2000, seed=0)
    assert res["p_spa"] < 0.01
    assert res["p_stepm"][0] < 0.01 and np.all(res["p_stepm"][1:5] > 0.05)
    assert np.all(res["p_stepm"] >= res["p_single"] - 1e-12)
    assert res["t"][5] == 0.0 and res["p_stepm"][5] == 1.0
    order = np.argsort(-res["t"][:5])
    assert np.all(np.diff(res["p_stepm"][order]) >= -1e-12)  # adjusted p-values follow the ranking
    null = S.benchmark_tests(rng.normal(size=(T, 6)) - 0.2, block=4, n_boot=2000, seed=0)
    assert null["p_spa"] > 0.1


def test_multi_horizon_tests_separate_average_and_uniform_gains():
    rng = np.random.default_rng(7)
    T, H = 80, 4
    everywhere = rng.normal(size=(T, H)) + 0.8
    p_avg, p_unif = S.multi_horizon_tests(everywhere, block=4, n_boot=2000)
    assert p_avg < 0.01 and p_unif < 0.01
    one_bad = everywhere.copy()
    one_bad[:, 0] -= 1.6  # worse than the benchmark at the first horizon
    p_avg, p_unif = S.multi_horizon_tests(one_bad, block=4, n_boot=2000)
    assert p_avg < 0.05 and p_unif > 0.5
