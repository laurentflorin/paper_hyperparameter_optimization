"""Tests for model-independent forecast inference (DM, HAC, bootstrap, Holm)."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common_hpo.inference import (
    bootstrap_ci,
    diebold_mariano,
    hac_long_run_variance,
    holm_adjust,
    moving_block_bootstrap_ci,
    stationary_bootstrap_ci,
)


# --------------------------------------------------------------------------- #
# HAC variance
# --------------------------------------------------------------------------- #
def test_hac_lag_zero_matches_population_variance():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # lag=0 -> gamma_0 = mean of squared deviations (population, ddof=0).
    expected = float(np.mean((x - x.mean()) ** 2))
    assert hac_long_run_variance(x, lag=0) == pytest.approx(expected)


def test_hac_uniform_lag_one_hand_computed():
    x = np.array([1.0, 3.0, 2.0, 4.0])
    c = x - x.mean()  # [-1.5, 0.5, -0.5, 1.5]
    n = 4
    gamma0 = float(np.dot(c, c) / n)
    gamma1 = float(np.dot(c[1:], c[:-1]) / n)
    expected = gamma0 + 2.0 * gamma1  # uniform kernel, lag 1
    assert hac_long_run_variance(x, lag=1, kernel="uniform") == pytest.approx(expected)


def test_hac_overlapping_horizon_uses_lag_h_minus_one():
    # For h-step forecasts the DM truncation lag is h-1; confirm the test wires
    # the lag through (bartlett weights shrink the lag-1 term).
    x = np.array([0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5])
    v0 = hac_long_run_variance(x, lag=0)
    v_h2 = hac_long_run_variance(x, lag=1, kernel="bartlett")
    # Strong negative autocovariance at lag 1 reduces the long-run variance.
    assert v_h2 < v0


# --------------------------------------------------------------------------- #
# Diebold-Mariano
# --------------------------------------------------------------------------- #
def test_dm_zero_variance_is_refused():
    # Identical losses -> zero differential variance -> no valid p-value.
    loss_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    result = diebold_mariano(loss_a, loss_a.copy(), horizon=1)
    assert result.valid is False
    assert result.p_value is None
    assert "zero" in result.reason.lower()


def test_dm_positive_differential_is_significant():
    rng = np.random.default_rng(0)
    base = rng.normal(size=40) ** 2
    # A is worse on average, with a genuinely varying differential.
    noise = rng.normal(scale=0.1, size=40)
    loss_a = base + 1.0 + noise
    loss_b = base
    result = diebold_mariano(loss_a, loss_b, horizon=1)
    assert result.valid is True
    assert result.mean_loss_differential == pytest.approx(float(np.mean(loss_a - loss_b)))
    assert result.dm_statistic > 0
    assert result.p_value < 0.05
    assert result.small_sample_corrected is True


def test_dm_small_sample_is_refused():
    loss_a = np.array([1.0, 2.0, 3.0])
    loss_b = np.array([1.1, 1.9, 3.2])
    result = diebold_mariano(loss_a, loss_b, horizon=1, min_observations=8)
    assert result.valid is False
    assert "too small" in result.reason.lower()


def test_dm_unpaired_lengths_refused():
    result = diebold_mariano(np.ones(10), np.ones(9), horizon=1)
    assert result.valid is False
    assert "unpaired" in result.reason.lower()


def test_dm_overlapping_horizon_sets_hac_lag():
    rng = np.random.default_rng(1)
    loss_a = rng.normal(size=60) ** 2 + 0.3
    loss_b = rng.normal(size=60) ** 2
    result = diebold_mariano(loss_a, loss_b, horizon=4)
    assert result.hac_lag == 3  # h - 1
    assert result.valid is True


def test_dm_normal_reference_without_correction():
    rng = np.random.default_rng(2)
    base = rng.normal(size=50) ** 2
    noise = rng.normal(scale=0.1, size=50)
    result = diebold_mariano(base + 0.5 + noise, base, horizon=1, small_sample_correction=False)
    assert result.small_sample_corrected is False
    assert result.valid is True
    assert 0.0 <= result.p_value <= 1.0


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def test_moving_block_bootstrap_is_reproducible():
    rng = np.random.default_rng(5)
    x = rng.normal(size=50)
    a = moving_block_bootstrap_ci(x, block_length=5, n_boot=200, seed=123)
    b = moving_block_bootstrap_ci(x, block_length=5, n_boot=200, seed=123)
    assert a.ci_lower == b.ci_lower
    assert a.ci_upper == b.ci_upper
    assert a.statistic == pytest.approx(float(x.mean()))


def test_moving_block_bootstrap_different_seed_differs():
    rng = np.random.default_rng(6)
    x = rng.normal(size=50)
    a = moving_block_bootstrap_ci(x, block_length=5, n_boot=200, seed=1)
    b = moving_block_bootstrap_ci(x, block_length=5, n_boot=200, seed=2)
    assert (a.ci_lower, a.ci_upper) != (b.ci_lower, b.ci_upper)


def test_stationary_bootstrap_is_reproducible_and_brackets_mean():
    rng = np.random.default_rng(7)
    x = rng.normal(loc=1.0, size=60)
    a = stationary_bootstrap_ci(x, mean_block_length=4, n_boot=300, seed=42)
    b = stationary_bootstrap_ci(x, mean_block_length=4, n_boot=300, seed=42)
    assert a.ci_lower == b.ci_lower and a.ci_upper == b.ci_upper
    assert a.ci_lower <= a.statistic <= a.ci_upper


def test_bootstrap_ci_dispatch():
    x = np.arange(20.0)
    mb = bootstrap_ci(x, method="moving_block", block_length=4, n_boot=100, seed=0)
    st = bootstrap_ci(x, method="stationary", block_length=4, n_boot=100, seed=0)
    assert mb.method == "moving_block"
    assert st.method == "stationary"
    with pytest.raises(ValueError):
        bootstrap_ci(x, method="bogus", block_length=4)


# --------------------------------------------------------------------------- #
# Holm
# --------------------------------------------------------------------------- #
def test_holm_adjustment_hand_computed():
    # Sorted p: 0.01, 0.02, 0.03, 0.04 with m=4.
    # Holm: 4*0.01=0.04, 3*0.02=0.06, 2*0.03=0.06->max(0.06)=0.06, 1*0.04=0.04->max=0.06.
    p = [0.04, 0.03, 0.02, 0.01]
    adjusted = holm_adjust(p)
    # Map back to original order.
    assert adjusted[3] == pytest.approx(0.04)  # smallest p
    assert adjusted[2] == pytest.approx(0.06)
    assert adjusted[1] == pytest.approx(0.06)
    assert adjusted[0] == pytest.approx(0.06)


def test_holm_is_monotone_and_capped_at_one():
    p = [0.5, 0.6, 0.7]
    adjusted = holm_adjust(p)
    assert all(v <= 1.0 for v in adjusted)
    # Non-decreasing in the sorted order.
    assert adjusted[0] <= adjusted[1] <= adjusted[2]


def test_holm_passes_through_invalid_entries():
    p = [0.01, None, float("nan"), 0.02]
    adjusted = holm_adjust(p)
    assert np.isnan(adjusted[1])
    assert np.isnan(adjusted[2])
    # Family size excludes the invalid entries (m=2): 2*0.01=0.02, then 0.02.
    assert adjusted[0] == pytest.approx(0.02)
    assert adjusted[3] == pytest.approx(0.02)


def test_holm_rejects_out_of_range():
    with pytest.raises(ValueError):
        holm_adjust([0.5, 1.5])
