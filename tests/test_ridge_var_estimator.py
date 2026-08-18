"""Tests for the lag-weighted ridge VAR estimator."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from regularized_var.design import build_lag_design
from regularized_var.estimators import (
    PenaltyConfig,
    UnstableVARError,
    build_penalty_weights,
    fit_ridge_var,
)


def _stable_var_series(T=200, seed=0):
    rng = np.random.default_rng(seed)
    n = 2
    A1 = np.array([[0.5, 0.1], [-0.2, 0.3]])
    c = np.array([0.4, -0.1])
    y = np.zeros((T, n))
    for t in range(1, T):
        y[t] = c + A1 @ y[t - 1] + rng.normal(scale=0.1, size=n)
    return y


def test_lambda_zero_matches_numpy_lstsq():
    y = _stable_var_series()
    X, Y, _ = build_lag_design(y, p=1, include_intercept=True)
    beta_ref, *_ = np.linalg.lstsq(X, Y, rcond=None)

    result = fit_ridge_var(y, p=1, lam=0.0)
    # Reassemble estimator coefficients into the design orientation B (k x n).
    intercept = result.intercept
    A1 = result.lag_coefficients[0]  # [eq, var]
    # Reference: row0 intercept, rows1..n lag1 predictors (var i -> eq j = beta[1+i, j]).
    np.testing.assert_allclose(intercept, beta_ref[0, :], atol=1e-8)
    # A1[j, i] should equal beta_ref[1 + i, j].
    for j in range(2):
        for i in range(2):
            assert A1[j, i] == pytest.approx(beta_ref[1 + i, j], abs=1e-8)


def test_increasing_lambda_reduces_penalized_norm():
    y = _stable_var_series()
    norms = []
    for lam in (0.0, 1.0, 10.0, 100.0):
        result = fit_ridge_var(y, p=2, lam=lam, alpha=1.0, kappa=1.0)
        # Penalized coefficient norm excludes the unpenalized intercept.
        norms.append(float(np.sum(result.lag_coefficients**2)))
    assert norms[0] > norms[1] > norms[2] > norms[3]


def test_lag_decay_ordering_penalizes_higher_lags_more():
    y = _stable_var_series(seed=3)
    # With a strong lag-decay exponent, higher lags are shrunk harder, so the
    # per-lag coefficient norm should be (weakly) decreasing in the lag index.
    result = fit_ridge_var(y, p=3, lam=50.0, alpha=3.0, kappa=1.0)
    lag_norms = [float(np.sum(result.lag_coefficients[l] ** 2)) for l in range(3)]
    assert lag_norms[0] >= lag_norms[1] >= lag_norms[2]


def test_penalty_weights_own_vs_cross():
    weights = build_penalty_weights(
        equation_index=0,
        n_vars=2,
        p=2,
        alpha=2.0,
        kappa=5.0,
        include_intercept=True,
    )
    # Layout: [intercept, lag1 var0, lag1 var1, lag2 var0, lag2 var1].
    assert weights[0] == 0.0  # intercept unpenalized
    assert weights[1] == pytest.approx(1.0**2.0)  # own lag1
    assert weights[2] == pytest.approx(5.0 * 1.0**2.0)  # cross lag1
    assert weights[3] == pytest.approx(2.0**2.0)  # own lag2
    assert weights[4] == pytest.approx(5.0 * 2.0**2.0)  # cross lag2


def test_penalty_weights_second_equation_shifts_own_index():
    weights = build_penalty_weights(
        equation_index=1,
        n_vars=2,
        p=1,
        alpha=0.0,
        kappa=3.0,
        include_intercept=False,
    )
    # lag1 var0 is cross for equation 1; lag1 var1 is own.
    assert weights[0] == pytest.approx(3.0)  # cross
    assert weights[1] == pytest.approx(1.0)  # own


def test_intercept_remains_unpenalized_under_large_lambda():
    y = _stable_var_series(seed=4)
    # Add a large constant so a meaningful intercept is required.
    y = y + 100.0
    result = fit_ridge_var(y, p=1, lam=1e6, alpha=1.0)
    # Lag coefficients are shrunk toward zero but the intercept is free to fit
    # the level (roughly the series mean), so it is far from zero.
    assert np.all(np.abs(result.lag_coefficients) < 1e-2)
    assert np.all(result.intercept > 50.0)


def test_result_fields_and_shapes():
    y = _stable_var_series()
    result = fit_ridge_var(y, p=2, lam=1.0, variable_names=["a", "b"])
    assert result.lag_coefficients.shape == (2, 2, 2)
    assert result.intercept.shape == (2,)
    assert result.lag_order == 2
    assert result.variable_names == ("a", "b")
    assert result.residuals.shape == (y.shape[0] - 2, 2)
    assert result.residual_covariance.shape == (2, 2)
    assert result.companion_matrix.shape == (4, 4)
    assert isinstance(result.max_companion_eigenvalue, float)
    assert isinstance(result.is_stable, bool)
    assert result.design_rank > 0
    assert np.isfinite(result.design_condition_number)


def test_companion_eigenvalue_known_ar1():
    # A scalar AR(1) with coefficient 0.9 has companion eigenvalue 0.9.
    rng = np.random.default_rng(9)
    T = 500
    y = np.zeros((T, 1))
    for t in range(1, T):
        y[t, 0] = 0.9 * y[t - 1, 0] + rng.normal(scale=0.01)
    result = fit_ridge_var(y, p=1, lam=0.0)
    assert result.max_companion_eigenvalue == pytest.approx(0.9, abs=0.02)
    assert result.is_stable


def test_companion_matrix_block_structure_var2():
    y = _stable_var_series()
    result = fit_ridge_var(y, p=2, lam=0.0)
    n = 2
    companion = result.companion_matrix
    # Top-left block equals A_1, next block equals A_2.
    np.testing.assert_allclose(companion[:n, :n], result.lag_coefficients[0])
    np.testing.assert_allclose(companion[:n, n : 2 * n], result.lag_coefficients[1])
    # Sub-diagonal identity.
    np.testing.assert_allclose(companion[n:, :n], np.eye(n))


def test_unstable_var_reported_not_discarded():
    # Construct an explosive AR(1) series.
    T = 100
    y = np.zeros((T, 1))
    y[0, 0] = 1.0
    for t in range(1, T):
        y[t, 0] = 1.1 * y[t - 1, 0]
    result = fit_ridge_var(y, p=1, lam=0.0)
    assert result.max_companion_eigenvalue > 1.0
    assert result.is_stable is False  # reported, not discarded


def test_reject_unstable_policy_raises():
    T = 100
    y = np.zeros((T, 1))
    y[0, 0] = 1.0
    for t in range(1, T):
        y[t, 0] = 1.1 * y[t - 1, 0]
    with pytest.raises(UnstableVARError, match="unstable"):
        fit_ridge_var(y, p=1, lam=0.0, reject_unstable=True)


def test_singular_design_is_handled():
    # Two identical columns make the design rank-deficient; lstsq still returns
    # a finite minimum-norm solution and the rank diagnostic reflects deficiency.
    rng = np.random.default_rng(7)
    base = rng.normal(size=(60, 1))
    y = np.hstack([base, base])  # perfectly collinear variables
    result = fit_ridge_var(y, p=1, lam=0.0)
    assert np.all(np.isfinite(result.lag_coefficients))
    # Design has an intercept + 2 lag columns but the two lag columns are
    # collinear, so the rank is below the column count.
    assert result.design_rank < result.companion_matrix.shape[0] + 1
    assert result.design_condition_number > 1e6


def test_nearly_singular_design_ridge_stabilizes():
    rng = np.random.default_rng(8)
    base = rng.normal(size=(80, 1))
    y = np.hstack([base, base + 1e-9 * rng.normal(size=(80, 1))])
    # lambda > 0 must still produce a finite, stable solve on an ill-conditioned
    # design without forming an explicit inverse.
    result = fit_ridge_var(y, p=1, lam=1.0, alpha=1.0)
    assert np.all(np.isfinite(result.lag_coefficients))
    assert np.all(np.isfinite(result.residual_covariance))


@pytest.mark.parametrize(
    "kwargs,exc",
    [
        ({"lam": -1.0}, ValueError),
        ({"alpha": -0.5}, ValueError),
        ({"kappa": 0.0}, ValueError),
        ({"kappa": -1.0}, ValueError),
        ({"lam": float("nan")}, ValueError),
    ],
)
def test_invalid_penalty_hyperparameters(kwargs, exc):
    y = _stable_var_series()
    with pytest.raises(exc):
        fit_ridge_var(y, p=1, **kwargs)


def test_invalid_lag_and_non_finite_data():
    y = _stable_var_series()
    with pytest.raises(ValueError):
        fit_ridge_var(y, p=0)
    bad = y.copy()
    bad[5, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        fit_ridge_var(bad, p=1)


def test_variable_names_length_mismatch():
    y = _stable_var_series()
    with pytest.raises(ValueError, match="length"):
        fit_ridge_var(y, p=1, variable_names=["only_one"])


def test_penalty_config_validation():
    with pytest.raises(ValueError):
        PenaltyConfig(lam=-1.0)
    with pytest.raises(ValueError):
        PenaltyConfig(kappa=0.0)
    cfg = PenaltyConfig(lam=1.0, alpha=2.0, kappa=3.0)
    assert cfg.to_dict()["kappa"] == 3.0
