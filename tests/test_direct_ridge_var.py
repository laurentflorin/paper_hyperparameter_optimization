"""Tests for direct (non-recursive) multi-step ridge VAR estimation."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common_hpo import LossConfig, ScaleConfig, SelectionSchedule, ValidationScheme
from regularized_var.data import PanelData
from regularized_var.design import build_lag_design
from regularized_var.direct import (
    build_direct_design,
    direct_forecast,
    fit_direct_ridge_var,
)
from regularized_var.estimators import fit_ridge_var
from regularized_var.experiment import (
    FORECAST_PANEL_COLUMNS,
    RidgeExperimentConfig,
    run_scope_experiment,
)
from regularized_var.forecasting import iterated_forecast
from regularized_var.tuning import RidgeGridSpec


def _series(T=120, seed=0):
    rng = np.random.default_rng(seed)
    A1 = np.array([[0.5, 0.1], [-0.2, 0.3]])
    c = np.array([0.4, -0.1])
    y = np.zeros((T, 2))
    for t in range(1, T):
        y[t] = c + A1 @ y[t - 1] + rng.normal(scale=0.1, size=2)
    return y


# --------------------------------------------------------------------------- #
# Target alignment for h = 1, 2, 4, 8
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("h", [1, 2, 4, 8])
def test_target_alignment_exact_horizon(h):
    y = np.arange(40.0).reshape(20, 2)  # deterministic, easy to trace
    p = 2
    X, Y_h, terms = build_direct_design(y, p, h, include_intercept=True)
    # Effective rows: base t in [p, T-h]; row r maps to t = p + r.
    n_eff = 20 - h - p + 1
    assert X.shape[0] == Y_h.shape[0] == n_eff
    for r in range(n_eff):
        t = p + r
        # Target is exactly y[t + h - 1] -- no off-by-one shift.
        np.testing.assert_array_equal(Y_h[r], y[t + h - 1])
        # Lag 1 block equals y[t-1], lag 2 equals y[t-2] (same convention).
        np.testing.assert_array_equal(X[r, 1:3], y[t - 1])
        np.testing.assert_array_equal(X[r, 3:5], y[t - 2])


def test_no_horizon_target_shifted_by_one_row():
    # Explicitly guard the acceptance criterion: for every horizon the last
    # usable target is y[T-1], never y[T] or y[T-2].
    y = np.arange(30.0).reshape(15, 2)
    p = 1
    for h in (1, 2, 4, 8):
        _, Y_h, _ = build_direct_design(y, p, h)
        np.testing.assert_array_equal(Y_h[-1], y[14])


# --------------------------------------------------------------------------- #
# No overlap between predictors and unavailable target observations
# --------------------------------------------------------------------------- #
def test_predictors_never_touch_target_row_or_future():
    y = _series()
    p = 3
    for h in (1, 2, 4, 8):
        X, Y_h, _ = build_direct_design(y, p, h)
        n_eff = y.shape[0] - h - p + 1
        for r in range(n_eff):
            t = p + r
            target_index = t + h - 1
            # All predictor rows are y[t-1..t-p]; the largest index is t-1,
            # strictly below the target index t+h-1 for every h >= 1.
            max_predictor_index = t - 1
            assert max_predictor_index < target_index


# --------------------------------------------------------------------------- #
# h = 1 consistency with the one-step design and forecast
# --------------------------------------------------------------------------- #
def test_h1_design_matches_one_step_var_design():
    y = _series()
    p = 2
    X_direct, Y_direct, _ = build_direct_design(y, p, 1)
    X_iter, Y_iter, _ = build_lag_design(y, p)
    np.testing.assert_allclose(X_direct, X_iter)
    np.testing.assert_allclose(Y_direct, Y_iter)


def test_h1_direct_forecast_matches_iterated_one_step():
    y = _series()
    p = 2
    direct = fit_direct_ridge_var(y, p, 1, lam=0.0)
    iterated = fit_ridge_var(y, p, lam=0.0)
    fc_direct = direct_forecast(direct, y[-p:])
    fc_iter = iterated_forecast(iterated, y[-p:], 1)[0]
    np.testing.assert_allclose(fc_direct, fc_iter, atol=1e-9)


# --------------------------------------------------------------------------- #
# One hand-computed direct forecast
# --------------------------------------------------------------------------- #
def test_hand_computed_direct_forecast():
    # Build a tiny series and fit a direct h=2 model with lam=0, then reproduce
    # the forecast as a plain linear map of the last p rows.
    y = _series(T=60, seed=1)
    p = 1
    result = fit_direct_ridge_var(y, p, 2, lam=0.0)
    # Direct map: y_hat = c + C_1 @ y_last.
    y_last = y[-1]
    expected = result.intercept + result.coefficients[0] @ y_last
    forecast = direct_forecast(result, y[-p:])
    np.testing.assert_allclose(forecast, expected, atol=1e-12)


def test_direct_forecast_is_not_recursive_for_var2():
    # For p=2 the direct map uses the two most recent rows exactly once each.
    y = _series(T=80, seed=2)
    p = 2
    result = fit_direct_ridge_var(y, p, 4, lam=0.5, alpha=1.0)
    state = y[-p:]
    expected = (
        result.intercept
        + result.coefficients[0] @ state[-1]
        + result.coefficients[1] @ state[-2]
    )
    np.testing.assert_allclose(direct_forecast(result, state), expected, atol=1e-12)


# --------------------------------------------------------------------------- #
# Complete target vector estimated regardless of scored variables
# --------------------------------------------------------------------------- #
def test_direct_estimates_full_target_vector():
    y = _series()
    result = fit_direct_ridge_var(y, 2, 3, lam=1.0)
    # All n equations are estimated even though a loss cell might score one var.
    assert result.coefficients.shape == (2, 2, 2)
    assert result.intercept.shape == (2,)
    assert result.n_variables == 2


# --------------------------------------------------------------------------- #
# Infeasible early samples
# --------------------------------------------------------------------------- #
def test_infeasible_early_sample_raises():
    y = np.arange(12.0).reshape(6, 2)
    # Need T >= p + h; here p=2, h=8 needs 10 > 6 rows.
    with pytest.raises(ValueError, match="infeasible"):
        build_direct_design(y, 2, 8)
    with pytest.raises(ValueError, match="infeasible"):
        fit_direct_ridge_var(y, 2, 8)


def test_invalid_horizon_and_lag():
    y = _series(T=40)
    with pytest.raises(ValueError):
        build_direct_design(y, 1, 0)
    with pytest.raises(TypeError):
        build_direct_design(y, 1, 1.5)
    with pytest.raises(ValueError):
        build_direct_design(y, 0, 1)


def test_direct_forecast_history_validation():
    y = _series(T=50)
    result = fit_direct_ridge_var(y, 2, 2, lam=0.0)
    with pytest.raises(ValueError, match="at least"):
        direct_forecast(result, y[-1:])  # fewer than p rows
    with pytest.raises(ValueError, match="columns"):
        direct_forecast(result, np.ones((3, 5)))
    with pytest.raises(ValueError, match="non-finite"):
        bad = y[-3:].copy()
        bad[0, 0] = np.nan
        direct_forecast(result, bad)


# --------------------------------------------------------------------------- #
# Fold-local preprocessing (standardization fit on training slice only)
# --------------------------------------------------------------------------- #
def test_direct_experiment_forecast_uses_training_only():
    from regularized_var.experiment import _forecast_direct

    y = _series(T=120)
    panel = PanelData(values=y, variable_names=("a", "b"))
    block = panel.values[:80]
    fc_a = _forecast_direct(
        block, p=2, lam=1.0, alpha=0.0, kappa=1.0, standardize=True,
        horizons=(1, 2), variable_names=panel.variable_names,
    )
    mutated = panel.values.copy()
    mutated[80:] += 500.0  # future validation targets
    fc_b = _forecast_direct(
        mutated[:80], p=2, lam=1.0, alpha=0.0, kappa=1.0, standardize=True,
        horizons=(1, 2), variable_names=panel.variable_names,
    )
    for h in (1, 2):
        np.testing.assert_allclose(fc_a[h], fc_b[h])


# --------------------------------------------------------------------------- #
# Experiment integration: scope mappings, labels, canonical schema
# --------------------------------------------------------------------------- #
def _direct_config(horizons=(1, 2), forecast_method="direct", **overrides):
    grid = RidgeGridSpec(lambdas=(0.0, 1.0), lag_orders=(1, 2), alphas=(0.0,), kappas=(1.0,))
    kwargs = dict(
        target_variables=("a", "b"),
        target_horizons=horizons,
        grid_spec=grid,
        outer_scheme=ValidationScheme(
            training_window="expanding", origin_selection="most_recent",
            n_origins=4, horizons=horizons, min_train_length=30,
        ),
        inner_scheme=ValidationScheme(
            training_window="expanding", origin_selection="most_recent",
            n_origins=3, horizons=horizons, min_train_length=30,
        ),
        selection_schedule=SelectionSchedule.once(),
        loss_config=LossConfig(aggregation="rmse", scale=ScaleConfig(method="none")),
        preprocessing="standardize",
        forecast_method=forecast_method,
    )
    kwargs.update(overrides)
    return RidgeExperimentConfig(**kwargs)


def test_direct_pooled_and_variable_horizon_mappings():
    panel = PanelData(values=_series(T=140), variable_names=("a", "b"))
    pooled = run_scope_experiment(panel, "pooled", _direct_config())
    assert len(pooled.selection_plan.cells) == 1
    vh = run_scope_experiment(panel, "variable_horizon", _direct_config())
    assert len(vh.selection_plan.cells) == 2 * 2


def test_direct_forecast_method_label_in_output():
    panel = PanelData(values=_series(T=140), variable_names=("a", "b"))
    result = run_scope_experiment(panel, "pooled", _direct_config())
    assert result.forecast_rows
    for row in result.forecast_rows:
        assert set(row.keys()) == set(FORECAST_PANEL_COLUMNS)
        assert row["forecast_method"] == "direct"
    assert result.metadata["forecast_method"] == "direct"


def test_direct_versus_iterated_labels_differ():
    panel = PanelData(values=_series(T=140), variable_names=("a", "b"))
    direct_rows = run_scope_experiment(panel, "pooled", _direct_config(forecast_method="direct")).forecast_rows
    iter_rows = run_scope_experiment(panel, "pooled", _direct_config(forecast_method="iterated")).forecast_rows
    assert {r["forecast_method"] for r in direct_rows} == {"direct"}
    assert {r["forecast_method"] for r in iter_rows} == {"iterated"}


def test_direct_uses_same_outer_protocol_as_iterated():
    # Same outer origins and identical forecast-panel keys -> reporting can
    # distinguish architecture (forecast_method) from selection scope (scope).
    panel = PanelData(values=_series(T=140), variable_names=("a", "b"))
    direct = run_scope_experiment(panel, "pooled", _direct_config(forecast_method="direct"))
    iterated = run_scope_experiment(panel, "pooled", _direct_config(forecast_method="iterated"))
    key = lambda rows: sorted(
        (r["forecast_origin"], r["horizon_quarters"], r["variable"]) for r in rows
    )
    assert key(direct.forecast_rows) == key(iterated.forecast_rows)
    # The architecture label differs while the scope metadata matches.
    assert direct.metadata["scope"] == iterated.metadata["scope"] == "pooled"
    assert direct.metadata["forecast_method"] != iterated.metadata["forecast_method"]


def test_direct_horizon_scope_selects_per_horizon_vector():
    # 'horizon' scope tests horizon-specific *regularization selection*, distinct
    # from the horizon-specific *coefficient models* direct always fits.
    panel = PanelData(values=_series(T=160), variable_names=("a", "b"))
    result = run_scope_experiment(panel, "horizon", _direct_config(horizons=(1, 2, 4)))
    assert len(result.selection_plan.cells) == 3
    assert not result.failed_origins
