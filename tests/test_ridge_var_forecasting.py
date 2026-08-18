"""Tests for deterministic iterated forecasting of the ridge VAR."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from regularized_var.estimators import RidgeVARResult, PenaltyConfig, fit_ridge_var
from regularized_var.forecasting import iterated_forecast


def _make_result(lag_coefficients, intercept, *, p, names):
    n = len(names)
    return RidgeVARResult(
        lag_coefficients=np.asarray(lag_coefficients, dtype=float),
        intercept=np.asarray(intercept, dtype=float),
        lag_order=p,
        variable_names=tuple(names),
        penalty=PenaltyConfig(),
        residuals=np.zeros((1, n)),
        residual_covariance=np.eye(n),
        design_rank=n,
        design_condition_number=1.0,
        companion_matrix=np.zeros((n * p, n * p)),
        max_companion_eigenvalue=0.0,
        is_stable=True,
    )


def test_one_step_forecast_by_hand_var1():
    # y_hat = c + A1 @ y_last
    A1 = np.array([[0.5, 0.1], [-0.2, 0.3]])
    c = np.array([1.0, 2.0])
    result = _make_result([A1], c, p=1, names=["a", "b"])
    history = np.array([[10.0, 20.0], [4.0, 8.0]])  # last row is most recent
    forecast = iterated_forecast(result, history, horizon=1)
    expected = c + A1 @ np.array([4.0, 8.0])
    assert forecast.shape == (1, 2)
    np.testing.assert_allclose(forecast[0], expected)


def test_multistep_recursion_by_hand_var1():
    A1 = np.array([[0.5, 0.0], [0.0, 0.2]])
    c = np.array([0.0, 0.0])
    result = _make_result([A1], c, p=1, names=["a", "b"])
    y0 = np.array([2.0, 5.0])
    history = np.array([[0.0, 0.0], y0])
    forecast = iterated_forecast(result, history, horizon=3)
    # Diagonal system: each variable decays independently.
    step1 = A1 @ y0
    step2 = A1 @ step1
    step3 = A1 @ step2
    np.testing.assert_allclose(forecast[0], step1)
    np.testing.assert_allclose(forecast[1], step2)
    np.testing.assert_allclose(forecast[2], step3)


def test_multistep_recursion_by_hand_var2():
    # VAR(2): y_hat_t = c + A1 y_{t-1} + A2 y_{t-2}.
    A1 = np.array([[0.4]])
    A2 = np.array([[0.2]])
    c = np.array([1.0])
    result = _make_result([A1, A2], c, p=2, names=["x"])
    # History oldest -> newest; last two rows seed the recursion.
    history = np.array([[3.0], [5.0]])  # y_{t-2}=3, y_{t-1}=5
    forecast = iterated_forecast(result, history, horizon=3)

    f1 = 1.0 + 0.4 * 5.0 + 0.2 * 3.0
    f2 = 1.0 + 0.4 * f1 + 0.2 * 5.0
    f3 = 1.0 + 0.4 * f2 + 0.2 * f1
    np.testing.assert_allclose(forecast[:, 0], [f1, f2, f3])


def test_deterministic_only_drops_intercept():
    A1 = np.array([[0.5]])
    c = np.array([10.0])
    result = _make_result([A1], c, p=1, names=["x"])
    history = np.array([[0.0], [2.0]])
    with_intercept = iterated_forecast(result, history, horizon=1)
    without_intercept = iterated_forecast(
        result, history, horizon=1, include_deterministic=False
    )
    assert with_intercept[0, 0] == pytest.approx(10.0 + 0.5 * 2.0)
    assert without_intercept[0, 0] == pytest.approx(0.5 * 2.0)


def test_forecast_uses_only_last_p_rows():
    A1 = np.array([[1.0]])
    c = np.array([0.0])
    result = _make_result([A1], c, p=1, names=["x"])
    long_history = np.array([[999.0], [7.0]])
    forecast = iterated_forecast(result, long_history, horizon=1)
    # Only the most recent row (7.0) matters for a VAR(1) identity map.
    assert forecast[0, 0] == pytest.approx(7.0)


def test_output_shape_and_variable_order():
    y = np.random.default_rng(0).normal(size=(120, 3))
    result = fit_ridge_var(y, p=2, lam=1.0, variable_names=["a", "b", "c"])
    forecast = iterated_forecast(result, y[-2:], horizon=5)
    assert forecast.shape == (5, 3)
    assert np.all(np.isfinite(forecast))


def test_forecast_input_validation():
    A1 = np.array([[0.5]])
    result = _make_result([A1], np.array([0.0]), p=1, names=["x"])
    with pytest.raises(ValueError, match="horizon"):
        iterated_forecast(result, np.array([[1.0]]), horizon=0)
    with pytest.raises(TypeError):
        iterated_forecast(result, np.array([[1.0]]), horizon=1.5)
    with pytest.raises(ValueError, match="at least p"):
        iterated_forecast(result, np.empty((0, 1)), horizon=1)
    with pytest.raises(ValueError, match="columns"):
        iterated_forecast(result, np.array([[1.0, 2.0]]), horizon=1)
    with pytest.raises(ValueError, match="non-finite"):
        iterated_forecast(result, np.array([[np.nan]]), horizon=1)


def test_stable_var_forecast_converges_to_mean():
    # A stable VAR forecast should settle toward its unconditional mean.
    A1 = np.array([[0.5]])
    c = np.array([1.0])  # mean = c / (1 - 0.5) = 2.0
    result = _make_result([A1], c, p=1, names=["x"])
    history = np.array([[0.0], [0.0]])
    forecast = iterated_forecast(result, history, horizon=40)
    assert forecast[-1, 0] == pytest.approx(2.0, abs=1e-6)
