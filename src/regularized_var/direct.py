"""Direct (non-recursive) multi-step ridge VAR estimation.

The iterated VAR in :mod:`regularized_var.forecasting` produces an ``h``-step
forecast by recursively chaining one-step predictions. A **direct** ``h``-step
model instead estimates a single mapping from the information available at time
``t`` straight to the target vector at time ``t + h`` -- no intermediate
forecasts are generated.

Two ideas are kept explicitly separate:

* A **horizon-specific coefficient model.** Direct forecasting *inherently*
  fits a different coefficient system for every horizon ``h`` because the
  regression target changes with ``h``. This is a structural property of the
  architecture, not a tuning choice.
* **Horizon-specific regularization selection.** Whether each horizon receives
  its *own hyperparameters* is a separate treatment governed by the
  :class:`~common_hpo.SelectionPlan` scope (``horizon`` /
  ``variable_horizon``). The direct estimator here accepts whatever
  hyperparameters the caller selected; it does not decide the selection scope.

Design and lag-vector convention
--------------------------------
The predictor row for a base time ``t`` is exactly the iterated lag design (see
:func:`regularized_var.design.build_lag_design`)::

    X[r] = [ intercept? , y[t-1] , y[t-2] , ... , y[t-p] ]

so **lag 1 is the most recent past** ``y[t-1]`` -- identical to the iterated
VAR. Only the target differs: the direct target at horizon ``h`` is

    Y_h[r] = y[t + h - 1]

Thus for ``h = 1`` the direct design coincides exactly with the one-step VAR
design (target ``y[t]``). Because ``h >= 1`` and every predictor index is
``<= t - 1`` while the target index is ``t + h - 1 >= t``, **no predictor ever
overlaps an unavailable (present or future) target observation**.

Forecasts are produced with a single linear map (no recursion), on the same
transformed / evaluation scale as the iterated model, and are labelled with
``forecast_method == "direct"``. Independently estimated direct horizons carry
**no** implied cross-horizon path coherence.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import numpy as np

from .design import n_predictors, predictor_terms, validate_observations
from .estimators import PenaltyConfig, _solve_equation, build_penalty_weights

__all__ = [
    "DirectRidgeResult",
    "build_direct_design",
    "fit_direct_ridge_var",
    "direct_forecast",
]

FORECAST_METHOD = "direct"


def _validate_horizon(horizon: object) -> int:
    if isinstance(horizon, bool) or not isinstance(horizon, Integral):
        raise TypeError("horizon must be an integer.")
    h = int(horizon)
    if h < 1:
        raise ValueError(f"horizon must be >= 1, got {h}.")
    return h


def _validate_lag_order_direct(p: object, *, n_obs: int, horizon: int) -> int:
    if isinstance(p, bool) or not isinstance(p, Integral):
        raise TypeError("lag order p must be an integer.")
    p_int = int(p)
    if p_int < 1:
        raise ValueError(f"lag order p must be >= 1, got {p_int}.")
    # A direct h-step design needs at least one effective row:
    #   n_eff = n_obs - horizon - p + 1 >= 1  <=>  n_obs >= p + horizon.
    if n_obs < p_int + horizon:
        raise ValueError(
            f"direct horizon-{horizon} design is infeasible: need at least "
            f"p + h = {p_int + horizon} observations, got T={n_obs}."
        )
    return p_int


def build_direct_design(
    y: object,
    p: int,
    horizon: int,
    *,
    include_intercept: bool = True,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Build the direct ``h``-step design ``(X, Y_h, terms)``.

    ``X`` uses the identical lag ordering as :func:`build_lag_design`; the target
    ``Y_h`` is horizon-aligned to ``y[t + h - 1]`` for base time ``t`` running
    over ``p .. T - h`` (inclusive). The effective sample has
    ``T - h - p + 1`` rows.
    """

    arr = validate_observations(y)
    n_obs, n_vars = arr.shape
    h = _validate_horizon(horizon)
    p_int = _validate_lag_order_direct(p, n_obs=n_obs, horizon=h)

    # Base time t runs over [p, T - h]; effective row r maps to t = p + r.
    n_eff = n_obs - h - p_int + 1
    k = n_predictors(n_vars, p_int, include_intercept)

    X = np.empty((n_eff, k), dtype=float)
    col = 0
    if include_intercept:
        X[:, 0] = 1.0
        col = 1
    for lag in range(1, p_int + 1):
        # Predictor block y[t - lag] for t in [p, T - h].
        block = arr[p_int - lag : n_obs - h - lag + 1, :]
        X[:, col : col + n_vars] = block
        col += n_vars

    # Horizon-aligned target y[t + h - 1] for t in [p, T - h] -> rows
    # [p + h - 1, T - 1].
    Y_h = arr[p_int + h - 1 : n_obs, :].copy()
    terms = predictor_terms(n_vars, p_int, include_intercept)
    return X, Y_h, terms


@dataclass(frozen=True)
class DirectRidgeResult:
    """Typed result of a direct ``h``-step ridge fit.

    ``coefficients`` has shape ``(p, n, n)`` indexed ``[lag - 1, equation,
    variable]`` exactly like the iterated ``lag_coefficients``: entry
    ``coefficients[l - 1, j, i]`` is the loading of variable ``i`` at lag ``l``
    onto the horizon-``h`` target for equation ``j``. There is **no** companion
    matrix or stability flag: a direct map is not a recursive dynamic system, so
    stability of an implied path is undefined and intentionally omitted.
    """

    coefficients: np.ndarray
    intercept: np.ndarray
    horizon: int
    lag_order: int
    variable_names: tuple[str, ...]
    penalty: PenaltyConfig
    residuals: np.ndarray
    residual_covariance: np.ndarray
    design_rank: int
    design_condition_number: float
    n_effective: int
    forecast_method: str = FORECAST_METHOD

    @property
    def n_variables(self) -> int:
        return len(self.variable_names)

    def to_dict(self) -> dict[str, object]:
        return {
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept.tolist(),
            "horizon": self.horizon,
            "lag_order": self.lag_order,
            "variable_names": list(self.variable_names),
            "penalty": self.penalty.to_dict(),
            "residual_covariance": self.residual_covariance.tolist(),
            "design_rank": self.design_rank,
            "design_condition_number": self.design_condition_number,
            "n_effective": self.n_effective,
            "forecast_method": self.forecast_method,
        }


def _resolve_variable_names(variable_names: Sequence[str] | None, n_vars: int) -> tuple[str, ...]:
    if variable_names is None:
        return tuple(f"y{i}" for i in range(n_vars))
    names = tuple(str(name) for name in variable_names)
    if len(names) != n_vars:
        raise ValueError(
            f"variable_names has length {len(names)} but y has {n_vars} columns."
        )
    if len(set(names)) != len(names):
        raise ValueError("variable_names must be unique.")
    return names


def fit_direct_ridge_var(
    y: object,
    p: int,
    horizon: int,
    *,
    lam: float = 0.0,
    alpha: float = 0.0,
    kappa: float = 1.0,
    include_intercept: bool = True,
    variable_names: Sequence[str] | None = None,
    ddof: int | None = None,
) -> DirectRidgeResult:
    """Fit a direct ``h``-step lag-weighted ridge model.

    The **complete** target vector is always estimated: every variable's
    horizon-``h`` equation is fit, regardless of which variables a downstream
    loss cell happens to score. The penalty structure is identical to the
    iterated estimator (unpenalized intercept; own-lag weight ``l**alpha``;
    cross weight ``kappa * l**alpha``) and is solved with the same augmented,
    inverse-free least-squares routine.
    """

    penalty = PenaltyConfig(
        lam=lam, alpha=alpha, kappa=kappa, include_intercept=include_intercept
    )
    h = _validate_horizon(horizon)

    X, Y, _terms = build_direct_design(y, p, h, include_intercept=include_intercept)
    n_eff, n_vars = Y.shape
    p_int = int(p)
    names = _resolve_variable_names(variable_names, n_vars)

    singular_values = np.linalg.svd(X, compute_uv=False)
    max_sv = float(singular_values[0]) if singular_values.size else 0.0
    min_sv = float(singular_values[-1]) if singular_values.size else 0.0
    rank_tol = max(X.shape) * np.finfo(float).eps * max_sv
    design_rank = int(np.sum(singular_values > rank_tol))
    condition_number = max_sv / min_sv if min_sv > 0.0 else float("inf")

    k = X.shape[1]
    B = np.empty((k, n_vars), dtype=float)
    if penalty.lam == 0.0:
        B, *_ = np.linalg.lstsq(X, Y, rcond=None)
    else:
        for j in range(n_vars):
            weights = build_penalty_weights(
                j,
                n_vars=n_vars,
                p=p_int,
                alpha=penalty.alpha,
                kappa=penalty.kappa,
                include_intercept=include_intercept,
            )
            B[:, j] = _solve_equation(X, Y[:, j], weights, penalty.lam)

    if include_intercept:
        intercept = B[0, :].copy()
        lag_block = B[1:, :]
    else:
        intercept = np.zeros(n_vars, dtype=float)
        lag_block = B

    # Orient as coefficients[lag, eq, var] like the iterated lag_coefficients.
    coefficients = np.empty((p_int, n_vars, n_vars), dtype=float)
    for lag in range(p_int):
        block = lag_block[lag * n_vars : (lag + 1) * n_vars, :]  # (var i, eq j)
        coefficients[lag] = block.T  # -> (eq j, var i)

    residuals = Y - X @ B
    if ddof is None:
        ddof = k
    denominator = n_eff - ddof
    if denominator <= 0:
        denominator = n_eff
    residual_covariance = (residuals.T @ residuals) / denominator

    return DirectRidgeResult(
        coefficients=coefficients,
        intercept=intercept,
        horizon=h,
        lag_order=p_int,
        variable_names=names,
        penalty=penalty,
        residuals=residuals,
        residual_covariance=residual_covariance,
        design_rank=design_rank,
        design_condition_number=float(condition_number),
        n_effective=n_eff,
    )


def direct_forecast(
    result: DirectRidgeResult,
    history: object,
    *,
    include_deterministic: bool = True,
) -> np.ndarray:
    """Produce a single direct ``h``-step point forecast for all variables.

    ``history`` is an ``m x n`` array with ``m >= p`` rows, most recent
    observation **last**. Only the final ``p`` rows seed the map. The forecast is
    the one-shot linear combination

        y_hat_{o+h} = c + sum_{l=1..p} C_l @ state[-l]

    where ``state[-1]`` is the most recent observation ``y[o]`` and
    ``C_l = coefficients[l - 1]``. No recursion is performed and no path is
    implied for the intermediate horizons.

    Returns a 1D array of shape ``(n,)`` in fitted variable order.
    """

    p = result.lag_order
    n = result.n_variables

    hist = np.asarray(history, dtype=float)
    if hist.ndim != 2:
        raise ValueError(f"history must be a two-dimensional m x n array, got ndim={hist.ndim}.")
    if hist.shape[1] != n:
        raise ValueError(
            f"history has {hist.shape[1]} columns but the fitted model has {n} variables."
        )
    if hist.shape[0] < p:
        raise ValueError(
            f"history needs at least p={p} rows, got {hist.shape[0]}."
        )
    if not np.all(np.isfinite(hist)):
        raise ValueError("history contains non-finite values; resolve them upstream.")

    intercept = result.intercept if include_deterministic else np.zeros(n, dtype=float)
    state = hist[hist.shape[0] - p : hist.shape[0]]  # oldest -> newest
    prediction = intercept.copy()
    for lag in range(1, p + 1):
        # C_1 multiplies the most recent vector state[-1] == y[o], matching the
        # lag-1 == most-recent convention of the iterated design.
        prediction = prediction + result.coefficients[lag - 1] @ state[-lag]
    return prediction
