"""Iterated (multi-step) point forecasting for the ridge VAR.

Point forecasts are purely deterministic: no shocks are added. Given a fitted
:class:`~regularized_var.estimators.RidgeVARResult` and a history of at least
``p`` observations, the recursion is

    y_hat_{T+h} = c + sum_{l=1..p} A_l * state_{h-l}

where ``state`` is the running sequence of the last ``p`` (observed then
forecast) vectors and ``A_l = lag_coefficients[l - 1]`` (orientation
``[equation, variable]``; see the estimator module). The intercept ``c`` is the
deterministic term and can be suppressed with ``include_deterministic=False`` to
obtain the zero-mean dynamics only.

Output shape and variable order
-------------------------------
The returned array has shape ``(horizon, n)`` with columns in the fitted
``variable_names`` order, so forecasts align with the input variable ordering.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np

from .estimators import RidgeVARResult

__all__ = ["iterated_forecast"]


def _validate_horizon(horizon: object) -> int:
    if isinstance(horizon, bool) or not isinstance(horizon, Integral):
        raise TypeError("horizon must be an integer.")
    h = int(horizon)
    if h < 1:
        raise ValueError(f"horizon must be >= 1, got {h}.")
    return h


def iterated_forecast(
    result: RidgeVARResult,
    history: object,
    horizon: int,
    *,
    include_deterministic: bool = True,
) -> np.ndarray:
    """Produce ``horizon``-step iterated point forecasts.

    Parameters
    ----------
    result:
        A fitted :class:`RidgeVARResult`.
    history:
        A ``m x n`` array with ``m >= p`` rows, most recent observation **last**.
        Only the final ``p`` rows are used to seed the recursion. Must be finite
        and match the fitted number of variables.
    horizon:
        Number of steps ``H >= 1``.
    include_deterministic:
        When ``True`` (default) the intercept is added at every step. When
        ``False`` the intercept is dropped (pure zero-mean dynamics).

    Returns
    -------
    numpy.ndarray
        ``(horizon, n)`` forecasts in fitted variable order.
    """

    h = _validate_horizon(horizon)
    p = result.lag_order
    n = result.n_variables

    hist = np.asarray(history, dtype=float)
    if hist.ndim != 2:
        raise ValueError(f"history must be a two-dimensional m x n array, got ndim={hist.ndim}.")
    if hist.shape[1] != n:
        raise ValueError(
            f"history has {hist.shape[1]} columns but the fitted VAR has {n} variables."
        )
    if hist.shape[0] < p:
        raise ValueError(
            f"history needs at least p={p} rows to seed the recursion, got {hist.shape[0]}."
        )
    if not np.all(np.isfinite(hist)):
        raise ValueError("history contains non-finite values; resolve them upstream.")

    intercept = result.intercept if include_deterministic else np.zeros(n, dtype=float)
    lag_coefficients = result.lag_coefficients  # (p, n, n) == A_l[eq, var]

    # ``state`` holds the most recent p vectors, ordered oldest -> newest.
    state = [hist[i].copy() for i in range(hist.shape[0] - p, hist.shape[0])]
    forecasts = np.empty((h, n), dtype=float)
    for step in range(h):
        prediction = intercept.copy()
        # A_1 multiplies the most recent vector (state[-1]), A_l multiplies
        # state[-l], preserving the documented lag-1 == most-recent convention.
        for lag in range(1, p + 1):
            prediction = prediction + lag_coefficients[lag - 1] @ state[-lag]
        forecasts[step] = prediction
        state.append(prediction)
        state.pop(0)
    return forecasts
