"""Lag-design construction for the regularized (ridge) VAR estimator.

This module is intentionally free of any Bayesian or optional heavy
dependencies. It only builds the stacked regression design used by the
lag-weighted ridge VAR.

Data contract
-------------
* Input ``y`` is a two-dimensional ``T x n`` numeric array. Row ``t`` is the
  observation vector at time ``t``; column ``i`` is variable ``i``. Variable
  order is explicit and preserved everywhere downstream.
* Missing values must be resolved by an upstream data adapter. This module
  fails descriptively on any non-finite value rather than imputing.

Design-matrix ordering (documented and stable)
----------------------------------------------
For a lag order ``p`` and ``T`` observations, the effective sample has
``T - p`` rows. Effective row ``r`` corresponds to time ``t = p + r``:

* ``Y[r] = y[t]`` (the contemporaneous target vector).
* ``X[r]`` is laid out as::

      [ intercept? , lag1 vars(0..n-1) , lag2 vars(0..n-1) , ... , lagp vars(0..n-1) ]

  where ``lag l`` block equals ``y[t - l]``. Thus **lag 1 is the most recent
  past** (``y[t-1]``) and the block for a given lag preserves the original
  variable order. When ``include_intercept`` is ``True`` the intercept is the
  first column and always equals ``1.0``.

The predictor term list returned alongside the matrices makes this ordering
introspectable: entry ``0`` is ``("intercept",)`` when present, and every lag
predictor is ``("lag", l, i)`` for lag ``l`` (1-based) and variable index ``i``.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Sequence

import numpy as np

__all__ = [
    "PredictorTerm",
    "validate_observations",
    "build_lag_design",
    "predictor_terms",
    "n_predictors",
]

# A predictor term is either ("intercept",) or ("lag", lag, variable_index).
PredictorTerm = tuple


def validate_observations(y: object) -> np.ndarray:
    """Return ``y`` as a validated ``float64`` ``T x n`` array.

    Raises ``ValueError`` on a non-2D shape, an empty dimension, or any
    non-finite value (NaN or infinity). Imputation is explicitly out of scope.
    """

    arr = np.asarray(y, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"y must be a two-dimensional T x n array, got ndim={arr.ndim}.")
    if arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(f"y must have at least one row and one column, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            "y contains non-finite values (NaN or inf); missing values must be "
            "resolved by an upstream data adapter before estimation."
        )
    return arr


def _validate_lag_order(p: object, n_obs: int) -> int:
    if isinstance(p, bool) or not isinstance(p, Integral):
        raise TypeError("lag order p must be an integer.")
    p_int = int(p)
    if p_int < 1:
        raise ValueError(f"lag order p must be >= 1, got {p_int}.")
    if p_int >= n_obs:
        raise ValueError(
            f"lag order p={p_int} leaves no effective sample for T={n_obs} observations; "
            "require p < T."
        )
    return p_int


def predictor_terms(n_vars: int, p: int, include_intercept: bool) -> list[PredictorTerm]:
    """Return the ordered predictor-term descriptors for the design columns."""

    terms: list[PredictorTerm] = []
    if include_intercept:
        terms.append(("intercept",))
    for lag in range(1, p + 1):
        for variable in range(n_vars):
            terms.append(("lag", lag, variable))
    return terms


def n_predictors(n_vars: int, p: int, include_intercept: bool) -> int:
    """Return the number of design columns (predictors)."""

    return n_vars * p + (1 if include_intercept else 0)


def build_lag_design(
    y: object,
    p: int,
    *,
    include_intercept: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[PredictorTerm]]:
    """Build the stacked lag design ``(X, Y, terms)`` for a VAR(``p``).

    Parameters
    ----------
    y:
        Validated or raw ``T x n`` observation array (revalidated here).
    p:
        Lag order (``>= 1`` and ``< T``).
    include_intercept:
        Whether to prepend an all-ones intercept column.

    Returns
    -------
    X:
        ``(T - p) x k`` design matrix with ``k = n * p (+1)`` following the
        documented ordering.
    Y:
        ``(T - p) x n`` contemporaneous target matrix.
    terms:
        The ordered predictor-term descriptors, one per column of ``X``.
    """

    arr = validate_observations(y)
    n_obs, n_vars = arr.shape
    p_int = _validate_lag_order(p, n_obs)

    n_eff = n_obs - p_int
    k = n_predictors(n_vars, p_int, include_intercept)

    X = np.empty((n_eff, k), dtype=float)
    col = 0
    if include_intercept:
        X[:, 0] = 1.0
        col = 1
    for lag in range(1, p_int + 1):
        # Effective row r maps to time t = p + r; this lag block is y[t - lag].
        block = arr[p_int - lag : n_obs - lag, :]
        X[:, col : col + n_vars] = block
        col += n_vars

    Y = arr[p_int:, :].copy()
    terms = predictor_terms(n_vars, p_int, include_intercept)
    return X, Y, terms
