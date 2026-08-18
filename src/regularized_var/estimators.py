"""Lag-weighted ridge VAR estimator.

This is the core estimator for the regularized VAR. It has **no** Bayesian or
optional heavy dependencies: only NumPy. The estimator solves, per target
equation, a penalized least-squares problem

    minimize_over_b  || Y_j - X b ||_2^2  +  lambda * sum_k d_{j,k} * b_k^2

where ``X`` is the documented lag design (see :mod:`.design`), ``Y_j`` is the
``j``-th target column, and ``d_{j,k}`` is the equation-specific penalty weight
for predictor ``k``:

* the intercept is **unpenalized** (``d = 0``);
* an own-lag coefficient at lag ``l`` (variable ``i`` in equation ``i``) has
  weight ``l**alpha``;
* a cross-variable coefficient (variable ``i != j``) at lag ``l`` has weight
  ``kappa * l**alpha``.

Numerical choices
-----------------
* The penalized problem is solved as an **augmented least-squares** system

      minimize || [ X ; sqrt(lambda) * R_j ] b - [ Y_j ; 0 ] ||_2^2

  with ``R_j = diag(sqrt(d_{j,:}))``. This is algebraically identical to the
  ridge normal equations ``(X^T X + lambda D_j) b = X^T Y_j`` but is solved with
  a QR-based least-squares routine (``numpy.linalg.lstsq``). We **never form an
  explicit inverse** and we avoid explicitly forming ``X^T X`` (which squares the
  condition number).
* ``lambda == 0`` reduces to a single stable least-squares solve of ``X b = Y``
  for all equations at once, matching ``numpy.linalg.lstsq``.
* Conditioning is reported via the singular values of ``X`` (rank at a relative
  tolerance and the 2-norm condition number).

Coefficient orientation (documented)
------------------------------------
The fitted lag coefficients are returned as ``lag_coefficients`` with shape
``(p, n, n)`` where ``lag_coefficients[l - 1, j, i]`` is the effect of variable
``i`` at lag ``l`` on equation (target variable) ``j``. This is exactly the
``A_l`` matrix in the VAR form

    y_t = c + sum_{l=1..p} A_l y_{t-l} + u_t

so ``A_l @ y_{t-l}`` maps lagged variables (index ``i``) to equations (index
``j``). The intercept ``c`` is returned separately as ``intercept`` of shape
``(n,)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Sequence

import numpy as np

from .design import build_lag_design, n_predictors

__all__ = [
    "PenaltyConfig",
    "RidgeVARResult",
    "fit_ridge_var",
    "build_penalty_weights",
    "UnstableVARError",
]


class UnstableVARError(ValueError):
    """Raised when a caller policy rejects an unstable fitted VAR."""


@dataclass(frozen=True)
class PenaltyConfig:
    """Penalty configuration for the lag-weighted ridge VAR.

    Attributes
    ----------
    lam:
        Ridge strength ``lambda >= 0``.
    alpha:
        Lag-decay exponent ``alpha >= 0``; lag ``l`` is penalized by ``l**alpha``.
    kappa:
        Cross-variable multiplier ``kappa > 0`` applied to off-diagonal (cross)
        coefficients relative to the own-lag penalty.
    include_intercept:
        Whether an unpenalized intercept is estimated.
    """

    lam: float = 0.0
    alpha: float = 0.0
    kappa: float = 1.0
    include_intercept: bool = True

    def __post_init__(self) -> None:
        for name in ("lam", "alpha", "kappa"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number.")
            object.__setattr__(self, name, float(value))
        if not np.isfinite(self.lam) or self.lam < 0.0:
            raise ValueError(f"lam must be a finite value >= 0, got {self.lam}.")
        if not np.isfinite(self.alpha) or self.alpha < 0.0:
            raise ValueError(f"alpha must be a finite value >= 0, got {self.alpha}.")
        if not np.isfinite(self.kappa) or self.kappa <= 0.0:
            raise ValueError(f"kappa must be a finite value > 0, got {self.kappa}.")

    def to_dict(self) -> dict[str, object]:
        return {
            "lam": self.lam,
            "alpha": self.alpha,
            "kappa": self.kappa,
            "include_intercept": self.include_intercept,
        }


@dataclass(frozen=True)
class RidgeVARResult:
    """Typed result of a lag-weighted ridge VAR fit.

    See the module docstring for the coefficient orientation. ``lag_coefficients``
    has shape ``(p, n, n)`` indexed ``[lag - 1, equation, variable]``.
    """

    lag_coefficients: np.ndarray
    intercept: np.ndarray
    lag_order: int
    variable_names: tuple[str, ...]
    penalty: PenaltyConfig
    residuals: np.ndarray
    residual_covariance: np.ndarray
    design_rank: int
    design_condition_number: float
    companion_matrix: np.ndarray
    max_companion_eigenvalue: float
    is_stable: bool

    @property
    def n_variables(self) -> int:
        return len(self.variable_names)

    def to_dict(self) -> dict[str, object]:
        return {
            "lag_coefficients": self.lag_coefficients.tolist(),
            "intercept": self.intercept.tolist(),
            "lag_order": self.lag_order,
            "variable_names": list(self.variable_names),
            "penalty": self.penalty.to_dict(),
            "residual_covariance": self.residual_covariance.tolist(),
            "design_rank": self.design_rank,
            "design_condition_number": self.design_condition_number,
            "max_companion_eigenvalue": self.max_companion_eigenvalue,
            "is_stable": self.is_stable,
        }


def _resolve_variable_names(
    variable_names: Sequence[str] | None, n_vars: int
) -> tuple[str, ...]:
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


def build_penalty_weights(
    equation_index: int,
    *,
    n_vars: int,
    p: int,
    alpha: float,
    kappa: float,
    include_intercept: bool,
) -> np.ndarray:
    """Return the diagonal penalty weights ``d_{j,:}`` for one equation.

    The returned vector aligns with the design columns: the intercept (if any)
    has weight ``0``; a predictor at lag ``l`` for variable ``i`` has weight
    ``l**alpha`` when ``i == equation_index`` and ``kappa * l**alpha`` otherwise.
    """

    k = n_predictors(n_vars, p, include_intercept)
    weights = np.empty(k, dtype=float)
    col = 0
    if include_intercept:
        weights[0] = 0.0
        col = 1
    for lag in range(1, p + 1):
        lag_penalty = float(lag) ** alpha
        for variable in range(n_vars):
            own = variable == equation_index
            weights[col] = lag_penalty if own else kappa * lag_penalty
            col += 1
    return weights


def _solve_equation(
    X: np.ndarray, y_col: np.ndarray, penalty_weights: np.ndarray, lam: float
) -> np.ndarray:
    """Solve one penalized equation via augmented least squares (no inverse)."""

    if lam == 0.0:
        beta, *_ = np.linalg.lstsq(X, y_col, rcond=None)
        return beta
    # Augment with sqrt(lambda * d) rows. Zero-weight (intercept) rows contribute
    # nothing and keep the intercept unpenalized.
    scale = np.sqrt(lam * penalty_weights)
    aug_rows = np.diag(scale)
    X_aug = np.vstack([X, aug_rows])
    y_aug = np.concatenate([y_col, np.zeros(X.shape[1])])
    beta, *_ = np.linalg.lstsq(X_aug, y_aug, rcond=None)
    return beta


def _companion_matrix(lag_coefficients: np.ndarray) -> np.ndarray:
    p, n, _ = lag_coefficients.shape
    size = n * p
    companion = np.zeros((size, size), dtype=float)
    for lag in range(p):
        companion[0:n, lag * n : (lag + 1) * n] = lag_coefficients[lag]
    if p > 1:
        companion[n:, : n * (p - 1)] = np.eye(n * (p - 1))
    return companion


def fit_ridge_var(
    y: object,
    p: int,
    *,
    lam: float = 0.0,
    alpha: float = 0.0,
    kappa: float = 1.0,
    include_intercept: bool = True,
    variable_names: Sequence[str] | None = None,
    ddof: int | None = None,
    stability_threshold: float = 1.0,
    reject_unstable: bool = False,
) -> RidgeVARResult:
    """Fit a lag-weighted ridge VAR and return a typed result.

    Parameters mirror :class:`PenaltyConfig` plus estimation controls:

    ``ddof``
        Denominator degrees-of-freedom subtracted when estimating the residual
        covariance (``(Y - XB)^T (Y - XB) / (n_eff - ddof)``). Defaults to the
        number of predictors ``k``; it falls back to ``n_eff`` if that would be
        non-positive.
    ``stability_threshold``
        A VAR is flagged stable when the maximum companion eigenvalue modulus is
        strictly below this threshold (default ``1.0``).
    ``reject_unstable``
        Caller policy. When ``True`` an unstable fit raises
        :class:`UnstableVARError`. The default (``False``) never discards an
        unstable fit; it only reports stability.
    """

    penalty = PenaltyConfig(
        lam=lam, alpha=alpha, kappa=kappa, include_intercept=include_intercept
    )

    X, Y, _terms = build_lag_design(y, p, include_intercept=include_intercept)
    n_eff, n_vars = Y.shape
    p_int = int(p)
    names = _resolve_variable_names(variable_names, n_vars)

    # Conditioning diagnostics from the design singular values.
    singular_values = np.linalg.svd(X, compute_uv=False)
    max_sv = float(singular_values[0]) if singular_values.size else 0.0
    min_sv = float(singular_values[-1]) if singular_values.size else 0.0
    rank_tol = max(X.shape) * np.finfo(float).eps * max_sv
    design_rank = int(np.sum(singular_values > rank_tol))
    if min_sv > 0.0:
        condition_number = max_sv / min_sv
    else:
        condition_number = float("inf")

    k = X.shape[1]
    B = np.empty((k, n_vars), dtype=float)
    if penalty.lam == 0.0:
        # Single stable least-squares solve for all equations at once.
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

    # Split intercept and lag blocks; orient lag coefficients as A_l[eq, var].
    if include_intercept:
        intercept = B[0, :].copy()
        lag_block = B[1:, :]
    else:
        intercept = np.zeros(n_vars, dtype=float)
        lag_block = B
    # lag_block rows are ordered [lag1 var0..n-1, lag2 var0..n-1, ...]; row for
    # (lag l, variable i) is predictor -> equation coefficient B[row, j].
    lag_coefficients = np.empty((p_int, n_vars, n_vars), dtype=float)
    for lag in range(p_int):
        block = lag_block[lag * n_vars : (lag + 1) * n_vars, :]  # (var i, eq j)
        lag_coefficients[lag] = block.T  # -> (eq j, var i) == A_{lag+1}

    residuals = Y - X @ B
    if ddof is None:
        ddof = k
    denominator = n_eff - ddof
    if denominator <= 0:
        denominator = n_eff
    residual_covariance = (residuals.T @ residuals) / denominator

    companion = _companion_matrix(lag_coefficients)
    if companion.size:
        eigenvalues = np.linalg.eigvals(companion)
        max_eig = float(np.max(np.abs(eigenvalues)))
    else:  # pragma: no cover - p>=1 guarantees a non-empty companion
        max_eig = 0.0
    is_stable = bool(max_eig < float(stability_threshold))

    if reject_unstable and not is_stable:
        raise UnstableVARError(
            f"fitted VAR is unstable: max companion eigenvalue {max_eig:.6f} >= "
            f"threshold {float(stability_threshold):.6f}."
        )

    return RidgeVARResult(
        lag_coefficients=lag_coefficients,
        intercept=intercept,
        lag_order=p_int,
        variable_names=names,
        penalty=penalty,
        residuals=residuals,
        residual_covariance=residual_covariance,
        design_rank=design_rank,
        design_condition_number=float(condition_number),
        companion_matrix=companion,
        max_companion_eigenvalue=max_eig,
        is_stable=is_stable,
    )
