"""Model-independent statistical inference for paired forecast comparisons.

This module is deliberately free of pandas and of any model/domain code. It
operates on **paired** loss differentials computed from aligned outer-sample
forecast errors and provides:

* :func:`diebold_mariano` -- the Diebold-Mariano test with a HAC (Newey-West)
  long-run variance appropriate for overlapping ``h``-step forecasts and the
  Harvey-Leybourne-Newbold (1997) small-sample correction.
* :func:`hac_long_run_variance` -- the Bartlett-kernel HAC estimator of the
  long-run variance of a mean.
* :func:`moving_block_bootstrap_ci` / :func:`stationary_bootstrap_ci` -- block
  bootstrap confidence intervals for the mean loss differential, reproducible
  from an explicit seed.
* :func:`holm_adjust` -- Holm's step-down multiple-comparison correction for a
  family of related p-values.

Refusal to report an invalid p-value
------------------------------------
:func:`diebold_mariano` returns a result with ``valid=False`` and a documented
``reason`` (never a fabricated p-value) when:

* the loss differential has zero (or numerically non-positive) estimated
  long-run variance (e.g. two identical forecasters);
* the effective sample is too small for the requested overlap;
* the inputs are not paired / not aligned (unequal lengths or non-finite
  values).

The Model Confidence Set is intentionally *not* implemented here; it is an
optional extension and must never be a hidden dependency of the core layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal, Sequence

import numpy as np


__all__ = [
    "DMResult",
    "BootstrapResult",
    "hac_long_run_variance",
    "diebold_mariano",
    "moving_block_bootstrap_ci",
    "stationary_bootstrap_ci",
    "bootstrap_ci",
    "holm_adjust",
]


# --------------------------------------------------------------------------- #
# Numerical helpers: Student-t survival via the regularized incomplete beta.
# --------------------------------------------------------------------------- #
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""

    tiny = 1e-30
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-12:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function ``I_x(a, b)``."""

    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(ln_beta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _student_t_two_sided_p(t_stat: float, dof: float) -> float:
    """Two-sided p-value for a Student-t statistic with ``dof`` > 0."""

    if dof <= 0.0:
        return float("nan")
    x = dof / (dof + t_stat * t_stat)
    # P(|T| > |t|) = I_x(dof/2, 1/2).
    return float(_betai(dof / 2.0, 0.5, x))


def _as_1d_finite(values: object, *, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{label} must be a 1D array, got ndim={arr.ndim}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} contains non-finite values.")
    return arr


def _as_positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{label} must be >= 1, got {result}.")
    return result


# --------------------------------------------------------------------------- #
# HAC long-run variance
# --------------------------------------------------------------------------- #
def hac_long_run_variance(
    x: Sequence[float],
    *,
    lag: int,
    kernel: Literal["bartlett", "uniform"] = "bartlett",
) -> float:
    """Return the HAC (Newey-West) long-run variance of the mean of ``x``.

    ``lag`` is the truncation lag. For an ``h``-step forecast comparison with
    overlapping errors the appropriate truncation lag is ``h - 1``. The Bartlett
    kernel (default) uses weights ``w_k = 1 - k/(lag + 1)`` and guarantees a
    non-negative estimate; the ``uniform`` (rectangular) kernel reproduces the
    original Diebold-Mariano formula but can be negative in small samples.

    The returned value estimates ``S = gamma_0 + 2 * sum_k w_k gamma_k`` (the
    long-run variance of a single observation), *not* divided by ``n``.
    """

    arr = _as_1d_finite(x, label="x")
    n = arr.size
    if n < 2:
        raise ValueError("at least two observations are required for a HAC variance.")
    if lag < 0:
        raise ValueError("lag must be non-negative.")
    lag = min(int(lag), n - 1)

    centered = arr - arr.mean()
    gamma0 = float(np.dot(centered, centered) / n)
    total = gamma0
    for k in range(1, lag + 1):
        gamma_k = float(np.dot(centered[k:], centered[:-k]) / n)
        if kernel == "bartlett":
            weight = 1.0 - k / (lag + 1.0)
        elif kernel == "uniform":
            weight = 1.0
        else:  # pragma: no cover - guarded by typing
            raise ValueError(f"unknown kernel {kernel!r}.")
        total += 2.0 * weight * gamma_k
    return total


# --------------------------------------------------------------------------- #
# Diebold-Mariano
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DMResult:
    """Outcome of a Diebold-Mariano test on a paired loss differential.

    ``mean_loss_differential`` is ``mean(loss_a - loss_b)``. A negative value
    means model A has the smaller average loss. ``valid`` is ``False`` when no
    trustworthy p-value could be computed; ``reason`` documents why.
    """

    n: int
    horizon: int
    hac_lag: int
    kernel: str
    mean_loss_differential: float
    dm_statistic: float | None
    p_value: float | None
    small_sample_corrected: bool
    valid: bool
    reason: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "horizon": self.horizon,
            "hac_lag": self.hac_lag,
            "kernel": self.kernel,
            "mean_loss_differential": self.mean_loss_differential,
            "dm_statistic": self.dm_statistic,
            "p_value": self.p_value,
            "small_sample_corrected": self.small_sample_corrected,
            "valid": self.valid,
            "reason": self.reason,
        }


def diebold_mariano(
    loss_a: Sequence[float],
    loss_b: Sequence[float],
    *,
    horizon: int = 1,
    kernel: Literal["bartlett", "uniform"] = "bartlett",
    small_sample_correction: bool = True,
    min_observations: int = 8,
    zero_variance_tol: float = 1e-12,
) -> DMResult:
    """Diebold-Mariano test comparing paired per-observation losses.

    Parameters
    ----------
    loss_a, loss_b:
        Equal-length arrays of per-observation losses (e.g. squared errors) for
        two models on the **same** aligned outer origins. The differential is
        ``d = loss_a - loss_b``.
    horizon:
        Forecast horizon ``h``. Overlapping ``h``-step forecasts induce serial
        correlation up to lag ``h - 1``, which sets the HAC truncation lag.
    small_sample_correction:
        Apply the Harvey-Leybourne-Newbold (1997) correction and use a Student-t
        reference distribution with ``n - 1`` degrees of freedom. When ``False``
        the standard-normal reference is used.
    min_observations:
        Minimum paired sample size below which no p-value is reported.

    Returns
    -------
    DMResult
        A result whose ``valid`` flag is ``False`` (with a documented ``reason``
        and ``p_value=None``) whenever a trustworthy p-value cannot be produced.
    """

    h = _as_positive_int(horizon, label="horizon")
    arr_a = _as_1d_finite(loss_a, label="loss_a")
    arr_b = _as_1d_finite(loss_b, label="loss_b")
    if arr_a.size != arr_b.size:
        return DMResult(
            n=min(arr_a.size, arr_b.size), horizon=h, hac_lag=h - 1, kernel=kernel,
            mean_loss_differential=float("nan"), dm_statistic=None, p_value=None,
            small_sample_corrected=False, valid=False,
            reason="unpaired comparison: loss_a and loss_b have different lengths.",
        )

    d = arr_a - arr_b
    n = d.size
    mean_d = float(d.mean())
    hac_lag = h - 1

    if n < max(int(min_observations), h + 1):
        return DMResult(
            n=n, horizon=h, hac_lag=hac_lag, kernel=kernel,
            mean_loss_differential=mean_d, dm_statistic=None, p_value=None,
            small_sample_corrected=False, valid=False,
            reason=(
                f"effective sample too small: n={n} < required "
                f"{max(int(min_observations), h + 1)} for horizon {h}."
            ),
        )

    long_run_var = hac_long_run_variance(d, lag=hac_lag, kernel=kernel)
    if not math.isfinite(long_run_var) or long_run_var <= zero_variance_tol:
        return DMResult(
            n=n, horizon=h, hac_lag=hac_lag, kernel=kernel,
            mean_loss_differential=mean_d, dm_statistic=None, p_value=None,
            small_sample_corrected=False, valid=False,
            reason="zero (or non-positive) estimated long-run variance of the differential.",
        )

    var_mean = long_run_var / n
    dm_stat = mean_d / math.sqrt(var_mean)

    corrected = False
    if small_sample_correction:
        # Harvey, Leybourne & Newbold (1997) finite-sample adjustment factor.
        factor = (n + 1 - 2 * h + h * (h - 1) / n) / n
        if factor <= 0.0:
            return DMResult(
                n=n, horizon=h, hac_lag=hac_lag, kernel=kernel,
                mean_loss_differential=mean_d, dm_statistic=dm_stat, p_value=None,
                small_sample_corrected=False, valid=False,
                reason="small-sample correction factor is non-positive; sample too small for horizon.",
            )
        dm_stat = dm_stat * math.sqrt(factor)
        p_value = _student_t_two_sided_p(dm_stat, dof=n - 1)
        corrected = True
    else:
        # Standard-normal two-sided p-value.
        p_value = math.erfc(abs(dm_stat) / math.sqrt(2.0))

    return DMResult(
        n=n, horizon=h, hac_lag=hac_lag, kernel=kernel,
        mean_loss_differential=mean_d, dm_statistic=float(dm_stat),
        p_value=float(p_value), small_sample_corrected=corrected,
        valid=True, reason=None,
    )


# --------------------------------------------------------------------------- #
# Block bootstrap confidence intervals
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BootstrapResult:
    method: str
    statistic: float
    ci_lower: float
    ci_upper: float
    confidence: float
    n_boot: int
    block_length: float
    seed: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "statistic": self.statistic,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "confidence": self.confidence,
            "n_boot": self.n_boot,
            "block_length": self.block_length,
            "seed": self.seed,
        }


def _percentile_ci(samples: np.ndarray, confidence: float) -> tuple[float, float]:
    alpha = (1.0 - confidence) / 2.0
    lower = float(np.quantile(samples, alpha))
    upper = float(np.quantile(samples, 1.0 - alpha))
    return lower, upper


def moving_block_bootstrap_ci(
    x: Sequence[float],
    *,
    block_length: int,
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> BootstrapResult:
    """Moving-block bootstrap CI for the mean of ``x`` (non-circular blocks)."""

    arr = _as_1d_finite(x, label="x")
    n = arr.size
    if n < 2:
        raise ValueError("at least two observations are required.")
    block = _as_positive_int(block_length, label="block_length")
    block = min(block, n)
    n_boot = _as_positive_int(n_boot, label="n_boot")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must lie in (0, 1).")

    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block)
    max_start = n - block  # inclusive
    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        pieces = [arr[s : s + block] for s in starts]
        sample = np.concatenate(pieces)[:n]
        means[b] = sample.mean()
    lower, upper = _percentile_ci(means, confidence)
    return BootstrapResult(
        method="moving_block", statistic=float(arr.mean()), ci_lower=lower,
        ci_upper=upper, confidence=confidence, n_boot=n_boot,
        block_length=float(block), seed=seed,
    )


def stationary_bootstrap_ci(
    x: Sequence[float],
    *,
    mean_block_length: float,
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> BootstrapResult:
    """Politis-Romano stationary bootstrap CI for the mean of ``x``.

    Block lengths are geometric with mean ``mean_block_length`` and blocks wrap
    around circularly, yielding a stationary resample.
    """

    arr = _as_1d_finite(x, label="x")
    n = arr.size
    if n < 2:
        raise ValueError("at least two observations are required.")
    if not isinstance(mean_block_length, Real) or float(mean_block_length) < 1.0:
        raise ValueError("mean_block_length must be a real number >= 1.")
    n_boot = _as_positive_int(n_boot, label="n_boot")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must lie in (0, 1).")

    p = 1.0 / float(mean_block_length)  # restart probability
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sample = np.empty(n, dtype=float)
        idx = int(rng.integers(0, n))
        for t in range(n):
            sample[t] = arr[idx]
            if rng.random() < p:
                idx = int(rng.integers(0, n))
            else:
                idx = (idx + 1) % n
        means[b] = sample.mean()
    lower, upper = _percentile_ci(means, confidence)
    return BootstrapResult(
        method="stationary", statistic=float(arr.mean()), ci_lower=lower,
        ci_upper=upper, confidence=confidence, n_boot=n_boot,
        block_length=float(mean_block_length), seed=seed,
    )


def bootstrap_ci(
    x: Sequence[float],
    *,
    method: Literal["moving_block", "stationary"] = "moving_block",
    block_length: float,
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> BootstrapResult:
    """Dispatch to a block bootstrap CI by ``method``."""

    if method == "moving_block":
        return moving_block_bootstrap_ci(
            x, block_length=int(block_length), n_boot=n_boot,
            confidence=confidence, seed=seed,
        )
    if method == "stationary":
        return stationary_bootstrap_ci(
            x, mean_block_length=float(block_length), n_boot=n_boot,
            confidence=confidence, seed=seed,
        )
    raise ValueError(f"unknown bootstrap method {method!r}.")


# --------------------------------------------------------------------------- #
# Holm correction
# --------------------------------------------------------------------------- #
def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm step-down adjusted p-values, preserving input order.

    ``None`` / NaN entries (comparisons for which no valid p-value exists) are
    passed through unchanged and are excluded from the family size, so an invalid
    comparison never inflates or deflates the correction of the valid ones.
    """

    indexed: list[tuple[int, float]] = []
    result: list[float] = [float("nan")] * len(p_values)
    for i, p in enumerate(p_values):
        if p is None:
            result[i] = float("nan")
            continue
        value = float(p)
        if math.isnan(value):
            result[i] = float("nan")
            continue
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"p-value at index {i} is outside [0, 1]: {value}.")
        indexed.append((i, value))

    m = len(indexed)
    if m == 0:
        return result

    order = sorted(indexed, key=lambda item: item[1])
    running_max = 0.0
    for rank, (orig_index, p) in enumerate(order):
        adjusted = (m - rank) * p
        adjusted = min(adjusted, 1.0)
        running_max = max(running_max, adjusted)  # enforce monotonicity
        result[orig_index] = running_max
    return result
