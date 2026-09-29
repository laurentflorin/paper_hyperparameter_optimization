"""Forecast-evaluation primitives for the GLP study.

Vintage-consistent targets
--------------------------
Forecasts are produced in the real-time vintage available at each origin and
scored against a fixed later evaluation vintage. For ``100 * log`` series the
two vintages can differ by a large *level* shift (NIPA base-year changes, unit
changes), which has nothing to do with forecast accuracy. Every score is
therefore computed on the change from the forecast jump-off:

    forecast change  = forecast_{T+h}      - y_T  (real-time vintage)
    realised change  = actual_{T+h}        - a_T  (evaluation vintage)

which is equivalent to comparing the level forecast with the *adjusted actual*
``actual - a_T + y_T``. Rate (``lin``) series are scored in levels.

Tests
-----
* Diebold-Mariano with a Newey-West long-run variance (``h - 1`` lags) and the
  Harvey-Leybourne-Newbold small-sample correction, Student-t reference.
* Holm step-down adjustment for families of DM tests.
* Model Confidence Set (Hansen, Lunde and Nason, 2011) with the ``T_max``
  statistic and a moving-block bootstrap.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

_SQRT_PI = np.sqrt(np.pi)


# --------------------------------------------------------------------------- #
# Vintage-consistent targets.
# --------------------------------------------------------------------------- #
def adjusted_actual(actual, actual_jumpoff, jumpoff, is_log) -> np.ndarray:
    """Realised value expressed in the forecast vintage's units (see module doc)."""
    actual = np.asarray(actual, dtype=float)
    shift = np.asarray(jumpoff, dtype=float) - np.asarray(actual_jumpoff, dtype=float)
    return np.where(np.asarray(is_log, dtype=bool), actual + shift, actual)


def growth_scale(horizon, is_log) -> np.ndarray:
    """Factor turning a cumulative ``100*log`` change error into annualized
    average-growth percentage points (``4/h``); 1 for rate series."""
    horizon = np.asarray(horizon, dtype=float)
    return np.where(np.asarray(is_log, dtype=bool), 4.0 / horizon, 1.0)


# --------------------------------------------------------------------------- #
# Density scores (lower is better for CRPS, higher is better for log score).
# --------------------------------------------------------------------------- #
def crps_sample(draws: np.ndarray, observation: np.ndarray) -> np.ndarray:
    """Empirical CRPS of an ensemble, column-wise.

    ``draws`` is ``(n_draws, n)`` and ``observation`` is ``(n,)``. Uses
    ``E|X - y| - 0.5 E|X - X'|`` with the sorted-sample identity for the second
    term, so the cost is ``O(m log m)`` per column.
    """
    draws = np.asarray(draws, dtype=float)
    observation = np.asarray(observation, dtype=float)
    m = draws.shape[0]
    first = np.mean(np.abs(draws - observation[None, :]), axis=0)
    ordered = np.sort(draws, axis=0)
    weights = (2.0 * np.arange(1, m + 1) - m - 1.0)[:, None]
    spread = 2.0 * np.sum(weights * ordered, axis=0) / (m * m)
    return first - 0.5 * spread


def crps_gaussian(mu, sigma, observation) -> np.ndarray:
    mu, sigma, observation = (np.asarray(v, dtype=float) for v in (mu, sigma, observation))
    z = (observation - mu) / sigma
    return sigma * (z * (2.0 * stats.norm.cdf(z) - 1.0) + 2.0 * stats.norm.pdf(z) - 1.0 / _SQRT_PI)


def log_score_gaussian(mu, sigma, observation) -> np.ndarray:
    return stats.norm.logpdf(np.asarray(observation, float), loc=np.asarray(mu, float), scale=np.asarray(sigma, float))


# --------------------------------------------------------------------------- #
# Pairwise and multiple-model tests.
# --------------------------------------------------------------------------- #
def _long_run_variance(d: np.ndarray, lags: int) -> float:
    d = d - d.mean()
    T = d.size
    lrv = float(np.dot(d, d) / T)
    for k in range(1, lags + 1):
        lrv += 2.0 * (1.0 - k / (lags + 1.0)) * float(np.dot(d[k:], d[:-k]) / T)
    return lrv if lrv > 0 else float(np.dot(d, d) / T)


def diebold_mariano(loss_model: Sequence[float], loss_baseline: Sequence[float], horizon: int) -> tuple[float, float]:
    """DM test of equal expected loss; negative statistic = model beats baseline.

    Returns ``(statistic, two-sided p-value)``; ``(nan, nan)`` when fewer than
    ten paired observations are available or the differential is degenerate.
    """
    a = np.asarray(loss_model, dtype=float)
    b = np.asarray(loss_baseline, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    T = d.size
    if T < 10 or np.allclose(d, 0.0):
        return float("nan"), float("nan")
    h = max(int(horizon), 1)
    stat = d.mean() / np.sqrt(_long_run_variance(d, h - 1) / T)
    stat *= np.sqrt(max(T + 1 - 2 * h + h * (h - 1) / T, 1e-12) / T)
    return float(stat), float(2.0 * stats.t.sf(abs(stat), T - 1))


def holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (NaNs are passed through)."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if valid.size == 0:
        return out
    order = valid[np.argsort(p[valid])]
    m = order.size
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        out[idx] = running
    return out


def _block_bootstrap_indices(T: int, block: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    n_blocks = int(np.ceil(T / block))
    starts = rng.integers(0, T - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :T]
    return idx


def model_confidence_set(
    losses: pd.DataFrame,
    *,
    alpha: float = 0.10,
    n_boot: int = 2000,
    block: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Hansen-Lunde-Nason MCS with the ``T_max`` elimination rule.

    ``losses`` has one column per model and one row per forecast origin (rows
    with any missing loss are dropped). Returns a frame indexed by model with
    the MCS p-value and an ``in_mcs`` flag at level ``alpha``.
    """
    frame = losses.dropna()
    models = list(frame.columns)
    L = frame.to_numpy(dtype=float)
    T = L.shape[0]
    if T < 10 or len(models) < 2:
        return pd.DataFrame({"mcs_pvalue": np.nan, "in_mcs": True}, index=models)
    block = block or max(1, int(round(T ** (1.0 / 3.0))))
    rng = np.random.default_rng(seed)
    idx = _block_bootstrap_indices(T, block, n_boot, rng)
    boot_means = L[idx].mean(axis=1)  # (n_boot, M)

    alive = list(range(len(models)))
    pvalues = np.zeros(len(models))
    running_p = 0.0
    while len(alive) > 1:
        sub = L[:, alive]
        dbar = sub.mean(axis=0) - sub.mean()  # d_i. = L_i - mean over alive models
        boot_sub = boot_means[:, alive]
        boot_d = boot_sub - boot_sub.mean(axis=1, keepdims=True)
        centred = boot_d - dbar[None, :]
        var = np.mean(centred**2, axis=0)
        var[var <= 0] = np.finfo(float).tiny
        t_obs = dbar / np.sqrt(var)
        t_boot = np.max(centred / np.sqrt(var)[None, :], axis=1)
        p = float(np.mean(t_boot >= np.max(t_obs)))
        running_p = max(running_p, p)
        worst = alive[int(np.argmax(t_obs))]
        pvalues[worst] = running_p
        alive.remove(worst)
    pvalues[alive[0]] = 1.0
    return pd.DataFrame({"mcs_pvalue": pvalues, "in_mcs": pvalues >= alpha}, index=models)
