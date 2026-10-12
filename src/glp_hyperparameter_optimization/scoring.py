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
* The same test pooled over horizons, and with fixed-smoothing long-run
  variances (equal-weighted cosine, Lazarus, Lewis, Stock and Watson, 2018;
  Daniell periodogram, Coroneo and Iacone, 2020) and Student-t references.
* Holm step-down adjustment for families of DM tests.
* Tests against a benchmark that account for searching over many criteria:
  Hansen's (2005) consistent SPA test and Romano and Wolf's (2005) step-down
  adjusted p-values, both with a moving-block bootstrap of EWC-studentized
  statistics; and, for one
  criterion over several horizons, the average and uniform multi-horizon
  tests of Quaedvlieg (2021), the latter as an intersection-union test.
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


def pooled_diebold_mariano(loss_model: np.ndarray, loss_baseline: np.ndarray) -> tuple[float, float]:
    """DM test of equal loss pooled over horizons.

    ``loss_model`` and ``loss_baseline`` are ``(T, H)`` arrays (origins x
    horizons 1..H). Each horizon's losses are scaled by the baseline's mean loss
    at that horizon, so every horizon carries equal weight, and the scaled
    differential is averaged over horizons; the averaged series overlaps for up
    to ``H - 1`` origins, which the test's Newey-West lags and HLN correction
    for horizon ``H`` account for. Origins with a missing horizon are dropped.
    """
    a = np.asarray(loss_model, dtype=float)
    b = np.asarray(loss_baseline, dtype=float)
    ok = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    a, b = a[ok], b[ok]
    if a.shape[0] < 10:
        return float("nan"), float("nan")
    return diebold_mariano(*pooled_losses(a, b), a.shape[1])


def pooled_losses(loss_model: np.ndarray, loss_baseline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average over horizons of ``(T, H)`` losses, each horizon scaled by the
    baseline's mean loss, so that every horizon carries equal weight."""
    a = np.asarray(loss_model, dtype=float)
    b = np.asarray(loss_baseline, dtype=float)
    scale = b.mean(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    return (a / scale).mean(axis=1), (b / scale).mean(axis=1)


def fixed_smoothing_dm(loss_model: Sequence[float], loss_baseline: Sequence[float], method: str = "ewc") -> tuple[float, float]:
    """DM test with a fixed-smoothing long-run variance and its Student-t reference.

    ``ewc``: equal-weighted cosine estimator with ``nu = floor(0.4 T^(2/3))``
    cosine terms, t with ``nu`` degrees of freedom (Lazarus, Lewis, Stock and
    Watson, 2018). ``wpe``: Daniell (averaged periodogram) estimator over the
    first ``m = floor(T^(1/3))`` Fourier frequencies, t with ``2m`` degrees of
    freedom (Coroneo and Iacone, 2020). Both keep the bandwidth a fixed share of
    the sample, which gives better size than Newey-West in small samples at the
    cost of some power. Returns ``(statistic, two-sided p-value)``; negative =
    model beats baseline.
    """
    a = np.asarray(loss_model, dtype=float)
    b = np.asarray(loss_baseline, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    T = d.size
    if T < 10 or np.allclose(d, 0.0):
        return float("nan"), float("nan")
    e = d - d.mean()
    t_index = np.arange(1, T + 1)
    if method == "ewc":
        dof = ewc_dof(T)
        lrv = float(_ewc_lrv(e[:, None], dof)[0])
    elif method == "wpe":
        m = max(1, int(np.floor(T ** (1.0 / 3.0))))
        dof = 2 * m
        j = np.arange(1, m + 1)[:, None]
        dft = np.exp(2j * np.pi * j * t_index[None, :] / T) @ e
        lrv = float(np.mean(np.abs(dft) ** 2) / T)
    else:
        raise ValueError(f"unknown method {method!r}")
    if lrv <= 0:
        return float("nan"), float("nan")
    stat = d.mean() / np.sqrt(lrv / T)
    return float(stat), float(2.0 * stats.t.sf(abs(stat), dof))


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


def ewc_dof(T: int) -> int:
    """Number of cosine terms of the EWC long-run variance, ``floor(0.4 T^(2/3))``."""
    return max(1, int(np.floor(0.4 * T ** (2.0 / 3.0))))


def _ewc_lrv(x: np.ndarray, dof: int) -> np.ndarray:
    """EWC long-run variance of each column of ``x`` (``... x T x K``): ``... x K``."""
    T = x.shape[-2]
    cos = np.sqrt(2.0 / T) * np.cos(np.pi * np.arange(1, dof + 1)[:, None] * (np.arange(1, T + 1)[None, :] - 0.5) / T)
    lam = np.einsum("jt,...tk->...jk", cos, x - x.mean(axis=-2, keepdims=True))
    return (lam**2).mean(axis=-2)


def benchmark_tests(diff: np.ndarray, *, block: int, n_boot: int = 5000, seed: int = 0, chunk: int = 250) -> dict[str, np.ndarray | float]:
    """Tests of K criteria against one benchmark that account for the search.

    ``diff`` is ``T x K``: benchmark loss minus criterion loss, so a positive
    mean favours the criterion; the null for criterion k is that its expected
    differential is at most zero. Statistics are ``sqrt(T) dbar_k / omega_k``
    with the EWC long-run variance, and their null distribution comes from a
    moving-block bootstrap in which the EWC variance is recomputed on every
    bootstrap sample (Goncalves and Vogelsang, 2011). Bootstrapping the means
    with a fixed standard error rejects far too often with 80 overlapping
    forecast errors; this version keeps close to its nominal size. Returns

    * ``t``: the studentized statistics;
    * ``p_single``: one-sided bootstrap p-value of each criterion on its own;
    * ``p_stepm``: Romano-Wolf (2005) step-down p-values, which control the
      familywise error rate across the K criteria and, unlike Holm, use the
      dependence between them (computed as in Romano and Wolf, 2016);
    * ``p_spa``: Hansen's (2005) consistent SPA p-value for the null that no
      criterion beats the benchmark.

    Criteria whose differential is identically zero get ``t = 0`` and p-values of one.
    """
    d = np.asarray(diff, dtype=float)
    T, K = d.shape
    dof = ewc_dof(T)
    dbar = d.mean(axis=0)
    omega = np.sqrt(_ewc_lrv(d, dof))
    live = omega > 1e-12 * np.maximum(1.0, np.abs(dbar))
    t = np.where(live, np.sqrt(T) * dbar / np.where(live, omega, 1.0), 0.0)
    keep = t >= -np.sqrt(2.0 * np.log(np.log(T)))  # Hansen's g_c: recentre unless clearly worse
    centre_spa = np.where(keep, dbar, 0.0)

    idx = _block_bootstrap_indices(T, max(1, int(block)), n_boot, np.random.default_rng(seed))
    centred, t_spa_boot = [], []
    for i in range(0, n_boot, chunk):
        xb = d[idx[i : i + chunk]]
        mb = xb.mean(axis=1)
        ob = np.sqrt(_ewc_lrv(xb, dof))
        ok = live & (ob > 0)
        ob = np.where(ok, ob, 1.0)
        centred.append(np.where(ok, np.sqrt(T) * (mb - dbar) / ob, 0.0))
        t_spa_boot.append(np.where(ok, np.sqrt(T) * (mb - centre_spa) / ob, -np.inf))
    centred, t_spa_boot = np.vstack(centred), np.vstack(t_spa_boot)

    p_single = np.where(live, (centred >= t).mean(axis=0), 1.0)
    p_stepm = np.ones(K)
    running = 0.0
    order = [k for k in np.argsort(-t) if live[k]]
    for i, k in enumerate(order):
        running = max(running, float((centred[:, order[i:]].max(axis=1) >= t[k]).mean()))
        p_stepm[k] = running

    t_spa = max(0.0, float(t[live].max())) if live.any() else 0.0
    p_spa = float((np.maximum(t_spa_boot.max(axis=1), 0.0) >= t_spa).mean()) if t_spa > 0 else 1.0
    return {"t": t, "p_single": p_single, "p_stepm": p_stepm, "p_spa": p_spa}


def multi_horizon_tests(diff: np.ndarray, *, block: int, n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
    """One criterion against the benchmark over H horizons (Quaedvlieg, 2021).

    ``diff`` is ``T x H`` (benchmark minus criterion loss, each horizon on a
    common scale). Returns one-sided p-values of the average test (the mean
    differential over horizons is positive) and of the uniform test (it is
    positive at every horizon). The uniform test is run as an intersection-union
    test: it rejects only if every horizon's own bootstrap test rejects, so its
    p-value is the largest of the per-horizon p-values. Bootstrapping the
    minimum statistic with every horizon recentred, as in Quaedvlieg (2021),
    over-rejects when only one horizon is at the null; Monschang, Trede and
    Wilfling (2026) show that per-horizon critical values are the right ones.
    """
    d = np.asarray(diff, dtype=float)
    res = benchmark_tests(np.column_stack([d.mean(axis=1), d]), block=block, n_boot=n_boot, seed=seed)
    p = res["p_single"]
    return float(p[0]), float(p[1:].max())


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
