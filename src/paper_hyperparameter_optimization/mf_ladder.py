"""Prior selection for the Schorfheide-Song mixed-frequency VAR.

The GLP ladder (:mod:`glp_hyperparameter_optimization.objectives`) relies on a
closed-form marginal likelihood. In the mixed-frequency VAR the quarterly
variables have latent monthly values, so the marginal likelihood of the observed
data is not available in closed form. Conditional on a draw of the latent
monthly values, however, the model is a conjugate monthly VAR with the
Sims-Zha dummy-observation prior of ``MBFVAR.mfbvar_funcs.varprior``: its
"complete-data" marginal likelihood and its pseudo-out-of-sample forecasts are
available in closed form.

We therefore run the Gibbs sampler once per forecast origin at the published
Schorfheide-Song hyperparameters, keep draws of the balanced monthly data, and
select the hyperparameters on those draws:

* ``ml``: one Monte Carlo EM step for the marginal likelihood, i.e. the
  hyperparameters that maximize the complete-data log marginal likelihood
  averaged over the draws of the latent monthly values;
* ``rmse``: GDP point-forecast RMSE over a short window of recent origins;
* ``rmsej``: point-forecast RMSE of all variables over a long window, each
  variable weighted by the inverse of its AR(1) residual variance.

The point criteria use the plug-in posterior mean on the balanced data averaged
over the draws; forecasts of the quarterly variables are the averages of the
monthly forecasts over the target quarter, matching the measurement equation.
Every criterion is optimized by multi-start Nelder-Mead from the ``ml``
selection (or the published values), so differences reflect the criterion.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import DEFAULT_PARAM_SPACE_BOUNDS, PAPER_HYPERPARAMETERS

MONTHS_PER_QUARTER = 3
HYPER_NAMES = ("lambda1_1", "lambda2_1", "lambda4_1", "lambda5_1")
PENALTY = 1e10


# --------------------------------------------------------------------------- #
# Hyperparameter vectors: MBFVAR uses [l1, l2, l3 (=1), l4, l5].
# --------------------------------------------------------------------------- #
def to_mbfvar(free: np.ndarray) -> list[float]:
    l1, l2, l4, l5 = (float(v) for v in free)
    return [l1, l2, 1.0, l4, l5]


def from_mbfvar(hyp: list[float]) -> np.ndarray:
    return np.array([hyp[0], hyp[1], hyp[3], hyp[4]], dtype=float)


# The Mango search box, with the constant (l4) and sum-of-coefficients (l5)
# dummy weights allowed up to 50: point criteria push both to the old bound of 10.
MF_BOUNDS = {**DEFAULT_PARAM_SPACE_BOUNDS, "lambda4_1": (0.01, 50.0), "lambda5_1": (0.01, 50.0)}
_BOUNDS = np.array([MF_BOUNDS[name] for name in HYPER_NAMES], dtype=float)


def _to_unbounded(free: np.ndarray) -> np.ndarray:
    lo, hi = np.log(_BOUNDS[:, 0]), np.log(_BOUNDS[:, 1])
    u = (np.log(np.clip(free, _BOUNDS[:, 0] * (1 + 1e-9), _BOUNDS[:, 1] * (1 - 1e-9))) - lo) / (hi - lo)
    return np.log(u / (1.0 - u))


def _to_bounded(z: np.ndarray) -> np.ndarray:
    lo, hi = np.log(_BOUNDS[:, 0]), np.log(_BOUNDS[:, 1])
    from scipy.special import expit

    return np.exp(lo + (hi - lo) * expit(np.asarray(z, dtype=float)))


# --------------------------------------------------------------------------- #
# Fast, exact replacement for MBFVAR's explosive-root check.
# --------------------------------------------------------------------------- #
# Grid of points just above one; det(x^p I - sum_i x^(p-i) A_i) is the
# characteristic polynomial of the companion matrix at x.
_ROOT_GRID = 1.0 + np.concatenate(([1e-7], np.geomspace(1e-5, 1.0, 40)))


def fast_is_explosive(Phi: np.ndarray, n: int, p: int) -> bool:
    """Same decision as ``MBFVAR.mfbvar_funcs.is_explosive`` (any companion
    eigenvalue with modulus above one), several times faster.

    The characteristic polynomial ``f(x) = det(x^p I - x^(p-1) A_1 - ... - A_p)``
    is evaluated on a grid of points above one and at infinity, where its sign
    is positive. A sign change certifies a real root above one, so the draw is
    explosive; this settles most candidate draws of a VAR in levels, whose
    offending root is a real root slightly above one. Only draws without a sign
    change get the full eigenvalue computation, so the decision is unchanged.
    """
    A = np.asarray(Phi[: n * p, :], dtype=float).reshape(p, n, n).transpose(0, 2, 1)
    powers = _ROOT_GRID[:, None] ** np.arange(p - 1, -1, -1)[None, :]  # x^(p-i), i = 1..p
    M = (_ROOT_GRID**p)[:, None, None] * np.eye(n) - np.einsum("gi,iab->gab", powers, A)
    signs = np.append(np.sign(np.linalg.det(M)), 1.0)
    if np.any(signs[:-1] * signs[1:] < 0):
        return True
    import scipy.linalg

    companion = np.zeros((n * p, n * p))
    companion[:n, :] = Phi[: n * p, :].T
    if p > 1:
        companion[n:, :-n] = np.eye(n * (p - 1))
    return bool(np.any(np.abs(scipy.linalg.eigvals(companion, check_finite=False, overwrite_a=True)) > 1))


@contextlib.contextmanager
def fast_explosive_check():
    """Use :func:`fast_is_explosive` inside MBFVAR's Gibbs sampler."""
    import MBFVAR._estimation as est

    original = est.is_explosive
    est.is_explosive = fast_is_explosive
    try:
        yield
    finally:
        est.is_explosive = original


# --------------------------------------------------------------------------- #
# Complete (balanced) monthly data captured from the Gibbs sampler.
# --------------------------------------------------------------------------- #
@dataclass
class CompleteData:
    """Draws of the balanced monthly data in model units (logs, rates / 100).

    ``draws`` has shape (K, T, n); ``spec`` is MBFVAR's (nlags, T0, nex, nv, nobs)
    with the first ``T0`` rows a pre-sample; ``dates`` are the month-ends of the
    T rows; ``n_observed`` is the number of leading rows up to and including the
    last month of the last observed quarter (later rows are nowcasts).
    """

    draws: np.ndarray
    spec: np.ndarray
    dates: pd.DatetimeIndex
    columns: list[str]
    n_observed: int

    @property
    def mean(self) -> np.ndarray:
        return self.draws.mean(axis=0)


def fit_capturing_complete_data(
    model: Any,
    data_in: Any,
    hyp: list[float],
    *,
    keep: int = 50,
    last_observed_quarter: pd.Timestamp,
    **fit_kwargs: Any,
) -> CompleteData:
    """Run ``model.fit`` and record the balanced monthly data of the post-burn-in
    iterations (thinned to ``keep`` draws). MBFVAR builds that dataset at every
    iteration and passes it to ``calc_yyact`` (or ``mdd_``), which we wrap."""
    import MBFVAR._estimation as est

    records: list[tuple[np.ndarray, np.ndarray]] = []
    original = {"calc_yyact": est.calc_yyact, "mdd_": est.mdd_}

    def recorder(name: str) -> Callable[..., Any]:
        def wrapped(hyp_, YY, spec):
            records.append((np.array(YY, dtype=float), np.array(spec)))
            return original[name](hyp_, YY, spec)

        return wrapped

    est.calc_yyact, est.mdd_ = recorder("calc_yyact"), recorder("mdd_")
    try:
        with fast_explosive_check(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            model.fit(data_in, [list(hyp)], **fit_kwargs)
    finally:
        est.calc_yyact, est.mdd_ = original["calc_yyact"], original["mdd_"]
    if not records:
        raise RuntimeError("MBFVAR did not expose the balanced monthly data.")

    post = records[len(records) // 2 :]
    picks = np.unique(np.linspace(0, len(post) - 1, num=min(keep, len(post))).round().astype(int))
    draws = np.stack([post[i][0] for i in picks])
    spec = post[-1][1]
    dates = pd.DatetimeIndex(model.index_list[-1])[-draws.shape[1] :]
    columns = [str(c) for c in model.varlist_list[-1]]
    n_observed = int((dates <= pd.Timestamp(last_observed_quarter) + pd.offsets.MonthEnd(0)).sum())
    return CompleteData(draws=draws, spec=spec, dates=dates, columns=columns, n_observed=n_observed)


# --------------------------------------------------------------------------- #
# Criteria.
# --------------------------------------------------------------------------- #
def complete_data_log_ml(free: np.ndarray, data: CompleteData) -> float:
    """Complete-data log marginal likelihood averaged over the latent draws."""
    from MBFVAR.mfbvar_funcs import mdd_

    hyp = to_mbfvar(free)
    return float(np.mean([mdd_(hyp, YY, data.spec)[0] for YY in data.draws]))


class CompleteDataVAR:
    """Conjugate monthly VAR on one balanced dataset with prefix sums of the
    sufficient statistics, so the posterior mean for any training sample
    ``Y[:cut]`` costs one small linear solve."""

    def __init__(self, Y: np.ndarray, spec: np.ndarray):
        self.Y = np.asarray(Y, dtype=float)
        self.nlags, self.T0, self.nex, self.n = (int(v) for v in spec[:4])
        T = self.Y.shape[0]
        pre = self.Y[: self.T0 + 16]
        self.premom = np.column_stack((pre.mean(axis=0), pre.std(axis=0, ddof=1)))
        rows = range(self.T0, T)
        X = np.array([self._regressors(t) for t in rows])
        Yact = self.Y[self.T0 :]
        k = X.shape[1]
        self.XtX = np.zeros((T + 1, k, k))
        self.XtY = np.zeros((T + 1, k, self.n))
        self.XtX[self.T0 + 1 :] = np.cumsum(X[:, :, None] * X[:, None, :], axis=0)
        self.XtY[self.T0 + 1 :] = np.cumsum(X[:, :, None] * Yact[:, None, :], axis=0)

    def _regressors(self, t: int) -> np.ndarray:
        lags = [self.Y[t - 1 - i] for i in range(self.nlags)]
        return np.concatenate(lags + [np.ones(self.nex)])

    def posterior_mean(self, free: np.ndarray, cut: int) -> np.ndarray:
        from MBFVAR.mfbvar_funcs import varprior

        Yd, Xd = varprior(self.n, self.nlags, self.nex, to_mbfvar(free), self.premom)
        return np.linalg.solve(Xd.T @ Xd + self.XtX[cut], Xd.T @ Yd + self.XtY[cut])

    def forecast(self, phi: np.ndarray, cut: int, months: int) -> np.ndarray:
        hist = list(self.Y[cut - self.nlags : cut])
        path = []
        for _ in range(months):
            x = np.concatenate([hist[-1 - i] for i in range(self.nlags)] + [np.ones(self.nex)])
            y = x @ phi
            path.append(y)
            hist.append(y)
        return np.array(path)


def quarter_end_cuts(data: CompleteData, horizon_q: int, window: int, min_rows: int = 120) -> list[int]:
    """Training-sample ends ``cut`` (rows ``Y[:cut]``) at quarter-end months
    whose ``horizon_q``-quarter-ahead target quarter is fully observed, most
    recent first, limited to ``window`` origins."""
    months = 3 * horizon_q
    cuts = [
        r + 1
        for r in range(data.n_observed - 1, -1, -1)
        if data.dates[r].month % 3 == 0 and r + 1 + months <= data.n_observed and r + 1 >= min_rows
    ]
    return cuts[:window]


def quarterly_ar1_variance(data: CompleteData) -> np.ndarray:
    """Residual variance of an AR(1) on each quarterly-averaged series."""
    Y = data.mean[: data.n_observed]
    ends = [r for r in range(2, Y.shape[0]) if data.dates[r].month % 3 == 0]
    Q = np.array([Y[r - 2 : r + 1].mean(axis=0) for r in ends])
    out = np.empty(Q.shape[1])
    for i in range(Q.shape[1]):
        x, y = np.column_stack((np.ones(len(Q) - 1), Q[:-1, i])), Q[1:, i]
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        out[i] = np.var(y - x @ beta, ddof=2)
    return out


def point_loss(
    var: CompleteDataVAR,
    cuts: list[int],
    horizon_q: int,
    columns: list[int],
    weights: np.ndarray,
) -> Callable[[np.ndarray], float]:
    """RMSE of the plug-in quarterly forecast ``horizon_q`` quarters ahead."""
    months = 3 * horizon_q
    actual = np.array([var.Y[c + months - 3 : c + months].mean(axis=0)[columns] for c in cuts])

    def loss(free: np.ndarray) -> float:
        sq = []
        for c, act in zip(cuts, actual):
            path = var.forecast(var.posterior_mean(free, c), c, months)
            err = path[-3:].mean(axis=0)[columns] - act
            sq.append(weights * err**2)
        return float(np.sqrt(np.mean(sq)))

    return loss


# --------------------------------------------------------------------------- #
# Optimizer: multi-start Nelder-Mead in the bounded-log space.
# --------------------------------------------------------------------------- #
def minimize_criterion(
    loss: Callable[[np.ndarray], float],
    start: np.ndarray,
    *,
    n_starts: int = 4,
    start_scale: float = 1.0,
    maxfev: int = 400,
    seed: int = 0,
) -> dict[str, Any]:
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    z0 = _to_unbounded(np.asarray(start, dtype=float))

    def f(z: np.ndarray) -> float:
        try:
            value = float(loss(_to_bounded(z)))
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return PENALTY
        return value if np.isfinite(value) else PENALTY

    starts = [z0] + [z0 + rng.normal(0.0, start_scale, z0.size) for _ in range(max(0, n_starts - 1))]
    runs = []
    for s in starts:
        res = minimize(f, s, method="Nelder-Mead", options={"maxfev": maxfev, "xatol": 1e-3, "fatol": 1e-6})
        runs.append((float(res.fun), res.x, int(res.nfev)))
    best = min(runs, key=lambda r: r[0])
    return {"free": _to_bounded(best[1]), "value": best[0], "value_at_start": f(z0), "nfev": sum(r[2] for r in runs)}


# --------------------------------------------------------------------------- #
# Selection at one origin.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MFCriterion:
    name: str  # ml | rmse | rmsej
    horizon_q: int = 4
    window: int = 12

    def label(self) -> str:
        return "ml" if self.name == "ml" else f"{self.name}_h{self.horizon_q}_w{self.window}"


DEFAULT_CRITERIA = (MFCriterion("ml"), MFCriterion("rmse", 4, 12), MFCriterion("rmsej", 4, 60))


def select_all(
    data: CompleteData,
    criteria: tuple[MFCriterion, ...] = DEFAULT_CRITERIA,
    *,
    n_starts: int = 4,
    maxfev: int = 400,
    seed: int = 0,
    target: str = "GDP",
) -> dict[str, dict[str, Any]]:
    """Select hyperparameters under every criterion from one set of draws."""
    paper = from_mbfvar(PAPER_HYPERPARAMETERS)
    ml = minimize_criterion(lambda v: -complete_data_log_ml(v, data), paper, n_starts=1, maxfev=maxfev, seed=seed)
    ml_free = ml["free"]
    out: dict[str, dict[str, Any]] = {}
    var = CompleteDataVAR(data.mean, data.spec)
    ar1 = quarterly_ar1_variance(data)
    for crit in criteria:
        if crit.name == "ml":
            res = {"free": ml_free, "value": -ml["value"], "value_at_start": -ml["value_at_start"], "nfev": ml["nfev"]}
        else:
            cuts = quarter_end_cuts(data, crit.horizon_q, crit.window)
            if crit.name == "rmsej":
                cols, weights = list(range(len(data.columns))), 1.0 / ar1
            else:
                cols, weights = [data.columns.index(target)], np.ones(1)
            loss = point_loss(var, cuts, crit.horizon_q, cols, weights)
            res = minimize_criterion(loss, ml_free, n_starts=n_starts, maxfev=maxfev, seed=seed)
            res["value_at_paper"] = loss(paper)
            res["n_cuts"] = len(cuts)
        res["complete_data_log_ml"] = complete_data_log_ml(res["free"], data)
        out[crit.label()] = res
    out["_paper_complete_data_log_ml"] = {"value": complete_data_log_ml(paper, data)}
    return out


# --------------------------------------------------------------------------- #
# Forecasts with per-draw quarterly metrics (for CRPS and the log score).
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _capture_quarterly_draws(sink: list[pd.DataFrame]):
    """``MBFVAR.aggregate`` builds one quarterly frame per posterior draw and
    only keeps their quantiles; record the first list it concatenates."""
    import MBFVAR._estimation as est

    real_pd = est.pd

    class _Proxy:
        def __getattr__(self, name: str) -> Any:
            return getattr(real_pd, name)

        def concat(self, objs: Any, *args: Any, **kwargs: Any) -> Any:
            if not sink and not isinstance(objs, (pd.DataFrame, pd.Series)):
                items = list(objs)
                if items and all(isinstance(o, pd.DataFrame) for o in items):
                    sink.extend(items)
                return real_pd.concat(items, *args, **kwargs)
            return real_pd.concat(objs, *args, **kwargs)

    est.pd = _Proxy()
    try:
        yield
    finally:
        est.pd = real_pd


def realtime_quarterly_history(quarterly: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    """Quarterly levels as published in the real-time vintage: the quarterly
    series as observed and the monthly series averaged over complete quarters."""
    q = quarterly.copy()
    q.index = pd.DatetimeIndex(q.index).to_period("Q")
    periods = pd.DatetimeIndex(monthly.index).to_period("Q")
    counts = monthly.groupby(periods).count()
    m = monthly.groupby(periods).mean().where(counts == MONTHS_PER_QUARTER)
    return q.join(m, how="outer")


def forecast_rows(
    label: str,
    origin_date: pd.Timestamp,
    model: Any,
    actual_levels: pd.DataFrame,
    history: pd.DataFrame,
    horizon_months: int,
) -> list[dict[str, Any]]:
    """Quarterly forecasts scored as in the real-time literature: every draw's
    quarterly path is anchored on the published real-time history, so growth
    rates are measured from the jump-off data the forecaster actually had, and
    the point forecast and quantiles are those of the per-draw metrics."""
    from glp_hyperparameter_optimization.scoring import crps_sample, log_score_gaussian

    from .config import MAX_FORECAST_HORIZON_QUARTERS, origin_group
    from .forecasting import compute_quarterly_metrics

    draws: list[pd.DataFrame] = []
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model.forecast(horizon_months)
        with _capture_quarterly_draws(draws):
            model.aggregate(frequency="Q")
    if not draws:
        raise RuntimeError("Could not capture the per-draw quarterly forecasts from MBFVAR.aggregate.")

    columns = list(draws[0].columns)
    periods = pd.DatetimeIndex(draws[0].index).to_period("Q")
    hist = history.reindex(index=periods, columns=columns)
    known = hist.notna().to_numpy()
    stacked = np.stack([np.where(known, hist.to_numpy(), d.to_numpy()) for d in draws])
    metrics = np.stack([compute_quarterly_metrics(pd.DataFrame(s, index=periods, columns=columns)).to_numpy() for s in stacked])
    actual = compute_quarterly_metrics(actual_levels)
    position = {p: i for i, p in enumerate(periods)}

    rows: list[dict[str, Any]] = []
    current = pd.Timestamp(origin_date).to_period("Q")
    for h in range(1, MAX_FORECAST_HORIZON_QUARTERS + 1):
        target = current + (h - 1)
        if target not in position or target not in actual.index:
            continue
        for j, variable in enumerate(columns):
            sample = metrics[:, position[target], j]
            sample = sample[np.isfinite(sample)]
            obs = float(actual.at[target, variable])
            if sample.size < 2 or not np.isfinite(obs):
                continue
            mean, std = float(sample.mean()), float(sample.std(ddof=1))
            q05, q16, q50, q84, q95 = np.quantile(sample, [0.05, 0.16, 0.5, 0.84, 0.95])
            rows.append(
                {
                    "strategy": label,
                    "forecast_origin": pd.Timestamp(origin_date).strftime("%Y-%m-%d"),
                    "group": origin_group(pd.Timestamp(origin_date)),
                    "target_quarter": str(target),
                    "horizon_quarters": h,
                    "variable": variable,
                    "actual_metric": obs,
                    "mean_metric": mean,
                    "median_metric": q50,
                    "p05_metric": q05,
                    "p16_metric": q16,
                    "p84_metric": q84,
                    "p95_metric": q95,
                    "std": std,
                    "error_metric": mean - obs,
                    "crps": float(crps_sample(sample[:, None], np.array([obs]))[0]),
                    "logscore": float(log_score_gaussian(mean, max(std, 1e-8), obs)),
                    "n_draws": int(sample.size),
                }
            )
    return rows


# --------------------------------------------------------------------------- #
# One forecast origin: capture, select, forecast under every criterion.
# --------------------------------------------------------------------------- #
def run_origin(task: dict[str, Any]) -> dict[str, Any]:
    import MBFVAR

    from .data_utils import build_model_input_frames, build_quarterly_evaluation_frame, load_realtime_panel
    from .forecasting import make_data_in
    from pathlib import Path

    origin = pd.Timestamp(task["origin_date"])
    panel = load_realtime_panel(Path(task["panel_path"]))
    actual = build_quarterly_evaluation_frame(panel, pd.Timestamp(task["actual_vintage"]))
    quarterly, monthly = build_model_input_frames(panel, origin)
    data_in = make_data_in(quarterly, monthly)
    history = realtime_quarterly_history(quarterly, monthly)
    seed = int(task["seed_base"]) + origin.year * 12 + origin.month
    fit_kwargs = {"temp_agg": task["temp_agg"], "check_explosive": task["check_explosive"]}

    def new_model() -> Any:
        np.random.seed(seed)  # common random numbers across criteria
        return MBFVAR.MixedFrequencyBVAR(task["fit_nsim"], task["nburn_perc"], task["nlags"], task["thining"])

    model = new_model()
    data = fit_capturing_complete_data(
        model, data_in, PAPER_HYPERPARAMETERS, keep=task["keep_draws"], last_observed_quarter=quarterly.index[-1], **fit_kwargs
    )
    rows = forecast_rows("paper", origin, model, actual, history, task["forecast_horizon_months"])
    base_record = {"forecast_origin": origin.strftime("%Y-%m-%d"), "n_complete_draws": int(data.draws.shape[0])}
    records = [{**base_record, "criterion": "paper", **dict(zip(HYPER_NAMES, from_mbfvar(PAPER_HYPERPARAMETERS)))}]

    criteria = tuple(MFCriterion(**c) for c in task["criteria"])
    selections = select_all(data, criteria, n_starts=task["n_starts"], maxfev=task["maxfev"], seed=seed)
    records[0]["complete_data_log_ml"] = selections.pop("_paper_complete_data_log_ml")["value"]
    for label, res in selections.items():
        model = new_model()
        with fast_explosive_check(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            model.fit(data_in, [to_mbfvar(res["free"])], **fit_kwargs)
        rows += forecast_rows(label, origin, model, actual, history, task["forecast_horizon_months"])
        records.append(
            {
                **base_record,
                "criterion": label,
                **dict(zip(HYPER_NAMES, res["free"])),
                **{k: res[k] for k in ("value", "value_at_start", "value_at_paper", "complete_data_log_ml", "nfev", "n_cuts") if k in res},
            }
        )
    return {"forecast_rows": rows, "hyperparameters": records}


# --------------------------------------------------------------------------- #
# Recursive experiment with per-origin checkpoints.
# --------------------------------------------------------------------------- #
def _run_origin_safe(task: dict[str, Any]) -> dict[str, Any]:
    try:
        return {**run_origin(task), "error": None}
    except Exception as exc:  # pragma: no cover - reported per origin
        return {"forecast_rows": [], "hyperparameters": [], "error": f"{type(exc).__name__}: {exc}"}


def run_mf_ladder(
    output_dir: "Path",
    *,
    origins: pd.DatetimeIndex,
    panel_path: "Path",
    actual_vintage: pd.Timestamp,
    criteria: tuple[MFCriterion, ...] = DEFAULT_CRITERIA,
    fit_nsim: int = 4000,
    nburn_perc: float = 0.5,
    nlags: list[int] | None = None,
    thining: int = 1,
    temp_agg: str = "mean",
    check_explosive: bool = True,
    keep_draws: int = 50,
    n_starts: int = 4,
    maxfev: int = 400,
    seed_base: int = 2015,
    forecast_horizon_months: int = 24,
    n_workers: int = 1,
) -> None:
    import hashlib
    import json
    import pickle
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from dataclasses import asdict
    from pathlib import Path

    from .config import PAPER_NLAGS

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "criteria": [asdict(c) for c in criteria],
        "fit_nsim": fit_nsim,
        "nburn_perc": nburn_perc,
        "nlags": list(nlags or PAPER_NLAGS),
        "thining": thining,
        "temp_agg": temp_agg,
        "check_explosive": check_explosive,
        "keep_draws": keep_draws,
        "n_starts": n_starts,
        "maxfev": maxfev,
        "seed_base": seed_base,
        "forecast_horizon_months": forecast_horizon_months,
        "actual_vintage": str(pd.Timestamp(actual_vintage).date()),
    }
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    ckpt_dir = output_dir / "_checkpoints" / key
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_metadata.json").write_text(json.dumps({**config, "checkpoint_key": key, "n_origins": len(origins)}, indent=2))

    tasks = [{**config, "origin_date": str(o.date()), "panel_path": str(panel_path)} for o in origins]
    done = {p.stem: p for p in ckpt_dir.glob("*.pkl")}
    pending = [t for t in tasks if t["origin_date"] not in done]
    print(f"[mf_ladder] {len(tasks)} origins, {len(tasks) - len(pending)} from checkpoints, {len(pending)} to run with {n_workers} worker(s).", flush=True)

    def save(task: dict[str, Any], result: dict[str, Any]) -> None:
        if result["error"] is None:
            tmp = ckpt_dir / f"{task['origin_date']}.tmp"
            with tmp.open("wb") as handle:
                pickle.dump(result, handle)
            tmp.replace(ckpt_dir / f"{task['origin_date']}.pkl")
        else:
            print(f"[mf_ladder] {task['origin_date']} failed: {result['error']}", flush=True)

    if n_workers <= 1:
        for i, task in enumerate(pending, 1):
            save(task, _run_origin_safe(task))
            print(f"[mf_ladder] {i}/{len(pending)} {task['origin_date']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_run_origin_safe, t): t for t in pending}
            for i, fut in enumerate(as_completed(futures), 1):
                save(futures[fut], fut.result())
                print(f"[mf_ladder] {i}/{len(pending)} {futures[fut]['origin_date']}", flush=True)

    results = []
    for task in tasks:
        path = ckpt_dir / f"{task['origin_date']}.pkl"
        if path.exists():
            with path.open("rb") as handle:
                results.append(pickle.load(handle))
    rows = [r for res in results for r in res["forecast_rows"]]
    records = [r for res in results for r in res["hyperparameters"]]
    pd.DataFrame(rows).to_csv(output_dir / "forecast_panel.csv", index=False)
    pd.DataFrame(records).to_csv(output_dir / "selected_hyperparameters.csv", index=False)
    missing = len(tasks) - len(results)
    print(f"[mf_ladder] wrote {len(rows)} forecast rows for {len(results)} origins ({missing} missing) to {output_dir}", flush=True)
