"""GLP (2015) BVAR driver and the three Mango hyperparameter optimizers.

The public ``covbayesvar.large_bvar.bvarGLP`` entry point crashes under numpy>=2
(it assigns a 2-D array into a scalar slot while building the prior mean), so
this module drives the estimation with the lower-level, numerically-correct
workhorses that *do* work:

* ``logMLVAR_formin``  -- log marginal likelihood + posterior-mode beta/sigma
  for a given hyperparameter vector (used for the point forecasts and, via
  ``scipy.optimize``, for GLP's own marginal-likelihood mode).
* ``logMLVAR_formcmc`` -- log posterior (``draw=0``) and posterior draws of
  beta/sigma (``draw=1``) for a given hyperparameter vector (used for the
  Mango MDD objective and for the full MCMC predictive densities).
* ``bvarFcst``         -- recursive point forecasts.

The optimized hyperparameters are GLP's three core quantities: ``lambda``
(overall Minnesota tightness), ``theta`` (sum-of-coefficients tightness) and
``miu`` (dummy-initial-observation tightness). ``psi`` (residual variances) and
``alpha`` (lag decay) are held fixed (``MNpsi=0``, ``MNalpha=0``), matching the
three-hyperparameter search of the paper.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .config import (
    GLP_HYPERPRIORS,
    GLP_MNALPHA,
    GLP_MNPSI,
    GLP_NOC,
    GLP_PARAM_SPACE_BOUNDS,
    GLP_SUR,
    GLP_VC,
)

# covbayesvar's MCMC draw path triggers a benign ComplexWarning from a complex
# eigendecomposition that is immediately cast back to real.
try:  # numpy>=2
    _ComplexWarning = np.exceptions.ComplexWarning
except AttributeError:  # pragma: no cover - older numpy
    _ComplexWarning = np.ComplexWarning  # type: ignore[attr-defined]
warnings.filterwarnings("ignore", category=_ComplexWarning)


def _bvar():
    """Lazily import ``covbayesvar.large_bvar`` (keeps the module import cheap
    and multiprocessing-friendly)."""
    import covbayesvar.large_bvar as bvar

    return bvar


# --------------------------------------------------------------------------- #
# Estimation context (hyperparameter-independent) and its construction.
# --------------------------------------------------------------------------- #
@dataclass
class GLPContext:
    """All hyperparameter-independent inputs required by the GLP workhorses."""

    y: np.ndarray
    x: np.ndarray
    lags: int
    T: int
    n: int
    k: int
    b: np.ndarray
    MIN: dict[str, Any]
    MAX: dict[str, Any]
    SS: np.ndarray
    Vc: float
    pos: list
    mn: dict[str, Any]
    sur: int
    noc: int
    y0: np.ndarray
    hyperpriors: int
    priorcoef: dict[str, Any]
    MCMCMsur: int
    long_run: np.ndarray


def prepare_glp_context(
    y: np.ndarray,
    lags: int,
    *,
    hyperpriors: int = GLP_HYPERPRIORS,
    sur: int = GLP_SUR,
    noc: int = GLP_NOC,
    mnpsi: int = GLP_MNPSI,
    mnalpha: int = GLP_MNALPHA,
    vc: float = GLP_VC,
    pos: Sequence[int] | None = None,
) -> GLPContext:
    """Replicate the ``bvarGLP`` preamble (regressors, prior mean, AR(1) scales)
    with the numpy>=2 fix for the prior-mean assignment."""
    bvar = _bvar()
    y = np.asarray(y, dtype=float)

    (
        _r,
        _mode,
        _sd,
        priorcoef,
        MIN,
        MAX,
        hyperpriors,
        Vc,
        pos_default,
        mn,
        _MNalpha,
        sur,
        noc,
        _Fcast,
        _hz,
        _mcmc,
        _M,
        _N,
        _const,
        _MCMCfcast,
        _MCMCstorecoeff,
        MCMCMsur,
        long_run,
    ) = bvar.set_priors(
        y,
        lags,
        hyperpriors=hyperpriors,
        mcmc=0,
        MNpsi=mnpsi,
        MNalpha=mnalpha,
        sur=sur,
        noc=noc,
        Vc=vc,
    )
    pos = list(pos) if pos is not None else list(pos_default or [])

    TT, n = y.shape
    k = n * lags + 1

    # Matrix of regressors: intercept + stacked lags.
    x = np.zeros((TT, k))
    x[:, 0] = 1.0
    for i in range(1, lags + 1):
        x[:, 1 + (i - 1) * n : 1 + i * n] = bvar.lag(y, i)

    y0 = np.mean(y[:lags, :], axis=0)
    x = x[lags:, :]
    y_trim = y[lags:, :]
    T = y_trim.shape[0]

    # Minnesota prior mean: unit coefficient on each own first lag (numpy>=2 fix
    # -- assign a plain float rather than a 1-element array).
    b = np.zeros((k, n))
    diagb = np.ones(n)
    for p in pos:
        diagb[int(p)] = 0.0
    for i in range(n):
        b[i + 1, i] = float(diagb[i]) if i < len(diagb) else 1.0

    # AR(1) residual variances (scale of the prior on the shock covariance).
    SS = np.zeros((n, 1))
    for i in range(n):
        X = np.concatenate([np.ones((T, 1)), x[:, i + 1].reshape(-1, 1)], axis=1)
        ar1 = bvar.ols1(y_trim[:, i].reshape(-1, 1), X)
        SS[i, 0] = float(np.ravel(ar1["sig2hatols"])[0])
    MIN["psi"] = SS / 100.0
    MAX["psi"] = SS * 100.0

    return GLPContext(
        y=y_trim,
        x=x,
        lags=lags,
        T=T,
        n=n,
        k=k,
        b=b,
        MIN=MIN,
        MAX=MAX,
        SS=SS,
        Vc=float(Vc),
        pos=pos,
        mn=mn,
        sur=int(sur),
        noc=int(noc),
        y0=y0,
        hyperpriors=int(hyperpriors),
        priorcoef=priorcoef,
        MCMCMsur=int(MCMCMsur),
        long_run=np.asarray(long_run, dtype=float),
    )


# --------------------------------------------------------------------------- #
# Hyperparameter <-> unconstrained transform ([lambda, theta, miu]).
# --------------------------------------------------------------------------- #
def _bounds_from_context(ctx: GLPContext) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([ctx.MIN["lambda"], ctx.MIN["theta"], ctx.MIN["miu"]], dtype=float)
    hi = np.array([ctx.MAX["lambda"], ctx.MAX["theta"], ctx.MAX["miu"]], dtype=float)
    return lo, hi


def to_transformed(natural: Sequence[float], ctx: GLPContext) -> np.ndarray:
    """Map natural ``[lambda, theta, miu]`` to the unconstrained real line."""
    lo, hi = _bounds_from_context(ctx)
    nat = np.clip(np.asarray(natural, dtype=float), lo + 1e-12, hi - 1e-12)
    return -np.log((hi - nat) / (nat - lo))


def to_natural(transformed: Sequence[float], ctx: GLPContext) -> np.ndarray:
    """Inverse of :func:`to_transformed`."""
    lo, hi = _bounds_from_context(ctx)
    t = np.asarray(transformed, dtype=float)
    return lo + (hi - lo) / (1.0 + np.exp(-t))


# --------------------------------------------------------------------------- #
# Objective evaluations built on the covbayesvar workhorses.
# --------------------------------------------------------------------------- #
def _formin(ctx: GLPContext, transformed_par: np.ndarray):
    bvar = _bvar()
    return bvar.logMLVAR_formin(
        np.asarray(transformed_par, dtype=float).ravel(),
        ctx.y,
        ctx.x,
        ctx.lags,
        ctx.T,
        ctx.n,
        ctx.b,
        ctx.MIN,
        ctx.MAX,
        ctx.SS,
        ctx.Vc,
        ctx.pos,
        ctx.mn,
        ctx.sur,
        ctx.noc,
        ctx.y0,
        ctx.hyperpriors,
        ctx.priorcoef,
        ctx.MCMCMsur,
        ctx.long_run,
    )


def _formcmc(ctx: GLPContext, natural_par: np.ndarray, draw: int):
    bvar = _bvar()
    return bvar.logMLVAR_formcmc(
        np.asarray(natural_par, dtype=float).ravel(),
        ctx.y,
        ctx.x,
        ctx.lags,
        ctx.T,
        ctx.n,
        ctx.b,
        ctx.MIN,
        ctx.MAX,
        ctx.SS,
        ctx.Vc,
        ctx.pos,
        ctx.mn,
        ctx.sur,
        ctx.noc,
        ctx.y0,
        draw,
        ctx.hyperpriors,
        ctx.priorcoef,
        ctx.MCMCMsur,
        ctx.long_run,
    )


def glp_neg_logposterior_transformed(transformed_par: np.ndarray, ctx: GLPContext) -> float:
    """Negative log posterior as a function of the unconstrained hyperparameters
    (the objective minimized to recover GLP's marginal-likelihood mode)."""
    value = _formin(ctx, transformed_par)[0]
    return float(np.ravel(value)[0])


def glp_logposterior(ctx: GLPContext, lam: float, theta: float, miu: float) -> float:
    """Log posterior (or pure log marginal likelihood when ``hyperpriors=0``)
    at natural hyperparameters -- the Mango MDD objective."""
    logml, _, _ = _formcmc(ctx, np.array([lam, theta, miu]), draw=0)
    return float(np.ravel(logml)[0])


def glp_mode_estimate(ctx: GLPContext, lam: float, theta: float, miu: float) -> tuple[np.ndarray, np.ndarray]:
    """Posterior-mode ``(betahat, sigmahat)`` for the given hyperparameters."""
    transformed = to_transformed([lam, theta, miu], ctx)
    _, betahat, sigmahat = _formin(ctx, transformed)
    return np.asarray(betahat, dtype=float), np.asarray(sigmahat, dtype=float)


def glp_draw(ctx: GLPContext, lam: float, theta: float, miu: float) -> tuple[np.ndarray, np.ndarray]:
    """A single posterior draw of ``(beta, sigma)`` at the given hyperparameters."""
    _, betadraw, drawsigma = _formcmc(ctx, np.array([lam, theta, miu]), draw=1)
    return np.asarray(betadraw, dtype=float), np.asarray(drawsigma, dtype=float)


# --------------------------------------------------------------------------- #
# Forecasting helpers.
# --------------------------------------------------------------------------- #
def point_forecast(y: np.ndarray, beta: np.ndarray, horizons: Sequence[int]) -> np.ndarray:
    """Recursive point forecast at the requested horizons (rows = horizons)."""
    bvar = _bvar()
    return np.asarray(bvar.bvarFcst(np.asarray(y, dtype=float), np.asarray(beta, dtype=float), list(horizons)), dtype=float)


def simulate_forecast_path(
    y: np.ndarray,
    beta: np.ndarray,
    sigma: np.ndarray,
    lags: int,
    max_horizon: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate one predictive path ``H`` quarters ahead by iterating the VAR
    with Gaussian innovations ``N(0, sigma)`` (mirrors bvarGLP's MCMC forecast
    block). Returns an ``(max_horizon, n)`` array of simulated levels."""
    y = np.asarray(y, dtype=float)
    beta = np.asarray(beta, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    T, n = y.shape
    k = beta.shape[0]
    Y = np.vstack([y, np.zeros((max_horizon, n))])
    for tau in range(1, max_horizon + 1):
        selected = Y[T + tau - 2 : T + tau - lags - 2 : -1, :].T
        xT = np.concatenate(([1.0], selected.flatten(order="F")))[:k]
        innovation = rng.multivariate_normal(np.zeros(n), sigma)
        Y[T + tau - 1, :] = xT @ beta + innovation
    return Y[T:, :]


# --------------------------------------------------------------------------- #
# GLP marginal-likelihood mode (the paper's own selection) via scipy.
# --------------------------------------------------------------------------- #
def glp_find_mode(
    ctx: GLPContext,
    *,
    start_natural: Sequence[float] | None = None,
    method: str = "Nelder-Mead",
    maxiter: int = 2000,
) -> dict[str, Any]:
    """Maximize the GLP (log) posterior over ``[lambda, theta, miu]`` and return
    the mode hyperparameters together with the posterior-mode beta/sigma."""
    from scipy.optimize import minimize

    if start_natural is None:
        start_natural = [0.2, 1.0, 1.0]
    x0 = to_transformed(start_natural, ctx)
    result = minimize(
        glp_neg_logposterior_transformed,
        x0,
        args=(ctx,),
        method=method,
        options={"maxiter": maxiter, "xatol": 1e-6, "fatol": 1e-8} if method == "Nelder-Mead" else {"maxiter": maxiter},
    )
    natural = to_natural(result.x, ctx)
    _, betahat, sigmahat = _formin(ctx, result.x)
    return {
        "lambda": float(natural[0]),
        "theta": float(natural[1]),
        "miu": float(natural[2]),
        "log_posterior": float(-result.fun),
        "betahat": np.asarray(betahat, dtype=float),
        "sigmahat": np.asarray(sigmahat, dtype=float),
        "transformed": np.asarray(result.x, dtype=float),
    }


# --------------------------------------------------------------------------- #
# Random-walk Metropolis over the hyperparameters (paper predictive density).
# --------------------------------------------------------------------------- #
def _numerical_hessian(fun: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = x.size
    hess = np.zeros((n, n))
    f0 = fun(x)
    steps = np.maximum(np.abs(x), 1.0) * eps
    for i in range(n):
        for j in range(i, n):
            xi = x.copy()
            xj = x.copy()
            xij = x.copy()
            xi[i] += steps[i]
            xj[j] += steps[j]
            xij[i] += steps[i]
            xij[j] += steps[j]
            value = (fun(xij) - fun(xi) - fun(xj) + f0) / (steps[i] * steps[j])
            hess[i, j] = value
            hess[j, i] = value
    return hess


def _proposal_covariance(ctx: GLPContext, mode_natural: np.ndarray) -> np.ndarray:
    """Inverse observed-information at the mode, regularized to be PD."""

    def neg_logpost(nat: np.ndarray) -> float:
        return -glp_logposterior(ctx, float(nat[0]), float(nat[1]), float(nat[2]))

    hess = _numerical_hessian(neg_logpost, mode_natural)
    try:
        cov = np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        cov = np.eye(mode_natural.size)
    cov = 0.5 * (cov + cov.T)
    eig, vec = np.linalg.eigh(cov)
    eig = np.abs(eig)
    eig[eig < 1e-10] = 1e-10
    return vec @ np.diag(eig) @ vec.T


def glp_metropolis_forecast_draws(
    ctx: GLPContext,
    mode_natural: Sequence[float],
    *,
    max_horizon: int,
    n_draws: int,
    n_discard: int,
    const: float = 1.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Random-walk Metropolis over ``[lambda, theta, miu]`` producing predictive
    forecast draws that integrate over hyperparameter uncertainty (the GLP
    hierarchical predictive density).

    Returns
    -------
    forecast_draws : ndarray, shape ``(n_kept, max_horizon, n)``
        Simulated forecast levels for each retained draw.
    hyper_draws : ndarray, shape ``(n_kept, 3)``
        The ``[lambda, theta, miu]`` value at each retained draw.
    """
    rng = np.random.default_rng(seed)
    if seed is not None:
        np.random.seed(int(seed) % (2**32 - 1))

    lo, hi = _bounds_from_context(ctx)
    mode_natural = np.asarray(mode_natural, dtype=float)
    cov = _proposal_covariance(ctx, mode_natural) * (const**2)

    current = mode_natural.copy()
    current_lp = glp_logposterior(ctx, *current)
    kept_fc: list[np.ndarray] = []
    kept_hyper: list[np.ndarray] = []

    total = n_discard + n_draws
    for it in range(total):
        proposal = rng.multivariate_normal(current, cov)
        if np.any(proposal <= lo) or np.any(proposal >= hi):
            accept = False
        else:
            proposal_lp = glp_logposterior(ctx, *proposal)
            accept = np.log(rng.uniform()) < (proposal_lp - current_lp)
            if accept:
                current, current_lp = proposal, proposal_lp
        if it >= n_discard:
            beta, sigma = glp_draw(ctx, *current)
            kept_fc.append(simulate_forecast_path(ctx.y, beta, sigma, ctx.lags, max_horizon, rng))
            kept_hyper.append(current.copy())

    return np.asarray(kept_fc), np.asarray(kept_hyper)


def glp_fixed_hyperparameter_forecast_draws(
    ctx: GLPContext,
    lam: float,
    theta: float,
    miu: float,
    *,
    max_horizon: int,
    n_draws: int,
    seed: int | None = None,
) -> np.ndarray:
    """Predictive forecast draws with the hyperparameters held fixed (used by the
    Mango strategies): repeatedly draw beta/sigma from the conditional posterior
    and simulate a forecast path. Returns ``(n_draws, max_horizon, n)``."""
    rng = np.random.default_rng(seed)
    if seed is not None:
        np.random.seed(int(seed) % (2**32 - 1))
    draws = np.empty((n_draws, max_horizon, ctx.n))
    for d in range(n_draws):
        beta, sigma = glp_draw(ctx, lam, theta, miu)
        draws[d] = simulate_forecast_path(ctx.y, beta, sigma, ctx.lags, max_horizon, rng)
    return draws


# --------------------------------------------------------------------------- #
# Mango parameter space (positional scipy.uniform, avoiding the Mango kwds bug).
# --------------------------------------------------------------------------- #
def make_param_space(bounds: dict[str, tuple[float, float]] | None = None) -> dict[str, Any]:
    from scipy.stats import uniform

    if bounds is None:
        bounds = GLP_PARAM_SPACE_BOUNDS
    # Rename the reserved word 'lambda' to 'lam' for the objective signature.
    renamed = {("lam" if key == "lambda" else key): value for key, value in bounds.items()}
    return {key: uniform(lower, upper - lower) for key, (lower, upper) in renamed.items()}


def _best_hyperparameters(best_params: dict[str, float]) -> dict[str, float]:
    return {
        "lambda": float(best_params["lam"]),
        "theta": float(best_params["theta"]),
        "miu": float(best_params["miu"]),
    }


# --------------------------------------------------------------------------- #
# The three Mango optimizers ported from MBFVAR, adapted to the GLP BVAR.
# --------------------------------------------------------------------------- #
def update_hyperparameters_mango(
    y: np.ndarray,
    lags: int,
    *,
    param_space: dict[str, Any] | None = None,
    init_points: int = 5,
    n_iter: int = 15,
    njobs: int = 1,
    hyperpriors: int = GLP_HYPERPRIORS,
    context: GLPContext | None = None,
    **prior_kwargs: Any,
) -> dict[str, float]:
    """Select ``[lambda, theta, miu]`` by MAXIMIZING the GLP (log) posterior /
    marginal data density with Bayesian optimization (Mango).

    This is the direct analogue of ``MBFVAR.update_hyperparameters_mango`` but on
    the single-frequency GLP BVAR: the objective is ``logMLVAR_formcmc(draw=0)``,
    i.e. the same quantity ``bvarGLP`` maximizes with ``csminwel`` -- only the
    optimizer differs.
    """
    from mango import Tuner, scheduler

    if param_space is None:
        param_space = make_param_space()
    ctx = context if context is not None else prepare_glp_context(y, lags, hyperpriors=hyperpriors, **prior_kwargs)

    @scheduler.parallel(n_jobs=njobs)
    def objective(lam: float, theta: float, miu: float) -> float:
        try:
            return glp_logposterior(ctx, float(lam), float(theta), float(miu))
        except Exception:  # pragma: no cover - a bad draw must not kill the run
            return -1.0e15

    conf = dict(num_iteration=n_iter, initial_random=init_points)
    results = Tuner(param_space, objective, conf).maximize()
    return _best_hyperparameters(results["best_params"])


def _rmse_eval_origins(
    T: int,
    H: int,
    n_eval: int,
    *,
    random: bool,
    min_t: int | None,
    random_seed: int | None,
) -> list[int]:
    """Return the training cut points for the RMSE evaluation origins.

    Origin ``k`` trains on ``y[:T - H - k]`` and scores the ``h``-step forecast
    against ``y[T - H - k + h - 1]``.
    """
    min_t = min_t if min_t is not None else max(4 * 5, H + 5)
    max_k = T - H - min_t
    if max_k < 0:
        raise ValueError(
            f"No valid RMSE evaluation origin: T={T}, H={H}, min_t={min_t}. Use a longer sample or smaller H/min_t."
        )
    n_valid = max_k + 1
    if random:
        if n_eval > n_valid:
            raise ValueError(f"n_eval={n_eval} exceeds the {n_valid} valid random origins.")
        rng = np.random.default_rng(random_seed)
        ks = sorted(rng.choice(n_valid, size=n_eval, replace=False).tolist())
    else:
        ks = [k for k in range(min(n_eval, n_valid))]
    return ks


def _build_rmse_origins(
    y: np.ndarray,
    lags: int,
    ks: Sequence[int],
    H: int,
    prior_kwargs: dict[str, Any],
) -> list[tuple[GLPContext, np.ndarray]]:
    """Pre-build the (hyperparameter-independent) context and holdout actuals for
    each evaluation origin."""
    y = np.asarray(y, dtype=float)
    T = y.shape[0]
    origins: list[tuple[GLPContext, np.ndarray]] = []
    for k in ks:
        cut = T - H - k
        train = y[:cut, :]
        actual = y[cut : cut + H, :]  # H future quarters (rows 0..H-1)
        ctx = prepare_glp_context(train, lags, **prior_kwargs)
        origins.append((ctx, actual))
    return origins


def _rmse_objective(
    origins: list[tuple[GLPContext, np.ndarray]],
    var_indices: Sequence[int],
    H: int,
    h_eval: int | None,
) -> Callable[[float, float, float], float]:
    horizons = list(range(1, H + 1))
    var_indices = list(var_indices)

    def calc_rmse(lam: float, theta: float, miu: float) -> float:
        squared_errors: list[float] = []
        try:
            for ctx, actual in origins:
                betahat, _ = glp_mode_estimate(ctx, float(lam), float(theta), float(miu))
                forecast = point_forecast(ctx.y, betahat, horizons)  # (H, n)
                rows = [h_eval - 1] if h_eval is not None else list(range(H))
                for row in rows:
                    if row >= actual.shape[0] or row >= forecast.shape[0]:
                        continue
                    for vi in var_indices:
                        error = forecast[row, vi] - actual[row, vi]
                        squared_errors.append(float(error) ** 2)
            if not squared_errors:
                return 1.0e10
            rmse = float(np.sqrt(np.mean(squared_errors)))
            if not np.isfinite(rmse):
                return 1.0e10
            return rmse
        except Exception:  # pragma: no cover
            return 1.0e10

    return calc_rmse


def update_hyperparameters_mango_rmse(
    y: np.ndarray,
    lags: int,
    *,
    model_codes: Sequence[str],
    var_of_interest: Sequence[str],
    H: int,
    param_space: dict[str, Any] | None = None,
    init_points: int = 5,
    n_iter: int = 15,
    njobs: int = 1,
    h_eval: int | None = None,
    n_eval: int = 1,
    min_t: int | None = None,
    hyperpriors: int = GLP_HYPERPRIORS,
    **prior_kwargs: Any,
) -> dict[str, float]:
    """Select ``[lambda, theta, miu]`` by MINIMIZING the rolling out-of-sample
    RMSE for ``var_of_interest`` at horizon ``h_eval`` (analogue of
    ``MBFVAR.update_hyperparameters_mango_rmse``)."""
    from mango import Tuner, scheduler

    if param_space is None:
        param_space = make_param_space()
    prior_kwargs = {"hyperpriors": hyperpriors, **prior_kwargs}
    var_indices = _resolve_var_indices(model_codes, var_of_interest)

    y = np.asarray(y, dtype=float)
    ks = _rmse_eval_origins(y.shape[0], H, n_eval, random=False, min_t=min_t, random_seed=None)
    origins = _build_rmse_origins(y, lags, ks, H, prior_kwargs)
    calc_rmse = _rmse_objective(origins, var_indices, H, h_eval)

    parallel_objective = scheduler.parallel(n_jobs=njobs)(calc_rmse)
    conf = dict(num_iteration=n_iter, initial_random=init_points)
    results = Tuner(param_space, parallel_objective, conf).minimize()
    return _best_hyperparameters(results["best_params"])


def update_hyperparameters_mango_rmse_random(
    y: np.ndarray,
    lags: int,
    *,
    model_codes: Sequence[str],
    var_of_interest: Sequence[str],
    H: int,
    param_space: dict[str, Any] | None = None,
    init_points: int = 5,
    n_iter: int = 15,
    njobs: int = 1,
    h_eval: int | None = None,
    n_eval: int = 1,
    min_t: int | None = None,
    random_seed: int | None = None,
    hyperpriors: int = GLP_HYPERPRIORS,
    **prior_kwargs: Any,
) -> dict[str, float]:
    """Like :func:`update_hyperparameters_mango_rmse` but the ``n_eval`` origins
    are drawn at random from the valid pool (analogue of
    ``MBFVAR.update_hyperparameters_mango_rmse_random``)."""
    from mango import Tuner, scheduler

    if param_space is None:
        param_space = make_param_space()
    prior_kwargs = {"hyperpriors": hyperpriors, **prior_kwargs}
    var_indices = _resolve_var_indices(model_codes, var_of_interest)

    y = np.asarray(y, dtype=float)
    ks = _rmse_eval_origins(y.shape[0], H, n_eval, random=True, min_t=min_t, random_seed=random_seed)
    origins = _build_rmse_origins(y, lags, ks, H, prior_kwargs)
    calc_rmse = _rmse_objective(origins, var_indices, H, h_eval)

    parallel_objective = scheduler.parallel(n_jobs=njobs)(calc_rmse)
    conf = dict(num_iteration=n_iter, initial_random=init_points)
    results = Tuner(param_space, parallel_objective, conf).minimize()
    return _best_hyperparameters(results["best_params"])


def _resolve_var_indices(model_codes: Sequence[str], var_of_interest: Sequence[str]) -> list[int]:
    codes = list(model_codes)
    indices: list[int] = []
    for code in var_of_interest:
        if code not in codes:
            raise ValueError(f"Variable {code!r} is not in the model block {codes}.")
        index = codes.index(code)
        if index not in indices:
            indices.append(index)
    if not indices:
        raise ValueError("var_of_interest resolved to an empty set of columns.")
    return indices
