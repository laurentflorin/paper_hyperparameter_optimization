"""Fast GLP posterior and the forecast-based hyperparameter objectives ("ladder").

Why a second implementation of the GLP posterior?
-------------------------------------------------
The forecast-based objectives re-estimate the BVAR at many pseudo out-of-sample
origins for every candidate hyperparameter vector. ``covbayesvar`` rebuilds
``X'X`` from the raw data on every call and draws one ``(beta, Sigma)`` pair per
call. Here the data cross-products of every sample prefix are precomputed once
(prefix sums), so the posterior at any origin costs ``O(k^2 n)`` instead of
``O(T k^2)``, and posterior draws are generated in vectorized batches with
common random numbers (fixed per origin), which makes every objective a smooth
deterministic function of the hyperparameters. The implementation mirrors
``covbayesvar.large_bvar.logMLVAR_formcmc`` line by line and is validated
against it in ``tests/test_glp_objectives.py``.

The ladder
----------
The GLP marginal likelihood is itself an out-of-sample criterion:
``log p(y_{1:T}) = sum_t log p(y_t | y_{1:t-1})`` -- the joint, one-step-ahead,
full-sample log predictive score. A targeted RMSE objective departs from it in
four ways (window, horizon, variables, point vs density). The objectives below
relax one departure at a time:

=====================  ==========  =======  =========  ========  =============
objective              window      horizon  variables  target    computation
=====================  ==========  =======  =========  ========  =============
``mdd``                full        1        joint      density   closed form
``logscore_joint_1``   recent W    1        joint      density   exact (ML ratio)
``logscore_joint``     recent W    h        joint      density   RB mixture
``logscore_marginal``  recent W    h        target     density   RB mixture
``rmse``               recent W    h        target     point     draw average
``rmse_joint``         recent W    h        joint      point     draw average
=====================  ==========  =======  =========  ========  =============

``rmse_joint`` scales each variable's squared error by its AR(1) residual
variance on the origin's sample (the ``SS`` scale of the GLP prior, which does
not depend on the hyperparameters), so no single volatile variable dominates.

"RB mixture" is the Rao-Blackwellized predictive density: conditional on a
posterior draw ``(beta, Sigma)`` the ``h``-step predictive is Gaussian with the
recursive mean and ``sum_j Psi_j Sigma Psi_j'`` covariance, so the predictive
density is estimated by a mixture of exact Gaussians over the draws. At
``h = 1`` it converges to the exact matric-t value used by ``logscore_joint_1``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from scipy.special import gammaln, logsumexp

from .glp_model import (
    GLPContext,
    _full_bounds,
    _psi_enabled,
    natural_vector_to_hyper,
    prepare_glp_context,
    to_natural,
    to_transformed,
)

LADDER_OBJECTIVES = ("mdd", "logscore_joint_1", "logscore_joint", "logscore_marginal", "rmse", "rmse_joint")
# Objectives whose value depends on the target horizon / target variables.
HORIZON_OBJECTIVES = {"logscore_joint", "logscore_marginal", "rmse", "rmse_joint"}
TARGET_OBJECTIVES = {"logscore_marginal", "rmse"}
PENALTY = 1.0e10


# --------------------------------------------------------------------------- #
# Hyperparameter unpacking (covbayesvar ordering [lambda, psi, theta, miu]).
# --------------------------------------------------------------------------- #
def _unpack(vec: np.ndarray, n: int, psi_enabled: bool, ss: np.ndarray) -> tuple[float, np.ndarray, float, float]:
    vec = np.asarray(vec, dtype=float).ravel()
    lam = float(vec[0])
    if psi_enabled:
        psi = vec[1 : n + 1].astype(float)
        theta, miu = float(vec[n + 1]), float(vec[n + 2])
    else:
        psi = np.ravel(ss).astype(float)  # (d - n - 1) = 1
        theta, miu = float(vec[1]), float(vec[2])
    return lam, psi, theta, miu


@dataclass
class Posterior:
    """Conjugate NIW posterior at one origin for one hyperparameter vector."""

    betahat: np.ndarray  # (k, n)
    chol_zzinv: np.ndarray  # (k, k) lower, L L' = (X'X + Omega^-1)^-1
    s_matrix: np.ndarray  # (n, n) posterior IW scale
    dof: int  # T + Td + d
    log_ml: float  # log marginal likelihood (no hyperprior)


@dataclass
class FastGLP:
    """Prefix-sum sufficient statistics of one real-time estimation sample.

    ``cut`` arguments are numbers of *rows of the raw data matrix* used for
    estimation (the same convention as ``y[:cut]`` in the RMSE objective): the
    posterior at ``cut`` uses the regression rows ``lags .. cut-1``.
    """

    y: np.ndarray  # raw data (T_raw, n)
    lags: int
    n: int
    k: int
    b: np.ndarray
    vc: float
    y0: np.ndarray
    sur: int
    noc: int
    pos: list
    alpha: float
    psi_enabled: bool
    ss: np.ndarray
    xx_cum: np.ndarray = field(repr=False)  # (T_eff + 1, k, k)
    xy_cum: np.ndarray = field(repr=False)  # (T_eff + 1, k, n)
    yy_cum: np.ndarray = field(repr=False)  # (T_eff + 1, n, n)

    @classmethod
    def from_context(cls, raw_y: np.ndarray, ctx: GLPContext) -> "FastGLP":
        if ctx.MCMCMsur:
            raise NotImplementedError("MCMCMsur=1 is not supported by the fast GLP posterior.")
        x, y = ctx.x, ctx.y
        zero_k = np.zeros((1, ctx.k, ctx.k))
        xx = np.concatenate([zero_k, np.cumsum(np.einsum("ti,tj->tij", x, x), axis=0)])
        xy = np.concatenate([np.zeros((1, ctx.k, ctx.n)), np.cumsum(np.einsum("ti,tj->tij", x, y), axis=0)])
        yy = np.concatenate([np.zeros((1, ctx.n, ctx.n)), np.cumsum(np.einsum("ti,tj->tij", y, y), axis=0)])
        return cls(
            y=np.asarray(raw_y, dtype=float),
            lags=ctx.lags,
            n=ctx.n,
            k=ctx.k,
            b=ctx.b,
            vc=ctx.Vc,
            y0=np.ravel(ctx.y0),
            sur=ctx.sur,
            noc=ctx.noc,
            pos=list(ctx.pos),
            alpha=2.0 if int(ctx.mn.get("alpha", 0)) == 0 else float("nan"),
            psi_enabled=_psi_enabled(ctx),
            ss=np.ravel(ctx.SS),
            xx_cum=xx,
            xy_cum=xy,
            yy_cum=yy,
        )

    # ------------------------------------------------------------------ #
    def _dummies(self, theta: float, miu: float) -> tuple[np.ndarray, np.ndarray]:
        n, lags = self.n, self.lags
        xd_rows, yd_rows = [], []
        if self.sur == 1:
            xd_rows.append(np.concatenate(([1.0 / theta], np.tile(self.y0, lags) / theta))[None, :])
            yd_rows.append((self.y0 / theta)[None, :])
        if self.noc == 1:
            ydnoc = np.diag(self.y0 / miu)
            for p in self.pos:
                ydnoc[int(p), int(p)] = 0.0
            xdnoc = np.hstack([np.zeros((n, 1)), np.tile(np.diag(self.y0 / miu), (1, lags))])
            xd_rows.append(xdnoc)
            yd_rows.append(ydnoc)
        if not xd_rows:
            return np.zeros((0, self.k)), np.zeros((0, n))
        return np.vstack(xd_rows), np.vstack(yd_rows)

    def _omega(self, lam: float, psi: np.ndarray) -> np.ndarray:
        omega = np.empty(self.k)
        omega[0] = self.vc
        for i in range(1, self.lags + 1):
            omega[1 + (i - 1) * self.n : 1 + i * self.n] = lam**2 / (i**self.alpha) / psi
        return omega

    @staticmethod
    def _log_ml_terms(n: int, T: int, d: int, psi, omega, xx, eps_quad) -> float:
        sq = np.sqrt(omega)
        aaa = sq[:, None] * xx * sq[None, :]
        _, logdet_a = np.linalg.slogdet(np.eye(xx.shape[0]) + aaa)
        ipsi = 1.0 / np.sqrt(psi)
        bbb = ipsi[:, None] * eps_quad * ipsi[None, :]
        _, logdet_b = np.linalg.slogdet(np.eye(n) + 0.5 * (bbb + bbb.T))
        i = np.arange(n)
        return float(
            -n * T * np.log(np.pi) / 2.0
            + np.sum(gammaln((T + d - i) / 2.0) - gammaln((d - i) / 2.0))
            - T * np.sum(np.log(psi)) / 2.0
            - n * logdet_a / 2.0
            - (T + d) * logdet_b / 2.0
        )

    def posterior(self, vec: Sequence[float], cut: int) -> Posterior:
        """Posterior using raw rows ``y[:cut]`` (regression rows ``lags..cut-1``)."""
        n, k = self.n, self.k
        lam, psi, theta, miu = _unpack(np.asarray(vec), n, self.psi_enabled, self.ss)
        t_eff = int(cut) - self.lags
        if t_eff < 1:
            raise ValueError(f"cut={cut} leaves no regression rows with {self.lags} lags.")
        xd, yd = self._dummies(theta, miu)
        xx = self.xx_cum[t_eff] + xd.T @ xd
        xy = self.xy_cum[t_eff] + xd.T @ yd
        yy = self.yy_cum[t_eff] + yd.T @ yd
        omega = self._omega(lam, psi)
        prec = xx + np.diag(1.0 / omega)
        chol_prec = np.linalg.cholesky(prec)
        rhs = xy + (1.0 / omega)[:, None] * self.b
        betahat = np.linalg.solve(chol_prec.T, np.linalg.solve(chol_prec, rhs))
        # eps'eps + (B - b)' Omega^-1 (B - b), via sufficient statistics.
        dev = betahat - self.b
        quad = yy - betahat.T @ xy - xy.T @ betahat + betahat.T @ xx @ betahat + dev.T @ ((1.0 / omega)[:, None] * dev)
        quad = 0.5 * (quad + quad.T)
        d = n + 2
        T_tot = t_eff + xd.shape[0]
        log_ml = self._log_ml_terms(n, T_tot, d, psi, omega, xx, quad)
        if xd.shape[0] > 0:
            epsd = yd - xd @ self.b
            log_ml -= self._log_ml_terms(n, xd.shape[0], d, psi, omega, xd.T @ xd, epsd.T @ epsd)
        # Lower Cholesky of prec^-1: prec^-1 = L^-T L^-1 -> chol via inverse of upper factor.
        inv_chol = np.linalg.solve(chol_prec, np.eye(k))  # L^-1
        zzinv = inv_chol.T @ inv_chol
        chol_zzinv = np.linalg.cholesky(0.5 * (zzinv + zzinv.T))
        s_matrix = np.diag(psi) + quad
        return Posterior(betahat, chol_zzinv, 0.5 * (s_matrix + s_matrix.T), T_tot + d, log_ml)

    def log_ml(self, vec: Sequence[float], cut: int) -> float:
        return self.posterior(vec, cut).log_ml


# --------------------------------------------------------------------------- #
# Posterior draws (vectorized, common random numbers) and predictive moments.
# --------------------------------------------------------------------------- #
@dataclass
class DrawNoise:
    """Standard-normal / chi-square noise reused for every hyperparameter
    candidate at one origin (common random numbers)."""

    z_beta: np.ndarray  # (D, k, n)
    z_offdiag: np.ndarray  # (D, n, n)
    chi: np.ndarray  # (D, n) sqrt(chi2(dof - i))

    @classmethod
    def generate(cls, n_draws: int, k: int, n: int, dof: int, seed: int) -> "DrawNoise":
        rng = np.random.default_rng(seed)
        chi = np.sqrt(rng.chisquare(dof - np.arange(n), size=(n_draws, n)))
        return cls(rng.standard_normal((n_draws, k, n)), rng.standard_normal((n_draws, n, n)), chi)


def draw_posterior(post: Posterior, noise: DrawNoise) -> tuple[np.ndarray, np.ndarray]:
    """``(beta, Sigma)`` draws, shapes ``(D, k, n)`` and ``(D, n, n)``.

    ``Sigma ~ IW(S, dof)`` via the Bartlett decomposition of ``W(S^-1, dof)``;
    ``vec(beta) | Sigma ~ N(vec(betahat), Sigma kron (X'X + Omega^-1)^-1)``.
    """
    n = post.s_matrix.shape[0]
    s_inv = np.linalg.inv(post.s_matrix)
    l_s = np.linalg.cholesky(0.5 * (s_inv + s_inv.T))
    bart = np.tril(noise.z_offdiag, -1)
    idx = np.arange(n)
    bart[:, idx, idx] = noise.chi
    la = l_s[None, :, :] @ bart  # (D, n, n); W = la la' ~ Wishart(S^-1, dof)
    la_inv = np.linalg.inv(la)  # (D, n, n)
    sigma = np.transpose(la_inv, (0, 2, 1)) @ la_inv
    # With U = la^-1, Sigma = U'U, so betahat + L Z U has covariance Sigma kron (L L').
    beta = post.betahat[None] + post.chol_zzinv[None] @ noise.z_beta @ la_inv
    return beta, sigma


def predictive_moments(
    y_hist: np.ndarray, beta: np.ndarray, sigma: np.ndarray | None, lags: int, horizon: int
) -> tuple[np.ndarray, np.ndarray | None]:
    """Per-draw ``horizon``-step predictive mean ``(D, n)`` and covariance
    ``(D, n, n)``; the covariance is skipped (``None``) when ``sigma`` is None."""
    D, k, n = beta.shape
    hist = [np.broadcast_to(y_hist[-j], (D, n)) for j in range(1, lags + 1)]  # y_T, y_{T-1}, ...
    const = beta[:, 0, :]
    a_mats = [np.transpose(beta[:, 1 + (i - 1) * n : 1 + i * n, :], (0, 2, 1)) for i in range(1, lags + 1)]
    means = []
    psi_list = [np.broadcast_to(np.eye(n), (D, n, n))]
    for step in range(1, horizon + 1):
        mean = const.copy()
        for i in range(lags):
            mean = mean + np.einsum("dij,dj->di", a_mats[i], hist[i])
        hist = [mean] + hist[:-1]
        means.append(mean)
        if sigma is not None and step < horizon:
            psi_j = np.zeros((D, n, n))
            for i in range(1, min(step, lags) + 1):
                psi_j = psi_j + a_mats[i - 1] @ psi_list[step - i]
            psi_list.append(psi_j)
    if sigma is None:
        return means[-1], None
    cov = np.zeros((D, n, n))
    for psi_j in psi_list:
        cov = cov + psi_j @ sigma @ np.transpose(psi_j, (0, 2, 1))
    return means[-1], 0.5 * (cov + np.transpose(cov, (0, 2, 1)))


def mixture_log_density(obs: np.ndarray, means: np.ndarray, covs: np.ndarray) -> float:
    """``log (1/D) sum_d N(obs; means_d, covs_d)`` for a Gaussian mixture."""
    D, m = means.shape
    chol = np.linalg.cholesky(covs)
    resid = obs[None, :] - means
    z = np.linalg.solve(chol, resid[:, :, None])[:, :, 0]
    logdet = 2.0 * np.sum(np.log(np.diagonal(chol, axis1=1, axis2=2)), axis=1)
    comp = -0.5 * (m * np.log(2.0 * np.pi) + logdet + np.sum(z * z, axis=1))
    return float(logsumexp(comp) - np.log(D))


# --------------------------------------------------------------------------- #
# Evaluation origins and objective construction.
# --------------------------------------------------------------------------- #
def evaluation_cuts(T_raw: int, lags: int, horizon: int, window: int | None, min_t: int | None = None) -> list[int]:
    """Training-sample ends ``cut`` (raw rows ``y[:cut]``) whose ``horizon``-step
    target ``y[cut + horizon - 1]`` is observed, most recent first.

    ``window=None`` uses every valid origin (expanding-window / prequential).
    """
    min_t = min_t if min_t is not None else max(4 * 5, horizon + 5) + lags
    last = T_raw - horizon
    cuts = list(range(last, min_t - 1, -1))
    if not cuts:
        raise ValueError(f"No valid evaluation origin: T={T_raw}, horizon={horizon}, min_t={min_t}.")
    return cuts if window is None else cuts[: int(window)]


@dataclass
class LadderSpec:
    objective: str
    horizon: int = 1
    window: int | None = 12  # None = all valid origins
    target_indices: tuple[int, ...] = (0,)
    n_draws: int = 200
    seed: int = 0
    min_t: int | None = None

    def label(self) -> str:
        win = "all" if self.window is None else f"w{self.window}"
        if self.objective == "mdd":
            return "mdd"
        if self.objective == "logscore_joint_1":
            return f"lsj1_{win}"
        short = {"logscore_joint": "lsj", "logscore_marginal": "lsm", "rmse": "rmse", "rmse_joint": "rmsej"}[self.objective]
        return f"{short}_h{self.horizon}_{win}"


def build_loss(fast: FastGLP, spec: LadderSpec, hyperpriors_ctx: GLPContext | None = None) -> Callable[[np.ndarray], float]:
    """Loss (to MINIMIZE) of a natural hyperparameter vector under ``spec``."""
    y = fast.y
    T_raw = y.shape[0]

    if spec.objective == "mdd":
        if hyperpriors_ctx is None:
            raise ValueError("The 'mdd' objective needs the origin context for its hyperprior terms.")
        from .glp_model import glp_logposterior

        return lambda vec: -glp_logposterior(hyperpriors_ctx, vec)

    if spec.objective == "logscore_joint_1":
        cuts = evaluation_cuts(T_raw, fast.lags, 1, spec.window, spec.min_t)
        first, last = min(cuts), max(cuts) + 1  # telescoping ML ratio

        def loss(vec: np.ndarray) -> float:
            return -(fast.log_ml(vec, last) - fast.log_ml(vec, first))

        return loss

    h = int(spec.horizon)
    cuts = evaluation_cuts(T_raw, fast.lags, h, spec.window, spec.min_t)
    targets = list(spec.target_indices)
    noises = [
        DrawNoise.generate(spec.n_draws, fast.k, fast.n, (cut - fast.lags) + _n_dummy(fast) + fast.n + 2, spec.seed + cut)
        for cut in cuts
    ]
    point = spec.objective in ("rmse", "rmse_joint")
    needs_cov = not point
    # Point-loss columns and their weights (1 for the univariate target RMSE).
    columns = list(range(fast.n)) if spec.objective == "rmse_joint" else targets
    weights = 1.0 / fast.ss[columns] if spec.objective == "rmse_joint" else np.ones(len(columns))

    def loss(vec: np.ndarray) -> float:
        total = 0.0
        sq_err: list[float] = []
        for cut, noise in zip(cuts, noises):
            post = fast.posterior(vec, cut)
            beta, sigma = draw_posterior(post, noise)
            actual = y[cut + h - 1]
            mean, cov = predictive_moments(y[:cut], beta, sigma if needs_cov else None, fast.lags, h)
            if spec.objective == "logscore_joint":
                total += mixture_log_density(actual, mean, cov)
            elif spec.objective == "logscore_marginal":
                sub = np.ix_(targets, targets)
                total += mixture_log_density(actual[targets], mean[:, targets], cov[:, sub[0], sub[1]])
            else:  # (weighted) rmse of the predictive mean
                err = mean[:, columns].mean(axis=0) - actual[columns]
                sq_err.extend((weights * err**2).tolist())
        if point:
            return float(np.sqrt(np.mean(sq_err)))
        return -total

    return loss


def _n_dummy(fast: FastGLP) -> int:
    return (1 if fast.sur == 1 else 0) + (fast.n if fast.noc == 1 else 0)


# --------------------------------------------------------------------------- #
# Optimizer: multi-start Nelder-Mead in the unconstrained space.
# --------------------------------------------------------------------------- #
def optimize_loss(
    loss: Callable[[np.ndarray], float],
    ctx: GLPContext,
    start_natural: np.ndarray,
    *,
    n_starts: int = 4,
    start_scale: float = 1.5,
    maxfev: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Minimize ``loss`` over the GLP box with Nelder-Mead from the supplied start
    (typically the marginal-likelihood mode) plus ``n_starts - 1`` random
    perturbations of it in the logistic-transformed space. Returns the best
    natural vector, its loss, and every start's result for diagnostics."""
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    x0 = to_transformed(start_natural, ctx)
    dim = x0.size
    maxfev = maxfev or 150 * dim

    def f(t: np.ndarray) -> float:
        try:
            value = float(loss(to_natural(t, ctx)))
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return PENALTY
        return value if np.isfinite(value) else PENALTY

    starts = [x0] + [x0 + rng.normal(0.0, start_scale, dim) for _ in range(max(0, n_starts - 1))]
    runs = []
    for s in starts:
        res = minimize(f, s, method="Nelder-Mead", options={"maxfev": maxfev, "xatol": 1e-4, "fatol": 1e-6})
        runs.append((float(res.fun), res.x, int(res.nfev)))
    best = min(runs, key=lambda r: r[0])
    return {
        "natural": to_natural(best[1], ctx),
        "loss": best[0],
        "start_losses": [r[0] for r in runs],
        "nfev": sum(r[2] for r in runs),
        "loss_at_start": f(x0),
    }


def select_ladder_hyperparameters(
    y: np.ndarray,
    lags: int,
    spec: LadderSpec,
    *,
    mdd_mode: dict[str, Any] | None = None,
    n_starts: int = 4,
    maxfev: int | None = None,
    hyperpriors: int = 1,
    **prior_kwargs: Any,
) -> dict[str, Any]:
    """Select GLP hyperparameters for one real-time sample under a ladder objective.

    Every rung starts its search at the marginal-likelihood mode (computed here
    when ``mdd_mode`` is not supplied) so differences across rungs reflect the
    objective, not the optimizer. The ``mdd`` rung returns that mode directly.
    """
    from .glp_model import glp_find_mode, glp_logposterior, hyper_to_natural_vector

    ctx = prepare_glp_context(y, lags, hyperpriors=hyperpriors, **prior_kwargs)
    mode = mdd_mode if mdd_mode is not None else glp_find_mode(ctx)
    mode_vec = hyper_to_natural_vector(mode, ctx)
    if spec.objective == "mdd":
        hyper = natural_vector_to_hyper(mode_vec, ctx)
        return {**hyper, "objective_value": float(-glp_logposterior(ctx, mode_vec)), "objective_at_mdd_mode": None}
    fast = FastGLP.from_context(y, ctx)
    loss = build_loss(fast, spec)
    result = optimize_loss(loss, ctx, mode_vec, n_starts=n_starts, maxfev=maxfev, seed=spec.seed)
    hyper = natural_vector_to_hyper(result["natural"], ctx)
    return {
        **hyper,
        "objective_value": result["loss"],
        "objective_at_mdd_mode": result["loss_at_start"],
        "objective_nfev": result["nfev"],
    }


# --------------------------------------------------------------------------- #
# Fixed-hyperparameter predictive draws for the final forecasts.
# --------------------------------------------------------------------------- #
def fast_forecast_draws(
    y: np.ndarray,
    ctx: GLPContext,
    hyper_vector: Sequence[float],
    *,
    max_horizon: int,
    n_draws: int,
    seed: int | None = None,
) -> np.ndarray:
    """Predictive paths ``(n_draws, max_horizon, n)`` with the hyperparameters
    held fixed: ``(beta, Sigma)`` from the conditional posterior on the full
    sample ``y`` and Gaussian shocks. Same distribution as
    :func:`glp_model.glp_fixed_hyperparameter_forecast_draws`, vectorized."""
    fast = FastGLP.from_context(y, ctx)
    post = fast.posterior(hyper_vector, y.shape[0])
    rng = np.random.default_rng(seed)
    noise = DrawNoise.generate(n_draws, fast.k, fast.n, post.dof, int(rng.integers(2**31)))
    beta, sigma = draw_posterior(post, noise)
    return simulate_paths(y, beta, sigma, fast.lags, max_horizon, rng)


def simulate_paths(
    y: np.ndarray, beta: np.ndarray, sigma: np.ndarray, lags: int, max_horizon: int, rng: np.random.Generator
) -> np.ndarray:
    """Vectorized analogue of :func:`glp_model.simulate_forecast_path` over draws."""
    D, k, n = beta.shape
    chol = np.linalg.cholesky(sigma)  # (D, n, n)
    shocks = np.einsum("dij,dhj->dhi", chol, rng.standard_normal((D, max_horizon, n)))
    hist = [np.broadcast_to(y[-j], (D, n)) for j in range(1, lags + 1)]
    const = beta[:, 0, :]
    a_mats = [np.transpose(beta[:, 1 + (i - 1) * n : 1 + i * n, :], (0, 2, 1)) for i in range(1, lags + 1)]
    out = np.empty((D, max_horizon, n))
    for step in range(max_horizon):
        level = const + shocks[:, step, :]
        for i in range(lags):
            level = level + np.einsum("dij,dj->di", a_mats[i], hist[i])
        out[:, step, :] = level
        hist = [level] + hist[:-1]
    return out


def glp_log_posterior_unbounded(y: np.ndarray, ctx: GLPContext, hyper_vector: Sequence[float]) -> float:
    """GLP log posterior (log ML + hyperprior when ``ctx.hyperpriors``) at any
    positive hyperparameter vector. Equals :func:`glp_model.glp_logposterior`
    inside the box, but -- unlike covbayesvar -- does not return the ``-1e15``
    sentinel when a selection reused from an earlier origin falls outside this
    sample's data-dependent psi bounds."""
    from covbayesvar.large_bvar import log_gamma_pdf, log_ig2pdf

    fast = FastGLP.from_context(y, ctx)
    value = fast.log_ml(hyper_vector, y.shape[0])
    if ctx.hyperpriors == 1:
        lam, psi, theta, miu = _unpack(np.asarray(hyper_vector), fast.n, fast.psi_enabled, fast.ss)
        pc = ctx.priorcoef
        value += float(log_gamma_pdf(lam, pc["lambda"]["k"], pc["lambda"]["theta"]))
        if ctx.sur == 1:
            value += float(log_gamma_pdf(theta, pc["theta"]["k"], pc["theta"]["theta"]))
        if ctx.noc == 1:
            value += float(log_gamma_pdf(miu, pc["miu"]["k"], pc["miu"]["theta"]))
        if fast.psi_enabled:
            value += float(np.sum(log_ig2pdf(psi, pc["alpha"]["PSI"], pc["beta"]["PSI"])))
    return float(value)
