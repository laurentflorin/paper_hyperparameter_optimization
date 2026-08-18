# GLP (2015) Prior Selection: Methodological Companion

> **Status (2026-08-18):** This document is current. The scope-grid
> architecture described in [docs/EXPERIMENT_DESIGN.md](docs/EXPERIMENT_DESIGN.md)
> supersedes the single-origin strategy descriptions in earlier versions of this
> file; the statistical derivation below remains unchanged and authoritative.
> Implementation commands are in [README.md](README.md) and
> [docs/PILOT_VALIDATION.md](docs/PILOT_VALIDATION.md).

This document is the **theoretical** companion to the GLP workflow in
`src/glp_hyperparameter_optimization` and `scripts/glp`. It explains every
hyperparameter‑selection strategy implemented in this repository, what each one
is doing statistically, and exactly how it relates to — or deliberately departs
from — the target paper:

> Giannone, D., Lenza, M., and Primiceri, G. E. (2015),
> *Prior Selection for Vector Autoregressions*,
> **Review of Economics and Statistics**, 97(2), 436–451.
> SSRN: <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2164591>

The intent is that this file can serve as the methodological skeleton of a
paper. It is written from a statistical/econometric standpoint; it does **not**
describe the Python API. Implementation names are mentioned only to let a reader
map each concept back to the code and the output files.

---

## 1. The problem GLP (2015) solve

A vector autoregression (VAR) with $n$ variables and $p$ lags has on the order of
$n^2 p$ autoregressive coefficients. As $n$ grows, the number of free parameters
explodes relative to the length of macroeconomic samples, so an unrestricted
(flat‑prior / OLS) VAR overfits badly: it fits the in‑sample noise and forecasts
poorly out of sample. The classical remedy is **shrinkage** through an
informative prior — most famously the *Minnesota prior* (Litterman, 1979; Doan,
Litterman, and Sims, 1984), which pulls the VAR toward a set of independent
random walks.

Any such prior is governed by a small set of **hyperparameters** that control how
much the data are allowed to move the coefficients away from the prior mean
(i.e., how aggressive the shrinkage is). Historically these hyperparameters were
set by rules of thumb or tuned informally on forecast performance. The central
methodological contribution of GLP (2015) is to put shrinkage on a principled,
fully Bayesian footing:

1. **Treat the hyperparameters as parameters.** Place a *hyperprior* on them and
   form a hierarchical model.
2. **Let the data choose the shrinkage** through the **marginal likelihood**
   (marginal data density, MDD), which is available in closed form under a
   conjugate prior and automatically embodies an Occam's‑razor penalty against
   overly loose priors.
3. **Show the practical payoff:** with data‑driven shrinkage, *large* Bayesian
   VARs forecast at least as well as — and often better than — small VARs and
   factor models, because the optimal amount of shrinkage rises with the
   cross‑sectional dimension $n$.

This repository reproduces GLP's own procedure and then asks a research question
GLP did not: **does selecting the shrinkage hyperparameters by the marginal
likelihood produce different forecasts than selecting them by directly
optimizing out‑of‑sample forecast accuracy?** Four strategies are implemented to
answer this, all on the *same* BVAR, the *same* data, and the *same* real‑time
evaluation design, so that the only thing that changes across strategies is the
**hyperparameter objective**.

---

## 2. The model: a conjugate Bayesian VAR

Let $y_t$ be the $n\times 1$ vector of (transformed) macro variables. The reduced‑form
VAR with $p$ lags and an intercept is

$$
y_t = c + A_1 y_{t-1} + A_2 y_{t-2} + \dots + A_p y_{t-p} + \varepsilon_t,
\qquad \varepsilon_t \sim \mathcal{N}(0, \Sigma).
$$

Stacking the coefficients into $\beta = \operatorname{vec}([c, A_1, \dots, A_p]')$,
the two unknowns are the coefficient vector $\beta$ and the innovation covariance
$\Sigma$. GLP use a **conjugate Normal–Inverse‑Wishart (NIW) prior**:

$$
\Sigma \sim \mathcal{IW}(\Psi, d), \qquad
\beta \mid \Sigma \sim \mathcal{N}\!\big(b,\ \Sigma \otimes \Omega\big).
$$

Conjugacy is what makes the whole GLP program tractable: the posterior of
$(\beta, \Sigma)$ is again Normal–Inverse‑Wishart, and — crucially — the
**marginal likelihood** obtained by integrating $(\beta,\Sigma)$ out has a
closed‑form matric‑variate‑$t$ expression. The prior mean $b$, the coefficient
prior‑covariance factor $\Omega$, and the scale matrix $\Psi$ are all functions
of the hyperparameters described next.

---

## 3. The GLP prior and its hyperparameters

The GLP prior is the **Minnesota prior** augmented with two sets of
**dummy‑observation priors**. Dummy (artificial) observations are a classical
device (Theil mixed estimation): adding fictitious data rows that the model is
asked to "fit" is equivalent to imposing a prior, and it keeps the posterior
conjugate.

### 3.1 The Minnesota core — tightness $\lambda$, scales $\psi$, and fixed $\alpha$

The Minnesota prior encodes the belief that each variable is *a priori* a random
walk and that recent lags matter more than distant lags:

- **Prior mean.** The coefficient on a variable's own first lag is centered at
  $1$ and every other coefficient at $0$ (a driftless random walk). Variables
  known to be stationary in levels (here the interest‑rate/unemployment‑type
  "rate" variables) are instead centered at $0$.
- **Prior variance.** The prior variance of the coefficient on lag $\ell$ of
  variable $j$ in the equation for variable $i$ is

$$
\operatorname{Var}\big[(A_\ell)_{ij}\big] \;\propto\; \frac{\lambda^{2}}{\ell^{\,\alpha}}\,\frac{\psi_i}{\psi_j}.
$$

The three ingredients here are the three "structural" Minnesota hyperparameters:

| Symbol | Name | Role | Status in this study |
| --- | --- | --- | --- |
| $\lambda$ | **Overall tightness** | Global scale of the prior standard deviation. $\lambda\to 0$ imposes the random‑walk prior dogmatically (maximal shrinkage); $\lambda\to\infty$ recovers the flat‑prior / OLS VAR (no shrinkage). | **Estimated** |
| $\alpha$ | **Lag‑decay** | How fast the prior tightens on higher lags. | **Fixed at $\alpha=2$** (the paper's $1/s^2$ decay) |
| $\psi$ | **Residual‑variance scale** | The diagonal scale matrix $\Psi=\operatorname{diag}(\psi)$ of the inverse‑Wishart; also rescales cross‑variable prior variances. | **Estimated** (an $n$‑vector of scales; `MNpsi=1`) |

$\lambda$ is *the* key object in GLP: it is the single number that most directly
governs the bias–variance trade‑off of the whole system.

> **Scope of the search.** Matching the paper — which states that "the setting of
> these priors depends on the hyperparameters $\lambda$, $\mu$, $\delta$ and
> $\psi$, which we treat as additional parameters" (GLP 2015, §3) — every strategy
> estimates the **full set**: $\lambda$, $\theta$ (the paper's $\delta$), $\mu$,
> and the residual‑variance vector $\psi$ (in code: `MNpsi = 1`). Only the
> Minnesota lag‑decay $\alpha$ is held fixed at $2$ (`MNalpha = 0`), which is
> itself paper‑consistent because the paper writes the Minnesota variance with a
> fixed $1/s^2$ decay. The common search space is therefore
> $[\lambda, \psi_1, \dots, \psi_n, \theta, \mu]$ — three scalars plus the
> $n$‑vector $\psi$.

### 3.2 The two additional dummy‑observation priors — $\theta$ and $\mu$

A flat‑prior VAR in levels tends to imply an implausible amount of deterministic
trending behaviour and to under‑represent the near‑unit‑root, cointegrated
character of macro data. GLP therefore add two further priors, each implemented
through dummy observations and each with its own tightness hyperparameter. The
identification below is taken **from the actual dummy‑observation structure** in
the estimation code, which is what determines the economics.

| Symbol | Prior | Dummy structure | Economic content | Tight ($\to 0$) limit |
| --- | --- | --- | --- | --- |
| $\theta$ | **Dummy‑initial‑observation** prior (a.k.a. *single‑unit‑root* / *co‑persistence*; Sims, 1993; Sims and Zha, 1998) | **One** dummy observation containing the intercept and a common level $\bar y_0$ across all variables | The system, started at its average initial level, tends to stay there — i.e. the variables may share **a single common stochastic trend** (cointegration is allowed / favoured) | Favours one common unit root and cointegration; disciplines the deterministic component |
| $\mu$ | **Sum‑of‑coefficients** prior (a.k.a. *no‑cointegration*; Doan, Litterman, and Sims, 1984) | **$n$** diagonal dummy observations, one per variable, with **no** intercept | Each variable's own lag coefficients sum to $1$ and other variables' sum to $0$ — i.e. **each series has its own independent unit root and there is no cointegration** | Imposes $n$ independent unit roots; rules out cointegration |

The two priors pull in *opposite* directions on the question of cointegration,
and GLP let the data adjudicate by estimating both tightnesses. As with
$\lambda$, **smaller values mean tighter (more dogmatic) priors** because in the
dummy rows the artificial observations enter scaled by $1/\theta$ and $1/\mu$.

> **Terminology note for the paper.** The code's switch names are `sur` (with
> hyperparameter $\theta$) and `noc` (with hyperparameter $\mu$). Structurally,
> and per the `covbayesvar` docstrings, `noc` is the **no‑cointegration /
> sum‑of‑coefficients** prior and `sur` is the **single‑unit‑root /
> dummy‑initial‑observation** prior. This is the identification used throughout
> this document, and the inline comments in
> `src/glp_hyperparameter_optimization/config.py` and the `glp_model` docstring
> now match it.

### 3.3 The hierarchical layer — hyperpriors on $(\lambda, \psi, \theta, \mu)$

What turns this from "a prior with knobs" into GLP's hierarchical model is the
**hyperprior** $p(\gamma)$ on the hyperparameter vector
$\gamma = (\lambda, \psi, \theta, \mu)$. The three tightness scalars each get an
independent **Gamma** hyperprior (parameterized by a mode and a standard
deviation), and the residual‑variance scales $\psi$ get a diffuse
**inverse‑Gamma** hyperprior:

| Hyperparameter | Hyperprior | Mode | Std. dev. |
| --- | --- | --- | --- |
| $\lambda$ | Gamma | $0.2$ | $0.4$ |
| $\theta$ | Gamma | $1.0$ | $1.0$ |
| $\mu$ | Gamma | $1.0$ | $1.0$ |
| $\psi/(d-n-1)$ | Inverse‑Gamma | $\approx(0.02)^2$ | scale = shape = $(0.02)^2$ |

These are exactly GLP's benchmark hyperprior settings. The $\lambda$ hyperprior
is deliberately loose and centered on a fairly tight value ($0.2$); the
inverse‑Gamma on $\psi$ (the prior mean of the diagonal of $\Sigma$) peaks near
$(0.02)^2$ because the data enter in annualized log terms, and is diffuse enough
to let the data speak. Setting `hyperpriors = 0` would drop this layer and reduce
the objective to the pure marginal likelihood; the default `hyperpriors = 1`
reproduces GLP's hierarchical prior.

---

## 4. The engine: marginal likelihood and two modes of inference

### 4.1 The marginal likelihood as a model‑selection criterion

For a fixed hyperparameter vector $\gamma$, integrating the VAR parameters out of
the likelihood gives the **marginal likelihood** (marginal data density)

$$
p(y \mid \gamma) \;=\; \int p(y \mid \beta, \Sigma)\, p(\beta, \Sigma \mid \gamma)\, \mathrm{d}\beta\, \mathrm{d}\Sigma,
$$

which under the conjugate NIW prior is available in closed form. This object is
the heart of GLP. Two properties make it the "right" criterion for choosing
shrinkage:

1. **It is a genuine out‑of‑sample criterion in disguise.** By the usual
   factorization $p(y\mid\gamma)=\prod_t p(y_t \mid y^{t-1}, \gamma)$, the
   marginal likelihood is the product of one‑step‑ahead *predictive* densities.
   Maximizing it therefore rewards a prior that would have predicted the sample
   well, not merely fit it.
2. **It has a built‑in Occam factor.** The marginal likelihood automatically
   penalizes priors that are too loose (too many effective degrees of freedom),
   so it trades off in‑sample fit against model complexity without any manual
   penalty. This is precisely why it can *select* the degree of shrinkage.

Combining it with the hyperprior gives the (log) **posterior of the
hyperparameters**, up to an additive constant:

$$
\log p(\gamma \mid y) \;=\; \log p(y \mid \gamma) \;+\; \log p(\gamma) \;+\; \text{const}.
$$

Everything below is a statement about how a given strategy *uses* this surface.

### 4.2 Two ways GLP turn the surface into forecasts

GLP describe (and this repo implements) two levels of Bayesian rigor:

- **ML‑II / empirical Bayes ("mode").** Maximize $p(\gamma\mid y)$ to get a
  single point estimate $\hat\gamma$, then condition on it as if it were known.
  Fast, and usually an excellent approximation for *point* forecasts.
- **Full hierarchical Bayes ("integrate out").** Treat $\gamma$ as genuinely
  uncertain: sample it from its posterior $p(\gamma\mid y)$ and average the
  predictive density over those draws. This propagates hyperparameter
  uncertainty into the forecast and produces better‑calibrated *density*
  forecasts.

The strategies in this repository differ along two axes: **which objective**
selects $\gamma$ (marginal likelihood vs. forecast loss) and **whether $\gamma$
is integrated out or plugged in** when forming the predictive density.

---

## 5. The four strategies

All four share the identical downstream forecasting machinery: given the selected
hyperparameters, the predictive density at each origin is built by drawing
$(\beta, \Sigma)$ from the conjugate posterior and iterating the VAR forward with
Gaussian innovations to simulate forecast paths $1,\dots,8$ quarters ahead. They
differ only in how $\gamma$ is chosen and (for the paper strategy) in whether
$\gamma$ is integrated out.

Output label mapping: `paper` → *GLP Marginal Likelihood*, `mango_mdd` →
*Mango MDD*, `mango_rmse` → *Mango RMSE*, `mango_rmse_random` →
*Mango RMSE Random*.

### 5.1 Strategy A — `paper`: GLP marginal‑likelihood selection (the benchmark)

**This is the faithful reproduction of GLP (2015).**

- **Objective.** The hyperparameters maximize the log posterior
  $\log p(\gamma\mid y) = \log p(y\mid\gamma) + \log p(\gamma)$ — the marginal
  data density combined with the Gamma hyperpriors.
- **Optimizer.** A local derivative‑free maximization of that surface
  (Nelder–Mead in this repo; GLP use `csminwel`). The result is the posterior
  **mode** $\hat\gamma$.
- **Predictive density — full hierarchical Bayes.** Rather than merely plugging
  in $\hat\gamma$, this strategy **integrates over hyperparameter uncertainty**
  with a **random‑walk Metropolis** sampler over $(\lambda,\psi,\theta,\mu)$:
  - the proposal covariance is the inverse observed‑information (the negative
    inverse Hessian of $\log p(\gamma\mid y)$ at the mode), regularized to be
    positive definite and scaled by a tuning constant;
  - at each retained draw of $\gamma$, a $(\beta,\Sigma)$ pair is drawn from the
    conditional NIW posterior and one forecast path is simulated.

  The resulting predictive density therefore integrates over **three** sources of
  uncertainty — the hyperparameters, the VAR parameters, and future shocks —
  which is exactly the GLP hierarchical predictive density.
- **Relation to the paper.** This *is* the paper's method, over the paper's full
  hyperparameter set $(\lambda, \psi, \theta, \mu)$. It is the **baseline**
  against which the other three strategies are measured (relative‑RMSE tables use
  `paper` as the denominator).

### 5.2 Strategy B — `mango_mdd`: same objective, global optimizer

**Question addressed: is GLP's local optimizer leaving marginal likelihood on the
table?**

- **Objective.** *Identical* to Strategy A: maximize the log marginal data
  density plus hyperprior, $\log p(\gamma\mid y)$. There is no change in the
  statistical criterion.
- **Optimizer.** Instead of a local gradient/simplex search, the surface is
  maximized with **Mango**, a parallel **Bayesian‑optimization** routine that
  fits a Gaussian‑process surrogate to the objective and uses an acquisition
  function to explore the bounded box
  $\lambda\in[10^{-4},5]$, $\theta\in[10^{-4},50]$, $\mu\in[10^{-4},50]$ and each
  $\psi_i\in[\mathrm{SS}_i/100,\ \mathrm{SS}_i\cdot 100]$ (the AR(1)‑based residual
  scale bounds). This is a **global**, derivative‑free search seeded with a handful of random
  evaluations (`init_points`) followed by guided iterations (`n_iter`).
- **Predictive density — plug‑in / ML‑II.** The hyperparameters are **fixed** at
  the located optimum $\hat\gamma$; $(\beta,\Sigma)$ are drawn from the
  conditional posterior and forecast paths simulated. Unlike Strategy A, this
  does **not** integrate over $\gamma$ — it is the empirical‑Bayes predictive
  density.
- **Relation to the paper.** Same target, different route to it. If the marginal
  likelihood is well‑behaved and unimodal, `mango_mdd` and `paper` should locate
  essentially the same $\hat\gamma$; systematic differences would flag either
  multimodality/ridges in the MDD surface or the practical cost of *not*
  integrating out $\gamma$. It isolates the **optimizer** and the
  **plug‑in‑vs‑integrate‑out** choices while holding the objective fixed.

### 5.3 Strategy C — `mango_rmse`: direct forecast‑accuracy selection

**Question addressed: does directly optimizing out‑of‑sample forecast accuracy
beat the marginal‑likelihood criterion?**

- **Objective.** A *fundamentally different*, decision‑theoretic criterion. For a
  candidate $\gamma$, forecasts are produced at a set of pseudo‑real‑time
  **evaluation origins** carved out of the current sample, and $\gamma$ is chosen
  to **minimize the root‑mean‑squared forecast error** of a chosen set of target
  variables at a chosen horizon $h$:

$$
\gamma^\star \;=\; \arg\min_{\gamma}\ \sqrt{\frac{1}{|\mathcal{O}|\,|\mathcal{V}|}\sum_{o\in\mathcal{O}}\sum_{v\in\mathcal{V}} \big(\hat y_{v,\,o+h}(\gamma) - y_{v,\,o+h}\big)^2 }.
$$

  Here $\mathcal{O}$ is a rolling set of the most recent valid origins (its size
  is `n_eval`), $\mathcal{V}$ is the set of target variables, and each forecast
  $\hat y(\gamma)$ is the posterior‑**mode** point forecast under $\gamma$. This
  is essentially **time‑series cross‑validation / a held‑out predictive‑loss
  criterion**.
- **Optimizer.** Mango Bayesian optimization again, this time **minimizing** the
  RMSE objective over the same bounded hyperparameter box.
- **Predictive density.** Hyperparameters fixed at the RMSE‑optimal
  $\gamma^\star$; $(\beta,\Sigma)$ drawn from the conditional posterior and paths
  simulated, exactly as in Strategy B.
- **Relation to the paper.** This is the principal **methodological contrast**
  with GLP. GLP's marginal likelihood optimizes a *global, one‑step, in‑sample*
  Bayesian criterion that is agnostic about which variable or horizon the
  forecaster cares about. The RMSE strategy instead **targets a specific loss,
  specific variables, and a specific horizon**. The comparison asks whether GLP's
  elegant, horizon‑agnostic criterion is left behind by brute‑force optimization
  of the loss the forecaster actually faces — and, if so, at what cost in
  robustness (a point‑forecast‑tuned prior need not deliver good density
  forecasts or good accuracy at other horizons/variables).
- **Practical notes for the write‑up.** Because the RMSE objective is expensive
  (it re‑estimates the model at several origins for every candidate $\gamma$), by
  default the hyperparameters are selected **once**, on the earliest real‑time
  origin, and then held fixed across the recursive experiment (a one‑time
  selection). The `--per-origin-selection` switch re‑optimizes at every origin.
  The repo also runs the optimization **separately per target horizon**
  (typically $h\in\{1,2,4,8\}$ quarters), so that each horizon gets its own
  loss‑minimizing prior; this is why the outputs contain `h1q`/`h2q`/`h4q`/`h8q`
  subdirectories.

### 5.4 Strategy D — `mango_rmse_random`: forecast‑accuracy selection on random origins

**Question addressed: how sensitive is the loss‑tuned prior to *which* evaluation
sample is used?**

- **Objective.** Identical in form to Strategy C, but the evaluation origins
  $\mathcal{O}$ are a **fixed random sample** drawn from the entire pool of valid
  origins, instead of the most recent contiguous block. A seed makes the sample
  reproducible; a minimum in‑sample length can be enforced.
- **Everything else** (Mango minimization, plug‑in predictive density, per‑
  horizon runs, one‑time selection by default) matches Strategy C.
- **Relation to the paper.** This is a **robustness variant** of the
  forecast‑accuracy approach. The rolling‑origin scheme in Strategy C weights the
  most recent (and possibly atypical, e.g. crisis‑era) part of the sample; random
  origins spread the evaluation across business‑cycle phases. Comparing C and D
  reveals how much the "optimal" shrinkage depends on the evaluation window — an
  instability that GLP's marginal‑likelihood criterion, being a single in‑sample
  quantity, does not suffer from. It sharpens the interpretation of any
  accuracy gains found in Strategy C.

### 5.5 Side‑by‑side summary

| | A. `paper` | B. `mango_mdd` | C. `mango_rmse` | D. `mango_rmse_random` |
| --- | --- | --- | --- | --- |
| **Objective** | Max log MDD + hyperprior | Max log MDD + hyperprior | Min out‑of‑sample RMSE | Min out‑of‑sample RMSE |
| **Statistical nature** | Marginal likelihood (Bayesian) | Marginal likelihood (Bayesian) | Predictive loss (decision‑theoretic) | Predictive loss (decision‑theoretic) |
| **Optimizer** | Local (Nelder–Mead / csminwel) | Global Bayesian opt. (Mango) | Global Bayesian opt. (Mango) | Global Bayesian opt. (Mango) |
| **Target of the objective** | Whole system, 1‑step, in‑sample | Whole system, 1‑step, in‑sample | Chosen variables & horizon | Chosen variables & horizon |
| **Evaluation origins** | — | — | Recent rolling block | Random sample of the pool |
| **$\gamma$ integrated out?** | **Yes** (RW‑Metropolis) | No (plug‑in) | No (plug‑in) | No (plug‑in) |
| **Selection frequency** | Every origin | Every origin | Once (or per origin) | Once (or per origin) |
| **Role in the study** | GLP benchmark | Objective held fixed, optimizer/plug‑in varied | Loss‑targeted alternative | Robustness of the loss‑targeted alternative |

The clean experimental design is the point: A vs. B isolates the *optimizer and
the integrate‑out step*; A/B vs. C isolates the *objective* (marginal likelihood
vs. forecast loss); C vs. D isolates the *evaluation sample*.

---

## 6. Real‑time recursive evaluation

All strategies are scored with a genuine **real‑time, recursive out‑of‑sample**
design, which is the standard in this literature and in GLP‑style forecast
horse‑races:

- **Real‑time vintages.** At each quarterly forecast origin the model sees only
  the data **vintage** that was actually available at that date (via ALFRED),
  so revisions and publication lags are respected and there is no look‑ahead.
- **Recursive (expanding) window.** The estimation sample grows as the origin
  advances; hyperparameters and coefficients are re‑estimated with the
  information available at each origin (subject to the one‑time‑selection caveat
  for the RMSE strategies).
- **Horizons.** Forecasts are made $1$ to $8$ quarters ahead.
- **Scoring vintage.** Realized values are taken from a single, fixed later
  vintage, so all strategies are graded against the same "truth".
- **Loss.** Accuracy is summarized by **RMSE** per variable and horizon, and by
  **relative RMSE** against the `paper` (GLP marginal‑likelihood) baseline; a
  value below $100\%$ means the alternative beats GLP.
- **Transform space.** Errors are computed on the model's transformed space
  ($100\times\log$ levels for log variables; levels for rates), matching the
  space in which the BVAR is estimated.

Density‑forecast objects (predictive quantiles) are also stored, so the study can
compare not just point accuracy but calibration — the dimension on which the
`paper` strategy's integrate‑out step is expected to matter most.

### 6.1 Selection scopes and the scope-grid runner

The Stage 6 workflow adds a **first-class scope-grid runner**,
`scripts/glp/run_glp_scope_grid.py`, for experiments that intentionally vary
*which targets share one hyperparameter vector* and *how often that vector is
retuned*. This is distinct from the legacy `paper` / `mango_*` scripts, which
remain available for backward compatibility and preserve their historical,
low-budget defaults.

The scope-grid runner works with four core selection scopes plus grouped cells:

- `pooled`: one hyperparameter vector is selected for every requested variable
  and horizon.
- `horizon`: one vector per horizon, shared across variables.
- `variable`: one vector per variable, shared across horizons.
- `variable_horizon`: one vector per `(variable, horizon)` pair.
- `group`: one vector per user-supplied variable group, optionally split by
  horizon.

The key output is the **canonical stitched forecast panel**. Each target
`(variable, horizon)` is assigned to exactly one selection cell by the chosen
scope. The runner then:

1. selects one system-wide hyperparameter vector for each active cell at each
   retuning event,
2. generates the full-system VAR forecast under that cell's selected vector,
3. stitches one canonical forecast row per requested target from the
   responsible cell, and
4. optionally saves `forecast_panel_all_cells.csv` as a diagnostic panel of the
   full-system forecasts for every cell.

This distinction matters empirically. A `variable_horizon` experiment can target
GDP at horizon four with a different shrinkage vector than GDP at horizon one,
while the final `forecast_panel.csv` still contains exactly one canonical row per
requested target and origin.

### 6.2 Legacy runners versus the study runner

The repository now has two GLP entry-point families:

- **Legacy strategy runners** (`run_glp_paper.py`, `run_glp_mango.py`,
  `run_glp_mango_rmse.py`, `run_glp_mango_rmse_random.py`): preserve the
  original single-cell workflows and their historical budgets for backward
  compatibility.
- **Study runner** (`run_glp_scope_grid.py`): validates the full experiment
  configuration up front, prints and saves a deterministic manifest, writes one
  self-contained run directory per scope, supports explicit retuning schedules,
  and defaults to the **recommended reduced search** that optimizes
  `lambda`, `theta`, and `miu` while holding `psi` fixed.

That reduced search is deliberate. Estimating `psi` makes the problem
$(3+n)$-dimensional, which is often too expensive for a broad scope study unless
the optimizer budget is raised substantially. The study runner therefore treats
`--no-optimize-psi` with `--fixed-psi-source context_ss` as the default design,
and warns prominently when a requested dimension-budget combination is too weak
to support scientific claims.

### 6.3 Recommended study examples

Recommended reduced-search scope study:

```bash
python scripts/glp/run_glp_scope_grid.py \
  --output-root outputs/glp/scope_study \
  --model-size medium \
  --selection-scopes pooled,horizon,variable,variable_horizon \
  --target-variables GDP,DEFL,FFR \
  --target-horizons 1,2,4,8 \
  --loss-metric rmse \
  --loss-scaling benchmark_rmse \
  --benchmark last_observation \
  --inner-n-origins 20 \
  --inner-origin-stride 2 \
  --selection-frequency 4 \
  --no-optimize-psi
```

Grouped-variable study with one residual block:

```bash
python scripts/glp/run_glp_scope_grid.py \
  --output-root outputs/glp/group_scope_study \
  --model-size medium \
  --selection-scopes group \
  --target-variables GDP,DEFL,FFR,CONS,INV \
  --target-horizons 1,4,8 \
  --variable-groups Real=GDP+CONS+INV;Prices=DEFL \
  --residual-group-name Rates \
  --group-separate-horizons \
  --selection-frequency per_origin \
  --no-optimize-psi
```

For a planning pass that requires **no optional Bayesian backend**, add
`--dry-run`. The runner still validates the configuration, estimates the number
of optimization cells and candidate evaluations, prints the manifest, and writes
per-scope run directories with manifests only.

---

## 7. Model size and the central GLP finding

The variable universe is **nested**, mirroring GLP's small/medium/large design:

- **small (3):** GDP, GDP deflator, Fed funds rate — the canonical monetary VAR.
- **medium (7):** adds consumption, investment, hours, and real wages — a
  Smets–Wouters‑style block.
- **large (~21):** adds labour, price, money/credit, and financial variables.

This nesting is not incidental; it is the vehicle for GLP's headline result. GLP
show that the marginal‑likelihood‑optimal $\lambda$ **falls as $n$ rises** — the
system optimally shrinks harder in higher dimensions — and that, with that
data‑driven shrinkage, the **large** BVAR forecasts as well as or better than the
small one and than factor models. Running each strategy across the three sizes
lets the paper (i) reproduce the $\hat\lambda$‑vs‑$n$ relationship for the
marginal‑likelihood strategies, and (ii) ask whether the loss‑targeted strategies
choose systematically different shrinkage as dimension grows.

---

## 8. Suggested paper narrative

A natural structure for a paper built on this repository:

1. **Motivation.** Prior selection is the binding constraint on large BVAR
   forecasting; GLP solved it with the marginal likelihood. But forecasters
   ultimately care about a *loss*, not a marginal likelihood. Do the two agree?
2. **Framework.** Present the conjugate BVAR, the Minnesota + dummy‑observation
   prior, and the hierarchical hyperprior (Sections 2–3). State the marginal
   likelihood and the mode‑vs‑integrate‑out distinction (Section 4).
3. **Strategies.** Define the four selection rules (Section 5) as a $2\times2$
   design: *objective* (marginal likelihood vs. forecast loss) $\times$
   *implementation* (local/integrated vs. global/plug‑in), plus the random‑origin
   robustness arm.
4. **Data and evaluation.** Real‑time recursive design across three model sizes
   (Sections 6–7).
5. **Results.** (i) $\hat\gamma$ paths and their dispersion across strategies;
   (ii) the $\hat\lambda$‑vs‑dimension relationship; (iii) RMSE and relative‑RMSE
   tables by variable and horizon; (iv) density‑forecast calibration, where the
   integrate‑out step of the `paper` strategy should pay off.
6. **Interpretation.** Whether the elegant, horizon‑agnostic marginal likelihood
   is competitive with brute‑force loss minimization; how stable the loss‑tuned
   prior is (C vs. D); and the practical recommendation for applied forecasters.

**Candidate research questions.**

- Does the marginal‑likelihood prior (`paper`) match a global optimizer on its
  own objective (`mango_mdd`), and does integrating out $\gamma$ improve density
  calibration?
- Can horizon‑ and variable‑specific RMSE tuning (`mango_rmse`) beat GLP on the
  targeted variable/horizon — and does it *lose* elsewhere (other variables,
  other horizons, density scores)?
- How much does the RMSE‑optimal shrinkage move when the evaluation origins
  change (`mango_rmse` vs. `mango_rmse_random`)? Is the marginal‑likelihood
  prior more stable by construction?
- Does the optimal shrinkage rise with dimension under *all* strategies, or only
  under the marginal‑likelihood criterion?

**Expected/plausible priors on the outcome (to be tested).** GLP's own results,
and the fact that the marginal likelihood is itself a predictive criterion,
suggest `paper` and `mango_mdd` should be very close and hard to beat on average;
loss‑tuned strategies may win narrowly on their targeted variable/horizon but at
the cost of robustness across horizons, variables, and density calibration — the
classic tension between a coherent Bayesian criterion and direct loss
minimization.

---

## 9. Notation and glossary

| Symbol / term | Meaning |
| --- | --- |
| $n$, $p$ | Number of variables; number of lags (here $p=5$). |
| $\beta, \Sigma$ | VAR coefficients; innovation covariance. |
| $\gamma=(\lambda,\psi,\theta,\mu)$ | The estimated hyperparameter vector ($\psi$ is an $n$‑vector). |
| $\lambda$ | Overall Minnesota tightness (global shrinkage). |
| $\theta$ | Tightness of the dummy‑initial‑observation / single‑unit‑root (co‑persistence) prior (code switch `sur`). |
| $\mu$ | Tightness of the sum‑of‑coefficients / no‑cointegration prior (code switch `noc`). |
| $\psi$ | Residual‑variance scales (diagonal of the IW scale matrix); **estimated** ($n$‑vector, `MNpsi=1`). |
| $\alpha$ | Minnesota lag‑decay exponent; **fixed at 2** (the paper's $1/s^2$, `MNalpha=0`). |
| MDD / marginal likelihood | $p(y\mid\gamma)=\int p(y\mid\beta,\Sigma)p(\beta,\Sigma\mid\gamma)\,\mathrm{d}\beta\,\mathrm{d}\Sigma$. |
| Hyperprior | $p(\gamma)$; Gamma priors on $\lambda,\theta,\mu$ and an inverse‑Gamma on $\psi$. |
| ML‑II / empirical Bayes | Selecting $\gamma$ by maximizing $p(\gamma\mid y)$ and conditioning on the mode. |
| NIW | Normal–Inverse‑Wishart, the conjugate prior/posterior family for $(\beta,\Sigma)$. |
| Dummy observations | Artificial data rows that implement a prior while preserving conjugacy. |
| Mango | A parallel Bayesian‑optimization library (Gaussian‑process surrogate + acquisition function) used as the global optimizer for the `mango_*` strategies. |
| Relative RMSE | $100\times \text{RMSE}_{\text{strategy}}/\text{RMSE}_{\text{paper}}$; $<100$ beats GLP. |

---

## 10. Caveats and honest limitations (for the methods/robustness section)

- **Hyperparameter set matches the paper.** The search covers $\lambda$, $\theta$,
  $\mu$ and the full residual‑variance vector $\psi$ (`MNpsi=1`); only the
  Minnesota lag‑decay $\alpha$ is fixed at $2$ (`MNalpha=0`), which the paper also
  does implicitly through its $1/s^2$ decay.
- **Computational cost of estimating $\psi$.** With $\psi$ estimated the search is
  $(3+n)$‑dimensional — $6$/$10$/$24$ dimensions for the small/medium/large
  models. This makes the Mango Bayesian optimization and the random‑walk
  Metropolis / observed‑information Hessian markedly heavier for the larger
  models. Any study that turns `--optimize-psi` back on should therefore raise
  the optimizer budget well above the legacy settings. The scope-grid runner
  uses a reduced-search default precisely because the full $\psi$ search is
  rarely defensible under a modest scope-study budget.
- **Plug‑in vs. integrate‑out is confounded with the objective.** Only the
  `paper` strategy integrates $\gamma$ out. If a difference appears between
  `paper` and `mango_mdd`, it may stem from the integrate‑out step rather than
  from the optimizer; a clean ablation would add a plug‑in‑at‑the‑mode variant of
  the marginal‑likelihood strategy.
- **One‑time selection for the RMSE strategies.** By default the loss‑tuned
  hyperparameters are chosen once and frozen, for compute reasons; this is not a
  fully recursive selection and should be stated (use `--per-origin-selection`
  for the fully recursive version).
- **MCMC size.** The recursive default uses a modest number of predictive draws
  for tractability; paper‑faithful density forecasts require raising it toward
  GLP's $20{,}000$ draws.
- **Data proxies.** A few large‑model series have limited real‑time ALFRED
  coverage, and the equity block uses the public OECD U.S. share‑price index as
  an explicit proxy for the S&P 500. Quantitative large‑model results should be
  read with this in mind.
- **Prior‑name bookkeeping.** As set out in Section 3.2, the estimated
  hyperparameters map to $\theta\to$ single‑unit‑root / dummy‑initial‑observation
  and $\mu\to$ sum‑of‑coefficients / no‑cointegration. This structural
  identification (verified against the dummy‑observation construction in
  `covbayesvar`) is the one to use when interpreting results; the code comments
  now reflect it.

---

## 11. References

- Giannone, D., Lenza, M., and Primiceri, G. E. (2015). *Prior Selection for
  Vector Autoregressions.* Review of Economics and Statistics, 97(2), 436–451.
- Litterman, R. B. (1979). *Techniques of Forecasting Using Vector
  Autoregressions.* Federal Reserve Bank of Minneapolis Working Paper.
- Doan, T., Litterman, R., and Sims, C. (1984). *Forecasting and Conditional
  Projection Using Realistic Prior Distributions.* Econometric Reviews, 3(1),
  1–100. *(Sum‑of‑coefficients / no‑cointegration prior.)*
- Sims, C. A. (1993). *A Nine‑Variable Probabilistic Macroeconomic Forecasting
  Model.* In Business Cycles, Indicators, and Forecasting, NBER. *(Dummy‑initial‑
  observation / single‑unit‑root prior.)*
- Sims, C. A., and Zha, T. (1998). *Bayesian Methods for Dynamic Multivariate
  Models.* International Economic Review, 39(4), 949–968.
- Kadiyala, K. R., and Karlsson, S. (1997). *Numerical Methods for Estimation and
  Inference in Bayesian VAR Models.* Journal of Applied Econometrics, 12(2),
  99–132. *(Conjugate NIW VAR.)*
- Stock, J. H., and Watson, M. W. (2008). *Phillips Curve Inflation Forecasts.*
  *(Monthly‑to‑quarterly aggregation convention used for the data.)*
