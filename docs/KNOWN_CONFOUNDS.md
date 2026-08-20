# Known Confounds and Threats to Validity

This document records design and implementation issues that a code review has
**confirmed** in the current state of the repository. They are threats to the
validity of the headline comparison — hyperparameter selection by marginal data
density / log posterior versus selection by out-of-sample forecast loss — and
should be read before any result from this repository is interpreted as
evidence about that comparison.

Each entry states the confound, the confirmed mechanism in the code, and the
change that would remove it.

---

## C-01. The MDD arm and the forecast-loss arm are not run under comparable conditions

The two arms currently differ in at least three ways that are unrelated to the
selection objective.

**Search budget and search dimension.**

| Arm | Evaluations | Search dimension |
|---|---|---|
| Legacy MDD path | `init_points=5` + `n_iter=15` = **20** | `3 + n` (ψ is always in the MDD search space because `GLP_MNPSI=1`): **10** for `medium`, **25** for `large` |
| Scope-grid forecast-loss path | `24` initial + `72` iterations = **96** | **3** (`--no-optimize-psi` is the default, so ψ is fixed) |

The forecast-loss arm therefore receives roughly five times the evaluations in
a search space an order of magnitude smaller. Coverage of the respective spaces
is not remotely equal.

**Selection schedule.**

| Arm | Re-selection frequency |
|---|---|
| Legacy RMSE | selected **once**, at the earliest forecast origin |
| Scope-grid RMSE | re-selected every **4** origins |
| MDD | re-selected at **every** origin |

The MDD arm adapts its hyperparameters to each origin; the legacy RMSE arm does
not adapt at all. Any difference in forecast accuracy over the evaluation
sample mixes the selection objective with the amount of adaptation allowed.

**Forecast construction.**

| Arm | Reported forecast |
|---|---|
| MDD | mean of simulated paths **with Gaussian innovations** |
| Scope-grid forecast loss | posterior-mean **point** forecast (default 25 draws), **no innovations** |

These are different estimators of the predictive distribution's centre and
carry different Monte Carlo error, independently of which hyperparameters were
chosen.

**Consequence.** An observed RMSE difference between the arms is **not
attributable to the selection objective alone.** Budget, dimension, schedule
and forecast construction are all confounded with it, and their combined effect
is plausibly of the same order as (or larger than) the effect of interest.

**Recommendation.** Run any headline comparison through **one** harness with an
identical parameter space, evaluation budget, selection schedule, inner fold
structure and forecast construction, varying **only** the objective:
`maximize(log posterior)` versus `minimize(forecast loss)`. The legacy paths
may remain for replication of prior work, but should not be used as the two
sides of the main claim.

---

## C-02. The "MDD" arm maximizes a log posterior, not a pure marginal data density

With hyperpriors enabled — which is the default — the objective maximized by
the "MDD" arm includes the Gamma / Inverse-Gamma hyperprior density terms on
the hyperparameters in addition to the marginal likelihood of the data.

This is consistent with the hierarchical treatment in Giannone, Lenza and
Primiceri (2015), where the hyperparameters carry proper priors and are chosen
at the mode of their posterior. It is **not** identical to the marginal data
density, and labelling it "MDD" overstates what is being maximized: with
hyperpriors on, the criterion is a penalized objective whose maximum can differ
from the marginal-likelihood maximum.

**Recommendation.** Refer to this arm as the **log posterior** or
**hierarchical criterion**, or state explicitly that "MDD" in this repository
is defined to include the hyperprior terms. Report whether hyperpriors were
enabled for every run; without that flag the objective is ambiguous.

---

## C-03. Selection-scope granularity is confounded with compute

The optimizer budget is applied **per cell**, not per run. A scope with
`n_cells` cells therefore consumes `n_cells` times the total number of model
fits that the pooled scope consumes. The `variable_horizon` scope, being the
finest, receives the largest total budget by a wide margin.

Consequently a "finer scope wins" result is **partly a budget effect**: the
finer scope has searched more of the hyperparameter space in aggregate, quite
apart from whether cell-specific hyperparameters are genuinely better.

**Recommendation.** Report **total model fits per scope** alongside any
scope-gain table, and — where feasible — include a compute-matched comparison
in which the pooled scope is given the same total number of evaluations as the
finest scope.

---

## C-04. The inner selection loss and the outer reported metric are not the same functional

The default inner loss scaling is `benchmark_rmse`: forecast errors inside the
selection objective are divided by the RMSE of an inner random-walk benchmark
before aggregation. The reported outer table is **raw per-target RMSE**.

- For **single-cell** scopes (one variable, one horizon), the benchmark divisor
  is a single positive constant, so the inner objective is a monotone
  reparameterization of the outer metric. The selected hyperparameters are
  unaffected. This case is harmless.
- For **pooled**, **horizon** and **variable** scopes, the optimized cell spans
  several variables and/or horizons, each with its own benchmark divisor.
  Scaling changes the **relative weight** of variables and horizons inside the
  aggregate that is optimized, relative to the aggregate that is reported. The
  selection can therefore be optimal for a weighting that the results table
  does not use.

**Recommendation.** Either report the scaled loss alongside raw RMSE, or run
the multi-cell scopes with `--loss-scaling none` for the headline table so that
the inner and outer functionals coincide.

---

## C-05. Single-seed practice

Multi-seed replication **is supported**: `scripts/run_scope_study.py` loops over
a list of seeds. The shipped `configs/paper_experiment.json`, however, specifies
a **single** seed, `[20150101]`.

The optimizers used here are stochastic. With one seed, the run-to-run
variability of the optimizer is fully confounded with the effect of the
selection rule, and a difference between arms cannot be distinguished from
optimizer noise.

**Recommendation.** Use **at least 10 seeds** for any reported comparison and
report dispersion (standard deviation or an interval across seeds), not just
the point estimate. Differences smaller than across-seed dispersion should not
be described as differences.

---

## C-06. Reproducibility caveat

`experiment_provenance.deterministic_rng_context` obtains determinism by
**monkey-patching `numpy.random.default_rng` process-globally** for the duration
of the context. It seeds the legacy global NumPy and Python RNGs and makes
unseeded `default_rng()` calls draw successive child `SeedSequence`s from the
run seed, restoring process-global state on exit.

Two consequences follow:

- The patch is **thread-unsafe by its own docstring** and must not be entered
  concurrently from overlapping threads. Process-level workers are isolated and
  therefore safe; thread-level parallelism is not.
- It is a **workaround** for an upstream package that exposes no seed path, not
  a supported seeding interface. It depends on where and how that package calls
  `default_rng`, which can change with the package version.

**Bit-for-bit reproduction is therefore not guaranteed** across environments,
package versions, or worker configurations. Reported results should be treated
as reproducible up to Monte Carlo error, and seeds/versions should be recorded
with every run.
