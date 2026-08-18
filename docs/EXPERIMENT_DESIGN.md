# Experiment Design

This document explains every design concept that governs the
hyperparameter-selection experiments in this repository. It is the
authoritative reference for the choices encoded in the scope-grid runners
(`scripts/regularized_var/run_ridge_scope_grid.py`,
`scripts/glp/run_glp_scope_grid.py`,
`scripts/mfvar/run_mfvar_scope_grid.py`) and the shared
`src/common_hpo` infrastructure.

---

## 1. Research question

Two foundational papers propose opposite strategies for hyperparameter
selection in Bayesian VARs:

* **Giannone, Lenza, and Primiceri (2015)** — *Prior Selection for Vector
  Autoregressions*, ReStat 97(2) — maximize the marginal likelihood (MDD)
  to select the Minnesota-prior hyperparameters. This is the **native**
  selection strategy.

* **Schorfheide and Song (2015)** — *Real-Time Forecasting With a Mixed-
  Frequency VAR*, JBES 33(3) — choose hyperparameters by trial-and-error
  or informal calibration against forecast data.

The central question this repository investigates is:

> **Does selecting hyperparameters by out-of-sample forecast loss
> produce better real-time macroeconomic forecasts than the marginal-
> likelihood rule, or is the native GLP/Bayesian selection already
> forecast-optimal?**

The same model, the same data, and the same real-time evaluation design
are used throughout; only the hyperparameter objective changes across
experimental arms.

---

## 2. Native selection vs. forecast-loss selection

| Criterion | Native (marginal likelihood / MDD) | Forecast-loss |
|---|---|---|
| Objective | Marginal log-likelihood of observed data | Mean squared (or absolute) error on held-out validation origins |
| Scope of information | Full in-sample history at each origin | Subset of pseudo-real-time validation splits |
| Theoretical motivation | Bayesian model comparison; automatic Occam's razor | Direct proxy for out-of-sample performance |
| Computational cost | Closed-form posterior → cheap; MDD needs MCMC | Grid search or Bayesian optimizer over inner splits |
| Implementation | `run_glp_scope_grid.py --benchmark none` (native / paper strategy is a separate legacy script) | All scope-grid runners via `--loss-metric rmse` |

In the ridge-VAR arm there is no closed-form native-selection criterion, so
only forecast-loss selection is offered. In the GLP arm both strategies are
present: the native MDD mode uses the `covbayesvar` log-MDD and a Mango
Bayesian optimizer, while the forecast-loss mode uses inner RMSE splits and
either a Mango optimizer or the deterministic ridge grid.

---

## 3. Selection scopes

A **selection scope** defines how forecast targets are grouped for the purpose
of choosing one set of hyperparameters.

| Scope | Cell definition | Selection events | Example cells |
|---|---|---|---|
| `pooled` | All targets share one cell | 1 per selection event | `pooled` |
| `horizon` | One cell per target horizon | H cells | `h1`, `h4`, `h8` |
| `variable` | One cell per target variable | V cells | `gdp`, `inv`, `cons` |
| `variable_horizon` | One cell per (variable, horizon) pair | V×H cells | `gdp-h1`, `gdp-h4`, `inv-h1` … |

All scopes are available in every runner via `--selection-scopes
pooled,horizon,variable,variable_horizon` (comma-separated list).

The scope-gain tables in the comparison output decompose RMSE reductions into
additive components (horizon gain, variable gain, interaction).

---

## 4. Why variable-specific tuning still estimates the full system

This is the most important conceptual subtlety: even when you choose the scope
`variable`, the VAR is **always estimated on all variables jointly**. The
selection scope affects only which combination of residuals enters the
validation loss that drives hyperparameter selection; it never changes the
model or the data that are passed to the estimator.

Concretely:

1. Suppose the target is `gdp` and the scope is `variable`.
2. The estimator receives the full multivariate panel (gdp, inv, cons, …).
3. The posterior is computed for the joint system.
4. Hyperparameters are chosen to minimize the inner RMSE of `gdp` predictions
   only.
5. At forecast time, the model again produces joint forecasts for all
   variables, but only the `gdp` column enters the reported evaluation.

This design preserves the econometric soundness of the Bayesian VAR
(which relies on the joint covariance structure) while still allowing
variable-specific weighting of the selection objective.

---

## 5. Inner vs. outer evaluation

The evaluation design has two nested layers to prevent look-ahead bias.

```
  Outer layer (reporting):   [train .......... | test_T ]  (one row per origin)
  Inner layer (selection):   [train . | val_T ]            (run at each selection event)
```

### Outer origins
Defined by `--outer-n-origins`, `--outer-origin-stride`, and
`--outer-origin-selection`. Each outer origin `t` generates a forecast for
horizons `h = 1, …, H` using all data up to and including `t`. The outer
test targets at horizons `h` are `y_{t+h}`, which must be in the panel.
**No outer test target ever enters any estimator**, including the inner-layer
scaling stats.

### Inner origins (selection splits)
Defined by `--inner-n-origins`, `--inner-origin-stride`,
`--inner-origin-selection`, and `--inner-window`. Each inner origin `s`
is strictly earlier than the outer test cutoff for the current outer origin.
The inner training window runs up to `s`; the inner holdout is `y_{s+1}, …,
y_{s+H}`. When scaling is enabled, it is fitted only on the inner training
window, never on data past `s`.

The shared `build_validation_splits` in `src/common_hpo/splits.py` enforces
this invariant. Unit tests in `tests/test_glp_scope_grid.py` and
`tests/test_ridge_experiment.py` verify it.

---

## 6. Vintage policies

A **vintage policy** specifies which version of each time series to use at
each forecast origin when the data are real-time (subject to revision).

| Policy | Meaning |
|---|---|
| `outer_vintage_consistent` | Each outer origin uses the vintage of the data that was available at that date. Inner splits within the same outer origin use the same vintage. This is the most realistic pseudo-real-time design. |
| `latest` | All origins use the most recent vintage. Ignores publication lags; useful for benchmarking data-revision effects. |

The ridge and MFVAR runners receive pre-loaded panels from the calling script;
the policy is recorded in `run_metadata.json` but is not enforced in the
runner. The GLP runner loads the panel once; the user is responsible for
supplying a point-in-time consistent panel when running the real experiment.

---

## 7. Selection schedules

A **selection schedule** determines at which outer origins the inner
validation is re-run and new hyperparameters are selected. Between selection
events, the last selected hyperparameters are reused.

| Schedule token | Meaning |
|---|---|
| `once` | Run inner validation once at the earliest feasible origin, then reuse those hyperparameters for all remaining outer origins. |
| `per_origin` | Re-run inner validation at every outer origin. |
| `annual_quarterly` | Re-run at the first origin of each calendar year. |
| `N` (integer) | Re-run every N outer origins. |

Specify via `--selection-frequency`. The `SelectionSchedule` in
`src/common_hpo/schedules.py` resolves the schedule and records its
interpretation in `run_metadata.json`.

---

## 8. Loss scaling

Raw validation loss aggregates error magnitudes across variables, which may
have very different variances. Loss scaling normalizes the loss before
aggregation.

| Mode | Effect |
|---|---|
| `none` | Raw RMSE, no normalization. Favours low-variance variables. |
| `target_std` (ridge) | Divide each variable's loss by its training-window standard deviation before averaging. Makes losses comparable across variables. |
| `benchmark_rmse` (GLP, MFVAR) | Divide by the RMSE of the selected benchmark (e.g., `no_change`, `last_observation`). Produces a relative loss. |

Specify via `--loss-scaling`. The ridge runner supports `none` and
`target_std`; GLP and MFVAR support `none` and `benchmark_rmse`.

---

## 9. Full vs. reduced GLP search

The GLP hyperparameter vector has four components: λ (overall tightness),
θ (sum-of-coefficients), μ (cross-variable tightness), and ψ (scale of
the IW prior's initial covariance).

| Mode | Optimized | Fixed | CLI flag |
|---|---|---|---|
| Full | λ, θ, μ, ψ | — | `--optimize-psi` |
| Reduced (default) | λ, θ, μ | ψ from in-sample steady-state | `--no-optimize-psi` (default) |

The reduced search is preferred because:

1. ψ is usually well-identified by the steady-state covariance of the data,
   which can be estimated without an optimizer.
2. Fixing ψ reduces the search dimension from 4 to 3, which requires fewer
   candidate evaluations for the same coverage quality.
3. GLP (2015) themselves treat ψ as derived from the data rather than freely
   optimized in their baseline specification.

The `--fixed-psi-source` flag controls whether ψ is taken from the
context-level steady state (`context_ss`, default) or supplied manually
(`supplied`, requires `--fixed-psi-values`).

---

## 10. Mixed-frequency forecast variables vs. objective variables

This distinction applies only to the MF-BVAR (`mfvar`) runner.

The MF-BVAR requires a **complete quarterly forecast block** to build its
internal mixed-frequency state representation. The block must include all
quarterly series (GDP, investment, government consumption, etc.) that the
model was designed to handle, regardless of which variables the research
question targets.

However, the **selection objective** can be restricted to a subset — for
example, GDP only — so that the inner RMSE loss is computed purely on
GDP forecasts while the model is still conditioned on the full joint system.

| Parameter | Meaning | CLI flag |
|---|---|---|
| Forecast variables | Full quarterly block passed to the MF-BVAR state | `--forecast-variables GDP,INVFIX,GOV,...` |
| Target variables | Subset used for the inner RMSE and outer reporting | `--target-variables GDP` |
| Objective horizon | Quarterly horizons used in the inner RMSE | `--optimization-horizon-quarters 1,4` |

Specifying fewer target variables than forecast variables does **not** change
the model — it only changes which residuals contribute to the selection loss.

---

## 11. Iterated vs. direct ridge forecasts

The ridge VAR runner (`run_ridge_scope_grid.py`) supports two forecast
architectures:

| Mode | How `h`-step forecast is produced | CLI flag |
|---|---|---|
| `iterated` | Fit one VAR(p) on the transformed series. Compute h-step forecast by iterating the companion form h times. | `--forecast-method iterated` |
| `direct` | Fit a separate regression of `y_{t+h}` on `(y_t, …, y_{t-p+1})` for each horizon h. No companion-form iteration. | `--forecast-method direct` |

Iterated forecasts are consistent with the VAR's own dynamics but can
compound misspecification errors. Direct forecasts are robust to short-
horizon misspecification but treat each horizon as an independent problem
(losing cross-horizon parameter sharing). The `direct` family appears as
`ridge_direct` in comparison tables to prevent pooling with iterated results.

---

## 12. Canonical output schemas

See [docs/OUTPUT_SCHEMA.md](OUTPUT_SCHEMA.md) for the full column-level
specification of every output file.

---

## 13. Known limitations

| ID | Area | Description |
|---|---|---|
| L-01 | Ridge / real data | `forecast_origin` and `target_quarter` are stored as integer row indices when the panel CSV has no date column. Real-data runs should supply a date column for interpretable outputs. |
| L-02 | Ridge | `actual_level` is always NaN because ridge operates on pre-transformed series and does not carry a back-transform. Only `actual_metric` (the transformed actual) is valid. |
| L-03 | All | There is no direct foreign-key column in `forecast_panel.csv` linking rows to `event_id` in `selected_hyperparameters.csv`. The linkage is implicit through the selection schedule recorded in `run_metadata.json`. |
| L-04 | GLP | The `update_hyperparameters_mango_rmse(_random)` functions are not yet exported from `glp_model.py`, blocking legacy RMSE strategies. The scope-grid runner (which uses the internal `evaluate_glp_candidate` path) is not affected. |
| L-05 | GLP / MFVAR | Both depend on optional packages (`covbayesvar`, `MBFVAR`). Missing dependencies are reported as collection-time skips under `--skip-optional`. |
| L-06 | MFVAR | `MBFVAR` creates fresh unseeded NumPy generators internally, so MFVAR results are not fully bit-reproducible across runs even with a fixed `--base-seed`. |
| L-07 | Comparison | The `compare_scope_study.py` comparison CLI requires the `family` field in the manifest to be a known value (`ridge`, `ridge_direct`, `glp`, `glp_legacy`, `mfbvar`, `paper_mf`, `minnesota`). Benchmark panels must use the same family as their corresponding tuned panel. |
