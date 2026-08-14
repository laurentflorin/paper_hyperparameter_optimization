# Review of the GLP `mango_rmse` Hyperparameter-Selection Approaches

**Scope.** This report reviews the two RMSE-based GLP hyperparameter strategies
— `mango_rmse` (contiguous tail origins) and `mango_rmse_random` (random
origins) — and answers three questions:

1. Do they actually use *real out-of-sample* forecasts to optimise the
   hyperparameters?
2. Does the RMSE approach, as currently implemented, make sense?
3. Would it be better to instead generate *pseudo out-of-sample forecasts using
   posterior draws* and compute the RMSE on those?

All code references are to the current tree under
[src/glp_hyperparameter_optimization](src/glp_hyperparameter_optimization) and
[scripts/glp](scripts/glp).

---

## TL;DR

- **Q1 — Real out-of-sample?** *Yes, in the strict sense.* The evaluation
  targets are genuinely held out of the estimation window (the model is
  re-estimated on `y[:cut]` and scored against `y[cut:cut+H]`), so the objective
  is a true pseudo-out-of-sample (POOS) forecast-error criterion. **But** the
  evaluation is thin and clustered: by default only 3 heavily-overlapping origins
  at the very end of one sample, and the whole selection is done **once** on the
  earliest forecast origin and then frozen.
- **Q2 — Does it make sense?** *Mostly yes, but there is one real internal
  inconsistency.* The objective scores the **posterior-mode point forecast**
  (`bvarFcst` at the mode of β), whereas the forecasts that are ultimately
  reported and scored in the comparison tables are the **mean of the posterior
  predictive draws**. You therefore tune the prior for a slightly different
  forecast object than the one you actually evaluate.
- **Q3 — Use posterior draws instead?** *Yes, this is a sound improvement* — with
  an important nuance. Scoring the **predictive mean** (β-averaged forecast)
  removes the mode-vs-mean inconsistency and matches the reported metric. The
  subtlety is that, for a *point* RMSE, simulating shock paths adds only Monte
  Carlo noise; what actually matters is averaging the forecast over the
  **posterior draws of β**. If the real goal is to reward the *density* (not just
  the point), RMSE is the wrong metric and a proper scoring rule (CRPS or mean
  log predictive score) is the better "use-the-draws" upgrade.

---

## 1. What the code actually does

### 1.1 Building the evaluation origins

Origins are constructed in
[`_rmse_eval_origins`](src/glp_hyperparameter_optimization/glp_model.py#L681) and
[`_build_rmse_origins`](src/glp_hyperparameter_optimization/glp_model.py#L712).
For a sample of length `T`, forecast horizon `H`, and origin index `k`:

```
cut    = T - H - k
train  = y[:cut, :]          # estimation window
actual = y[cut:cut+H, :]     # H held-out future quarters
ctx    = prepare_glp_context(train, lags, ...)
```

The held-out `actual` rows are **not** in `train`, and the forecast is seeded
from the end of `train`, so the horizon-`h` forecast row aligns correctly with
`actual[h-1]`. This is a correct POOS split.

- `mango_rmse` uses the **contiguous most-recent** origins `k = 0, 1, …, n_eval-1`
  (default `n_eval = 3` → `k ∈ {0,1,2}`), i.e. `cut ∈ {T-H, T-H-1, T-H-2}`.
- `mango_rmse_random` draws `n_eval` origins **at random** from the whole valid
  pool (`random=True`), which spreads them across the sample.

### 1.2 The objective: posterior-mode point forecast

The objective is
[`_rmse_objective`](src/glp_hyperparameter_optimization/glp_model.py#L733):

```python
betahat, _ = glp_mode_estimate(ctx, vec)          # posterior MODE of beta
forecast   = point_forecast(ctx.y, betahat, horizons)  # deterministic bvarFcst
error      = forecast[row, vi] - actual[row, vi]
# rmse = sqrt(mean(error**2))
```

- [`glp_mode_estimate`](src/glp_hyperparameter_optimization/glp_model.py#L385)
  returns the **posterior-mode** `betahat` (via `logMLVAR_formin`).
- [`point_forecast`](src/glp_hyperparameter_optimization/glp_model.py#L401) is a
  **deterministic** recursive forecast (`bvarFcst`) — no draws, no shocks.
- Only the single horizon `h_eval` is scored (the driver sets
  `optimization_eval_horizon_quarters = horizon`), and each horizon in
  `{1,2,4,8}` gets its **own** optimisation (the `h1q/h2q/h4q/h8q`
  subdirectories).

So inside the optimiser, each candidate γ is evaluated by a **single
deterministic point forecast per origin** — no posterior draws are involved.

### 1.3 How the tuned prior is used (selection frozen once)

In
[`run_glp_experiment`](src/glp_hyperparameter_optimization/forecasting.py#L407),
for the RMSE strategies (`ONE_TIME_OPTIMIZATION_STRATEGIES`) the hyperparameters
are selected **once**, on the **earliest** origin's real-time sample, and then
reused for every forecast origin:

```python
y0, codes0, _ = build_glp_estimation_matrix(panel, origins[0], size)
shared_hyperparameters = select_hyperparameters(strategy, y0, codes0, task_template)
# ... reused via task["fixed_hyperparameters"] for all origins
```

By contrast, `paper` and `mango_mdd` re-select **per origin**. `--per-origin-selection`
exists but is not the default.

### 1.4 How the final forecasts differ from the objective

The forecasts that are actually reported/scored are produced by
[`predictive_draws`](src/glp_hyperparameter_optimization/forecasting.py#L226) →
[`glp_fixed_hyperparameter_forecast_draws`](src/glp_hyperparameter_optimization/glp_model.py#L573):
β and Σ are drawn from the conditional posterior, forecast paths are simulated
with Gaussian shocks, and the reported point forecast is the **mean across
draws** in
[`_forecast_rows`](src/glp_hyperparameter_optimization/forecasting.py#L250):

```python
mean = layer.mean(axis=0)              # forecasting.py:264
row["error"] = mean[vi] - actual       # scored in the comparison tables
```

The comparison RMSE tables aggregate exactly this `error` column
([reporting.py `compute_rmse_table`](src/glp_hyperparameter_optimization/reporting.py#L102)).

**Key point:** objective = RMSE of `bvarFcst(β_mode)`; reported metric = RMSE of
`mean_d bvarFcst-with-shocks(β_d, Σ_d)`. These are **not the same forecast
object**.

---

## 2. Q1 — Are these real out-of-sample forecasts?

**Yes, strictly speaking.** The targets are held out of the estimation window,
the model is re-estimated for each origin, and the horizon alignment is correct.
This is genuine time-series cross-validation, not an in-sample fit.

Four caveats qualify how *informative* that out-of-sample signal is:

1. **Thin, overlapping evaluation set.** With `n_eval = 3` and contiguous origins,
   `mango_rmse` scores only 3 windows at the very end of the sample, whose
   holdout periods overlap heavily. That is a high-variance objective, and
   "rolling out-of-sample RMSE" overstates it (the origins are the last three,
   not a rolling pass over the sample). `mango_rmse_random` mitigates the
   *clustering* by spreading origins, but still uses only `n_eval` of them.
2. **Selection is frozen on the earliest origin.** The tuned prior is chosen once
   on the **smallest** real-time window and applied to the entire recursive
   experiment, so across the actual out-of-sample period the hyperparameters are
   never re-optimised. (Documented as a known limitation; `--per-origin-selection`
   is the recursive alternative.)
3. **Inner vs. outer vintage mismatch.** The objective scores against later
   observations of the **same real-time vintage** used for training, whereas the
   final comparison scores against a fixed later **actual vintage**
   (`GLP_ACTUAL_VINTAGE = 2023-01-01`). Optimising toward the real-time vintage
   is defensible for real-time forecasting, but it differs from the reported
   evaluation target and should be a conscious choice.
4. **Point forecast, not the reported density mean** (see §3).

**Verdict:** it is real POOS, but the objective is a *small, tail-weighted,
frozen-once* slice of out-of-sample performance rather than a rich recursive one.

---

## 3. Q2 — Does the RMSE approach make sense?

**As a point-forecast-accuracy criterion, yes** — the mechanics (split,
alignment, horizon indexing, Bayesian optimisation over the bounded box) are
correct, and horizon-specific tuning is a legitimate design choice.

**The one substantive weakness is internal inconsistency:** the objective uses
the **posterior-mode** point forecast, while the reported forecasts use the
**mean of the posterior predictive draws**. Consequences:

- At **h = 1** these nearly coincide: `bvarFcst` is linear in β at one step, so
  `E_β[bvarFcst(β)] = bvarFcst(E[β])`, and for a roughly symmetric posterior the
  mode ≈ mean. The inconsistency is negligible.
- At **longer horizons** the multi-step recursion is **nonlinear in β**, so a
  Jensen gap opens between `bvarFcst(β_mode)` (objective) and
  `E_β[bvarFcst(β)]` (reported). You can then select a γ that is optimal for the
  mode path but not for the density mean you actually report — precisely at the
  horizons (h = 4, 8) where the two most diverge.

So the approach "makes sense" but is tuning a proxy for the reported metric
rather than the reported metric itself.

---

## 4. Q3 — Should the objective use posterior draws instead?

**Yes — scoring the predictive mean (β-averaged forecast) is the right
consistency fix.** But there are three points to get right.

### 4.1 It aligns the objective with the reported metric

The reported point forecast is `mean_d(simulated draws) → E_β[bvarFcst(β)]` in
the large-draw limit. Making the objective score the *same* β-averaged forecast
removes the mode-vs-mean gap of §3 and is, under squared-error loss, the
**Bayes-optimal point predictor** (the posterior predictive mean minimises
expected squared error). This is the theoretically correct forecast to pair with
an RMSE objective.

### 4.2 The subtlety: shocks add only noise; β-averaging is what matters

For a **linear** VAR, the shock innovations are mean-zero and propagate linearly,
so for any fixed draw `E_shocks[simulate_path(β, Σ)] = bvarFcst(β)` exactly.
Therefore:

$$
\mathbb{E}_{\text{draws}}\big[\text{simulated path}\big] \;=\; \mathbb{E}_\beta\big[\text{bvarFcst}(\beta)\big].
$$

The shock simulation contributes **only Monte Carlo variance** to a *point* RMSE;
it does not change the target. The efficient, lower-variance implementation is
therefore:

> for each candidate γ and each origin, draw `β_d` (say 200–500 draws), compute
> `bvarFcst(β_d)`, average → predictive-mean forecast, then RMSE.

i.e. you do **not** need to simulate shock paths just to fix the point-RMSE
inconsistency — averaging `bvarFcst` over posterior β draws is enough and matches
the reported mean in the large-draw limit. (If exact bit-for-bit agreement with
the reported simulator is wanted, reuse the same draw-and-simulate path at the
cost of extra variance.)

### 4.3 If you care about the density, change the metric, not just the inputs

Feeding draws into an RMSE only exploits them through the β-average. If the aim
is to reward the **whole predictive density** (calibration, tails), RMSE is the
wrong loss. The principled "use-the-draws" upgrade is a proper scoring rule:

- **CRPS** (empirical, from the draws) — strictly proper, robust, easy to compute
  from the simulated ensemble.
- **Mean log predictive score** — strictly proper, density-aware, but more
  sensitive to tail draws / requires a density estimate.

This is a *different research question* (best density prior vs. best point-RMSE
prior) and would be a genuine methodological addition, not just a bug fix.

### 4.4 Cost

The β-averaged objective multiplies the per-candidate cost by the number of
draws per origin. Bayesian optimisation already evaluates ~`init_points + n_iter`
candidates × `n_eval` origins; a modest draw count (200–500) keeps this tractable
and can be subsampled. The MDD objective is cheaper; this trade-off should be
noted in the write-up.

### 4.5 Verdict

- **Do** switch the RMSE objective to score the **β-averaged predictive mean**
  (§4.1–4.2). It is a correct, low-risk consistency improvement, most impactful
  at h = 4, 8.
- **Consider** adding a **CRPS / log-score** objective variant if density quality
  (not just point accuracy) is of interest (§4.3).
- Keep the deterministic-mode objective available as a fast baseline so the two
  can be compared.

---

## 5. Recommendations (tiered)

| Tier                              | Change                                                                                                                                                              | Why                                                                               | Effort           |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | ---------------- |
| **1 (consistency)**         | Score the**β-averaged predictive mean** in the RMSE objective instead of the single posterior-mode forecast.                                                 | Aligns the objective with the reported metric; removes the multi-step Jensen gap. | Low–medium      |
| **2 (robustness)**          | Use**more, better-spread** evaluation origins (raise `n_eval`; optionally non-overlapping / rolling across the sample).                                     | The default 3 tail-clustered overlapping origins give a high-variance objective.  | Low              |
| **3 (density-aware)**       | Add an optional**CRPS / log-score** objective computed from the draws.                                                                                        | Rewards the full predictive density, not just the point.                          | Medium           |
| **4 (recursive selection)** | Make periodic/`--per-origin-selection` re-optimisation the documented default for headline results, or at least report both.                                      | One-time-on-earliest-origin selection is not fully recursive.                     | Medium (compute) |
| **5 (vintage)**             | Decide and document whether the inner objective should target the**real-time** or the **final** actual vintage, consistently with the outer evaluation. | Removes an unstated inner/outer target mismatch.                                  | Low              |

---

## 6. Ready-to-use agent prompts

Paste any of these to an agent to implement the corresponding change. They are
written to be self-contained and to preserve the existing tests.

### Prompt A — Align the RMSE objective with the reported predictive mean (Tier 1)

```
In src/glp_hyperparameter_optimization/glp_model.py, change the RMSE
hyperparameter objective so it scores the Bayesian PREDICTIVE-MEAN forecast
(averaged over posterior draws of beta) instead of the single posterior-mode
point forecast, matching how final forecasts are aggregated in
forecasting.py `_forecast_rows` (mean of draws).

Requirements:
1. In `_rmse_objective` (around line 733), replace the `glp_mode_estimate` +
   `point_forecast` call with an average over `n_obj_draws` posterior draws:
   for each origin, draw (beta_d, sigma_d) via `glp_draw(ctx, vec)` `n_obj_draws`
   times, compute `point_forecast(ctx.y, beta_d, horizons)` for each, and average
   the forecasts across draws (beta-averaging only; do NOT simulate shock paths —
   for a point RMSE the mean-zero shocks only add Monte Carlo variance).
2. Add an `n_obj_draws: int = 200` parameter threaded from
   `update_hyperparameters_mango_rmse` and
   `update_hyperparameters_mango_rmse_random` (default 200). When
   `n_obj_draws <= 1`, fall back to the existing deterministic posterior-mode
   forecast so the old behaviour remains available and cheap.
3. Use a fixed, per-origin RNG seed so the objective is deterministic across
   Mango candidate evaluations (otherwise the surrogate sees noise). Seed each
   origin's draws from a base seed + origin index.
4. Expose `--optimization-n-obj-draws` on the RMSE scripts
   (scripts/glp/run_glp_mango_rmse.py and run_glp_mango_rmse_random.py) and thread
   it through `forecasting.run_from_namespace` / `run_glp_experiment` and the task
   dict, plus record it in `run_metadata.json`.
5. Keep all existing tests in tests/test_glp_model.py green (the
   `test_mango_rmse_variants_return_in_bounds` and `_rmse_eval_origins` tests).
   Add a unit test asserting that with `n_obj_draws=1` the objective equals the
   current posterior-mode RMSE, and that `n_obj_draws>1` runs and stays in bounds.

Do not change the paper / mango_mdd strategies. Run the GLP test suite
(tests/test_glp_*.py) and report results.
```

### Prompt B — Add a CRPS / log-score density objective variant (Tier 3)

```
Add an optional density-scoring objective to the GLP RMSE optimisers so the
prior can be tuned to the full predictive density instead of a point forecast.

In src/glp_hyperparameter_optimization/glp_model.py:
1. Add an `objective_metric` argument ("rmse" | "crps" | "logscore",
   default "rmse") to `update_hyperparameters_mango_rmse` and
   `update_hyperparameters_mango_rmse_random`, threaded into `_rmse_objective`.
2. For "crps"/"logscore", generate a predictive ENSEMBLE per origin using the
   existing `glp_fixed_hyperparameter_forecast_draws` machinery (draw beta/sigma
   and simulate paths with `simulate_forecast_path`), then:
   - CRPS: compute the empirical CRPS of the ensemble vs the held-out actual for
     each scored variable/horizon and average (use the energy-form estimator
     mean|X-y| - 0.5 mean|X-X'|).
   - logscore: fit a Gaussian to the ensemble mean/variance per variable/horizon
     and return the NEGATIVE mean log predictive density (so Mango .minimize()
     still applies).
   Keep Mango minimising for all three metrics.
3. Add `--optimization-objective {rmse,crps,logscore}` to
   scripts/glp/run_glp_mango_rmse.py and run_glp_mango_rmse_random.py, thread it
   through forecasting.run_from_namespace / run_glp_experiment, and record it in
   run_metadata.json.
4. Add unit tests in tests/test_glp_model.py that each metric runs on a small
   synthetic sample, returns in-bounds hyperparameters, and that CRPS/logscore
   use draws (seeded, deterministic).

Report the test results and briefly note the extra compute cost of the density
objectives versus rmse.
```

### Prompt C — Strengthen the evaluation origins (Tier 2)

```
Improve the GLP RMSE evaluation-origin scheme in
src/glp_hyperparameter_optimization/glp_model.py `_rmse_eval_origins`
(around line 681).

1. Raise the default `n_eval` used by the RMSE strategies from 3 to a larger
   value (e.g. 8), configurable via the existing --optimization-n-eval flag.
2. Add a `stride` option so contiguous origins can be spaced (k = 0, stride,
   2*stride, ...) to reduce holdout overlap, plumbed via a new
   --optimization-origin-stride flag (default 1 preserves current behaviour).
3. Validate feasibility (enough origins for n_eval*stride given T, H, min_t) and
   raise a clear error otherwise, mirroring the existing infeasibility test.
4. Record n_eval and stride in run_metadata.json.
5. Keep `test_rmse_eval_origins_rolling_and_random` and
   `test_rmse_eval_origins_raises_when_infeasible` green; add a test for the
   stride behaviour.

Do not change random-origin sampling semantics beyond honouring the new stride
only in the non-random (contiguous) path. Run tests/test_glp_model.py.
```

### Prompt D — Make recursive (per-origin) RMSE selection a first-class, documented mode (Tier 4)

```
In src/glp_hyperparameter_optimization/forecasting.py, make recursive RMSE
hyperparameter selection easy to run and clearly reported.

1. Add a `selection_frequency` option ("once" | "per_origin" | integer N) to
   run_glp_experiment: "once" = current behaviour (select on origins[0]);
   "per_origin" = select at every origin; N = re-select every N origins, reusing
   the most recent selection in between.
2. Expose it as --selection-frequency on run_glp_mango_rmse.py and
   run_glp_mango_rmse_random.py (default "once" to preserve behaviour), thread it
   through run_from_namespace, and record it in run_metadata.json (extend the
   existing "hyperparameters_selected_once" / "hyperparameter_selection_origin"
   fields).
3. Ensure no look-ahead: selection at origin t may only use data available at t.
4. Add a test that "per_origin" produces a per-origin selection record and that
   "once" matches current output on a small synthetic run (monkeypatch the
   optimiser as in tests/test_optimizer_variable_resolution.py
   `test_mango_rmse_optimizes_once_for_full_recursive_run`).

Run the GLP test suite and report results.
```

### Prompt E — Resolve the inner/outer evaluation-vintage consistency (Tier 5)

```
Clarify and make configurable which data vintage the GLP RMSE objective scores
against.

Currently the inner RMSE objective in
src/glp_hyperparameter_optimization/glp_model.py scores forecasts against later
observations of the SAME real-time estimation vintage, while the outer comparison
(reporting.py / compare_glp_forecasts.py) scores against the fixed
GLP_ACTUAL_VINTAGE (2023-01-01).

1. Add an `objective_actual` option ("realtime" | "final", default "realtime")
   to update_hyperparameters_mango_rmse / _random and _build_rmse_origins. When
   "final", the held-out actuals for each origin should come from the fixed final
   actual vintage frame (build_glp_actual_frame) aligned by target quarter,
   instead of from the real-time y matrix.
2. Plumb it through forecasting.run_glp_experiment (which already loads the actual
   vintage) and expose --optimization-actual {realtime,final} on the two RMSE
   scripts; record the choice in run_metadata.json.
3. Add a docstring/section note explaining the trade-off (real-time is honest for
   real-time forecasting; final matches the reported evaluation target).
4. Add a unit test covering both modes on a small synthetic panel.

Run tests/test_glp_*.py and report results.
```

---

## 7. Appendix — code map

| Concept                                 | Location                                                                                                                                                                                            |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Evaluation-origin indices               | [`_rmse_eval_origins`](src/glp_hyperparameter_optimization/glp_model.py#L681)                                                                                                                      |
| Per-origin train/holdout + context      | [`_build_rmse_origins`](src/glp_hyperparameter_optimization/glp_model.py#L712)                                                                                                                     |
| RMSE objective (mode point forecast)    | [`_rmse_objective`](src/glp_hyperparameter_optimization/glp_model.py#L733)                                                                                                                         |
| Posterior-mode β                       | [`glp_mode_estimate`](src/glp_hyperparameter_optimization/glp_model.py#L385)                                                                                                                       |
| Deterministic recursive forecast        | [`point_forecast`](src/glp_hyperparameter_optimization/glp_model.py#L401)                                                                                                                          |
| Fixed-γ predictive draws               | [`glp_fixed_hyperparameter_forecast_draws`](src/glp_hyperparameter_optimization/glp_model.py#L573)                                                                                                 |
| RMSE optimisers                         | [`update_hyperparameters_mango_rmse`](src/glp_hyperparameter_optimization/glp_model.py#L769), [`update_hyperparameters_mango_rmse_random`](src/glp_hyperparameter_optimization/glp_model.py#L808) |
| One-time selection on earliest origin   | [forecasting.py L407–L425](src/glp_hyperparameter_optimization/forecasting.py#L407)                                                                                                                 |
| Reported point forecast = mean of draws | [`_forecast_rows`](src/glp_hyperparameter_optimization/forecasting.py#L250) (mean at [L264](src/glp_hyperparameter_optimization/forecasting.py#L264))                                               |
| Comparison RMSE on the`error` column  | [`compute_rmse_table`](src/glp_hyperparameter_optimization/reporting.py#L102)                                                                                                                      |
| Existing methodology description        | [README_GLP_METHODOLOGY.md §5.3](README_GLP_METHODOLOGY.md)                                                                                                                                         |
