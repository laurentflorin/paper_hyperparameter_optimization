# Forecasting, Hyperparameter-Optimization, and Ranking Audit

Audit date: 2026-08-18  
Audited repository commit: `afa5977ac90d617e681bd5fe7d9b1c200864a50f`  
Primary focus: GLP code, in-/out-of-sample indices, look-ahead bias, and consistency of the rolling and random RMSE selectors and downstream rankings.

## Executive conclusion

The basic GLP inner holdout arithmetic is correct: for origin offset `k`, the code uses `cut = T - H - k`, fits on `y[:cut]`, holds out `y[cut:cut+H]`, and compares forecast row `h-1` with actual row `h-1`. The fixed final actual vintage is used only after fitting. I found no off-by-one error or direct future-outer-outcome leakage in that narrow split.

The overall implementation is nevertheless not safe to use for headline comparisons without fixes. The most important problems are:

1. The GLP large model stops receiving complete data after 2015Q4 but continues to label runs through 2019 as new forecast origins. It repeatedly “forecasts” 2016Q1–2017Q4 after those outcomes were already known.
2. The stored GLP real-time panel violates its point-in-time contract, and an empty-ALFRED fallback deliberately substitutes current revised FRED history.
3. The non-GLP RMSE optimizer minimizes raw level errors, while the final tables rank growth-rate errors.
4. The non-GLP reporter pools separately tuned `h1q`, `h2q`, `h4q`, and `h8q` models into one synthetic model.
5. Both reporters can compare models on different origin samples. A checked-in non-GLP result changes winner after the samples are paired.
6. The non-GLP 24-month forecast is sometimes too short to produce the row labeled quarter 8; 28 of 51 `+0 months` paper origins in the checked-in artifact stop at quarter 7.

Accordingly, existing large-GLP and non-GLP comparison tables should be treated as invalid until the critical/high findings are fixed and all affected forecasts are regenerated.

## What “the two ranking algorithms” means in this audit

There is no standalone `rank()` function in the repository. I treated the phrase as covering:

- the two RMSE hyperparameter selectors, `mango_rmse` (recent contiguous origins) and `mango_rmse_random` (random origins); and
- the two reporting implementations that turn forecast errors into rankings: the GLP and non-GLP reporters.

The rolling and random GLP variants share the same forecast loss and have correct train/holdout index arithmetic. Their origin validation and failure semantics are not consistent, however, and the non-GLP dependency has additional feasibility and stochastic-objective problems described below.

## Severity summary

| ID | Severity | Finding | Evidence status |
|---|---|---|---|
| GLP-01 | Critical | Large-model nominal origins become hindsight forecasts of stale 2016–2017 targets | Confirmed in committed panel and outputs |
| GLP-02 | High | GLP vintage ingestion does not enforce a point-in-time invariant | Confirmed future rows; fallback is active code |
| GLP-03 | Medium | Inner RMSE validation is retrospective, not pseudo-real-time | Confirmed by data flow |
| GLP-04 | High | Holdouts influence `psi` search bounds and candidates are silently changed across folds | Confirmed by code path |
| GLP-05 | Medium | Declared 1959 sample start is ignored; samples expand backward | Confirmed in committed panel |
| GLP-06 | Low/medium | Origin seed offset is always zero and RMSE feasibility hardcodes five lags | Confirmed by arithmetic/code |
| MF-01 | High | Calendar horizons are detached from the effective data endpoint; h8 is missing | Confirmed in committed outputs |
| MF-02 | Critical | RMSE tuning uses raw levels while final ranking uses growth rates | Confirmed in local and dependency code |
| RANK-01 | Critical | Non-GLP reporting pools distinct horizon-tuned models | Confirmed in code and artifacts |
| RANK-02 | High | Models are ranked on unequal OOS samples | Confirmed rank reversal in artifacts |
| SEL-01 | High | Invalid/all-penalty optimization can return arbitrary parameters; rolling/random feasibility differs | Confirmed in code/dependency |
| SEL-02 | Medium/high | Strategy comparisons confound objective, update schedule, and variable universe | Confirmed and test-locked |
| REPRO-01 | High | Scientific dependency and stochastic objectives are not reproducible | Confirmed in requirements/dependency |
| MF-03 | Medium, conditional | Custom data windows can backfill from a future vintage and score partial actual quarters | Reproduced; defaults currently avoid it |
| MF-04 | Medium | Growth predictive intervals are not quantiles of growth draws | Confirmed by code |
| OPS-01 | Medium/high | Bad explicit comparison paths silently select stale outputs; compatibility is unchecked | Reproduced |
| OPS-02 | High, operational | Documented Euler script uses wrong paths and omits its baseline | Confirmed by static trace |
| DOC-01 | Low | Methodology and relative-RMSE documentation contradict current code | Confirmed |

---

## Detailed findings and repair prompts

### GLP-01 — Critical: the large model produces hindsight forecasts of past targets

#### Evidence

- The large block includes `PPIFGS` as `PPI` in [`config.py`](src/glp_hyperparameter_optimization/config.py#L158). In the committed panel it ends at 2015Q4.
- [`_complete_window`](src/glp_hyperparameter_optimization/data_utils.py#L378) silently selects the last fully complete quarter across every series.
- [`_forecast_rows`](src/glp_hyperparameter_optimization/forecasting.py#L252) defines targets as `last_quarter + h`, without checking the nominal `forecast_origin`.
- A direct panel diagnostic found that the large-model endpoint is 2015Q4 for every nominal origin from 2016-03-31 through 2019-12-31.
- [`outputs/glp/all_large/paper/selected_hyperparameters.csv`](outputs/glp/all_large/paper/selected_hyperparameters.csv) records `last_quarter=2015Q4` at the 2019-12-31 origin.
- [`outputs/glp/all_large/paper/forecast_panel.csv`](outputs/glp/all_large/paper/forecast_panel.csv) then assigns that origin the targets 2016Q1 through 2017Q4. Each of those target quarters is scored 16 times under different nominal origins.

This is not merely stale data. By a 2018 or 2019 nominal origin, the 2016–2017 outcomes being scored were already known. The evaluation is therefore not real-time out-of-sample and overweights the same target realizations.

#### Repair prompt

```text
Fix the GLP large-model stale-origin bug.

Scope:
- src/glp_hyperparameter_optimization/data_utils.py
- src/glp_hyperparameter_optimization/forecasting.py
- src/glp_hyperparameter_optimization/config.py
- GLP tests and run metadata

Requirements:
1. Distinguish the nominal vintage/forecast-origin date from the effective information-set quarter returned by build_glp_estimation_matrix.
2. Add a configurable expected data lag. For the current quarterly GLP workflow, validate that the last complete estimation quarter is exactly the intended lag behind the nominal origin. Reject an origin with a clear error when it is stale; never silently move the effective origin backward.
3. Ensure every emitted target is genuinely after the information set and consistent with the nominal origin. Add uniqueness/monotonicity assertions so repeated stale origins cannot rescore the same target block.
4. Resolve PPIFGS ending in 2015: replace it only with a scientifically justified continuing series, or explicitly cap/version the large-model evaluation window. Do not forward-fill a discontinued series.
5. Record information_set_quarter and origin_data_lag_quarters in selected_hyperparameters.csv and run metadata.
6. Add a synthetic discontinued-series test that must fail before fitting, plus an integration test over all configured origins/model sizes asserting the expected information lag and target mapping.
7. Invalidate and regenerate affected large-model outputs. Document that prior large-model comparisons are not usable.
```

### GLP-02 — High: GLP vintage ingestion is not point-in-time safe

There are two independent paths.

First, a successful ALFRED response is never filtered or validated after download in [`download_series_vintage`](src/glp_hyperparameter_optimization/data_utils.py#L167). The committed `glp_realtime_panel.csv.gz` contains 4,305 `TWEXMMTH` rows whose `observation_date` is after their stated `vintage_date`; affected vintages run from 2005-06-30 through 2013-12-31, and some contain observations through 2019-12-01. Other missing series currently stop these future rows from extending the joint complete window, but the panel contract itself is broken.

Second, when ALFRED returns an empty body, [`download_series_vintage`](src/glp_hyperparameter_optimization/data_utils.py#L177) substitutes the current/latest FRED history and truncates only on `observation_date`. That does not remove revisions and does not reproduce historical release availability. The behavior is explicitly blessed by [`test_glp_data_utils.py`](tests/test_glp_data_utils.py#L23). For a revisable quarterly series, an observation dated before the vintage may still have been unreleased then, and every older value may carry later revisions.

#### Repair prompt

```text
Make the GLP data pipeline fail closed and enforce a point-in-time invariant.

Requirements:
1. In download_series_vintage, apply and validate observation_date <= vintage_date after every source path, including a nominally successful ALFRED response. Raise a data-integrity error if the server returns future rows; do not silently trust or cache them.
2. Remove the generic latest-FRED fallback for missing historical ALFRED vintages. For revisable series, failure to obtain a true vintage must stop the run. If any non-revisable-series fallback is retained, require an explicit reviewed allowlist and document the release-lag assumption.
3. Persist source URL/type, requested vintage, download time, fallback status, and a data hash in cache and processed metadata. Never store a fallback indistinguishably from a true vintage.
4. Defensively revalidate cached frames in _load_or_download_realtime_vintage and revalidate the invariant in build_quarterly_levels so old contaminated caches cannot enter fitting.
5. Invalidate/regenerate affected caches and data/processed/glp_realtime_panel.csv.gz, especially TWEXMMTH.
6. Replace the existing fallback test. Add tests for (a) a successful response with post-vintage observations, (b) a latest series containing revised historical values, and (c) an observation dated before the vintage but released afterward. None may silently enter a real-time vintage.
7. Add a whole-panel integration invariant: every row must satisfy observation_date <= vintage_date and every series/vintage must have recorded provenance.
```

### GLP-03 — Medium: inner RMSE validation is retrospective rather than pseudo-real-time

At an outer origin, [`_run_origin_task`](src/glp_hyperparameter_optimization/forecasting.py#L287) builds one matrix from that outer origin's vintage. [`_build_rmse_origins`](src/glp_hyperparameter_optimization/glp_model.py#L712) creates historical folds by slicing the same matrix. An internal 1995 origin evaluated at a 2000 outer origin therefore sees 1995 history as revised by 2000, not the values available in 1995.

This is not leakage beyond the outer forecast date—the 2000 forecaster is allowed to know revisions released by 2000. It is nevertheless look-ahead relative to the claimed internal pseudo-real-time origins. The method is ordinary retrospective time-series cross-validation unless each inner fold loads its own vintage.

#### Repair prompt

```text
Make GLP RMSE validation-vintage semantics explicit and implement a true real-time option.

Requirements:
1. Add validation_vintage_mode with at least retrospective and realtime values.
2. Retrospective mode may keep slicing the outer-origin matrix, but rename/document it accurately and record the mode in metadata.
3. In realtime mode, represent inner origins by calendar dates. Build each training matrix from that inner origin's own ALFRED vintage rather than slicing a later vintage.
4. Add an explicit validation-target-vintage policy (for example first_release, inner_vintage_later_release, or fixed_final) and align holdout quarters by keys, not row offsets alone.
5. Restrict the origin pool to vintages actually available in the panel, or extend the downloader; fail clearly when a true inner vintage is absent.
6. Add a revision-sensitive fixture where a historical value changes between vintages. Assert realtime mode never observes the later revision and retrospective mode does, with both behaviors documented.
7. Update README_GLP_METHODOLOGY.md so it does not call retrospective folds pseudo-real-time without qualification.
```

### GLP-04 — High: validation targets affect `psi` search support and one candidate changes across folds

[`prepare_glp_context`](src/glp_hyperparameter_optimization/glp_model.py#L109) estimates AR residual variances `SS` and sets `psi` bounds to `SS/100` and `SS*100`. The RMSE optimizers build `ctx_ref` and Mango's search space from full `y` in [`update_hyperparameters_mango_rmse`](src/glp_hyperparameter_optimization/glp_model.py#L805) and its random counterpart, even though full `y` contains the inner validation targets.

Each training fold has its own `SS` and bounds. The candidate vector created under `ctx_ref` is then silently clipped by [`to_transformed`](src/glp_hyperparameter_optimization/glp_model.py#L262) or [`_clip_natural`](src/glp_hyperparameter_optimization/glp_model.py#L307). Consequently:

- changing only held-out targets changes the optimizer's search domain;
- the same named Mango candidate can mean a different effective `psi` in every fold; and
- one-time fixed `psi` values can be clipped again at later outer origins while the original, not effective, values are recorded.

#### Repair prompt

```text
Redesign GLP RMSE psi tuning to remove validation leakage and silent candidate mutation.

Requirements:
1. Do not derive any search-space statistic from inner holdout rows.
2. Give one optimizer candidate a stable interpretation across folds. Prefer dimensionless log multipliers relative to each training fold's SS, then map the selected multiplier to the final forecast context. A documented common intersection derived only from training contexts is an acceptable alternative.
3. Remove silent clipping. Reject an invalid candidate explicitly or expose a deterministic mapping that is part of the parameterization.
4. Record both selected optimizer coordinates and effective natural psi values used for every final forecast origin.
5. Ensure one-time fixed hyperparameters are truly fixed under the documented parameterization; do not report an unclipped vector while forecasting with a clipped one.
6. Add tests with deliberately different fold variances/bounds. Mutating only holdout values must not change the search domain, and one candidate must retain the documented meaning in every fold.
7. Add a later-outer-origin regression test proving recorded effective values equal the values passed to posterior drawing.
```

### GLP-05 — Medium: `GLP_SAMPLE_START` is declared but not enforced

[`config.py`](src/glp_hyperparameter_optimization/config.py#L94) declares a 1959-01-01 start. [`build_glp_estimation_matrix`](src/glp_hyperparameter_optimization/data_utils.py#L390) instead accepts the first fully complete row, with no lower bound.

The committed small and medium panels demonstrate backward expansion:

- at 2000-03-31, the sample is 1959Q1–1999Q4 (`n=164`);
- at 2000-06-30, it becomes 1954Q3–2000Q1 (`n=183`).

Advancing one quarter therefore adds 19 observations because the start moves backward. This changes the estimation population and prior scales, rather than forming a stable expanding window.

#### Repair prompt

```text
Enforce a stable GLP estimation window.

Requirements:
1. Thread sample_start through build_glp_estimation_matrix and clip the quarterly frame to GLP_SAMPLE_START before selecting the complete window. Expose an explicit override for alternative studies.
2. Decide and document whether GLP_SAMPLE_END applies only to the original replication window or to recursive forecasts; do not leave it as dead configuration.
3. Store first_estimation_quarter, last_estimation_quarter, and n_obs in run metadata and per-origin records.
4. Add a test with a later vintage that backfills older rows. The first estimation quarter must remain fixed and the sample may grow only at its end unless an explicit missing-data policy says otherwise.
5. Add an integration assertion over committed/configured origins that the first quarter is stable for each model definition.
6. Regenerate hyperparameters and forecasts affected by the changed estimation sample.
```

### GLP-06 — Low/medium: seed and feasibility calculations ignore their intended inputs

Two smaller implementation bugs remain:

- [`forecasting.py`](src/glp_hyperparameter_optimization/forecasting.py#L305) computes the per-origin offset as `Timestamp.value % 100000`. Quarter-end timestamps are at midnight and the result is zero, so every origin receives the same `seed_base`, contrary to the documented offset behavior.
- [`_rmse_eval_origins`](src/glp_hyperparameter_optimization/glp_model.py#L681) defaults `min_t` with `4 * 5`, hardcoding five rather than using the caller's `lags`. The CLI permits other lag orders.

#### Repair prompt

```text
Harden GLP seed derivation and RMSE-origin feasibility.

Requirements:
1. Replace Timestamp.value % 100000 with a stable origin identifier, such as quarterly Period.ordinal, combined with seed_base through numpy.random.SeedSequence. Do not use Python hash(). Preserve None as explicitly unseeded behavior.
2. Record the seed policy and effective per-origin seed in metadata sufficient for reproduction.
3. Pass lags into _rmse_eval_origins and derive a documented minimum training length from the actual lag order and posterior-context requirements. Remove the hardcoded five.
4. Validate positive H/n_eval/lags, h_eval within 1..H, and every cut before constructing contexts.
5. Add tests proving distinct origins get distinct but repeatable seeds in serial and parallel scheduling.
6. Add feasibility tests for lag orders below and above five, including a random candidate at the oldest permissible origin.
```

### MF-01 — High: labeled horizons are not anchored to the calendar origin

The non-GLP builder discards the ragged edge in [`_complete_window`](src/paper_hyperparameter_optimization/data_utils.py#L484) and [`build_model_input_frames`](src/paper_hyperparameter_optimization/data_utils.py#L496). The model then always forecasts a fixed number of months in [`_run_origin_task`](src/paper_hyperparameter_optimization/forecasting.py#L301), while [`extract_forecasts`](src/paper_hyperparameter_optimization/forecasting.py#L254) labels quarters from the calendar origin.

For the 1997-07-31 vintage, the returned balanced monthly frame ends at 1997-05-31. A 24-month simulation reaches May 1999, whose last complete aggregate quarter is 1999Q1; labeled horizon 8 should be 1999Q2. The checked-in paper artifact has an h8 row for only 23 of 51 `+0 months` origins; 28 silently stop at h7.

The same trimming also discards released ragged-edge information. At 1997-07-31, June observations exist for several monthly variables and 1997Q2 exists for the quarterly block, but a lagging monthly PCE observation makes the returned monthly block stop in May and the quarterly block stop in 1997Q1. This is conservative rather than look-ahead, but it means the nominal origin does not describe the model's information set.

#### Repair prompt

```text
Fix non-GLP calendar-origin and forecast-horizon alignment.

Requirements:
1. Preserve the real-time ragged edge if the pinned MBFVAR implementation supports trailing NaNs: retain released cells and align blocks by calendar date rather than len(monthly)//3.
2. If a balanced-panel design is retained, explicitly compute/store the effective data cutoff and do not claim it is the nominal origin's full information set.
3. Compute the required monthly simulation length from the effective highest-frequency endpoint through the end of origin.to_period('Q') + (max_horizon - 1). A fixed 3*H is insufficient when the endpoint lags the calendar origin.
4. After aggregation, require exactly every requested (origin, variable, horizon) key or fail the origin with a precise coverage error. Remove silent continue behavior for requested target quarters.
5. Add regression tests for July, August, and September origins with different ragged endpoints. Every successful run must emit h1..h8 for every variable and map each label to the intended calendar quarter.
6. Regenerate non-GLP outputs and report sample counts by horizon.
```

### MF-02 — Critical: RMSE tuning uses raw levels but final rankings use growth rates

All required quarterly RMSE targets (`GDP`, `INVFIX`, `GOV`) are marked as growth variables in [`config.py`](src/paper_hyperparameter_optimization/config.py#L62). Final evaluation applies `100 * log(level).diff()` in [`compute_quarterly_metrics`](src/paper_hyperparameter_optimization/forecasting.py#L88).

The wrapper delegates tuning at [`select_hyperparameters`](src/paper_hyperparameter_optimization/forecasting.py#L196). In the audited MBFVAR dependency commit `5b06f93272cd6ebf370fbf2aac3b3573c7830493`, both RMSE methods subtract raw back-transformed forecast levels from raw held-out levels:

- [rolling objective](https://github.com/laurentflorin/MBFVAR/blob/5b06f93272cd6ebf370fbf2aac3b3573c7830493/MBFVAR/_hyp_opt.py#L464-L488)
- [random objective](https://github.com/laurentflorin/MBFVAR/blob/5b06f93272cd6ebf370fbf2aac3b3573c7830493/MBFVAR/_hyp_opt.py#L758-L780)

It also pools squared level errors across three variables with very different units/scales. Selected parameters therefore optimize neither the final metric nor a scale-invariant joint loss; GDP levels can dominate.

#### Repair prompt

```text
Make the non-GLP RMSE hyperparameter objective exactly match final scoring.

Requirements:
1. Implement a repository-controlled validation loss instead of relying on MBFVAR's raw-level RMSE, or contribute and pin a verified upstream implementation.
2. Extract one canonical transformation/error helper and reuse it in tuning and extract_forecasts. Growth variables must use the same 100*diff(log(level)) definition; level variables must remain in levels.
3. Retain the preceding observed/nowcast level required to compute the first forecast growth rate without leaking a held-out future value.
4. Define a documented multi-variable aggregation scheme. Use equal variable weights after normalization, or another justified weighting, so changing measurement units cannot change candidate ranking.
5. Preserve disjoint rolling/random train and holdout blocks and exact h_eval alignment.
6. Add hand-calculated one- and multi-step tests whose tuning loss equals final error_metric, plus a scale-invariance test where multiplying one level series by 100 leaves growth-loss rankings unchanged.
7. Record objective metric, transforms, variable weights, and inspected dependency revision in run metadata. Regenerate all RMSE-selected results.
```

### RANK-01 — Critical: non-GLP reporting pools distinct horizon-tuned models

The batch scripts write separate `h1q`, `h2q`, `h4q`, and `h8q` models. [`load_forecast_panels`](src/paper_hyperparameter_optimization/reporting.py#L28) concatenates those children with only the common label `mango_rmse` or `mango_rmse_random`. [`compute_rmse_table`](src/paper_hyperparameter_optimization/reporting.py#L79) then pools their errors. Hyperparameters and paths are pooled in the same way.

In the checked-in Euler artifact, GDP / `+0 months` / evaluated h1 has 51 paper rows but 153 `mango_rmse` rows for the same 51 origins—one copy from each nonempty h1q, h2q, and h4q child. The h8q child is empty. The resulting RMSE is not the RMSE of any implemented strategy.

The GLP reporter correctly retains `optimization_horizon`, so the two reporting algorithms are inconsistent.

#### Repair prompt

```text
Correct horizon-specific model identity in non-GLP reporting.

Requirements:
1. Read optimization_eval_horizon_quarters from each child run_metadata.json, with a strictly validated h<N>q directory-name fallback only for legacy runs.
2. Add optimization_horizon to loaded forecast and hyperparameter rows and to grouping, summaries, sorting, tables, plot labels, and output filenames.
3. Choose and document one reporting semantic: show every optimization target as a separate model variant, or score only the evaluation horizon matching that run's tuning horizon. Do not pool variants.
4. Validate uniqueness of the full forecast key after model variant is included; duplicate keys without an explicit variant dimension must raise.
5. Mirror the already-correct GLP variant handling where practical.
6. Add temporary-directory tests with h1q and h4q children having deliberately different errors/hyperparameters. Assert separate RMSE rows and plots and prove no pooled row exists.
7. Regenerate outputs/comparison; prior non-GLP comparison tables should be marked invalid.
```

### RANK-02 — High: rankings use unequal out-of-sample samples and can reverse

Both reporters drop missing errors and aggregate each model independently:

- [`paper_hyperparameter_optimization/reporting.py`](src/paper_hyperparameter_optimization/reporting.py#L79)
- [`glp_hyperparameter_optimization/reporting.py`](src/glp_hyperparameter_optimization/reporting.py#L102)

Neither aligns model/baseline keys before RMSE. This matters because failed origins are omitted. A direct diagnostic on checked-in paper versus Mango-h1 results found an actual reversal for CPI / `+2 months` / h1:

| Calculation | Paper | Mango h1 | Winner |
|---|---:|---:|---|
| Current independent samples (`n=50` vs `n=48`) | 0.198736 | 0.187039 | Mango |
| Common 48 origins | 0.184755 | 0.187039 | Paper |

The diagnostic found 27 such winner flips for the h1q artifact and 35 for h2q. The GLP committed medium artifacts currently have no failed origins, so this is latent there, but the reporter is still unsafe.

#### Repair prompt

```text
Make GLP and non-GLP forecast rankings strictly sample-paired.

Requirements:
1. Define a canonical unique observation key containing, as applicable: model size/group, optimization variant, forecast origin, target quarter, variable, and forecast horizon.
2. Validate uniqueness and identical actual values across models before scoring.
3. For every competitor/baseline cell, inner-join on valid keys and recompute both RMSEs on that identical set. Alternatively require one global common sample, but document the choice.
4. Output n_model, n_baseline, n_common, excluded-key counts, and an exclusion/failure audit table. Warn or fail when coverage falls below a configured threshold.
5. Ensure relative RMSE never combines independently estimated aggregates.
6. Add tests for a competitor missing a hard origin, duplicate keys, model-specific NaNs, mismatched actuals, and a constructed rank reversal.
7. Recompute all comparison tables after pairing; do not compare the old and new numbers as if only formatting changed.
```

### SEL-01 — High: invalid/all-penalty optimizations can return arbitrary parameters, and rolling/random feasibility differs

GLP objectives convert broad exceptions into fixed penalties and then accept `best_params` without confirming any valid evaluation:

- MDD: [`glp_model.py`](src/glp_hyperparameter_optimization/glp_model.py#L669)
- RMSE objective: [`glp_model.py`](src/glp_hyperparameter_optimization/glp_model.py#L780)
- optimizer result acceptance: [`glp_model.py`](src/glp_hyperparameter_optimization/glp_model.py#L846)
- `glp_find_mode` also ignores `scipy.optimize`'s `result.success`.

Bounds-only tests cannot detect an all-penalty result because an arbitrary returned candidate remains in bounds.

Origin semantics also differ:

- GLP random selection raises when `n_eval > n_valid`, but rolling silently truncates to `min(n_eval, n_valid)`.
- The upstream MBFVAR rolling method skips infeasible folds, whereas random pre-validates a nominal pool.
- Non-GLP random RMSE defaults `min_T=None`. At the audited MBFVAR revision, its “valid” pool can include a fold with one lowest-frequency training observation. If such a fold is sampled, every candidate can collapse to the `1e10` penalty and Mango still returns a candidate.
- Requested rather than effective fold counts are recorded.

#### Repair prompt

```text
Add shared, strict preflight and postconditions for every hyperparameter optimizer.

Requirements:
1. Build one validated origin-selection helper per workflow and use it for rolling and random modes. Derive minimum training length from lags, frequencies, priors, and selected variables.
2. Require exactly n_eval feasible origins by default. If partial evaluation is supported, require an explicit flag and record requested/effective counts plus exact origin dates/cuts.
3. Persist random sampled origins before optimization so a run is auditable and resumable.
4. Distinguish expected numerical candidate failures from programming/configuration errors; do not catch every Exception as a penalty.
5. Track valid, penalized, exceptional, and non-finite evaluations. Re-evaluate best_params and raise a dedicated error if no valid candidate exists or the best score equals the sentinel penalty.
6. Check scipy mode result.success and record status/message/evaluation counts.
7. Add tests for n_eval==n_valid, n_eval>n_valid, an oldest infeasible random fold, one failed candidate, all failed candidates, NaN, and tied penalties. Verify that all-penalty optimization fails rather than returning in-bounds parameters.
```

### SEL-02 — Medium/high: strategy rankings confound more than the selection algorithm

For the non-GLP workflow:

- MDD defaults to GDP, while RMSE is forced to GDP/INVFIX/GOV in [`forecasting.py`](src/paper_hyperparameter_optimization/forecasting.py#L119).
- RMSE variants select once on the first outer origin, while MDD reselects every origin in [`run_recursive_experiment`](src/paper_hyperparameter_optimization/forecasting.py#L348).
- The final model is fitted without the MDD optimizer's `var_of_interest`, so parameters selected on a reduced objective/system are applied to the full model.
- [`test_optimizer_variable_resolution.py`](tests/test_optimizer_variable_resolution.py#L101) locks the different call schedules in.

GLP similarly defaults RMSE selection to once and MDD selection per origin, though it records the choice and offers `--per-origin-selection`.

This is not look-ahead bias. It does mean a “ranking of selection algorithms” also ranks different update schedules and objective variable universes.

#### Repair prompt

```text
Make strategy comparisons factorial and apples-to-apples.

Requirements:
1. Expose selection_schedule (first_origin, per_origin, or explicit periodic schedule) as an orthogonal option supported by MDD, rolling RMSE, and random RMSE.
2. Expose objective_variables/model_universe explicitly for all strategies. Do not silently optimize a reduced GDP system and apply the result to a full system unless that experiment has a distinct label.
3. Record schedule, objective variables, final-fit variables, and selection dates in metadata and model labels.
4. Define a fair headline comparison with the same schedule and variable universe; report alternative schedules as separate robustness experiments.
5. Add tests asserting equal optimizer-call counts and equal objective/final variable sets when fair-comparison mode is selected.
6. Update documentation and tables so differences caused by schedule or universe are never attributed solely to MDD versus RMSE ranking logic.
```

### REPRO-01 — High: scientific objectives and rankings are not reproducible

[`requirements.txt`](requirements.txt) pulls MBFVAR from mutable GitHub `master`; `covbayesvar` and all other packages are unpinned. The inspected MBFVAR revision was `5b06f93272cd6ebf370fbf2aac3b3573c7830493`, but another installation of the same requirements can implement different splits or losses.

Randomness is also incompletely controlled:

- GLP Mango tuners have no explicit candidate-generation seed.
- GLP final `seed_base` and random-fold seed default to `None`.
- Non-GLP has no complete fit/forecast/optimizer RNG plumbing.
- At the inspected MBFVAR revision, MDD uses stochastic fitting, RMSE uses stochastic posterior forecasts, and forecast innovations construct fresh unseeded generators. `random_seed` controls only fold sampling, not the candidate, fitting, or forecast randomness.

Thus candidates can be ranked partly on different Monte Carlo noise, and serial/parallel executions need not agree.

#### Repair prompt

```text
Make the forecasting experiments computationally reproducible.

Requirements:
1. Pin MBFVAR to a reviewed full commit SHA and create a lock file with exact versions/hashes for all scientific dependencies.
2. Record repository commit, dependency versions/SHAs, data hashes, and platform details in every run metadata file.
3. Introduce distinct recorded seeds for fold sampling, optimizer candidate generation, posterior fitting/objective draws, and final forecasts.
4. Thread explicit numpy.random.Generator or SeedSequence-derived streams through local code and the pinned MBFVAR implementation. Remove internal fresh default_rng() calls with no supplied seed.
5. Use common random numbers across hyperparameter candidates so objective differences are not candidate-specific Monte Carlo noise. Define how parallel workers receive deterministic child streams.
6. Replace a last-draw/noisy MDD estimate with a deterministic or appropriately averaged criterion, or quantify and model its noise explicitly.
7. Add same-seed serial-versus-parallel reproducibility tests and different-seed sensitivity tests. Persist sampled folds and effective seeds.
```

### MF-03 — Medium, conditional: custom windows can use future-derived backfills or partial actual quarters

Two conditional data paths are unsafe outside the checked default window.

[`repair_short_history_vintages`](src/paper_hyperparameter_optimization/data_utils.py#L258) uses the latest FRED frame if no earlier repaired vintage reaches the estimation start. A custom run beginning with a short-history vintage can therefore derive pre-sample growth from a reference published after the target vintage. The checked default panel does not trigger this branch for PCEC96/FPIC1 because its earliest vintages already reach before 1967; truncated/custom origin sets can.

[`build_quarterly_evaluation_frame`](src/paper_hyperparameter_optimization/data_utils.py#L513) averages whatever months exist. It can treat a one- or two-month partial quarter as a realized quarterly mean. The current 2012 actual frame, for example, has a 2012Q1 SP500 value based only on January. Default targets end in 2011Q4, so the standard run avoids this case.

#### Repair prompt

```text
Enforce point-in-time provenance for non-GLP backfills and completeness for realized quarterly actuals.

Requirements:
1. Every backfill reference must have a known vintage date no later than the target vintage. Remove the implicit latest-FRED fallback or require an explicitly dated historical seed vintage.
2. Record provenance for every synthesized span: target vintage, reference vintage, anchor date, and transformation.
3. Add a sentinel test where the latest frame contains unique future-only growth rates and prove they cannot appear in an earlier repaired vintage. Retain a default-panel test proving later short histories use earlier real-time vintages.
4. When aggregating monthly actuals to quarters, compute per-variable counts and require all expected months. Set partial quarters to missing and record the reason.
5. In extract_forecasts, require a complete/non-null actual metric before scoring; do not treat a partial average as final truth.
6. Add one-, two-, and three-month quarter tests and a custom end-date integration test reaching the evaluation vintage's partial final quarter.
```

### MF-04 — Medium: reported growth intervals are not quantiles of growth draws

[`extract_forecasts`](src/paper_hyperparameter_optimization/forecasting.py#L254) applies the growth transform separately to the mean, median, p95, p84, p16, and p05 level curves. In general,

`log(p95(level_t)) - log(p95(level_t-1))`

is not the 95th percentile of `log(level_t / level_t-1)`. The current interval columns discard cross-horizon dependence. Point-RMSE ranking uses `mean_metric`, so this does not cause the rank reversals above, but the published predictive intervals are statistically invalid.

#### Repair prompt

```text
Compute transformed predictive summaries from joint posterior paths.

Requirements:
1. Retain or expose forecast draws with their draw identity across target periods.
2. For growth variables, calculate 100*log(Y_t/Y_t-1) draw by draw using paired adjacent levels, then compute the mean, median, and requested quantiles from those transformed draws.
3. Do not difference marginal quantile curves. Keep level summaries separate from transformed-metric summaries.
4. Ensure the first growth horizon uses the correct preceding observed/nowcast level for each draw without future leakage.
5. Add a two-period correlated-draw fixture where the correct growth p95 is hand-computable and demonstrably differs from the old result.
6. Rename or remove any legacy interval columns that cannot be reconstructed correctly and document regenerated outputs.
```

### OPS-01 — Medium/high: comparison input resolution can silently select stale, incompatible runs

[`resolve_experiment_dir`](scripts/compare_forecasts.py#L24) tries a checked-in Euler fallback even when the user supplied an explicit invalid path. A typo such as `/definitely/not/the/requested/run` therefore resolves to unrelated stale output if the fallback exists.

The reporting layer also does not validate run metadata, actual vintage, date range, aggregation, horizon semantics, dependency revision, data hash, or identical actuals. This combines badly with the repository state: root `outputs/paper_hyperparameters/forecast_panel.csv` is a one-byte empty file while populated comparison tables exist, so the published tables are not reproducible solely from their nominal root inputs.

#### Repair prompt

```text
Make comparison input resolution strict and provenance-aware.

Requirements:
1. An explicit nonexistent, empty, or incomplete path must raise. Remove implicit Euler fallback for explicit arguments; if legacy discovery remains, make it opt-in and print the resolved source prominently.
2. Load run_metadata.json for every input and require compatibility across actual vintage, origin range, horizon semantics, aggregation/transforms, model size/universe, selection schedule, data fingerprint, and dependency revision.
3. Verify identical actual values on common forecast keys and reject duplicate keys before scoring.
4. Add a machine-readable comparison manifest listing every exact input directory/file hash and exclusion decision.
5. Add tests for a typo path, empty one-byte panel, incompatible actual vintages, mismatched actual values, and stale fallback discovery.
6. Ensure checked-in comparison outputs can be regenerated from checked-in or explicitly documented external artifacts; otherwise remove or clearly label orphaned derived tables.
```

### OPS-02 — High operational: the documented Euler entry point is broken

[`scripts/run_everything_euler.sh`](scripts/run_everything_euler.sh) changes to `SLURM_SUBMIT_DIR` and invokes `python download_data.py` and other root-level filenames, although the entry points live under `scripts/`. It comments out the paper baseline, then asks the comparison stage to load that baseline. [`README.md`](README.md) documents invoking this shell script from the repository root and claims the baseline runs.

#### Repair prompt

```text
Repair and test the Euler batch entry point.

Requirements:
1. Derive REPO_ROOT from the shell script's own location, not SLURM_SUBMIT_DIR, and invoke every Python entry point through an absolute repository path such as $REPO_ROOT/scripts/<name>.py.
2. Add explicit stage controls. Run the paper baseline by default, or refuse the comparison stage when a required baseline was not requested/completed.
3. Use strict shell settings and stop on a failed stage. Write a stage manifest/status file and do not compare partial stale outputs.
4. Quote paths and make output directories explicit.
5. Add a static/shell smoke test launched both from the repository root and a different temporary submit directory, with Python commands stubbed so no long fit runs.
6. Update README.md so its invocation, stages, and output locations exactly match the script.
```

### DOC-01 — Low: methodology text no longer matches the code

The existing [`GLP_RMSE_REVIEW.md`](GLP_RMSE_REVIEW.md) and [`README_GLP_METHODOLOGY.md`](README_GLP_METHODOLOGY.md) describe the RMSE objective as a posterior-mode forecast. Current defaults average 200 posterior beta draws in [`glp_model.py`](src/glp_hyperparameter_optimization/glp_model.py#L733) and the GLP launchers. The prior review's main recommended change has already been implemented, so that file is stale.

The methodology also defines relative RMSE as `100 * RMSE / baseline` (baseline 100), while both reporters implement percentage change `(RMSE - baseline) / baseline * 100` (baseline 0). This does not change ordering, but it changes interpretation and table labels.

#### Repair prompt

```text
Synchronize methodology and output definitions with the current implementation.

Requirements:
1. Update GLP_RMSE_REVIEW.md and README_GLP_METHODOLOGY.md to describe the current predictive-mean objective, n_obj_draws behavior, deterministic common-random-number seeding, and the retrospective-versus-realtime validation distinction.
2. Choose one relative-RMSE convention: ratio index with baseline 100, or percent change with baseline 0. Make code, column names, plot axes, tests, and both READMEs agree.
3. Mark superseded findings/recommendations explicitly rather than leaving them as current conclusions.
4. Add small tests for the chosen relative-RMSE definition and a documentation assertion/example showing the expected baseline and winning direction.
5. Cross-link this implementation audit and state which old output tables require regeneration after the code fixes.
```

---

## Confirmed-correct index and leakage properties

The following concerns were investigated and ruled out for the current code paths:

1. **GLP inner split arithmetic:** [`_rmse_eval_origins`](src/glp_hyperparameter_optimization/glp_model.py#L681), [`_build_rmse_origins`](src/glp_hyperparameter_optimization/glp_model.py#L712), and [`_rmse_objective`](src/glp_hyperparameter_optimization/glp_model.py#L733) correctly separate training and holdout rows. There is no overlap and `h_eval` is converted correctly from one-based to zero-based indexing.
2. **GLP forecast seed observation:** trimming the initial lag rows in `prepare_glp_context` does not remove the last observation; recursive forecasts start from the correct final training values.
3. **GLP objective directions:** the paper method minimizes the negative log posterior, Mango MDD maximizes the positive log posterior, and both RMSE variants minimize. There is no max/min sign inversion.
4. **GLP variable selection:** `_resolve_var_indices` aligns by variable name, deduplicates, and rejects missing/empty targets correctly.
5. **Random fold reuse:** random origins are sampled once without replacement before candidate evaluations and reused for all candidates.
6. **Outer final-vintage isolation:** the fixed final actual vintage is loaded for scoring only and is not passed into fitting or hyperparameter selection in either local workflow.
7. **Small/medium GLP target alignment:** for every committed small/medium origin, the effective last complete quarter is one quarter behind the quarter-end vintage, so `last_quarter + 1` is the vintage's calendar quarter. In that convention h1 is a nowcast; it is not an off-by-one error.
8. **One-time RMSE selection:** selecting on the earliest requested outer origin and reusing parameters later is stale/adaptive only by design, but it does not use later outer evaluation outcomes. It should still be compared under an equal selection schedule as noted in SEL-02.
9. **Non-GLP outer vintage lookup:** `pivot_vintage_panel` uses exact vintage equality; it does not silently use the latest panel.
10. **Inspected MBFVAR train/holdout slices:** at dependency commit `5b06f93272cd6ebf370fbf2aac3b3573c7830493`, rolling and random RMSE training blocks end before their H-quarter holdouts and their `h_eval-1` row alignment is correct. The major non-GLP defects are the metric, feasibility, calendar endpoint, and stochastic evaluation—not direct overlap of these slices.

## Validation performed and limitations

The audit used:

- complete static tracing of `src/`, `scripts/`, and relevant tests;
- independent GLP, ranking, and non-GLP reviews;
- direct diagnostics of both processed panels and committed forecast artifacts;
- an audit of the MBFVAR source at commit `5b06f93272cd6ebf370fbf2aac3b3573c7830493`, the upstream HEAD retrieved during this review;
- explicit reproductions of future-dated GLP rows, stale large-model targets, missing non-GLP h8 rows, pooled horizon variants, and paired-sample rank reversals.

The Python test suite could not be executed in the supplied runtime because `pytest`, `covbayesvar`, and `MBFVAR` are not installed. `python -m pytest --version` fails with `No module named pytest`. Findings that depend on model behavior were therefore checked through source tracing and committed artifacts rather than a fresh full fit. This limitation does not affect the direct data/output reproductions or the pure index arithmetic conclusions.

## Recommended remediation order

1. Stop using existing large-GLP and non-GLP comparison tables for conclusions.
2. Fix GLP-01/02 and MF-01/02, then invalidate affected data and forecast artifacts.
3. Fix RANK-01/02 before producing any new comparison table.
4. Add SEL-01 fail-fast checks so bad optimizations cannot quietly create plausible-looking parameters.
5. Pin dependencies and RNG behavior (REPRO-01), then regenerate every strategy from one compatible manifest.
6. Address the medium/conditional data and inference issues, update documentation, and rerun paired reports with coverage counts.
