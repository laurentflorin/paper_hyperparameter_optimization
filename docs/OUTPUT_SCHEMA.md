# Output Schema Reference

Every scope-grid runner writes canonical output files to each run directory.
This document is the authoritative schema reference. All three runners
(ridge, GLP, MFVAR) share the same column layout for the key outputs, which
makes them directly comparable by `scripts/compare_scope_study.py`.

---

## Run-directory layout

```
<output_root>/
  scope_<scope>/              # one sub-directory per selection scope
    run_manifest.json         # written BEFORE expensive work
    run_status.json           # partial | failed | cancelled | complete
    run_complete.json         # written only after schema validation passes
    run_metadata.json         # final provenance record
    forecast_panel.csv
    selected_hyperparameters.csv
    failed_origins.csv
  benchmarks/
    <strategy>/               # one sub-directory per benchmark strategy
      run_manifest.json
      run_status.json
      run_complete.json
      run_metadata.json
      forecast_panel.csv
      selected_hyperparameters.csv   # benchmark parameter choices (may be empty)
      failed_origins.csv
```

---

## `forecast_panel.csv`

One row per (forecast_origin, variable, horizon). This is the only file
that enters RMSE/MAE/relative-RMSE computation.

| Column | Type | Notes |
|---|---|---|
| `strategy` | string | Runner-assigned model label, e.g. `ridge_var`, `glp`, `mfbvar`. |
| `forecast_origin` | string or integer | For real-data runs: ISO date of the outer evaluation origin. For synthetic runs without a date column: integer row index of the last training observation. |
| `group` | string | Group name for variable-group scopes; `all` when no grouping is active. |
| `target_quarter` | string or integer | Label of the forecast target. ISO date or row index. |
| `horizon_quarters` | integer | Forecast horizon in quarters (1, 2, 4, 8 in most studies). |
| `variable` | string | Variable name as it appears in the input panel. |
| `forecast_method` | string | `iterated` or `direct` (ridge); `mcmc_draws` (GLP/MFVAR). |
| `actual_level` | float or NaN | Back-transformed level-scale actual. NaN when the runner operates on already-transformed data (ridge). |
| `actual_metric` | float | Transformed actual at the evaluation scale. Always populated. |
| `mean_level` | float or NaN | Posterior mean forecast at level scale. NaN for ridge. |
| `mean_metric` | float | Posterior mean (or point) forecast at the evaluation scale. The primary forecast for RMSE computation. |
| `median_level` | float or NaN | Posterior median at level scale. NaN for ridge (mean = median for point forecasts). |
| `median_metric` | float | Posterior median at evaluation scale. |
| `error_metric` | float | `actual_metric − mean_metric`. Positive = overprediction. |

Additional quantile columns (`p95_metric`, `p84_metric`, `p16_metric`,
`p05_metric`) are present in GLP and MFVAR outputs but not in ridge outputs.

---

## `selected_hyperparameters.csv`

One row per selection event. Each selection event covers one (scope-cell,
outer-origin) combination where inner validation was re-run.

### Ridge

| Column | Notes |
|---|---|
| `forecast_origin` | Outer origin at which this selection event was triggered. |
| `group` | Group label (see `forecast_panel.csv`). |
| `strategy` | `ridge_var`. |
| `cell_id` | Scope cell this event belongs to (e.g. `pooled`, `gdp`, `gdp-h1`). |
| `event_id` | Unique event identifier: `sel-NNN-oMMM`. |
| `param_lam` | Selected ridge regularization λ. |
| `param_p` | Selected VAR lag order p. |
| `param_alpha` | Selected α (mixing weight; `0.0` = pure ridge). |
| `param_kappa` | Selected κ (lag-decay; `1.0` = uniform). |
| `selection_loss` | Inner validation RMSE achieved by the selected candidate. |
| `n_tied` | Number of grid candidates tied at the best loss (tie-breaking selects the smallest λ). |

### GLP

| Column | Notes |
|---|---|
| `forecast_origin` | Outer origin at which this event was triggered. |
| `group` | Group label. |
| `cell_id` | Scope cell identifier. |
| `selection_event_id` | Unique event identifier. |
| `selection_loss` | Inner validation RMSE. |
| `param_lambda` | Selected overall tightness λ. |
| `param_theta` | Selected sum-of-coefficients θ. |
| `param_miu` | Selected cross-variable shrinkage μ. |
| `param_psi` | Selected or fixed ψ (IW prior scale). |

### MFVAR

| Column | Notes |
|---|---|
| `forecast_origin` | Outer origin at which this event was triggered. |
| `group` | Group label. |
| `cell_id` | Scope cell identifier. |
| `selection_event_id` | Unique event identifier. |
| `selection_loss` | Inner validation RMSE. |
| `param_lambda1_1` | Selected Minnesota tightness λ₁. |
| `param_lambda2_1` | Selected sum-of-coefficients λ₂. |
| `param_lambda3_1` | Fixed at `1.0`. |
| `param_lambda4_1` | Selected lag-decay λ₄. |
| `param_lambda5_1` | Selected lag-decay λ₅. |

### Benchmark `selected_hyperparameters.csv`

Benchmark strategies (`no_change`, `var_aic`, `var_bic`, `ar_univariate`,
`var_nested_loss`) write a parameter table in a different shape:

| Column | Notes |
|---|---|
| `forecast_origin` | Outer origin. |
| `strategy` | Benchmark name. |
| `group` | Group label. |
| `parameter` | Parameter name (e.g. `lag_order` for VAR-based benchmarks). |
| `value` | Selected parameter value. |

For the `no_change` benchmark there are no free parameters and this file is
written with zero data rows (header only).

---

## `failed_origins.csv`

One row per failed forecast origin or failed benchmark run.

### Scope failures

| Column | Notes |
|---|---|
| `forecast_origin` | Origin where the failure occurred. |
| `cell_id` | Cell affected. |
| `stage` | `selection`, `forecast`, or `benchmark`. |
| `failure_category` | Structured error class (see below). |
| `error` | Full exception message. |

### Benchmark failures

| Column | Notes |
|---|---|
| `forecast_origin` | Origin where the failure occurred. |
| `stage` | `benchmark`. |
| `failure_category` | Structured error class. |
| `error` | Full exception message. |

### `failure_category` values

| Category | Meaning |
|---|---|
| `data_insufficient` | Not enough training rows to fit the model at this split. |
| `singular_matrix` | Covariance or design matrix is singular or near-singular. |
| `numerical_divergence` | MCMC or optimizer returned non-finite values. |
| `optimization_failed` | Bayesian optimizer produced a degenerate result. |
| `forecast_invalid` | Forecast row produced NaN or Inf outputs. |
| `timeout` | Wall-time limit exceeded (not yet enforced in runners). |
| `unknown` | Catch-all for exceptions that did not match a known category. |

---

## `run_manifest.json`

Written atomically before expensive work. Records the full configuration
that identifies the run for resume/skip decisions. Key fields:

| Field | Type | Notes |
|---|---|---|
| `configuration_hash` | string (SHA-256) | Stable hash of the entire scientific configuration, including code state, package versions, and input fingerprints. Used to detect incompatible resumes. |
| `runner` | string | e.g. `regularized_var_scope_grid`. |
| `command_line` | string | Full CLI invocation. |
| `argv` | array | Raw `sys.argv` at launch time. |
| `repository_commit` | string | Git commit hash at the time of the run. |
| `repository_dirty` | bool | Whether the worktree had uncommitted changes. |
| `utc_started_at` | string | ISO 8601 Z timestamp. |
| `python_version` | string | Python interpreter version. |
| `platform` | string | OS/hardware string. |
| `relevant_packages` | object | `{package: version}` for scientific dependencies. |
| `model_family` | string | `ridge`, `glp`, `mfbvar`, etc. |
| `data_source` | object | Panel path, format, and vintage identifier. |
| `input_files` | array | Per-file SHA-256 fingerprints. |
| `search_space` | object | Grid or optimizer bounds. |
| `optimizer_budget` | object | Candidate evaluation counts. |
| `random_seeds` | object | All relevant seeds. |
| `output_schema_version` | string | Schema version string (currently `"1"`). |

---

## `run_status.json`

Written after each state change. Contains:

```json
{
  "status": "partial | failed | cancelled | complete",
  "configuration_hash": "<sha256>",
  "utc_updated_at": "<ISO8601Z>"
}
```

---

## `run_complete.json`

Written only after all required output files pass schema validation. Contains
the same fields as `run_status.json` with `"status": "complete"`. The
presence of this file is the authoritative "this run is safe to use"
signal for downstream reporting.

---

## Comparison output files

`scripts/compare_scope_study.py` reads a manifest of panels and writes:

| File | Content |
|---|---|
| `rmse_by_target.csv` | RMSE per (model, variable, horizon), computed on the cell-wise common sample. Columns `n` (observations used), `n_common` (shared observations in the cell), `n_model_total` (the model's own observations), `n_excluded` (`n_model_total - n_common`). |
| `mae_by_target.csv` | MAE per (model, variable, horizon), same sample and same `n`/`n_common`/`n_model_total`/`n_excluded` columns as `rmse_by_target.csv`. |
| `relative_rmse.csv` | Relative RMSE vs. baseline per (model, variable, horizon). Both numerator and denominator are recomputed on the sample the model shares **with the baseline**, which is not necessarily the sample shared with all models: `n` (model observations used), `n_baseline` (baseline observations used, equal to `n` under the pairwise basis), `n_common`, and `sample_basis` (`pairwise_common_with_baseline` or `as_supplied`). |
| `average_ranks.csv` | Average rank across (variable, horizon) cells per model, plus `n_cells`, `n`, `n_common` (summed over cells) and `sample_basis` (`common_sample` or `unpaired_per_model`). Ranking refuses cells whose models rest on different sample sizes, except under `--coverage-policy advisory`, where nothing is restricted by design and the ranks are flagged `unpaired_per_model`. |
| `scope_gains.csv` | Additive scope-gain decomposition (horizon, variable, interaction). Losses are compared within one forecasting system (family, size, forecast_method) on that system's common sample: `n_common`, `n_models`, `n_excluded`, `sample_basis`. |
| `scope_gain_summary.csv` | Summary statistics of scope gains. |
| `hyperparameter_summary.csv` | Descriptive statistics of selected parameters. |
| `selection_stability.csv` | Variation of selected parameters across selection events. |
| `failure_summary.csv` | Failure counts by model and stage. |
| `computational_cost.csv` | Elapsed times from `run_metadata.json`. |
| `dm_tests.csv` | Diebold-Mariano test statistics and p-values for pairwise comparisons. |
| `bootstrap_intervals.csv` | Block-bootstrap confidence intervals for the **paired mean loss differential** `mean(L(model_a) - L(model_b))`, where `L` is the per-observation squared or absolute error (column `loss`). This is a difference of losses, **not** a ratio such as relative RMSE. Columns: `mean_diff`, `ci_lower`, `ci_upper`, `n` (paired observations), `method`, `block_length`, `n_boot`, `seed`, `valid`. A negative `mean_diff` favours `model_a`. |
| `common_sample.csv` | Common-sample audit: one row per model plus an `__all__` row, with `policy` (`restrict`/`raise`/`advisory`), `coverage` (retained share of observations), `n_common_keys`, `n_excluded_keys`, and per-model `n_model_total`/`n_common`/`n_excluded`. |
| `comparison_summary.md` | Human-readable summary of key findings, including the coverage policy and the number of excluded observation keys. |

Every table above is built from a single frame that `--coverage-policy` /
`--min-coverage` are threaded into consistently: under `restrict` (default) the
frame is restricted once and each per-table restriction is an exact no-op; under
`advisory` the shortfall is reported in `common_sample.csv` but **no** table
drops observations, so `n` is each model's own row count and `sample_basis`
reads `unpaired_per_model`.
