# Current State Audit

Audit date: 2026-08-18  
Repository path: `/home/u80856195/git/paper_hyperparameter_optimization`  
Audited HEAD commit: `afa5977ac90d617e681bd5fe7d9b1c200864a50f`  
Audit basis: the live dirty worktree, not a clean checkout of `HEAD`  
Runtime/code changes made during this audit: none  
Validation interpreter: `/home/u80856195/.virtualenvs/venv/bin/python` (Python 3.12.13)

## Executive summary

The repository is currently in a transitional but clearly staged state. Several of the most important fixes proposed in [IMPLEMENTATION_AUDIT.md](../IMPLEMENTATION_AUDIT.md) are already present in uncommitted source: strict paired comparison/reporting, shared comparison manifests and exclusion audits, GLP point-in-time validation helpers, GLP fixed sample-start enforcement, mixed-frequency predictive-mean RMSE objective code, reproducibility helpers, and a repaired Euler batch launcher.

The same worktree is not runnable end-to-end yet. Three concrete blockers stop full pytest collection today:

1. [src/paper_hyperparameter_optimization/forecasting.py](../src/paper_hyperparameter_optimization/forecasting.py#L361) contains a malformed `hyperparameter_record` block and raises an `IndentationError` at [src/paper_hyperparameter_optimization/forecasting.py](../src/paper_hyperparameter_optimization/forecasting.py#L373).
2. [src/glp_hyperparameter_optimization/forecasting.py](../src/glp_hyperparameter_optimization/forecasting.py#L59) imports `update_hyperparameters_mango_rmse` and `update_hyperparameters_mango_rmse_random`, but [src/glp_hyperparameter_optimization/glp_model.py](../src/glp_hyperparameter_optimization/glp_model.py#L1118) still contains the `# RMSE_UPDATERS_IMPLEMENTATION` placeholder instead of those exported functions.
3. [scripts/glp/compare_glp_forecasts.py](../scripts/glp/compare_glp_forecasts.py#L118) contains duplicate `_resolve_strategy_dir` and `main` definitions, which is not yet breaking import, but is active code drift and already visible in tests.

Observed outputs are also mixed-generation. The GLP checked-in run artifacts under [outputs/glp](../outputs/glp) look populated and broadly aligned with the current code shape. The root mixed-frequency run artifact under [outputs/paper_hyperparameters](../outputs/paper_hyperparameters) is incomplete (`n_origins_completed = 0`, blank CSV payloads), and the checked-in mixed-frequency comparison artifacts under [outputs/comparison](../outputs/comparison) appear older than the current strict reporting implementation.

## Snapshot of the live worktree

### Dirty-state categories

| Category | Paths | Why it matters |
|---|---|---|
| Modified runtime files | `requirements.txt`, `scripts/compare_forecasts.py`, `scripts/glp/compare_glp_forecasts.py`, `scripts/run_everything_euler.sh`, `src/paper_hyperparameter_optimization/*.py`, `src/glp_hyperparameter_optimization/*.py` | Current behavior depends on these local modifications rather than clean `HEAD`. |
| Untracked runtime files | `requirements.lock`, `src/experiment_provenance.py`, `src/forecast_comparison.py`, `src/paper_hyperparameter_optimization/_strict_reporting.py`, `src/glp_hyperparameter_optimization/_strict_reporting.py` | These are substantive new modules; the repository’s effective behavior already depends on them. |
| Untracked snapshots / merge remnants | `*.orig`, `scripts/glp/compare_glp_forecasts.py.rej` | Evidence of in-progress migration; do not overwrite or re-derive blindly. |
| Generated state | `__pycache__/`, `.pytest_cache/`, new pytest bytecode files | Noise for implementation, but useful evidence that local test execution already occurred. |
| New test files | `tests/test_experiment_provenance.py`, `tests/test_reporting_integrity.py`, `tests/test_reporting_ops.py` and `.orig` variants | The local worktree already contains tests for the new comparison and provenance layer. |

### Current repository map

| Area | Primary paths | Current role | Current state |
|---|---|---|---|
| Mixed-frequency workflow | [src/paper_hyperparameter_optimization](../src/paper_hyperparameter_optimization) | Schorfheide-Song recursive forecasting, local Mango objectives, reporting | Richly extended, but currently import-broken by syntax/import drift. |
| GLP workflow | [src/glp_hyperparameter_optimization](../src/glp_hyperparameter_optimization) | GLP recursive forecasting, data preparation, reporting | Data and reporting layers are advanced; RMSE strategy entrypoints are currently import-broken. |
| Shared comparison layer | [src/forecast_comparison.py](../src/forecast_comparison.py) | Strict run discovery, metadata compatibility, paired RMSE, manifest writing | New and active; both workflows now route reporting through it. |
| Reproducibility layer | [src/experiment_provenance.py](../src/experiment_provenance.py) | Repo/dependency/data hashes, MBFVAR revision checks, deterministic seeds | Implemented and tested, but not fully wired into the mixed-frequency runtime. |
| Mixed-frequency CLIs | [scripts](../scripts) | Download, run paper/MDD/RMSE strategies, compare runs, Euler batch entrypoint | Present and mostly coherent; compare script is strict/fail-closed. |
| GLP CLIs | [scripts/glp](../scripts/glp) | Download, run paper/MDD/RMSE strategies, compare runs, batch orchestration | Present; batch orchestration is modernized, compare script has duplicate defs. |
| Checked-in artifacts | [outputs](../outputs) | Example outputs and comparison tables | Heterogeneous generation state; mixed-frequency root outputs are incomplete. |
| Tests | [tests](../tests) | Coverage for configs, data utilities, reporting, CLI behavior, provenance | Good coverage for new reporting/provenance layer; forecasting layer blocked by current code errors. |

## Workflow call graphs

### Mixed-frequency data pipeline

```text
scripts/download_data.py
  -> paper_hyperparameter_optimization.data_utils.download_realtime_panel(...)
     -> per-series ALFRED/FRED/Stooq caching
     -> realtime_panel.csv.gz
     -> latest_panel.csv.gz
     -> download_metadata.json
```

### Mixed-frequency forecasting pipeline

```text
scripts/run_paper_hyperparameters.py
  -> paper_hyperparameter_optimization.forecasting.run_from_namespace("paper", ...)
     -> run_recursive_experiment(...)
        -> _run_origin_task(...)
           -> load_realtime_panel(...)
           -> build_model_input_frames(...)
           -> build_quarterly_evaluation_frame(...)
           -> MBFVAR fit/forecast/aggregate
           -> extract_forecasts(...)
           -> forecast_panel.csv / selected_hyperparameters.csv / failed_origins.csv / run_metadata.json

scripts/run_mango_mdd.py
  -> run_from_namespace("mango_mdd", ...)
     -> run_recursive_experiment(...)
        -> select_hyperparameters(...)
           -> MBFVAR update_hyperparameters_mango(...)

scripts/run_mango_rmse.py and scripts/run_mango_rmse_random.py
  -> per-horizon wrapper loop (h1q, h2q, h4q, h8q)
  -> run_from_namespace("mango_rmse[_random]", ...)
     -> run_recursive_experiment(...)
        -> _run_local_mango_optimizer(...)
           -> build_rmse_validation_folds(...)
           -> _rmse_candidate_score(...)
              -> aggregate_quarterly_posterior_draws(...)
              -> summarize_quarterly_draws(...)
              -> compute_quarterly_metrics(...)
```

Current breakpoints in this path:

- [src/paper_hyperparameter_optimization/forecasting.py](../src/paper_hyperparameter_optimization/forecasting.py#L26) imports `DEFAULT_RANDOM_SEED`, but [src/paper_hyperparameter_optimization/config.py](../src/paper_hyperparameter_optimization/config.py) does not define it.
- [src/paper_hyperparameter_optimization/forecasting.py](../src/paper_hyperparameter_optimization/forecasting.py#L19) imports `runtime_provenance` and `validate_mbfvar_revision`, but neither name is currently used in the file.

### Mixed-frequency comparison pipeline

```text
scripts/compare_forecasts.py
  -> resolve_experiment_dir(...)
     -> forecast_comparison.discover_run_directories(...)
  -> paper_hyperparameter_optimization.reporting.create_comparison_report(...)
     -> paper_hyperparameter_optimization._strict_reporting.create_comparison_report(...)
        -> forecast_comparison.validate_metadata_compatibility(...)
        -> forecast_comparison.compute_paired_rmse_table(...)
        -> forecast_comparison.write_comparison_manifest(...)
```

The effective runtime implementation is the strict overlay imported at the end of [src/paper_hyperparameter_optimization/reporting.py](../src/paper_hyperparameter_optimization/reporting.py#L248), not the legacy logic earlier in that file.

### GLP data pipeline

```text
scripts/glp/download_glp_data.py
  -> glp_hyperparameter_optimization.data_utils.download_glp_realtime_panel(...)
     -> download_series_vintage(...)
     -> validate_glp_realtime_panel(...)
     -> glp_realtime_panel.csv.gz
     -> glp_latest_panel.csv.gz
     -> glp_download_metadata.json
```

### GLP forecasting pipeline

```text
scripts/glp/run_glp_paper.py or scripts/glp/run_glp_mango.py
  -> glp_hyperparameter_optimization.forecasting.run_from_namespace(...)
     -> run_glp_experiment(...)
        -> build_glp_estimation_matrix(...)
        -> select_hyperparameters(...)
        -> predictive_draws(...)
        -> _forecast_rows(...)
        -> forecast_panel.csv / selected_hyperparameters.csv / failed_origins.csv / run_metadata.json

scripts/glp/run_glp_mango_rmse.py, scripts/glp/run_glp_mango_rmse_random.py, scripts/glp/run_glp_all.py
  -> per-horizon wrapper loop
  -> run_glp_experiment(...)
     -> select_hyperparameters(...)
        -> expected to call update_hyperparameters_mango_rmse(_random)
        -> currently blocked because those exports are missing from glp_model.py
```

Important partially implemented GLP internals that already exist below the breakage line:

- true calendar-keyed inner-vintage helper [src/glp_hyperparameter_optimization/glp_model.py](../src/glp_hyperparameter_optimization/glp_model.py#L911)
- predictive-mean capable RMSE objective [src/glp_hyperparameter_optimization/glp_model.py](../src/glp_hyperparameter_optimization/glp_model.py#L985)
- fixed sample start and information-lag enforcement in [src/glp_hyperparameter_optimization/data_utils.py](../src/glp_hyperparameter_optimization/data_utils.py#L659) and [src/glp_hyperparameter_optimization/data_utils.py](../src/glp_hyperparameter_optimization/data_utils.py#L681)

### GLP comparison pipeline

```text
scripts/glp/compare_glp_forecasts.py or scripts/glp/run_glp_all.py --stages compare
  -> glp_hyperparameter_optimization.reporting.create_glp_comparison_report(...)
     -> glp_hyperparameter_optimization._strict_reporting.create_glp_comparison_report(...)
        -> shared forecast_comparison helpers
```

The effective runtime implementation is the strict overlay imported at the end of [src/glp_hyperparameter_optimization/reporting.py](../src/glp_hyperparameter_optimization/reporting.py#L306).

## Public scripts and current status

| Script | Role | Primary implementation target | Current status |
|---|---|---|---|
| [scripts/download_data.py](../scripts/download_data.py) | Mixed-frequency real-time data download | `download_realtime_panel` | Runnable in principle; no import blocker observed. |
| [scripts/run_paper_hyperparameters.py](../scripts/run_paper_hyperparameters.py) | Mixed-frequency paper strategy | `run_from_namespace("paper")` | Blocked by mixed-frequency forecasting syntax/import errors. |
| [scripts/run_mango_mdd.py](../scripts/run_mango_mdd.py) | Mixed-frequency Mango MDD | `run_from_namespace("mango_mdd")` | Blocked by same mixed-frequency forecasting issues. |
| [scripts/run_mango_rmse.py](../scripts/run_mango_rmse.py) | Mixed-frequency RMSE batch | horizon loop + `run_from_namespace("mango_rmse")` | Logic is present; currently blocked by mixed-frequency forecasting import failure. |
| [scripts/run_mango_rmse_random.py](../scripts/run_mango_rmse_random.py) | Mixed-frequency random-RMSE batch | horizon loop + `run_from_namespace("mango_rmse_random")` | Logic is present; currently blocked by mixed-frequency forecasting import failure. |
| [scripts/compare_forecasts.py](../scripts/compare_forecasts.py) | Mixed-frequency paired comparison | strict reporting via shared helpers | Runnable and tested; explicit paths now fail closed. |
| [scripts/run_everything_euler.sh](../scripts/run_everything_euler.sh) | Mixed-frequency batch orchestration | script-derived `REPO_ROOT`, stage manifest, compare gating | Repaired relative to older audit; current smoke tests pass. |
| [scripts/glp/download_glp_data.py](../scripts/glp/download_glp_data.py) | GLP real-time data download | `download_glp_realtime_panel` | Runnable in principle; strict vintage validation is implemented. |
| [scripts/glp/run_glp_paper.py](../scripts/glp/run_glp_paper.py) | GLP paper strategy | `run_from_namespace("paper")` | Forecasting module imports, but whole suite blocked elsewhere. |
| [scripts/glp/run_glp_mango.py](../scripts/glp/run_glp_mango.py) | GLP Mango MDD | `run_from_namespace("mango_mdd")` | Same as above. |
| [scripts/glp/run_glp_mango_rmse.py](../scripts/glp/run_glp_mango_rmse.py) | GLP RMSE batch | horizon loop + `run_from_namespace("mango_rmse")` | Broken by missing updater exports. |
| [scripts/glp/run_glp_mango_rmse_random.py](../scripts/glp/run_glp_mango_rmse_random.py) | GLP random-RMSE batch | horizon loop + `run_from_namespace("mango_rmse_random")` | Broken by missing updater exports. |
| [scripts/glp/run_glp_all.py](../scripts/glp/run_glp_all.py) | GLP shared batch runner | `run_glp_experiment`, `create_glp_comparison_report` | Good shared CLI surface; `mango_rmse*` stages inherit the missing-export blocker. |
| [scripts/glp/compare_glp_forecasts.py](../scripts/glp/compare_glp_forecasts.py) | GLP paired comparison | strict reporting via shared helpers | Functional intent is clear, but file contains duplicate definitions and a failing explicit-path test. |

## Output schemas and observed artifact state

### Per-run artifacts currently observed on disk

| Workflow | Directory | Forecast artifact | Hyperparameter artifact | Metadata observations |
|---|---|---|---|---|
| Mixed-frequency paper | [outputs/paper_hyperparameters](../outputs/paper_hyperparameters) | `forecast_panel.csv` exists but contains only whitespace | `selected_hyperparameters.csv` exists but contains only whitespace | `run_metadata.json` records `strategy=paper`, `actual_vintage=2012-01-31`, `optimization_variables=["GDP"]`, `n_origins_requested=151`, `n_origins_completed=0`; `panel_path` points to an external absolute path. |
| GLP paper small | [outputs/glp/paper_small](../outputs/glp/paper_small) | Header observed: `strategy,model_size,forecast_origin,target_quarter,horizon_quarters,variable,mean,actual,error,p05,p16,median,p84,p95` | Header observed: `forecast_origin,strategy,model_size,last_quarter,n_obs,lambda,theta,miu` | `run_metadata.json` records `strategy`, `model_size`, `actual_vintage`, `lags`, `mcmc_draws`, optimization horizon fields, and completed-origin counts. |

### Comparison artifacts currently observed on disk

| Directory | Observed schema/status | Interpretation |
|---|---|---|
| [outputs/comparison/rmse_all_variables.csv](../outputs/comparison/rmse_all_variables.csv) | Header: `model,group,variable,horizon_quarters,rmse` | This looks older than the current strict reporter; it lacks manifest/audit outputs and no variant column is present. |
| [outputs/comparison/relative_rmse_vs_paper.csv](../outputs/comparison/relative_rmse_vs_paper.csv) | Header: `model,group,variable,horizon_quarters,rmse,baseline_rmse,relative_rmse_pct` | Relative RMSE is already stored as percent change, not baseline-index-100. |
| [outputs/comparison/hyperparameter_summary.csv](../outputs/comparison/hyperparameter_summary.csv) | Aggregates `lambda1_1..lambda5_1` by `model` only | Another sign this predates horizon-variant-aware strict reporting. |
| [outputs/glp/comparison/rmse_all_variables.csv](../outputs/glp/comparison/rmse_all_variables.csv) | Header: `model,model_size,variable,horizon_quarters,optimization_horizon,rmse,n` | GLP comparison artifacts already preserve optimization horizon on disk. |
| [outputs/glp/comparison/relative_rmse_vs_glp.csv](../outputs/glp/comparison/relative_rmse_vs_glp.csv) | Header: `model,model_size,variable,horizon_quarters,optimization_horizon,rmse,n,baseline_rmse,relative_rmse_pct` | Broadly consistent with the strict paired-reporting direction. |
| [outputs/glp/comparison/hyperparameter_summary.csv](../outputs/glp/comparison/hyperparameter_summary.csv) | Groups by `model`, `model_size`, `optimization_horizon` | Variant awareness is already reflected in these artifacts. |

### Effective reporter contract in the current worktree

If the comparison scripts were rerun from the current dirty worktree, both workflows would route through [src/forecast_comparison.py](../src/forecast_comparison.py) and the strict overlays in [src/paper_hyperparameter_optimization/_strict_reporting.py](../src/paper_hyperparameter_optimization/_strict_reporting.py) and [src/glp_hyperparameter_optimization/_strict_reporting.py](../src/glp_hyperparameter_optimization/_strict_reporting.py). That code now expects to write:

- paired RMSE tables
- paired relative RMSE tables
- `comparison_exclusion_audit.csv`
- `comparison_manifest.json`
- variant-aware plots
- hyperparameter summaries that preserve optimization-horizon variants where applicable

The mixed-frequency checked-in comparison directory does not yet reflect that stricter output contract.

## Scientific defaults currently encoded in source

### Mixed-frequency defaults

| Setting | Value | Source |
|---|---|---|
| Forecast origins | `1997-07-31` to `2010-01-31` | [src/paper_hyperparameter_optimization/config.py](../src/paper_hyperparameter_optimization/config.py#L39) |
| Actual vintage | `2012-01-31` | [src/paper_hyperparameter_optimization/config.py](../src/paper_hyperparameter_optimization/config.py#L41) |
| Estimation start | `1967-01-01` | [src/paper_hyperparameter_optimization/config.py](../src/paper_hyperparameter_optimization/config.py#L42) |
| Max horizon | `24` months / `8` quarters | [src/paper_hyperparameter_optimization/config.py](../src/paper_hyperparameter_optimization/config.py#L44) |
| Paper fit draws | `PAPER_NSIM = 20000` | [src/paper_hyperparameter_optimization/config.py](../src/paper_hyperparameter_optimization/config.py#L57) |
| Paper lags | `[6]` | [src/paper_hyperparameter_optimization/config.py](../src/paper_hyperparameter_optimization/config.py#L60) |
| Paper hyperparameters | `[0.09, 4.30, 1.0, 2.70, 4.30]` | [src/paper_hyperparameter_optimization/config.py](../src/paper_hyperparameter_optimization/config.py#L61) |
| Optimization draws | `DEFAULT_OPTIMIZATION_NSIM = 5000` | [src/paper_hyperparameter_optimization/config.py](../src/paper_hyperparameter_optimization/config.py#L63) |
| Default selection schedule | `first_origin` | [src/paper_hyperparameter_optimization/config.py](../src/paper_hyperparameter_optimization/config.py#L50) |
| Valid selection schedules | `("first_origin", "per_origin")` | [src/paper_hyperparameter_optimization/config.py](../src/paper_hyperparameter_optimization/config.py#L51) |
| RMSE objective variable rule | full quarterly block only | [src/paper_hyperparameter_optimization/forecasting.py](../src/paper_hyperparameter_optimization/forecasting.py#L300) |

Notable drift: [src/paper_hyperparameter_optimization/config.py](../src/paper_hyperparameter_optimization/config.py#L53) documents a recorded master seed, but no `DEFAULT_RANDOM_SEED` symbol is actually defined.

### GLP defaults

| Setting | Value | Source |
|---|---|---|
| Forecast origins | `2000-03-31` to `2019-12-31` | [src/glp_hyperparameter_optimization/config.py](../src/glp_hyperparameter_optimization/config.py#L116) |
| Actual vintage | `2023-01-01` | [src/glp_hyperparameter_optimization/config.py](../src/glp_hyperparameter_optimization/config.py#L134) |
| Expected information lag | `1` quarter | [src/glp_hyperparameter_optimization/config.py](../src/glp_hyperparameter_optimization/config.py#L121) |
| Sample start, small/medium | `1959-01-01` | [src/glp_hyperparameter_optimization/config.py](../src/glp_hyperparameter_optimization/config.py#L107) |
| Sample start, large | `1973-01-01` | [src/glp_hyperparameter_optimization/config.py](../src/glp_hyperparameter_optimization/config.py#L110) |
| Declared large-model forecast end cap | `2016-03-31` | [src/glp_hyperparameter_optimization/config.py](../src/glp_hyperparameter_optimization/config.py#L128) |
| Default lags | `5` | [src/glp_hyperparameter_optimization/config.py](../src/glp_hyperparameter_optimization/config.py#L136) |
| Evaluation horizons | `[1, 2, 4, 8]` | [src/glp_hyperparameter_optimization/config.py](../src/glp_hyperparameter_optimization/config.py#L138) |
| Forecast MCMC draws/discard in driver | `2000 / 1000` | [src/glp_hyperparameter_optimization/forecasting.py](../src/glp_hyperparameter_optimization/forecasting.py#L68) |
| RMSE objective beta draws | `200` | [scripts/glp/run_glp_all.py](../scripts/glp/run_glp_all.py#L155) |

Notable drift: `GLP_MODEL_FORECAST_END` is declared, but it is not referenced anywhere outside [src/glp_hyperparameter_optimization/config.py](../src/glp_hyperparameter_optimization/config.py#L128).

## Test coverage and validation performed

### Commands run during this audit

```bash
/home/u80856195/.virtualenvs/venv/bin/python -m pytest --collect-only -q
/home/u80856195/.virtualenvs/venv/bin/python -m pytest \
  tests/test_compare_forecasts_defaults.py \
  tests/test_experiment_provenance.py \
  tests/test_glp_compare_forecasts.py \
  tests/test_glp_config.py \
  tests/test_glp_data_utils.py \
  tests/test_glp_model.py \
  tests/test_reporting_integrity.py \
  tests/test_reporting_ops.py -q
/home/u80856195/.virtualenvs/venv/bin/python -m pytest tests/test_glp_model.py -q -ra
```

### Full collection result

`pytest --collect-only -q` reached 30 collected tests and then stopped on 3 collection errors:

| Blocker | Failing test modules | Current root cause |
|---|---|---|
| GLP RMSE updater exports missing | `tests/test_glp_forecasting.py`, `tests/test_glp_run_all.py` | [src/glp_hyperparameter_optimization/forecasting.py](../src/glp_hyperparameter_optimization/forecasting.py#L59) imports names that do not exist in [src/glp_hyperparameter_optimization/glp_model.py](../src/glp_hyperparameter_optimization/glp_model.py#L1118). |
| Mixed-frequency syntax error | `tests/test_optimizer_variable_resolution.py` | [src/paper_hyperparameter_optimization/forecasting.py](../src/paper_hyperparameter_optimization/forecasting.py#L373) raises `IndentationError`. |

### Runnable independent subset result

The collection-independent subset finished with:

- `28 passed`
- `2 failed`
- `1 skipped`

Passing or skip-capable coverage came from:

| Test file | Area covered | Result |
|---|---|---|
| [tests/test_compare_forecasts_defaults.py](../tests/test_compare_forecasts_defaults.py) | mixed-frequency compare CLI defaults | passed |
| [tests/test_experiment_provenance.py](../tests/test_experiment_provenance.py) | provenance hashing, MBFVAR revision check, deterministic RNG helpers | passed |
| [tests/test_glp_compare_forecasts.py](../tests/test_glp_compare_forecasts.py) | GLP compare script discovery behavior | 2 pass, 1 fail |
| [tests/test_glp_config.py](../tests/test_glp_config.py) | GLP model hierarchy and transform conventions | passed |
| [tests/test_glp_data_utils.py](../tests/test_glp_data_utils.py) | GLP download parsing, fail-closed data behavior, proxy-series handling | 3 pass, 1 fail |
| [tests/test_glp_model.py](../tests/test_glp_model.py) | optional GLP model math/optimizer tests | skipped: `covbayesvar` missing |
| [tests/test_reporting_integrity.py](../tests/test_reporting_integrity.py) | shared paired-reporting logic and manifest/audit writing | passed |
| [tests/test_reporting_ops.py](../tests/test_reporting_ops.py) | compare CLI strict path handling and Euler dry-run behavior | passed |

The two current failing tests are both informative rather than random:

1. [tests/test_glp_compare_forecasts.py](../tests/test_glp_compare_forecasts.py#L45) still expects an explicit override directory to be accepted when empty; the current code now validates explicit directories via [src/forecast_comparison.py](../src/forecast_comparison.py#L55) and raises `FileNotFoundError` unless a complete run is present.
2. [tests/test_glp_data_utils.py](../tests/test_glp_data_utils.py#L23) still expects empty-ALFRED fallback to latest FRED; the current code now intentionally raises `DataDownloadError` in [src/glp_hyperparameter_optimization/data_utils.py](../src/glp_hyperparameter_optimization/data_utils.py#L298).

The optional blocker is explicit and reproducible:

- [tests/test_glp_model.py](../tests/test_glp_model.py) is skipped because `covbayesvar` is not importable in the configured environment.

## What is already implemented and should not be reimplemented

The largest risk in follow-on work is duplicating the local migration that is already present in this dirty worktree. The following items are already implemented enough that future work should finish wiring and validation rather than redesign them from scratch.

| Older finding / recommendation | Current status in live worktree | Do not reimplement; next step |
|---|---|---|
| `RANK-01`, `RANK-02`, much of `OPS-01` in [IMPLEMENTATION_AUDIT.md](../IMPLEMENTATION_AUDIT.md) | Implemented via [src/forecast_comparison.py](../src/forecast_comparison.py), plus strict overlays in both workflows | Keep the shared paired-reporting layer; fix forecasting breakages, then rerun and regenerate outputs. |
| Mixed-frequency strict reporting migration | Implemented; [src/paper_hyperparameter_optimization/reporting.py](../src/paper_hyperparameter_optimization/reporting.py#L248) is overlayed by [src/paper_hyperparameter_optimization/_strict_reporting.py](../src/paper_hyperparameter_optimization/_strict_reporting.py#L413) | Do not add another comparison implementation. Use the strict overlay already present. |
| GLP strict reporting migration | Implemented; [src/glp_hyperparameter_optimization/reporting.py](../src/glp_hyperparameter_optimization/reporting.py#L306) is overlayed by [src/glp_hyperparameter_optimization/_strict_reporting.py](../src/glp_hyperparameter_optimization/_strict_reporting.py#L216) | Same guidance: keep this path and remove legacy duplication only after tests are green. |
| `GLP-02` fail-closed point-in-time ingestion | Mostly implemented in [src/glp_hyperparameter_optimization/data_utils.py](../src/glp_hyperparameter_optimization/data_utils.py#L298) and [src/glp_hyperparameter_optimization/data_utils.py](../src/glp_hyperparameter_optimization/data_utils.py#L447) | Update stale tests and regenerate caches/artifacts; do not restore latest-FRED fallback. |
| `GLP-01` / `GLP-05` information-lag and sample-start enforcement | Mostly implemented in [src/glp_hyperparameter_optimization/data_utils.py](../src/glp_hyperparameter_optimization/data_utils.py#L659) and [src/glp_hyperparameter_optimization/data_utils.py](../src/glp_hyperparameter_optimization/data_utils.py#L681), with size-specific starts in config | Finish end-to-end wiring and decide how to enforce the declared large-model forecast cap. |
| `GLP-03` and `GLP-04` groundwork for real-time predictive-mean RMSE tuning | Helper layer already exists in [src/glp_hyperparameter_optimization/glp_model.py](../src/glp_hyperparameter_optimization/glp_model.py#L911) and [src/glp_hyperparameter_optimization/glp_model.py](../src/glp_hyperparameter_optimization/glp_model.py#L985) | Implement the missing exported updater functions around this helper layer instead of starting over. |
| `MF-02` final-metric-consistent RMSE objective | Largely implemented in [src/paper_hyperparameter_optimization/forecasting.py](../src/paper_hyperparameter_optimization/forecasting.py#L420), [src/paper_hyperparameter_optimization/forecasting.py](../src/paper_hyperparameter_optimization/forecasting.py#L549), and draw-summary helpers above them | Fix syntax/import drift, then validate end to end. Do not revert to upstream raw-level RMSE. |
| `MF-04` transformed-draw summaries | Implemented in [src/paper_hyperparameter_optimization/forecasting.py](../src/paper_hyperparameter_optimization/forecasting.py#L102) and [src/paper_hyperparameter_optimization/forecasting.py](../src/paper_hyperparameter_optimization/forecasting.py#L176) | Preserve this draw-based summary path. |
| `REPRO-01` provenance/reproducibility scaffolding | Partially implemented in [src/experiment_provenance.py](../src/experiment_provenance.py), `requirements.lock`, and deterministic RNG helpers | Wire it into live run metadata and seed plumbing; do not create a second provenance mechanism. |
| `OPS-02` Euler batch repair | Implemented in [scripts/run_everything_euler.sh](../scripts/run_everything_euler.sh) and validated by [tests/test_reporting_ops.py](../tests/test_reporting_ops.py) | Keep this script-derived `REPO_ROOT` / stage-manifest design. |
| [GLP_RMSE_REVIEW.md](../GLP_RMSE_REVIEW.md) recommendation to move from posterior-mode RMSE to predictive-mean RMSE | Partially/mostly implemented in GLP helper code and fully exposed in the GLP batch CLI through `--optimization-n-obj-draws` | Update docs and finish missing updater exports; do not re-argue the design as if it were absent. |

## Still open, regressed, or newly visible issues

### Highest-priority runtime blockers

1. Mixed-frequency forecasting must become importable again.
   - Fix the malformed block at [src/paper_hyperparameter_optimization/forecasting.py](../src/paper_hyperparameter_optimization/forecasting.py#L361).
   - Reconcile the missing `DEFAULT_RANDOM_SEED` contract between [src/paper_hyperparameter_optimization/forecasting.py](../src/paper_hyperparameter_optimization/forecasting.py#L26) and [src/paper_hyperparameter_optimization/config.py](../src/paper_hyperparameter_optimization/config.py).
2. GLP RMSE updater entrypoints must be restored.
   - The helper layer exists, but the exported functions expected by [src/glp_hyperparameter_optimization/forecasting.py](../src/glp_hyperparameter_optimization/forecasting.py#L191) and [src/glp_hyperparameter_optimization/forecasting.py](../src/glp_hyperparameter_optimization/forecasting.py#L208) are still absent.
3. [scripts/glp/compare_glp_forecasts.py](../scripts/glp/compare_glp_forecasts.py) should be deduplicated to one `_resolve_strategy_dir` and one `main`.

### High-priority consistency gaps after importability

4. Decide whether `GLP_MODEL_FORECAST_END` is policy-only or should be enforced in `run_glp_experiment` / `build_glp_estimation_matrix`.
5. Finish wiring mixed-frequency provenance into `run_recursive_experiment`, since [src/experiment_provenance.py](../src/experiment_provenance.py) is present but currently unused there.
6. Update stale tests that still expect fallback behavior the code intentionally removed.
7. Regenerate or clearly quarantine checked-in mixed-frequency outputs and comparison tables; they do not reflect the current strict reporting and local-objective code path.

### Medium-priority remaining scientific / operational items

8. Review whether mixed-frequency custom-window backfill and partial-quarter actual handling are still intentionally open (`MF-03` in the older audit).
9. Reconcile the stale documentation in [README_GLP_METHODOLOGY.md](../README_GLP_METHODOLOGY.md), [GLP_RMSE_REVIEW.md](../GLP_RMSE_REVIEW.md), and [README.md](../README.md) with the current predictive-mean, fail-closed, paired-reporting design.

## Unresolved pre-implementation questions

1. Should mixed-frequency reproducibility be driven by a real `DEFAULT_RANDOM_SEED` constant in config, or should forecasting stop importing that symbol and treat seeding as always explicit?
2. Should the declared GLP large-model cap in `GLP_MODEL_FORECAST_END` be enforced in the driver, or is the intended scientific design to drop the large model entirely after 2016Q1?
3. Are the strict fail-closed behaviors now in the compare and GLP data scripts the desired final semantics? The current code says yes; two tests still say no.
4. Should blank/stale checked-in mixed-frequency artifacts under [outputs/paper_hyperparameters](../outputs/paper_hyperparameters) and [outputs/comparison](../outputs/comparison) be removed, regenerated, or explicitly labeled historical?
5. Is `covbayesvar` meant to remain optional in local development, or should the validated environment definition include it so [tests/test_glp_model.py](../tests/test_glp_model.py) becomes part of the default lightweight suite?

## Bottom line

The current local worktree is materially ahead of the previously written planning documents. Shared strict comparison logic, fail-closed GLP data handling, GLP information-set validation, predictive-mean RMSE objective scaffolding, provenance helpers, and the repaired Euler launcher already exist and should be treated as the new base layer.

The next implementation stage should therefore be narrow:

1. restore importability;
2. finish wiring the already-present GLP and mixed-frequency helper layers;
3. update the stale tests to the stricter semantics now in code; and
4. only then rerun/regenerate outputs and documentation.

Any plan that starts by redesigning reporting, provenance, GLP fail-closed ingestion, or predictive-mean RMSE logic from scratch would be duplicating work that is already in this repository state.