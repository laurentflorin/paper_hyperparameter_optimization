# paper_hyperparameter_optimization

Reproducible Python experiments for comparing two strategies for selecting
the shrinkage hyperparameters of Bayesian VARs on out-of-sample forecast
accuracy:

1. **Native selection** — maximize the marginal data density (MDD / marginal
   likelihood) as in Giannone, Lenza, and Primiceri (2015).
2. **Forecast-loss selection** — minimize pseudo-out-of-sample RMSE on inner
   validation splits.

> **Research question:** does selecting hyperparameters by forecast loss
> produce better real-time macroeconomic forecasts than the principled
> Bayesian (MDD) rule?

### Target papers

| Paper | Abbr |
|---|---|
| Giannone, Lenza, Primiceri (2015), *Prior Selection for VARs*, ReStat 97(2) | GLP |
| Schorfheide, Song (2015), *Real-Time Forecasting With a Mixed-Frequency VAR*, JBES 33(3) | SS |

---

## Documentation

| Document | Contents |
|---|---|
| [docs/EXPERIMENT_DESIGN.md](docs/EXPERIMENT_DESIGN.md) | Research question, selection scopes, inner/outer evaluation, vintage policies, selection schedules, loss scaling, forecast methods, known limitations |
| [docs/OUTPUT_SCHEMA.md](docs/OUTPUT_SCHEMA.md) | Full column-level schema for all output files |
| [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | Configuration hashes, resume semantics, test tiers, CI setup |
| [docs/PILOT_VALIDATION.md](docs/PILOT_VALIDATION.md) | Exact pilot commands, timing, output counts, validation results, dependency status |
| [README_GLP_METHODOLOGY.md](README_GLP_METHODOLOGY.md) | Statistical derivation of the GLP model and hyperparameter strategies |
| [docs/CURRENT_STATE_AUDIT.md](docs/CURRENT_STATE_AUDIT.md) | Repository audit (2026-08-18): known blockers, output quality, code state |
| [IMPLEMENTATION_AUDIT.md](IMPLEMENTATION_AUDIT.md) | Implementation audit with severity-tagged findings; see status notes for resolved items |

---

## Environment

```bash
# Full environment (required for GLP and MFVAR workflows)
pip install -r requirements.txt

# Minimal CI environment (ridge + shared infrastructure only; no optional deps)
pip install -r requirements-dev.txt
```

`MBFVAR` requires a C/C++ build toolchain and is pinned to commit
`5b06f93272cd6ebf370fbf2aac3b3573c7830493`. `covbayesvar` is required for GLP.

---

## Quick start: synthetic ridge integration (no downloads required)

```bash
# 1. Generate synthetic panel
python - <<'EOF'
import numpy as np, pandas as pd
from pathlib import Path
rng = np.random.default_rng(42)
T = 120
A = np.array([[0.6,0.1,0.0],[-0.1,0.5,0.1],[0.0,0.1,0.4]])
c = np.array([0.2,-0.1,0.1])
y = np.zeros((T,3))
for t in range(1,T):
    y[t] = c + A@y[t-1] + rng.normal(scale=0.15,size=3)
Path("/tmp/pilot_a").mkdir(exist_ok=True)
pd.DataFrame(y, columns=["gdp","inv","cons"]).to_csv("/tmp/pilot_a/panel.csv", index=False)
EOF

# 2. Iterated ridge, selected once, all four scopes
python scripts/regularized_var/run_ridge_scope_grid.py \
  --output-root /tmp/pilot_a/iterated \
  --panel-path  /tmp/pilot_a/panel.csv \
  --target-variables gdp,inv,cons --target-horizons 1,2,4,8 \
  --selection-scopes pooled,horizon,variable,variable_horizon \
  --forecast-method iterated \
  --grid-lambdas 0.001,0.01,0.1,1.0 --grid-lag-orders 1,2 \
  --grid-alphas 0.0 --grid-kappas 1.0 \
  --outer-n-origins 8 --inner-n-origins 4 --min-train-length 30 \
  --selection-frequency once --benchmarks no_change,var_aic --overwrite
```

See [docs/PILOT_VALIDATION.md](docs/PILOT_VALIDATION.md) for the full
command set, comparison manifest, validation script, and all expected output
row counts.

---

## Scope-grid runners (primary experiment entrypoints)

Each runner supports all four selection scopes (`pooled`, `horizon`,
`variable`, `variable_horizon`), configurable schedules, loss scaling, and
canonical run-state markers with configuration-hash-based resume logic.

### Ridge VAR (no optional dependencies)

```bash
python scripts/regularized_var/run_ridge_scope_grid.py \
  --output-root outputs/regularized/<study> \
  --panel-path  <panel.csv> \
  --target-variables <v1,v2,...> \
  --target-horizons  1,2,4,8 \
  --selection-scopes pooled,horizon,variable,variable_horizon \
  --forecast-method  iterated \    # or: direct
  --selection-frequency once \     # or: per_origin, annual_quarterly, N
  --benchmarks no_change,var_aic \
  --overwrite
```

### GLP (requires `covbayesvar`)

```bash
python scripts/glp/run_glp_scope_grid.py \
  --output-root outputs/glp/<study> \
  --panel-path  data/processed/glp_realtime_panel.csv.gz \
  --model-size  small \            # or: medium, large
  --start 2000-03-31 --end 2019-12-31 \
  --selection-scopes pooled,horizon,variable,variable_horizon \
  --no-optimize-psi \              # reduced search (recommended)
  --selection-frequency once \
  --overwrite
```

### MFVAR / Schorfheide-Song (requires `MBFVAR`)

```bash
python scripts/mfvar/run_mfvar_scope_grid.py \
  --output-root outputs/mfvar/<study> \
  --panel-path  data/processed/realtime_panel.csv.gz \
  --forecast-variables GDP,INVFIX,GOV,UNR,HRS,CPI,IP,PCE,FF,TB,SP500 \
  --target-variables   GDP \
  --target-horizons    1,4 \
  --selection-scopes   pooled,variable \
  --selection-frequency once \
  --overwrite
```

`--forecast-variables` is the full quarterly block required by the model.
`--target-variables` restricts the inner RMSE loss without changing the model.

### Comparison

```bash
python scripts/compare_scope_study.py \
  --manifest <manifest.json> \
  --output-dir outputs/<study>/comparison
```

The manifest is a JSON file listing each panel directory with its model
label and family tag. See [docs/OUTPUT_SCHEMA.md](docs/OUTPUT_SCHEMA.md)
for the manifest format and comparison output schemas.

---

## Tests

```bash
# Minimal CI (no optional deps)
pytest --skip-optional -q -rs

# Full suite
pytest -q

# Unit tests only
pytest -m unit -q
```

CI workflow: [.github/workflows/ci.yml](.github/workflows/ci.yml).
Test tiers: [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md#9-test-tiers).

---

## Legacy scripts

The sections below document the original single-origin scripts that predate
the scope-grid architecture. They remain operational but do not write
configuration hashes or run-state files.

### Data

## Legacy scripts

The sections below document the original single-origin scripts. They remain
operational but do not write configuration hashes or run-state files.

### Schorfheide-Song data setup

The download script reproduces the paper's 11-variable ALFRED mapping:

- Quarterly: `GDPC1`, `FPIC1`, `GCEC1`
- Monthly: `UNRATE`, `AWHI`, `CPIAUCSL`, `INDPRO`, `PCEC96`, `FEDFUNDS`, `GS10`, `SP500`

It pulls:

- one real-time vintage for each paper forecast origin from `1997-07-31` through `2010-01-31`,
- one evaluation vintage for `2012-01-31`,
- latest FRED series for backcasting incomplete `PCEC96` and `FPIC1` histories in the same spirit as the paper appendix.

If the ALFRED `SP500` vintage endpoint is unavailable, the downloader falls back to Stooq monthly S&P 500 closes and truncates that non-revised history at each vintage date.

### Environment

Create a Python environment with the dependencies in [requirements.txt](requirements.txt).

`MBFVAR` compiles native extensions, so a working C/C++ build toolchain is required.

### Scripts

All script path arguments are resolved relative to the repository root.

### 1. Download data

```bash
python scripts/download_data.py
```

If `data/processed/realtime_panel.csv.gz`, `data/processed/latest_panel.csv.gz`, and `data/processed/download_metadata.json` already match the requested date range and actual vintage, the script exits without downloading again unless `--force` is provided.

This writes:

- `data/processed/realtime_panel.csv.gz`
- `data/processed/latest_panel.csv.gz`
- `data/processed/download_metadata.json`

### 2. Recreate paper-hyperparameter forecasts

```bash
python scripts/run_paper_hyperparameters.py \
  --output-dir outputs/paper_hyperparameters
```

Defaults match the paper's MF-VAR setup:

- monthly state VAR with `6` lags,
- `20,000` posterior draws,
- burn-in share `0.5`,
- paper MF-VAR hyperparameters `[0.09, 4.30, 1.0, 2.70, 4.30]`,
- quarterly aggregation by mean,
- recursive forecast origins from `1997-07` to `2010-01`,
- forecast horizon `24` months / `8` quarters.

### 3. Re-optimize with `update_hyperparameters_mango`

```bash
python scripts/run_mango_mdd.py \
  --output-dir outputs/mango_mdd \
  --optimization-nsim 1000 \
  --optimization-init-points 5 \
  --optimization-iterations 15 \
  --optimization-variables GDP
```

This uses Mango to maximize the marginal data density with one quarterly variable of interest by default: `GDP`.

### 4. Re-optimize with `update_hyperparameters_mango_rmse`

```bash
python scripts/run_mango_rmse.py \
  --output-dir outputs/mango_rmse \
  --optimization-nsim 1000 \
  --optimization-init-points 5 \
  --optimization-iterations 15 \
  --optimization-variables GDP,INVFIX,GOV
```

By default this now runs four horizon-specific RMSE optimizations for `1`, `2`, `4`, and `8` quarters ahead, using `n_eval=3` rolling evaluation origins as a compromise between runtime and objective stability. Each run is written to its own subdirectory under `outputs/mango_rmse` such as `h1q` and `h8q`.

This uses Mango to minimize an internal rolling-holdout RMSE criterion through `MBFVAR.update_hyperparameters_mango_rmse`, passing both the target evaluation horizon and `n_eval`.

The RMSE optimizers must currently use the full quarterly block `GDP,INVFIX,GOV`. Smaller quarterly subsets such as `GDP` alone trigger an upstream `MBFVAR` forecast dimension mismatch and degenerate to the fixed `1e10` penalty that shows up as the optimizer's best score.

To run a single target horizon instead of the default batch:

```bash
python scripts/run_mango_rmse.py \
  --output-dir outputs/mango_rmse_h4 \
  --optimization-eval-horizons-quarters 4 \
  --optimization-nsim 1000 \
  --optimization-init-points 5 \
  --optimization-iterations 15 \
  --optimization-n-eval 3 \
  --optimization-variables GDP,INVFIX,GOV
```

### 5. Re-optimize with `update_hyperparameters_mango_rmse_random`

```bash
python scripts/run_mango_rmse_random.py \
  --output-dir outputs/mango_rmse_random \
  --optimization-nsim 1000 \
  --optimization-init-points 5 \
  --optimization-iterations 15 \
  --optimization-n-eval 3 \
  --optimization-random-seed 123 \
  --optimization-variables GDP,INVFIX,GOV
```

This mirrors the horizon-specific RMSE workflow but uses `MBFVAR.update_hyperparameters_mango_rmse_random`, which evaluates each hyperparameter proposal on a fixed random sample of valid forecast origins instead of the last `n_eval` rolling origins.

By default this also runs four horizon-specific optimizations for `1`, `2`, `4`, and `8` quarters ahead, writing one subdirectory per target horizon under `outputs/mango_rmse_random`.

Optional controls:

- `--optimization-min-t` enforces a minimum lowest-frequency in-sample size for candidate evaluation origins.
- `--optimization-random-seed` makes the sampled evaluation origins reproducible.

### 6. Compare the forecast sets

```bash
python scripts/compare_forecasts.py \
  --paper-dir outputs/paper_hyperparameters \
  --mango-mdd-dir outputs/mango_mdd \
  --mango-rmse-dir outputs/mango_rmse \
  --mango-rmse-random-dir outputs/mango_rmse_random \
  --output-dir outputs/comparison
```

This creates:

- RMSE tables for all variables and headline variables,
- relative-RMSE tables versus the paper hyperparameters,
- hyperparameter summary tables for the optimized models,
- paper-style PNG figures for relative RMSE by group and hyperparameter paths.

## GLP Workflow

This repository also contains a separate quarterly BVAR workflow for Giannone, Lenza and Primiceri (2015) under `src/glp_hyperparameter_optimization` and `scripts/glp`. Those commands are independent of the Schorfheide-Song MF-VAR workflow above and write to `outputs/glp/...` by convention.

### Choosing `small`, `medium`, or `large`

All GLP scripts expose `--model-size small|medium|large`, but it affects two different layers:

- `scripts/glp/download_glp_data.py` chooses which series are downloaded into the processed GLP panel and defaults to `large`.
- The recursive forecast scripts choose which nested model is estimated from that panel and default to `medium`.

The model sizes are nested:

- `small`: `GDP, DEFL, FFR`
- `medium`: `small` plus `CONS, INV, HOURS, WAGE`
- `large`: `medium` plus `EMP, UNR, AHE, CPI, PPI, COMM, M1, M2, MBASE, TOTRES, NBRES, SP500, TB10, REER`

Practical rule:

- Download `large` once if you want to switch between model sizes later.
- Download only `small` or `medium` only if you know you will stay at that size, because larger runs need the additional series to build a complete quarterly estimation window.

### 1. Download GLP data

Download the full GLP panel once and then reuse it for any of the three model sizes:

```bash
python scripts/glp/download_glp_data.py \
  --model-size large
```

If you only want one smaller universe, change `--model-size` accordingly:

```bash
python scripts/glp/download_glp_data.py \
  --model-size small
```

The downloader also supports:

- `--start` / `--end` to change the recursive forecast-origin vintages that are cached,
- `--actual-vintage` to change the fixed evaluation vintage,
- `--output-panel`, `--output-latest`, and `--metadata-path` to redirect the processed outputs,
- `--force` to redownload even when the processed files already match the requested setup.

### 2. Common GLP run parameters

All recursive GLP forecast scripts share these controls:

- `--output-dir` is required and determines where `forecast_panel.csv`, `selected_hyperparameters.csv`, `failed_origins.csv`, and `run_metadata.json` are written.
- `--model-size` selects the estimated model size for that run.
- `--panel-path` points to the processed GLP real-time panel and defaults to `data/processed/glp_realtime_panel.csv.gz`.
- `--start` / `--end` change the recursive forecast window. The defaults are `2000-03-31` through `2019-12-31`.
- `--actual-vintage` changes the scoring vintage. The default is `2023-01-01`.
- `--lags` changes the BVAR lag order. The default is `5`.
- `--mcmc-draws`, `--mcmc-discard`, and `--mcmc-const` change the predictive-density simulation settings.
- `--seed-base` sets a reproducible base RNG seed that is offset per forecast origin.
- `--n-workers` controls process-level parallelism across forecast origins.

The Mango-based scripts also expose:

- `--optimization-init-points`, `--optimization-iterations`, and `--optimization-njobs` for the Bayesian-optimization budget.

The RMSE-based GLP scripts additionally expose:

- `--variables` for the objective variables, e.g. `GDP,DEFL`,
- `--optimization-eval-horizons-quarters` for the target horizon batch, e.g. `1,2,4,8` or just `4`,
- `--optimization-n-eval` for the number of evaluation origins inside the RMSE objective,
- `--optimization-min-t` and `--optimization-random-seed` for the random-origin RMSE variant,
- `--per-origin-selection` to re-select RMSE hyperparameters at every origin instead of once on the first origin.

### Batch-run the GLP forecast scripts

If you want one command that runs the four forecast scripts (`paper`, `mango_mdd`,
`mango_rmse`, `mango_rmse_random`) with one shared flag surface, use:

```bash
python scripts/glp/run_glp_all.py \
  --stages paper,mango_mdd,mango_rmse,mango_rmse_random,compare \
  --output-root outputs/glp/all_medium \
  --model-size medium \
  --variables GDP,DEFL \
  --optimization-eval-horizons-quarters 1,2,4,8 \
  --optimization-n-eval 3 \
  --optimization-random-seed 123 \
  --optimization-init-points 5 \
  --optimization-iterations 15 \
  --optimization-njobs 4 \
  --n-workers 4
```

Notes:

- `--stages` defaults to `paper,mango_mdd,mango_rmse,mango_rmse_random`; add `compare` if you also want the comparison report.
- `--output-root` becomes the base directory; the wrapper writes `paper/`, `mango_mdd/`, `mango_rmse/`, `mango_rmse_random/`, and optionally `comparison/` under it unless you override the per-stage directories with `--paper-dir`, `--mango-mdd-dir`, `--mango-rmse-dir`, `--mango-rmse-random-dir`, or `--comparison-dir`.
- The wrapper also exposes the shared model-prior switches `--hyperpriors`, `--sur`, `--noc`, `--mnpsi`, `--mnalpha`, and `--vc` in case you want to override the defaults programmatically used by the individual scripts.

### 3. Run the GLP paper strategy

```bash
python scripts/glp/run_glp_paper.py \
  --output-dir outputs/glp/paper_medium \
  --model-size medium
```

This is the hierarchical GLP predictive density: hyperparameters are selected by marginal likelihood and integrated out with a random-walk Metropolis step.

### 4. Run the GLP Mango MDD strategy

```bash
python scripts/glp/run_glp_mango.py \
  --output-dir outputs/glp/mango_mdd_medium \
  --model-size medium \
  --optimization-init-points 5 \
  --optimization-iterations 15
```

Despite the shorter filename, `run_glp_mango.py` is the GLP MDD / posterior-optimization workflow.

### 5. Run the GLP Mango RMSE strategy

```bash
python scripts/glp/run_glp_mango_rmse.py \
  --output-dir outputs/glp/mango_rmse_medium \
  --model-size medium \
  --variables GDP,DEFL \
  --optimization-eval-horizons-quarters 1,2,4,8 \
  --optimization-n-eval 3
```

By default this writes one subdirectory per target horizon such as `h1q`, `h2q`, `h4q`, and `h8q` under the chosen output directory. To optimize only one horizon, pass a single value such as `--optimization-eval-horizons-quarters 4`.

### 6. Run the GLP Mango RMSE-random strategy

```bash
python scripts/glp/run_glp_mango_rmse_random.py \
  --output-dir outputs/glp/mango_rmse_random_medium \
  --model-size medium \
  --variables GDP,DEFL \
  --optimization-eval-horizons-quarters 1,2,4,8 \
  --optimization-n-eval 3 \
  --optimization-random-seed 123
```

This mirrors the previous command but samples the RMSE evaluation origins at random from the valid pool.

### 7. Compare the GLP forecast sets

```bash
python scripts/glp/compare_glp_forecasts.py \
  --paper-dir outputs/glp/paper_medium \
  --mango-mdd-dir outputs/glp/mango_mdd_medium \
  --mango-rmse-dir outputs/glp/mango_rmse_medium \
  --mango-rmse-random-dir outputs/glp/mango_rmse_random_medium \
  --output-dir outputs/glp/comparison_medium
```

If you instead write the runs to the unsuffixed defaults `outputs/glp/paper`, `outputs/glp/mango_mdd`, `outputs/glp/mango_rmse`, and `outputs/glp/mango_rmse_random`, you can omit the directory flags and keep only `--output-dir`.

If you keep size-suffixed directories such as `outputs/glp/paper_small` and
`outputs/glp/mango_rmse_small`, the compare script now auto-discovers them from
their `run_metadata.json` files under `outputs/glp`. If multiple model sizes are
present, pass `--model-size small|medium|large` to disambiguate, or keep using
the explicit `--paper-dir`, `--mango-mdd-dir`, `--mango-rmse-dir`, and
`--mango-rmse-random-dir` flags.

## Parallelization

The recursive forecast scripts in both workflows support process-level parallelization across forecast origins:

```bash
python scripts/run_mango_mdd.py \
  --output-dir outputs/mango_mdd \
  --n-workers 4 \
  --optimization-njobs 2
```

Use this carefully because each worker runs a full `MBFVAR` estimation and memory use can become substantial.

When the scripts run inside a Slurm allocation, `--n-workers` now defaults to the full Slurm task allocation and `--optimization-njobs` defaults to the remaining per-origin share of that allocation. This lets a 48-core Euler job use all 48 cores without multiplying into `48 x 48` nested oversubscription. You can still override either flag manually.

## Euler / Slurm

Use [scripts/run_everything_euler.sh](scripts/run_everything_euler.sh) to run the full Schorfheide-Song workflow on Euler:

```bash
sbatch scripts/run_everything_euler.sh
```

The batch script:

- requests `48` single-core Slurm tasks on one node,
- loads the requested Euler module stack,
- pins BLAS/OpenMP thread pools to `1` per Python worker to avoid oversubscription,
- runs data download, paper hyperparameters, Mango MDD, Mango RMSE, Mango RMSE random, and the final comparison report in sequence,
- writes outputs under `outputs/euler` by default.

There is no dedicated GLP Euler wrapper yet, so run the `scripts/glp/*.py` commands you need inside your own batch script and pass `--model-size`, output paths, and optimizer flags explicitly.

To write results somewhere else, override `OUTPUT_ROOT` at submission time:

```bash
OUTPUT_ROOT=/cluster/scratch/$USER/paper_hpo sbatch scripts/run_everything_euler.slurm
```

## Output Files

Each forecast script writes:

- `forecast_panel.csv`: one row per model/origin/variable/horizon with point forecasts, intervals, actuals, and errors,
- `selected_hyperparameters.csv`: one row per forecast origin with the chosen hyperparameters,
- `failed_origins.csv`: any origins that failed,
- `run_metadata.json`: run configuration for reproducibility.

## Notes

- The repository implements the paper's MF-VAR setup, not the paper's quarterly-frequency VAR or MIDAS benchmark models.
- The comparison script evaluates quarterly averages. For `GDP`, `INVFIX`, `GOV`, `HRS`, `CPI`, `IP`, `PCE`, and `SP500`, forecast errors are computed on quarter-on-quarter log growth rates in percent. For `UNR`, `FF`, and `TB`, they are computed on quarterly-average levels.
- `MBFVAR` currently uses `numpy.random.default_rng()` internally without a user-exposed seed path in the package API, so exact bit-for-bit replication across runs may require patching the upstream package.
