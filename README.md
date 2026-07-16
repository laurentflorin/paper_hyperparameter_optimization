# paper_hyperparameter_optimization

This repository contains a reproducible Python workflow for:

1. downloading the Schorfheide-Song real-time macro dataset from ALFRED/FRED,
2. recreating the mixed-frequency BVAR forecasts with the paper hyperparameters,
3. re-optimizing the hyperparameters with `update_hyperparameters_mango`,
4. re-optimizing the hyperparameters with `update_hyperparameters_mango_rmse`,
5. re-optimizing the hyperparameters with `update_hyperparameters_mango_rmse_random`,
6. comparing the out-of-sample forecasts from the analysis runs with paper-ready tables and figures.

The target paper is:

- Schorfheide, F. and Song, D. (2015), *Real-Time Forecasting With a Mixed-Frequency VAR*, *Journal of Business & Economic Statistics*, 33(3), 366-380.

The workflow uses the `MBFVAR` package from [laurentflorin/MBFVAR](https://github.com/laurentflorin/MBFVAR.git).

## Data Setup

The download script reproduces the paper's 11-variable ALFRED mapping:

- Quarterly: `GDPC1`, `FPIC1`, `GCEC1`
- Monthly: `UNRATE`, `AWHI`, `CPIAUCSL`, `INDPRO`, `PCEC96`, `FEDFUNDS`, `GS10`, `SP500`

It pulls:

- one real-time vintage for each paper forecast origin from `1997-07-31` through `2010-01-31`,
- one evaluation vintage for `2012-01-31`,
- latest FRED series for backcasting incomplete `PCEC96` and `FPIC1` histories in the same spirit as the paper appendix.

If the ALFRED `SP500` vintage endpoint is unavailable, the downloader falls back to Stooq monthly S&P 500 closes and truncates that non-revised history at each vintage date.

## Environment

Create a Python environment with the dependencies in [requirements.txt](/workspaces/paper_hyperparameter_optimization/requirements.txt).

`MBFVAR` compiles native extensions, so a working C/C++ build toolchain is required.

## Scripts

### 1. Download data

```bash
python scripts/download_data.py
```

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
  --optimization-variables GDP
```

By default this now runs four horizon-specific RMSE optimizations for `1`, `2`, `4`, and `8` quarters ahead, using `n_eval=3` rolling evaluation origins as a compromise between runtime and objective stability. Each run is written to its own subdirectory under `outputs/mango_rmse` such as `h1q` and `h8q`.

This uses Mango to minimize an internal rolling-holdout RMSE criterion through `MBFVAR.update_hyperparameters_mango_rmse`, passing both the target evaluation horizon and `n_eval`.

To run a single target horizon instead of the default batch:

```bash
python scripts/run_mango_rmse.py \
  --output-dir outputs/mango_rmse_h4 \
  --optimization-eval-horizons-quarters 4 \
  --optimization-nsim 1000 \
  --optimization-init-points 5 \
  --optimization-iterations 15 \
  --optimization-n-eval 3 \
  --optimization-variables GDP
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
  --optimization-variables GDP
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

## Parallelization

The recursive forecast scripts support process-level parallelization across forecast origins:

```bash
python scripts/run_mango_mdd.py \
  --output-dir outputs/mango_mdd \
  --n-workers 4 \
  --optimization-njobs 2
```

Use this carefully because each worker runs a full `MBFVAR` estimation and memory use can become substantial.

When the scripts run inside a Slurm allocation, `--n-workers` now defaults to the full Slurm task allocation and `--optimization-njobs` defaults to the remaining per-origin share of that allocation. This lets a 48-core Euler job use all 48 cores without multiplying into `48 x 48` nested oversubscription. You can still override either flag manually.

## Euler / Slurm

Use [scripts/run_everything_euler.slurm](/home/runner/work/paper_hyperparameter_optimization/paper_hyperparameter_optimization/scripts/run_everything_euler.slurm) to run the full workflow on Euler:

```bash
sbatch scripts/run_everything_euler.slurm
```

The batch script:

- requests `48` single-core Slurm tasks on one node,
- loads the requested Euler module stack,
- pins BLAS/OpenMP thread pools to `1` per Python worker to avoid oversubscription,
- runs data download, paper hyperparameters, Mango MDD, Mango RMSE, Mango RMSE random, and the final comparison report in sequence,
- writes outputs under `outputs/euler` by default.

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
