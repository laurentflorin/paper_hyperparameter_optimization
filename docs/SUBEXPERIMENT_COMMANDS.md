# Subexperiment Run Commands

This file lists copy-paste Python commands to run each model-family subexperiment.

## 0) Environment

```bash
pip install -r requirements.txt
```

## 1) GLP (Giannone-Lenza-Primiceri)

GLP supports explicit model-size variants: `small`, `medium`, `large`.

### 1.1 Download GLP data by size

```bash
python scripts/glp/download_glp_data.py --model-size small
python scripts/glp/download_glp_data.py --model-size medium
python scripts/glp/download_glp_data.py --model-size large
```

Practical default: download `large` once, then run any model size from the same processed panel.

### 1.2 Run GLP scope-grid (forecast-loss, psi fixed) by size

```bash
python scripts/glp/run_glp_scope_grid.py \
  --output-root outputs/glp/scope_small \
  --panel-path data/processed/glp_realtime_panel.csv.gz \
  --model-size small \
  --start 2000-03-31 --end 2019-12-31 \
  --selection-scopes pooled,horizon,variable,variable_horizon \
  --target-horizons 1,2,4,8 \
  --selection-frequency once \
  --optimization-init-points 24 \
  --optimization-iterations 72 \
  --objective-posterior-draws 25 \
  --no-optimize-psi \
  --overwrite

python scripts/glp/run_glp_scope_grid.py \
  --output-root outputs/glp/scope_medium \
  --panel-path data/processed/glp_realtime_panel.csv.gz \
  --model-size medium \
  --start 2000-03-31 --end 2019-12-31 \
  --selection-scopes pooled,horizon,variable,variable_horizon \
  --target-horizons 1,2,4,8 \
  --selection-frequency once \
  --optimization-init-points 24 \
  --optimization-iterations 72 \
  --objective-posterior-draws 25 \
  --no-optimize-psi \
  --overwrite

python scripts/glp/run_glp_scope_grid.py \
  --output-root outputs/glp/scope_large \
  --panel-path data/processed/glp_realtime_panel.csv.gz \
  --model-size large \
  --start 2000-03-31 --end 2019-12-31 \
  --selection-scopes pooled,horizon,variable,variable_horizon \
  --target-horizons 1,2,4,8 \
  --selection-frequency once \
  --optimization-init-points 24 \
  --optimization-iterations 72 \
  --objective-posterior-draws 25 \
  --no-optimize-psi \
  --overwrite
```

### 1.3 GLP commands for different `--selection-frequency` values

Valid values are `once`, `per_origin`, `annual_quarterly`, or an integer `N`
(meaning re-select every `N` origins). Below uses `model-size medium`; switch
to `small` or `large` as needed.

```bash
# Re-select once at the first origin (most common paper setting)
python scripts/glp/run_glp_scope_grid.py \
  --output-root outputs/glp/scope_medium_once \
  --panel-path data/processed/glp_realtime_panel.csv.gz \
  --model-size medium \
  --start 2000-03-31 --end 2019-12-31 \
  --selection-scopes pooled,horizon,variable,variable_horizon \
  --target-horizons 1,2,4,8 \
  --selection-frequency once \
  --optimization-init-points 24 \
  --optimization-iterations 72 \
  --objective-posterior-draws 25 \
  --no-optimize-psi \
  --overwrite

# Re-select at every forecast origin (most expensive)
python scripts/glp/run_glp_scope_grid.py \
  --output-root outputs/glp/scope_medium_per_origin \
  --panel-path data/processed/glp_realtime_panel.csv.gz \
  --model-size medium \
  --start 2000-03-31 --end 2019-12-31 \
  --selection-scopes pooled,horizon,variable,variable_horizon \
  --target-horizons 1,2,4,8 \
  --selection-frequency per_origin \
  --optimization-init-points 24 \
  --optimization-iterations 72 \
  --objective-posterior-draws 25 \
  --no-optimize-psi \
  --overwrite

# Re-select annually in quarterly data (every 4 origins)
python scripts/glp/run_glp_scope_grid.py \
  --output-root outputs/glp/scope_medium_annual \
  --panel-path data/processed/glp_realtime_panel.csv.gz \
  --model-size medium \
  --start 2000-03-31 --end 2019-12-31 \
  --selection-scopes pooled,horizon,variable,variable_horizon \
  --target-horizons 1,2,4,8 \
  --selection-frequency annual_quarterly \
  --optimization-init-points 24 \
  --optimization-iterations 72 \
  --objective-posterior-draws 25 \
  --no-optimize-psi \
  --overwrite

# Re-select every N origins (example: N=8)
python scripts/glp/run_glp_scope_grid.py \
  --output-root outputs/glp/scope_medium_every8 \
  --panel-path data/processed/glp_realtime_panel.csv.gz \
  --model-size medium \
  --start 2000-03-31 --end 2019-12-31 \
  --selection-scopes pooled,horizon,variable,variable_horizon \
  --target-horizons 1,2,4,8 \
  --selection-frequency 8 \
  --optimization-init-points 24 \
  --optimization-iterations 72 \
  --objective-posterior-draws 25 \
  --no-optimize-psi \
  --overwrite
```

### 1.4 Are GLP Bayesian-optimization defaults good enough?

For the GLP scope-grid runner, the defaults are:

- `--optimization-init-points 24`
- `--optimization-iterations 72`
- `--objective-posterior-draws 25`

These are generally good baseline settings for the full experiment (they are
not the old low-budget legacy defaults). Keep them for your main runs unless
you are explicitly doing a quick smoke test.

If you need a stronger robustness run (more compute):

```bash
--optimization-init-points 40 --optimization-iterations 120 --objective-posterior-draws 50
```

If you only need a quick sanity check (faster, lower fidelity):

```bash
--optimization-init-points 8 --optimization-iterations 16 --objective-posterior-draws 10
```

## 2) Schorfheide-Song (MFVAR)

MFVAR does not expose `small|medium|large` model-size flags. The model block is controlled by `--forecast-variables`, and evaluation focus by `--target-variables`.

### 2.1 Download MFVAR/SS data

```bash
python scripts/download_data.py
```

### 2.2 Run MFVAR scope-grid

```bash
python scripts/mfvar/run_mfvar_scope_grid.py \
  --output-root outputs/mfvar/scope_full \
  --panel-path data/processed/realtime_panel.csv.gz \
  --forecast-variables GDP,INVFIX,GOV,UNR,HRS,CPI,IP,PCE,FF,TB,SP500 \
  --target-variables GDP,INVFIX,GOV,UNR,HRS,CPI,IP,PCE,FF,TB,SP500 \
  --target-horizons 1,2,4,8 \
  --selection-scopes pooled,horizon,variable,variable_horizon \
  --selection-frequency once \
  --overwrite
```

## 3) Regularized VAR (Ridge)

Regularized VAR also does not expose `small|medium|large` flags. Use `--target-variables` to define the variable set.

### 3.1 Download/prepare panel data

```bash
python scripts/download_data.py
```

### 3.2 Run ridge scope-grid

```bash
python scripts/regularized_var/run_ridge_scope_grid.py \
  --output-root outputs/regularized_var/scope_full \
  --panel-path data/processed/realtime_panel.csv.gz \
  --target-variables GDP,INVFIX,GOV,UNR,HRS,CPI,IP,PCE,FF,TB,SP500 \
  --target-horizons 1,2,4,8 \
  --selection-scopes pooled,horizon,variable,variable_horizon \
  --forecast-method iterated \
  --selection-frequency once \
  --benchmarks no_change,var_aic \
  --overwrite
```

## 4) Optional: run the configured paper subexperiments by family

If you want the exact family-level matrix configured in `configs/paper_experiment.json`:

```bash
python scripts/run_scope_study.py --config configs/paper_experiment.json --filter-family glp --resume --output-root outputs/scope_study
python scripts/run_scope_study.py --config configs/paper_experiment.json --filter-family mfvar --resume --output-root outputs/scope_study
python scripts/run_scope_study.py --config configs/paper_experiment.json --filter-family ridge --resume --output-root outputs/scope_study
```
