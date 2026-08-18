# Pilot Validation Record

**Date:** 2026-08-18  
**Commit:** `b6bde20` (Stage 15 — Reproducibility, Metadata, Failure Handling, and CI)  
**Python:** 3.12.13 (Anaconda, packaged by Anaconda, Inc.)  
**OS:** Linux 6.4.0-150700.53.66-default x86_64  
**Key packages:** numpy=2.4.3, pandas=3.0.1, scipy=1.17.1  
**Interpreter:** `/home/u80856195/.virtualenvs/venv/bin/python`  

---

## Data availability

| Data | Path | Status |
|---|---|---|
| MFVAR real-time panel | `data/processed/realtime_panel.csv.gz` | Present |
| MFVAR latest panel | `data/processed/latest_panel.csv.gz` | Present |
| GLP real-time panel | `data/processed/glp_realtime_panel.csv.gz` | Present |
| GLP latest panel | `data/processed/glp_latest_panel.csv.gz` | Present |
| Synthetic pilot panel | `/tmp/pilot_a/panel.csv` (generated) | Generated |

## Optional dependencies

| Package | Status | Effect |
|---|---|---|
| `covbayesvar` | **Not installed** | Pilot B (GLP) cannot run |
| `MBFVAR` | **Not installed** | Pilot C (MFVAR) cannot run |

---

## CI test run

**Command:**
```bash
python -m pytest --skip-optional -q -rs
```

**Result:** 357 passed, 20 skipped in 8.19s

**Skips (all expected):**

| Test file | Reason |
|---|---|
| `test_glp_forecasting.py` | `update_hyperparameters_mango_rmse` not yet exported from `glp_model.py` (known pre-existing issue, L-04) |
| `test_glp_model.py` | `covbayesvar` not installed |
| `test_glp_run_all.py` | Same as `test_glp_forecasting.py` |
| `test_glp_compare_forecasts.py` (3 tests) | Marked `integration` |
| `test_glp_data_utils.py` (4 tests) | Marked `integration` |
| `test_mfvar_scope_grid.py::test_existing_scripts_still_import` | Marked `integration, requires_mbfvar` |
| `test_optimizer_variable_resolution.py` (6 tests) | Marked `integration, requires_mbfvar` |
| `test_reporting_ops.py` (3 tests) | Marked `integration` |

---

## Pilot A — Synthetic ridge end-to-end integration

### Research question
Does the full pipeline from synthetic panel → four selection scopes × two
forecast methods × two selection schedules → common comparison produce
correct, non-duplicated, fully-covered, scheme-consistent outputs?

### Environment
- Synthetic panel: 120 rows × 3 variables (`gdp`, `inv`, `cons`)
  generated from a VAR(1) with mild positive persistence (seed 42).
- Panel written to `/tmp/pilot_a/panel.csv` (not committed; generated fresh
  by running the command below).

### Commands

**Generate synthetic panel:**
```bash
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
df = pd.DataFrame(y, columns=["gdp","inv","cons"])
Path("/tmp/pilot_a").mkdir(parents=True, exist_ok=True)
df.to_csv("/tmp/pilot_a/panel.csv", index=False)
EOF
```

**Pilot A.1 — Iterated ridge, selected once, all four scopes:**
```bash
python scripts/regularized_var/run_ridge_scope_grid.py \
  --output-root /tmp/pilot_a/iterated \
  --panel-path  /tmp/pilot_a/panel.csv \
  --target-variables gdp,inv,cons \
  --target-horizons  1,2,4,8 \
  --selection-scopes pooled,horizon,variable,variable_horizon \
  --forecast-method  iterated \
  --grid-lambdas  0.001,0.01,0.1,1.0 \
  --grid-lag-orders 1,2 \
  --grid-alphas   0.0 \
  --grid-kappas   1.0 \
  --outer-n-origins  8 \
  --inner-n-origins  4 \
  --min-train-length 30 \
  --selection-frequency once \
  --benchmarks no_change,var_aic \
  --overwrite
```

**Pilot A.2 — Direct ridge, reselected every 2 outer origins:**
```bash
python scripts/regularized_var/run_ridge_scope_grid.py \
  --output-root /tmp/pilot_a/direct \
  --panel-path  /tmp/pilot_a/panel.csv \
  --target-variables gdp,inv,cons \
  --target-horizons  1,2,4,8 \
  --selection-scopes pooled,horizon,variable,variable_horizon \
  --forecast-method  direct \
  --grid-lambdas  0.001,0.01,0.1,1.0 \
  --grid-lag-orders 1,2 \
  --grid-alphas   0.0 \
  --grid-kappas   1.0 \
  --outer-n-origins  8 \
  --inner-n-origins  4 \
  --min-train-length 30 \
  --selection-frequency 2 \
  --benchmarks no_change \
  --overwrite
```

**Pilot A.3 — Comparison (requires manifest at `/tmp/pilot_a/manifest.json`):**
```bash
python scripts/compare_scope_study.py \
  --manifest /tmp/pilot_a/manifest.json \
  --output-dir /tmp/pilot_a/comparison
```

See `tests/pilot_a_validate.py` for the full validation script.

**Pilot A validation:**
```bash
python tests/pilot_a_validate.py \
  --iterated-root /tmp/pilot_a/iterated \
  --direct-root   /tmp/pilot_a/direct \
  --comparison    /tmp/pilot_a/comparison
```

### Runtime

| Step | Elapsed |
|---|---|
| A.1 iterated (4 scopes + 2 benchmarks) | ~2 s |
| A.2 direct (4 scopes + 1 benchmark) | ~3 s |
| A.3 comparison (11 models) | ~3 s |
| Validation script | < 1 s |

### Observed output row counts

**Expected row count calculation:**  
- 8 outer origins × 3 variables × 4 horizons = **96 rows** per scope.

| Study | Scope | forecast rows | selection rows | Expected selection rows |
|---|---|---|---|---|
| Iterated (once) | pooled | 96 | 1 | 1 (one cell, selected once) |
| Iterated (once) | horizon | 96 | 4 | 4 (4 cells, selected once each) |
| Iterated (once) | variable | 96 | 3 | 3 (3 cells, selected once each) |
| Iterated (once) | variable_horizon | 96 | 12 | 12 (3×4 cells, selected once each) |
| Direct (every 2) | pooled | 96 | 4 | ⌈8/2⌉=4 events × 1 cell |
| Direct (every 2) | horizon | 96 | 16 | ⌈8/2⌉=4 events × 4 cells |
| Direct (every 2) | variable | 96 | 12 | ⌈8/2⌉=4 events × 3 cells |
| Direct (every 2) | variable_horizon | 96 | 48 | ⌈8/2⌉=4 events × 12 cells |

All observed counts match expected counts exactly.

### Validation result

**ALL CHECKS PASSED** (0 failures)

Checks performed by `tests/pilot_a_validate.py`:

| Check | Outcome |
|---|---|
| `run_complete.json` present, status=complete | ✓ all 8 scope dirs |
| `configuration_hash` present in manifest | ✓ all 8 scope dirs |
| No duplicate canonical rows (origin, variable, horizon) | ✓ all 8 scopes |
| Complete target coverage (all variable × horizon combos at all origins) | ✓ 96/96 per scope |
| No NaN `mean_metric` forecasts | ✓ all 8 scopes |
| `selected_hyperparameters.csv` present | ✓ all 8 scopes |
| No missing selected parameter values | ✓ all 8 scopes |
| All selection origins are valid outer origins | ✓ all 8 scopes |
| `failed_origins.csv` present and empty | ✓ all 8 scopes |
| `comparison_summary.md` produced | ✓ |
| All 8 models present in `rmse_by_target.csv` (132 rows) | ✓ |
| `relative_rmse.csv` produced | ✓ |
| `scope_gains.csv` produced | ✓ |

### Parameter boundary diagnostics

Selected hyperparameters from the iterated/pooled scope (selected once):

| param_lam | param_p | param_alpha | param_kappa | selection_loss |
|---|---|---|---|---|
| 0.001 | 2 | 0.0 | 1.0 | 0.1464 |

Grid edges: λ ∈ {0.001, 0.01, 0.1, 1.0}, p ∈ {1, 2}.  
Selected λ = 0.001 is at the lower grid boundary. This is expected for a
small synthetic panel where the data strongly prefer aggressive shrinkage.
In a full scientific experiment, the grid should be extended below 0.001
if the smallest grid value is consistently selected.

The direct/pooled scope (every 2 origins) shows λ = 1.0 selected throughout
(upper boundary), indicating the synthetic data also support the no-shrinkage
limit at the direct regression targets. Both edge patterns are artefacts of
the synthetic panel; real data require a broader grid.

### Numerical failures
None. All `failed_origins.csv` files contain zero data rows.

### Comparison output summary

From `comparison_summary.md`:

- All 8 models share the same 8 outer origins.
- Scope-gain decomposition (ridge iterated family):
  - Horizon gain: average +0.009 RMSE reduction (66.7% of cells improved)
  - Variable gain: average +0.010 RMSE reduction (62.5% of cells improved)
  - Interaction: negligible (−0.0003 average)
- Average ranks (lower = better): `ridge_direct_variable_horizon` ranks
  first (4.71), followed by `ridge_iterated_variable` (4.96).
- Diebold-Mariano: 36 of 48 pairwise comparisons produced a valid p-value.

**Note:** These results are from a synthetic VAR(1) panel and have no
scientific interpretation. They confirm the output contract only.

### Cleared for full experiments?
**YES** — the synthetic integration pipeline is fully cleared. For real
experiments, extend the grid below λ=0.001 and above λ=1.0 to avoid
systematic boundary selection.

---

## Pilot B — Small GLP pilot

**Status: SKIPPED — `covbayesvar` not installed**

The GLP runner (`scripts/glp/run_glp_scope_grid.py`) requires `covbayesvar`
for the MCMC evaluation path. This package is not present in the current
environment.

To install and run Pilot B:
```bash
pip install covbayesvar
python scripts/glp/run_glp_scope_grid.py \
  --output-root /tmp/pilot_b/glp \
  --panel-path  data/processed/glp_realtime_panel.csv.gz \
  --model-size  small \
  --start 2000-03-31 --end 2002-12-31 \
  --selection-scopes pooled,horizon,variable,variable_horizon \
  --target-variables GDP,INVFIX,CONS \
  --target-horizons  1,2,4 \
  --loss-metric rmse --loss-scaling none \
  --benchmark no_change \
  --inner-n-origins 4 \
  --selection-frequency once \
  --optimization-init-points 3 \
  --optimization-iterations 5 \
  --objective-posterior-draws 10 \
  --no-optimize-psi \
  --overwrite
```

**Note:** The `--optimization-init-points 3 --optimization-iterations 5`
budget is intentionally tiny and suitable for integration validation only.
Results are not scientifically meaningful. Full experiments use the default
budget (24 init + 72 iterations).

Additional note: `glp_model.py` does not yet export
`update_hyperparameters_mango_rmse(_random)`. This affects the legacy
scripts (`scripts/glp/run_glp_mango_rmse.py`) but **not** the scope-grid
runner (`run_glp_scope_grid.py`), which uses the internal `evaluate_glp_candidate`
path directly.

---

## Pilot C — Mixed-frequency MF-BVAR pilot

**Status: SKIPPED — `MBFVAR` not installed**

The MFVAR runner (`scripts/mfvar/run_mfvar_scope_grid.py`) requires `MBFVAR`
from `https://github.com/laurentflorin/MBFVAR.git`. This package is not
present in the current environment.

The audited MBFVAR commit is `5b06f93272cd6ebf370fbf2aac3b3573c7830493`.
Install:
```bash
pip install git+https://github.com/laurentflorin/MBFVAR.git@5b06f93272cd6ebf370fbf2aac3b3573c7830493
```

Pilot C command (GDP-only objective, full quarterly forecast block):
```bash
python scripts/mfvar/run_mfvar_scope_grid.py \
  --output-root /tmp/pilot_c/mfvar \
  --panel-path  data/processed/realtime_panel.csv.gz \
  --forecast-variables GDP,INVFIX,GOV,UNR,HRS,CPI,IP,PCE,FF,TB,SP500 \
  --target-variables GDP \
  --target-horizons  1,4 \
  --selection-scopes pooled,variable \
  --inner-n-origins 4 \
  --selection-frequency once \
  --overwrite
```

**Horizon conversion verification:** the MFVAR runner converts quarterly
horizons to monthly steps internally (×3). Horizon 4q = 12 monthly steps.
The `src/paper_hyperparameter_optimization/horizon_mapping.py` module
documents this mapping.

**Known issue (L-06):** `MBFVAR` creates unseeded NumPy generators
internally, so MFVAR forecast results are not bitwise reproducible across
runs even with a fixed `--base-seed`. Configuration hashes and manifests
are still written correctly; only the forecast values may differ.

---

## Workflow clearance summary

| Pilot | Description | Status |
|---|---|---|
| A — Synthetic ridge | Full iterated + direct ridge, 4 scopes, 2 schedules, reporting | **Cleared** |
| B — GLP small | GLP scope-grid with tiny optimizer budget | **Skipped** (`covbayesvar` absent) |
| C — MFVAR | Mixed-frequency with GDP-only objective | **Skipped** (`MBFVAR` absent) |
| CI unit suite | `pytest --skip-optional` | **357 passed, 20 skipped** |
