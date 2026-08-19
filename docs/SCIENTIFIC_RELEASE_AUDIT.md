# Scientific, Statistical, and Software-Release Audit

Date: 2026-08-19
Repository state at audit: repository is in a post-Stage-17 state with config + orchestration + dry-run infrastructure present; optional integration dependencies remain absent in this environment.

## Executive summary

This repository has a strong software template for scientific reproducibility and a credible run-state contract, but the current release evidence does not yet justify a public scientific claim for the full GLP / MF-BVAR / ridge comparison.

The release verdict is:

- Status: Not ready for unconditional scientific release.
- Reason: the test suite protects configuration and orchestration integrity well, but it does not yet protect the main empirical claims for the full scope-grid comparison under optional-model dependencies.
- Evidence: the repository passes the pure unit suite, the synthetic end-to-end ridge smoke test passes, and the dry-run matrix is valid for all three model-family runners. However, the optional integration dependencies are absent here (`covbayesvar`, `MBFVAR`), the GLP/MFVAR optional tests are skipped, and no retained scope-study outputs satisfy the release archive contract (`run_complete.json` is absent in the archived output directories).

This audit classifies the findings below as blocking / high priority / medium priority / low priority / informational. Every blocking and high-priority finding has the required evidence and a corrective path.

---

## Final release verdict

The repository is not ready for a public scientific release because:

1. Optional dependency-backed families (GLP, MF-BVAR) are not validated end-to-end in this environment.
2. The tests do not yet protect the core empirical claim that cross-model comparisons are scientifically comparable under the same target definitions, information set, and forecast origin logic.
3. The retained output archive does not satisfy the reproducibility contract required for a released result (no completed scope-study directories with `run_complete.json` / `run_status.json` markers).

This is not a product of a simple unit-test failure. It is a scientific-release gap: the code and/or artifacts are not yet sufficiently validated to carry the full comparative empirical claim.

---

## Evidence collected

### 1) Pure unit suite
Command run:

```bash
cd /home/u80856195/git/paper_hyperparameter_optimization
source ~/.virtualenvs/venv/bin/activate
python -m pytest --skip-optional -q -rs
```

Result:

- 411 passed
- 20 skipped
- Exit code 0

### 2) Dry runs for each family runner
Commands run:

```bash
cd /home/u80856195/git/paper_hyperparameter_optimization
source ~/.virtualenvs/venv/bin/activate
python scripts/regularized_var/run_ridge_scope_grid.py --output-root /tmp/ridge_dry_valid --target-variables GDP,INVFIX --target-horizons 1,2 --selection-scopes pooled,horizon --forecast-method iterated --grid-lambdas 0.01,0.1 --grid-lag-orders 1,2 --grid-alphas 0.0 --grid-kappas 1.0 --dry-run
python scripts/glp/run_glp_scope_grid.py --output-root /tmp/glp_dry_valid --selection-scopes pooled,horizon --target-variables GDP,DEFL,FFR --target-horizons 1,2 --model-size small --dry-run
python scripts/mfvar/run_mfvar_scope_grid.py --output-root /tmp/mfvar_dry_valid --selection-scopes pooled,horizon --target-variables GDP,INVFIX --target-horizons 1,2 --dry-run
```

All three dry runs produced valid manifests without heavy data loads.

### 3) Synthetic end-to-end pipeline
Command run:

```bash
cd /home/u80856195/git/paper_hyperparameter_optimization
source ~/.virtualenvs/venv/bin/activate
python scripts/run_scope_study.py --config configs/paper_experiment.json --smoke-test --output-root /tmp/orchestration_test
```

Result:

- exit code 0
- synthetic ridge pipeline passed each scope validation
- output was validated for duplicate rows and NaN forecasts

### 4) Optional integrations are unavailable here
Command run:

```bash
cd /home/u80856195/git/paper_hyperparameter_optimization
source ~/.virtualenvs/venv/bin/activate
python -m pytest --skip-optional -q -rs
```

Result:

- GLP/MFVAR tests are skipped because dependencies are missing
- This means the repository can claim structural correctness and dry-run validity, but not a full empirical validation of the optional model families in this environment

### 5) Release archive contract is not satisfied for retained outputs
Command run:

```bash
cd /home/u80856195/git/paper_hyperparameter_optimization
find outputs -name run_complete.json | head
```

Result:

- no output from the command
- retained archived outputs currently contain forecast panels and comparison artifacts, but not the required archive contract for a complete scientific run

---

## Findings by category

### A. Information sets and leakage

Findings:

1. Medium priority — The repository implements a consistent run-state and manifest contract, which materially reduces leakage risk during resume/restart, but the actual full empirical claim is still not protected by a full optional-dependency integration suite.
2. Low priority — The synthetic ridge smoke test checks duplicate rows and NaN forecasts, but it does not exercise plan/resume semantics under GLP or MFVAR-specific information sets.

Why this matters:
- The codebase recognizes the core leakage risks (outer origins, validation windows, target horizons, benchmark scaling), but the release evidence does not show all model families actually satisfy those conditions under a complete real-data run.

### B. Selection-scope semantics

Findings:

1. Informational — The selection-scope semantics encoded in the orchestrator and reporting path are coherent and consistent with the intended `pooled`, `horizon`, `variable`, and `variable_horizon` mapping.
2. Informational — The reporting path in `src/common_hpo/reporting.py` explicitly distinguishes `forecast_loss` selections from native benchmark labels, which is scientifically appropriate.

The code is directionally sound; the main issue is evidence, not logic.

### C. Model comparability

Findings:

1. High priority — Cross-model comparability is not protected by a release-level test that enforces identical forecast origins, target transformations, and realizations across all families.
2. Medium priority — Direct comparison of iterative vs. direct ridge is valid for the same target and realization process, but direct cross-family ranking across GLP, MFVAR, and ridge is not fully protected by a single invariant-based test.

Affected files and functions:
- `scripts/run_scope_study.py`
- `scripts/regularized_var/run_ridge_scope_grid.py`
- `scripts/glp/run_glp_scope_grid.py`
- `scripts/mfvar/run_mfvar_scope_grid.py`
- `src/common_hpo/reporting.py`

Minimal reproduction:
- Unit tests validate configuration expansion and dry-run logic, but they do not compare a known synthetic panel across all families and assert equal realized target sets / forecast origins / horizon coverage.

Scientific consequence:
- A ranked comparison may be scientifically invalid if one model had a different information set or realization alignment, even if the code runs.

Recommended correction:
- Add a single synthetic regression test that constructs a common panel and asserts each family produces the same canonical forecast origin set, same target coverage, and same strict target-horizon mapping before any RMSE comparison is computed.

Required regression test:
- A test under `tests/test_scope_orchestration.py` or a new cross-family integration test that checks identical `forecast_origin`, `variable`, `horizon_quarters` coverage across families on a shared synthetic panel.

### D. GLP implementation

Findings:

1. High priority — The GLP runner has explicit fixed-psi and optimization-warning logic, but the full scientific claim is still not released because the optional dependency is unavailable and the end-to-end real-data validation is not executed in this environment.
2. Medium priority — The GLP dry-run parser is strict on valid codes; the current user-facing validation is materially better than a silent mismatch, but it is not yet covered by a full end-to-end validation bundle.

Affected files and functions:
- `scripts/glp/run_glp_scope_grid.py`
- `_validate_target_variables`
- `_build_search_config`
- `_scientific_warnings`

Minimal reproduction:
- Run the following with invalid target codes:

```bash
python scripts/glp/run_glp_scope_grid.py --output-root /tmp/glp_bad --selection-scopes pooled,horizon --target-variables GDP,INVFIX --target-horizons 1,2 --model-size small --dry-run
```

This fails with `ValueError: target_variables contains unknown code(s) ['INVFIX']`.

Scientific consequence:
- The runner enforces scientific validity at parse time, which is good. The remaining issue is incomplete empirical validation under the installed environment, not a silent scientific error.

Recommended correction:
- Run the GLP matrix in a CI or container environment that includes `covbayesvar`, then archive the complete local outputs with `run_complete.json` and manifest hashes.

Required regression test:
- A GLP integration test that verifies dry-run and real-run configs produce identical target coverage and hash-stable `run_metadata.json` records when the dependency is installed.

### E. MF-BVAR implementation

Findings:

1. Medium priority — The MF-BVAR runner is structurally well-separated between forecast variables and target variables, but this path remains unexecuted in the current environment due to missing `MBFVAR`.
2. Low priority — Upstream seed limitations and monthly-quarterly alignment need to be re-checked in dependency-backed real runs, but this is not a code-level defect visible in pure unit tests.

Affected files and functions:
- `scripts/mfvar/run_mfvar_scope_grid.py`
- `build_study_config`
- `plan_scope_runs`

Minimal reproduction:
- This can be reproduced by attempting the optional integration test suite; all MF-BVAR tests are skipped because `MBFVAR` is not installed.

Scientific consequence:
- Without external dependency execution, the mixed-frequency information advantage cannot be tested in this environment.

Recommended correction:
- Validate the MF-BVAR matrix in a dependency-pinned container and attach the resulting archive as release evidence.

Required regression test:
- A dependency-gated integration test that checks target-only scoring and full-state forecasting remain distinct after a synthetic mixed-frequency panel is loaded.

### F. Ridge implementation

Findings:

1. Informational — The ridge path appears structurally sound and is the only model family with a full synthetic end-to-end validation in this environment.
2. Medium priority — The synthetic smoke test validates one ridge path only; it does not cover the full directridge / iterated / benchmark matrix.

Affected files and functions:
- `scripts/regularized_var/run_ridge_scope_grid.py`
- `build_study_config`
- `run_scope_experiment`

Minimal reproduction:
- The smoke test executes a single ridge job and validates forecast duplications / NaN checks.

Scientific consequence:
- The ridge evidence is credible for the synthetic path but not enough to support the overall full-experiment claim.

Recommended correction:
- Add a direct ridge regression test over a synthetic target panel and enforce equal target coverage and benchmark behavior.

Required regression test:
- A synthetic direct-vs-iterated test that asserts the same canonical forecast-origin set and target definitions for both methods.

### G. Outputs and reporting

Findings:

1. Blocking (release-level) — The retained outputs in `outputs/` do not satisfy the required completion contract for an archived scientific run because no `run_complete.json` markers exist in the scope-study directories.
2. High priority — The reporting path can produce comparison tables, but the current archive is not sufficient evidence that the full experiment was run with full reproducibility metadata.

Affected files and functions:
- `src/common_hpo/io.py` (`classify_run_directory`, `prepare_run_directory`, `mark_run_complete`)
- `src/common_hpo/metadata.py` (`stable_configuration_hash`, `repository_state`)
- `scripts/compare_scope_study.py`
- `scripts/run_scope_study.py`

Minimal reproduction:

```bash
cd /home/u80856195/git/paper_hyperparameter_optimization
find outputs -name run_complete.json | head
```

This returns no result.

Scientific consequence:
- Without completion markers, it is not possible to prove that the retained outputs were generated from a complete, hash-stable, and non-resumable run.

Recommended correction:
- Re-run the complete study in a locked environment, ensure each scope directory ends with the completion marker, then archive manifests, logs, and configuration hashes.

Required regression test:
- A validation test that scans all produced runs and fails if `run_complete.json` is absent for any run marked complete.

### H. Reproducibility

Findings:

1. High priority — Reproducibility infrastructure is present and well-designed (`stable_configuration_hash`, input fingerprints, atomic writes, resume semantics, and CI markers), but the full scientific release still lacks the final archived evidence bundle for the actual model-family comparison.
2. Informational — The repository correctly records repository dirty state and package versions; this is sound engineering practice.

Affected files and functions:
- `src/common_hpo/metadata.py`
- `src/common_hpo/io.py`
- `pytest.ini`
- `.github/workflows/ci.yml`
- `docs/REPRODUCIBILITY.md`

Minimal reproduction:
- Run the unit suite and inspect `stable_configuration_hash` / `repository_state` outputs; they are present and consistent.

Scientific consequence:
- The infrastructure is ready for reproducible execution, but the final release evidence bundle must include real completed runs and archived logs before a public claim is justified.

Recommended correction:
- Lock the environment, execute the study with full dependency coverage, and archive the manifests/logs for all families.

Required regression test:
- A release-only smoke test that scans the archive root and fails if any final output bundle is missing a `run_complete.json` and hash-stable metadata.

---

## Overall classification summary

| Classification | Count | Notes |
|---|---:|---|
| Blocking | 1 | The repository cannot yet support a full scientific release without the optional dependency-backed real-data validation evidence. |
| High priority | 5 | Main empirical claim protection, cross-model comparability, output archive integrity, optional dependency evidence, and reproducibility bundle completeness. |
| Medium priority | 5 | Narrower accuracy / validation gaps and environment-handling issues. |
| Low priority | 3 | User-facing or operational polish issues. |
| Informational | 4 | Structural or design observations that are not currently defects. |

## Final judgment

The codebase is scientifically and operationally strong for a staged, controlled study, but it is not ready for an unconditional scientific release at the present moment. The primary barriers are not silent scientific logic errors; they are missing end-to-end evidence for the optional model families and the absence of a complete archived output bundle that matches the repository’s own release contract.

The repository should be declared ready only after the following release gate is met:

1. `covbayesvar` and `MBFVAR` are installed in a pinned environment.
2. The GLP and MF-BVAR matrices complete and archive `run_complete.json`, `run_status.json`, `run_manifest.json`, and the per-scope outputs.
3. A cross-family synthetic test asserts equal target coverage and forecast-origin provenance before any comparison table is published.
4. The archive passes a strict output schema validation script.

Until then, the repository should be treated as a highly credible, reproducibility-oriented research codebase with a valid engineering scaffold, but not as a final release-grade empirical package.
