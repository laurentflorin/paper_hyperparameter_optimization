# Full Experiment Checklist

This is the release checklist in the exact order the repository should be executed for a real end-to-end empirical run. The sequence is intentionally conservative: it validates system dependencies, fingerprints inputs, executes each model family, validates outputs, and only then generates inferential outputs and archives the evidence bundle.

## 1) Validate local dependencies

```bash
cd /home/u80856195/git/paper_hyperparameter_optimization
source ~/.virtualenvs/venv/bin/activate
python -m pip install -r requirements.txt
python - <<'PY'
import importlib
mods = [
    'numpy','scipy','pandas','matplotlib','joblib','requests',
    'plotly','seaborn','covbayesvar','MBFVAR','sklearn'
]
for name in mods:
    try:
        importlib.import_module(name)
        print(f'OK {name}')
    except Exception as exc:
        print(f'MISSING {name}: {type(exc).__name__}: {exc}')
PY
```

Expected outcome:
- `numpy`, `scipy`, `pandas`, and the required scientific packages are importable.
- If `covbayesvar` or `MBFVAR` are not available, the release must be treated as dependency-gated and not as a full empirical release.

## 2) Validate data fingerprints

```bash
cd /home/u80856195/git/paper_hyperparameter_optimization
source ~/.virtualenvs/venv/bin/activate
python - <<'PY'
from pathlib import Path
from common_hpo.metadata import fingerprint_input_files
root = Path('data')
paths = [
    root / 'processed' / 'download_metadata.json',
    root / 'raw' / 'alfred_realtime',
    root / 'raw' / 'fred_latest',
    root / 'raw' / 'glp_alfred_realtime',
    root / 'raw' / 'glp_fred_latest',
]
# Keep only files that actually exist; the function will fail on missing paths.
real = [p for p in paths if p.exists()]
print(fingerprint_input_files(real, root=Path('.')))
PY
```

Expected outcome:
- SHA-256 hashes and file metadata are printed for the data inputs.
- This is the baseline input evidence for the experiment.

## 3) GLP matrix dry run and execution

```bash
cd /home/u80856195/git/paper_hyperparameter_optimization
source ~/.virtualenvs/venv/bin/activate
python scripts/run_scope_study.py --config configs/paper_experiment.json --filter-family glp --dry-run
python scripts/run_scope_study.py --config configs/paper_experiment.json --filter-family glp --resume --output-root outputs/scope_study
```

Expected outcome:
- The GLP dry-run prints the exact 4-job matrix.
- The GLP execution produces per-job directories under `outputs/scope_study/jobs` and writes `study_status.json` after completion.

## 4) MF-BVAR matrix dry run and execution

```bash
cd /home/u80856195/git/paper_hyperparameter_optimization
source ~/.virtualenvs/venv/bin/activate
python scripts/run_scope_study.py --config configs/paper_experiment.json --filter-family mfvar --dry-run
python scripts/run_scope_study.py --config configs/paper_experiment.json --filter-family mfvar --resume --output-root outputs/scope_study
```

Expected outcome:
- The MF-BVAR dry-run prints the exact job matrix.
- The MF-BVAR execution produces `job_manifest.json`, `job.log`, and scope-level outputs.

## 5) Ridge and direct ridge dry run and execution

```bash
cd /home/u80856195/git/paper_hyperparameter_optimization
source ~/.virtualenvs/venv/bin/activate
python scripts/run_scope_study.py --config configs/paper_experiment.json --filter-family ridge --dry-run
python scripts/run_scope_study.py --config configs/paper_experiment.json --filter-family ridge --resume --output-root outputs/scope_study
```

Expected outcome:
- The ridge run executes both `iterated` and `direct` forecast-method jobs and respects the `forecast_loss` configuration.
- Each ridge job produces scope directories with `forecast_panel.csv`, `selected_hyperparameters.csv`, and completion markers.

## 6) Optional Minnesota benchmarks

```bash
cd /home/u80856195/git/paper_hyperparameter_optimization
source ~/.virtualenvs/venv/bin/activate
python scripts/run_scope_study.py --config configs/paper_experiment.json --filter-family ridge --dry-run
python scripts/compare_scope_study.py --config configs/paper_experiment.json --output-root outputs/scope_study --benchmark minnesota
```

Expected outcome:
- The benchmark panel is built only if the optional Minnesota benchmark implementation is available.
- If benchmark dependencies are unavailable, the run must be documented as a scientific limitation and excluded from the public comparison panel.

## 7) Validate outputs

```bash
cd /home/u80856195/git/paper_hyperparameter_optimization
source ~/.virtualenvs/venv/bin/activate
python -m pytest tests/test_experiment_manifest.py tests/test_scope_orchestration.py -q --skip-optional
python tests/pilot_a_validate.py --iterated-root /tmp/pilot_a/iterated --direct-root /tmp/pilot_a/direct --comparison /tmp/pilot_a/comparison
python scripts/inspect_scope_study.py --config configs/paper_experiment.json --output-root outputs/scope_study --json > outputs/scope_study/inspection.json
```

Expected outcome:
- All orchestration tests pass.
- Every selected output bundle has valid structure and coverage.
- `inspection.json` shows the scope-study status counts and storage usage.

## 8) Generate comparison tables

```bash
cd /home/u80856195/git/paper_hyperparameter_optimization
source ~/.virtualenvs/venv/bin/activate
python scripts/compare_scope_study.py --manifest outputs/scope_study/study_status.json --output-dir outputs/comparison
```

Expected outcome:
- Comparison tables are written under `outputs/comparison/`.
- The resulting files should be checked for canonical uniqueness, target coverage, and matched forecast origins.

## 9) Run inference

```bash
cd /home/u80856195/git/paper_hyperparameter_optimization
source ~/.virtualenvs/venv/bin/activate
python -m pytest --skip-optional -q -rs
python scripts/inspect_scope_study.py --config configs/paper_experiment.json --output-root outputs/scope_study --summary
```

Expected outcome:
- The code passes the full unit path and optional integration is only skipped when dependencies are unavailable.
- The study summary indicates whether the complete run is structurally valid and what remains unvalidated.

## 10) Archive manifests and logs

```bash
cd /home/u80856195/git/paper_hyperparameter_optimization
source ~/.virtualenvs/venv/bin/activate
mkdir -p artifacts/archive
cp -R outputs/scope_study artifacts/archive/
cp -R outputs/comparison artifacts/archive/
find outputs/scope_study -type f \( -name 'job_manifest.json' -o -name 'job.log' -o -name 'run_manifest.json' -o -name 'run_status.json' -o -name 'run_complete.json' -o -name 'study_status.json' \) | sort > artifacts/archive/manifest_file_list.txt
git rev-parse HEAD > artifacts/archive/git_commit.txt
git status --porcelain --untracked-files=no > artifacts/archive/git_dirty_state.txt
```

Expected outcome:
- The archive contains the full run-state bundle, logs, and the final comparison outputs.
- The scientific archive is ready for inspection and re-use.

---

## Release gate

The experiment is release-ready only if:

- all dependency checks pass in the pinned environment;
- the GLP, MF-BVAR, and ridge matrices have run and archived complete markers;
- all output validation checks pass;
- the comparison tables are generated from matching information sets;
- the manifest and log bundle is archived.

If any optional dependency is missing, the result must be documented as a scientific limitation and not presented as a full-model family release claim.
