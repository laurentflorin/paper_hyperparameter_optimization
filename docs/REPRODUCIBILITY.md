# Reproducibility Guide

This document explains how to reproduce a run exactly, how to resume an
interrupted run safely, and how to verify that an existing output was
generated under the same conditions as a reference run.

---

## 1. What is recorded and where

Every scope-grid runner writes a `run_manifest.json` to each scope
sub-directory *before* expensive work starts. This file contains the
complete provenance record:

| Category | Fields recorded |
|---|---|
| Code state | `repository_commit`, `repository_dirty` |
| Python environment | `python_version`, `platform`, `relevant_packages` (with versions) |
| Model family | `model_family`, `model_version` |
| Data source | `data_source`, `data_vintage_identifiers` |
| Input fingerprints | `input_files` (SHA-256 per file, relative path, byte size, last-modified) |
| Scientific config | `transformation_configuration`, `variable_order`, `target_variables`, `target_horizons`, `selection_plan`, `validation_scheme`, `vintage_policy`, `selection_schedule`, `loss_configuration`, `search_space`, `optimizer_budget`, `random_seeds` |
| Run-time | `utc_started_at`, `command_line`, `argv` |
| Schema version | `output_schema_version` (currently `"1"`) |

All of the above fields are combined into a single **configuration hash**
(SHA-256 of the canonical JSON serialization). This hash is the primary key
for resume decisions.

---

## 2. Configuration hash

```
configuration_hash = SHA-256(canonical_json(full_manifest_minus_timestamps))
```

The hash is stable across JSON key ordering. It changes whenever:

- The input data file changes (SHA-256 mismatch).
- Any scientific parameter changes (`--grid-lambdas`, `--selection-scopes`, etc.).
- A relevant package version changes.
- The git commit changes (including switching to a clean commit from dirty).
- The worktree has uncommitted changes (`repository_dirty = true`).

Running the same command twice on the same machine with the same
uncommitted code produces the same hash as long as no files have changed.

---

## 3. Resume semantics

Use `--resume` to continue an interrupted run. The runner reads the
existing `run_manifest.json`, computes the configuration hash of the
*current* invocation, and compares:

| Existing state | Hashes match | Action |
|---|---|---|
| `missing` or `empty` | — | Start fresh |
| `partial`, `failed`, `cancelled` | yes | Re-run (reuses existing partial work where possible) |
| `partial`, `failed`, `cancelled` | no | **Reject** (`ResumeRejectedError`): different configuration |
| `complete` | yes | **Skip** (outputs already valid) |
| `complete` | no | **Reject**: cannot overwrite a completed run with a different configuration |
| legacy (no `run_manifest.json`) | — | **Reject**: pre-hash directory cannot be safely resumed |

Use `--overwrite` to unconditionally regenerate any directory.

**Never use `--overwrite` to reuse a directory as the "same" run when the
configuration has changed.** The outputs will be tagged with the new hash
but the old outputs will be silently replaced.

---

## 4. Reproducing a run from a manifest

Given a `run_manifest.json`, a run can be re-launched as follows:

```bash
# 1. Check out the exact commit
git checkout $(jq -r .repository_commit run_manifest.json)

# 2. Verify no dirty-tree contamination (must be false)
python -c "import json; m=json.load(open('run_manifest.json')); assert not m['repository_dirty'], 'worktree was dirty'"

# 3. Install the exact package versions
pip install $(python -c "import json; m=json.load(open('run_manifest.json')); print(' '.join(f\"{k}=={v}\" for k,v in m['relevant_packages'].items() if v))")

# 4. Re-run with the same argv
python $(jq -r '.command_line' run_manifest.json) $(jq -r '.argv[]' run_manifest.json)
```

For GLP runs where `MBFVAR` uses unseeded internal RNG, bitwise
reproducibility cannot be guaranteed (see [Known limitations, L-06](EXPERIMENT_DESIGN.md#known-limitations)).

---

## 5. Verifying an existing output

To check that an existing output directory matches a stored hash:

```python
import json
from pathlib import Path
import sys
sys.path.insert(0, "src")
from common_hpo.metadata import stable_configuration_hash

manifest = json.loads(Path("run_manifest.json").read_text())
stored_hash = manifest["configuration_hash"]
# Remove timestamps and hash before recomputing
check_manifest = {k: v for k, v in manifest.items()
                  if k not in ("configuration_hash", "utc_started_at", "utc_finished_at", "generated_utc")}
recomputed = stable_configuration_hash(check_manifest)
assert recomputed == stored_hash, f"hash mismatch: {recomputed} != {stored_hash}"
print("OK:", stored_hash[:16], "...")
```

---

## 6. Input file fingerprints

Each input file is recorded as:

```json
{
  "relative_path": "data/processed/panel.csv.gz",
  "sha256": "abc123...",
  "size_bytes": 123456,
  "last_modified_utc": "2026-01-01T00:00:00Z"
}
```

The SHA-256 digest changes whenever file content changes. The
`last_modified_utc` is informational only and not part of the hash.

---

## 7. Dirty-worktree policy

When `repository_dirty = true`:

- The run is still recorded and can complete.
- The configuration hash includes the dirty flag, so the hash differs from
  an otherwise-identical clean run.
- Resume from a dirty-tree run is possible as long as the hash matches
  (i.e., the worktree state at re-run time produces the same hash).
- Results from dirty-tree runs should be treated as exploratory and not
  cited in publications without re-running on a clean checkout.

---

## 8. Minimal CI environment

A minimal test environment without the optional scientific packages is
defined in `requirements-dev.txt`. Install it for CI:

```bash
pip install -r requirements-dev.txt
pytest --skip-optional -q -rs
```

The `--skip-optional` flag skips tests marked `integration`, `slow`,
`requires_covbayesvar`, or `requires_mbfvar`. The CI workflow in
`.github/workflows/ci.yml` uses this command.

See [docs/PILOT_VALIDATION.md](PILOT_VALIDATION.md) for a record of which
tests pass in the minimal environment.

---

## 9. Test tiers

| Marker | Meaning | Runs in minimal CI |
|---|---|---|
| `unit` | Auto-assigned to all tests not marked optional. Pure unit test, no heavy optional dependencies. | Yes |
| `integration` | Cross-module or real-data workflow test. | No (`--skip-optional`) |
| `slow` | Long-running computation. | No |
| `requires_covbayesvar` | Needs `covbayesvar` (GLP model). | No |
| `requires_mbfvar` | Needs `MBFVAR` (Schorfheide-Song MF-BVAR). | No |

Run the full suite (requires optional packages):
```bash
pytest -q
```

Run only fast unit tests:
```bash
pytest -m unit -q
```

Run only integration tests (requires optional packages):
```bash
pytest -m integration -q
```
