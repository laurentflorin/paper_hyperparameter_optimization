"""Tests for experiment manifest loading, validation, and job expansion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for p in (str(SRC_ROOT), str(SCRIPTS_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Load orchestrator module from file to avoid sys.modules pollution
_spec = importlib.util.spec_from_file_location(
    "run_scope_study", str(REPO_ROOT / "scripts" / "run_scope_study.py")
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["run_scope_study"] = _mod
_spec.loader.exec_module(_mod)

load_config = _mod.load_config
validate_config = _mod.validate_config
expand_jobs = _mod.expand_jobs
_expand_env = _mod._expand_env
_make_job_id = _mod._make_job_id
_scientific_config = _mod._scientific_config
_job_output_dir = _mod._job_output_dir
JobSpec = _mod.JobSpec


CONFIG_PATH = REPO_ROOT / "configs" / "paper_experiment.json"

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def minimal_config():
    """A minimal valid config for unit-testing without optional packages."""
    return {
        "study": {
            "name": "test",
            "version": "1",
            "output_root": "/tmp/test_study",
            "seed_base": 42,
        },
        "families": [
            {
                "family": "ridge",
                "enabled": True,
                "runner": "scripts/regularized_var/run_ridge_scope_grid.py",
                "panel_path": "/tmp/panel.csv",
                "scope_dir_pattern": "scope_{scope}",
                "requires": [],
                "sizes": [None],
                "forecast_methods": ["iterated"],
                "scopes": ["pooled", "variable"],
                "target_variables": ["gdp", "inv"],
                "target_horizons": [1, 4],
                "outer_origins": {"n_origins": 4, "stride": 1, "origin_selection": "recent"},
                "execution": {"n_workers": "1"},
                "variants": [
                    {
                        "name": "forecast_loss",
                        "enabled": True,
                        "preprocessing": "none",
                        "grid": {"lambdas": [0.1, 1.0], "lag_orders": [1]},
                        "loss_metric": "rmse",
                        "loss_scaling": "none",
                        "inner_validation": {"window": "expanding", "n_origins": 3},
                        "selection_schedule": "once",
                        "benchmarks": [],
                        "base_seed": 42,
                    }
                ],
            }
        ],
        "seeds": [42],
        "parallelism": {"max_concurrent_jobs": "1", "max_nested_workers": "1"},
    }


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #

def test_load_config_reads_real_file():
    config = load_config(CONFIG_PATH)
    assert "study" in config
    assert "families" in config
    assert config["study"]["name"] == "paper_experiment"


def test_expand_env_substitutes_env_var(monkeypatch):
    monkeypatch.setenv("MY_ROOT", "/cluster/scratch")
    result = _expand_env("${MY_ROOT:-default}")
    assert result == "/cluster/scratch"


def test_expand_env_uses_default_when_var_absent(monkeypatch):
    monkeypatch.delenv("ABSENT_VAR_XYZ", raising=False)
    result = _expand_env("${ABSENT_VAR_XYZ:-fallback_value}")
    assert result == "fallback_value"


def test_expand_env_recursive():
    d = {"a": "${ABSENT_XYZ:-x}", "b": ["${ABSENT_XYZ2:-y}"]}
    result = _expand_env(d)
    assert result == {"a": "x", "b": ["y"]}


def test_expand_env_leaves_non_strings_unchanged():
    assert _expand_env(42) == 42
    assert _expand_env(None) is None
    assert _expand_env(3.14) == 3.14


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def test_validate_minimal_config_passes(minimal_config):
    errors = validate_config(minimal_config)
    assert errors == []


def test_validate_real_config_passes():
    config = load_config(CONFIG_PATH)
    errors = validate_config(config)
    assert errors == [], f"Validation errors: {errors}"


def test_validate_missing_study_key_fails():
    config = {"families": [], "seeds": [1], "parallelism": {}}
    errors = validate_config(config)
    assert any("study" in e for e in errors)


def test_validate_missing_families_key_fails():
    config = {"study": {"name": "x", "version": "1", "output_root": "/tmp", "seed_base": 1},
              "seeds": [1], "parallelism": {}}
    errors = validate_config(config)
    assert any("families" in e for e in errors)


def test_validate_empty_variants_fails():
    config = {
        "study": {"name": "x", "version": "1", "output_root": "/tmp", "seed_base": 1},
        "families": [{"family": "ridge", "runner": "r.py", "variants": []}],
        "seeds": [1],
        "parallelism": {},
    }
    errors = validate_config(config)
    assert any("variant" in e for e in errors)


def test_validate_duplicate_family_fails():
    base = {
        "study": {"name": "x", "version": "1", "output_root": "/tmp", "seed_base": 1},
        "seeds": [1],
        "parallelism": {},
    }
    fam = {"family": "ridge", "runner": "r.py", "variants": [{"name": "v"}]}
    base["families"] = [fam, fam]
    errors = validate_config(base)
    assert any("duplicate" in e.lower() for e in errors)


# --------------------------------------------------------------------------- #
# Job expansion
# --------------------------------------------------------------------------- #

def test_expand_jobs_ridge_minimal(minimal_config, tmp_path):
    jobs = expand_jobs(minimal_config, tmp_path)
    # 1 family × 1 method × 1 variant × 1 seed = 1 job
    assert len(jobs) == 1
    job = jobs[0]
    assert job.family == "ridge"
    assert job.variant == "forecast_loss"
    assert job.scopes == ["pooled", "variable"]
    assert job.seed == 42


def test_expand_jobs_disabled_variant_excluded(minimal_config, tmp_path):
    minimal_config["families"][0]["variants"][0]["enabled"] = False
    jobs = expand_jobs(minimal_config, tmp_path)
    assert len(jobs) == 0


def test_expand_jobs_disabled_family_excluded(minimal_config, tmp_path):
    minimal_config["families"][0]["enabled"] = False
    jobs = expand_jobs(minimal_config, tmp_path)
    assert len(jobs) == 0


def test_expand_jobs_multiple_seeds(minimal_config, tmp_path):
    minimal_config["seeds"] = [1, 2, 3]
    jobs = expand_jobs(minimal_config, tmp_path)
    assert len(jobs) == 3
    seeds = {j.seed for j in jobs}
    assert seeds == {1, 2, 3}


def test_expand_jobs_multiple_methods(minimal_config, tmp_path):
    minimal_config["families"][0]["forecast_methods"] = ["iterated", "direct"]
    jobs = expand_jobs(minimal_config, tmp_path)
    assert len(jobs) == 2
    methods = {j.forecast_method for j in jobs}
    assert methods == {"iterated", "direct"}


def test_job_id_is_deterministic(minimal_config, tmp_path):
    jobs1 = expand_jobs(minimal_config, tmp_path)
    jobs2 = expand_jobs(minimal_config, tmp_path)
    assert [j.job_id for j in jobs1] == [j.job_id for j in jobs2]


def test_configuration_hash_is_stable(minimal_config, tmp_path):
    jobs1 = expand_jobs(minimal_config, tmp_path)
    jobs2 = expand_jobs(minimal_config, tmp_path / "other_root")  # different path
    # Same scientific config → same hash regardless of output root
    assert jobs1[0].configuration_hash == jobs2[0].configuration_hash


def test_configuration_hash_changes_with_variant(minimal_config, tmp_path):
    jobs_a = expand_jobs(minimal_config, tmp_path)
    minimal_config["families"][0]["variants"][0]["grid"]["lambdas"] = [0.5, 2.0]
    jobs_b = expand_jobs(minimal_config, tmp_path)
    assert jobs_a[0].configuration_hash != jobs_b[0].configuration_hash


def test_job_cli_contains_output_root_placeholder(minimal_config, tmp_path):
    jobs = expand_jobs(minimal_config, tmp_path)
    assert "__OUTPUT_ROOT__" in jobs[0].cli_command


def test_job_cli_contains_target_variables(minimal_config, tmp_path):
    jobs = expand_jobs(minimal_config, tmp_path)
    cli = " ".join(jobs[0].cli_command)
    assert "gdp" in cli.lower()


def test_scope_output_dir_uses_pattern(tmp_path, minimal_config):
    jobs = expand_jobs(minimal_config, tmp_path)
    job = jobs[0]
    expected = job.output_dir / "scope_pooled"
    assert job.scope_output_dir("pooled") == expected


def test_scope_output_dir_glp_uses_hyphens(minimal_config, tmp_path):
    minimal_config["families"][0]["scope_dir_pattern"] = "scope-{scope_hyphen}"
    minimal_config["families"][0]["scopes"] = ["variable_horizon"]
    jobs = expand_jobs(minimal_config, tmp_path)
    job = jobs[0]
    expected = job.output_dir / "scope-variable-horizon"
    assert job.scope_output_dir("variable_horizon") == expected


def test_real_config_expands_without_optional_deps():
    """The real config should expand all enabled jobs; GLP/MFVAR will be included."""
    config = load_config(CONFIG_PATH)
    jobs = expand_jobs(config, Path("/tmp/dry_run_test"))
    assert len(jobs) > 0
    families = {j.family for j in jobs}
    # Ridge should always be present (no optional deps)
    assert "ridge" in families


def test_real_config_job_ids_are_unique():
    config = load_config(CONFIG_PATH)
    jobs = expand_jobs(config, Path("/tmp/dry_run_test"))
    ids = [j.job_id for j in jobs]
    assert len(ids) == len(set(ids)), "Duplicate job IDs found"


def test_real_config_hashes_are_unique_per_job():
    config = load_config(CONFIG_PATH)
    jobs = expand_jobs(config, Path("/tmp/dry_run_test"))
    hashes = [j.configuration_hash for j in jobs]
    assert len(hashes) == len(set(hashes)), "Duplicate configuration hashes found"
