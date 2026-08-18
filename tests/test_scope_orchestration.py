"""Tests for the scope-study orchestrator: planning, execution, filtering.

All tests mock subprocess.run so no real GLP/MFVAR invocations are made.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for p in (str(SRC_ROOT),):
    if p not in sys.path:
        sys.path.insert(0, p)

# Load orchestrator module from file
_spec = importlib.util.spec_from_file_location(
    "run_scope_study_orch",
    str(REPO_ROOT / "scripts" / "run_scope_study.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["run_scope_study_orch"] = _mod  # required for dataclass __module__ lookup
_spec.loader.exec_module(_mod)

JobSpec = _mod.JobSpec
PlannedJob = _mod.PlannedJob
JobResult = _mod.JobResult
plan_job = _mod.plan_job
plan_jobs = _mod.plan_jobs
execute_job = _mod.execute_job
write_study_status = _mod.write_study_status
run_smoke_test = _mod.run_smoke_test
expand_jobs = _mod.expand_jobs
load_config = _mod.load_config


CONFIG_PATH = REPO_ROOT / "configs" / "paper_experiment.json"

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _make_spec(
    tmp_path: Path,
    family: str = "ridge",
    variant: str = "v1",
    scopes: list[str] | None = None,
    seed: int = 20150101,
    requires: list[str] | None = None,
    method: str | None = "iterated",
    scope_dir_pattern: str = "scope_{scope}",
) -> JobSpec:
    scopes = scopes or ["pooled", "variable"]
    job_dir = tmp_path / "jobs" / f"{family}_{variant}"
    sci = {"family": family, "variant": variant, "seed": seed}
    return JobSpec(
        job_id=f"{family}_{variant}_{seed}",
        family=family,
        variant=variant,
        size=None,
        forecast_method=method,
        scopes=scopes,
        seed=seed,
        output_dir=job_dir,
        scope_dir_pattern=scope_dir_pattern,
        cli_command=[sys.executable, "scripts/fake_runner.py", "--output-root", "__OUTPUT_ROOT__"],
        scientific_config=sci,
        requires=requires or [],
    )


def _make_complete_scope(scope_dir: Path) -> None:
    """Create a minimal directory that classify_run_directory marks as complete."""
    scope_dir.mkdir(parents=True, exist_ok=True)
    (scope_dir / "run_manifest.json").write_text(
        json.dumps({"configuration_hash": "aaa", "n_origins": 4, "n_variables": 2,
                    "n_horizons": 4, "scopes": [scope_dir.name]}),
        encoding="utf-8"
    )
    (scope_dir / "run_status.json").write_text(
        json.dumps({"status": "complete", "n_origins_complete": 4, "n_origins_failed": 0}),
        encoding="utf-8"
    )
    (scope_dir / "run_complete.json").write_text(
        json.dumps({"utc_completed": "2026-01-01T00:00:00Z", "configuration_hash": "aaa"}),
        encoding="utf-8"
    )
    (scope_dir / "forecast_panel.csv").write_text(
        "forecast_origin,variable,horizon_quarters,mean_metric\n"
        "2020-01-01,gdp,1,0.5\n",
        encoding="utf-8"
    )
    (scope_dir / "selected_hyperparameters.csv").write_text(
        "forecast_origin,variable,horizon_quarters\n"
        "2020-01-01,gdp,1\n",
        encoding="utf-8"
    )
    (scope_dir / "failed_origins.csv").write_text(
        "forecast_origin,variable,horizon_quarters,failure_category\n",
        encoding="utf-8"
    )


def _mock_subprocess_success(cmd, **kwargs):
    m = MagicMock()
    m.returncode = 0
    m.stdout = "mock runner ok\n"
    return m


def _mock_subprocess_failure(cmd, **kwargs):
    m = MagicMock()
    m.returncode = 1
    m.stdout = "mock runner error\n"
    return m


# --------------------------------------------------------------------------- #
# plan_job
# --------------------------------------------------------------------------- #

class TestPlanJob:
    def test_no_prior_run_returns_run(self, tmp_path):
        spec = _make_spec(tmp_path)
        planned = plan_job(spec, "error")
        assert planned.action == "run"
        assert planned.prior_status == "missing"

    def test_complete_with_error_policy_returns_skip(self, tmp_path):
        spec = _make_spec(tmp_path)
        for scope in spec.scopes:
            _make_complete_scope(spec.scope_output_dir(scope))
        planned = plan_job(spec, "error")
        assert planned.action == "skip"

    def test_complete_with_overwrite_policy_returns_run(self, tmp_path):
        spec = _make_spec(tmp_path)
        for scope in spec.scopes:
            _make_complete_scope(spec.scope_output_dir(scope))
        planned = plan_job(spec, "overwrite")
        assert planned.action == "run"

    def test_missing_dep_returns_skip_missing_dep(self, tmp_path):
        spec = _make_spec(tmp_path, requires=["__definitely_not_installed_pkg_xyz__"])
        planned = plan_job(spec, "error")
        assert planned.action == "skip_missing_dep"
        assert "__definitely_not_installed_pkg_xyz__" in planned.reason

    def test_partial_with_resume_returns_resume(self, tmp_path):
        spec = _make_spec(tmp_path, scopes=["pooled", "variable"])
        # Make only pooled complete
        _make_complete_scope(spec.scope_output_dir("pooled"))
        # Leave variable missing
        # Write job manifest with matching hash
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        (spec.output_dir / "job_manifest.json").write_text(
            json.dumps({"configuration_hash": spec.configuration_hash}),
            encoding="utf-8",
        )
        planned = plan_job(spec, "resume")
        assert planned.action == "resume"

    def test_partial_without_resume_returns_reject(self, tmp_path):
        spec = _make_spec(tmp_path, scopes=["pooled", "variable"])
        _make_complete_scope(spec.scope_output_dir("pooled"))
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        (spec.output_dir / "job_manifest.json").write_text(
            json.dumps({"configuration_hash": spec.configuration_hash}),
            encoding="utf-8",
        )
        planned = plan_job(spec, "error")
        assert planned.action == "reject"

    def test_hash_mismatch_returns_reject(self, tmp_path):
        spec = _make_spec(tmp_path)
        for scope in spec.scopes:
            _make_complete_scope(spec.scope_output_dir(scope))
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        (spec.output_dir / "job_manifest.json").write_text(
            json.dumps({"configuration_hash": "completely_different_hash_xyz"}),
            encoding="utf-8",
        )
        planned = plan_job(spec, "overwrite")
        assert planned.action == "reject"
        assert "hash mismatch" in planned.reason.lower()


# --------------------------------------------------------------------------- #
# execute_job
# --------------------------------------------------------------------------- #

class TestExecuteJob:
    def test_skip_action_returns_skipped(self, tmp_path):
        spec = _make_spec(tmp_path)
        planned = PlannedJob(spec, "skip", "already done", "complete")
        result = execute_job(planned, tmp_path / "logs")
        assert result.status == "skipped"
        assert result.exit_code is None

    def test_reject_action_returns_rejected(self, tmp_path):
        spec = _make_spec(tmp_path)
        planned = PlannedJob(spec, "reject", "hash mismatch", "incompatible")
        result = execute_job(planned, tmp_path / "logs")
        assert result.status == "rejected"

    def test_run_action_calls_subprocess(self, tmp_path):
        spec = _make_spec(tmp_path)
        planned = PlannedJob(spec, "run", "fresh run", "missing")
        called_cmds = []

        def mock_fn(cmd, **kwargs):
            called_cmds.append(cmd)
            # Create scope dirs to simulate complete scopes
            for scope in spec.scopes:
                _make_complete_scope(spec.scope_output_dir(scope))
            m = MagicMock()
            m.returncode = 0
            m.stdout = "done\n"
            return m

        result = execute_job(planned, tmp_path / "logs", _subprocess_fn=mock_fn)
        assert len(called_cmds) == 1
        assert result.exit_code == 0
        assert result.status == "complete"

    def test_run_substitutes_output_root(self, tmp_path):
        spec = _make_spec(tmp_path)
        planned = PlannedJob(spec, "run", "fresh run", "missing")
        captured_cmd = []

        def mock_fn(cmd, **kwargs):
            captured_cmd.extend(cmd)
            for scope in spec.scopes:
                _make_complete_scope(spec.scope_output_dir(scope))
            m = MagicMock()
            m.returncode = 0
            m.stdout = "done\n"
            return m

        execute_job(planned, tmp_path / "logs", _subprocess_fn=mock_fn)
        assert "__OUTPUT_ROOT__" not in captured_cmd
        assert str(spec.output_dir) in captured_cmd

    def test_failed_subprocess_records_failed_status(self, tmp_path):
        spec = _make_spec(tmp_path)
        planned = PlannedJob(spec, "run", "fresh run", "missing")

        result = execute_job(planned, tmp_path / "logs", _subprocess_fn=_mock_subprocess_failure)
        assert result.status == "failed"
        assert result.exit_code == 1

    def test_log_file_is_written(self, tmp_path):
        spec = _make_spec(tmp_path)
        planned = PlannedJob(spec, "run", "fresh run", "missing")

        execute_job(planned, tmp_path / "logs", _subprocess_fn=_mock_subprocess_success)
        log_path = tmp_path / "logs" / f"{spec.job_id}.log"
        assert log_path.exists()
        content = log_path.read_text()
        assert spec.job_id in content

    def test_job_manifest_written_after_run(self, tmp_path):
        spec = _make_spec(tmp_path)
        planned = PlannedJob(spec, "run", "fresh run", "missing")

        def mock_fn(cmd, **kwargs):
            for scope in spec.scopes:
                _make_complete_scope(spec.scope_output_dir(scope))
            m = MagicMock()
            m.returncode = 0
            m.stdout = "done\n"
            return m

        execute_job(planned, tmp_path / "logs", _subprocess_fn=mock_fn)
        manifest_path = spec.output_dir / "job_manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["job_id"] == spec.job_id
        assert manifest["status"] == "complete"


# --------------------------------------------------------------------------- #
# write_study_status
# --------------------------------------------------------------------------- #

class TestWriteStudyStatus:
    def test_creates_study_status_json(self, tmp_path):
        spec = _make_spec(tmp_path)
        results = [
            JobResult(spec, "complete", 0, 1.5, None, {"pooled": "complete"}),
        ]
        write_study_status(results, tmp_path)
        status_path = tmp_path / "study_status.json"
        assert status_path.exists()

    def test_summary_counts_correct(self, tmp_path):
        spec1 = _make_spec(tmp_path, variant="v1")
        spec2 = _make_spec(tmp_path, variant="v2")
        spec3 = _make_spec(tmp_path, variant="v3")
        results = [
            JobResult(spec1, "complete", 0, 1.0, None),
            JobResult(spec2, "failed", 1, 0.5, None),
            JobResult(spec3, "skipped", None, 0.0, None),
        ]
        write_study_status(results, tmp_path)
        data = json.loads((tmp_path / "study_status.json").read_text())
        assert data["summary"]["complete"] == 1
        assert data["summary"]["failed"] == 1
        assert data["summary"]["skipped"] == 1

    def test_job_records_include_hash(self, tmp_path):
        spec = _make_spec(tmp_path)
        results = [JobResult(spec, "complete", 0, 2.0, None)]
        write_study_status(results, tmp_path)
        data = json.loads((tmp_path / "study_status.json").read_text())
        assert data["jobs"][0]["configuration_hash"]


# --------------------------------------------------------------------------- #
# plan_jobs filtering
# --------------------------------------------------------------------------- #

class TestPlanJobsFiltering:
    def _make_specs(self, tmp_path) -> list[JobSpec]:
        return [
            _make_spec(tmp_path, family="ridge", variant="v1"),
            _make_spec(tmp_path, family="glp", variant="v1"),
            _make_spec(tmp_path, family="mfvar", variant="v1"),
        ]

    def test_filter_family(self, tmp_path):
        specs = self._make_specs(tmp_path)
        planned = plan_jobs(specs, "error", filter_family=["ridge"])
        assert all(p.spec.family == "ridge" for p in planned)
        assert len(planned) == 1

    def test_filter_multiple_families(self, tmp_path):
        specs = self._make_specs(tmp_path)
        planned = plan_jobs(specs, "error", filter_family=["ridge", "glp"])
        families = {p.spec.family for p in planned}
        assert families == {"ridge", "glp"}

    def test_filter_scope(self, tmp_path):
        specs = [
            _make_spec(tmp_path, variant="v1", scopes=["pooled"]),
            _make_spec(tmp_path, variant="v2", scopes=["variable"]),
        ]
        planned = plan_jobs(specs, "error", filter_scope=["pooled"])
        assert all("pooled" in p.spec.scopes for p in planned)
        assert len(planned) == 1

    def test_job_index_selects_one(self, tmp_path):
        specs = self._make_specs(tmp_path)
        planned = plan_jobs(specs, "error", job_index=1)
        assert len(planned) == 1
        assert planned[0].spec.family == "glp"

    def test_job_index_out_of_range_raises(self, tmp_path):
        specs = self._make_specs(tmp_path)
        with pytest.raises(IndexError):
            plan_jobs(specs, "error", job_index=99)


# --------------------------------------------------------------------------- #
# Smoke test (mocked subprocess)
# --------------------------------------------------------------------------- #

@pytest.mark.unit
def test_smoke_test_mocked_success(tmp_path, monkeypatch):
    """Smoke test with a mocked subprocess that creates complete scope dirs."""

    def mock_runner(cmd, **kwargs):
        # Find the output root (first arg after --output-root)
        idx = cmd.index("--output-root")
        job_dir = Path(cmd[idx + 1])
        for scope in ("pooled", "horizon", "variable", "variable_horizon"):
            _make_complete_scope(job_dir / f"scope_{scope}")
        m = MagicMock()
        m.returncode = 0
        return m

    rc = run_smoke_test(tmp_path, _subprocess_fn=mock_runner)
    assert rc == 0


@pytest.mark.unit
def test_smoke_test_mocked_failure(tmp_path):
    """Smoke test reports failure when subprocess fails."""

    def mock_runner(cmd, **kwargs):
        m = MagicMock()
        m.returncode = 1
        return m

    rc = run_smoke_test(tmp_path, _subprocess_fn=mock_runner)
    assert rc == 1


@pytest.mark.unit
def test_smoke_test_mocked_incomplete_scope(tmp_path):
    """Smoke test reports failure when a scope is not complete."""

    def mock_runner(cmd, **kwargs):
        idx = cmd.index("--output-root")
        job_dir = Path(cmd[idx + 1])
        # Only create pooled, omit others
        _make_complete_scope(job_dir / "scope_pooled")
        m = MagicMock()
        m.returncode = 0
        return m

    rc = run_smoke_test(tmp_path, _subprocess_fn=mock_runner)
    assert rc == 1  # should fail because horizon/variable/variable_horizon missing


# --------------------------------------------------------------------------- #
# Integration: dry-run on real config
# --------------------------------------------------------------------------- #

@pytest.mark.unit
def test_real_config_dry_run_produces_jobs():
    """Expand all jobs from the real config — must not raise."""
    config = load_config(CONFIG_PATH)
    jobs = expand_jobs(config, Path("/tmp/inspect_test"))
    assert len(jobs) > 0


@pytest.mark.unit
def test_real_config_all_cli_commands_have_python(tmp_path):
    config = load_config(CONFIG_PATH)
    jobs = expand_jobs(config, tmp_path)
    for job in jobs:
        assert job.cli_command[0] == sys.executable, (
            f"job {job.job_id}: cli_command[0] is not sys.executable"
        )


@pytest.mark.unit
def test_real_config_all_cli_commands_have_output_root(tmp_path):
    config = load_config(CONFIG_PATH)
    jobs = expand_jobs(config, tmp_path)
    for job in jobs:
        assert "__OUTPUT_ROOT__" in job.cli_command, (
            f"job {job.job_id}: missing __OUTPUT_ROOT__ placeholder"
        )
