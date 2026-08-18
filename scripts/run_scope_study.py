"""Declarative, resumable orchestrator for the complete scope study.

Usage examples::

    # Validate config only
    python scripts/run_scope_study.py --config configs/paper_experiment.json --validate

    # Dry-run: print full job matrix with estimated counts
    python scripts/run_scope_study.py --config configs/paper_experiment.json --dry-run

    # Plan only (show what would happen without running)
    python scripts/run_scope_study.py --config configs/paper_experiment.json --plan

    # Run all enabled jobs
    python scripts/run_scope_study.py --config configs/paper_experiment.json \
        --output-root outputs/scope_study --overwrite

    # Resume incomplete/failed jobs
    python scripts/run_scope_study.py --config configs/paper_experiment.json --resume

    # Filter by family and scope
    python scripts/run_scope_study.py --config configs/paper_experiment.json \
        --filter-family ridge --filter-scope pooled,horizon --resume

    # Cluster array mode: execute exactly one job by 0-based index
    python scripts/run_scope_study.py --config configs/paper_experiment.json \
        --job-index 3 --resume

    # Smoke test: one tiny synthetic ridge job, validates full pipeline
    python scripts/run_scope_study.py --config configs/paper_experiment.json \
        --smoke-test --output-root /tmp/smoke_test
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common_hpo.io import classify_run_directory  # noqa: E402
from common_hpo.metadata import stable_configuration_hash  # noqa: E402


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

STUDY_STATUS_FILENAME = "study_status.json"
JOB_MANIFEST_FILENAME = "job_manifest.json"
JOB_LOG_FILENAME = "job.log"
SCHEMA_PATH = PROJECT_ROOT / "configs" / "paper_experiment.schema.json"

_SCOPE_HYPHEN = {"variable_horizon": "variable-horizon"}  # for GLP dir naming


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #

@dataclass
class JobSpec:
    """Complete, self-contained specification for one scope-grid runner invocation."""
    job_id: str                          # deterministic slug
    family: str                          # glp | mfvar | ridge
    variant: str                         # variant name from config
    size: str | None                     # GLP model size; None for ridge/mfvar
    forecast_method: str | None          # iterated | direct for ridge; None else
    scopes: list[str]                    # e.g. ["pooled", "horizon", "variable", "variable_horizon"]
    seed: int
    output_dir: Path                     # the --output-root for this job
    scope_dir_pattern: str               # e.g. "scope_{scope}" or "scope-{scope_hyphen}"
    cli_command: list[str]               # full argv (python + script + args, no output-root)
    scientific_config: dict[str, Any]    # serialisable; used to compute config hash
    requires: list[str]                  # optional packages needed (for skip-if-absent logic)
    configuration_hash: str = field(default="")  # computed after construction

    def __post_init__(self) -> None:
        if not self.configuration_hash:
            self.configuration_hash = stable_configuration_hash(self.scientific_config)

    def scope_output_dir(self, scope: str) -> Path:
        """Return the expected scope sub-directory inside this job's output_dir."""
        scope_hyphen = scope.replace("_", "-")
        return self.output_dir / self.scope_dir_pattern.format(
            scope=scope, scope_hyphen=scope_hyphen
        )

    def benchmark_output_dir(self, strategy: str) -> Path:
        return self.output_dir / "benchmarks" / strategy


@dataclass
class PlannedJob:
    spec: JobSpec
    action: str          # run | skip | resume | reject | skip_missing_dep
    reason: str
    prior_status: str    # complete | partial | failed | cancelled | missing | incompatible | unknown


@dataclass
class JobResult:
    spec: JobSpec
    status: str          # complete | partial | failed | cancelled | rejected | skipped
    exit_code: int | None
    elapsed_seconds: float
    log_path: Path | None
    scope_statuses: dict[str, str] = field(default_factory=dict)  # scope -> status
    error_summary: str = ""


# --------------------------------------------------------------------------- #
# Config loading and validation
# --------------------------------------------------------------------------- #

_ENV_RE = re.compile(r"\$\{(\w+)(?::-(.*?))?\}")


def _expand_env(value: object) -> object:
    """Recursively expand ${VAR:-default} placeholders in string values."""
    if isinstance(value, str):
        def _replace(m: re.Match) -> str:
            var, default = m.group(1), m.group(2) or ""
            return os.environ.get(var, default)
        return _ENV_RE.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return _expand_env(raw)  # type: ignore[return-value]


def validate_config(config: dict[str, Any]) -> list[str]:
    """Lightweight structural validation. Returns list of error strings."""
    errors: list[str] = []
    if "study" not in config:
        errors.append("missing top-level 'study' key")
    if "families" not in config or not isinstance(config["families"], list):
        errors.append("'families' must be a non-empty list")
        return errors
    seen_families: set[str] = set()
    for i, fam in enumerate(config["families"]):
        fname = fam.get("family", f"<family[{i}]>")
        if fname in seen_families:
            errors.append(f"duplicate family {fname!r}")
        seen_families.add(fname)
        if "runner" not in fam:
            errors.append(f"{fname}: missing 'runner'")
        if "variants" not in fam or not fam["variants"]:
            errors.append(f"{fname}: 'variants' must be a non-empty list")
        for j, var in enumerate(fam.get("variants", [])):
            vname = var.get("name", f"<variant[{j}]>")
            if "name" not in var:
                errors.append(f"{fname}.variants[{j}]: missing 'name'")
    if "seeds" not in config or not config["seeds"]:
        errors.append("'seeds' must be a non-empty list of integers")
    return errors


# --------------------------------------------------------------------------- #
# Job expansion
# --------------------------------------------------------------------------- #

def _int_or_env(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _str_list(items: Any) -> list[str]:
    if items is None:
        return []
    return [str(x) for x in items]


def _build_glp_command(
    runner: str,
    fam: dict[str, Any],
    var: dict[str, Any],
    size: str,
    seed: int,
    nested_workers: int,
) -> list[str]:
    argv = [sys.executable, runner]
    argv += ["--output-root", "__OUTPUT_ROOT__"]  # placeholder replaced by orchestrator
    argv += ["--panel-path", fam["panel_path"]]
    argv += ["--model-size", size]
    oo = fam.get("outer_origins", {})
    if oo.get("start"):
        argv += ["--start", str(oo["start"])]
    if oo.get("end"):
        argv += ["--end", str(oo["end"])]
    argv += ["--selection-scopes", ",".join(fam["scopes"])]
    if fam.get("target_variables"):
        argv += ["--target-variables", ",".join(fam["target_variables"])]
    if fam.get("target_horizons"):
        argv += ["--target-horizons", ",".join(str(h) for h in fam["target_horizons"])]
    argv += ["--loss-metric", var.get("loss_metric", "rmse")]
    argv += ["--loss-scaling", var.get("loss_scaling", "none")]
    if var.get("benchmark"):
        argv += ["--benchmark", var["benchmark"]]
    iv = var.get("inner_validation", {})
    if iv.get("window"):
        argv += ["--inner-window", iv["window"]]
    if iv.get("n_origins"):
        argv += ["--inner-n-origins", str(iv["n_origins"])]
    if iv.get("stride"):
        argv += ["--inner-origin-stride", str(iv["stride"])]
    if iv.get("origin_selection"):
        argv += ["--inner-origin-selection", iv["origin_selection"]]
    argv += ["--selection-frequency", str(var.get("selection_schedule", "once"))]
    budget = var.get("optimizer_budget", {})
    if budget.get("init_points"):
        argv += ["--optimization-init-points", str(budget["init_points"])]
    if budget.get("iterations"):
        argv += ["--optimization-iterations", str(budget["iterations"])]
    if budget.get("posterior_draws"):
        argv += ["--objective-posterior-draws", str(budget["posterior_draws"])]
    optimizer_seed = _int_or_env(budget.get("optimizer_seed"), seed)
    argv += ["--optimizer-seed", str(optimizer_seed)]
    if var.get("optimize_psi"):
        argv += ["--optimize-psi"]
    else:
        argv += ["--no-optimize-psi"]
    if var.get("fixed_psi_source"):
        argv += ["--fixed-psi-source", var["fixed_psi_source"]]
    exec_cfg = fam.get("execution", {})
    wc = _int_or_env(exec_cfg.get("worker_count"), nested_workers)
    wc = min(wc, nested_workers) if nested_workers > 0 else 1
    if wc > 1:
        argv += ["--execution-mode", "parallel", "--worker-count", str(wc)]
    else:
        argv += ["--execution-mode", "serial"]
    return argv


def _build_mfvar_command(
    runner: str,
    fam: dict[str, Any],
    var: dict[str, Any],
    seed: int,
) -> list[str]:
    argv = [sys.executable, runner]
    argv += ["--output-root", "__OUTPUT_ROOT__"]
    argv += ["--panel-path", fam["panel_path"]]
    if fam.get("forecast_variables"):
        argv += ["--forecast-variables", ",".join(fam["forecast_variables"])]
    argv += ["--selection-scopes", ",".join(fam["scopes"])]
    if fam.get("target_variables"):
        argv += ["--target-variables", ",".join(fam["target_variables"])]
    if fam.get("target_horizons"):
        argv += ["--target-horizons", ",".join(str(h) for h in fam["target_horizons"])]
    argv += ["--loss-metric", var.get("loss_metric", "rmse")]
    argv += ["--loss-scaling", var.get("loss_scaling", "none")]
    iv = var.get("inner_validation", {})
    if iv.get("window"):
        argv += ["--inner-window", iv["window"]]
    if iv.get("n_origins"):
        argv += ["--inner-n-origins", str(iv["n_origins"])]
    if iv.get("stride"):
        argv += ["--inner-origin-stride", str(iv["stride"])]
    argv += ["--selection-frequency", str(var.get("selection_schedule", "once"))]
    budget = var.get("optimizer_budget", {})
    if budget.get("optimization_horizon_quarters"):
        argv += ["--optimization-horizon-quarters", str(budget["optimization_horizon_quarters"])]
    if budget.get("optimization_eval_horizon_quarters"):
        argv += ["--optimization-eval-horizon-quarters", str(budget["optimization_eval_horizon_quarters"])]
    if budget.get("optimization_n_eval"):
        argv += ["--optimization-n-eval", str(budget["optimization_n_eval"])]
    base_seed = _int_or_env(var.get("base_seed"), seed)
    argv += ["--base-seed", str(base_seed)]
    return argv


def _build_ridge_command(
    runner: str,
    fam: dict[str, Any],
    var: dict[str, Any],
    forecast_method: str,
    seed: int,
) -> list[str]:
    argv = [sys.executable, runner]
    argv += ["--output-root", "__OUTPUT_ROOT__"]
    argv += ["--panel-path", fam["panel_path"]]
    argv += ["--target-variables", ",".join(fam["target_variables"])]
    argv += ["--target-horizons", ",".join(str(h) for h in fam["target_horizons"])]
    argv += ["--selection-scopes", ",".join(fam["scopes"])]
    argv += ["--forecast-method", forecast_method]
    grid = var.get("grid", {})
    if grid.get("lambdas"):
        argv += ["--grid-lambdas", ",".join(str(v) for v in grid["lambdas"])]
    if grid.get("lag_orders"):
        argv += ["--grid-lag-orders", ",".join(str(v) for v in grid["lag_orders"])]
    if grid.get("alphas"):
        argv += ["--grid-alphas", ",".join(str(v) for v in grid["alphas"])]
    if grid.get("kappas"):
        argv += ["--grid-kappas", ",".join(str(v) for v in grid["kappas"])]
    if var.get("preprocessing"):
        argv += ["--preprocessing", var["preprocessing"]]
    argv += ["--loss-metric", var.get("loss_metric", "rmse")]
    argv += ["--loss-scaling", var.get("loss_scaling", "none")]
    iv = var.get("inner_validation", {})
    if iv.get("window"):
        argv += ["--inner-window", iv["window"]]
    if iv.get("n_origins"):
        argv += ["--inner-n-origins", str(iv["n_origins"])]
    if iv.get("stride"):
        argv += ["--inner-origin-stride", str(iv["stride"])]
    if iv.get("origin_selection"):
        argv += ["--inner-origin-selection", iv["origin_selection"]]
    argv += ["--selection-frequency", str(var.get("selection_schedule", "once"))]
    oo = fam.get("outer_origins", {})
    if oo.get("n_origins"):
        argv += ["--outer-n-origins", str(oo["n_origins"])]
    if oo.get("stride"):
        argv += ["--outer-origin-stride", str(oo["stride"])]
    if oo.get("origin_selection"):
        argv += ["--outer-origin-selection", str(oo["origin_selection"])]
    benchmarks = var.get("benchmarks", [])
    if benchmarks:
        argv += ["--benchmarks", ",".join(benchmarks)]
    base_seed = _int_or_env(var.get("base_seed"), seed)
    argv += ["--base-seed", str(base_seed)]
    return argv


def _make_job_id(family: str, variant: str, size: str | None, method: str | None, seed: int) -> str:
    parts = [family]
    if size:
        parts.append(size)
    if method:
        parts.append(method)
    parts.append(variant)
    parts.append(str(seed))
    return "_".join(parts)


def _scientific_config(
    family: str,
    variant: dict[str, Any],
    size: str | None,
    method: str | None,
    fam: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Extract the scientifically relevant subset of the config for hashing."""
    return {
        "family": family,
        "variant": variant["name"],
        "size": size,
        "forecast_method": method,
        "scopes": sorted(fam.get("scopes", [])),
        "target_variables": sorted(_str_list(fam.get("target_variables"))),
        "target_horizons": sorted(fam.get("target_horizons", [])),
        "outer_origins": fam.get("outer_origins", {}),
        "seed": seed,
        "variant_params": {
            k: v for k, v in variant.items()
            if k not in ("name", "label", "enabled")
        },
    }


def expand_jobs(config: dict[str, Any], output_root: Path) -> list[JobSpec]:
    """Expand the declarative config into a flat list of JobSpec objects."""
    jobs: list[JobSpec] = []
    seeds = config.get("seeds", [config["study"]["seed_base"]])
    nested_workers = _int_or_env(
        config.get("parallelism", {}).get("max_nested_workers"), 1
    )

    for fam in config["families"]:
        if not fam.get("enabled", True):
            continue
        family = fam["family"]
        runner = str(PROJECT_ROOT / fam["runner"])
        scope_pattern = fam.get("scope_dir_pattern", "scope_{scope}")
        requires = fam.get("requires", [])
        sizes = fam.get("sizes", [None])
        methods = fam.get("forecast_methods", [None])

        for seed in seeds:
            for var in fam.get("variants", []):
                if not var.get("enabled", True):
                    continue
                for size in sizes:
                    for method in methods:
                        job_id = _make_job_id(family, var["name"], size, method, seed)
                        sci = _scientific_config(family, var, size, method, fam, seed)
                        job_dir = _job_output_dir(output_root, family, var["name"], size, method, seed)

                        if family == "glp":
                            cli = _build_glp_command(runner, fam, var, size or "small", seed, nested_workers)
                        elif family == "mfvar":
                            cli = _build_mfvar_command(runner, fam, var, seed)
                        else:
                            cli = _build_ridge_command(runner, fam, var, method or "iterated", seed)

                        jobs.append(JobSpec(
                            job_id=job_id,
                            family=family,
                            variant=var["name"],
                            size=size,
                            forecast_method=method,
                            scopes=list(fam.get("scopes", [])),
                            seed=seed,
                            output_dir=job_dir,
                            scope_dir_pattern=scope_pattern,
                            cli_command=cli,
                            scientific_config=sci,
                            requires=requires,
                        ))
    return jobs


def _job_output_dir(
    output_root: Path,
    family: str,
    variant: str,
    size: str | None,
    method: str | None,
    seed: int,
) -> Path:
    parts = [family]
    if size:
        parts.append(size)
    if method:
        parts.append(method)
    parts.append(variant)
    if seed != 20150101:
        parts.append(f"seed{seed}")
    return output_root / "jobs" / "_".join(parts)


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #

def _check_required_packages(requires: list[str]) -> list[str]:
    missing = []
    for pkg in requires:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


def _read_job_manifest(job_dir: Path) -> dict[str, Any] | None:
    manifest_path = job_dir / JOB_MANIFEST_FILENAME
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _overall_job_status(spec: JobSpec) -> str:
    """Determine the overall status of a job by inspecting its scope directories."""
    if not spec.output_dir.exists():
        return "missing"
    scope_statuses = []
    for scope in spec.scopes:
        sd = spec.scope_output_dir(scope)
        if not sd.exists():
            scope_statuses.append("missing")
        else:
            state = classify_run_directory(sd)
            scope_statuses.append(state.status)

    if not scope_statuses or all(s == "missing" for s in scope_statuses):
        return "missing"
    if all(s == "complete" for s in scope_statuses):
        return "complete"
    if any(s == "failed" for s in scope_statuses):
        return "failed"
    if any(s in {"partial", "cancelled"} for s in scope_statuses):
        return "partial"
    if any(s == "complete" for s in scope_statuses):
        return "partial"
    return "missing"


def plan_job(spec: JobSpec, if_exists_policy: str) -> PlannedJob:
    """Decide what to do with a single job."""
    missing_deps = _check_required_packages(spec.requires)
    if missing_deps:
        return PlannedJob(spec, "skip_missing_dep",
                          f"missing optional package(s): {missing_deps}", "missing")

    prior = _overall_job_status(spec)

    if prior == "missing":
        return PlannedJob(spec, "run", "no prior run found", "missing")

    manifest = _read_job_manifest(spec.output_dir)
    stored_hash = (manifest or {}).get("configuration_hash")

    if stored_hash and stored_hash != spec.configuration_hash:
        return PlannedJob(spec, "reject",
                          f"configuration hash mismatch (stored={stored_hash[:12]}, "
                          f"current={spec.configuration_hash[:12]})",
                          "incompatible")

    if prior == "complete":
        if if_exists_policy == "overwrite":
            return PlannedJob(spec, "run", "overwrite requested", "complete")
        return PlannedJob(spec, "skip", "all scopes already complete", "complete")

    if prior in {"partial", "cancelled"}:
        if if_exists_policy in {"resume", "overwrite"}:
            return PlannedJob(spec, "resume",
                              f"prior status={prior}, hash compatible", prior)
        return PlannedJob(spec, "reject",
                          f"prior status={prior}; pass --resume or --overwrite to continue", prior)

    if prior == "failed":
        # Never silently treat failed as missing
        if if_exists_policy in {"resume", "overwrite"}:
            return PlannedJob(spec, "resume",
                              f"prior status=failed, hash compatible", "failed")
        return PlannedJob(spec, "reject",
                          "prior run failed; pass --resume or --overwrite to retry", "failed")

    return PlannedJob(spec, "run", f"prior status={prior}", prior)


def plan_jobs(
    specs: list[JobSpec],
    if_exists_policy: str,
    filter_family: list[str] | None = None,
    filter_scope: list[str] | None = None,
    filter_variant: list[str] | None = None,
    filter_status: list[str] | None = None,
    job_index: int | None = None,
) -> list[PlannedJob]:
    filtered = specs
    if filter_family:
        filtered = [s for s in filtered if s.family in filter_family]
    if filter_scope:
        filtered = [s for s in filtered if any(sc in s.scopes for sc in filter_scope)]
    if filter_variant:
        filtered = [s for s in filtered if s.variant in filter_variant]
    planned = [plan_job(s, if_exists_policy) for s in filtered]
    if filter_status:
        planned = [p for p in planned if p.prior_status in filter_status or p.action in filter_status]
    if job_index is not None:
        if 0 <= job_index < len(planned):
            planned = [planned[job_index]]
        else:
            raise IndexError(f"--job-index {job_index} out of range [0, {len(planned)-1}]")
    return planned


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #

def _write_job_manifest(spec: JobSpec, status: str) -> None:
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "job_id": spec.job_id,
        "configuration_hash": spec.configuration_hash,
        "family": spec.family,
        "variant": spec.variant,
        "size": spec.size,
        "forecast_method": spec.forecast_method,
        "scopes": spec.scopes,
        "seed": spec.seed,
        "status": status,
        "utc_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    manifest_path = spec.output_dir / JOB_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def execute_job(
    planned: PlannedJob,
    log_root: Path,
    *,
    _subprocess_fn=None,
) -> JobResult:
    """Run one planned job, capturing stdout+stderr to a log file."""
    spec = planned.spec
    if planned.action in ("skip", "skip_missing_dep"):
        return JobResult(spec, "skipped", None, 0.0, None,
                         error_summary=planned.reason)
    if planned.action == "reject":
        return JobResult(spec, "rejected", None, 0.0, None,
                         error_summary=planned.reason)

    # Build actual command with output-root substituted
    cmd = [
        (str(spec.output_dir) if tok == "__OUTPUT_ROOT__" else tok)
        for tok in spec.cli_command
    ]
    # Add resume/overwrite flag
    if planned.action == "resume":
        cmd.append("--resume")
    else:
        cmd.append("--overwrite")

    # Prepare log file
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{spec.job_id}.log"

    # Write pre-run manifest
    _write_job_manifest(spec, "running")

    t0 = time.monotonic()
    exit_code = -1
    try:
        fn = _subprocess_fn or subprocess.run
        result = fn(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        exit_code = result.returncode
        log_content = (
            f"# job_id: {spec.job_id}\n"
            f"# command: {shlex.join(cmd)}\n"
            f"# started: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
            f"# exit_code: {exit_code}\n\n"
        ) + (result.stdout or "")
        log_path.write_text(log_content, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        log_path.write_text(f"# orchestrator error: {exc}\n", encoding="utf-8")
        _write_job_manifest(spec, "failed")
        return JobResult(spec, "failed", -1, time.monotonic() - t0, log_path,
                         error_summary=str(exc))

    elapsed = time.monotonic() - t0

    # Determine final status from scope directories
    scope_statuses = {}
    for scope in spec.scopes:
        sd = spec.scope_output_dir(scope)
        if sd.exists():
            state = classify_run_directory(sd)
            scope_statuses[scope] = state.status
        else:
            scope_statuses[scope] = "missing"

    if exit_code == 0 and all(s == "complete" for s in scope_statuses.values()):
        final_status = "complete"
    elif exit_code != 0:
        final_status = "failed"
    elif any(s == "failed" for s in scope_statuses.values()):
        final_status = "failed"
    else:
        final_status = "partial"

    _write_job_manifest(spec, final_status)

    return JobResult(
        spec=spec,
        status=final_status,
        exit_code=exit_code,
        elapsed_seconds=elapsed,
        log_path=log_path,
        scope_statuses=scope_statuses,
        error_summary="" if exit_code == 0 else f"exit_code={exit_code}",
    )


# --------------------------------------------------------------------------- #
# Study status
# --------------------------------------------------------------------------- #

def write_study_status(results: list[JobResult], output_root: Path) -> None:
    from collections import Counter
    summary: Counter = Counter()
    job_records = []
    for r in results:
        summary[r.status] += 1
        job_records.append({
            "job_id": r.spec.job_id,
            "family": r.spec.family,
            "variant": r.spec.variant,
            "size": r.spec.size,
            "forecast_method": r.spec.forecast_method,
            "seed": r.spec.seed,
            "status": r.status,
            "exit_code": r.exit_code,
            "elapsed_seconds": round(r.elapsed_seconds, 1),
            "scopes_complete": sum(1 for s in r.scope_statuses.values() if s == "complete"),
            "scopes_total": len(r.spec.scopes),
            "configuration_hash": r.spec.configuration_hash[:16],
            "log": str(r.log_path) if r.log_path else None,
            "error_summary": r.error_summary or None,
        })
    status_path = output_root / STUDY_STATUS_FILENAME
    output_root.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps({
            "utc_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "summary": dict(summary),
            "jobs": job_records,
        }, indent=2),
        encoding="utf-8",
    )


def print_study_summary(results: list[JobResult]) -> None:
    from collections import Counter
    counts: Counter = Counter(r.status for r in results)
    total = len(results)
    print(f"\nStudy summary ({total} jobs):")
    for status in ("complete", "partial", "failed", "rejected", "skipped"):
        n = counts.get(status, 0)
        if n:
            print(f"  {status:12s}: {n}")
    rejected = [r for r in results if r.status == "rejected"]
    failed = [r for r in results if r.status == "failed"]
    if rejected:
        print(f"\nRejected jobs ({len(rejected)}):")
        for r in rejected:
            print(f"  {r.spec.job_id}: {r.error_summary}")
    if failed:
        print(f"\nFailed jobs ({len(failed)}):")
        for r in failed:
            print(f"  {r.spec.job_id}: {r.error_summary}")


# --------------------------------------------------------------------------- #
# Dry-run reporting
# --------------------------------------------------------------------------- #

def print_dry_run(jobs: list[JobSpec]) -> None:
    print(f"\nDry-run job matrix ({len(jobs)} jobs):\n")
    enabled = [j for j in jobs if True]
    print(f"{'#':>4}  {'job_id':<52}  {'family':<8}  {'scopes':>6}  {'hash[:12]'}")
    print("-" * 100)
    for i, j in enumerate(enabled):
        print(f"{i:>4}  {j.job_id:<52}  {j.family:<8}  {len(j.scopes):>6}  {j.configuration_hash[:12]}")
    print()
    # Count by family
    from collections import Counter
    family_counts: Counter = Counter(j.family for j in enabled)
    for fam, n in sorted(family_counts.items()):
        print(f"  {fam}: {n} jobs")


def print_plan(planned: list[PlannedJob]) -> None:
    print(f"\nPlan ({len(planned)} jobs):\n")
    print(f"{'action':<18}  {'job_id':<52}  reason")
    print("-" * 100)
    for p in planned:
        print(f"{p.action:<18}  {p.spec.job_id:<52}  {p.reason}")


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #

def run_smoke_test(output_root: Path, *, _subprocess_fn=None) -> int:
    """Run one tiny synthetic ridge job to validate the full pipeline."""
    import numpy as np
    import pandas as pd

    print("\n=== Smoke test: synthetic ridge integration ===")
    smoke_dir = output_root / "smoke_test"
    panel_path = smoke_dir / "panel.csv"
    smoke_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic panel
    rng = np.random.default_rng(42)
    T = 60
    A = np.array([[0.6, 0.1, 0.0], [-0.1, 0.5, 0.1], [0.0, 0.1, 0.4]])
    c = np.array([0.2, -0.1, 0.1])
    y = np.zeros((T, 3))
    for t in range(1, T):
        y[t] = c + A @ y[t - 1] + rng.normal(scale=0.15, size=3)
    pd.DataFrame(y, columns=["gdp", "inv", "cons"]).to_csv(panel_path, index=False)

    runner = str(PROJECT_ROOT / "scripts" / "regularized_var" / "run_ridge_scope_grid.py")
    job_dir = smoke_dir / "ridge_iterated_forecast_loss"
    cmd = [
        sys.executable, runner,
        "--output-root", str(job_dir),
        "--panel-path", str(panel_path),
        "--target-variables", "gdp,inv,cons",
        "--target-horizons", "1,2,4,8",
        "--selection-scopes", "pooled,horizon,variable,variable_horizon",
        "--forecast-method", "iterated",
        "--grid-lambdas", "0.01,0.1",
        "--grid-lag-orders", "1,2",
        "--grid-alphas", "0.0",
        "--grid-kappas", "1.0",
        "--outer-n-origins", "4",
        "--inner-n-origins", "3",
        "--min-train-length", "20",
        "--selection-frequency", "once",
        "--overwrite",
    ]

    print(f"Command: {shlex.join(cmd)}\n")
    t0 = time.monotonic()
    fn = _subprocess_fn or subprocess.run
    result = fn(cmd, text=True, capture_output=False)
    elapsed = time.monotonic() - t0
    print(f"\nElapsed: {elapsed:.1f}s  exit_code={result.returncode}")

    if result.returncode != 0:
        print("SMOKE TEST FAILED: non-zero exit code")
        return 1

    # Validate outputs
    import pandas as _pd

    errors: list[str] = []
    for scope in ("pooled", "horizon", "variable", "variable_horizon"):
        scope_dir = job_dir / f"scope_{scope}"
        state = classify_run_directory(scope_dir)
        if state.status != "complete":
            errors.append(f"scope_{scope}: status={state.status}, expected complete")
        else:
            fp = _pd.read_csv(scope_dir / "forecast_panel.csv")
            if fp.duplicated(subset=["forecast_origin", "variable", "horizon_quarters"]).any():
                errors.append(f"scope_{scope}: duplicate forecast rows")
            if fp["mean_metric"].isna().any():
                errors.append(f"scope_{scope}: NaN forecasts")

    if errors:
        print("SMOKE TEST FAILED:")
        for e in errors:
            print(f"  ✗  {e}")
        return 1

    print("\n✓  SMOKE TEST PASSED — synthetic ridge pipeline validated")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Declarative scope-study orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "paper_experiment.json"),
                   help="Path to the experiment JSON config.")
    p.add_argument("--output-root",
                   help="Override study output root (overrides config/env).")
    p.add_argument("--validate", action="store_true",
                   help="Validate config then exit.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print full job matrix and exit without running.")
    p.add_argument("--plan", action="store_true",
                   help="Show what the orchestrator would do, then exit.")
    p.add_argument("--smoke-test", action="store_true",
                   help="Run one tiny synthetic ridge job to validate the pipeline.")
    p.add_argument("--resume", action="store_true",
                   help="Resume partial/failed jobs that are hash-compatible.")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite all existing outputs unconditionally.")
    p.add_argument("--filter-family", default=None,
                   help="Comma-separated list of families to include.")
    p.add_argument("--filter-scope", default=None,
                   help="Comma-separated list of scopes to include.")
    p.add_argument("--filter-variant", default=None,
                   help="Comma-separated list of variant names to include.")
    p.add_argument("--filter-status", default=None,
                   help="Comma-separated list of prior statuses to include.")
    p.add_argument("--job-index", type=int, default=None,
                   help="Run exactly one job by 0-based index (cluster-array mode).")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    config = load_config(args.config)
    errors = validate_config(config)
    if errors:
        for e in errors:
            print(f"error: config error: {e}", file=sys.stderr)
        return 1
    if args.validate:
        print("Config valid.")
        return 0

    output_root = Path(args.output_root) if args.output_root else Path(config["study"]["output_root"])

    if args.smoke_test:
        return run_smoke_test(output_root)

    jobs = expand_jobs(config, output_root)

    if args.dry_run:
        print_dry_run(jobs)
        return 0

    if_exists_policy = "error"
    if args.overwrite:
        if_exists_policy = "overwrite"
    elif args.resume:
        if_exists_policy = "resume"

    planned = plan_jobs(
        jobs,
        if_exists_policy,
        filter_family=[f.strip() for f in args.filter_family.split(",")] if args.filter_family else None,
        filter_scope=[s.strip() for s in args.filter_scope.split(",")] if args.filter_scope else None,
        filter_variant=[v.strip() for v in args.filter_variant.split(",")] if args.filter_variant else None,
        filter_status=[s.strip() for s in args.filter_status.split(",")] if args.filter_status else None,
        job_index=args.job_index,
    )

    if args.plan:
        print_plan(planned)
        return 0

    log_root = Path(config["study"].get("log_dir", str(output_root / "logs")))
    results: list[JobResult] = []

    runnable = [p for p in planned if p.action in ("run", "resume")]
    skipped = [p for p in planned if p.action not in ("run", "resume")]

    for p in skipped:
        results.append(JobResult(
            p.spec,
            "skipped" if p.action in ("skip", "skip_missing_dep") else "rejected",
            None, 0.0, None, error_summary=p.reason,
        ))

    print(f"Running {len(runnable)} job(s), skipping {len(skipped)}...")
    for p in runnable:
        print(f"\n[job] {p.spec.job_id}  action={p.action}")
        r = execute_job(p, log_root)
        results.append(r)
        print(f"[done] {r.spec.job_id}  status={r.status}  elapsed={r.elapsed_seconds:.1f}s")

    write_study_status(results, output_root)
    print_study_summary(results)

    failed_or_rejected = [r for r in results if r.status in ("failed", "rejected")]
    return 1 if failed_or_rejected else 0


if __name__ == "__main__":
    sys.exit(main())
