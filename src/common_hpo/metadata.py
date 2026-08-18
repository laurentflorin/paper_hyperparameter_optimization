"""Common reproducibility metadata for hyperparameter-optimization runs.

This module centralizes the machine-readable metadata that every scientific run
directory should record before and after expensive work. The configuration hash
is intentionally stronger than a simple CLI hash: it includes repository state,
relevant package versions, input-file fingerprints, and the scientific
configuration so an interrupted run cannot be resumed under silently different
code, data, or dependency states.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


__all__ = [
    "DEFAULT_RELEVANT_PACKAGES",
    "OUTPUT_SCHEMA_VERSION",
    "build_run_metadata",
    "classify_failure",
    "fingerprint_input_file",
    "fingerprint_input_files",
    "package_versions",
    "repository_state",
    "sha256_file",
    "stable_configuration_hash",
    "stable_json_dumps",
    "summarize_failures",
    "utc_now",
]


DEFAULT_RELEVANT_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "requests",
    "joblib",
    "openpyxl",
    "xlsxwriter",
    "tqdm",
    "plotly",
    "seaborn",
    "arm-mango",
    "bayesian-optimization",
    "covbayesvar",
    "MBFVAR",
)
OUTPUT_SCHEMA_VERSION = "1"


def utc_now() -> str:
    """Return the current UTC time as an ISO 8601 ``Z`` timestamp."""

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stable_json_dumps(value: object) -> str:
    """Serialize *value* canonically for hashing and reproducibility."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def stable_configuration_hash(configuration: object) -> str:
    """Return a SHA-256 digest of *configuration*'s canonical JSON form."""

    return hashlib.sha256(stable_json_dumps(configuration).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for *path*."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path, *, root: Path | None) -> str:
    if root is not None:
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass
    return str(path)


def fingerprint_input_file(path: str | Path, *, root: str | Path | None = None) -> dict[str, object]:
    """Describe one input file by path, size, and SHA-256 fingerprint."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"input file does not exist: {resolved}")
    root_path = None if root is None else Path(root).expanduser().resolve()
    return {
        "path": _display_path(resolved, root=root_path),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def fingerprint_input_files(
    paths: Iterable[str | Path],
    *,
    root: str | Path | None = None,
) -> list[dict[str, object]]:
    """Return stable, de-duplicated file fingerprints for *paths*."""

    root_path = None if root is None else Path(root).expanduser().resolve()
    unique = sorted({Path(path).expanduser().resolve() for path in paths}, key=str)
    return [fingerprint_input_file(path, root=root_path) for path in unique]


def _git_output(project_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip()


def repository_state(project_root: str | Path) -> dict[str, object]:
    """Describe the repository revision and dirtiness without failing outside Git."""

    root = Path(project_root).expanduser().resolve()
    commit = _git_output(root, "rev-parse", "HEAD")
    dirty = _git_output(root, "status", "--porcelain", "--untracked-files=no")
    return {
        "repository_commit": commit,
        "repository_dirty": None if dirty is None else bool(dirty),
    }


def package_versions(names: Sequence[str] = DEFAULT_RELEVANT_PACKAGES) -> dict[str, str | None]:
    """Return installed versions for *names*, preserving the requested order."""

    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[str(name)] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[str(name)] = None
    return versions


def classify_failure(value: object) -> str:
    """Map an exception or failure record to a structured category label."""

    if isinstance(value, Mapping):
        explicit = value.get("failure_category")
        if explicit:
            return str(explicit)
        return classify_failure(value.get("error") or value.get("reason") or value.get("exception"))

    if isinstance(value, BaseException):
        if isinstance(value, KeyboardInterrupt):
            return "cancelled"
        if isinstance(value, (ImportError, ModuleNotFoundError)):
            return "dependency"
        if isinstance(value, FileNotFoundError):
            return "missing_input"
        if isinstance(value, PermissionError):
            return "permission"
        if isinstance(value, OSError):
            return "io"
        if isinstance(value, MemoryError):
            return "memory"
        if isinstance(value, TimeoutError):
            return "timeout"
        if isinstance(value, (FloatingPointError, OverflowError, ZeroDivisionError, ArithmeticError)):
            return "numerical"
        if type(value).__name__ == "LinAlgError":
            return "numerical"
        if isinstance(value, ValueError):
            return "invalid_configuration"
        return "runtime"

    if value is None:
        return "unknown"

    text = str(value).strip().lower()
    if not text:
        return "unknown"
    if "keyboardinterrupt" in text or "cancel" in text:
        return "cancelled"
    if "non-finite" in text or "nonfinite" in text:
        return "nonfinite_forecast"
    if "lin alg" in text or "linalg" in text:
        return "numerical"
    if "overflow" in text or "divide" in text or "nan" in text:
        return "numerical"
    if "missing" in text or "not found" in text:
        return "missing_input"
    if "import" in text or "module" in text or "dependency" in text:
        return "dependency"
    if "permission" in text:
        return "permission"
    if "timeout" in text:
        return "timeout"
    if "benchmark_rmse" in text or "requires" in text or "invalid" in text or "mismatch" in text:
        return "invalid_configuration"
    if "forecast" in text and "infeasible" in text:
        return "forecast_invalid"
    return "runtime"


def summarize_failures(records: Sequence[Mapping[str, object]] | None) -> dict[str, object]:
    """Aggregate failure counts by stage and category."""

    if not records:
        return {
            "total": 0,
            "by_stage": {},
            "by_category": {},
        }

    by_stage: Counter[str] = Counter()
    by_category: Counter[str] = Counter()
    for record in records:
        stage = str(record.get("stage") or "unknown")
        by_stage[stage] += 1
        by_category[classify_failure(record)] += 1

    return {
        "total": int(sum(by_category.values())),
        "by_stage": dict(sorted(by_stage.items())),
        "by_category": dict(sorted(by_category.items())),
    }


def build_run_metadata(
    *,
    project_root: str | Path,
    command_line: str,
    started_utc: str,
    finished_utc: str | None,
    completion_status: str,
    model_family: str,
    model_version: str,
    data_source: object,
    data_vintage_identifiers: Mapping[str, object] | None = None,
    input_files: Iterable[str | Path] = (),
    transformation_configuration: Mapping[str, object] | None = None,
    variable_order: Sequence[str] | None = None,
    target_variables: Sequence[str] = (),
    target_horizons: Sequence[int] = (),
    selection_plan: Mapping[str, object] | None = None,
    validation_scheme: Mapping[str, object] | None = None,
    vintage_policy: object | None = None,
    selection_schedule: Mapping[str, object] | str | None = None,
    loss_configuration: Mapping[str, object] | None = None,
    search_space: Mapping[str, object] | None = None,
    optimizer_budget: Mapping[str, object] | None = None,
    random_seeds: Mapping[str, object] | None = None,
    parallel_worker_count: int | None = None,
    output_schema_version: str = OUTPUT_SCHEMA_VERSION,
    failure_records: Sequence[Mapping[str, object]] | None = None,
    relevant_packages: Sequence[str] = DEFAULT_RELEVANT_PACKAGES,
    configuration_extra: Mapping[str, object] | None = None,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the standard run-metadata payload for one output directory."""

    project_root = Path(project_root).expanduser().resolve()
    repo = repository_state(project_root)
    packages = package_versions(relevant_packages)
    input_fingerprints = fingerprint_input_files(input_files, root=project_root)
    failures = summarize_failures(failure_records)

    configuration: dict[str, object] = {
        "repository_commit": repo["repository_commit"],
        "repository_dirty": repo["repository_dirty"],
        "package_versions": packages,
        "model_family": model_family,
        "model_version": model_version,
        "data_source": data_source,
        "data_vintage_identifiers": dict(data_vintage_identifiers or {}),
        "input_fingerprints": input_fingerprints,
        "transformation_configuration": dict(transformation_configuration or {}),
        "variable_order": list(variable_order or ()),
        "target_variables": list(target_variables),
        "target_horizons": [int(h) for h in target_horizons],
        "selection_plan": dict(selection_plan or {}),
        "validation_scheme": dict(validation_scheme or {}),
        "vintage_policy": vintage_policy,
        "selection_schedule": selection_schedule,
        "loss_configuration": dict(loss_configuration or {}),
        "search_space": dict(search_space or {}),
        "optimizer_budget": dict(optimizer_budget or {}),
        "random_seeds": dict(random_seeds or {}),
        "parallel_worker_count": parallel_worker_count,
        "output_schema_version": output_schema_version,
    }
    if configuration_extra:
        configuration.update(dict(configuration_extra))

    metadata: dict[str, object] = {
        **repo,
        "command_line": command_line,
        "utc_started_at": started_utc,
        "utc_finished_at": finished_utc,
        "python_version": sys.version,
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python_implementation": platform.python_implementation(),
        },
        "package_versions": packages,
        "model_family": model_family,
        "model_version": model_version,
        "data_source": data_source,
        "data_vintage_identifiers": dict(data_vintage_identifiers or {}),
        "input_fingerprints": input_fingerprints,
        "transformation_configuration": dict(transformation_configuration or {}),
        "variable_order": list(variable_order or ()),
        "target_variables": list(target_variables),
        "target_horizons": [int(h) for h in target_horizons],
        "selection_plan": dict(selection_plan or {}),
        "validation_scheme": dict(validation_scheme or {}),
        "vintage_policy": vintage_policy,
        "selection_schedule": selection_schedule,
        "loss_configuration": dict(loss_configuration or {}),
        "search_space": dict(search_space or {}),
        "optimizer_budget": dict(optimizer_budget or {}),
        "random_seeds": dict(random_seeds or {}),
        "parallel_worker_count": parallel_worker_count,
        "output_schema_version": output_schema_version,
        "configuration_hash": stable_configuration_hash(configuration),
        "completion_status": completion_status,
        "failure_counts": {
            "total": failures["total"],
            "by_stage": failures["by_stage"],
        },
        "failure_categories": failures["by_category"],
        "scientific_configuration": configuration,
    }
    if extra:
        metadata.update(dict(extra))
    return metadata