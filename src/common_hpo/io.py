"""Atomic output writers and run-directory lifecycle helpers.

The shared run state is intentionally explicit:

* ``run_manifest.json`` is written before expensive work starts;
* ``run_status.json`` tracks ``partial``, ``failed``, ``cancelled``, or
  ``complete``;
* ``run_complete.json`` is written only after all required outputs validate.

New run discovery logic may therefore reject directories that contain metadata
but no completion marker, while legacy directories without the new state files
continue to fall back to the historical file-presence checks.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence
from uuid import uuid4

from .metadata import stable_json_dumps, utc_now


__all__ = [
    "CSVSchema",
    "JSONSchema",
    "OutputValidationError",
    "ResumeRejectedError",
    "RUN_COMPLETE_FILENAME",
    "RUN_MANIFEST_FILENAME",
    "RUN_STATUS_FILENAME",
    "RunDirectoryState",
    "atomic_write_csv_rows",
    "atomic_write_dataframe_csv",
    "atomic_write_json",
    "atomic_write_text",
    "classify_run_directory",
    "mark_run_cancelled",
    "mark_run_complete",
    "mark_run_failed",
    "prepare_run_directory",
    "resolve_run_directory_policy",
    "validate_csv_schema",
    "validate_json_schema",
    "validate_outputs",
]


RUN_MANIFEST_FILENAME = "run_manifest.json"
RUN_STATUS_FILENAME = "run_status.json"
RUN_COMPLETE_FILENAME = "run_complete.json"
_KNOWN_STATES = {"missing", "empty", "partial", "failed", "cancelled", "complete", "legacy_complete", "unknown"}


class ResumeRejectedError(RuntimeError):
    """Raised when a requested resume is not scientifically compatible."""


class OutputValidationError(RuntimeError):
    """Raised when a required output fails schema validation."""


@dataclass(frozen=True)
class CSVSchema:
    path: str | Path
    required_columns: tuple[str, ...]
    min_rows: int = 0


@dataclass(frozen=True)
class JSONSchema:
    path: str | Path
    required_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunDirectoryState:
    status: str
    configuration_hash: str | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.status not in _KNOWN_STATES:
            raise ValueError(f"unknown run-directory status {self.status!r}")


def _temp_path(path: Path) -> Path:
    return path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid4().hex}"


def atomic_write_text(path: str | Path, content: str) -> None:
    """Atomically replace *path* with *content*."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temp_path(target)
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    """Atomically replace *path* with an indented JSON object."""

    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    atomic_write_text(path, text + "\n")


def atomic_write_dataframe_csv(path: str | Path, frame, *, index: bool = False) -> None:
    """Atomically replace *path* with a DataFrame CSV export."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temp_path(target)
    try:
        frame.to_csv(temp_path, index=index)
        os.replace(temp_path, target)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_write_csv_rows(
    path: str | Path,
    columns: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Atomically replace *path* with CSV rows written by column contract."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temp_path(target)
    try:
        with temp_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns))
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in columns})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _read_json_object(path: Path) -> dict[str, object] | None:
    if not path.exists() or path.stat().st_size <= 1:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def classify_run_directory(path: str | Path) -> RunDirectoryState:
    """Classify a run directory under the shared state-machine contract."""

    directory = Path(path)
    if not directory.exists():
        return RunDirectoryState(status="missing")
    if not directory.is_dir():
        raise NotADirectoryError(f"run directory path is not a directory: {directory}")
    if not list(directory.iterdir()):
        return RunDirectoryState(status="empty")

    manifest = _read_json_object(directory / RUN_MANIFEST_FILENAME)
    status_payload = _read_json_object(directory / RUN_STATUS_FILENAME)
    completion_payload = _read_json_object(directory / RUN_COMPLETE_FILENAME)
    configuration_hash = None
    for payload in (completion_payload, status_payload, manifest):
        if isinstance(payload, dict) and payload.get("configuration_hash"):
            configuration_hash = str(payload["configuration_hash"])
            break

    if completion_payload is not None:
        state = str(completion_payload.get("status") or "complete")
        return RunDirectoryState(
            status="complete" if state == "complete" else state,
            configuration_hash=configuration_hash,
            reason="completion marker present",
        )

    if status_payload is not None:
        state = str(status_payload.get("status") or "partial")
        return RunDirectoryState(
            status=state if state in _KNOWN_STATES else "partial",
            configuration_hash=configuration_hash,
            reason="status file present without completion marker",
        )

    if manifest is not None:
        return RunDirectoryState(
            status="partial",
            configuration_hash=configuration_hash,
            reason="manifest present without status/completion marker",
        )

    forecast_path = directory / "forecast_panel.csv"
    metadata_path = directory / "run_metadata.json"
    if forecast_path.exists() and metadata_path.exists():
        if forecast_path.stat().st_size > 1 and metadata_path.stat().st_size > 1:
            return RunDirectoryState(status="legacy_complete", reason="legacy file-presence completion")
        return RunDirectoryState(status="partial", reason="legacy run has empty required files")
    return RunDirectoryState(status="unknown", reason="directory is non-empty but has no recognized markers")


def resolve_run_directory_policy(
    output_dir: str | Path,
    *,
    if_exists_policy: str,
    configuration_hash: str,
) -> str:
    """Resolve whether a run directory should be planned, skipped, or refused."""

    state = classify_run_directory(output_dir)
    if if_exists_policy == "overwrite":
        return "planned"

    if if_exists_policy == "resume":
        if state.status == "complete":
            if state.configuration_hash != configuration_hash:
                raise ResumeRejectedError(
                    f"run directory {output_dir} is complete but its configuration hash differs."
                )
            return "resume_skip"
        if state.status in {"missing", "empty"}:
            return "planned"
        if state.status == "legacy_complete":
            raise ResumeRejectedError(
                f"run directory {output_dir} predates configuration hashes; safe resume is impossible."
            )
        if state.status in {"partial", "failed", "cancelled"}:
            if state.configuration_hash is None:
                raise ResumeRejectedError(
                    f"run directory {output_dir} lacks a configuration hash; safe resume is impossible."
                )
            if state.configuration_hash != configuration_hash:
                raise ResumeRejectedError(
                    f"run directory {output_dir} has configuration hash {state.configuration_hash}, "
                    f"expected {configuration_hash}."
                )
            return "planned"
        raise FileExistsError(f"run directory {output_dir} is non-empty and cannot be resumed safely.")

    if state.status in {"missing", "empty"}:
        return "planned"
    raise FileExistsError(
        f"run directory {output_dir} already exists and is not empty. "
        "Use --resume or --overwrite explicitly."
    )


def prepare_run_directory(
    output_dir: str | Path,
    *,
    manifest: Mapping[str, object],
    if_exists_policy: str,
    expected_outputs: Sequence[str],
) -> str:
    """Write pre-run manifest/state and return ``planned`` or ``resume_skip``."""

    if "configuration_hash" not in manifest:
        raise ValueError("manifest must contain configuration_hash")
    output_dir = Path(output_dir)
    action = resolve_run_directory_policy(
        output_dir,
        if_exists_policy=if_exists_policy,
        configuration_hash=str(manifest["configuration_hash"]),
    )
    if action == "resume_skip":
        return action

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / RUN_MANIFEST_FILENAME, dict(manifest))
    status_payload = {
        "status": "partial",
        "configuration_hash": manifest["configuration_hash"],
        "expected_outputs": list(expected_outputs),
        "utc_started_at": manifest.get("utc_started_at"),
        "utc_updated_at": utc_now(),
    }
    atomic_write_json(output_dir / RUN_STATUS_FILENAME, status_payload)
    complete_path = output_dir / RUN_COMPLETE_FILENAME
    if complete_path.exists():
        complete_path.unlink()
    return action


def validate_csv_schema(path: str | Path, spec: CSVSchema) -> None:
    """Validate header presence and a minimum row count for a CSV file."""

    file_path = Path(path)
    if not file_path.exists():
        raise OutputValidationError(f"required CSV output is missing: {file_path}")
    with file_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise OutputValidationError(f"CSV output is empty: {file_path}") from exc
        missing = [column for column in spec.required_columns if column not in header]
        if missing:
            raise OutputValidationError(
                f"CSV output {file_path} is missing required columns: {missing}."
            )
        n_rows = sum(1 for _ in reader)
    if n_rows < int(spec.min_rows):
        raise OutputValidationError(
            f"CSV output {file_path} has {n_rows} data rows; expected at least {spec.min_rows}."
        )


def validate_json_schema(path: str | Path, spec: JSONSchema) -> None:
    """Validate that a JSON file exists and contains the required top-level keys."""

    file_path = Path(path)
    if not file_path.exists():
        raise OutputValidationError(f"required JSON output is missing: {file_path}")
    payload = _read_json_object(file_path)
    if payload is None:
        raise OutputValidationError(f"JSON output is empty or invalid: {file_path}")
    missing = [key for key in spec.required_keys if key not in payload]
    if missing:
        raise OutputValidationError(
            f"JSON output {file_path} is missing required keys: {missing}."
        )


def validate_outputs(
    output_dir: str | Path,
    *,
    csv_schemas: Sequence[CSVSchema] = (),
    json_schemas: Sequence[JSONSchema] = (),
) -> None:
    """Validate every required output beneath *output_dir*."""

    output_dir = Path(output_dir)
    for spec in csv_schemas:
        validate_csv_schema(output_dir / Path(spec.path), spec)
    for spec in json_schemas:
        validate_json_schema(output_dir / Path(spec.path), spec)


def _update_status(output_dir: Path, *, status: str, payload: Mapping[str, object] | None = None) -> None:
    base = _read_json_object(output_dir / RUN_STATUS_FILENAME) or {}
    base.update(dict(payload or {}))
    base["status"] = status
    base["utc_updated_at"] = utc_now()
    atomic_write_json(output_dir / RUN_STATUS_FILENAME, base)


def mark_run_failed(
    output_dir: str | Path,
    *,
    configuration_hash: str,
    reason: str,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Record a failed run state and optionally persist failure metadata."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if metadata is not None:
        atomic_write_json(output_dir / "run_metadata.json", dict(metadata))
    _update_status(
        output_dir,
        status="failed",
        payload={"configuration_hash": configuration_hash, "failure_reason": reason},
    )
    complete_path = output_dir / RUN_COMPLETE_FILENAME
    if complete_path.exists():
        complete_path.unlink()


def mark_run_cancelled(
    output_dir: str | Path,
    *,
    configuration_hash: str,
    reason: str = "cancelled",
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Record a cancelled run state and optionally persist cancellation metadata."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if metadata is not None:
        atomic_write_json(output_dir / "run_metadata.json", dict(metadata))
    _update_status(
        output_dir,
        status="cancelled",
        payload={"configuration_hash": configuration_hash, "failure_reason": reason},
    )
    complete_path = output_dir / RUN_COMPLETE_FILENAME
    if complete_path.exists():
        complete_path.unlink()


def mark_run_complete(
    output_dir: str | Path,
    *,
    configuration_hash: str,
    csv_schemas: Sequence[CSVSchema] = (),
    json_schemas: Sequence[JSONSchema] = (),
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Validate outputs, write final metadata, and create the completion marker."""

    output_dir = Path(output_dir)
    if metadata is not None:
        atomic_write_json(output_dir / "run_metadata.json", dict(metadata))
    validate_outputs(output_dir, csv_schemas=csv_schemas, json_schemas=json_schemas)
    _update_status(output_dir, status="complete", payload={"configuration_hash": configuration_hash})
    atomic_write_json(
        output_dir / RUN_COMPLETE_FILENAME,
        {
            "status": "complete",
            "configuration_hash": configuration_hash,
            "validated_csv_outputs": [str(Path(spec.path)) for spec in csv_schemas],
            "validated_json_outputs": [str(Path(spec.path)) for spec in json_schemas],
            "completed_utc": utc_now(),
        },
    )