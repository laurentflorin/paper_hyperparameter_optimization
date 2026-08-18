"""Shared, fail-closed helpers for forecast-comparison reporting.

The two forecasting workflows have different schemas, but their comparison
contracts are the same: every input run must be complete and attributable, a
model variant must have unique forecast keys, and relative RMSE must use a
baseline recomputed on the exact same observations as the competitor.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd

from common_hpo.io import classify_run_directory


OPTIMIZATION_HORIZON_RE = re.compile(r"^h([1-9][0-9]*)q$")
RUN_FILES = (
    "forecast_panel.csv",
    "selected_hyperparameters.csv",
    "failed_origins.csv",
    "run_metadata.json",
)


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _complete_run_directory(path: Path) -> tuple[bool, str]:
    state = classify_run_directory(path)
    if state.status == "complete":
        return True, ""
    if state.status in {"partial", "failed", "cancelled"}:
        return False, state.reason or f"run state is {state.status}"

    forecast_path = path / "forecast_panel.csv"
    metadata_path = path / "run_metadata.json"
    if not forecast_path.exists():
        return False, "forecast_panel.csv is missing"
    if forecast_path.stat().st_size <= 1:
        return False, "forecast_panel.csv is empty"
    if not metadata_path.exists():
        return False, "run_metadata.json is missing"
    if metadata_path.stat().st_size <= 1:
        return False, "run_metadata.json is empty"
    return True, ""


def discover_run_directories(directory: Path) -> list[Path]:
    """Resolve a direct run or a one-level horizon batch, rejecting partial runs.

    A directory containing either of the two required direct-run markers is
    treated as a direct run. Otherwise its child directories are inspected. If
    a child contains either marker, it must contain a non-empty forecast panel
    and metadata file; partial horizon runs are never silently skipped.
    """
    directory = Path(directory).expanduser().resolve()
    if not directory.exists():
        raise FileNotFoundError(f"Experiment directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Experiment path is not a directory: {directory}")

    direct_markers = [directory / "forecast_panel.csv", directory / "run_metadata.json"]
    if any(path.exists() for path in direct_markers):
        complete, reason = _complete_run_directory(directory)
        if not complete:
            raise ValueError(f"Incomplete experiment directory {directory}: {reason}.")
        return [directory]

    candidates: list[Path] = []
    incomplete: list[str] = []
    for child in sorted(directory.iterdir()):
        if not child.is_dir():
            continue
        markers = [child / "forecast_panel.csv", child / "run_metadata.json"]
        if not any(path.exists() for path in markers):
            continue
        complete, reason = _complete_run_directory(child)
        if complete:
            candidates.append(child)
        else:
            incomplete.append(f"{child.name}: {reason}")

    if incomplete:
        details = "; ".join(incomplete)
        raise ValueError(f"Incomplete child run(s) under {directory}: {details}.")
    if not candidates:
        raise FileNotFoundError(
            f"No complete run was found in {directory}; expected a non-empty "
            "forecast_panel.csv and run_metadata.json."
        )
    return candidates


def load_run_metadata(directory: Path) -> dict[str, Any]:
    path = Path(directory) / "run_metadata.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Run metadata must be a JSON object: {path}")
    return value


def optimization_horizon_label(
    directory: Path,
    metadata: Mapping[str, Any],
    *,
    strategy: str,
) -> tuple[str | None, str | None]:
    """Read a variant horizon from metadata, with a validated legacy fallback."""
    name_match = OPTIMIZATION_HORIZON_RE.fullmatch(Path(directory).name)
    directory_horizon = int(name_match.group(1)) if name_match else None
    raw_horizon = metadata.get("optimization_eval_horizon_quarters")

    if raw_horizon is not None:
        try:
            horizon = int(raw_horizon)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid optimization_eval_horizon_quarters={raw_horizon!r} in {directory}."
            ) from exc
        if horizon <= 0 or float(raw_horizon) != float(horizon):
            raise ValueError(
                f"optimization_eval_horizon_quarters must be a positive integer in {directory}."
            )
        if directory_horizon is not None and directory_horizon != horizon:
            raise ValueError(
                f"Optimization-horizon mismatch in {directory}: metadata says h{horizon}q "
                f"but the directory is named h{directory_horizon}q."
            )
        return f"h{horizon}q", "run_metadata"

    if directory_horizon is None:
        return None, None
    if strategy not in {"mango_rmse", "mango_rmse_random"}:
        raise ValueError(
            f"Legacy h<N>q directory fallback is only valid for RMSE strategies: {directory}."
        )
    legacy_horizon = metadata.get("optimization_horizon_quarters")
    if legacy_horizon is not None and int(legacy_horizon) != directory_horizon:
        raise ValueError(
            f"Legacy optimization-horizon mismatch in {directory}: metadata says "
            f"{legacy_horizon}, directory says {directory_horizon}."
        )
    return f"h{directory_horizon}q", "directory_name_legacy_fallback"


def build_input_record(
    *,
    model: str,
    directory: Path,
    metadata: Mapping[str, Any],
    optimization_horizon: str | None,
    horizon_source: str | None,
) -> dict[str, Any]:
    directory = Path(directory).resolve()
    files: dict[str, dict[str, Any]] = {}
    for filename in RUN_FILES:
        path = directory / filename
        if not path.exists():
            continue
        files[filename] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    directory_digest = hashlib.sha256()
    for filename, description in sorted(files.items()):
        directory_digest.update(filename.encode("utf-8"))
        directory_digest.update(description["sha256"].encode("ascii"))
    return {
        "model": model,
        "optimization_horizon": optimization_horizon,
        "optimization_horizon_source": horizon_source,
        "directory": str(directory),
        "directory_input_sha256": directory_digest.hexdigest(),
        "files": files,
        "metadata": dict(metadata),
    }


def _json_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def validate_metadata_compatibility(
    records: Sequence[Mapping[str, Any]],
    field_extractors: Mapping[str, Callable[[Mapping[str, Any]], Any]],
    *,
    allow_legacy_metadata: bool = False,
) -> dict[str, Any]:
    """Validate that provenance-bearing metadata are present and compatible."""
    if not records:
        raise ValueError("No comparison input records were supplied.")
    compatibility: dict[str, Any] = {}
    missing: dict[str, list[str]] = {}
    incompatible: dict[str, dict[str, Any]] = {}

    for field, extractor in field_extractors.items():
        values: list[tuple[str, Any]] = []
        for record in records:
            label = str(record["model"])
            if record.get("optimization_horizon"):
                label += f"/{record['optimization_horizon']}"
            value = extractor(record["metadata"])
            if value is None:
                missing.setdefault(field, []).append(label)
            else:
                values.append((label, value))

        distinct = {_json_key(value): value for _, value in values}
        if len(distinct) > 1:
            incompatible[field] = {label: value for label, value in values}
        elif values:
            compatibility[field] = values[0][1]
        else:
            compatibility[field] = None

    if incompatible:
        details = "; ".join(
            f"{field}={values}" for field, values in sorted(incompatible.items())
        )
        raise ValueError(f"Incompatible comparison run metadata: {details}.")
    if missing and not allow_legacy_metadata:
        details = "; ".join(
            f"{field} missing for {', '.join(labels)}" for field, labels in sorted(missing.items())
        )
        raise ValueError(
            "Comparison provenance is incomplete: "
            f"{details}. Regenerate the runs or explicitly enable legacy metadata mode."
        )
    compatibility["legacy_missing_fields"] = missing
    compatibility["allow_legacy_metadata"] = bool(allow_legacy_metadata)
    return compatibility


def _variant_key(frame: pd.DataFrame) -> pd.Series:
    if "optimization_horizon" not in frame.columns:
        return pd.Series("", index=frame.index, dtype="object")
    return frame["optimization_horizon"].fillna("").astype(str)


def _format_duplicate_sample(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    return frame.loc[:, list(columns)].head(5).to_dict(orient="records").__repr__()


def compute_paired_rmse_table(
    forecasts: pd.DataFrame,
    *,
    cell_columns: Sequence[str],
    observation_columns: Sequence[str],
    error_column: str,
    actual_column: str,
    baseline_model: str,
    minimum_common_coverage: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute pairwise RMSEs and a key-level exclusion audit.

    Each non-baseline model/optimization-horizon variant is inner-joined to the
    baseline on ``observation_columns``. Both RMSEs on that row are therefore
    calculated from the same observations. Baseline self-rows use their own
    valid sample and have relative RMSE zero downstream.
    """
    if not 0.0 <= minimum_common_coverage <= 1.0:
        raise ValueError("minimum_common_coverage must lie in [0, 1].")
    required = {
        "model",
        *cell_columns,
        *observation_columns,
        error_column,
        actual_column,
    }
    missing_columns = sorted(required.difference(forecasts.columns))
    if missing_columns:
        raise ValueError(f"Forecast panel is missing required comparison columns: {missing_columns}.")

    frame = forecasts.copy()
    if "optimization_horizon" not in frame.columns:
        frame["optimization_horizon"] = pd.NA
    frame["_optimization_variant"] = _variant_key(frame)

    identity_columns = ["model", "_optimization_variant", *observation_columns]
    duplicates = frame.duplicated(identity_columns, keep=False)
    if duplicates.any():
        sample = _format_duplicate_sample(frame.loc[duplicates], identity_columns)
        raise ValueError(f"Duplicate forecast keys within a model variant: {sample}.")

    numeric_actual = pd.to_numeric(frame[actual_column], errors="coerce")
    nonnumeric_actual = frame[actual_column].notna() & numeric_actual.isna()
    if nonnumeric_actual.any():
        raise ValueError(f"Column {actual_column!r} contains non-numeric realized values.")
    frame["_actual_numeric"] = numeric_actual
    mismatched_keys: list[dict[str, Any]] = []
    for key, group in frame.dropna(subset=["_actual_numeric"]).groupby(
        list(observation_columns), dropna=False, sort=False
    ):
        values = group["_actual_numeric"].to_numpy(dtype=float)
        if values.size > 1 and not np.allclose(values, values[0], rtol=1.0e-10, atol=1.0e-12):
            key_tuple = key if isinstance(key, tuple) else (key,)
            mismatched_keys.append(
                {**dict(zip(observation_columns, key_tuple)), "actual_values": sorted(set(values.tolist()))}
            )
            if len(mismatched_keys) >= 5:
                break
    if mismatched_keys:
        raise ValueError(f"Models contain mismatched actual values on common forecast keys: {mismatched_keys}.")

    baseline = frame[frame["model"] == baseline_model].copy()
    if baseline.empty:
        raise ValueError(f"Baseline model {baseline_model!r} has no forecasts.")
    baseline_variants = baseline["_optimization_variant"].unique().tolist()
    if len(baseline_variants) != 1:
        raise ValueError(
            f"Baseline model {baseline_model!r} must have exactly one variant; found {baseline_variants}."
        )

    score_rows: list[dict[str, Any]] = []
    audit_frames: list[pd.DataFrame] = []

    for cell_key, baseline_cell in baseline.groupby(list(cell_columns), dropna=False, sort=False):
        cell_tuple = cell_key if isinstance(cell_key, tuple) else (cell_key,)
        cell_values = dict(zip(cell_columns, cell_tuple))
        valid = baseline_cell.dropna(subset=[error_column])
        if valid.empty:
            continue
        rmse = float(np.sqrt(np.mean(np.square(valid[error_column].astype(float)))))
        score_rows.append(
            {
                "model": baseline_model,
                "optimization_horizon": (
                    baseline_cell["optimization_horizon"].iloc[0]
                    if baseline_cell["_optimization_variant"].iloc[0]
                    else pd.NA
                ),
                **cell_values,
                "rmse": rmse,
                "baseline_rmse": rmse,
                "n_model": int(valid.shape[0]),
                "n_baseline": int(valid.shape[0]),
                "n_common": int(valid.shape[0]),
                "n_model_total_keys": int(baseline_cell.shape[0]),
                "n_baseline_total_keys": int(baseline_cell.shape[0]),
                "excluded_model_keys": 0,
                "excluded_baseline_keys": 0,
                "excluded_union_keys": 0,
                "common_coverage": 1.0,
            }
        )

    competitor = frame[frame["model"] != baseline_model]
    competitor_groups = ["model", "_optimization_variant", *cell_columns]
    for group_key, model_cell in competitor.groupby(competitor_groups, dropna=False, sort=False):
        model_name, variant, *cell_tuple = group_key
        cell_values = dict(zip(cell_columns, cell_tuple))
        mask = pd.Series(True, index=baseline.index)
        for column, value in cell_values.items():
            if pd.isna(value):
                mask &= baseline[column].isna()
            else:
                mask &= baseline[column] == value
        baseline_cell = baseline.loc[mask]
        if baseline_cell.empty:
            raise ValueError(
                f"No {baseline_model!r} baseline cell for model={model_name!r}, "
                f"optimization_horizon={variant or None!r}, values={cell_values}."
            )

        left = model_cell.loc[:, [*observation_columns, error_column, actual_column]].rename(
            columns={error_column: "model_error", actual_column: "model_actual"}
        )
        right = baseline_cell.loc[:, [*observation_columns, error_column, actual_column]].rename(
            columns={error_column: "baseline_error", actual_column: "baseline_actual"}
        )
        left["_model_present"] = True
        right["_baseline_present"] = True
        merged = left.merge(right, on=list(observation_columns), how="outer", validate="one_to_one")

        model_present = merged["_model_present"].fillna(False).astype(bool)
        baseline_present = merged["_baseline_present"].fillna(False).astype(bool)
        model_valid = model_present & merged["model_error"].notna()
        baseline_valid = baseline_present & merged["baseline_error"].notna()
        common = model_valid & baseline_valid

        status = np.select(
            [
                ~model_present,
                ~baseline_present,
                model_present & baseline_present & merged["model_error"].isna() & merged["baseline_error"].isna(),
                model_present & baseline_present & merged["model_error"].isna(),
                model_present & baseline_present & merged["baseline_error"].isna(),
            ],
            [
                "missing_model_key",
                "missing_baseline_key",
                "both_errors_missing",
                "model_error_missing",
                "baseline_error_missing",
            ],
            default="common",
        )
        merged["status"] = status

        n_model = int(model_valid.sum())
        n_baseline = int(baseline_valid.sum())
        n_common = int(common.sum())
        if n_common == 0:
            raise ValueError(
                f"Model {model_name!r}/{variant or 'default'} has no valid observations in common "
                f"with baseline {baseline_model!r} for {cell_values}."
            )
        common_coverage = n_common / max(n_model, n_baseline)
        if common_coverage < minimum_common_coverage:
            warnings.warn(
                f"Common-sample coverage {common_coverage:.1%} is below the configured "
                f"{minimum_common_coverage:.1%} for {model_name}/{variant or 'default'} {cell_values}.",
                RuntimeWarning,
                stacklevel=2,
            )

        model_rmse = float(np.sqrt(np.mean(np.square(merged.loc[common, "model_error"].astype(float)))))
        baseline_rmse = float(
            np.sqrt(np.mean(np.square(merged.loc[common, "baseline_error"].astype(float))))
        )
        score_rows.append(
            {
                "model": model_name,
                "optimization_horizon": variant if variant else pd.NA,
                **cell_values,
                "rmse": model_rmse,
                "baseline_rmse": baseline_rmse,
                "n_model": n_model,
                "n_baseline": n_baseline,
                "n_common": n_common,
                "n_model_total_keys": int(model_present.sum()),
                "n_baseline_total_keys": int(baseline_present.sum()),
                "excluded_model_keys": n_model - n_common,
                "excluded_baseline_keys": n_baseline - n_common,
                "excluded_union_keys": int((~common).sum()),
                "common_coverage": common_coverage,
            }
        )

        excluded = merged.loc[~common].copy()
        if not excluded.empty:
            excluded.insert(0, "model", model_name)
            excluded.insert(1, "optimization_horizon", variant if variant else pd.NA)
            audit_frames.append(excluded)

    scores = pd.DataFrame(score_rows)
    if scores.empty:
        raise ValueError("No valid RMSE cells could be computed.")
    sort_columns = [*cell_columns, "model", "optimization_horizon"]
    scores = scores.sort_values(sort_columns, na_position="first").reset_index(drop=True)
    audit = pd.concat(audit_frames, ignore_index=True) if audit_frames else pd.DataFrame()
    return scores, audit


def append_run_failures(audit: pd.DataFrame, input_records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Append explicit ``failed_origins.csv`` records to an exclusion audit."""
    failures: list[pd.DataFrame] = []
    for record in input_records:
        file_record = record.get("files", {}).get("failed_origins.csv")
        if not file_record or int(file_record.get("size_bytes", 0)) <= 1:
            continue
        path = Path(file_record["path"])
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if frame.empty:
            continue
        frame.insert(0, "model", record["model"])
        frame.insert(1, "optimization_horizon", record.get("optimization_horizon"))
        frame["status"] = "run_failure"
        failures.append(frame)
    frames = [frame for frame in [audit, *failures] if not frame.empty]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def compute_relative_rmse_percent_change(scores: pd.DataFrame, *, baseline_model: str) -> pd.DataFrame:
    """Add percent-change relative RMSE (baseline = 0; negative is better)."""
    if "baseline_rmse" not in scores.columns:
        raise ValueError("Paired score table must contain baseline_rmse.")
    result = scores.copy()
    denominator = result["baseline_rmse"].astype(float)
    numerator = result["rmse"].astype(float) - denominator
    with np.errstate(divide="ignore", invalid="ignore"):
        result["relative_rmse_pct"] = 100.0 * numerator / denominator
    result.loc[result["model"] == baseline_model, "relative_rmse_pct"] = 0.0
    return result


def write_comparison_manifest(
    path: Path,
    *,
    workflow: str,
    baseline_model: str,
    input_records: Sequence[Mapping[str, Any]],
    compatibility: Mapping[str, Any],
    minimum_common_coverage: float,
    exclusion_audit: pd.DataFrame,
    output_files: Iterable[Path],
) -> Path:
    outputs: dict[str, dict[str, Any]] = {}
    for output_path in output_files:
        output_path = Path(output_path)
        if not output_path.exists():
            continue
        outputs[output_path.name] = {
            "path": str(output_path.resolve()),
            "size_bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        }
    exclusion_counts = (
        exclusion_audit["status"].value_counts(dropna=False).astype(int).to_dict()
        if not exclusion_audit.empty and "status" in exclusion_audit.columns
        else {}
    )
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "workflow": workflow,
        "baseline_model": baseline_model,
        "pairing_policy": "pairwise inner join on unique valid forecast keys",
        "relative_rmse_definition": (
            "100 * (model_rmse - paired_baseline_rmse) / paired_baseline_rmse; "
            "baseline is 0 and negative values favor the model"
        ),
        "minimum_common_coverage": minimum_common_coverage,
        "compatibility": dict(compatibility),
        "inputs": list(input_records),
        "exclusion_decisions": {
            "audit_file": "comparison_exclusion_audit.csv",
            "n_rows": int(exclusion_audit.shape[0]),
            "status_counts": exclusion_counts,
        },
        "outputs": outputs,
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path
