"""Strict sample-paired reporting for the quarterly GLP workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from forecast_comparison import (
    append_run_failures,
    build_input_record,
    compute_paired_rmse_table,
    compute_relative_rmse_percent_change,
    discover_run_directories,
    load_run_metadata,
    optimization_horizon_label,
    validate_metadata_compatibility,
    write_comparison_manifest,
)

from .config import resolve_project_path


PAIRING_CELL_COLUMNS = ["model_size", "variable", "horizon_quarters"]
PAIRING_OBSERVATION_COLUMNS = [
    "model_size",
    "forecast_origin",
    "target_quarter",
    "variable",
    "horizon_quarters",
]


def _load_run_frame(
    model_name: str, candidate: Path, filename: str
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    metadata = load_run_metadata(candidate)
    metadata_strategy = metadata.get("strategy")
    if metadata_strategy != model_name:
        raise ValueError(
            f"Strategy mismatch for {candidate}: comparison labels it {model_name!r}, "
            f"run_metadata.json says {metadata_strategy!r}."
        )
    optimization_horizon, horizon_source = optimization_horizon_label(
        candidate, metadata, strategy=model_name
    )
    record = build_input_record(
        model=model_name,
        directory=candidate,
        metadata=metadata,
        optimization_horizon=optimization_horizon,
        horizon_source=horizon_source,
    )
    path = candidate / filename
    if not path.exists() or path.stat().st_size <= 1:
        return None, record
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{path} has headers but no data rows.")
    frame["model"] = model_name
    frame["optimization_horizon"] = optimization_horizon
    return frame, record


def load_forecast_panels(
    experiment_dirs: dict[str, Path | None]
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    for model_name, directory in experiment_dirs.items():
        if directory is None:
            continue
        for candidate in discover_run_directories(
            resolve_project_path(directory)
        ):
            frame, record = _load_run_frame(
                model_name, candidate, "forecast_panel.csv"
            )
            records.append(record)
            if frame is not None:
                frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            "No readable forecast_panel.csv files were found in the supplied directories."
        )
    result = pd.concat(frames, ignore_index=True)
    result.attrs["comparison_inputs"] = records
    return result


def load_hyperparameters(
    experiment_dirs: dict[str, Path | None]
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model_name, directory in experiment_dirs.items():
        if directory is None:
            continue
        for candidate in discover_run_directories(
            resolve_project_path(directory)
        ):
            frame, _ = _load_run_frame(
                model_name, candidate, "selected_hyperparameters.csv"
            )
            if frame is not None:
                frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compute_rmse_table(
    forecasts: pd.DataFrame,
    baseline_model: str = "paper",
    *,
    minimum_common_coverage: float = 0.8,
    return_audit: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    scores, audit = compute_paired_rmse_table(
        forecasts,
        cell_columns=PAIRING_CELL_COLUMNS,
        observation_columns=PAIRING_OBSERVATION_COLUMNS,
        error_column="error",
        actual_column="actual",
        baseline_model=baseline_model,
        minimum_common_coverage=minimum_common_coverage,
    )
    return (scores, audit) if return_audit else scores


def compute_relative_rmse(
    rmse_table: pd.DataFrame, baseline_model: str = "paper"
) -> pd.DataFrame:
    if "baseline_rmse" in rmse_table.columns:
        return compute_relative_rmse_percent_change(
            rmse_table, baseline_model=baseline_model
        )
    baseline_keys = ["model_size", "variable", "horizon_quarters"]
    baseline = (
        rmse_table[rmse_table["model"] == baseline_model]
        .groupby(baseline_keys, as_index=False, dropna=False)
        .agg(baseline_rmse=("rmse", "mean"))
    )
    merged = rmse_table.merge(baseline, on=baseline_keys, how="left")
    return compute_relative_rmse_percent_change(
        merged, baseline_model=baseline_model
    )


def _first(metadata: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metadata and metadata[key] is not None:
            return metadata[key]
    return None


def _selection_schedule(metadata: Mapping[str, Any]) -> Any:
    explicit = _first(
        metadata, "selection_schedule", "hyperparameter_selection_schedule"
    )
    if explicit is not None:
        return explicit
    selected_once = metadata.get("hyperparameters_selected_once")
    if selected_once is not None:
        return "first_origin" if bool(selected_once) else "per_origin"
    return None


def _compatibility_extractors() -> dict[str, Any]:
    return {
        "actual_vintage": lambda m: _first(m, "actual_vintage"),
        "forecast_origin_start": lambda m: _first(
            m, "forecast_origin_start", "origin_start", "start"
        ),
        "forecast_origin_end": lambda m: _first(
            m, "forecast_origin_end", "origin_end", "end"
        ),
        "horizon_semantics": lambda m: _first(
            m, "horizon_semantics", "target_horizon_semantics"
        ),
        "forecast_horizon_quarters": lambda m: _first(
            m, "forecast_horizon_quarters", "max_forecast_horizon_quarters"
        ),
        "aggregation": lambda m: _first(
            m, "aggregation", "quarterly_aggregation"
        ),
        "evaluation_transforms": lambda m: _first(
            m,
            "evaluation_transforms",
            "evaluation_transform_policy",
            "transform_policy",
        ),
        "model_universe": lambda m: _first(
            m, "model_universe", "model_codes", "model_size"
        ),
        "lags": lambda m: _first(m, "lags"),
        "selection_schedule": _selection_schedule,
        "data_fingerprint": lambda m: _first(
            m,
            "data_fingerprint",
            "panel_sha256",
            "panel_hash",
            "data_sha256",
        ),
        "dependency_revision": lambda m: _first(
            m,
            "dependency_revision",
            "covbayesvar_revision",
            "covbayesvar_commit",
            "dependency_versions",
        ),
        "repository_commit": lambda m: _first(
            m, "repository_commit", "repo_commit", "git_commit"
        ),
    }


def create_glp_comparison_report(
    experiment_dirs: dict[str, Path | None],
    output_dir: Path,
    *,
    minimum_common_coverage: float = 0.8,
    allow_legacy_metadata: bool = False,
) -> Path:
    from . import reporting as public_reporting

    output_dir = resolve_project_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    forecasts = load_forecast_panels(experiment_dirs)
    input_records = forecasts.attrs.get("comparison_inputs", [])
    compatibility = validate_metadata_compatibility(
        input_records,
        _compatibility_extractors(),
        allow_legacy_metadata=allow_legacy_metadata,
    )
    hyperparameters = load_hyperparameters(experiment_dirs)
    rmse_table, audit = compute_rmse_table(
        forecasts,
        minimum_common_coverage=minimum_common_coverage,
        return_audit=True,
    )
    audit = append_run_failures(audit, input_records)
    relative_rmse = compute_relative_rmse(rmse_table)
    hyper_summary = public_reporting.compute_hyperparameter_summary(
        hyperparameters
    )

    headline_rmse = rmse_table[
        rmse_table["variable"].isin(public_reporting.HEADLINE_VARIABLES)
    ].copy()
    output_files: list[Path] = []
    for frame, stem in [
        (rmse_table, "rmse_all_variables"),
        (headline_rmse, "rmse_headline_variables"),
        (relative_rmse, "relative_rmse_vs_glp"),
    ]:
        output_stem = output_dir / stem
        public_reporting.save_table_variants(frame, output_stem)
        output_files.extend(
            output_stem.with_suffix(suffix)
            for suffix in (".csv", ".tex", ".md")
        )
    if not hyper_summary.empty:
        output_stem = output_dir / "hyperparameter_summary"
        public_reporting.save_table_variants(hyper_summary, output_stem)
        output_files.extend(
            output_stem.with_suffix(suffix)
            for suffix in (".csv", ".tex", ".md")
        )

    audit_path = output_dir / "comparison_exclusion_audit.csv"
    audit.to_csv(audit_path, index=False)
    output_files.append(audit_path)
    public_reporting.plot_relative_rmse(relative_rmse, output_dir)
    public_reporting.plot_hyperparameter_paths(hyperparameters, output_dir)
    output_files.extend(sorted(output_dir.glob("*.png")))

    write_comparison_manifest(
        output_dir / "comparison_manifest.json",
        workflow="glp_quarterly_bvar",
        baseline_model="paper",
        input_records=input_records,
        compatibility=compatibility,
        minimum_common_coverage=minimum_common_coverage,
        exclusion_audit=audit,
        output_files=output_files,
    )
    return output_dir


__all__ = [
    "load_forecast_panels",
    "load_hyperparameters",
    "compute_rmse_table",
    "compute_relative_rmse",
    "create_glp_comparison_report",
]
