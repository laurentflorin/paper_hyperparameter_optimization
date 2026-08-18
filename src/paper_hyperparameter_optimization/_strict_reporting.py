"""Strict sample-paired reporting for the mixed-frequency workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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


HEADLINE_VARIABLES = ["GDP", "CPI", "UNR", "FF"]
MODEL_ORDER = ["paper", "mango_mdd", "mango_rmse", "mango_rmse_random"]
MODEL_LABELS = {
    "paper": "Paper Hyperparameters",
    "mango_mdd": "Mango MDD",
    "mango_rmse": "Mango RMSE",
    "mango_rmse_random": "Mango RMSE Random",
}
MODEL_COLORS = {
    "paper": "#4c566a",
    "mango_mdd": "#c0392b",
    "mango_rmse": "#1f618d",
    "mango_rmse_random": "#117864",
}
OPTIMIZATION_HORIZON_LINESTYLES = {"h1q": "-", "h2q": "--", "h4q": ":", "h8q": "-."}
PAIRING_CELL_COLUMNS = ["group", "variable", "horizon_quarters"]
PAIRING_OBSERVATION_COLUMNS = [
    "group",
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


def load_forecast_panels(experiment_dirs: dict[str, Path | None]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    for model_name, directory in experiment_dirs.items():
        if directory is None:
            continue
        for candidate in discover_run_directories(resolve_project_path(directory)):
            frame, record = _load_run_frame(model_name, candidate, "forecast_panel.csv")
            records.append(record)
            if frame is not None:
                frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            "No readable forecast_panel.csv files were found in the supplied experiment directories."
        )
    result = pd.concat(frames, ignore_index=True)
    result.attrs["comparison_inputs"] = records
    return result


def load_hyperparameters(experiment_dirs: dict[str, Path | None]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model_name, directory in experiment_dirs.items():
        if directory is None:
            continue
        for candidate in discover_run_directories(resolve_project_path(directory)):
            frame, _ = _load_run_frame(model_name, candidate, "selected_hyperparameters.csv")
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
        error_column="error_metric",
        actual_column="actual_metric",
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
    baseline_keys = ["group", "variable", "horizon_quarters"]
    baseline = rmse_table[rmse_table["model"] == baseline_model].rename(
        columns={"rmse": "baseline_rmse"}
    )
    merged = rmse_table.merge(
        baseline[[*baseline_keys, "baseline_rmse"]],
        on=baseline_keys,
        how="left",
    )
    return compute_relative_rmse_percent_change(merged, baseline_model=baseline_model)


def compute_hyperparameter_summary(hyperparameters: pd.DataFrame) -> pd.DataFrame:
    if hyperparameters.empty:
        return hyperparameters
    value_columns = [
        column for column in hyperparameters.columns if column.startswith("lambda")
    ]
    if not value_columns:
        return pd.DataFrame()
    group_columns = ["model"]
    if "optimization_horizon" in hyperparameters.columns:
        group_columns.append("optimization_horizon")
    summary = hyperparameters.groupby(group_columns, dropna=False)[value_columns].agg(
        ["mean", "std", "min", "max"]
    )
    summary.columns = [
        "_".join(column).strip() for column in summary.columns.to_flat_index()
    ]
    return summary.reset_index()


def ordered_models(models: list[str] | pd.Series | np.ndarray) -> list[str]:
    unique_models = list(dict.fromkeys(str(model) for model in models))
    known_models = [model for model in MODEL_ORDER if model in unique_models]
    unknown_models = sorted(model for model in unique_models if model not in MODEL_ORDER)
    return known_models + unknown_models


def _optimization_horizon_sort_key(value: object) -> tuple[int, int, str]:
    if pd.isna(value):
        return (0, 0, "")
    label = str(value)
    if label.startswith("h") and label.endswith("q"):
        try:
            return (1, int(label[1:-1]), label)
        except ValueError:
            pass
    return (1, 10**9, label)


def _variant_label(model_name: str, optimization_horizon: object) -> str:
    base = MODEL_LABELS.get(model_name, model_name)
    return base if pd.isna(optimization_horizon) else f"{base} ({optimization_horizon})"


def save_table_variants(
    frame: pd.DataFrame, output_stem: Path, index: bool = False
) -> None:
    frame.to_csv(output_stem.with_suffix(".csv"), index=index)
    text_value = frame.to_string(index=index)
    output_stem.with_suffix(".tex").write_text(text_value, encoding="utf-8")
    output_stem.with_suffix(".md").write_text(text_value, encoding="utf-8")


def _plot_variant_frames(group_frame: pd.DataFrame):
    horizons = sorted(
        group_frame["optimization_horizon"].dropna().unique(),
        key=_optimization_horizon_sort_key,
    )
    if not horizons:
        yield "all", group_frame
        return
    fixed = group_frame[group_frame["optimization_horizon"].isna()]
    for optimization_horizon in horizons:
        yield str(optimization_horizon), pd.concat(
            [
                fixed,
                group_frame[
                    group_frame["optimization_horizon"] == optimization_horizon
                ],
            ],
            ignore_index=True,
        )


def plot_relative_rmse_by_group(
    relative_rmse: pd.DataFrame, output_dir: Path
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    headline = relative_rmse[
        relative_rmse["variable"].isin(HEADLINE_VARIABLES)
    ]
    for group, group_frame in headline.groupby("group", dropna=False):
        for variant_stem, plot_frame in _plot_variant_frames(group_frame):
            fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
            axes = axes.flatten()
            for axis, variable in zip(axes, HEADLINE_VARIABLES):
                var_frame = plot_frame[plot_frame["variable"] == variable]
                axis.axhline(
                    0.0, color="#999999", linewidth=1.0, linestyle="--"
                )
                for (
                    model_name,
                    optimization_horizon,
                ), model_frame in var_frame.groupby(
                    ["model", "optimization_horizon"], dropna=False
                ):
                    model_frame = model_frame.sort_values("horizon_quarters")
                    axis.plot(
                        model_frame["horizon_quarters"],
                        model_frame["relative_rmse_pct"],
                        color=MODEL_COLORS.get(model_name, "#2c3e50"),
                        linewidth=1.5 if model_name == "paper" else 2.0,
                        linestyle=OPTIMIZATION_HORIZON_LINESTYLES.get(
                            str(optimization_horizon), "-"
                        ),
                        label=_variant_label(model_name, optimization_horizon),
                    )
                axis.set_title(variable)
                axis.set_xlabel("Forecast Horizon (quarters)")
                axis.set_ylabel("RMSE change vs paired paper baseline (%)")
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
            fig.suptitle(
                f"Paired Relative RMSE by Horizon: {group} ({variant_stem})"
            )
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            group_slug = (
                str(group).replace(" ", "_").replace("+", "plus_")
            )
            fig.savefig(
                output_dir
                / f"relative_rmse_{group_slug}_{variant_stem}.png",
                bbox_inches="tight",
            )
            plt.close(fig)


def plot_hyperparameter_paths(
    hyperparameters: pd.DataFrame, output_dir: Path
) -> None:
    if hyperparameters.empty or "forecast_origin" not in hyperparameters.columns:
        return
    plt.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    for (
        model_name,
        optimization_horizon,
    ), model_frame in hyperparameters.groupby(
        ["model", "optimization_horizon"], dropna=False
    ):
        if model_name == "paper":
            continue
        model_frame = model_frame.copy()
        model_frame["forecast_origin"] = pd.to_datetime(
            model_frame["forecast_origin"]
        )
        fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
        axes = axes.flatten()
        for axis, parameter in zip(
            axes, ["lambda1_1", "lambda2_1", "lambda4_1", "lambda5_1"]
        ):
            if parameter not in model_frame.columns:
                axis.set_visible(False)
                continue
            axis.plot(
                model_frame["forecast_origin"],
                model_frame[parameter],
                color=MODEL_COLORS.get(model_name, "#2c3e50"),
                linewidth=1.8,
            )
            axis.set_title(parameter)
            axis.set_xlabel("Forecast origin")
        fig.suptitle(
            f"Selected Hyperparameters: "
            f"{_variant_label(model_name, optimization_horizon)}"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        suffix = (
            model_name
            if pd.isna(optimization_horizon)
            else f"{model_name}_{optimization_horizon}"
        )
        fig.savefig(
            output_dir / f"{suffix}_hyperparameter_paths.png",
            bbox_inches="tight",
        )
        plt.close(fig)


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
        "forecast_horizon_months": lambda m: _first(
            m, "forecast_horizon_months"
        ),
        "aggregation": lambda m: _first(m, "temp_agg", "aggregation"),
        "evaluation_transforms": lambda m: _first(
            m,
            "evaluation_transforms",
            "evaluation_transform_policy",
            "transform_policy",
        ),
        "model_universe": lambda m: _first(
            m, "model_universe", "final_fit_variables", "model_codes"
        ),
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
            "mbfvar_revision",
            "mbfvar_commit",
            "dependency_versions",
        ),
        "repository_commit": lambda m: _first(
            m, "repository_commit", "repo_commit", "git_commit"
        ),
    }


def create_comparison_report(
    experiment_dirs: dict[str, Path | None],
    output_dir: Path,
    *,
    minimum_common_coverage: float = 0.8,
    allow_legacy_metadata: bool = False,
) -> Path:
    resolved_dirs = {
        model_name: resolve_project_path(directory)
        for model_name, directory in experiment_dirs.items()
        if directory is not None
    }
    output_dir = resolve_project_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    forecasts = load_forecast_panels(resolved_dirs)
    input_records = forecasts.attrs.get("comparison_inputs", [])
    compatibility = validate_metadata_compatibility(
        input_records,
        _compatibility_extractors(),
        allow_legacy_metadata=allow_legacy_metadata,
    )
    hyperparameters = load_hyperparameters(resolved_dirs)
    rmse_table, audit = compute_rmse_table(
        forecasts,
        minimum_common_coverage=minimum_common_coverage,
        return_audit=True,
    )
    audit = append_run_failures(audit, input_records)
    relative_rmse = compute_relative_rmse(rmse_table)
    hyper_summary = compute_hyperparameter_summary(hyperparameters)

    headline_rmse = rmse_table[
        rmse_table["variable"].isin(HEADLINE_VARIABLES)
    ].copy()
    headline_relative = relative_rmse[
        relative_rmse["variable"].isin(HEADLINE_VARIABLES)
    ].copy()

    output_files: list[Path] = []
    for frame, stem in [
        (rmse_table, "rmse_all_variables"),
        (headline_rmse, "rmse_headline_variables"),
        (relative_rmse, "relative_rmse_vs_paper"),
    ]:
        output_stem = output_dir / stem
        save_table_variants(frame, output_stem)
        output_files.extend(
            output_stem.with_suffix(suffix)
            for suffix in (".csv", ".tex", ".md")
        )
    if not hyper_summary.empty:
        output_stem = output_dir / "hyperparameter_summary"
        save_table_variants(hyper_summary, output_stem)
        output_files.extend(
            output_stem.with_suffix(suffix)
            for suffix in (".csv", ".tex", ".md")
        )

    audit_path = output_dir / "comparison_exclusion_audit.csv"
    audit.to_csv(audit_path, index=False)
    output_files.append(audit_path)
    plot_relative_rmse_by_group(headline_relative, output_dir)
    plot_hyperparameter_paths(hyperparameters, output_dir)
    output_files.extend(sorted(output_dir.glob("*.png")))

    write_comparison_manifest(
        output_dir / "comparison_manifest.json",
        workflow="mixed_frequency_bvar",
        baseline_model="paper",
        input_records=input_records,
        compatibility=compatibility,
        minimum_common_coverage=minimum_common_coverage,
        exclusion_audit=audit,
        output_files=output_files,
    )
    return output_dir


__all__ = [
    "HEADLINE_VARIABLES",
    "MODEL_ORDER",
    "MODEL_LABELS",
    "MODEL_COLORS",
    "load_forecast_panels",
    "load_hyperparameters",
    "compute_rmse_table",
    "compute_relative_rmse",
    "compute_hyperparameter_summary",
    "ordered_models",
    "save_table_variants",
    "plot_relative_rmse_by_group",
    "plot_hyperparameter_paths",
    "create_comparison_report",
]
