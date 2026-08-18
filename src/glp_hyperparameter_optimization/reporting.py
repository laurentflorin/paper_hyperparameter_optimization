"""Comparison tables and figures for the GLP hyperparameter-selection study.

Scores the recursive out-of-sample forecasts of the four strategies (``paper``,
``mango_mdd``, ``mango_rmse``, ``mango_rmse_random``) on the transformed model
space (``100 * log`` levels for log variables, levels for rate variables) and
produces RMSE / relative-RMSE tables plus paper-style figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import resolve_project_path

MODEL_ORDER = ["paper", "mango_mdd", "mango_rmse", "mango_rmse_random"]
MODEL_LABELS = {
    "paper": "GLP Marginal Likelihood",
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
OPTIMIZATION_HORIZON_LINESTYLES = {
    "h1q": "-",
    "h2q": "--",
    "h4q": ":",
    "h8q": "-.",
}
# Variables shown in the headline figures (all present in every model size where
# possible; extras are only plotted when available).
HEADLINE_VARIABLES = ["GDP", "DEFL", "FFR", "CPI"]


def _iter_forecast_dirs(directory: Path) -> list[Path]:
    """Yield the directories that actually contain a forecast panel, descending
    one level into horizon subdirectories (e.g. ``h1q``/``h4q``) when needed."""
    directory = resolve_project_path(directory)
    if (directory / "forecast_panel.csv").exists() and (directory / "forecast_panel.csv").stat().st_size > 1:
        return [directory]
    if not directory.exists():
        return []
    return [
        child
        for child in sorted(directory.iterdir())
        if child.is_dir()
        and (child / "forecast_panel.csv").exists()
        and (child / "forecast_panel.csv").stat().st_size > 1
    ]


def load_forecast_panels(experiment_dirs: dict[str, Path | None]) -> pd.DataFrame:
    frames = []
    for model_name, directory in experiment_dirs.items():
        if directory is None:
            continue
        for candidate in _iter_forecast_dirs(Path(directory)):
            frame = pd.read_csv(candidate / "forecast_panel.csv")
            frame["model"] = model_name
            # Tag the optimization horizon when reading a per-horizon subdir.
            if candidate.name.startswith("h") and candidate.name.endswith("q"):
                frame["optimization_horizon"] = candidate.name
            frames.append(frame)
    if not frames:
        raise FileNotFoundError("No readable forecast_panel.csv files were found in the supplied directories.")
    return pd.concat(frames, ignore_index=True)


def load_hyperparameters(experiment_dirs: dict[str, Path | None]) -> pd.DataFrame:
    frames = []
    for model_name, directory in experiment_dirs.items():
        if directory is None:
            continue
        directory = resolve_project_path(directory)
        candidates = [directory]
        if not (directory / "selected_hyperparameters.csv").exists() and directory.exists():
            candidates = [child for child in sorted(directory.iterdir()) if child.is_dir()]
        for candidate in candidates:
            path = candidate / "selected_hyperparameters.csv"
            if path.exists() and path.stat().st_size > 1:
                frame = pd.read_csv(path)
                frame["model"] = model_name
                if candidate.name.startswith("h") and candidate.name.endswith("q"):
                    frame["optimization_horizon"] = candidate.name
                frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def compute_rmse_table(forecasts: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["model", "model_size", "variable", "horizon_quarters"]
    if "optimization_horizon" in forecasts.columns:
        group_cols.append("optimization_horizon")
    return (
        forecasts.dropna(subset=["error"])
        .groupby(group_cols, as_index=False, dropna=False)
        .agg(rmse=("error", lambda s: float(np.sqrt(np.mean(np.square(s))))), n=("error", "size"))
    )


def compute_relative_rmse(rmse_table: pd.DataFrame, baseline_model: str = "paper") -> pd.DataFrame:
    # The paper / mango_mdd runs do not usually have an optimization_horizon label,
    # while the RMSE workflows do (h1q/h2q/h4q/h8q). Relative RMSE should still
    # compare every row to the same paper baseline at the corresponding forecast
    # horizon, so baseline matching intentionally ignores optimization_horizon.
    baseline_keys = ["model_size", "variable", "horizon_quarters"]
    baseline = (
        rmse_table[rmse_table["model"] == baseline_model]
        .groupby(baseline_keys, as_index=False, dropna=False)
        .agg(baseline_rmse=("rmse", "mean"))
    )
    merged = rmse_table.merge(baseline, on=baseline_keys, how="left")
    merged["relative_rmse_pct"] = 100.0 * (merged["rmse"] - merged["baseline_rmse"]) / merged["baseline_rmse"]
    return merged


def compute_hyperparameter_summary(hyperparameters: pd.DataFrame) -> pd.DataFrame:
    if hyperparameters.empty:
        return hyperparameters
    value_columns = [c for c in ("lambda", "theta", "miu") if c in hyperparameters.columns]
    group_cols = ["model", "model_size"] if "model_size" in hyperparameters.columns else ["model"]
    if "optimization_horizon" in hyperparameters.columns:
        group_cols.append("optimization_horizon")
    summary = hyperparameters.groupby(group_cols, dropna=False)[value_columns].agg(["mean", "std", "min", "max"])
    summary.columns = ["_".join(col).strip() for col in summary.columns.to_flat_index()]
    summary = summary.reset_index()
    if "optimization_horizon" in summary.columns:
        summary["_opt_order"] = summary["optimization_horizon"].apply(lambda v: _optimization_horizon_sort_key(v)[1])
        summary = summary.sort_values(["model_size", "model", "_opt_order", "optimization_horizon"], na_position="first")
        summary = summary.drop(columns="_opt_order")
    return summary


def ordered_models(models) -> list[str]:
    unique = list(dict.fromkeys(str(m) for m in models))
    known = [m for m in MODEL_ORDER if m in unique]
    unknown = sorted(m for m in unique if m not in MODEL_ORDER)
    return known + unknown


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
    if pd.isna(optimization_horizon):
        return base
    return f"{base} ({optimization_horizon})"


def _variant_stem(model_name: str, optimization_horizon: object) -> str:
    if pd.isna(optimization_horizon):
        return model_name
    return f"{model_name}_{optimization_horizon}"


def _iter_model_variants(frame: pd.DataFrame):
    if "optimization_horizon" not in frame.columns:
        for model_name in ordered_models(frame["model"].tolist()):
            model_frame = frame[frame["model"] == model_name]
            if not model_frame.empty:
                yield model_name, np.nan, model_frame
        return

    for model_name in ordered_models(frame["model"].tolist()):
        model_frame = frame[frame["model"] == model_name]
        if model_frame.empty:
            continue
        no_opt_frame = model_frame[model_frame["optimization_horizon"].isna()]
        if not no_opt_frame.empty:
            yield model_name, np.nan, no_opt_frame
        optimization_horizons = sorted(model_frame["optimization_horizon"].dropna().unique(), key=_optimization_horizon_sort_key)
        for optimization_horizon in optimization_horizons:
            variant_frame = model_frame[model_frame["optimization_horizon"] == optimization_horizon]
            if not variant_frame.empty:
                yield model_name, optimization_horizon, variant_frame


def save_table_variants(frame: pd.DataFrame, output_stem: Path, index: bool = False) -> None:
    frame.to_csv(output_stem.with_suffix(".csv"), index=index)
    output_stem.with_suffix(".tex").write_text(frame.to_string(index=index), encoding="utf-8")
    output_stem.with_suffix(".md").write_text(frame.to_string(index=index), encoding="utf-8")


def plot_relative_rmse(relative_rmse: pd.DataFrame, output_dir: Path) -> None:
    plt.rcParams.update(
        {"font.family": "serif", "figure.dpi": 180, "axes.spines.top": False, "axes.spines.right": False}
    )
    for size, size_frame in relative_rmse.groupby("model_size"):
        variables = [v for v in HEADLINE_VARIABLES if v in size_frame["variable"].unique()]
        if not variables:
            variables = sorted(size_frame["variable"].unique())[:4]
        if not variables:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
        axes = axes.flatten()
        for axis, variable in zip(axes, variables):
            var_frame = size_frame[size_frame["variable"] == variable]
            axis.axhline(0.0, color="#999999", linewidth=1.0, linestyle="--")
            for model_name, optimization_horizon, model_frame in _iter_model_variants(var_frame):
                model_frame = model_frame.sort_values("horizon_quarters")
                axis.plot(
                    model_frame["horizon_quarters"],
                    model_frame["relative_rmse_pct"],
                    color=MODEL_COLORS.get(model_name, "#2c3e50"),
                    linewidth=2.0 if model_name != "paper" else 1.5,
                    linestyle=OPTIMIZATION_HORIZON_LINESTYLES.get(str(optimization_horizon), "-")
                    if not pd.isna(optimization_horizon)
                    else "-",
                    label=_variant_label(model_name, optimization_horizon),
                )
            axis.set_title(variable)
            axis.set_xlabel("Forecast Horizon (quarters)")
            axis.set_ylabel("Relative RMSE vs GLP (%)")
        for axis in axes[len(variables):]:
            axis.set_visible(False)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
        fig.suptitle(f"Relative RMSE by Horizon ({size} model)")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(output_dir / f"relative_rmse_{size}.png", bbox_inches="tight")
        plt.close(fig)


def plot_hyperparameter_paths(hyperparameters: pd.DataFrame, output_dir: Path) -> None:
    if hyperparameters.empty or "forecast_origin" not in hyperparameters.columns:
        return
    plt.rcParams.update(
        {"font.family": "serif", "figure.dpi": 180, "axes.spines.top": False, "axes.spines.right": False}
    )
    for existing in output_dir.glob("*_hyperparameter_paths.png"):
        existing.unlink()
    for model_name, optimization_horizon, model_frame in _iter_model_variants(hyperparameters):
        model_frame = model_frame.copy()
        model_frame["forecast_origin"] = pd.to_datetime(model_frame["forecast_origin"])
        model_frame = model_frame.sort_values("forecast_origin")
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
        for axis, parameter in zip(axes, ["lambda", "theta", "miu"]):
            if parameter not in model_frame.columns:
                continue
            axis.plot(
                model_frame["forecast_origin"],
                model_frame[parameter],
                color=MODEL_COLORS.get(model_name, "#2c3e50"),
                linewidth=1.8,
            )
            axis.set_title(parameter)
            axis.set_xlabel("Forecast origin")
        fig.suptitle(f"Selected Hyperparameters Over Time: {_variant_label(model_name, optimization_horizon)}")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(output_dir / f"{_variant_stem(model_name, optimization_horizon)}_hyperparameter_paths.png", bbox_inches="tight")
        plt.close(fig)


def create_glp_comparison_report(experiment_dirs: dict[str, Path | None], output_dir: Path) -> Path:
    output_dir = resolve_project_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    forecasts = load_forecast_panels(experiment_dirs)
    hyperparameters = load_hyperparameters(experiment_dirs)

    rmse_table = compute_rmse_table(forecasts)
    relative_rmse = compute_relative_rmse(rmse_table)
    hyper_summary = compute_hyperparameter_summary(hyperparameters)

    headline_rmse = rmse_table[rmse_table["variable"].isin(HEADLINE_VARIABLES)].copy()
    sort_cols = ["model_size", "variable", "horizon_quarters", "model"]
    if "optimization_horizon" in headline_rmse.columns:
        sort_cols.append("optimization_horizon")
    headline_rmse = headline_rmse.sort_values(sort_cols)

    save_table_variants(rmse_table, output_dir / "rmse_all_variables")
    save_table_variants(headline_rmse, output_dir / "rmse_headline_variables")
    save_table_variants(relative_rmse, output_dir / "relative_rmse_vs_glp")
    if not hyper_summary.empty:
        save_table_variants(hyper_summary, output_dir / "hyperparameter_summary")

    plot_relative_rmse(relative_rmse, output_dir)
    plot_hyperparameter_paths(hyperparameters, output_dir)
    return output_dir


# The strict implementation supersedes the legacy definitions above. Keeping
# this import at the module boundary preserves existing import paths.
from ._strict_reporting import *  # noqa: E402,F401,F403
