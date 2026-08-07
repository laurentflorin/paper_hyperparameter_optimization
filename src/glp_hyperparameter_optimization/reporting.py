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
        .groupby(group_cols, as_index=False)
        .agg(rmse=("error", lambda s: float(np.sqrt(np.mean(np.square(s))))), n=("error", "size"))
    )


def compute_relative_rmse(rmse_table: pd.DataFrame, baseline_model: str = "paper") -> pd.DataFrame:
    keys = ["model_size", "variable", "horizon_quarters"]
    if "optimization_horizon" in rmse_table.columns:
        keys.append("optimization_horizon")
    baseline = rmse_table[rmse_table["model"] == baseline_model].rename(columns={"rmse": "baseline_rmse"})
    merged = rmse_table.merge(baseline[keys + ["baseline_rmse"]], on=keys, how="left")
    merged["relative_rmse_pct"] = 100.0 * (merged["rmse"] - merged["baseline_rmse"]) / merged["baseline_rmse"]
    return merged


def compute_hyperparameter_summary(hyperparameters: pd.DataFrame) -> pd.DataFrame:
    if hyperparameters.empty:
        return hyperparameters
    value_columns = [c for c in ("lambda", "theta", "miu") if c in hyperparameters.columns]
    group_cols = ["model", "model_size"] if "model_size" in hyperparameters.columns else ["model"]
    summary = hyperparameters.groupby(group_cols)[value_columns].agg(["mean", "std", "min", "max"])
    summary.columns = ["_".join(col).strip() for col in summary.columns.to_flat_index()]
    return summary.reset_index()


def ordered_models(models) -> list[str]:
    unique = list(dict.fromkeys(str(m) for m in models))
    known = [m for m in MODEL_ORDER if m in unique]
    unknown = sorted(m for m in unique if m not in MODEL_ORDER)
    return known + unknown


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
            for model_name in ordered_models(var_frame["model"].tolist()):
                model_frame = var_frame[var_frame["model"] == model_name].sort_values("horizon_quarters")
                if model_frame.empty:
                    continue
                axis.plot(
                    model_frame["horizon_quarters"],
                    model_frame["relative_rmse_pct"],
                    color=MODEL_COLORS.get(model_name, "#2c3e50"),
                    linewidth=2.0 if model_name != "paper" else 1.5,
                    label=MODEL_LABELS.get(model_name, model_name),
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
    for model_name in ordered_models(hyperparameters["model"].tolist()):
        model_frame = hyperparameters[hyperparameters["model"] == model_name].copy()
        if model_frame.empty:
            continue
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
        fig.suptitle(f"Selected Hyperparameters Over Time: {MODEL_LABELS.get(model_name, model_name)}")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(output_dir / f"{model_name}_hyperparameter_paths.png", bbox_inches="tight")
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
    headline_rmse = headline_rmse.sort_values(["model_size", "variable", "horizon_quarters", "model"])

    save_table_variants(rmse_table, output_dir / "rmse_all_variables")
    save_table_variants(headline_rmse, output_dir / "rmse_headline_variables")
    save_table_variants(relative_rmse, output_dir / "relative_rmse_vs_glp")
    if not hyper_summary.empty:
        save_table_variants(hyper_summary, output_dir / "hyperparameter_summary")

    plot_relative_rmse(relative_rmse, output_dir)
    plot_hyperparameter_paths(hyperparameters, output_dir)
    return output_dir
