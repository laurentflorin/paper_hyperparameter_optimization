from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HEADLINE_VARIABLES = ["GDP", "CPI", "UNR", "FF"]
MODEL_ORDER = ["paper", "mango_mdd", "mango_rmse"]
MODEL_LABELS = {
    "paper": "Paper Hyperparameters",
    "mango_mdd": "Mango MDD",
    "mango_rmse": "Mango RMSE",
}
MODEL_COLORS = {
    "paper": "#4c566a",
    "mango_mdd": "#c0392b",
    "mango_rmse": "#1f618d",
}


def load_forecast_panels(experiment_dirs: dict[str, Path]) -> pd.DataFrame:
    frames = []
    for model_name, directory in experiment_dirs.items():
        frame = pd.read_csv(directory / "forecast_panel.csv")
        frame["model"] = model_name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_hyperparameters(experiment_dirs: dict[str, Path]) -> pd.DataFrame:
    frames = []
    for model_name, directory in experiment_dirs.items():
        hyper_path = directory / "selected_hyperparameters.csv"
        if hyper_path.exists():
            frame = pd.read_csv(hyper_path)
            frame["model"] = model_name
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def compute_rmse_table(forecasts: pd.DataFrame) -> pd.DataFrame:
    return (
        forecasts.dropna(subset=["error_metric"])
        .groupby(["model", "group", "variable", "horizon_quarters"], as_index=False)
        .agg(rmse=("error_metric", lambda series: float(np.sqrt(np.mean(np.square(series))))))
    )


def compute_relative_rmse(rmse_table: pd.DataFrame, baseline_model: str = "paper") -> pd.DataFrame:
    baseline = rmse_table[rmse_table["model"] == baseline_model].rename(columns={"rmse": "baseline_rmse"})
    merged = rmse_table.merge(
        baseline[["group", "variable", "horizon_quarters", "baseline_rmse"]],
        on=["group", "variable", "horizon_quarters"],
        how="left",
    )
    merged["relative_rmse_pct"] = 100.0 * (merged["rmse"] - merged["baseline_rmse"]) / merged["baseline_rmse"]
    return merged


def compute_hyperparameter_summary(hyperparameters: pd.DataFrame) -> pd.DataFrame:
    if hyperparameters.empty:
        return hyperparameters
    value_columns = [column for column in hyperparameters.columns if column.startswith("lambda")]
    summary = hyperparameters.groupby("model")[value_columns].agg(["mean", "std", "min", "max"])
    summary.columns = ["_".join(column).strip() for column in summary.columns.to_flat_index()]
    summary = summary.reset_index()
    return summary


def save_table_variants(frame: pd.DataFrame, output_stem: Path, index: bool = False) -> None:
    frame.to_csv(output_stem.with_suffix(".csv"), index=index)
    output_stem.with_suffix(".tex").write_text(frame.to_latex(index=index, float_format="%.4f"), encoding="utf-8")
    output_stem.with_suffix(".md").write_text(frame.to_markdown(index=index), encoding="utf-8")


def plot_relative_rmse_by_group(relative_rmse: pd.DataFrame, output_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    for group, group_frame in relative_rmse[relative_rmse["variable"].isin(HEADLINE_VARIABLES)].groupby("group"):
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
        axes = axes.flatten()

        for axis, variable in zip(axes, HEADLINE_VARIABLES):
            var_frame = group_frame[group_frame["variable"] == variable]
            baseline = var_frame[var_frame["model"] == "paper"]
            axis.axhline(0.0, color="#999999", linewidth=1.0, linestyle="--")
            if not baseline.empty:
                axis.plot(
                    baseline["horizon_quarters"],
                    baseline["relative_rmse_pct"],
                    color=MODEL_COLORS["paper"],
                    linewidth=1.5,
                    label=MODEL_LABELS["paper"],
                )
            for model_name in ("mango_mdd", "mango_rmse"):
                model_frame = var_frame[var_frame["model"] == model_name]
                if model_frame.empty:
                    continue
                axis.plot(
                    model_frame["horizon_quarters"],
                    model_frame["relative_rmse_pct"],
                    color=MODEL_COLORS[model_name],
                    linewidth=2.0,
                    label=MODEL_LABELS[model_name],
                )
            axis.set_title(variable)
            axis.set_xlabel("Forecast Horizon (quarters)")
            axis.set_ylabel("Relative RMSE vs paper (%)")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
        fig.suptitle(f"Relative RMSE by Horizon: {group}")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(output_dir / f"relative_rmse_{group.replace(' ', '_').replace('+', 'plus_')}.png", bbox_inches="tight")
        plt.close(fig)


def plot_hyperparameter_paths(hyperparameters: pd.DataFrame, output_dir: Path) -> None:
    if hyperparameters.empty:
        return

    plt.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    for model_name in ("mango_mdd", "mango_rmse"):
        model_frame = hyperparameters[hyperparameters["model"] == model_name].copy()
        if model_frame.empty:
            continue
        model_frame["forecast_origin"] = pd.to_datetime(model_frame["forecast_origin"])
        fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
        axes = axes.flatten()
        for axis, parameter in zip(axes, ["lambda1_1", "lambda2_1", "lambda4_1", "lambda5_1"]):
            axis.plot(model_frame["forecast_origin"], model_frame[parameter], color=MODEL_COLORS[model_name], linewidth=1.8)
            axis.set_title(parameter)
            axis.set_xlabel("Forecast origin")
        fig.suptitle(f"Selected Hyperparameters Over Time: {MODEL_LABELS[model_name]}")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(output_dir / f"{model_name}_hyperparameter_paths.png", bbox_inches="tight")
        plt.close(fig)


def create_comparison_report(
    experiment_dirs: dict[str, Path],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    forecasts = load_forecast_panels(experiment_dirs)
    hyperparameters = load_hyperparameters(experiment_dirs)
    rmse_table = compute_rmse_table(forecasts)
    relative_rmse = compute_relative_rmse(rmse_table)
    hyper_summary = compute_hyperparameter_summary(hyperparameters)

    headline_rmse = rmse_table[rmse_table["variable"].isin(HEADLINE_VARIABLES)].copy()
    headline_rmse = headline_rmse.sort_values(["group", "variable", "horizon_quarters", "model"])

    headline_relative = relative_rmse[relative_rmse["variable"].isin(HEADLINE_VARIABLES)].copy()
    headline_relative = headline_relative.sort_values(["group", "variable", "horizon_quarters", "model"])

    save_table_variants(rmse_table, output_dir / "rmse_all_variables")
    save_table_variants(headline_rmse, output_dir / "rmse_headline_variables")
    save_table_variants(relative_rmse, output_dir / "relative_rmse_vs_paper")
    if not hyper_summary.empty:
        save_table_variants(hyper_summary, output_dir / "hyperparameter_summary")

    plot_relative_rmse_by_group(headline_relative, output_dir)
    plot_hyperparameter_paths(hyperparameters, output_dir)

    return output_dir
