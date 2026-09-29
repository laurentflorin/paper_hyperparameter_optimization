"""Comparison tables and figures for the GLP hyperparameter-selection study.

Scores the recursive out-of-sample forecasts of every strategy on
vintage-consistent targets (see :mod:`scoring`): ``100 * log`` series are scored
on the change from the forecast jump-off, reported as annualized average growth
over the horizon; rate series are scored in levels. Raw level errors against the
evaluation vintage are dominated by base-year / unit changes between vintages and
are kept only as ``rmse_level`` for reference.

Outputs: score tables (RMSE, CRPS, log score, 68/90% coverage), relative scores
versus the GLP baseline with Diebold-Mariano tests (Holm-adjusted within each
model size x variable x metric family), Model Confidence Sets, hyperparameter
summaries including the achieved log MDD, and paper-style figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import model_series, resolve_project_path
from .scoring import (
    adjusted_actual,
    crps_gaussian,
    diebold_mariano,
    growth_scale,
    holm_adjust,
    log_score_gaussian,
    model_confidence_set,
)

MODEL_ORDER = ["paper", "paper_plugin", "mango_mdd", "mango_rmse", "mango_rmse_random"]
MODEL_LABELS = {
    "paper": "GLP Marginal Likelihood",
    "paper_plugin": "GLP ML mode (plug-in)",
    "mango_mdd": "Mango MDD",
    "mango_rmse": "Mango RMSE",
    "mango_rmse_random": "Mango RMSE Random",
}
_Z68 = 0.9944578832097535  # N(0,1) quantile for a central 68% band (p16/p84)
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


def _transform_map() -> dict[str, bool]:
    return {spec.code: spec.transform == "log" for spec in model_series("large")}


def _jumpoff_frame(forecasts: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Real-time jump-off values for (origin, size, variable) rebuilt from the panel."""
    from .data_utils import build_glp_estimation_matrix

    rows = []
    for (origin, size), _ in forecasts.groupby(["forecast_origin", "model_size"]):
        y, codes, _q = build_glp_estimation_matrix(panel, pd.Timestamp(origin), size)
        rows.extend({"forecast_origin": origin, "model_size": size, "variable": c, "jumpoff": v} for c, v in zip(codes, y[-1]))
    return pd.DataFrame(rows)


def prepare_scores(forecasts: pd.DataFrame, panel: pd.DataFrame | None = None, actual_vintage: str | None = None) -> pd.DataFrame:
    """Add vintage-consistent error and density-score columns to a forecast panel.

    Panels written before the jump-off columns existed are backfilled from the
    real-time ``panel`` (required in that case). Adds ``err_g`` (annualized
    growth / level error), ``crps_g`` (CRPS in the same units), ``logscore``,
    ``cov68`` / ``cov90`` and ``density_source`` (``draws`` when the CRPS was
    computed from the predictive draws, ``gaussian`` when approximated from the
    stored quantiles).
    """
    frame = forecasts.copy()
    is_log_map = _transform_map()
    frame["is_log"] = frame["variable"].map(is_log_map).fillna(False).astype(bool)
    if "error_adj" not in frame.columns or frame["error_adj"].isna().all():
        if panel is None:
            raise ValueError("Forecast panels without jump-off columns need the real-time data panel to be scored.")
        from .data_utils import build_glp_actual_frame

        frame = frame.drop(columns=[c for c in ("jumpoff", "actual_jumpoff") if c in frame.columns])
        frame = frame.merge(_jumpoff_frame(frame, panel), on=["forecast_origin", "model_size", "variable"], how="left")
        vintage = pd.Timestamp(actual_vintage or "2023-01-01")
        jump_q = pd.PeriodIndex(frame["target_quarter"], freq="Q") - frame["horizon_quarters"].to_numpy()
        actual_jump = np.full(len(frame), np.nan)
        for size, idx in frame.groupby("model_size").groups.items():
            actual = build_glp_actual_frame(panel, vintage, size)
            lookup = actual.stack()
            keys = list(zip(jump_q[frame.index.get_indexer(idx)], frame.loc[idx, "variable"]))
            actual_jump[frame.index.get_indexer(idx)] = [lookup.get(k, np.nan) for k in keys]
        frame["actual_jumpoff"] = actual_jump
        frame["actual_adj"] = adjusted_actual(frame["actual"], frame["actual_jumpoff"], frame["jumpoff"], frame["is_log"])
        frame["error_adj"] = frame["mean"] - frame["actual_adj"]
    scale = growth_scale(frame["horizon_quarters"], frame["is_log"])
    frame["err_g"] = frame["error_adj"] * scale
    if "crps" in frame.columns and frame["crps"].notna().any():
        frame["crps_g"] = frame["crps"] * scale
        frame["density_source"] = "draws"
    else:
        sd = (frame["p84"] - frame["p16"]) / (2.0 * _Z68)
        frame["crps_g"] = crps_gaussian(frame["mean"], sd, frame["actual_adj"]) * scale
        frame["logscore"] = log_score_gaussian(frame["mean"], sd, frame["actual_adj"])
        frame["density_source"] = "gaussian"
    target = frame["actual_adj"]
    frame["cov68"] = ((target >= frame["p16"]) & (target <= frame["p84"])).astype(float).where(target.notna())
    frame["cov90"] = ((target >= frame["p05"]) & (target <= frame["p95"])).astype(float).where(target.notna())
    return frame


def _group_cols(frame: pd.DataFrame) -> list[str]:
    cols = ["model", "model_size", "variable", "horizon_quarters"]
    if "optimization_horizon" in frame.columns:
        cols.append("optimization_horizon")
    return cols


def variant_key(frame: pd.DataFrame) -> pd.Series:
    """``model`` or ``model_<optimization horizon>`` -- one key per forecast set."""
    if "optimization_horizon" not in frame.columns:
        return frame["model"].astype(str)
    opt = frame["optimization_horizon"]
    return np.where(opt.isna(), frame["model"].astype(str), frame["model"].astype(str) + "_" + opt.astype(str))


def compute_rmse_table(forecasts: pd.DataFrame) -> pd.DataFrame:
    """RMSE per model/variable/horizon on ``err_g`` when available (see
    :func:`prepare_scores`), otherwise on the raw ``error`` column."""
    column = "err_g" if "err_g" in forecasts.columns else "error"
    return (
        forecasts.dropna(subset=[column])
        .groupby(_group_cols(forecasts), as_index=False, dropna=False)
        .agg(rmse=(column, lambda s: float(np.sqrt(np.mean(np.square(s))))), n=(column, "size"))
    )


def compute_score_table(scored: pd.DataFrame) -> pd.DataFrame:
    """RMSE, mean CRPS, mean log score, coverage and the raw level RMSE."""
    valid = scored.dropna(subset=["err_g"])
    return valid.groupby(_group_cols(valid), as_index=False, dropna=False).agg(
        rmse=("err_g", lambda s: float(np.sqrt(np.mean(np.square(s))))),
        crps=("crps_g", "mean"),
        logscore=("logscore", "mean"),
        cov68=("cov68", "mean"),
        cov90=("cov90", "mean"),
        bias=("err_g", "mean"),
        rmse_level=("error", lambda s: float(np.sqrt(np.mean(np.square(s))))),
        n=("err_g", "size"),
    )


_METRICS = {  # metric -> (per-origin loss column, loss transform); lower loss is better
    "sq_error": ("err_g", np.square),
    "crps": ("crps_g", lambda x: x),
    "logscore": ("logscore", lambda x: -x),
}


def compute_relative_scores(scored: pd.DataFrame, baseline_model: str = "paper") -> pd.DataFrame:
    """Every forecast set versus ``baseline_model`` on matched origins.

    ``rel_rmse_pct`` / ``rel_crps_pct`` are percentage changes (negative =
    better than the baseline); ``d_logscore`` is the mean log-score gain.
    ``dm_*`` are Diebold-Mariano statistics on the loss differential (negative =
    better) with two-sided p-values, Holm-adjusted (``*_p_holm``) within each
    model size x variable x metric family.
    """
    frame = scored.dropna(subset=["err_g"]).copy()
    frame["variant"] = variant_key(frame)
    keys = ["forecast_origin", "model_size", "variable", "horizon_quarters"]
    base = frame[frame["model"] == baseline_model].set_index(keys)
    rows = []
    for (variant, size, variable, horizon), group in frame.groupby(["variant", "model_size", "variable", "horizon_quarters"]):
        if group["model"].iloc[0] == baseline_model:
            continue
        g = group.set_index(keys)
        b = base.reindex(g.index)
        ok = b["err_g"].notna().to_numpy()
        if ok.sum() == 0:
            continue
        g, b = g[ok], b[ok]
        row = {
            "variant": variant,
            "model": group["model"].iloc[0],
            "optimization_horizon": group["optimization_horizon"].iloc[0] if "optimization_horizon" in group else np.nan,
            "model_size": size,
            "variable": variable,
            "horizon_quarters": horizon,
            "n": int(ok.sum()),
            "rmse": float(np.sqrt(np.mean(g["err_g"] ** 2))),
            "baseline_rmse": float(np.sqrt(np.mean(b["err_g"] ** 2))),
            "crps": float(g["crps_g"].mean()),
            "baseline_crps": float(b["crps_g"].mean()),
            "d_logscore": float(g["logscore"].mean() - b["logscore"].mean()),
        }
        row["rel_rmse_pct"] = 100.0 * (row["rmse"] / row["baseline_rmse"] - 1.0)
        row["rel_crps_pct"] = 100.0 * (row["crps"] / row["baseline_crps"] - 1.0)
        for metric, (column, loss) in _METRICS.items():
            stat, pvalue = diebold_mariano(loss(g[column].to_numpy()), loss(b[column].to_numpy()), int(horizon))
            row[f"dm_{metric}"] = stat
            row[f"dm_{metric}_p"] = pvalue
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for metric in _METRICS:
        out[f"dm_{metric}_p_holm"] = np.nan
        for _, idx in out.groupby(["model_size", "variable"]).groups.items():
            out.loc[idx, f"dm_{metric}_p_holm"] = holm_adjust(out.loc[idx, f"dm_{metric}_p"].to_numpy())
    return out.sort_values(["model_size", "variable", "horizon_quarters", "variant"]).reset_index(drop=True)


def compute_mcs_table(scored: pd.DataFrame, metric: str = "sq_error", alpha: float = 0.10, seed: int = 0) -> pd.DataFrame:
    """Model Confidence Set per model size x variable x horizon across all forecast sets."""
    column, loss = _METRICS[metric]
    frame = scored.dropna(subset=[column]).copy()
    frame["variant"] = variant_key(frame)
    frame["loss"] = loss(frame[column].to_numpy())
    rows = []
    for (size, variable, horizon), group in frame.groupby(["model_size", "variable", "horizon_quarters"]):
        losses = group.pivot_table(index="forecast_origin", columns="variant", values="loss")
        mcs = model_confidence_set(losses, alpha=alpha, seed=seed)
        mean_loss = losses.dropna().mean()
        for variant, rec in mcs.iterrows():
            rows.append(
                {
                    "model_size": size,
                    "variable": variable,
                    "horizon_quarters": horizon,
                    "variant": variant,
                    "mean_loss": float(mean_loss.get(variant, np.nan)),
                    "mcs_pvalue": float(rec["mcs_pvalue"]),
                    "in_mcs": bool(rec["in_mcs"]),
                }
            )
    return pd.DataFrame(rows)


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
    """Mean/std/min/max of the selected hyperparameters and, when recorded, of the
    achieved log MDD and of its gap to the ``paper`` (marginal-likelihood mode)
    value at the same origin (``log_mdd_gap``; <= 0 up to optimizer precision)."""
    if hyperparameters.empty:
        return hyperparameters
    hyperparameters = hyperparameters.copy()
    if "log_mdd" in hyperparameters.columns:
        paper = hyperparameters[hyperparameters["model"] == "paper"]
        if not paper.empty:
            ref = paper.set_index(["model_size", "forecast_origin"])["log_mdd"]
            keys = pd.MultiIndex.from_frame(hyperparameters[["model_size", "forecast_origin"]])
            hyperparameters["log_mdd_gap"] = hyperparameters["log_mdd"].to_numpy() - ref.reindex(keys).to_numpy()
    value_columns = [c for c in ("lambda", "theta", "miu", "log_mdd", "log_mdd_gap") if c in hyperparameters.columns]
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
    """Relative RMSE by horizon; accepts the output of :func:`compute_relative_rmse`."""
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


def create_glp_comparison_report(
    experiment_dirs: dict[str, Path | None],
    output_dir: Path,
    *,
    panel_path: Path | None = None,
    baseline_model: str = "paper",
    mcs_alpha: float = 0.10,
) -> Path:
    output_dir = resolve_project_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    forecasts = load_forecast_panels(experiment_dirs)
    hyperparameters = load_hyperparameters(experiment_dirs)

    panel = None
    if "error_adj" not in forecasts.columns or forecasts["error_adj"].isna().any():
        from .config import GLP_REALTIME_PANEL_PATH
        from .data_utils import load_glp_realtime_panel

        panel = load_glp_realtime_panel(panel_path or GLP_REALTIME_PANEL_PATH)
    if "error_adj" in forecasts.columns and forecasts["error_adj"].isna().any():
        # Mixed old/new panels: rescore the old ones from the data panel.
        old = forecasts["error_adj"].isna()
        forecasts = pd.concat(
            [prepare_scores(forecasts[~old]), prepare_scores(forecasts[old].drop(columns=["error_adj"]), panel)],
            ignore_index=True,
        )
        scored = forecasts
    else:
        scored = prepare_scores(forecasts, panel)
    forecasts = scored

    rmse_table = compute_rmse_table(forecasts)
    relative_rmse = compute_relative_rmse(rmse_table, baseline_model)
    hyper_summary = compute_hyperparameter_summary(hyperparameters)
    score_table = compute_score_table(scored)
    relative_scores = compute_relative_scores(scored, baseline_model)
    save_table_variants(score_table, output_dir / "scores_all_variables")
    if not relative_scores.empty:
        save_table_variants(relative_scores, output_dir / "relative_scores_vs_glp")
    for metric in ("sq_error", "crps"):
        save_table_variants(compute_mcs_table(scored, metric, mcs_alpha), output_dir / f"mcs_{metric}")

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
