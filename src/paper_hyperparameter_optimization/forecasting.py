from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_PARAM_SPACE_BOUNDS,
    MBFVAR_TRANSFORMS,
    MAX_FORECAST_HORIZON_MONTHS,
    MAX_FORECAST_HORIZON_QUARTERS,
    PAPER_ACTUAL_VINTAGE,
    PAPER_HYPERPARAMETERS,
    PAPER_NBURN_PERC,
    PAPER_NLAGS,
    PAPER_NSIM,
    PAPER_TEMPORAL_AGGREGATION,
    PAPER_THINING,
    QUARTERLY_SERIES,
    REALTIME_PANEL_PATH,
    SERIES_BY_CODE,
    forecast_origin_dates,
    origin_group,
    resolve_project_path,
)
from .data_utils import build_model_input_frames, build_quarterly_evaluation_frame, load_realtime_panel


def parse_csv_list(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_csv_int_list(value: str | None, default: list[int]) -> list[int]:
    if not value:
        return default
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_positive_int(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"\d+", value)
    if not match:
        return None
    parsed = int(match.group())
    return parsed if parsed > 0 else None


def detect_slurm_parallel_slots() -> int | None:
    for variable_name in ("SLURM_NTASKS", "SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE", "SLURM_CPUS_PER_TASK"):
        parsed = parse_positive_int(os.getenv(variable_name))
        if parsed is not None:
            return parsed
    return None


def resolve_parallel_settings(
    n_origins: int,
    requested_n_workers: int | None,
    requested_optimization_njobs: int | None,
) -> tuple[int, int]:
    slurm_parallel_slots = detect_slurm_parallel_slots()
    default_parallel_slots = slurm_parallel_slots or 1

    n_workers = requested_n_workers if requested_n_workers and requested_n_workers > 0 else default_parallel_slots
    n_workers = max(1, min(n_workers, n_origins))

    if requested_optimization_njobs and requested_optimization_njobs > 0:
        optimization_njobs = requested_optimization_njobs
    elif slurm_parallel_slots:
        optimization_njobs = max(1, slurm_parallel_slots // n_workers)
    else:
        optimization_njobs = 1

    return n_workers, optimization_njobs


def compute_quarterly_metrics(level_frame: pd.DataFrame) -> pd.DataFrame:
    metrics = level_frame.copy()
    for column in metrics.columns:
        spec = SERIES_BY_CODE[column]
        if spec.evaluation_transform == "growth":
            metrics[column] = 100.0 * np.log(metrics[column]).diff()
    return metrics


def make_param_space(bounds: dict[str, tuple[float, float]]) -> dict[str, Any]:
    from scipy.stats import uniform

    return {
        # Mango reads scipy distribution parameters from dist.args.
        # uniform(loc=..., scale=...) stores these in kwds, which Mango
        # fails to unpack. Positional construction keeps args=(loc, scale).
        key: uniform(lower, upper - lower)
        for key, (lower, upper) in bounds.items()
    }


def make_data_in(quarterly: pd.DataFrame, monthly: pd.DataFrame):
    import MBFVAR

    return MBFVAR.mbfvar_data(
        [quarterly, monthly],
        MBFVAR_TRANSFORMS,
        ["Q", "M"],
    )


DEFAULT_MDD_OPTIMIZATION_VARIABLES = ["GDP"]
RMSE_REQUIRED_OPTIMIZATION_VARIABLES = [spec.paper_code for spec in QUARTERLY_SERIES]
ONE_TIME_OPTIMIZATION_STRATEGIES = {"mango_rmse", "mango_rmse_random"}


def default_optimization_variables(strategy: str) -> list[str]:
    if strategy in {"mango_rmse", "mango_rmse_random"}:
        return RMSE_REQUIRED_OPTIMIZATION_VARIABLES.copy()
    return DEFAULT_MDD_OPTIMIZATION_VARIABLES.copy()


def resolve_optimization_variables(strategy: str, optimization_variables: list[str] | None) -> list[str]:
    resolved = []
    for variable in optimization_variables or default_optimization_variables(strategy):
        if variable not in resolved:
            resolved.append(variable)

    if strategy in {"mango_rmse", "mango_rmse_random"}:
        if set(resolved) != set(RMSE_REQUIRED_OPTIMIZATION_VARIABLES):
            required = ",".join(RMSE_REQUIRED_OPTIMIZATION_VARIABLES)
            raise ValueError(
                "MBFVAR's RMSE hyperparameter objective currently requires the full quarterly variable block "
                f"{required}. Smaller quarterly subsets trigger an upstream forecast dimension mismatch and collapse "
                "the optimizer to the fixed 1e10 penalty."
            )
        return RMSE_REQUIRED_OPTIMIZATION_VARIABLES.copy()

    return resolved


def optimize_once_per_experiment(strategy: str) -> bool:
    return strategy in ONE_TIME_OPTIMIZATION_STRATEGIES


def select_initial_hyperparameters(
    strategy: str,
    task_template: dict[str, Any],
    origin_date: pd.Timestamp,
) -> list[list[float]] | None:
    if not optimize_once_per_experiment(strategy):
        return None

    panel = load_realtime_panel(Path(task_template["panel_path"]))
    quarterly, monthly = build_model_input_frames(panel, origin_date)
    data_in = make_data_in(quarterly, monthly)

    import MBFVAR

    optimizer_model = MBFVAR.MixedFrequencyBVAR(
        task_template["optimization_nsim"],
        task_template["nburn_perc"],
        task_template["nlags"],
        task_template["thining"],
    )
    return select_hyperparameters(strategy, optimizer_model, data_in, task_template)


def hyperparameter_record(origin_date: pd.Timestamp, strategy: str, hyperparameters: list[list[float]]) -> dict[str, Any]:
    base = {
        "forecast_origin": pd.Timestamp(origin_date).strftime("%Y-%m-%d"),
        "group": origin_group(pd.Timestamp(origin_date)),
        "strategy": strategy,
    }
    if hyperparameters:
        values = hyperparameters[0]
        base.update(
            {
                "lambda1_1": float(values[0]),
                "lambda2_1": float(values[1]),
                "lambda3_1": float(values[2]),
                "lambda4_1": float(values[3]),
                "lambda5_1": float(values[4]),
            }
        )
    return base


def select_hyperparameters(
    strategy: str,
    model,
    data_in,
    args: dict[str, Any],
) -> list[list[float]]:
    if strategy == "paper":
        return [PAPER_HYPERPARAMETERS]

    param_space = make_param_space(DEFAULT_PARAM_SPACE_BOUNDS)
    optimization_vars = args["optimization_variables"]
    if strategy == "mango_mdd":
        return model.update_hyperparameters_mango(
            data_in,
            param_space=param_space,
            init_points=args["optimization_init_points"],
            n_iter=args["optimization_iterations"],
            nsim=args["optimization_nsim"],
            njobs=args["optimization_njobs"],
            var_of_interest=optimization_vars,
            temp_agg=args["temp_agg"],
            save=False,
        )
    if strategy == "mango_rmse":
        return model.update_hyperparameters_mango_rmse(
            data_in,
            param_space=param_space,
            H=args["optimization_horizon_quarters"],
            init_points=args["optimization_init_points"],
            n_iter=args["optimization_iterations"],
            nsim=args["optimization_nsim"],
            njobs=args["optimization_njobs"],
            var_of_interest=optimization_vars,
            temp_agg=args["temp_agg"],
            h_eval=args["optimization_eval_horizon_quarters"],
            n_eval=args["optimization_n_eval"],
            save=False,
        )
    if strategy == "mango_rmse_random":
        return model.update_hyperparameters_mango_rmse_random(
            data_in,
            param_space=param_space,
            H=args["optimization_horizon_quarters"],
            init_points=args["optimization_init_points"],
            n_iter=args["optimization_iterations"],
            nsim=args["optimization_nsim"],
            njobs=args["optimization_njobs"],
            var_of_interest=optimization_vars,
            temp_agg=args["temp_agg"],
            h_eval=args["optimization_eval_horizon_quarters"],
            n_eval=args["optimization_n_eval"],
            min_T=args["optimization_min_t"],
            random_seed=args["optimization_random_seed"],
            save=False,
        )
    raise ValueError(f"Unsupported strategy: {strategy}")


def extract_forecasts(
    strategy: str,
    origin_date: pd.Timestamp,
    model,
    actual_levels: pd.DataFrame,
) -> pd.DataFrame:
    current_quarter = origin_date.to_period("Q")
    prediction_frames = {
        "mean": model.YY_mean_agg,
        "median": model.YY_median_agg,
        "p95": model.YY_095_agg,
        "p84": model.YY_084_agg,
        "p16": model.YY_016_agg,
        "p05": model.YY_005_agg,
    }
    metric_frames = {name: compute_quarterly_metrics(frame) for name, frame in prediction_frames.items()}
    actual_metrics = compute_quarterly_metrics(actual_levels)

    rows: list[dict[str, Any]] = []
    for horizon_quarters in range(1, MAX_FORECAST_HORIZON_QUARTERS + 1):
        target_quarter = current_quarter + (horizon_quarters - 1)
        if target_quarter not in actual_levels.index:
            continue
        if target_quarter not in prediction_frames["mean"].index:
            continue

        for variable in prediction_frames["mean"].columns:
            row = {
                "strategy": strategy,
                "forecast_origin": origin_date.strftime("%Y-%m-%d"),
                "group": origin_group(origin_date),
                "target_quarter": str(target_quarter),
                "horizon_quarters": horizon_quarters,
                "variable": variable,
                "actual_level": actual_levels.at[target_quarter, variable],
                "actual_metric": actual_metrics.at[target_quarter, variable],
            }
            for frame_name, frame in prediction_frames.items():
                row[f"{frame_name}_level"] = frame.at[target_quarter, variable]
            for frame_name, frame in metric_frames.items():
                row[f"{frame_name}_metric"] = frame.at[target_quarter, variable]
            row["error_metric"] = row["mean_metric"] - row["actual_metric"]
            rows.append(row)

    return pd.DataFrame(rows)


def _run_origin_task(task: dict[str, Any]) -> dict[str, Any]:
    try:
        origin_date = pd.Timestamp(task["origin_date"])
        panel = load_realtime_panel(Path(task["panel_path"]))
        actual_levels = build_quarterly_evaluation_frame(panel, pd.Timestamp(task["actual_vintage"]))
        quarterly, monthly = build_model_input_frames(panel, origin_date)

        import MBFVAR

        data_in = make_data_in(quarterly, monthly)
        fixed_hyperparameters = task.get("fixed_hyperparameters")
        if fixed_hyperparameters is None:
            optimizer_model = MBFVAR.MixedFrequencyBVAR(
                task["optimization_nsim"],
                task["nburn_perc"],
                task["nlags"],
                task["thining"],
            )
            hyperparameters = select_hyperparameters(task["strategy"], optimizer_model, data_in, task)
        else:
            hyperparameters = fixed_hyperparameters

        model = MBFVAR.MixedFrequencyBVAR(
            task["fit_nsim"],
            task["nburn_perc"],
            task["nlags"],
            task["thining"],
        )
        model.fit(data_in, hyp=hyperparameters, temp_agg=task["temp_agg"])
        model.forecast(task["forecast_horizon_months"])
        model.aggregate(frequency="Q")

        forecasts = extract_forecasts(task["strategy"], origin_date, model, actual_levels)
        hyper_record = hyperparameter_record(origin_date, task["strategy"], hyperparameters)
        return {
            "forecast_rows": forecasts.to_dict(orient="records"),
            "hyperparameters": hyper_record,
            "error": None,
        }
    except Exception as exc:  # pragma: no cover
        return {
            "forecast_rows": [],
            "hyperparameters": {},
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_recursive_experiment(
    strategy: str,
    output_dir: Path,
    panel_path: Path = REALTIME_PANEL_PATH,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    fit_nsim: int = PAPER_NSIM,
    nburn_perc: float = PAPER_NBURN_PERC,
    nlags: list[int] | None = None,
    thining: int = PAPER_THINING,
    forecast_horizon_months: int = MAX_FORECAST_HORIZON_MONTHS,
    actual_vintage: pd.Timestamp = PAPER_ACTUAL_VINTAGE,
    optimization_nsim: int = 1000,
    optimization_init_points: int = 5,
    optimization_iterations: int = 15,
    optimization_njobs: int | None = None,
    optimization_horizon_quarters: int = 4,
    optimization_eval_horizon_quarters: int | None = None,
    optimization_n_eval: int = 3,
    optimization_min_t: int | None = None,
    optimization_random_seed: int | None = None,
    optimization_variables: list[str] | None = None,
    temp_agg: str = PAPER_TEMPORAL_AGGREGATION,
    n_workers: int | None = None,
) -> Path:
    output_dir = resolve_project_path(output_dir)
    panel_path = resolve_project_path(panel_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    available_origins = forecast_origin_dates()
    resolved_start = start if start is not None else available_origins[0]
    resolved_end = end if end is not None else available_origins[-1]
    origins = forecast_origin_dates(resolved_start, resolved_end)
    resolved_n_workers, resolved_optimization_njobs = resolve_parallel_settings(
        len(origins),
        requested_n_workers=n_workers,
        requested_optimization_njobs=optimization_njobs,
    )
    task_template = {
        "strategy": strategy,
        "panel_path": str(panel_path),
        "actual_vintage": str(pd.Timestamp(actual_vintage).date()),
        "fit_nsim": fit_nsim,
        "nburn_perc": nburn_perc,
        "nlags": nlags or PAPER_NLAGS,
        "thining": thining,
        "forecast_horizon_months": forecast_horizon_months,
        "optimization_nsim": optimization_nsim,
        "optimization_init_points": optimization_init_points,
        "optimization_iterations": optimization_iterations,
        "optimization_njobs": resolved_optimization_njobs,
        "optimization_horizon_quarters": optimization_horizon_quarters,
        "optimization_eval_horizon_quarters": optimization_eval_horizon_quarters,
        "optimization_n_eval": optimization_n_eval,
        "optimization_min_t": optimization_min_t,
        "optimization_random_seed": optimization_random_seed,
        "optimization_variables": resolve_optimization_variables(strategy, optimization_variables),
        "temp_agg": temp_agg,
    }

    has_origins = len(origins) > 0
    shared_hyperparameters = None
    if has_origins:
        shared_hyperparameters = select_initial_hyperparameters(strategy, task_template, origins[0])

    tasks = [
        {
            **task_template,
            "origin_date": origin.strftime("%Y-%m-%d"),
            "fixed_hyperparameters": shared_hyperparameters,
        }
        for origin in origins
    ]

    forecast_rows: list[dict[str, Any]] = []
    hyper_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    if resolved_n_workers == 1:
        iterator = ((task["origin_date"], _run_origin_task(task)) for task in tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=resolved_n_workers)
        futures = {executor.submit(_run_origin_task, task): task for task in tasks}

        def _iterator():
            try:
                for future in as_completed(futures):
                    yield futures[future]["origin_date"], future.result()
            finally:
                executor.shutdown(wait=True)

        iterator = _iterator()

    for origin_label, result in iterator:
        if result["error"]:
            errors.append({"forecast_origin": origin_label, "error": result["error"]})
            continue
        forecast_rows.extend(result["forecast_rows"])
        hyper_rows.append(result["hyperparameters"])

    forecasts = pd.DataFrame(forecast_rows)
    if not forecasts.empty:
        forecasts = forecasts.sort_values(["forecast_origin", "variable", "horizon_quarters"])
    hyperparameters = pd.DataFrame(hyper_rows)
    if not hyperparameters.empty:
        hyperparameters = hyperparameters.sort_values("forecast_origin")
    forecast_path = output_dir / "forecast_panel.csv"
    hyper_path = output_dir / "selected_hyperparameters.csv"
    error_path = output_dir / "failed_origins.csv"
    metadata_path = output_dir / "run_metadata.json"

    forecasts.to_csv(forecast_path, index=False)
    hyperparameters.to_csv(hyper_path, index=False)
    pd.DataFrame(errors).to_csv(error_path, index=False)

    metadata = {
        "strategy": strategy,
        "panel_path": str(panel_path),
        "actual_vintage": pd.Timestamp(actual_vintage).strftime("%Y-%m-%d"),
        "fit_nsim": fit_nsim,
        "nburn_perc": nburn_perc,
        "nlags": nlags or PAPER_NLAGS,
        "thining": thining,
        "forecast_horizon_months": forecast_horizon_months,
        "optimization_nsim": optimization_nsim,
        "optimization_init_points": optimization_init_points,
        "optimization_iterations": optimization_iterations,
        "optimization_njobs": resolved_optimization_njobs,
        "optimization_horizon_quarters": optimization_horizon_quarters,
        "optimization_eval_horizon_quarters": optimization_eval_horizon_quarters,
        "optimization_n_eval": optimization_n_eval,
        "optimization_min_t": optimization_min_t,
        "optimization_random_seed": optimization_random_seed,
        "optimization_variables": resolve_optimization_variables(strategy, optimization_variables),
        "hyperparameters_selected_once": optimize_once_per_experiment(strategy),
        "hyperparameter_selection_origin": (origins[0].strftime("%Y-%m-%d") if has_origins and optimize_once_per_experiment(strategy) else None),
        "temp_agg": temp_agg,
        "n_workers": resolved_n_workers,
        "n_origins_requested": len(origins),
        "n_origins_completed": int(hyperparameters.shape[0]),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_dir


def build_common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--panel-path", type=Path, default=REALTIME_PANEL_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--fit-nsim", type=int, default=PAPER_NSIM)
    parser.add_argument("--nburn-perc", type=float, default=PAPER_NBURN_PERC)
    parser.add_argument("--forecast-horizon-months", type=int, default=MAX_FORECAST_HORIZON_MONTHS)
    parser.add_argument("--actual-vintage", type=str, default=PAPER_ACTUAL_VINTAGE.strftime("%Y-%m-%d"))
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Forecast-origin worker count. Defaults to the full Slurm task allocation when available, otherwise 1.",
    )
    return parser


def build_optimizer_parser(description: str) -> argparse.ArgumentParser:
    parser = build_common_parser(description)
    parser.add_argument("--optimization-nsim", type=int, default=1000)
    parser.add_argument("--optimization-init-points", type=int, default=5)
    parser.add_argument("--optimization-iterations", type=int, default=15)
    parser.add_argument(
        "--optimization-njobs",
        type=int,
        default=None,
        help="Per-origin optimizer parallelism. Defaults to the remaining Slurm allocation after splitting across workers, otherwise 1.",
    )
    parser.add_argument("--optimization-horizon-quarters", type=int, default=4)
    parser.add_argument(
        "--optimization-variables",
        type=str,
        default=None,
        help=(
            "Comma-separated variables passed to the MBFVAR hyperparameter objective. "
            "Mango MDD defaults to GDP. Mango RMSE variants default to the full quarterly block GDP,INVFIX,GOV "
            "and reject smaller subsets because the upstream MBFVAR forecast code fails on them."
        ),
    )
    return parser


def parse_dates(start: str | None, end: str | None) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    start_ts = pd.Timestamp(start) if start else None
    end_ts = pd.Timestamp(end) if end else None
    return start_ts, end_ts


def run_from_namespace(strategy: str, namespace: argparse.Namespace) -> Path:
    start, end = parse_dates(namespace.start, namespace.end)
    kwargs = {
        "strategy": strategy,
        "output_dir": namespace.output_dir,
        "panel_path": namespace.panel_path,
        "start": start,
        "end": end,
        "fit_nsim": namespace.fit_nsim,
        "nburn_perc": namespace.nburn_perc,
        "forecast_horizon_months": namespace.forecast_horizon_months,
        "actual_vintage": pd.Timestamp(namespace.actual_vintage),
        "n_workers": namespace.n_workers,
    }
    if hasattr(namespace, "optimization_nsim"):
        kwargs.update(
            {
                "optimization_nsim": namespace.optimization_nsim,
                "optimization_init_points": namespace.optimization_init_points,
                "optimization_iterations": namespace.optimization_iterations,
                "optimization_njobs": namespace.optimization_njobs,
                "optimization_horizon_quarters": namespace.optimization_horizon_quarters,
                "optimization_eval_horizon_quarters": getattr(namespace, "optimization_eval_horizon_quarters", None),
                "optimization_n_eval": getattr(namespace, "optimization_n_eval", 3),
                "optimization_min_t": getattr(namespace, "optimization_min_t", None),
                "optimization_random_seed": getattr(namespace, "optimization_random_seed", None),
                "optimization_variables": parse_csv_list(namespace.optimization_variables, []),
            }
        )
    return run_recursive_experiment(**kwargs)
