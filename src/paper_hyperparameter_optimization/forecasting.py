from __future__ import annotations

import argparse
import json
import copy
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common_hpo.metadata import classify_failure


from experiment_provenance import (
    deterministic_rng_context,
    runtime_provenance,
    stable_child_seed,
    validate_mbfvar_revision,
)
from .config import (
    DEFAULT_OPTIMIZATION_NSIM,
    DEFAULT_PARAM_SPACE_BOUNDS,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SELECTION_SCHEDULE,
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
    PROJECT_ROOT,
    QUARTERLY_SERIES,
    REALTIME_PANEL_PATH,
    SERIES_BY_CODE,
    SERIES_SPECS,
    VALID_SELECTION_SCHEDULES,
    forecast_origin_dates,
    origin_group,
    resolve_project_path,
)
from .data_utils import build_model_input_frames, build_quarterly_evaluation_frame, load_realtime_panel
from .horizon_mapping import FREQ_RATIO


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
        if column not in SERIES_BY_CODE:
            raise KeyError(f"No evaluation transform is configured for {column}.")
        spec = SERIES_BY_CODE[column]
        if spec.evaluation_transform == "growth":
            invalid = metrics[column].notna() & (metrics[column] <= 0)
            if invalid.any():
                raise ValueError(
                    f"Growth metric for {column} requires strictly positive levels."
                )
            metrics[column] = 100.0 * np.log(metrics[column]).diff()
    return metrics


SUMMARY_QUANTILES = {
    "median": 0.50,
    "p95": 0.95,
    "p84": 0.84,
    "p16": 0.16,
    "p05": 0.05,
}


def summarize_quarterly_draws(
    draw_frames: list[pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Summarize levels and evaluation metrics from joint posterior paths."""
    if not draw_frames:
        raise ValueError("At least one posterior draw is required.")

    index = pd.PeriodIndex(draw_frames[0].index, freq="Q")
    columns = draw_frames[0].columns
    normalized: list[pd.DataFrame] = []
    for draw in draw_frames:
        current = draw.copy()
        current.index = pd.PeriodIndex(current.index, freq="Q")
        if not current.index.equals(index) or not current.columns.equals(columns):
            raise ValueError("Posterior draw paths must have identical quarter and variable keys.")
        normalized.append(current)

    level_stack = np.stack([draw.to_numpy(dtype=float) for draw in normalized], axis=0)
    metric_stack = level_stack.copy()
    for column_index, column in enumerate(columns):
        if column not in SERIES_BY_CODE:
            raise KeyError(f"No evaluation transform is configured for {column}.")
        if SERIES_BY_CODE[column].evaluation_transform == "growth":
            values = level_stack[:, :, column_index]
            invalid = np.isfinite(values) & (values <= 0)
            if invalid.any():
                raise ValueError(f"Growth metric for {column} requires strictly positive posterior levels.")
            metric_stack[:, 0, column_index] = np.nan
            metric_stack[:, 1:, column_index] = 100.0 * np.diff(np.log(values), axis=1)

    def frame_from(values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(values, index=index, columns=columns)

    level_summaries = {"mean": frame_from(np.nanmean(level_stack, axis=0))}
    metric_summaries = {"mean": frame_from(np.nanmean(metric_stack, axis=0))}
    for name, quantile in SUMMARY_QUANTILES.items():
        level_summaries[name] = frame_from(np.nanquantile(level_stack, quantile, axis=0))
        metric_summaries[name] = frame_from(np.nanquantile(metric_stack, quantile, axis=0))
    return level_summaries, metric_summaries


def _back_transform_draw_block(values: np.ndarray, transform_codes: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    codes = np.asarray(transform_codes)
    out[..., codes == 0] = np.exp(out[..., codes == 0])
    out[..., codes == 1] = 100.0 * out[..., codes == 1]
    return out


def aggregate_quarterly_posterior_draws(model) -> list[pd.DataFrame]:
    """Reconstruct complete quarterly level paths while retaining draw identity."""
    cached = getattr(model, "_repo_quarterly_draw_frames", None)
    if cached is not None:
        return [frame.copy() for frame in cached]

    required = (
        "forecast_draws_list",
        "YYactsim_list",
        "lstate_list",
        "YMh_list",
        "valid_draws",
        "freq_ratio_list",
        "Nm_list",
        "select_m_list",
        "select_q",
        "select_list",
        "varlist_list",
        "index_list",
        "temp_agg",
    )
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise RuntimeError(
            "The installed MBFVAR model does not expose the joint draw state "
            f"required for valid transformed intervals: {', '.join(missing)}."
        )

    ratio = int(model.freq_ratio_list[-1])
    if ratio != 3:
        raise NotImplementedError("Repository draw aggregation currently supports monthly-to-quarterly models only.")

    forecast_draws = np.asarray(model.forecast_draws_list[-1])
    valid_draws = [int(draw) for draw in model.valid_draws]
    if forecast_draws.shape[0] != len(valid_draws):
        raise RuntimeError("Forecast draw count does not match MBFVAR's retained posterior draw identifiers.")

    yynow = np.asarray(model.YYactsim_list[-1])
    lstate = np.asarray(model.lstate_list[-1])
    ymh = np.asarray(model.YMh_list[-1])
    if not ymh.size:
        raise RuntimeError("Joint draw reconstruction requires a non-empty monthly observation block.")

    monthly_codes = np.asarray(model.select_m_list[-1])
    quarterly_codes = np.asarray(model.select_q[-1])
    all_codes = np.asarray(model.select_list[-1])
    n_monthly = int(model.Nm_list[-1])
    columns = pd.Index(model.varlist_list[-1])
    full_index = pd.DatetimeIndex(model.index_list[-1])
    draw_frames: list[pd.DataFrame] = []

    for draw_position, posterior_index in enumerate(valid_draws):
        lstate_levels = _back_transform_draw_block(lstate[posterior_index].T, quarterly_codes)
        nowcast_levels = _back_transform_draw_block(
            yynow[posterior_index, 1 : ratio + 1, :n_monthly],
            monthly_codes,
        )
        forecast_levels = _back_transform_draw_block(forecast_draws[draw_position], all_codes)

        correction = int(ymh.shape[0] - lstate_levels[:-ratio].shape[0])
        observed_monthly = ymh[correction:]
        latent_history = lstate_levels[:-ratio]
        if observed_monthly.shape[0] != latent_history.shape[0]:
            raise RuntimeError("Monthly observations and latent quarterly paths are not calendar-aligned.")

        history = np.hstack((observed_monthly, latent_history))
        nowcast = np.hstack((nowcast_levels, lstate_levels[-ratio:]))
        full_values = np.vstack((history, nowcast, forecast_levels))
        if full_values.shape != (len(full_index), len(columns)):
            raise RuntimeError(
                "Reconstructed posterior path shape does not match MBFVAR's forecast calendar."
            )

        monthly_frame = pd.DataFrame(full_values, index=full_index, columns=columns)
        quarter_keys = pd.PeriodIndex(monthly_frame.index, freq="Q")
        row_counts = pd.Series(1, index=quarter_keys).groupby(level=0).sum()
        if model.temp_agg == "mean":
            quarterly_frame = monthly_frame.groupby(quarter_keys).mean()
        elif model.temp_agg == "sum":
            quarterly_frame = monthly_frame.groupby(quarter_keys).sum()
        else:
            raise ValueError(f"Unsupported temporal aggregation: {model.temp_agg}.")
        complete_quarters = row_counts.index[row_counts.eq(ratio)]
        quarterly_frame = quarterly_frame.loc[quarterly_frame.index.intersection(complete_quarters)]
        draw_frames.append(quarterly_frame)

    model._repo_quarterly_draw_frames = [frame.copy() for frame in draw_frames]
    return draw_frames


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


FAIR_OPTIMIZATION_VARIABLES = [spec.paper_code for spec in QUARTERLY_SERIES]
DEFAULT_MDD_OPTIMIZATION_VARIABLES = FAIR_OPTIMIZATION_VARIABLES.copy()
RMSE_REQUIRED_OPTIMIZATION_VARIABLES = FAIR_OPTIMIZATION_VARIABLES.copy()
# The complete quarterly model block that the audited MBFVAR revision requires to
# construct a dimensionally valid mixed-frequency state forecast. A reduced block
# cannot be forecast safely, so this set is always used to build the forecast state.
RMSE_REQUIRED_FORECAST_VARIABLES = FAIR_OPTIMIZATION_VARIABLES.copy()
OPTIMIZED_STRATEGIES = {"mango_mdd", "mango_rmse", "mango_rmse_random"}
RMSE_STRATEGIES = {"mango_rmse", "mango_rmse_random"}

# The selection schedule is an orthogonal option, but each strategy keeps the
# baseline update frequency of the original exercise unless a run overrides it.
# The rolling-RMSE objective needs a held-out evaluation span, so the RMSE
# variants select once on the first origin. The marginal data density is a
# function of the sample available at each origin, so MDD re-selects per origin.
# Pass an explicit selection_schedule to force one common schedule when the
# point of the run is to compare objectives rather than update frequencies.
DEFAULT_SELECTION_SCHEDULE_BY_STRATEGY = {
    "mango_mdd": "per_origin",
    "mango_rmse": "first_origin",
    "mango_rmse_random": "first_origin",
}


def default_selection_schedule(strategy: str) -> str:
    """Return the baseline selection schedule for one strategy."""
    return DEFAULT_SELECTION_SCHEDULE_BY_STRATEGY.get(strategy, DEFAULT_SELECTION_SCHEDULE)


def resolve_selection_schedule(strategy: str, selection_schedule: str | None) -> str:
    """Resolve and validate the schedule, falling back to the strategy baseline."""
    if selection_schedule is None:
        selection_schedule = default_selection_schedule(strategy)
    return validate_selection_schedule(selection_schedule)


def default_optimization_variables(strategy: str) -> list[str]:
    if strategy in OPTIMIZED_STRATEGIES:
        return FAIR_OPTIMIZATION_VARIABLES.copy()
    return []


def _dedupe_preserving_order(variables: list[str]) -> list[str]:
    resolved: list[str] = []
    for variable in variables:
        if variable not in resolved:
            resolved.append(variable)
    return resolved


def resolve_forecast_objective_variables(
    strategy: str,
    *,
    optimization_variables: list[str] | None = None,
    forecast_variables: list[str] | None = None,
    objective_variables: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Resolve the forecast-state block and the objective (loss) subset.

    Two distinct concepts are returned:

    * ``forecast_variables`` -- the complete set required to construct a
      dimensionally valid mixed-frequency state forecast. For the RMSE objective
      the audited MBFVAR revision cannot forecast a reduced block, so this is
      always the full quarterly block ``GDP,INVFIX,GOV``.
    * ``objective_variables`` -- the subset whose forecast errors enter the
      optimization objective. It may contain only ``GDP`` or any valid subset of
      ``forecast_variables``.

    Backward compatibility: the legacy single ``optimization_variables`` argument
    maps to the objective subset while the forecast block is expanded to the full
    quarterly block when dimensionally required. Passing ``optimization_variables``
    together with either explicit argument is rejected.
    """
    legacy_supplied = bool(optimization_variables)
    explicit_supplied = forecast_variables is not None or objective_variables is not None
    if legacy_supplied and explicit_supplied:
        raise ValueError(
            "Pass either the legacy optimization_variables argument or the explicit "
            "forecast_variables/objective_variables pair, not both."
        )

    valid_quarterly = set(FAIR_OPTIMIZATION_VARIABLES)

    if strategy not in OPTIMIZED_STRATEGIES:
        # Non-optimized strategies (e.g. "paper") do not tune on a variable set.
        resolved = _dedupe_preserving_order(optimization_variables or objective_variables or [])
        return resolved, resolved

    if strategy in RMSE_STRATEGIES:
        # Forecast state must span the full quarterly block.
        if forecast_variables is not None:
            resolved_forecast = _dedupe_preserving_order(forecast_variables)
            if set(resolved_forecast) != valid_quarterly:
                required = ",".join(RMSE_REQUIRED_FORECAST_VARIABLES)
                raise ValueError(
                    "The repository RMSE objective requires the full quarterly forecast "
                    f"block {required} because the audited MBFVAR revision cannot forecast "
                    "a reduced block safely."
                )
            resolved_forecast = RMSE_REQUIRED_FORECAST_VARIABLES.copy()
        else:
            resolved_forecast = RMSE_REQUIRED_FORECAST_VARIABLES.copy()

        # Objective subset defaults to the legacy argument, else the full block.
        if objective_variables is not None:
            resolved_objective = _dedupe_preserving_order(objective_variables)
        elif legacy_supplied:
            resolved_objective = _dedupe_preserving_order(optimization_variables)
        else:
            resolved_objective = resolved_forecast.copy()

        if not resolved_objective:
            raise ValueError("At least one objective variable is required.")
        non_subset = [v for v in resolved_objective if v not in set(resolved_forecast)]
        if non_subset:
            raise ValueError(
                "objective_variables must be a subset of the forecast block "
                f"{resolved_forecast}; invalid entries: {non_subset}."
            )
        return resolved_forecast, resolved_objective

    # Marginal-data-density strategy: var_of_interest reduces the fitted system,
    # so the forecast block and objective subset coincide.
    resolved = _dedupe_preserving_order(
        forecast_variables
        if forecast_variables is not None
        else objective_variables
        if objective_variables is not None
        else optimization_variables
        or default_optimization_variables(strategy)
    )
    invalid = [variable for variable in resolved if variable not in valid_quarterly]
    if invalid:
        raise ValueError(f"Optimization variables must belong to the quarterly block: {invalid}.")
    if not resolved:
        raise ValueError("At least one objective variable is required.")
    return resolved, resolved


def resolve_optimization_variables(strategy: str, optimization_variables: list[str] | None) -> list[str]:
    """Backward-compatible helper returning the objective (loss) variable subset."""
    _, objective = resolve_forecast_objective_variables(
        strategy, optimization_variables=optimization_variables
    )
    return objective


def validate_selection_schedule(selection_schedule: str) -> str:
    if selection_schedule not in VALID_SELECTION_SCHEDULES:
        allowed = ", ".join(VALID_SELECTION_SCHEDULES)
        raise ValueError(f"selection_schedule must be one of {allowed}, got {selection_schedule}.")
    return selection_schedule


def optimize_once_per_experiment(
    strategy: str,
    selection_schedule: str = DEFAULT_SELECTION_SCHEDULE,
) -> bool:
    validate_selection_schedule(selection_schedule)
    return strategy in OPTIMIZED_STRATEGIES and selection_schedule == "first_origin"


def select_initial_hyperparameters(
    strategy: str,
    task_template: dict[str, Any],
    origin_date: pd.Timestamp,
) -> list[list[float]] | None:
    if not optimize_once_per_experiment(strategy, task_template["selection_schedule"]):
        return None
    task_template["selection_origin"] = pd.Timestamp(origin_date).strftime("%Y-%m-%d")

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


def hyperparameter_record(
    origin_date: pd.Timestamp,
    strategy: str,
    hyperparameters: list[list[float]],
    information_set: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base = {
        "forecast_origin": pd.Timestamp(origin_date).strftime("%Y-%m-%d"),
        "group": origin_group(pd.Timestamp(origin_date)),
        "strategy": strategy,
    }
    if information_set:
        base.update(information_set)
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


OPTIMIZATION_PENALTY = 1.0e10
MDD_INVALID_FLOOR = -1.0e15
EXPECTED_NUMERICAL_FAILURES = (
    np.linalg.LinAlgError,
    FloatingPointError,
    OverflowError,
    ZeroDivisionError,
)


def derive_minimum_training_quarters(
    nlags: list[int],
    n_objective_variables: int,
    *,
    frequency_ratio: int = 3,
) -> int:
    if not nlags or any(not isinstance(lag, (int, np.integer)) or lag < 1 for lag in nlags):
        raise ValueError("nlags must contain positive integers.")
    if n_objective_variables < 1:
        raise ValueError("At least one objective variable is required.")
    if frequency_ratio < 1:
        raise ValueError("frequency_ratio must be positive.")
    # One full lag history plus a predecessor for the first growth target is
    # required. The variable term prevents a one-observation low-frequency fit
    # when a larger quarterly block is selected.
    return max(
        2,
        math.ceil((max(nlags) + 1) / frequency_ratio),
        math.ceil(n_objective_variables / frequency_ratio),
    )


def build_rmse_validation_folds(
    data_in,
    *,
    horizon_quarters: int,
    h_eval: int | None,
    n_eval: int,
    forecast_variables: list[str],
    objective_variables: list[str],
    nlags: list[int],
    selection: str,
    min_train_quarters: int | None,
    fold_seed: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(horizon_quarters, int) or horizon_quarters < 1:
        raise ValueError("optimization_horizon_quarters must be a positive integer.")
    if h_eval is not None and (
        not isinstance(h_eval, int) or h_eval < 1 or h_eval > horizon_quarters
    ):
        raise ValueError(
            f"optimization_eval_horizon_quarters must be in 1..{horizon_quarters}."
        )
    if not isinstance(n_eval, int) or n_eval < 1:
        raise ValueError("optimization_n_eval must be a positive integer.")
    if selection not in {"rolling", "random"}:
        raise ValueError(f"Unknown validation-origin selection: {selection}.")

    non_subset = [v for v in objective_variables if v not in set(forecast_variables)]
    if non_subset:
        raise ValueError(
            "objective_variables must be a subset of forecast_variables; "
            f"invalid entries: {non_subset}."
        )

    frequencies = list(data_in.frequencies)
    if frequencies != ["Q", "M"] or int(data_in.freq_ratio_list[-1]) != 3:
        raise NotImplementedError("The repository RMSE objective currently supports Q/M models only.")

    quarterly = data_in.input_data_Q.copy()
    monthly = list(data_in.input_data)[-1].copy()
    missing_variables = [
        variable
        for variable in forecast_variables
        if variable not in quarterly
    ]
    if missing_variables:
        raise ValueError(
            f"Forecast variables are missing from the quarterly data: {missing_variables}."
        )
    missing_objective = [
        variable for variable in objective_variables if variable not in quarterly
    ]
    if missing_objective:
        raise ValueError(
            f"Objective variables are missing from the quarterly data: {missing_objective}."
        )

    # The fitted mixed-frequency system spans the full forecast block, so the
    # minimum training length is derived from that dimension rather than the
    # (possibly smaller) objective subset.
    derived_minimum = derive_minimum_training_quarters(nlags, len(forecast_variables))
    if min_train_quarters is None:
        effective_minimum = derived_minimum
    else:
        if not isinstance(min_train_quarters, int) or min_train_quarters < derived_minimum:
            raise ValueError(
                "optimization_min_t must be an integer at least "
                f"{derived_minimum} for the configured lags and variables."
            )
        effective_minimum = min_train_quarters

    eligible: list[dict[str, Any]] = []
    maximum_cut = len(quarterly) - horizon_quarters
    for cut in range(effective_minimum, maximum_cut + 1):
        train_quarterly = quarterly.iloc[:cut].copy()
        holdout = quarterly.iloc[cut : cut + horizon_quarters].copy()
        # Only the objective subset must be observed in the holdout, since the
        # forecast block is predicted rather than read from the holdout.
        if len(holdout) != horizon_quarters or holdout[objective_variables].isna().any().any():
            continue
        invalid_growth = False
        for variable in objective_variables:
            if SERIES_BY_CODE[variable].evaluation_transform == "growth":
                values = pd.concat(
                    [train_quarterly[variable].iloc[-1:], holdout[variable]]
                )
                if values.isna().any() or (values <= 0).any():
                    invalid_growth = True
                    break
        if invalid_growth:
            continue

        train_end_quarter = pd.Timestamp(train_quarterly.index[-1]).to_period("Q")
        train_monthly = monthly.loc[
            pd.PeriodIndex(monthly.index, freq="Q") <= train_end_quarter
        ].copy()
        if len(train_monthly) <= max(nlags) or train_monthly.iloc[-1].isna().any():
            continue

        target_quarters = pd.PeriodIndex(holdout.index, freq="Q")
        eligible.append(
            {
                "cut": cut,
                "training_end_quarter": train_end_quarter,
                "target_quarters": target_quarters,
                "quarterly": train_quarterly,
                "monthly": train_monthly,
                "holdout": holdout,
            }
        )

    if n_eval > len(eligible):
        raise ValueError(
            f"optimization_n_eval={n_eval} exceeds the {len(eligible)} strictly feasible "
            "validation origins; no silent truncation is allowed."
        )
    if selection == "rolling":
        folds = eligible[-n_eval:]
    else:
        rng = np.random.default_rng(fold_seed)
        positions = sorted(rng.choice(len(eligible), size=n_eval, replace=False).tolist())
        folds = [eligible[position] for position in positions]

    diagnostics = {
        "selection": selection,
        "requested_n_eval": n_eval,
        "effective_n_eval": len(folds),
        "derived_min_train_quarters": derived_minimum,
        "effective_min_train_quarters": effective_minimum,
        "fold_seed": fold_seed,
        "origins": [
            {
                "training_end_quarter": str(fold["training_end_quarter"]),
                "target_start_quarter": str(fold["target_quarters"][0]),
                "target_end_quarter": str(fold["target_quarters"][-1]),
                "cut": fold["cut"],
            }
            for fold in folds
        ],
    }
    return folds, diagnostics


def _candidate_hyperparameters(params: dict[str, Any]) -> list[list[float]]:
    values: list[float] = []
    for name in ("lambda1_1", "lambda2_1", "lambda4_1", "lambda5_1"):
        if name not in params:
            raise KeyError(f"Optimizer candidate is missing {name}.")
        value = float(params[name])
        lower, upper = DEFAULT_PARAM_SPACE_BOUNDS[name]
        if not np.isfinite(value) or value < lower or value > upper:
            raise ValueError(f"Optimizer candidate {name}={value} is outside [{lower}, {upper}].")
        values.append(value)
    return [[values[0], values[1], 1.0, values[2], values[3]]]


def _rmse_candidate_score(
    params: dict[str, Any],
    *,
    model_class,
    folds: list[dict[str, Any]],
    forecast_variables: list[str],
    objective_variables: list[str],
    horizon_quarters: int,
    h_eval: int | None,
    nsim: int,
    nburn_perc: float,
    nlags: list[int],
    thining: int,
    temp_agg: str,
    objective_seed: int | None,
) -> float:
    hyperparameters = _candidate_hyperparameters(params)
    errors_by_variable: dict[str, list[float]] = {
        variable: [] for variable in objective_variables
    }
    evaluated_horizons = (
        [h_eval] if h_eval is not None else list(range(1, horizon_quarters + 1))
    )

    for fold_index, fold in enumerate(folds):
        fold_data = make_data_in(fold["quarterly"], fold["monthly"])
        fold_model = model_class(nsim, nburn_perc, nlags, thining)
        seed = stable_child_seed(objective_seed, "rmse_fold", fold_index)
        with deterministic_rng_context(seed):
            # The forecast state is always built from the full forecast block so a
            # reduced objective subset never collapses the mixed-frequency system.
            fold_model.fit(
                fold_data,
                hyp=hyperparameters,
                var_of_interest=forecast_variables,
                temp_agg=temp_agg,
                check_explosive=False,
            )
            # Folds are non-ragged by construction (each fold's monthly block
            # ends exactly at its quarter end), so the calendar-nominal length
            # is exact here; FREQ_RATIO keeps the ratio centralized.
            fold_model.forecast(horizon_quarters * FREQ_RATIO)

        draw_frames = aggregate_quarterly_posterior_draws(fold_model)
        _, metric_summaries = summarize_quarterly_draws(draw_frames)
        predicted_metrics = metric_summaries["mean"]

        actual_levels = pd.concat(
            [fold["quarterly"].iloc[-1:], fold["holdout"]]
        ).copy()
        actual_levels.index = pd.PeriodIndex(actual_levels.index, freq="Q")
        actual_metrics = compute_quarterly_metrics(actual_levels)

        for horizon in evaluated_horizons:
            target = fold["target_quarters"][horizon - 1]
            if target not in predicted_metrics.index or target not in actual_metrics.index:
                raise ValueError(
                    f"Validation forecast is missing target {target} at horizon {horizon}."
                )
            # Only the objective subset contributes to the loss.
            for variable in objective_variables:
                prediction = float(predicted_metrics.at[target, variable])
                actual = float(actual_metrics.at[target, variable])
                if not np.isfinite(prediction) or not np.isfinite(actual):
                    raise FloatingPointError(
                        f"Non-finite validation metric for {variable} at {target}."
                    )
                errors_by_variable[variable].append(prediction - actual)

    variable_mses = []
    for variable, errors in errors_by_variable.items():
        if not errors:
            raise ValueError(f"No validation errors were produced for {variable}.")
        variable_mses.append(float(np.mean(np.square(errors))))
    return float(np.sqrt(np.mean(variable_mses)))


def _mdd_candidate_score(
    params: dict[str, Any],
    *,
    model_class,
    data_in,
    variables: list[str],
    nsim: int,
    nburn_perc: float,
    nlags: list[int],
    thining: int,
    temp_agg: str,
    objective_seed: int | None,
    n_replicates: int,
) -> float:
    hyperparameters = _candidate_hyperparameters(params)
    mdd_values: list[float] = []
    for replicate in range(n_replicates):
        replicate_model = model_class(nsim, nburn_perc, nlags, thining)
        seed = stable_child_seed(objective_seed, "mdd_replicate", replicate)
        with deterministic_rng_context(seed):
            value = replicate_model.fit(
                copy.deepcopy(data_in),
                hyp=hyperparameters,
                var_of_interest=variables,
                temp_agg=temp_agg,
                check_explosive=False,
                return_mdd=True,
            )
        value = float(value)
        if not np.isfinite(value) or value <= MDD_INVALID_FLOOR:
            raise FloatingPointError(f"Invalid MDD objective value: {value}.")
        mdd_values.append(value)
    return -float(np.mean(mdd_values))


def _run_local_mango_optimizer(
    strategy: str,
    model,
    data_in,
    args: dict[str, Any],
) -> list[list[float]]:
    from mango import Tuner

    # objective_variables drive the loss; forecast_variables build the state.
    objective_variables = args.get("objective_variables") or args["optimization_variables"]
    forecast_variables = args.get("forecast_variables") or objective_variables
    variables = objective_variables
    diagnostics: dict[str, Any] = {
        "objective": (
            "equal_variable_weight_rmse_of_evaluation_metric"
            if strategy in {"mango_rmse", "mango_rmse_random"}
            else "negative_mean_log_marginal_data_density"
        ),
        "forecast_variables": forecast_variables,
        "objective_variables": objective_variables,
        "variable_weights": {variable: 1.0 / len(variables) for variable in variables},
        "valid_evaluations": 0,
        "penalized_evaluations": 0,
        "exceptional_evaluations": 0,
        "nonfinite_evaluations": 0,
        "penalty": OPTIMIZATION_PENALTY,
        "candidate_seed": args.get("optimization_candidate_seed"),
        "objective_seed": args.get("optimization_objective_seed"),
        "optimizer_njobs_effective": 1,
    }

    folds: list[dict[str, Any]] | None = None
    if strategy in {"mango_rmse", "mango_rmse_random"}:
        folds, fold_diagnostics = build_rmse_validation_folds(
            data_in,
            horizon_quarters=args["optimization_horizon_quarters"],
            h_eval=args["optimization_eval_horizon_quarters"],
            n_eval=args["optimization_n_eval"],
            forecast_variables=forecast_variables,
            objective_variables=objective_variables,
            nlags=args["nlags"],
            selection="random" if strategy == "mango_rmse_random" else "rolling",
            min_train_quarters=args["optimization_min_t"],
            fold_seed=args.get("optimization_fold_seed"),
        )
        diagnostics["validation"] = fold_diagnostics
        diagnostics["metric_transforms"] = {
            variable: SERIES_BY_CODE[variable].evaluation_transform
            for variable in variables
        }

    def score_one(params: dict[str, Any]) -> float:
        try:
            if strategy in {"mango_rmse", "mango_rmse_random"}:
                assert folds is not None
                score = _rmse_candidate_score(
                    params,
                    model_class=model.__class__,
                    folds=folds,
                    forecast_variables=forecast_variables,
                    objective_variables=objective_variables,
                    horizon_quarters=args["optimization_horizon_quarters"],
                    h_eval=args["optimization_eval_horizon_quarters"],
                    nsim=args["optimization_nsim"],
                    nburn_perc=args["nburn_perc"],
                    nlags=args["nlags"],
                    thining=args["thining"],
                    temp_agg=args["temp_agg"],
                    objective_seed=args.get("optimization_objective_seed"),
                )
            else:
                score = _mdd_candidate_score(
                    params,
                    model_class=model.__class__,
                    data_in=data_in,
                    variables=variables,
                    nsim=args["optimization_nsim"],
                    nburn_perc=args["nburn_perc"],
                    nlags=args["nlags"],
                    thining=args["thining"],
                    temp_agg=args["temp_agg"],
                    objective_seed=args.get("optimization_objective_seed"),
                    n_replicates=args["optimization_objective_replicates"],
                )
        except EXPECTED_NUMERICAL_FAILURES:
            diagnostics["exceptional_evaluations"] += 1
            diagnostics["penalized_evaluations"] += 1
            return OPTIMIZATION_PENALTY
        if not np.isfinite(score):
            diagnostics["nonfinite_evaluations"] += 1
            diagnostics["penalized_evaluations"] += 1
            return OPTIMIZATION_PENALTY
        diagnostics["valid_evaluations"] += 1
        return float(score)

    def batch_objective(params_batch: list[dict[str, Any]]) -> list[float]:
        return [score_one(params) for params in params_batch]

    configuration = {
        "num_iteration": args["optimization_iterations"],
        "initial_random": args["optimization_init_points"],
        "batch_size": 1,
    }
    candidate_seed = args.get("optimization_candidate_seed")
    with deterministic_rng_context(candidate_seed):
        results = Tuner(
            make_param_space(DEFAULT_PARAM_SPACE_BOUNDS),
            batch_objective,
            configuration,
        ).minimize()

    best_params = results.get("best_params")
    best_objective = results.get("best_objective")
    if not isinstance(best_params, dict) or best_objective is None:
        raise RuntimeError("Mango did not return a best candidate and objective value.")
    best_objective = float(best_objective)
    postcheck_score = score_one(best_params)
    if (
        diagnostics["valid_evaluations"] == 0
        or not np.isfinite(best_objective)
        or best_objective >= OPTIMIZATION_PENALTY
        or not np.isfinite(postcheck_score)
        or postcheck_score >= OPTIMIZATION_PENALTY
    ):
        raise RuntimeError(
            "Hyperparameter optimization produced no valid candidate; "
            "an all-penalty result is never accepted."
        )

    diagnostics["best_objective"] = best_objective
    diagnostics["postcheck_objective"] = postcheck_score
    diagnostics["best_optimizer_coordinates"] = {
        name: float(value) for name, value in best_params.items()
    }
    args["_selection_diagnostics"] = diagnostics
    return _candidate_hyperparameters(best_params)


def select_hyperparameters(
    strategy: str,
    model,
    data_in,
    args: dict[str, Any],
) -> list[list[float]]:
    if strategy == "paper":
        return [PAPER_HYPERPARAMETERS]
    if strategy in OPTIMIZED_STRATEGIES:
        return _run_local_mango_optimizer(strategy, model, data_in, args)

    raise ValueError(f"Unsupported strategy: {strategy}")


def extract_forecasts(
    strategy: str,
    origin_date: pd.Timestamp,
    model,
    actual_levels: pd.DataFrame,
) -> pd.DataFrame:
    current_quarter = pd.Timestamp(origin_date).to_period("Q")
    draw_frames = aggregate_quarterly_posterior_draws(model)
    prediction_frames, metric_frames = summarize_quarterly_draws(draw_frames)

    actual_levels = actual_levels.copy()
    actual_levels.index = pd.PeriodIndex(actual_levels.index, freq="Q")
    actual_metrics = compute_quarterly_metrics(actual_levels)
    variables = [spec.paper_code for spec in SERIES_SPECS]
    prediction_variables = prediction_frames["mean"].columns.tolist()
    if set(prediction_variables) != set(variables):
        missing = sorted(set(variables) - set(prediction_variables))
        extra = sorted(set(prediction_variables) - set(variables))
        raise ValueError(
            f"Forecast variable coverage mismatch; missing={missing}, extra={extra}."
        )

    target_quarters = [
        current_quarter + (horizon - 1)
        for horizon in range(1, MAX_FORECAST_HORIZON_QUARTERS + 1)
    ]
    missing_prediction_targets = [
        str(target)
        for target in target_quarters
        if target not in prediction_frames["mean"].index
    ]
    missing_actual_targets = [
        str(target) for target in target_quarters if target not in actual_levels.index
    ]
    if missing_prediction_targets or missing_actual_targets:
        raise ValueError(
            "Requested forecast coverage is incomplete: "
            f"missing predictions={missing_prediction_targets}, "
            f"missing actuals={missing_actual_targets}."
        )

    rows: list[dict[str, Any]] = []
    for horizon_quarters, target_quarter in enumerate(target_quarters, start=1):
        for variable in variables:
            actual_level = float(actual_levels.at[target_quarter, variable])
            actual_metric = float(actual_metrics.at[target_quarter, variable])
            if not np.isfinite(actual_level) or not np.isfinite(actual_metric):
                raise ValueError(
                    f"Actual {variable} for {target_quarter} is missing, partial, or non-finite."
                )

            row = {
                "strategy": strategy,
                "forecast_origin": pd.Timestamp(origin_date).strftime("%Y-%m-%d"),
                "group": origin_group(pd.Timestamp(origin_date)),
                "target_quarter": str(target_quarter),
                "horizon_quarters": horizon_quarters,
                "variable": variable,
                "actual_level": actual_level,
                "actual_metric": actual_metric,
            }
            for frame_name, frame in prediction_frames.items():
                value = float(frame.at[target_quarter, variable])
                if not np.isfinite(value):
                    raise ValueError(
                        f"Posterior {frame_name} level is non-finite for {variable} at {target_quarter}."
                    )
                row[f"{frame_name}_level"] = value
            for frame_name, frame in metric_frames.items():
                value = float(frame.at[target_quarter, variable])
                if not np.isfinite(value):
                    raise ValueError(
                        f"Posterior {frame_name} metric is non-finite for {variable} at {target_quarter}."
                    )
                row[f"{frame_name}_metric"] = value
            row["error_metric"] = row["mean_metric"] - row["actual_metric"]
            rows.append(row)

    forecasts = pd.DataFrame(rows)
    expected_rows = MAX_FORECAST_HORIZON_QUARTERS * len(variables)
    key_columns = ["forecast_origin", "target_quarter", "horizon_quarters", "variable"]
    if len(forecasts) != expected_rows or forecasts.duplicated(key_columns).any():
        raise RuntimeError(
            f"Forecast extraction produced {len(forecasts)} rows; expected {expected_rows} unique keys."
        )
    return forecasts


def required_forecast_months(
    monthly_frame: pd.DataFrame,
    origin_date: pd.Timestamp,
    *,
    max_horizon_quarters: int = MAX_FORECAST_HORIZON_QUARTERS,
) -> int:
    if monthly_frame.empty:
        raise ValueError("Cannot derive a forecast horizon from an empty monthly block.")
    if max_horizon_quarters < 1:
        raise ValueError("max_horizon_quarters must be positive.")
    model_endpoint = pd.Timestamp(monthly_frame.index[-1]).to_period("M")
    # ``Period.add`` does not exist in supported pandas versions; use operator
    # arithmetic, which is period-aware across year/quarter boundaries.
    final_target_month = (
        pd.Timestamp(origin_date).to_period("Q") + (max_horizon_quarters - 1)
    ).asfreq("M", how="end")
    required = final_target_month.ordinal - model_endpoint.ordinal
    if required < 1:
        raise ValueError(
            f"Model calendar endpoint {model_endpoint} is not before final target month {final_target_month}."
        )
    return required


def _origin_information_set(
    quarterly: pd.DataFrame,
    monthly: pd.DataFrame,
    origin_date: pd.Timestamp,
    effective_forecast_months: int,
) -> dict[str, Any]:
    last_released_month = pd.Timestamp(
        monthly.attrs.get("last_released_month", monthly.dropna(how="all").index[-1])
    )
    quarterly_observed = quarterly.dropna(how="all")
    last_quarter = pd.Timestamp(quarterly_observed.index[-1]).to_period("Q")
    origin_month = pd.Timestamp(origin_date).to_period("M")
    return {
        "quarterly_information_set_end": str(last_quarter),
        "monthly_last_released": last_released_month.strftime("%Y-%m-%d"),
        "model_calendar_endpoint": pd.Timestamp(monthly.index[-1]).strftime("%Y-%m-%d"),
        "origin_data_lag_months": origin_month.ordinal - last_released_month.to_period("M").ordinal,
        "forecast_horizon_months_effective": effective_forecast_months,
    }


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
            "failure_category": classify_failure(exc),
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
    optimization_nsim: int = DEFAULT_OPTIMIZATION_NSIM,
    optimization_init_points: int = 5,
    optimization_iterations: int = 15,
    optimization_njobs: int | None = None,
    optimization_horizon_quarters: int = 4,
    optimization_eval_horizon_quarters: int | None = None,
    optimization_n_eval: int = 3,
    optimization_min_t: int | None = None,
    optimization_random_seed: int | None = None,
    optimization_variables: list[str] | None = None,
    forecast_variables: list[str] | None = None,
    objective_variables: list[str] | None = None,
    temp_agg: str = PAPER_TEMPORAL_AGGREGATION,
    n_workers: int | None = None,
    selection_schedule: str | None = None,
) -> Path:
    resolved_selection_schedule = resolve_selection_schedule(strategy, selection_schedule)
    output_dir = resolve_project_path(output_dir)
    panel_path = resolve_project_path(panel_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_forecast_variables, resolved_objective_variables = resolve_forecast_objective_variables(
        strategy,
        optimization_variables=optimization_variables,
        forecast_variables=forecast_variables,
        objective_variables=objective_variables,
    )

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
        "optimization_variables": resolved_objective_variables,
        "forecast_variables": resolved_forecast_variables,
        "objective_variables": resolved_objective_variables,
        "temp_agg": temp_agg,
        "selection_schedule": resolved_selection_schedule,
    }

    has_origins = len(origins) > 0
    selected_once = optimize_once_per_experiment(strategy, resolved_selection_schedule)
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
        "optimization_variables": resolved_objective_variables,
        "forecast_variables": resolved_forecast_variables,
        "objective_variables": resolved_objective_variables,
        "selection_schedule": resolved_selection_schedule,
        "hyperparameters_selected_once": selected_once,
        "hyperparameter_selection_origin": (origins[0].strftime("%Y-%m-%d") if has_origins and selected_once else None),
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
    parser.add_argument("--optimization-nsim", type=int, default=DEFAULT_OPTIMIZATION_NSIM)
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
        "--selection-schedule",
        type=str,
        default=None,
        choices=list(VALID_SELECTION_SCHEDULES),
        help=(
            "How often hyperparameters are re-selected. Defaults to the baseline schedule "
            "for the strategy: first_origin for the Mango RMSE variants and per_origin for "
            "Mango MDD. Set the same value for every strategy to compare objectives without "
            "also comparing update frequencies."
        ),
    )
    parser.add_argument(
        "--optimization-variables",
        type=str,
        default=None,
        help=(
            "Legacy comma-separated objective subset for the MBFVAR hyperparameter objective. "
            "Mango MDD defaults to GDP. For Mango RMSE variants this maps to the objective "
            "(loss) subset while the forecast state always uses the full quarterly block "
            "GDP,INVFIX,GOV. Mutually exclusive with --forecast-variables/--objective-variables."
        ),
    )
    parser.add_argument(
        "--forecast-variables",
        type=str,
        default=None,
        help=(
            "Comma-separated block used to build the mixed-frequency forecast state. "
            "For Mango RMSE variants this must be the full quarterly block GDP,INVFIX,GOV."
        ),
    )
    parser.add_argument(
        "--objective-variables",
        type=str,
        default=None,
        help=(
            "Comma-separated subset of the forecast block whose forecast errors enter the "
            "optimization objective. May be a single target such as GDP."
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
                "forecast_variables": parse_csv_list(getattr(namespace, "forecast_variables", None), []) or None,
                "objective_variables": parse_csv_list(getattr(namespace, "objective_variables", None), []) or None,
                "selection_schedule": getattr(namespace, "selection_schedule", None),
            }
        )
    return run_recursive_experiment(**kwargs)
