"""Real-run execution for the mixed-frequency scope-grid study.

This module is imported lazily by ``scripts/mfvar/run_mfvar_scope_grid.py`` only
for real (non-dry-run) execution, so the dry-run and manifest paths never import
data-loading code or MBFVAR. It wires the model-independent adapter
(:func:`run_mfvar_selection_experiment`) to the real mixed-frequency objective
selector and a real MBFVAR forecast generator, then writes the canonical output
contract shared with GLP.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, Sequence

import pandas as pd

from common_hpo import (
    CSVSchema,
    JSONSchema,
    atomic_write_csv_rows,
    atomic_write_dataframe_csv,
    atomic_write_json,
    build_run_metadata,
    classify_failure,
    mark_run_cancelled,
    mark_run_complete,
    mark_run_failed,
    prepare_run_directory,
    utc_now,
)

from .config import (
    PAPER_NBURN_PERC,
    PAPER_NLAGS,
    PAPER_NSIM,
    PAPER_TEMPORAL_AGGREGATION,
    PAPER_THINING,
    DEFAULT_OPTIMIZATION_NSIM,
    forecast_origin_dates,
    param_space_metadata,
)
from .horizon_mapping import (
    quarterly_horizon_to_state_rows,
    target_quarter_for_origin,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .selection_experiment import MFVARForecastRequest


_FAILED_ORIGINS_COLUMNS = ("forecast_origin", "stage", "failure_category", "error")
_FORECAST_PANEL_COLUMNS = (
    "forecast_origin",
    "variable",
    "horizon",
    "forecast",
)
_SELECTION_COLUMNS = (
    "cell_id",
    "selection_event_id",
    "selection_loss",
)


def _run_metadata(config, plan, *, started_utc: str, finished_utc: str | None, completion_status: str, extra=None):
    expected_outputs = [
        "forecast_panel.csv",
        "selected_hyperparameters.csv",
        "failed_origins.csv",
        "run_metadata.json",
    ]
    if config.save_all_cell_forecasts:
        expected_outputs.append("forecast_panel_all_cells.csv")
    return build_run_metadata(
        project_root=Path(__file__).resolve().parents[2],
        command_line=config.command_line,
        started_utc=started_utc,
        finished_utc=finished_utc,
        completion_status=completion_status,
        model_family="mfbvar",
        model_version="mfvar_scope_grid",
        data_source={"panel_path": str(config.panel_path), "format": "realtime_panel"},
        data_vintage_identifiers={
            "outer_vintage_policy": "outer_vintage_consistent",
            "optimization_horizon_quarters": config.optimization_horizon_quarters,
            "optimization_eval_horizon_quarters": config.optimization_eval_horizon_quarters,
        },
        input_files=(config.panel_path,),
        transformation_configuration={
            "forecast_variables": list(config.forecast_variables),
            "quarterly_transform_pipeline": "paper_mfvar_default",
        },
        variable_order=config.forecast_variables,
        target_variables=config.target_variables,
        target_horizons=config.target_horizons,
        selection_plan=plan.selection_plan.to_dict(),
        validation_scheme={
            "window": config.inner_window,
            "n_origins": config.inner_n_origins,
            "origin_stride": config.inner_origin_stride,
            "origin_selection": config.inner_origin_selection,
            "rolling_window_length": config.rolling_window_length,
        },
        vintage_policy="outer_vintage_consistent",
        selection_schedule={"requested": config.selection_frequency},
        loss_configuration={"metric": config.loss_metric, "scaling": config.loss_scaling},
        search_space={
            "forecast_variables": list(config.forecast_variables),
            "variable_groups": list(config.variable_groups or ()),
            "residual_group_name": config.residual_group_name,
            "group_separate_horizons": config.group_separate_horizons,
            **param_space_metadata(),
        },
        optimizer_budget={
            "n_eval": config.optimization_n_eval,
            "optimization_horizon_quarters": config.optimization_horizon_quarters,
            "optimization_eval_horizon_quarters": config.optimization_eval_horizon_quarters,
        },
        random_seeds={"base_seed": config.base_seed, "inner_random_seed": config.inner_random_seed},
        parallel_worker_count=1,
        configuration_extra={
            "scope": plan.scope,
            "expected_outputs": expected_outputs,
            "output_dir": str(plan.output_dir),
            "save_all_cell_forecasts": config.save_all_cell_forecasts,
        },
        extra=extra,
    )


def _output_schemas(save_all_cell_forecasts: bool):
    csv_schemas = [
        CSVSchema("forecast_panel.csv", _FORECAST_PANEL_COLUMNS, min_rows=1),
        CSVSchema("selected_hyperparameters.csv", _SELECTION_COLUMNS, min_rows=1),
        CSVSchema("failed_origins.csv", _FAILED_ORIGINS_COLUMNS, min_rows=0),
    ]
    if save_all_cell_forecasts:
        csv_schemas.append(CSVSchema("forecast_panel_all_cells.csv", _FORECAST_PANEL_COLUMNS, min_rows=1))
    json_schemas = [
        JSONSchema(
            "run_metadata.json",
            (
                "configuration_hash",
                "completion_status",
                "repository_commit",
                "target_variables",
                "target_horizons",
            ),
        )
    ]
    return tuple(csv_schemas), tuple(json_schemas)


def _write_outputs(output_dir: Path, result, *, save_all_cell_forecasts: bool, metadata_override=None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_dataframe_csv(output_dir / "forecast_panel.csv", pd.DataFrame(result.forecast_panel), index=False)
    atomic_write_dataframe_csv(
        output_dir / "selected_hyperparameters.csv",
        pd.DataFrame(result.selected_hyperparameters),
        index=False,
    )
    atomic_write_csv_rows(output_dir / "failed_origins.csv", _FAILED_ORIGINS_COLUMNS, ())
    atomic_write_json(output_dir / "run_metadata.json", dict(metadata_override or result.run_metadata))
    if save_all_cell_forecasts:
        atomic_write_dataframe_csv(
            output_dir / "forecast_panel_all_cells.csv",
            pd.DataFrame(result.forecast_panel_all_cells),
            index=False,
        )


def _build_real_forecast_generator(panel_path: Path, forecast_variables: Sequence[str]):
    """Return a forecast generator producing quarterly predictive-mean forecasts.

    The heavy MBFVAR + data imports stay inside this factory so they are only
    resolved during a genuine run.
    """

    from . import forecasting
    from .data_utils import build_model_input_frames, load_realtime_panel

    panel = load_realtime_panel(panel_path)

    def forecast_generator(request: "MFVARForecastRequest"):
        import MBFVAR

        origin_date = pd.Timestamp(request.origin_label)
        quarterly, monthly = build_model_input_frames(panel, origin_date)
        data_in = forecasting.make_data_in(quarterly, monthly)
        model = MBFVAR.MixedFrequencyBVAR(
            PAPER_NSIM, PAPER_NBURN_PERC, PAPER_NLAGS, PAPER_THINING
        )
        model.fit(
            data_in,
            hyp=[list(request.hyperparameter_vector)],
            temp_agg=PAPER_TEMPORAL_AGGREGATION,
        )
        # Endpoint-aware forecast length (MF-01): the monthly block is ragged in
        # real time, so its last calendar month can lag the nominal origin
        # quarter. A nominal ``max_horizon * 3`` would then stop short of the
        # final target quarter and the extraction below would raise KeyError.
        model.forecast(
            forecasting.required_forecast_months(
                monthly,
                origin_date,
                max_horizon_quarters=max(request.system_horizons),
            )
        )
        model.aggregate(frequency="Q")

        draw_frames = forecasting.aggregate_quarterly_posterior_draws(model)
        _, metric_summaries = forecasting.summarize_quarterly_draws(draw_frames)
        predicted = metric_summaries["mean"]

        rows = len(request.system_horizons)
        cols = len(request.system_variables)
        import numpy as np

        out = np.empty((rows, cols), dtype=float)
        for row, horizon in enumerate(request.system_horizons):
            target = target_quarter_for_origin(origin_date, horizon)
            for col, variable in enumerate(request.system_variables):
                out[row, col] = float(predicted.at[target, variable])
        return out

    return forecast_generator


def execute_scope_runs(config, plans) -> None:
    """Execute each planned scope run and write the canonical outputs."""

    from . import forecasting
    from .data_utils import build_model_input_frames, load_realtime_panel
    from .loss_engine import build_mfvar_objective_selector
    from .selection_experiment import run_mfvar_selection_experiment

    schedule = _build_schedule(config.selection_frequency)
    origins = [
        pd.Timestamp(origin) for origin in forecast_origin_dates()
    ]

    panel = load_realtime_panel(config.panel_path)
    forecast_generator = _build_real_forecast_generator(
        config.panel_path, config.forecast_variables
    )

    # Selection data_in is built at the first origin (selection is point-in-time
    # at each event's origin inside the adapter; here we use the earliest origin
    # to construct the fold data consistent with the once/annual schedules).
    first_quarterly, first_monthly = build_model_input_frames(panel, origins[0])
    data_in = forecasting.make_data_in(first_quarterly, first_monthly)

    loss_config = _build_loss_config(config)
    validation_scheme = _build_validation_scheme(config)

    for plan in plans:
        if plan.existing_policy == "resume_skip":
            continue
        started_utc = utc_now()
        started_monotonic = time.monotonic()
        initial_metadata = _run_metadata(
            config,
            plan,
            started_utc=started_utc,
            finished_utc=None,
            completion_status="partial",
        )
        expected_outputs = list(EXPECTED_OUTPUT_FILES)
        if config.save_all_cell_forecasts:
            expected_outputs.append("forecast_panel_all_cells.csv")
        prepare_run_directory(
            plan.output_dir,
            manifest=initial_metadata,
            if_exists_policy=config.if_exists_policy,
            expected_outputs=expected_outputs,
        )
        selector = build_mfvar_objective_selector(
            data_in=data_in,
            model_class=__import__("MBFVAR").MixedFrequencyBVAR,
            nsim=DEFAULT_OPTIMIZATION_NSIM,
            nburn_perc=PAPER_NBURN_PERC,
            nlags=PAPER_NLAGS,
            thining=PAPER_THINING,
            temp_agg=PAPER_TEMPORAL_AGGREGATION,
            horizon_quarters=config.optimization_horizon_quarters,
            eval_horizon_quarters=config.optimization_eval_horizon_quarters,
            n_eval=config.optimization_n_eval,
            objective_seed=config.base_seed,
        )
        result = run_mfvar_selection_experiment(
            origins,
            selector=selector,
            forecast_generator=forecast_generator,
            target_variables=config.target_variables,
            target_horizons=config.target_horizons,
            forecast_variables=config.forecast_variables,
            loss_config=loss_config,
            validation_scheme=validation_scheme,
            plan=plan.selection_plan,
            schedule=schedule,
            retain_off_target=config.save_all_cell_forecasts,
            base_seed=config.base_seed,
        )
        try:
            final_metadata = _run_metadata(
                config,
                plan,
                started_utc=started_utc,
                finished_utc=utc_now(),
                completion_status="complete",
                extra={
                    "runner": "mfvar_scope_grid",
                    "cache_stats": result.cache_stats,
                    "selection_event_count": len(result.run_metadata.get("selection_events", [])),
                    "wall_time_seconds": time.monotonic() - started_monotonic,
                },
            )
            _write_outputs(
                plan.output_dir,
                result,
                save_all_cell_forecasts=config.save_all_cell_forecasts,
                metadata_override=final_metadata,
            )
            csv_schemas, json_schemas = _output_schemas(config.save_all_cell_forecasts)
            mark_run_complete(
                plan.output_dir,
                configuration_hash=str(final_metadata["configuration_hash"]),
                csv_schemas=csv_schemas,
                json_schemas=json_schemas,
            )
        except KeyboardInterrupt:
            cancelled_metadata = _run_metadata(
                config,
                plan,
                started_utc=started_utc,
                finished_utc=utc_now(),
                completion_status="cancelled",
                extra={
                    "runner": "mfvar_scope_grid",
                    "wall_time_seconds": time.monotonic() - started_monotonic,
                },
            )
            mark_run_cancelled(
                plan.output_dir,
                configuration_hash=str(initial_metadata["configuration_hash"]),
                metadata=cancelled_metadata,
            )
            raise
        except Exception as exc:  # noqa: BLE001
            failed_metadata = _run_metadata(
                config,
                plan,
                started_utc=started_utc,
                finished_utc=utc_now(),
                completion_status="failed",
                extra={
                    "runner": "mfvar_scope_grid",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                    "failure_category": classify_failure(exc),
                    "wall_time_seconds": time.monotonic() - started_monotonic,
                },
            )
            mark_run_failed(
                plan.output_dir,
                configuration_hash=str(initial_metadata["configuration_hash"]),
                reason=f"{type(exc).__name__}: {exc}",
                metadata=failed_metadata,
            )
            raise


def _build_schedule(selection_frequency: str):
    from scripts.mfvar.run_mfvar_scope_grid import build_selection_schedule  # pragma: no cover

    return build_selection_schedule(selection_frequency)


def _build_loss_config(config):
    from common_hpo import LossConfig, ScaleConfig

    if config.loss_scaling == "none":
        return LossConfig(aggregation=config.loss_metric)
    return LossConfig(aggregation=config.loss_metric, scale=ScaleConfig(method="benchmark_rmse"))


def _build_validation_scheme(config):
    from common_hpo import ValidationScheme
    from common_hpo.splits import VintagePolicy

    origin_selection = {
        "recent": "most_recent",
        "evenly_spaced": "evenly_spaced",
        "random": "random",
    }[config.inner_origin_selection]
    return ValidationScheme(
        training_window=config.inner_window,
        origin_selection=origin_selection,
        n_origins=config.inner_n_origins,
        horizons=config.target_horizons,
        min_train_length=max(2, config.optimization_horizon_quarters),
        origin_stride=config.inner_origin_stride,
        rolling_window_length=config.rolling_window_length,
        random_seed=config.inner_random_seed,
        vintage_policy=VintagePolicy.OUTER_VINTAGE_CONSISTENT,
    )


__all__ = ["execute_scope_runs"]
