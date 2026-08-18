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
from typing import TYPE_CHECKING, Any, Sequence

import pandas as pd

from .config import (
    PAPER_NBURN_PERC,
    PAPER_NLAGS,
    PAPER_NSIM,
    PAPER_TEMPORAL_AGGREGATION,
    PAPER_THINING,
    DEFAULT_OPTIMIZATION_NSIM,
    forecast_origin_dates,
)
from .horizon_mapping import (
    quarterly_horizon_to_state_rows,
    required_forecast_months,
    target_quarter_for_origin,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .selection_experiment import MFVARForecastRequest


def _write_outputs(output_dir: Path, result, *, save_all_cell_forecasts: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result.forecast_panel).to_csv(output_dir / "forecast_panel.csv", index=False)
    pd.DataFrame(result.selected_hyperparameters).to_csv(
        output_dir / "selected_hyperparameters.csv", index=False
    )
    pd.DataFrame(columns=["forecast_origin", "error"]).to_csv(
        output_dir / "failed_origins.csv", index=False
    )
    (output_dir / "run_metadata.json").write_text(
        json.dumps(result.run_metadata, indent=2, default=str), encoding="utf-8"
    )
    if save_all_cell_forecasts:
        pd.DataFrame(result.forecast_panel_all_cells).to_csv(
            output_dir / "forecast_panel_all_cells.csv", index=False
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
        model.forecast(required_forecast_months(max(request.system_horizons)))
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
        _write_outputs(
            plan.output_dir, result, save_all_cell_forecasts=config.save_all_cell_forecasts
        )


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
