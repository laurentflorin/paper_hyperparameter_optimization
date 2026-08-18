"""Scheduled, multi-cell hyperparameter-selection orchestration for MF-BVAR.

This is the mixed-frequency (Schorfheide-Song) *adapter* onto the shared
``common_hpo`` abstractions. It deliberately does **not** import any GLP code:
it reuses the model-independent contracts (:class:`SelectionPlan`,
:class:`SelectionSchedule`, :class:`LossConfig`, :class:`ValidationScheme`) and
mirrors the Stage 6-7 GLP design conventions (callback-injected selection and
forecasting, forecast caching by a stable system hash, canonical stitched
output, per-cell diagnostics, deterministic ordering, and rich run metadata).

Model-specific concepts are kept in their own fields rather than overloaded onto
the shared columns:

* ``forecast_variables`` -- the complete quarterly block used to build a
  dimensionally valid mixed-frequency state forecast (``GDP,INVFIX,GOV``).
* ``objective_variables`` -- the (possibly smaller) subset whose forecast errors
  enter the optimization objective.

The heavy model work is injected through two callbacks so the orchestration is
unit-testable without MBFVAR:

* ``selector(request) -> MFVARCellSelection`` tunes one hyperparameter vector for
  one :class:`~common_hpo.selection_scope.TargetCell` at one selection event.
* ``forecast_generator(request) -> numpy.ndarray`` produces the complete-system
  quarterly forecast (``system_horizons x system_variables``) for a selected
  hyperparameter vector at one outer origin.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from common_hpo.losses import LossConfig
from common_hpo.schedules import SelectionEvent, SelectionSchedule
from common_hpo.selection_scope import SelectionPlan, build_selection_plan
from common_hpo.splits import ValidationScheme


MODEL_NAME = "mfvar"
FORECAST_METHOD = "posterior_predictive_mean"


# --------------------------------------------------------------------------- #
# Requests and results exchanged with the injected callbacks.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MFVARCellSelectionRequest:
    """Everything a selector needs to tune one cell at one event.

    ``objective_variables`` are the cell's target variables (the loss subset).
    ``forecast_variables`` is the full quarterly block that the mixed-frequency
    state forecast is always built from, so a reduced objective never collapses
    the state dimension.
    """

    event: SelectionEvent
    cell_id: str
    objective_variables: tuple[str, ...]
    forecast_variables: tuple[str, ...]
    horizons: tuple[int, ...]
    loss_config: LossConfig
    validation_scheme: ValidationScheme
    origin_index: int
    origin_label: object
    seed: int | None


@dataclass(frozen=True)
class MFVARCellSelection:
    """A selector's result for one cell: the selected system plus provenance."""

    hyperparameter_vector: tuple[float, ...]
    named_parameters: Mapping[str, Any]
    selection_loss: float
    forecast_variables: tuple[str, ...] | None = None
    objective_variables: tuple[str, ...] | None = None
    optimizer_seed: int | None = None
    optimizer_budget: Mapping[str, Any] | None = None
    inner_window_start: int | None = None
    inner_window_end: int | None = None
    n_inner_origins: int | None = None
    validation_stride: int | None = None
    objective_draw_count: int | None = None
    failure_counts: Mapping[str, int] | None = None
    runtime_seconds: float | None = None
    seed_uncontrolled: bool = False
    seed_uncontrolled_reason: str | None = None

    def cache_token(self) -> tuple[float, ...]:
        return tuple(float(v) for v in self.hyperparameter_vector)


@dataclass(frozen=True)
class MFVARForecastRequest:
    """Everything a forecast generator needs to forecast one system at one origin."""

    hyperparameter_vector: tuple[float, ...]
    origin_index: int
    origin_label: object
    system_variables: tuple[str, ...]
    system_horizons: tuple[int, ...]
    forecast_variables: tuple[str, ...]
    cell_id: str
    event_id: str


Selector = Callable[[MFVARCellSelectionRequest], MFVARCellSelection]
ForecastGenerator = Callable[[MFVARForecastRequest], np.ndarray]


# --------------------------------------------------------------------------- #
# Result container.
# --------------------------------------------------------------------------- #
@dataclass
class MFVARExperimentResult:
    """The canonical panel, hyperparameter records, diagnostics, and metadata."""

    forecast_panel: list[dict[str, Any]]
    selected_hyperparameters: list[dict[str, Any]]
    forecast_panel_all_cells: list[dict[str, Any]]
    run_metadata: dict[str, Any]
    cache_stats: dict[str, int]


def _stable_system_hash(
    *,
    model: str,
    vintage_token: str,
    hyperparameter_vector: Sequence[float],
    origin_label: object,
) -> str:
    payload = {
        "model": model,
        "vintage_token": vintage_token,
        "hyperparameter_vector": [round(float(v), 12) for v in hyperparameter_vector],
        "origin_label": str(origin_label),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _default_seed(base_seed: int | None, event_number: int, cell_id: str) -> int | None:
    if base_seed is None:
        return None
    from experiment_provenance import stable_child_seed

    return stable_child_seed(base_seed, "mfvar-selection", event_number, cell_id)


def run_mfvar_selection_experiment(
    origin_labels: Sequence[object],
    *,
    selector: Selector,
    forecast_generator: ForecastGenerator,
    target_variables: Sequence[str],
    target_horizons: Sequence[int],
    forecast_variables: Sequence[str],
    loss_config: LossConfig,
    validation_scheme: ValidationScheme,
    plan: SelectionPlan | None = None,
    schedule: SelectionSchedule | None = None,
    system_variables: Sequence[str] | None = None,
    system_horizons: Sequence[int] | None = None,
    retain_off_target: bool = False,
    model_size: str = "paper",
    vintage_token: str = "default",
    loss_config_id: str = "loss-default",
    base_seed: int | None = None,
    vintage_policy: str = "outer_vintage_consistent",
) -> MFVARExperimentResult:
    """Run a scheduled, multi-cell MF-BVAR selection experiment.

    ``forecast_variables`` is the complete quarterly block from which the
    mixed-frequency state forecast is built for every cell. Each cell's target
    variables form its ``objective_variables`` (the loss subset), which must be a
    subset of ``forecast_variables``.

    When ``plan`` is ``None`` the experiment falls back to a single pooled cell
    over all target variables and horizons. When ``schedule`` is ``None``
    selection happens once (at the first origin) and is reused thereafter.
    """

    origin_labels = list(origin_labels)
    if not origin_labels:
        raise ValueError("origin_labels must be non-empty.")

    target_variables = tuple(str(v) for v in target_variables)
    target_horizons = tuple(int(h) for h in target_horizons)
    forecast_variables = tuple(str(v) for v in forecast_variables)

    if plan is None:
        plan = build_selection_plan("pooled", target_variables, target_horizons)
    if schedule is None:
        schedule = SelectionSchedule.once()

    if set(plan.target_variables) != set(target_variables):
        raise ValueError("plan target variables must match the requested target variables.")
    if set(plan.target_horizons) != set(target_horizons):
        raise ValueError("plan target horizons must match the requested target horizons.")

    # The objective subset of every cell must be resolvable in the full forecast
    # block, and every requested target must be forecastable by the full system.
    forecast_variable_set = set(forecast_variables)
    non_subset = [v for v in target_variables if v not in forecast_variable_set]
    if non_subset:
        raise ValueError(
            "target (objective) variables must be a subset of forecast_variables; "
            f"invalid entries: {non_subset}."
        )

    system_variables = (
        tuple(system_variables) if system_variables is not None else forecast_variables
    )
    system_horizons = (
        tuple(system_horizons) if system_horizons is not None else target_horizons
    )
    system_var_index = {v: i for i, v in enumerate(system_variables)}
    system_hor_index = {h: i for i, h in enumerate(system_horizons)}

    for variable in target_variables:
        if variable not in system_var_index:
            raise ValueError(f"target variable {variable!r} is not in the system output.")
    for variable in forecast_variables:
        if variable not in system_var_index:
            raise ValueError(f"forecast variable {variable!r} is not in the system output.")
    for horizon in target_horizons:
        if horizon not in system_hor_index:
            raise ValueError(f"target horizon {horizon} is not in the system output.")

    events = schedule.resolve(origin_labels)

    # ---- Select one system per (event, cell); record one row each. -------- #
    selections: dict[tuple[int, str], MFVARCellSelection] = {}
    selected_hyperparameters: list[dict[str, Any]] = []
    seed_uncontrolled_flags: list[dict[str, Any]] = []
    for event in events:
        for cell in plan.cells:
            seed = _default_seed(base_seed, event.event_number, cell.cell_id)
            request = MFVARCellSelectionRequest(
                event=event,
                cell_id=cell.cell_id,
                objective_variables=cell.variables,
                forecast_variables=forecast_variables,
                horizons=cell.horizons,
                loss_config=loss_config,
                validation_scheme=validation_scheme,
                origin_index=event.origin_index,
                origin_label=event.origin_label,
                seed=seed,
            )
            selection = selector(request)
            selections[(event.event_number, cell.cell_id)] = selection
            if selection.seed_uncontrolled:
                seed_uncontrolled_flags.append(
                    {
                        "cell_id": cell.cell_id,
                        "selection_event_id": event.event_id,
                        "reason": selection.seed_uncontrolled_reason,
                    }
                )
            selected_hyperparameters.append(
                {
                    "model": MODEL_NAME,
                    "model_size": model_size,
                    "selection_scope": plan.scope,
                    "cell_id": cell.cell_id,
                    "selection_event_id": event.event_id,
                    "tuned_for_variables": list(cell.variables),
                    "tuned_for_horizons": list(cell.horizons),
                    "selection_origin_index": event.origin_index,
                    "selected_on_origin": event.origin_label,
                    "applies_from_origin": origin_labels[event.applies_from_index],
                    "applies_to_origin": origin_labels[event.applies_to_index],
                    "applies_from_index": event.applies_from_index,
                    "applies_to_index": event.applies_to_index,
                    "selection_loss": selection.selection_loss,
                    # Model-specific provenance (kept in their own columns).
                    "hyperparameters": dict(selection.named_parameters),
                    "hyperparameter_vector": list(selection.hyperparameter_vector),
                    "forecast_variables": list(
                        selection.forecast_variables
                        if selection.forecast_variables is not None
                        else forecast_variables
                    ),
                    "objective_variables": list(
                        selection.objective_variables
                        if selection.objective_variables is not None
                        else cell.variables
                    ),
                    "optimizer_seed": selection.optimizer_seed,
                    "optimizer_budget": (
                        dict(selection.optimizer_budget)
                        if selection.optimizer_budget is not None
                        else None
                    ),
                    "inner_window_start": selection.inner_window_start,
                    "inner_window_end": selection.inner_window_end,
                    "n_inner_origins": selection.n_inner_origins,
                    "validation_stride": selection.validation_stride,
                    "objective_draw_count": selection.objective_draw_count,
                    "failure_counts": (
                        dict(selection.failure_counts)
                        if selection.failure_counts is not None
                        else None
                    ),
                    "runtime_seconds": selection.runtime_seconds,
                    "seed_uncontrolled": selection.seed_uncontrolled,
                    "seed_uncontrolled_reason": selection.seed_uncontrolled_reason,
                    "loss_config_id": loss_config_id,
                }
            )

    # ---- Generate forecasts per origin with caching; stitch panel. -------- #
    forecast_cache: dict[str, np.ndarray] = {}
    cache_stats = {"hits": 0, "misses": 0}

    def _system_forecast(
        selection: MFVARCellSelection,
        event: SelectionEvent,
        cell_id: str,
        origin_index: int,
    ) -> np.ndarray:
        origin_label = origin_labels[origin_index]
        key = _stable_system_hash(
            model=MODEL_NAME,
            vintage_token=vintage_token,
            hyperparameter_vector=selection.cache_token(),
            origin_label=origin_label,
        )
        if key in forecast_cache:
            cache_stats["hits"] += 1
            return forecast_cache[key]
        cache_stats["misses"] += 1
        request = MFVARForecastRequest(
            hyperparameter_vector=selection.cache_token(),
            origin_index=origin_index,
            origin_label=origin_label,
            system_variables=system_variables,
            system_horizons=system_horizons,
            forecast_variables=forecast_variables,
            cell_id=cell_id,
            event_id=event.event_id,
        )
        forecast = np.asarray(forecast_generator(request), dtype=float)
        expected = (len(system_horizons), len(system_variables))
        if forecast.shape != expected:
            raise ValueError(
                f"forecast generator returned shape {forecast.shape}, expected {expected}."
            )
        forecast_cache[key] = forecast
        return forecast

    canonical_rows: list[dict[str, Any]] = []
    all_cell_rows: list[dict[str, Any]] = []

    for origin_index, origin_label in enumerate(origin_labels):
        event = schedule.event_for_origin(origin_index, events)

        cell_forecasts: dict[str, np.ndarray] = {}
        for cell in plan.cells:
            selection = selections[(event.event_number, cell.cell_id)]
            cell_forecasts[cell.cell_id] = _system_forecast(
                selection, event, cell.cell_id, origin_index
            )

        for variable in target_variables:
            for horizon in target_horizons:
                responsible_cell = plan.cell_for(variable, horizon)
                forecast = cell_forecasts[responsible_cell.cell_id]
                value = float(
                    forecast[system_hor_index[horizon], system_var_index[variable]]
                )
                canonical_rows.append(
                    {
                        "model": MODEL_NAME,
                        "model_size": model_size,
                        "forecast_origin": origin_label,
                        "origin_index": origin_index,
                        "variable": variable,
                        "horizon": horizon,
                        "forecast": value,
                        "selection_scope": plan.scope,
                        "cell_id": responsible_cell.cell_id,
                        "selection_event_id": event.event_id,
                        "selected_on_origin": event.origin_label,
                        "tuned_for_variables": list(responsible_cell.variables),
                        "tuned_for_horizons": list(responsible_cell.horizons),
                        "forecast_method": FORECAST_METHOD,
                        "loss_config_id": loss_config_id,
                    }
                )

        if retain_off_target:
            for cell in plan.cells:
                forecast = cell_forecasts[cell.cell_id]
                for variable in system_variables:
                    for horizon in system_horizons:
                        is_canonical = (
                            variable in target_variables
                            and horizon in target_horizons
                            and plan.cell_for(variable, horizon).cell_id == cell.cell_id
                        )
                        all_cell_rows.append(
                            {
                                "model": MODEL_NAME,
                                "model_size": model_size,
                                "forecast_origin": origin_label,
                                "origin_index": origin_index,
                                "variable": variable,
                                "horizon": horizon,
                                "forecast": float(
                                    forecast[system_hor_index[horizon], system_var_index[variable]]
                                ),
                                "selection_scope": plan.scope,
                                "cell_id": cell.cell_id,
                                "selection_event_id": event.event_id,
                                "is_canonical": bool(is_canonical),
                                "diagnostic_only": True,
                            }
                        )

    canonical_rows.sort(
        key=lambda row: (row["model"], row["origin_index"], row["variable"], row["horizon"])
    )
    selected_hyperparameters.sort(key=lambda row: (row["applies_from_index"], row["cell_id"]))
    all_cell_rows.sort(
        key=lambda row: (
            row["model"],
            row["origin_index"],
            row["cell_id"],
            row["variable"],
            row["horizon"],
        )
    )

    run_metadata = {
        "model": MODEL_NAME,
        "model_size": model_size,
        "selection_plan": plan.to_dict(),
        "schedule": schedule.to_dict(),
        "selection_events": [event.to_dict() for event in events],
        "loss_config": loss_config.to_dict(),
        "validation_scheme": validation_scheme.to_dict(),
        "loss_config_id": loss_config_id,
        "target_variables": list(target_variables),
        "target_horizons": list(target_horizons),
        "forecast_variables": list(forecast_variables),
        "system_variables": list(system_variables),
        "system_horizons": list(system_horizons),
        "retain_off_target": retain_off_target,
        "forecast_method": FORECAST_METHOD,
        "base_seed": base_seed,
        "vintage_token": vintage_token,
        "vintage_policy": vintage_policy,
        "n_outer_origins": len(origin_labels),
        "seed_uncontrolled_events": seed_uncontrolled_flags,
        "reproducibility_limitations": (
            [
                "One or more selection events reported an uncontrollable upstream "
                "random generator; see seed_uncontrolled_events."
            ]
            if seed_uncontrolled_flags
            else []
        ),
    }

    return MFVARExperimentResult(
        forecast_panel=canonical_rows,
        selected_hyperparameters=selected_hyperparameters,
        forecast_panel_all_cells=all_cell_rows,
        run_metadata=run_metadata,
        cache_stats=cache_stats,
    )


__all__ = [
    "MODEL_NAME",
    "FORECAST_METHOD",
    "MFVARCellSelection",
    "MFVARCellSelectionRequest",
    "MFVARForecastRequest",
    "MFVARExperimentResult",
    "Selector",
    "ForecastGenerator",
    "run_mfvar_selection_experiment",
]
