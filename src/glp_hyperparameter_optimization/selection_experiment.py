"""Multi-cell, scheduled hyperparameter-selection orchestration for GLP.

This orchestrator ties together the shared abstractions -- a
:class:`~common_hpo.selection_scope.SelectionPlan`, a
:class:`~common_hpo.schedules.SelectionSchedule`, a
:class:`~glp_hyperparameter_optimization.search_config.GLPSearchConfig`, a
:class:`~common_hpo.losses.LossConfig`, and an inner
:class:`~common_hpo.splits.ValidationScheme` -- into a recursive experiment that
can produce a canonical stitched forecast panel for any selection scope
(pooled / horizon / variable / variable_horizon).

The heavy model work (hyperparameter selection and complete-system forecast
generation) is injected through two callbacks so the orchestration logic is
model-independent and unit-testable without covbayesvar:

* ``selector(request) -> CellSelection`` optimizes one hyperparameter vector for
  one :class:`~common_hpo.selection_scope.TargetCell` at one selection event.
* ``forecast_generator(request) -> numpy.ndarray`` produces the complete-system
  forecast (``system_horizons x system_variables``) under a selected
  hyperparameter vector at one outer origin.

Forecasts are cached by a stable hash of the model configuration, data-vintage
token, natural hyperparameter vector, and origin, so two cells that select
identical values do not refit the same system, while incompatible data vintages
never share cache entries.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from common_hpo.losses import LossConfig
from common_hpo.schedules import SelectionEvent, SelectionSchedule
from common_hpo.selection_scope import SelectionPlan, build_selection_plan
from common_hpo.splits import ValidationScheme

from .search_config import GLPSearchConfig


# --------------------------------------------------------------------------- #
# Requests and results exchanged with the injected callbacks.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CellSelectionRequest:
    """Everything a selector needs to tune one cell at one event."""

    event: SelectionEvent
    cell_id: str
    variables: tuple[str, ...]
    horizons: tuple[int, ...]
    search_config: GLPSearchConfig
    loss_config: LossConfig
    validation_scheme: ValidationScheme
    origin_index: int
    origin_label: object
    seed: int | None


@dataclass(frozen=True)
class CellSelection:
    """A selector's result for one cell: the selected system plus provenance.

    ``natural_vector`` is the covbayesvar-ordered natural hyperparameter vector
    used both for forecasting and for cache identity. ``named_parameters`` is a
    serializable mapping of selected natural parameters for reporting.
    """

    natural_vector: tuple[float, ...]
    named_parameters: Mapping[str, Any]
    selection_loss: float
    fixed_psi_source: str | None = None
    fixed_psi_values: tuple[float, ...] | None = None
    search_dimension: int | None = None
    inner_window_start: int | None = None
    inner_window_end: int | None = None
    n_inner_origins: int | None = None
    validation_stride: int | None = None
    optimizer_seed: int | None = None
    optimizer_budget: Mapping[str, Any] | None = None
    objective_draw_count: int | None = None
    failure_counts: Mapping[str, int] | None = None
    runtime_seconds: float | None = None

    def cache_token(self) -> tuple[float, ...]:
        return tuple(float(v) for v in self.natural_vector)


@dataclass(frozen=True)
class ForecastRequest:
    """Everything a forecast generator needs to forecast one system at one origin."""

    natural_vector: tuple[float, ...]
    origin_index: int
    origin_label: object
    system_variables: tuple[str, ...]
    system_horizons: tuple[int, ...]
    cell_id: str
    event_id: str


Selector = Callable[[CellSelectionRequest], CellSelection]
ForecastGenerator = Callable[[ForecastRequest], np.ndarray]


# --------------------------------------------------------------------------- #
# Result container.
# --------------------------------------------------------------------------- #
@dataclass
class GLPExperimentResult:
    """The canonical panel, hyperparameter records, diagnostics, and metadata."""

    forecast_panel: list[dict[str, Any]]
    selected_hyperparameters: list[dict[str, Any]]
    forecast_panel_all_cells: list[dict[str, Any]]
    run_metadata: dict[str, Any]
    cache_stats: dict[str, int]


def _stable_system_hash(
    *,
    model: str,
    model_size: str,
    vintage_token: str,
    natural_vector: Sequence[float],
    origin_label: object,
) -> str:
    payload = {
        "model": model,
        "model_size": model_size,
        "vintage_token": vintage_token,
        # Round to keep floating-point identity stable across equal selections.
        "natural_vector": [round(float(v), 12) for v in natural_vector],
        "origin_label": str(origin_label),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _default_seed(base_seed: int | None, event_number: int, cell_id: str) -> int | None:
    if base_seed is None:
        return None
    from experiment_provenance import stable_child_seed

    return stable_child_seed(base_seed, "glp-selection", event_number, cell_id)


def run_glp_selection_experiment(
    origin_labels: Sequence[object],
    *,
    selector: Selector,
    forecast_generator: ForecastGenerator,
    target_variables: Sequence[str],
    target_horizons: Sequence[int],
    search_config: GLPSearchConfig,
    loss_config: LossConfig,
    validation_scheme: ValidationScheme,
    plan: SelectionPlan | None = None,
    schedule: SelectionSchedule | None = None,
    system_variables: Sequence[str] | None = None,
    system_horizons: Sequence[int] | None = None,
    retain_off_target: bool = False,
    model: str = "glp",
    model_size: str = "small",
    vintage_token: str = "default",
    search_config_id: str = "search-default",
    loss_config_id: str = "loss-default",
    forecast_method: str = "posterior_predictive_mean",
    base_seed: int | None = None,
    vintage_policy: str = "outer_vintage_consistent",
) -> GLPExperimentResult:
    """Run a scheduled, multi-cell GLP selection experiment.

    When ``plan`` is ``None`` the experiment falls back to the legacy one-cell
    behavior: a single pooled cell over all target variables and horizons. When
    ``schedule`` is ``None`` selection happens once (at the first origin).
    """

    origin_labels = list(origin_labels)
    if not origin_labels:
        raise ValueError("origin_labels must be non-empty.")

    target_variables = tuple(str(v) for v in target_variables)
    target_horizons = tuple(int(h) for h in target_horizons)

    if plan is None:
        plan = build_selection_plan("pooled", target_variables, target_horizons)
    if schedule is None:
        schedule = SelectionSchedule.once()

    if set(plan.target_variables) != set(target_variables):
        raise ValueError("plan target variables must match the requested target variables.")
    if set(plan.target_horizons) != set(target_horizons):
        raise ValueError("plan target horizons must match the requested target horizons.")

    system_variables = tuple(system_variables) if system_variables is not None else target_variables
    system_horizons = tuple(system_horizons) if system_horizons is not None else target_horizons
    system_var_index = {v: i for i, v in enumerate(system_variables)}
    system_hor_index = {h: i for i, h in enumerate(system_horizons)}

    # Every target must be producible by the complete system.
    for variable in target_variables:
        if variable not in system_var_index:
            raise ValueError(f"target variable {variable!r} is not in the system output.")
    for horizon in target_horizons:
        if horizon not in system_hor_index:
            raise ValueError(f"target horizon {horizon} is not in the system output.")

    events = schedule.resolve(origin_labels)

    # ---- 1-2. Select one system per (event, cell); record one row each. ---- #
    selections: dict[tuple[int, str], CellSelection] = {}
    selected_hyperparameters: list[dict[str, Any]] = []
    for event in events:
        for cell in plan.cells:
            seed = _default_seed(base_seed, event.event_number, cell.cell_id)
            request = CellSelectionRequest(
                event=event,
                cell_id=cell.cell_id,
                variables=cell.variables,
                horizons=cell.horizons,
                search_config=search_config,
                loss_config=loss_config,
                validation_scheme=validation_scheme,
                origin_index=event.origin_index,
                origin_label=event.origin_label,
                seed=seed,
            )
            selection = selector(request)
            selections[(event.event_number, cell.cell_id)] = selection
            selected_hyperparameters.append(
                {
                    "model": model,
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
                    "natural_parameters": dict(selection.named_parameters),
                    "natural_vector": list(selection.natural_vector),
                    "fixed_psi_source": selection.fixed_psi_source,
                    "fixed_psi_values": (
                        list(selection.fixed_psi_values)
                        if selection.fixed_psi_values is not None
                        else None
                    ),
                    "search_dimension": selection.search_dimension,
                    "inner_window_start": selection.inner_window_start,
                    "inner_window_end": selection.inner_window_end,
                    "n_inner_origins": selection.n_inner_origins,
                    "validation_stride": selection.validation_stride,
                    "selection_loss": selection.selection_loss,
                    "optimizer_seed": selection.optimizer_seed,
                    "optimizer_budget": (
                        dict(selection.optimizer_budget)
                        if selection.optimizer_budget is not None
                        else None
                    ),
                    "objective_draw_count": selection.objective_draw_count,
                    "failure_counts": (
                        dict(selection.failure_counts)
                        if selection.failure_counts is not None
                        else None
                    ),
                    "runtime_seconds": selection.runtime_seconds,
                    "search_config_id": search_config_id,
                    "loss_config_id": loss_config_id,
                }
            )

    # ---- 3-5. Generate forecasts per origin with caching; stitch panel. ---- #
    forecast_cache: dict[str, np.ndarray] = {}
    cache_stats = {"hits": 0, "misses": 0}

    def _system_forecast(
        selection: CellSelection, event: SelectionEvent, cell_id: str, origin_index: int
    ) -> np.ndarray:
        origin_label = origin_labels[origin_index]
        key = _stable_system_hash(
            model=model,
            model_size=model_size,
            vintage_token=vintage_token,
            natural_vector=selection.cache_token(),
            origin_label=origin_label,
        )
        if key in forecast_cache:
            cache_stats["hits"] += 1
            return forecast_cache[key]
        cache_stats["misses"] += 1
        request = ForecastRequest(
            natural_vector=selection.cache_token(),
            origin_index=origin_index,
            origin_label=origin_label,
            system_variables=system_variables,
            system_horizons=system_horizons,
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

        # Forecast each cell's complete system once per origin (cached).
        cell_forecasts: dict[str, np.ndarray] = {}
        for cell in plan.cells:
            selection = selections[(event.event_number, cell.cell_id)]
            cell_forecasts[cell.cell_id] = _system_forecast(
                selection, event, cell.cell_id, origin_index
            )

        # Canonical stitched panel: one row per target (variable, horizon).
        for variable in target_variables:
            for horizon in target_horizons:
                responsible_cell = plan.cell_for(variable, horizon)
                selection = selections[(event.event_number, responsible_cell.cell_id)]
                forecast = cell_forecasts[responsible_cell.cell_id]
                value = float(
                    forecast[system_hor_index[horizon], system_var_index[variable]]
                )
                canonical_rows.append(
                    {
                        "model": model,
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
                        "forecast_method": forecast_method,
                        "search_config_id": search_config_id,
                        "loss_config_id": loss_config_id,
                    }
                )

        # Optional diagnostic panel: every cell's complete-system forecast.
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
                                "model": model,
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

    # ---- Deterministic ordering. ---- #
    canonical_rows.sort(
        key=lambda row: (row["model"], row["origin_index"], row["variable"], row["horizon"])
    )
    selected_hyperparameters.sort(
        key=lambda row: (row["applies_from_index"], row["cell_id"])
    )
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
        "model": model,
        "model_size": model_size,
        "selection_plan": {
            "scope": plan.scope,
            "target_variables": list(plan.target_variables),
            "target_horizons": list(plan.target_horizons),
            "cells": [cell.to_dict() for cell in plan.cells],
        },
        "schedule": schedule.to_dict(),
        "selection_events": [event.to_dict() for event in events],
        "loss_config": loss_config.to_dict(),
        "validation_scheme": validation_scheme.to_dict(),
        "search_config": search_config.to_dict(),
        "search_config_id": search_config_id,
        "loss_config_id": loss_config_id,
        "target_variables": list(target_variables),
        "target_horizons": list(target_horizons),
        "system_variables": list(system_variables),
        "system_horizons": list(system_horizons),
        "retain_off_target": retain_off_target,
        "forecast_method": forecast_method,
        "base_seed": base_seed,
        "vintage_token": vintage_token,
        "vintage_policy": vintage_policy,
        "n_outer_origins": len(origin_labels),
    }

    return GLPExperimentResult(
        forecast_panel=canonical_rows,
        selected_hyperparameters=selected_hyperparameters,
        forecast_panel_all_cells=all_cell_rows,
        run_metadata=run_metadata,
        cache_stats=cache_stats,
    )


__all__ = [
    "CellSelection",
    "CellSelectionRequest",
    "ForecastGenerator",
    "ForecastRequest",
    "GLPExperimentResult",
    "Selector",
    "run_glp_selection_experiment",
]
