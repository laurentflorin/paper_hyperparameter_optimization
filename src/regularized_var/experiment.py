"""Nested hyperparameter selection and recursive experiments for the ridge VAR.

This module wires the NumPy-only ridge VAR estimator (:mod:`regularized_var`)
into the shared, model-independent selection contracts in :mod:`common_hpo`:

* :class:`~common_hpo.SelectionPlan` -- groups forecast targets into the four
  core scopes (``pooled``, ``horizon``, ``variable``, ``variable_horizon``).
* :class:`~common_hpo.ValidationScheme` / ``build_validation_splits`` -- builds
  leakage-safe inner-validation origins for selection and outer origins for
  reporting.
* :class:`~common_hpo.SelectionSchedule` -- decides at which outer origins
  hyperparameters are re-selected and reused.
* :class:`~common_hpo.LossConfig` / ``evaluate_selection_loss`` -- computes the
  scalar validation loss minimized during selection.

The scientific search is the deterministic grid in :mod:`regularized_var.tuning`
-- no Bayesian optimization and no Mango dependency. Standardization, when
enabled, is fit strictly on each training fold and validation inputs are
transformed with the training-fold statistics; forecasts are mapped back to the
evaluation scale before any error is computed.

Benchmark models reuse the same outer origins and output schema:

* ``var_aic`` / ``var_bic`` -- OLS VAR with lag chosen by AIC / BIC.
* ``var_nested_loss`` -- OLS VAR with lag chosen by nested forecast loss.
* ``ar_univariate`` -- per-variable AR with an AIC lag rule.
* ``no_change`` -- naive persistence forecast on the transformed series.

None of the benchmarks use the outer test sample to choose lag lengths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping, Sequence

import numpy as np

from common_hpo import (
    ForecastErrorRecord,
    LossConfig,
    ScaleConfig,
    SelectionPlan,
    SelectionSchedule,
    ValidationScheme,
    build_selection_plan,
    build_validation_splits,
    evaluate_selection_loss,
)

from .data import PanelData, Standardizer
from .direct import direct_forecast, fit_direct_ridge_var
from .estimators import fit_ridge_var
from .forecasting import iterated_forecast
from .tuning import (
    DEFAULT_TIE_TOLERANCE,
    RidgeCandidate,
    RidgeGridSpec,
    RidgeSelection,
    default_grid_spec,
    enumerate_grid,
    grid_size,
    select_best_candidate,
)


__all__ = [
    "RidgeExperimentConfig",
    "CellSelection",
    "ScopeExperimentResult",
    "BenchmarkResult",
    "FORECAST_PANEL_COLUMNS",
    "BENCHMARK_STRATEGIES",
    "select_for_cell",
    "run_scope_experiment",
    "run_benchmark",
    "estimate_fit_counts",
    "write_scope_outputs",
    "write_benchmark_outputs",
    "default_experiment_config",
]


# Canonical forecast-panel columns. This is a schema-compatible subset of the
# GLP / MF-BVAR ``forecast_panel.csv`` (same key columns and the mean/median
# point-forecast and error columns). Ridge point forecasts populate the
# ``*_metric`` columns; the level columns are recorded but left as NaN because
# the experiment operates on an already-transformed series.
FORECAST_PANEL_COLUMNS = (
    "strategy",
    "forecast_origin",
    "group",
    "target_quarter",
    "horizon_quarters",
    "variable",
    "forecast_method",
    "actual_level",
    "actual_metric",
    "mean_level",
    "mean_metric",
    "median_level",
    "median_metric",
    "error_metric",
)

BENCHMARK_STRATEGIES = (
    "var_aic",
    "var_bic",
    "var_nested_loss",
    "ar_univariate",
    "no_change",
)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RidgeExperimentConfig:
    """Everything needed to run a ridge scope experiment reproducibly."""

    target_variables: tuple[str, ...]
    target_horizons: tuple[int, ...]
    grid_spec: RidgeGridSpec
    outer_scheme: ValidationScheme
    inner_scheme: ValidationScheme
    selection_schedule: SelectionSchedule
    loss_config: LossConfig
    preprocessing: str = "standardize"
    forecast_method: str = "iterated"
    horizon_row_offset: int = 1
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE
    benchmark_lag_orders: tuple[int, ...] = (1, 2, 4)
    base_seed: int | None = None

    def __post_init__(self) -> None:
        if self.preprocessing not in ("none", "standardize"):
            raise ValueError("preprocessing must be 'none' or 'standardize'.")
        if self.forecast_method not in ("iterated", "direct"):
            raise ValueError("forecast_method must be 'iterated' or 'direct'.")
        if int(self.horizon_row_offset) < 1:
            raise ValueError("horizon_row_offset must be a positive integer.")
        object.__setattr__(self, "target_variables", tuple(self.target_variables))
        object.__setattr__(self, "target_horizons", tuple(int(h) for h in self.target_horizons))
        object.__setattr__(
            self, "benchmark_lag_orders", tuple(int(p) for p in self.benchmark_lag_orders)
        )

    @property
    def standardize(self) -> bool:
        return self.preprocessing == "standardize"

    def horizon_offsets(self):
        offset = int(self.horizon_row_offset)
        return {int(h): int(h) * offset for h in self.target_horizons}

    def to_metadata(self) -> dict[str, object]:
        return {
            "target_variables": list(self.target_variables),
            "target_horizons": list(self.target_horizons),
            "grid_spec": self.grid_spec.to_dict(),
            "grid_size": grid_size(self.grid_spec),
            "outer_validation_scheme": self.outer_scheme.to_dict(),
            "inner_validation_scheme": self.inner_scheme.to_dict(),
            "selection_schedule": self.selection_schedule.to_dict(),
            "loss_config": self.loss_config.to_dict(),
            "preprocessing": self.preprocessing,
            "forecast_method": self.forecast_method,
            "horizon_row_offset": int(self.horizon_row_offset),
            "tie_tolerance": self.tie_tolerance,
            "benchmark_lag_orders": list(self.benchmark_lag_orders),
            "base_seed": self.base_seed,
        }


def default_experiment_config(
    target_variables: Sequence[str],
    target_horizons: Sequence[int],
    *,
    preprocessing: str = "standardize",
    forecast_method: str = "iterated",
    outer_n_origins: int = 8,
    inner_n_origins: int = 4,
    min_train_length: int = 40,
    selection_schedule: SelectionSchedule | None = None,
    grid_spec: RidgeGridSpec | None = None,
    loss_metric: str = "rmse",
    loss_scaling: str = "none",
) -> RidgeExperimentConfig:
    """Build a documented default configuration for a single-frequency panel."""

    horizons = tuple(int(h) for h in target_horizons)
    outer_scheme = ValidationScheme(
        training_window="expanding",
        origin_selection="most_recent",
        n_origins=int(outer_n_origins),
        horizons=horizons,
        min_train_length=int(min_train_length),
    )
    inner_scheme = ValidationScheme(
        training_window="expanding",
        origin_selection="most_recent",
        n_origins=int(inner_n_origins),
        horizons=horizons,
        min_train_length=int(min_train_length),
    )
    scale = ScaleConfig(method="none") if loss_scaling == "none" else ScaleConfig(method=loss_scaling)
    return RidgeExperimentConfig(
        target_variables=tuple(target_variables),
        target_horizons=horizons,
        grid_spec=grid_spec or default_grid_spec(),
        outer_scheme=outer_scheme,
        inner_scheme=inner_scheme,
        selection_schedule=selection_schedule or SelectionSchedule.once(),
        loss_config=LossConfig(aggregation=loss_metric, scale=scale),
        preprocessing=preprocessing,
        forecast_method=forecast_method,
    )


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CellSelection:
    """A selected candidate for one cell, at one selection event."""

    event_id: str
    origin_index: int
    origin_label: object
    cell_id: str
    group: str
    selection: RidgeSelection


@dataclass(frozen=True)
class ScopeExperimentResult:
    scope: str
    selection_plan: SelectionPlan
    forecast_rows: tuple[dict[str, object], ...]
    selection_rows: tuple[CellSelection, ...]
    failed_origins: tuple[dict[str, object], ...]
    metadata: dict[str, object]


@dataclass(frozen=True)
class BenchmarkResult:
    strategy: str
    forecast_rows: tuple[dict[str, object], ...]
    selection_rows: tuple[dict[str, object], ...]
    failed_origins: tuple[dict[str, object], ...]
    metadata: dict[str, object]


# --------------------------------------------------------------------------- #
# Core forecasting helpers (all NumPy)
# --------------------------------------------------------------------------- #
def _forecast_ridge(
    block: np.ndarray,
    *,
    p: int,
    lam: float,
    alpha: float,
    kappa: float,
    standardize: bool,
    horizons: Sequence[int],
    variable_names: Sequence[str],
) -> dict[int, np.ndarray] | None:
    """Fit a ridge VAR on ``block`` and return raw-scale forecasts per horizon.

    Returns ``None`` when the block is too short for ``p`` lags or when the
    resulting forecasts are non-finite (e.g. an explosive candidate), so callers
    can reject the candidate rather than crash.
    """

    if block.shape[0] <= p:
        return None
    max_h = max(int(h) for h in horizons)
    std = Standardizer.fit(block, enabled=standardize)
    z = std.transform(block)
    result = fit_ridge_var(z, p, lam=lam, alpha=alpha, kappa=kappa, variable_names=variable_names)
    fc_z = iterated_forecast(result, z[-p:], max_h)
    fc = std.inverse_transform(fc_z)
    if not np.all(np.isfinite(fc)):
        return None
    return {int(h): fc[int(h) - 1] for h in horizons}


def _forecast_direct(
    block: np.ndarray,
    *,
    p: int,
    lam: float,
    alpha: float,
    kappa: float,
    standardize: bool,
    horizons: Sequence[int],
    variable_names: Sequence[str],
) -> dict[int, np.ndarray] | None:
    """Fit an independent direct model per horizon and return raw-scale forecasts.

    Each horizon gets its own coefficient system (a structural requirement of
    direct forecasting) but the caller supplies a single hyperparameter vector.
    Standardization is fit fold-locally on ``block`` and forecasts are mapped
    back to the evaluation scale. Returns ``None`` when any requested horizon is
    infeasible for the block or produces non-finite forecasts.
    """

    std = Standardizer.fit(block, enabled=standardize)
    z = std.transform(block)
    out: dict[int, np.ndarray] = {}
    for horizon in horizons:
        h = int(horizon)
        if block.shape[0] < p + h:
            return None
        result = fit_direct_ridge_var(
            z, p, h, lam=lam, alpha=alpha, kappa=kappa, variable_names=variable_names
        )
        fc_z = direct_forecast(result, z[-p:])
        fc = std.inverse_transform(fc_z)
        if not np.all(np.isfinite(fc)):
            return None
        out[h] = fc
    return out


def _forecast_cell(
    method: str,
    block: np.ndarray,
    *,
    p: int,
    lam: float,
    alpha: float,
    kappa: float,
    standardize: bool,
    horizons: Sequence[int],
    variable_names: Sequence[str],
) -> dict[int, np.ndarray] | None:
    """Dispatch to the iterated or direct forecaster by ``method``."""

    if method == "direct":
        return _forecast_direct(
            block, p=p, lam=lam, alpha=alpha, kappa=kappa,
            standardize=standardize, horizons=horizons, variable_names=variable_names,
        )
    return _forecast_ridge(
        block, p=p, lam=lam, alpha=alpha, kappa=kappa,
        standardize=standardize, horizons=horizons, variable_names=variable_names,
    )


def _no_change_forecast(
    block: np.ndarray, horizons: Sequence[int]
) -> dict[int, np.ndarray]:
    last = block[-1]
    return {int(h): last.copy() for h in horizons}


def _var_information_criteria(
    block: np.ndarray, p: int, *, standardize: bool, variable_names: Sequence[str]
) -> tuple[float, float]:
    """Return ``(aic, bic)`` for an OLS VAR(p) fit on ``block``.

    Uses the Lütkepohl small-sample forms based on the maximum-likelihood
    residual covariance (``ddof=0``):

    ``AIC(p) = ln det(Sigma_p) + 2 * p * n^2 / T_eff``
    ``BIC(p) = ln det(Sigma_p) + ln(T_eff) * p * n^2 / T_eff``
    """

    std = Standardizer.fit(block, enabled=standardize)
    z = std.transform(block)
    result = fit_ridge_var(z, p, lam=0.0, ddof=0, variable_names=variable_names)
    sigma = result.residual_covariance
    _, logdet = np.linalg.slogdet(sigma)
    t_eff = block.shape[0] - p
    n = block.shape[1]
    penalty_terms = p * n * n
    aic = float(logdet + 2.0 * penalty_terms / t_eff)
    bic = float(logdet + np.log(t_eff) * penalty_terms / t_eff)
    return aic, bic


# --------------------------------------------------------------------------- #
# Nested selection for one cell
# --------------------------------------------------------------------------- #
def select_for_cell(
    panel: PanelData,
    variables: Sequence[str],
    horizons: Sequence[int],
    *,
    candidates: Sequence[RidgeCandidate],
    inner_scheme: ValidationScheme,
    loss_config: LossConfig,
    standardize: bool,
    outer_info_cutoff: int,
    horizon_offsets: Mapping[int, int],
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE,
    forecast_method: str = "iterated",
) -> RidgeSelection:
    """Select the best ridge candidate for one cell by nested validation.

    Inner-validation splits are built with an information cutoff at the outer
    origin, so no outer target can leak into selection. Standardization is fit
    fold-locally inside the forecaster. Selection uses the same forecasting
    architecture (``forecast_method``) that will generate the outer forecasts,
    so hyperparameters are chosen for the architecture actually deployed.
    """

    splits = build_validation_splits(
        panel.n_observations,
        inner_scheme,
        horizon_offsets,
        outer_info_cutoff=int(outer_info_cutoff),
        date_labels=panel.date_labels,
    )
    var_indices = {var: panel.column_index(var) for var in variables}

    evaluated: list[tuple[RidgeCandidate, float]] = []
    for candidate in candidates:
        records: list[ForecastErrorRecord] = []
        feasible = True
        for split in splits:
            block = panel.values[split.train_start : split.train_end + 1]
            forecasts = _forecast_cell(
                forecast_method,
                block,
                p=candidate.p,
                lam=candidate.lam,
                alpha=candidate.alpha,
                kappa=candidate.kappa,
                standardize=standardize,
                horizons=horizons,
                variable_names=panel.variable_names,
            )
            if forecasts is None:
                feasible = False
                break
            for horizon in horizons:
                target_row = split.target_for(int(horizon))
                for var in variables:
                    j = var_indices[var]
                    records.append(
                        ForecastErrorRecord(
                            origin=split.origin,
                            variable=var,
                            horizon=int(horizon),
                            forecast=float(forecasts[int(horizon)][j]),
                            realization=float(panel.values[target_row, j]),
                        )
                    )
        if not feasible or not records:
            evaluated.append((candidate, float("inf")))
            continue
        loss = evaluate_selection_loss(records, loss_config).value
        evaluated.append((candidate, float(loss)))

    return select_best_candidate(evaluated, tolerance=tie_tolerance)


# --------------------------------------------------------------------------- #
# Outer recursive experiment
# --------------------------------------------------------------------------- #
def _origin_label(panel: PanelData, row: int) -> object:
    return panel.label_for(row)


def _forecast_row(
    *,
    strategy: str,
    origin_label: object,
    group: str,
    target_label: object,
    horizon: int,
    variable: str,
    forecast: float,
    realization: float,
    forecast_method: str = "iterated",
) -> dict[str, object]:
    return {
        "strategy": strategy,
        "forecast_origin": origin_label,
        "group": group,
        "target_quarter": target_label,
        "horizon_quarters": int(horizon),
        "variable": variable,
        "forecast_method": forecast_method,
        "actual_level": float("nan"),
        "actual_metric": float(realization),
        "mean_level": float("nan"),
        "mean_metric": float(forecast),
        "median_level": float("nan"),
        "median_metric": float(forecast),
        # GLP convention: error = forecast - actual (mean_metric - actual_metric).
        "error_metric": float(forecast) - float(realization),
    }


def run_scope_experiment(
    panel: PanelData,
    scope: str,
    config: RidgeExperimentConfig,
    *,
    strategy: str = "ridge_var",
) -> ScopeExperimentResult:
    """Run one scope's nested-selection experiment over all outer origins."""

    plan = build_selection_plan(scope, config.target_variables, config.target_horizons)
    offsets = config.horizon_offsets()
    outer_splits = build_validation_splits(
        panel.n_observations,
        config.outer_scheme,
        offsets,
        date_labels=panel.date_labels,
    )
    candidates = enumerate_grid(config.grid_spec)

    origin_labels = [_origin_label(panel, split.origin) for split in outer_splits]
    events = config.selection_schedule.resolve(origin_labels)
    event_by_origin = {
        origin_index: config.selection_schedule.event_for_origin(origin_index, events)
        for origin_index in range(len(outer_splits))
    }
    selection_event_origins = {event.origin_index for event in events}

    forecast_rows: list[dict[str, object]] = []
    selection_rows: list[CellSelection] = []
    failed: list[dict[str, object]] = []

    current: dict[str, RidgeSelection] = {}
    for origin_index, split in enumerate(outer_splits):
        event = event_by_origin[origin_index]
        origin_label = origin_labels[origin_index]
        if origin_index in selection_event_origins:
            current = {}
            for cell in plan.cells:
                group = cell.group_name or "all"
                try:
                    selection = select_for_cell(
                        panel,
                        cell.variables,
                        cell.horizons,
                        candidates=candidates,
                        inner_scheme=config.inner_scheme,
                        loss_config=config.loss_config,
                        standardize=config.standardize,
                        outer_info_cutoff=split.origin,
                        horizon_offsets=offsets,
                        tie_tolerance=config.tie_tolerance,
                        forecast_method=config.forecast_method,
                    )
                except Exception as exc:  # noqa: BLE001 - record and continue
                    failed.append(
                        {
                            "forecast_origin": origin_label,
                            "cell_id": cell.cell_id,
                            "stage": "selection",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                current[cell.cell_id] = selection
                selection_rows.append(
                    CellSelection(
                        event_id=event.event_id,
                        origin_index=origin_index,
                        origin_label=origin_label,
                        cell_id=cell.cell_id,
                        group=group,
                        selection=selection,
                    )
                )

        block = panel.values[: split.origin + 1]
        for cell in plan.cells:
            selection = current.get(cell.cell_id)
            if selection is None:
                continue
            group = cell.group_name or "all"
            candidate = selection.candidate
            forecasts = _forecast_cell(
                config.forecast_method,
                block,
                p=candidate.p,
                lam=candidate.lam,
                alpha=candidate.alpha,
                kappa=candidate.kappa,
                standardize=config.standardize,
                horizons=cell.horizons,
                variable_names=panel.variable_names,
            )
            if forecasts is None:
                failed.append(
                    {
                        "forecast_origin": origin_label,
                        "cell_id": cell.cell_id,
                        "stage": "forecast",
                        "error": "non-finite or infeasible ridge forecast",
                    }
                )
                continue
            for horizon in cell.horizons:
                target_row = split.target_for(int(horizon))
                target_label = _origin_label(panel, target_row)
                for var in cell.variables:
                    j = panel.column_index(var)
                    forecast_rows.append(
                        _forecast_row(
                            strategy=strategy,
                            origin_label=origin_label,
                            group=group,
                            target_label=target_label,
                            horizon=int(horizon),
                            variable=var,
                            forecast=float(forecasts[int(horizon)][j]),
                            realization=float(panel.values[target_row, j]),
                            forecast_method=config.forecast_method,
                        )
                    )

    metadata = {
        "strategy": strategy,
        "scope": scope,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_outer_origins": len(outer_splits),
        "n_selection_events": len(events),
        "n_target_cells": len(plan.cells),
        "selection_plan": plan.to_dict(),
        "panel": panel.to_metadata(),
        **config.to_metadata(),
    }
    return ScopeExperimentResult(
        scope=scope,
        selection_plan=plan,
        forecast_rows=tuple(forecast_rows),
        selection_rows=tuple(selection_rows),
        failed_origins=tuple(failed),
        metadata=metadata,
    )


# --------------------------------------------------------------------------- #
# Benchmarks
# --------------------------------------------------------------------------- #
def _select_var_lag(
    block: np.ndarray,
    *,
    criterion: str,
    lag_orders: Sequence[int],
    standardize: bool,
    variable_names: Sequence[str],
) -> int:
    """Choose a VAR lag by AIC or BIC on ``block`` only (no outer test leakage)."""

    scored: list[tuple[float, int]] = []
    for p in sorted(dict.fromkeys(int(p) for p in lag_orders)):
        if block.shape[0] <= p + 1:
            continue
        aic, bic = _var_information_criteria(
            block, p, standardize=standardize, variable_names=variable_names
        )
        value = aic if criterion == "aic" else bic
        scored.append((float(value), p))
    if not scored:
        return 1
    # Deterministic tie-break: smaller criterion, then smaller lag.
    scored.sort(key=lambda item: (item[0], item[1]))
    return scored[0][1]


def _select_var_lag_nested(
    panel: PanelData,
    *,
    lag_orders: Sequence[int],
    inner_scheme: ValidationScheme,
    loss_config: LossConfig,
    standardize: bool,
    outer_info_cutoff: int,
    horizon_offsets: Mapping[int, int],
    variables: Sequence[str],
    horizons: Sequence[int],
) -> int:
    """Choose a VAR lag by nested forecast loss (OLS VAR, lam=0)."""

    candidates = [RidgeCandidate(lam=0.0, p=int(p), alpha=0.0, kappa=1.0) for p in lag_orders]
    selection = select_for_cell(
        panel,
        variables,
        horizons,
        candidates=candidates,
        inner_scheme=inner_scheme,
        loss_config=loss_config,
        standardize=standardize,
        outer_info_cutoff=outer_info_cutoff,
        horizon_offsets=horizon_offsets,
    )
    return selection.candidate.p


def _select_ar_lag(
    column_block: np.ndarray,
    *,
    lag_orders: Sequence[int],
    standardize: bool,
) -> int:
    """Choose a univariate AR lag by AIC on the single series (documented rule)."""

    block = column_block.reshape(-1, 1)
    scored: list[tuple[float, int]] = []
    for p in sorted(dict.fromkeys(int(p) for p in lag_orders)):
        if block.shape[0] <= p + 1:
            continue
        aic, _ = _var_information_criteria(
            block, p, standardize=standardize, variable_names=("series",)
        )
        scored.append((float(aic), p))
    if not scored:
        return 1
    scored.sort(key=lambda item: (item[0], item[1]))
    return scored[0][1]


def run_benchmark(
    panel: PanelData,
    strategy: str,
    config: RidgeExperimentConfig,
) -> BenchmarkResult:
    """Run one benchmark strategy over the same outer origins and schema."""

    if strategy not in BENCHMARK_STRATEGIES:
        raise ValueError(
            f"unknown benchmark strategy {strategy!r}; expected one of {BENCHMARK_STRATEGIES}."
        )

    offsets = config.horizon_offsets()
    outer_splits = build_validation_splits(
        panel.n_observations,
        config.outer_scheme,
        offsets,
        date_labels=panel.date_labels,
    )
    variables = config.target_variables
    horizons = config.target_horizons

    forecast_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []

    for split in outer_splits:
        origin_label = _origin_label(panel, split.origin)
        block = panel.values[: split.origin + 1]

        try:
            forecasts, chosen = _benchmark_forecasts(
                panel, strategy, block, config, offsets, split.origin
            )
        except Exception as exc:  # noqa: BLE001
            failed.append(
                {
                    "forecast_origin": origin_label,
                    "stage": "forecast",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        for name, value in chosen.items():
            selection_rows.append(
                {
                    "forecast_origin": origin_label,
                    "strategy": strategy,
                    "group": "all",
                    "parameter": name,
                    "value": value,
                }
            )

        for horizon in horizons:
            target_row = split.target_for(int(horizon))
            target_label = _origin_label(panel, target_row)
            for var in variables:
                j = panel.column_index(var)
                forecast_rows.append(
                    _forecast_row(
                        strategy=strategy,
                        origin_label=origin_label,
                        group="all",
                        target_label=target_label,
                        horizon=int(horizon),
                        variable=var,
                        forecast=float(forecasts[int(horizon)][j]),
                        realization=float(panel.values[target_row, j]),
                    )
                )

    metadata = {
        "strategy": strategy,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_outer_origins": len(outer_splits),
        "benchmark_lag_orders": list(config.benchmark_lag_orders),
        "preprocessing": config.preprocessing,
        "outer_validation_scheme": config.outer_scheme.to_dict(),
        "panel": panel.to_metadata(),
    }
    return BenchmarkResult(
        strategy=strategy,
        forecast_rows=tuple(forecast_rows),
        selection_rows=tuple(selection_rows),
        failed_origins=tuple(failed),
        metadata=metadata,
    )


def _benchmark_forecasts(
    panel: PanelData,
    strategy: str,
    block: np.ndarray,
    config: RidgeExperimentConfig,
    offsets: Mapping[int, int],
    outer_info_cutoff: int,
) -> tuple[dict[int, np.ndarray], dict[str, object]]:
    horizons = config.target_horizons
    names = panel.variable_names

    if strategy == "no_change":
        return _no_change_forecast(block, horizons), {}

    if strategy in ("var_aic", "var_bic"):
        criterion = "aic" if strategy == "var_aic" else "bic"
        p = _select_var_lag(
            block,
            criterion=criterion,
            lag_orders=config.benchmark_lag_orders,
            standardize=config.standardize,
            variable_names=names,
        )
        forecasts = _forecast_ridge(
            block, p=p, lam=0.0, alpha=0.0, kappa=1.0,
            standardize=config.standardize, horizons=horizons, variable_names=names,
        )
        if forecasts is None:
            raise ValueError(f"{strategy}: infeasible VAR({p}) forecast.")
        return forecasts, {"lag_order": p}

    if strategy == "var_nested_loss":
        p = _select_var_lag_nested(
            panel,
            lag_orders=config.benchmark_lag_orders,
            inner_scheme=config.inner_scheme,
            loss_config=config.loss_config,
            standardize=config.standardize,
            outer_info_cutoff=outer_info_cutoff,
            horizon_offsets=offsets,
            variables=config.target_variables,
            horizons=horizons,
        )
        forecasts = _forecast_ridge(
            block, p=p, lam=0.0, alpha=0.0, kappa=1.0,
            standardize=config.standardize, horizons=horizons, variable_names=names,
        )
        if forecasts is None:
            raise ValueError(f"var_nested_loss: infeasible VAR({p}) forecast.")
        return forecasts, {"lag_order": p}

    # ar_univariate: an independent AR per variable, forecast on its own history.
    max_h = max(int(h) for h in horizons)
    per_horizon = {int(h): np.empty(panel.n_variables) for h in horizons}
    chosen: dict[str, object] = {}
    for j, var in enumerate(names):
        column = block[:, j : j + 1]
        p = _select_ar_lag(
            column, lag_orders=config.benchmark_lag_orders, standardize=config.standardize
        )
        chosen[f"ar_lag_{var}"] = p
        forecasts = _forecast_ridge(
            column, p=p, lam=0.0, alpha=0.0, kappa=1.0,
            standardize=config.standardize, horizons=horizons, variable_names=("series",),
        )
        if forecasts is None:
            raise ValueError(f"ar_univariate: infeasible AR({p}) for {var}.")
        for h in horizons:
            per_horizon[int(h)][j] = float(forecasts[int(h)][0])
    return per_horizon, chosen


# --------------------------------------------------------------------------- #
# Fit-count estimation (for dry-run manifests)
# --------------------------------------------------------------------------- #
def estimate_fit_counts(
    panel_n_observations: int,
    scope: str,
    config: RidgeExperimentConfig,
) -> dict[str, int]:
    """Estimate the number of model fits for one scope run (manifest only)."""

    plan = build_selection_plan(scope, config.target_variables, config.target_horizons)
    n_cells = len(plan.cells)
    n_outer = config.outer_scheme.n_origins
    events = config.selection_schedule.resolve(list(range(n_outer)))
    n_events = len(events)
    n_grid = grid_size(config.grid_spec)
    n_inner = config.inner_scheme.n_origins

    selection_fits = n_events * n_cells * n_grid * n_inner
    outer_fits = n_outer * n_cells
    return {
        "grid_size": n_grid,
        "n_target_cells": n_cells,
        "n_outer_origins": n_outer,
        "n_selection_events": n_events,
        "n_inner_origins": n_inner,
        "selection_fits": selection_fits,
        "outer_forecast_fits": outer_fits,
        "total_fits": selection_fits + outer_fits,
    }


# --------------------------------------------------------------------------- #
# Output writing (canonical files matching GLP / MF-BVAR)
# --------------------------------------------------------------------------- #
def _write_csv(path, columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})


def write_scope_outputs(result: ScopeExperimentResult, output_dir) -> dict[str, object]:
    """Write the canonical output files for one scope run."""

    import json
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(output_dir / "forecast_panel.csv", FORECAST_PANEL_COLUMNS, result.forecast_rows)

    selection_columns = (
        "forecast_origin",
        "group",
        "strategy",
        "cell_id",
        "event_id",
        "param_lam",
        "param_p",
        "param_alpha",
        "param_kappa",
        "selection_loss",
        "n_tied",
    )
    selection_records = []
    for row in result.selection_rows:
        params = row.selection.candidate.to_dict()
        selection_records.append(
            {
                "forecast_origin": row.origin_label,
                "group": row.group,
                "strategy": result.metadata.get("strategy", "ridge_var"),
                "cell_id": row.cell_id,
                "event_id": row.event_id,
                "param_lam": params["lam"],
                "param_p": params["p"],
                "param_alpha": params["alpha"],
                "param_kappa": params["kappa"],
                "selection_loss": row.selection.loss,
                "n_tied": row.selection.n_tied,
            }
        )
    _write_csv(output_dir / "selected_hyperparameters.csv", selection_columns, selection_records)

    _write_csv(
        output_dir / "failed_origins.csv",
        ("forecast_origin", "cell_id", "stage", "error"),
        result.failed_origins,
    )

    metadata = dict(result.metadata)
    metadata["n_forecast_rows"] = len(result.forecast_rows)
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=str)
    return metadata


def write_benchmark_outputs(result: BenchmarkResult, output_dir) -> dict[str, object]:
    """Write the canonical output files for one benchmark strategy."""

    import json
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(output_dir / "forecast_panel.csv", FORECAST_PANEL_COLUMNS, result.forecast_rows)
    _write_csv(
        output_dir / "selected_hyperparameters.csv",
        ("forecast_origin", "strategy", "group", "parameter", "value"),
        result.selection_rows,
    )
    _write_csv(
        output_dir / "failed_origins.csv",
        ("forecast_origin", "stage", "error"),
        result.failed_origins,
    )
    metadata = dict(result.metadata)
    metadata["n_forecast_rows"] = len(result.forecast_rows)
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=str)
    return metadata
