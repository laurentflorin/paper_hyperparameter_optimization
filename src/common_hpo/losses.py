"""Model-independent forecast-loss framework.

This module computes the scalar loss that hyperparameter selection minimizes on
*inner* validation splits, and it is kept strictly separate from the metrics
that are later reported on the *outer* test sample.

Two distinct entry points make that separation explicit:

* :func:`evaluate_selection_loss` computes the standardized, weighted objective
  used to *choose* hyperparameters. Any scaling it applies must be derived only
  from information permitted at selection time (inner training samples or inner
  validation benchmark errors).
* :func:`compute_outer_report_metrics` computes plain, unscaled MSE / RMSE / MAE
  for *reporting* on the outer test sample. It never applies inner-derived
  scaling and never feeds back into selection.

Scientific guarantees enforced here:

* A target scale is never estimated from the complete outer test sample; the
  caller supplies inner training samples for ``target_std``.
* A benchmark scale never uses future outer realizations; ``benchmark_rmse`` is
  computed only from benchmark errors contained in the supplied inner records.
* Standardized loss is invariant to multiplying a variable's target, forecast,
  realization, and target scale by the same positive constant.
* Pooled objectives default to equal contribution *per variable-horizon cell*
  after scaling (``equal_cell``), not equal contribution per raw residual.
* The legacy raw-RMSE computation is reproducible (``scale='none'`` with
  ``equal_observation`` aggregation and uniform weights).

The module has no pandas dependency and does not embed any AR or no-change
benchmark model. Benchmark forecasts are supplied through the
:class:`BenchmarkForecaster` protocol / callback interface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Literal, Mapping, Protocol, Sequence, runtime_checkable


PointMetric = Literal["squared_error", "absolute_error"]
Aggregation = Literal["mse", "rmse", "mae"]
ScaleMethod = Literal["none", "target_std", "benchmark_rmse", "supplied"]
CellAggregation = Literal["equal_cell", "equal_observation"]

DEFAULT_MIN_SCALE = 1e-8
# Documented default: cell/target weights are normalized so that the reported
# loss is a weighted average (weights sum to one) rather than a weighted sum.
DEFAULT_NORMALIZED = True


class LossConfigurationError(ValueError):
    """Raised when a loss or scale configuration is invalid or infeasible."""


class DuplicateErrorRecord(ValueError):
    """Raised when two forecast-error records identify the same target."""


class MissingCellError(ValueError):
    """Raised when required information for a target cell is unavailable."""


@runtime_checkable
class BenchmarkForecaster(Protocol):
    """Callback interface supplying benchmark forecasts.

    Implementations return the benchmark point forecast for one target. This
    keeps any particular benchmark model (AR, no-change, ...) out of this module.
    """

    def __call__(self, *, variable: str, horizon: int, origin: object) -> float:
        ...


def squared_error(forecast: float, realization: float) -> float:
    """Return the squared point error ``(realization - forecast) ** 2``."""

    error = float(realization) - float(forecast)
    return error * error


def absolute_error(forecast: float, realization: float) -> float:
    """Return the absolute point error ``|realization - forecast|``."""

    return abs(float(realization) - float(forecast))


_POINT_METRICS = {
    "squared_error": squared_error,
    "absolute_error": absolute_error,
}


def _as_real(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {result!r}.")
    return result


def _as_positive_real(value: object, *, label: str) -> float:
    result = _as_real(value, label=label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive, got {result}.")
    return result


def _as_horizon(value: object, *, label: str = "horizon") -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer.")
    return int(value)


@dataclass(frozen=True)
class ForecastErrorRecord:
    """One forecast/realization pair for a single validation target.

    ``origin`` identifies the validation split or pseudo-forecast origin (any
    hashable label). ``raw_error`` defaults to ``realization - forecast`` and can
    be supplied explicitly for pre-computed pipelines. Benchmark fields are
    optional and only required by the ``benchmark_rmse`` scale method.
    """

    origin: object
    variable: str
    horizon: int
    forecast: float
    realization: float
    raw_error: float | None = None
    benchmark_forecast: float | None = None
    benchmark_error: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.variable, str) or not self.variable.strip():
            raise ValueError("variable must be a non-empty string.")
        object.__setattr__(self, "variable", self.variable.strip())
        object.__setattr__(self, "horizon", _as_horizon(self.horizon))
        if self.horizon <= 0:
            raise ValueError("horizon must be a positive integer.")
        object.__setattr__(self, "forecast", _as_real(self.forecast, label="forecast"))
        object.__setattr__(self, "realization", _as_real(self.realization, label="realization"))

        if self.raw_error is None:
            object.__setattr__(self, "raw_error", self.realization - self.forecast)
        else:
            object.__setattr__(self, "raw_error", _as_real(self.raw_error, label="raw_error"))

        if self.benchmark_forecast is not None:
            object.__setattr__(
                self,
                "benchmark_forecast",
                _as_real(self.benchmark_forecast, label="benchmark_forecast"),
            )
        if self.benchmark_error is None:
            if self.benchmark_forecast is not None:
                object.__setattr__(
                    self, "benchmark_error", self.realization - self.benchmark_forecast
                )
        else:
            object.__setattr__(
                self, "benchmark_error", _as_real(self.benchmark_error, label="benchmark_error")
            )

    @property
    def cell(self) -> tuple[str, int]:
        """Return the ``(variable, horizon)`` cell key."""

        return (self.variable, self.horizon)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "origin": self.origin,
            "variable": self.variable,
            "horizon": self.horizon,
            "forecast": self.forecast,
            "realization": self.realization,
            "raw_error": self.raw_error,
            "benchmark_forecast": self.benchmark_forecast,
            "benchmark_error": self.benchmark_error,
        }


@dataclass(frozen=True)
class ScaleConfig:
    """How raw errors are standardized before aggregation.

    Methods
    -------
    none:
        Use raw errors unchanged.
    target_std:
        Divide errors by a per-variable scale estimated from the inner training
        sample supplied in ``target_samples`` (sample standard deviation, ddof=1).
    benchmark_rmse:
        Divide errors by the per-cell RMSE of the benchmark errors contained in
        the supplied inner records (permitted inner validation information only).
    supplied:
        Divide errors by explicit positive scales in ``supplied_scales`` keyed by
        ``(variable, horizon)``.

    ``min_scale`` is an explicit safeguard: any scale below it is floored to
    ``min_scale`` and the flooring is recorded in the diagnostics.
    """

    method: ScaleMethod = "none"
    supplied_scales: Mapping[tuple[str, int], float] | None = None
    target_samples: Mapping[str, Sequence[float]] | None = None
    min_scale: float = DEFAULT_MIN_SCALE

    def __post_init__(self) -> None:
        if self.method not in ("none", "target_std", "benchmark_rmse", "supplied"):
            raise LossConfigurationError(f"unknown scale method {self.method!r}.")
        object.__setattr__(self, "min_scale", _as_positive_real(self.min_scale, label="min_scale"))

        if self.method == "supplied":
            if not self.supplied_scales:
                raise LossConfigurationError(
                    "scale method 'supplied' requires supplied_scales."
                )
            normalized: dict[tuple[str, int], float] = {}
            for key, value in self.supplied_scales.items():
                variable, horizon = key
                cell = (str(variable), _as_horizon(horizon))
                normalized[cell] = _as_positive_real(
                    value, label=f"supplied scale for {cell}"
                )
            object.__setattr__(self, "supplied_scales", normalized)
        elif self.supplied_scales is not None:
            raise LossConfigurationError(
                "supplied_scales is only valid for scale method 'supplied'."
            )

        if self.method == "target_std":
            if not self.target_samples:
                raise LossConfigurationError(
                    "scale method 'target_std' requires target_samples."
                )
            normalized_samples: dict[str, tuple[float, ...]] = {}
            for variable, sample in self.target_samples.items():
                values = tuple(_as_real(v, label=f"target sample for {variable!r}") for v in sample)
                if len(values) < 2:
                    raise LossConfigurationError(
                        f"target sample for {variable!r} needs at least two "
                        "observations to estimate a scale."
                    )
                normalized_samples[str(variable)] = values
            object.__setattr__(self, "target_samples", normalized_samples)
        elif self.target_samples is not None:
            raise LossConfigurationError(
                "target_samples is only valid for scale method 'target_std'."
            )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "method": self.method,
            "supplied_scales": (
                {f"{v}|{h}": s for (v, h), s in self.supplied_scales.items()}
                if self.supplied_scales
                else None
            ),
            "target_samples": (
                {v: list(s) for v, s in self.target_samples.items()}
                if self.target_samples
                else None
            ),
            "min_scale": self.min_scale,
        }


@dataclass(frozen=True)
class LossConfig:
    """Configuration for the selection loss.

    ``aggregation`` selects the reported statistic and the implied point metric:
    ``mse`` / ``rmse`` use squared errors, ``mae`` uses absolute errors.

    ``cell_aggregation`` defaults to ``equal_cell``: every variable-horizon cell
    contributes equally after scaling (times any explicit weights), regardless of
    how many observations it holds. ``equal_observation`` weights each cell by its
    observation count and, with uniform weights and ``scale='none'``, reproduces
    the legacy raw calculation.

    Weights (``variable_weights``, ``horizon_weights``, ``origin_weights``)
    default to ``1.0`` for any missing key. ``normalized`` (default ``True``)
    controls whether the final loss divides by the total cell weight (a weighted
    average) or not (a weighted sum).
    """

    aggregation: Aggregation = "rmse"
    scale: ScaleConfig = field(default_factory=ScaleConfig)
    cell_aggregation: CellAggregation = "equal_cell"
    variable_weights: Mapping[str, float] | None = None
    horizon_weights: Mapping[int, float] | None = None
    origin_weights: Mapping[object, float] | None = None
    normalized: bool = DEFAULT_NORMALIZED

    def __post_init__(self) -> None:
        if self.aggregation not in ("mse", "rmse", "mae"):
            raise LossConfigurationError(f"unknown aggregation {self.aggregation!r}.")
        if self.cell_aggregation not in ("equal_cell", "equal_observation"):
            raise LossConfigurationError(
                f"unknown cell_aggregation {self.cell_aggregation!r}."
            )
        if not isinstance(self.scale, ScaleConfig):
            raise LossConfigurationError("scale must be a ScaleConfig instance.")

        object.__setattr__(self, "variable_weights", _normalize_weight_map(
            self.variable_weights, key_cast=str, label="variable_weights"
        ))
        object.__setattr__(self, "horizon_weights", _normalize_weight_map(
            self.horizon_weights, key_cast=lambda k: _as_horizon(k), label="horizon_weights"
        ))
        object.__setattr__(self, "origin_weights", _normalize_weight_map(
            self.origin_weights, key_cast=lambda k: k, label="origin_weights"
        ))

    @property
    def point_metric(self) -> PointMetric:
        """Return the point-error metric implied by the aggregation."""

        return "absolute_error" if self.aggregation == "mae" else "squared_error"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "aggregation": self.aggregation,
            "point_metric": self.point_metric,
            "scale": self.scale.to_dict(),
            "cell_aggregation": self.cell_aggregation,
            "variable_weights": dict(self.variable_weights) if self.variable_weights else {},
            "horizon_weights": (
                {str(k): v for k, v in self.horizon_weights.items()}
                if self.horizon_weights
                else {}
            ),
            "origin_weights": (
                {str(k): v for k, v in self.origin_weights.items()}
                if self.origin_weights
                else {}
            ),
            "normalized": self.normalized,
        }


def _normalize_weight_map(mapping, *, key_cast, label):
    if mapping is None:
        return {}
    normalized = {}
    for key, value in mapping.items():
        weight = _as_real(value, label=f"{label}[{key!r}]")
        if weight < 0.0:
            raise LossConfigurationError(f"{label}[{key!r}] must be non-negative.")
        normalized[key_cast(key)] = weight
    return normalized


@dataclass(frozen=True)
class CellDiagnostic:
    """Per-cell contribution to the selection loss (for debugging)."""

    variable: str
    horizon: int
    n_observations: int
    scale: float
    scale_floored: bool
    cell_value: float
    cell_weight: float
    contribution: float
    contribution_fraction: float

    def to_dict(self) -> dict[str, object]:
        return {
            "variable": self.variable,
            "horizon": self.horizon,
            "n_observations": self.n_observations,
            "scale": self.scale,
            "scale_floored": self.scale_floored,
            "cell_value": self.cell_value,
            "cell_weight": self.cell_weight,
            "contribution": self.contribution,
            "contribution_fraction": self.contribution_fraction,
        }


@dataclass(frozen=True)
class LossResult:
    """The scalar selection loss plus deterministic per-cell diagnostics."""

    value: float
    aggregation: Aggregation
    point_metric: PointMetric
    scale_method: ScaleMethod
    cell_aggregation: CellAggregation
    normalized: bool
    n_cells: int
    n_observations: int
    cells: tuple[CellDiagnostic, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "value": self.value,
            "aggregation": self.aggregation,
            "point_metric": self.point_metric,
            "scale_method": self.scale_method,
            "cell_aggregation": self.cell_aggregation,
            "normalized": self.normalized,
            "n_cells": self.n_cells,
            "n_observations": self.n_observations,
            "cells": [cell.to_dict() for cell in self.cells],
        }


def attach_benchmark_errors(
    records: Sequence[ForecastErrorRecord],
    benchmark: BenchmarkForecaster,
) -> tuple[ForecastErrorRecord, ...]:
    """Return copies of ``records`` with benchmark forecasts/errors attached.

    The benchmark model itself lives entirely in the supplied ``benchmark``
    callback, keeping this module model-independent.
    """

    enriched: list[ForecastErrorRecord] = []
    for record in records:
        forecast = float(
            benchmark(variable=record.variable, horizon=record.horizon, origin=record.origin)
        )
        enriched.append(
            ForecastErrorRecord(
                origin=record.origin,
                variable=record.variable,
                horizon=record.horizon,
                forecast=record.forecast,
                realization=record.realization,
                raw_error=record.raw_error,
                benchmark_forecast=forecast,
            )
        )
    return tuple(enriched)


def _group_cells(
    records: Sequence[ForecastErrorRecord],
) -> dict[tuple[str, int], list[ForecastErrorRecord]]:
    cells: dict[tuple[str, int], list[ForecastErrorRecord]] = {}
    seen: set[tuple[object, str, int]] = set()
    for record in records:
        identity = (record.origin, record.variable, record.horizon)
        if identity in seen:
            raise DuplicateErrorRecord(
                "duplicate forecast-error record for origin "
                f"{record.origin!r}, variable {record.variable!r}, horizon "
                f"{record.horizon}."
            )
        seen.add(identity)
        cells.setdefault(record.cell, []).append(record)
    return cells


def _sample_std(values: Sequence[float]) -> float:
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(variance)


def _cell_scale(
    cell: tuple[str, int],
    cell_records: Sequence[ForecastErrorRecord],
    scale_config: ScaleConfig,
) -> tuple[float, bool]:
    variable, horizon = cell
    method = scale_config.method

    if method == "none":
        raw_scale = 1.0
    elif method == "supplied":
        if cell not in scale_config.supplied_scales:
            raise MissingCellError(
                f"no supplied scale for cell (variable={variable!r}, horizon={horizon})."
            )
        raw_scale = scale_config.supplied_scales[cell]
    elif method == "target_std":
        if variable not in scale_config.target_samples:
            raise MissingCellError(
                f"no target sample supplied for variable {variable!r}."
            )
        raw_scale = _sample_std(scale_config.target_samples[variable])
    else:  # benchmark_rmse
        benchmark_errors = [r.benchmark_error for r in cell_records if r.benchmark_error is not None]
        if len(benchmark_errors) != len(cell_records):
            raise MissingCellError(
                f"benchmark_rmse scaling requires a benchmark error on every record "
                f"in cell (variable={variable!r}, horizon={horizon})."
            )
        raw_scale = math.sqrt(sum(e * e for e in benchmark_errors) / len(benchmark_errors))

    floored = raw_scale < scale_config.min_scale
    scale = scale_config.min_scale if floored else raw_scale
    return scale, floored


def _cell_value(
    cell_records: Sequence[ForecastErrorRecord],
    scale: float,
    point_metric: PointMetric,
    origin_weights: Mapping[object, float],
) -> float:
    metric_fn = _POINT_METRICS[point_metric]
    weight_sum = 0.0
    weighted = 0.0
    for record in cell_records:
        origin_weight = origin_weights.get(record.origin, 1.0)
        standardized_error = record.raw_error / scale
        # Apply the point metric to the standardized error by comparing a
        # standardized "forecast" of zero against the standardized error.
        point = metric_fn(0.0, standardized_error)
        weighted += origin_weight * point
        weight_sum += origin_weight
    if weight_sum <= 0.0:
        raise LossConfigurationError(
            "origin weights within a cell sum to zero; cannot average."
        )
    return weighted / weight_sum


def evaluate_selection_loss(
    records: Sequence[ForecastErrorRecord],
    config: LossConfig | None = None,
) -> LossResult:
    """Compute the standardized, weighted loss minimized during selection.

    The returned :class:`LossResult` carries deterministic per-cell diagnostics
    sorted by ``(variable, horizon)``. Scaling uses only selection-time
    information as configured in :class:`ScaleConfig`.
    """

    if config is None:
        config = LossConfig()
    if not records:
        raise LossConfigurationError("at least one forecast-error record is required.")

    cells = _group_cells(records)
    point_metric = config.point_metric

    ordered_keys = sorted(cells.keys(), key=lambda key: (key[0], key[1]))

    cell_values: dict[tuple[str, int], float] = {}
    cell_weights: dict[tuple[str, int], float] = {}
    cell_scales: dict[tuple[str, int], tuple[float, bool]] = {}

    for cell in ordered_keys:
        cell_records = cells[cell]
        variable, horizon = cell
        scale, floored = _cell_scale(cell, cell_records, config.scale)
        cell_scales[cell] = (scale, floored)
        cell_values[cell] = _cell_value(
            cell_records, scale, point_metric, config.origin_weights
        )
        base_weight = (
            config.variable_weights.get(variable, 1.0)
            * config.horizon_weights.get(horizon, 1.0)
        )
        if config.cell_aggregation == "equal_observation":
            base_weight *= len(cell_records)
        cell_weights[cell] = base_weight

    total_weight = sum(cell_weights[cell] for cell in ordered_keys)
    if total_weight <= 0.0:
        raise LossConfigurationError("total cell weight is zero; cannot aggregate.")

    numerator = sum(cell_weights[cell] * cell_values[cell] for cell in ordered_keys)
    aggregate = numerator / total_weight if config.normalized else numerator

    if config.aggregation == "rmse":
        value = math.sqrt(aggregate)
    else:
        value = aggregate

    diagnostics: list[CellDiagnostic] = []
    for cell in ordered_keys:
        variable, horizon = cell
        scale, floored = cell_scales[cell]
        contribution = cell_weights[cell] * cell_values[cell]
        diagnostics.append(
            CellDiagnostic(
                variable=variable,
                horizon=horizon,
                n_observations=len(cells[cell]),
                scale=scale,
                scale_floored=floored,
                cell_value=cell_values[cell],
                cell_weight=cell_weights[cell],
                contribution=contribution,
                contribution_fraction=contribution / numerator if numerator != 0.0 else 0.0,
            )
        )

    return LossResult(
        value=value,
        aggregation=config.aggregation,
        point_metric=point_metric,
        scale_method=config.scale.method,
        cell_aggregation=config.cell_aggregation,
        normalized=config.normalized,
        n_cells=len(ordered_keys),
        n_observations=len(records),
        cells=tuple(diagnostics),
    )


def compute_outer_report_metrics(
    records: Sequence[ForecastErrorRecord],
    aggregation: Aggregation = "rmse",
) -> dict[str, object]:
    """Compute plain, unscaled outer-sample metrics for reporting only.

    This is intentionally distinct from :func:`evaluate_selection_loss`: it
    applies no inner-derived scaling and no cell reweighting. It reports a pooled
    statistic together with a deterministic per-cell breakdown, and must never be
    used to select hyperparameters.
    """

    if aggregation not in ("mse", "rmse", "mae"):
        raise LossConfigurationError(f"unknown aggregation {aggregation!r}.")
    if not records:
        raise LossConfigurationError("at least one forecast-error record is required.")

    point_metric: PointMetric = "absolute_error" if aggregation == "mae" else "squared_error"
    metric_fn = _POINT_METRICS[point_metric]

    cells = _group_cells(records)
    ordered_keys = sorted(cells.keys(), key=lambda key: (key[0], key[1]))

    per_cell: list[dict[str, object]] = []
    pooled_points: list[float] = []
    for cell in ordered_keys:
        variable, horizon = cell
        points = [metric_fn(0.0, record.raw_error) for record in cells[cell]]
        pooled_points.extend(points)
        cell_mean = sum(points) / len(points)
        per_cell.append(
            {
                "variable": variable,
                "horizon": horizon,
                "n_observations": len(points),
                "value": math.sqrt(cell_mean) if aggregation == "rmse" else cell_mean,
            }
        )

    pooled_mean = sum(pooled_points) / len(pooled_points)
    pooled_value = math.sqrt(pooled_mean) if aggregation == "rmse" else pooled_mean

    return {
        "aggregation": aggregation,
        "point_metric": point_metric,
        "purpose": "outer_report_only",
        "pooled_value": pooled_value,
        "n_observations": len(records),
        "cells": per_cell,
    }


__all__ = [
    "Aggregation",
    "BenchmarkForecaster",
    "CellAggregation",
    "CellDiagnostic",
    "DEFAULT_MIN_SCALE",
    "DEFAULT_NORMALIZED",
    "DuplicateErrorRecord",
    "ForecastErrorRecord",
    "LossConfig",
    "LossConfigurationError",
    "LossResult",
    "MissingCellError",
    "PointMetric",
    "ScaleConfig",
    "ScaleMethod",
    "attach_benchmark_errors",
    "absolute_error",
    "compute_outer_report_metrics",
    "evaluate_selection_loss",
    "squared_error",
]
