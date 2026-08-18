"""GLP forecast-loss objective built on the shared ``common_hpo`` abstractions.

This module refactors the GLP hyperparameter-selection loss so that a single
``TargetCell`` -- one or several variables at one or several horizons -- is
scored in one candidate evaluation and aggregated through the shared
:mod:`common_hpo.losses` framework. It reuses the *existing* GLP
posterior-predictive-mean primitives (``glp_mode_estimate``, ``glp_draw`` and
``point_forecast``) rather than reimplementing them, so the predictive-mean
calculation is not duplicated in two divergent implementations.

Preserved behavior (delegated to :mod:`.glp_model`):

* beta is drawn from its posterior and deterministic conditional forecasts are
  formed for each draw, then averaged into a point forecast;
* no mean-zero future shocks are added to the point-forecast objective;
* draws use deterministic common random numbers keyed on the validation split
  and draw index -- never on the candidate hyperparameter values -- and the
  caller's global NumPy / Python random state is restored afterward.

The design decomposition is:

* :func:`prepare_glp_validation_contexts` -- wrap pre-built inner folds
  (context + holdout actuals) into split-aware evaluation contexts;
* :func:`build_glp_error_records` -- turn one context's forecast into
  :class:`~common_hpo.losses.ForecastErrorRecord` objects with strict target
  alignment;
* :func:`evaluate_glp_candidate` -- score one candidate across all contexts and
  return structured diagnostics;
* :func:`make_glp_loss_objective` -- expose an optimizer-friendly callable that
  returns a finite penalty on failure while retaining the failure reason.

The module has no hard dependency on Mango and does not embed any benchmark
model. Forecast production is injected through ``forecast_fn`` so unit tests can
supply synthetic contexts without running a full recursive experiment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, MutableSequence, Sequence

import numpy as np

from experiment_provenance import deterministic_rng_context, stable_child_seed

from common_hpo.losses import (
    ForecastErrorRecord,
    LossConfig,
    LossConfigurationError,
    LossResult,
    MissingCellError,
    evaluate_selection_loss,
)
from common_hpo.splits import ValidationSplit

from . import glp_model
from .glp_model import RMSE_PENALTY, InvalidHyperparameterError


# Numerical failures that a robust optimizer objective must convert into a
# finite penalty rather than propagate.
_NUMERICAL_FAILURES = (
    InvalidHyperparameterError,
    np.linalg.LinAlgError,
    FloatingPointError,
    OverflowError,
    ZeroDivisionError,
)


ForecastFn = Callable[..., np.ndarray]


class _NonFiniteForecast(RuntimeError):
    """Internal signal that a candidate produced a non-finite forecast."""


@dataclass(frozen=True)
class GLPCellSpec:
    """One target cell to score in a single candidate evaluation.

    Parameters
    ----------
    variables:
        Target variable codes scored by this cell (one or several).
    horizons:
        Canonical horizons scored by this cell (one or several), expressed in
        the model's own horizon units (quarters for GLP).
    n_obj_draws:
        Number of posterior beta draws averaged into the point forecast. ``1``
        (or less) uses the deterministic posterior mode.
    seed_base:
        Local seed base for common random numbers. ``None`` disables seeding.
    loss_config:
        The shared :class:`~common_hpo.losses.LossConfig`. Defaults to the
        equal-cell RMSE objective (equal contribution per variable-horizon cell
        after scaling).
    """

    variables: tuple[str, ...]
    horizons: tuple[int, ...]
    n_obj_draws: int = 1
    seed_base: int | None = 0
    loss_config: LossConfig = field(default_factory=LossConfig)

    def __post_init__(self) -> None:
        if not self.variables:
            raise ValueError("GLPCellSpec requires at least one variable.")
        object.__setattr__(self, "variables", tuple(str(v) for v in self.variables))
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("GLPCellSpec variables must be unique.")

        horizons = tuple(int(h) for h in self.horizons)
        if not horizons:
            raise ValueError("GLPCellSpec requires at least one horizon.")
        if any(h <= 0 for h in horizons):
            raise ValueError("GLPCellSpec horizons must be positive integers.")
        if len(set(horizons)) != len(horizons):
            raise ValueError("GLPCellSpec horizons must be unique.")
        object.__setattr__(self, "horizons", horizons)

        if not isinstance(self.loss_config, LossConfig):
            raise TypeError("loss_config must be a LossConfig instance.")

    @property
    def ordered_horizons(self) -> list[int]:
        """Return horizons in ascending order (deterministic forecast layout)."""

        return sorted(self.horizons)

    @property
    def uses_draws(self) -> bool:
        return int(self.n_obj_draws) > 1


@dataclass(frozen=True)
class GLPValidationContext:
    """One inner validation fold ready for candidate scoring.

    ``context`` is the hyperparameter-independent training context (a
    ``GLPContext`` in production, or any object exposing ``.y`` in tests).
    ``actual`` holds the future holdout levels with row ``h - 1`` corresponding
    to canonical horizon ``h`` (the single documented alignment convention).
    """

    split_id: str
    origin: object
    context: Any
    codes: tuple[str, ...]
    actual: np.ndarray
    horizon_rows: Mapping[int, int]
    split: ValidationSplit | None = None

    def variable_index(self, variable: str) -> int:
        codes = list(self.codes)
        if variable not in codes:
            raise MissingCellError(
                f"variable {variable!r} is not in the model block {codes}."
            )
        return codes.index(variable)

    def actual_for(self, variable: str, horizon: int) -> float:
        if horizon not in self.horizon_rows:
            raise MissingCellError(
                f"horizon {horizon} is not aligned in split {self.split_id!r}."
            )
        row = self.horizon_rows[horizon]
        return float(self.actual[row, self.variable_index(variable)])


@dataclass
class GLPCandidateEvaluation:
    """Structured diagnostics for one scored candidate."""

    failed: bool
    total_loss: float
    loss_by_cell: dict[tuple[str, int], float]
    n_valid_records: int
    numerical_failures: int
    nonfinite_forecasts: int
    scale_problems: int
    failure_reason: str | None = None
    records: tuple[ForecastErrorRecord, ...] = ()
    loss_result: LossResult | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "failed": self.failed,
            "total_loss": self.total_loss,
            "loss_by_cell": {f"{v}|{h}": val for (v, h), val in self.loss_by_cell.items()},
            "n_valid_records": self.n_valid_records,
            "numerical_failures": self.numerical_failures,
            "nonfinite_forecasts": self.nonfinite_forecasts,
            "scale_problems": self.scale_problems,
            "failure_reason": self.failure_reason,
            "records": [record.to_dict() for record in self.records],
            "loss_result": self.loss_result.to_dict() if self.loss_result is not None else None,
        }


def default_glp_forecast_fn(
    context: Any,
    natural_vec: np.ndarray,
    horizons: Sequence[int],
    *,
    draw_index: int,
    use_draws: bool,
) -> np.ndarray:
    """Produce the posterior (mode or draw) point forecast for all horizons.

    This delegates to the existing GLP primitives via module attribute access so
    the predictive-mean logic is never duplicated and remains monkeypatchable in
    tests. A single beta (mode or draw) serves every requested horizon.
    """

    if use_draws:
        beta, _ = glp_model.glp_draw(context, natural_vec)
    else:
        beta, _ = glp_model.glp_mode_estimate(context, natural_vec)
    forecast = glp_model.point_forecast(context.y, beta, list(horizons))
    return np.asarray(forecast, dtype=float)


def prepare_glp_validation_contexts(
    origins: Sequence[tuple[Any, np.ndarray]],
    codes: Sequence[str],
    *,
    max_horizon: int,
    splits: Sequence[ValidationSplit] | None = None,
    origin_labels: Sequence[object] | None = None,
) -> tuple[GLPValidationContext, ...]:
    """Wrap pre-built ``(context, actual)`` inner folds into split-aware contexts.

    ``origins`` is exactly the structure produced by the existing GLP origin
    builders, so this reuses (rather than replaces) the current fold
    construction. Row ``h - 1`` of each holdout maps to canonical horizon ``h``.
    """

    codes = tuple(str(code) for code in codes)
    if max_horizon <= 0:
        raise ValueError("max_horizon must be a positive integer.")
    if splits is not None and len(splits) != len(origins):
        raise ValueError("splits must align one-to-one with origins.")
    if origin_labels is not None and len(origin_labels) != len(origins):
        raise ValueError("origin_labels must align one-to-one with origins.")

    horizon_rows = {h: h - 1 for h in range(1, int(max_horizon) + 1)}
    contexts: list[GLPValidationContext] = []
    for index, (context, actual) in enumerate(origins):
        actual_array = np.asarray(actual, dtype=float)
        if actual_array.ndim != 2:
            raise ValueError(
                f"holdout actual for origin {index} must be 2-D, got shape "
                f"{actual_array.shape}."
            )
        if actual_array.shape[0] < int(max_horizon):
            raise ValueError(
                f"holdout actual for origin {index} has {actual_array.shape[0]} rows "
                f"but {max_horizon} horizons are requested."
            )
        if actual_array.shape[1] != len(codes):
            raise ValueError(
                f"holdout actual for origin {index} has {actual_array.shape[1]} "
                f"columns but {len(codes)} variable codes were supplied."
            )
        split = splits[index] if splits is not None else None
        if split is not None:
            split_id = split.split_id
            origin = split.origin
        else:
            split_id = f"glp-split-{index:03d}"
            origin = origin_labels[index] if origin_labels is not None else index
        contexts.append(
            GLPValidationContext(
                split_id=split_id,
                origin=origin,
                context=context,
                codes=codes,
                actual=actual_array,
                horizon_rows=horizon_rows,
                split=split,
            )
        )
    return tuple(contexts)


def build_glp_error_records(
    context: GLPValidationContext,
    forecast: np.ndarray,
    spec: GLPCellSpec,
) -> list[ForecastErrorRecord]:
    """Build leakage-safe error records for one context's forecast.

    ``forecast`` has one row per ascending horizon in ``spec.ordered_horizons``.
    Each requested variable-horizon pair yields exactly one record with the
    holdout realization aligned by the single row = horizon - 1 convention.
    """

    forecast = np.asarray(forecast, dtype=float)
    horizons = spec.ordered_horizons
    if forecast.shape[0] != len(horizons):
        raise ValueError(
            f"forecast has {forecast.shape[0]} horizon rows but {len(horizons)} "
            "were requested."
        )

    records: list[ForecastErrorRecord] = []
    for row_index, horizon in enumerate(horizons):
        target_row = context.horizon_rows[horizon]
        if not 0 <= target_row < context.actual.shape[0]:
            raise MissingCellError(
                f"target row for horizon {horizon} is out of range in split "
                f"{context.split_id!r}."
            )
        for variable in spec.variables:
            vidx = context.variable_index(variable)
            forecast_value = float(forecast[row_index, vidx])
            realization = float(context.actual[target_row, vidx])
            records.append(
                ForecastErrorRecord(
                    origin=context.split_id,
                    variable=variable,
                    horizon=horizon,
                    forecast=forecast_value,
                    realization=realization,
                )
            )
    return records


def _predict_mean(
    context: GLPValidationContext,
    natural_vec: np.ndarray,
    spec: GLPCellSpec,
    split_index: int,
    forecast_fn: ForecastFn,
) -> np.ndarray:
    horizons = spec.ordered_horizons
    if not spec.uses_draws:
        forecast = forecast_fn(
            context.context, natural_vec, horizons, draw_index=0, use_draws=False
        )
        return np.asarray(forecast, dtype=float)

    total: np.ndarray | None = None
    for draw_index in range(int(spec.n_obj_draws)):
        # Common random numbers: the seed depends only on the split and draw
        # index, never on the candidate hyperparameters, and the global RNG is
        # restored on exit.
        child_seed = stable_child_seed(
            spec.seed_base, "glp-loss-objective", split_index, draw_index
        )
        with deterministic_rng_context(child_seed):
            forecast = np.asarray(
                forecast_fn(
                    context.context,
                    natural_vec,
                    horizons,
                    draw_index=draw_index,
                    use_draws=True,
                ),
                dtype=float,
            )
        total = forecast if total is None else total + forecast
    assert total is not None
    return total / float(spec.n_obj_draws)


def evaluate_glp_candidate(
    params: Mapping[str, float],
    contexts: Sequence[GLPValidationContext],
    spec: GLPCellSpec,
    *,
    forecast_fn: ForecastFn = default_glp_forecast_fn,
    to_natural: Callable[[Mapping[str, float], Any], np.ndarray] = glp_model._params_to_natural,
    penalty: float = RMSE_PENALTY,
) -> GLPCandidateEvaluation:
    """Score one candidate across every context and return structured diagnostics.

    The evaluation never raises for numerical failures: instead it records the
    failure reason and returns a failed evaluation carrying the finite penalty.
    """

    if not contexts:
        raise ValueError("at least one validation context is required.")

    numerical_failures = 0
    nonfinite_forecasts = 0
    records: list[ForecastErrorRecord] = []

    try:
        for split_index, context in enumerate(contexts):
            natural_vec = to_natural(params, context.context)
            forecast = _predict_mean(context, natural_vec, spec, split_index, forecast_fn)
            if not np.all(np.isfinite(forecast)):
                nonfinite_forecasts += 1
                raise _NonFiniteForecast(
                    f"non-finite forecast in split {context.split_id!r}."
                )
            records.extend(build_glp_error_records(context, forecast, spec))
    except _NUMERICAL_FAILURES as exc:
        numerical_failures += 1
        return GLPCandidateEvaluation(
            failed=True,
            total_loss=float(penalty),
            loss_by_cell={},
            n_valid_records=0,
            numerical_failures=numerical_failures,
            nonfinite_forecasts=nonfinite_forecasts,
            scale_problems=0,
            failure_reason=f"{type(exc).__name__}: {exc}",
        )
    except MissingCellError as exc:
        return GLPCandidateEvaluation(
            failed=True,
            total_loss=float(penalty),
            loss_by_cell={},
            n_valid_records=0,
            numerical_failures=numerical_failures,
            nonfinite_forecasts=nonfinite_forecasts,
            scale_problems=1,
            failure_reason=f"{type(exc).__name__}: {exc}",
        )
    except _NonFiniteForecast as exc:
        return GLPCandidateEvaluation(
            failed=True,
            total_loss=float(penalty),
            loss_by_cell={},
            n_valid_records=0,
            numerical_failures=numerical_failures,
            nonfinite_forecasts=nonfinite_forecasts,
            scale_problems=0,
            failure_reason=str(exc),
        )

    try:
        result = evaluate_selection_loss(records, spec.loss_config)
    except (MissingCellError, LossConfigurationError) as exc:
        return GLPCandidateEvaluation(
            failed=True,
            total_loss=float(penalty),
            loss_by_cell={},
            n_valid_records=len(records),
            numerical_failures=numerical_failures,
            nonfinite_forecasts=nonfinite_forecasts,
            scale_problems=1,
            failure_reason=f"{type(exc).__name__}: {exc}",
        )

    if not math.isfinite(result.value):
        return GLPCandidateEvaluation(
            failed=True,
            total_loss=float(penalty),
            loss_by_cell={
                (cell.variable, cell.horizon): cell.cell_value for cell in result.cells
            },
            n_valid_records=len(records),
            numerical_failures=numerical_failures,
            nonfinite_forecasts=nonfinite_forecasts,
            scale_problems=sum(1 for cell in result.cells if cell.scale_floored),
            failure_reason="non-finite aggregated loss.",
            records=tuple(records),
            loss_result=result,
        )

    return GLPCandidateEvaluation(
        failed=False,
        total_loss=result.value,
        loss_by_cell={
            (cell.variable, cell.horizon): cell.cell_value for cell in result.cells
        },
        n_valid_records=len(records),
        numerical_failures=0,
        nonfinite_forecasts=0,
        scale_problems=sum(1 for cell in result.cells if cell.scale_floored),
        failure_reason=None,
        records=tuple(records),
        loss_result=result,
    )


def make_glp_loss_objective(
    contexts: Sequence[GLPValidationContext],
    spec: GLPCellSpec,
    *,
    forecast_fn: ForecastFn = default_glp_forecast_fn,
    to_natural: Callable[[Mapping[str, float], Any], np.ndarray] = glp_model._params_to_natural,
    penalty: float = RMSE_PENALTY,
    diagnostics_collector: MutableSequence[dict[str, object]] | None = None,
) -> Callable[..., float]:
    """Return an optimizer-friendly objective for one target cell.

    The returned callable accepts optimizer keyword coordinates and returns the
    scalar selection loss, or a finite ``penalty`` on failure. Every failure
    reason is retained on the callable's ``diagnostics`` attribute and, when
    supplied, appended to ``diagnostics_collector`` so an optimizer that only
    sees a finite penalty does not lose the underlying cause.
    """

    diagnostics = {
        "valid": 0,
        "penalized": 0,
        "numerical_failures": 0,
        "nonfinite_forecasts": 0,
        "scale_problems": 0,
        "last_failure_reason": None,
    }

    def objective(**params: float) -> float:
        evaluation = evaluate_glp_candidate(
            params,
            contexts,
            spec,
            forecast_fn=forecast_fn,
            to_natural=to_natural,
            penalty=penalty,
        )
        diagnostics["numerical_failures"] += evaluation.numerical_failures
        diagnostics["nonfinite_forecasts"] += evaluation.nonfinite_forecasts
        diagnostics["scale_problems"] += evaluation.scale_problems
        if evaluation.failed:
            diagnostics["penalized"] += 1
            diagnostics["last_failure_reason"] = evaluation.failure_reason
            if diagnostics_collector is not None:
                diagnostics_collector.append(
                    {
                        "params": {str(k): float(v) for k, v in params.items()},
                        "reason": evaluation.failure_reason,
                        "numerical_failures": evaluation.numerical_failures,
                        "nonfinite_forecasts": evaluation.nonfinite_forecasts,
                        "scale_problems": evaluation.scale_problems,
                    }
                )
            return float(penalty)
        diagnostics["valid"] += 1
        return float(evaluation.total_loss)

    objective.diagnostics = diagnostics  # type: ignore[attr-defined]
    return objective


__all__ = [
    "ForecastFn",
    "GLPCandidateEvaluation",
    "GLPCellSpec",
    "GLPValidationContext",
    "build_glp_error_records",
    "default_glp_forecast_fn",
    "evaluate_glp_candidate",
    "make_glp_loss_objective",
    "prepare_glp_validation_contexts",
]
