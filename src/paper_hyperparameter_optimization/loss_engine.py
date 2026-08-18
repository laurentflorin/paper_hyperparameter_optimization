"""Adapter selector that drives the *real* mixed-frequency objective path.

This bridges the model-independent
:func:`~paper_hyperparameter_optimization.selection_experiment.run_mfvar_selection_experiment`
orchestrator to the existing Schorfheide-Song RMSE objective in
:mod:`paper_hyperparameter_optimization.forecasting`. It reuses -- rather than
reimplements -- ``resolve_forecast_objective_variables``,
``build_rmse_validation_folds`` and ``_rmse_candidate_score`` so a GDP-only
objective is scored through exactly the same fold construction and posterior
draw aggregation as a full-block objective. The forecast state is always fit on
the full ``forecast_variables`` block, while only the cell's
``objective_variables`` enter the loss.

The MBFVAR model class is injected (``model_class``) so unit tests can exercise
the real objective code with a lightweight fake, and the candidate search is
supplied as an explicit list of natural coordinate dictionaries so no dependency
on Mango is required to run the objective.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np

from . import forecasting
from .selection_experiment import MFVARCellSelection, MFVARCellSelectionRequest


# The audited MBFVAR revision constructs fresh, unseeded NumPy generators inside
# its fit/forecast routines. The repository can control fold sampling and Mango
# candidate order, but not that internal draw stream. This is surfaced (never
# concealed) as a reproducibility limitation on every real selection.
UPSTREAM_SEED_UNCONTROLLED_REASON = (
    "The pinned MBFVAR revision creates fresh unseeded NumPy generators inside "
    "its fit/forecast routines, so posterior-draw randomness is not fully "
    "reproducible from the supplied seed until MBFVAR exposes an injectable "
    "generator."
)


def _candidate_param_grid() -> tuple[dict[str, float], ...]:
    """A small deterministic default candidate grid over the natural bounds."""

    grid: list[dict[str, float]] = []
    for lambda1 in (0.05, 0.2):
        grid.append(
            {
                "lambda1_1": lambda1,
                "lambda2_1": 1.0,
                "lambda4_1": 1.0,
                "lambda5_1": 1.0,
            }
        )
    return tuple(grid)


def build_mfvar_objective_selector(
    *,
    data_in: Any,
    model_class: Any,
    candidate_params: Sequence[Mapping[str, float]] | None = None,
    nsim: int,
    nburn_perc: float,
    nlags: list[int],
    thining: int,
    temp_agg: str,
    horizon_quarters: int,
    eval_horizon_quarters: int | None,
    n_eval: int,
    min_train_quarters: int | None = None,
    selection: str = "rolling",
    fold_seed: int | None = None,
    objective_seed: int | None = None,
    strategy: str = "mango_rmse",
    report_seed_uncontrolled: bool = True,
):
    """Return a ``Selector`` that scores candidates via the real RMSE objective.

    The returned callable resolves the full forecast block for the cell's
    objective subset, builds leakage-safe RMSE validation folds once per cell,
    scores each supplied candidate through ``_rmse_candidate_score`` (fitting the
    full forecast block, scoring only the objective subset), and returns the
    best-scoring candidate as an :class:`MFVARCellSelection`.
    """

    grid = tuple(dict(params) for params in (candidate_params or _candidate_param_grid()))
    if not grid:
        raise ValueError("candidate_params must be non-empty.")

    def selector(request: MFVARCellSelectionRequest) -> MFVARCellSelection:
        forecast_variables, objective_variables = forecasting.resolve_forecast_objective_variables(
            strategy,
            forecast_variables=list(request.forecast_variables),
            objective_variables=list(request.objective_variables),
        )

        folds, fold_diagnostics = forecasting.build_rmse_validation_folds(
            data_in,
            horizon_quarters=horizon_quarters,
            h_eval=eval_horizon_quarters,
            n_eval=n_eval,
            forecast_variables=forecast_variables,
            objective_variables=objective_variables,
            nlags=nlags,
            selection=selection,
            min_train_quarters=min_train_quarters,
            fold_seed=fold_seed if fold_seed is not None else request.seed,
        )

        start = time.perf_counter()
        best_params: dict[str, float] | None = None
        best_score = np.inf
        failure_counts = {"numerical": 0, "nonfinite": 0}
        for params in grid:
            try:
                score = forecasting._rmse_candidate_score(
                    params,
                    model_class=model_class,
                    folds=folds,
                    forecast_variables=forecast_variables,
                    objective_variables=objective_variables,
                    horizon_quarters=horizon_quarters,
                    h_eval=eval_horizon_quarters,
                    nsim=nsim,
                    nburn_perc=nburn_perc,
                    nlags=nlags,
                    thining=thining,
                    temp_agg=temp_agg,
                    objective_seed=objective_seed if objective_seed is not None else request.seed,
                )
            except forecasting.EXPECTED_NUMERICAL_FAILURES:
                failure_counts["numerical"] += 1
                continue
            if not np.isfinite(score):
                failure_counts["nonfinite"] += 1
                continue
            if score < best_score:
                best_score = score
                best_params = dict(params)

        runtime = time.perf_counter() - start
        if best_params is None:
            raise RuntimeError(
                "No candidate produced a finite objective for cell "
                f"{request.cell_id!r}; an all-penalty selection is never accepted."
            )

        natural_vector = tuple(
            float(v) for v in forecasting._candidate_hyperparameters(best_params)[0]
        )

        return MFVARCellSelection(
            hyperparameter_vector=natural_vector,
            named_parameters={
                "lambda1_1": natural_vector[0],
                "lambda2_1": natural_vector[1],
                "lambda3_1": natural_vector[2],
                "lambda4_1": natural_vector[3],
                "lambda5_1": natural_vector[4],
            },
            selection_loss=float(best_score),
            forecast_variables=tuple(forecast_variables),
            objective_variables=tuple(objective_variables),
            optimizer_seed=request.seed,
            optimizer_budget={"candidate_grid_size": len(grid)},
            n_inner_origins=fold_diagnostics["effective_n_eval"],
            validation_stride=None,
            objective_draw_count=int(nsim),
            failure_counts=failure_counts,
            runtime_seconds=runtime,
            seed_uncontrolled=bool(report_seed_uncontrolled),
            seed_uncontrolled_reason=(
                UPSTREAM_SEED_UNCONTROLLED_REASON if report_seed_uncontrolled else None
            ),
        )

    return selector


__all__ = [
    "UPSTREAM_SEED_UNCONTROLLED_REASON",
    "build_mfvar_objective_selector",
]
