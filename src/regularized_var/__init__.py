"""Regularized (lag-weighted ridge) VAR estimator and iterated forecasting.

This package implements a self-contained VAR estimator with a lag-decay ridge
penalty and deterministic multi-step forecasting. It imports **only** NumPy and
has no Bayesian or other optional heavy dependencies, so it can be used in pure,
fast unit tests.

Public API
----------
* :func:`build_lag_design` -- construct the documented lag design.
* :class:`PenaltyConfig` -- lag-weighted ridge penalty configuration.
* :class:`RidgeVARResult` -- typed fit result.
* :func:`fit_ridge_var` -- fit the estimator.
* :func:`iterated_forecast` -- deterministic multi-step point forecasts.
* :class:`UnstableVARError` -- raised by the optional reject-unstable policy.
"""

from __future__ import annotations

from .design import build_lag_design, predictor_terms, n_predictors, validate_observations
from .estimators import (
    PenaltyConfig,
    RidgeVARResult,
    UnstableVARError,
    build_penalty_weights,
    fit_ridge_var,
)
from .forecasting import iterated_forecast
from .data import PanelData, Standardizer, load_panel_csv
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
from .experiment import (
    BENCHMARK_STRATEGIES,
    FORECAST_PANEL_COLUMNS,
    BenchmarkResult,
    CellSelection,
    RidgeExperimentConfig,
    ScopeExperimentResult,
    default_experiment_config,
    estimate_fit_counts,
    run_benchmark,
    run_scope_experiment,
    select_for_cell,
    write_benchmark_outputs,
    write_scope_outputs,
)

__all__ = [
    "build_lag_design",
    "predictor_terms",
    "n_predictors",
    "validate_observations",
    "PenaltyConfig",
    "RidgeVARResult",
    "UnstableVARError",
    "build_penalty_weights",
    "fit_ridge_var",
    "iterated_forecast",
    # data
    "PanelData",
    "Standardizer",
    "load_panel_csv",
    # tuning
    "DEFAULT_TIE_TOLERANCE",
    "RidgeCandidate",
    "RidgeGridSpec",
    "RidgeSelection",
    "default_grid_spec",
    "enumerate_grid",
    "grid_size",
    "select_best_candidate",
    # experiment
    "BENCHMARK_STRATEGIES",
    "FORECAST_PANEL_COLUMNS",
    "BenchmarkResult",
    "CellSelection",
    "RidgeExperimentConfig",
    "ScopeExperimentResult",
    "default_experiment_config",
    "estimate_fit_counts",
    "run_benchmark",
    "run_scope_experiment",
    "select_for_cell",
    "write_benchmark_outputs",
    "write_scope_outputs",
]
