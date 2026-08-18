"""Run GLP forecast-loss studies across multiple selection scopes.

This runner is the first-class study entry point for Stage 6's scheduled,
multi-cell orchestration. The legacy GLP scripts remain available for backward
compatibility and continue to expose their original low-budget defaults. This
study runner does not silently inherit those low-budget settings.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common_hpo import LossConfig, ScaleConfig, SelectionPlan, SelectionSchedule, ValidationScheme, build_selection_plan  # noqa: E402
from common_hpo.schedules import ScheduleError  # noqa: E402
from experiment_provenance import deterministic_rng_context, stable_child_seed  # noqa: E402
from glp_hyperparameter_optimization.config import (  # noqa: E402
    EVAL_HORIZONS_QUARTERS,
    GLP_ACTUAL_VINTAGE,
    GLP_FORECAST_START,
    GLP_LAGS,
    GLP_MODEL_FORECAST_END,
    GLP_REALTIME_PANEL_PATH,
    MAX_FORECAST_HORIZON_QUARTERS,
    forecast_origin_dates,
    model_codes,
    resolve_project_path,
)
from glp_hyperparameter_optimization.loss_engine import GLPCellSpec, GLPValidationContext, evaluate_glp_candidate, make_glp_loss_objective  # noqa: E402
from glp_hyperparameter_optimization.search_config import GLPSearchConfig  # noqa: E402
from glp_hyperparameter_optimization.selection_experiment import CellSelection, GLPExperimentResult, run_glp_selection_experiment  # noqa: E402


RUNNER_NAME = "glp_scope_grid"
SUPPORTED_SELECTION_SCOPES = (
    "pooled",
    "horizon",
    "variable",
    "variable_horizon",
    "group",
)
LEGACY_LOW_BUDGET_TOTAL = 20
DEFAULT_OPTIMIZATION_INIT_POINTS = 24
DEFAULT_OPTIMIZATION_ITERATIONS = 72
DEFAULT_INNER_N_ORIGINS = 20
DEFAULT_INNER_ORIGIN_STRIDE = 2
DEFAULT_OBJECTIVE_POSTERIOR_DRAWS = 25
DEFAULT_SELECTION_FREQUENCY = "4"
EXPECTED_OUTPUT_FILES = (
    "forecast_panel.csv",
    "selected_hyperparameters.csv",
    "run_metadata.json",
)


@dataclass(frozen=True)
class ScopeRunPlan:
    scope: str
    output_dir: Path
    selection_plan: SelectionPlan
    n_selection_events: int
    n_target_cells: int
    estimated_optimization_cells: int
    estimated_candidate_evaluations: int
    estimated_validation_split_evaluations: int
    status: str

    def to_manifest(self) -> dict[str, object]:
        return {
            "scope": self.scope,
            "output_dir": str(self.output_dir),
            "run_directory_name": self.output_dir.name,
            "status": self.status,
            "selection_plan": self.selection_plan.to_dict(),
            "n_selection_events": self.n_selection_events,
            "n_target_cells": self.n_target_cells,
            "estimated_optimization_cells": self.estimated_optimization_cells,
            "estimated_candidate_evaluations": self.estimated_candidate_evaluations,
            "estimated_validation_split_evaluations": self.estimated_validation_split_evaluations,
        }


@dataclass(frozen=True)
class ScopeGridConfig:
    output_root: Path
    panel_path: Path
    model_size: str
    start: pd.Timestamp
    end: pd.Timestamp
    actual_vintage: pd.Timestamp
    lags: int
    selection_scopes: tuple[str, ...]
    target_variables: tuple[str, ...]
    target_horizons: tuple[int, ...]
    variable_groups: tuple[tuple[str, tuple[str, ...]], ...]
    residual_group_name: str | None
    separate_group_horizons: bool
    loss_metric: str
    loss_scaling: str
    benchmark: str | None
    validation_scheme: ValidationScheme
    schedule: SelectionSchedule
    search_config: GLPSearchConfig
    search_config_id: str
    loss_config: LossConfig
    loss_config_id: str
    optimizer_init_points: int
    optimizer_iterations: int
    optimizer_seed: int
    objective_posterior_draws: int
    save_all_cell_forecasts: bool
    execution_mode: str
    worker_count: int
    dry_run: bool
    if_exists_policy: str
    command_line: str
    argv: tuple[str, ...]
    outer_origins: tuple[pd.Timestamp, ...]
    warnings: tuple[str, ...]
    scope_plans: tuple[ScopeRunPlan, ...]


@dataclass(frozen=True)
class _OuterOriginBundle:
    y: np.ndarray
    codes: tuple[str, ...]
    quarter_index: pd.PeriodIndex
    context: Any


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_dumps(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def _csv_items(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _csv_ints(value: str | None) -> list[int]:
    return [int(item) for item in _csv_items(value)]


def _csv_floats(value: str | None) -> list[float]:
    return [float(item) for item in _csv_items(value)]


def _unique_preserving_order(values: Iterable[Any]) -> tuple[Any, ...]:
    seen: list[Any] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return tuple(seen)


def parse_selection_scopes(value: str) -> tuple[str, ...]:
    scopes = _unique_preserving_order(_csv_items(value))
    if not scopes:
        raise ValueError("selection_scopes must be non-empty.")
    invalid = [scope for scope in scopes if scope not in SUPPORTED_SELECTION_SCOPES]
    if invalid:
        raise ValueError(
            "unknown selection scope(s) "
            f"{invalid}; choose from {', '.join(SUPPORTED_SELECTION_SCOPES)}."
        )
    return tuple(str(scope) for scope in scopes)


def parse_variable_groups(value: str | None) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not value:
        return ()
    groups: list[tuple[str, tuple[str, ...]]] = []
    names: set[str] = set()
    for chunk in value.split(";"):
        item = chunk.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                "variable_groups must use the form 'Name=VAR1+VAR2;Other=VAR3'."
            )
        raw_name, raw_members = item.split("=", 1)
        name = raw_name.strip()
        if not name:
            raise ValueError("group names must be non-empty.")
        if name in names:
            raise ValueError(f"duplicate group name {name!r}.")
        members = tuple(
            part.strip()
            for part in raw_members.replace("+", ",").split(",")
            if part.strip()
        )
        if not members:
            raise ValueError(f"group {name!r} must contain at least one variable.")
        groups.append((name, members))
        names.add(name)
    return tuple(groups)


def parse_selection_frequency(value: str) -> SelectionSchedule:
    normalized = str(value).strip().lower()
    if normalized == "":
        raise ValueError("selection_frequency must be non-empty.")
    if normalized in {"once", "single"}:
        return SelectionSchedule.once()
    if normalized in {"per_origin", "every_origin"}:
        return SelectionSchedule.every_origin()
    if normalized in {"annual", "annual_quarterly"}:
        return SelectionSchedule.annual_quarterly()
    try:
        every_n = int(normalized)
    except ValueError as exc:
        raise ValueError(
            "selection_frequency must be 'once', 'per_origin', 'annual_quarterly', "
            "or a positive integer number of origins."
        ) from exc
    return SelectionSchedule.every_n_origins(every_n)


def _selection_frequency_label(schedule: SelectionSchedule) -> str:
    if schedule.kind == "once":
        return "once"
    if schedule.kind == "every_origin":
        return "per_origin"
    if schedule.kind == "every_n_origins" and schedule.n is not None:
        return str(schedule.n)
    if schedule.kind == "explicit_indices":
        return "explicit_indices"
    return schedule.kind


def _search_config_id(config: GLPSearchConfig) -> str:
    if config.optimize_psi:
        psi_label = f"psi-{config.psi_parameterization}"
    else:
        psi_label = f"psi-fixed-{config.fixed_psi_source}"
    return f"glp-{config.mode}-{psi_label}"


def _loss_config_id(metric: str, scaling: str, benchmark: str | None) -> str:
    parts = [metric, scaling]
    if scaling == "benchmark_rmse" and benchmark is not None:
        parts.append(benchmark)
    return "-".join(parts)


def _scope_output_dir(output_root: Path, scope: str) -> Path:
    return output_root / f"scope-{scope.replace('_', '-')}"


def _model_codes(size: str) -> tuple[str, ...]:
    return tuple(model_codes(size))


def _validate_target_variables(
    target_variables: Sequence[str],
    *,
    available_codes: Sequence[str],
) -> tuple[str, ...]:
    variables = _unique_preserving_order(
        str(value).strip() for value in target_variables if str(value).strip()
    )
    if not variables:
        raise ValueError("target_variables must be non-empty.")
    missing = [value for value in variables if value not in available_codes]
    if missing:
        raise ValueError(
            f"target_variables contains unknown code(s) {missing}; "
            f"available codes for this model are {list(available_codes)}."
        )
    return tuple(variables)


def _validate_target_horizons(target_horizons: Sequence[int]) -> tuple[int, ...]:
    horizons = _unique_preserving_order(int(value) for value in target_horizons)
    if not horizons:
        raise ValueError("target_horizons must be non-empty.")
    invalid = [value for value in horizons if value < 1 or value > MAX_FORECAST_HORIZON_QUARTERS]
    if invalid:
        raise ValueError(
            "target_horizons must lie within "
            f"1..{MAX_FORECAST_HORIZON_QUARTERS}, got {invalid}."
        )
    return tuple(horizons)


def _build_loss_config(metric: str, scaling: str) -> LossConfig:
    if scaling == "none":
        return LossConfig(aggregation=metric)
    return LossConfig(aggregation=metric, scale=ScaleConfig(method="benchmark_rmse"))


def _build_search_config(
    *,
    optimize_psi: bool,
    fixed_psi_source: str | None,
    fixed_psi_values: Sequence[float] | None,
) -> GLPSearchConfig:
    if optimize_psi:
        return GLPSearchConfig.legacy_full()
    return GLPSearchConfig.reduced_lambda_theta_miu(
        fixed_psi_source=fixed_psi_source or "context_ss",
        fixed_psi_values=fixed_psi_values,
    )


def _selection_search_dimension(search_config: GLPSearchConfig, model_size: str) -> int:
    dimension = 0
    dimension += int(search_config.optimize_lambda)
    dimension += int(search_config.optimize_theta)
    dimension += int(search_config.optimize_miu)
    if search_config.optimize_psi:
        dimension += len(_model_codes(model_size))
    return dimension


def _scientific_warnings(
    *,
    search_dimension: int,
    total_budget: int,
    n_inner_origins: int,
    optimize_psi: bool,
    model_size: str,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if total_budget <= LEGACY_LOW_BUDGET_TOTAL:
        warnings.append(
            "The requested optimizer budget matches or falls below the legacy GLP "
            "scripts' low-budget settings. Those settings are retained only for "
            "backward compatibility and should not be treated as study-quality defaults."
        )
    if total_budget < max(24, 8 * search_dimension):
        warnings.append(
            "The optimizer budget is scientifically weak relative to the requested "
            f"search dimension ({search_dimension} dimensions, {total_budget} total "
            "candidate evaluations per optimization cell)."
        )
    if optimize_psi and model_size in {"medium", "large"} and total_budget < 12 * search_dimension:
        warnings.append(
            "Optimizing psi in the medium or large GLP model expands the search space "
            f"to {search_dimension} dimensions. The requested budget is too small to "
            "treat this as a paper-quality search."
        )
    if n_inner_origins < 8:
        warnings.append(
            "Fewer than eight inner validation origins were requested. That is a thin "
            "basis for a scope study and should be treated as exploratory only."
        )
    return tuple(warnings)


def _classify_existing_directory(output_dir: Path, *, save_all_cell_forecasts: bool) -> str:
    if not output_dir.exists():
        return "missing"
    if not output_dir.is_dir():
        raise FileExistsError(f"run directory path exists but is not a directory: {output_dir}")
    entries = list(output_dir.iterdir())
    if not entries:
        return "empty"
    expected = list(EXPECTED_OUTPUT_FILES)
    if save_all_cell_forecasts:
        expected.append("forecast_panel_all_cells.csv")
    complete = all((output_dir / name).exists() for name in expected)
    return "complete" if complete else "partial"


def _resolve_existing_policy(
    output_dir: Path,
    *,
    if_exists_policy: str,
    save_all_cell_forecasts: bool,
) -> str:
    state = _classify_existing_directory(
        output_dir,
        save_all_cell_forecasts=save_all_cell_forecasts,
    )
    if if_exists_policy == "overwrite":
        return "planned"
    if if_exists_policy == "resume":
        if state == "complete":
            return "resume_skip"
        if state in {"missing", "empty"}:
            return "planned"
        raise FileExistsError(
            f"run directory {output_dir} already contains partial outputs; "
            "refusing an ambiguous resume. Use --overwrite to replace it."
        )
    if state in {"missing", "empty"}:
        return "planned"
    raise FileExistsError(
        f"run directory {output_dir} already exists and is not empty. "
        "Use --resume or --overwrite explicitly."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run GLP forecast-loss experiments across multiple selection scopes."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--panel-path", type=Path, default=GLP_REALTIME_PANEL_PATH)
    parser.add_argument("--model-size", choices=("small", "medium", "large"), required=True)
    parser.add_argument("--start", type=str, default=GLP_FORECAST_START.strftime("%Y-%m-%d"))
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--actual-vintage", type=str, default=GLP_ACTUAL_VINTAGE.strftime("%Y-%m-%d"))
    parser.add_argument("--lags", type=int, default=GLP_LAGS)
    parser.add_argument("--selection-scopes", type=str, required=True)
    parser.add_argument("--target-variables", type=str, default=None)
    parser.add_argument(
        "--target-horizons",
        type=str,
        default=",".join(str(value) for value in EVAL_HORIZONS_QUARTERS),
    )
    parser.add_argument("--variable-groups", type=str, default=None)
    parser.add_argument("--residual-group-name", type=str, default=None)
    parser.add_argument("--group-separate-horizons", action="store_true")
    parser.add_argument("--loss-metric", choices=("rmse", "mse", "mae"), default="rmse")
    parser.add_argument(
        "--loss-scaling",
        choices=("none", "benchmark_rmse"),
        default="benchmark_rmse",
    )
    parser.add_argument(
        "--benchmark",
        choices=("none", "last_observation", "no_change"),
        default="last_observation",
    )
    parser.add_argument(
        "--inner-window",
        choices=("expanding", "rolling"),
        default="expanding",
    )
    parser.add_argument("--inner-n-origins", type=int, default=DEFAULT_INNER_N_ORIGINS)
    parser.add_argument("--inner-origin-stride", type=int, default=DEFAULT_INNER_ORIGIN_STRIDE)
    parser.add_argument(
        "--inner-origin-selection",
        choices=("recent", "evenly_spaced", "random"),
        default="recent",
    )
    parser.add_argument("--inner-random-seed", type=int, default=None)
    parser.add_argument("--rolling-window-length", type=int, default=None)
    parser.add_argument(
        "--selection-frequency",
        type=str,
        default=DEFAULT_SELECTION_FREQUENCY,
        help="Use 'once', 'per_origin', 'annual_quarterly', or an integer N for every N origins.",
    )
    parser.add_argument("--optimization-init-points", type=int, default=DEFAULT_OPTIMIZATION_INIT_POINTS)
    parser.add_argument("--optimization-iterations", type=int, default=DEFAULT_OPTIMIZATION_ITERATIONS)
    parser.add_argument("--optimizer-seed", type=int, default=0)
    parser.add_argument("--objective-posterior-draws", type=int, default=DEFAULT_OBJECTIVE_POSTERIOR_DRAWS)
    psi_group = parser.add_mutually_exclusive_group()
    psi_group.add_argument("--optimize-psi", dest="optimize_psi", action="store_true")
    psi_group.add_argument("--no-optimize-psi", dest="optimize_psi", action="store_false")
    parser.set_defaults(optimize_psi=False)
    parser.add_argument(
        "--fixed-psi-source",
        choices=("context_ss", "supplied"),
        default="context_ss",
    )
    parser.add_argument("--fixed-psi-values", type=str, default=None)
    parser.add_argument("--save-all-cell-forecasts", action="store_true")
    parser.add_argument("--execution-mode", choices=("serial", "parallel"), default="serial")
    parser.add_argument("--worker-count", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    existing_group = parser.add_mutually_exclusive_group()
    existing_group.add_argument("--resume", action="store_true")
    existing_group.add_argument("--overwrite", action="store_true")
    return parser


def build_study_config(
    args: argparse.Namespace,
    *,
    argv: Sequence[str] | None = None,
    program: str = "scripts/glp/run_glp_scope_grid.py",
) -> ScopeGridConfig:
    output_root = resolve_project_path(args.output_root)
    panel_path = resolve_project_path(args.panel_path)
    model_size = str(args.model_size)
    available_codes = _model_codes(model_size)
    selection_scopes = parse_selection_scopes(args.selection_scopes)
    target_variables_raw = _csv_items(args.target_variables)
    if not target_variables_raw:
        if any(scope in {"variable", "variable_horizon", "group"} for scope in selection_scopes):
            raise ValueError(
                "variable, variable_horizon, and group scope studies require explicit "
                "--target-variables; otherwise the scope-to-cell mapping is ambiguous."
            )
        target_variables = available_codes
    else:
        target_variables = _validate_target_variables(
            target_variables_raw,
            available_codes=available_codes,
        )

    target_horizons = _validate_target_horizons(_csv_ints(args.target_horizons))
    variable_groups = parse_variable_groups(args.variable_groups)
    residual_group_name = str(args.residual_group_name).strip() if args.residual_group_name else None

    if "group" in selection_scopes and not variable_groups and residual_group_name is None:
        raise ValueError(
            "group scope requires --variable-groups, --residual-group-name, or both."
        )
    if "group" not in selection_scopes and (variable_groups or residual_group_name or args.group_separate_horizons):
        raise ValueError(
            "group-specific options are only valid when selection_scopes includes 'group'."
        )

    if_exists_policy = "overwrite" if args.overwrite else "resume" if args.resume else "error"

    benchmark = None if args.benchmark == "none" else "last_observation" if args.benchmark == "no_change" else args.benchmark
    if args.loss_scaling == "benchmark_rmse" and benchmark is None:
        raise ValueError(
            "benchmark_rmse loss scaling requires an explicit benchmark selection."
        )
    if args.loss_scaling != "benchmark_rmse" and benchmark is not None:
        raise ValueError(
            "benchmark selection is only valid when --loss-scaling=benchmark_rmse."
        )

    search_config = _build_search_config(
        optimize_psi=bool(args.optimize_psi),
        fixed_psi_source=None if args.optimize_psi else args.fixed_psi_source,
        fixed_psi_values=tuple(_csv_floats(args.fixed_psi_values)) if args.fixed_psi_values else None,
    )
    loss_config = _build_loss_config(args.loss_metric, args.loss_scaling)

    origin_selection_map = {
        "recent": "most_recent",
        "evenly_spaced": "evenly_spaced",
        "random": "random",
    }
    if args.inner_origin_selection == "random" and args.inner_random_seed is None:
        raise ValueError(
            "random inner-origin selection requires --inner-random-seed for reproducibility."
        )
    validation_scheme = ValidationScheme(
        training_window=args.inner_window,
        origin_selection=origin_selection_map[args.inner_origin_selection],
        n_origins=int(args.inner_n_origins),
        horizons=target_horizons,
        min_train_length=max(4 * int(args.lags), max(target_horizons) + int(args.lags), int(args.lags) + 3),
        origin_stride=int(args.inner_origin_stride),
        rolling_window_length=args.rolling_window_length,
        random_seed=args.inner_random_seed,
    )

    try:
        schedule = parse_selection_frequency(args.selection_frequency)
    except ScheduleError as exc:
        raise ValueError(str(exc)) from exc

    requested_start = pd.Timestamp(args.start)
    model_end = pd.Timestamp(GLP_MODEL_FORECAST_END[model_size])
    requested_end = pd.Timestamp(args.end) if args.end else model_end
    if requested_end > model_end:
        raise ValueError(
            f"end date {requested_end:%Y-%m-%d} exceeds the supported {model_size} "
            f"model endpoint {model_end:%Y-%m-%d}."
        )
    if requested_start > requested_end:
        raise ValueError("start must be less than or equal to end.")
    outer_origins = tuple(pd.Timestamp(origin) for origin in forecast_origin_dates(requested_start, requested_end))
    if not outer_origins:
        raise ValueError("the requested outer-origin range produced no forecast origins.")

    total_budget = int(args.optimization_init_points) + int(args.optimization_iterations)
    search_dimension = _selection_search_dimension(search_config, model_size)
    warnings = _scientific_warnings(
        search_dimension=search_dimension,
        total_budget=total_budget,
        n_inner_origins=validation_scheme.n_origins,
        optimize_psi=search_config.optimize_psi,
        model_size=model_size,
    )

    n_events = len(schedule.resolve(outer_origins))
    scope_plans: list[ScopeRunPlan] = []
    for scope in selection_scopes:
        if scope == "group":
            selection_plan = build_selection_plan(
                scope,
                target_variables,
                target_horizons,
                variable_groups=variable_groups,
                separate_group_horizons=bool(args.group_separate_horizons),
                residual_group_name=residual_group_name,
            )
        else:
            selection_plan = build_selection_plan(scope, target_variables, target_horizons)
        output_dir = _scope_output_dir(output_root, scope)
        status = _resolve_existing_policy(
            output_dir,
            if_exists_policy=if_exists_policy,
            save_all_cell_forecasts=bool(args.save_all_cell_forecasts),
        )
        n_cells = len(selection_plan.cells)
        optimization_cells = n_events * n_cells
        candidate_evaluations = optimization_cells * total_budget
        scope_plans.append(
            ScopeRunPlan(
                scope=scope,
                output_dir=output_dir,
                selection_plan=selection_plan,
                n_selection_events=n_events,
                n_target_cells=n_cells,
                estimated_optimization_cells=optimization_cells,
                estimated_candidate_evaluations=candidate_evaluations,
                estimated_validation_split_evaluations=(candidate_evaluations * validation_scheme.n_origins),
                status=status,
            )
        )

    argv_tokens = tuple(str(value) for value in (argv if argv is not None else sys.argv[1:]))
    command_line = shlex.join((program, *argv_tokens))

    worker_count = 1
    if args.execution_mode == "parallel":
        requested_workers = int(args.worker_count) if args.worker_count is not None else len(scope_plans)
        worker_count = max(1, min(requested_workers, len(scope_plans)))

    return ScopeGridConfig(
        output_root=output_root,
        panel_path=panel_path,
        model_size=model_size,
        start=requested_start,
        end=requested_end,
        actual_vintage=pd.Timestamp(args.actual_vintage),
        lags=int(args.lags),
        selection_scopes=selection_scopes,
        target_variables=target_variables,
        target_horizons=target_horizons,
        variable_groups=variable_groups,
        residual_group_name=residual_group_name,
        separate_group_horizons=bool(args.group_separate_horizons),
        loss_metric=str(args.loss_metric),
        loss_scaling=str(args.loss_scaling),
        benchmark=benchmark,
        validation_scheme=validation_scheme,
        schedule=schedule,
        search_config=search_config,
        search_config_id=_search_config_id(search_config),
        loss_config=loss_config,
        loss_config_id=_loss_config_id(str(args.loss_metric), str(args.loss_scaling), benchmark),
        optimizer_init_points=int(args.optimization_init_points),
        optimizer_iterations=int(args.optimization_iterations),
        optimizer_seed=int(args.optimizer_seed),
        objective_posterior_draws=int(args.objective_posterior_draws),
        save_all_cell_forecasts=bool(args.save_all_cell_forecasts),
        execution_mode=str(args.execution_mode),
        worker_count=worker_count,
        dry_run=bool(args.dry_run),
        if_exists_policy=if_exists_policy,
        command_line=command_line,
        argv=argv_tokens,
        outer_origins=outer_origins,
        warnings=warnings,
        scope_plans=tuple(scope_plans),
    )


def manifest_for_config(config: ScopeGridConfig) -> dict[str, object]:
    return {
        "runner": RUNNER_NAME,
        "generated_utc": _timestamp_utc(),
        "command_line": config.command_line,
        "argv": list(config.argv),
        "output_root": str(config.output_root),
        "panel_path": str(config.panel_path),
        "model_size": config.model_size,
        "outer_origins": {
            "start": config.start.strftime("%Y-%m-%d"),
            "end": config.end.strftime("%Y-%m-%d"),
            "count": len(config.outer_origins),
            "labels": [pd.Timestamp(value).strftime("%Y-%m-%d") for value in config.outer_origins],
        },
        "actual_vintage": config.actual_vintage.strftime("%Y-%m-%d"),
        "lags": config.lags,
        "selection_scopes": list(config.selection_scopes),
        "target_variables": list(config.target_variables),
        "target_horizons": list(config.target_horizons),
        "variable_groups": [
            {"name": name, "variables": list(variables)}
            for name, variables in config.variable_groups
        ],
        "residual_group_name": config.residual_group_name,
        "separate_group_horizons": config.separate_group_horizons,
        "loss_request": {
            "metric": config.loss_metric,
            "scaling": config.loss_scaling,
            "benchmark": config.benchmark,
            "loss_config_id": config.loss_config_id,
        },
        "selection_schedule": {
            "requested": _selection_frequency_label(config.schedule),
            "resolved": config.schedule.to_dict(),
        },
        "validation_scheme": config.validation_scheme.to_dict(),
        "search_config": config.search_config.to_dict(),
        "search_config_id": config.search_config_id,
        "optimizer_budget": {
            "init_points": config.optimizer_init_points,
            "iterations": config.optimizer_iterations,
            "total_candidates_per_cell": config.optimizer_init_points + config.optimizer_iterations,
            "optimizer_seed": config.optimizer_seed,
            "objective_posterior_draws": config.objective_posterior_draws,
        },
        "study_defaults_note": (
            "This scope-grid runner defaults to the recommended reduced search "
            "(lambda, theta, miu with psi fixed) and does not treat the legacy "
            "scripts' low-budget settings as recommended study defaults."
        ),
        "save_all_cell_forecasts": config.save_all_cell_forecasts,
        "execution": {
            "mode": config.execution_mode,
            "worker_count": config.worker_count,
            "dry_run": config.dry_run,
        },
        "if_exists_policy": config.if_exists_policy,
        "warnings": list(config.warnings),
        "planned_runs": [plan.to_manifest() for plan in config.scope_plans],
    }


def _scope_manifest(config: ScopeGridConfig, plan: ScopeRunPlan) -> dict[str, object]:
    return {
        "runner": RUNNER_NAME,
        "generated_utc": _timestamp_utc(),
        "command_line": config.command_line,
        "argv": list(config.argv),
        "scope": plan.scope,
        "output_dir": str(plan.output_dir),
        "selection_plan": plan.selection_plan.to_dict(),
        "selection_schedule": config.schedule.to_dict(),
        "validation_scheme": config.validation_scheme.to_dict(),
        "search_config": config.search_config.to_dict(),
        "search_config_id": config.search_config_id,
        "loss_request": {
            "metric": config.loss_metric,
            "scaling": config.loss_scaling,
            "benchmark": config.benchmark,
            "loss_config_id": config.loss_config_id,
        },
        "optimizer_budget": {
            "init_points": config.optimizer_init_points,
            "iterations": config.optimizer_iterations,
            "optimizer_seed": config.optimizer_seed,
            "objective_posterior_draws": config.objective_posterior_draws,
            "total_candidates_per_cell": config.optimizer_init_points + config.optimizer_iterations,
        },
        "outer_origins": [pd.Timestamp(value).strftime("%Y-%m-%d") for value in config.outer_origins],
        "planned_counts": {
            "n_selection_events": plan.n_selection_events,
            "n_target_cells": plan.n_target_cells,
            "estimated_optimization_cells": plan.estimated_optimization_cells,
            "estimated_candidate_evaluations": plan.estimated_candidate_evaluations,
            "estimated_validation_split_evaluations": plan.estimated_validation_split_evaluations,
        },
        "status": plan.status,
        "warnings": list(config.warnings),
    }


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_manifests(config: ScopeGridConfig) -> None:
    config.output_root.mkdir(parents=True, exist_ok=True)
    _write_text(config.output_root / "scope_grid_manifest.json", _json_dumps(manifest_for_config(config)))
    _write_text(
        config.output_root / "batch_metadata.json",
        _json_dumps(
            {
                "runner": RUNNER_NAME,
                "runs": [
                    {
                        "scope": plan.scope,
                        "output_dir": str(plan.output_dir),
                        "status": plan.status,
                    }
                    for plan in config.scope_plans
                ],
            }
        ),
    )
    for plan in config.scope_plans:
        plan.output_dir.mkdir(parents=True, exist_ok=True)
        _write_text(plan.output_dir / "experiment_manifest.json", _json_dumps(_scope_manifest(config, plan)))
        _write_text(plan.output_dir / "command.txt", config.command_line + "\n")


def _build_benchmark_callback(
    benchmark: str | None,
    contexts: Sequence[GLPValidationContext],
) -> Callable[..., float] | None:
    if benchmark is None:
        return None
    if benchmark != "last_observation":
        raise ValueError(f"unsupported benchmark {benchmark!r}.")
    by_split = {context.split_id: context for context in contexts}

    def callback(*, variable: str, horizon: int, origin: object) -> float:
        del horizon
        context = by_split[str(origin)]
        column = context.variable_index(variable)
        return float(context.context.y[-1, column])

    return callback


def load_scope_grid_panel(panel_path: Path):
    from glp_hyperparameter_optimization.data_utils import load_glp_realtime_panel

    return load_glp_realtime_panel(panel_path)


def _load_outer_origin_bundle(
    panel,
    *,
    origin_label: object,
    model_size: str,
    lags: int,
) -> _OuterOriginBundle:
    from glp_hyperparameter_optimization.data_utils import build_glp_estimation_matrix
    from glp_hyperparameter_optimization.glp_model import prepare_glp_context

    origin = pd.Timestamp(origin_label)
    y, codes, quarter_index = build_glp_estimation_matrix(panel, origin, model_size)
    context = prepare_glp_context(y, lags)
    return _OuterOriginBundle(
        y=np.asarray(y, dtype=float),
        codes=tuple(codes),
        quarter_index=pd.PeriodIndex(quarter_index, freq="Q"),
        context=context,
    )


def _objective_contexts(
    *,
    bundle: _OuterOriginBundle,
    lags: int,
    validation_scheme: ValidationScheme,
) -> tuple[GLPValidationContext, ...]:
    from common_hpo import build_validation_splits
    from glp_hyperparameter_optimization.glp_model import prepare_glp_context

    splits = build_validation_splits(
        bundle.y.shape[0],
        validation_scheme,
        {horizon: horizon for horizon in validation_scheme.horizons},
        outer_info_cutoff=bundle.y.shape[0] - 1,
        date_labels=bundle.quarter_index,
    )
    horizon_rows = {horizon: index for index, horizon in enumerate(validation_scheme.horizons)}
    contexts: list[GLPValidationContext] = []
    for split in splits:
        training = bundle.y[split.train_start: split.train_end + 1, :]
        actual = np.vstack(
            [bundle.y[split.target_for(horizon), :] for horizon in validation_scheme.horizons]
        )
        contexts.append(
            GLPValidationContext(
                split_id=split.split_id,
                origin=split.origin,
                context=prepare_glp_context(training, lags),
                codes=bundle.codes,
                actual=actual,
                horizon_rows=horizon_rows,
                split=split,
            )
        )
    return tuple(contexts)


def _posterior_mean_forecast(
    *,
    context,
    natural_vector: Sequence[float],
    horizons: Sequence[int],
    draw_count: int,
    seed_base: int,
    origin_index: int,
    cell_id: str,
    event_id: str,
) -> np.ndarray:
    from glp_hyperparameter_optimization.glp_model import glp_draw, glp_mode_estimate, point_forecast

    natural = np.asarray(natural_vector, dtype=float)
    ordered_horizons = list(horizons)
    if draw_count <= 1:
        beta, _ = glp_mode_estimate(context, natural)
        return np.asarray(point_forecast(context.y, beta, ordered_horizons), dtype=float)

    total: np.ndarray | None = None
    for draw_index in range(int(draw_count)):
        child_seed = stable_child_seed(
            seed_base,
            "glp-scope-grid-forecast",
            int(origin_index),
            cell_id,
            event_id,
            int(draw_index),
        )
        with deterministic_rng_context(child_seed):
            beta, _ = glp_draw(context, natural)
            forecast = np.asarray(point_forecast(context.y, beta, ordered_horizons), dtype=float)
        total = forecast if total is None else total + forecast
    assert total is not None
    return total / float(draw_count)


def _selection_named_parameters(natural_vector: Sequence[float], context) -> dict[str, object]:
    from glp_hyperparameter_optimization.glp_model import natural_vector_to_hyper

    raw = natural_vector_to_hyper(natural_vector, context)
    named: dict[str, object] = {}
    for key, value in raw.items():
        if isinstance(value, np.ndarray):
            named[str(key)] = [float(item) for item in value.ravel()]
        elif isinstance(value, (np.floating, float)):
            named[str(key)] = float(value)
        elif isinstance(value, (np.integer, int)):
            named[str(key)] = int(value)
        elif isinstance(value, list):
            named[str(key)] = [float(item) for item in value]
        else:
            named[str(key)] = value
    return named


def run_scope_study(plan: ScopeRunPlan, config: ScopeGridConfig, panel) -> GLPExperimentResult:
    from glp_hyperparameter_optimization.glp_model import RMSE_PENALTY
    from mango import Tuner, scheduler

    outer_cache: dict[pd.Timestamp, _OuterOriginBundle] = {}

    def outer_bundle(origin_label: object) -> _OuterOriginBundle:
        origin = pd.Timestamp(origin_label)
        if origin not in outer_cache:
            outer_cache[origin] = _load_outer_origin_bundle(
                panel,
                origin_label=origin,
                model_size=config.model_size,
                lags=config.lags,
            )
        return outer_cache[origin]

    def selector(request) -> CellSelection:
        start_time = time.perf_counter()
        bundle = outer_bundle(request.origin_label)
        resolved_search = request.search_config.resolve(bundle.context)
        contexts = _objective_contexts(
            bundle=bundle,
            lags=config.lags,
            validation_scheme=request.validation_scheme,
        )
        benchmark = _build_benchmark_callback(config.benchmark, contexts)
        spec = GLPCellSpec(
            variables=request.variables,
            horizons=request.horizons,
            n_obj_draws=config.objective_posterior_draws,
            seed_base=request.seed,
            loss_config=request.loss_config,
        )
        objective = make_glp_loss_objective(
            contexts,
            spec,
            to_natural=resolved_search.to_natural,
            penalty=RMSE_PENALTY,
            benchmark=benchmark,
        )
        objective_runner = scheduler.parallel(n_jobs=1)(objective)
        with deterministic_rng_context(request.seed):
            results = Tuner(
                resolved_search.mango_param_space(),
                objective_runner,
                {
                    "initial_random": config.optimizer_init_points,
                    "num_iteration": config.optimizer_iterations,
                },
            ).minimize()
        best_params = results.get("best_params") if isinstance(results, dict) else None
        if not isinstance(best_params, dict) or not best_params:
            raise RuntimeError("scope-grid optimization returned no best_params mapping.")
        evaluation = evaluate_glp_candidate(
            best_params,
            contexts,
            spec,
            to_natural=resolved_search.to_natural,
            penalty=RMSE_PENALTY,
            benchmark=benchmark,
        )
        if evaluation.failed:
            raise RuntimeError(
                "best candidate did not survive objective re-evaluation: "
                f"{evaluation.failure_reason}"
            )

        natural_vector = tuple(float(value) for value in resolved_search.to_natural(best_params, bundle.context))
        diagnostics = getattr(objective, "diagnostics", {})
        splits = [context.split for context in contexts if context.split is not None]
        return CellSelection(
            natural_vector=natural_vector,
            named_parameters=_selection_named_parameters(natural_vector, bundle.context),
            selection_loss=float(evaluation.total_loss),
            fixed_psi_source=request.search_config.fixed_psi_source,
            fixed_psi_values=(
                tuple(float(value) for value in np.ravel(resolved_search.fixed_psi))
                if resolved_search.fixed_psi is not None
                else None
            ),
            search_dimension=resolved_search.search_dimension,
            inner_window_start=min(split.train_start for split in splits) if splits else None,
            inner_window_end=max(split.train_end for split in splits) if splits else None,
            n_inner_origins=len(contexts),
            validation_stride=request.validation_scheme.origin_stride,
            optimizer_seed=request.seed,
            optimizer_budget={
                "init_points": config.optimizer_init_points,
                "n_iter": config.optimizer_iterations,
                "total_candidates_per_cell": config.optimizer_init_points + config.optimizer_iterations,
            },
            objective_draw_count=config.objective_posterior_draws,
            failure_counts={
                "valid": int(diagnostics.get("valid", 0)),
                "penalized": int(diagnostics.get("penalized", 0)),
                "numerical_failures": int(diagnostics.get("numerical_failures", 0)),
                "nonfinite_forecasts": int(diagnostics.get("nonfinite_forecasts", 0)),
                "scale_problems": int(diagnostics.get("scale_problems", 0)),
            },
            runtime_seconds=time.perf_counter() - start_time,
        )

    def forecast_generator(request) -> np.ndarray:
        bundle = outer_bundle(request.origin_label)
        return _posterior_mean_forecast(
            context=bundle.context,
            natural_vector=request.natural_vector,
            horizons=request.system_horizons,
            draw_count=config.objective_posterior_draws,
            seed_base=config.optimizer_seed,
            origin_index=request.origin_index,
            cell_id=request.cell_id,
            event_id=request.event_id,
        )

    forecast_method = (
        "posterior_mode_point_forecast"
        if config.objective_posterior_draws <= 1
        else "posterior_predictive_mean"
    )
    return run_glp_selection_experiment(
        config.outer_origins,
        selector=selector,
        forecast_generator=forecast_generator,
        target_variables=config.target_variables,
        target_horizons=config.target_horizons,
        search_config=config.search_config,
        loss_config=config.loss_config,
        validation_scheme=config.validation_scheme,
        plan=plan.selection_plan,
        schedule=config.schedule,
        system_variables=_model_codes(config.model_size),
        system_horizons=config.target_horizons,
        retain_off_target=config.save_all_cell_forecasts,
        model="glp_scope_grid",
        model_size=config.model_size,
        vintage_token="glp_outer_vintage",
        search_config_id=config.search_config_id,
        loss_config_id=config.loss_config_id,
        forecast_method=forecast_method,
        base_seed=config.optimizer_seed,
        vintage_policy=config.validation_scheme.vintage_policy.value,
    )


def _write_scope_outputs(plan: ScopeRunPlan, config: ScopeGridConfig, result: GLPExperimentResult) -> None:
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result.forecast_panel).to_csv(plan.output_dir / "forecast_panel.csv", index=False)
    pd.DataFrame(result.selected_hyperparameters).to_csv(
        plan.output_dir / "selected_hyperparameters.csv",
        index=False,
    )
    if config.save_all_cell_forecasts:
        pd.DataFrame(result.forecast_panel_all_cells).to_csv(
            plan.output_dir / "forecast_panel_all_cells.csv",
            index=False,
        )
    metadata = dict(result.run_metadata)
    metadata.update(
        {
            "runner": RUNNER_NAME,
            "scope": plan.scope,
            "output_dir": str(plan.output_dir),
            "output_root": str(config.output_root),
            "command_line": config.command_line,
            "argv": list(config.argv),
            "actual_vintage": config.actual_vintage.strftime("%Y-%m-%d"),
            "loss_request": {
                "metric": config.loss_metric,
                "scaling": config.loss_scaling,
                "benchmark": config.benchmark,
                "loss_config_id": config.loss_config_id,
            },
            "selection_schedule_requested": _selection_frequency_label(config.schedule),
            "search_config_id": config.search_config_id,
            "optimizer_budget": {
                "init_points": config.optimizer_init_points,
                "iterations": config.optimizer_iterations,
                "total_candidates_per_cell": config.optimizer_init_points + config.optimizer_iterations,
                "optimizer_seed": config.optimizer_seed,
                "objective_posterior_draws": config.objective_posterior_draws,
            },
            "execution_mode": config.execution_mode,
            "worker_count": config.worker_count,
            "if_exists_policy": config.if_exists_policy,
            "estimated_optimization_cells": plan.estimated_optimization_cells,
            "estimated_candidate_evaluations": plan.estimated_candidate_evaluations,
            "estimated_validation_split_evaluations": plan.estimated_validation_split_evaluations,
            "save_all_cell_forecasts": config.save_all_cell_forecasts,
            "warnings": list(config.warnings),
            "completed_utc": _timestamp_utc(),
        }
    )
    _write_text(plan.output_dir / "run_metadata.json", _json_dumps(metadata))


def execute_study(config: ScopeGridConfig) -> dict[str, object]:
    manifest = manifest_for_config(config)
    for warning in config.warnings:
        print(f"WARNING: {warning}", file=sys.stderr, flush=True)
    print(_json_dumps(manifest), flush=True)
    _write_manifests(config)

    pending = [plan for plan in config.scope_plans if plan.status == "planned"]
    skipped = [plan.scope for plan in config.scope_plans if plan.status == "resume_skip"]
    if config.dry_run or not pending:
        return {
            "executed_scopes": [],
            "skipped_scopes": skipped,
            "manifest_path": str(config.output_root / "scope_grid_manifest.json"),
        }

    panel = load_scope_grid_panel(config.panel_path)
    executed: list[str] = []

    def _run_one(plan: ScopeRunPlan) -> tuple[str, GLPExperimentResult]:
        result = run_scope_study(plan, config, panel)
        _write_scope_outputs(plan, config, result)
        return plan.scope, result

    if config.execution_mode == "parallel" and len(pending) > 1:
        with ThreadPoolExecutor(max_workers=config.worker_count) as executor:
            futures = {executor.submit(_run_one, plan): plan for plan in pending}
            for future in as_completed(futures):
                scope, _result = future.result()
                executed.append(scope)
    else:
        for plan in pending:
            scope, _result = _run_one(plan)
            executed.append(scope)

    _write_text(
        config.output_root / "batch_metadata.json",
        _json_dumps(
            {
                "runner": RUNNER_NAME,
                "runs": [
                    {
                        "scope": plan.scope,
                        "output_dir": str(plan.output_dir),
                        "status": "completed" if plan.scope in executed else "skipped",
                    }
                    for plan in config.scope_plans
                ],
            }
        ),
    )
    return {
        "executed_scopes": executed,
        "skipped_scopes": skipped,
        "manifest_path": str(config.output_root / "scope_grid_manifest.json"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(argv)
    config = build_study_config(
        parsed,
        argv=tuple(argv) if argv is not None else tuple(sys.argv[1:]),
    )
    execute_study(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())