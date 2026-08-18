"""Run mixed-frequency (MF-BVAR) forecast-loss studies across selection scopes.

This is the mixed-frequency counterpart to ``scripts/glp/run_glp_scope_grid.py``.
It accepts options parallel to the GLP scope runner wherever the concepts are
equivalent (selection scopes, target variables/horizons, variable groups,
selection schedules, standardized/benchmark-scaled loss, explicit inner
validation, dry-run manifests, resume/overwrite policy) and keeps model-specific
options (the mixed-frequency forecast block, MBFVAR fit sizes) separate.

The runner reuses the shared ``common_hpo`` contracts and the mixed-frequency
adapter in :mod:`paper_hyperparameter_optimization.selection_experiment`. It does
not import any GLP code. Heavy model work (MBFVAR fitting/forecasting) is only
imported lazily inside a real run, so ``--dry-run`` needs no data download and no
MBFVAR install.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common_hpo import (  # noqa: E402
    LossConfig,
    ScaleConfig,
    SelectionPlan,
    SelectionSchedule,
    ValidationScheme,
    build_selection_plan,
)
from common_hpo.schedules import ScheduleError  # noqa: E402
from common_hpo.splits import VintagePolicy  # noqa: E402
from paper_hyperparameter_optimization.config import (  # noqa: E402
    MAX_FORECAST_HORIZON_QUARTERS,
    QUARTERLY_SERIES,
    REALTIME_PANEL_PATH,
    resolve_project_path,
)
from paper_hyperparameter_optimization.horizon_mapping import required_forecast_months  # noqa: E402


RUNNER_NAME = "mfvar_scope_grid"
SUPPORTED_SELECTION_SCOPES = (
    "pooled",
    "horizon",
    "variable",
    "variable_horizon",
    "group",
)
FULL_QUARTERLY_BLOCK = tuple(spec.paper_code for spec in QUARTERLY_SERIES)
DEFAULT_TARGET_HORIZONS = (1, 2, 4, 8)
DEFAULT_INNER_N_ORIGINS = 3
DEFAULT_INNER_ORIGIN_STRIDE = 1
EXPECTED_OUTPUT_FILES = (
    "forecast_panel.csv",
    "selected_hyperparameters.csv",
    "run_metadata.json",
    "failed_origins.csv",
)


class ScopeGridConfigError(ValueError):
    """Raised when the requested scope-grid study configuration is invalid."""


@dataclass(frozen=True)
class ScopeRunPlan:
    scope: str
    output_dir: Path
    selection_plan: SelectionPlan
    n_selection_events: int
    n_target_cells: int
    existing_policy: str


@dataclass(frozen=True)
class ScopeGridConfig:
    output_root: Path
    panel_path: Path
    forecast_variables: tuple[str, ...]
    target_variables: tuple[str, ...]
    target_horizons: tuple[int, ...]
    selection_scopes: tuple[str, ...]
    variable_groups: tuple[tuple[str, tuple[str, ...]], ...] | None
    residual_group_name: str | None
    group_separate_horizons: bool
    loss_metric: str
    loss_scaling: str
    inner_window: str
    inner_n_origins: int
    inner_origin_stride: int
    inner_origin_selection: str
    inner_random_seed: int | None
    rolling_window_length: int | None
    selection_frequency: str
    optimization_horizon_quarters: int
    optimization_eval_horizon_quarters: int | None
    optimization_n_eval: int
    base_seed: int | None
    save_all_cell_forecasts: bool
    if_exists_policy: str
    dry_run: bool
    command_line: str
    warnings: tuple[str, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------- #
# Parsing helpers (parallel to the GLP runner).
# --------------------------------------------------------------------------- #
def _csv_items(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _csv_ints(value: str | None) -> list[int]:
    return [int(item) for item in _csv_items(value)]


def parse_selection_scopes(value: str) -> tuple[str, ...]:
    scopes = _csv_items(value)
    if not scopes:
        raise ScopeGridConfigError("at least one selection scope is required.")
    invalid = [scope for scope in scopes if scope not in SUPPORTED_SELECTION_SCOPES]
    if invalid:
        allowed = ", ".join(SUPPORTED_SELECTION_SCOPES)
        raise ScopeGridConfigError(f"unknown selection scopes {invalid}; allowed: {allowed}.")
    ordered = tuple(dict.fromkeys(scopes))
    return ordered


def parse_variable_groups(
    value: str | None,
) -> tuple[tuple[str, tuple[str, ...]], ...] | None:
    if not value:
        return None
    groups: list[tuple[str, tuple[str, ...]]] = []
    for group_spec in value.split(";"):
        group_spec = group_spec.strip()
        if not group_spec:
            continue
        if "=" not in group_spec:
            raise ScopeGridConfigError(
                f"variable group {group_spec!r} must be formatted as name=VAR1,VAR2."
            )
        name, raw_variables = group_spec.split("=", 1)
        variables = tuple(_csv_items(raw_variables))
        if not variables:
            raise ScopeGridConfigError(f"variable group {name!r} must list at least one variable.")
        groups.append((name.strip(), variables))
    return tuple(groups) if groups else None


def _validate_target_variables(
    requested: Sequence[str],
    *,
    forecast_variables: Sequence[str],
) -> tuple[str, ...]:
    available = set(forecast_variables)
    invalid = [variable for variable in requested if variable not in available]
    if invalid:
        raise ScopeGridConfigError(
            f"target variables must be a subset of the forecast block {list(forecast_variables)}; "
            f"invalid entries: {invalid}."
        )
    ordered = tuple(dict.fromkeys(requested))
    return ordered


def _validate_target_horizons(horizons: Sequence[int]) -> tuple[int, ...]:
    if not horizons:
        raise ScopeGridConfigError("at least one target horizon is required.")
    invalid = [h for h in horizons if h < 1 or h > MAX_FORECAST_HORIZON_QUARTERS]
    if invalid:
        raise ScopeGridConfigError(
            f"target horizons must lie within 1..{MAX_FORECAST_HORIZON_QUARTERS}, got {invalid}."
        )
    return tuple(dict.fromkeys(horizons))


def build_selection_schedule(value: str) -> SelectionSchedule:
    token = str(value).strip().lower()
    if token in {"once", "first_origin"}:
        return SelectionSchedule.once()
    if token in {"per_origin", "every_origin"}:
        return SelectionSchedule.every_origin()
    if token in {"annual_quarterly", "annual"}:
        return SelectionSchedule.every_n_origins(4, interpretation="annual (every 4 quarterly origins)")
    try:
        n = int(token)
    except ValueError as exc:
        raise ScopeGridConfigError(
            "selection-frequency must be 'once', 'per_origin', 'annual_quarterly', or an integer N."
        ) from exc
    return SelectionSchedule.every_n_origins(n)


def _build_loss_config(metric: str, scaling: str) -> LossConfig:
    if scaling == "none":
        return LossConfig(aggregation=metric)
    return LossConfig(aggregation=metric, scale=ScaleConfig(method="benchmark_rmse"))


def _build_validation_scheme(config: ScopeGridConfig) -> ValidationScheme:
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


def _build_selection_plan(config: ScopeGridConfig, scope: str) -> SelectionPlan:
    if scope == "group":
        return build_selection_plan(
            scope,
            config.target_variables,
            config.target_horizons,
            variable_groups=config.variable_groups,
            residual_group_name=config.residual_group_name,
            separate_group_horizons=config.group_separate_horizons,
        )
    return build_selection_plan(scope, config.target_variables, config.target_horizons)


# --------------------------------------------------------------------------- #
# Config construction and manifests.
# --------------------------------------------------------------------------- #
def build_study_config(
    args: argparse.Namespace,
    *,
    argv: Sequence[str] | None = None,
    program: str = "scripts/mfvar/run_mfvar_scope_grid.py",
) -> ScopeGridConfig:
    output_root = resolve_project_path(args.output_root)
    panel_path = resolve_project_path(args.panel_path)

    forecast_variables = tuple(_csv_items(args.forecast_variables)) or FULL_QUARTERLY_BLOCK
    selection_scopes = parse_selection_scopes(args.selection_scopes)

    target_variables_raw = _csv_items(args.target_variables)
    if not target_variables_raw:
        if any(scope in {"variable", "variable_horizon", "group"} for scope in selection_scopes):
            raise ScopeGridConfigError(
                "variable, variable_horizon, and group scope studies require explicit "
                "--target-variables; otherwise the scope-to-cell mapping is ambiguous."
            )
        target_variables = forecast_variables
    else:
        target_variables = _validate_target_variables(
            target_variables_raw, forecast_variables=forecast_variables
        )

    target_horizons = _validate_target_horizons(
        _csv_ints(args.target_horizons) or list(DEFAULT_TARGET_HORIZONS)
    )
    variable_groups = parse_variable_groups(args.variable_groups)
    residual_group_name = str(args.residual_group_name).strip() if args.residual_group_name else None

    if args.optimization_eval_horizon_quarters is not None:
        eval_horizon = int(args.optimization_eval_horizon_quarters)
    else:
        eval_horizon = None

    required_months = required_forecast_months(config_max_horizon := max(target_horizons))
    if args.optimization_horizon_quarters < config_max_horizon:
        raise ScopeGridConfigError(
            "optimization-horizon-quarters must be at least the largest target horizon "
            f"({config_max_horizon}); got {args.optimization_horizon_quarters}."
        )

    if_exists_policy = "overwrite" if args.overwrite else "resume" if args.resume else "fail"

    warnings: list[str] = []
    if args.inner_n_origins < 3:
        warnings.append(
            "Fewer than three inner validation origins were requested; treat this as exploratory only."
        )

    resolved_argv = list(argv) if argv is not None else sys.argv[1:]
    command_line = " ".join([program, *(shlex.quote(token) for token in resolved_argv)])

    return ScopeGridConfig(
        output_root=output_root,
        panel_path=panel_path,
        forecast_variables=forecast_variables,
        target_variables=target_variables,
        target_horizons=target_horizons,
        selection_scopes=selection_scopes,
        variable_groups=variable_groups,
        residual_group_name=residual_group_name,
        group_separate_horizons=bool(args.group_separate_horizons),
        loss_metric=str(args.loss_metric),
        loss_scaling=str(args.loss_scaling),
        inner_window=str(args.inner_window),
        inner_n_origins=int(args.inner_n_origins),
        inner_origin_stride=int(args.inner_origin_stride),
        inner_origin_selection=str(args.inner_origin_selection),
        inner_random_seed=args.inner_random_seed,
        rolling_window_length=args.rolling_window_length,
        selection_frequency=str(args.selection_frequency),
        optimization_horizon_quarters=int(args.optimization_horizon_quarters),
        optimization_eval_horizon_quarters=eval_horizon,
        optimization_n_eval=int(args.optimization_n_eval),
        base_seed=args.base_seed,
        save_all_cell_forecasts=bool(args.save_all_cell_forecasts),
        if_exists_policy=if_exists_policy,
        dry_run=bool(args.dry_run),
        command_line=command_line,
        warnings=tuple(warnings),
    )


def _classify_existing_directory(output_dir: Path, *, save_all_cell_forecasts: bool) -> str:
    if not output_dir.exists():
        return "missing"
    if not output_dir.is_dir():
        raise FileExistsError(f"run directory path exists but is not a directory: {output_dir}")
    if not list(output_dir.iterdir()):
        return "empty"
    expected = list(EXPECTED_OUTPUT_FILES)
    if save_all_cell_forecasts:
        expected.append("forecast_panel_all_cells.csv")
    complete = all((output_dir / name).exists() for name in expected)
    return "complete" if complete else "partial"


def _resolve_existing_policy(
    output_dir: Path, *, if_exists_policy: str, save_all_cell_forecasts: bool
) -> str:
    state = _classify_existing_directory(
        output_dir, save_all_cell_forecasts=save_all_cell_forecasts
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


def plan_scope_runs(config: ScopeGridConfig) -> list[ScopeRunPlan]:
    schedule = build_selection_schedule(config.selection_frequency)
    plans: list[ScopeRunPlan] = []
    for scope in config.selection_scopes:
        selection_plan = _build_selection_plan(config, scope)
        # One representative origin sequence is enough to resolve event count for
        # the manifest; the real run resolves against actual origins.
        n_events = _estimate_selection_events(schedule, config)
        output_dir = config.output_root / f"scope_{scope}"
        existing_policy = _resolve_existing_policy(
            output_dir,
            if_exists_policy=config.if_exists_policy,
            save_all_cell_forecasts=config.save_all_cell_forecasts,
        )
        plans.append(
            ScopeRunPlan(
                scope=scope,
                output_dir=output_dir,
                selection_plan=selection_plan,
                n_selection_events=n_events,
                n_target_cells=len(selection_plan.cells),
                existing_policy=existing_policy,
            )
        )
    return plans


def _estimate_selection_events(schedule: SelectionSchedule, config: ScopeGridConfig) -> int:
    # Estimate against a nominal outer-origin count for the manifest only.
    nominal_origins = list(range(max(config.optimization_n_eval, 12)))
    try:
        return len(schedule.resolve(nominal_origins))
    except ScheduleError:
        return 1


def build_manifest(config: ScopeGridConfig, plans: Sequence[ScopeRunPlan]) -> dict[str, Any]:
    return {
        "runner": RUNNER_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command_line": config.command_line,
        "forecast_variables": list(config.forecast_variables),
        "target_variables": list(config.target_variables),
        "target_horizons": list(config.target_horizons),
        "selection_scopes": list(config.selection_scopes),
        "selection_frequency": config.selection_frequency,
        "loss_metric": config.loss_metric,
        "loss_scaling": config.loss_scaling,
        "inner_validation": {
            "window": config.inner_window,
            "n_origins": config.inner_n_origins,
            "origin_stride": config.inner_origin_stride,
            "origin_selection": config.inner_origin_selection,
            "random_seed": config.inner_random_seed,
            "rolling_window_length": config.rolling_window_length,
        },
        "optimization": {
            "horizon_quarters": config.optimization_horizon_quarters,
            "eval_horizon_quarters": config.optimization_eval_horizon_quarters,
            "n_eval": config.optimization_n_eval,
        },
        "base_seed": config.base_seed,
        "save_all_cell_forecasts": config.save_all_cell_forecasts,
        "if_exists_policy": config.if_exists_policy,
        "dry_run": config.dry_run,
        "warnings": list(config.warnings),
        "reproducibility_limitations": [
            "MBFVAR posterior-draw randomness is not fully seed-controllable at the "
            "pinned revision; per-cell selections record seed_uncontrolled=True."
        ],
        "scopes": [
            {
                "scope": plan.scope,
                "output_dir": str(plan.output_dir),
                "n_selection_events": plan.n_selection_events,
                "n_target_cells": plan.n_target_cells,
                "existing_policy": plan.existing_policy,
                "selection_plan": plan.selection_plan.to_dict(),
            }
            for plan in plans
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run mixed-frequency MF-BVAR forecast-loss experiments across selection scopes."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--panel-path", type=Path, default=REALTIME_PANEL_PATH)
    parser.add_argument(
        "--forecast-variables",
        type=str,
        default=",".join(FULL_QUARTERLY_BLOCK),
        help="Full quarterly block used to build the mixed-frequency state forecast.",
    )
    parser.add_argument("--selection-scopes", type=str, required=True)
    parser.add_argument("--target-variables", type=str, default=None)
    parser.add_argument(
        "--target-horizons",
        type=str,
        default=",".join(str(v) for v in DEFAULT_TARGET_HORIZONS),
    )
    parser.add_argument("--variable-groups", type=str, default=None)
    parser.add_argument("--residual-group-name", type=str, default=None)
    parser.add_argument("--group-separate-horizons", action="store_true")
    parser.add_argument("--loss-metric", choices=("rmse", "mse", "mae"), default="rmse")
    parser.add_argument(
        "--loss-scaling", choices=("none", "benchmark_rmse"), default="benchmark_rmse"
    )
    parser.add_argument("--inner-window", choices=("expanding", "rolling"), default="expanding")
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
        default="once",
        help="Use 'once', 'per_origin', 'annual_quarterly', or an integer N for every N origins.",
    )
    parser.add_argument("--optimization-horizon-quarters", type=int, default=8)
    parser.add_argument("--optimization-eval-horizon-quarters", type=int, default=None)
    parser.add_argument("--optimization-n-eval", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=None)
    parser.add_argument("--save-all-cell-forecasts", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    existing_group = parser.add_mutually_exclusive_group()
    existing_group.add_argument("--resume", action="store_true")
    existing_group.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = build_study_config(args, argv=argv)
    plans = plan_scope_runs(config)
    manifest = build_manifest(config, plans)

    if config.dry_run:
        # Dry-run emits the manifest to stdout without loading data or MBFVAR.
        print(json.dumps(manifest, indent=2, default=str))
        return 0

    config.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = config.output_root / "scope_grid_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    # A real run requires data + MBFVAR; those heavy imports stay lazy so the
    # dry-run and manifest paths never need them.
    from paper_hyperparameter_optimization.scope_grid_execution import execute_scope_runs

    execute_scope_runs(config, plans)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
