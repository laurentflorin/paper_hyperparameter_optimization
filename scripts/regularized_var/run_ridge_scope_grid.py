"""Run deterministic ridge VAR forecast-loss studies across selection scopes.

This is the regularized-VAR counterpart to ``scripts/glp/run_glp_scope_grid.py``
and ``scripts/mfvar/run_mfvar_scope_grid.py``. It reuses the shared
``common_hpo`` contracts and the NumPy-only ridge experiment in
:mod:`regularized_var.experiment`. The scientific search is a deterministic
grid, so runs are reproducible without Mango or any Bayesian optimizer.

The runner supports:

* a ``--dry-run`` manifest with grid-size and fit-count estimation;
* configurable target variables and horizons;
* a CSV panel adapter (data source);
* the preprocessing mode (``none`` or ``standardize``);
* an explicit inner-validation scheme and selection schedule;
* a parallel worker count (recorded; the default engine runs serially);
* resume/overwrite safeguards; and
* canonical output files (``forecast_panel.csv``, ``selected_hyperparameters.csv``,
  ``run_metadata.json``, ``failed_origins.csv``) matching GLP and MF-BVAR.

Heavy work stays NumPy-only, so ``--dry-run`` needs no data at all.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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

from common_hpo import LossConfig, ScaleConfig, SelectionSchedule, ValidationScheme  # noqa: E402
from common_hpo import (  # noqa: E402
    CSVSchema,
    JSONSchema,
    build_run_metadata,
    build_selection_plan,
    classify_failure,
    mark_run_cancelled,
    mark_run_complete,
    mark_run_failed,
    prepare_run_directory,
    resolve_run_directory_policy,
    utc_now,
)
from common_hpo.schedules import ScheduleError  # noqa: E402
from regularized_var.experiment import (  # noqa: E402
    BENCHMARK_STRATEGIES,
    FORECAST_PANEL_COLUMNS,
    RidgeExperimentConfig,
    estimate_fit_counts,
)
from regularized_var.tuning import RidgeGridSpec, default_grid_spec, grid_size  # noqa: E402


RUNNER_NAME = "regularized_var_scope_grid"
SUPPORTED_SELECTION_SCOPES = ("pooled", "horizon", "variable", "variable_horizon")
EXPECTED_OUTPUT_FILES = (
    "forecast_panel.csv",
    "selected_hyperparameters.csv",
    "run_metadata.json",
    "failed_origins.csv",
)
DEFAULT_TARGET_HORIZONS = (1, 2, 4, 8)
_SELECTION_COLUMNS = (
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
_BENCHMARK_SELECTION_COLUMNS = (
    "forecast_origin",
    "strategy",
    "group",
    "parameter",
    "value",
)
_FAILED_SCOPE_COLUMNS = ("forecast_origin", "cell_id", "stage", "failure_category", "error")
_FAILED_BENCHMARK_COLUMNS = ("forecast_origin", "stage", "failure_category", "error")


class ScopeGridConfigError(ValueError):
    """Raised when the requested scope-grid study configuration is invalid."""


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def _parse_str_list(raw: str | None, *, label: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    items = tuple(token.strip() for token in raw.split(",") if token.strip())
    if not items:
        raise ScopeGridConfigError(f"{label} must contain at least one entry.")
    return items


def _parse_int_list(raw: str | None, *, label: str) -> tuple[int, ...]:
    if raw is None:
        return ()
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(int(token))
        except ValueError as exc:
            raise ScopeGridConfigError(f"{label} entry {token!r} is not an integer.") from exc
    if not values:
        raise ScopeGridConfigError(f"{label} must contain at least one integer.")
    return tuple(values)


def _parse_float_list(raw: str | None, *, label: str) -> tuple[float, ...]:
    if raw is None:
        return ()
    values: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError as exc:
            raise ScopeGridConfigError(f"{label} entry {token!r} is not a number.") from exc
    if not values:
        raise ScopeGridConfigError(f"{label} must contain at least one number.")
    return tuple(values)


def build_selection_schedule(raw: str) -> SelectionSchedule:
    text = str(raw).strip().lower()
    if text in ("once", "select_once"):
        return SelectionSchedule.once()
    if text in ("per_origin", "every_origin"):
        return SelectionSchedule.every_origin()
    if text in ("annual_quarterly", "annual"):
        return SelectionSchedule.annual_quarterly()
    try:
        n = int(text)
    except ValueError as exc:
        raise ScopeGridConfigError(
            f"unrecognized selection frequency {raw!r}; use 'once', 'per_origin', "
            "'annual_quarterly', or an integer."
        ) from exc
    return SelectionSchedule.every_n_origins(n)


def build_grid_spec(args: argparse.Namespace) -> RidgeGridSpec:
    default = default_grid_spec()
    lambdas = _parse_float_list(args.grid_lambdas, label="grid-lambdas") or default.lambdas
    lag_orders = _parse_int_list(args.grid_lag_orders, label="grid-lag-orders") or default.lag_orders
    alphas = _parse_float_list(args.grid_alphas, label="grid-alphas") or default.alphas
    kappas = _parse_float_list(args.grid_kappas, label="grid-kappas") or default.kappas
    return RidgeGridSpec(lambdas=lambdas, lag_orders=lag_orders, alphas=alphas, kappas=kappas)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScopeGridConfig:
    output_root: Path
    panel_path: Path | None
    target_variables: tuple[str, ...]
    target_horizons: tuple[int, ...]
    selection_scopes: tuple[str, ...]
    experiment: RidgeExperimentConfig
    benchmarks: tuple[str, ...]
    n_workers: int
    if_exists_policy: str
    dry_run: bool
    command_line: str
    warnings: tuple[str, ...]


def build_study_config(
    args: argparse.Namespace,
    argv: Sequence[str] | None = None,
    program: str = "scripts/regularized_var/run_ridge_scope_grid.py",
) -> ScopeGridConfig:
    selection_scopes = _parse_str_list(args.selection_scopes, label="selection-scopes")
    unknown = [s for s in selection_scopes if s not in SUPPORTED_SELECTION_SCOPES]
    if unknown:
        raise ScopeGridConfigError(
            f"unsupported selection scopes {unknown}; expected any of "
            f"{SUPPORTED_SELECTION_SCOPES}."
        )

    target_variables = _parse_str_list(args.target_variables, label="target-variables")
    if not target_variables:
        raise ScopeGridConfigError("--target-variables is required.")
    target_horizons = _parse_int_list(args.target_horizons, label="target-horizons")
    if not target_horizons:
        target_horizons = DEFAULT_TARGET_HORIZONS

    if args.preprocessing not in ("none", "standardize"):
        raise ScopeGridConfigError("--preprocessing must be 'none' or 'standardize'.")
    if args.forecast_method not in ("iterated", "direct"):
        raise ScopeGridConfigError("--forecast-method must be 'iterated' or 'direct'.")

    schedule = build_selection_schedule(args.selection_frequency)
    grid_spec = build_grid_spec(args)

    if args.inner_window == "rolling" and args.rolling_window_length is None:
        raise ScopeGridConfigError("--rolling-window-length is required for a rolling window.")

    outer_scheme = ValidationScheme(
        training_window=args.inner_window,
        origin_selection=_map_origin_selection(args.outer_origin_selection),
        n_origins=int(args.outer_n_origins),
        horizons=target_horizons,
        min_train_length=int(args.min_train_length),
        origin_stride=int(args.outer_origin_stride),
        rolling_window_length=args.rolling_window_length,
        random_seed=args.inner_random_seed,
    )
    inner_scheme = ValidationScheme(
        training_window=args.inner_window,
        origin_selection=_map_origin_selection(args.inner_origin_selection),
        n_origins=int(args.inner_n_origins),
        horizons=target_horizons,
        min_train_length=int(args.min_train_length),
        origin_stride=int(args.inner_origin_stride),
        rolling_window_length=args.rolling_window_length,
        random_seed=args.inner_random_seed,
    )

    scale = ScaleConfig(method="none") if args.loss_scaling == "none" else ScaleConfig(method=args.loss_scaling)
    loss_config = LossConfig(aggregation=args.loss_metric, scale=scale)

    experiment = RidgeExperimentConfig(
        target_variables=target_variables,
        target_horizons=target_horizons,
        grid_spec=grid_spec,
        outer_scheme=outer_scheme,
        inner_scheme=inner_scheme,
        selection_schedule=schedule,
        loss_config=loss_config,
        preprocessing=args.preprocessing,
        forecast_method=args.forecast_method,
        horizon_row_offset=int(args.horizon_row_offset),
        benchmark_lag_orders=_parse_int_list(args.benchmark_lag_orders, label="benchmark-lag-orders")
        or (1, 2, 4),
        base_seed=args.base_seed,
    )

    benchmarks = _parse_str_list(args.benchmarks, label="benchmarks") if args.benchmarks else ()
    unknown_bench = [b for b in benchmarks if b not in BENCHMARK_STRATEGIES]
    if unknown_bench:
        raise ScopeGridConfigError(
            f"unknown benchmarks {unknown_bench}; expected any of {BENCHMARK_STRATEGIES}."
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
        output_root=Path(args.output_root),
        panel_path=Path(args.panel_path) if args.panel_path else None,
        target_variables=target_variables,
        target_horizons=target_horizons,
        selection_scopes=selection_scopes,
        experiment=experiment,
        benchmarks=benchmarks,
        n_workers=int(args.n_workers),
        if_exists_policy=if_exists_policy,
        dry_run=bool(args.dry_run),
        command_line=command_line,
        warnings=tuple(warnings),
    )


def _map_origin_selection(value: str) -> str:
    mapping = {
        "recent": "most_recent",
        "most_recent": "most_recent",
        "evenly_spaced": "evenly_spaced",
        "random": "random",
    }
    if value not in mapping:
        raise ScopeGridConfigError(f"unknown origin selection {value!r}.")
    return mapping[value]


# --------------------------------------------------------------------------- #
# Manifest / resume
# --------------------------------------------------------------------------- #
def _classify_existing_directory(output_dir: Path) -> str:
    if not output_dir.exists():
        return "missing"
    if not output_dir.is_dir():
        raise FileExistsError(f"run directory path exists but is not a directory: {output_dir}")
    if not list(output_dir.iterdir()):
        return "empty"
    complete = all((output_dir / name).exists() for name in EXPECTED_OUTPUT_FILES)
    return "complete" if complete else "partial"


def _resolve_existing_policy(output_dir: Path, *, if_exists_policy: str) -> str:
    state = _classify_existing_directory(output_dir)
    if if_exists_policy == "overwrite":
        return "planned"
    if if_exists_policy == "resume":
        if state == "complete":
            return "resume_skip"
        if state in {"missing", "empty"}:
            return "planned"
        raise FileExistsError(
            f"run directory {output_dir} contains partial outputs; refusing an "
            "ambiguous resume. Use --overwrite to replace it."
        )
    if state in {"missing", "empty"}:
        return "planned"
    raise FileExistsError(
        f"run directory {output_dir} already exists and is not empty. "
        "Use --resume or --overwrite explicitly."
    )


def _scope_selection_plan(config: ScopeGridConfig, scope: str) -> dict[str, object]:
    return build_selection_plan(scope, config.target_variables, config.target_horizons).to_dict()


def _scope_run_metadata(
    config: ScopeGridConfig,
    *,
    scope: str,
    output_dir: Path,
    started_utc: str,
    finished_utc: str | None,
    completion_status: str,
    failure_records: Sequence[Mapping[str, object]] = (),
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return build_run_metadata(
        project_root=PROJECT_ROOT,
        command_line=config.command_line,
        started_utc=started_utc,
        finished_utc=finished_utc,
        completion_status=completion_status,
        model_family="ridge",
        model_version=RUNNER_NAME,
        data_source={"panel_path": str(config.panel_path) if config.panel_path else None, "format": "csv"},
        data_vintage_identifiers={"policy": "non_vintage_csv"},
        input_files=(() if config.panel_path is None else (config.panel_path,)),
        transformation_configuration={
            "preprocessing": config.experiment.preprocessing,
            "forecast_method": config.experiment.forecast_method,
            "horizon_row_offset": config.experiment.horizon_row_offset,
        },
        variable_order=config.target_variables,
        target_variables=config.target_variables,
        target_horizons=config.target_horizons,
        selection_plan=_scope_selection_plan(config, scope),
        validation_scheme={
            "inner": config.experiment.inner_scheme.to_dict(),
            "outer": config.experiment.outer_scheme.to_dict(),
        },
        vintage_policy={
            "inner": getattr(config.experiment.inner_scheme.vintage_policy, "value", None),
            "outer": getattr(config.experiment.outer_scheme.vintage_policy, "value", None),
        },
        selection_schedule=config.experiment.selection_schedule.to_dict(),
        loss_configuration=config.experiment.loss_config.to_dict(),
        search_space=config.experiment.grid_spec.to_dict(),
        optimizer_budget={
            "search_strategy": "deterministic_grid",
            "grid_size": grid_size(config.experiment.grid_spec),
            "benchmark_lag_orders": list(config.experiment.benchmark_lag_orders),
        },
        random_seeds={
            "base_seed": config.experiment.base_seed,
            "inner_random_seed": config.experiment.inner_scheme.random_seed,
        },
        parallel_worker_count=config.n_workers,
        failure_records=failure_records,
        configuration_extra={"scope": scope, "output_dir": str(output_dir)},
        extra=extra,
    )


def _benchmark_run_metadata(
    config: ScopeGridConfig,
    *,
    benchmark: str,
    output_dir: Path,
    started_utc: str,
    finished_utc: str | None,
    completion_status: str,
    failure_records: Sequence[Mapping[str, object]] = (),
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return build_run_metadata(
        project_root=PROJECT_ROOT,
        command_line=config.command_line,
        started_utc=started_utc,
        finished_utc=finished_utc,
        completion_status=completion_status,
        model_family="ridge_benchmark",
        model_version=benchmark,
        data_source={"panel_path": str(config.panel_path) if config.panel_path else None, "format": "csv"},
        data_vintage_identifiers={"policy": "non_vintage_csv"},
        input_files=(() if config.panel_path is None else (config.panel_path,)),
        transformation_configuration={
            "preprocessing": config.experiment.preprocessing,
            "forecast_method": config.experiment.forecast_method,
            "horizon_row_offset": config.experiment.horizon_row_offset,
        },
        variable_order=config.target_variables,
        target_variables=config.target_variables,
        target_horizons=config.target_horizons,
        selection_plan={"scope": "benchmark", "strategy": benchmark},
        validation_scheme={"outer": config.experiment.outer_scheme.to_dict()},
        vintage_policy={
            "outer": getattr(config.experiment.outer_scheme.vintage_policy, "value", None),
        },
        selection_schedule="benchmark_per_origin",
        loss_configuration=config.experiment.loss_config.to_dict(),
        search_space={"benchmark_lag_orders": list(config.experiment.benchmark_lag_orders)},
        optimizer_budget={"search_strategy": "benchmark_rule"},
        random_seeds={"base_seed": config.experiment.base_seed},
        parallel_worker_count=config.n_workers,
        failure_records=failure_records,
        configuration_extra={"benchmark": benchmark, "output_dir": str(output_dir)},
        extra=extra,
    )


def build_manifest(config: ScopeGridConfig) -> dict[str, Any]:
    scopes: list[dict[str, Any]] = []
    total_fits = 0
    for scope in config.selection_scopes:
        counts = estimate_fit_counts(
            config.experiment.outer_scheme.min_train_length + config.experiment.outer_scheme.n_origins,
            scope,
            config.experiment,
        )
        total_fits += counts["total_fits"]
        output_dir = config.output_root / f"scope_{scope}"
        started_utc = utc_now()
        manifest_metadata = _scope_run_metadata(
            config,
            scope=scope,
            output_dir=output_dir,
            started_utc=started_utc,
            finished_utc=None,
            completion_status="partial",
        )
        existing_policy = resolve_run_directory_policy(
            output_dir,
            if_exists_policy=config.if_exists_policy,
            configuration_hash=str(manifest_metadata["configuration_hash"]),
        )
        scopes.append(
            {
                "scope": scope,
                "output_dir": str(output_dir),
                "existing_policy": existing_policy,
                "configuration_hash": manifest_metadata["configuration_hash"],
                "fit_counts": counts,
            }
        )

    return {
        "runner": RUNNER_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command_line": config.command_line,
        "panel_path": str(config.panel_path) if config.panel_path else None,
        "target_variables": list(config.target_variables),
        "target_horizons": list(config.target_horizons),
        "selection_scopes": list(config.selection_scopes),
        "benchmarks": list(config.benchmarks),
        "preprocessing": config.experiment.preprocessing,
        "forecast_method": config.experiment.forecast_method,
        "grid_spec": config.experiment.grid_spec.to_dict(),
        "grid_size": grid_size(config.experiment.grid_spec),
        "selection_schedule": config.experiment.selection_schedule.to_dict(),
        "inner_validation_scheme": config.experiment.inner_scheme.to_dict(),
        "outer_validation_scheme": config.experiment.outer_scheme.to_dict(),
        "loss_config": config.experiment.loss_config.to_dict(),
        "n_workers": config.n_workers,
        "if_exists_policy": config.if_exists_policy,
        "dry_run": config.dry_run,
        "warnings": list(config.warnings),
        "estimated_total_fits": total_fits,
        "reproducibility": "deterministic grid search; no Mango / Bayesian optimizer required.",
        "scopes": scopes,
    }


def _scope_output_schemas() -> tuple[tuple[CSVSchema, ...], tuple[JSONSchema, ...]]:
    return (
        (
            CSVSchema("forecast_panel.csv", tuple(FORECAST_PANEL_COLUMNS), min_rows=1),
            CSVSchema("selected_hyperparameters.csv", _SELECTION_COLUMNS, min_rows=1),
            CSVSchema("failed_origins.csv", _FAILED_SCOPE_COLUMNS, min_rows=0),
        ),
        (
            JSONSchema(
                "run_metadata.json",
                (
                    "configuration_hash",
                    "completion_status",
                    "repository_commit",
                    "target_variables",
                    "target_horizons",
                ),
            ),
        ),
    )


def _benchmark_output_schemas() -> tuple[tuple[CSVSchema, ...], tuple[JSONSchema, ...]]:
    return (
        (
            CSVSchema("forecast_panel.csv", tuple(FORECAST_PANEL_COLUMNS), min_rows=1),
            CSVSchema("selected_hyperparameters.csv", _BENCHMARK_SELECTION_COLUMNS, min_rows=0),
            CSVSchema("failed_origins.csv", _FAILED_BENCHMARK_COLUMNS, min_rows=0),
        ),
        (
            JSONSchema(
                "run_metadata.json",
                (
                    "configuration_hash",
                    "completion_status",
                    "repository_commit",
                    "target_variables",
                    "target_horizons",
                ),
            ),
        ),
    )


# --------------------------------------------------------------------------- #
# Parser & main
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic ridge VAR forecast-loss studies across selection scopes."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--panel-path", type=Path, default=None)
    parser.add_argument("--target-variables", type=str, required=True)
    parser.add_argument("--target-horizons", type=str, default="1,2,4,8")
    parser.add_argument("--selection-scopes", type=str, required=True)
    parser.add_argument("--preprocessing", choices=("none", "standardize"), default="standardize")
    parser.add_argument("--forecast-method", choices=("iterated", "direct"), default="iterated")
    parser.add_argument("--horizon-row-offset", type=int, default=1)
    # grid
    parser.add_argument("--grid-lambdas", type=str, default=None)
    parser.add_argument("--grid-lag-orders", type=str, default=None)
    parser.add_argument("--grid-alphas", type=str, default=None)
    parser.add_argument("--grid-kappas", type=str, default=None)
    # loss
    parser.add_argument("--loss-metric", choices=("rmse", "mse", "mae"), default="rmse")
    parser.add_argument("--loss-scaling", choices=("none", "target_std"), default="none")
    # validation
    parser.add_argument("--inner-window", choices=("expanding", "rolling"), default="expanding")
    parser.add_argument("--rolling-window-length", type=int, default=None)
    parser.add_argument("--min-train-length", type=int, default=40)
    parser.add_argument("--outer-n-origins", type=int, default=8)
    parser.add_argument("--outer-origin-stride", type=int, default=1)
    parser.add_argument("--outer-origin-selection", choices=("recent", "evenly_spaced", "random"), default="recent")
    parser.add_argument("--inner-n-origins", type=int, default=4)
    parser.add_argument("--inner-origin-stride", type=int, default=1)
    parser.add_argument("--inner-origin-selection", choices=("recent", "evenly_spaced", "random"), default="recent")
    parser.add_argument("--inner-random-seed", type=int, default=None)
    parser.add_argument("--selection-frequency", type=str, default="once")
    # benchmarks
    parser.add_argument("--benchmarks", type=str, default=None,
                        help="comma-separated: var_aic,var_bic,var_nested_loss,ar_univariate,no_change")
    parser.add_argument("--benchmark-lag-orders", type=str, default=None)
    # execution
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = build_study_config(args, argv=argv)
        manifest = build_manifest(config)
    except (ScopeGridConfigError, ScheduleError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if config.dry_run:
        print(json.dumps(manifest, indent=2, default=str))
        return 0

    if config.panel_path is None:
        print("error: --panel-path is required for a real run.", file=sys.stderr)
        return 2

    # Lazy import keeps --dry-run free of any data/IO dependency.
    from regularized_var.data import load_panel_csv
    from regularized_var.experiment import (
        run_benchmark,
        run_scope_experiment,
        write_benchmark_outputs,
        write_scope_outputs,
    )

    panel = load_panel_csv(config.panel_path, variables=config.target_variables)

    try:
        for scope in config.selection_scopes:
            output_dir = config.output_root / f"scope_{scope}"
            started_utc = utc_now()
            initial_metadata = _scope_run_metadata(
                config,
                scope=scope,
                output_dir=output_dir,
                started_utc=started_utc,
                finished_utc=None,
                completion_status="partial",
            )
            action = prepare_run_directory(
                output_dir,
                manifest=initial_metadata,
                if_exists_policy=config.if_exists_policy,
                expected_outputs=EXPECTED_OUTPUT_FILES,
            )
            if action == "resume_skip":
                print(f"[resume] skipping complete scope {scope} -> {output_dir}")
                continue
            try:
                result = run_scope_experiment(panel, scope, config.experiment)
                final_metadata = _scope_run_metadata(
                    config,
                    scope=scope,
                    output_dir=output_dir,
                    started_utc=started_utc,
                    finished_utc=utc_now(),
                    completion_status="complete",
                    failure_records=result.failed_origins,
                    extra={
                        "runner": RUNNER_NAME,
                        "scope": scope,
                        "panel": result.metadata.get("panel"),
                        "n_outer_origins": result.metadata.get("n_outer_origins"),
                        "n_selection_events": result.metadata.get("n_selection_events"),
                        "n_target_cells": result.metadata.get("n_target_cells"),
                        "preprocessing": result.metadata.get("preprocessing"),
                    },
                )
                write_scope_outputs(result, output_dir, metadata_override=final_metadata)
                csv_schemas, json_schemas = _scope_output_schemas()
                mark_run_complete(
                    output_dir,
                    configuration_hash=str(final_metadata["configuration_hash"]),
                    csv_schemas=csv_schemas,
                    json_schemas=json_schemas,
                )
                print(f"[done] scope {scope}: {len(result.forecast_rows)} forecast rows -> {output_dir}")
            except KeyboardInterrupt:
                cancelled_metadata = _scope_run_metadata(
                    config,
                    scope=scope,
                    output_dir=output_dir,
                    started_utc=started_utc,
                    finished_utc=utc_now(),
                    completion_status="cancelled",
                    failure_records=({"stage": "run", "error": "KeyboardInterrupt", "failure_category": "cancelled"},),
                )
                mark_run_cancelled(
                    output_dir,
                    configuration_hash=str(initial_metadata["configuration_hash"]),
                    metadata=cancelled_metadata,
                )
                raise
            except Exception as exc:  # noqa: BLE001
                failed_metadata = _scope_run_metadata(
                    config,
                    scope=scope,
                    output_dir=output_dir,
                    started_utc=started_utc,
                    finished_utc=utc_now(),
                    completion_status="failed",
                    failure_records=({"stage": "run", "error": f"{type(exc).__name__}: {exc}", "failure_category": classify_failure(exc)},),
                )
                mark_run_failed(
                    output_dir,
                    configuration_hash=str(initial_metadata["configuration_hash"]),
                    reason=f"{type(exc).__name__}: {exc}",
                    metadata=failed_metadata,
                )
                raise

        for benchmark in config.benchmarks:
            output_dir = config.output_root / "benchmarks" / benchmark
            started_utc = utc_now()
            initial_metadata = _benchmark_run_metadata(
                config,
                benchmark=benchmark,
                output_dir=output_dir,
                started_utc=started_utc,
                finished_utc=None,
                completion_status="partial",
            )
            action = prepare_run_directory(
                output_dir,
                manifest=initial_metadata,
                if_exists_policy=config.if_exists_policy,
                expected_outputs=EXPECTED_OUTPUT_FILES,
            )
            if action == "resume_skip":
                print(f"[resume] skipping complete benchmark {benchmark} -> {output_dir}")
                continue
            try:
                result = run_benchmark(panel, benchmark, config.experiment)
                final_metadata = _benchmark_run_metadata(
                    config,
                    benchmark=benchmark,
                    output_dir=output_dir,
                    started_utc=started_utc,
                    finished_utc=utc_now(),
                    completion_status="complete",
                    failure_records=result.failed_origins,
                    extra={
                        "runner": RUNNER_NAME,
                        "strategy": benchmark,
                        "panel": result.metadata.get("panel"),
                        "n_outer_origins": result.metadata.get("n_outer_origins"),
                    },
                )
                write_benchmark_outputs(result, output_dir, metadata_override=final_metadata)
                csv_schemas, json_schemas = _benchmark_output_schemas()
                mark_run_complete(
                    output_dir,
                    configuration_hash=str(final_metadata["configuration_hash"]),
                    csv_schemas=csv_schemas,
                    json_schemas=json_schemas,
                )
                print(f"[done] benchmark {benchmark}: {len(result.forecast_rows)} forecast rows -> {output_dir}")
            except KeyboardInterrupt:
                cancelled_metadata = _benchmark_run_metadata(
                    config,
                    benchmark=benchmark,
                    output_dir=output_dir,
                    started_utc=started_utc,
                    finished_utc=utc_now(),
                    completion_status="cancelled",
                    failure_records=({"stage": "run", "error": "KeyboardInterrupt", "failure_category": "cancelled"},),
                )
                mark_run_cancelled(
                    output_dir,
                    configuration_hash=str(initial_metadata["configuration_hash"]),
                    metadata=cancelled_metadata,
                )
                raise
            except Exception as exc:  # noqa: BLE001
                failed_metadata = _benchmark_run_metadata(
                    config,
                    benchmark=benchmark,
                    output_dir=output_dir,
                    started_utc=started_utc,
                    finished_utc=utc_now(),
                    completion_status="failed",
                    failure_records=({"stage": "run", "error": f"{type(exc).__name__}: {exc}", "failure_category": classify_failure(exc)},),
                )
                mark_run_failed(
                    output_dir,
                    configuration_hash=str(initial_metadata["configuration_hash"]),
                    reason=f"{type(exc).__name__}: {exc}",
                    metadata=failed_metadata,
                )
                raise
    except KeyboardInterrupt:
        print("error: run cancelled by user.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    from common_hpo.io import atomic_write_json

    atomic_write_json(config.output_root / "scope_grid_manifest.json", manifest)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
