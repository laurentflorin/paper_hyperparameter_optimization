"""Run the GLP forecast strategies from one command.

By default this runs the four forecast scripts

* ``run_glp_paper.py``
* ``run_glp_mango.py``
* ``run_glp_mango_rmse.py``
* ``run_glp_mango_rmse_random.py``

with one shared CLI surface. The optional ``compare`` stage reproduces
``compare_glp_forecasts.py`` after the forecast runs finish.

Two further stages are available:

* ``paper_plugin`` -- the GLP marginal-likelihood mode with the hyperparameters
  held fixed (no integration over them); isolates the effect of integrating out
  the hyperparameters in ``paper``.
* ``ladder`` -- one run per rung in ``--ladder-rungs`` (see
  :mod:`glp_hyperparameter_optimization.objectives`); each rung writes to
  ``<output-root>/ladder/<rung label>``. Rung syntax: ``OBJ[:hH][:wW|:wall|:rN[:pP][:sS]]``
  with ``OBJ`` in ``mdd, lsj1, lsj, lsm, rmse, rmsej`` (e.g. ``lsj1:w12,lsm:h4:w12,rmse:h4:wall``).
  ``rN`` scores ``N`` origins drawn at random from the ``P`` most recent (default: all)
  instead of the last ``N``; ``sS`` picks the random subset (e.g. ``rmse:h4:r12:p80:s1``).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from glp_hyperparameter_optimization.config import (  # noqa: E402
    EVAL_HORIZONS_QUARTERS,
    GLP_ACTUAL_VINTAGE,
    GLP_LAGS,
    GLP_MCMC_CONST,
    GLP_REALTIME_PANEL_PATH,
    MAX_FORECAST_HORIZON_QUARTERS,
    MODEL_SIZES,
    resolve_project_path,
)
from glp_hyperparameter_optimization.forecasting import (  # noqa: E402
    DEFAULT_MCMC_DISCARD,
    DEFAULT_MCMC_DRAWS,
    add_selection_arguments,
    parse_csv_int_list,
    parse_csv_list,
    run_glp_experiment,
)
from glp_hyperparameter_optimization.reporting import create_glp_comparison_report  # noqa: E402

VALID_STAGES = ("paper", "mango_mdd", "mango_rmse", "mango_rmse_random", "paper_plugin", "ladder", "compare")
DEFAULT_STAGES = ("paper", "mango_mdd", "mango_rmse", "mango_rmse_random")
LADDER_SHORT_NAMES = {
    "mdd": "mdd",
    "lsj1": "logscore_joint_1",
    "lsj": "logscore_joint",
    "lsm": "logscore_marginal",
    "rmse": "rmse",
    "rmsej": "rmse_joint",
}


def _log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{timestamp}] {message}", flush=True)


def parse_stage_list(value: str | None) -> list[str]:
    if not value:
        return list(DEFAULT_STAGES)
    parsed: list[str] = []
    for item in value.split(","):
        stage = item.strip()
        if not stage:
            continue
        if stage not in VALID_STAGES:
            raise ValueError(f"Unknown stage {stage!r}. Choose from {', '.join(VALID_STAGES)}.")
        if stage not in parsed:
            parsed.append(stage)
    if not parsed:
        raise ValueError("stages resolved to an empty set.")
    return parsed


def validate_eval_horizons(horizons: list[int]) -> list[int]:
    unique: list[int] = []
    for horizon in horizons:
        if horizon < 1 or horizon > MAX_FORECAST_HORIZON_QUARTERS:
            raise ValueError(f"eval horizon must be within 1..{MAX_FORECAST_HORIZON_QUARTERS}, got {horizon}.")
        if horizon not in unique:
            unique.append(horizon)
    return unique


def parse_ladder_rungs(value: str | None) -> list[dict[str, object]]:
    """Parse ``OBJ[:hH][:wW|:wall]`` rung specs into ladder dicts (see module doc)."""
    rungs: list[dict[str, object]] = []
    for item in (value or "").split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if parts[0] not in LADDER_SHORT_NAMES:
            raise ValueError(f"Unknown ladder objective {parts[0]!r}; choose from {', '.join(LADDER_SHORT_NAMES)}.")
        rung: dict[str, object] = {"objective": LADDER_SHORT_NAMES[parts[0]], "horizon": 1, "window": 12}
        for part in parts[1:]:
            if part.startswith("h"):
                rung["horizon"] = validate_eval_horizons([int(part[1:])])[0]
            elif part == "wall":
                rung["window"] = None
            elif part.startswith("w"):
                rung["window"] = int(part[1:])
            elif part.startswith("r"):
                rung["window"] = int(part[1:])
                rung["random_origins"] = True
            elif part.startswith("p"):
                rung["pool"] = int(part[1:])
            elif part.startswith("s"):
                rung["subset_seed"] = int(part[1:])
            else:
                raise ValueError(f"Cannot parse ladder rung component {part!r} in {item!r}.")
        if ("pool" in rung or "subset_seed" in rung) and not rung.get("random_origins"):
            raise ValueError(f"Pool (pP) and subset seed (sS) need random origins (rN) in {item!r}.")
        rungs.append(rung)
    return rungs


def ladder_label(rung: dict[str, object]) -> str:
    from glp_hyperparameter_optimization.objectives import LadderSpec

    return LadderSpec(
        str(rung["objective"]),
        int(rung["horizon"]),  # type: ignore[arg-type]
        rung["window"],  # type: ignore[arg-type]
        random_origins=bool(rung.get("random_origins", False)),
        pool=rung.get("pool"),  # type: ignore[arg-type]
        subset_seed=int(rung.get("subset_seed", 0)),  # type: ignore[arg-type]
    ).label()


def resolve_stage_output_dir(output_root: Path, explicit: Path | None, leaf: str) -> Path:
    path = explicit if explicit is not None else output_root / leaf
    return resolve_project_path(path)


def horizon_output_dir(base: Path, horizon: int, multiple: bool) -> Path:
    return base if not multiple else base / f"h{horizon}q"


def write_batch_metadata(base_dir: Path, manifest: list[dict[str, object]]) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "batch_metadata.json").write_text(json.dumps({"runs": manifest}, indent=2), encoding="utf-8")


def _existing(path: Path | None) -> Path | None:
    if path is None:
        return None
    resolved = resolve_project_path(path)
    return resolved if resolved.exists() else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the GLP forecast scripts from one command.")
    parser.add_argument(
        "--stages",
        type=str,
        default=",".join(DEFAULT_STAGES),
        help=(
            "Comma-separated stages to run. Defaults to paper,mango_mdd,mango_rmse,"
            "mango_rmse_random. Add compare to run the comparison report afterward."
        ),
    )

    parser.add_argument("--output-root", type=Path, default=Path("outputs/glp"))
    parser.add_argument("--paper-dir", type=Path, default=None)
    parser.add_argument("--mango-mdd-dir", type=Path, default=None)
    parser.add_argument("--mango-rmse-dir", type=Path, default=None)
    parser.add_argument("--mango-rmse-random-dir", type=Path, default=None)
    parser.add_argument("--comparison-dir", type=Path, default=None)

    parser.add_argument("--panel-path", type=Path, default=GLP_REALTIME_PANEL_PATH)
    parser.add_argument("--model-size", choices=MODEL_SIZES, default="medium")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--actual-vintage", type=str, default=GLP_ACTUAL_VINTAGE.strftime("%Y-%m-%d"))
    parser.add_argument("--lags", type=int, default=GLP_LAGS)
    parser.add_argument("--mcmc-draws", type=int, default=DEFAULT_MCMC_DRAWS)
    parser.add_argument("--mcmc-discard", type=int, default=DEFAULT_MCMC_DISCARD)
    parser.add_argument("--mcmc-const", type=float, default=GLP_MCMC_CONST)
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Forecast-origin worker count. Defaults to the Slurm allocation when available, otherwise 1.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable the per-origin progress bar and progress logging from the underlying experiment runner.",
    )

    parser.add_argument("--hyperpriors", type=int, default=1)
    parser.add_argument("--sur", type=int, default=1)
    parser.add_argument("--noc", type=int, default=1)
    parser.add_argument("--mnpsi", type=int, default=1)
    parser.add_argument("--mnalpha", type=int, default=0)
    parser.add_argument("--vc", type=float, default=1.0e7)

    parser.add_argument("--optimization-init-points", type=int, default=5)
    parser.add_argument("--optimization-iterations", type=int, default=15)
    parser.add_argument("--optimization-njobs", type=int, default=None)
    parser.add_argument(
        "--optimization-eval-horizons-quarters",
        type=str,
        default=",".join(str(h) for h in EVAL_HORIZONS_QUARTERS),
        help="Comma-separated target horizons for the RMSE strategies; each writes to its own output subdirectory.",
    )
    parser.add_argument(
        "--optimization-n-eval",
        type=int,
        default=3,
        help="Number of rolling/random evaluation origins used inside the RMSE objectives.",
    )
    parser.add_argument(
        "--optimization-n-obj-draws",
        type=int,
        default=200,
        help=(
            "Posterior beta draws averaged into the predictive-mean RMSE objective "
            "(<=1 uses the deterministic posterior-mode point forecast)."
        ),
    )
    parser.add_argument("--optimization-min-t", type=int, default=None)
    parser.add_argument("--optimization-random-seed", type=int, default=None)
    parser.add_argument(
        "--variables",
        type=str,
        default=None,
        help="Comma-separated model variables used by the RMSE objective (e.g. GDP,DEFL). Defaults to the first variable.",
    )
    parser.add_argument(
        "--per-origin-selection",
        action="store_true",
        help="Re-select the RMSE hyperparameters at every origin instead of once on the first origin.",
    )
    add_selection_arguments(parser)
    parser.add_argument(
        "--ladder-rungs",
        type=str,
        default="mdd,lsj1:w12,lsj:h4:w12,lsm:h4:w12,rmse:h4:w12",
        help="Comma-separated ladder rungs OBJ[:hH][:wW|:wall|:rN[:pP][:sS]] for the 'ladder' stage.",
    )
    parser.add_argument("--ladder-draws", type=int, default=200, help="Posterior draws per origin in the ladder objectives.")
    parser.add_argument("--ladder-starts", type=int, default=4, help="Nelder-Mead starts per ladder selection.")
    parser.add_argument("--ladder-maxfev", type=int, default=None, help="Max objective evaluations per Nelder-Mead start.")
    return parser


def common_run_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "size": args.model_size,
        "panel_path": resolve_project_path(args.panel_path),
        "start": pd.Timestamp(args.start) if args.start else None,
        "end": pd.Timestamp(args.end) if args.end else None,
        "actual_vintage": pd.Timestamp(args.actual_vintage),
        "lags": args.lags,
        "hyperpriors": args.hyperpriors,
        "sur": args.sur,
        "noc": args.noc,
        "mnpsi": args.mnpsi,
        "mnalpha": args.mnalpha,
        "vc": args.vc,
        "mcmc_draws": args.mcmc_draws,
        "mcmc_discard": args.mcmc_discard,
        "mcmc_const": args.mcmc_const,
        "init_points": args.optimization_init_points,
        "n_iter": args.optimization_iterations,
        "optimization_njobs": args.optimization_njobs,
        "n_eval": args.optimization_n_eval,
        "n_obj_draws": args.optimization_n_obj_draws,
        "min_t": args.optimization_min_t,
        "random_seed": args.optimization_random_seed,
        "variables": parse_csv_list(args.variables, []),
        "per_origin_selection": args.per_origin_selection,
        "seed_base": args.seed_base,
        "n_workers": args.n_workers,
        "show_progress": not args.quiet,
        "optimization_seed": args.optimization_seed,
        "local_polish": args.local_polish,
        "reselect_every": args.reselect_every,
    }


def run_simple_stage(strategy: str, output_dir: Path, args: argparse.Namespace) -> Path:
    kwargs = common_run_kwargs(args)
    kwargs["strategy"] = strategy
    kwargs["output_dir"] = output_dir
    return run_glp_experiment(**kwargs)


def run_rmse_stage(strategy: str, base_output_dir: Path, args: argparse.Namespace) -> Path:
    kwargs = common_run_kwargs(args)
    kwargs["strategy"] = strategy
    horizons = validate_eval_horizons(
        parse_csv_int_list(args.optimization_eval_horizons_quarters, list(EVAL_HORIZONS_QUARTERS))
    )
    multiple = len(horizons) > 1
    manifest: list[dict[str, object]] = []
    for horizon in horizons:
        output_dir = horizon_output_dir(base_output_dir, horizon, multiple)
        _log(f"Running GLP {strategy} at horizon {horizon}q -> {output_dir}")
        run_glp_experiment(
            **kwargs,
            output_dir=output_dir,
            optimization_horizon_quarters=horizon,
            optimization_eval_horizon_quarters=horizon,
        )
        manifest.append({"horizon_quarters": horizon, "output_dir": str(output_dir)})
    if multiple:
        write_batch_metadata(base_output_dir, manifest)
    return base_output_dir


def run_ladder_stage(base_output_dir: Path, args: argparse.Namespace) -> Path:
    kwargs = common_run_kwargs(args)
    manifest: list[dict[str, object]] = []
    for rung in parse_ladder_rungs(args.ladder_rungs):
        label = ladder_label(rung)
        output_dir = base_output_dir / label
        ladder = {**rung, "n_draws": args.ladder_draws, "n_starts": args.ladder_starts, "maxfev": args.ladder_maxfev}
        _log(f"Running GLP ladder rung {label} -> {output_dir}")
        run_glp_experiment(**kwargs, strategy="ladder", output_dir=output_dir, ladder=ladder)
        manifest.append({"rung": label, **rung, "output_dir": str(output_dir)})
    write_batch_metadata(base_output_dir, manifest)
    return base_output_dir


def run_compare(args: argparse.Namespace, output_root: Path) -> Path:
    paper_dir = resolve_stage_output_dir(output_root, args.paper_dir, "paper")
    mango_mdd_dir = resolve_stage_output_dir(output_root, args.mango_mdd_dir, "mango_mdd")
    mango_rmse_dir = resolve_stage_output_dir(output_root, args.mango_rmse_dir, "mango_rmse")
    mango_rmse_random_dir = resolve_stage_output_dir(output_root, args.mango_rmse_random_dir, "mango_rmse_random")
    comparison_dir = resolve_stage_output_dir(output_root, args.comparison_dir, "comparison")
    experiment_dirs = {
        "paper": _existing(paper_dir),
        "mango_mdd": _existing(mango_mdd_dir),
        "mango_rmse": _existing(mango_rmse_dir),
        "mango_rmse_random": _existing(mango_rmse_random_dir),
    }
    experiment_dirs["paper_plugin"] = _existing(output_root / "paper_plugin")
    ladder_root = output_root / "ladder"
    if ladder_root.exists():
        for child in sorted(ladder_root.iterdir()):
            if child.is_dir() and (child / "forecast_panel.csv").exists():
                experiment_dirs[f"ladder_{child.name}"] = child
    if not any(experiment_dirs.values()):
        raise FileNotFoundError("No GLP forecast outputs were found for comparison.")
    _log(f"Creating GLP comparison report -> {comparison_dir}")
    return create_glp_comparison_report(experiment_dirs, comparison_dir, panel_path=resolve_project_path(args.panel_path))


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    stages = parse_stage_list(args.stages)
    output_root = resolve_project_path(args.output_root)

    stage_dirs = {
        "paper": resolve_stage_output_dir(output_root, args.paper_dir, "paper"),
        "mango_mdd": resolve_stage_output_dir(output_root, args.mango_mdd_dir, "mango_mdd"),
        "mango_rmse": resolve_stage_output_dir(output_root, args.mango_rmse_dir, "mango_rmse"),
        "mango_rmse_random": resolve_stage_output_dir(output_root, args.mango_rmse_random_dir, "mango_rmse_random"),
    }

    _log(f"Requested GLP stages: {', '.join(stages)}")

    if "paper" in stages:
        _log(f"Running GLP paper strategy -> {stage_dirs['paper']}")
        run_simple_stage("paper", stage_dirs["paper"], args)

    if "mango_mdd" in stages:
        _log(f"Running GLP mango_mdd strategy -> {stage_dirs['mango_mdd']}")
        run_simple_stage("mango_mdd", stage_dirs["mango_mdd"], args)

    if "mango_rmse" in stages:
        run_rmse_stage("mango_rmse", stage_dirs["mango_rmse"], args)

    if "mango_rmse_random" in stages:
        run_rmse_stage("mango_rmse_random", stage_dirs["mango_rmse_random"], args)

    if "paper_plugin" in stages:
        _log(f"Running GLP paper_plugin strategy -> {output_root / 'paper_plugin'}")
        run_simple_stage("paper_plugin", output_root / "paper_plugin", args)

    if "ladder" in stages:
        run_ladder_stage(output_root / "ladder", args)

    if "compare" in stages:
        run_compare(args, output_root)

    _log("All requested GLP stages completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())