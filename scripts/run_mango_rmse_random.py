from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_hyperparameter_optimization.config import resolve_project_path
from paper_hyperparameter_optimization.forecasting import (
    MAX_FORECAST_HORIZON_QUARTERS,
    build_optimizer_parser,
    parse_csv_int_list,
    run_from_namespace,
)


DEFAULT_RMSE_EVAL_HORIZONS = [1, 2, 4, 8]
DEFAULT_RMSE_N_EVAL = 3


def build_parser():
    parser = build_optimizer_parser(
        "Run recursive MF-VAR forecasts with hyperparameters selected by update_hyperparameters_mango_rmse_random."
    )
    parser.add_argument(
        "--optimization-eval-horizons-quarters",
        type=str,
        default="1,2,4,8",
        help="Comma-separated target forecast horizons to optimize for. Each horizon writes to its own output subdirectory.",
    )
    parser.add_argument(
        "--optimization-n-eval",
        type=int,
        default=DEFAULT_RMSE_N_EVAL,
        help="Number of forecast origins sampled inside the RMSE objective.",
    )
    parser.add_argument(
        "--optimization-min-t",
        type=int,
        default=None,
        help="Optional minimum number of lowest-frequency in-sample observations for valid evaluation origins.",
    )
    parser.add_argument(
        "--optimization-random-seed",
        type=int,
        default=None,
        help="Optional RNG seed used to sample the RMSE evaluation origins.",
    )
    return parser


def horizon_output_dir(base_output_dir: Path, horizon_quarters: int, multiple_horizons: bool) -> Path:
    if not multiple_horizons:
        return base_output_dir
    return base_output_dir / f"h{horizon_quarters}q"


def validate_eval_horizons(horizons: list[int], forecast_horizon_months: int) -> list[int]:
    unique_horizons = []
    for horizon in horizons:
        if horizon < 1 or horizon > MAX_FORECAST_HORIZON_QUARTERS:
            raise ValueError(
                f"optimization-eval-horizons-quarters must be between 1 and {MAX_FORECAST_HORIZON_QUARTERS}, got {horizon}."
            )
        if horizon not in unique_horizons:
            unique_horizons.append(horizon)

    required_months = max(unique_horizons) * 3
    if forecast_horizon_months < required_months:
        raise ValueError(
            f"forecast-horizon-months must be at least {required_months} to evaluate horizon {max(unique_horizons)} quarters."
        )
    return unique_horizons


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.panel_path = resolve_project_path(args.panel_path)
    args.output_dir = resolve_project_path(args.output_dir)

    eval_horizons = validate_eval_horizons(
        parse_csv_int_list(args.optimization_eval_horizons_quarters, DEFAULT_RMSE_EVAL_HORIZONS),
        args.forecast_horizon_months,
    )
    multiple_horizons = len(eval_horizons) > 1
    run_manifest: list[dict[str, object]] = []

    for horizon_quarters in eval_horizons:
        run_args = SimpleNamespace(**vars(args))
        run_args.output_dir = horizon_output_dir(args.output_dir, horizon_quarters, multiple_horizons)
        run_args.optimization_horizon_quarters = horizon_quarters
        run_args.optimization_eval_horizon_quarters = horizon_quarters
        run_args.optimization_n_eval = args.optimization_n_eval
        run_args.optimization_min_t = args.optimization_min_t
        run_args.optimization_random_seed = args.optimization_random_seed
        run_from_namespace("mango_rmse_random", run_args)
        run_manifest.append(
            {
                "optimization_horizon_quarters": horizon_quarters,
                "optimization_eval_horizon_quarters": horizon_quarters,
                "optimization_n_eval": args.optimization_n_eval,
                "optimization_min_t": args.optimization_min_t,
                "optimization_random_seed": args.optimization_random_seed,
                "output_dir": str(run_args.output_dir),
            }
        )

    if multiple_horizons:
        manifest_path = args.output_dir / "batch_metadata.json"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps({"runs": run_manifest}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
