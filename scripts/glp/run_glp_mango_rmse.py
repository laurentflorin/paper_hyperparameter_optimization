"""Recursive GLP forecasts with hyperparameters selected by update_hyperparameters_mango_rmse.

The "mango_rmse" strategy selects ``[lambda, theta, miu]`` by minimizing the
rolling out-of-sample RMSE of the chosen variable(s) at a target horizon. By
default it runs four horizon-specific optimizations (1, 2, 4 and 8 quarters
ahead), writing each to its own subdirectory (``h1q`` .. ``h8q``).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from glp_hyperparameter_optimization.config import MAX_FORECAST_HORIZON_QUARTERS, resolve_project_path
from glp_hyperparameter_optimization.forecasting import (
    build_optimizer_parser,
    parse_csv_int_list,
    run_from_namespace,
)

DEFAULT_EVAL_HORIZONS = [1, 2, 4, 8]
DEFAULT_N_EVAL = 3
DEFAULT_N_OBJ_DRAWS = 200


def build_parser():
    parser = build_optimizer_parser(
        "Run recursive GLP forecasts with hyperparameters selected by update_hyperparameters_mango_rmse."
    )
    parser.add_argument(
        "--optimization-eval-horizons-quarters",
        type=str,
        default="1,2,4,8",
        help="Comma-separated target horizons to optimize for; each writes to its own output subdirectory.",
    )
    parser.add_argument("--optimization-n-eval", type=int, default=DEFAULT_N_EVAL,
                        help="Number of rolling evaluation origins used inside the RMSE objective.")
    parser.add_argument("--optimization-n-obj-draws", type=int, default=DEFAULT_N_OBJ_DRAWS,
                        help="Posterior beta draws averaged into the predictive-mean RMSE objective "
                             "(<=1 uses the deterministic posterior-mode point forecast).")
    return parser


def horizon_output_dir(base: Path, horizon: int, multiple: bool) -> Path:
    return base if not multiple else base / f"h{horizon}q"


def validate_eval_horizons(horizons: list[int]) -> list[int]:
    unique: list[int] = []
    for horizon in horizons:
        if horizon < 1 or horizon > MAX_FORECAST_HORIZON_QUARTERS:
            raise ValueError(f"eval horizon must be within 1..{MAX_FORECAST_HORIZON_QUARTERS}, got {horizon}.")
        if horizon not in unique:
            unique.append(horizon)
    return unique


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.output_dir = resolve_project_path(args.output_dir)

    horizons = validate_eval_horizons(parse_csv_int_list(args.optimization_eval_horizons_quarters, DEFAULT_EVAL_HORIZONS))
    multiple = len(horizons) > 1
    manifest: list[dict[str, object]] = []

    for horizon in horizons:
        run_args = SimpleNamespace(**vars(args))
        run_args.output_dir = horizon_output_dir(args.output_dir, horizon, multiple)
        run_args.optimization_horizon_quarters = horizon
        run_args.optimization_eval_horizon_quarters = horizon
        run_from_namespace("mango_rmse", run_args)
        manifest.append({"horizon_quarters": horizon, "output_dir": str(run_args.output_dir)})

    if multiple:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "batch_metadata.json").write_text(json.dumps({"runs": manifest}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
