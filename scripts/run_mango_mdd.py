from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_hyperparameter_optimization.forecasting import build_optimizer_parser, run_from_namespace


def main() -> int:
    parser = build_optimizer_parser("Run recursive MF-VAR forecasts with hyperparameters selected by update_hyperparameters_mango.")
    args = parser.parse_args()
    run_from_namespace("mango_mdd", args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
