from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_hyperparameter_optimization.config import resolve_project_path
from paper_hyperparameter_optimization.forecasting import build_common_parser, run_from_namespace


def main() -> int:
    parser = build_common_parser("Run recursive MF-VAR forecasts with the paper hyperparameters.")
    args = parser.parse_args()
    args.panel_path = resolve_project_path(args.panel_path)
    args.output_dir = resolve_project_path(args.output_dir)
    run_from_namespace("paper", args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
