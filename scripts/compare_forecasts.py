from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_hyperparameter_optimization.config import resolve_project_path
from paper_hyperparameter_optimization.reporting import create_comparison_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare out-of-sample forecasts from the MF-VAR experiments.")
    parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--mango-mdd-dir", type=Path, required=True)
    parser.add_argument("--mango-rmse-dir", type=Path, required=True)
    parser.add_argument("--mango-rmse-random-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.paper_dir = resolve_project_path(args.paper_dir)
    args.mango_mdd_dir = resolve_project_path(args.mango_mdd_dir)
    args.mango_rmse_dir = resolve_project_path(args.mango_rmse_dir)
    if args.mango_rmse_random_dir is not None:
        args.mango_rmse_random_dir = resolve_project_path(args.mango_rmse_random_dir)
    args.output_dir = resolve_project_path(args.output_dir)
    experiment_dirs = {
        "paper": args.paper_dir,
        "mango_mdd": args.mango_mdd_dir,
        "mango_rmse": args.mango_rmse_dir,
    }
    if args.mango_rmse_random_dir is not None:
        experiment_dirs["mango_rmse_random"] = args.mango_rmse_random_dir
    create_comparison_report(experiment_dirs, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
