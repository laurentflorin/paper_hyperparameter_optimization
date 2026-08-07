"""Compare the recursive out-of-sample forecasts of the GLP strategies."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from glp_hyperparameter_optimization.config import resolve_project_path
from glp_hyperparameter_optimization.reporting import create_glp_comparison_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare out-of-sample forecasts from the GLP experiments.")
    parser.add_argument("--paper-dir", type=Path, default=Path("outputs/glp/paper"))
    parser.add_argument("--mango-mdd-dir", type=Path, default=Path("outputs/glp/mango_mdd"))
    parser.add_argument("--mango-rmse-dir", type=Path, default=Path("outputs/glp/mango_rmse"))
    parser.add_argument("--mango-rmse-random-dir", type=Path, default=Path("outputs/glp/mango_rmse_random"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/glp/comparison"))
    return parser


def _existing(path: Path | None) -> Path | None:
    if path is None:
        return None
    resolved = resolve_project_path(path)
    return resolved if resolved.exists() else None


def main() -> int:
    args = build_parser().parse_args()
    experiment_dirs = {
        "paper": _existing(args.paper_dir),
        "mango_mdd": _existing(args.mango_mdd_dir),
        "mango_rmse": _existing(args.mango_rmse_dir),
        "mango_rmse_random": _existing(args.mango_rmse_random_dir),
    }
    create_glp_comparison_report(experiment_dirs, resolve_project_path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
