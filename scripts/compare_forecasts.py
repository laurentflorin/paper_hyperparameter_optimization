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
    parser.add_argument("--paper-dir", type=Path, default=Path("outputs/paper_hyperparameters"))
    parser.add_argument("--mango-mdd-dir", type=Path, default=Path("outputs/mango_mdd"))
    parser.add_argument("--mango-rmse-dir", type=Path, default=Path("outputs/mango_rmse"))
    parser.add_argument("--mango-rmse-random-dir", type=Path, default=Path("outputs/mango_rmse_random"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/comparison"))
    return parser


def resolve_experiment_dir(path: Path | None, fallback_name: str) -> Path | None:
    if path is None:
        return None

    candidates = [resolve_project_path(path)]
    fallback = resolve_project_path(Path("scripts/outputs/euler") / fallback_name)
    if fallback not in candidates:
        candidates.append(fallback)

    for candidate in candidates:
        if not candidate.exists():
            continue
        if (candidate / "forecast_panel.csv").exists() and (candidate / "forecast_panel.csv").stat().st_size > 1:
            return candidate
        child_candidates = [
            child
            for child in sorted(candidate.iterdir())
            if child.is_dir() and (child / "forecast_panel.csv").exists() and (child / "forecast_panel.csv").stat().st_size > 1
        ]
        if child_candidates:
            return candidate

    return None


def main() -> int:
    args = build_parser().parse_args()
    paper_dir = resolve_experiment_dir(args.paper_dir, "paper_hyperparameters")
    mango_mdd_dir = resolve_experiment_dir(args.mango_mdd_dir, "mango_mdd")
    mango_rmse_dir = resolve_experiment_dir(args.mango_rmse_dir, "mango_rmse")
    mango_rmse_random_dir = resolve_experiment_dir(args.mango_rmse_random_dir, "mango_rmse_random")
    args.output_dir = resolve_project_path(args.output_dir)
    experiment_dirs = {
        "paper": paper_dir,
        "mango_mdd": mango_mdd_dir,
        "mango_rmse": mango_rmse_dir,
    }
    if mango_rmse_random_dir is not None:
        experiment_dirs["mango_rmse_random"] = mango_rmse_random_dir
    create_comparison_report(experiment_dirs, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
