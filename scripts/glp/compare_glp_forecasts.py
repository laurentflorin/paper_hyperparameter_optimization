"""Compare the recursive out-of-sample forecasts of the GLP strategies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from forecast_comparison import discover_run_directories
from glp_hyperparameter_optimization.config import resolve_project_path
from glp_hyperparameter_optimization.reporting import create_glp_comparison_report

STRATEGY_ARG_NAMES = {
    "paper": "paper_dir",
    "mango_mdd": "mango_mdd_dir",
    "mango_rmse": "mango_rmse_dir",
    "mango_rmse_random": "mango_rmse_random_dir",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare out-of-sample forecasts from the GLP experiments.")
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path("outputs/glp"),
        help="Base directory scanned for GLP runs when per-strategy directories are not provided.",
    )
    parser.add_argument(
        "--model-size",
        choices=("small", "medium", "large"),
        default=None,
        help="Optional model-size filter used during auto-discovery (recommended if multiple sizes exist).",
    )
    parser.add_argument("--paper-dir", type=Path, default=None)
    parser.add_argument("--mango-mdd-dir", type=Path, default=None)
    parser.add_argument("--mango-rmse-dir", type=Path, default=None)
    parser.add_argument("--mango-rmse-random-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/glp/comparison"))
    parser.add_argument(
        "--minimum-common-coverage",
        type=float,
        default=0.8,
        help="Warn when pairwise valid-key coverage is below this fraction.",
    )
    parser.add_argument(
        "--allow-legacy-metadata",
        action="store_true",
        help="Allow missing legacy provenance fields; known incompatibilities still fail.",
    )
    return parser


def _existing(path: Path | None) -> Path | None:
    if path is None:
        return None
    resolved = resolve_project_path(path)
    return resolved if resolved.exists() else None


def _load_run_metadata(path: Path) -> dict[str, Any] | None:
    metadata_path = path / "run_metadata.json"
    if not metadata_path.exists():
        return None
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _strategy_metadata(path: Path) -> dict[str, Any] | None:
    metadata = _load_run_metadata(path)
    if metadata is not None:
        return metadata
    if not path.exists() or not path.is_dir():
        return None
    child_metadata = [_load_run_metadata(child) for child in sorted(path.iterdir()) if child.is_dir()]
    child_metadata = [meta for meta in child_metadata if meta is not None]
    if not child_metadata:
        return None
    first = child_metadata[0]
    same_strategy = all(meta.get("strategy") == first.get("strategy") for meta in child_metadata)
    same_size = all(meta.get("model_size") == first.get("model_size") for meta in child_metadata)
    return first if same_strategy and same_size else None


def _discover_strategy_dir(root_dir: Path, strategy: str, model_size: str | None) -> Path | None:
    if not root_dir.exists():
        return None
    candidates: list[Path] = []
    for child in sorted(root_dir.iterdir()):
        if not child.is_dir():
            continue
        metadata = _strategy_metadata(child)
        if metadata is None:
            continue
        if metadata.get("strategy") != strategy:
            continue
        if model_size is not None and metadata.get("model_size") != model_size:
            continue
        candidates.append(child)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        names = ", ".join(str(path) for path in candidates)
        raise ValueError(
            f"Multiple candidate directories found for strategy {strategy!r}: {names}. "
            f"Pass --{STRATEGY_ARG_NAMES[strategy].replace('_', '-')} or --model-size to disambiguate."
        )
    return None



def _resolve_strategy_dir(explicit: Path | None, *, strategy: str, root_dir: Path, model_size: str | None) -> Path | None:
    """Resolve explicit paths strictly; retain metadata-based auto-discovery."""
    if explicit is not None:
        resolved = _existing(explicit)
        if resolved is None:
            raise FileNotFoundError(
                f"Provided directory for strategy {strategy!r} does not exist: {explicit}"
            )
        discover_run_directories(resolved)
        return resolved
    return _discover_strategy_dir(root_dir, strategy, model_size)


def main() -> int:
    args = build_parser().parse_args()
    root_dir = resolve_project_path(args.root_dir)
    experiment_dirs = {
        "paper": _resolve_strategy_dir(
            args.paper_dir, strategy="paper", root_dir=root_dir, model_size=args.model_size
        ),
        "mango_mdd": _resolve_strategy_dir(
            args.mango_mdd_dir, strategy="mango_mdd", root_dir=root_dir, model_size=args.model_size
        ),
        "mango_rmse": _resolve_strategy_dir(
            args.mango_rmse_dir, strategy="mango_rmse", root_dir=root_dir, model_size=args.model_size
        ),
        "mango_rmse_random": _resolve_strategy_dir(
            args.mango_rmse_random_dir,
            strategy="mango_rmse_random",
            root_dir=root_dir,
            model_size=args.model_size,
        ),
    }
    if not any(experiment_dirs.values()):
        raise FileNotFoundError(
            f"No GLP forecast outputs were found under {root_dir}. "
            "Pass --model-size or explicit per-strategy paths."
        )
    create_glp_comparison_report(
        experiment_dirs,
        resolve_project_path(args.output_dir),
        minimum_common_coverage=args.minimum_common_coverage,
        allow_legacy_metadata=args.allow_legacy_metadata,
    )
    resolved_inputs = ", ".join(
        str(path) for path in experiment_dirs.values() if path is not None
    )
    print(f"Comparison inputs: {resolved_inputs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
