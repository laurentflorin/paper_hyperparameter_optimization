"""Prior selection for the Schorfheide-Song mixed-frequency VAR in real time.

At every forecast origin the Gibbs sampler runs once at the published
Schorfheide-Song hyperparameters; the hyperparameters are then selected on the
captured draws of the balanced monthly data under each criterion (see
:mod:`paper_hyperparameter_optimization.mf_ladder`), and the model is re-fit and
forecast with every selected prior. Results resume from per-origin checkpoints.

Example (Euler, 48 cores)::

    python scripts/run_mf_ladder.py --output-dir outputs/mf_ladder --n-workers 48
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from paper_hyperparameter_optimization.config import (  # noqa: E402
    PAPER_ACTUAL_VINTAGE,
    PAPER_FORECAST_END,
    PAPER_FORECAST_START,
    REALTIME_PANEL_PATH,
    forecast_origin_dates,
    resolve_project_path,
)
from paper_hyperparameter_optimization.mf_ladder import MFCriterion, run_mf_ladder  # noqa: E402


def parse_criteria(value: str) -> tuple[MFCriterion, ...]:
    """``ml,rmse:h4:w12,rmsej:h4:w60`` -> criteria."""
    out = []
    for item in value.split(","):
        parts = item.strip().split(":")
        kwargs: dict[str, int | str] = {"name": parts[0]}
        for part in parts[1:]:
            if part.startswith("h"):
                kwargs["horizon_q"] = int(part[1:])
            elif part.startswith("w"):
                kwargs["window"] = int(part[1:])
            else:
                raise ValueError(f"Cannot parse criterion component {part!r} in {item!r}.")
        if kwargs["name"] not in ("ml", "rmse", "rmsej"):
            raise ValueError(f"Unknown criterion {kwargs['name']!r}.")
        out.append(MFCriterion(**kwargs))  # type: ignore[arg-type]
    return tuple(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/mf_ladder"))
    p.add_argument("--panel-path", type=Path, default=REALTIME_PANEL_PATH)
    p.add_argument("--start", default=PAPER_FORECAST_START.strftime("%Y-%m-%d"))
    p.add_argument("--end", default=PAPER_FORECAST_END.strftime("%Y-%m-%d"))
    p.add_argument("--every", type=int, default=1, help="Use every k-th monthly origin.")
    p.add_argument("--actual-vintage", default=PAPER_ACTUAL_VINTAGE.strftime("%Y-%m-%d"))
    p.add_argument("--criteria", default="ml,rmse:h4:w12,rmsej:h4:w60")
    p.add_argument("--fit-nsim", type=int, default=4000)
    p.add_argument("--keep-draws", type=int, default=50, help="Draws of the balanced monthly data kept for selection.")
    p.add_argument(
        "--explosive-check",
        action="store_true",
        help="Truncate the posterior to stable VARs (MBFVAR's check_explosive). Off by default: the criteria and the "
        "GLP ladder use the untruncated posterior, and loss-selected priors can leave no stable draw at crisis origins.",
    )
    p.add_argument(
        "--max-it-stable",
        type=int,
        default=5000,
        help="With --explosive-check: redraws per Gibbs iteration before MBFVAR abandons it (MBFVAR default 1000).",
    )
    p.add_argument("--n-starts", type=int, default=4)
    p.add_argument("--maxfev", type=int, default=400)
    p.add_argument("--seed-base", type=int, default=2015)
    p.add_argument("--n-workers", type=int, default=1)
    return p


def check_mbfvar_version(minimum: tuple[int, ...] = (0, 9, 1)) -> None:
    """Needs MBFVAR >= 0.9.1: seeded fit()/forecast(), retained per-draw
    aggregated paths (0.9.0) and the screened explosive-root check (0.9.1)."""
    import importlib.metadata

    version = importlib.metadata.version("MBFVAR")
    print(f"[mf_ladder] MBFVAR {version}", flush=True)
    if tuple(int(p) for p in version.split(".")[:3]) < minimum:
        raise SystemExit(
            f"MBFVAR {version} is too old; install >= {'.'.join(map(str, minimum))}: "
            'pip install --upgrade "MBFVAR @ git+https://github.com/laurentflorin/MBFVAR.git"'
        )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    check_mbfvar_version()
    origins = forecast_origin_dates(pd.Timestamp(args.start), pd.Timestamp(args.end))[:: args.every]
    run_mf_ladder(
        resolve_project_path(args.output_dir),
        origins=origins,
        panel_path=resolve_project_path(args.panel_path),
        actual_vintage=pd.Timestamp(args.actual_vintage),
        criteria=parse_criteria(args.criteria),
        fit_nsim=args.fit_nsim,
        check_explosive=args.explosive_check,
        max_it_stable=args.max_it_stable,
        keep_draws=args.keep_draws,
        n_starts=args.n_starts,
        maxfev=args.maxfev,
        seed_base=args.seed_base,
        n_workers=args.n_workers,
    )


if __name__ == "__main__":
    main()
