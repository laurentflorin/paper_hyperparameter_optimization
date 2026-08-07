"""Recursive GLP forecasts with hyperparameters selected by update_hyperparameters_mango.

The "mango_mdd" strategy maximizes the same GLP (log) posterior / marginal data
density as the paper, but with Mango Bayesian optimization instead of the
gradient optimizer -- isolating the effect of the optimizer choice.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from glp_hyperparameter_optimization.forecasting import build_optimizer_parser, run_from_namespace


def main() -> int:
    parser = build_optimizer_parser(
        "Run recursive GLP forecasts with hyperparameters selected by update_hyperparameters_mango (MDD)."
    )
    args = parser.parse_args()
    run_from_namespace("mango_mdd", args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
