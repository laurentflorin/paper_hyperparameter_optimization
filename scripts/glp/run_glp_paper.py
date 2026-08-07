"""Recursive GLP forecasts using the paper's marginal-likelihood prior selection.

This is the "paper" strategy: at each real-time origin the hyperparameters
``[lambda, theta, miu]`` are chosen by maximizing the GLP (log) posterior, and
the predictive density integrates over hyperparameter uncertainty via a
random-walk Metropolis sampler (the GLP hierarchical predictive density).
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from glp_hyperparameter_optimization.forecasting import build_common_parser, run_from_namespace


def main() -> int:
    parser = build_common_parser("Run recursive GLP forecasts with marginal-likelihood prior selection.")
    args = parser.parse_args()
    run_from_namespace("paper", args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
