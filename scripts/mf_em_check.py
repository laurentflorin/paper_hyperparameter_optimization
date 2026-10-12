"""Check the one-step Monte Carlo EM selection of the MF-VAR marginal likelihood.

At a subset of forecast origins, iterate the EM from the published
Schorfheide-Song hyperparameters. At every step, record the gradient of the
observed-data log marginal likelihood at the current point by Fisher's identity,
with its Monte Carlo standard error (``mf_ladder.observed_score``), and an
importance-sampling estimate over the latent monthly draws of how much the step
raises the observed-data log marginal likelihood, with its effective sample size
(``mf_ladder.log_ml_ratio``). Writes ``em_check.csv``.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from paper_hyperparameter_optimization.config import PAPER_NLAGS, REALTIME_PANEL_PATH, forecast_origin_dates  # noqa: E402
from paper_hyperparameter_optimization.mf_ladder import em_check_origin  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "mf_ladder" / "em_check.csv")
    p.add_argument("--n-origins", type=int, default=16)
    p.add_argument("--iterations", type=int, default=4)
    p.add_argument("--fit-nsim", type=int, default=2000)
    p.add_argument("--keep-draws", type=int, default=200)
    p.add_argument("--maxfev", type=int, default=400)
    p.add_argument("--seed-base", type=int, default=2015)
    p.add_argument("--n-workers", type=int, default=8)
    args = p.parse_args(argv)

    origins = forecast_origin_dates()
    picks = origins[:: max(1, len(origins) // args.n_origins)][: args.n_origins]
    tasks = [
        {"origin_date": str(o.date()), "panel_path": str(REALTIME_PANEL_PATH), "iterations": args.iterations,
         "fit_nsim": args.fit_nsim, "keep_draws": args.keep_draws, "maxfev": args.maxfev,
         "seed_base": args.seed_base, "nlags": PAPER_NLAGS}
        for o in picks
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        for i, res in enumerate(pool.map(em_check_origin, tasks), 1):
            rows += res["rows"]
            print(f"[em_check] {i}/{len(tasks)} {res['rows'][0]['forecast_origin']}", flush=True)
    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    cols = ["q_gain", "log_ml_gain", "log_ml_gain_cumulative", "ess"] + [c for c in out if c.startswith("score_lambda")]
    print(out.groupby("iteration")[cols].median().round(3).to_string())


if __name__ == "__main__":
    main()
