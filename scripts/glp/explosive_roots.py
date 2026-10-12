"""Largest root of the VAR at the posterior mode, by forecast origin.

For every run directory given (a strategy output with ``selected_hyperparameters.csv``),
rebuilds the real-time estimation sample at each origin, computes the posterior
mode of the VAR coefficients at the selected hyperparameters and reports the
largest modulus of the companion-matrix eigenvalues. Used to document why the
reserve and monetary-base series are left out of the baseline large model.

Example::

    python scripts/glp/explosive_roots.py outputs/glp/canonical_large/paper outputs/glp/canonical_large_nr/paper
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from glp_hyperparameter_optimization.config import GLP_REALTIME_PANEL_PATH  # noqa: E402
from glp_hyperparameter_optimization.data_utils import build_glp_estimation_matrix, load_glp_realtime_panel  # noqa: E402
from glp_hyperparameter_optimization.glp_model import glp_mode_estimate, hyper_to_natural_vector, prepare_glp_context  # noqa: E402


def max_root(beta: np.ndarray, n: int, lags: int) -> float:
    """Largest eigenvalue modulus of the companion matrix; ``beta`` is (1 + n*lags, n)
    with the constant in the first row."""
    A = beta[1:].T  # n x n*lags
    companion = np.zeros((n * lags, n * lags))
    companion[:n] = A
    companion[n:, : n * (lags - 1)] = np.eye(n * (lags - 1))
    return float(np.abs(np.linalg.eigvals(companion)).max())


def roots(run_dir: Path, panel: pd.DataFrame) -> pd.DataFrame:
    meta = pd.read_json(run_dir / "run_metadata.json", typ="series")
    hyp = pd.read_csv(run_dir / "selected_hyperparameters.csv")
    rows = []
    for _, r in hyp.iterrows():
        origin = pd.Timestamp(r["forecast_origin"])
        y, codes, _ = build_glp_estimation_matrix(panel, origin, meta["model_size"])
        ctx = prepare_glp_context(
            y, int(meta["lags"]), hyperpriors=int(meta["hyperpriors"]),
            sur=int(meta["sur"]), noc=int(meta["noc"]), mnpsi=int(meta["mnpsi"]), mnalpha=int(meta["mnalpha"]),
        )
        psi = r.get("psi")
        hyper = {"lambda": r["lambda"], "theta": r["theta"], "miu": r["miu"],
                 "psi": ast.literal_eval(psi) if isinstance(psi, str) else None}
        beta, _ = glp_mode_estimate(ctx, hyper_to_natural_vector(hyper, ctx))
        rows.append({"run": str(run_dir.relative_to(REPO_ROOT)) if run_dir.is_relative_to(REPO_ROOT) else str(run_dir),
                     "forecast_origin": r["forecast_origin"], "max_root": max_root(beta, len(codes), int(meta["lags"]))})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="+", type=Path)
    p.add_argument("--panel", type=Path, default=GLP_REALTIME_PANEL_PATH)
    p.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "glp" / "explosive_roots.csv")
    args = p.parse_args(argv)

    panel = load_glp_realtime_panel(args.panel)
    out = pd.concat([roots(d.resolve(), panel) for d in args.run_dirs], ignore_index=True)
    out.to_csv(args.output, index=False)
    wide = out.pivot(index="forecast_origin", columns="run", values="max_root")
    print(wide.loc["2007-12-31":"2011-12-31"].round(3).to_string())
    print("\nMaximum over origins:\n" + wide.max().round(3).to_string())
    print("Origins with a root above 1:\n" + (wide > 1).sum().to_string())
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
