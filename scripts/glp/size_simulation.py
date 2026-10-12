"""Size of the forecast-comparison tests in samples like ours.

Simulates loss differentials with mean zero that overlap like eight-quarter
forecast errors (an MA(7) with equal weights, Gaussian or Student-t(4)
innovations) for T = 80 and T = 50 origins, and reports how often each test
rejects at the nominal 10% level (two-sided):

* ``NW+HLN``: Diebold-Mariano with Newey-West (7 lags) and the HLN correction;
* ``EWC`` / ``WPE``: fixed-smoothing variances with Student-t references;
* ``boot-mean``: a moving-block bootstrap of the mean with a fixed standard
  error (blocks of 8), for comparison;
* ``boot-EWC``: the moving-block bootstrap with the EWC variance recomputed in
  every replication (blocks of 8), as used by ``scoring.benchmark_tests``.

The two bootstrap tests are one-sided; the two-sided rejection is the union of
the two one-sided tests at 5%.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from glp_hyperparameter_optimization import scoring as S  # noqa: E402


def differential(rng: np.random.Generator, T: int, heavy: bool, overlap: int = 8) -> np.ndarray:
    e = rng.standard_t(4, T + overlap) if heavy else rng.normal(size=T + overlap)
    return np.convolve(e, np.ones(overlap) / np.sqrt(overlap), "valid")[:T]


def boot_mean_p(d: np.ndarray, block: int, n_boot: int, seed: int) -> float:
    """Two-sided-by-union p-value of the naive block bootstrap of the mean (fixed standard error)."""
    idx = S._block_bootstrap_indices(d.size, block, n_boot, np.random.default_rng(seed))
    centred = d[idx].mean(axis=1) - d.mean()
    return 2 * min((centred >= d.mean()).mean(), (centred <= d.mean()).mean())


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-sim", type=int, default=1000)
    p.add_argument("--n-boot", type=int, default=499)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args(argv)
    rng = np.random.default_rng(args.seed)

    rows = []
    for T in (80, 50):
        for heavy in (False, True):
            hits = {k: 0 for k in ("NW+HLN", "EWC", "WPE", "boot-mean", "boot-EWC")}
            for _ in range(args.n_sim):
                d = differential(rng, T, heavy)
                z = np.zeros(T)
                seed = int(rng.integers(2**31))
                hits["NW+HLN"] += S.diebold_mariano(d, z, 8)[1] < 0.10
                hits["EWC"] += S.fixed_smoothing_dm(d, z, "ewc")[1] < 0.10
                hits["WPE"] += S.fixed_smoothing_dm(d, z, "wpe")[1] < 0.10
                hits["boot-mean"] += boot_mean_p(d, 8, args.n_boot, seed) < 0.10
                p_one = S.benchmark_tests(np.column_stack([d, -d]), block=8, n_boot=args.n_boot, seed=seed)["p_single"]
                hits["boot-EWC"] += p_one.min() < 0.05
            rows.append({"T": T, "innovations": "t(4)" if heavy else "normal", **{k: v / args.n_sim for k, v in hits.items()}})
    out = pd.DataFrame(rows)
    out.to_csv(REPO_ROOT / "outputs" / "glp" / "size_simulation.csv", index=False)
    print("Rejection rates at the nominal 10% level:\n" + out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
