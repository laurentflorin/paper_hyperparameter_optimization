"""Tests of the ladder criteria against GLP that account for the search over criteria.

For every model size, target and criterion, the squared-error differential to
GLP at horizons 1-8 is scaled at each horizon by GLP's mean squared error
(so every horizon carries equal weight, as in ``pooled_tests.py``). Targets are
the single variables and a joint loss that averages, over all variables, each
variable's squared error scaled by GLP's mean squared error at that variable
and horizon. With a moving-block bootstrap (blocks of 8 quarters, the longest
overlap of the forecast errors):

* ``p_average`` / ``p_uniform``: Quaedvlieg's (2021) multi-horizon tests of one
  criterion against GLP, on average over horizons and at every horizon
  (intersection-union); one-sided, small = the criterion beats GLP;
* ``p_stepm``: Romano-Wolf step-down p-values of the average test across all
  criteria of a model size, which control the familywise error rate and use
  the dependence between criteria;
* ``p_spa`` (family file): Hansen's (2005) consistent SPA test of the null that
  no criterion beats GLP, with block lengths 4, 8 and 12.

The same is done for the mixed-frequency VAR (median forecasts, by month within
the quarter, against the published hyperparameters). Writes
``outputs/glp/spa_tests.csv``, ``outputs/glp/spa_family.csv`` and the
``spa_*.csv`` counterparts in ``outputs/mf_ladder/comparison``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from glp_hyperparameter_optimization.scoring import benchmark_tests, multi_horizon_tests  # noqa: E402

SIZES = {"small": "canonical_small", "medium": "canonical_medium", "large_nr": "canonical_large_nr"}
BLOCKS = (4, 8, 12)
MAIN_BLOCK = 8


def squared_errors(panel: pd.DataFrame, error: str) -> dict[str, pd.DataFrame]:
    """Per variable, an (origin x horizon) frame of squared errors."""
    p = panel.assign(sq=panel[error] ** 2)
    wide = p.pivot_table(index=["variable", "forecast_origin"], columns="horizon_quarters", values="sq")
    return {v: wide.loc[v] for v in wide.index.get_level_values(0).unique()}


def targets(base: dict[str, pd.DataFrame], runs: dict[str, dict[str, pd.DataFrame]]):
    """Yield ``(target, baseline (T x H), {criterion: (T x H)})`` on common origins,
    for each variable and for the joint loss."""
    names = [v for v in base if all(v in r for r in runs.values())]
    origins = sorted(set.intersection(*[set(base[v].dropna().index) for v in names],
                                      *[set(r[v].dropna().index) for r in runs.values() for v in names]))
    joint_b, joint_r = 0.0, {k: 0.0 for k in runs}
    for v in names:
        b = base[v].loc[origins].to_numpy()
        crit = {k: r[v].loc[origins].to_numpy() for k, r in runs.items()}
        yield v, b, crit
        scale = b.mean(axis=0)
        joint_b = joint_b + b / scale / len(names)
        for k in runs:
            joint_r[k] = joint_r[k] + crit[k] / scale / len(names)
    yield "joint", joint_b, joint_r


def evaluate(base: np.ndarray, crit: dict[str, np.ndarray], n_boot: int) -> tuple[pd.DataFrame, list[dict]]:
    names = list(crit)
    scale = base.mean(axis=0)
    diffs = {k: (base - crit[k]) / scale for k in names}
    rows = []
    for k in names:
        p_avg, p_unif = multi_horizon_tests(diffs[k], block=MAIN_BLOCK, n_boot=n_boot)
        rel = 100 * (np.sqrt(crit[k].mean(axis=0)) / np.sqrt(scale) - 1)
        rows.append({"variant": k, "rel_rmse_pct": float(rel.mean()), "p_average": p_avg, "p_uniform": p_unif})
    avg = np.column_stack([diffs[k].mean(axis=1) for k in names])
    family = []
    for block in BLOCKS:
        res = benchmark_tests(avg, block=block, n_boot=n_boot)
        if block == MAIN_BLOCK:
            for row, t, p in zip(rows, res["t"], res["p_stepm"]):
                row.update({"t_stat": float(t), "p_stepm": float(p)})
        best = int(np.argmax(res["t"]))
        family.append({"block": block, "n_criteria": len(names), "p_spa": res["p_spa"], "best_variant": names[best],
                       "min_p_stepm": float(res["p_stepm"].min()), "n_stepm_10": int((res["p_stepm"] < 0.10).sum())})
    return pd.DataFrame(rows), family


def run(groups, n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``groups``: iterable of ``(keys, base, runs)`` with ``keys`` a dict of labels."""
    tests, families = [], []
    for keys, base, runs in groups:
        for target, b, crit in targets(base, runs):
            frame, family = evaluate(b, crit, n_boot)
            tests.append(frame.assign(**keys, target=target, n_origins=b.shape[0]))
            families += [{**keys, "target": target, **f} for f in family]
    tests = pd.concat(tests, ignore_index=True)
    first = list(keys) + ["target", "variant"]
    return tests[first + [c for c in tests.columns if c not in first]], pd.DataFrame(families)


def glp_groups():
    for size, folder in SIZES.items():
        root = REPO_ROOT / "outputs" / "glp" / folder
        base = squared_errors(pd.read_csv(root / "paper" / "forecast_panel.csv"), "error_adj")
        dirs = {"paper_plugin": root / "paper_plugin"} | {
            f"ladder_{d.name}": d for d in sorted((root / "ladder").iterdir()) if (d / "forecast_panel.csv").exists()}
        runs = {k: squared_errors(pd.read_csv(d / "forecast_panel.csv"), "error_adj") for k, d in dirs.items()}
        print(f"[spa] {size}: {len(runs)} criteria", flush=True)
        yield {"model_size": size}, base, runs


def mf_groups():
    panel = pd.read_csv(REPO_ROOT / "outputs" / "mf_ladder" / "forecast_panel.csv")
    panel["med_err"] = panel["median_metric"] - panel["actual_metric"]
    for group, g in panel.groupby("group"):
        base = squared_errors(g[g.strategy == "paper"], "med_err")
        runs = {s: squared_errors(g[g.strategy == s], "med_err") for s in sorted(set(g.strategy) - {"paper"})}
        yield {"group": group}, base, runs


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-boot", type=int, default=5000)
    args = p.parse_args(argv)

    tests, family = run(glp_groups(), args.n_boot)
    tests.to_csv(REPO_ROOT / "outputs" / "glp" / "spa_tests.csv", index=False)
    family.to_csv(REPO_ROOT / "outputs" / "glp" / "spa_family.csv", index=False)
    mf_tests, mf_family = run(mf_groups(), args.n_boot)
    out = REPO_ROOT / "outputs" / "mf_ladder" / "comparison"
    mf_tests.to_csv(out / "spa_tests.csv", index=False)
    mf_family.to_csv(out / "spa_family.csv", index=False)

    main_family = family[family.block == MAIN_BLOCK]
    print(main_family[main_family.target.isin(["GDP", "joint"])].round(3).to_string(index=False))
    sig = tests[tests.p_stepm < 0.10]
    print("\nStepM < 0.10:\n" + sig[["model_size", "target", "variant", "rel_rmse_pct", "p_average", "p_uniform", "p_stepm"]].round(3).to_string(index=False))
    print("\nMF:\n" + mf_family[(mf_family.block == MAIN_BLOCK)].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
