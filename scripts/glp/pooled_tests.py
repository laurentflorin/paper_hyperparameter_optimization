"""Diebold-Mariano tests pooled over horizons 1-8, against the GLP benchmark.

For every criterion (ladder run) and variable, the squared-error differential
to GLP is scaled by GLP's mean squared error at each horizon, averaged over
horizons and tested once (``scoring.pooled_diebold_mariano``); p-values are
Holm-adjusted across criteria within each model size and variable. The same is
done for the mixed-frequency VAR (median forecasts, by month within the quarter,
against the published hyperparameters). Writes ``outputs/glp/pooled_tests.csv``
and ``outputs/mf_ladder/comparison/pooled_tests.csv``.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from glp_hyperparameter_optimization.scoring import holm_adjust, pooled_diebold_mariano  # noqa: E402

SIZES = {"small": "canonical_small", "medium": "canonical_medium", "large_nr": "canonical_large_nr"}


def losses(panel: pd.DataFrame, error: str) -> pd.DataFrame:
    """Squared errors as an (origin x horizon) frame per variable."""
    p = panel.assign(sq=panel[error] ** 2)
    return p.pivot_table(index=["variable", "forecast_origin"], columns="horizon_quarters", values="sq")


def pooled(base: pd.DataFrame, model: pd.DataFrame) -> list[dict]:
    rows = []
    for variable in base.index.get_level_values(0).unique():
        b, m = base.loc[variable], model.loc[variable] if variable in model.index.get_level_values(0) else None
        if m is None:
            continue
        joint = b.join(m, lsuffix="_b", rsuffix="_m", how="inner")
        H = b.shape[1]
        stat, p = pooled_diebold_mariano(joint.iloc[:, H:].to_numpy(), joint.iloc[:, :H].to_numpy())
        rel = 100 * (np.sqrt(joint.iloc[:, H:].mean().to_numpy()) / np.sqrt(joint.iloc[:, :H].mean().to_numpy()) - 1)
        rows.append({"variable": variable, "rel_rmse_pct": float(rel.mean()), "pooled_stat": stat, "pooled_p": p, "n_origins": len(joint)})
    return rows


def holm(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    frame["pooled_p_holm"] = np.nan
    for _, idx in frame.groupby(keys).groups.items():
        frame.loc[idx, "pooled_p_holm"] = holm_adjust(frame.loc[idx, "pooled_p"].to_numpy())
    return frame


def main() -> None:
    rows = []
    for size, folder in SIZES.items():
        root = REPO_ROOT / "outputs" / "glp" / folder
        base = losses(pd.read_csv(root / "paper" / "forecast_panel.csv"), "error_adj")
        runs = {"paper_plugin": root / "paper_plugin"} | {f"ladder_{d.name}": d for d in sorted((root / "ladder").iterdir()) if (d / "forecast_panel.csv").exists()}
        for name, path in runs.items():
            for r in pooled(base, losses(pd.read_csv(path / "forecast_panel.csv"), "error_adj")):
                rows.append({"model_size": size, "variant": name, **r})
    glp = holm(pd.DataFrame(rows), ["model_size", "variable"])
    glp.to_csv(REPO_ROOT / "outputs" / "glp" / "pooled_tests.csv", index=False)

    mf_panel = pd.read_csv(REPO_ROOT / "outputs" / "mf_ladder" / "forecast_panel.csv")
    mf_panel["med_err"] = mf_panel["median_metric"] - mf_panel["actual_metric"]
    mf_rows = []
    for group, g in mf_panel.groupby("group"):
        base = losses(g[g.strategy == "paper"], "med_err")
        for strategy in sorted(set(g.strategy) - {"paper"}):
            for r in pooled(base, losses(g[g.strategy == strategy], "med_err")):
                mf_rows.append({"group": group, "strategy": strategy, **r})
    mf = holm(pd.DataFrame(mf_rows), ["group", "variable"])
    mf.to_csv(REPO_ROOT / "outputs" / "mf_ladder" / "comparison" / "pooled_tests.csv", index=False)

    gdp = glp[glp.variable == "GDP"].sort_values("pooled_p")
    print(gdp.groupby("model_size").head(5)[["model_size", "variant", "rel_rmse_pct", "pooled_p", "pooled_p_holm"]].round(3).to_string(index=False))
    print("\nHolm < 0.10 by model size and variable (gains / losses):")
    sig = glp[glp.pooled_p_holm < 0.10]
    print(sig.assign(sign=np.where(sig.pooled_stat < 0, "gain", "loss")).groupby(["model_size", "variable", "sign"]).size().unstack(fill_value=0).to_string())
    print("\nMF:")
    print(mf[mf.pooled_p_holm < 0.10][["group", "strategy", "variable", "rel_rmse_pct", "pooled_p_holm"]].round(3).to_string(index=False))
    print(mf[mf.variable == "GDP"][["group", "strategy", "rel_rmse_pct", "pooled_p", "pooled_p_holm"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
