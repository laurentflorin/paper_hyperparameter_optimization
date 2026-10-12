"""Real-time choice of the evaluation window from past forecast performance.

At each forecast origin, pick the evaluation window W of the joint RMSE
criterion (target horizon four quarters) whose forecasts have had the lowest
mean squared error relative to GLP so far, averaged over the variables, using
only forecasts whose target quarter ended before the origin; until at least
``--min-realized`` origins have realized one-quarter-ahead targets, use the
longest window. The selected window's forecasts are then scored like any other
criterion. Writes ``outputs/glp/window_selection.csv`` and prints relative RMSEs.
Past errors use the final scoring vintage (a mild look-ahead in data revisions,
not in the choice of window).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SETS = {
    "small": ("canonical_small", [12, 20, 40, 60, 80]),
    "medium": ("canonical_medium", [12, 40, 60, 80]),
    "large_nr": ("canonical_large_nr", [12, 60, 80]),
}


def load(path: Path) -> pd.DataFrame:
    p = pd.read_csv(path, usecols=["forecast_origin", "target_quarter", "horizon_quarters", "variable", "error_adj"])
    p["origin_q"] = pd.PeriodIndex(pd.to_datetime(p.forecast_origin), freq="Q")
    p["target_q"] = pd.PeriodIndex(p.target_quarter, freq="Q")
    p["sq"] = p.error_adj**2
    return p


def run(size: str, folder: str, windows: list[int], min_realized: int = 8) -> tuple[pd.DataFrame, pd.Series]:
    root = REPO_ROOT / "outputs" / "glp" / folder
    glp = load(root / "paper" / "forecast_panel.csv")
    cands = {w: load(root / "ladder" / f"rmsej_h4_w{w}" / "forecast_panel.csv") for w in windows}
    key = ["forecast_origin", "horizon_quarters", "variable"]
    glp_sq = glp.set_index(key)
    origins = sorted(glp.origin_q.unique())
    chosen, picked_rows = [], []
    for t in origins:
        scores = {}
        for w, c in cands.items():
            past = c[c.target_q < t]
            if past.origin_q.nunique() < min_realized:
                continue
            g = glp_sq.loc[past.set_index(key).index]
            rel = past.groupby("variable").sq.sum() / g.groupby("variable").sq.sum()
            scores[w] = rel.mean()
        w_star = min(scores, key=scores.get) if len(scores) == len(cands) else max(windows)
        chosen.append({"model_size": size, "origin": str(t), "window": w_star})
        picked_rows.append(cands[w_star][cands[w_star].origin_q == t])
    rule = pd.concat(picked_rows)
    rel = {}
    for name, panel in {**{f"last {w}": cands[w] for w in windows}, "selected": rule}.items():
        m = panel.groupby(["variable", "horizon_quarters"]).sq.mean() ** 0.5
        b = glp.groupby(["variable", "horizon_quarters"]).sq.mean() ** 0.5
        rel[name] = (100 * (m / b - 1)).groupby("variable").mean()
    return pd.DataFrame(chosen), pd.DataFrame(rel).T


def main() -> None:
    choices, tables = [], []
    for size, (folder, windows) in SETS.items():
        ch, tab = run(size, folder, windows)
        choices.append(ch)
        tab["model_size"] = size
        tables.append(tab)
        print(f"== {size}: windows chosen {ch.window.value_counts().sort_index().to_dict()}")
        cols = [c for c in ["GDP", "CONS", "INV", "DEFL", "FFR", "HOURS", "CPI", "EMP"] if c in tab]
        tab["All"] = tab.drop(columns="model_size").mean(axis=1)
        print(tab[cols + ["All"]].round(1).to_string())
    pd.concat(choices).to_csv(REPO_ROOT / "outputs" / "glp" / "window_selection.csv", index=False)
    pd.concat(tables).to_csv(REPO_ROOT / "outputs" / "glp" / "window_selection_scores.csv")


if __name__ == "__main__":
    main()
