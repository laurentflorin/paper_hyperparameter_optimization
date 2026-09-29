"""Score the mixed-frequency prior-selection experiment (``run_mf_ladder.py``).

Writes to ``<run-dir>/comparison``:

* ``relative_scores.csv``: RMSE and mean CRPS of every criterion relative to the
  baseline (the published Schorfheide-Song hyperparameters by default), by
  variable, horizon and month within the quarter, with Diebold-Mariano
  (Newey-West, HLN) p-values, Holm-adjusted within variable x metric x group;
* ``summary_rmse.md`` / ``summary_crps.md``: relative scores averaged over
  horizons 1-8;
* ``mcs.csv``: Model Confidence Set (T_max, block bootstrap) on squared errors
  and CRPS per variable, horizon and group;
* ``hyperparameters.md``: medians of the selected hyperparameters and the
  complete-data log marginal likelihood gap to the ``ml`` selection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from glp_hyperparameter_optimization.scoring import diebold_mariano, holm_adjust, model_confidence_set  # noqa: E402

KEYS = ["group", "variable", "horizon_quarters"]
HEADLINE = ["GDP", "CPI", "FF", "UNR", "INVFIX", "PCE", "IP", "HRS", "TB", "GOV", "SP500"]


def losses(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["sq_error"] = panel["error_metric"] ** 2
    return panel


def relative_scores(panel: pd.DataFrame, baseline: str) -> pd.DataFrame:
    rows = []
    for (group, variable, h), block in panel.groupby(KEYS):
        wide = {m: block.pivot_table(index="forecast_origin", columns="strategy", values=m) for m in ("sq_error", "crps")}
        if baseline not in wide["sq_error"]:
            continue
        for strategy in wide["sq_error"].columns:
            if strategy == baseline:
                continue
            row = {"group": group, "variable": variable, "horizon_quarters": h, "strategy": strategy}
            for metric, frame in wide.items():
                pair = frame[[strategy, baseline]].dropna()
                model, base = pair[strategy].to_numpy(), pair[baseline].to_numpy()
                if metric == "sq_error":
                    row["relative_rmse_pct"] = 100 * (np.sqrt(model.mean()) / np.sqrt(base.mean()) - 1)
                else:
                    row["relative_crps_pct"] = 100 * (model.mean() / base.mean() - 1)
                row[f"dm_p_{metric}"] = diebold_mariano(model, base, int(h))[1]
                row["n_origins"] = len(pair)
            rows.append(row)
    out = pd.DataFrame(rows)
    for metric in ("sq_error", "crps"):
        out[f"holm_p_{metric}"] = np.nan
        for _, idx in out.groupby(["variable", "group", "strategy"]).groups.items():
            out.loc[idx, f"holm_p_{metric}"] = holm_adjust(out.loc[idx, f"dm_p_{metric}"].to_numpy())
    return out


def mcs_table(panel: pd.DataFrame, alpha: float) -> pd.DataFrame:
    rows = []
    for metric in ("sq_error", "crps"):
        for (group, variable, h), block in panel.groupby(KEYS):
            frame = block.pivot_table(index="forecast_origin", columns="strategy", values=metric)
            res = model_confidence_set(frame, alpha=alpha, block=max(1, int(h)))
            for strategy, r in res.iterrows():
                rows.append({"metric": metric, "group": group, "variable": variable, "horizon_quarters": h, "strategy": strategy, **r.to_dict()})
    return pd.DataFrame(rows)


def summary(rel: pd.DataFrame, column: str) -> pd.DataFrame:
    table = rel.groupby(["strategy", "group", "variable"])[column].mean().unstack("variable")
    return table[[v for v in HEADLINE if v in table.columns]].round(1)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, default=REPO_ROOT / "outputs" / "mf_ladder")
    p.add_argument("--baseline", default="paper")
    p.add_argument("--mcs-alpha", type=float, default=0.10)
    args = p.parse_args(argv)

    out_dir = args.run_dir / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = losses(pd.read_csv(args.run_dir / "forecast_panel.csv"))

    rel = relative_scores(panel, args.baseline)
    rel.to_csv(out_dir / "relative_scores.csv", index=False)
    for column, name in (("relative_rmse_pct", "summary_rmse"), ("relative_crps_pct", "summary_crps")):
        table = summary(rel, column)
        table.to_csv(out_dir / f"{name}.csv")
        (out_dir / f"{name}.md").write_text(
            f"Relative {column.split('_')[1].upper()} versus `{args.baseline}` (%), average over horizons 1-8; negative is better.\n\n"
            + table.to_string()
            + "\n"
        )
    mcs_table(panel, args.mcs_alpha).to_csv(out_dir / "mcs.csv", index=False)

    hyp_path = args.run_dir / "selected_hyperparameters.csv"
    if hyp_path.exists():
        hyp = pd.read_csv(hyp_path).replace([np.inf, -np.inf], np.nan)
        ml = hyp[hyp.criterion == "ml"].set_index("forecast_origin")["complete_data_log_ml"]
        hyp["log_ml_gap_to_ml"] = hyp["complete_data_log_ml"] - hyp["forecast_origin"].map(ml)
        cols = ["lambda1_1", "lambda2_1", "lambda4_1", "lambda5_1", "log_ml_gap_to_ml"]
        med = hyp.groupby("criterion")[cols].median().round(3)
        (out_dir / "hyperparameters.md").write_text("Median selected hyperparameters across origins.\n\n" + med.to_string() + "\n")
    print(summary(rel, "relative_rmse_pct").to_string())
    print(f"Wrote comparison tables to {out_dir}")


if __name__ == "__main__":
    main()
