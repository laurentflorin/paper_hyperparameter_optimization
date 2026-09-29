"""Re-score all finished small-model ladder runs and inject the cross-horizon
matrix data and the status line into the blueprint page (same folder)."""
import json, re, subprocess
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
PAGE = Path(__file__).with_name("prior_selection_blueprint.html")
C = REPO / "outputs/glp/canonical_small"

extra = []
for d in sorted((C / "ladder").iterdir()):
    if d.name != "mdd" and (d / "forecast_panel.csv").exists() and (d / "run_metadata.json").exists():
        extra += ["--extra-dir", f"L_{d.name}={d}"]
subprocess.run([str(REPO / ".venv/bin/python"), "scripts/glp/compare_glp_forecasts.py", "--root-dir", str(C),
                "--paper-dir", str(C / "paper"), "--extra-dir", f"paper_plugin={C / 'paper_plugin'}", *extra,
                "--output-dir", str(C / "comparison_all")], cwd=REPO, check=True, capture_output=True)
rel = pd.read_csv(C / "comparison_all/relative_scores_vs_glp.csv")
xh = {}
for variant, g in rel.groupby("variant"):
    m = re.match(r"L_(rmsej|rmse|lsj|lsm)_h(\d)_(w\d+|all)$", variant) or re.match(r"L_(lsj)(1)_(w\d+|all)$", variant)
    if not m:
        continue
    obj, th, win = m.groups()
    for var, gv in g.groupby("variable"):
        if var in ("GDP", "DEFL", "FFR"):
            gv = gv.sort_values("horizon_quarters")
            xh.setdefault(obj, {}).setdefault(win, {}).setdefault(var, {})[th] = [
                [round(float(r.rel_rmse_pct), 1), round(float(r.dm_sq_error_p), 3)] for r in gv.itertuples()]

done = sum(1 for d in (C / "ladder").iterdir() if (d / "forecast_panel.csv").exists())
medium = sum(1 for d in (REPO / "outputs/glp/canonical_medium/ladder").iterdir() if (d / "forecast_panel.csv").exists())
large_dir = REPO / "outputs/glp/canonical_large_nr/ladder"
large = sum(1 for d in large_dir.iterdir() if (d / "forecast_panel.csv").exists()) if large_dir.exists() else 0
stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
status = f"Updated {stamp} · ladder runs complete: {done} (3 variables), {medium} (7 variables), {large} (18 variables)"

s = PAGE.read_text()
s = re.sub(r"/\*XH_START\*/.*?/\*XH_END\*/", lambda _: "/*XH_START*/" + json.dumps(xh, separators=(",", ":")) + "/*XH_END*/", s, flags=re.S)
s = re.sub(r"<!--STATUS_START-->.*?<!--STATUS_END-->", lambda _: f"<!--STATUS_START-->{status}<!--STATUS_END-->", s, flags=re.S)
PAGE.write_text(s)
print(status)
