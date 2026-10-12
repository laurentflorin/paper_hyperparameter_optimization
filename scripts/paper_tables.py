"""Build the paper's tables, figure data and quoted numbers from ``outputs/``.

Writes to ``outputs/paper``:

* ``tab_*.tex``: the body rows (between ``\\midrule`` and ``\\bottomrule``) of
  every results table in the paper, in the paper's number format;
* ``fig_windows.tex``, ``fig_crosshorizon.tex``, ``fig_lambda.tex``: the pgfplots
  figures, complete (they are ``\\input`` by the paper as they are);
* ``numbers.md``: the numbers quoted in the text, by section, so that the text
  can be checked against the outputs after any rerun.

Inputs: ``outputs/glp/canonical_{small,medium,large_nr,large}`` (ladder runs and
their ``comparison``), ``outputs/glp/pooled_tests.csv``,
``outputs/glp/spa_tests.csv``, ``outputs/glp/spa_family.csv``,
``outputs/glp/window_selection_scores.csv``, ``outputs/glp/explosive_roots.csv``,
``outputs/mf_ladder`` (``comparison`` and ``em_check.csv``). Run
``scripts/glp/pooled_tests.py``, ``scripts/glp/spa_tests.py``,
``scripts/glp/window_selection.py`` and ``scripts/glp/explosive_roots.py`` first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from glp_hyperparameter_optimization.scoring import holm_adjust, model_confidence_set  # noqa: E402

GLP = REPO_ROOT / "outputs" / "glp"
MF = REPO_ROOT / "outputs" / "mf_ladder"
SIZES = {"small": "canonical_small", "medium": "canonical_medium", "large_nr": "canonical_large_nr"}
N_VARIABLES = {"small": 3, "medium": 7, "large_nr": 18}
LABELS = {
    "GDP": "Real GDP", "CONS": "Consumption", "INV": "Investment", "DEFL": "GDP deflator",
    "FFR": "Federal funds rate", "HOURS": "Hours", "WAGE": "Real compensation", "EMP": "Employment",
    "UNR": "Unemployment rate", "AHE": "Hourly earnings", "CPI": "CPI", "PPI": "PPI, finished goods",
    "COMM": "PPI, commodities", "M1": "M1", "M2": "M2", "SP500": "Share prices", "TB10": "10-year yield",
    "REER": "Trade-weighted dollar",
}
APPENDIX_COLUMNS = [
    ("GLP mode", "paper_plugin"), ("J1 80", "ladder_lsj1_w80"), ("J4 80", "ladder_lsj_h4_w80"),
    ("JR1 80", "ladder_rmsej_h1_w80"), ("JR4 60", "ladder_rmsej_h4_w60"), ("JR4 80", "ladder_rmsej_h4_w80"),
    ("GR1 80", "ladder_rmse_h1_w80"), ("GR4 60", "ladder_rmse_h4_w60"), ("GR4 12", "ladder_rmse_h4_w12"),
]


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def pm(x: float, digits: int = 1) -> str:
    """Signed number in the paper's text-table macros: \\dn{1.2}, \\pos{1.2}, $0.0$."""
    r = round(float(x), digits)
    if r == 0:
        return f"${0:.{digits}f}$"
    return rf"\dn{{{abs(r):.{digits}f}}}" if r < 0 else rf"\pos{{{r:.{digits}f}}}"


def app(x: float, star: bool = False) -> str:
    """Signed number in the appendix format: $-1.2$, ${+}1.2$, $0.0$."""
    r = round(float(x), 1)
    s = "$0.0$" if r == 0 else (f"${r:.1f}$" if r < 0 else f"${{+}}{r:.1f}$")
    return s + ("$^{*}$" if star else "")


def fill(template: str, values: dict[str, str]) -> str:
    """Substitute ``%(key)s`` placeholders (the templates contain literal % signs)."""
    for key, value in values.items():
        template = template.replace(f"%({key})s", value)
    return template


def rows(lines: list[list[str]]) -> str:
    return "\n".join(" & ".join(cells) + r" \\" for cells in lines) + "\n"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
class Size:
    """Comparison outputs of one model size."""

    def __init__(self, size: str):
        root = GLP / SIZES[size]
        self.size, self.root = size, root
        rel = pd.read_csv(root / "comparison" / "relative_rmse_vs_glp.csv")
        self.rel = rel.pivot_table(index=["model", "variable"], columns="horizon_quarters", values="relative_rmse_pct")
        self.glp_rmse = rel.drop_duplicates(["variable", "horizon_quarters"]).pivot(
            index="variable", columns="horizon_quarters", values="baseline_rmse")
        self.scores = pd.read_csv(root / "comparison" / "relative_scores_vs_glp.csv")
        self.crps = self.scores.pivot_table(index=["model", "variable"], columns="horizon_quarters", values="rel_crps_pct")
        self.mcs = pd.read_csv(root / "comparison" / "mcs_sq_error.csv")

    def avg(self, model: str, variable: str) -> float:
        """Relative RMSE (%) averaged over horizons 1-8."""
        return float(self.rel.loc[(model, variable)].mean())

    def at(self, model: str, variable: str, h: int) -> float:
        return float(self.rel.loc[(model, variable), h])

    def crps_avg(self, model: str, variable: str) -> float:
        return float(self.crps.loc[(model, variable)].mean())

    def has(self, model: str) -> bool:
        return model in self.rel.index.get_level_values(0)

    def models(self, prefix: str = "ladder_") -> list[str]:
        return sorted(m for m in self.rel.index.get_level_values(0).unique() if m.startswith(prefix))

    def variables(self) -> list[str]:
        return list(self.glp_rmse.index)

    def hyper(self, rung: str) -> pd.DataFrame:
        """Selected hyperparameters of a ladder rung (or ``paper``), with the log
        posterior gap to the marginal-likelihood mode at the same origin."""
        path = self.root / ("paper" if rung == "paper" else f"ladder/{rung}") / "selected_hyperparameters.csv"
        h = pd.read_csv(path).set_index("forecast_origin")
        mode = pd.read_csv(self.root / "paper" / "selected_hyperparameters.csv").set_index("forecast_origin")["log_mdd"]
        return h.assign(log_ml_gap=h["log_mdd"] - mode.reindex(h.index))


def dm_p(size: Size, model: str, variable: str, column: str = "dm_sq_error_p") -> pd.Series:
    s = size.scores[(size.scores.model == model) & (size.scores.variable == variable)]
    return s.set_index("horizon_quarters")[column].sort_index()


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def tab_ladder_results(s: Size) -> str:
    spec = [
        ("GLP mode, plug-in", "paper_plugin", None),
        ("Joint 1-step score, full sample", "ladder_lsj1_all", "lsj1_all"),
        (r"\quad + window (last 12 quarters)", "ladder_lsj1_w12", "lsj1_w12"),
        (r"\quad + horizon (4 quarters)", "ladder_lsj_h4_w12", "lsj_h4_w12"),
        (r"\quad + GDP only (density)", "ladder_lsm_h4_w12", "lsm_h4_w12"),
        (r"\quad + point forecast (GDP RMSE)", "ladder_rmse_h4_w12", "rmse_h4_w12"),
    ]
    out = []
    for label, model, rung in spec:
        gap = 0 if rung is None else s.hyper(rung)["log_ml_gap"].median()
        out.append([label] + [pm(s.avg(model, v)) for v in ("GDP", "DEFL", "FFR")] + [f"${gap:.0f}$".replace("$-0$", "$0$")])
    return rows(out)


def tab_windows(s: Size) -> str:
    out = []
    for w, label in [(12, "12 quarters"), (20, "20 quarters"), (40, "40 quarters"), (60, "60 quarters"),
                     (80, "80 quarters"), ("all", "Full sample")]:
        g, j = f"ladder_rmse_h4_w{w}" if w != "all" else "ladder_rmse_h4_all", f"ladder_rmsej_h4_w{w}"
        cells = [label] + [pm(s.avg(g, v)) for v in ("GDP", "DEFL")]
        cells += [pm(s.avg(j, v)) for v in ("GDP", "DEFL", "FFR")] if s.has(j) else ["---"] * 3
        out.append(cells)
    return rows(out)


def tab_crps(s: Size) -> str:
    paper = s.hyper("paper")
    out = [["GLP (marginal likelihood)", "$0$", "$0$", "$0$", f"{paper['lambda'].median():.2f}", f"{paper['miu'].median():.2f}", "$0$"]]
    for label, rung in [("GDP-only RMSE, 60q", "rmse_h4_w60"), ("Joint RMSE, 40q", "rmsej_h4_w40"),
                        ("Joint RMSE, 60q", "rmsej_h4_w60"), ("Joint RMSE, 80q", "rmsej_h4_w80"),
                        ("Joint log score, 60q", "lsj_h4_w60")]:
        h = s.hyper(rung)
        out.append([label] + [pm(s.crps_avg(f"ladder_{rung}", v)) for v in ("GDP", "DEFL", "FFR")]
                   + [f"{h['lambda'].median():.2f}", f"{h['miu'].median():.2f}", f"${h['log_ml_gap'].median():.0f}$"])
    return rows(out)


def tab_medium_horizons(s: Size) -> str:
    out = []
    for w in (60, 80):
        for i, h in enumerate((1, 2, 4, 8)):
            m = f"ladder_rmsej_h{h}_w{w}"
            first = f"{w} quarters" if i == 0 else ""
            tuned = f"{h} quarter" + ("s" if h > 1 else "")
            cells = [first, tuned] + [pm(s.at(m, "GDP", e)) for e in (1, 4, 8)] + [pm(s.avg(m, "GDP"))]
            out.append(cells + [pm(s.avg(m, v)) for v in ("INV", "DEFL", "FFR")])
    return rows(out)


def tab_size(sizes: dict[str, Size]) -> str:
    out = []
    for size, s in sizes.items():
        r = s.glp_rmse.loc["GDP"]
        out.append([f"{N_VARIABLES[size]} variables", f"{r[1]:.2f} / {r[4]:.2f} / {r[8]:.2f}"])
    return rows(out)


def tab_mf_results() -> str:
    t = pd.read_csv(MF / "comparison" / "summary_rmse.csv").set_index(["strategy", "group"])
    groups = ["+0 months", "+1 month", "+2 months"]
    out = []
    for label, strategy in [("Marginal likelihood (one EM step)", "ml"), ("GDP RMSE, 12 quarters", "rmse_h4_w12"),
                            ("Joint RMSE, 60 quarters", "rmsej_h4_w60")]:
        for i, g in enumerate(groups):
            r = t.loc[(strategy, g)]
            out.append([label if i == 0 else "", str(i + 1)] + [pm(r[v]) for v in ("GDP", "INVFIX", "CPI", "FF", "UNR", "HRS")])
    return rows(out)


def tab_random_origins(s: Size) -> str:
    spec = [
        ("Last 12 quarters", "w12"), ("12 drawn from the last 80, draw 1", "r12p80"),
        (r"\phantom{12 drawn from the last 80,} draw 2", "r12p80s1"), (r"\phantom{12 drawn from the last 80,} draw 3", "r12p80s2"),
        ("Last 20 quarters", "w20"), ("20 drawn from the last 80", "r20p80"), ("40 drawn from the last 80", "r40p80"),
        ("Last 40 quarters", "w40"), ("Last 80 quarters", "w80"), ("12 drawn from the whole history", "r12"),
    ]
    out = []
    for label, tag in spec:
        out.append([label] + [pm(s.avg(f"ladder_{k}_h4_{tag}", v)) for k in ("rmsej", "rmse") for v in ("GDP", "DEFL")])
    return rows(out)


def tab_appendix(s: Size, spa: pd.DataFrame) -> str:
    """Star: gain over GLP significant at 10% by the Romano-Wolf step-down test across all criteria."""
    p = spa[spa.model_size == s.size].set_index(["variant", "target"])["p_stepm"]
    out = []
    for v in [v for v in LABELS if v in s.variables()]:
        out.append([LABELS[v]] + [app(s.avg(m, v), bool(p.get((m, v), 1.0) < 0.10)) for _, m in APPENDIX_COLUMNS])
    return rows(out)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
FIG_WINDOWS = r"""% Generated by scripts/paper_tables.py from outputs/glp/canonical_*/comparison; joint RMSE criterion, target horizon 4 quarters.
\begin{tikzpicture}
\begin{groupplot}[
    group style={group size=2 by 1, horizontal sep=1.4cm},
    width=0.44\textwidth, height=5.2cm,
    xmin=8, xmax=84, xtick={12,40,60,80},
    xlabel={Evaluation window (quarters)},
    grid=major, major grid style={draw=vizgrid, line width=0.3pt},
    axis line style={draw=vizaxis}, tick style={draw=vizaxis},
    tick label style={font=\footnotesize, text=vizink2}, label style={font=\footnotesize, text=vizink2},
    title style={font=\small},
    every axis plot/.append style={line width=1.2pt, mark size=2.2pt},
  ]
\nextgroupplot[title={GDP}, ylabel={RMSE relative to GLP (\%)}, legend to name=windowlegend, legend columns=3, legend style={font=\footnotesize, draw=none, /tikz/every even column/.append style={column sep=0.6cm}}]
\addplot[vizaxis, line width=0.8pt, forget plot] coordinates {(8,0) (84,0)};
\addplot[color=series1, mark=*] coordinates {%(small,GDP)s};
\addlegendentry{3 variables}
\addplot[color=series2, mark=square*] coordinates {%(medium,GDP)s};
\addlegendentry{7 variables}
\addplot[color=series3, mark=triangle*] coordinates {%(large_nr,GDP)s};
\addlegendentry{18 variables}
\nextgroupplot[title={GDP deflator}]
\addplot[vizaxis, line width=0.8pt, forget plot] coordinates {(8,0) (84,0)};
\addplot[color=series1, mark=*] coordinates {%(small,DEFL)s};
\addplot[color=series2, mark=square*] coordinates {%(medium,DEFL)s};
\addplot[color=series3, mark=triangle*] coordinates {%(large_nr,DEFL)s};
\end{groupplot}
\node[anchor=south] at ($(group c1r1.north)!0.5!(group c2r1.north)+(0,0.55cm)$) {\pgfplotslegendfromname{windowlegend}};
\end{tikzpicture}
"""


def fig_windows(sizes: dict[str, Size]) -> str:
    values = {}
    for size, s in sizes.items():
        for v in ("GDP", "DEFL"):
            pts = [f"({w},{s.avg(f'ladder_rmsej_h4_w{w}', v):.1f})" for w in (12, 40, 60, 80) if s.has(f"ladder_rmsej_h4_w{w}")]
            values[f"{size},{v}"] = " ".join(pts)
    return fill(FIG_WINDOWS, values)


FIG_CROSS_HEAD = r"""% Generated by scripts/paper_tables.py from outputs/glp/canonical_*/comparison; joint RMSE criterion, 60-quarter window, GDP.
\begin{tikzpicture}
\begin{groupplot}[
    group style={group size=1 by 3, vertical sep=1.05cm, x descriptions at=edge bottom},
    width=0.8\textwidth, height=3.6cm,
    colormap name=vizdiverging, point meta min=-15, point meta max=15,
    enlargelimits=false, axis on top,
    xmin=0.5, xmax=8.5, ymin=-0.5, ymax=3.5,
    xtick={1,...,8}, ytick={0,1,2,3}, yticklabels={1,2,4,8},
    y dir=reverse,
    xlabel={Evaluation horizon (quarters)}, ylabel={Tuned for (quarters)},
    tick label style={font=\footnotesize, text=vizink2}, label style={font=\footnotesize, text=vizink2},
    title style={font=\small, yshift=-0.1cm}, axis line style={draw=vizaxis}, tick style={draw=none},
    nodes near coords={\pgfmathprintnumber[fixed, fixed zerofill, precision=1, showpos]{\pgfplotspointmeta}},
    nodes near coords align={center}, every node near coord/.append style={font=\scriptsize, text=vizink1},
  ]
"""
FIG_CROSS_PANEL = {
    "small": r"\nextgroupplot[title={3 variables}]",
    "medium": r"\nextgroupplot[title={7 variables}, colorbar, colorbar style={width=0.25cm, ytick={-15,-10,-5,0,5,10,15}, yticklabel style={font=\scriptsize, text=vizink2}, title={\scriptsize \%}}]",
    "large_nr": r"\nextgroupplot[title={18 variables}]",
}


def fig_crosshorizon(sizes: dict[str, Size]) -> str:
    out = FIG_CROSS_HEAD
    for size, s in sizes.items():
        out += FIG_CROSS_PANEL[size] + "\n"
        out += "\\addplot[matrix plot*, mesh/cols=8, mesh/rows=4, point meta=explicit] table[meta index=2] {\nx y z\n"
        for y, h in enumerate((1, 2, 4, 8)):
            for e in range(1, 9):
                out += f"{e} {y} {s.at(f'ladder_rmsej_h{h}_w60', 'GDP', e):.1f}\n"
        out += "};\n"
    return out + "\\end{groupplot}\n\\end{tikzpicture}\n"


FIG_LAMBDA = r"""% Generated by scripts/paper_tables.py from outputs/glp/canonical_small/ladder/*/selected_hyperparameters.csv (3-variable model).
\begin{tikzpicture}
\begin{axis}[
    width=0.95\textwidth, height=5.6cm, ymode=log, ymin=5e-5, ymax=20,
    xmin=1999.75, xmax=2020, xtick={2000,2004,...,2020}, xticklabel style={/pgf/number format/1000 sep={}},
    ylabel={Overall tightness $\lambda$ (log scale)}, xlabel={Forecast origin},
    grid=major, major grid style={draw=vizgrid, line width=0.3pt},
    axis line style={draw=vizaxis}, tick style={draw=vizaxis},
    tick label style={font=\footnotesize, text=vizink2}, label style={font=\footnotesize, text=vizink2},
    legend style={font=\footnotesize, draw=none, at={(0.5,1.03)}, anchor=south, legend columns=3, /tikz/every even column/.append style={column sep=0.35cm}},
    every axis plot/.append style={line width=1.2pt, mark size=2pt, mark repeat=4},
  ]
\addplot[vizaxis, dashed, line width=0.6pt, forget plot, mark=none] coordinates {(1999.75,5) (2020,5)};
\addplot[vizaxis, dashed, line width=0.6pt, forget plot, mark=none] coordinates {(1999.75,1e-4) (2020,1e-4)};
\addplot[color=series1, mark=*] coordinates {%(mdd)s};
\addlegendentry{Marginal likelihood}
\addplot[color=series2, mark=square*] coordinates {%(rmsej_h4_w60)s};
\addlegendentry{Joint RMSE, 60 quarters}
\addplot[color=series3, mark=triangle*] coordinates {%(rmse_h4_w12)s};
\addlegendentry{GDP RMSE, 12 quarters}
\end{axis}
\end{tikzpicture}
"""


def fig_lambda(s: Size) -> str:
    def path(rung: str) -> str:
        lam = s.hyper(rung)["lambda"]
        t = pd.to_datetime(lam.index)
        return " ".join(f"({d.year + (d.quarter - 1) / 4:.2f},{x:.5f})" for d, x in zip(t, lam))
    return fill(FIG_LAMBDA, {k: path(k) for k in ("mdd", "rmsej_h4_w60", "rmse_h4_w12")})


# --------------------------------------------------------------------------- #
# Numbers quoted in the text
# --------------------------------------------------------------------------- #
def rng(values) -> str:
    v = np.asarray(list(values), dtype=float)
    return f"{v.min():.1f} to {v.max():.1f}"


def mcs_excludes_glp(s: Size, variable: str, variants: list[str] | None, alpha: float = 0.10) -> dict[int, bool]:
    """Whether the MCS at level alpha excludes GLP, by horizon, among GLP and ``variants``
    (all criteria when None). Uses the stored set when all criteria are compared."""
    if variants is None:
        m = s.mcs[(s.mcs.variable == variable) & (s.mcs.variant == "paper")]
        return {int(h): not bool(r) for h, r in zip(m.horizon_quarters, m.in_mcs)}
    out = {}
    panels = {}
    for v in ["paper"] + variants:
        d = "paper" if v == "paper" else f"ladder/{v.removeprefix('ladder_')}"
        p = pd.read_csv(s.root / d / "forecast_panel.csv")
        panels[v] = p[p.variable == variable].set_index(["forecast_origin", "horizon_quarters"])["error_adj"] ** 2
    frame = pd.DataFrame(panels)
    for h, block in frame.groupby(level="horizon_quarters"):
        res = model_confidence_set(block.droplevel("horizon_quarters"), alpha=alpha, block=max(1, int(h)))
        out[int(h)] = not bool(res.loc["paper", "in_mcs"])
    return out


def reserves(add) -> None:
    """Section 8: the 21-variable model with reserves and the monetary base."""
    full, base = GLP / "canonical_large", GLP / "canonical_large_nr"
    if not (full / "comparison" / "rmse_all_variables.csv").exists():
        return
    def rmse(root: Path) -> pd.DataFrame:
        r = pd.read_csv(root / "comparison" / "rmse_all_variables.csv")
        return r.pivot_table(index=["model", "variable"], columns="horizon_quarters", values="rmse")
    f, b = rmse(full), rmse(base)
    for model in ("paper", "ladder_lsj1_w80", "ladder_rmsej_h4_w80"):
        vs_glp = (f.loc[model] / f.loc["paper"] - 1).mul(100).mean(axis=1)
        vs_base = (f.loc[model] / b.loc[model] - 1).mul(100).mean(axis=1).dropna()
        add("8", f"reserves {model}: GDP vs own GLP {vs_glp['GDP']:.1f}, vs same criterion without reserves {vs_base['GDP']:.1f}; "
            f"other variables vs own GLP {rng(vs_glp.drop('GDP'))}")
    roots = GLP / "explosive_roots.csv"
    if roots.exists():
        r = pd.read_csv(roots)
        for run, g in r.groupby("run"):
            big = g[g.max_root > 1.05]
            add("8", f"roots {run}: max {g.max_root.max():.3f}, origins above 1.05: {len(big)} ({', '.join(big.forecast_origin)})")


def numbers(sizes: dict[str, Size], pooled: pd.DataFrame, with_mcs_subsets: bool) -> str:
    s, m, l = sizes["small"], sizes["medium"], sizes["large_nr"]
    L = []

    def add(section: str, text: str) -> None:
        L.append(f"- [{section}] {text}")

    # Section 5: 3-variable model.
    for rung in ("rmse_h4_w12", "rmse_h4_w20", "rmsej_h4_w12"):
        h = s.hyper(rung)
        ratio = (h["objective_value"] / h["objective_at_mdd_mode"]).median()
        add("5.2", f"small {rung}: median in-window loss at the selection / at the ML mode = {ratio:.2f}")
    add("5.2", f"small joint RMSE 40-80q GDP: {rng(s.avg(f'ladder_rmsej_h4_w{w}', 'GDP') for w in (40, 60, 80))}; "
        f"GDP-only 40-80q GDP: {rng(s.avg(f'ladder_rmse_h4_w{w}', 'GDP') for w in (40, 60, 80))}, "
        f"deflator {rng(s.avg(f'ladder_rmse_h4_w{w}', 'DEFL') for w in (40, 60, 80))}")
    for rung in ("mdd", "paper", "rmsej_h4_w60"):
        h = s.hyper(rung)
        add("5.2", f"small {rung}: median lambda {h['lambda'].median():.3f}, mu {h['miu'].median():.3f}")
    for h in (1, 2, 4, 8):
        add("5.3", f"small rmsej_h{h}_w60 GDP avg {s.avg(f'ladder_rmsej_h{h}_w60', 'GDP'):.1f}; "
            f"lsm_h{h}_w60 DEFL avg {s.avg(f'ladder_lsm_h{h}_w60', 'DEFL'):.1f}")
    for h in (1, 2, 4, 8):
        p = dm_p(s, f"ladder_rmsej_h{h}_w60", "GDP")
        add("5.5", f"small rmsej_h{h}_w60 GDP DM p by horizon: " + ", ".join(f"{k}:{v:.3f}" for k, v in p.items())
            + f"; min Holm {dm_p(s, f'ladder_rmsej_h{h}_w60', 'GDP', 'dm_sq_error_p_holm').min():.3f}")
    gdp_holm = s.scores[(s.scores.variable == "GDP") & s.scores.model.str.startswith("ladder_")]["dm_sq_error_p_holm"].min()
    add("5.5", f"small: smallest Holm-adjusted DM p for GDP across all criteria and horizons = {gdp_holm:.3f}")
    n_crit = s.mcs[(s.mcs.variable == "GDP") & (s.mcs.horizon_quarters == 1)].variant.nunique() - 1
    add("5.5", f"small: number of criteria in the all-criteria MCS (excluding GLP) = {n_crit}; "
        f"GLP excluded at GDP horizons {[h for h, e in mcs_excludes_glp(s, 'GDP', None).items() if e]}")
    if with_mcs_subsets:
        rmse_only = [mm for mm in s.models() if mm.startswith(("ladder_rmse_", "ladder_rmsej_"))]
        ex = mcs_excludes_glp(s, "GDP", rmse_only)
        add("5.5", f"small: MCS of GLP + the {len(rmse_only)} RMSE criteria excludes GLP for GDP at horizons {[h for h, e in ex.items() if e]}")
    gdp_only = [mm for mm in s.models() if mm.startswith(("ladder_rmse_h", "ladder_lsm_"))]
    sig = s.scores[s.scores.model.isin(gdp_only) & (s.scores.variable == "DEFL") & (s.scores.dm_crps_p_holm < 0.05)]
    add("5.5", f"small: GDP-only criteria with a Holm-significant (5%) deflator CRPS difference: {sorted(sig.model.unique())}")
    fam = pooled[(pooled.model_size == "small") & (pooled.variable == "GDP") & pooled.variant.str.match(r"ladder_rmsej_h\d_w(40|60|80)$")]
    add("5.5", f"small pooled, 12 joint 40-80q: GDP {rng(fam.rel_rmse_pct)}, p {fam.pooled_p.min():.3f}-{fam.pooled_p.max():.3f}, "
        f"Holm (all criteria) min {fam.pooled_p_holm.min():.3f}; Holm within family min {holm_adjust(fam.pooled_p.to_numpy()).min():.3f}")
    add("5.5", f"small pooled EWC, 12 joint 40-80q: p {fam.pooled_p_ewc.min():.3f}-{fam.pooled_p_ewc.max():.3f}; WPE {fam.pooled_p_wpe.min():.3f}-{fam.pooled_p_wpe.max():.3f}")
    for size in ("medium", "large_nr"):
        pm_ = pooled[(pooled.model_size == size) & (pooled.variable == "GDP")].nsmallest(3, "pooled_p_ewc")
        add("5.5", f"{size} pooled GDP, three smallest EWC p: " + "; ".join(
            f"{r.variant} {r.rel_rmse_pct:.1f}% EWC={r.pooled_p_ewc:.3f} NW={r.pooled_p:.3f}" for r in pm_.itertuples()))
    spa_family = pd.read_csv(GLP / "spa_family.csv")
    spa = pd.read_csv(GLP / "spa_tests.csv")
    for target in ("GDP", "joint"):
        f = spa_family[spa_family.target == target].pivot(index="model_size", columns="block", values="p_spa")
        add("5.5", f"SPA p ({target}) by size x block: " + "; ".join(f"{s} " + "/".join(f"{v:.2f}" for v in f.loc[s]) for s in f.index))
        sm = spa[spa.target == target].groupby("model_size").p_stepm.min()
        add("5.5", f"StepM smallest p ({target}): " + ", ".join(f"{s} {v:.3f}" for s, v in sm.items()))
    sig = spa[spa.p_stepm < 0.10]
    add("5.5", "StepM < 0.10 (any target): " + "; ".join(f"{r.model_size} {r.target} {r.variant} {r.rel_rmse_pct:.1f}% p={r.p_stepm:.3f}" for r in sig.itertuples()))
    u = spa[(spa.model_size == "medium") & (spa.target == "GDP") & spa.variant.isin(["ladder_rmsej_h1_w60", "ladder_rmsej_h1_w80"])]
    add("5.5", "medium joint h=1 GDP uniform-test p: " + ", ".join(f"{r.variant} {r.p_uniform:.3f}" for r in u.itertuples()))
    sig_any = pooled[pooled.pooled_p_holm < 0.10]
    add("5.5/App", "pooled tests Holm-significant at 10%: " + "; ".join(f"{r.model_size} {r.variant} {r.variable} {r.rel_rmse_pct:.1f}%" for r in sig_any.itertuples()))

    # Section 6: model size.
    best3 = max(((mm, -s.avg(mm, "GDP")) for mm in s.models() if "_w4" in mm or "_w6" in mm or "_w8" in mm), key=lambda x: x[1])
    add("6", f"small best GDP gain at 40-80q windows: {best3[0]} {-best3[1]:.1f}")
    for size, sz in sizes.items():
        g = [mm for mm in sz.models() if mm.startswith("ladder_rmse_h")]
        add("6", f"{size} GDP-only RMSE (all windows/targets): GDP {rng(sz.avg(mm, 'GDP') for mm in g)}, deflator {rng(sz.avg(mm, 'DEFL') for mm in g)}")
        gl = [mm for mm in sz.models() if mm.startswith(("ladder_rmse_h", "ladder_lsm_")) and "_r" not in mm.rsplit("_", 1)[1]]
        add("6", f"{size} GDP-only RMSE+density (fixed windows, n={len(gl)}): GDP {rng(sz.avg(mm, 'GDP') for mm in gl)}, "
            f"deflator {rng(sz.avg(mm, 'DEFL') for mm in gl)}, #GDP gains {sum(sz.avg(mm, 'GDP') < 0 for mm in gl)}")
    for w in (60, 80):
        mm = f"ladder_rmsej_h1_w{w}"
        add("6", f"medium {mm}: GDP avg {m.avg(mm, 'GDP'):.1f}, by horizon {rng(m.rel.loc[(mm, 'GDP')])}, "
            + ", ".join(f"{v} {m.avg(mm, v):.1f}" for v in ("CONS", "INV", "DEFL", "FFR", "HOURS", "WAGE")))
    for w in (60, 80):
        for h in (4, 8):
            mm = f"ladder_rmsej_h{h}_w{w}"
            add("6", f"medium {mm}: GDP h1 {m.at(mm, 'GDP', 1):.1f}, INV {m.avg(mm, 'INV'):.1f}, HOURS {m.avg(mm, 'HOURS'):.1f}")
    for w in (60, 80):
        mm = f"ladder_lsj1_w{w}"
        others = {v: m.avg(mm, v) for v in m.variables() if v != "GDP"}
        add("6", f"medium {mm}: GDP {m.avg(mm, 'GDP'):.1f}, worst other {max(others.values()):.1f}")
    p80 = [dm_p(m, mm, "GDP").loc[[7, 8]] for mm in m.models() if mm.endswith("_w80")]
    allp = pd.concat(p80)
    add("6", f"medium 80q criteria GDP DM p at h=7-8: {allp.min():.4f}-{allp.max():.3f}")
    add("6", f"medium: GLP excluded from all-criteria MCS for GDP at horizons {[h for h, e in mcs_excludes_glp(m, 'GDP', None).items() if e]}; "
        f"smallest GDP Holm p {m.scores[(m.scores.variable == 'GDP') & m.scores.model.str.startswith('ladder_')]['dm_sq_error_p_holm'].min():.3f}")
    add("6", f"medium rmse_h1_w60 GDP {m.avg('ladder_rmse_h1_w60', 'GDP'):.1f}; GDP-only deflator by criterion: "
        + ", ".join(f"{mm.removeprefix('ladder_')} {m.avg(mm, 'DEFL'):.1f}" for mm in m.models() if mm.startswith("ladder_rmse_h") and "_r" not in mm.rsplit("_", 1)[1]))
    sigm = m.scores[m.scores.model.str.startswith(("ladder_rmse_h", "ladder_lsm_")) & (m.scores.variable == "DEFL") & (m.scores.dm_crps_p_holm < 0.05)]
    add("6", f"medium GDP-only criteria with Holm-significant deflator CRPS differences: {sorted(sigm.model.unique())}")
    for w in (12, 60, 80):
        gaps = l.hyper(f"lsj1_w{w}")["log_ml_gap"]
        add("6", f"large_nr lsj1_w{w}: median log-ML gap to GLP mode {gaps.median():.2f}, min {gaps.min():.2f}")
    for h in (1, 2, 4, 8):
        mm = f"ladder_rmsej_h{h}_w80"
        add("6", f"large_nr {mm}: mean over 18 variables {np.mean([l.avg(mm, v) for v in l.variables()]):.2f}, GDP {l.avg(mm, 'GDP'):.1f}, "
            f"EMP {l.avg(mm, 'EMP'):.1f}, HOURS {l.avg(mm, 'HOURS'):.1f}")
    add("6", f"large_nr rmsej_h1_w60 GDP {l.avg('ladder_rmsej_h1_w60', 'GDP'):.1f}, by horizon {rng(l.rel.loc[('ladder_rmsej_h1_w60', 'GDP')])}")
    sigl = l.scores[l.scores.model.str.startswith(("ladder_rmse_h", "ladder_lsm_")) & (l.scores[["dm_sq_error_p_holm", "dm_crps_p_holm"]].min(axis=1) < 0.05) & (l.scores.rel_rmse_pct > 0)]
    add("6", f"large_nr GDP-only: variables with a Holm-significant (5%, RMSE or CRPS, some horizon) loss: {sorted(sigl.variable.unique())}")
    add("6", f"large_nr: GLP excluded from all-criteria MCS for GDP at horizons {[h for h, e in mcs_excludes_glp(l, 'GDP', None).items() if e]}; "
        f"smallest GDP Holm p {l.scores[(l.scores.variable == 'GDP') & l.scores.model.str.startswith('ladder_')]['dm_sq_error_p_holm'].min():.3f}")

    # Section 7: mixed-frequency VAR.
    mf = pd.read_csv(MF / "comparison" / "relative_scores.csv")
    for strategy in ("rmse_h4_w12", "rmsej_h4_w60"):
        x = mf[mf.strategy == strategy]
        now = x[(x.variable == "GDP") & (x.horizon_quarters == 1)].set_index("group")["relative_rmse_pct"]
        add("7", f"MF {strategy}: GDP nowcast (h=1) by month " + ", ".join(f"{g}: {v:.1f}" for g, v in now.items()))
        gdp = x[x.variable == "GDP"]
        add("7", f"MF {strategy}: GDP smallest DM p {gdp.dm_p_sq_error.min():.3f} at h={int(gdp.loc[gdp.dm_p_sq_error.idxmin(), 'horizon_quarters'])}, "
            f"smallest Holm {gdp.holm_p_sq_error.min():.3f}")
        for v in ("CPI", "FF", "UNR", "HRS", "TB"):
            xv = x[x.variable == v]
            add("7", f"MF {strategy} {v}: avg {xv.groupby('group').relative_rmse_pct.mean().round(1).to_dict()}, "
                f"Holm RMSE<0.05 at {int((xv.holm_p_sq_error < 0.05).sum())} cells, Holm CRPS<0.05 at {int((xv.holm_p_crps < 0.05).sum())} cells")
    mf_spa = pd.read_csv(MF / "comparison" / "spa_tests.csv")
    for r in mf_spa[mf_spa.p_stepm < 0.10].itertuples():
        add("7", f"MF StepM<0.10: {r.group} {r.target} {r.variant} {r.rel_rmse_pct:.1f}% p_stepm={r.p_stepm:.3f} p_uniform={r.p_uniform:.3f}")
    g = mf_spa[(mf_spa.target == "GDP") & (mf_spa.variant == "rmsej_h4_w60")]
    add("7", "MF joint GDP StepM p by month: " + ", ".join(f"{r.group} {r.p_stepm:.3f}" for r in g.itertuples()))
    mcs = pd.read_csv(MF / "comparison" / "mcs.csv")
    out_paper = mcs[(mcs.strategy == "paper") & (mcs.variable == "GDP") & (mcs.metric == "sq_error") & ~mcs.in_mcs]
    add("7", f"MF: cells where the MCS excludes the published prior for GDP: {len(out_paper)}")
    em = pd.read_csv(MF / "em_check.csv")
    first = em[em.iteration == 1]
    t = np.abs(first[[c for c in first if c.startswith("score_lambda")]].to_numpy() / first[[c for c in first if c.startswith("score_se_")]].to_numpy())
    add("7", f"EM check: |t| > 1.96 in {(t > 1.96).sum()} of {t.size}; median q_gain by iteration "
        f"{em.groupby('iteration').q_gain.median().round(2).to_dict()}; ESS range {em.ess.min():.1f}-{em.ess.max():.1f}")

    # Section 8: robustness.
    for size, sz in sizes.items():
        plug = [sz.avg("paper_plugin", v) for v in sz.variables()]
        add("8", f"{size} plug-in vs GLP: {rng(plug)} (averaged over horizons, all variables)")
    ws = pd.read_csv(GLP / "window_selection_scores.csv")
    add("8", "window selection rows: " + "; ".join(f"{r['model_size']} {r.iloc[0]}: GDP {r['GDP']:.1f}" for _, r in ws.iterrows() if str(r.iloc[0]).startswith(("real", "select", "rule"))))
    reserves(add)
    m40 = m.avg("ladder_rmsej_h4_r40p80", "GDP"), m.avg("ladder_rmsej_h4_r40p80", "CONS")
    add("8", f"medium rmsej_h4_r40p80: GDP {m40[0]:.1f}, CONS {m40[1]:.1f}; last 40 GDP {m.avg('ladder_rmsej_h4_w40', 'GDP'):.1f}, CONS {m.avg('ladder_rmsej_h4_w40', 'CONS'):.1f}")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "paper")
    p.add_argument("--skip-mcs-subsets", action="store_true", help="Skip the slower MCS among a subset of criteria.")
    args = p.parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    sizes = {k: Size(k) for k in SIZES}
    pooled = pd.read_csv(GLP / "pooled_tests.csv")
    spa = pd.read_csv(GLP / "spa_tests.csv")
    files = {
        "tab_ladder_results.tex": tab_ladder_results(sizes["small"]),
        "tab_windows.tex": tab_windows(sizes["small"]),
        "tab_crps.tex": tab_crps(sizes["small"]),
        "tab_medium_horizons.tex": tab_medium_horizons(sizes["medium"]),
        "tab_size.tex": tab_size(sizes),
        "tab_mf_results.tex": tab_mf_results(),
        "tab_random_origins.tex": tab_random_origins(sizes["small"]),
        "tab_app_medium.tex": tab_appendix(sizes["medium"], spa),
        "tab_app_large_nr.tex": tab_appendix(sizes["large_nr"], spa),
        "fig_windows.tex": fig_windows(sizes),
        "fig_crosshorizon.tex": fig_crosshorizon(sizes),
        "fig_lambda.tex": fig_lambda(sizes["small"]),
        "numbers.md": "# Numbers quoted in the paper, by section\n\n" + numbers(sizes, pooled, not args.skip_mcs_subsets),
    }
    for name, text in files.items():
        (out / name).write_text(text)
    print(f"Wrote {len(files)} files to {out}")


if __name__ == "__main__":
    main()
