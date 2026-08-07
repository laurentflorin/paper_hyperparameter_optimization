"""Configuration for the Giannone-Lenza-Primiceri (2015) prior-selection study.

This package builds a workflow that is SEPARATE from the Schorfheide-Song MF-VAR
workflow in ``paper_hyperparameter_optimization``. It compares the marginal-
likelihood prior selection of Giannone, Lenza and Primiceri (2015) -- as
implemented natively in the ``covbayesvar`` package -- against three
Bayesian-optimization (Mango) hyperparameter strategies ported from ``MBFVAR``
(``mango`` / MDD, ``mango_rmse`` and ``mango_rmse_random``).

The BVAR is the single-frequency quarterly hierarchical BVAR of
"Prior Selection for Vector Autoregressions" (2015), estimated with
``covbayesvar.large_bvar``.

NOTE ON THE VARIABLE UNIVERSE
-----------------------------
The FRED/ALFRED series identifiers below are a best-effort mapping of the
paper's small (3), medium (7) and large model configurations. A few of the
large-model series (real effective exchange rate, non-borrowed reserves,
reserves) have limited real-time ALFRED coverage before the 1990s, so the
effective large-model sample begins where every selected series/vintage is
available. The exact 22-variable large set from the paper's appendix should be
verified against the original replication files before drawing quantitative
conclusions; the mapping here is intentionally easy to edit.

The stock-price block deserves a special note: historical ALFRED vintages for
the copyrighted ``SP500`` series are no longer publicly available, and the old
Stooq CSV fallback is now protected by a browser challenge. The large model
therefore uses the public OECD monthly U.S. share-price index
``SPASTT01USM661N`` as an explicit proxy while keeping the short code ``SP500``
so downstream comparisons still line up with the paper's variable list.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GLPSeriesSpec:
    """One macro series in the GLP variable universe.

    Attributes
    ----------
    series_id:
        FRED / ALFRED identifier used for the real-time download.
    code:
        Short mnemonic used inside the model and output tables.
    label:
        Human-readable description.
    frequency:
        ``"Q"`` for natively quarterly series, ``"M"`` for monthly series that
        are aggregated to quarterly by simple averaging (Stock & Watson, 2008).
    transform:
        ``"log"`` applies ``100 * log(level)`` (the ``covbayesvar`` convention,
        matching ``covbayesvar.large_bvar.transform_data``); ``"lin"`` keeps the
        series in levels (used for variables already expressed in annualized
        rates such as interest rates and the unemployment rate).
    """

    series_id: str
    code: str
    label: str
    frequency: str
    transform: str


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

GLP_REALTIME_PANEL_PATH = PROCESSED_DATA_DIR / "glp_realtime_panel.csv.gz"
GLP_LATEST_PANEL_PATH = PROCESSED_DATA_DIR / "glp_latest_panel.csv.gz"
GLP_DOWNLOAD_METADATA_PATH = PROCESSED_DATA_DIR / "glp_download_metadata.json"

GLP_ALFRED_CACHE_DIR = RAW_DATA_DIR / "glp_alfred_realtime"
GLP_FRED_CACHE_DIR = RAW_DATA_DIR / "glp_fred_latest"


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


# --------------------------------------------------------------------------- #
# Sample, recursive out-of-sample window, and estimation settings.
# --------------------------------------------------------------------------- #
# Full GLP (2015) estimation sample.
GLP_SAMPLE_START = pd.Timestamp("1959-01-01")
GLP_SAMPLE_END = pd.Timestamp("2008-12-31")

# Recursive real-time out-of-sample forecast origins (quarter-end vintage dates).
# The default window sits inside the span with reliable real-time ALFRED
# coverage; override with --start/--end on the scripts.
GLP_FORECAST_START = pd.Timestamp("2000-03-31")
GLP_FORECAST_END = pd.Timestamp("2019-12-31")
# Fixed later vintage used to score the recursive forecasts.
GLP_ACTUAL_VINTAGE = pd.Timestamp("2023-01-01")

GLP_LAGS = 5
MAX_FORECAST_HORIZON_QUARTERS = 8
EVAL_HORIZONS_QUARTERS = [1, 2, 4, 8]

# GLP prior / estimation switches (see covbayesvar.large_bvar.set_priors).
GLP_SUR = 1          # sum-of-coefficients ("theta") prior on.
GLP_NOC = 1          # dummy-initial-observation ("miu") prior on.
GLP_MNPSI = 0        # keep psi fixed at the AR(1) residual variances.
GLP_MNALPHA = 0      # keep the lag-decay exponent alpha fixed at 2.
GLP_HYPERPRIORS = 1  # use Gamma hyperpriors (the GLP hierarchical prior).
GLP_VC = 1.0e7       # prior variance of the intercept.

# MCMC settings for the full predictive densities.
GLP_NDRAWS = 20_000
GLP_NDRAWS_DISCARD = 10_000
GLP_MCMC_CONST = 1.0

# --------------------------------------------------------------------------- #
# Hyperparameter search space for the Mango strategies.
# Bounds equal the GLP maximization bounds (set_priors MIN/MAX).
# --------------------------------------------------------------------------- #
GLP_PARAM_SPACE_BOUNDS: dict[str, tuple[float, float]] = {
    "lambda": (1.0e-4, 5.0),
    "theta": (1.0e-4, 50.0),
    "miu": (1.0e-4, 50.0),
}
# Gamma hyperprior modes / sds used by GLP (mirrors set_priors defaults).
GLP_HYPERPRIOR_MODE = {"lambda": 0.2, "theta": 1.0, "miu": 1.0}
GLP_HYPERPRIOR_SD = {"lambda": 0.4, "theta": 1.0, "miu": 1.0}


# --------------------------------------------------------------------------- #
# Variable universe (nested: small subset of medium subset of large).
# --------------------------------------------------------------------------- #
SERIES_SPECS: tuple[GLPSeriesSpec, ...] = (
    # --- Small (3): the canonical monetary-policy VAR -----------------------
    GLPSeriesSpec("GDPC1", "GDP", "Real Gross Domestic Product", "Q", "log"),
    GLPSeriesSpec("GDPDEF", "DEFL", "GDP Deflator", "Q", "log"),
    GLPSeriesSpec("FEDFUNDS", "FFR", "Federal Funds Rate", "M", "lin"),
    # --- Medium (+4 = 7): the Smets-Wouters (2007) DSGE block ---------------
    GLPSeriesSpec("PCECC96", "CONS", "Real Personal Consumption Expenditures", "Q", "log"),
    GLPSeriesSpec("GPDIC1", "INV", "Real Gross Private Domestic Investment", "Q", "log"),
    GLPSeriesSpec("HOANBS", "HOURS", "Nonfarm Business Sector: Hours Worked", "Q", "log"),
    GLPSeriesSpec("COMPRNFB", "WAGE", "Nonfarm Business Sector: Real Compensation per Hour", "Q", "log"),
    # --- Large (+14): labour, prices, money/credit, financial ---------------
    GLPSeriesSpec("PAYEMS", "EMP", "Total Nonfarm Employment", "M", "log"),
    GLPSeriesSpec("UNRATE", "UNR", "Unemployment Rate", "M", "lin"),
    GLPSeriesSpec("AHETPI", "AHE", "Average Hourly Earnings (Production Workers)", "M", "log"),
    GLPSeriesSpec("CPIAUCSL", "CPI", "Consumer Price Index (All Urban)", "M", "log"),
    GLPSeriesSpec("PPIFGS", "PPI", "Producer Price Index (Finished Goods)", "M", "log"),
    GLPSeriesSpec("PPIACO", "COMM", "Producer Price Index (All Commodities)", "M", "log"),
    GLPSeriesSpec("M1SL", "M1", "M1 Money Stock", "M", "log"),
    GLPSeriesSpec("M2SL", "M2", "M2 Money Stock", "M", "log"),
    GLPSeriesSpec("BOGMBASE", "MBASE", "Monetary Base", "M", "log"),
    GLPSeriesSpec("TOTRESNS", "TOTRES", "Total Reserves of Depository Institutions", "M", "log"),
    GLPSeriesSpec("NONBORRES", "NBRES", "Non-Borrowed Reserves", "M", "lin"),
    GLPSeriesSpec(
        "SPASTT01USM661N",
        "SP500",
        "Financial Market: Share Prices for United States (proxy for S&P 500)",
        "M",
        "log",
    ),
    GLPSeriesSpec("GS10", "TB10", "10-Year Treasury Bond Yield", "M", "lin"),
    GLPSeriesSpec("TWEXMMTH", "REER", "Trade-Weighted US Dollar Index (Major Currencies)", "M", "log"),
)

SMALL_CODES: tuple[str, ...] = ("GDP", "DEFL", "FFR")
MEDIUM_CODES: tuple[str, ...] = SMALL_CODES + ("CONS", "INV", "HOURS", "WAGE")
LARGE_CODES: tuple[str, ...] = MEDIUM_CODES + (
    "EMP",
    "UNR",
    "AHE",
    "CPI",
    "PPI",
    "COMM",
    "M1",
    "M2",
    "MBASE",
    "TOTRES",
    "NBRES",
    "SP500",
    "TB10",
    "REER",
)

MODEL_SIZE_CODES: dict[str, tuple[str, ...]] = {
    "small": SMALL_CODES,
    "medium": MEDIUM_CODES,
    "large": LARGE_CODES,
}

SERIES_BY_CODE: dict[str, GLPSeriesSpec] = {spec.code: spec for spec in SERIES_SPECS}
SERIES_BY_ID: dict[str, GLPSeriesSpec] = {spec.series_id: spec for spec in SERIES_SPECS}


def model_series(size: str) -> tuple[GLPSeriesSpec, ...]:
    """Return the ordered series specs for a model size (small/medium/large)."""
    if size not in MODEL_SIZE_CODES:
        raise ValueError(f"Unknown model size {size!r}. Choose from {sorted(MODEL_SIZE_CODES)}.")
    codes = MODEL_SIZE_CODES[size]
    return tuple(SERIES_BY_CODE[code] for code in codes)


def model_codes(size: str) -> list[str]:
    return list(MODEL_SIZE_CODES[size])


def forecast_origin_dates(
    start: pd.Timestamp = GLP_FORECAST_START,
    end: pd.Timestamp = GLP_FORECAST_END,
) -> pd.DatetimeIndex:
    """Quarter-end real-time forecast origins (also used as ALFRED vintages)."""
    return pd.date_range(start=start, end=end, freq="QE")


def serialise_series_specs(size: str | None = None) -> list[dict[str, object]]:
    specs = SERIES_SPECS if size is None else model_series(size)
    return [asdict(spec) for spec in specs]


def mbfvar_style_bounds() -> dict[str, tuple[float, float]]:
    """Alias kept for symmetry with the MBFVAR workflow's DEFAULT_PARAM_SPACE_BOUNDS."""
    return dict(GLP_PARAM_SPACE_BOUNDS)


# Transform code applied to a level series once it is at quarterly frequency.
def apply_transform(levels: "pd.Series", transform: str) -> "pd.Series":
    if transform == "log":
        return 100.0 * np.log(levels)
    if transform == "lin":
        return levels.astype(float)
    raise ValueError(f"Unknown transform {transform!r}; expected 'log' or 'lin'.")
