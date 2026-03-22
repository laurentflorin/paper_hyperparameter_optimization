from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SeriesSpec:
    series_id: str
    paper_code: str
    label: str
    frequency: str
    mbfvar_transform: int
    evaluation_transform: str


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

REALTIME_PANEL_PATH = PROCESSED_DATA_DIR / "realtime_panel.csv.gz"
LATEST_PANEL_PATH = PROCESSED_DATA_DIR / "latest_panel.csv.gz"
DOWNLOAD_METADATA_PATH = PROCESSED_DATA_DIR / "download_metadata.json"

PAPER_FORECAST_START = pd.Timestamp("1997-07-31")
PAPER_FORECAST_END = pd.Timestamp("2010-01-31")
PAPER_ACTUAL_VINTAGE = pd.Timestamp("2012-01-31")
PAPER_ESTIMATION_START = pd.Timestamp("1967-01-01")

MAX_FORECAST_HORIZON_MONTHS = 24
MAX_FORECAST_HORIZON_QUARTERS = 8

PAPER_NSIM = 20_000
PAPER_NBURN_PERC = 0.5
PAPER_THINING = 1
PAPER_NLAGS = [6]
PAPER_HYPERPARAMETERS = [0.09, 4.30, 1.0, 2.70, 4.30]
PAPER_TEMPORAL_AGGREGATION = "mean"

# Explosive VAR handling
# The paper hyperparameters (especially λ1=0.09) can produce explosive VAR draws
# during MCMC sampling. The MBFVAR package attempts up to max_it_stable draws
# to find non-explosive coefficients. If all attempts fail, it raises an error.
#
# The default max_it_stable=1000 may be insufficient for weak priors like λ1=0.09.
# We increase this to 10000 to give more attempts at finding stable draws.
#
# Alternative solutions:
# 1. Increase λ1 for stronger shrinkage (reduces explosive draws)
# 2. Use MDD-optimized hyperparameters instead of fixed paper values
# 3. Accept that some origins may fail with very weak priors
MAX_IT_STABLE = 10_000  # Increased from default 1000

DEFAULT_PARAM_SPACE_BOUNDS = {
    "lambda1_1": (0.001, 20.0),
    "lambda2_1": (0.01, 10.0),
    "lambda4_1": (0.01, 10.0),
    "lambda5_1": (0.01, 10.0),
}

SERIES_SPECS = (
    SeriesSpec("GDPC1", "GDP", "Gross Domestic Product", "Q", 0, "growth"),
    SeriesSpec("FPIC1", "INVFIX", "Fixed Investment", "Q", 0, "growth"),
    SeriesSpec("GCEC1", "GOV", "Government Expenditures", "Q", 0, "growth"),
    SeriesSpec("UNRATE", "UNR", "Unemployment Rate", "M", 1, "level"),
    SeriesSpec("AWHI", "HRS", "Hours Worked", "M", 0, "growth"),
    SeriesSpec("CPIAUCSL", "CPI", "Consumer Price Index", "M", 0, "growth"),
    SeriesSpec("INDPRO", "IP", "Industrial Production Index", "M", 0, "growth"),
    SeriesSpec("PCEC96", "PCE", "Personal Consumption Expenditure", "M", 0, "growth"),
    SeriesSpec("FEDFUNDS", "FF", "Federal Funds Rate", "M", 1, "level"),
    SeriesSpec("GS10", "TB", "Treasury Bond Yield", "M", 1, "level"),
    SeriesSpec("SP500", "SP500", "S&P 500", "M", 0, "growth"),
)

QUARTERLY_SERIES = tuple(spec for spec in SERIES_SPECS if spec.frequency == "Q")
MONTHLY_SERIES = tuple(spec for spec in SERIES_SPECS if spec.frequency == "M")
SERIES_BY_ID = {spec.series_id: spec for spec in SERIES_SPECS}
SERIES_BY_CODE = {spec.paper_code: spec for spec in SERIES_SPECS}

MBFVAR_TRANSFORMS = [
    np.array([spec.mbfvar_transform for spec in QUARTERLY_SERIES], dtype=int),
    np.array([spec.mbfvar_transform for spec in MONTHLY_SERIES], dtype=int),
]


def forecast_origin_dates(
    start: pd.Timestamp = PAPER_FORECAST_START,
    end: pd.Timestamp = PAPER_FORECAST_END,
) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="ME")


def origin_group(origin_date: pd.Timestamp) -> str:
    month_mod = origin_date.month % 3
    if month_mod == 1:
        return "+0 months"
    if month_mod == 2:
        return "+1 month"
    return "+2 months"


def origin_group_slug(origin_date: pd.Timestamp) -> str:
    return origin_group(origin_date).replace(" ", "_").replace("+", "plus_")


def serialise_series_specs() -> list[dict[str, object]]:
    return [asdict(spec) for spec in SERIES_SPECS]
