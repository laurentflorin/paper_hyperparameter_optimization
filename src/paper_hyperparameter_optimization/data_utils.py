from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from .config import (
    DOWNLOAD_METADATA_PATH,
    LATEST_PANEL_PATH,
    PAPER_ACTUAL_VINTAGE,
    PAPER_ESTIMATION_START,
    PROCESSED_DATA_DIR,
    QUARTERLY_SERIES,
    RAW_DATA_DIR,
    REALTIME_PANEL_PATH,
    SERIES_SPECS,
    SERIES_BY_ID,
    forecast_origin_dates,
    origin_group,
    serialise_series_specs,
)


ALFRED_GRAPH_URL = "https://alfred.stlouisfed.org/graph/alfredgraph.csv"
FRED_GRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
REQUEST_TIMEOUT = 120


@dataclass(frozen=True)
class DownloadTask:
    series_id: str
    vintage_date: pd.Timestamp


def ensure_data_directories() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def alfred_vintage_url(series_id: str, vintage_date: pd.Timestamp) -> str:
    return f"{ALFRED_GRAPH_URL}?id={series_id}&vintage_date={vintage_date.strftime('%Y-%m-%d')}"


def fred_latest_url(series_id: str) -> str:
    return f"{FRED_GRAPH_URL}?id={series_id}"


def _read_csv_from_url(url: str, session: requests.Session) -> pd.DataFrame:
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text))
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.rename(columns={frame.columns[0]: "observation_date", frame.columns[1]: "value"})
    frame["observation_date"] = pd.to_datetime(frame["observation_date"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["value"]).sort_values("observation_date")
    return frame


def download_series_vintage(series_id: str, vintage_date: pd.Timestamp, session: requests.Session) -> pd.DataFrame:
    frame = _read_csv_from_url(alfred_vintage_url(series_id, vintage_date), session)
    frame["series_id"] = series_id
    frame["vintage_date"] = pd.Timestamp(vintage_date)
    return frame[["series_id", "vintage_date", "observation_date", "value"]]


def download_latest_series(series_id: str, session: requests.Session) -> pd.DataFrame:
    frame = _read_csv_from_url(fred_latest_url(series_id), session)
    frame["series_id"] = series_id
    return frame[["series_id", "observation_date", "value"]]


def backcast_from_latest(
    vintage_frame: pd.DataFrame,
    latest_frame: pd.DataFrame,
    start_date: pd.Timestamp = PAPER_ESTIMATION_START,
) -> pd.DataFrame:
    vintage = vintage_frame.sort_values("observation_date").copy()
    latest = latest_frame.sort_values("observation_date").copy()
    if vintage.empty:
        return vintage

    first_observation = vintage["observation_date"].min()
    if first_observation <= start_date:
        return vintage[vintage["observation_date"] >= start_date].copy()

    latest = latest[latest["observation_date"] >= start_date].copy()
    if latest.empty:
        return vintage

    latest_series = latest.set_index("observation_date")["value"]
    vintage_series = vintage.set_index("observation_date")["value"]
    anchor_date = first_observation

    if anchor_date not in latest_series.index:
        return vintage[vintage["observation_date"] >= start_date].copy()

    history = vintage_series.copy()
    prior_dates = latest_series.index[latest_series.index < anchor_date]
    current_value = history.loc[anchor_date]

    for date in reversed(prior_dates):
        next_date = latest_series.index[latest_series.index.get_loc(date) + 1]
        growth_ratio = latest_series.loc[next_date] / latest_series.loc[date]
        if pd.isna(growth_ratio) or growth_ratio == 0:
            continue
        current_value = current_value / growth_ratio
        history.loc[date] = current_value

    history = history.sort_index()
    out = history.to_frame("value").reset_index().rename(columns={"index": "observation_date"})
    out["series_id"] = vintage["series_id"].iloc[0]
    out["vintage_date"] = vintage["vintage_date"].iloc[0]
    out = out[out["observation_date"] >= start_date]
    return out[["series_id", "vintage_date", "observation_date", "value"]]


def _download_task(task: DownloadTask) -> pd.DataFrame:
    with requests.Session() as session:
        session.headers.update({"User-Agent": "paper-hyperparameter-optimization/1.0"})
        return download_series_vintage(task.series_id, task.vintage_date, session)


def download_realtime_panel(
    output_path: Path = REALTIME_PANEL_PATH,
    latest_output_path: Path = LATEST_PANEL_PATH,
    metadata_path: Path = DOWNLOAD_METADATA_PATH,
    forecast_origins: Iterable[pd.Timestamp] | None = None,
    actual_vintage: pd.Timestamp = PAPER_ACTUAL_VINTAGE,
    max_workers: int = 8,
) -> tuple[Path, Path]:
    ensure_data_directories()

    origins = list(forecast_origins or forecast_origin_dates())
    vintage_dates = sorted({*origins, pd.Timestamp(actual_vintage)})
    tasks = [DownloadTask(spec.series_id, vintage_date) for spec in SERIES_SPECS for vintage_date in vintage_dates]

    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_download_task, task): task for task in tasks}
        for future in as_completed(future_map):
            frames.append(future.result())

    realtime_panel = pd.concat(frames, ignore_index=True).sort_values(
        ["series_id", "vintage_date", "observation_date"]
    )

    with requests.Session() as session:
        session.headers.update({"User-Agent": "paper-hyperparameter-optimization/1.0"})
        latest_frames = [download_latest_series(spec.series_id, session) for spec in SERIES_SPECS]
    latest_panel = pd.concat(latest_frames, ignore_index=True).sort_values(["series_id", "observation_date"])

    for series_id in ("PCEC96", "FPIC1"):
        latest_series = latest_panel[latest_panel["series_id"] == series_id]
        updated_vintages = []
        for vintage_date, vintage_frame in realtime_panel[realtime_panel["series_id"] == series_id].groupby("vintage_date"):
            updated_vintages.append(backcast_from_latest(vintage_frame, latest_series))
        realtime_panel = realtime_panel[realtime_panel["series_id"] != series_id]
        realtime_panel = pd.concat([realtime_panel, *updated_vintages], ignore_index=True)

    realtime_panel = realtime_panel.sort_values(["series_id", "vintage_date", "observation_date"])
    realtime_panel.to_csv(output_path, index=False, compression="gzip")
    latest_panel.to_csv(latest_output_path, index=False, compression="gzip")

    metadata = {
        "forecast_origin_start": min(origins).strftime("%Y-%m-%d"),
        "forecast_origin_end": max(origins).strftime("%Y-%m-%d"),
        "actual_vintage": pd.Timestamp(actual_vintage).strftime("%Y-%m-%d"),
        "n_forecast_origins": len(origins),
        "series": serialise_series_specs(),
        "origin_groups": [
            {"forecast_origin": origin.strftime("%Y-%m-%d"), "group": origin_group(origin)}
            for origin in origins
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path, latest_output_path


def load_realtime_panel(path: Path = REALTIME_PANEL_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path, compression="gzip")
    frame["vintage_date"] = pd.to_datetime(frame["vintage_date"])
    frame["observation_date"] = pd.to_datetime(frame["observation_date"])
    return frame


def load_latest_panel(path: Path = LATEST_PANEL_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path, compression="gzip")
    frame["observation_date"] = pd.to_datetime(frame["observation_date"])
    return frame


def pivot_vintage_panel(panel: pd.DataFrame, vintage_date: pd.Timestamp) -> pd.DataFrame:
    vintage_date = pd.Timestamp(vintage_date)
    subset = panel.loc[panel["vintage_date"] == vintage_date, ["series_id", "observation_date", "value"]].copy()
    if subset.empty:
        raise FileNotFoundError(f"No data available for vintage {vintage_date:%Y-%m-%d}.")
    wide = subset.pivot(index="observation_date", columns="series_id", values="value").sort_index()
    rename_map = {series_id: SERIES_BY_ID[series_id].paper_code for series_id in wide.columns}
    wide = wide.rename(columns=rename_map)
    return wide


def build_model_input_frames(panel: pd.DataFrame, vintage_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = pivot_vintage_panel(panel, vintage_date)
    quarterly = wide[[spec.paper_code for spec in QUARTERLY_SERIES]].dropna(how="all")
    monthly = wide[[spec.paper_code for spec in SERIES_SPECS if spec.frequency == "M"]].dropna(how="all")
    return quarterly, monthly


def build_quarterly_evaluation_frame(panel: pd.DataFrame, vintage_date: pd.Timestamp) -> pd.DataFrame:
    quarterly, monthly = build_model_input_frames(panel, vintage_date)
    quarterly = quarterly.copy()
    quarterly.index = pd.PeriodIndex(quarterly.index, freq="Q")

    monthly = monthly.copy()
    monthly = monthly.groupby(pd.PeriodIndex(monthly.index, freq="Q")).mean()

    combined = quarterly.join(monthly, how="outer").sort_index()
    combined.index.name = "quarter"
    return combined
