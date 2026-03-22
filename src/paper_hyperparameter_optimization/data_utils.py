from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
import subprocess
import time
from typing import Callable, Iterable

import pandas as pd

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
STOOQ_SP500_MONTHLY_URL = "https://stooq.com/q/d/l/?s=%5Espx&i=m"
REQUEST_TIMEOUT_SECONDS = 120
MAX_REQUEST_ATTEMPTS = 4
RETRY_BACKOFF_SECONDS = 2.0
ALFRED_CACHE_DIR = RAW_DATA_DIR / "alfred_realtime"
FRED_CACHE_DIR = RAW_DATA_DIR / "fred_latest"


class DataDownloadError(RuntimeError):
    """Raised when a FRED/ALFRED download fails after retries."""


def ensure_data_directories() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ALFRED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FRED_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def alfred_vintage_url(series_id: str, vintage_date: pd.Timestamp) -> str:
    return f"{ALFRED_GRAPH_URL}?id={series_id}&vintage_date={vintage_date.strftime('%Y-%m-%d')}"


def fred_latest_url(series_id: str) -> str:
    return f"{FRED_GRAPH_URL}?id={series_id}"


def realtime_cache_path(series_id: str, vintage_date: pd.Timestamp) -> Path:
    return ALFRED_CACHE_DIR / series_id / f"{pd.Timestamp(vintage_date):%Y-%m-%d}.csv.gz"


def latest_cache_path(series_id: str) -> Path:
    return FRED_CACHE_DIR / f"{series_id}.csv.gz"


def _retry_delay_seconds(attempt: int) -> float:
    return RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))


def _write_cached_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression="gzip")


def _read_cached_frame(path: Path, date_columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, compression="gzip")
    for column in date_columns:
        frame[column] = pd.to_datetime(frame[column])
    return frame


def _read_csv_from_url(url: str) -> pd.DataFrame:
    text = _download_text_with_curl(url)
    frame = pd.read_csv(StringIO(text))
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.rename(columns={frame.columns[0]: "observation_date", frame.columns[1]: "value"})
    frame["observation_date"] = pd.to_datetime(frame["observation_date"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["value"]).sort_values("observation_date")
    return frame


def _download_text_with_curl(url: str) -> str:
    last_error: Exception | None = None
    for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
        try:
            completed = subprocess.run(
                [
                    "curl",
                    "-L",
                    "--silent",
                    "--show-error",
                    url,
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            return completed.stdout
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            last_error = exc

        if attempt == MAX_REQUEST_ATTEMPTS:
            assert last_error is not None
            raise DataDownloadError(f"Failed to download {url}") from last_error

        time.sleep(_retry_delay_seconds(attempt))

    raise AssertionError("unreachable")


def download_sp500_stooq_monthly() -> pd.DataFrame:
    text = _download_text_with_curl(STOOQ_SP500_MONTHLY_URL)
    frame = pd.read_csv(StringIO(text))
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.rename(columns={"Date": "observation_date", "Close": "value"})
    frame["observation_date"] = pd.to_datetime(frame["observation_date"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["value"]).sort_values("observation_date")
    frame = frame.loc[frame["observation_date"] >= PAPER_ESTIMATION_START].copy()
    return frame[["observation_date", "value"]]


def download_series_vintage(series_id: str, vintage_date: pd.Timestamp) -> pd.DataFrame:
    if series_id == "SP500":
        frame = download_sp500_stooq_monthly()
        frame = frame.loc[frame["observation_date"] <= pd.Timestamp(vintage_date)].copy()
    else:
        frame = _read_csv_from_url(alfred_vintage_url(series_id, vintage_date))
    frame["series_id"] = series_id
    frame["vintage_date"] = pd.Timestamp(vintage_date)
    return frame[["series_id", "vintage_date", "observation_date", "value"]]


def download_latest_series(series_id: str) -> pd.DataFrame:
    if series_id == "SP500":
        frame = download_sp500_stooq_monthly()
    else:
        frame = _read_csv_from_url(fred_latest_url(series_id))
    frame["series_id"] = series_id
    return frame[["series_id", "observation_date", "value"]]


def _load_or_download_realtime_vintage(
    series_id: str,
    vintage_date: pd.Timestamp,
    *,
    force: bool = False,
) -> tuple[pd.DataFrame, bool]:
    path = realtime_cache_path(series_id, vintage_date)
    if path.exists() and not force:
        return _read_cached_frame(path, ["vintage_date", "observation_date"]), True

    frame = download_series_vintage(series_id, vintage_date)
    _write_cached_frame(frame, path)
    return frame, False


def _load_or_download_latest_series(
    series_id: str,
    *,
    force: bool = False,
) -> tuple[pd.DataFrame, bool]:
    path = latest_cache_path(series_id)
    if path.exists() and not force:
        return _read_cached_frame(path, ["observation_date"]), True

    frame = download_latest_series(series_id)
    _write_cached_frame(frame, path)
    return frame, False


def backcast_from_reference(
    vintage_frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
    start_date: pd.Timestamp = PAPER_ESTIMATION_START,
) -> pd.DataFrame:
    vintage = vintage_frame.sort_values("observation_date").copy()
    reference = reference_frame.sort_values("observation_date").copy()
    if vintage.empty:
        return vintage

    vintage = vintage[vintage["observation_date"] >= start_date].copy()
    reference = reference[reference["observation_date"] >= start_date].copy()
    if vintage.empty or reference.empty:
        return vintage

    first_observation = vintage["observation_date"].min()
    if first_observation <= start_date:
        return vintage

    vintage_series = vintage.set_index("observation_date")["value"]
    reference_series = reference.set_index("observation_date")["value"]
    overlapping_dates = vintage_series.index.intersection(reference_series.index)
    if overlapping_dates.empty:
        return vintage

    anchor_date = overlapping_dates.min()
    history = vintage_series.copy()
    prior_dates = reference_series.index[reference_series.index < anchor_date]
    current_value = history.loc[anchor_date]

    for date in reversed(prior_dates):
        next_date = reference_series.index[reference_series.index.get_loc(date) + 1]
        growth_ratio = reference_series.loc[next_date] / reference_series.loc[date]
        if pd.isna(growth_ratio) or growth_ratio == 0:
            history.loc[date] = current_value
            continue
        current_value = current_value / growth_ratio
        history.loc[date] = current_value

    history = history.sort_index()
    out = history.to_frame("value").reset_index().rename(columns={"index": "observation_date"})
    out["series_id"] = vintage["series_id"].iloc[0]
    out["vintage_date"] = vintage["vintage_date"].iloc[0]
    return out[["series_id", "vintage_date", "observation_date", "value"]]


def repair_short_history_vintages(
    series_panel: pd.DataFrame,
    latest_series_frame: pd.DataFrame | None = None,
    start_date: pd.Timestamp = PAPER_ESTIMATION_START,
) -> pd.DataFrame:
    repaired_vintages: list[pd.DataFrame] = []
    for _, vintage_frame in series_panel.groupby("vintage_date", sort=True):
        vintage = vintage_frame.sort_values("observation_date").copy()
        if vintage.empty:
            repaired_vintages.append(vintage)
            continue

        first_observation = vintage["observation_date"].min()
        if first_observation <= start_date:
            repaired_vintages.append(vintage[vintage["observation_date"] >= start_date].copy())
            continue

        reference_frame = next(
            (
                prior
                for prior in reversed(repaired_vintages)
                if not prior.empty and prior["observation_date"].min() <= start_date
            ),
            None,
        )
        if reference_frame is None:
            reference_frame = latest_series_frame
        if reference_frame is None:
            repaired_vintages.append(vintage[vintage["observation_date"] >= start_date].copy())
            continue

        repaired_vintages.append(backcast_from_reference(vintage, reference_frame, start_date=start_date))

    return pd.concat(repaired_vintages, ignore_index=True)


def download_realtime_panel(
    output_path: Path = REALTIME_PANEL_PATH,
    latest_output_path: Path = LATEST_PANEL_PATH,
    metadata_path: Path = DOWNLOAD_METADATA_PATH,
    forecast_origins: Iterable[pd.Timestamp] | None = None,
    actual_vintage: pd.Timestamp = PAPER_ACTUAL_VINTAGE,
    max_workers: int = 1,
    progress_callback: Callable[[str], None] | None = None,
    force: bool = False,
) -> tuple[Path, Path]:
    ensure_data_directories()
    if max_workers != 1:
        report = progress_callback or (lambda message: None)
        report("Ignoring max_workers because downloads now run sequentially by series for resumable caching.")
    else:
        report = progress_callback or (lambda message: None)

    origins = list(forecast_origin_dates() if forecast_origins is None else forecast_origins)
    vintage_dates = sorted({*origins, pd.Timestamp(actual_vintage)})

    report(
        "Preparing resumable downloads "
        f"for {len(SERIES_SPECS)} series across {len(vintage_dates)} vintages."
    )

    frames: list[pd.DataFrame] = []
    for series_idx, spec in enumerate(SERIES_SPECS, start=1):
        cached_vintages = 0 if force else sum(realtime_cache_path(spec.series_id, vintage_date).exists() for vintage_date in vintage_dates)
        report(
            f"Series {series_idx}/{len(SERIES_SPECS)} {spec.series_id}: "
            f"{cached_vintages}/{len(vintage_dates)} vintages already cached."
        )

        series_frames: list[pd.DataFrame] = []
        ready = 0
        next_report = 0
        for vintage_date in vintage_dates:
            try:
                frame, from_cache = _load_or_download_realtime_vintage(
                    spec.series_id,
                    vintage_date,
                    force=force,
                )
            except DataDownloadError as exc:
                raise RuntimeError(
                    "Failed to download "
                    f"{spec.series_id} for vintage {vintage_date:%Y-%m-%d} "
                    f"after {MAX_REQUEST_ATTEMPTS} attempts. "
                    "Rerunning the command will resume from the cached vintages."
                ) from exc

            series_frames.append(frame)
            ready += 1
            progress_pct = int((ready / len(vintage_dates)) * 100)
            if ready == len(vintage_dates) or progress_pct >= next_report:
                source = "cached/downloaded"
                if from_cache:
                    source = "cached"
                elif force:
                    source = "redownloaded"
                else:
                    source = "downloaded"
                report(
                    f"Series {spec.series_id}: {ready}/{len(vintage_dates)} vintages ready "
                    f"({progress_pct}%, latest {source}: {vintage_date:%Y-%m-%d})."
                )
                next_report = progress_pct + 10

        frames.extend(series_frames)

    realtime_panel = pd.concat(frames, ignore_index=True).sort_values(
        ["series_id", "vintage_date", "observation_date"]
    )

    report(f"Preparing latest FRED series for backfilling and evaluation ({len(SERIES_SPECS)} series).")
    latest_frames: list[pd.DataFrame] = []
    for series_idx, spec in enumerate(SERIES_SPECS, start=1):
        try:
            frame, from_cache = _load_or_download_latest_series(spec.series_id, force=force)
        except DataDownloadError as exc:
            raise RuntimeError(
                f"Failed to download latest FRED series {spec.series_id} after {MAX_REQUEST_ATTEMPTS} attempts. "
                "Rerunning the command will resume from the cached files."
            ) from exc
        latest_frames.append(frame)
        source = "cached" if from_cache else ("redownloaded" if force else "downloaded")
        report(f"Latest series {series_idx}/{len(SERIES_SPECS)} {spec.series_id}: {source}.")
    latest_panel = pd.concat(latest_frames, ignore_index=True).sort_values(["series_id", "observation_date"])

    report("Backfilling incomplete ALFRED histories for PCEC96 and FPIC1.")
    for series_id in ("PCEC96", "FPIC1"):
        series_panel = realtime_panel[realtime_panel["series_id"] == series_id].copy()
        latest_series = latest_panel[latest_panel["series_id"] == series_id].copy()
        realtime_panel = realtime_panel[realtime_panel["series_id"] != series_id]
        repaired_panel = repair_short_history_vintages(
            series_panel=series_panel,
            latest_series_frame=latest_series,
        )
        realtime_panel = pd.concat([realtime_panel, repaired_panel], ignore_index=True)

    realtime_panel = realtime_panel.sort_values(["series_id", "vintage_date", "observation_date"])
    report(f"Writing real-time panel to {output_path}.")
    realtime_panel.to_csv(output_path, index=False, compression="gzip")
    report(f"Writing latest panel to {latest_output_path}.")
    latest_panel.to_csv(latest_output_path, index=False, compression="gzip")

    metadata = {
        "forecast_origin_start": min(origins).strftime("%Y-%m-%d"),
        "forecast_origin_end": max(origins).strftime("%Y-%m-%d"),
        "actual_vintage": pd.Timestamp(actual_vintage).strftime("%Y-%m-%d"),
        "n_forecast_origins": len(origins),
        "force_refresh": force,
        "realtime_cache_dir": str(ALFRED_CACHE_DIR),
        "latest_cache_dir": str(FRED_CACHE_DIR),
        "series": serialise_series_specs(),
        "origin_groups": [
            {"forecast_origin": origin.strftime("%Y-%m-%d"), "group": origin_group(origin)}
            for origin in origins
        ],
    }
    report(f"Writing download metadata to {metadata_path}.")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report("Download pipeline finished.")
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
    monthly_ids = {spec.series_id for spec in SERIES_SPECS if spec.frequency == "M"}
    quarterly_ids = {spec.series_id for spec in QUARTERLY_SERIES}
    monthly_mask = subset["series_id"].isin(monthly_ids)
    quarterly_mask = subset["series_id"].isin(quarterly_ids)

    # FRED monthly/quarterly series and Stooq market data use different day-of-period
    # conventions. Align them before pivoting so one calendar period maps to one row.
    subset.loc[monthly_mask, "observation_date"] = (
        subset.loc[monthly_mask, "observation_date"]
        .dt.to_period("M")
        .dt.to_timestamp(how="end")
        .dt.normalize()
    )
    subset.loc[quarterly_mask, "observation_date"] = (
        subset.loc[quarterly_mask, "observation_date"]
        .dt.to_period("Q")
        .dt.to_timestamp(how="end")
        .dt.normalize()
    )
    wide = subset.pivot(index="observation_date", columns="series_id", values="value").sort_index()
    rename_map = {series_id: SERIES_BY_ID[series_id].paper_code for series_id in wide.columns}
    wide = wide.rename(columns=rename_map)
    return wide


def _complete_window(frame: pd.DataFrame) -> pd.DataFrame:
    complete_rows = frame.notna().all(axis=1)
    if not complete_rows.any():
        raise ValueError("No fully observed sample window is available for the selected vintage.")
    first_complete = complete_rows[complete_rows].index[0]
    last_complete = complete_rows[complete_rows].index[-1]
    window = frame.loc[first_complete:last_complete].copy()
    if window.isna().any().any():
        raise ValueError("The selected vintage contains internal gaps inside the complete sample window.")
    return window


def build_model_input_frames(panel: pd.DataFrame, vintage_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = pivot_vintage_panel(panel, vintage_date)
    quarterly = wide[[spec.paper_code for spec in QUARTERLY_SERIES]].dropna(how="all")
    monthly = wide[[spec.paper_code for spec in SERIES_SPECS if spec.frequency == "M"]].dropna(how="all")

    quarterly = _complete_window(quarterly)
    monthly = _complete_window(monthly)

    # Keep all fully observed monthly rows, even within an incomplete quarter.
    # The quarterly block only needs to be capped so it fits inside the monthly sample.
    max_quarters = len(monthly) // 3
    if max_quarters == 0:
        raise ValueError("Not enough monthly history is available after trimming the complete sample window.")
    quarterly = quarterly.iloc[: min(len(quarterly), max_quarters)].copy()
    return quarterly, monthly


def build_quarterly_evaluation_frame(panel: pd.DataFrame, vintage_date: pd.Timestamp) -> pd.DataFrame:
    wide = pivot_vintage_panel(panel, vintage_date)
    quarterly = wide[[spec.paper_code for spec in QUARTERLY_SERIES]].dropna(how="all")
    monthly = wide[[spec.paper_code for spec in SERIES_SPECS if spec.frequency == "M"]].dropna(how="all")
    quarterly = quarterly.copy()
    quarterly.index = pd.PeriodIndex(quarterly.index, freq="Q")

    monthly = monthly.copy()
    monthly = monthly.groupby(pd.PeriodIndex(monthly.index, freq="Q")).mean()

    combined = quarterly.join(monthly, how="outer").sort_index()
    combined.index.name = "quarter"
    return combined
