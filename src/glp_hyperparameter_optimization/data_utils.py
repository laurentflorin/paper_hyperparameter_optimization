"""Real-time (ALFRED) data pipeline for the GLP (2015) replication.

This mirrors the resumable ALFRED/FRED download approach used by the
Schorfheide-Song workflow but is self-contained and specialized for the GLP
quarterly variable universe: monthly series are aggregated to quarterly by
simple averaging (Stock & Watson, 2008) and each series is transformed with the
``covbayesvar`` convention (``100 * log`` for level series, identity for rates).
"""

from __future__ import annotations

import json
import subprocess
import time
from io import StringIO
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .config import (
    GLP_ACTUAL_VINTAGE,
    GLP_ALFRED_CACHE_DIR,
    GLP_DOWNLOAD_METADATA_PATH,
    GLP_FRED_CACHE_DIR,
    GLP_LATEST_PANEL_PATH,
    GLP_REALTIME_PANEL_PATH,
    GLP_SAMPLE_START,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SERIES_BY_ID,
    SERIES_SPECS,
    apply_transform,
    forecast_origin_dates,
    model_series,
    resolve_project_path,
    serialise_series_specs,
)

ALFRED_GRAPH_URL = "https://alfred.stlouisfed.org/graph/alfredgraph.csv"
FRED_GRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
STOOQ_SP500_MONTHLY_URL = "https://stooq.com/q/d/l/?s=%5Espx&i=m"
REQUEST_TIMEOUT_SECONDS = 120
MAX_REQUEST_ATTEMPTS = 4
RETRY_BACKOFF_SECONDS = 2.0


class DataDownloadError(RuntimeError):
    """Raised when a FRED/ALFRED download fails after retries."""


class EmptyRemoteDataError(DataDownloadError):
    """Raised when a remote FRED/ALFRED endpoint returns an empty or unusable body."""


# --------------------------------------------------------------------------- #
# Low-level download helpers.
# --------------------------------------------------------------------------- #
def ensure_data_directories() -> None:
    for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, GLP_ALFRED_CACHE_DIR, GLP_FRED_CACHE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def _retry_delay_seconds(attempt: int) -> float:
    return RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))


def _download_text_with_curl(url: str) -> str:
    last_error: Exception | None = None
    for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
        try:
            completed = subprocess.run(
                ["curl", "-L", "--silent", "--show-error", url],
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


def _parse_remote_csv_text(text: str, url: str) -> pd.DataFrame:
    if not text.strip():
        raise EmptyRemoteDataError(f"Empty response body from {url}")

    try:
        frame = pd.read_csv(StringIO(text))
    except pd.errors.EmptyDataError as exc:
        raise EmptyRemoteDataError(f"No CSV columns were returned by {url}") from exc

    if frame.shape[1] < 2:
        raise EmptyRemoteDataError(
            f"Expected at least 2 CSV columns from {url}, got {frame.shape[1]}."
        )

    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.rename(columns={frame.columns[0]: "observation_date", frame.columns[1]: "value"})
    frame["observation_date"] = pd.to_datetime(frame["observation_date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["observation_date", "value"]).sort_values("observation_date")
    if frame.empty:
        raise EmptyRemoteDataError(f"No usable observations were returned by {url}")
    return frame


def _read_csv_from_url(url: str) -> pd.DataFrame:
    text = _download_text_with_curl(url)
    return _parse_remote_csv_text(text, url)


def download_sp500_stooq_monthly() -> pd.DataFrame:
    text = _download_text_with_curl(STOOQ_SP500_MONTHLY_URL)
    if text.lstrip().startswith("<!DOCTYPE html"):
        raise DataDownloadError(
            "Stooq returned an anti-bot HTML challenge instead of the historical SP500 CSV."
        )
    frame = pd.read_csv(StringIO(text))
    frame.columns = [str(column).strip() for column in frame.columns]
    if "Date" not in frame.columns or "Close" not in frame.columns:
        raise DataDownloadError(
            "Stooq did not return the expected Date/Close CSV columns for the SP500 fallback."
        )
    frame = frame.rename(columns={"Date": "observation_date", "Close": "value"})
    frame["observation_date"] = pd.to_datetime(frame["observation_date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["observation_date", "value"]).sort_values("observation_date")
    frame = frame.loc[frame["observation_date"] >= GLP_SAMPLE_START].copy()
    return frame[["observation_date", "value"]]


def alfred_vintage_url(series_id: str, vintage_date: pd.Timestamp) -> str:
    return f"{ALFRED_GRAPH_URL}?id={series_id}&vintage_date={pd.Timestamp(vintage_date).strftime('%Y-%m-%d')}"


def fred_latest_url(series_id: str) -> str:
    return f"{FRED_GRAPH_URL}?id={series_id}"


def realtime_cache_path(series_id: str, vintage_date: pd.Timestamp) -> Path:
    return GLP_ALFRED_CACHE_DIR / series_id / f"{pd.Timestamp(vintage_date):%Y-%m-%d}.csv.gz"


def latest_cache_path(series_id: str) -> Path:
    return GLP_FRED_CACHE_DIR / f"{series_id}.csv.gz"


def _write_cached_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression="gzip")


def _read_cached_frame(path: Path, date_columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, compression="gzip")
    for column in date_columns:
        frame[column] = pd.to_datetime(frame[column])
    return frame


def download_series_vintage(
    series_id: str,
    vintage_date: pd.Timestamp,
    *,
    force_latest_fallback: bool = False,
) -> pd.DataFrame:
    if series_id == "SP500":
        frame = download_sp500_stooq_monthly()
        frame = frame.loc[frame["observation_date"] <= pd.Timestamp(vintage_date)].copy()
    else:
        try:
            frame = _read_csv_from_url(alfred_vintage_url(series_id, vintage_date))
        except EmptyRemoteDataError:
            # Some ALFRED vintages return a zero-byte body for valid historical
            # series rather than an HTTP error. When that happens, fall back to
            # the latest available FRED history and truncate it at the requested
            # vintage so the downloader can still build a complete panel.
            frame, _ = _load_or_download_latest_series(series_id, force=force_latest_fallback)
            frame = frame.loc[frame["observation_date"] <= pd.Timestamp(vintage_date)].copy()
            if frame.empty:
                raise DataDownloadError(
                    f"ALFRED returned no vintage data for {series_id} at {pd.Timestamp(vintage_date):%Y-%m-%d}, "
                    "and the latest FRED history also has no observations by that date."
                )
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


def _load_or_download_realtime_vintage(series_id: str, vintage_date: pd.Timestamp, *, force: bool = False):
    path = realtime_cache_path(series_id, vintage_date)
    if path.exists() and not force:
        return _read_cached_frame(path, ["vintage_date", "observation_date"]), True
    frame = download_series_vintage(series_id, vintage_date, force_latest_fallback=force)
    _write_cached_frame(frame, path)
    return frame, False


def _load_or_download_latest_series(series_id: str, *, force: bool = False):
    path = latest_cache_path(series_id)
    if path.exists() and not force:
        return _read_cached_frame(path, ["observation_date"]), True
    frame = download_latest_series(series_id)
    _write_cached_frame(frame, path)
    return frame, False


# --------------------------------------------------------------------------- #
# Panel download orchestration.
# --------------------------------------------------------------------------- #
def _existing_outputs_match(
    output_path: Path,
    latest_output_path: Path,
    metadata_path: Path,
    series_ids: list[str],
    origins: list[pd.Timestamp],
    actual_vintage: pd.Timestamp,
) -> bool:
    if not (output_path.exists() and latest_output_path.exists() and metadata_path.exists()):
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("forecast_origin_start") == min(origins).strftime("%Y-%m-%d")
        and metadata.get("forecast_origin_end") == max(origins).strftime("%Y-%m-%d")
        and metadata.get("actual_vintage") == pd.Timestamp(actual_vintage).strftime("%Y-%m-%d")
        and metadata.get("n_forecast_origins") == len(origins)
        and set(metadata.get("series_ids", [])) == set(series_ids)
    )


def download_glp_realtime_panel(
    *,
    series_ids: Iterable[str] | None = None,
    output_path: Path = GLP_REALTIME_PANEL_PATH,
    latest_output_path: Path = GLP_LATEST_PANEL_PATH,
    metadata_path: Path = GLP_DOWNLOAD_METADATA_PATH,
    forecast_origins: Iterable[pd.Timestamp] | None = None,
    actual_vintage: pd.Timestamp = GLP_ACTUAL_VINTAGE,
    progress_callback: Callable[[str], None] | None = None,
    force: bool = False,
) -> tuple[Path, Path]:
    """Download real-time ALFRED vintages for the GLP variable universe and the
    latest FRED series, storing long-format panels + metadata."""
    output_path = resolve_project_path(output_path)
    latest_output_path = resolve_project_path(latest_output_path)
    metadata_path = resolve_project_path(metadata_path)
    ensure_data_directories()
    report = progress_callback or (lambda message: None)

    specs = SERIES_SPECS if series_ids is None else tuple(SERIES_BY_ID[s] for s in series_ids)
    resolved_ids = [spec.series_id for spec in specs]
    origins = list(forecast_origin_dates() if forecast_origins is None else forecast_origins)
    vintage_dates = sorted({*origins, pd.Timestamp(actual_vintage)})

    if not force and _existing_outputs_match(
        output_path, latest_output_path, metadata_path, resolved_ids, origins, pd.Timestamp(actual_vintage)
    ):
        report("Skipping download: cached GLP panels already match this request.")
        return output_path, latest_output_path

    report(f"Downloading {len(specs)} series across {len(vintage_dates)} vintages (resumable).")
    frames: list[pd.DataFrame] = []
    for series_index, spec in enumerate(specs, start=1):
        cached = 0 if force else sum(realtime_cache_path(spec.series_id, v).exists() for v in vintage_dates)
        report(f"Series {series_index}/{len(specs)} {spec.series_id}: {cached}/{len(vintage_dates)} vintages cached.")
        for vintage_date in vintage_dates:
            try:
                frame, _ = _load_or_download_realtime_vintage(spec.series_id, vintage_date, force=force)
            except DataDownloadError as exc:
                raise RuntimeError(
                    f"Failed to download {spec.series_id} for vintage {vintage_date:%Y-%m-%d}. "
                    "Rerun to resume from the cached vintages."
                ) from exc
            frames.append(frame)

    realtime_panel = pd.concat(frames, ignore_index=True).sort_values(["series_id", "vintage_date", "observation_date"])

    report(f"Downloading latest FRED series ({len(specs)}).")
    latest_frames: list[pd.DataFrame] = []
    for spec in specs:
        try:
            frame, _ = _load_or_download_latest_series(spec.series_id, force=force)
        except DataDownloadError as exc:
            raise RuntimeError(f"Failed to download latest FRED series {spec.series_id}.") from exc
        latest_frames.append(frame)
    latest_panel = pd.concat(latest_frames, ignore_index=True).sort_values(["series_id", "observation_date"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    realtime_panel.to_csv(output_path, index=False, compression="gzip")
    latest_panel.to_csv(latest_output_path, index=False, compression="gzip")

    metadata = {
        "forecast_origin_start": min(origins).strftime("%Y-%m-%d"),
        "forecast_origin_end": max(origins).strftime("%Y-%m-%d"),
        "actual_vintage": pd.Timestamp(actual_vintage).strftime("%Y-%m-%d"),
        "n_forecast_origins": len(origins),
        "series_ids": resolved_ids,
        "series": serialise_series_specs(),
        "force_refresh": force,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report("GLP download pipeline finished.")
    return output_path, latest_output_path


# --------------------------------------------------------------------------- #
# Loading and quarterly panel construction.
# --------------------------------------------------------------------------- #
def load_glp_realtime_panel(path: Path = GLP_REALTIME_PANEL_PATH) -> pd.DataFrame:
    path = resolve_project_path(path)
    frame = pd.read_csv(path, compression="gzip")
    frame["vintage_date"] = pd.to_datetime(frame["vintage_date"])
    frame["observation_date"] = pd.to_datetime(frame["observation_date"])
    return frame


def load_glp_latest_panel(path: Path = GLP_LATEST_PANEL_PATH) -> pd.DataFrame:
    path = resolve_project_path(path)
    frame = pd.read_csv(path, compression="gzip")
    frame["observation_date"] = pd.to_datetime(frame["observation_date"])
    return frame


def build_quarterly_levels(panel: pd.DataFrame, vintage_date: pd.Timestamp, size: str) -> pd.DataFrame:
    """Wide quarterly LEVEL frame (untransformed) for one vintage and model size.

    Quarterly series are used as-is; monthly series are averaged within the
    quarter (Stock & Watson, 2008). The frame is indexed by a quarterly Period.
    """
    vintage_date = pd.Timestamp(vintage_date)
    specs = model_series(size)
    subset = panel.loc[panel["vintage_date"] == vintage_date, ["series_id", "observation_date", "value"]]
    columns: dict[str, pd.Series] = {}
    for spec in specs:
        series = subset.loc[subset["series_id"] == spec.series_id, ["observation_date", "value"]]
        if series.empty:
            columns[spec.code] = pd.Series(dtype=float)
            continue
        series = series.set_index("observation_date")["value"].sort_index()
        quarters = pd.PeriodIndex(series.index, freq="Q")
        if spec.frequency == "M":
            quarterly = series.groupby(quarters).mean()
        else:
            quarterly = series.groupby(quarters).last()
        columns[spec.code] = quarterly
    wide = pd.DataFrame(columns).sort_index()
    wide.index.name = "quarter"
    # Preserve the model's column order.
    return wide[[spec.code for spec in specs]]


def transform_quarterly_levels(levels: pd.DataFrame, size: str) -> pd.DataFrame:
    specs = {spec.code: spec for spec in model_series(size)}
    transformed = levels.copy()
    for code in transformed.columns:
        transformed[code] = apply_transform(levels[code], specs[code].transform)
    return transformed


def _complete_window(frame: pd.DataFrame) -> pd.DataFrame:
    complete_rows = frame.notna().all(axis=1)
    if not complete_rows.any():
        raise ValueError("No fully observed quarterly window is available for the selected vintage/model size.")
    first = complete_rows[complete_rows].index[0]
    last = complete_rows[complete_rows].index[-1]
    window = frame.loc[first:last].copy()
    if window.isna().any().any():
        raise ValueError("The selected vintage has interior gaps inside the complete quarterly window.")
    return window


def build_glp_estimation_matrix(
    panel: pd.DataFrame,
    vintage_date: pd.Timestamp,
    size: str,
) -> tuple[np.ndarray, list[str], pd.PeriodIndex]:
    """Return ``(y, codes, quarter_index)`` for estimation at ``vintage_date``.

    ``y`` is the transformed, complete-window quarterly matrix (rows = quarters,
    columns = model variables in canonical order).
    """
    levels = build_quarterly_levels(panel, vintage_date, size)
    transformed = transform_quarterly_levels(levels, size)
    window = _complete_window(transformed)
    codes = list(window.columns)
    return window.to_numpy(dtype=float), codes, pd.PeriodIndex(window.index, freq="Q")


def build_glp_actual_frame(panel: pd.DataFrame, actual_vintage: pd.Timestamp, size: str) -> pd.DataFrame:
    """Transformed quarterly frame from the evaluation vintage (used to score
    forecasts). Indexed by quarterly Period, columns in canonical order."""
    levels = build_quarterly_levels(panel, actual_vintage, size)
    transformed = transform_quarterly_levels(levels, size)
    return transformed.dropna(how="all")
