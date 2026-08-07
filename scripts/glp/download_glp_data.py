"""Download real-time ALFRED/FRED data for the GLP (2015) replication."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from glp_hyperparameter_optimization.config import (
    GLP_ACTUAL_VINTAGE,
    GLP_DOWNLOAD_METADATA_PATH,
    GLP_FORECAST_END,
    GLP_FORECAST_START,
    GLP_LATEST_PANEL_PATH,
    GLP_REALTIME_PANEL_PATH,
    forecast_origin_dates,
    model_series,
    resolve_project_path,
)
from glp_hyperparameter_optimization.data_utils import download_glp_realtime_panel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download ALFRED/FRED data for the GLP (2015) replication.")
    parser.add_argument("--model-size", choices=("small", "medium", "large"), default="large",
                        help="Which variable universe to download. 'large' (default) covers every model size.")
    parser.add_argument("--output-panel", type=Path, default=GLP_REALTIME_PANEL_PATH)
    parser.add_argument("--output-latest", type=Path, default=GLP_LATEST_PANEL_PATH)
    parser.add_argument("--metadata-path", type=Path, default=GLP_DOWNLOAD_METADATA_PATH)
    parser.add_argument("--start", type=str, default=GLP_FORECAST_START.strftime("%Y-%m-%d"))
    parser.add_argument("--end", type=str, default=GLP_FORECAST_END.strftime("%Y-%m-%d"))
    parser.add_argument("--actual-vintage", type=str, default=GLP_ACTUAL_VINTAGE.strftime("%Y-%m-%d"))
    parser.add_argument("--force", action="store_true", help="Redownload even if a cached copy already exists.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    origins = forecast_origin_dates(pd.Timestamp(args.start), pd.Timestamp(args.end))
    series_ids = [spec.series_id for spec in model_series(args.model_size)]

    def report(message: str) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"[{timestamp}] {message}", flush=True)

    report(
        f"Downloading GLP '{args.model_size}' data ({len(series_ids)} series) for {len(origins)} origins "
        f"{args.start}..{args.end} with actual vintage {args.actual_vintage}."
    )
    download_glp_realtime_panel(
        series_ids=series_ids,
        output_path=resolve_project_path(args.output_panel),
        latest_output_path=resolve_project_path(args.output_latest),
        metadata_path=resolve_project_path(args.metadata_path),
        forecast_origins=origins,
        actual_vintage=pd.Timestamp(args.actual_vintage),
        progress_callback=report,
        force=args.force,
    )
    report("All downloads completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
