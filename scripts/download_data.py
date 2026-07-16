from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from paper_hyperparameter_optimization.config import (
    DOWNLOAD_METADATA_PATH,
    LATEST_PANEL_PATH,
    PAPER_ACTUAL_VINTAGE,
    PAPER_FORECAST_END,
    PAPER_FORECAST_START,
    REALTIME_PANEL_PATH,
    forecast_origin_dates,
    resolve_project_path,
)
from paper_hyperparameter_optimization.data_utils import download_realtime_panel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download ALFRED/FRED data for the Schorfheide-Song replication.")
    parser.add_argument("--output-panel", type=Path, default=REALTIME_PANEL_PATH)
    parser.add_argument("--output-latest", type=Path, default=LATEST_PANEL_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DOWNLOAD_METADATA_PATH)
    parser.add_argument("--start", type=str, default=PAPER_FORECAST_START.strftime("%Y-%m-%d"))
    parser.add_argument("--end", type=str, default=PAPER_FORECAST_END.strftime("%Y-%m-%d"))
    parser.add_argument("--actual-vintage", type=str, default=PAPER_ACTUAL_VINTAGE.strftime("%Y-%m-%d"))
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even if a cached copy already exists.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output_panel = resolve_project_path(args.output_panel)
    args.output_latest = resolve_project_path(args.output_latest)
    args.metadata_path = resolve_project_path(args.metadata_path)
    origins = forecast_origin_dates(pd.Timestamp(args.start), pd.Timestamp(args.end))

    def report(message: str) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"[{timestamp}] {message}", flush=True)

    report(
        "Starting data download "
        f"for {len(origins)} forecast origins from {args.start} to {args.end} "
        f"with actual vintage {args.actual_vintage}."
    )
    download_realtime_panel(
        output_path=args.output_panel,
        latest_output_path=args.output_latest,
        metadata_path=args.metadata_path,
        forecast_origins=origins,
        actual_vintage=pd.Timestamp(args.actual_vintage),
        max_workers=args.max_workers,
        progress_callback=report,
        force=args.force,
    )
    report("All downloads completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
