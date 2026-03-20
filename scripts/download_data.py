from __future__ import annotations

import argparse
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
    parser.add_argument("--max-workers", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    origins = forecast_origin_dates(pd.Timestamp(args.start), pd.Timestamp(args.end))
    download_realtime_panel(
        output_path=args.output_panel,
        latest_output_path=args.output_latest,
        metadata_path=args.metadata_path,
        forecast_origins=origins,
        actual_vintage=pd.Timestamp(args.actual_vintage),
        max_workers=args.max_workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
