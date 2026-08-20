import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from glp_hyperparameter_optimization import data_utils as du
from glp_hyperparameter_optimization.config import model_series


def test_parse_remote_csv_text_rejects_empty_and_headerless_bodies():
    with pytest.raises(du.EmptyRemoteDataError):
        du._parse_remote_csv_text("", "https://example.com/empty")
    with pytest.raises(du.EmptyRemoteDataError):
        du._parse_remote_csv_text("only_one_column\nvalue\n", "https://example.com/bad")


def test_download_series_vintage_fails_closed_when_alfred_body_is_empty(monkeypatch):
    """An empty ALFRED body must raise instead of substituting latest FRED data.

    Latest FRED history is revised data. Silently returning it in place of a
    true point-in-time vintage would leak post-origin information into a
    real-time forecast exercise, so the download fails closed.
    """

    def fake_read_csv(url: str):
        raise du.EmptyRemoteDataError("empty")

    def unexpected_latest(series_id: str, force: bool = False):
        raise AssertionError("latest FRED history must never be used as a vintage substitute")

    monkeypatch.setattr(du, "_read_csv_from_url", fake_read_csv)
    monkeypatch.setattr(du, "_load_or_download_latest_series", unexpected_latest)

    with pytest.raises(du.DataDownloadError, match="not a point-in-time substitute"):
        du.download_series_vintage("BOGMBASE", pd.Timestamp("2000-03-31"))


def test_download_series_vintage_fails_closed_even_with_force_latest_fallback(monkeypatch):
    """The legacy force_latest_fallback flag no longer authorizes the unsafe fallback."""

    def fake_read_csv(url: str):
        raise du.EmptyRemoteDataError("empty")

    def unexpected_latest(series_id: str, force: bool = False):
        raise AssertionError("latest FRED history must never be used as a vintage substitute")

    monkeypatch.setattr(du, "_read_csv_from_url", fake_read_csv)
    monkeypatch.setattr(du, "_load_or_download_latest_series", unexpected_latest)

    with pytest.raises(du.DataDownloadError):
        du.download_series_vintage(
            "BOGMBASE", pd.Timestamp("2000-03-31"), force_latest_fallback=True
        )


def test_build_quarterly_levels_uses_proxy_series_for_large_model():
    specs = {spec.code: spec for spec in model_series("large")}
    assert specs["SP500"].series_id == "SPASTT01USM661N"
    assert "proxy" in specs["SP500"].label.lower()


def test_download_sp500_stooq_monthly_rejects_html_challenge(monkeypatch):
    monkeypatch.setattr(du, "_download_text_with_curl", lambda url: "<!DOCTYPE html><html><body>challenge</body></html>")
    with pytest.raises(du.DataDownloadError):
        du.download_sp500_stooq_monthly()
