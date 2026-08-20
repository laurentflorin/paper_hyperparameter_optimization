"""Tests for the canonical quarterly<->monthly horizon mapping."""

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from paper_hyperparameter_optimization.horizon_mapping import (
    FREQ_RATIO,
    quarterly_horizon_to_state_rows,
    nominal_forecast_months,
    state_rows_for_max_horizon,
    target_quarter_for_origin,
    target_quarters_for_origin,
)


def test_frequency_ratio_is_three():
    assert FREQ_RATIO == 3


@pytest.mark.parametrize(
    "horizon,expected_rows",
    [(1, 3), (2, 6), (4, 12), (8, 24)],
)
def test_quarterly_horizon_to_state_rows(horizon, expected_rows):
    assert quarterly_horizon_to_state_rows(horizon) == expected_rows
    assert state_rows_for_max_horizon(horizon) == expected_rows
    assert nominal_forecast_months(horizon) == expected_rows


def test_invalid_horizon_rejected():
    with pytest.raises(ValueError):
        quarterly_horizon_to_state_rows(0)
    with pytest.raises(TypeError):
        quarterly_horizon_to_state_rows(1.5)
    with pytest.raises(TypeError):
        quarterly_horizon_to_state_rows(True)


def test_target_quarter_within_year():
    # Origin in Q1, horizon 1 is the origin quarter itself (nowcast).
    assert target_quarter_for_origin("2005-01-31", 1) == pd.Period("2005Q1", freq="Q")
    assert target_quarter_for_origin("2005-01-31", 2) == pd.Period("2005Q2", freq="Q")
    assert target_quarter_for_origin("2005-01-31", 4) == pd.Period("2005Q4", freq="Q")


def test_target_quarter_crosses_year_boundary():
    # Origin in Q4 2005; horizon 2 must roll into 2006Q1.
    assert target_quarter_for_origin("2005-11-30", 1) == pd.Period("2005Q4", freq="Q")
    assert target_quarter_for_origin("2005-11-30", 2) == pd.Period("2006Q1", freq="Q")
    assert target_quarter_for_origin("2005-12-31", 5) == pd.Period("2006Q4", freq="Q")
    # A long horizon from Q3 spanning multiple years.
    assert target_quarter_for_origin("2005-08-31", 8) == pd.Period("2007Q2", freq="Q")


def test_target_quarter_accepts_period_and_timestamp():
    assert target_quarter_for_origin(pd.Period("2005Q4", freq="Q"), 2) == pd.Period(
        "2006Q1", freq="Q"
    )
    assert target_quarter_for_origin(pd.Timestamp("2005-10-15"), 1) == pd.Period(
        "2005Q4", freq="Q"
    )


def test_target_quarter_mid_quarter_origin_maps_to_enclosing_quarter():
    # Any month within a quarter maps to that same quarter for horizon 1.
    for month in ("2005-04-30", "2005-05-31", "2005-06-30"):
        assert target_quarter_for_origin(month, 1) == pd.Period("2005Q2", freq="Q")


def test_target_quarters_for_origin_sequence():
    quarters = target_quarters_for_origin("2005-12-31", [1, 2, 4])
    assert quarters == (
        pd.Period("2005Q4", freq="Q"),
        pd.Period("2006Q1", freq="Q"),
        pd.Period("2006Q3", freq="Q"),
    )
