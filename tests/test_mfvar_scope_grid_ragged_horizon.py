"""Regression test for MF-01 in the scope-grid forecast generator.

The scope-grid runner used to size ``model.forecast(...)`` with the
calendar-nominal ``max_horizon * 3`` months, which ignores where the model's
monthly calendar actually ends. Whenever the outer origin's monthly block lags
the nominal origin quarter (the normal real-time ragged edge) the simulation
stopped short of the final target quarter and forecast extraction raised
``KeyError``. The generator must instead use the endpoint-aware length.
"""

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from paper_hyperparameter_optimization import forecasting, scope_grid_execution
from paper_hyperparameter_optimization.horizon_mapping import nominal_forecast_months
from paper_hyperparameter_optimization.selection_experiment import MFVARForecastRequest


ORIGIN = pd.Timestamp("2005-01-31")
# Ragged edge: the last released monthly observation lags the origin quarter,
# so the model's monthly calendar ends in November 2004.
MONTHLY_ENDPOINT = pd.Timestamp("2004-11-30")
VARIABLES = ("gdp", "infl")
HORIZONS = (1, 2, 3, 4)


def _ragged_frames():
    monthly_index = pd.date_range("2000-01-31", MONTHLY_ENDPOINT, freq="ME")
    monthly = pd.DataFrame(
        np.zeros((len(monthly_index), len(VARIABLES))),
        index=monthly_index,
        columns=list(VARIABLES),
    )
    quarterly_index = pd.date_range("2000-03-31", "2004-09-30", freq="QE")
    quarterly = pd.DataFrame(
        np.zeros((len(quarterly_index), len(VARIABLES))),
        index=quarterly_index,
        columns=list(VARIABLES),
    )
    return quarterly, monthly


class _FakeModel:
    """Model stub whose predictive quarters follow the simulated month count."""

    def __init__(self, *args, **kwargs):
        self.forecast_months = None

    def fit(self, *args, **kwargs):
        return self

    def forecast(self, months):
        self.forecast_months = int(months)

    def aggregate(self, frequency="Q"):
        return self

    def predicted_quarters(self):
        # Simulated months run from the calendar endpoint forward; only fully
        # simulated quarters survive quarterly aggregation.
        last_month = MONTHLY_ENDPOINT.to_period("M") + self.forecast_months
        first_quarter = (MONTHLY_ENDPOINT.to_period("M") + 1).asfreq("Q")
        last_quarter = last_month.asfreq("Q")
        if last_month != last_quarter.asfreq("M", how="end"):
            last_quarter -= 1
        return pd.period_range(first_quarter, last_quarter, freq="Q")


@pytest.fixture()
def patched_generator(monkeypatch):
    quarterly, monthly = _ragged_frames()
    created: list[_FakeModel] = []

    class _Recording(_FakeModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    fake_mbfvar = types.ModuleType("MBFVAR")
    fake_mbfvar.MixedFrequencyBVAR = _Recording
    monkeypatch.setitem(sys.modules, "MBFVAR", fake_mbfvar)

    monkeypatch.setattr(
        "paper_hyperparameter_optimization.data_utils.load_realtime_panel",
        lambda path: pd.DataFrame(),
    )
    monkeypatch.setattr(
        "paper_hyperparameter_optimization.data_utils.build_model_input_frames",
        lambda panel, origin_date: (quarterly, monthly),
    )
    monkeypatch.setattr(forecasting, "make_data_in", lambda q, m: {"quarterly": q, "monthly": m})
    monkeypatch.setattr(
        forecasting, "aggregate_quarterly_posterior_draws", lambda model: model
    )

    def _summarize(model):
        quarters = model.predicted_quarters()
        frame = pd.DataFrame(
            np.arange(len(quarters) * len(VARIABLES), dtype=float).reshape(
                len(quarters), len(VARIABLES)
            ),
            index=quarters,
            columns=list(VARIABLES),
        )
        return {}, {"mean": frame}

    monkeypatch.setattr(forecasting, "summarize_quarterly_draws", _summarize)

    generator = scope_grid_execution._build_real_forecast_generator(
        Path("unused.csv"), VARIABLES
    )
    return generator, created


def _request():
    return MFVARForecastRequest(
        hyperparameter_vector=(0.2, 1.0, 1.0, 1.0, 1.0),
        origin_index=0,
        origin_label=ORIGIN,
        system_variables=VARIABLES,
        system_horizons=HORIZONS,
        forecast_variables=VARIABLES,
        cell_id="pooled",
        event_id="event-0",
    )


def test_ragged_origin_forecast_reaches_final_target(patched_generator):
    generator, created = patched_generator
    out = generator(_request())

    assert out.shape == (len(HORIZONS), len(VARIABLES))
    assert np.isfinite(out).all()

    (model,) = created
    endpoint_aware = forecasting.required_forecast_months(
        _ragged_frames()[1], ORIGIN, max_horizon_quarters=max(HORIZONS)
    )
    assert model.forecast_months >= endpoint_aware
    # The old, horizon-only behaviour would have simulated too few months.
    assert model.forecast_months > nominal_forecast_months(max(HORIZONS))


def test_nominal_length_would_miss_the_final_target():
    """Guard the premise: the nominal length is short for this ragged origin."""
    _, monthly = _ragged_frames()
    endpoint_aware = forecasting.required_forecast_months(
        monthly, ORIGIN, max_horizon_quarters=max(HORIZONS)
    )
    assert endpoint_aware == 13
    assert nominal_forecast_months(max(HORIZONS)) == 12
