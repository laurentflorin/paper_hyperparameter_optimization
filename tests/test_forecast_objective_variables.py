"""Tests for separating forecast_variables from objective_variables.

These cover the two-variable-set contract that fixes the mixed-frequency
target-dimension limitation: the forecast state is always built from the full
quarterly block, while only a (possibly smaller) objective subset enters the loss.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from paper_hyperparameter_optimization import forecasting
from paper_hyperparameter_optimization.forecasting import (
    RMSE_REQUIRED_FORECAST_VARIABLES,
    _rmse_candidate_score,
    build_rmse_validation_folds,
    resolve_forecast_objective_variables,
)


FULL_BLOCK = list(RMSE_REQUIRED_FORECAST_VARIABLES)


# ---------------------------------------------------------------------------
# Resolution contract
# ---------------------------------------------------------------------------
def test_forecast_variables_default_to_full_block():
    forecast_variables, objective_variables = resolve_forecast_objective_variables("mango_rmse")
    assert forecast_variables == FULL_BLOCK
    assert objective_variables == FULL_BLOCK


def test_objective_variables_may_be_gdp_only():
    forecast_variables, objective_variables = resolve_forecast_objective_variables(
        "mango_rmse", objective_variables=["GDP"]
    )
    assert forecast_variables == FULL_BLOCK
    assert objective_variables == ["GDP"]


def test_legacy_single_argument_maps_to_objective_subset():
    forecast_variables, objective_variables = resolve_forecast_objective_variables(
        "mango_rmse", optimization_variables=["GDP"]
    )
    assert forecast_variables == FULL_BLOCK
    assert objective_variables == ["GDP"]


def test_legacy_full_block_call_unchanged():
    forecast_variables, objective_variables = resolve_forecast_objective_variables(
        "mango_rmse", optimization_variables=FULL_BLOCK
    )
    assert forecast_variables == FULL_BLOCK
    assert objective_variables == FULL_BLOCK


def test_non_subset_objective_rejected():
    with pytest.raises(ValueError, match="subset"):
        resolve_forecast_objective_variables(
            "mango_rmse", forecast_variables=FULL_BLOCK, objective_variables=["NOT_A_VAR"]
        )


def test_reduced_forecast_block_rejected_for_rmse():
    with pytest.raises(ValueError, match="full quarterly forecast block"):
        resolve_forecast_objective_variables("mango_rmse", forecast_variables=["GDP"])


def test_legacy_and_explicit_are_mutually_exclusive():
    with pytest.raises(ValueError, match="not both"):
        resolve_forecast_objective_variables(
            "mango_rmse", optimization_variables=["GDP"], objective_variables=["GDP"]
        )


# ---------------------------------------------------------------------------
# Objective path with a lightweight fake MBFVAR model
# ---------------------------------------------------------------------------
class _RecordingModel:
    """Captures the var_of_interest passed to fit and exposes fixed forecasts."""

    fit_calls: list[list[str]] = []

    def __init__(self, *args, **kwargs):
        pass

    def fit(self, data_in, *, hyp, var_of_interest, temp_agg, check_explosive):
        type(self).fit_calls.append(list(var_of_interest))

    def forecast(self, horizon_months):
        pass


def _valid_params():
    return {
        "lambda1_1": 0.1,
        "lambda2_1": 1.0,
        "lambda4_1": 1.0,
        "lambda5_1": 1.0,
    }


def _single_fold():
    target = pd.PeriodIndex(["2005Q1"], freq="Q")
    quarterly = pd.DataFrame(
        {code: [100.0, 101.0] for code in FULL_BLOCK},
        index=pd.PeriodIndex(["2004Q3", "2004Q4"], freq="Q"),
    )
    holdout = pd.DataFrame(
        {code: [102.0] for code in FULL_BLOCK}, index=target
    )
    return {
        "cut": 2,
        "target_quarters": target,
        "quarterly": quarterly,
        "monthly": pd.DataFrame(),
        "holdout": holdout,
    }


def test_objective_uses_full_block_but_scores_only_gdp(monkeypatch):
    _RecordingModel.fit_calls = []
    target = pd.PeriodIndex(["2005Q1"], freq="Q")

    # Predicted metrics for the full forecast block; INVFIX/GOV are far off but
    # must not enter the loss because the objective is GDP only.
    predicted = pd.DataFrame(
        {"GDP": [1.0], "INVFIX": [99.0], "GOV": [99.0]}, index=target
    )
    actual = pd.DataFrame(
        {"GDP": [1.5], "INVFIX": [2.0], "GOV": [2.0]}, index=target
    )

    monkeypatch.setattr(forecasting, "make_data_in", lambda q, m: object())
    monkeypatch.setattr(
        forecasting, "aggregate_quarterly_posterior_draws", lambda model: [predicted.copy()]
    )
    monkeypatch.setattr(
        forecasting,
        "summarize_quarterly_draws",
        lambda frames: ({}, {"mean": predicted.copy()}),
    )
    monkeypatch.setattr(forecasting, "compute_quarterly_metrics", lambda frame: actual.copy())

    score = _rmse_candidate_score(
        _valid_params(),
        model_class=_RecordingModel,
        folds=[_single_fold()],
        forecast_variables=FULL_BLOCK,
        objective_variables=["GDP"],
        horizon_quarters=1,
        h_eval=1,
        nsim=1,
        nburn_perc=0.5,
        nlags=[1],
        thining=1,
        temp_agg="mean",
        objective_seed=0,
    )

    # The forecast state is fit on the full quarterly block, not the reduced target.
    assert _RecordingModel.fit_calls == [FULL_BLOCK]
    # Only the GDP error (|1.0 - 1.5| = 0.5) contributes; INVFIX/GOV are ignored.
    assert score == pytest.approx(0.5)
    # A finite, small score proves the GDP-only objective no longer collapses to a penalty.
    assert np.isfinite(score)
    assert score < forecasting.OPTIMIZATION_PENALTY


def test_horizon_selection_picks_correct_quarter(monkeypatch):
    _RecordingModel.fit_calls = []
    targets = pd.PeriodIndex(["2005Q1", "2005Q2"], freq="Q")
    predicted = pd.DataFrame({"GDP": [1.0, 5.0]}, index=targets)
    actual = pd.DataFrame({"GDP": [1.0, 2.0]}, index=targets)

    fold = _single_fold()
    fold["target_quarters"] = targets
    fold["holdout"] = pd.DataFrame(
        {code: [102.0, 103.0] for code in FULL_BLOCK}, index=targets
    )

    monkeypatch.setattr(forecasting, "make_data_in", lambda q, m: object())
    monkeypatch.setattr(
        forecasting, "aggregate_quarterly_posterior_draws", lambda model: [predicted.copy()]
    )
    monkeypatch.setattr(
        forecasting,
        "summarize_quarterly_draws",
        lambda frames: ({}, {"mean": predicted.copy()}),
    )
    monkeypatch.setattr(forecasting, "compute_quarterly_metrics", lambda frame: actual.copy())

    # Scoring horizon 2 selects 2005Q2 where the error is |5.0 - 2.0| = 3.0.
    score = _rmse_candidate_score(
        _valid_params(),
        model_class=_RecordingModel,
        folds=[fold],
        forecast_variables=FULL_BLOCK,
        objective_variables=["GDP"],
        horizon_quarters=2,
        h_eval=2,
        nsim=1,
        nburn_perc=0.5,
        nlags=[1],
        thining=1,
        temp_agg="mean",
        objective_seed=0,
    )
    assert score == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Fold construction validates the subset before any expensive fitting
# ---------------------------------------------------------------------------
class _FakeDataIn:
    frequencies = ["Q", "M"]
    freq_ratio_list = [3]

    def __init__(self, quarterly, monthly):
        self.input_data_Q = quarterly
        self.input_data = [monthly]


def _build_fake_data_in():
    quarters = pd.period_range("2000Q1", periods=20, freq="Q").to_timestamp(how="end")
    quarterly = pd.DataFrame(
        {code: np.linspace(100.0, 200.0, len(quarters)) for code in FULL_BLOCK},
        index=quarters,
    )
    months = pd.period_range("2000-01", periods=60, freq="M").to_timestamp(how="end")
    monthly = pd.DataFrame({"m": np.linspace(1.0, 2.0, len(months))}, index=months)
    return _FakeDataIn(quarterly, monthly)


def test_invalid_subset_fails_before_fold_building():
    with pytest.raises(ValueError, match="subset"):
        build_rmse_validation_folds(
            _build_fake_data_in(),
            horizon_quarters=2,
            h_eval=None,
            n_eval=1,
            forecast_variables=FULL_BLOCK,
            objective_variables=["NOT_A_VAR"],
            nlags=[1],
            selection="rolling",
            min_train_quarters=None,
            fold_seed=0,
        )


def test_folds_derive_min_train_from_forecast_block():
    folds, diagnostics = build_rmse_validation_folds(
        _build_fake_data_in(),
        horizon_quarters=2,
        h_eval=None,
        n_eval=1,
        forecast_variables=FULL_BLOCK,
        objective_variables=["GDP"],
        nlags=[1],
        selection="rolling",
        min_train_quarters=None,
        fold_seed=0,
    )
    assert len(folds) == 1
    # The full three-variable forecast block drives the minimum training length.
    assert diagnostics["derived_min_train_quarters"] >= 2
