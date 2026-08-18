"""Tests for the MF-BVAR selection-experiment adapter and real objective path."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common_hpo import (
    LossConfig,
    SelectionSchedule,
    ValidationScheme,
    build_selection_plan,
)
from paper_hyperparameter_optimization import forecasting
from paper_hyperparameter_optimization.selection_experiment import (
    MFVARCellSelection,
    MFVARCellSelectionRequest,
    MFVARForecastRequest,
    run_mfvar_selection_experiment,
)
from paper_hyperparameter_optimization.loss_engine import build_mfvar_objective_selector


FORECAST_BLOCK = ("GDP", "INVFIX", "GOV")
HORIZONS = (1, 2, 4)


def _validation_scheme():
    return ValidationScheme(
        training_window="expanding",
        origin_selection="most_recent",
        n_origins=2,
        horizons=HORIZONS,
        min_train_length=2,
    )


def _fake_selector_factory(record):
    """Selector recording each request and returning a vector encoding the cell."""

    def selector(request: MFVARCellSelectionRequest) -> MFVARCellSelection:
        record.append(request)
        # Encode the cell id length so the forecast generator can identify which
        # cell's selection produced a value.
        marker = float(len(request.cell_id))
        return MFVARCellSelection(
            hyperparameter_vector=(marker, 1.0, 1.0, 1.0, 1.0),
            named_parameters={"marker": marker},
            selection_loss=marker,
            forecast_variables=request.forecast_variables,
            objective_variables=request.objective_variables,
            optimizer_seed=request.seed,
        )

    return selector


def _encoding_forecast_generator(request: MFVARForecastRequest) -> np.ndarray:
    """Encode (origin, cell marker, variable, horizon) into each forecast cell."""

    marker = request.hyperparameter_vector[0]
    rows = len(request.system_horizons)
    cols = len(request.system_variables)
    out = np.empty((rows, cols), dtype=float)
    for r, horizon in enumerate(request.system_horizons):
        for c, _variable in enumerate(request.system_variables):
            out[r, c] = 1000 * request.origin_index + 100 * marker + 10 * r + c
    return out


ORIGINS = [
    pd.Timestamp("2000-01-31"),
    pd.Timestamp("2000-02-29"),
    pd.Timestamp("2000-03-31"),
    pd.Timestamp("2000-04-30"),
]


def _run(plan, schedule=None, **kwargs):
    record: list = []
    selector = _fake_selector_factory(record)
    result = run_mfvar_selection_experiment(
        ORIGINS,
        selector=selector,
        forecast_generator=_encoding_forecast_generator,
        target_variables=plan.target_variables,
        target_horizons=plan.target_horizons,
        forecast_variables=FORECAST_BLOCK,
        loss_config=LossConfig(),
        validation_scheme=_validation_scheme(),
        plan=plan,
        schedule=schedule,
        base_seed=20240101,
        **kwargs,
    )
    return result, record


def test_shared_plan_describes_mfvar_and_glp():
    # The exact same SelectionPlan object is valid for both workflows.
    plan = build_selection_plan("variable", ("GDP", "INVFIX"), (1, 2))
    assert plan.scope == "variable"
    # Usable by the MF-BVAR adapter unchanged.
    result, _ = _run(plan)
    assert result.run_metadata["model"] == "mfvar"
    assert {row["variable"] for row in result.forecast_panel} == {"GDP", "INVFIX"}


def test_pooled_mapping_single_cell():
    plan = build_selection_plan("pooled", FORECAST_BLOCK, HORIZONS)
    result, record = _run(plan)
    # One cell, one selection event -> one selection row.
    assert len({r.cell_id for r in record}) == 1
    assert len(result.selected_hyperparameters) == 1
    scopes = {row["selection_scope"] for row in result.forecast_panel}
    assert scopes == {"pooled"}


def test_horizon_mapping_one_cell_per_horizon():
    plan = build_selection_plan("horizon", FORECAST_BLOCK, HORIZONS)
    result, _ = _run(plan)
    cell_ids = {row["cell_id"] for row in result.forecast_panel}
    assert cell_ids == {"horizon-h1", "horizon-h2", "horizon-h4"}
    # Each canonical row's responsible cell matches its horizon.
    for row in result.forecast_panel:
        assert row["cell_id"] == f"horizon-h{row['horizon']}"


def test_variable_mapping_one_cell_per_variable():
    plan = build_selection_plan("variable", FORECAST_BLOCK, HORIZONS)
    result, _ = _run(plan)
    for row in result.forecast_panel:
        assert row["cell_id"] == f"variable-{row['variable'].lower()}"


def test_variable_horizon_mapping_one_cell_per_pair():
    plan = build_selection_plan("variable_horizon", FORECAST_BLOCK, HORIZONS)
    result, _ = _run(plan)
    n_pairs = len(FORECAST_BLOCK) * len(HORIZONS)
    assert len(result.selected_hyperparameters) == n_pairs
    for row in result.forecast_panel:
        assert row["cell_id"] == f"variable-{row['variable'].lower()}-h{row['horizon']}"


def test_selection_event_reuse_once_schedule():
    plan = build_selection_plan("pooled", FORECAST_BLOCK, HORIZONS)
    result, record = _run(plan, schedule=SelectionSchedule.once())
    # Selection happens once regardless of the four origins.
    assert len(record) == 1
    # But every origin appears in the canonical panel.
    origins = {row["forecast_origin"] for row in result.forecast_panel}
    assert len(origins) == len(ORIGINS)
    # All rows reuse the single selection event.
    assert {row["selection_event_id"] for row in result.forecast_panel} == {"sel-000-o0000"}


def test_selection_event_reuse_every_n_origins():
    plan = build_selection_plan("pooled", FORECAST_BLOCK, HORIZONS)
    result, record = _run(plan, schedule=SelectionSchedule.every_n_origins(2))
    # Origins 0 and 2 are selection events -> two selection rows.
    assert len(record) == 2
    events = {row["selection_event_id"] for row in result.forecast_panel}
    assert len(events) == 2


def test_output_uniqueness_and_completeness():
    plan = build_selection_plan("variable_horizon", FORECAST_BLOCK, HORIZONS)
    result, _ = _run(plan)
    keys = {
        (row["forecast_origin"], row["variable"], row["horizon"])
        for row in result.forecast_panel
    }
    expected = len(ORIGINS) * len(FORECAST_BLOCK) * len(HORIZONS)
    assert len(result.forecast_panel) == expected
    assert len(keys) == expected  # no duplicates


def test_deterministic_seed_propagation():
    plan = build_selection_plan("variable", FORECAST_BLOCK, HORIZONS)
    _, record_a = _run(plan)
    _, record_b = _run(plan)
    seeds_a = [r.seed for r in record_a]
    seeds_b = [r.seed for r in record_b]
    assert seeds_a == seeds_b
    assert all(seed is not None for seed in seeds_a)


def test_all_cells_panel_marks_canonical():
    plan = build_selection_plan("variable", FORECAST_BLOCK, HORIZONS)
    result, _ = _run(plan, retain_off_target=True)
    assert result.forecast_panel_all_cells
    canonical = [row for row in result.forecast_panel_all_cells if row["is_canonical"]]
    # Canonical diagnostic rows equal the canonical panel size.
    assert len(canonical) == len(result.forecast_panel)


def test_seed_uncontrolled_recorded_in_metadata():
    plan = build_selection_plan("pooled", FORECAST_BLOCK, HORIZONS)

    def flagged_selector(request):
        return MFVARCellSelection(
            hyperparameter_vector=(0.1, 1.0, 1.0, 1.0, 1.0),
            named_parameters={},
            selection_loss=1.0,
            seed_uncontrolled=True,
            seed_uncontrolled_reason="upstream unseeded generator",
        )

    result = run_mfvar_selection_experiment(
        ORIGINS,
        selector=flagged_selector,
        forecast_generator=_encoding_forecast_generator,
        target_variables=plan.target_variables,
        target_horizons=plan.target_horizons,
        forecast_variables=FORECAST_BLOCK,
        loss_config=LossConfig(),
        validation_scheme=_validation_scheme(),
        plan=plan,
    )
    assert result.run_metadata["seed_uncontrolled_events"]
    assert result.run_metadata["reproducibility_limitations"]


def test_non_subset_target_rejected():
    plan = build_selection_plan("pooled", ("GDP", "NOTAVAR"), HORIZONS)
    with pytest.raises(ValueError, match="subset of forecast_variables"):
        run_mfvar_selection_experiment(
            ORIGINS,
            selector=_fake_selector_factory([]),
            forecast_generator=_encoding_forecast_generator,
            target_variables=("GDP", "NOTAVAR"),
            target_horizons=HORIZONS,
            forecast_variables=FORECAST_BLOCK,
            loss_config=LossConfig(),
            validation_scheme=_validation_scheme(),
            plan=plan,
        )


# --------------------------------------------------------------------------- #
# Real mixed-frequency objective path (GDP-only) with a lightweight fake MBFVAR.
# --------------------------------------------------------------------------- #
class _RecordingModel:
    fit_calls: list = []

    def __init__(self, *args, **kwargs):
        pass

    def fit(self, data_in, *, hyp, var_of_interest, temp_agg, check_explosive):
        type(self).fit_calls.append(list(var_of_interest))

    def forecast(self, horizon_months):
        pass


class _FakeDataIn:
    frequencies = ["Q", "M"]
    freq_ratio_list = [3]

    def __init__(self, quarterly, monthly):
        self.input_data_Q = quarterly
        self.input_data = [monthly]


def _build_fake_data_in():
    quarters = pd.period_range("2000Q1", periods=20, freq="Q").to_timestamp(how="end")
    quarterly = pd.DataFrame(
        {code: np.linspace(100.0, 200.0, len(quarters)) for code in FORECAST_BLOCK},
        index=quarters,
    )
    months = pd.period_range("2000-01", periods=60, freq="M").to_timestamp(how="end")
    monthly = pd.DataFrame({"m": np.linspace(1.0, 2.0, len(months))}, index=months)
    return _FakeDataIn(quarterly, monthly)


def test_gdp_only_objective_runs_through_real_path(monkeypatch):
    _RecordingModel.fit_calls = []
    target = pd.PeriodIndex(["2005Q1", "2005Q2"], freq="Q")
    # INVFIX/GOV predictions are far off; only GDP should enter the loss.
    predicted = pd.DataFrame(
        {"GDP": [1.0, 1.0], "INVFIX": [99.0, 99.0], "GOV": [99.0, 99.0]}, index=target
    )
    actual = pd.DataFrame(
        {"GDP": [1.5, 1.5], "INVFIX": [2.0, 2.0], "GOV": [2.0, 2.0]}, index=target
    )

    # Patch only the heavy MBFVAR-facing primitives so the real fold-building and
    # real _rmse_candidate_score run end to end.
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

    # Make fold target quarters line up with the patched predicted/actual index.
    real_build = forecasting.build_rmse_validation_folds

    def patched_folds(*args, **kwargs):
        folds, diagnostics = real_build(*args, **kwargs)
        for fold in folds:
            fold["target_quarters"] = target
        return folds, diagnostics

    monkeypatch.setattr(forecasting, "build_rmse_validation_folds", patched_folds)

    selector = build_mfvar_objective_selector(
        data_in=_build_fake_data_in(),
        model_class=_RecordingModel,
        candidate_params=[{"lambda1_1": 0.1, "lambda2_1": 1.0, "lambda4_1": 1.0, "lambda5_1": 1.0}],
        nsim=1,
        nburn_perc=0.5,
        nlags=[1],
        thining=1,
        temp_agg="mean",
        horizon_quarters=2,
        eval_horizon_quarters=2,
        n_eval=1,
    )

    request = MFVARCellSelectionRequest(
        event=SelectionSchedule.once().resolve(ORIGINS)[0],
        cell_id="variable-gdp",
        objective_variables=("GDP",),
        forecast_variables=FORECAST_BLOCK,
        horizons=(1, 2),
        loss_config=LossConfig(),
        validation_scheme=_validation_scheme(),
        origin_index=0,
        origin_label=ORIGINS[0],
        seed=7,
    )
    selection = selector(request)

    # The forecast state was fit on the full block, not the GDP-only objective.
    assert _RecordingModel.fit_calls
    assert all(call == list(FORECAST_BLOCK) for call in _RecordingModel.fit_calls)
    # GDP-only objective yields the small, finite score (|1.0 - 1.5| = 0.5),
    # proving only GDP errors enter the loss (INVFIX/GOV would give ~97).
    assert selection.selection_loss == pytest.approx(0.5)
    assert selection.objective_variables == ("GDP",)
    assert selection.forecast_variables == FORECAST_BLOCK
    # The upstream unseeded generator is surfaced, not concealed.
    assert selection.seed_uncontrolled is True
    assert selection.seed_uncontrolled_reason
