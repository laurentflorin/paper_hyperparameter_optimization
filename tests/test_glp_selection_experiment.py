import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common_hpo.losses import LossConfig
from common_hpo.schedules import SelectionSchedule
from common_hpo.selection_scope import build_selection_plan
from common_hpo.splits import ValidationScheme
from glp_hyperparameter_optimization.search_config import GLPSearchConfig
from glp_hyperparameter_optimization.selection_experiment import (
    CellSelection,
    run_glp_selection_experiment,
)


SYSTEM_VARS = ("GDP", "DEFL", "FFR")
SYSTEM_HORS = (1, 2, 4)
ORIGINS = [f"{2000 + i // 4}Q{i % 4 + 1}" for i in range(8)]


def _scheme():
    return ValidationScheme(
        training_window="expanding",
        origin_selection="most_recent",
        n_origins=2,
        horizons=SYSTEM_HORS,
        min_train_length=8,
    )


def _make_selector(call_log=None):
    """A fake selector: encodes the cell into distinct, deterministic values."""

    def selector(request):
        call_log and call_log.append((request.event.event_number, request.cell_id))
        # Encode a per-cell deterministic natural vector so different cells
        # produce distinguishable forecasts.
        base = float(abs(hash(request.cell_id)) % 1000) / 1000.0 + 0.1
        natural = (base, 1.0, 1.0)
        return CellSelection(
            natural_vector=natural,
            named_parameters={"lambda": base, "theta": 1.0, "miu": 1.0},
            selection_loss=base,
            fixed_psi_source="context_ss",
            search_dimension=3,
            inner_window_start=0,
            inner_window_end=10,
            n_inner_origins=2,
            validation_stride=1,
            optimizer_seed=request.seed,
            optimizer_budget={"init_points": 2, "n_iter": 2},
            objective_draw_count=1,
            failure_counts={"penalized": 0},
            runtime_seconds=0.01,
        )

    return selector


def _make_forecast_generator(call_counter=None):
    """A fake generator: forecast depends on hyper, origin, variable, horizon.

    The dependence on the natural vector's first element lets tests confirm which
    cell's system produced each canonical value.
    """

    def generator(request):
        if call_counter is not None:
            call_counter.append(request.natural_vector)
        lam = request.natural_vector[0]
        rows = []
        for h in request.system_horizons:
            row = []
            for vi, _v in enumerate(request.system_variables):
                row.append(lam * 100.0 + h + vi * 0.01 + request.origin_index * 0.001)
            rows.append(row)
        return np.asarray(rows, dtype=float)

    return generator


def _run(scope, schedule=None, **kwargs):
    plan = build_selection_plan(scope, ["GDP", "DEFL"], [1, 2])
    return run_glp_selection_experiment(
        ORIGINS,
        selector=_make_selector(),
        forecast_generator=_make_forecast_generator(),
        target_variables=["GDP", "DEFL"],
        target_horizons=[1, 2],
        search_config=GLPSearchConfig.reduced_lambda_theta_miu(),
        loss_config=LossConfig(),
        validation_scheme=_scheme(),
        plan=plan,
        schedule=schedule or SelectionSchedule.once(),
        system_variables=SYSTEM_VARS,
        system_horizons=SYSTEM_HORS,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Cell-to-forecast mapping for the four scopes.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scope", ["pooled", "horizon", "variable", "variable_horizon"])
def test_canonical_panel_full_coverage_no_duplicates(scope):
    result = _run(scope)
    rows = result.forecast_panel
    keys = [(r["forecast_origin"], r["variable"], r["horizon"]) for r in rows]
    # Full coverage: 8 origins * 2 vars * 2 horizons = 32 canonical rows.
    assert len(rows) == 32
    assert len(set(keys)) == len(keys)  # no duplicate canonical rows
    covered = {(r["variable"], r["horizon"]) for r in rows}
    assert covered == {("GDP", 1), ("GDP", 2), ("DEFL", 1), ("DEFL", 2)}


def test_pooled_scope_all_targets_from_one_cell():
    result = _run("pooled")
    cell_ids = {r["cell_id"] for r in result.forecast_panel}
    assert cell_ids == {"pooled"}


def test_horizon_scope_maps_by_horizon():
    result = _run("horizon")
    for row in result.forecast_panel:
        assert row["cell_id"] == f"horizon-h{row['horizon']}"


def test_variable_scope_maps_by_variable():
    result = _run("variable")
    for row in result.forecast_panel:
        assert row["cell_id"] == f"variable-{row['variable'].lower()}"


def test_variable_horizon_scope_maps_by_pair():
    result = _run("variable_horizon")
    for row in result.forecast_panel:
        expected = f"variable-{row['variable'].lower()}-h{row['horizon']}"
        assert row["cell_id"] == expected


# --------------------------------------------------------------------------- #
# Schedules and selection-event reuse.
# --------------------------------------------------------------------------- #
def test_selection_once_produces_single_event():
    result = _run("pooled", SelectionSchedule.once())
    event_ids = {r["selection_event_id"] for r in result.forecast_panel}
    assert len(event_ids) == 1
    assert len(result.selected_hyperparameters) == 1  # 1 event * 1 cell


def test_selection_every_origin():
    result = _run("pooled", SelectionSchedule.every_origin())
    event_ids = {r["selection_event_id"] for r in result.forecast_panel}
    assert len(event_ids) == len(ORIGINS)
    assert len(result.selected_hyperparameters) == len(ORIGINS)


def test_selection_every_n_origins_reuses_event_ids():
    result = _run("pooled", SelectionSchedule.every_n_origins(4))
    # 8 origins, every 4 -> events at origin 0 and 4.
    events = sorted({r["selection_event_id"] for r in result.forecast_panel})
    assert len(events) == 2
    # Origins 0-3 use event 0, origins 4-7 use event 1.
    by_origin = {r["origin_index"]: r["selection_event_id"] for r in result.forecast_panel}
    assert by_origin[0] == by_origin[3]
    assert by_origin[4] == by_origin[7]
    assert by_origin[0] != by_origin[4]


def test_variable_horizon_every_n_origins_record_count():
    plan = build_selection_plan("variable_horizon", ["GDP", "DEFL"], [1, 2])
    result = run_glp_selection_experiment(
        ORIGINS,
        selector=_make_selector(),
        forecast_generator=_make_forecast_generator(),
        target_variables=["GDP", "DEFL"],
        target_horizons=[1, 2],
        search_config=GLPSearchConfig.reduced_lambda_theta_miu(),
        loss_config=LossConfig(),
        validation_scheme=_scheme(),
        plan=plan,
        schedule=SelectionSchedule.every_n_origins(4),
        system_variables=SYSTEM_VARS,
        system_horizons=SYSTEM_HORS,
    )
    # 2 events * 4 cells = 8 hyperparameter records.
    assert len(result.selected_hyperparameters) == 8


# --------------------------------------------------------------------------- #
# Caching.
# --------------------------------------------------------------------------- #
def test_cache_hit_for_identical_hyperparameters():
    # A pooled plan reuses one system across origins within an event; the same
    # (system, origin) never regenerates.
    call_counter = []
    plan = build_selection_plan("pooled", ["GDP", "DEFL"], [1, 2])
    run_glp_selection_experiment(
        ORIGINS,
        selector=_make_selector(),
        forecast_generator=_make_forecast_generator(call_counter),
        target_variables=["GDP", "DEFL"],
        target_horizons=[1, 2],
        search_config=GLPSearchConfig.reduced_lambda_theta_miu(),
        loss_config=LossConfig(),
        validation_scheme=_scheme(),
        plan=plan,
        schedule=SelectionSchedule.once(),
        system_variables=SYSTEM_VARS,
        system_horizons=SYSTEM_HORS,
    )
    # One pooled cell, 8 origins -> exactly 8 generations (one per origin).
    assert len(call_counter) == 8


def test_identical_cells_do_not_refit_same_system():
    # Force two cells to select identical values, then confirm one generation
    # per origin (a cache hit for the second identical cell).
    def constant_selector(request):
        return CellSelection(
            natural_vector=(0.2, 1.0, 1.0),
            named_parameters={"lambda": 0.2, "theta": 1.0, "miu": 1.0},
            selection_loss=0.2,
        )

    counter = []
    plan = build_selection_plan("variable", ["GDP", "DEFL"], [1, 2])
    result = run_glp_selection_experiment(
        ORIGINS[:2],
        selector=constant_selector,
        forecast_generator=_make_forecast_generator(counter),
        target_variables=["GDP", "DEFL"],
        target_horizons=[1, 2],
        search_config=GLPSearchConfig.reduced_lambda_theta_miu(),
        loss_config=LossConfig(),
        validation_scheme=_scheme(),
        plan=plan,
        schedule=SelectionSchedule.once(),
        system_variables=SYSTEM_VARS,
        system_horizons=SYSTEM_HORS,
    )
    # 2 cells select identical systems -> 1 generation per origin, 2 origins.
    assert len(counter) == 2
    assert result.cache_stats["hits"] == 2  # second cell hits per origin


def test_cache_invalidation_across_vintages():
    counter_a = []
    counter_b = []
    common = dict(
        selector=_make_selector(),
        target_variables=["GDP", "DEFL"],
        target_horizons=[1, 2],
        search_config=GLPSearchConfig.reduced_lambda_theta_miu(),
        loss_config=LossConfig(),
        validation_scheme=_scheme(),
        plan=build_selection_plan("pooled", ["GDP", "DEFL"], [1, 2]),
        schedule=SelectionSchedule.once(),
        system_variables=SYSTEM_VARS,
        system_horizons=SYSTEM_HORS,
    )
    run_glp_selection_experiment(
        ORIGINS[:2], forecast_generator=_make_forecast_generator(counter_a),
        vintage_token="2020-01-01", **common,
    )
    run_glp_selection_experiment(
        ORIGINS[:2], forecast_generator=_make_forecast_generator(counter_b),
        vintage_token="2021-01-01", **common,
    )
    # Different vintages never share cache entries: each run regenerates fully.
    assert len(counter_a) == 2
    assert len(counter_b) == 2


# --------------------------------------------------------------------------- #
# Off-target diagnostics and output linkage.
# --------------------------------------------------------------------------- #
def test_off_target_forecasts_excluded_from_canonical_panel():
    result = _run("variable", retain_off_target=True)
    # Canonical panel only holds target variables/horizons.
    for row in result.forecast_panel:
        assert row["variable"] in {"GDP", "DEFL"}
        assert row["horizon"] in {1, 2}
    # Diagnostic panel holds the full system (including FFR and horizon 4).
    diag_vars = {r["variable"] for r in result.forecast_panel_all_cells}
    diag_hors = {r["horizon"] for r in result.forecast_panel_all_cells}
    assert "FFR" in diag_vars
    assert 4 in diag_hors
    assert all(r["diagnostic_only"] for r in result.forecast_panel_all_cells)


def test_every_forecast_row_links_to_a_hyperparameter_record():
    result = _run("variable_horizon", SelectionSchedule.every_n_origins(4))
    keys = {
        (r["selection_event_id"], r["cell_id"]) for r in result.selected_hyperparameters
    }
    for row in result.forecast_panel:
        assert (row["selection_event_id"], row["cell_id"]) in keys


def test_deterministic_output_ordering():
    result = _run("variable_horizon", SelectionSchedule.every_origin())
    panel_keys = [
        (r["model"], r["origin_index"], r["variable"], r["horizon"])
        for r in result.forecast_panel
    ]
    assert panel_keys == sorted(panel_keys)
    hyper_keys = [
        (r["applies_from_index"], r["cell_id"]) for r in result.selected_hyperparameters
    ]
    assert hyper_keys == sorted(hyper_keys)


# --------------------------------------------------------------------------- #
# Legacy one-cell behavior and metadata.
# --------------------------------------------------------------------------- #
def test_legacy_no_plan_defaults_to_single_pooled_cell():
    result = run_glp_selection_experiment(
        ORIGINS,
        selector=_make_selector(),
        forecast_generator=_make_forecast_generator(),
        target_variables=["GDP", "DEFL"],
        target_horizons=[1, 2],
        search_config=GLPSearchConfig.reduced_lambda_theta_miu(),
        loss_config=LossConfig(),
        validation_scheme=_scheme(),
        system_variables=SYSTEM_VARS,
        system_horizons=SYSTEM_HORS,
    )
    assert result.run_metadata["selection_plan"]["scope"] == "pooled"
    assert {r["cell_id"] for r in result.forecast_panel} == {"pooled"}
    assert len(result.selected_hyperparameters) == 1


def test_run_metadata_serializes_full_configuration():
    import json

    result = _run("horizon", SelectionSchedule.annual_quarterly())
    meta = result.run_metadata
    assert meta["selection_plan"]["scope"] == "horizon"
    assert meta["schedule"]["kind"] == "every_n_origins"
    assert "loss_config" in meta
    assert "validation_scheme" in meta
    assert "search_config" in meta
    assert meta["vintage_policy"] == "outer_vintage_consistent"
    # Fully JSON-serializable.
    json.dumps(meta)


def test_serial_and_repeated_runs_are_identical():
    first = _run("variable_horizon", SelectionSchedule.every_n_origins(4))
    second = _run("variable_horizon", SelectionSchedule.every_n_origins(4))
    assert first.forecast_panel == second.forecast_panel
    assert first.selected_hyperparameters == second.selected_hyperparameters
