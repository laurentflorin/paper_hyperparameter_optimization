"""Tests for the model-independent reporting and comparison layer."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common_hpo.reporting import (
    DuplicateForecastError,
    PanelSpec,
    RealizationMismatchError,
    ScopeContrast,
    average_ranks,
    bootstrap_intervals,
    check_origin_alignment,
    combine_panels,
    computational_cost,
    dm_tests,
    failure_summary,
    hyperparameter_summary,
    load_failed_origins,
    load_forecast_panel,
    load_selected_hyperparameters,
    mae_by_target,
    relative_rmse,
    rmse_by_target,
    scope_gain_summary,
    scope_gains,
    selection_stability,
    standard_scope_contrasts,
    write_comparison_summary,
)


# --------------------------------------------------------------------------- #
# Panel builders
# --------------------------------------------------------------------------- #
def _metric_panel(strategy, origins, variables, horizons, error_fn, *, method="iterated"):
    rows = []
    for oi, origin in enumerate(origins):
        for variable in variables:
            for h in horizons:
                err = error_fn(oi, variable, h)
                rows.append(
                    {
                        "strategy": strategy,
                        "forecast_origin": origin,
                        "group": "all",
                        "target_quarter": f"{origin}+{h}",
                        "horizon_quarters": h,
                        "variable": variable,
                        "forecast_method": method,
                        "actual_level": np.nan,
                        "actual_metric": 0.0,
                        "mean_level": np.nan,
                        "mean_metric": err,  # actual=0 => error = forecast
                        "median_level": np.nan,
                        "median_metric": err,
                        "error_metric": err,
                    }
                )
    return pd.DataFrame(rows)


def _glp_panel(strategy, origins, variables, horizons, error_fn, *, size="small"):
    rows = []
    for oi, origin in enumerate(origins):
        for variable in variables:
            for h in horizons:
                err = error_fn(oi, variable, h)
                rows.append(
                    {
                        "strategy": strategy,
                        "model_size": size,
                        "forecast_origin": origin,
                        "target_quarter": f"{origin}+{h}",
                        "horizon_quarters": h,
                        "variable": variable,
                        "mean": err,
                        "actual": 0.0,
                        "error": err,
                    }
                )
    return pd.DataFrame(rows)


ORIGINS = [f"2000Q{i}" for i in range(1, 13)]


# --------------------------------------------------------------------------- #
# RMSE / MAE hand checks
# --------------------------------------------------------------------------- #
def test_rmse_and_mae_hand_computed():
    errors = [1.0, -1.0, 2.0, -2.0]
    df = _metric_panel("m", ORIGINS[:4], ["x"], [1], lambda oi, v, h: errors[oi])
    spec = PanelSpec(model="m", family="ridge", scope="pooled", selection="forecast_loss",
                     forecast_method="iterated")
    combined = combine_panels([load_forecast_panel(df, spec)])
    rmse = rmse_by_target(combined)
    mae = mae_by_target(combined)
    assert rmse.loc[0, "rmse"] == pytest.approx(np.sqrt((1 + 1 + 4 + 4) / 4))
    assert mae.loc[0, "mae"] == pytest.approx((1 + 1 + 2 + 2) / 4)
    assert rmse.loc[0, "n"] == 4


# --------------------------------------------------------------------------- #
# Relative RMSE
# --------------------------------------------------------------------------- #
def test_relative_rmse_against_baseline():
    base = _metric_panel("base", ORIGINS[:4], ["x"], [1], lambda oi, v, h: 1.0)  # rmse 1
    comp = _metric_panel("comp", ORIGINS[:4], ["x"], [1], lambda oi, v, h: 2.0)  # rmse 2
    b = load_forecast_panel(base, PanelSpec(model="base", family="ridge"))
    c = load_forecast_panel(comp, PanelSpec(model="comp", family="ridge"))
    combined = combine_panels([b, c])
    rmse = rmse_by_target(combined)
    rel = relative_rmse(rmse, baseline_model="base")
    comp_row = rel[rel["model"] == "comp"].iloc[0]
    assert comp_row["relative_rmse"] == pytest.approx(2.0)
    assert comp_row["relative_rmse_pct"] == pytest.approx(100.0)


# --------------------------------------------------------------------------- #
# Ranks and ties
# --------------------------------------------------------------------------- #
def test_average_ranks_with_ties():
    a = _metric_panel("a", ORIGINS[:4], ["x"], [1], lambda oi, v, h: 1.0)  # rmse 1
    b = _metric_panel("b", ORIGINS[:4], ["x"], [1], lambda oi, v, h: 1.0)  # rmse 1 (tie)
    c = _metric_panel("c", ORIGINS[:4], ["x"], [1], lambda oi, v, h: 3.0)  # rmse 3
    frames = [
        load_forecast_panel(a, PanelSpec(model="a", family="ridge")),
        load_forecast_panel(b, PanelSpec(model="b", family="ridge")),
        load_forecast_panel(c, PanelSpec(model="c", family="ridge")),
    ]
    combined = combine_panels(frames)
    ranks = average_ranks(rmse_by_target(combined))
    ranks = ranks.set_index("model")
    # a and b tie for ranks 1,2 -> average 1.5; c gets rank 3.
    assert ranks.loc["a", "average_rank"] == pytest.approx(1.5)
    assert ranks.loc["b", "average_rank"] == pytest.approx(1.5)
    assert ranks.loc["c", "average_rank"] == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# Scope-gain signs
# --------------------------------------------------------------------------- #
def _ridge_scope_panel(scope, error_value):
    df = _metric_panel(f"ridge_{scope}", ORIGINS, ["x"], [1],
                       lambda oi, v, h: error_value)
    return load_forecast_panel(
        df,
        PanelSpec(model=f"ridge_{scope}", family="ridge", size="small",
                  scope=scope, selection="forecast_loss", forecast_method="iterated"),
    )


def test_scope_gain_signs():
    # pooled worse (err 2) than horizon/variable (err 1) and variable_horizon best (err 0.5).
    panels = [
        _ridge_scope_panel("pooled", 2.0),
        _ridge_scope_panel("horizon", 1.0),
        _ridge_scope_panel("variable", 1.0),
        _ridge_scope_panel("variable_horizon", 0.5),
    ]
    combined = combine_panels(panels)
    gains = scope_gains(combined)
    row = gains.iloc[0]
    assert row["horizon_gain"] == pytest.approx(1.0)  # 2 - 1 > 0 improvement
    assert row["variable_gain"] == pytest.approx(1.0)
    assert row["interaction_gain"] == pytest.approx(0.5)  # min(1,1) - 0.5
    assert row["vh_vs_pooled_gain"] == pytest.approx(1.5)
    summary = scope_gain_summary(gains)
    hor = summary[summary["gain"] == "horizon_gain"].iloc[0]
    assert hor["proportion_improved"] == pytest.approx(1.0)
    assert hor["worst_deterioration"] == pytest.approx(0.0)


def test_scope_gain_reports_deterioration():
    # horizon-specific selection is worse than pooled -> negative horizon gain.
    panels = [
        _ridge_scope_panel("pooled", 1.0),
        _ridge_scope_panel("horizon", 2.0),
    ]
    combined = combine_panels(panels)
    gains = scope_gains(combined)
    assert gains.iloc[0]["horizon_gain"] == pytest.approx(-1.0)
    summary = scope_gain_summary(gains)
    hor = summary[summary["gain"] == "horizon_gain"].iloc[0]
    assert hor["worst_deterioration"] == pytest.approx(-1.0)
    assert hor["proportion_improved"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Duplicate-row rejection
# --------------------------------------------------------------------------- #
def test_duplicate_rows_are_rejected():
    df = _metric_panel("m", ORIGINS[:2], ["x"], [1], lambda oi, v, h: 1.0)
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # duplicate one key
    with pytest.raises(DuplicateForecastError):
        load_forecast_panel(dup, PanelSpec(model="m", family="ridge"))


# --------------------------------------------------------------------------- #
# Mismatched-origin detection
# --------------------------------------------------------------------------- #
def test_mismatched_origin_detection():
    a = _metric_panel("a", ORIGINS[:6], ["x"], [1], lambda oi, v, h: 1.0)
    b = _metric_panel("b", ORIGINS[3:9], ["x"], [1], lambda oi, v, h: 1.0)
    combined = combine_panels([
        load_forecast_panel(a, PanelSpec(model="a", family="ridge")),
        load_forecast_panel(b, PanelSpec(model="b", family="ridge")),
    ])
    report = check_origin_alignment(combined)
    assert not report.is_aligned
    assert len(report.unmatched["a"]) == 3  # origins 0,1,2 not shared
    assert len(report.unmatched["b"]) == 3  # origins 6,7,8 not shared
    assert len(report.common_origins) == 3


# --------------------------------------------------------------------------- #
# Realization definition mismatch
# --------------------------------------------------------------------------- #
def test_realization_mismatch_is_rejected():
    a = _metric_panel("a", ORIGINS[:4], ["x"], [1], lambda oi, v, h: 1.0)
    b = _metric_panel("b", ORIGINS[:4], ["x"], [1], lambda oi, v, h: 1.0)
    b["actual_metric"] = 5.0  # different realization definition
    with pytest.raises(RealizationMismatchError):
        combine_panels([
            load_forecast_panel(a, PanelSpec(model="a", family="ridge")),
            load_forecast_panel(b, PanelSpec(model="b", family="ridge")),
        ])


# --------------------------------------------------------------------------- #
# Legacy GLP directory adapter
# --------------------------------------------------------------------------- #
def test_legacy_glp_adapter_normalizes_schema():
    glp = _glp_panel("paper", ORIGINS[:4], ["x"], [1], lambda oi, v, h: 1.5, size="small")
    loaded = load_forecast_panel(glp, PanelSpec(model="glp_paper", family="glp",
                                                scope="native", selection="native"))
    assert set(loaded.columns) == set(
        ["model", "family", "size", "scope", "selection", "forecast_method",
         "group", "forecast_origin", "target_quarter", "horizon", "variable",
         "forecast", "actual", "error"]
    )
    assert (loaded["forecast_method"] == "native").all()
    assert (loaded["group"] == "all").all()
    assert (loaded["size"] == "small").all()
    # GLP mean/actual/error mapped through.
    assert loaded["forecast"].iloc[0] == pytest.approx(1.5)


def test_glp_and_metric_panels_combine_when_realizations_match():
    glp = _glp_panel("paper", ORIGINS[:6], ["x"], [1], lambda oi, v, h: 1.0)
    ridge = _metric_panel("ridge", ORIGINS[:6], ["x"], [1], lambda oi, v, h: 0.8)
    combined = combine_panels([
        load_forecast_panel(glp, PanelSpec(model="glp", family="glp")),
        load_forecast_panel(ridge, PanelSpec(model="ridge", family="ridge",
                                             scope="pooled", selection="forecast_loss",
                                             forecast_method="iterated")),
    ])
    rmse = rmse_by_target(combined)
    assert set(rmse["model"]) == {"glp", "ridge"}


# --------------------------------------------------------------------------- #
# DM tests via reporting layer
# --------------------------------------------------------------------------- #
def test_dm_overlapping_horizon_lag_and_holm_present():
    rng = np.random.default_rng(0)
    err_a = {oi: rng.normal() for oi in range(len(ORIGINS))}
    a = _metric_panel("a", ORIGINS, ["x"], [4], lambda oi, v, h: err_a[oi] + 0.5)
    b = _metric_panel("b", ORIGINS, ["x"], [4], lambda oi, v, h: err_a[oi])
    combined = combine_panels([
        load_forecast_panel(a, PanelSpec(model="a", family="ridge")),
        load_forecast_panel(b, PanelSpec(model="b", family="ridge")),
    ])
    dm = dm_tests(combined, [ScopeContrast("a_vs_b", "a", "b")])
    row = dm.iloc[0]
    assert row["hac_lag"] == 3  # horizon 4 -> h-1
    assert "holm_p_value" in dm.columns


def test_dm_zero_variance_cell_is_invalid():
    a = _metric_panel("a", ORIGINS, ["x"], [1], lambda oi, v, h: 1.0)
    b = _metric_panel("b", ORIGINS, ["x"], [1], lambda oi, v, h: 1.0)  # identical
    combined = combine_panels([
        load_forecast_panel(a, PanelSpec(model="a", family="ridge")),
        load_forecast_panel(b, PanelSpec(model="b", family="ridge")),
    ])
    dm = dm_tests(combined, [ScopeContrast("a_vs_b", "a", "b")])
    row = dm.iloc[0]
    assert row["valid"] is False or bool(row["valid"]) is False
    assert row["p_value"] is None or pd.isna(row["p_value"])
    assert pd.isna(row["holm_p_value"])


def test_dm_non_paired_cell_is_invalid():
    # No shared origins for the comparison -> non-paired, refused.
    a = _metric_panel("a", ORIGINS[:6], ["x"], [1], lambda oi, v, h: 1.0)
    b = _metric_panel("b", ORIGINS[6:], ["x"], [1], lambda oi, v, h: 2.0)
    combined = combine_panels([
        load_forecast_panel(a, PanelSpec(model="a", family="ridge")),
        load_forecast_panel(b, PanelSpec(model="b", family="ridge")),
    ])
    dm = dm_tests(combined, [ScopeContrast("a_vs_b", "a", "b")])
    row = dm.iloc[0]
    assert bool(row["valid"]) is False
    assert row["n"] == 0


# --------------------------------------------------------------------------- #
# Bootstrap reproducibility (reporting layer)
# --------------------------------------------------------------------------- #
def test_bootstrap_intervals_reproducible():
    rng = np.random.default_rng(1)
    err = {oi: rng.normal() for oi in range(len(ORIGINS))}
    a = _metric_panel("a", ORIGINS, ["x"], [1], lambda oi, v, h: err[oi] + 0.3)
    b = _metric_panel("b", ORIGINS, ["x"], [1], lambda oi, v, h: err[oi])
    combined = combine_panels([
        load_forecast_panel(a, PanelSpec(model="a", family="ridge")),
        load_forecast_panel(b, PanelSpec(model="b", family="ridge")),
    ])
    contrast = [ScopeContrast("a_vs_b", "a", "b")]
    r1 = bootstrap_intervals(combined, contrast, block_length=4, n_boot=200, seed=7)
    r2 = bootstrap_intervals(combined, contrast, block_length=4, n_boot=200, seed=7)
    pd.testing.assert_frame_equal(r1, r2)


# --------------------------------------------------------------------------- #
# Direct-versus-iterated filtering
# --------------------------------------------------------------------------- #
def test_direct_versus_iterated_contrast_is_built_and_scoped():
    iterated = _metric_panel("ridge_iter", ORIGINS, ["x"], [1],
                             lambda oi, v, h: 1.0, method="iterated")
    direct = _metric_panel("ridge_direct", ORIGINS, ["x"], [1],
                           lambda oi, v, h: 0.8, method="direct")
    combined = combine_panels([
        load_forecast_panel(iterated, PanelSpec(model="ridge_iter", family="ridge",
                                                scope="pooled", selection="forecast_loss",
                                                forecast_method="iterated")),
        load_forecast_panel(direct, PanelSpec(model="ridge_direct", family="ridge_direct",
                                              scope="pooled", selection="forecast_loss",
                                              forecast_method="direct")),
    ])
    contrasts = standard_scope_contrasts(combined)
    names = [c.name for c in contrasts]
    assert any("direct_vs_iterated[ridge:pooled]" == n for n in names)
    dv = next(c for c in contrasts if c.name.startswith("direct_vs_iterated"))
    # The contrast pairs the direct model against the iterated one.
    assert dv.model_a == "ridge_direct" and dv.model_b == "ridge_iter"


def test_native_and_forecast_loss_pooled_not_conflated():
    # A native-selection ridge and a forecast-loss pooled ridge are distinct
    # models; the pooled_vs_native contrast pairs them without merging.
    native = _metric_panel("ridge_native", ORIGINS, ["x"], [1], lambda oi, v, h: 1.2)
    pooled = _metric_panel("ridge_pooled", ORIGINS, ["x"], [1], lambda oi, v, h: 1.0)
    combined = combine_panels([
        load_forecast_panel(native, PanelSpec(model="ridge_native", family="ridge",
                                              size="small", scope="native", selection="native",
                                              forecast_method="iterated")),
        load_forecast_panel(pooled, PanelSpec(model="ridge_pooled", family="ridge",
                                              size="small", scope="pooled", selection="forecast_loss",
                                              forecast_method="iterated")),
    ])
    contrasts = standard_scope_contrasts(combined)
    pvn = [c for c in contrasts if c.name.startswith("pooled_vs_native")]
    assert pvn
    assert pvn[0].model_a == "ridge_pooled" and pvn[0].model_b == "ridge_native"


# --------------------------------------------------------------------------- #
# Hyperparameter summary, stability, failures, cost
# --------------------------------------------------------------------------- #
def test_hyperparameter_summary_and_stability():
    hp = pd.DataFrame(
        {
            "forecast_origin": ORIGINS[:4],
            "strategy": "ridge_var",
            "cell_id": "pooled",
            "param_lam": [1.0, 1.0, 10.0, 10.0],
            "param_p": [2, 2, 2, 2],
        }
    )
    loaded = load_selected_hyperparameters(hp, PanelSpec(model="ridge", family="ridge",
                                                         scope="pooled", selection="forecast_loss"))
    summary = hyperparameter_summary(loaded)
    lam = summary[summary["parameter"] == "param_lam"].iloc[0]
    assert lam["min"] == pytest.approx(1.0)
    assert lam["max"] == pytest.approx(10.0)
    assert lam["n"] == 4

    stability = selection_stability(loaded)
    lam_stab = stability[stability["parameter"] == "param_lam"].iloc[0]
    assert lam_stab["n_unique"] == 2
    assert lam_stab["n_switches"] == 1  # one change from 1 to 10 across ordered origins
    p_stab = stability[stability["parameter"] == "param_p"].iloc[0]
    assert p_stab["n_switches"] == 0
    assert p_stab["modal_fraction"] == pytest.approx(1.0)


def test_failure_summary_counts_by_stage():
    failures = pd.DataFrame(
        {
            "forecast_origin": ["o1", "o2", "o3"],
            "cell_id": ["c", "c", "c"],
            "stage": ["selection", "forecast", "forecast"],
            "error": ["e", "e", "e"],
        }
    )
    loaded = load_failed_origins(failures, PanelSpec(model="ridge", family="ridge"))
    summary = failure_summary(loaded)
    forecast_row = summary[(summary["model"] == "ridge") & (summary["stage"] == "forecast")].iloc[0]
    assert forecast_row["n_failures"] == 2
    total_row = summary[(summary["model"] == "ridge") & (summary["stage"] == "__total__")].iloc[0]
    assert total_row["n_failures"] == 3


def test_failure_summary_glp_without_stage():
    failures = pd.DataFrame({"forecast_origin": ["o1", "o2"], "error": ["e", "e"]})
    loaded = load_failed_origins(failures, PanelSpec(model="glp", family="glp"))
    assert (loaded["stage"] == "unknown").all()


def test_computational_cost_extracts_documented_keys():
    metadata = {
        "ridge": {"strategy": "ridge_var", "n_outer_origins": 8, "grid_size": 36,
                  "n_target_cells": 1},
        "glp": {"strategy": "paper", "n_workers": 4, "n_origins_completed": 100},
    }
    cost = computational_cost(metadata)
    assert set(cost["model"]) == {"ridge", "glp"}
    ridge_row = cost[cost["model"] == "ridge"].iloc[0]
    assert ridge_row["grid_size"] == 36
    assert pd.isna(ridge_row["n_workers"])  # missing -> NaN, stable schema


# --------------------------------------------------------------------------- #
# Markdown summary + end-to-end via runner
# --------------------------------------------------------------------------- #
def test_write_comparison_summary(tmp_path):
    panels = [
        _ridge_scope_panel("pooled", 2.0),
        _ridge_scope_panel("horizon", 1.0),
    ]
    combined = combine_panels(panels)
    gains = scope_gains(combined)
    summary = scope_gain_summary(gains)
    ranks = average_ranks(rmse_by_target(combined))
    contrasts = standard_scope_contrasts(combined)
    dm = dm_tests(combined, contrasts) if contrasts else pd.DataFrame(columns=["valid"])
    alignment = check_origin_alignment(combined)
    text = write_comparison_summary(
        tmp_path / "comparison_summary.md",
        gains=gains, gain_summary=summary, average_rank_table=ranks,
        dm_table=dm, alignment=alignment,
    )
    assert (tmp_path / "comparison_summary.md").exists()
    assert "Scope-gain decomposition" in text


def test_end_to_end_runner(tmp_path):
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import compare_scope_study as runner

    # Lay out two ridge scope directories with canonical files.
    import json

    manifest = {"baseline_model": "ridge_pooled", "panels": []}
    for scope, err in (("pooled", 2.0), ("variable_horizon", 1.0)):
        d = tmp_path / f"scope_{scope}"
        d.mkdir()
        panel = _metric_panel(f"ridge_{scope}", ORIGINS, ["x", "y"], [1, 2],
                              lambda oi, v, h: err)
        panel.to_csv(d / "forecast_panel.csv", index=False)
        pd.DataFrame(
            {"forecast_origin": ORIGINS, "strategy": "ridge_var", "group": "all",
             "cell_id": scope, "param_lam": 1.0, "param_p": 2}
        ).to_csv(d / "selected_hyperparameters.csv", index=False)
        (d / "run_metadata.json").write_text(json.dumps(
            {"strategy": "ridge_var", "n_outer_origins": len(ORIGINS), "grid_size": 4}
        ))
        manifest["panels"].append(
            {"model": f"ridge_{scope}", "family": "ridge", "scope": scope,
             "selection": "forecast_loss", "forecast_method": "iterated",
             "size": "small", "dir": f"scope_{scope}"}
        )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    rc = runner.main([
        "--manifest", str(manifest_path),
        "--root", str(tmp_path),
        "--output-dir", str(tmp_path / "out"),
    ])
    assert rc == 0
    out = tmp_path / "out"
    for name in (
        "rmse_by_target.csv", "mae_by_target.csv", "relative_rmse.csv",
        "average_ranks.csv", "scope_gains.csv", "hyperparameter_summary.csv",
        "selection_stability.csv", "failure_summary.csv", "computational_cost.csv",
        "dm_tests.csv", "bootstrap_intervals.csv", "comparison_summary.md",
    ):
        assert (out / name).exists(), f"missing {name}"

    # scope_gains.csv answers the paper question directly.
    gains = pd.read_csv(out / "scope_gains.csv")
    assert "vh_vs_pooled_gain" in gains.columns
    assert (gains["vh_vs_pooled_gain"] > 0).all()  # variable_horizon beats pooled
