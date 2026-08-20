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

import common_hpo.reporting as reporting_module
from common_hpo.reporting import (
    CoverageError,
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
    restrict_to_common_sample,
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


def test_computational_cost_accepts_legacy_total_fits_alias():
    metadata = {
        "ridge": {"strategy": "ridge_var", "estimated_total_fits": 120,
                  "wall_time_seconds": 3.5},
        "glp": {"strategy": "paper", "total_fits": 7},
    }
    cost = computational_cost(metadata)
    ridge_row = cost[cost["model"] == "ridge"].iloc[0]
    glp_row = cost[cost["model"] == "glp"].iloc[0]
    assert ridge_row["total_fits"] == 120
    assert ridge_row["wall_time_seconds"] == 3.5
    assert glp_row["total_fits"] == 7


# --------------------------------------------------------------------------- #
# Common-sample restriction (prior-audit RANK-02 regression)
# --------------------------------------------------------------------------- #
def _unbalanced_scope_panels():
    """Two scopes of one ridge system; the second misses the first 2 of 8 origins.

    Model ``a`` (pooled) has errors 10, 10 on exactly the two origins that model
    ``b`` (horizon) is missing, and 1.0 elsewhere; model ``b`` has 2.0 everywhere.
    Unpaired:      RMSE(a) = 5.074 on n=8, RMSE(b) = 2.0 on n=6  -> b "wins".
    Common sample: RMSE(a) = 1.0   on n=6, RMSE(b) = 2.0 on n=6  -> a wins.
    """

    origins = ORIGINS[:8]
    a = _metric_panel("ridge_pooled", origins, ["x"], [1],
                      lambda oi, v, h: 10.0 if oi < 2 else 1.0)
    b = _metric_panel("ridge_horizon", origins[2:], ["x"], [1],
                      lambda oi, v, h: 2.0)
    return [
        load_forecast_panel(a, PanelSpec(model="ridge_pooled", family="ridge", size="small",
                                         scope="pooled", selection="forecast_loss",
                                         forecast_method="iterated")),
        load_forecast_panel(b, PanelSpec(model="ridge_horizon", family="ridge", size="small",
                                         scope="horizon", selection="forecast_loss",
                                         forecast_method="iterated")),
    ]


def test_unpaired_vs_common_sample_winner_reversal():
    combined = combine_panels(_unbalanced_scope_panels())

    raw = rmse_by_target(combined, common_sample=False).set_index("model")
    assert raw.loc["ridge_pooled", "rmse"] == pytest.approx(np.sqrt(206.0 / 8))  # 5.0744
    assert raw.loc["ridge_pooled", "n"] == 8
    assert raw.loc["ridge_horizon", "rmse"] == pytest.approx(2.0)
    assert raw.loc["ridge_horizon", "n"] == 6
    # Unpaired aggregation picks the WRONG winner (b) ...
    assert raw["rmse"].idxmin() == "ridge_horizon"

    common = rmse_by_target(combined).set_index("model")
    assert common.loc["ridge_pooled", "rmse"] == pytest.approx(1.0)
    assert common.loc["ridge_horizon", "rmse"] == pytest.approx(2.0)
    assert (common["n"] == 6).all()
    # ... the common sample picks a.
    assert common["rmse"].idxmin() == "ridge_pooled"

    # And the headline scope gain flips sign: L(pooled) - L(horizon).
    unpaired_gain = scope_gains(combined, common_sample=False).iloc[0]["horizon_gain"]
    paired_gain = scope_gains(combined).iloc[0]["horizon_gain"]
    assert unpaired_gain > 0
    assert unpaired_gain == pytest.approx(np.sqrt(206.0 / 8) - 2.0)
    assert paired_gain < 0
    assert paired_gain == pytest.approx(-1.0)


def test_common_sample_counts_report_exclusions():
    combined = combine_panels(_unbalanced_scope_panels())
    rmse = rmse_by_target(combined).set_index("model")
    assert rmse.loc["ridge_pooled", "n_common"] == 6
    assert rmse.loc["ridge_pooled", "n_model_total"] == 8
    assert rmse.loc["ridge_pooled", "n_excluded"] == 2
    assert rmse.loc["ridge_horizon", "n_common"] == 6
    assert rmse.loc["ridge_horizon", "n_excluded"] == 0

    _, report = restrict_to_common_sample(combined)
    assert report.n_common_keys == 6
    assert report.n_excluded_keys == 2
    assert report.coverage == pytest.approx(12.0 / 14.0)


def test_scope_gains_report_the_paired_sample_size():
    combined = combine_panels(_unbalanced_scope_panels())
    gains = scope_gains(combined)
    assert "n_common" in gains.columns
    row = gains.iloc[0]
    _, _, n_paired = reporting_module._paired_errors(
        combined, "ridge_pooled", "ridge_horizon", "x", 1
    )
    assert n_paired == 6
    assert int(row["n_common"]) == n_paired
    assert int(row["n_models"]) == 2
    assert int(row["n_excluded"]) == 2


def test_coverage_policies_and_min_coverage():
    combined = combine_panels(_unbalanced_scope_panels())
    with pytest.raises(CoverageError):
        restrict_to_common_sample(combined, policy="raise")
    with pytest.raises(CoverageError):
        restrict_to_common_sample(combined, min_coverage=0.99)
    advisory, report = restrict_to_common_sample(combined, policy="advisory")
    assert len(advisory) == len(combined)  # nothing dropped ...
    assert report.n_excluded_keys == 2  # ... but the shortfall is reported.


def test_disjoint_cells_are_not_annihilated():
    """A model absent from a cell must not shrink that cell's sample."""

    a = _metric_panel("a", ORIGINS[:4], ["x"], [1], lambda oi, v, h: 1.0)
    b = _metric_panel("b", ORIGINS[:4], ["y"], [1], lambda oi, v, h: 2.0)
    combined = combine_panels([
        load_forecast_panel(a, PanelSpec(model="a", family="ridge")),
        load_forecast_panel(b, PanelSpec(model="b", family="ridge")),
    ])
    restricted, report = restrict_to_common_sample(combined)
    assert len(restricted) == len(combined)
    assert report.n_excluded_keys == 0


def test_average_ranks_refuses_mixed_sample_sizes():
    combined = combine_panels(_unbalanced_scope_panels())
    unpaired = rmse_by_target(combined, common_sample=False)
    with pytest.raises(CoverageError):
        average_ranks(unpaired)
    ranks = average_ranks(rmse_by_target(combined))
    assert ranks.iloc[0]["model"] == "ridge_pooled"
    assert int(ranks.iloc[0]["n_common"]) == 6


def test_relative_rmse_pairwise_common_sample():
    combined = combine_panels(_unbalanced_scope_panels())
    rmse = rmse_by_target(combined, common_sample=False)
    rel = relative_rmse(rmse, baseline_model="ridge_horizon", combined=combined)
    row = rel[rel["model"] == "ridge_pooled"].iloc[0]
    assert row["rmse"] == pytest.approx(1.0)  # recomputed on the paired sample
    assert row["baseline_rmse"] == pytest.approx(2.0)
    assert row["relative_rmse"] == pytest.approx(0.5)
    assert int(row["n"]) == 6
    assert int(row["n_baseline"]) == 6
    assert row["sample_basis"] == "pairwise_common_with_baseline"


def test_runner_writes_common_sample_table(tmp_path):
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import compare_scope_study as runner
    import json

    manifest = {"baseline_model": "ridge_pooled", "panels": []}
    for scope, origins, err in (
        ("pooled", ORIGINS[:8], lambda oi, v, h: 10.0 if oi < 2 else 1.0),
        ("horizon", ORIGINS[2:8], lambda oi, v, h: 2.0),
    ):
        d = tmp_path / f"scope_{scope}"
        d.mkdir()
        _metric_panel(f"ridge_{scope}", origins, ["x"], [1], err).to_csv(
            d / "forecast_panel.csv", index=False
        )
        manifest["panels"].append(
            {"model": f"ridge_{scope}", "family": "ridge", "scope": scope,
             "selection": "forecast_loss", "forecast_method": "iterated",
             "size": "small", "dir": f"scope_{scope}"}
        )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    out = tmp_path / "out"
    rc = runner.main([
        "--manifest", str(manifest_path), "--root", str(tmp_path),
        "--output-dir", str(out), "--coverage-policy", "restrict",
    ])
    assert rc == 0
    sample = pd.read_csv(out / "common_sample.csv").set_index("model")
    assert (sample["policy"] == "restrict").all()
    assert int(sample.loc["ridge_pooled", "n_excluded"]) == 2
    assert int(sample.loc["__all__", "n_common_keys"]) == 6
    # The globally restricted frame feeds every table: gains and DM agree on n.
    gains = pd.read_csv(out / "scope_gains.csv")
    dm = pd.read_csv(out / "dm_tests.csv")
    assert int(gains.iloc[0]["n_common"]) == int(dm.iloc[0]["n"]) == 6
    assert gains.iloc[0]["horizon_gain"] == pytest.approx(-1.0)

    # A strict policy refuses the unbalanced study outright.
    rc_raise = runner.main([
        "--manifest", str(manifest_path), "--root", str(tmp_path),
        "--output-dir", str(tmp_path / "out_raise"), "--coverage-policy", "raise",
    ])
    assert rc_raise == 1


def _unequal_coverage_study(tmp_path):
    """Two ridge panels with unequal origin coverage (8 vs 6) plus a manifest."""
    import json

    manifest = {"baseline_model": "ridge_pooled", "panels": []}
    for scope, origins, err in (
        ("pooled", ORIGINS[:8], lambda oi, v, h: 10.0 if oi < 2 else 1.0),
        ("horizon", ORIGINS[2:8], lambda oi, v, h: 2.0),
    ):
        d = tmp_path / f"scope_{scope}"
        d.mkdir()
        _metric_panel(f"ridge_{scope}", origins, ["x"], [1], err).to_csv(
            d / "forecast_panel.csv", index=False
        )
        manifest["panels"].append(
            {"model": f"ridge_{scope}", "family": "ridge", "scope": scope,
             "selection": "forecast_loss", "forecast_method": "iterated",
             "size": "small", "dir": f"scope_{scope}"}
        )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


def _runner():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import compare_scope_study as runner

    return runner


def test_cli_advisory_policy_does_not_restrict_any_table(tmp_path):
    """`--coverage-policy advisory` must report, never restrict -- in every table."""
    runner = _runner()
    manifest_path = _unequal_coverage_study(tmp_path)
    out = tmp_path / "out_advisory"
    rc = runner.main([
        "--manifest", str(manifest_path), "--root", str(tmp_path),
        "--output-dir", str(out), "--coverage-policy", "advisory",
    ])
    assert rc == 0

    # common_sample.csv still *reports* the shortfall...
    sample = pd.read_csv(out / "common_sample.csv").set_index("model")
    assert (sample["policy"] == "advisory").all()
    assert int(sample.loc["ridge_pooled", "n_excluded"]) == 2

    # ...but no table is allowed to act on it: n is each model's own row count.
    rmse = pd.read_csv(out / "rmse_by_target.csv").set_index("model")
    assert int(rmse.loc["ridge_pooled", "n"]) == 8
    assert int(rmse.loc["ridge_horizon", "n"]) == 6
    # unrestricted pooled RMSE keeps the two large early errors
    assert rmse.loc["ridge_pooled", "rmse"] == pytest.approx(np.sqrt(206.0 / 8.0))

    mae = pd.read_csv(out / "mae_by_target.csv").set_index("model")
    assert int(mae.loc["ridge_pooled", "n"]) == 8
    assert int(mae.loc["ridge_horizon", "n"]) == 6

    rel = pd.read_csv(out / "relative_rmse.csv").set_index("model")
    assert int(rel.loc["ridge_pooled", "n"]) == 8
    assert int(rel.loc["ridge_horizon", "n"]) == 6

    gains = pd.read_csv(out / "scope_gains.csv")
    assert gains.iloc[0]["L_pooled"] == pytest.approx(np.sqrt(206.0 / 8.0))
    assert gains.iloc[0]["L_horizon"] == pytest.approx(2.0)
    assert gains.iloc[0]["sample_basis"] == "unpaired_per_model"

    # ranks must not blow up merely because the samples differ under advisory
    ranks = pd.read_csv(out / "average_ranks.csv")
    assert set(ranks["model"]) == {"ridge_pooled", "ridge_horizon"}
    assert (ranks["sample_basis"] == "unpaired_per_model").all()


def test_cli_restrict_policy_restricts_every_table(tmp_path):
    """`--coverage-policy restrict` puts every table on the one common sample."""
    runner = _runner()
    manifest_path = _unequal_coverage_study(tmp_path)
    out = tmp_path / "out_restrict"
    rc = runner.main([
        "--manifest", str(manifest_path), "--root", str(tmp_path),
        "--output-dir", str(out), "--coverage-policy", "restrict",
    ])
    assert rc == 0

    for name, column in (("rmse_by_target", "rmse"), ("mae_by_target", "mae")):
        table = pd.read_csv(out / f"{name}.csv")
        assert set(table["n"]) == {6}
        assert set(table["n_common"]) == {6}
        del column

    rmse = pd.read_csv(out / "rmse_by_target.csv").set_index("model")
    assert rmse.loc["ridge_pooled", "rmse"] == pytest.approx(1.0)  # the 10.0s are gone

    rel = pd.read_csv(out / "relative_rmse.csv")
    assert set(rel["n"]) == {6}

    gains = pd.read_csv(out / "scope_gains.csv")
    assert int(gains.iloc[0]["n_common"]) == 6
    assert gains.iloc[0]["sample_basis"] == "system_common_sample"
    assert gains.iloc[0]["horizon_gain"] == pytest.approx(-1.0)

    ranks = pd.read_csv(out / "average_ranks.csv")
    assert set(ranks["n_common"]) == {6}
    assert (ranks["sample_basis"] == "common_sample").all()
