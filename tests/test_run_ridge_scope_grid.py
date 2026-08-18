"""Tests for the ridge VAR scope-grid runner (dry-run manifest & CLI)."""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts" / "regularized_var"
for path in (str(SRC_ROOT), str(SCRIPTS_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import run_ridge_scope_grid as runner


def _base_argv(output_root):
    return [
        "--output-root", str(output_root),
        "--target-variables", "gdp,inv,cons",
        "--target-horizons", "1,2",
        "--selection-scopes", "pooled,horizon,variable,variable_horizon",
        "--grid-lambdas", "0.0,1.0",
        "--grid-lag-orders", "1,2",
        "--grid-alphas", "0.0,1.0",
        "--grid-kappas", "1.0",
        "--outer-n-origins", "4",
        "--inner-n-origins", "3",
        "--min-train-length", "30",
        "--dry-run",
    ]


def test_dry_run_manifest_reports_grid_and_fit_counts(tmp_path, capsys):
    rc = runner.main(_base_argv(tmp_path))
    assert rc == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["runner"] == "regularized_var_scope_grid"
    assert manifest["grid_size"] == 8
    assert manifest["dry_run"] is True
    assert [s["scope"] for s in manifest["scopes"]] == [
        "pooled", "horizon", "variable", "variable_horizon"
    ]
    # Pooled scope fit-count estimation is hand-checkable.
    pooled = next(s for s in manifest["scopes"] if s["scope"] == "pooled")
    assert pooled["fit_counts"]["selection_fits"] == 1 * 1 * 8 * 3
    assert pooled["fit_counts"]["outer_forecast_fits"] == 4 * 1
    assert manifest["estimated_total_fits"] > 0
    # No output directories are created in a dry run.
    assert not list(tmp_path.glob("scope_*"))


def test_dry_run_records_selection_schedule_and_preprocessing(tmp_path, capsys):
    argv = _base_argv(tmp_path) + ["--selection-frequency", "2", "--preprocessing", "none"]
    assert runner.main(argv) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["selection_schedule"]["kind"] == "every_n_origins"
    assert manifest["selection_schedule"]["n"] == 2
    assert manifest["preprocessing"] == "none"


def test_unknown_scope_is_rejected(tmp_path, capsys):
    argv = [
        "--output-root", str(tmp_path),
        "--target-variables", "gdp",
        "--selection-scopes", "bogus",
        "--dry-run",
    ]
    rc = runner.main(argv)
    assert rc == 2
    assert "unsupported selection scopes" in capsys.readouterr().err


def test_end_to_end_real_run_writes_canonical_outputs(tmp_path):
    # Build a small synthetic panel CSV and run without --dry-run.
    import numpy as np

    rng = np.random.default_rng(0)
    A1 = np.array([[0.5, 0.1, 0.0], [-0.2, 0.3, 0.1], [0.0, 0.1, 0.4]])
    c = np.array([0.4, -0.1, 0.2])
    T = 160
    y = np.zeros((T, 3))
    for t in range(1, T):
        y[t] = c + A1 @ y[t - 1] + rng.normal(scale=0.1, size=3)

    panel_path = tmp_path / "panel.csv"
    header = "gdp,inv,cons\n"
    lines = [",".join(f"{v:.10f}" for v in row) for row in y]
    panel_path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")

    output_root = tmp_path / "out"
    argv = [
        "--output-root", str(output_root),
        "--panel-path", str(panel_path),
        "--target-variables", "gdp,inv,cons",
        "--target-horizons", "1,2",
        "--selection-scopes", "pooled,variable",
        "--grid-lambdas", "0.0,1.0",
        "--grid-lag-orders", "1,2",
        "--grid-alphas", "0.0",
        "--grid-kappas", "1.0",
        "--outer-n-origins", "4",
        "--inner-n-origins", "3",
        "--min-train-length", "30",
        "--benchmarks", "no_change,var_aic",
    ]
    rc = runner.main(argv)
    assert rc == 0
    for scope in ("pooled", "variable"):
        scope_dir = output_root / f"scope_{scope}"
        assert (scope_dir / "forecast_panel.csv").exists()
        assert (scope_dir / "selected_hyperparameters.csv").exists()
        assert (scope_dir / "run_metadata.json").exists()
    for benchmark in ("no_change", "var_aic"):
        assert (output_root / "benchmarks" / benchmark / "forecast_panel.csv").exists()
    assert (output_root / "scope_grid_manifest.json").exists()
