"""Tests for the MF-BVAR scope-grid CLI runner (dry-run, planning, manifests)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Load the script module by path (scripts/ is not an importable package).
_SCRIPT_PATH = REPO_ROOT / "scripts" / "mfvar" / "run_mfvar_scope_grid.py"
_spec = importlib.util.spec_from_file_location("run_mfvar_scope_grid", _SCRIPT_PATH)
runner = importlib.util.module_from_spec(_spec)
sys.modules["run_mfvar_scope_grid"] = runner
_spec.loader.exec_module(runner)


def _args(tmp_path, **overrides):
    argv = [
        "--output-root",
        str(tmp_path / "study"),
        "--selection-scopes",
        overrides.pop("selection_scopes", "pooled"),
    ]
    for key, value in overrides.items():
        argv.append(f"--{key.replace('_', '-')}")
        if value is not None:
            argv.append(str(value))
    return argv


def test_dry_run_emits_manifest_without_data(tmp_path, capsys):
    argv = _args(tmp_path, selection_scopes="pooled,horizon", dry_run=None)
    rc = runner.main(argv)
    assert rc == 0
    out = capsys.readouterr().out
    manifest = json.loads(out)
    assert manifest["runner"] == "mfvar_scope_grid"
    assert manifest["selection_scopes"] == ["pooled", "horizon"]
    # No study directory is written during a dry run.
    assert not (tmp_path / "study").exists()
    # Reproducibility limitation is always surfaced.
    assert manifest["reproducibility_limitations"]


def test_pooled_scope_plan_single_cell(tmp_path):
    argv = _args(tmp_path, selection_scopes="pooled")
    args = runner.build_parser().parse_args(argv)
    config = runner.build_study_config(args, argv=argv)
    plans = runner.plan_scope_runs(config)
    assert len(plans) == 1
    assert plans[0].n_target_cells == 1


def test_variable_horizon_scope_cell_count(tmp_path):
    argv = _args(
        tmp_path,
        selection_scopes="variable_horizon",
        target_variables="GDP,INVFIX",
        target_horizons="1,2",
    )
    args = runner.build_parser().parse_args(argv)
    config = runner.build_study_config(args, argv=argv)
    plans = runner.plan_scope_runs(config)
    assert plans[0].n_target_cells == 4


def test_variable_scope_requires_explicit_targets(tmp_path):
    argv = _args(tmp_path, selection_scopes="variable")
    args = runner.build_parser().parse_args(argv)
    with pytest.raises(runner.ScopeGridConfigError, match="explicit"):
        runner.build_study_config(args, argv=argv)


def test_group_scope_with_residual(tmp_path):
    argv = _args(
        tmp_path,
        selection_scopes="group",
        target_variables="GDP,INVFIX,GOV",
        variable_groups="real=GDP,INVFIX",
        residual_group_name="other",
    )
    args = runner.build_parser().parse_args(argv)
    config = runner.build_study_config(args, argv=argv)
    plans = runner.plan_scope_runs(config)
    # One explicit group plus one residual group.
    assert plans[0].n_target_cells == 2


def test_invalid_target_variable_rejected(tmp_path):
    argv = _args(
        tmp_path,
        selection_scopes="variable",
        target_variables="GDP,NOTAVAR",
    )
    args = runner.build_parser().parse_args(argv)
    with pytest.raises(runner.ScopeGridConfigError, match="subset of the forecast block"):
        runner.build_study_config(args, argv=argv)


def test_horizon_above_max_rejected(tmp_path):
    argv = _args(tmp_path, selection_scopes="pooled", target_horizons="1,99")
    args = runner.build_parser().parse_args(argv)
    with pytest.raises(runner.ScopeGridConfigError, match="target horizons"):
        runner.build_study_config(args, argv=argv)


def test_selection_frequency_variants(tmp_path):
    for token, kind in [
        ("once", "once"),
        ("per_origin", "every_origin"),
        ("annual_quarterly", "every_n_origins"),
        ("4", "every_n_origins"),
    ]:
        schedule = runner.build_selection_schedule(token)
        assert schedule.kind == kind


def test_manifest_records_selection_plan(tmp_path):
    argv = _args(tmp_path, selection_scopes="horizon", target_horizons="1,2,4")
    args = runner.build_parser().parse_args(argv)
    config = runner.build_study_config(args, argv=argv)
    plans = runner.plan_scope_runs(config)
    manifest = runner.build_manifest(config, plans)
    scope_entry = manifest["scopes"][0]
    assert scope_entry["scope"] == "horizon"
    assert scope_entry["selection_plan"]["scope"] == "horizon"
    assert len(scope_entry["selection_plan"]["cells"]) == 3


def test_default_forecast_block_is_full_quarterly(tmp_path):
    argv = _args(tmp_path, selection_scopes="pooled")
    args = runner.build_parser().parse_args(argv)
    config = runner.build_study_config(args, argv=argv)
    assert config.forecast_variables == ("GDP", "INVFIX", "GOV")


def test_existing_scripts_still_import():
    # Compatibility: the legacy mixed-frequency scripts import cleanly.
    for name in ("run_mango_rmse.py", "run_paper_hyperparameters.py"):
        path = REPO_ROOT / "scripts" / name
        spec = importlib.util.spec_from_file_location(f"legacy_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert hasattr(module, "build_parser") or hasattr(module, "main")


def test_run_metadata_records_search_space_bounds_and_scaling(tmp_path):
    """Peer-review requirement: exact bounds + scaling must be in run metadata."""

    from paper_hyperparameter_optimization.config import DEFAULT_PARAM_SPACE_BOUNDS

    argv = _args(tmp_path, selection_scopes="pooled")
    args = runner.build_parser().parse_args(argv)
    config = runner.build_study_config(args, argv=argv)
    plan = runner.plan_scope_runs(config)[0]
    metadata = runner._plan_run_metadata(
        config,
        plan,
        started_utc="2020-01-01T00:00:00+00:00",
        finished_utc=None,
        completion_status="partial",
    )
    search_space = metadata["search_space"]
    assert search_space["bounds"] == {
        name: [float(lo), float(hi)]
        for name, (lo, hi) in DEFAULT_PARAM_SPACE_BOUNDS.items()
    }
    assert set(search_space["scaling"].values()) == {"uniform"}
    assert search_space["log_transform"] is False
