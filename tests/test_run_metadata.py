"""Tests for reproducibility metadata and hash-aware run resumption."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common_hpo.io import ResumeRejectedError, classify_run_directory, mark_run_cancelled, prepare_run_directory  # noqa: E402
from common_hpo.metadata import (  # noqa: E402
    build_run_metadata,
    fingerprint_input_file,
    stable_configuration_hash,
)
from glp_hyperparameter_optimization.selection_experiment import GLPExperimentResult  # noqa: E402

_GLP_SCOPE_GRID_PATH = REPO_ROOT / "scripts" / "glp" / "run_glp_scope_grid.py"
_GLP_SCOPE_GRID_SPEC = importlib.util.spec_from_file_location("run_glp_scope_grid_metadata", _GLP_SCOPE_GRID_PATH)
scope_grid = importlib.util.module_from_spec(_GLP_SCOPE_GRID_SPEC)
sys.modules["run_glp_scope_grid_metadata"] = scope_grid
_GLP_SCOPE_GRID_SPEC.loader.exec_module(scope_grid)


def _patch_stable_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    import common_hpo.metadata as metadata

    monkeypatch.setattr(
        metadata,
        "repository_state",
        lambda _root: {"repository_commit": "deadbeef", "repository_dirty": False},
    )
    monkeypatch.setattr(metadata, "package_versions", lambda _names=(): {})


def _manifest(tmp_path: Path, *, input_files=(), dirty: bool = False) -> dict[str, object]:
    return build_run_metadata(
        project_root=tmp_path,
        command_line="python run.py --scope pooled",
        started_utc="2026-01-01T00:00:00Z",
        finished_utc=None,
        completion_status="partial",
        model_family="test",
        model_version="v1",
        data_source={"kind": "synthetic"},
        data_vintage_identifiers={"policy": "outer_vintage_consistent"},
        input_files=input_files,
        transformation_configuration={"transform": "none"},
        variable_order=("GDP", "INVFIX"),
        target_variables=("GDP",),
        target_horizons=(1,),
        selection_plan={"scope": "pooled"},
        validation_scheme={"window": "expanding"},
        vintage_policy="outer_vintage_consistent",
        selection_schedule={"kind": "once"},
        loss_configuration={"metric": "rmse"},
        search_space={"grid": [1, 2]},
        optimizer_budget={"budget": 2},
        random_seeds={"seed": 0},
        relevant_packages=(),
        configuration_extra={"dirty_override": dirty},
    )


def test_configuration_hash_is_stable_across_key_order():
    a = {"b": [2, 1], "a": {"x": 1, "y": 2}}
    b = {"a": {"y": 2, "x": 1}, "b": [2, 1]}
    assert stable_configuration_hash(a) == stable_configuration_hash(b)


def test_input_file_hash_changes_update_configuration_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _patch_stable_environment(monkeypatch)
    data_path = tmp_path / "input.txt"
    data_path.write_text("first", encoding="utf-8")
    first = _manifest(tmp_path, input_files=(data_path,))
    first_fp = fingerprint_input_file(data_path, root=tmp_path)

    data_path.write_text("second", encoding="utf-8")
    second = _manifest(tmp_path, input_files=(data_path,))
    second_fp = fingerprint_input_file(data_path, root=tmp_path)

    assert first_fp["sha256"] != second_fp["sha256"]
    assert first["configuration_hash"] != second["configuration_hash"]


def test_dirty_tree_metadata_is_recorded_and_affects_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import common_hpo.metadata as metadata

    monkeypatch.setattr(metadata, "package_versions", lambda _names=(): {})
    monkeypatch.setattr(
        metadata,
        "repository_state",
        lambda _root: {"repository_commit": "deadbeef", "repository_dirty": False},
    )
    clean = _manifest(tmp_path)
    monkeypatch.setattr(
        metadata,
        "repository_state",
        lambda _root: {"repository_commit": "deadbeef", "repository_dirty": True},
    )
    dirty = _manifest(tmp_path)
    assert clean["repository_dirty"] is False
    assert dirty["repository_dirty"] is True
    assert clean["configuration_hash"] != dirty["configuration_hash"]


def test_compatible_resume_allows_incomplete_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _patch_stable_environment(monkeypatch)
    manifest = _manifest(tmp_path)
    run_dir = tmp_path / "run"

    first = prepare_run_directory(
        run_dir,
        manifest=manifest,
        if_exists_policy="overwrite",
        expected_outputs=("forecast_panel.csv", "run_metadata.json"),
    )
    second = prepare_run_directory(
        run_dir,
        manifest=manifest,
        if_exists_policy="resume",
        expected_outputs=("forecast_panel.csv", "run_metadata.json"),
    )

    assert first == "planned"
    assert second == "planned"
    assert classify_run_directory(run_dir).status == "partial"


def test_incompatible_resume_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _patch_stable_environment(monkeypatch)
    first_input = tmp_path / "input.txt"
    first_input.write_text("first", encoding="utf-8")
    first = _manifest(tmp_path, input_files=(first_input,))
    run_dir = tmp_path / "run"

    prepare_run_directory(
        run_dir,
        manifest=first,
        if_exists_policy="overwrite",
        expected_outputs=("forecast_panel.csv", "run_metadata.json"),
    )

    first_input.write_text("second", encoding="utf-8")
    second = _manifest(tmp_path, input_files=(first_input,))
    with pytest.raises(ResumeRejectedError, match="configuration hash"):
        prepare_run_directory(
            run_dir,
            manifest=second,
            if_exists_policy="resume",
            expected_outputs=("forecast_panel.csv", "run_metadata.json"),
        )


def test_interrupted_run_status_is_cancelled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _patch_stable_environment(monkeypatch)
    manifest = _manifest(tmp_path)
    run_dir = tmp_path / "run"
    prepare_run_directory(
        run_dir,
        manifest=manifest,
        if_exists_policy="overwrite",
        expected_outputs=("forecast_panel.csv", "run_metadata.json"),
    )
    cancelled = dict(manifest)
    cancelled["completion_status"] = "cancelled"
    cancelled["utc_finished_at"] = "2026-01-01T00:01:00Z"
    mark_run_cancelled(
        run_dir,
        configuration_hash=str(manifest["configuration_hash"]),
        metadata=cancelled,
    )
    assert classify_run_directory(run_dir).status == "cancelled"


def test_serial_and_parallel_mocked_glp_workflows_are_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    panel_sentinel = object()
    monkeypatch.setattr(scope_grid, "load_scope_grid_panel", lambda _path: panel_sentinel)

    def fake_runner(plan, config, panel):
        assert panel is panel_sentinel
        value = 1.0 if plan.scope == "pooled" else 2.0
        return GLPExperimentResult(
            forecast_panel=[
                {
                    "forecast_origin": "2000-03-31",
                    "variable": "GDP",
                    "horizon": 1,
                    "forecast": value,
                }
            ],
            selected_hyperparameters=[
                {
                    "cell_id": plan.scope,
                    "selection_event_id": f"{plan.scope}-sel-0",
                    "selection_loss": value / 10.0,
                }
            ],
            forecast_panel_all_cells=[],
            run_metadata={"selection_events": []},
            cache_stats={"hits": 0, "misses": 1},
        )

    monkeypatch.setattr(scope_grid, "run_scope_study", fake_runner)

    def build_config(output_root: Path, mode: str):
        args = scope_grid.build_parser().parse_args(
            [
                "--output-root", str(output_root),
                "--model-size", "small",
                "--selection-scopes", "pooled,horizon",
                "--target-variables", "GDP",
                "--target-horizons", "1",
                "--loss-scaling", "none",
                "--benchmark", "none",
                "--start", "2000-03-31",
                "--end", "2000-06-30",
                "--execution-mode", mode,
                "--worker-count", "2",
            ]
        )
        return scope_grid.build_study_config(args, argv=())

    serial_config = build_config(tmp_path / "serial", "serial")
    parallel_config = build_config(tmp_path / "parallel", "parallel")

    assert scope_grid.execute_study(serial_config)["executed_scopes"] == ["pooled", "horizon"]
    executed_parallel = scope_grid.execute_study(parallel_config)["executed_scopes"]
    assert set(executed_parallel) == {"pooled", "horizon"}

    for scope in ("pooled", "horizon"):
        pd.testing.assert_frame_equal(
            pd.read_csv(serial_config.output_root / f"scope-{scope}" / "forecast_panel.csv"),
            pd.read_csv(parallel_config.output_root / f"scope-{scope}" / "forecast_panel.csv"),
        )
        pd.testing.assert_frame_equal(
            pd.read_csv(serial_config.output_root / f"scope-{scope}" / "selected_hyperparameters.csv"),
            pd.read_csv(parallel_config.output_root / f"scope-{scope}" / "selected_hyperparameters.csv"),
        )