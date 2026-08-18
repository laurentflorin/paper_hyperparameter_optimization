"""Tests for atomic output writers and completion markers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common_hpo.io import (  # noqa: E402
    CSVSchema,
    JSONSchema,
    OutputValidationError,
    RUN_COMPLETE_FILENAME,
    atomic_write_dataframe_csv,
    atomic_write_json,
    classify_run_directory,
    mark_run_complete,
    prepare_run_directory,
)
from common_hpo.metadata import build_run_metadata  # noqa: E402


def _fixed_manifest(tmp_path: Path) -> dict[str, object]:
    return build_run_metadata(
        project_root=tmp_path,
        command_line="test-command",
        started_utc="2026-01-01T00:00:00Z",
        finished_utc=None,
        completion_status="partial",
        model_family="test",
        model_version="unit",
        data_source={"kind": "synthetic"},
        target_variables=("GDP",),
        target_horizons=(1,),
        selection_plan={"scope": "pooled"},
        validation_scheme={"kind": "synthetic"},
        selection_schedule={"kind": "once"},
        loss_configuration={"metric": "rmse"},
        search_space={"kind": "grid"},
        optimizer_budget={"budget": 1},
        random_seeds={"seed": 0},
        relevant_packages=(),
    )


def test_atomic_write_json_replaces_existing_file(tmp_path: Path):
    path = tmp_path / "payload.json"
    atomic_write_json(path, {"old": 1})
    atomic_write_json(path, {"new": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"new": 2}
    assert not list(tmp_path.glob("*.tmp-*"))


def test_atomic_write_dataframe_csv_replaces_existing_file(tmp_path: Path):
    path = tmp_path / "table.csv"
    atomic_write_dataframe_csv(path, pd.DataFrame({"x": [1], "y": [2]}), index=False)
    atomic_write_dataframe_csv(path, pd.DataFrame({"x": [3], "y": [4]}), index=False)
    pd.testing.assert_frame_equal(pd.read_csv(path), pd.DataFrame({"x": [3], "y": [4]}))
    assert not list(tmp_path.glob("*.tmp-*"))


def test_partial_run_without_completion_marker_is_not_complete(tmp_path: Path):
    manifest = _fixed_manifest(tmp_path)
    run_dir = tmp_path / "run"
    prepare_run_directory(
        run_dir,
        manifest=manifest,
        if_exists_policy="overwrite",
        expected_outputs=("forecast_panel.csv", "run_metadata.json"),
    )
    atomic_write_dataframe_csv(
        run_dir / "forecast_panel.csv",
        pd.DataFrame({"forecast_origin": ["o1"], "variable": ["GDP"], "horizon": [1], "forecast": [1.0]}),
        index=False,
    )
    atomic_write_json(run_dir / "run_metadata.json", manifest)
    state = classify_run_directory(run_dir)
    assert state.status == "partial"
    assert not (run_dir / RUN_COMPLETE_FILENAME).exists()


def test_schema_validation_blocks_completion_marker(tmp_path: Path):
    manifest = _fixed_manifest(tmp_path)
    run_dir = tmp_path / "run"
    prepare_run_directory(
        run_dir,
        manifest=manifest,
        if_exists_policy="overwrite",
        expected_outputs=("forecast_panel.csv", "selected_hyperparameters.csv", "run_metadata.json"),
    )
    atomic_write_dataframe_csv(
        run_dir / "forecast_panel.csv",
        pd.DataFrame({"forecast_origin": ["o1"], "variable": ["GDP"], "horizon": [1]}),
        index=False,
    )
    atomic_write_dataframe_csv(
        run_dir / "selected_hyperparameters.csv",
        pd.DataFrame({"cell_id": ["pooled"], "selection_event_id": ["e1"], "selection_loss": [0.1]}),
        index=False,
    )
    atomic_write_json(run_dir / "run_metadata.json", manifest)

    with pytest.raises(OutputValidationError, match="missing required columns"):
        mark_run_complete(
            run_dir,
            configuration_hash=str(manifest["configuration_hash"]),
            csv_schemas=(
                CSVSchema("forecast_panel.csv", ("forecast_origin", "variable", "horizon", "forecast"), min_rows=1),
                CSVSchema("selected_hyperparameters.csv", ("cell_id", "selection_event_id", "selection_loss"), min_rows=1),
            ),
            json_schemas=(JSONSchema("run_metadata.json", ("configuration_hash",)),),
        )

    assert not (run_dir / RUN_COMPLETE_FILENAME).exists()
    assert classify_run_directory(run_dir).status == "partial"


def test_successful_completion_writes_marker_and_complete_status(tmp_path: Path):
    manifest = _fixed_manifest(tmp_path)
    run_dir = tmp_path / "run"
    prepare_run_directory(
        run_dir,
        manifest=manifest,
        if_exists_policy="overwrite",
        expected_outputs=("forecast_panel.csv", "selected_hyperparameters.csv", "run_metadata.json"),
    )
    atomic_write_dataframe_csv(
        run_dir / "forecast_panel.csv",
        pd.DataFrame({"forecast_origin": ["o1"], "variable": ["GDP"], "horizon": [1], "forecast": [1.0]}),
        index=False,
    )
    atomic_write_dataframe_csv(
        run_dir / "selected_hyperparameters.csv",
        pd.DataFrame({"cell_id": ["pooled"], "selection_event_id": ["e1"], "selection_loss": [0.1]}),
        index=False,
    )
    complete_metadata = dict(manifest)
    complete_metadata["completion_status"] = "complete"
    complete_metadata["utc_finished_at"] = "2026-01-01T00:01:00Z"
    atomic_write_json(run_dir / "run_metadata.json", complete_metadata)

    mark_run_complete(
        run_dir,
        configuration_hash=str(manifest["configuration_hash"]),
        csv_schemas=(
            CSVSchema("forecast_panel.csv", ("forecast_origin", "variable", "horizon", "forecast"), min_rows=1),
            CSVSchema("selected_hyperparameters.csv", ("cell_id", "selection_event_id", "selection_loss"), min_rows=1),
        ),
        json_schemas=(JSONSchema("run_metadata.json", ("configuration_hash", "completion_status")),),
    )

    assert (run_dir / RUN_COMPLETE_FILENAME).exists()
    assert classify_run_directory(run_dir).status == "complete"