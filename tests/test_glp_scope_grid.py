import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts" / "glp"
SRC_ROOT = REPO_ROOT / "src"
for root in (SRC_ROOT, SCRIPT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import run_glp_scope_grid as scope_grid
from glp_hyperparameter_optimization.selection_experiment import GLPExperimentResult


def _args(tmp_path: Path, *extra: str):
    parser = scope_grid.build_parser()
    return parser.parse_args(
        [
            "--output-root",
            str(tmp_path / "scope-study"),
            "--model-size",
            "small",
            "--selection-scopes",
            "pooled",
            "--target-horizons",
            "1,4",
            *extra,
        ]
    )


def test_build_study_config_parses_recommended_scope_grid_cli(tmp_path: Path):
    args = _args(
        tmp_path,
        "--selection-scopes",
        "pooled,horizon,variable,variable_horizon",
        "--target-variables",
        "GDP,DEFL,FFR",
        "--loss-metric",
        "rmse",
        "--loss-scaling",
        "benchmark_rmse",
        "--benchmark",
        "last_observation",
        "--selection-frequency",
        "4",
        "--no-optimize-psi",
    )

    config = scope_grid.build_study_config(args, argv=())

    assert config.selection_scopes == ("pooled", "horizon", "variable", "variable_horizon")
    assert config.target_variables == ("GDP", "DEFL", "FFR")
    assert config.target_horizons == (1, 4)
    assert config.schedule.kind == "every_n_origins"
    assert config.schedule.n == 4
    assert config.validation_scheme.origin_selection == "most_recent"
    assert config.loss_config.scale.method == "benchmark_rmse"
    assert config.search_config.optimize_psi is False


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (
            [
                "--selection-scopes",
                "variable_horizon",
            ],
            "require explicit --target-variables",
        ),
        (
            [
                "--target-variables",
                "GDP",
                "--loss-scaling",
                "benchmark_rmse",
                "--benchmark",
                "none",
            ],
            "requires an explicit benchmark selection",
        ),
        (
            [
                "--selection-scopes",
                "group",
                "--target-variables",
                "GDP,DEFL",
            ],
            "group scope requires",
        ),
        (
            [
                "--target-variables",
                "GDP",
                "--inner-origin-selection",
                "random",
            ],
            "requires --inner-random-seed",
        ),
    ],
)
def test_build_study_config_rejects_invalid_combinations(
    tmp_path: Path,
    extra: list[str],
    message: str,
):
    args = _args(tmp_path, *extra)
    with pytest.raises(ValueError, match=message):
        scope_grid.build_study_config(args, argv=())


def test_scope_expansion_includes_group_plan(tmp_path: Path):
    args = _args(
        tmp_path,
        "--selection-scopes",
        "group",
        "--target-variables",
        "GDP,DEFL,FFR",
        "--variable-groups",
        "Real=GDP+DEFL",
        "--residual-group-name",
        "Rates",
        "--group-separate-horizons",
    )

    config = scope_grid.build_study_config(args, argv=())
    plan = config.scope_plans[0].selection_plan

    assert plan.scope == "group"
    assert [cell.cell_id for cell in plan.cells] == [
        "group-real-h1",
        "group-real-h4",
        "group-rates-h1",
        "group-rates-h4",
    ]


def test_deterministic_run_directory_naming(tmp_path: Path):
    args = _args(
        tmp_path,
        "--selection-scopes",
        "pooled,variable_horizon",
        "--target-variables",
        "GDP,DEFL",
    )

    first = scope_grid.build_study_config(args, argv=())
    second = scope_grid.build_study_config(args, argv=())

    assert [plan.output_dir.name for plan in first.scope_plans] == [
        "scope-pooled",
        "scope-variable-horizon",
    ]
    assert [plan.output_dir.name for plan in first.scope_plans] == [
        plan.output_dir.name for plan in second.scope_plans
    ]


def test_budget_calculations_are_reported_per_scope(tmp_path: Path):
    parser = scope_grid.build_parser()
    args = parser.parse_args(
        [
            "--output-root",
            str(tmp_path / "scope-study"),
            "--model-size",
            "small",
            "--selection-scopes",
            "pooled,variable",
            "--target-variables",
            "GDP,DEFL",
            "--target-horizons",
            "1,4",
            "--start",
            "2000-03-31",
            "--end",
            "2000-12-31",
            "--selection-frequency",
            "2",
            "--optimization-init-points",
            "3",
            "--optimization-iterations",
            "5",
            "--inner-n-origins",
            "2",
        ]
    )

    config = scope_grid.build_study_config(args, argv=())
    pooled, variable = config.scope_plans

    assert pooled.estimated_optimization_cells == 2
    assert pooled.estimated_candidate_evaluations == 16
    assert pooled.estimated_validation_split_evaluations == 32
    assert variable.estimated_optimization_cells == 4
    assert variable.estimated_candidate_evaluations == 32
    assert variable.estimated_validation_split_evaluations == 64


def test_dry_run_writes_manifest_without_loading_data(tmp_path: Path, monkeypatch):
    args = _args(
        tmp_path,
        "--selection-scopes",
        "pooled,horizon",
        "--dry-run",
    )
    config = scope_grid.build_study_config(args, argv=("--dry-run",))

    def fail_loader(_path):
        raise AssertionError("dry-run must not load the realtime panel")

    monkeypatch.setattr(scope_grid, "load_scope_grid_panel", fail_loader)

    summary = scope_grid.execute_study(config)
    manifest_path = Path(summary["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert summary["executed_scopes"] == []
    assert manifest["execution"]["dry_run"] is True
    assert len(manifest["planned_runs"]) == 2
    assert (config.output_root / "batch_metadata.json").exists()
    assert (config.scope_plans[0].output_dir / "experiment_manifest.json").exists()
    assert not (config.scope_plans[0].output_dir / "forecast_panel.csv").exists()


def test_execute_study_delegates_to_scope_experiment_runner(tmp_path: Path, monkeypatch):
    args = _args(
        tmp_path,
        "--target-variables",
        "GDP",
        "--loss-scaling",
        "none",
        "--benchmark",
        "none",
        "--selection-scopes",
        "pooled",
    )
    config = scope_grid.build_study_config(args, argv=())
    panel_sentinel = object()
    calls: list[tuple[str, object]] = []

    def fake_loader(path):
        calls.append(("load", path))
        return panel_sentinel

    def fake_runner(plan, configured, panel):
        calls.append((plan.scope, panel))
        assert configured is config
        assert panel is panel_sentinel
        return GLPExperimentResult(
            forecast_panel=[
                {
                    "model": "glp_scope_grid",
                    "origin_index": 0,
                    "forecast_origin": "2000-03-31",
                    "variable": "GDP",
                    "horizon": 1,
                    "forecast": 1.23,
                }
            ],
            selected_hyperparameters=[
                {
                    "model": "glp_scope_grid",
                    "cell_id": "pooled",
                    "selection_event_id": "sel-000-o0000",
                    "selection_loss": 0.5,
                }
            ],
            forecast_panel_all_cells=[],
            run_metadata={"selection_plan": {"scope": plan.scope}},
            cache_stats={"hits": 0, "misses": 1},
        )

    monkeypatch.setattr(scope_grid, "load_scope_grid_panel", fake_loader)
    monkeypatch.setattr(scope_grid, "run_scope_study", fake_runner)

    summary = scope_grid.execute_study(config)
    run_dir = config.scope_plans[0].output_dir

    assert summary["executed_scopes"] == ["pooled"]
    assert calls[0][0] == "load"
    assert calls[1] == ("pooled", panel_sentinel)
    assert (run_dir / "forecast_panel.csv").exists()
    assert (run_dir / "selected_hyperparameters.csv").exists()
    assert (run_dir / "run_metadata.json").exists()