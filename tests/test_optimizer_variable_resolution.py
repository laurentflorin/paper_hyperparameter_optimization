import inspect
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from paper_hyperparameter_optimization import forecasting
from paper_hyperparameter_optimization.config import DEFAULT_OPTIMIZATION_NSIM
from paper_hyperparameter_optimization.forecasting import (
    DEFAULT_MDD_OPTIMIZATION_VARIABLES,
    RMSE_REQUIRED_OPTIMIZATION_VARIABLES,
    resolve_optimization_variables,
)


def test_mango_mdd_defaults_to_gdp():
    assert resolve_optimization_variables("mango_mdd", []) == DEFAULT_MDD_OPTIMIZATION_VARIABLES


def test_optimization_nsim_default_is_single_sourced(tmp_path: Path):
    parser = forecasting.build_optimizer_parser("test")
    # --output-dir is deliberately required: a run must never write to an implicit path.
    args = parser.parse_args(["--output-dir", str(tmp_path)])
    parameter = inspect.signature(forecasting.run_recursive_experiment).parameters["optimization_nsim"]

    assert args.optimization_nsim == DEFAULT_OPTIMIZATION_NSIM
    assert parameter.default == DEFAULT_OPTIMIZATION_NSIM


def test_optimizer_parser_requires_output_dir():
    """The optimizer CLI must fail closed when no output directory is given."""
    parser = forecasting.build_optimizer_parser("test")
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_mango_rmse_defaults_to_full_quarterly_block():
    assert resolve_optimization_variables("mango_rmse", []) == RMSE_REQUIRED_OPTIMIZATION_VARIABLES


def test_mango_rmse_legacy_subset_maps_to_objective_only():
    # The legacy single argument now maps to the objective (loss) subset while the
    # forecast state still spans the full quarterly block.
    forecast_variables, objective_variables = forecasting.resolve_forecast_objective_variables(
        "mango_rmse", optimization_variables=["GDP"]
    )
    assert forecast_variables == RMSE_REQUIRED_OPTIMIZATION_VARIABLES
    assert objective_variables == ["GDP"]
    assert resolve_optimization_variables("mango_rmse", ["GDP"]) == ["GDP"]


def _run_recursive_experiment_with_stubbed_optimizer(
    monkeypatch,
    tmp_path: Path,
    strategy: str,
    optimization_variables: list[str],
    selection_schedule: str | None = None,
):
    origins = pd.DatetimeIndex(
        [
            pd.Timestamp("2000-01-31"),
            pd.Timestamp("2000-02-29"),
            pd.Timestamp("2000-03-31"),
        ]
    )
    optimization_calls: list[str] = []

    class FakeModel:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            pass

        def forecast(self, *args, **kwargs):
            pass

        def aggregate(self, *args, **kwargs):
            pass

    fake_mbfvar = types.ModuleType("MBFVAR")
    fake_mbfvar.MixedFrequencyBVAR = FakeModel
    monkeypatch.setitem(sys.modules, "MBFVAR", fake_mbfvar)

    monkeypatch.setattr(forecasting, "forecast_origin_dates", lambda *args: origins)
    monkeypatch.setattr(forecasting, "resolve_parallel_settings", lambda *args, **kwargs: (1, 1))
    monkeypatch.setattr(forecasting, "load_realtime_panel", lambda path: object())
    monkeypatch.setattr(forecasting, "build_quarterly_evaluation_frame", lambda panel, vintage: pd.DataFrame())
    monkeypatch.setattr(forecasting, "build_model_input_frames", lambda panel, origin: (pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(forecasting, "make_data_in", lambda quarterly, monthly: object())
    monkeypatch.setattr(forecasting, "extract_forecasts", lambda *args, **kwargs: pd.DataFrame())

    def fake_select_hyperparameters(strategy_name, model, data_in, args):
        optimization_calls.append(args.get("origin_date", "initial"))
        return [[1.0, 2.0, 1.0, 4.0, 5.0]]

    monkeypatch.setattr(forecasting, "select_hyperparameters", fake_select_hyperparameters)

    output_dir = forecasting.run_recursive_experiment(
        strategy=strategy,
        output_dir=tmp_path / strategy,
        panel_path=tmp_path / "panel.csv",
        start=origins[0],
        end=origins[-1],
        optimization_variables=optimization_variables,
        n_workers=1,
        selection_schedule=selection_schedule,
    )
    return optimization_calls, output_dir


def test_mango_rmse_optimizes_once_for_full_recursive_run(monkeypatch, tmp_path: Path):
    optimization_calls, output_dir = _run_recursive_experiment_with_stubbed_optimizer(
        monkeypatch,
        tmp_path,
        "mango_rmse",
        RMSE_REQUIRED_OPTIMIZATION_VARIABLES,
    )

    assert optimization_calls == ["initial"]

    selected_hyperparameters = pd.read_csv(output_dir / "selected_hyperparameters.csv")
    assert len(selected_hyperparameters) == 3
    assert (
        selected_hyperparameters[["lambda1_1", "lambda2_1", "lambda3_1", "lambda4_1", "lambda5_1"]]
        .drop_duplicates()
        .shape[0]
        == 1
    )

    metadata = pd.read_json(output_dir / "run_metadata.json", typ="series")
    assert bool(metadata["hyperparameters_selected_once"])
    assert metadata["hyperparameter_selection_origin"] == "2000-01-31"


def test_mango_mdd_still_optimizes_per_origin(monkeypatch, tmp_path: Path):
    optimization_calls, _ = _run_recursive_experiment_with_stubbed_optimizer(
        monkeypatch,
        tmp_path,
        "mango_mdd",
        DEFAULT_MDD_OPTIMIZATION_VARIABLES,
    )

    assert optimization_calls == ["2000-01-31", "2000-02-29", "2000-03-31"]

def test_selection_schedule_is_threaded_into_the_task_template(monkeypatch, tmp_path: Path):
    """An explicit per_origin schedule must override the strategy baseline.

    The schedule is consumed from the task template, so a schedule that never
    reaches the template silently reverts every run to the strategy default.
    """
    optimization_calls, output_dir = _run_recursive_experiment_with_stubbed_optimizer(
        monkeypatch,
        tmp_path,
        "mango_rmse",
        RMSE_REQUIRED_OPTIMIZATION_VARIABLES,
        selection_schedule="per_origin",
    )

    assert optimization_calls == ["2000-01-31", "2000-02-29", "2000-03-31"]

    metadata = pd.read_json(output_dir / "run_metadata.json", typ="series")
    assert metadata["selection_schedule"] == "per_origin"
    assert not bool(metadata["hyperparameters_selected_once"])
    assert metadata["hyperparameter_selection_origin"] is None


def test_default_selection_schedule_matches_the_baseline_exercise():
    assert forecasting.default_selection_schedule("mango_mdd") == "per_origin"
    assert forecasting.default_selection_schedule("mango_rmse") == "first_origin"
    assert forecasting.default_selection_schedule("mango_rmse_random") == "first_origin"


def test_resolve_selection_schedule_rejects_unknown_values():
    with pytest.raises(ValueError, match="selection_schedule must be one of"):
        forecasting.resolve_selection_schedule("mango_rmse", "every_other_tuesday")


def test_mango_mdd_records_its_per_origin_schedule(monkeypatch, tmp_path: Path):
    """MDD metadata must not claim a single selection when it reselects per origin."""
    _, output_dir = _run_recursive_experiment_with_stubbed_optimizer(
        monkeypatch,
        tmp_path,
        "mango_mdd",
        DEFAULT_MDD_OPTIMIZATION_VARIABLES,
    )

    metadata = pd.read_json(output_dir / "run_metadata.json", typ="series")
    assert metadata["selection_schedule"] == "per_origin"
    assert not bool(metadata["hyperparameters_selected_once"])
