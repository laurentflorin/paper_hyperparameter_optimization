import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from glp_hyperparameter_optimization import forecasting as F
from glp_hyperparameter_optimization import reporting as R


def test_parse_helpers():
    assert F.parse_csv_list("GDP, DEFL ,", ["X"]) == ["GDP", "DEFL"]
    assert F.parse_csv_list("", ["X"]) == ["X"]
    assert F.parse_csv_int_list("1,2,4,8", [1]) == [1, 2, 4, 8]
    assert F.parse_positive_int("cpu-4") == 4
    assert F.parse_positive_int(None) is None


def test_resolve_parallel_settings_defaults_to_single(monkeypatch):
    for name in ("SLURM_NTASKS", "SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE", "SLURM_CPUS_PER_TASK"):
        monkeypatch.delenv(name, raising=False)
    workers, njobs = F.resolve_parallel_settings(10, None, None)
    assert workers == 1 and njobs == 1


def test_resolve_parallel_settings_splits_slurm_allocation(monkeypatch):
    monkeypatch.setenv("SLURM_NTASKS", "8")
    workers, njobs = F.resolve_parallel_settings(4, None, None)
    assert workers == 4
    assert njobs == 2


def test_one_time_strategies_declared():
    assert F.ONE_TIME_OPTIMIZATION_STRATEGIES == {"mango_rmse", "mango_rmse_random"}
    assert set(F.STRATEGIES) == {"paper", "mango_mdd", "mango_rmse", "mango_rmse_random"}


def _forecast_frame() -> pd.DataFrame:
    rows = []
    for model, bias in [("paper", 0.0), ("mango_mdd", 1.0)]:
        for horizon in (1, 2):
            rows.append(
                {
                    "model": model,
                    "model_size": "small",
                    "forecast_origin": "2020-03-31",
                    "target_quarter": f"2020Q{horizon}",
                    "variable": "GDP",
                    "horizon_quarters": horizon,
                    "actual": 10.0,
                    "error": bias + horizon,
                }
            )
    return pd.DataFrame(rows)


def test_rmse_and_relative_tables():
    rmse = R.compute_rmse_table(_forecast_frame())
    paper_h1 = rmse[(rmse.model == "paper") & (rmse.horizon_quarters == 1)]["rmse"].iloc[0]
    assert paper_h1 == 1.0  # sqrt(mean([1^2]))
    relative = R.compute_relative_rmse(rmse)
    paper_rows = relative[relative.model == "paper"]
    assert np.allclose(paper_rows["relative_rmse_pct"], 0.0)
    mdd_h1 = relative[(relative.model == "mango_mdd") & (relative.horizon_quarters == 1)]["relative_rmse_pct"].iloc[0]
    assert mdd_h1 == 100.0  # rmse 2.0 vs baseline 1.0


def test_relative_rmse_works_when_only_rmse_models_have_optimization_horizon():
    frame = pd.DataFrame(
        [
            {
                "model": "paper",
                "model_size": "small",
                "forecast_origin": "2020-03-31",
                "target_quarter": "2020Q1",
                "variable": "GDP",
                "horizon_quarters": 1,
                "actual": 10.0,
                "error": 1.0,
            },
            {
                "model": "mango_rmse",
                "model_size": "small",
                "forecast_origin": "2020-03-31",
                "target_quarter": "2020Q1",
                "variable": "GDP",
                "horizon_quarters": 1,
                "actual": 10.0,
                "optimization_horizon": "h1q",
                "error": 2.0,
            },
            {
                "model": "mango_rmse",
                "model_size": "small",
                "forecast_origin": "2020-03-31",
                "target_quarter": "2020Q1",
                "variable": "GDP",
                "horizon_quarters": 1,
                "actual": 10.0,
                "optimization_horizon": "h2q",
                "error": 3.0,
            },
        ]
    )
    rmse = R.compute_rmse_table(frame)
    assert set(rmse["model"]) == {"paper", "mango_rmse"}

    relative = R.compute_relative_rmse(rmse)
    paper_row = relative[(relative.model == "paper") & (relative.horizon_quarters == 1)].iloc[0]
    assert paper_row["baseline_rmse"] == 1.0
    assert paper_row["relative_rmse_pct"] == 0.0

    rmse_rows = relative[(relative.model == "mango_rmse") & (relative.horizon_quarters == 1)].sort_values(
        "optimization_horizon"
    )
    assert list(rmse_rows["optimization_horizon"]) == ["h1q", "h2q"]
    assert np.allclose(rmse_rows["baseline_rmse"], 1.0)
    assert np.allclose(rmse_rows["relative_rmse_pct"], [100.0, 200.0])


def test_hyperparameter_summary_keeps_optimization_horizons_separate():
    hyper = pd.DataFrame(
        [
            {
                "model": "paper",
                "model_size": "small",
                "forecast_origin": "2020-03-31",
                "lambda": 0.5,
                "theta": 1.0,
                "miu": 2.0,
            },
            {
                "model": "mango_rmse",
                "model_size": "small",
                "optimization_horizon": "h1q",
                "forecast_origin": "2020-03-31",
                "lambda": 1.0,
                "theta": 2.0,
                "miu": 3.0,
            },
            {
                "model": "mango_rmse",
                "model_size": "small",
                "optimization_horizon": "h2q",
                "forecast_origin": "2020-03-31",
                "lambda": 4.0,
                "theta": 5.0,
                "miu": 6.0,
            },
        ]
    )
    summary = R.compute_hyperparameter_summary(hyper)
    assert len(summary) == 3
    paper_row = summary[summary.model == "paper"].iloc[0]
    assert pd.isna(paper_row["optimization_horizon"])
    rmse_rows = summary[summary.model == "mango_rmse"]
    assert list(rmse_rows["optimization_horizon"]) == ["h1q", "h2q"]


def test_plot_hyperparameter_paths_separates_optimization_horizons(tmp_path):
    hyper = pd.DataFrame(
        [
            {
                "model": "paper",
                "model_size": "small",
                "forecast_origin": "2020-03-31",
                "lambda": 0.5,
                "theta": 1.0,
                "miu": 2.0,
            },
            {
                "model": "mango_rmse",
                "model_size": "small",
                "optimization_horizon": "h1q",
                "forecast_origin": "2020-03-31",
                "lambda": 1.0,
                "theta": 2.0,
                "miu": 3.0,
            },
            {
                "model": "mango_rmse",
                "model_size": "small",
                "optimization_horizon": "h2q",
                "forecast_origin": "2020-03-31",
                "lambda": 4.0,
                "theta": 5.0,
                "miu": 6.0,
            },
        ]
    )
    stale = tmp_path / "mango_rmse_hyperparameter_paths.png"
    stale.write_text("stale", encoding="utf-8")
    R.plot_hyperparameter_paths(hyper, tmp_path)
    assert (tmp_path / "paper_hyperparameter_paths.png").exists()
    assert (tmp_path / "mango_rmse_h1q_hyperparameter_paths.png").exists()
    assert (tmp_path / "mango_rmse_h2q_hyperparameter_paths.png").exists()
    assert not stale.exists()


def test_ordered_models_puts_paper_first():
    assert R.ordered_models(["mango_rmse", "paper", "zzz"]) == ["paper", "mango_rmse", "zzz"]
