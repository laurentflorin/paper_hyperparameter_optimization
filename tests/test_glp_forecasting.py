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
                    "variable": "GDP",
                    "horizon_quarters": horizon,
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


def test_ordered_models_puts_paper_first():
    assert R.ordered_models(["mango_rmse", "paper", "zzz"]) == ["paper", "mango_rmse", "zzz"]
