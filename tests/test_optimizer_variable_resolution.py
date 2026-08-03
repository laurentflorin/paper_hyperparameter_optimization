import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from paper_hyperparameter_optimization.forecasting import (
    DEFAULT_MDD_OPTIMIZATION_VARIABLES,
    RMSE_REQUIRED_OPTIMIZATION_VARIABLES,
    resolve_optimization_variables,
)


def test_mango_mdd_defaults_to_gdp():
    assert resolve_optimization_variables("mango_mdd", []) == DEFAULT_MDD_OPTIMIZATION_VARIABLES


def test_mango_rmse_defaults_to_full_quarterly_block():
    assert resolve_optimization_variables("mango_rmse", []) == RMSE_REQUIRED_OPTIMIZATION_VARIABLES


def test_mango_rmse_rejects_quarterly_subset():
    try:
        resolve_optimization_variables("mango_rmse", ["GDP"])
    except ValueError as exc:
        assert "full quarterly variable block" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected RMSE optimizer variable validation to fail for GDP-only input.")