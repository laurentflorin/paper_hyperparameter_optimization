import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from glp_hyperparameter_optimization import config


def test_model_sizes_are_nested():
    small = set(config.MODEL_SIZE_CODES["small"])
    medium = set(config.MODEL_SIZE_CODES["medium"])
    large = set(config.MODEL_SIZE_CODES["large"])
    assert small < medium < large
    assert len(config.MODEL_SIZE_CODES["small"]) == 3
    assert len(config.MODEL_SIZE_CODES["medium"]) == 7


def test_model_series_preserves_order_and_membership():
    specs = config.model_series("medium")
    assert [spec.code for spec in specs] == list(config.MEDIUM_CODES)
    for spec in specs:
        assert config.SERIES_BY_CODE[spec.code] is spec


def test_every_code_has_a_spec():
    for code in config.LARGE_CODES:
        assert code in config.SERIES_BY_CODE


def test_apply_transform_matches_covbayesvar_convention():
    levels = pd.Series([1.0, np.e, np.e**2])
    logged = config.apply_transform(levels, "log")
    np.testing.assert_allclose(logged.to_numpy(), [0.0, 100.0, 200.0])
    linear = config.apply_transform(levels, "lin")
    np.testing.assert_allclose(linear.to_numpy(), levels.to_numpy())


def test_param_space_bounds_cover_glp_hyperparameters():
    assert set(config.GLP_PARAM_SPACE_BOUNDS) == {"lambda", "theta", "miu"}
    for lower, upper in config.GLP_PARAM_SPACE_BOUNDS.values():
        assert lower < upper


def test_forecast_origins_are_quarter_ends():
    origins = config.forecast_origin_dates(pd.Timestamp("2001-01-01"), pd.Timestamp("2001-12-31"))
    assert list(origins) == [
        pd.Timestamp("2001-03-31"),
        pd.Timestamp("2001-06-30"),
        pd.Timestamp("2001-09-30"),
        pd.Timestamp("2001-12-31"),
    ]
