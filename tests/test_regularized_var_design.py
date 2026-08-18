"""Tests for the ridge VAR lag-design construction."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from regularized_var.design import (
    build_lag_design,
    n_predictors,
    predictor_terms,
    validate_observations,
)


def test_exact_design_matrix_two_variable_two_lag():
    # y rows are time-ordered; columns are variables.
    y = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ]
    )
    X, Y, terms = build_lag_design(y, p=2, include_intercept=True)

    # Effective rows for t = 2, 3.
    assert Y.tolist() == [[3.0, 30.0], [4.0, 40.0]]
    # Row t=2: [intercept, lag1=y[1], lag2=y[0]].
    assert X[0].tolist() == [1.0, 2.0, 20.0, 1.0, 10.0]
    # Row t=3: [intercept, lag1=y[2], lag2=y[1]].
    assert X[1].tolist() == [1.0, 3.0, 30.0, 2.0, 20.0]
    assert terms[0] == ("intercept",)
    assert terms[1] == ("lag", 1, 0)
    assert terms[2] == ("lag", 1, 1)
    assert terms[3] == ("lag", 2, 0)
    assert terms[4] == ("lag", 2, 1)


def test_lag_ordering_lag1_is_most_recent():
    y = np.arange(1, 11, dtype=float).reshape(-1, 1)  # single variable 1..10
    X, Y, terms = build_lag_design(y, p=3, include_intercept=False)
    # For t=3 (first effective row): lag1=y[2]=3, lag2=y[1]=2, lag3=y[0]=1.
    assert X[0].tolist() == [3.0, 2.0, 1.0]
    assert terms == [("lag", 1, 0), ("lag", 2, 0), ("lag", 3, 0)]


def test_intercept_toggle_changes_columns():
    y = np.random.default_rng(0).normal(size=(20, 3))
    X_with, _, terms_with = build_lag_design(y, p=2, include_intercept=True)
    X_without, _, terms_without = build_lag_design(y, p=2, include_intercept=False)
    assert X_with.shape[1] == n_predictors(3, 2, True) == 7
    assert X_without.shape[1] == n_predictors(3, 2, False) == 6
    assert np.all(X_with[:, 0] == 1.0)
    assert terms_with[0] == ("intercept",)
    assert ("intercept",) not in terms_without


def test_effective_sample_size():
    y = np.random.default_rng(1).normal(size=(15, 2))
    X, Y, _ = build_lag_design(y, p=4, include_intercept=True)
    assert X.shape[0] == Y.shape[0] == 15 - 4


def test_predictor_terms_ordering():
    terms = predictor_terms(2, 2, include_intercept=True)
    assert terms == [
        ("intercept",),
        ("lag", 1, 0),
        ("lag", 1, 1),
        ("lag", 2, 0),
        ("lag", 2, 1),
    ]


def test_non_finite_rejected():
    y = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, 5.0]])
    with pytest.raises(ValueError, match="non-finite"):
        validate_observations(y)
    with pytest.raises(ValueError, match="non-finite"):
        build_lag_design(y, p=1)


def test_infinite_rejected():
    y = np.array([[1.0], [np.inf], [3.0]])
    with pytest.raises(ValueError, match="non-finite"):
        build_lag_design(y, p=1)


def test_bad_shape_rejected():
    with pytest.raises(ValueError, match="two-dimensional"):
        validate_observations(np.arange(5.0))


def test_invalid_lag_order():
    y = np.random.default_rng(2).normal(size=(6, 2))
    with pytest.raises(ValueError, match=">= 1"):
        build_lag_design(y, p=0)
    with pytest.raises(TypeError):
        build_lag_design(y, p=1.5)
    with pytest.raises(ValueError, match="p < T"):
        build_lag_design(y, p=6)
