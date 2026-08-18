"""Tests for the ridge VAR panel adapter and fold-local standardization."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from regularized_var.data import PanelData, Standardizer, load_panel_csv


def test_panel_validation_and_metadata():
    values = np.arange(12.0).reshape(6, 2)
    panel = PanelData(values=values, variable_names=["a", "b"])
    assert panel.n_observations == 6
    assert panel.n_variables == 2
    assert panel.column_index("b") == 1
    assert panel.label_for(3) == 3
    meta = panel.to_metadata()
    assert meta["variable_names"] == ["a", "b"]


def test_panel_rejects_non_finite():
    values = np.ones((4, 2))
    values[1, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        PanelData(values=values, variable_names=["a", "b"])


def test_panel_rejects_name_mismatch_and_duplicates():
    with pytest.raises(ValueError, match="length"):
        PanelData(values=np.ones((4, 2)), variable_names=["only"])
    with pytest.raises(ValueError, match="unique"):
        PanelData(values=np.ones((4, 2)), variable_names=["x", "x"])


def test_standardizer_is_fold_local():
    rng = np.random.default_rng(0)
    train = rng.normal(loc=5.0, scale=2.0, size=(50, 2))
    std = Standardizer.fit(train, enabled=True)
    z = std.transform(train)
    # Fitted on training only: standardized training has ~zero mean, ~unit std.
    np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(z.std(axis=0, ddof=1), 1.0, atol=1e-8)


def test_standardizer_transform_uses_training_statistics_only():
    train = np.array([[0.0], [2.0], [4.0]])  # mean 2, std 2 (ddof=1)
    std = Standardizer.fit(train, enabled=True)
    # A brand-new validation input is transformed with training stats, so its
    # standardized value is deterministic and independent of other validation
    # rows (no leakage).
    val = np.array([[6.0]])
    np.testing.assert_allclose(std.transform(val), np.array([[2.0]]))


def test_standardizer_inverse_round_trip():
    rng = np.random.default_rng(1)
    train = rng.normal(size=(30, 3))
    std = Standardizer.fit(train, enabled=True)
    forecasts_z = rng.normal(size=(5, 3))
    back = std.inverse_transform(forecasts_z)
    np.testing.assert_allclose(std.transform(back), forecasts_z, atol=1e-10)


def test_standardizer_disabled_is_identity():
    train = np.array([[1.0, 2.0], [3.0, 4.0]])
    std = Standardizer.fit(train, enabled=False)
    assert std.enabled is False
    np.testing.assert_array_equal(std.transform(train), train)
    np.testing.assert_array_equal(std.inverse_transform(train), train)


def test_standardizer_floors_constant_column_scale():
    train = np.array([[3.0, 1.0], [3.0, 2.0], [3.0, 3.0]])  # first column constant
    std = Standardizer.fit(train, enabled=True, min_scale=1e-6)
    assert std.scale[0] == pytest.approx(1e-6)
    assert np.all(np.isfinite(std.transform(train)))


def test_load_panel_csv(tmp_path):
    csv_path = tmp_path / "panel.csv"
    csv_path.write_text("date,a,b\n2000Q1,1.0,2.0\n2000Q2,3.0,4.0\n", encoding="utf-8")
    panel = load_panel_csv(csv_path, variables=["a", "b"], date_column="date")
    assert panel.variable_names == ("a", "b")
    assert panel.date_labels == ("2000Q1", "2000Q2")
    np.testing.assert_array_equal(panel.values, np.array([[1.0, 2.0], [3.0, 4.0]]))
