import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

pytest.importorskip("MBFVAR")
from MBFVAR.mfbvar_funcs import mdd_  # noqa: E402

from paper_hyperparameter_optimization import mf_ladder as L  # noqa: E402
from paper_hyperparameter_optimization.config import PAPER_HYPERPARAMETERS  # noqa: E402


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(4)
    T, n, draws = 240, 4, 3
    base = np.cumsum(rng.normal(0.002, 0.01, (T, n)), axis=0) + np.array([4.0, 3.0, 0.05, 8.0])
    Y = np.stack([base + rng.normal(0, 0.001, (T, n)) for _ in range(draws)])
    dates = pd.date_range("1980-01-31", periods=T, freq="ME")
    spec = np.array([3, 3, 1, n, T - 3])
    return L.CompleteData(draws=Y, spec=spec, dates=dates, columns=["A", "B", "C", "GDP"], n_observed=T - 2)


def test_hyperparameter_round_trip_and_bounds():
    free = L.from_mbfvar(PAPER_HYPERPARAMETERS)
    assert L.to_mbfvar(free) == [0.09, 4.30, 1.0, 2.70, 4.30]
    assert np.allclose(L._to_bounded(L._to_unbounded(free)), free)
    assert np.all(L._to_bounded(np.array([50.0, -50.0, 0.0, 3.0])) <= L._BOUNDS[:, 1])


def test_posterior_mean_matches_mbfvar_dummy_regression(data):
    Y = data.draws[0]
    var = L.CompleteDataVAR(Y, data.spec)
    free = L.from_mbfvar(PAPER_HYPERPARAMETERS)
    _, YYact, YYdum, XXact, XXdum = mdd_(L.to_mbfvar(free), Y, data.spec)
    X, Yf = np.vstack((XXdum, XXact)), np.vstack((YYdum, YYact))
    expected = np.linalg.solve(X.T @ X, X.T @ Yf)
    assert np.allclose(var.posterior_mean(free, Y.shape[0]), expected, atol=1e-8)
    # A prefix of the sample equals re-running the regression on Y[:cut].
    cut = 150
    _, Ya, Yd, Xa, Xd = mdd_(L.to_mbfvar(free), Y[:cut], np.r_[data.spec[:4], cut - data.spec[1]])
    Xc, Yc = np.vstack((Xd, Xa)), np.vstack((Yd, Ya))
    var_cut = L.CompleteDataVAR(Y[:cut], np.r_[data.spec[:4], cut - data.spec[1]])
    assert np.allclose(var.posterior_mean(free, cut), var_cut.posterior_mean(free, cut), atol=1e-8)
    assert np.allclose(var.posterior_mean(free, cut), np.linalg.solve(Xc.T @ Xc, Xc.T @ Yc), atol=1e-8)


def test_complete_data_log_ml_averages_mbfvar_mdd(data):
    free = L.from_mbfvar(PAPER_HYPERPARAMETERS)
    expected = np.mean([mdd_(L.to_mbfvar(free), Y, data.spec)[0] for Y in data.draws])
    assert L.complete_data_log_ml(free, data) == pytest.approx(expected)


def test_quarter_end_cuts_respect_observed_sample_and_window(data):
    cuts = L.quarter_end_cuts(data, horizon_q=4, window=5)
    assert len(cuts) == 5 and cuts == sorted(cuts, reverse=True)
    assert all(data.dates[c - 1].month % 3 == 0 for c in cuts)
    assert all(c + 12 <= data.n_observed for c in cuts)


def test_point_loss_is_deterministic_and_positive(data):
    var = L.CompleteDataVAR(data.mean, data.spec)
    cuts = L.quarter_end_cuts(data, 2, 6)
    loss = L.point_loss(var, cuts, 2, [3], np.ones(1))
    free = L.from_mbfvar(PAPER_HYPERPARAMETERS)
    assert loss(free) == loss(free) > 0


def test_select_all_returns_every_criterion(data):
    crits = (L.MFCriterion("ml"), L.MFCriterion("rmse", 2, 6), L.MFCriterion("rmsej", 2, 10))
    out = L.select_all(data, crits, n_starts=1, maxfev=60)
    assert set(out) == {"ml", "rmse_h2_w6", "rmsej_h2_w10", "_paper_complete_data_log_ml"}
    # The ML step never lowers the complete-data log ML relative to the published values.
    assert out["ml"]["complete_data_log_ml"] >= out["_paper_complete_data_log_ml"]["value"] - 1e-6
    for key in ("rmse_h2_w6", "rmsej_h2_w10"):
        assert out[key]["value"] <= out[key]["value_at_start"] + 1e-12


def test_realtime_quarterly_history_uses_complete_quarters_only():
    q = pd.DataFrame({"GDP": [100.0, 101.0]}, index=pd.to_datetime(["2005-03-31", "2005-06-30"]))
    m = pd.DataFrame({"CPI": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=pd.date_range("2005-01-31", periods=5, freq="ME"))
    hist = L.realtime_quarterly_history(q, m)
    assert hist.loc[pd.Period("2005Q1"), "CPI"] == pytest.approx(2.0)
    assert np.isnan(hist.loc[pd.Period("2005Q2"), "CPI"])  # only two months observed
    assert hist.loc[pd.Period("2005Q2"), "GDP"] == 101.0


def test_fast_is_explosive_matches_mbfvar_check():
    from MBFVAR.mfbvar_funcs import is_explosive

    rng = np.random.default_rng(7)
    n, p = 4, 3
    cases = []
    for scale in (0.0005, 0.002, 0.01, 0.05, 0.3):
        for _ in range(60):
            phi = rng.normal(0, scale, (n * p + 1, n))
            phi[:n, :] += np.eye(n)  # unit roots perturbed in both directions
            cases.append(phi)
    rot = np.zeros((n * p + 1, n))  # complex pair just outside the unit circle
    theta, r = 0.4, 1.001
    rot[:2, :2] = r * np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rot[2, 2] = rot[3, 3] = 0.5
    cases.append(rot)
    decisions = [(bool(is_explosive(c, n, p)), L.fast_is_explosive(c, n, p)) for c in cases]
    assert all(a == b for a, b in decisions)
    assert 0 < sum(a for a, _ in decisions) < len(decisions)  # both outcomes covered
    assert L.fast_is_explosive(rot, n, p)
