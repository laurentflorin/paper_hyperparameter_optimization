import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from glp_hyperparameter_optimization import scoring as S


def test_adjusted_actual_removes_vintage_level_shift_for_log_series_only():
    adj = S.adjusted_actual([120.0, 5.0], [118.0, 4.0], [100.0, 4.5], [True, False])
    assert np.allclose(adj, [102.0, 5.0])  # log: 120 - 118 + 100; rate: unchanged


def test_crps_sample_matches_gaussian_closed_form():
    rng = np.random.default_rng(0)
    draws = rng.normal(1.0, 2.0, size=(200000, 1))
    assert S.crps_sample(draws, np.array([0.3]))[0] == pytest.approx(S.crps_gaussian(1.0, 2.0, 0.3), rel=0.01)


def test_diebold_mariano_sign_and_degenerate_cases():
    rng = np.random.default_rng(1)
    base = rng.normal(size=200) ** 2
    better = base * 0.5
    stat, p = S.diebold_mariano(better, base, 1)
    assert stat < 0 and p < 0.01
    assert np.isnan(S.diebold_mariano(base, base, 1)[0])
    assert np.isnan(S.diebold_mariano(base[:5], base[:5] * 2, 1)[0])


def test_holm_adjust_is_monotone_and_capped():
    adj = S.holm_adjust([0.01, 0.04, 0.03, np.nan])
    assert np.allclose(adj[:3], [0.03, 0.06, 0.06]) and np.isnan(adj[3])


def test_mcs_excludes_clearly_worse_model():
    rng = np.random.default_rng(2)
    good = rng.normal(1.0, 0.2, 300)
    losses = pd.DataFrame({"a": good, "b": good + rng.normal(0, 0.05, 300), "bad": good + 1.0})
    mcs = S.model_confidence_set(losses, alpha=0.1, n_boot=500)
    assert not mcs.loc["bad", "in_mcs"]
    assert mcs.loc[["a", "b"], "in_mcs"].all()
