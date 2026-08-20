"""Regression tests for the Mango RMSE hyperparameter updaters.

These cover the two failure modes that previously broke the GLP arm:

1. ``glp_hyperparameter_optimization.forecasting`` failed to import because
   ``update_hyperparameters_mango_rmse`` / ``_random`` were missing from
   ``glp_model``.
2. A Mango run whose every evaluation hit the sentinel penalty could return
   in-bounds junk hyperparameters instead of failing loudly.

None of these tests require the optional ``covbayesvar`` dependency.
"""

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from glp_hyperparameter_optimization import glp_model as gm


def test_rmse_updaters_are_exported():
    assert callable(gm.update_hyperparameters_mango_rmse)
    assert callable(gm.update_hyperparameters_mango_rmse_random)


def test_forecasting_module_imports():
    # The ImportError this guards against killed every GLP runner, including
    # the MDD arm of the headline comparison.
    from glp_hyperparameter_optimization import forecasting

    assert forecasting.update_hyperparameters_mango_rmse is (
        gm.update_hyperparameters_mango_rmse
    )
    assert forecasting.update_hyperparameters_mango_rmse_random is (
        gm.update_hyperparameters_mango_rmse_random
    )


@pytest.mark.parametrize(
    "func, extra",
    [
        (gm.update_hyperparameters_mango_rmse, {}),
        (gm.update_hyperparameters_mango_rmse_random, {"random_seed": 7}),
    ],
)
def test_updater_signatures_accept_forecasting_call(func, extra):
    # Keyword set used by forecasting.select_hyperparameters.
    kwargs = dict(
        model_codes=["GDP"],
        var_of_interest=["GDP"],
        H=4,
        h_eval=4,
        n_eval=2,
        min_t=None,
        n_obj_draws=200,
        init_points=2,
        n_iter=2,
        njobs=1,
        hyperpriors=1,
        sur=1,
        noc=1,
        mnpsi=1,
        mnalpha=0,
        vc=10e6,
        **extra,
    )
    inspect.signature(func).bind(np.zeros((90, 1)), 4, **kwargs)


@pytest.mark.parametrize(
    "func, extra",
    [
        (gm.update_hyperparameters_mango_rmse, {}),
        (gm.update_hyperparameters_mango_rmse_random, {"random_seed": 7}),
    ],
)
def test_updaters_reject_non_positive_budget(func, extra):
    with pytest.raises(ValueError):
        func(
            np.zeros((90, 1)),
            4,
            model_codes=["GDP"],
            var_of_interest=["GDP"],
            H=4,
            init_points=0,
            **extra,
        )


def _objective(valid: int, score: float):
    def calc(**params):
        return score

    calc.diagnostics = {"valid": valid, "penalized": 3, "nonfinite": 0}
    return calc


def test_finalize_rejects_missing_best_params():
    with pytest.raises(gm.HyperparameterOptimizationError, match="best_params"):
        gm._finalize_rmse_result(
            {}, _objective(5, 1.0), None,
            label="Mango RMSE", optimizer_seed=None, n_obj_draws=1, origin_ks=[0],
        )


def test_finalize_rejects_run_without_any_valid_evaluation():
    with pytest.raises(gm.HyperparameterOptimizationError, match="valid finite"):
        gm._finalize_rmse_result(
            {"best_params": {"lam": 0.2}}, _objective(0, 1.0), None,
            label="Mango RMSE", optimizer_seed=None, n_obj_draws=1, origin_ks=[0],
        )


@pytest.mark.parametrize("score", [gm.RMSE_PENALTY, gm.RMSE_PENALTY * 10, np.nan, np.inf])
def test_finalize_rejects_penalty_as_best_score(score):
    # Fail closed: never return in-bounds junk when the best recorded score is
    # the sentinel penalty (peer-review finding).
    with pytest.raises(gm.HyperparameterOptimizationError):
        gm._finalize_rmse_result(
            {"best_params": {"lam": 0.2}}, _objective(5, float(score)), None,
            label="Mango RMSE", optimizer_seed=None, n_obj_draws=1, origin_ks=[0],
        )


def test_finalize_accepts_valid_score(monkeypatch):
    monkeypatch.setattr(gm, "_best_hyperparameters", lambda params, ctx: {"lambda": 0.2})
    hyper = gm._finalize_rmse_result(
        {"best_params": {"lam": 0.2}}, _objective(5, 0.75), None,
        label="Mango RMSE", optimizer_seed=11, n_obj_draws=4, origin_ks=[0, 1],
    )
    diagnostics = hyper["optimization_diagnostics"]
    assert diagnostics["best_score"] == pytest.approx(0.75)
    assert diagnostics["objective_direction"] == "minimize"
    assert diagnostics["penalty"] == gm.RMSE_PENALTY
    assert diagnostics["postcondition_revalidated"] is True
    assert diagnostics["evaluation_origins"] == [0, 1]
    assert diagnostics["valid_evaluations_observed_in_process"] == 5
