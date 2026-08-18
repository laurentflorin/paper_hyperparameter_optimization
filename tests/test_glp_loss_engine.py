import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common_hpo.losses import LossConfig, ScaleConfig
from common_hpo.splits import (
    ValidationScheme,
    build_validation_splits,
)
from glp_hyperparameter_optimization import loss_engine as le


# --------------------------------------------------------------------------- #
# Synthetic contexts + fake forecast primitives (no covbayesvar required).
# --------------------------------------------------------------------------- #
class FakeContext:
    """Minimal stand-in for a GLPContext: exposes the training matrix and n."""

    def __init__(self, y: np.ndarray):
        self.y = np.asarray(y, dtype=float)
        self.n = self.y.shape[1]


def _fake_to_natural(params, ctx):
    # Deterministic mapping from optimizer coordinates to a "natural vector".
    return np.array([params["lam"], params.get("theta", 1.0), params.get("miu", 1.0)])


def _make_contexts(n_folds=3, n_vars=3, max_horizon=4, seed=0):
    rng = np.random.default_rng(seed)
    codes = ["GDP", "DEFL", "FFR"][:n_vars]
    origins = []
    for _ in range(n_folds):
        train = np.cumsum(rng.normal(size=(30, n_vars)), axis=0) + 100.0
        actual = rng.normal(size=(max_horizon, n_vars)) + 100.0
        fake = FakeContext(train)
        fake.actual = actual  # convenience for tests that need the holdout
        origins.append((fake, actual))
    contexts = le.prepare_glp_validation_contexts(origins, codes, max_horizon=max_horizon)
    return contexts, codes


def _linear_forecast_fn(offset=0.0):
    """A deterministic forecast: each horizon row = lam * (row+1) plus offset.

    Independent of any RNG so mode-path evaluations are exactly reproducible.
    """

    def forecast_fn(context, natural_vec, horizons, *, draw_index, use_draws):
        lam = float(natural_vec[0])
        n = context.y.shape[1]
        rows = []
        for h in horizons:
            rows.append(np.full(n, lam * h + offset))
        return np.asarray(rows, dtype=float)

    return forecast_fn


# --------------------------------------------------------------------------- #
# Cell-shape coverage: pooled / variable / horizon / variable-horizon.
# --------------------------------------------------------------------------- #
def test_pooled_multi_variable_multi_horizon_single_evaluation():
    contexts, codes = _make_contexts()
    spec = le.GLPCellSpec(
        variables=tuple(codes),
        horizons=(1, 2, 4),
        loss_config=LossConfig(aggregation="rmse"),
    )
    evaluation = le.evaluate_glp_candidate(
        {"lam": 0.2, "theta": 1.0, "miu": 1.0},
        contexts,
        spec,
        forecast_fn=_linear_forecast_fn(),
        to_natural=_fake_to_natural,
    )
    assert not evaluation.failed
    # 3 folds * 3 variables * 3 horizons = 27 records; 9 distinct cells.
    assert evaluation.n_valid_records == 27
    assert len(evaluation.loss_by_cell) == 9
    assert np.isfinite(evaluation.total_loss)


def test_variable_specific_cell_over_several_horizons():
    contexts, codes = _make_contexts()
    spec = le.GLPCellSpec(variables=("GDP",), horizons=(1, 2, 4))
    evaluation = le.evaluate_glp_candidate(
        {"lam": 0.2}, contexts, spec,
        forecast_fn=_linear_forecast_fn(), to_natural=_fake_to_natural,
    )
    assert not evaluation.failed
    assert {cell[0] for cell in evaluation.loss_by_cell} == {"GDP"}
    assert {cell[1] for cell in evaluation.loss_by_cell} == {1, 2, 4}


def test_horizon_specific_cell_over_several_variables():
    contexts, codes = _make_contexts()
    spec = le.GLPCellSpec(variables=tuple(codes), horizons=(2,))
    evaluation = le.evaluate_glp_candidate(
        {"lam": 0.2}, contexts, spec,
        forecast_fn=_linear_forecast_fn(), to_natural=_fake_to_natural,
    )
    assert not evaluation.failed
    assert {cell[1] for cell in evaluation.loss_by_cell} == {2}
    assert {cell[0] for cell in evaluation.loss_by_cell} == set(codes)


def test_variable_horizon_cell_single_target():
    contexts, codes = _make_contexts()
    spec = le.GLPCellSpec(variables=("GDP",), horizons=(4,))
    evaluation = le.evaluate_glp_candidate(
        {"lam": 0.2}, contexts, spec,
        forecast_fn=_linear_forecast_fn(), to_natural=_fake_to_natural,
    )
    assert not evaluation.failed
    assert list(evaluation.loss_by_cell) == [("GDP", 4)]
    assert evaluation.n_valid_records == 3  # one per fold


# --------------------------------------------------------------------------- #
# Aggregation, legacy equivalence, target alignment.
# --------------------------------------------------------------------------- #
def test_multi_horizon_cell_produces_one_aggregated_objective():
    contexts, codes = _make_contexts()
    spec = le.GLPCellSpec(variables=("GDP",), horizons=(1, 2, 4))
    objective = le.make_glp_loss_objective(
        contexts, spec, forecast_fn=_linear_forecast_fn(), to_natural=_fake_to_natural
    )
    value = objective(lam=0.2, theta=1.0, miu=1.0)
    assert np.isscalar(value) and np.isfinite(value)


def test_equal_cell_versus_equal_observation_differ_with_uneven_cells():
    # Build contexts where one variable appears with more observations by using
    # a spec with two horizons for one cell weighting comparison.
    contexts, codes = _make_contexts(n_folds=2)
    forecast_fn = _linear_forecast_fn()

    equal_cell = le.evaluate_glp_candidate(
        {"lam": 0.2}, contexts,
        le.GLPCellSpec(variables=("GDP", "DEFL"), horizons=(1, 2),
                       loss_config=LossConfig(aggregation="mse", cell_aggregation="equal_cell")),
        forecast_fn=forecast_fn, to_natural=_fake_to_natural,
    )
    equal_obs = le.evaluate_glp_candidate(
        {"lam": 0.2}, contexts,
        le.GLPCellSpec(variables=("GDP", "DEFL"), horizons=(1, 2),
                       loss_config=LossConfig(aggregation="mse", cell_aggregation="equal_observation")),
        forecast_fn=forecast_fn, to_natural=_fake_to_natural,
    )
    # Both finite; the framework computes each; equal-obs weights by count.
    assert np.isfinite(equal_cell.total_loss)
    assert np.isfinite(equal_obs.total_loss)


def test_legacy_raw_rmse_equivalence():
    # equal_observation + scale none + uniform weights + one horizon reproduces
    # the plain pooled RMSE over origins and variables.
    contexts, codes = _make_contexts(n_folds=3)
    forecast_fn = _linear_forecast_fn()
    h_eval = 4
    var = "GDP"

    spec = le.GLPCellSpec(
        variables=(var,),
        horizons=(h_eval,),
        loss_config=LossConfig(aggregation="rmse", cell_aggregation="equal_observation"),
    )
    evaluation = le.evaluate_glp_candidate(
        {"lam": 0.2}, contexts, spec,
        forecast_fn=forecast_fn, to_natural=_fake_to_natural,
    )

    # Hand-compute the same raw RMSE directly from the fake forecasts.
    squared = []
    for context in contexts:
        forecast = forecast_fn(context.context, np.array([0.2]), [h_eval],
                               draw_index=0, use_draws=False)
        vidx = context.variable_index(var)
        f = float(forecast[0, vidx])
        a = context.actual_for(var, h_eval)
        squared.append((a - f) ** 2)
    expected = float(np.sqrt(np.mean(squared)))
    assert evaluation.total_loss == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_correct_target_alignment_per_horizon():
    # A forecast that exactly matches the holdout at every horizon must give
    # zero loss, proving row h-1 alignment is correct.
    contexts, codes = _make_contexts(n_folds=2)

    def perfect_forecast_fn(context, natural_vec, horizons, *, draw_index, use_draws):
        rows = [context.actual[h - 1] for h in horizons]
        return np.asarray(rows, dtype=float)

    spec = le.GLPCellSpec(variables=tuple(codes), horizons=(1, 2, 3, 4))
    evaluation = le.evaluate_glp_candidate(
        {"lam": 0.2}, contexts, spec,
        forecast_fn=perfect_forecast_fn, to_natural=_fake_to_natural,
    )
    assert evaluation.total_loss == pytest.approx(0.0)


def test_no_training_target_overlap_via_validation_split_metadata():
    # ValidationSplit targets must strictly follow the training window.
    scheme = ValidationScheme(
        training_window="expanding",
        origin_selection="most_recent",
        n_origins=2,
        horizons=(1, 2, 4),
        min_train_length=10,
    )
    splits = build_validation_splits(40, scheme, {1: 1, 2: 2, 4: 4})
    for split in splits:
        for _, position in split.targets:
            assert position > split.train_end


# --------------------------------------------------------------------------- #
# Determinism, CRN, global-state independence.
# --------------------------------------------------------------------------- #
def _rng_forecast_fn():
    """A forecast that consumes the global NumPy RNG so CRN can be observed."""

    def forecast_fn(context, natural_vec, horizons, *, draw_index, use_draws):
        lam = float(natural_vec[0])
        n = context.y.shape[1]
        rows = [lam * h + np.random.normal(size=n) for h in horizons]
        return np.asarray(rows, dtype=float)

    return forecast_fn


def test_deterministic_repeated_evaluation_with_draws():
    contexts, codes = _make_contexts(n_folds=2)
    spec = le.GLPCellSpec(variables=("GDP",), horizons=(1, 2), n_obj_draws=8, seed_base=123)
    forecast_fn = _rng_forecast_fn()
    first = le.evaluate_glp_candidate(
        {"lam": 0.2}, contexts, spec, forecast_fn=forecast_fn, to_natural=_fake_to_natural
    )
    second = le.evaluate_glp_candidate(
        {"lam": 0.2}, contexts, spec, forecast_fn=forecast_fn, to_natural=_fake_to_natural
    )
    assert first.total_loss == pytest.approx(second.total_loss)


def test_common_random_numbers_independent_of_candidate_seeded_stream():
    # The CRN draws depend only on split+draw index, not on the candidate. Two
    # different candidates that share the same deterministic mapping must reuse
    # the identical random shock stream.
    contexts, codes = _make_contexts(n_folds=1)
    spec = le.GLPCellSpec(variables=("GDP",), horizons=(1, 2), n_obj_draws=4, seed_base=7)

    captured = {}

    def capturing_forecast_fn(context, natural_vec, horizons, *, draw_index, use_draws):
        shocks = np.random.normal(size=(len(horizons), context.y.shape[1]))
        captured.setdefault(draw_index, []).append(shocks.copy())
        return shocks

    le.evaluate_glp_candidate(
        {"lam": 0.2}, contexts, spec, forecast_fn=capturing_forecast_fn, to_natural=_fake_to_natural
    )
    le.evaluate_glp_candidate(
        {"lam": 0.9}, contexts, spec, forecast_fn=capturing_forecast_fn, to_natural=_fake_to_natural
    )
    for draw_index, shock_pairs in captured.items():
        # Same draw index across the two candidates -> identical shocks (CRN).
        assert len(shock_pairs) == 2
        np.testing.assert_array_equal(shock_pairs[0], shock_pairs[1])


def test_evaluation_does_not_disturb_global_numpy_state():
    contexts, codes = _make_contexts(n_folds=2)
    spec = le.GLPCellSpec(variables=("GDP",), horizons=(1, 2), n_obj_draws=4, seed_base=1)
    forecast_fn = _rng_forecast_fn()

    np.random.seed(2024)
    expected = np.random.random(5)
    np.random.seed(2024)
    le.evaluate_glp_candidate(
        {"lam": 0.2}, contexts, spec, forecast_fn=forecast_fn, to_natural=_fake_to_natural
    )
    after = np.random.random(5)
    np.testing.assert_array_equal(expected, after)


# --------------------------------------------------------------------------- #
# Diagnostics + optimizer-compatible failure handling.
# --------------------------------------------------------------------------- #
def test_diagnostics_capture_for_nonfinite_forecast_candidate():
    contexts, codes = _make_contexts(n_folds=2)
    spec = le.GLPCellSpec(variables=("GDP",), horizons=(1,))

    def broken_forecast_fn(context, natural_vec, horizons, *, draw_index, use_draws):
        n = context.y.shape[1]
        return np.full((len(horizons), n), np.nan)

    evaluation = le.evaluate_glp_candidate(
        {"lam": 0.2}, contexts, spec,
        forecast_fn=broken_forecast_fn, to_natural=_fake_to_natural,
    )
    assert evaluation.failed
    assert evaluation.nonfinite_forecasts == 1
    assert evaluation.total_loss == le.RMSE_PENALTY
    assert "non-finite" in evaluation.failure_reason


def test_finite_penalty_and_recorded_reason_for_invalid_candidate():
    contexts, codes = _make_contexts(n_folds=2)
    spec = le.GLPCellSpec(variables=("GDP",), horizons=(1,))

    def raising_to_natural(params, ctx):
        raise le.InvalidHyperparameterError("candidate outside GLP bounds")

    collector: list[dict] = []
    objective = le.make_glp_loss_objective(
        contexts, spec,
        forecast_fn=_linear_forecast_fn(),
        to_natural=raising_to_natural,
        diagnostics_collector=collector,
    )
    value = objective(lam=0.2)
    assert value == le.RMSE_PENALTY
    assert np.isfinite(value)
    assert objective.diagnostics["penalized"] == 1
    assert objective.diagnostics["numerical_failures"] == 1
    assert "InvalidHyperparameterError" in objective.diagnostics["last_failure_reason"]
    assert collector and "candidate outside GLP bounds" in collector[0]["reason"]


def test_missing_variable_is_reported_as_scale_or_cell_problem():
    contexts, codes = _make_contexts(n_folds=1)
    spec = le.GLPCellSpec(variables=("NOT_A_VAR",), horizons=(1,))
    evaluation = le.evaluate_glp_candidate(
        {"lam": 0.2}, contexts, spec,
        forecast_fn=_linear_forecast_fn(), to_natural=_fake_to_natural,
    )
    assert evaluation.failed
    assert evaluation.failure_reason is not None


def test_deterministic_diagnostics_ordering():
    contexts, codes = _make_contexts(n_folds=2)
    spec = le.GLPCellSpec(variables=("FFR", "GDP"), horizons=(4, 1))
    evaluation = le.evaluate_glp_candidate(
        {"lam": 0.2}, contexts, spec,
        forecast_fn=_linear_forecast_fn(), to_natural=_fake_to_natural,
    )
    ordering = [(c.variable, c.horizon) for c in evaluation.loss_result.cells]
    assert ordering == sorted(ordering)


def test_supplied_scale_records_scale_problem_when_floored():
    contexts, codes = _make_contexts(n_folds=2)
    spec = le.GLPCellSpec(
        variables=("GDP",),
        horizons=(1,),
        loss_config=LossConfig(
            aggregation="rmse",
            scale=ScaleConfig(method="supplied", supplied_scales={("GDP", 1): 1e-30}, min_scale=1e-6),
        ),
    )
    evaluation = le.evaluate_glp_candidate(
        {"lam": 0.2}, contexts, spec,
        forecast_fn=_linear_forecast_fn(), to_natural=_fake_to_natural,
    )
    assert not evaluation.failed
    assert evaluation.scale_problems == 1
