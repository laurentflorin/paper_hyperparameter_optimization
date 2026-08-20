"""Regression tests for GLP prior-scale leakage (audit finding GLP-04 / H3).

The GLP prior scale ``psi`` is derived from the AR(1) residual variances
``ctx.SS`` of the estimation sample. When an inner validation fold is scored,
every ``SS``-derived quantity must come from *that fold's own training rows*.
Resolving the search configuration once against the outer sample and reusing it
across folds injects the inner holdout targets into the prior used to forecast
them -- a one-sided leak that flatters the forecast-loss arm.

These tests exercise the search-config / psi-resolution layer directly and do
not require covbayesvar.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from glp_hyperparameter_optimization.glp_model import (
    PSI_LOG_MULTIPLIER_BOUNDS,
    InvalidHyperparameterError,
)
from glp_hyperparameter_optimization.search_config import (
    GLPSearchConfig,
    GLPSearchConfigError,
)


# --------------------------------------------------------------------------- #
# A covbayesvar-free stand-in whose SS is a genuine function of the sample.
# --------------------------------------------------------------------------- #
def _ar1_residual_variances(y: np.ndarray) -> np.ndarray:
    """Per-variable AR(1) OLS residual variance -- the semantics of ``ctx.SS``."""

    y = np.asarray(y, dtype=float)
    variances = []
    for column in range(y.shape[1]):
        series = y[:, column]
        lhs = series[1:]
        rhs = np.column_stack([np.ones(lhs.size), series[:-1]])
        coef, *_ = np.linalg.lstsq(rhs, lhs, rcond=None)
        residual = lhs - rhs @ coef
        variances.append(float(residual @ residual) / max(lhs.size, 1))
    return np.asarray(variances, dtype=float)


def _context_from_sample(y: np.ndarray) -> SimpleNamespace:
    """Build a GLPContext-like object whose SS is estimated from ``y`` alone."""

    y = np.asarray(y, dtype=float)
    ss = _ar1_residual_variances(y)
    return SimpleNamespace(
        n=int(y.shape[1]),
        y=y,
        mn={"psi": 1},
        sur=1,
        noc=1,
        SS=ss.reshape(-1, 1),
        MIN={"lambda": 1.0e-4, "theta": 1.0e-4, "miu": 1.0e-4, "psi": ss * 0.01},
        MAX={"lambda": 5.0, "theta": 50.0, "miu": 50.0, "psi": ss * 100.0},
    )


def _outer_sample(seed: int = 7, rows: int = 80, cols: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(size=(rows, cols)), axis=0)


TRAIN_END = 60  # inner fold trains on rows [:TRAIN_END]; the rest is its holdout.


def _mutate_holdout_only(y: np.ndarray) -> np.ndarray:
    """Perturb ONLY the inner-holdout rows (train_end .. end) of the outer sample."""

    mutated = np.array(y, dtype=float, copy=True)
    rng = np.random.default_rng(1234)
    mutated[TRAIN_END:, :] += 2.0 * rng.normal(size=mutated[TRAIN_END:, :].shape)
    return mutated


def _psi_block(natural_vec: np.ndarray, n: int) -> np.ndarray:
    """Extract the psi block from ``[lambda, psi..., theta, miu]``."""

    return np.asarray(natural_vec, dtype=float)[1 : 1 + n]


# --------------------------------------------------------------------------- #
# GLP-04: the inner-fold prior scale must not see the inner holdout.
# --------------------------------------------------------------------------- #
def test_holdout_rows_do_not_change_inner_fold_psi_under_fixed_context_ss():
    """Mutating only the holdout rows must leave the inner-fold psi unchanged.

    This is the DEFAULT configuration (``--no-optimize-psi
    --fixed-psi-source context_ss``). It fails if the search configuration is
    resolved once against the outer context and reused across folds.
    """

    config = GLPSearchConfig.reduced_lambda_theta_miu(fixed_psi_source="context_ss")
    params = {"lam": 0.2, "theta": 1.0, "miu": 1.0}

    baseline = _outer_sample()
    mutated = _mutate_holdout_only(baseline)

    # The inner fold's own training rows are identical in both worlds ...
    np.testing.assert_allclose(baseline[:TRAIN_END], mutated[:TRAIN_END])
    # ... while the outer sample (which spans the holdout) is not.
    assert not np.allclose(
        _ar1_residual_variances(baseline), _ar1_residual_variances(mutated)
    )

    psi_by_world = []
    for sample in (baseline, mutated):
        outer_ctx = _context_from_sample(sample)
        fold_ctx = _context_from_sample(sample[:TRAIN_END])
        to_natural = config.fold_resolving_to_natural(
            reference=config.resolve(outer_ctx)
        )
        psi_by_world.append(_psi_block(to_natural(params, fold_ctx), fold_ctx.n))

    np.testing.assert_allclose(psi_by_world[0], psi_by_world[1], rtol=0, atol=0)


def test_prebinding_the_outer_resolution_is_the_leak_this_guards_against():
    """Document the defect: a pre-bound outer resolution does leak the holdout."""

    config = GLPSearchConfig.reduced_lambda_theta_miu(fixed_psi_source="context_ss")
    params = {"lam": 0.2, "theta": 1.0, "miu": 1.0}

    baseline = _outer_sample()
    mutated = _mutate_holdout_only(baseline)

    leaky_psi = []
    for sample in (baseline, mutated):
        outer_ctx = _context_from_sample(sample)
        fold_ctx = _context_from_sample(sample[:TRAIN_END])
        resolved = config.resolve(outer_ctx)  # bound to the OUTER sample
        # The outer-derived psi is injected verbatim into the fold's natural
        # vector (or, for a large enough perturbation, is rejected by the fold's
        # own bounds -- either way it is holdout-dependent).
        try:
            psi = _psi_block(resolved.to_natural(params, fold_ctx), fold_ctx.n)
        except InvalidHyperparameterError:
            psi = np.ravel(resolved.fixed_psi)
        leaky_psi.append(psi)

    assert not np.allclose(leaky_psi[0], leaky_psi[1]), (
        "expected the pre-bound outer resolution to carry holdout information; "
        "if this passes the fixture no longer exercises the leak"
    )


def test_holdout_rows_do_not_change_inner_fold_search_domain():
    """The optimizer's per-fold search domain must be holdout-independent."""

    config = GLPSearchConfig.legacy_full(psi_parameterization="ss_log_multiplier")

    baseline = _outer_sample()
    mutated = _mutate_holdout_only(baseline)

    domains = []
    for sample in (baseline, mutated):
        fold_ctx = _context_from_sample(sample[:TRAIN_END])
        resolved = config.resolve(fold_ctx)
        domains.append(
            (resolved.optimized_names, resolved.transformed_bounds, resolved.natural_bounds)
        )

    assert domains[0][0] == domains[1][0]
    assert domains[0][1] == domains[1][1]
    assert domains[0][2] == domains[1][2]


def test_absolute_psi_bounds_are_holdout_contaminated_when_taken_from_outer_ctx():
    """The absolute parameterization's bounds are context-derived, hence leaky."""

    config = GLPSearchConfig.legacy_full(psi_parameterization="absolute")
    baseline = _outer_sample()
    mutated = _mutate_holdout_only(baseline)

    outer_bounds = [
        config.resolve(_context_from_sample(sample)).transformed_bounds["psi_0"]
        for sample in (baseline, mutated)
    ]
    assert outer_bounds[0] != outer_bounds[1]
    assert config.has_context_dependent_domain

    # ss_log_multiplier is invariant to the same perturbation.
    stable = GLPSearchConfig.legacy_full(psi_parameterization="ss_log_multiplier")
    stable_bounds = [
        stable.resolve(_context_from_sample(sample)).transformed_bounds[
            "psi_log_multiplier_0"
        ]
        for sample in (baseline, mutated)
    ]
    assert stable_bounds[0] == stable_bounds[1] == PSI_LOG_MULTIPLIER_BOUNDS
    assert not stable.has_context_dependent_domain


# --------------------------------------------------------------------------- #
# H3: one coordinate, one documented meaning, in every fold.
# --------------------------------------------------------------------------- #
def test_ss_log_multiplier_coordinate_has_the_documented_meaning_in_every_fold():
    """``psi_i = SS_i * exp(psi_log_multiplier_i)`` using each fold's own SS."""

    config = GLPSearchConfig.legacy_full(psi_parameterization="ss_log_multiplier")
    sample = _outer_sample()
    outer_ctx = _context_from_sample(sample)
    to_natural = config.fold_resolving_to_natural(reference=config.resolve(outer_ctx))

    log_multipliers = np.array([-1.5, 0.0, 0.75])
    params = {"lam": 0.2, "theta": 1.0, "miu": 1.0}
    params.update(
        {f"psi_log_multiplier_{i}": float(v) for i, v in enumerate(log_multipliers)}
    )

    for train_end in (40, 50, 60, 70):
        fold_ctx = _context_from_sample(sample[:train_end])
        psi = _psi_block(to_natural(params, fold_ctx), fold_ctx.n)
        expected = np.ravel(fold_ctx.SS) * np.exp(log_multipliers)
        np.testing.assert_allclose(psi, expected, rtol=1e-12)


def test_absolute_coordinate_meaning_is_not_fold_stable():
    """The same absolute coordinate can be feasible in one fold and rejected in another."""

    config = GLPSearchConfig.legacy_full(psi_parameterization="absolute")
    sample = _outer_sample()
    to_natural = config.fold_resolving_to_natural()

    fold_a = _context_from_sample(sample[:30])
    fold_b = _context_from_sample(sample[:TRAIN_END])
    # A psi level sitting inside fold_a's admissible band.
    psi_values = np.ravel(fold_a.SS) * 0.02
    params = {"lam": 0.2, "theta": 1.0, "miu": 1.0}
    params.update({f"psi_{i}": float(v) for i, v in enumerate(psi_values)})

    to_natural(params, fold_a)  # feasible here
    if np.any(psi_values <= np.ravel(fold_b.MIN["psi"])):
        with pytest.raises(InvalidHyperparameterError):
            to_natural(params, fold_b)


def test_fold_resolver_rejects_coordinate_name_drift():
    """A fold whose coordinates differ from the optimizer's space must raise."""

    config = GLPSearchConfig.legacy_full(psi_parameterization="ss_log_multiplier")
    sample = _outer_sample(cols=3)
    reference = config.resolve(_context_from_sample(sample))
    to_natural = config.fold_resolving_to_natural(reference=reference)

    narrow_fold = _context_from_sample(sample[:TRAIN_END, :2])
    params = {"lam": 0.2, "theta": 1.0, "miu": 1.0}
    params.update({f"psi_log_multiplier_{i}": 0.0 for i in range(3)})
    with pytest.raises(GLPSearchConfigError):
        to_natural(params, narrow_fold)


def test_fold_resolver_memoizes_per_context():
    """Repeated calls with the same context reuse one resolution."""

    config = GLPSearchConfig.reduced_lambda_theta_miu(fixed_psi_source="context_ss")
    sample = _outer_sample()
    fold_ctx = _context_from_sample(sample[:TRAIN_END])
    calls = {"n": 0}
    original = GLPSearchConfig.resolve

    def counting_resolve(self, ctx):
        calls["n"] += 1
        return original(self, ctx)

    GLPSearchConfig.resolve = counting_resolve
    try:
        to_natural = config.fold_resolving_to_natural()
        params = {"lam": 0.2, "theta": 1.0, "miu": 1.0}
        to_natural(params, fold_ctx)
        to_natural(params, fold_ctx)
    finally:
        GLPSearchConfig.resolve = original
    assert calls["n"] == 1


# --------------------------------------------------------------------------- #
# Production wiring: the scope-grid selector must score folds fold-locally.
# --------------------------------------------------------------------------- #
def _import_scope_grid():
    script_root = REPO_ROOT / "scripts" / "glp"
    if str(script_root) not in sys.path:
        sys.path.insert(0, str(script_root))
    import run_glp_scope_grid as scope_grid

    return scope_grid


def _run_selector_capturing_to_natural(tmp_path, monkeypatch, *extra_cli):
    """Drive the real scope-grid selector and return (captured_to_natural, ...)."""

    from types import SimpleNamespace

    import mango

    from common_hpo.losses import LossConfig
    from glp_hyperparameter_optimization.loss_engine import (
        GLPCandidateEvaluation,
        prepare_glp_validation_contexts,
    )

    scope_grid = _import_scope_grid()

    parser = scope_grid.build_parser()
    args = parser.parse_args(
        [
            "--output-root",
            str(tmp_path / "scope-study"),
            "--model-size",
            "small",
            "--selection-scopes",
            "pooled",
            "--target-horizons",
            "1",
            *extra_cli,
        ]
    )
    config = scope_grid.build_study_config(args, argv=[], program="test")

    sample = _outer_sample(cols=3)
    outer_ctx = _context_from_sample(sample)
    fold_ends = (40, 50, TRAIN_END)
    fold_contexts = [_context_from_sample(sample[:end]) for end in fold_ends]
    codes = tuple(scope_grid._model_codes("small"))[: outer_ctx.n]

    bundle = SimpleNamespace(
        y=sample,
        codes=codes,
        quarter_index=None,
        context=outer_ctx,
    )
    monkeypatch.setattr(
        scope_grid, "_load_outer_origin_bundle", lambda *a, **k: bundle
    )

    validation_contexts = prepare_glp_validation_contexts(
        [(ctx, np.zeros((1, len(codes)))) for ctx in fold_contexts],
        codes,
        max_horizon=1,
    )
    monkeypatch.setattr(
        scope_grid, "_objective_contexts", lambda **k: validation_contexts
    )

    captured: dict[str, object] = {}

    def fake_make_objective(contexts, spec, *, to_natural, **kwargs):
        captured["to_natural"] = to_natural

        def objective(**params):
            return 0.0

        objective.diagnostics = {}
        return objective

    monkeypatch.setattr(scope_grid, "make_glp_loss_objective", fake_make_objective)
    monkeypatch.setattr(
        scope_grid,
        "evaluate_glp_candidate",
        lambda *a, **k: GLPCandidateEvaluation(
            failed=False,
            total_loss=0.5,
            loss_by_cell={},
            n_valid_records=1,
            numerical_failures=0,
            nonfinite_forecasts=0,
            scale_problems=0,
        ),
    )

    best_params = {"lam": 0.2, "theta": 1.0, "miu": 1.0}

    class FakeTuner:
        def __init__(self, space, objective, conf):
            self.space = space

        def minimize(self):
            return {"best_params": dict(best_params)}

    monkeypatch.setattr(mango, "Tuner", FakeTuner)

    holder: dict[str, object] = {}

    def fake_experiment(*a, **kwargs):
        holder["selector"] = kwargs["selector"]
        return "sentinel"

    monkeypatch.setattr(scope_grid, "run_glp_selection_experiment", fake_experiment)

    scope_grid.run_scope_study(
        SimpleNamespace(selection_plan=None), config, panel=object()
    )

    request = SimpleNamespace(
        origin_label="2000-01-01",
        search_config=config.search_config,
        validation_scheme=config.validation_scheme,
        variables=codes[:1],
        horizons=(1,),
        loss_config=LossConfig(),
        seed=0,
    )
    selection = holder["selector"](request)
    return captured["to_natural"], best_params, fold_contexts, outer_ctx, selection


def test_scope_grid_selector_resolves_psi_per_inner_fold(tmp_path, monkeypatch):
    """The selector's objective must map candidates through each fold's own SS.

    Fails against the leaky behavior, where the objective was handed an
    outer-sample ``ResolvedGLPSearch.to_natural`` whose fixed psi is the outer
    ``SS`` -- estimated partly on the inner holdout targets.
    """

    to_natural, params, fold_contexts, outer_ctx, selection = (
        _run_selector_capturing_to_natural(tmp_path, monkeypatch)
    )

    for fold_ctx in fold_contexts:
        psi = _psi_block(to_natural(params, fold_ctx), fold_ctx.n)
        np.testing.assert_allclose(psi, np.ravel(fold_ctx.SS), rtol=1e-12)
        assert not np.allclose(psi, np.ravel(outer_ctx.SS))

    # The final outer forecast still uses the outer context's own psi.
    np.testing.assert_allclose(
        _psi_block(np.asarray(selection.natural_vector), outer_ctx.n),
        np.ravel(outer_ctx.SS),
        rtol=1e-12,
    )


def _scope_config(tmp_path, *extra_cli):
    scope_grid = _import_scope_grid()
    parser = scope_grid.build_parser()
    args = parser.parse_args(
        [
            "--output-root",
            str(tmp_path / "scope-study"),
            "--model-size",
            "small",
            "--selection-scopes",
            "pooled",
            "--target-horizons",
            "1",
            *extra_cli,
        ]
    )
    return scope_grid.build_study_config(args, argv=[], program="test")


def test_psi_parameterization_defaults_to_ss_log_multiplier(tmp_path):
    config = _scope_config(tmp_path)
    assert config.search_config.psi_parameterization == "ss_log_multiplier"
    assert config.search_config.to_dict()["psi_parameterization"] == "ss_log_multiplier"


def test_psi_parameterization_flag_is_wired_through(tmp_path):
    config = _scope_config(tmp_path, "--optimize-psi", "--psi-parameterization", "absolute")
    assert config.search_config.optimize_psi
    assert config.search_config.psi_parameterization == "absolute"
    assert config.search_config.has_context_dependent_domain
    assert any("ss_log_multiplier" in warning for warning in config.warnings)

    stable = _scope_config(tmp_path, "--optimize-psi")
    assert stable.search_config.psi_parameterization == "ss_log_multiplier"
    assert not stable.search_config.has_context_dependent_domain
    assert not any("ss_log_multiplier" in warning for warning in stable.warnings)
