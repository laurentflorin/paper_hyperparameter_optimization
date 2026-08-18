"""Tests for the deterministic ridge VAR grid and tie-breaking."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from regularized_var.tuning import (
    RidgeCandidate,
    RidgeGridSpec,
    default_grid_spec,
    enumerate_grid,
    grid_size,
    select_best_candidate,
)


def test_exact_grid_enumeration_order_and_size():
    spec = RidgeGridSpec(lambdas=(1.0, 0.0), lag_orders=(2, 1), alphas=(1.0, 0.0), kappas=(1.0,))
    candidates = enumerate_grid(spec)
    assert grid_size(spec) == len(candidates) == 2 * 2 * 2 * 1
    # Enumeration is p (outer) -> lambda -> alpha -> kappa, each ascending, after
    # the spec canonicalizes each axis to ascending unique order.
    coords = [(c.p, c.lam, c.alpha, c.kappa) for c in candidates]
    assert coords == [
        (1, 0.0, 0.0, 1.0),
        (1, 0.0, 1.0, 1.0),
        (1, 1.0, 0.0, 1.0),
        (1, 1.0, 1.0, 1.0),
        (2, 0.0, 0.0, 1.0),
        (2, 0.0, 1.0, 1.0),
        (2, 1.0, 0.0, 1.0),
        (2, 1.0, 1.0, 1.0),
    ]


def test_default_grid_includes_zero_lambda():
    spec = default_grid_spec()
    assert 0.0 in spec.lambdas
    assert spec.lag_orders[0] == 1
    assert 0.0 in spec.alphas
    assert 1.0 in spec.kappas


def test_grid_spec_deduplicates_and_sorts():
    spec = RidgeGridSpec(lambdas=(1.0, 1.0, 0.0), lag_orders=(2, 1, 2), alphas=(0.0,), kappas=(2.0, 1.0))
    assert spec.lambdas == (0.0, 1.0)
    assert spec.lag_orders == (1, 2)
    assert spec.kappas == (1.0, 2.0)


def test_tie_break_prefers_smaller_lag_order():
    a = RidgeCandidate(lam=1.0, p=1, alpha=0.0, kappa=1.0)
    b = RidgeCandidate(lam=1.0, p=2, alpha=0.0, kappa=1.0)
    result = select_best_candidate([(b, 1.0), (a, 1.0)])
    assert result.candidate == a
    assert result.n_tied == 2


def test_tie_break_prefers_stronger_regularization():
    weak = RidgeCandidate(lam=0.0, p=1, alpha=0.0, kappa=1.0)
    strong = RidgeCandidate(lam=10.0, p=1, alpha=0.0, kappa=1.0)
    result = select_best_candidate([(weak, 2.0), (strong, 2.0)])
    assert result.candidate == strong


def test_tie_break_prefers_fewer_penalty_distinctions():
    plain = RidgeCandidate(lam=1.0, p=1, alpha=0.0, kappa=1.0)
    lag_decay = RidgeCandidate(lam=1.0, p=1, alpha=2.0, kappa=1.0)
    cross = RidgeCandidate(lam=1.0, p=1, alpha=0.0, kappa=3.0)
    result = select_best_candidate([(lag_decay, 1.0), (cross, 1.0), (plain, 1.0)])
    assert result.candidate == plain


def test_tie_break_uses_tolerance_band():
    simple = RidgeCandidate(lam=1.0, p=1, alpha=0.0, kappa=1.0)
    complex_but_slightly_better = RidgeCandidate(lam=0.0, p=4, alpha=2.0, kappa=3.0)
    # The complex model is better by less than the tolerance, so the simpler
    # model wins.
    result = select_best_candidate(
        [(complex_but_slightly_better, 1.0 - 1e-10), (simple, 1.0)], tolerance=1e-9
    )
    assert result.candidate == simple


def test_tie_break_is_order_independent():
    candidates = [
        (RidgeCandidate(lam=0.0, p=2, alpha=1.0, kappa=2.0), 1.0),
        (RidgeCandidate(lam=1.0, p=1, alpha=0.0, kappa=1.0), 1.0),
        (RidgeCandidate(lam=10.0, p=1, alpha=0.0, kappa=1.0), 1.0),
    ]
    winners = set()
    import itertools

    for permutation in itertools.permutations(candidates):
        winners.add(select_best_candidate(list(permutation)).candidate)
    assert len(winners) == 1
    winner = winners.pop()
    assert winner == RidgeCandidate(lam=10.0, p=1, alpha=0.0, kappa=1.0)


def test_strict_best_loss_wins_outside_tolerance():
    good = RidgeCandidate(lam=0.0, p=4, alpha=2.0, kappa=3.0)
    worse_but_simple = RidgeCandidate(lam=1.0, p=1, alpha=0.0, kappa=1.0)
    result = select_best_candidate([(good, 0.5), (worse_but_simple, 1.0)], tolerance=1e-9)
    assert result.candidate == good


def test_invalid_candidate_parameters():
    with pytest.raises(ValueError):
        RidgeCandidate(lam=-1.0, p=1, alpha=0.0, kappa=1.0)
    with pytest.raises(ValueError):
        RidgeCandidate(lam=0.0, p=0, alpha=0.0, kappa=1.0)
    with pytest.raises(ValueError):
        RidgeCandidate(lam=0.0, p=1, alpha=0.0, kappa=0.0)


def test_empty_evaluation_rejected():
    with pytest.raises(ValueError):
        select_best_candidate([])
