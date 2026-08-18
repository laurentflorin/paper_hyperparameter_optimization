"""Deterministic hyperparameter grids and tie-breaking for the ridge VAR.

This module defines the *scientific search* over the four ridge VAR dimensions
without any Bayesian optimization dependency:

* ``lambda`` -- ridge penalty strength (``lam`` in the estimator).
* ``p`` -- lag order.
* ``alpha`` -- lag-decay exponent applied as ``lag ** alpha``.
* ``kappa`` -- cross-variable penalty multiplier.

The default grid is a deterministic (staged) grid. Enumeration order is fully
documented and stable, and -- crucially -- the *winner* of a search never
depends on incidental dictionary or parallel-execution order. When two
candidates tie within a documented numerical tolerance the simpler model is
chosen by an explicit structural rule.

Definition of "simpler" (applied in this priority order):

1. Smaller lag order ``p``.
2. Stronger regularization (larger ``lambda``).
3. Fewer special penalty distinctions. A model with ``alpha == 0`` makes no
   lag-decay distinction and a model with ``kappa == 1`` makes no own/cross
   distinction; both are considered simpler. Ties are then broken by the
   smaller ``alpha`` and then by ``kappa`` closest to one.

All of the tie-breaking is a pure function of the candidate coordinates and the
set of losses, so serial and parallel evaluation yield identical selections.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Iterable, Mapping, Sequence


__all__ = [
    "RidgeGridSpec",
    "RidgeCandidate",
    "RidgeSelection",
    "default_grid_spec",
    "enumerate_grid",
    "grid_size",
    "select_best_candidate",
    "DEFAULT_TIE_TOLERANCE",
]


# Documented default numerical tolerance for treating two losses as tied.
# Selection prefers the simpler model when losses fall within this absolute band
# of the best observed loss.
DEFAULT_TIE_TOLERANCE = 1e-9


def _as_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number, got {value!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {result!r}.")
    return result


def _as_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer, got {value!r}.")
    return int(value)


@dataclass(frozen=True)
class RidgeCandidate:
    """One point in the ridge VAR hyperparameter grid."""

    lam: float
    p: int
    alpha: float
    kappa: float

    def __post_init__(self) -> None:
        lam = _as_float(self.lam, label="lam")
        alpha = _as_float(self.alpha, label="alpha")
        kappa = _as_float(self.kappa, label="kappa")
        p = _as_int(self.p, label="p")
        if lam < 0.0:
            raise ValueError("lam must be non-negative.")
        if alpha < 0.0:
            raise ValueError("alpha must be non-negative.")
        if kappa <= 0.0:
            raise ValueError("kappa must be strictly positive.")
        if p < 1:
            raise ValueError("p must be a positive integer.")
        object.__setattr__(self, "lam", lam)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "kappa", kappa)
        object.__setattr__(self, "p", p)

    def to_dict(self) -> dict[str, float | int]:
        return {"lam": self.lam, "p": self.p, "alpha": self.alpha, "kappa": self.kappa}


@dataclass(frozen=True)
class RidgeGridSpec:
    """The finite search space over the four ridge VAR dimensions."""

    lambdas: tuple[float, ...]
    lag_orders: tuple[int, ...]
    alphas: tuple[float, ...]
    kappas: tuple[float, ...]

    def __post_init__(self) -> None:
        lambdas = tuple(_as_float(v, label="lambda") for v in self.lambdas)
        lag_orders = tuple(_as_int(v, label="p") for v in self.lag_orders)
        alphas = tuple(_as_float(v, label="alpha") for v in self.alphas)
        kappas = tuple(_as_float(v, label="kappa") for v in self.kappas)

        if not lambdas:
            raise ValueError("lambdas must be non-empty.")
        if not lag_orders:
            raise ValueError("lag_orders must be non-empty.")
        if not alphas:
            raise ValueError("alphas must be non-empty.")
        if not kappas:
            raise ValueError("kappas must be non-empty.")

        if any(v < 0.0 for v in lambdas):
            raise ValueError("lambdas must be non-negative.")
        if any(v < 1 for v in lag_orders):
            raise ValueError("lag_orders must be positive integers.")
        if any(v < 0.0 for v in alphas):
            raise ValueError("alphas must be non-negative.")
        if any(v <= 0.0 for v in kappas):
            raise ValueError("kappas must be strictly positive.")

        # Deduplicate while keeping a canonical ascending order so enumeration is
        # reproducible regardless of how the caller supplied the values.
        object.__setattr__(self, "lambdas", tuple(sorted(dict.fromkeys(lambdas))))
        object.__setattr__(self, "lag_orders", tuple(sorted(dict.fromkeys(lag_orders))))
        object.__setattr__(self, "alphas", tuple(sorted(dict.fromkeys(alphas))))
        object.__setattr__(self, "kappas", tuple(sorted(dict.fromkeys(kappas))))

    def to_dict(self) -> dict[str, list[float | int]]:
        return {
            "lambdas": list(self.lambdas),
            "lag_orders": list(self.lag_orders),
            "alphas": list(self.alphas),
            "kappas": list(self.kappas),
        }


def default_grid_spec() -> RidgeGridSpec:
    """Return the documented default staged grid.

    * ``lambda`` -- a logarithmic grid that explicitly includes ``0.0`` (the
      unpenalized OLS VAR) alongside a decade ladder.
    * ``p`` -- a small finite set of lag orders.
    * ``alpha`` -- a small documented set of lag-decay exponents (``0`` disables
      lag decay).
    * ``kappa`` -- a small documented set of cross-variable multipliers (``1``
      disables the own/cross distinction).
    """

    return RidgeGridSpec(
        lambdas=(0.0, 1e-2, 1e-1, 1.0, 10.0, 100.0),
        lag_orders=(1, 2, 4),
        alphas=(0.0, 1.0, 2.0),
        kappas=(1.0, 2.0),
    )


def grid_size(spec: RidgeGridSpec) -> int:
    """Return the number of candidates the grid enumerates."""

    return (
        len(spec.lambdas)
        * len(spec.lag_orders)
        * len(spec.alphas)
        * len(spec.kappas)
    )


def enumerate_grid(spec: RidgeGridSpec) -> tuple[RidgeCandidate, ...]:
    """Enumerate every grid candidate in a stable, documented order.

    The nested iteration order is ``p`` (outermost) then ``lambda`` then
    ``alpha`` then ``kappa`` (innermost), each ascending. This order is a pure
    convenience for humans reading logs; the selected winner does not depend on
    it (see :func:`select_best_candidate`).
    """

    candidates: list[RidgeCandidate] = []
    for p in spec.lag_orders:
        for lam in spec.lambdas:
            for alpha in spec.alphas:
                for kappa in spec.kappas:
                    candidates.append(
                        RidgeCandidate(lam=lam, p=p, alpha=alpha, kappa=kappa)
                    )
    return tuple(candidates)


def _simplicity_key(candidate: RidgeCandidate) -> tuple[float, ...]:
    """Return the structural simplicity key (smaller is simpler).

    Priority order matches the documented definition of "simpler":

    1. Smaller lag order ``p``.
    2. Stronger regularization -> larger ``lambda`` -> sort by ``-lambda``.
    3. Fewer special penalty distinctions -> count of active distinctions
       (``alpha != 0`` and ``kappa != 1``), then smaller ``alpha``, then
       ``kappa`` closest to one, then raw ``kappa`` for total determinism.
    """

    n_distinctions = int(candidate.alpha != 0.0) + int(candidate.kappa != 1.0)
    return (
        float(candidate.p),
        -candidate.lam,
        float(n_distinctions),
        candidate.alpha,
        abs(candidate.kappa - 1.0),
        candidate.kappa,
    )


@dataclass(frozen=True)
class RidgeSelection:
    """The outcome of a deterministic grid search for one selection cell."""

    candidate: RidgeCandidate
    loss: float
    n_candidates: int
    n_tied: int
    tolerance: float

    def to_dict(self) -> dict[str, object]:
        return {
            **{f"param_{k}": v for k, v in self.candidate.to_dict().items()},
            "selection_loss": self.loss,
            "n_candidates": self.n_candidates,
            "n_tied": self.n_tied,
            "tie_tolerance": self.tolerance,
        }


def select_best_candidate(
    evaluated: Iterable[tuple[RidgeCandidate, float]]
    | Mapping[RidgeCandidate, float]
    | Sequence[tuple[RidgeCandidate, float]],
    *,
    tolerance: float = DEFAULT_TIE_TOLERANCE,
) -> RidgeSelection:
    """Pick the best candidate with deterministic, order-independent tie-breaking.

    ``evaluated`` maps (or pairs) each candidate to its scalar validation loss.
    The candidate with the smallest loss wins; any candidate whose loss is within
    ``tolerance`` of the best loss is considered tied, and among the tied
    candidates the structurally simplest one (see :func:`_simplicity_key`) is
    selected. The result is invariant to the iteration order of ``evaluated``.
    """

    tol = _as_float(tolerance, label="tolerance")
    if tol < 0.0:
        raise ValueError("tolerance must be non-negative.")

    if isinstance(evaluated, Mapping):
        items = list(evaluated.items())
    else:
        items = list(evaluated)
    if not items:
        raise ValueError("at least one evaluated candidate is required.")

    losses: list[float] = []
    for candidate, loss in items:
        if not isinstance(candidate, RidgeCandidate):
            raise TypeError("each evaluated entry must pair a RidgeCandidate with a loss.")
        losses.append(_as_float(loss, label="loss"))

    best_loss = min(losses)
    tied = [
        candidate
        for (candidate, _), loss in zip(items, losses)
        if loss <= best_loss + tol
    ]
    winner = min(tied, key=_simplicity_key)
    # Report the winner's own loss (the loss stored against the chosen candidate)
    # rather than the group minimum, so diagnostics reflect the selected model.
    winner_loss = next(
        loss for (candidate, _), loss in zip(items, losses) if candidate == winner
    )
    return RidgeSelection(
        candidate=winner,
        loss=winner_loss,
        n_candidates=len(items),
        n_tied=len(tied),
        tolerance=tol,
    )
