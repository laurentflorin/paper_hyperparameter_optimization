"""Model-independent, leakage-safe inner-validation split engine.

This module builds inner cross-validation splits used to select hyperparameters
by pseudo-out-of-sample evaluation. It is deliberately independent of any
forecasting model, data frequency, GLP/MBFVAR code, Mango, pandas, or data
loading. The only third-party dependency is NumPy, used solely for reproducible
random origin selection.

Indexing convention (single, explicit, no hidden off-by-one)
------------------------------------------------------------
* Every position is a 0-based integer row index into the model's aligned data
  matrix. Row ``0`` is the earliest observation.
* A training window is the closed interval ``[train_start, train_end]``. Both
  endpoints are inclusive and denote rows actually used to fit the model.
* The pseudo-forecast origin ``origin`` is the most recent observation in the
  information set and, by convention, coincides with ``train_end`` (the
  inclusive-origin convention). The forecast is formed "standing at" ``origin``.
* For each canonical horizon ``h`` the target row is ``origin + offset`` where
  ``offset = horizon_row_offsets[h] >= 1``. Targets therefore always lie
  strictly after the origin, so no target can leak into its own training sample.
* The outer information-set cutoff ``info_cutoff`` is an inclusive upper bound:
  every target must satisfy ``target <= info_cutoff``.

The engine never learns whether one canonical horizon means a month, a quarter,
or anything else. Callers supply a model-specific mapping from each canonical
horizon to a number of data rows (``horizon_row_offsets``). A monthly model can
map ``h -> h``; a monthly-data/quarterly-horizon model can map ``h -> 3 * h``.

Vintage policy
--------------
``VintagePolicy.OUTER_VINTAGE_CONSISTENT`` (implemented) reuses the data of the
current outer-origin vintage for all inner splits.
``VintagePolicy.STRICT_INNER_REAL_TIME`` is a documented extension point only:
requesting it raises ``NotImplementedError`` because the per-origin historical
vintages are not loaded here. This module does not pretend to implement strict
real-time validation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral
from typing import Callable, Literal, Mapping, Sequence

import numpy as np


TrainingWindow = Literal["expanding", "rolling"]
OriginSelection = Literal["most_recent", "evenly_spaced", "random"]

HorizonRowOffsets = Mapping[int, int] | Callable[[int], int]


class InfeasibleValidationDesign(ValueError):
    """Raised when a requested validation design cannot be satisfied.

    The engine fails closed with this error rather than silently returning
    fewer splits than requested.
    """


class VintagePolicy(Enum):
    """How each inner split sees the data across vintages.

    ``OUTER_VINTAGE_CONSISTENT`` is implemented: every inner split uses the data
    contained in the current outer-origin vintage.

    ``STRICT_INNER_REAL_TIME`` is a reserved extension point: it would use the
    historical vintage available at each inner origin. It is not implemented in
    this module and selecting it raises ``NotImplementedError``.
    """

    OUTER_VINTAGE_CONSISTENT = "outer_vintage_consistent"
    STRICT_INNER_REAL_TIME = "strict_inner_real_time"

    @property
    def is_implemented(self) -> bool:
        return self is VintagePolicy.OUTER_VINTAGE_CONSISTENT

    def metadata(self) -> dict[str, object]:
        """Return serializable metadata describing this policy."""

        return {
            "policy": self.value,
            "implemented": self.is_implemented,
            "description": {
                VintagePolicy.OUTER_VINTAGE_CONSISTENT: (
                    "Use the data contained in the current outer-origin vintage "
                    "for all inner splits."
                ),
                VintagePolicy.STRICT_INNER_REAL_TIME: (
                    "Use the historical vintage available at each inner origin. "
                    "Extension point only; requires per-origin vintages that are "
                    "not loaded here."
                ),
            }[self],
        }


def _as_positive_int(value: object, *, label: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{label} must be a positive integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be a positive integer, got {result}.")
    return result


def _as_nonneg_int(value: object, *, label: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{label} must be a non-negative integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{label} must be a non-negative integer, got {result}.")
    return result


@dataclass(frozen=True)
class ValidationScheme:
    """Configuration describing how inner-validation splits are generated.

    Parameters
    ----------
    training_window:
        ``"expanding"`` grows the training window from row ``0``;
        ``"rolling"`` uses a fixed-length window of ``rolling_window_length``.
    origin_selection:
        ``"most_recent"`` takes the latest feasible origins; ``"evenly_spaced"``
        spreads origins across the feasible range; ``"random"`` draws origins
        reproducibly from a fixed ``random_seed``.
    n_origins:
        The exact number of pseudo-forecast origins to return.
    horizons:
        Canonical horizons to evaluate. These are opaque integers; their meaning
        in data rows is supplied separately via ``horizon_row_offsets``.
    min_train_length:
        Minimum number of training observations required at any origin.
    origin_stride:
        Spacing (in rows) of the candidate-origin grid. Defaults to ``1``.
    rolling_window_length:
        Fixed training length for ``"rolling"`` windows. Required and only valid
        for rolling windows.
    recency_decay:
        Optional decay factor in ``(0, 1]`` for exponentially decaying recency
        weights. ``None`` disables weighting (all ``origin_weight`` are ``None``).
    random_seed:
        Local seed for reproducible ``"random"`` origin selection.
    vintage_policy:
        Vintage handling; see :class:`VintagePolicy`.
    """

    training_window: TrainingWindow
    origin_selection: OriginSelection
    n_origins: int
    horizons: tuple[int, ...]
    min_train_length: int
    origin_stride: int = 1
    rolling_window_length: int | None = None
    recency_decay: float | None = None
    random_seed: int | None = None
    vintage_policy: VintagePolicy = VintagePolicy.OUTER_VINTAGE_CONSISTENT

    def __post_init__(self) -> None:
        if self.training_window not in ("expanding", "rolling"):
            raise ValueError(
                "training_window must be 'expanding' or 'rolling', got "
                f"{self.training_window!r}."
            )
        if self.origin_selection not in ("most_recent", "evenly_spaced", "random"):
            raise ValueError(
                "origin_selection must be 'most_recent', 'evenly_spaced', or "
                f"'random', got {self.origin_selection!r}."
            )

        object.__setattr__(self, "n_origins", _as_positive_int(self.n_origins, label="n_origins"))
        object.__setattr__(
            self,
            "min_train_length",
            _as_positive_int(self.min_train_length, label="min_train_length"),
        )
        object.__setattr__(
            self,
            "origin_stride",
            _as_positive_int(self.origin_stride, label="origin_stride"),
        )

        if not isinstance(self.horizons, Sequence) or isinstance(self.horizons, (str, bytes)):
            raise TypeError("horizons must be a sequence of positive integers.")
        horizons: list[int] = []
        seen: set[int] = set()
        for value in self.horizons:
            horizon = _as_positive_int(value, label="horizon")
            if horizon in seen:
                raise ValueError(f"horizons must be unique; found duplicate {horizon}.")
            horizons.append(horizon)
            seen.add(horizon)
        if not horizons:
            raise ValueError("horizons must be non-empty.")
        object.__setattr__(self, "horizons", tuple(horizons))

        if self.training_window == "rolling":
            if self.rolling_window_length is None:
                raise ValueError("rolling_window_length is required for rolling windows.")
            length = _as_positive_int(self.rolling_window_length, label="rolling_window_length")
            if length < self.min_train_length:
                raise ValueError(
                    "rolling_window_length must be at least min_train_length "
                    f"({length} < {self.min_train_length})."
                )
            object.__setattr__(self, "rolling_window_length", length)
        else:
            if self.rolling_window_length is not None:
                raise ValueError(
                    "rolling_window_length is only valid for rolling windows."
                )

        if self.recency_decay is not None:
            decay = float(self.recency_decay)
            if not (0.0 < decay <= 1.0):
                raise ValueError("recency_decay must lie in the interval (0, 1].")
            object.__setattr__(self, "recency_decay", decay)

        if self.random_seed is not None:
            object.__setattr__(
                self,
                "random_seed",
                _as_nonneg_int(self.random_seed, label="random_seed"),
            )

        if not isinstance(self.vintage_policy, VintagePolicy):
            raise TypeError("vintage_policy must be a VintagePolicy member.")

    def to_dict(self) -> dict[str, object]:
        """Return a stable serialized representation."""

        return {
            "training_window": self.training_window,
            "origin_selection": self.origin_selection,
            "n_origins": self.n_origins,
            "horizons": list(self.horizons),
            "min_train_length": self.min_train_length,
            "origin_stride": self.origin_stride,
            "rolling_window_length": self.rolling_window_length,
            "recency_decay": self.recency_decay,
            "random_seed": self.random_seed,
            "vintage_policy": self.vintage_policy.value,
        }


@dataclass(frozen=True)
class ValidationSplit:
    """One leakage-safe inner-validation split.

    See the module docstring for the indexing convention. ``origin`` coincides
    with ``train_end`` (inclusive-origin). ``targets`` maps each canonical
    horizon to its target row position, and every target lies strictly after the
    origin and at or before ``info_cutoff``.
    """

    split_id: str
    train_start: int
    train_end: int
    origin: int
    targets: tuple[tuple[int, int], ...]
    info_cutoff: int
    origin_weight: float | None = None
    date_labels: Mapping[str, object] | None = None

    def target_for(self, horizon: int) -> int:
        """Return the target row position for one canonical horizon."""

        for stored_horizon, position in self.targets:
            if stored_horizon == horizon:
                return position
        raise KeyError(f"Unknown horizon {horizon!r} for split {self.split_id!r}.")

    def training_positions(self) -> range:
        """Return the inclusive range of training row positions."""

        return range(self.train_start, self.train_end + 1)

    def to_dict(self) -> dict[str, object]:
        """Return a stable serialized representation."""

        return {
            "split_id": self.split_id,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "origin": self.origin,
            "targets": [list(pair) for pair in self.targets],
            "info_cutoff": self.info_cutoff,
            "origin_weight": self.origin_weight,
            "date_labels": dict(self.date_labels) if self.date_labels is not None else None,
        }


def resolve_horizon_offsets(
    horizons: Sequence[int],
    horizon_row_offsets: HorizonRowOffsets,
) -> dict[int, int]:
    """Resolve the model-specific canonical-horizon-to-rows mapping.

    ``horizon_row_offsets`` may be a mapping or a callable. Every resolved
    offset must be a positive integer number of rows.
    """

    resolved: dict[int, int] = {}
    for horizon in horizons:
        if callable(horizon_row_offsets):
            raw = horizon_row_offsets(horizon)
        else:
            if horizon not in horizon_row_offsets:
                raise KeyError(f"horizon_row_offsets is missing horizon {horizon!r}.")
            raw = horizon_row_offsets[horizon]
        resolved[horizon] = _as_positive_int(raw, label=f"row offset for horizon {horizon}")
    return resolved


def _candidate_origins(origin_min: int, origin_max: int, stride: int) -> list[int]:
    # Grid anchored at the most recent feasible origin so stride behavior is
    # stable regardless of how many origins are ultimately selected.
    descending = list(range(origin_max, origin_min - 1, -stride))
    return sorted(descending)


def _derive_local_rng(seed: int | None, rng: np.random.Generator | None) -> np.random.Generator:
    if seed is not None:
        return np.random.default_rng(seed)
    if rng is not None:
        # Read the generator state without advancing it (accessing the ``state``
        # property does not mutate the generator), then derive an independent,
        # reproducible local generator from a stable digest of that state.
        state_repr = json.dumps(rng.bit_generator.state, sort_keys=True, default=str)
        digest = hashlib.sha256(state_repr.encode("utf-8")).hexdigest()
        return np.random.default_rng(int(digest[:16], 16))
    raise ValueError("random origin selection requires a seed or a NumPy generator.")


def _select_origins(
    grid: list[int],
    scheme: ValidationScheme,
    rng: np.random.Generator | None,
) -> list[int]:
    available = len(grid)
    requested = scheme.n_origins
    if available < requested:
        raise InfeasibleValidationDesign(
            "requested more validation origins than are feasible: asked for "
            f"{requested} but only {available} candidate origins exist for this "
            "design."
        )

    if scheme.origin_selection == "most_recent":
        chosen = grid[available - requested:]
    elif scheme.origin_selection == "evenly_spaced":
        raw_indices = np.rint(np.linspace(0, available - 1, requested)).astype(int)
        unique_indices = sorted(dict.fromkeys(int(i) for i in raw_indices))
        if len(unique_indices) != requested:
            raise InfeasibleValidationDesign(
                "cannot place the requested number of evenly spaced origins "
                f"({requested}) within {available} candidate origins without "
                "collisions; widen the range or reduce n_origins."
            )
        chosen = [grid[i] for i in unique_indices]
    else:  # random
        local_rng = _derive_local_rng(scheme.random_seed, rng)
        picked = local_rng.choice(available, size=requested, replace=False)
        chosen = [grid[int(i)] for i in sorted(int(p) for p in picked)]

    return sorted(chosen)


def _recency_weights(origins: Sequence[int], decay: float | None) -> dict[int, float | None]:
    if decay is None:
        return {origin: None for origin in origins}
    # Most recent origin has rank 0 and the largest raw weight.
    descending = sorted(origins, reverse=True)
    raw = {origin: decay ** rank for rank, origin in enumerate(descending)}
    total = sum(raw.values())
    return {origin: raw[origin] / total for origin in origins}


def build_validation_splits(
    n_positions: int,
    scheme: ValidationScheme,
    horizon_row_offsets: HorizonRowOffsets,
    *,
    outer_info_cutoff: int | None = None,
    date_labels: Sequence[object] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[ValidationSplit, ...]:
    """Build deterministic, leakage-safe inner-validation splits.

    Parameters
    ----------
    n_positions:
        Number of aligned data rows available in the current outer vintage.
    scheme:
        The :class:`ValidationScheme` describing the design.
    horizon_row_offsets:
        Model-specific mapping (or callable) from each canonical horizon to a
        positive number of data rows.
    outer_info_cutoff:
        Inclusive upper-bound row for the outer information set. Defaults to the
        last available row (``n_positions - 1``).
    date_labels:
        Optional per-row labels (length ``>= n_positions``). When supplied, each
        split records the labels for its train start, origin, targets, and
        information cutoff.
    rng:
        Optional NumPy generator used only when ``origin_selection == "random"``
        and ``scheme.random_seed`` is ``None``. The generator is read without
        being advanced, so callers observe no mutation.

    Returns
    -------
    tuple[ValidationSplit, ...]
        Splits sorted deterministically by origin.

    Raises
    ------
    InfeasibleValidationDesign
        If the requested design cannot yield ``scheme.n_origins`` origins.
    NotImplementedError
        If ``scheme.vintage_policy`` is a reserved, unimplemented policy.
    """

    n_positions = _as_positive_int(n_positions, label="n_positions")

    if not scheme.vintage_policy.is_implemented:
        raise NotImplementedError(
            f"vintage policy {scheme.vintage_policy.value!r} is an extension point "
            "and is not implemented: strict inner real-time validation requires "
            "per-origin historical vintages that are not loaded here."
        )

    offsets = resolve_horizon_offsets(scheme.horizons, horizon_row_offsets)
    max_offset = max(offsets.values())

    if outer_info_cutoff is None:
        info_cutoff = n_positions - 1
    else:
        info_cutoff = _as_nonneg_int(outer_info_cutoff, label="outer_info_cutoff")
        if info_cutoff >= n_positions:
            raise ValueError(
                f"outer_info_cutoff ({info_cutoff}) must be < n_positions ({n_positions})."
            )

    if date_labels is not None and len(date_labels) < n_positions:
        raise ValueError(
            f"date_labels has length {len(date_labels)} but must cover all "
            f"{n_positions} positions."
        )

    origin_max = info_cutoff - max_offset
    if scheme.training_window == "rolling":
        origin_min = max(scheme.min_train_length, scheme.rolling_window_length) - 1
    else:
        origin_min = scheme.min_train_length - 1

    if origin_max < origin_min:
        raise InfeasibleValidationDesign(
            "no feasible validation origin exists: the earliest origin allowed by "
            f"the training requirements is {origin_min}, but the latest origin that "
            f"keeps every target at or before the information cutoff is {origin_max}."
        )

    grid = _candidate_origins(origin_min, origin_max, scheme.origin_stride)
    origins = _select_origins(grid, scheme, rng)
    weights = _recency_weights(origins, scheme.recency_decay)

    splits: list[ValidationSplit] = []
    for origin in origins:
        if scheme.training_window == "rolling":
            train_start = origin - scheme.rolling_window_length + 1
        else:
            train_start = 0

        targets = tuple((horizon, origin + offsets[horizon]) for horizon in scheme.horizons)

        labels: dict[str, object] | None = None
        if date_labels is not None:
            labels = {
                "train_start": date_labels[train_start],
                "origin": date_labels[origin],
                "info_cutoff": date_labels[info_cutoff],
                "targets": {horizon: date_labels[position] for horizon, position in targets},
            }

        splits.append(
            ValidationSplit(
                split_id=f"split-o{origin:05d}",
                train_start=train_start,
                train_end=origin,
                origin=origin,
                targets=targets,
                info_cutoff=info_cutoff,
                origin_weight=weights[origin],
                date_labels=labels,
            )
        )

    splits.sort(key=lambda split: (split.origin, split.split_id))
    return tuple(splits)


def assert_split_is_leakage_safe(
    split: ValidationSplit,
    horizon_row_offsets: Mapping[int, int],
) -> None:
    """Assert the core leakage-safety invariants for a single split.

    Checks that:

    * every training observation precedes the pseudo-forecast origin;
    * every validation target follows the origin by exactly the configured rows;
    * every validation target is available at or before the information cutoff;
    * no validation target enters its own training sample.
    """

    if split.train_start > split.train_end:
        raise AssertionError(
            f"split {split.split_id!r}: train_start ({split.train_start}) exceeds "
            f"train_end ({split.train_end})."
        )
    if split.origin != split.train_end:
        raise AssertionError(
            f"split {split.split_id!r}: origin ({split.origin}) must coincide with "
            f"train_end ({split.train_end}) under the inclusive-origin convention."
        )

    # Every training observation strictly precedes the origin except the origin
    # row itself, which is the boundary of the information set; no training row
    # may sit at or beyond the first target.
    for horizon, position in split.targets:
        expected_offset = horizon_row_offsets[horizon]
        actual_offset = position - split.origin
        if actual_offset != expected_offset:
            raise AssertionError(
                f"split {split.split_id!r}: target for horizon {horizon} follows the "
                f"origin by {actual_offset} rows but {expected_offset} were configured."
            )
        if position <= split.origin:
            raise AssertionError(
                f"split {split.split_id!r}: target for horizon {horizon} at position "
                f"{position} does not follow origin {split.origin}."
            )
        if position > split.info_cutoff:
            raise AssertionError(
                f"split {split.split_id!r}: target for horizon {horizon} at position "
                f"{position} lies beyond the information cutoff {split.info_cutoff}."
            )
        if split.train_start <= position <= split.train_end:
            raise AssertionError(
                f"split {split.split_id!r}: target for horizon {horizon} at position "
                f"{position} falls inside its own training window."
            )


def assert_rolling_window_length(
    splits: Sequence[ValidationSplit],
    rolling_window_length: int,
) -> None:
    """Assert every split has exactly the requested rolling training length."""

    for split in splits:
        length = split.train_end - split.train_start + 1
        if length != rolling_window_length:
            raise AssertionError(
                f"split {split.split_id!r}: rolling window length {length} does not "
                f"match the requested {rolling_window_length}."
            )


def assert_sorted_deterministically(splits: Sequence[ValidationSplit]) -> None:
    """Assert splits are sorted by ``(origin, split_id)``."""

    keys = [(split.origin, split.split_id) for split in splits]
    if keys != sorted(keys):
        raise AssertionError("splits are not sorted deterministically by origin.")


def verify_validation_splits(
    splits: Sequence[ValidationSplit],
    scheme: ValidationScheme,
    horizon_row_offsets: HorizonRowOffsets,
) -> None:
    """Run all leakage-safety and structural checks for a set of splits."""

    offsets = resolve_horizon_offsets(scheme.horizons, horizon_row_offsets)
    assert_sorted_deterministically(splits)
    if len(splits) != scheme.n_origins:
        raise AssertionError(
            f"expected {scheme.n_origins} splits but received {len(splits)}."
        )
    for split in splits:
        assert_split_is_leakage_safe(split, offsets)
    if scheme.training_window == "rolling":
        assert_rolling_window_length(splits, scheme.rolling_window_length)


__all__ = [
    "HorizonRowOffsets",
    "InfeasibleValidationDesign",
    "OriginSelection",
    "TrainingWindow",
    "ValidationScheme",
    "ValidationSplit",
    "VintagePolicy",
    "assert_rolling_window_length",
    "assert_sorted_deterministically",
    "assert_split_is_leakage_safe",
    "build_validation_splits",
    "resolve_horizon_offsets",
    "verify_validation_splits",
]
