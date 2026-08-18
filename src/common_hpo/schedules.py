"""Model-independent hyperparameter-selection (retuning) schedules.

A :class:`SelectionSchedule` decides, given an ordered list of outer forecast
origins, at which origins hyperparameters are re-selected. Between selection
events the previously selected hyperparameters are reused.

The abstraction is deliberately calendar-agnostic. An "annual" quarterly
retuning cadence is expressed as ``every_n_origins(4)`` -- four outer forecast
origins -- rather than by hardcoding calendar assumptions. The chosen
interpretation is recorded in :meth:`SelectionSchedule.to_dict` so run metadata
captures exactly what "annual" meant for a given run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Literal, Mapping, Sequence


ScheduleKind = Literal[
    "once",
    "every_origin",
    "every_n_origins",
    "explicit_indices",
    "explicit_labels",
]


class ScheduleError(ValueError):
    """Raised when a selection schedule is invalid or cannot be resolved."""


def _as_index(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ScheduleError(f"{label} must be an integer.")
    return int(value)


@dataclass(frozen=True)
class SelectionEvent:
    """One resolved selection event over a concrete origin sequence.

    ``event_id`` is a stable, human-readable identifier. ``origin_index`` is the
    0-based position in the outer-origin sequence at which selection happens, and
    ``origin_label`` is the caller-supplied label for that origin (a date, an
    integer, etc.). ``applies_from_index`` / ``applies_to_index`` give the
    inclusive span of outer origins that reuse this event's hyperparameters.
    """

    event_id: str
    event_number: int
    origin_index: int
    origin_label: object
    applies_from_index: int
    applies_to_index: int

    def applies_to(self, origin_index: int) -> bool:
        """Return whether ``origin_index`` falls in this event's reuse span."""

        return self.applies_from_index <= origin_index <= self.applies_to_index

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "event_number": self.event_number,
            "origin_index": self.origin_index,
            "origin_label": self.origin_label,
            "applies_from_index": self.applies_from_index,
            "applies_to_index": self.applies_to_index,
        }


@dataclass(frozen=True)
class SelectionSchedule:
    """A calendar-agnostic hyperparameter re-selection cadence.

    Construct via the classmethods :meth:`once`, :meth:`every_origin`,
    :meth:`every_n_origins`, :meth:`explicit_indices`, or
    :meth:`explicit_labels`. Resolve against a concrete outer-origin sequence
    with :meth:`resolve`.
    """

    kind: ScheduleKind
    n: int | None = None
    indices: tuple[int, ...] | None = None
    labels: tuple[object, ...] | None = None
    interpretation: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in (
            "once",
            "every_origin",
            "every_n_origins",
            "explicit_indices",
            "explicit_labels",
        ):
            raise ScheduleError(f"unknown schedule kind {self.kind!r}.")

        if self.kind == "every_n_origins":
            if self.n is None:
                raise ScheduleError("every_n_origins requires n.")
            n = _as_index(self.n, label="n")
            if n <= 0:
                raise ScheduleError("every_n_origins requires a positive n.")
            object.__setattr__(self, "n", n)
        elif self.n is not None:
            raise ScheduleError("n is only valid for every_n_origins.")

        if self.kind == "explicit_indices":
            if not self.indices:
                raise ScheduleError("explicit_indices requires a non-empty indices sequence.")
            indices = tuple(_as_index(i, label="index") for i in self.indices)
            if any(i < 0 for i in indices):
                raise ScheduleError("explicit indices must be non-negative.")
            ordered = tuple(sorted(dict.fromkeys(indices)))
            object.__setattr__(self, "indices", ordered)
        elif self.indices is not None:
            raise ScheduleError("indices is only valid for explicit_indices.")

        if self.kind == "explicit_labels":
            if not self.labels:
                raise ScheduleError("explicit_labels requires a non-empty labels sequence.")
            object.__setattr__(self, "labels", tuple(self.labels))
        elif self.labels is not None:
            raise ScheduleError("labels is only valid for explicit_labels.")

    # -- constructors ------------------------------------------------------- #
    @classmethod
    def once(cls) -> "SelectionSchedule":
        """Select once, at the first outer origin, and reuse thereafter."""

        return cls(kind="once", interpretation="select once at the first origin")

    @classmethod
    def every_origin(cls) -> "SelectionSchedule":
        """Re-select at every outer origin."""

        return cls(kind="every_origin", interpretation="re-select at every origin")

    @classmethod
    def every_n_origins(cls, n: int, *, interpretation: str | None = None) -> "SelectionSchedule":
        """Re-select every ``n`` outer origins (annual quarterly cadence -> n=4)."""

        return cls(
            kind="every_n_origins",
            n=n,
            interpretation=interpretation or f"re-select every {int(n)} outer origins",
        )

    @classmethod
    def annual_quarterly(cls) -> "SelectionSchedule":
        """Annual retuning for a quarterly origin cadence: every four origins."""

        return cls(
            kind="every_n_origins",
            n=4,
            interpretation="annual cadence expressed as every 4 quarterly outer origins",
        )

    @classmethod
    def explicit_indices(
        cls, indices: Sequence[int], *, interpretation: str | None = None
    ) -> "SelectionSchedule":
        """Re-select at explicit 0-based outer-origin indices."""

        return cls(
            kind="explicit_indices",
            indices=tuple(indices),
            interpretation=interpretation or "re-select at explicit origin indices",
        )

    @classmethod
    def explicit_labels(
        cls, labels: Sequence[object], *, interpretation: str | None = None
    ) -> "SelectionSchedule":
        """Re-select at explicit outer-origin labels (e.g. dates)."""

        return cls(
            kind="explicit_labels",
            labels=tuple(labels),
            interpretation=interpretation or "re-select at explicit origin labels",
        )

    # -- resolution --------------------------------------------------------- #
    def _selection_indices(self, origin_labels: Sequence[object]) -> list[int]:
        n_origins = len(origin_labels)
        if n_origins == 0:
            raise ScheduleError("cannot resolve a schedule over an empty origin sequence.")

        if self.kind == "once":
            return [0]
        if self.kind == "every_origin":
            return list(range(n_origins))
        if self.kind == "every_n_origins":
            return list(range(0, n_origins, self.n))
        if self.kind == "explicit_indices":
            out_of_range = [i for i in self.indices if i >= n_origins]
            if out_of_range:
                raise ScheduleError(
                    f"explicit indices {out_of_range} are out of range for "
                    f"{n_origins} origins."
                )
            if 0 not in self.indices:
                raise ScheduleError(
                    "explicit_indices must include index 0 so the first origins "
                    "have selected hyperparameters."
                )
            return list(self.indices)
        # explicit_labels
        label_to_index: dict[object, int] = {}
        for index, label in enumerate(origin_labels):
            label_to_index.setdefault(label, index)
        resolved: list[int] = []
        for label in self.labels:
            if label not in label_to_index:
                raise ScheduleError(f"schedule label {label!r} is not an outer origin.")
            resolved.append(label_to_index[label])
        resolved = sorted(dict.fromkeys(resolved))
        if resolved[0] != 0:
            raise ScheduleError(
                "explicit_labels must include the first outer origin so early "
                "origins have selected hyperparameters."
            )
        return resolved

    def resolve(self, origin_labels: Sequence[object]) -> tuple[SelectionEvent, ...]:
        """Resolve to deterministic selection events over ``origin_labels``.

        Each event's reuse span runs from its own origin up to (but not
        including) the next event's origin.
        """

        origin_labels = list(origin_labels)
        indices = self._selection_indices(origin_labels)
        n_origins = len(origin_labels)

        events: list[SelectionEvent] = []
        for event_number, origin_index in enumerate(indices):
            next_origin = indices[event_number + 1] if event_number + 1 < len(indices) else n_origins
            applies_to = next_origin - 1
            events.append(
                SelectionEvent(
                    event_id=f"sel-{event_number:03d}-o{origin_index:04d}",
                    event_number=event_number,
                    origin_index=origin_index,
                    origin_label=origin_labels[origin_index],
                    applies_from_index=origin_index,
                    applies_to_index=applies_to,
                )
            )
        return tuple(events)

    def event_for_origin(
        self, origin_index: int, events: Sequence[SelectionEvent]
    ) -> SelectionEvent:
        """Return the selection event whose reuse span covers ``origin_index``."""

        for event in events:
            if event.applies_to(origin_index):
                return event
        raise ScheduleError(
            f"no selection event covers origin index {origin_index}; the schedule "
            "must select at or before the first origin."
        )

    def to_dict(self) -> dict[str, object]:
        """Return a stable serialized representation including the interpretation."""

        return {
            "kind": self.kind,
            "n": self.n,
            "indices": list(self.indices) if self.indices is not None else None,
            "labels": list(self.labels) if self.labels is not None else None,
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SelectionSchedule":
        return cls(
            kind=data["kind"],  # type: ignore[arg-type]
            n=data.get("n"),
            indices=tuple(data["indices"]) if data.get("indices") is not None else None,
            labels=tuple(data["labels"]) if data.get("labels") is not None else None,
            interpretation=data.get("interpretation"),
        )


__all__ = [
    "ScheduleError",
    "ScheduleKind",
    "SelectionEvent",
    "SelectionSchedule",
]
