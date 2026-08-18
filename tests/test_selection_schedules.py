import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common_hpo.schedules import ScheduleError, SelectionSchedule


ORIGINS = [f"2000Q{i % 4 + 1}" for i in range(10)]


def test_once_selects_first_origin_only():
    events = SelectionSchedule.once().resolve(ORIGINS)
    assert len(events) == 1
    assert events[0].origin_index == 0
    assert events[0].applies_from_index == 0
    assert events[0].applies_to_index == len(ORIGINS) - 1


def test_every_origin_selects_all():
    events = SelectionSchedule.every_origin().resolve(ORIGINS)
    assert len(events) == len(ORIGINS)
    assert [e.origin_index for e in events] == list(range(len(ORIGINS)))
    assert all(e.applies_from_index == e.applies_to_index for e in events)


def test_every_n_origins_annual_quarterly():
    schedule = SelectionSchedule.annual_quarterly()
    events = schedule.resolve(ORIGINS)
    assert [e.origin_index for e in events] == [0, 4, 8]
    # Interpretation recorded (no hardcoded calendar).
    assert "every 4" in schedule.to_dict()["interpretation"]
    # Reuse spans cover all origins with no gaps.
    assert events[0].applies_to_index == 3
    assert events[1].applies_to_index == 7
    assert events[2].applies_to_index == 9


def test_every_n_origins_custom():
    events = SelectionSchedule.every_n_origins(3).resolve(ORIGINS)
    assert [e.origin_index for e in events] == [0, 3, 6, 9]


def test_explicit_indices_requires_zero():
    with pytest.raises(ScheduleError, match="must include index 0"):
        SelectionSchedule.explicit_indices([2, 5]).resolve(ORIGINS)


def test_explicit_indices_resolves_spans():
    events = SelectionSchedule.explicit_indices([0, 5]).resolve(ORIGINS)
    assert [e.origin_index for e in events] == [0, 5]
    assert events[0].applies_to_index == 4
    assert events[1].applies_to_index == 9


def test_explicit_labels_resolves():
    unique = [f"Y{i:02d}" for i in range(10)]
    events = SelectionSchedule.explicit_labels([unique[0], unique[4]]).resolve(unique)
    assert [e.origin_index for e in events] == [0, 4]


def test_explicit_indices_out_of_range():
    with pytest.raises(ScheduleError, match="out of range"):
        SelectionSchedule.explicit_indices([0, 99]).resolve(ORIGINS)


def test_event_for_origin_covers_reuse_span():
    schedule = SelectionSchedule.annual_quarterly()
    events = schedule.resolve(ORIGINS)
    assert schedule.event_for_origin(2, events).origin_index == 0
    assert schedule.event_for_origin(5, events).origin_index == 4
    assert schedule.event_for_origin(9, events).origin_index == 8


def test_schedule_round_trip():
    schedule = SelectionSchedule.every_n_origins(4)
    restored = SelectionSchedule.from_dict(schedule.to_dict())
    assert restored == schedule


def test_empty_origins_rejected():
    with pytest.raises(ScheduleError, match="empty origin sequence"):
        SelectionSchedule.once().resolve([])


def test_event_ids_are_stable_and_unique():
    events = SelectionSchedule.every_n_origins(4).resolve(ORIGINS)
    ids = [e.event_id for e in events]
    assert len(set(ids)) == len(ids)
    # Deterministic format.
    assert ids[0] == "sel-000-o0000"
    assert ids[1] == "sel-001-o0004"
