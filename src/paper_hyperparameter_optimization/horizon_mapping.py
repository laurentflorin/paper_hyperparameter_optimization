"""Canonical horizon handling for the mixed-frequency (Schorfheide-Song) model.

User-facing forecast horizons are always expressed in **quarters**. The MBFVAR
state space evolves at the **monthly** frequency, so a quarterly horizon must be
converted to monthly state-space rows before it is handed to the model, and a
quarterly target date must be derived from the forecast origin.

This module is the single place that knows the monthly-to-quarterly frequency
ratio. The multiplication by three is defined once here (``FREQ_RATIO``) instead
of being spread as a hardcoded ``* 3`` through several forecasting functions.
All conversions are pure and calendar-correct across year and quarter
boundaries, so they can be unit-tested without loading any data or the model.
"""

from __future__ import annotations

from numbers import Integral

import pandas as pd

# The monthly-to-quarterly frequency ratio for the Schorfheide-Song model.
# Defined once so no other function hardcodes a bare ``* 3``.
FREQ_RATIO = 3


def _as_positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be a positive integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{label} must be a positive integer, got {result}.")
    return result


def quarterly_horizon_to_state_rows(horizon_quarters: int) -> int:
    """Return the number of monthly state-space rows for a quarterly horizon.

    A one-quarter horizon spans ``FREQ_RATIO`` monthly rows, a two-quarter
    horizon spans ``2 * FREQ_RATIO``, and so on.
    """

    return _as_positive_int(horizon_quarters, label="horizon_quarters") * FREQ_RATIO


def state_rows_for_max_horizon(max_horizon_quarters: int) -> int:
    """Return the monthly state-space rows needed to cover the longest horizon."""

    return quarterly_horizon_to_state_rows(max_horizon_quarters)


def required_forecast_months(max_horizon_quarters: int) -> int:
    """Return the minimum monthly forecast length covering ``max_horizon_quarters``."""

    return quarterly_horizon_to_state_rows(max_horizon_quarters)


def target_quarter_for_origin(
    origin: object,
    horizon_quarters: int,
) -> pd.Period:
    """Return the quarterly target period for ``horizon_quarters`` from ``origin``.

    ``origin`` may be any value pandas can interpret as a timestamp or quarterly
    period (a ``Timestamp``, ``Period``, or date string). Horizon ``1`` is the
    origin quarter itself (a nowcast of the current quarter), horizon ``2`` is
    the next quarter, and so on, so the offset is ``horizon - 1`` quarters. The
    arithmetic is period-based and therefore correct across year boundaries.
    """

    horizon = _as_positive_int(horizon_quarters, label="horizon_quarters")
    if isinstance(origin, pd.Period):
        origin_quarter = origin.asfreq("Q")
    else:
        origin_quarter = pd.Timestamp(origin).to_period("Q")
    return origin_quarter + (horizon - 1)


def target_quarters_for_origin(
    origin: object,
    horizons_quarters,
) -> tuple[pd.Period, ...]:
    """Return the ordered quarterly target periods for several horizons."""

    return tuple(
        target_quarter_for_origin(origin, horizon) for horizon in horizons_quarters
    )


__all__ = [
    "FREQ_RATIO",
    "quarterly_horizon_to_state_rows",
    "state_rows_for_max_horizon",
    "required_forecast_months",
    "target_quarter_for_origin",
    "target_quarters_for_origin",
]
