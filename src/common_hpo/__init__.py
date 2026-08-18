"""Model-independent hyperparameter-selection planning utilities.

The public API in this package describes how forecast targets are grouped into
selection cells. Each cell stands for one system-wide hyperparameter vector
shared by all targets assigned to that cell.
"""

from .selection_scope import SelectionPlan, TargetCell, TargetKey, build_selection_plan
from .splits import (
    InfeasibleValidationDesign,
    ValidationScheme,
    ValidationSplit,
    VintagePolicy,
    assert_rolling_window_length,
    assert_sorted_deterministically,
    assert_split_is_leakage_safe,
    build_validation_splits,
    resolve_horizon_offsets,
    verify_validation_splits,
)

__all__ = [
    "TargetKey",
    "TargetCell",
    "SelectionPlan",
    "build_selection_plan",
    "InfeasibleValidationDesign",
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