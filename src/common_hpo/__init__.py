"""Model-independent hyperparameter-selection planning utilities.

The public API in this package describes how forecast targets are grouped into
selection cells. Each cell stands for one system-wide hyperparameter vector
shared by all targets assigned to that cell.
"""

from .selection_scope import SelectionPlan, TargetCell, TargetKey, build_selection_plan

__all__ = ["TargetKey", "TargetCell", "SelectionPlan", "build_selection_plan"]