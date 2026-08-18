"""Selection-scope planning for model-independent hyperparameter selection.

This module is intentionally independent of pandas, GLP, MBFVAR, Mango, and
all data-loading code. It only describes how forecast targets are partitioned
into selection cells.

In this vocabulary, a "variable-specific" scope means selecting a separate
system-wide hyperparameter vector for each forecast target variable. It does
not mean estimating isolated univariate models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from re import sub
from types import MappingProxyType
from typing import Iterable, Literal, Mapping, Sequence


SelectionScope = Literal[
    "pooled",
    "horizon",
    "variable",
    "variable_horizon",
    "group",
]

SUPPORTED_SCOPES: tuple[str, ...] = (
    "pooled",
    "horizon",
    "variable",
    "variable_horizon",
    "group",
)


def _normalize_label(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} must be a non-empty string.")
    return normalized


def _normalize_scope(scope: object) -> str:
    normalized = _normalize_label(scope, label="scope").lower()
    if normalized not in SUPPORTED_SCOPES:
        allowed = ", ".join(SUPPORTED_SCOPES)
        raise ValueError(f"scope must be one of {allowed}, got {scope!r}.")
    return normalized


def _normalize_variables(values: object, *, label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must be a sequence of strings, not a scalar string.")
    try:
        iterator = iter(values)
    except TypeError as exc:  # pragma: no cover - defensive
        raise TypeError(f"{label} must be an iterable of strings.") from exc

    normalized: list[str] = []
    seen: set[str] = set()
    for value in iterator:
        variable = _normalize_label(value, label=label[:-1] if label.endswith("s") else label)
        if variable in seen:
            raise ValueError(f"{label} must be unique; found duplicate {variable!r}.")
        normalized.append(variable)
        seen.add(variable)

    if not normalized:
        raise ValueError(f"{label} must be non-empty.")
    return tuple(normalized)


def _normalize_horizons(values: object, *, label: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must be a sequence of positive integers, not a scalar string.")
    try:
        iterator = iter(values)
    except TypeError as exc:  # pragma: no cover - defensive
        raise TypeError(f"{label} must be an iterable of positive integers.") from exc

    normalized: list[int] = []
    seen: set[int] = set()
    for value in iterator:
        if not isinstance(value, Integral) or isinstance(value, bool):
            raise TypeError(f"{label} entries must be positive integers.")
        horizon = int(value)
        if horizon <= 0:
            raise ValueError(f"{label} entries must be positive integers.")
        if horizon in seen:
            raise ValueError(f"{label} must be unique; found duplicate {horizon!r}.")
        normalized.append(horizon)
        seen.add(horizon)

    if not normalized:
        raise ValueError(f"{label} must be non-empty.")
    return tuple(normalized)


def _slugify(label: str) -> str:
    slug = sub(r"[^a-z0-9]+", "-", label.strip().lower()).strip("-")
    if not slug:
        raise ValueError(f"Could not generate a stable slug for {label!r}.")
    return slug


def _iter_group_items(
    variable_groups: Mapping[str, Sequence[str]] | Iterable[tuple[str, Sequence[str]]],
) -> list[tuple[str, Sequence[str]]]:
    if isinstance(variable_groups, Mapping):
        return list(variable_groups.items())
    return list(variable_groups)


def _normalize_group_definitions(
    variable_groups: Mapping[str, Sequence[str]] | Iterable[tuple[str, Sequence[str]]] | None,
    *,
    target_variables: tuple[str, ...],
    residual_group_name: str | None,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if variable_groups is None:
        items: list[tuple[str, Sequence[str]]] = []
    else:
        items = _iter_group_items(variable_groups)

    target_set = set(target_variables)
    normalized: list[tuple[str, tuple[str, ...]]] = []
    seen_names: set[str] = set()
    seen_variables: dict[str, str] = {}

    for raw_name, raw_variables in items:
        name = _normalize_label(raw_name, label="group name")
        if name in seen_names:
            raise ValueError(f"group names must be unique; found duplicate {name!r}.")
        group_variables = _normalize_variables(
            raw_variables,
            label=f"variables for group {name!r}",
        )
        unknown = [variable for variable in group_variables if variable not in target_set]
        if unknown:
            raise ValueError(
                f"group {name!r} references unknown requested variables: {unknown}."
            )
        overlaps = [
            variable
            for variable in group_variables
            if variable in seen_variables
        ]
        if overlaps:
            details = ", ".join(
                f"{variable!r} in {seen_variables[variable]!r} and {name!r}"
                for variable in overlaps
            )
            raise ValueError(f"group definitions may not overlap: {details}.")

        ordered_variables = tuple(
            variable for variable in target_variables if variable in set(group_variables)
        )
        normalized.append((name, ordered_variables))
        seen_names.add(name)
        for variable in ordered_variables:
            seen_variables[variable] = name

    missing = [variable for variable in target_variables if variable not in seen_variables]
    if missing:
        if residual_group_name is None:
            raise ValueError(
                "group definitions must cover every requested variable unless a residual "
                f"group is requested; missing {missing}."
            )
        residual_name = _normalize_label(residual_group_name, label="residual_group_name")
        if residual_name in seen_names:
            raise ValueError(
                f"residual_group_name {residual_name!r} conflicts with a supplied group name."
            )
        normalized.append((residual_name, tuple(missing)))

    if not normalized:
        raise ValueError("group scope requires at least one explicit or residual group.")
    return tuple(normalized)


@dataclass(frozen=True, order=True)
class TargetKey:
    """A single forecast target identified by variable name and horizon.

    The horizon is an application-defined positive integer. The structure is
    immutable and suitable for stable serialization and hashing.
    """

    variable: str
    horizon: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "variable", _normalize_label(self.variable, label="variable"))
        if not isinstance(self.horizon, Integral) or isinstance(self.horizon, bool):
            raise TypeError("horizon must be a positive integer.")
        horizon = int(self.horizon)
        if horizon <= 0:
            raise ValueError("horizon must be a positive integer.")
        object.__setattr__(self, "horizon", horizon)

    def to_dict(self) -> dict[str, object]:
        """Return a stable plain-Python representation."""

        return {"variable": self.variable, "horizon": self.horizon}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "TargetKey":
        """Reconstruct a key from its serialized representation."""

        return cls(variable=data["variable"], horizon=data["horizon"])


@dataclass(frozen=True)
class TargetCell:
    """A selection cell covering one or more target keys.

    A cell defines the set of target variables and horizons that share one
    system-wide hyperparameter vector. It does not describe estimation of a
    reduced or univariate model.
    """

    cell_id: str
    variables: tuple[str, ...]
    horizons: tuple[int, ...]
    group_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "cell_id", _normalize_label(self.cell_id, label="cell_id"))
        object.__setattr__(self, "variables", _normalize_variables(self.variables, label="variables"))
        object.__setattr__(self, "horizons", _normalize_horizons(self.horizons, label="horizons"))
        if self.group_name is not None:
            object.__setattr__(
                self,
                "group_name",
                _normalize_label(self.group_name, label="group_name"),
            )

    def targets(self) -> tuple[TargetKey, ...]:
        """Expand this cell to its ordered target-key cross product."""

        return tuple(
            TargetKey(variable=variable, horizon=horizon)
            for variable in self.variables
            for horizon in self.horizons
        )

    def to_dict(self) -> dict[str, object]:
        """Return a stable plain-Python representation."""

        return {
            "cell_id": self.cell_id,
            "variables": list(self.variables),
            "horizons": list(self.horizons),
            "group_name": self.group_name,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "TargetCell":
        """Reconstruct a cell from its serialized representation."""

        return cls(
            cell_id=data["cell_id"],
            variables=tuple(data["variables"]),
            horizons=tuple(data["horizons"]),
            group_name=data.get("group_name"),
        )


@dataclass(frozen=True)
class SelectionPlan:
    """An immutable mapping from forecast targets to selection cells.

    The plan stores the requested target universe and a deterministic partition
    of that universe into cells. Every requested `(variable, horizon)` pair must
    map to exactly one cell.
    """

    scope: SelectionScope | str
    target_variables: tuple[str, ...]
    target_horizons: tuple[int, ...]
    cells: tuple[TargetCell, ...]
    _cell_by_id: Mapping[str, TargetCell] = field(init=False, repr=False, compare=False)
    _cell_targets: Mapping[str, tuple[TargetKey, ...]] = field(init=False, repr=False, compare=False)
    _target_to_cell_id: Mapping[TargetKey, str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        scope = _normalize_scope(self.scope)
        target_variables = _normalize_variables(self.target_variables, label="target_variables")
        target_horizons = _normalize_horizons(self.target_horizons, label="target_horizons")

        if isinstance(self.cells, TargetCell):
            raise TypeError("cells must be a sequence of TargetCell instances, not a single cell.")
        cells = tuple(self.cells)
        if not cells:
            raise ValueError("cells must be non-empty.")
        if any(not isinstance(cell, TargetCell) for cell in cells):
            raise TypeError("cells must contain only TargetCell instances.")

        requested_targets = tuple(
            TargetKey(variable=variable, horizon=horizon)
            for variable in target_variables
            for horizon in target_horizons
        )
        target_variable_set = set(target_variables)
        target_horizon_set = set(target_horizons)
        requested_target_set = set(requested_targets)
        cell_by_id: dict[str, TargetCell] = {}
        cell_targets: dict[str, tuple[TargetKey, ...]] = {}
        target_to_cell_id: dict[TargetKey, str] = {}

        for cell in cells:
            if cell.cell_id in cell_by_id:
                raise ValueError(f"cell_id values must be unique; found duplicate {cell.cell_id!r}.")
            unknown_variables = [variable for variable in cell.variables if variable not in target_variable_set]
            if unknown_variables:
                raise ValueError(
                    f"cell {cell.cell_id!r} references unknown target variables: {unknown_variables}."
                )
            unknown_horizons = [horizon for horizon in cell.horizons if horizon not in target_horizon_set]
            if unknown_horizons:
                raise ValueError(
                    f"cell {cell.cell_id!r} references unknown target horizons: {unknown_horizons}."
                )

            targets = cell.targets()
            for target in targets:
                if target not in requested_target_set:
                    raise ValueError(
                        f"cell {cell.cell_id!r} references target outside the requested universe: {target}."
                    )
                if target in target_to_cell_id:
                    previous_cell_id = target_to_cell_id[target]
                    raise ValueError(
                        "every requested target pair must map to exactly one cell; "
                        f"{target.variable!r} at horizon {target.horizon} maps to both "
                        f"{previous_cell_id!r} and {cell.cell_id!r}."
                    )
                target_to_cell_id[target] = cell.cell_id

            cell_by_id[cell.cell_id] = cell
            cell_targets[cell.cell_id] = targets

        missing_targets = [target for target in requested_targets if target not in target_to_cell_id]
        if missing_targets:
            details = ", ".join(f"{target.variable}@h{target.horizon}" for target in missing_targets)
            raise ValueError(
                "every requested target pair must map to exactly one cell; missing target mappings for "
                f"{details}."
            )

        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "target_variables", target_variables)
        object.__setattr__(self, "target_horizons", target_horizons)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "_cell_by_id", MappingProxyType(cell_by_id))
        object.__setattr__(self, "_cell_targets", MappingProxyType(cell_targets))
        object.__setattr__(self, "_target_to_cell_id", MappingProxyType(target_to_cell_id))

    def cell_for(self, variable: str, horizon: int) -> TargetCell:
        """Return the cell covering one requested target pair.

        The lookup only resolves targets already declared in the plan. Unknown
        variables or horizons raise `KeyError` with a descriptive message.
        """

        normalized_variable = _normalize_label(variable, label="variable")
        if normalized_variable not in self.target_variables:
            raise KeyError(f"Unknown variable {normalized_variable!r}.")
        if not isinstance(horizon, Integral) or isinstance(horizon, bool):
            raise KeyError(f"Unknown horizon {horizon!r}.")
        normalized_horizon = int(horizon)
        if normalized_horizon not in self.target_horizons:
            raise KeyError(f"Unknown horizon {normalized_horizon!r}.")

        target = TargetKey(variable=normalized_variable, horizon=normalized_horizon)
        cell_id = self._target_to_cell_id[target]
        return self._cell_by_id[cell_id]

    def targets_for(self, cell_id: str) -> tuple[TargetKey, ...]:
        """Return the ordered target keys covered by one cell."""

        normalized_cell_id = _normalize_label(cell_id, label="cell_id")
        if normalized_cell_id not in self._cell_targets:
            raise KeyError(f"Unknown cell_id {normalized_cell_id!r}.")
        return self._cell_targets[normalized_cell_id]

    def to_dict(self) -> dict[str, object]:
        """Return a stable serialized representation.

        The returned dictionary preserves insertion order and only contains
        built-in Python types so callers can hash or JSON-serialize it for run
        metadata.
        """

        return {
            "scope": self.scope,
            "target_variables": list(self.target_variables),
            "target_horizons": list(self.target_horizons),
            "cells": [cell.to_dict() for cell in self.cells],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SelectionPlan":
        """Reconstruct a plan from `to_dict()` output."""

        if not isinstance(data, Mapping):
            raise TypeError("SelectionPlan.from_dict expects a mapping.")
        cells_raw = data["cells"]
        if isinstance(cells_raw, (str, bytes)):
            raise TypeError("cells must be a sequence of mappings, not a scalar string.")
        cells = tuple(TargetCell.from_dict(cell_data) for cell_data in cells_raw)
        return cls(
            scope=data["scope"],
            target_variables=tuple(data["target_variables"]),
            target_horizons=tuple(data["target_horizons"]),
            cells=cells,
        )


def build_selection_plan(
    scope: SelectionScope | str,
    target_variables: Sequence[str],
    target_horizons: Sequence[int],
    *,
    variable_groups: Mapping[str, Sequence[str]] | Iterable[tuple[str, Sequence[str]]] | None = None,
    separate_group_horizons: bool = False,
    residual_group_name: str | None = None,
) -> SelectionPlan:
    """Build a deterministic `SelectionPlan` for one target universe.

    Scope semantics:

    - `pooled`: one cell covers every requested variable and horizon.
    - `horizon`: one cell per horizon, pooled across all requested variables.
    - `variable`: one cell per variable, pooled across all requested horizons.
    - `variable_horizon`: one cell per `(variable, horizon)` pair.
    - `group`: one cell per supplied variable group, optionally split by horizon.

    A variable-oriented scope still means selecting a separate system-wide
    hyperparameter vector for each target variable. It does not imply fitting
    an isolated univariate model.
    """

    normalized_scope = _normalize_scope(scope)
    normalized_variables = _normalize_variables(target_variables, label="target_variables")
    normalized_horizons = _normalize_horizons(target_horizons, label="target_horizons")

    if normalized_scope != "group":
        if variable_groups is not None:
            raise ValueError("variable_groups is only valid for scope='group'.")
        if residual_group_name is not None:
            raise ValueError("residual_group_name is only valid for scope='group'.")
        if separate_group_horizons:
            raise ValueError("separate_group_horizons is only valid for scope='group'.")

    cells: tuple[TargetCell, ...]
    if normalized_scope == "pooled":
        cells = (TargetCell("pooled", normalized_variables, normalized_horizons),)
    elif normalized_scope == "horizon":
        cells = tuple(
            TargetCell(
                cell_id=f"horizon-h{horizon}",
                variables=normalized_variables,
                horizons=(horizon,),
            )
            for horizon in normalized_horizons
        )
    elif normalized_scope == "variable":
        cells = tuple(
            TargetCell(
                cell_id=f"variable-{_slugify(variable)}",
                variables=(variable,),
                horizons=normalized_horizons,
            )
            for variable in normalized_variables
        )
    elif normalized_scope == "variable_horizon":
        cells = tuple(
            TargetCell(
                cell_id=f"variable-{_slugify(variable)}-h{horizon}",
                variables=(variable,),
                horizons=(horizon,),
            )
            for variable in normalized_variables
            for horizon in normalized_horizons
        )
    else:
        groups = _normalize_group_definitions(
            variable_groups,
            target_variables=normalized_variables,
            residual_group_name=residual_group_name,
        )
        built_cells: list[TargetCell] = []
        for group_name, group_variables in groups:
            if separate_group_horizons:
                for horizon in normalized_horizons:
                    built_cells.append(
                        TargetCell(
                            cell_id=f"group-{_slugify(group_name)}-h{horizon}",
                            variables=group_variables,
                            horizons=(horizon,),
                            group_name=group_name,
                        )
                    )
            else:
                built_cells.append(
                    TargetCell(
                        cell_id=f"group-{_slugify(group_name)}",
                        variables=group_variables,
                        horizons=normalized_horizons,
                        group_name=group_name,
                    )
                )
        cells = tuple(built_cells)

    return SelectionPlan(
        scope=normalized_scope,
        target_variables=normalized_variables,
        target_horizons=normalized_horizons,
        cells=cells,
    )


__all__ = [
    "SUPPORTED_SCOPES",
    "SelectionPlan",
    "SelectionScope",
    "TargetCell",
    "TargetKey",
    "build_selection_plan",
]