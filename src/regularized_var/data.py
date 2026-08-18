"""Clean, NumPy-only panel adapter and fold-local standardization.

The regularized VAR experiments operate on a single, already-transformed
macroeconomic panel: a 2D ``(T, n)`` numeric array with an explicit variable
order and optional per-row date labels. This module provides:

* :class:`PanelData` -- a small immutable container for the transformed panel,
  the variable order, and optional row labels.
* :class:`Standardizer` -- fold-local mean/scale standardization that is fit
  **only** on a training slice and then applied to validation inputs, with an
  inverse transform used to return forecasts to the evaluation scale before any
  forecast error is computed.

Both are deliberately free of pandas and Bayesian dependencies. A thin optional
``load_panel_csv`` helper is provided for convenience but imports pandas lazily
so the core module stays dependency-light.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Sequence

import numpy as np


__all__ = [
    "PanelData",
    "Standardizer",
    "PREPROCESSING_MODES",
    "load_panel_csv",
]


# Documented preprocessing modes recorded in run metadata.
PREPROCESSING_MODES = ("none", "standardize")


def _as_2d_float(values: object, *, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{label} must be a 2D (T, n) array, got shape {array.shape}.")
    if array.size == 0:
        raise ValueError(f"{label} must be non-empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite values; no imputation is performed.")
    return array


@dataclass(frozen=True)
class PanelData:
    """An immutable, validated transformed panel.

    ``values`` is a ``(T, n)`` float array in the evaluation (transformed) scale.
    ``variable_names`` gives the explicit column order. ``date_labels`` optionally
    labels each row (dates, integers, ...) and is used only for reporting.
    """

    values: np.ndarray
    variable_names: tuple[str, ...]
    date_labels: tuple[object, ...] | None = None

    def __post_init__(self) -> None:
        values = _as_2d_float(self.values, label="values")
        names = tuple(str(name) for name in self.variable_names)
        if len(names) != values.shape[1]:
            raise ValueError(
                f"variable_names has length {len(names)} but the panel has "
                f"{values.shape[1]} columns."
            )
        if len(set(names)) != len(names):
            raise ValueError("variable_names must be unique.")
        if self.date_labels is not None:
            labels = tuple(self.date_labels)
            if len(labels) != values.shape[0]:
                raise ValueError(
                    f"date_labels has length {len(labels)} but the panel has "
                    f"{values.shape[0]} rows."
                )
            object.__setattr__(self, "date_labels", labels)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "variable_names", names)

    @property
    def n_observations(self) -> int:
        return self.values.shape[0]

    @property
    def n_variables(self) -> int:
        return self.values.shape[1]

    def column_index(self, variable: str) -> int:
        try:
            return self.variable_names.index(variable)
        except ValueError as exc:  # pragma: no cover - defensive
            raise KeyError(f"unknown variable {variable!r}.") from exc

    def label_for(self, row: int) -> object:
        if self.date_labels is None:
            return int(row)
        return self.date_labels[row]

    def to_metadata(self) -> dict[str, object]:
        return {
            "n_observations": self.n_observations,
            "n_variables": self.n_variables,
            "variable_names": list(self.variable_names),
            "has_date_labels": self.date_labels is not None,
        }


@dataclass(frozen=True)
class Standardizer:
    """Fold-local mean/scale standardization.

    Fit :meth:`fit` on a *training* slice only. :meth:`transform` applies the
    training-fold statistics to any inputs (e.g. validation rows), and
    :meth:`inverse_transform` maps standardized forecasts back to the evaluation
    scale so forecast errors are always computed on the original transformed
    series. When ``enabled`` is ``False`` the standardizer is the identity map,
    which keeps the code path uniform while recording that no standardization was
    applied.
    """

    mean: np.ndarray
    scale: np.ndarray
    enabled: bool

    @classmethod
    def fit(
        cls,
        training_values: object,
        *,
        enabled: bool,
        min_scale: float = 1e-8,
    ) -> "Standardizer":
        train = _as_2d_float(training_values, label="training_values")
        n = train.shape[1]
        if not enabled:
            return cls(mean=np.zeros(n), scale=np.ones(n), enabled=False)
        if not isinstance(min_scale, Real) or float(min_scale) <= 0.0:
            raise ValueError("min_scale must be a positive real number.")
        mean = train.mean(axis=0)
        # Sample standard deviation (ddof=1) when possible; floor tiny scales so a
        # constant column does not produce a division by zero.
        if train.shape[0] > 1:
            scale = train.std(axis=0, ddof=1)
        else:  # pragma: no cover - guarded by validation elsewhere
            scale = np.zeros(n)
        scale = np.where(scale < float(min_scale), float(min_scale), scale)
        return cls(mean=mean, scale=scale, enabled=True)

    def transform(self, values: object) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        return (array - self.mean) / self.scale

    def inverse_transform(self, values: object) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        return array * self.scale + self.mean

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
        }


def load_panel_csv(
    path: object,
    *,
    variables: Sequence[str] | None = None,
    date_column: str | None = None,
) -> PanelData:
    """Load a transformed panel from a wide CSV file.

    Each row is one time period and each selected column is one variable. When
    ``date_column`` is provided it is used for row labels and excluded from the
    numeric block. pandas is imported lazily so the core module has no hard
    pandas dependency.
    """

    import pandas as pd  # local import keeps the module pandas-free by default

    frame = pd.read_csv(path)
    labels: tuple[object, ...] | None = None
    if date_column is not None:
        if date_column not in frame.columns:
            raise KeyError(f"date_column {date_column!r} is not in the CSV.")
        labels = tuple(frame[date_column].tolist())
        frame = frame.drop(columns=[date_column])
    if variables is not None:
        missing = [name for name in variables if name not in frame.columns]
        if missing:
            raise KeyError(f"requested variables missing from CSV: {missing}.")
        frame = frame[list(variables)]
    names = tuple(str(col) for col in frame.columns)
    values = frame.to_numpy(dtype=float)
    return PanelData(values=values, variable_names=names, date_labels=labels)
