"""Model-independent reporting and statistical-comparison layer.

This module loads *canonical* forecast panels produced by heterogeneous models
(GLP, MF-BVAR, ridge, direct ridge, and optionally a Minnesota BVAR), normalizes
harmless legacy schema differences through **explicit** adapters, and produces
paired, leakage-safe comparison tables. It never silently collapses duplicate
forecasts and it separates two things the paper must not conflate:

* the **forecasting architecture** (``forecast_method`` -- iterated vs direct),
  and
* the **hyperparameter-selection scope** (``scope`` -- pooled / horizon /
  variable / variable_horizon) or a model's **native** selection method.

Statistical inference (Diebold-Mariano, block bootstrap, Holm) is delegated to
:mod:`common_hpo.inference` and always operates on *paired* outer forecast
errors only.

The authoritative outputs are CSV tables (plots are optional). In particular
``scope_gains.csv`` answers the main paper question directly, without manually
stitching directories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .inference import (
    BootstrapResult,
    DMResult,
    bootstrap_ci,
    diebold_mariano,
    holm_adjust,
)


__all__ = [
    "ReportingError",
    "SchemaError",
    "DuplicateForecastError",
    "RealizationMismatchError",
    "CoverageError",
    "PanelSpec",
    "AlignmentReport",
    "CommonSampleReport",
    "restrict_to_common_sample",
    "CANONICAL_COLUMNS",
    "KEY_COLUMNS",
    "load_forecast_panel",
    "load_selected_hyperparameters",
    "load_failed_origins",
    "load_run_metadata",
    "combine_panels",
    "check_origin_alignment",
    "cell_losses",
    "rmse_by_target",
    "mae_by_target",
    "relative_rmse",
    "average_ranks",
    "scope_gains",
    "scope_gain_summary",
    "hyperparameter_summary",
    "selection_stability",
    "failure_summary",
    "computational_cost",
    "dm_tests",
    "bootstrap_intervals",
    "write_comparison_summary",
    "ScopeContrast",
    "standard_scope_contrasts",
]


class ReportingError(ValueError):
    """Base class for reporting errors."""


class SchemaError(ReportingError):
    """Raised when a forecast panel is missing required columns."""


class DuplicateForecastError(ReportingError):
    """Raised when a panel contains duplicate forecasts for one target."""


class RealizationMismatchError(ReportingError):
    """Raised when two models disagree on a realization for a shared target."""


class CoverageError(ReportingError):
    """Raised when common-sample coverage is unacceptable.

    Either the caller demanded ``policy="raise"`` and some observations are not
    shared by every model in a cell, or the retained share of observations fell
    below ``min_coverage``.
    """


# Canonical long-form columns after adaptation.
CANONICAL_COLUMNS = (
    "model",
    "family",
    "size",
    "scope",
    "selection",
    "forecast_method",
    "group",
    "forecast_origin",
    "target_quarter",
    "horizon",
    "variable",
    "forecast",
    "actual",
    "error",
)

# Uniqueness key for a single forecast observation within one model.
KEY_COLUMNS = ("forecast_origin", "target_quarter", "horizon", "variable", "group")

# Families whose panels use the metric schema (mean_metric/actual_metric/...).
_METRIC_FAMILIES = frozenset({"ridge", "ridge_direct", "mfbvar", "paper_mf", "minnesota"})
# Families whose panels use the legacy GLP schema (mean/actual/error/model_size).
_GLP_FAMILIES = frozenset({"glp", "glp_legacy"})


# --------------------------------------------------------------------------- #
# Panel specification
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PanelSpec:
    """Describes one panel to load and how to tag it for comparison.

    ``family`` selects the schema adapter. ``scope`` records the
    hyperparameter-selection scope (``pooled`` / ``horizon`` / ``variable`` /
    ``variable_horizon``) or ``native`` for a model's built-in selection.
    ``selection`` is ``forecast_loss`` for the scope-grid selection or ``native``
    for the model's own method (e.g. Mango-RMSE, fixed paper priors). These are
    kept distinct so native selection and forecast-loss pooled selection are
    never conflated.
    """

    model: str
    family: str
    scope: str = "native"
    selection: str = "native"
    forecast_method: str = "native"
    size: str | None = None
    native_method: str | None = None
    path: Path | None = None

    def __post_init__(self) -> None:
        if not str(self.model).strip():
            raise ReportingError("model label must be a non-empty string.")
        if self.family not in _METRIC_FAMILIES and self.family not in _GLP_FAMILIES:
            raise ReportingError(
                f"unknown family {self.family!r}; expected one of "
                f"{sorted(_METRIC_FAMILIES | _GLP_FAMILIES)}."
            )

    def tags(self) -> dict[str, object]:
        return {
            "model": self.model,
            "family": self.family,
            "size": self.size,
            "scope": self.scope,
            "selection": self.selection,
            "forecast_method": self.forecast_method,
            "native_method": self.native_method,
        }


# --------------------------------------------------------------------------- #
# Loading & adaptation
# --------------------------------------------------------------------------- #
def _read_frame(source: object) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    return pd.read_csv(source)


def _require_columns(df: pd.DataFrame, required: Sequence[str], *, context: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"{context} is missing required columns: {missing}.")


def _adapt_metric_panel(df: pd.DataFrame, spec: PanelSpec) -> pd.DataFrame:
    _require_columns(
        df,
        ("forecast_origin", "target_quarter", "horizon_quarters", "variable",
         "mean_metric", "actual_metric"),
        context=f"metric panel for model {spec.model!r}",
    )
    out = pd.DataFrame()
    out["forecast_origin"] = df["forecast_origin"].astype(str)
    out["target_quarter"] = df["target_quarter"].astype(str)
    out["horizon"] = df["horizon_quarters"].astype(int)
    out["variable"] = df["variable"].astype(str)
    out["group"] = df["group"].astype(str) if "group" in df.columns else "all"
    out["forecast"] = df["mean_metric"].astype(float)
    out["actual"] = df["actual_metric"].astype(float)
    if "error_metric" in df.columns:
        out["error"] = df["error_metric"].astype(float)
    else:
        out["error"] = out["forecast"] - out["actual"]
    if "forecast_method" in df.columns:
        out["forecast_method"] = df["forecast_method"].astype(str)
    else:
        out["forecast_method"] = spec.forecast_method
    return out


def _adapt_glp_panel(df: pd.DataFrame, spec: PanelSpec) -> pd.DataFrame:
    _require_columns(
        df,
        ("forecast_origin", "target_quarter", "horizon_quarters", "variable",
         "mean", "actual"),
        context=f"GLP panel for model {spec.model!r}",
    )
    out = pd.DataFrame()
    out["forecast_origin"] = df["forecast_origin"].astype(str)
    out["target_quarter"] = df["target_quarter"].astype(str)
    out["horizon"] = df["horizon_quarters"].astype(int)
    out["variable"] = df["variable"].astype(str)
    # Legacy GLP panels carry no per-origin group split.
    out["group"] = df["group"].astype(str) if "group" in df.columns else "all"
    out["forecast"] = df["mean"].astype(float)
    out["actual"] = df["actual"].astype(float)
    if "error" in df.columns:
        out["error"] = df["error"].astype(float)
    else:
        out["error"] = out["forecast"] - out["actual"]
    # GLP panels never record a forecast architecture; treat as native/iterated.
    out["forecast_method"] = spec.forecast_method
    return out


def load_forecast_panel(source: object, spec: PanelSpec) -> pd.DataFrame:
    """Load and adapt one forecast panel to the canonical long format.

    ``source`` may be a path to ``forecast_panel.csv`` or a preloaded
    ``DataFrame``. Duplicate forecasts for a single target raise
    :class:`DuplicateForecastError` -- they are never silently collapsed.
    """

    df = _read_frame(source)
    if spec.family in _METRIC_FAMILIES:
        canonical = _adapt_metric_panel(df, spec)
    else:
        canonical = _adapt_glp_panel(df, spec)

    # Attach tags.
    canonical["model"] = spec.model
    canonical["family"] = spec.family
    canonical["scope"] = spec.scope
    canonical["selection"] = spec.selection
    # Size: prefer explicit spec, else legacy model_size column.
    if spec.size is not None:
        canonical["size"] = spec.size
    elif "model_size" in df.columns:
        canonical["size"] = df["model_size"].astype(str)
    else:
        canonical["size"] = None

    # Reject duplicate observations before any aggregation.
    dup_mask = canonical.duplicated(list(KEY_COLUMNS), keep=False)
    if dup_mask.any():
        n_dup = int(dup_mask.sum())
        example = canonical.loc[dup_mask, list(KEY_COLUMNS)].iloc[0].to_dict()
        raise DuplicateForecastError(
            f"model {spec.model!r} has {n_dup} duplicate forecast rows for keys "
            f"{list(KEY_COLUMNS)}; first offending key: {example}."
        )

    canonical = canonical[list(CANONICAL_COLUMNS)].reset_index(drop=True)
    return canonical


def load_run_metadata(source: object) -> dict[str, object]:
    """Load a ``run_metadata.json`` file (or pass through a dict)."""

    if isinstance(source, Mapping):
        return dict(source)
    with Path(source).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_selected_hyperparameters(source: object, spec: PanelSpec) -> pd.DataFrame:
    """Load selected hyperparameters and tag them with the model label."""

    df = _read_frame(source)
    df = df.copy()
    df["model"] = spec.model
    df["family"] = spec.family
    df["scope"] = spec.scope
    df["selection"] = spec.selection
    return df


def load_failed_origins(source: object, spec: PanelSpec) -> pd.DataFrame:
    """Load a failed-origins table, normalizing the optional ``stage`` column."""

    df = _read_frame(source)
    df = df.copy()
    if "stage" not in df.columns:
        df["stage"] = "unknown"
    if "forecast_origin" not in df.columns:
        raise SchemaError(
            f"failed-origins table for {spec.model!r} lacks forecast_origin."
        )
    df["model"] = spec.model
    return df


# --------------------------------------------------------------------------- #
# Combining & alignment
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AlignmentReport:
    """Outer-origin alignment across the combined models."""

    common_origins: tuple[str, ...]
    per_model_origins: Mapping[str, tuple[str, ...]]
    unmatched: Mapping[str, tuple[str, ...]]
    # Common-sample diagnostics (defaults keep legacy constructions valid).
    policy: str = "restrict"
    n_common_keys: int = 0
    n_excluded_keys: int = 0
    coverage: float = float("nan")

    @property
    def is_aligned(self) -> bool:
        return all(len(v) == 0 for v in self.unmatched.values())

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for model, origins in self.unmatched.items():
            for origin in origins:
                rows.append({"model": model, "unmatched_origin": origin})
        return pd.DataFrame(rows, columns=["model", "unmatched_origin"])


def combine_panels(
    panels: Iterable[pd.DataFrame],
    *,
    require_matched_realizations: bool = True,
    realization_tolerance: float = 1e-6,
) -> pd.DataFrame:
    """Concatenate canonical panels and validate cross-model consistency.

    When ``require_matched_realizations`` is set, any shared observation key with
    disagreeing ``actual`` values (beyond ``realization_tolerance``) raises
    :class:`RealizationMismatchError`. Squared- and absolute-error columns are
    added for downstream loss computation.
    """

    frames = [p for p in panels]
    if not frames:
        raise ReportingError("at least one panel is required.")
    combined = pd.concat(frames, ignore_index=True)

    # Cross-model realization-definition check on shared observation keys.
    if require_matched_realizations:
        obs_key = ["forecast_origin", "target_quarter", "horizon", "variable"]
        spread = combined.groupby(obs_key)["actual"].agg(["min", "max"])
        disagree = spread[(spread["max"] - spread["min"]).abs() > realization_tolerance]
        if not disagree.empty:
            example = disagree.iloc[0]
            raise RealizationMismatchError(
                "models disagree on realizations for shared targets "
                f"(e.g. spread {float(example['max'] - example['min']):.6g}); "
                "the compared models do not share a realization definition."
            )

    combined["sq_error"] = combined["error"].astype(float) ** 2
    combined["abs_error"] = combined["error"].astype(float).abs()
    return combined


def check_origin_alignment(
    combined: pd.DataFrame,
    *,
    policy: str = "restrict",
    min_coverage: float = 0.0,
) -> AlignmentReport:
    """Report whether all models use the same set of outer forecast origins.

    In addition to the origin-level view, the report carries the *cell-wise*
    common-sample diagnostics used by the loss tables (``n_common_keys``,
    ``n_excluded_keys``, ``coverage``), computed under ``policy`` and
    ``min_coverage`` exactly as :func:`restrict_to_common_sample` would.
    """

    per_model: dict[str, tuple[str, ...]] = {}
    for model, group in combined.groupby("model"):
        per_model[str(model)] = tuple(sorted(group["forecast_origin"].astype(str).unique()))

    if per_model:
        common = set.intersection(*(set(v) for v in per_model.values()))
    else:  # pragma: no cover
        common = set()
    unmatched = {
        model: tuple(sorted(set(origins) - common))
        for model, origins in per_model.items()
    }
    _, sample_report = restrict_to_common_sample(
        combined, policy=policy, min_coverage=min_coverage
    )
    return AlignmentReport(
        common_origins=tuple(sorted(common)),
        per_model_origins=per_model,
        unmatched=unmatched,
        policy=policy,
        n_common_keys=sample_report.n_common_keys,
        n_excluded_keys=sample_report.n_excluded_keys,
        coverage=sample_report.coverage,
    )


# --------------------------------------------------------------------------- #
# Common-sample restriction
# --------------------------------------------------------------------------- #
# Observation key used for *pairing* models within one (variable, horizon) cell.
# It is deliberately the same tuple used by :func:`_paired_errors`, so point
# estimates (RMSE / ranks / scope gains) and paired inference (DM, bootstrap)
# describe the same estimand on the same sample.
COMMON_SAMPLE_KEY = ("forecast_origin", "target_quarter", "group")

_COMMON_SAMPLE_POLICIES = ("restrict", "raise", "advisory")


@dataclass
class CommonSampleReport:
    """Diagnostics for a cell-wise common-sample restriction."""

    policy: str
    models: tuple[str, ...]
    n_rows_input: int
    n_rows_common: int
    n_rows_excluded: int
    n_common_keys: int
    n_excluded_keys: int
    coverage: float
    counts: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_frame(self) -> pd.DataFrame:
        """One row per model plus an ``__all__`` row, for ``common_sample.csv``."""

        base = {
            "policy": self.policy,
            "coverage": self.coverage,
            "n_common_keys": self.n_common_keys,
            "n_excluded_keys": self.n_excluded_keys,
        }
        rows: list[dict[str, object]] = []
        if not self.counts.empty:
            per_model = self.counts.groupby("model", dropna=False)[
                ["n_model_total", "n_common", "n_excluded"]
            ].sum()
            for model, row in per_model.iterrows():
                rows.append(
                    {
                        "model": str(model),
                        "n_model_total": int(row["n_model_total"]),
                        "n_common": int(row["n_common"]),
                        "n_excluded": int(row["n_excluded"]),
                        **base,
                    }
                )
        rows.append(
            {
                "model": "__all__",
                "n_model_total": int(self.n_rows_input),
                "n_common": int(self.n_rows_common),
                "n_excluded": int(self.n_rows_excluded),
                **base,
            }
        )
        return pd.DataFrame(rows)


def _key_tuples(frame: pd.DataFrame) -> pd.Series:
    keys = frame[list(COMMON_SAMPLE_KEY)].astype(str)
    return pd.Series(list(map(tuple, keys.to_numpy())), index=frame.index, dtype=object)


def restrict_to_common_sample(
    combined: pd.DataFrame,
    *,
    models: Sequence[str] | None = None,
    policy: str = "restrict",
    min_coverage: float = 0.0,
) -> tuple[pd.DataFrame, CommonSampleReport]:
    """Restrict ``combined`` to the **cell-wise** common sample across models.

    The convention is identical to :func:`_paired_errors`: within each
    ``(variable, horizon)`` cell, only the observation keys
    ``(forecast_origin, target_quarter, group)`` that are present for *every
    model appearing in that cell* are retained. Models covering disjoint cells
    are therefore never annihilated -- a model that is simply absent from a cell
    does not shrink that cell's sample.

    Without this restriction, per-model aggregates (RMSE, MAE, ranks, scope
    gains) are estimated on *different* samples, so their ratios and differences
    are not comparable and can even reverse the sign of the headline number.

    ``policy``:

    * ``"restrict"`` (default) -- drop the non-shared rows,
    * ``"raise"``  -- raise :class:`CoverageError` if anything would be dropped,
    * ``"advisory"`` -- return the frame unchanged but report what *would* be
      dropped (diagnostics only; aggregates stay unpaired).

    ``min_coverage`` raises :class:`CoverageError` when the retained share of
    rows falls below the given threshold.

    Returns the (possibly restricted) frame and a :class:`CommonSampleReport`.
    """

    if policy not in _COMMON_SAMPLE_POLICIES:
        raise ReportingError(
            f"unknown coverage policy {policy!r}; expected one of "
            f"{list(_COMMON_SAMPLE_POLICIES)}."
        )
    if not 0.0 <= float(min_coverage) <= 1.0:
        raise ReportingError("min_coverage must lie in [0, 1].")

    frame = combined
    if models is not None:
        wanted = {str(m) for m in models}
        frame = frame[frame["model"].astype(str).isin(wanted)]

    keep = pd.Series(False, index=frame.index)
    n_common_keys = 0
    n_excluded_keys = 0
    count_rows: list[dict[str, object]] = []

    if not frame.empty:
        for (variable, horizon), block in frame.groupby(["variable", "horizon"], dropna=False):
            key_series = _key_tuples(block)
            per_model_keys: dict[object, set] = {}
            for model, model_block in block.groupby("model", dropna=False):
                per_model_keys[model] = set(key_series.loc[model_block.index])
            all_keys: set = set().union(*per_model_keys.values())
            common: set = set.intersection(*per_model_keys.values())
            n_common_keys += len(common)
            n_excluded_keys += len(all_keys - common)
            keep.loc[block.index] = key_series.isin(common).to_numpy()
            for model, model_keys in per_model_keys.items():
                count_rows.append(
                    {
                        "model": model,
                        "variable": variable,
                        "horizon": horizon,
                        "n_model_total": int(len(model_keys)),
                        "n_common": int(len(common)),
                        "n_excluded": int(len(model_keys) - len(common)),
                    }
                )

    counts = pd.DataFrame(
        count_rows,
        columns=["model", "variable", "horizon", "n_model_total", "n_common", "n_excluded"],
    )
    n_rows_input = int(len(frame))
    n_rows_common = int(keep.sum())
    coverage = float(n_rows_common / n_rows_input) if n_rows_input else float("nan")

    report = CommonSampleReport(
        policy=policy,
        models=tuple(sorted(str(m) for m in frame["model"].astype(str).unique())) if n_rows_input else (),
        n_rows_input=n_rows_input,
        n_rows_common=n_rows_common,
        n_rows_excluded=n_rows_input - n_rows_common,
        n_common_keys=n_common_keys,
        n_excluded_keys=n_excluded_keys,
        coverage=coverage,
        counts=counts,
    )

    if policy == "raise" and report.n_rows_excluded > 0:
        raise CoverageError(
            f"{report.n_rows_excluded} observation(s) across {n_excluded_keys} key(s) "
            "are not shared by every model in their (variable, horizon) cell; "
            'coverage policy is "raise".'
        )
    if n_rows_input and coverage < float(min_coverage):
        raise CoverageError(
            f"common-sample coverage {coverage:.4f} is below the required "
            f"minimum {float(min_coverage):.4f}."
        )

    restricted = frame if policy == "advisory" else frame[keep]
    return restricted.copy(), report


def _restricted(
    combined: pd.DataFrame,
    *,
    common_sample: bool,
    policy: str,
    min_coverage: float,
) -> tuple[pd.DataFrame, CommonSampleReport | None]:
    """Apply the restriction (or not) and return the frame plus its report."""

    if not common_sample:
        return combined, None
    return restrict_to_common_sample(combined, policy=policy, min_coverage=min_coverage)


def _count_lookup(report: CommonSampleReport | None) -> dict[tuple, tuple[int, int, int]]:
    if report is None or report.counts.empty:
        return {}
    return {
        (str(r.model), str(r.variable), int(r.horizon)):
            (int(r.n_common), int(r.n_model_total), int(r.n_excluded))
        for r in report.counts.itertuples(index=False)
    }


def _count_fields(
    lookup: dict[tuple, tuple[int, int, int]],
    model: object,
    variable: object,
    horizon: object,
    n_used: int,
) -> dict[str, int]:
    key = (str(model), str(variable), int(horizon))
    if key in lookup:
        n_common, n_total, n_excluded = lookup[key]
    else:
        n_common, n_total, n_excluded = n_used, n_used, 0
    return {"n_common": n_common, "n_model_total": n_total, "n_excluded": n_excluded}


# --------------------------------------------------------------------------- #
# Loss tables
# --------------------------------------------------------------------------- #
def _rmse(values: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(values.astype(float)))))


def cell_losses(
    combined: pd.DataFrame,
    *,
    loss: str = "rmse",
    common_sample: bool = True,
    policy: str = "restrict",
    min_coverage: float = 0.0,
) -> pd.DataFrame:
    """Return per-``(model, variable, horizon)`` loss with model tags preserved.

    ``loss`` is ``"rmse"``, ``"mse"``, or ``"mae"``. The result carries the
    ``family``, ``size``, ``scope``, ``selection`` and ``forecast_method`` tags so
    scope gains can be computed without re-joining directories.

    With ``common_sample=True`` (default) every model in a ``(variable,
    horizon)`` cell is evaluated on the *same* observation keys -- see
    :func:`restrict_to_common_sample`. The reported ``n`` is the sample actually
    used; ``n_common``/``n_model_total``/``n_excluded`` document the restriction.

    ``policy``/``min_coverage`` are forwarded to
    :func:`restrict_to_common_sample`; with ``policy="advisory"`` nothing is
    dropped, so ``n`` is each model's own row count while ``n_common`` still
    reports the shared sample that *would* have been used.
    """

    frame, report = _restricted(
        combined, common_sample=common_sample, policy=policy, min_coverage=min_coverage
    )
    lookup = _count_lookup(report)

    tag_columns = ["model", "family", "size", "scope", "selection", "forecast_method"]
    grouped = frame.groupby(tag_columns + ["variable", "horizon"], dropna=False)
    rows = []
    for keys, block in grouped:
        record = dict(zip(tag_columns + ["variable", "horizon"], keys))
        errors = block["error"].astype(float)
        if loss == "rmse":
            value = _rmse(errors)
        elif loss == "mse":
            value = float(np.mean(np.square(errors)))
        elif loss == "mae":
            value = float(np.mean(np.abs(errors)))
        else:
            raise ReportingError(f"unknown loss {loss!r}; expected rmse|mse|mae.")
        record["loss"] = loss
        record["loss_value"] = value
        record["n"] = int(len(block))
        record.update(
            _count_fields(lookup, record["model"], record["variable"],
                          record["horizon"], int(len(block)))
        )
        rows.append(record)
    return pd.DataFrame(rows)


def rmse_by_target(
    combined: pd.DataFrame,
    *,
    common_sample: bool = True,
    policy: str = "restrict",
    min_coverage: float = 0.0,
) -> pd.DataFrame:
    """RMSE per ``(model, variable, horizon)`` on the cell-wise common sample.

    See :func:`restrict_to_common_sample`. Without the restriction the RMSEs of
    two models in the same cell may be estimated on different observations, so
    their ratio, difference or rank is not interpretable.
    """

    frame, report = _restricted(
        combined, common_sample=common_sample, policy=policy, min_coverage=min_coverage
    )
    lookup = _count_lookup(report)

    rows = []
    for (model, variable, horizon), block in frame.groupby(["model", "variable", "horizon"]):
        rows.append(
            {
                "model": model,
                "variable": variable,
                "horizon": int(horizon),
                "rmse": _rmse(block["error"]),
                "n": int(len(block)),
                **_count_fields(lookup, model, variable, horizon, int(len(block))),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "variable", "horizon"]).reset_index(drop=True)


def mae_by_target(
    combined: pd.DataFrame,
    *,
    common_sample: bool = True,
    policy: str = "restrict",
    min_coverage: float = 0.0,
) -> pd.DataFrame:
    """MAE per ``(model, variable, horizon)`` on the cell-wise common sample."""

    frame, report = _restricted(
        combined, common_sample=common_sample, policy=policy, min_coverage=min_coverage
    )
    lookup = _count_lookup(report)

    rows = []
    for (model, variable, horizon), block in frame.groupby(["model", "variable", "horizon"]):
        rows.append(
            {
                "model": model,
                "variable": variable,
                "horizon": int(horizon),
                "mae": float(np.mean(np.abs(block["error"].astype(float)))),
                "n": int(len(block)),
                **_count_fields(lookup, model, variable, horizon, int(len(block))),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "variable", "horizon"]).reset_index(drop=True)


def relative_rmse(
    rmse_table: pd.DataFrame,
    *,
    baseline_model: str,
    combined: pd.DataFrame | None = None,
    policy: str = "restrict",
    min_coverage: float = 0.0,
) -> pd.DataFrame:
    """Relative RMSE of each model versus ``baseline_model`` per target cell.

    When ``combined`` is supplied, **both** the model RMSE and the baseline RMSE
    are recomputed on the *pairwise* common sample shared by that model and the
    baseline, so the ratio is a like-for-like comparison.

    Note explicitly that "common with the baseline" is **not** the same sample as
    "common with all models": each row of the resulting table may rest on a
    different (pairwise) sample, which is why ``n``, ``n_common`` and
    ``n_baseline`` are reported per row. If you need one single sample basis for
    every model, restrict ``combined`` once with
    :func:`restrict_to_common_sample` before building any table -- then the
    pairwise restriction below is a no-op.
    """

    if baseline_model not in set(rmse_table["model"]):
        raise ReportingError(f"baseline model {baseline_model!r} is not in the RMSE table.")

    if combined is not None:
        return _relative_rmse_pairwise(
            combined, baseline_model=baseline_model, policy=policy, min_coverage=min_coverage
        )

    baseline_rows = rmse_table[rmse_table["model"] == baseline_model].set_index(
        ["variable", "horizon"]
    )
    baseline = baseline_rows["rmse"]
    baseline_n = baseline_rows["n"] if "n" in baseline_rows.columns else None
    rows = []
    for _, row in rmse_table.iterrows():
        key = (row["variable"], int(row["horizon"]))
        base = float(baseline.get(key, np.nan))
        rmse = float(row["rmse"])
        if np.isfinite(base) and base > 0.0:
            ratio = rmse / base
            pct = 100.0 * (rmse - base) / base
        else:
            ratio = float("nan")
            pct = float("nan")
        n_used = int(row["n"]) if "n" in rmse_table.columns else -1
        n_common = int(row["n_common"]) if "n_common" in rmse_table.columns else n_used
        n_baseline = (
            int(baseline_n.get(key, n_common)) if baseline_n is not None else n_common
        )
        rows.append(
            {
                "model": row["model"],
                "variable": row["variable"],
                "horizon": int(row["horizon"]),
                "rmse": rmse,
                "baseline_model": baseline_model,
                "baseline_rmse": base,
                "relative_rmse": ratio,
                "relative_rmse_pct": pct,
                "n": n_used,
                "n_common": n_common,
                "n_baseline": n_baseline,
                "sample_basis": "as_supplied",
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "variable", "horizon"]).reset_index(drop=True)


def _relative_rmse_pairwise(
    combined: pd.DataFrame,
    *,
    baseline_model: str,
    policy: str = "restrict",
    min_coverage: float = 0.0,
) -> pd.DataFrame:
    """Relative RMSE where each model is paired with the baseline separately."""

    if baseline_model not in set(combined["model"].astype(str)):
        raise ReportingError(f"baseline model {baseline_model!r} is not in the panel.")

    rows = []
    for model in sorted(combined["model"].astype(str).unique()):
        models = [model] if model == baseline_model else [model, baseline_model]
        pair, report = restrict_to_common_sample(
            combined, models=models, policy=policy, min_coverage=min_coverage
        )
        lookup = _count_lookup(report)
        for (variable, horizon), block in pair.groupby(["variable", "horizon"]):
            model_block = block[block["model"].astype(str) == model]
            base_block = block[block["model"].astype(str) == baseline_model]
            if model_block.empty:
                continue
            rmse = _rmse(model_block["error"])
            base = _rmse(base_block["error"]) if not base_block.empty else float("nan")
            if np.isfinite(base) and base > 0.0:
                ratio = rmse / base
                pct = 100.0 * (rmse - base) / base
            else:
                ratio = float("nan")
                pct = float("nan")
            counts = _count_fields(lookup, model, variable, horizon, int(len(model_block)))
            rows.append(
                {
                    "model": model,
                    "variable": variable,
                    "horizon": int(horizon),
                    "rmse": rmse,
                    "baseline_model": baseline_model,
                    "baseline_rmse": base,
                    "relative_rmse": ratio,
                    "relative_rmse_pct": pct,
                    "n": int(len(model_block)),
                    "n_common": counts["n_common"],
                    "n_baseline": int(len(base_block)),
                    "sample_basis": "pairwise_common_with_baseline",
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "variable", "horizon"]).reset_index(drop=True)


def average_ranks(
    rmse_table: pd.DataFrame, *, require_common_sample: bool = True
) -> pd.DataFrame:
    """Average rank of each model across target cells (ties share average rank).

    Ranking losses that were estimated on different samples is meaningless, so
    with ``require_common_sample=True`` (default) a cell whose models disagree on
    their sample size raises :class:`CoverageError`. Pass a table built by
    :func:`rmse_by_target` with ``common_sample=True`` (the default), or set the
    flag to ``False`` to rank explicitly unpaired losses.

    Callers running under the ``"advisory"`` coverage policy must pass
    ``require_common_sample=False``: advisory means "report the coverage
    shortfall, restrict nothing", so the per-model samples legitimately differ
    and raising here would contradict the policy. The resulting ranks are then
    explicitly unpaired and the emitted ``sample_basis`` column says so.
    """

    count_columns = [c for c in ("n", "n_common") if c in rmse_table.columns]
    ranked_frames = []
    for (variable, horizon), block in rmse_table.groupby(["variable", "horizon"]):
        if require_common_sample and count_columns and len(block) > 1:
            for count_column in count_columns:
                distinct = sorted({int(v) for v in block[count_column]})
                if len(distinct) > 1:
                    raise CoverageError(
                        f"cell (variable={variable!r}, horizon={int(horizon)}) mixes models "
                        f"estimated on different sample sizes ({count_column}={distinct}); "
                        "ranks would compare losses on different samples. Build the RMSE "
                        "table with common_sample=True or pass require_common_sample=False."
                    )
        block = block.copy()
        block["rank"] = block["rmse"].rank(method="average", ascending=True)
        ranked_frames.append(block)
    ranked = pd.concat(ranked_frames, ignore_index=True)
    rows = []
    for model, block in ranked.groupby("model"):
        n_used = int(block["n"].sum()) if "n" in block.columns else -1
        n_common = (
            int(block["n_common"].sum()) if "n_common" in block.columns else n_used
        )
        rows.append(
            {
                "model": model,
                "average_rank": float(block["rank"].mean()),
                "n_cells": int(len(block)),
                "n": n_used,
                "n_common": n_common,
                "sample_basis": (
                    "common_sample" if require_common_sample else "unpaired_per_model"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("average_rank").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Scope gains -- the authoritative table
# --------------------------------------------------------------------------- #
_SCOPES = ("pooled", "horizon", "variable", "variable_horizon", "native")


def _min_ignore_nan(a: float, b: float) -> float:
    """Return the smaller of two values, ignoring NaN; NaN if both are NaN."""

    if np.isnan(a) and np.isnan(b):
        return float("nan")
    if np.isnan(a):
        return float(b)
    if np.isnan(b):
        return float(a)
    return float(min(a, b))


def scope_gains(
    combined: pd.DataFrame,
    *,
    loss: str = "rmse",
    common_sample: bool = True,
    policy: str = "restrict",
    min_coverage: float = 0.0,
) -> pd.DataFrame:
    """Compute scope-gain decompositions per forecasting *system* and target cell.

    A *system* is a ``(family, size, forecast_method)`` triple. Within a system
    the loss ``L(scope)`` is computed for each available selection scope. The
    decompositions are:

    * ``horizon_gain      = L(pooled) - L(horizon)``
    * ``variable_gain     = L(pooled) - L(variable)``
    * ``interaction_gain  = min[L(horizon), L(variable)] - L(variable_horizon)``
    * ``vh_vs_pooled_gain = L(pooled) - L(variable_horizon)``
    * ``pooled_vs_native_gain = L(native) - L(pooled)``

    A **positive** gain always denotes a loss reduction (an improvement). Missing
    scopes yield ``NaN`` gains rather than fabricated numbers.

    With ``common_sample=True`` (default) the restriction is applied **per
    system**: within each system, all of that system's scopes are evaluated on
    the observation keys they share in a given cell. A gain is then a difference
    of losses measured on one and the same sample, which is what the paper
    claims. ``n_common``/``n_models``/``n_excluded`` document that sample.
    """

    system_cols = ["family", "size", "forecast_method", "variable", "horizon"]
    frames = []
    reports: dict[tuple, CommonSampleReport] = {}
    for keys, system_block in combined.groupby(
        ["family", "size", "forecast_method"], dropna=False
    ):
        if common_sample:
            restricted, report = restrict_to_common_sample(
                system_block, policy=policy, min_coverage=min_coverage
            )
        else:
            restricted, report = system_block, None
        block_losses = cell_losses(restricted, loss=loss, common_sample=False)
        frames.append(block_losses)
        if report is not None:
            reports[keys] = report

    losses = (
        pd.concat(frames, ignore_index=True) if frames else cell_losses(combined, loss=loss)
    )

    def _system_counts(keys: tuple, variable: object, horizon: object) -> dict[str, object]:
        report = reports.get(keys)
        if report is None or report.counts.empty:
            return {"n_common": -1, "n_excluded": -1}
        counts = report.counts
        sel = counts[
            (counts["variable"].astype(str) == str(variable))
            & (counts["horizon"].astype(int) == int(horizon))
        ]
        if sel.empty:
            return {"n_common": 0, "n_excluded": 0}
        return {
            "n_common": int(sel["n_common"].iloc[0]),
            "n_excluded": int(sel["n_excluded"].sum()),
        }

    rows = []
    for keys, block in losses.groupby(system_cols, dropna=False):
        record = dict(zip(system_cols, keys))
        by_scope = {
            scope: float(block.loc[block["scope"] == scope, "loss_value"].mean())
            if (block["scope"] == scope).any()
            else float("nan")
            for scope in _SCOPES
        }
        L_pooled = by_scope["pooled"]
        L_horizon = by_scope["horizon"]
        L_variable = by_scope["variable"]
        L_vh = by_scope["variable_horizon"]
        L_native = by_scope["native"]

        system_key = (record["family"], record["size"], record["forecast_method"])
        counts = _system_counts(system_key, record["variable"], record["horizon"])

        record.update(
            {
                "loss": loss,
                "L_pooled": L_pooled,
                "L_horizon": L_horizon,
                "L_variable": L_variable,
                "L_variable_horizon": L_vh,
                "L_native": L_native,
                "horizon_gain": L_pooled - L_horizon,
                "variable_gain": L_pooled - L_variable,
                "interaction_gain": _min_ignore_nan(L_horizon, L_variable) - L_vh,
                "vh_vs_pooled_gain": L_pooled - L_vh,
                "vh_vs_best_marginal_gain": _min_ignore_nan(L_horizon, L_variable) - L_vh,
                "pooled_vs_native_gain": L_native - L_pooled,
                "n_common": counts["n_common"],
                "n_models": int(block["model"].nunique()),
                "n_excluded": counts["n_excluded"],
                "sample_basis": (
                    "system_common_sample"
                    if common_sample and policy != "advisory"
                    else "unpaired_per_model"
                ),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows).sort_values(system_cols).reset_index(drop=True)


_GAIN_COLUMNS = (
    "horizon_gain",
    "variable_gain",
    "interaction_gain",
    "vh_vs_pooled_gain",
    "pooled_vs_native_gain",
)


def scope_gain_summary(gains: pd.DataFrame) -> pd.DataFrame:
    """Aggregate scope gains: average, median, worst deterioration, share improved."""

    rows = []
    for gain in _GAIN_COLUMNS:
        if gain not in gains.columns:
            continue
        values = gains[gain].astype(float).dropna()
        if values.empty:
            rows.append(
                {
                    "gain": gain, "n_cells": 0, "average_gain": float("nan"),
                    "median_gain": float("nan"), "worst_deterioration": float("nan"),
                    "proportion_improved": float("nan"),
                }
            )
            continue
        rows.append(
            {
                "gain": gain,
                "n_cells": int(values.size),
                "average_gain": float(values.mean()),
                "median_gain": float(values.median()),
                # Worst deterioration is the most negative gain (0 if all >= 0).
                "worst_deterioration": float(min(values.min(), 0.0)),
                "proportion_improved": float((values > 0.0).mean()),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Hyperparameter summary & selection stability
# --------------------------------------------------------------------------- #
def _numeric_parameter_columns(df: pd.DataFrame) -> list[str]:
    skip = {"forecast_origin", "n_obs", "n_tied"}
    numeric = []
    for column in df.columns:
        if column in ("model", "family", "scope", "selection", "strategy",
                      "group", "cell_id", "event_id", "model_size", "last_quarter"):
            continue
        if column in skip:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            numeric.append(column)
    return numeric


def hyperparameter_summary(hyperparameters: pd.DataFrame) -> pd.DataFrame:
    """Aggregate selected hyperparameters (mean/std/min/max) per model and parameter."""

    rows = []
    for model, block in hyperparameters.groupby("model"):
        for parameter in _numeric_parameter_columns(block):
            values = block[parameter].astype(float).dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "model": model,
                    "parameter": parameter,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "n": int(values.size),
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "parameter"]).reset_index(drop=True)


def selection_stability(hyperparameters: pd.DataFrame) -> pd.DataFrame:
    """Quantify how stable selected hyperparameters are across origins.

    Groups by ``model`` and, when present, ``cell_id`` (so per-cell selection is
    not averaged away). Reports, per numeric parameter, the number of origins,
    number of distinct selected values, the modal fraction, and the number of
    switches across origins ordered by ``forecast_origin``.
    """

    has_cell = "cell_id" in hyperparameters.columns
    group_cols = ["model"] + (["cell_id"] if has_cell else [])
    rows = []
    for keys, block in hyperparameters.groupby(group_cols):
        keys_tuple = keys if isinstance(keys, tuple) else (keys,)
        record_base = dict(zip(group_cols, keys_tuple))
        ordered = (
            block.sort_values("forecast_origin")
            if "forecast_origin" in block.columns
            else block
        )
        for parameter in _numeric_parameter_columns(block):
            series = ordered[parameter].astype(float).dropna()
            if series.empty:
                continue
            counts = series.value_counts()
            modal_fraction = float(counts.iloc[0] / series.size)
            n_switches = int((series.values[1:] != series.values[:-1]).sum())
            rows.append(
                {
                    **record_base,
                    "parameter": parameter,
                    "n_origins": int(series.size),
                    "n_unique": int(series.nunique()),
                    "modal_fraction": modal_fraction,
                    "n_switches": n_switches,
                }
            )
    return pd.DataFrame(rows)


def failure_summary(failures: pd.DataFrame) -> pd.DataFrame:
    """Count failures per ``(model, stage)`` with a per-model total."""

    if failures.empty:
        return pd.DataFrame(columns=["model", "stage", "n_failures"])
    rows = []
    for (model, stage), block in failures.groupby(["model", "stage"]):
        rows.append({"model": model, "stage": stage, "n_failures": int(len(block))})
    frame = pd.DataFrame(rows)
    totals = (
        frame.groupby("model")["n_failures"].sum().reset_index()
        .assign(stage="__total__")
    )
    return pd.concat([frame, totals], ignore_index=True).sort_values(
        ["model", "stage"]
    ).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Computational cost
# --------------------------------------------------------------------------- #
_COST_KEYS = (
    "n_workers",
    "n_origins_completed",
    "n_origins_requested",
    "n_outer_origins",
    "n_selection_events",
    "n_target_cells",
    "grid_size",
    "wall_time_seconds",
    "total_fits",
)


# Legacy / producer-specific spellings accepted for a canonical cost key.
_COST_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "total_fits": ("estimated_total_fits",),
}


def _cost_value(metadata: Mapping[str, object], key: str) -> object:
    """Return ``metadata[key]``, falling back to documented alias spellings."""

    if key in metadata:
        return metadata[key]
    for alias in _COST_KEY_ALIASES.get(key, ()):
        if alias in metadata:
            return metadata[alias]
    return np.nan


def computational_cost(metadata_by_model: Mapping[str, Mapping[str, object]]) -> pd.DataFrame:
    """Extract a documented set of cost fields from each model's run metadata.

    Missing keys are reported as ``NaN`` rather than dropped, so the table has a
    stable schema across heterogeneous producers.
    """

    rows = []
    for model, metadata in metadata_by_model.items():
        record: dict[str, object] = {"model": model}
        for key in _COST_KEYS:
            record[key] = _cost_value(metadata, key) if metadata is not None else np.nan
        record["strategy"] = metadata.get("strategy") if metadata else None
        rows.append(record)
    columns = ["model", "strategy", *_COST_KEYS]
    return pd.DataFrame(rows, columns=columns)


# --------------------------------------------------------------------------- #
# Paired statistical inference
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScopeContrast:
    """A named paired comparison between two model labels."""

    name: str
    model_a: str
    model_b: str


def _paired_errors(
    combined: pd.DataFrame, model_a: str, model_b: str, variable: str, horizon: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return per-origin errors for two models on their **matched** origins."""

    obs_key = list(COMMON_SAMPLE_KEY)
    mask = (combined["variable"] == variable) & (combined["horizon"] == int(horizon))
    a = combined[mask & (combined["model"] == model_a)].set_index(obs_key)["error"].astype(float)
    b = combined[mask & (combined["model"] == model_b)].set_index(obs_key)["error"].astype(float)
    common = a.index.intersection(b.index)
    common = common.sort_values()
    return a.loc[common].to_numpy(), b.loc[common].to_numpy(), len(common)


def dm_tests(
    combined: pd.DataFrame,
    contrasts: Sequence[ScopeContrast],
    *,
    loss: str = "squared",
    kernel: str = "bartlett",
    small_sample_correction: bool = True,
    min_observations: int = 8,
) -> pd.DataFrame:
    """Run Diebold-Mariano tests for each contrast and target cell.

    Losses are ``squared`` (default) or ``absolute`` per-observation errors on
    **paired** origins. The HAC truncation lag is ``horizon - 1`` (overlapping
    ``h``-step forecasts). Within each contrast, p-values across target-horizon
    cells form a family and receive a Holm adjustment; invalid comparisons carry
    ``NaN`` and are excluded from the family size.
    """

    cells = combined[["variable", "horizon"]].drop_duplicates().itertuples(index=False)
    cell_list = [(str(v), int(h)) for v, h in cells]

    records: list[dict[str, object]] = []
    per_contrast_indices: dict[str, list[int]] = {}
    for contrast in contrasts:
        per_contrast_indices.setdefault(contrast.name, [])
        for variable, horizon in cell_list:
            err_a, err_b, n = _paired_errors(
                combined, contrast.model_a, contrast.model_b, variable, horizon
            )
            if n == 0:
                result = DMResult(
                    n=0, horizon=horizon, hac_lag=horizon - 1, kernel=kernel,
                    mean_loss_differential=float("nan"), dm_statistic=None,
                    p_value=None, small_sample_corrected=False, valid=False,
                    reason="non-paired comparison: no shared origins for this cell.",
                )
            else:
                if loss == "squared":
                    la, lb = err_a ** 2, err_b ** 2
                elif loss == "absolute":
                    la, lb = np.abs(err_a), np.abs(err_b)
                else:
                    raise ReportingError(f"unknown loss {loss!r}; expected squared|absolute.")
                result = diebold_mariano(
                    la, lb, horizon=horizon, kernel=kernel,
                    small_sample_correction=small_sample_correction,
                    min_observations=min_observations,
                )
            idx = len(records)
            per_contrast_indices[contrast.name].append(idx)
            records.append(
                {
                    "comparison": contrast.name,
                    "model_a": contrast.model_a,
                    "model_b": contrast.model_b,
                    "variable": variable,
                    "horizon": horizon,
                    "n": result.n,
                    "loss": loss,
                    "mean_loss_differential": result.mean_loss_differential,
                    "dm_statistic": result.dm_statistic,
                    "p_value": result.p_value,
                    "hac_lag": result.hac_lag,
                    "kernel": result.kernel,
                    "small_sample_corrected": result.small_sample_corrected,
                    "valid": result.valid,
                    "reason": result.reason,
                }
            )

    # Holm-adjust within each contrast family.
    frame = pd.DataFrame(records)
    frame["holm_p_value"] = np.nan
    for name, indices in per_contrast_indices.items():
        p_values = [records[i]["p_value"] for i in indices]
        adjusted = holm_adjust(p_values)
        for i, adj in zip(indices, adjusted):
            frame.loc[i, "holm_p_value"] = adj
    return frame


def bootstrap_intervals(
    combined: pd.DataFrame,
    contrasts: Sequence[ScopeContrast],
    *,
    loss: str = "squared",
    method: str = "moving_block",
    block_length: float = 4,
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int | None = 0,
) -> pd.DataFrame:
    """Block-bootstrap CIs for the mean loss differential of each contrast/cell."""

    cells = combined[["variable", "horizon"]].drop_duplicates().itertuples(index=False)
    cell_list = [(str(v), int(h)) for v, h in cells]

    rows: list[dict[str, object]] = []
    for contrast in contrasts:
        for variable, horizon in cell_list:
            err_a, err_b, n = _paired_errors(
                combined, contrast.model_a, contrast.model_b, variable, horizon
            )
            base = {
                "comparison": contrast.name,
                "model_a": contrast.model_a,
                "model_b": contrast.model_b,
                "variable": variable,
                "horizon": horizon,
                "loss": loss,
                "method": method,
                "block_length": float(block_length),
                "n_boot": int(n_boot),
                "seed": seed,
            }
            if n < 2:
                rows.append({**base, "n": n, "mean_diff": float("nan"),
                             "ci_lower": float("nan"), "ci_upper": float("nan"),
                             "valid": False})
                continue
            if loss == "squared":
                diff = err_a ** 2 - err_b ** 2
            elif loss == "absolute":
                diff = np.abs(err_a) - np.abs(err_b)
            else:
                raise ReportingError(f"unknown loss {loss!r}; expected squared|absolute.")
            result = bootstrap_ci(
                diff, method=method, block_length=block_length,
                n_boot=n_boot, confidence=confidence, seed=seed,
            )
            rows.append({**base, "n": n, "mean_diff": result.statistic,
                         "ci_lower": result.ci_lower, "ci_upper": result.ci_upper,
                         "valid": True})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Standard contrasts derived from tags
# --------------------------------------------------------------------------- #
def standard_scope_contrasts(combined: pd.DataFrame) -> list[ScopeContrast]:
    """Derive the paper's required contrasts from the model tags.

    Produces, where the corresponding models are present:

    1. forecast-loss pooled vs native selection,
    2/3/4. horizon / variable / variable_horizon vs forecast-loss pooled,
    5. variable_horizon vs the better marginal (variable or horizon),
    6. direct vs iterated ridge under the same scope.

    Each contrast pairs the two model *labels*; contrast #5 is represented
    against both marginals so the Holm family covers the related comparisons.
    """

    tags = combined[[
        "model", "family", "size", "scope", "selection", "forecast_method"
    ]].drop_duplicates()

    def _find(**criteria) -> list[str]:
        mask = pd.Series(True, index=tags.index)
        for key, value in criteria.items():
            mask &= tags[key] == value
        return list(tags.loc[mask, "model"].unique())

    contrasts: list[ScopeContrast] = []

    # Group by system (family, size, forecast_method).
    systems = tags[["family", "size", "forecast_method"]].drop_duplicates()
    for _, sys_row in systems.iterrows():
        fam, size, method = sys_row["family"], sys_row["size"], sys_row["forecast_method"]
        crit = {"family": fam, "size": size, "forecast_method": method}

        pooled = _find(**crit, scope="pooled", selection="forecast_loss")
        native = _find(**crit, scope="native", selection="native")
        horizon = _find(**crit, scope="horizon", selection="forecast_loss")
        variable = _find(**crit, scope="variable", selection="forecast_loss")
        vh = _find(**crit, scope="variable_horizon", selection="forecast_loss")

        label = f"{fam}:{size}:{method}"
        if pooled and native:
            contrasts.append(ScopeContrast(f"pooled_vs_native[{label}]", pooled[0], native[0]))
        if horizon and pooled:
            contrasts.append(ScopeContrast(f"horizon_vs_pooled[{label}]", horizon[0], pooled[0]))
        if variable and pooled:
            contrasts.append(ScopeContrast(f"variable_vs_pooled[{label}]", variable[0], pooled[0]))
        if vh and pooled:
            contrasts.append(ScopeContrast(f"vh_vs_pooled[{label}]", vh[0], pooled[0]))
        if vh and horizon:
            contrasts.append(ScopeContrast(f"vh_vs_horizon[{label}]", vh[0], horizon[0]))
        if vh and variable:
            contrasts.append(ScopeContrast(f"vh_vs_variable[{label}]", vh[0], variable[0]))

    # Direct vs iterated ridge under the same scope.
    ridge_tags = tags[tags["family"].isin(["ridge", "ridge_direct"])]
    for scope in ridge_tags["scope"].unique():
        iterated = list(ridge_tags[(ridge_tags["scope"] == scope) &
                                   (ridge_tags["forecast_method"] == "iterated")]["model"].unique())
        direct = list(ridge_tags[(ridge_tags["scope"] == scope) &
                                 (ridge_tags["forecast_method"] == "direct")]["model"].unique())
        if iterated and direct:
            contrasts.append(
                ScopeContrast(f"direct_vs_iterated[ridge:{scope}]", direct[0], iterated[0])
            )
    return contrasts


# --------------------------------------------------------------------------- #
# Markdown summary
# --------------------------------------------------------------------------- #
def write_comparison_summary(
    output_path: object,
    *,
    gains: pd.DataFrame,
    gain_summary: pd.DataFrame,
    average_rank_table: pd.DataFrame,
    dm_table: pd.DataFrame,
    alignment: AlignmentReport,
) -> str:
    """Write a human-readable ``comparison_summary.md`` and return its text."""

    lines: list[str] = []
    lines.append("# Scope-study comparison summary\n")
    lines.append("## Outer-origin alignment\n")
    if alignment.is_aligned:
        lines.append(f"- All models share {len(alignment.common_origins)} outer origins.\n")
    else:
        lines.append("- **Warning:** models do not share identical outer origins.\n")
        for model, unmatched in alignment.unmatched.items():
            if unmatched:
                lines.append(f"  - `{model}`: {len(unmatched)} unmatched origin(s).\n")
    lines.append(f"- Coverage policy: `{alignment.policy}`.\n")
    lines.append(
        f"- Common-sample keys: {alignment.n_common_keys}; "
        f"excluded keys: {alignment.n_excluded_keys}"
        + (
            f" (coverage {alignment.coverage:.2%}).\n"
            if alignment.coverage == alignment.coverage
            else ".\n"
        )
    )
    lines.append(
        "- All loss tables, ranks, scope gains and paired tests use the same "
        "cell-wise common sample.\n"
    )

    lines.append("\n## Scope-gain decomposition (loss reduction; positive = better)\n")
    if gain_summary.empty:
        lines.append("- No scope gains available.\n")
    else:
        lines.append("| gain | n | average | median | worst deterioration | share improved |\n")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |\n")
        for _, row in gain_summary.iterrows():
            lines.append(
                f"| {row['gain']} | {int(row['n_cells'])} | {row['average_gain']:.4g} | "
                f"{row['median_gain']:.4g} | {row['worst_deterioration']:.4g} | "
                f"{row['proportion_improved']:.2%} |\n"
            )

    lines.append("\n## Average ranks (lower is better)\n")
    for _, row in average_rank_table.iterrows():
        lines.append(f"- `{row['model']}`: {row['average_rank']:.3f} over {int(row['n_cells'])} cells\n")

    lines.append("\n## Diebold-Mariano tests (paired outer errors)\n")
    valid = dm_table[dm_table["valid"]] if not dm_table.empty else dm_table
    n_valid = int(len(valid))
    n_total = int(len(dm_table))
    lines.append(f"- {n_valid} of {n_total} comparisons produced a valid p-value.\n")
    if n_total - n_valid > 0:
        lines.append(
            f"- {n_total - n_valid} comparison(s) were refused a p-value "
            "(zero-variance, too few paired origins, or non-paired cells).\n"
        )

    text = "".join(lines)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(text, encoding="utf-8")
    return text
