"""Typed configuration for the GLP hyperparameter search space.

This module turns the previously implicit search-space construction (the
``make_param_space`` / ``_params_to_natural`` pair driven by context flags) into
an explicit, serializable :class:`GLPSearchConfig`. The configuration decides
which hyperparameters the optimizer searches, how psi is handled when it is not
searched, and records enough metadata for run provenance to distinguish a full
search from a reduced one *without parsing command strings*.

Design notes
------------
* The covbayesvar workhorses always consume the full natural vector
  ``[lambda, psi..., theta, miu]`` when psi is active in the estimation context
  (``mn['psi'] == 1``). "Reduced" mode therefore keeps psi *in the model* but
  injects fixed psi values and removes psi from the optimizer's search
  dimensions -- genuinely excluding psi from the optimizer.
* Natural-parameter conversion, bound validation, and the psi log-multiplier map
  are reused from :mod:`.glp_model` rather than reimplemented here.
* **Leakage safety.** Any quantity derived from an estimation context's data
  (the AR(1) residual scale ``ctx.SS``, and the ``ctx.MIN``/``ctx.MAX`` psi
  bounds derived from it) is context-specific. A :class:`ResolvedGLPSearch`
  therefore binds such quantities to *one* context and must never be reused
  across a different one. When scoring inner validation folds, resolve the
  configuration against each fold's own training context -- use
  :meth:`GLPSearchConfig.fold_resolving_to_natural` rather than pre-binding a
  single outer-sample :meth:`ResolvedGLPSearch.to_natural`. Reusing an outer
  resolution injects outer-sample (and hence inner-holdout) information into the
  prior scale used to forecast that same holdout.
* **Fold stability.** With ``psi_parameterization='ss_log_multiplier'`` every
  optimizer coordinate has the same interpretation in every fold (a log
  multiplier of *that fold's* ``SS``), so the search domain is a fixed box and
  the feasible set does not depend on fold composition. With ``'absolute'`` the
  psi coordinates are raw prior scales whose admissible range is
  context-dependent, so a single coordinate means different things in different
  folds and candidates feasible in one fold can be rejected in another.
* alpha optimization is intentionally *not* added in this stage. The backend
  holds alpha fixed (``MNalpha=0``); adding an alpha search would require a
  separate, isolated change to the estimation context and is documented here as
  a possible later extension only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .config import GLP_PARAM_SPACE_BOUNDS
from .glp_model import (
    PSI_LOG_MULTIPLIER_BOUNDS,
    GLPContext,
    InvalidHyperparameterError,
    _psi_enabled,
    _psi_from_log_multipliers,
    _validate_natural_vector,
)

_SCALAR_DEFAULTS = {"lambda": 0.2, "theta": 1.0, "miu": 1.0}


class GLPSearchConfigError(ValueError):
    """Raised when a GLP search-space configuration is invalid or infeasible."""


def _check_bounds(bounds: Mapping[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    checked: dict[str, tuple[float, float]] = {}
    for key in ("lambda", "theta", "miu"):
        if key not in bounds:
            raise GLPSearchConfigError(f"bounds is missing required key {key!r}.")
        lower, upper = bounds[key]
        lower_f = float(lower)
        upper_f = float(upper)
        if not (np.isfinite(lower_f) and np.isfinite(upper_f)):
            raise GLPSearchConfigError(f"bounds[{key!r}] must be finite.")
        if lower_f <= 0.0:
            raise GLPSearchConfigError(
                f"bounds[{key!r}] lower bound must be positive, got {lower_f}."
            )
        if not lower_f < upper_f:
            raise GLPSearchConfigError(
                f"bounds[{key!r}] must satisfy lower < upper, got ({lower_f}, {upper_f})."
            )
        checked[key] = (lower_f, upper_f)
    return checked


def _validate_fixed_psi_values(values: Sequence[float]) -> tuple[float, ...]:
    array = np.asarray(values, dtype=float).ravel()
    if array.size == 0:
        raise GLPSearchConfigError("fixed_psi_values must be non-empty.")
    if not np.all(np.isfinite(array)):
        raise GLPSearchConfigError("fixed_psi_values must all be finite.")
    if np.any(array <= 0.0):
        raise GLPSearchConfigError("fixed_psi_values must all be strictly positive.")
    return tuple(float(v) for v in array)


@dataclass(frozen=True)
class GLPSearchConfig:
    """Explicit configuration of the GLP optimizer search space.

    Parameters
    ----------
    optimize_lambda, optimize_theta, optimize_miu, optimize_psi:
        Which hyperparameters the optimizer searches. Search dimensions are
        derived *only* from these flags (and the estimation context's active
        prior components), never inferred indirectly from unrelated flags.
    fixed_psi_source:
        When ``optimize_psi`` is ``False``, where the fixed psi comes from:
        ``"context_ss"`` (the training-only AR residual scale, ``ctx.SS``) or
        ``"supplied"`` (``fixed_psi_values``). An MDD-mode psi vector may be
        passed as supplied values; this never triggers an implicit
        marginal-likelihood optimization.
    fixed_psi_values:
        Explicit positive, finite psi values used when
        ``fixed_psi_source == "supplied"``.
    bounds:
        Natural-parameter bounds for lambda/theta/miu.
    psi_parameterization:
        The psi transformation when psi *is* optimized: ``"absolute"`` (search
        psi directly) or ``"ss_log_multiplier"`` (search log multipliers of
        ``ctx.SS``). ``"ss_log_multiplier"`` is the fold-stable, leakage-safe
        choice: coordinate ``psi_log_multiplier_i`` always means
        ``psi_i = SS_i * exp(coordinate)`` evaluated on the context being
        scored, so the optimizer's box is the constant
        ``PSI_LOG_MULTIPLIER_BOUNDS`` in every fold. ``"absolute"`` makes the
        search box depend on the resolving context's ``MIN``/``MAX`` psi and is
        therefore neither fold-stable nor safe to pre-bind to an outer sample.
    initial_values:
        Optional supplied initial/fixed values for scalar parameters
        (``lambda``/``theta``/``miu``). Required as a fixed value when a scalar
        is active but not optimized.
    """

    optimize_lambda: bool = True
    optimize_theta: bool = True
    optimize_miu: bool = True
    optimize_psi: bool = True
    fixed_psi_source: str | None = None
    fixed_psi_values: tuple[float, ...] | None = None
    bounds: Mapping[str, tuple[float, float]] = field(default_factory=lambda: dict(GLP_PARAM_SPACE_BOUNDS))
    psi_parameterization: str = "absolute"
    initial_values: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "bounds", _check_bounds(self.bounds))

        if self.psi_parameterization not in ("absolute", "ss_log_multiplier"):
            raise GLPSearchConfigError(
                "psi_parameterization must be 'absolute' or 'ss_log_multiplier', "
                f"got {self.psi_parameterization!r}."
            )

        if self.optimize_psi:
            if self.fixed_psi_source is not None:
                raise GLPSearchConfigError(
                    "fixed_psi_source must be None when optimize_psi is True."
                )
            if self.fixed_psi_values is not None:
                raise GLPSearchConfigError(
                    "fixed_psi_values must be None when optimize_psi is True."
                )
        else:
            if self.fixed_psi_source is None:
                raise GLPSearchConfigError(
                    "optimize_psi is False; a fixed_psi_source ('context_ss' or "
                    "'supplied') is required."
                )
            if self.fixed_psi_source not in ("context_ss", "supplied"):
                raise GLPSearchConfigError(
                    "fixed_psi_source must be 'context_ss' or 'supplied', got "
                    f"{self.fixed_psi_source!r}."
                )
            if self.fixed_psi_source == "supplied":
                if self.fixed_psi_values is None:
                    raise GLPSearchConfigError(
                        "fixed_psi_source='supplied' requires fixed_psi_values."
                    )
                object.__setattr__(
                    self, "fixed_psi_values", _validate_fixed_psi_values(self.fixed_psi_values)
                )
            else:  # context_ss
                if self.fixed_psi_values is not None:
                    raise GLPSearchConfigError(
                        "fixed_psi_values must be None when fixed_psi_source='context_ss'."
                    )

        if self.initial_values is not None:
            normalized: dict[str, float] = {}
            for key, value in self.initial_values.items():
                if key not in ("lambda", "theta", "miu"):
                    raise GLPSearchConfigError(
                        f"initial_values key {key!r} must be one of lambda/theta/miu."
                    )
                fvalue = float(value)
                if not np.isfinite(fvalue):
                    raise GLPSearchConfigError(f"initial_values[{key!r}] must be finite.")
                normalized[key] = fvalue
            object.__setattr__(self, "initial_values", normalized)

        if not (self.optimize_lambda or self.optimize_theta or self.optimize_miu or self.optimize_psi):
            raise GLPSearchConfigError(
                "at least one parameter must be optimized; the search space is empty."
            )

    # -- convenience constructors ------------------------------------------- #
    @classmethod
    def legacy_full(cls, *, psi_parameterization: str = "absolute", **kwargs: Any) -> "GLPSearchConfig":
        """Full/legacy search: optimize lambda, every active psi element, theta, miu."""

        return cls(
            optimize_lambda=True,
            optimize_theta=True,
            optimize_miu=True,
            optimize_psi=True,
            psi_parameterization=psi_parameterization,
            **kwargs,
        )

    @classmethod
    def reduced_lambda_theta_miu(
        cls,
        *,
        fixed_psi_source: str = "context_ss",
        fixed_psi_values: Sequence[float] | None = None,
        **kwargs: Any,
    ) -> "GLPSearchConfig":
        """Recommended reduced main-study search: optimize lambda, theta, miu; hold psi fixed.

        By default psi is held at the training-only AR residual scale
        (``context_ss``). Supply ``fixed_psi_source='supplied'`` with
        ``fixed_psi_values`` (for example an MDD-mode psi vector) to pin psi to
        explicit values instead.
        """

        return cls(
            optimize_lambda=True,
            optimize_theta=True,
            optimize_miu=True,
            optimize_psi=False,
            fixed_psi_source=fixed_psi_source,
            fixed_psi_values=tuple(fixed_psi_values) if fixed_psi_values is not None else None,
            **kwargs,
        )

    # -- classification ----------------------------------------------------- #
    @property
    def mode(self) -> str:
        """Return ``'full'``, ``'reduced'``, or ``'custom'``."""

        if self.optimize_lambda and self.optimize_theta and self.optimize_miu and self.optimize_psi:
            return "full"
        if (
            self.optimize_lambda
            and self.optimize_theta
            and self.optimize_miu
            and not self.optimize_psi
        ):
            return "reduced"
        return "custom"

    # -- serialization ------------------------------------------------------ #
    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "optimize_lambda": self.optimize_lambda,
            "optimize_theta": self.optimize_theta,
            "optimize_miu": self.optimize_miu,
            "optimize_psi": self.optimize_psi,
            "fixed_psi_source": self.fixed_psi_source,
            "fixed_psi_values": list(self.fixed_psi_values) if self.fixed_psi_values is not None else None,
            "bounds": {key: list(value) for key, value in self.bounds.items()},
            "psi_parameterization": self.psi_parameterization,
            "initial_values": dict(self.initial_values) if self.initial_values else None,
            "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "GLPSearchConfig":
        """Reconstruct a config from :meth:`to_dict` output (ignores ``mode``)."""

        return cls(
            optimize_lambda=bool(data["optimize_lambda"]),
            optimize_theta=bool(data["optimize_theta"]),
            optimize_miu=bool(data["optimize_miu"]),
            optimize_psi=bool(data["optimize_psi"]),
            fixed_psi_source=data.get("fixed_psi_source"),
            fixed_psi_values=(
                tuple(data["fixed_psi_values"]) if data.get("fixed_psi_values") is not None else None
            ),
            bounds={key: tuple(value) for key, value in data["bounds"].items()},
            psi_parameterization=str(data.get("psi_parameterization", "absolute")),
            initial_values=data.get("initial_values"),
        )

    # -- resolution against an estimation context --------------------------- #
    def resolve(self, ctx: GLPContext) -> "ResolvedGLPSearch":
        """Resolve the configuration against one estimation context.

        This validates activity of prior components, fixed psi length, and the
        deterministic parameter ordering, and records the search metadata.
        """

        theta_active = int(getattr(ctx, "sur", 0)) == 1
        miu_active = int(getattr(ctx, "noc", 0)) == 1
        psi_active = _psi_enabled(ctx)

        if self.optimize_theta and not theta_active:
            raise GLPSearchConfigError(
                "optimize_theta is True but the single-unit-root (theta) prior is "
                "inactive in this context (sur=0)."
            )
        if self.optimize_miu and not miu_active:
            raise GLPSearchConfigError(
                "optimize_miu is True but the sum-of-coefficients (miu) prior is "
                "inactive in this context (noc=0)."
            )
        if self.optimize_psi and not psi_active:
            raise GLPSearchConfigError(
                "optimize_psi is True but psi is not estimated in this context "
                "(mn['psi'] != 1)."
            )

        optimized_names: list[str] = []
        transformed_bounds: dict[str, tuple[float, float]] = {}
        natural_bounds: dict[str, tuple[float, float]] = {}

        if self.optimize_lambda:
            optimized_names.append("lam")
            lo, hi = self.bounds["lambda"]
            transformed_bounds["lam"] = (lo, hi)
            natural_bounds["lam"] = (lo, hi)

        if psi_active and self.optimize_psi:
            ss = np.ravel(np.asarray(ctx.SS, dtype=float))
            if self.psi_parameterization == "ss_log_multiplier":
                tlo, thi = PSI_LOG_MULTIPLIER_BOUNDS
                for i in range(ctx.n):
                    name = f"psi_log_multiplier_{i}"
                    optimized_names.append(name)
                    transformed_bounds[name] = (float(tlo), float(thi))
                    natural_bounds[name] = (
                        float(ss[i] * np.exp(tlo)),
                        float(ss[i] * np.exp(thi)),
                    )
            else:
                psi_lo = np.ravel(np.asarray(ctx.MIN["psi"], dtype=float))
                psi_hi = np.ravel(np.asarray(ctx.MAX["psi"], dtype=float))
                for i in range(ctx.n):
                    name = f"psi_{i}"
                    optimized_names.append(name)
                    transformed_bounds[name] = (float(psi_lo[i]), float(psi_hi[i]))
                    natural_bounds[name] = (float(psi_lo[i]), float(psi_hi[i]))

        for scalar in ("theta", "miu"):
            optimize_scalar = getattr(self, f"optimize_{scalar}")
            if optimize_scalar:
                optimized_names.append(scalar)
                lo, hi = self.bounds[scalar]
                transformed_bounds[scalar] = (lo, hi)
                natural_bounds[scalar] = (lo, hi)

        if not optimized_names:
            raise GLPSearchConfigError(
                "resolved search space is empty for this context; nothing to optimize."
            )

        fixed_scalars = self._resolve_fixed_scalars(theta_active, miu_active)
        fixed_psi = self._resolve_fixed_psi(ctx, psi_active)

        return ResolvedGLPSearch(
            config=self,
            optimized_names=tuple(optimized_names),
            transformed_bounds=transformed_bounds,
            natural_bounds=natural_bounds,
            fixed_scalars=fixed_scalars,
            fixed_psi=fixed_psi,
            psi_active=psi_active,
            n=int(ctx.n),
        )

    def _resolve_fixed_scalars(self, theta_active: bool, miu_active: bool) -> dict[str, float]:
        initial = dict(self.initial_values or {})
        fixed: dict[str, float] = {}

        if not self.optimize_lambda:
            if "lambda" not in initial:
                raise GLPSearchConfigError(
                    "optimize_lambda is False; a fixed lambda must be supplied via "
                    "initial_values['lambda']."
                )
            fixed["lambda"] = float(initial["lambda"])

        for scalar, active in (("theta", theta_active), ("miu", miu_active)):
            optimize_scalar = getattr(self, f"optimize_{scalar}")
            if optimize_scalar:
                continue
            if active:
                if scalar not in initial:
                    raise GLPSearchConfigError(
                        f"optimize_{scalar} is False and the {scalar} prior is active; "
                        f"a fixed value must be supplied via initial_values[{scalar!r}]."
                    )
                fixed[scalar] = float(initial[scalar])
            else:
                # Inactive prior: value is ignored by the model but must be a valid
                # in-bounds placeholder in the natural vector.
                fixed[scalar] = float(initial.get(scalar, _SCALAR_DEFAULTS[scalar]))
        return fixed

    def _resolve_fixed_psi(self, ctx: GLPContext, psi_active: bool) -> np.ndarray | None:
        if not psi_active or self.optimize_psi:
            return None
        if self.fixed_psi_source == "context_ss":
            return np.ravel(np.asarray(ctx.SS, dtype=float)).astype(float)
        # supplied
        values = np.asarray(self.fixed_psi_values, dtype=float).ravel()
        if values.size != int(ctx.n):
            raise GLPSearchConfigError(
                f"fixed_psi_values has length {values.size} but the context has "
                f"{ctx.n} variables."
            )
        return values

    # -- fold-local resolution --------------------------------------------- #
    @property
    def has_context_dependent_domain(self) -> bool:
        """Whether the optimizer's search box depends on the resolving context.

        ``True`` only when psi is optimized in the ``"absolute"``
        parameterization, whose psi bounds come from ``ctx.MIN``/``ctx.MAX``.
        In that case one optimizer coordinate does *not* have a stable
        interpretation across inner folds.
        """

        return bool(self.optimize_psi and self.psi_parameterization == "absolute")

    def to_natural_for(self, params: Mapping[str, float], ctx: GLPContext) -> np.ndarray:
        """Resolve against ``ctx`` and assemble the natural vector in one step.

        This is the leakage-safe entry point for scoring a validation fold: the
        fixed psi and the psi bounds are taken from ``ctx`` itself, never from a
        wider sample.
        """

        return self.resolve(ctx).to_natural(params, ctx)

    def fold_resolving_to_natural(
        self,
        *,
        reference: "ResolvedGLPSearch | None" = None,
    ) -> Callable[[Mapping[str, float], GLPContext], np.ndarray]:
        """Return a ``(params, ctx) -> natural_vector`` callable resolved per context.

        The returned callable resolves this configuration against *each* context
        it is handed (memoized by context identity) instead of reusing a single
        pre-bound resolution. Every ``SS``-derived quantity -- fixed psi and the
        absolute psi bounds -- therefore comes from that context's own training
        rows only.

        Parameters
        ----------
        reference:
            Optional resolution whose optimized coordinate names define the
            optimizer's parameter space. When supplied, each fold-local
            resolution is checked to expose the *same* coordinate names, so a
            coordinate keeps its documented meaning in every fold.
        """

        cache: dict[int, tuple[Any, ResolvedGLPSearch]] = {}

        def _resolved(ctx: GLPContext) -> ResolvedGLPSearch:
            cached = cache.get(id(ctx))
            if cached is not None and cached[0] is ctx:
                return cached[1]
            resolved = self.resolve(ctx)
            if reference is not None and resolved.optimized_names != reference.optimized_names:
                raise GLPSearchConfigError(
                    "fold-local search resolution exposes coordinates "
                    f"{list(resolved.optimized_names)} but the optimizer searches "
                    f"{list(reference.optimized_names)}; the coordinate meaning is "
                    "not stable across folds."
                )
            cache[id(ctx)] = (ctx, resolved)
            return resolved

        def to_natural(params: Mapping[str, float], ctx: GLPContext) -> np.ndarray:
            return _resolved(ctx).to_natural(params, ctx)

        return to_natural


@dataclass(frozen=True)
class ResolvedGLPSearch:
    """A :class:`GLPSearchConfig` resolved against *one* estimation context.

    The resolution captures data-dependent quantities of that context -- the
    fixed psi (when ``fixed_psi_source='context_ss'``) and the absolute psi
    bounds -- so an instance is only valid for the context it was resolved
    against. Do **not** reuse an outer-sample resolution to score inner
    validation folds; call :meth:`GLPSearchConfig.fold_resolving_to_natural`
    instead.
    """

    config: GLPSearchConfig
    optimized_names: tuple[str, ...]
    transformed_bounds: dict[str, tuple[float, float]]
    natural_bounds: dict[str, tuple[float, float]]
    fixed_scalars: dict[str, float]
    fixed_psi: np.ndarray | None
    psi_active: bool
    n: int

    @property
    def search_dimension(self) -> int:
        """Total number of optimizer search dimensions."""

        return len(self.optimized_names)

    def mango_param_space(self) -> dict[str, Any]:
        """Build the Mango parameter space for the optimized dimensions only."""

        from scipy.stats import uniform

        space: dict[str, Any] = {}
        for name in self.optimized_names:
            lower, upper = self.transformed_bounds[name]
            space[name] = uniform(lower, upper - lower)
        return space

    def to_natural(self, params: Mapping[str, float], ctx: GLPContext) -> np.ndarray:
        """Assemble the full covbayesvar-ordered natural vector for a candidate.

        Optimized coordinates come from ``params``; fixed scalars and fixed psi
        come from the resolved configuration. Works whether psi is optimized or
        fixed, and validates the assembled vector against the context bounds.

        ``ctx`` must be the *same* context this search was resolved against.
        Passing a different context silently mixes that context's data with the
        resolution's ``SS``-derived psi and is a leakage bug; use
        :meth:`GLPSearchConfig.fold_resolving_to_natural` for multi-fold
        scoring. Out-of-bounds candidates raise
        :class:`~.glp_model.InvalidHyperparameterError`; values are never
        silently clipped.
        """

        # lambda
        if self.config.optimize_lambda:
            try:
                vec = [float(params["lam"])]
            except KeyError as exc:
                raise InvalidHyperparameterError("missing optimized coordinate 'lam'.") from exc
        else:
            vec = [self.fixed_scalars["lambda"]]

        # psi block (only present when psi is active in the context)
        if _psi_enabled(ctx):
            if self.config.optimize_psi:
                if self.config.psi_parameterization == "ss_log_multiplier":
                    multiplier_keys = [f"psi_log_multiplier_{i}" for i in range(ctx.n)]
                    if not all(key in params for key in multiplier_keys):
                        raise InvalidHyperparameterError(
                            "incomplete psi log-multiplier candidate."
                        )
                    psi = _psi_from_log_multipliers(
                        [float(params[key]) for key in multiplier_keys], ctx
                    )
                    vec.extend(psi.astype(float).tolist())
                else:
                    try:
                        vec.extend(float(params[f"psi_{i}"]) for i in range(ctx.n))
                    except KeyError as exc:
                        raise InvalidHyperparameterError(
                            "incomplete absolute psi candidate."
                        ) from exc
            else:
                if self.fixed_psi is None or self.fixed_psi.size != ctx.n:
                    raise InvalidHyperparameterError(
                        "fixed psi is unavailable or has the wrong length for this context."
                    )
                vec.extend(float(v) for v in self.fixed_psi)

        # theta, miu (always present in the natural vector)
        for scalar in ("theta", "miu"):
            if getattr(self.config, f"optimize_{scalar}"):
                try:
                    vec.append(float(params[scalar]))
                except KeyError as exc:
                    raise InvalidHyperparameterError(
                        f"missing optimized coordinate {scalar!r}."
                    ) from exc
            else:
                vec.append(self.fixed_scalars[scalar])

        return _validate_natural_vector(vec, ctx)

    def metadata(self) -> dict[str, object]:
        """Return run-provenance metadata for this resolved search."""

        fixed_parameters: dict[str, object] = dict(self.fixed_scalars)
        if self.fixed_psi is not None:
            fixed_parameters["psi"] = [float(v) for v in self.fixed_psi]
            fixed_parameters["psi_source"] = self.config.fixed_psi_source

        return {
            "mode": self.config.mode,
            "optimized_parameters": list(self.optimized_names),
            "fixed_parameters": fixed_parameters,
            "transformed_bounds": {k: list(v) for k, v in self.transformed_bounds.items()},
            "natural_bounds": {k: list(v) for k, v in self.natural_bounds.items()},
            "search_dimension": self.search_dimension,
            "psi_parameterization": self.config.psi_parameterization,
            "psi_active": self.psi_active,
            "n_variables": self.n,
        }


__all__ = [
    "GLPSearchConfig",
    "GLPSearchConfigError",
    "ResolvedGLPSearch",
]
