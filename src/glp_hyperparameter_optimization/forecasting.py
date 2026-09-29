"""Recursive real-time out-of-sample forecasting engine for the GLP study.

For every real-time quarterly origin the engine (1) selects hyperparameters with
the requested strategy, (2) draws the full MCMC predictive density, (3) forecasts
1..8 quarters ahead and (4) scores against a fixed later evaluation vintage.

Predictive densities:
* ``paper``            -- random-walk Metropolis over ``[lambda, theta, miu]``
  (integrates over hyperparameter uncertainty, the GLP hierarchical density).
* ``mango_mdd``        -- hyperparameters fixed at the MDD/posterior optimum,
  beta/sigma drawn from the conditional posterior.
* ``mango_rmse``       -- hyperparameters fixed at the rolling-RMSE optimum.
* ``mango_rmse_random``-- hyperparameters fixed at the random-origin RMSE optimum.
* ``paper_plugin``     -- hyperparameters fixed at the GLP marginal-likelihood
  mode (isolates integrating out the hyperparameters from the ``paper`` density).
* ``ladder``           -- hyperparameters fixed at the optimum of one of the
  forecast-based "ladder" objectives in :mod:`objectives`.

Every fixed-hyperparameter density is simulated with the vectorized conjugate
posterior of :mod:`objectives` (same distribution as the covbayesvar draws).
Each forecast row stores the real-time jump-off and the evaluation-vintage
value of the jump-off quarter, so scores can be computed on changes from the
jump-off (see :mod:`scoring`), plus draw-based CRPS / log score.

The recursive origins are processed in parallel with a process pool, mirroring the
Slurm-aware parallelization of the Schorfheide-Song workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd

from .config import (
    EVAL_HORIZONS_QUARTERS,
    GLP_ACTUAL_VINTAGE,
    GLP_FORECAST_END,
    GLP_FORECAST_START,
    GLP_LAGS,
    GLP_MCMC_CONST,
    GLP_REALTIME_PANEL_PATH,
    MAX_FORECAST_HORIZON_QUARTERS,
    MODEL_SIZES,
    forecast_origin_dates,
    model_codes,
    model_series,
    resolve_project_path,
)
from .data_utils import (
    build_glp_actual_frame,
    build_glp_estimation_matrix,
    load_glp_realtime_panel,
)
from .glp_model import (
    glp_find_mode,
    glp_metropolis_forecast_draws,
    hyper_to_natural_vector,
    prepare_glp_context,
    update_hyperparameters_mango,
    update_hyperparameters_mango_rmse,
    update_hyperparameters_mango_rmse_random,
)
from .objectives import (
    LadderSpec,
    fast_forecast_draws,
    glp_log_posterior_unbounded,
    select_ladder_hyperparameters,
)
from .scoring import adjusted_actual, crps_sample, log_score_gaussian

STRATEGIES = ("paper", "mango_mdd", "mango_rmse", "mango_rmse_random", "paper_plugin", "ladder")
ONE_TIME_OPTIMIZATION_STRATEGIES = {"mango_rmse", "mango_rmse_random"}
# Strategies whose selection is an optimization that may be re-run only every
# ``reselect_every`` origins (paper-type strategies always re-select).
RESELECTABLE_STRATEGIES = {"mango_mdd", "mango_rmse", "mango_rmse_random", "ladder"}

# Default modest MCMC size for the recursive experiment (raise via CLI toward the
# paper's 20,000 draws for fully paper-faithful densities).
DEFAULT_MCMC_DRAWS = 2000
DEFAULT_MCMC_DISCARD = 1000


# --------------------------------------------------------------------------- #
# CLI / parallel helpers (self-contained; no cross-workflow coupling).
# --------------------------------------------------------------------------- #
def parse_csv_list(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_csv_int_list(value: str | None, default: list[int]) -> list[int]:
    if not value:
        return list(default)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_positive_int(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"\d+", value)
    if not match:
        return None
    parsed = int(match.group())
    return parsed if parsed > 0 else None


def detect_slurm_parallel_slots() -> int | None:
    for name in ("SLURM_NTASKS", "SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE", "SLURM_CPUS_PER_TASK"):
        parsed = parse_positive_int(os.getenv(name))
        if parsed is not None:
            return parsed
    return None


def resolve_parallel_settings(
    n_origins: int,
    requested_n_workers: int | None,
    requested_optimization_njobs: int | None,
) -> tuple[int, int]:
    slurm_slots = detect_slurm_parallel_slots()
    default_slots = slurm_slots or 1
    n_workers = requested_n_workers if requested_n_workers and requested_n_workers > 0 else default_slots
    n_workers = max(1, min(n_workers, max(1, n_origins)))
    if requested_optimization_njobs and requested_optimization_njobs > 0:
        optimization_njobs = requested_optimization_njobs
    elif slurm_slots:
        optimization_njobs = max(1, slurm_slots // n_workers)
    else:
        optimization_njobs = 1
    return n_workers, optimization_njobs


# --------------------------------------------------------------------------- #
# Progress reporting helpers.
# --------------------------------------------------------------------------- #
def _log_progress(message: str) -> None:
    """Print a timestamped progress line to stderr, cooperating with an active
    tqdm bar when one is present."""
    stamped = f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S UTC}] {message}"
    try:
        from tqdm.auto import tqdm

        tqdm.write(stamped, file=sys.stderr)
    except Exception:  # pragma: no cover - tqdm is a declared dependency
        print(stamped, file=sys.stderr, flush=True)


def _iter_with_progress(
    iterator: Iterable[tuple[str, dict[str, Any]]],
    *,
    total: int,
    desc: str,
    enabled: bool,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(origin_label, result)`` pairs while driving a tqdm progress bar
    that tracks completed vs. failed origins. Degrades to a plain pass-through
    when progress is disabled or tqdm is unavailable."""
    if not enabled:
        yield from iterator
        return
    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover - tqdm is a declared dependency
        yield from iterator
        return

    completed = 0
    failed = 0
    with tqdm(total=total, desc=desc, unit="origin") as bar:
        for origin_label, result in iterator:
            if result.get("error"):
                failed += 1
            else:
                completed += 1
            bar.set_postfix(ok=completed, failed=failed, refresh=False)
            bar.update(1)
            yield origin_label, result


# --------------------------------------------------------------------------- #
# Hyperparameter selection and predictive densities.
# --------------------------------------------------------------------------- #
def _optimizer_seed(task: dict[str, Any], origin_date: pd.Timestamp | str | None) -> int | None:
    """Per-origin optimizer seed derived from ``optimization_seed`` (None = unseeded)."""
    base = task.get("optimization_seed")
    if base is None:
        return None
    offset = 0 if origin_date is None else int(pd.Timestamp(origin_date).strftime("%Y%m%d"))
    return (int(base) * 1_000_003 + offset) % (2**31 - 1)


def select_hyperparameters(
    strategy: str, y: np.ndarray, codes: list[str], task: dict[str, Any], origin_date: pd.Timestamp | str | None = None
) -> dict[str, float]:
    lags = task["lags"]
    prior_kwargs = dict(sur=task["sur"], noc=task["noc"], mnpsi=task["mnpsi"], mnalpha=task["mnalpha"], vc=task["vc"])
    opt_seed = _optimizer_seed(task, origin_date)
    if strategy in ("paper", "paper_plugin"):
        ctx = prepare_glp_context(y, lags, hyperpriors=task["hyperpriors"], **prior_kwargs)
        mode = glp_find_mode(ctx)
        return {"lambda": mode["lambda"], "theta": mode["theta"], "miu": mode["miu"], "psi": mode.get("psi")}
    if strategy == "ladder":
        ladder = dict(task["ladder"])
        spec = LadderSpec(
            objective=ladder["objective"],
            horizon=int(ladder.get("horizon", 1)),
            window=ladder.get("window"),
            target_indices=tuple(codes.index(v) for v in task["variables"]),
            n_draws=int(ladder.get("n_draws", 200)),
            seed=int(opt_seed or 0),
            min_t=task.get("min_t"),
        )
        return select_ladder_hyperparameters(
            y,
            lags,
            spec,
            n_starts=int(ladder.get("n_starts", 4)),
            maxfev=ladder.get("maxfev"),
            hyperpriors=task["hyperpriors"],
            **prior_kwargs,
        )
    if strategy == "mango_mdd":
        return update_hyperparameters_mango(
            y,
            lags,
            init_points=task["init_points"],
            n_iter=task["n_iter"],
            njobs=task["optimization_njobs"],
            hyperpriors=task["hyperpriors"],
            opt_seed=opt_seed,
            local_polish=bool(task.get("local_polish", False)),
            **prior_kwargs,
        )
    if strategy == "mango_rmse":
        return update_hyperparameters_mango_rmse(
            y,
            lags,
            model_codes=codes,
            var_of_interest=task["variables"],
            H=task["optimization_horizon_quarters"],
            h_eval=task["optimization_eval_horizon_quarters"],
            n_eval=task["n_eval"],
            min_t=task["min_t"],
            n_obj_draws=task.get("n_obj_draws", 200),
            init_points=task["init_points"],
            n_iter=task["n_iter"],
            njobs=task["optimization_njobs"],
            hyperpriors=task["hyperpriors"],
            opt_seed=opt_seed,
            **prior_kwargs,
        )
    if strategy == "mango_rmse_random":
        return update_hyperparameters_mango_rmse_random(
            y,
            lags,
            model_codes=codes,
            var_of_interest=task["variables"],
            H=task["optimization_horizon_quarters"],
            h_eval=task["optimization_eval_horizon_quarters"],
            n_eval=task["n_eval"],
            min_t=task["min_t"],
            random_seed=task["random_seed"],
            n_obj_draws=task.get("n_obj_draws", 200),
            init_points=task["init_points"],
            n_iter=task["n_iter"],
            njobs=task["optimization_njobs"],
            hyperpriors=task["hyperpriors"],
            opt_seed=opt_seed,
            **prior_kwargs,
        )
    raise ValueError(f"Unsupported strategy: {strategy}")


def predictive_draws(
    strategy: str, ctx, hyper: dict[str, Any], task: dict[str, Any], seed: int | None, y: np.ndarray | None = None
) -> np.ndarray:
    """Return an ``(n_draws, H, n)`` array of simulated forecast levels.

    ``y`` is the raw estimation matrix behind ``ctx`` (required for the
    fixed-hyperparameter strategies, which use the fast conjugate sampler)."""
    horizon = MAX_FORECAST_HORIZON_QUARTERS
    hyper_vector = hyper_to_natural_vector(hyper, ctx)
    if strategy == "paper":
        draws, _ = glp_metropolis_forecast_draws(
            ctx,
            hyper_vector,
            max_horizon=horizon,
            n_draws=task["mcmc_draws"],
            n_discard=task["mcmc_discard"],
            const=task["mcmc_const"],
            seed=seed,
        )
        return draws
    return fast_forecast_draws(
        y,
        ctx,
        hyper_vector,
        max_horizon=horizon,
        n_draws=task["mcmc_draws"],
        seed=seed,
    )


def _forecast_rows(
    strategy: str,
    size: str,
    origin_date: pd.Timestamp,
    last_quarter: pd.Period,
    codes: list[str],
    draws: np.ndarray,
    actual_frame: pd.DataFrame,
    jumpoff: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """One row per horizon/variable with the density summary and its scores.

    Scores (``error_adj``, ``crps``, ``logscore``) compare the forecast with the
    evaluation-vintage outcome re-expressed in the forecast vintage's units
    (:func:`scoring.adjusted_actual`); ``error`` keeps the raw level error.
    """
    rows: list[dict[str, Any]] = []
    quantiles = {"p05": 5, "p16": 16, "median": 50, "p84": 84, "p95": 95}
    is_log = {spec.code: spec.transform == "log" for spec in model_series(size)}
    has_last = last_quarter in actual_frame.index
    for h in range(1, MAX_FORECAST_HORIZON_QUARTERS + 1):
        target_quarter = last_quarter + h
        layer = draws[:, h - 1, :]  # (n_draws, n)
        mean = layer.mean(axis=0)
        std = layer.std(axis=0, ddof=1)
        qvals = {name: np.percentile(layer, pct, axis=0) for name, pct in quantiles.items()}
        actual_vec = np.array(
            [float(actual_frame.at[target_quarter, c]) if target_quarter in actual_frame.index else np.nan for c in codes]
        )
        actual_jump = np.array([float(actual_frame.at[last_quarter, c]) if has_last else np.nan for c in codes])
        jump = np.full(len(codes), np.nan) if jumpoff is None else np.asarray(jumpoff, dtype=float)
        adj = adjusted_actual(actual_vec, actual_jump, jump, [is_log[c] for c in codes])
        finite = np.isfinite(adj)
        crps = np.full(len(codes), np.nan)
        if finite.any():
            crps[finite] = crps_sample(layer[:, finite], adj[finite])
        logscore = log_score_gaussian(mean, std, adj)
        for vi, code in enumerate(codes):
            actual = actual_vec[vi]
            row = {
                "strategy": strategy,
                "model_size": size,
                "forecast_origin": pd.Timestamp(origin_date).strftime("%Y-%m-%d"),
                "target_quarter": str(target_quarter),
                "horizon_quarters": h,
                "variable": code,
                "mean": float(mean[vi]),
                "actual": actual,
                "error": float(mean[vi] - actual) if np.isfinite(actual) else np.nan,
                "transform": "log" if is_log[code] else "lin",
                "jumpoff": float(jump[vi]),
                "actual_jumpoff": float(actual_jump[vi]),
                "actual_adj": float(adj[vi]),
                "error_adj": float(mean[vi] - adj[vi]),
                "std": float(std[vi]),
                "crps": float(crps[vi]),
                "logscore": float(logscore[vi]),
            }
            for name in quantiles:
                row[name] = float(qvals[name][vi])
            rows.append(row)
    return rows


def _run_selection_task(task: dict[str, Any]) -> dict[str, Any]:
    """Select hyperparameters on one origin's real-time sample (no forecasting)."""
    try:
        origin_date = pd.Timestamp(task["origin_date"])
        panel = load_glp_realtime_panel(Path(task["panel_path"]))
        y, codes, _ = build_glp_estimation_matrix(panel, origin_date, task["size"])
        hyper = select_hyperparameters(task["strategy"], y, codes, task, origin_date)
        hyper["selection_origin"] = origin_date.strftime("%Y-%m-%d")
        return {"hyperparameters": hyper, "error": None}
    except Exception as exc:  # pragma: no cover - logged by the caller
        return {"hyperparameters": None, "error": f"{type(exc).__name__}: {exc}"}


class _Checkpoints:
    """Per-origin results saved as soon as they finish, so an interrupted run
    resumes where it stopped instead of losing the whole stage.

    Files live under ``<output_dir>/_checkpoints/<config hash>/<kind>/<origin>.pkl``;
    the hash covers every setting that changes results (not the worker count),
    so a run with different settings never picks up stale checkpoints. They are
    kept after completion, which makes re-running a finished stage instant.
    """

    def __init__(self, output_dir: Path, config: dict[str, Any]):
        digest = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
        self.root = output_dir / "_checkpoints" / digest

    def load(self, kind: str, label: str) -> Any | None:
        path = self.root / kind / f"{label}.pkl"
        if not path.exists():
            return None
        try:
            with path.open("rb") as handle:
                return pickle.load(handle)
        except (OSError, pickle.UnpicklingError, EOFError):
            return None

    def save(self, kind: str, label: str, value: Any) -> None:
        folder = self.root / kind
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / f".{label}.pkl.tmp"
        with tmp.open("wb") as handle:
            pickle.dump(value, handle)
        tmp.replace(folder / f"{label}.pkl")


def _map_tasks(function, tasks: list[dict[str, Any]], n_workers: int) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Yield ``(task, result)`` pairs, in a process pool when ``n_workers > 1``."""
    if n_workers == 1:
        for task in tasks:
            yield task, function(task)
        return
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(function, task): task for task in tasks}
        for future in as_completed(futures):
            yield futures[future], future.result()


def _run_origin_task(task: dict[str, Any]) -> dict[str, Any]:
    try:
        origin_date = pd.Timestamp(task["origin_date"])
        panel = load_glp_realtime_panel(Path(task["panel_path"]))
        actual_frame = build_glp_actual_frame(panel, pd.Timestamp(task["actual_vintage"]), task["size"])
        y, codes, qidx = build_glp_estimation_matrix(panel, origin_date, task["size"])

        hyper = task.get("fixed_hyperparameters") or select_hyperparameters(
            task["strategy"], y, codes, task, origin_date
        )
        ctx = prepare_glp_context(
            y,
            task["lags"],
            hyperpriors=task["hyperpriors"],
            sur=task["sur"],
            noc=task["noc"],
            mnpsi=task["mnpsi"],
            mnalpha=task["mnalpha"],
            vc=task["vc"],
        )
        seed = task.get("seed_base")
        seed = None if seed is None else int(seed) + int(pd.Timestamp(origin_date).value % 100000)
        draws = predictive_draws(task["strategy"], ctx, hyper, task, seed, y)
        rows = _forecast_rows(
            task["strategy"], task["size"], origin_date, qidx[-1], codes, draws, actual_frame, jumpoff=y[-1]
        )

        hyper_record = {
            "forecast_origin": pd.Timestamp(origin_date).strftime("%Y-%m-%d"),
            "strategy": task["strategy"],
            "model_size": task["size"],
            "last_quarter": str(qidx[-1]),
            "n_obs": int(y.shape[0]),
            "lambda": float(hyper["lambda"]),
            "theta": float(hyper["theta"]),
            "miu": float(hyper["miu"]),
        }
        psi_values = hyper.get("psi")
        if psi_values is not None:
            hyper_record["psi"] = json.dumps([float(v) for v in psi_values])
        # Achieved GLP log posterior (log MDD + hyperprior) at the selected
        # hyperparameters on this origin's sample, comparable across strategies.
        hyper_record["log_mdd"] = glp_log_posterior_unbounded(y, ctx, hyper_to_natural_vector(hyper, ctx))
        for key in ("log_mdd_mango", "objective_value", "objective_at_mdd_mode", "objective_nfev", "selection_origin"):
            if hyper.get(key) is not None:
                hyper_record[key] = hyper[key]
        return {"forecast_rows": rows, "hyperparameters": hyper_record, "error": None}
    except Exception as exc:  # pragma: no cover - per-origin failures are logged
        return {"forecast_rows": [], "hyperparameters": {}, "error": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------- #
# Experiment driver.
# --------------------------------------------------------------------------- #
def run_glp_experiment(
    strategy: str,
    size: str,
    output_dir: Path,
    *,
    panel_path: Path = GLP_REALTIME_PANEL_PATH,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    actual_vintage: pd.Timestamp = GLP_ACTUAL_VINTAGE,
    lags: int = GLP_LAGS,
    hyperpriors: int = 1,
    sur: int = 1,
    noc: int = 1,
    mnpsi: int = 1,
    mnalpha: int = 0,
    vc: float = 1.0e7,
    mcmc_draws: int = DEFAULT_MCMC_DRAWS,
    mcmc_discard: int = DEFAULT_MCMC_DISCARD,
    mcmc_const: float = GLP_MCMC_CONST,
    init_points: int = 5,
    n_iter: int = 15,
    optimization_njobs: int | None = None,
    optimization_horizon_quarters: int = 4,
    optimization_eval_horizon_quarters: int | None = None,
    n_eval: int = 3,
    n_obj_draws: int = 200,
    min_t: int | None = None,
    random_seed: int | None = None,
    variables: list[str] | None = None,
    per_origin_selection: bool = False,
    seed_base: int | None = None,
    n_workers: int | None = None,
    show_progress: bool = True,
    optimization_seed: int | None = None,
    local_polish: bool = False,
    ladder: dict[str, Any] | None = None,
    reselect_every: int = 1,
) -> Path:
    output_dir = resolve_project_path(output_dir)
    panel_path = resolve_project_path(panel_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    codes = model_codes(size)
    resolved_variables = variables or [codes[0]]

    available = forecast_origin_dates()
    resolved_start = start if start is not None else available[0]
    resolved_end = end if end is not None else available[-1]
    origins = forecast_origin_dates(resolved_start, resolved_end)

    n_workers_resolved, optimization_njobs_resolved = resolve_parallel_settings(
        len(origins), n_workers, optimization_njobs
    )

    task_template = {
        "strategy": strategy,
        "size": size,
        "panel_path": str(panel_path),
        "actual_vintage": str(pd.Timestamp(actual_vintage).date()),
        "lags": lags,
        "hyperpriors": hyperpriors,
        "sur": sur,
        "noc": noc,
        "mnpsi": mnpsi,
        "mnalpha": mnalpha,
        "vc": vc,
        "mcmc_draws": mcmc_draws,
        "mcmc_discard": mcmc_discard,
        "mcmc_const": mcmc_const,
        "init_points": init_points,
        "n_iter": n_iter,
        "optimization_njobs": optimization_njobs_resolved,
        "optimization_horizon_quarters": optimization_horizon_quarters,
        "optimization_eval_horizon_quarters": optimization_eval_horizon_quarters,
        "n_eval": n_eval,
        "n_obj_draws": n_obj_draws,
        "min_t": min_t,
        "random_seed": random_seed,
        "variables": resolved_variables,
        "seed_base": seed_base,
        "optimization_seed": optimization_seed,
        "local_polish": local_polish,
        "ladder": ladder,
    }
    if strategy == "ladder" and not ladder:
        raise ValueError("The 'ladder' strategy needs a ladder specification (objective, horizon, window, ...).")

    # One-time (shared) hyperparameter selection for the RMSE strategies, using
    # the earliest origin's real-time sample (unless per-origin was requested).
    shared_hyperparameters: dict[str, float] | None = None
    selection_origin: str | None = None
    if len(origins) > 0 and strategy in ONE_TIME_OPTIMIZATION_STRATEGIES and not per_origin_selection:
        if show_progress:
            _log_progress(f"Selecting shared {strategy} hyperparameters on {origins[0]:%Y-%m-%d} ...")
        panel = load_glp_realtime_panel(panel_path)
        y0, codes0, _ = build_glp_estimation_matrix(panel, origins[0], size)
        shared_hyperparameters = select_hyperparameters(strategy, y0, codes0, task_template)
        selection_origin = pd.Timestamp(origins[0]).strftime("%Y-%m-%d")
        if show_progress:
            _log_progress(
                "Selected shared hyperparameters: "
                f"lambda={shared_hyperparameters['lambda']:.4g}, "
                f"theta={shared_hyperparameters['theta']:.4g}, "
                f"miu={shared_hyperparameters['miu']:.4g}."
            )

    checkpoint_config = {
        key: value for key, value in task_template.items() if key not in ("optimization_njobs", "panel_path")
    }
    checkpoint_config.update(
        {"per_origin_selection": per_origin_selection, "reselect_every": int(reselect_every), "shared": shared_hyperparameters}
    )
    checkpoints = _Checkpoints(output_dir, checkpoint_config)

    # Periodic re-selection: optimize only every ``reselect_every`` origins and
    # reuse the latest selection (made with data available at that earlier
    # origin, so there is no look-ahead) for the origins in between.
    fixed_by_origin: dict[str, dict[str, Any]] = {}
    periodic = (
        shared_hyperparameters is None
        and strategy in RESELECTABLE_STRATEGIES
        and int(reselect_every) > 1
        and len(origins) > 0
    )
    if periodic:
        selection_origins = list(origins[:: int(reselect_every)])
        selection_tasks = [{**task_template, "origin_date": o.strftime("%Y-%m-%d")} for o in selection_origins]
        if show_progress:
            _log_progress(f"Selecting {strategy} hyperparameters at {len(selection_tasks)} origins (every {reselect_every}).")
        selected: dict[str, dict[str, Any]] = {}
        pending_selections = []
        for task in selection_tasks:
            cached = checkpoints.load("selection", task["origin_date"])
            if cached is not None:
                selected[task["origin_date"]] = cached
            else:
                pending_selections.append(task)
        if show_progress and len(pending_selections) < len(selection_tasks):
            _log_progress(f"Resuming: {len(selection_tasks) - len(pending_selections)} selections loaded from checkpoints.")
        for task, result in _map_tasks(_run_selection_task, pending_selections, n_workers_resolved):
            if result["error"]:
                raise RuntimeError(f"Hyperparameter selection failed at {task['origin_date']}: {result['error']}")
            selected[task["origin_date"]] = result["hyperparameters"]
            checkpoints.save("selection", task["origin_date"], result["hyperparameters"])
        current: dict[str, Any] | None = None
        for origin in origins:
            label = origin.strftime("%Y-%m-%d")
            current = selected.get(label, current)
            fixed_by_origin[label] = current  # type: ignore[assignment]

    tasks = [
        {
            **task_template,
            "origin_date": origin.strftime("%Y-%m-%d"),
            "fixed_hyperparameters": fixed_by_origin.get(origin.strftime("%Y-%m-%d"), shared_hyperparameters),
        }
        for origin in origins
    ]

    forecast_rows: list[dict[str, Any]] = []
    hyper_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    pending_tasks = []
    for task in tasks:
        cached = checkpoints.load("origin", task["origin_date"])
        if cached is not None:
            forecast_rows.extend(cached["forecast_rows"])
            hyper_rows.append(cached["hyperparameters"])
        else:
            pending_tasks.append(task)
    if show_progress and len(pending_tasks) < len(tasks):
        _log_progress(f"Resuming: {len(tasks) - len(pending_tasks)} of {len(tasks)} origins loaded from checkpoints.")
    tasks = pending_tasks

    iterator = (
        (task["origin_date"], result) for task, result in _map_tasks(_run_origin_task, tasks, n_workers_resolved)
    )

    if show_progress:
        _log_progress(
            f"Running {len(tasks)} forecast origins with {n_workers_resolved} worker(s) [GLP {strategy}/{size}]."
        )

    tracked_iterator = _iter_with_progress(
        iterator, total=len(tasks), desc=f"GLP {strategy}/{size}", enabled=show_progress
    )

    for origin_label, result in tracked_iterator:
        if result["error"]:
            errors.append({"forecast_origin": origin_label, "error": result["error"]})
            continue
        forecast_rows.extend(result["forecast_rows"])
        hyper_rows.append(result["hyperparameters"])
        checkpoints.save("origin", origin_label, result)

    forecasts = pd.DataFrame(forecast_rows)
    if not forecasts.empty:
        forecasts = forecasts.sort_values(["forecast_origin", "variable", "horizon_quarters"])
    hyperparameters = pd.DataFrame(hyper_rows)
    if not hyperparameters.empty:
        hyperparameters = hyperparameters.sort_values("forecast_origin")

    forecasts.to_csv(output_dir / "forecast_panel.csv", index=False)
    hyperparameters.to_csv(output_dir / "selected_hyperparameters.csv", index=False)
    pd.DataFrame(errors).to_csv(output_dir / "failed_origins.csv", index=False)

    metadata = {
        "strategy": strategy,
        "model_size": size,
        "panel_path": str(panel_path),
        "actual_vintage": pd.Timestamp(actual_vintage).strftime("%Y-%m-%d"),
        "lags": lags,
        "variables_of_interest": resolved_variables,
        "hyperpriors": hyperpriors,
        "sur": sur,
        "noc": noc,
        "mnpsi": mnpsi,
        "mnalpha": mnalpha,
        "mcmc_draws": mcmc_draws,
        "mcmc_discard": mcmc_discard,
        "mcmc_const": mcmc_const,
        "optimization_init_points": init_points,
        "optimization_iterations": n_iter,
        "optimization_njobs": optimization_njobs_resolved,
        "optimization_horizon_quarters": optimization_horizon_quarters,
        "optimization_eval_horizon_quarters": optimization_eval_horizon_quarters,
        "n_eval": n_eval,
        "n_obj_draws": n_obj_draws,
        "min_t": min_t,
        "random_seed": random_seed,
        "optimization_seed": optimization_seed,
        "local_polish": local_polish,
        "ladder": ladder,
        "reselect_every": int(reselect_every) if periodic else 1,
        "fixed_draw_sampler": "objectives.fast_forecast_draws",
        "hyperparameters_selected_once": strategy in ONE_TIME_OPTIMIZATION_STRATEGIES and not per_origin_selection,
        "hyperparameter_selection_origin": selection_origin,
        "n_workers": n_workers_resolved,
        "n_origins_requested": len(origins),
        "n_origins_completed": int(hyperparameters.shape[0]),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if show_progress:
        _log_progress(
            f"Completed {len(hyper_rows)}/{len(origins)} origins "
            f"({len(errors)} failed). Outputs written to {output_dir}."
        )
    return output_dir


# --------------------------------------------------------------------------- #
# Argument parsers shared by the strategy scripts.
# --------------------------------------------------------------------------- #
def build_common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--panel-path", type=Path, default=GLP_REALTIME_PANEL_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-size", choices=MODEL_SIZES, default="medium")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--actual-vintage", type=str, default=GLP_ACTUAL_VINTAGE.strftime("%Y-%m-%d"))
    parser.add_argument("--lags", type=int, default=GLP_LAGS)
    parser.add_argument("--mcmc-draws", type=int, default=DEFAULT_MCMC_DRAWS)
    parser.add_argument("--mcmc-discard", type=int, default=DEFAULT_MCMC_DISCARD)
    parser.add_argument("--mcmc-const", type=float, default=GLP_MCMC_CONST)
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Forecast-origin worker count. Defaults to the Slurm allocation when available, otherwise 1.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable the per-origin progress bar and progress logging.",
    )
    return parser


def build_optimizer_parser(description: str) -> argparse.ArgumentParser:
    parser = build_common_parser(description)
    parser.add_argument("--optimization-init-points", type=int, default=5)
    parser.add_argument("--optimization-iterations", type=int, default=15)
    parser.add_argument("--optimization-njobs", type=int, default=None)
    parser.add_argument("--optimization-horizon-quarters", type=int, default=4)
    parser.add_argument(
        "--variables",
        type=str,
        default=None,
        help="Comma-separated model variables used by the RMSE objective (e.g. GDP,DEFL). Defaults to the first variable.",
    )
    parser.add_argument(
        "--per-origin-selection",
        action="store_true",
        help="Re-select the RMSE hyperparameters at every origin instead of once on the first origin.",
    )
    add_selection_arguments(parser)
    return parser


def add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    """Optimizer reproducibility / precision flags shared by every optimizing script."""
    parser.add_argument(
        "--optimization-seed",
        type=int,
        default=None,
        help="Base seed for the hyperparameter optimizers (offset per origin); unseeded when omitted.",
    )
    parser.add_argument(
        "--local-polish",
        action="store_true",
        help="Refine the Mango MDD optimum with Nelder-Mead on the same closed-form objective.",
    )
    parser.add_argument(
        "--reselect-every",
        type=int,
        default=1,
        help="Re-run the hyperparameter optimization only every N origins (reusing the latest selection).",
    )


def parse_dates(start: str | None, end: str | None) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    return (pd.Timestamp(start) if start else None, pd.Timestamp(end) if end else None)


def run_from_namespace(strategy: str, namespace: argparse.Namespace) -> Path:
    start, end = parse_dates(namespace.start, namespace.end)
    kwargs: dict[str, Any] = {
        "strategy": strategy,
        "size": namespace.model_size,
        "output_dir": resolve_project_path(namespace.output_dir),
        "panel_path": resolve_project_path(namespace.panel_path),
        "start": start,
        "end": end,
        "actual_vintage": pd.Timestamp(namespace.actual_vintage),
        "lags": namespace.lags,
        "mcmc_draws": namespace.mcmc_draws,
        "mcmc_discard": namespace.mcmc_discard,
        "mcmc_const": namespace.mcmc_const,
        "seed_base": namespace.seed_base,
        "n_workers": namespace.n_workers,
        "show_progress": not getattr(namespace, "quiet", False),
    }
    if hasattr(namespace, "optimization_init_points"):
        kwargs.update(
            {
                "init_points": namespace.optimization_init_points,
                "n_iter": namespace.optimization_iterations,
                "optimization_njobs": namespace.optimization_njobs,
                "optimization_horizon_quarters": namespace.optimization_horizon_quarters,
                "optimization_eval_horizon_quarters": getattr(namespace, "optimization_eval_horizon_quarters", None),
                "n_eval": getattr(namespace, "optimization_n_eval", 3),
                "n_obj_draws": getattr(namespace, "optimization_n_obj_draws", 200),
                "min_t": getattr(namespace, "optimization_min_t", None),
                "random_seed": getattr(namespace, "optimization_random_seed", None),
                "variables": parse_csv_list(namespace.variables, []),
                "per_origin_selection": getattr(namespace, "per_origin_selection", False),
                "optimization_seed": getattr(namespace, "optimization_seed", None),
                "local_polish": getattr(namespace, "local_polish", False),
                "reselect_every": getattr(namespace, "reselect_every", 1),
            }
        )
    return run_glp_experiment(**kwargs)
