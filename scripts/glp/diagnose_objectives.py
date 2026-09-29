"""Diagnose the hyperparameter objectives on a few real-time samples.

Writes to ``--output-dir``:

* ``profiles.csv`` / ``profiles_<origin>.png`` -- every ladder objective along
  one hyperparameter at a time (others at the marginal-likelihood mode):
  lambda, theta, miu and a common scale factor on psi. Shows how sharply each
  objective identifies each hyperparameter.
* ``mc_noise.csv`` -- Monte Carlo standard deviation of the draw-based
  objectives at the mode across seeds, next to the objective's range along the
  profiles (signal-to-noise).
* ``roughness.csv`` -- the original ``mango_rmse`` predictive-mean objective
  (covbayesvar draws, fixed seed) versus the common-random-numbers ladder RMSE
  on a fine theta grid: mean absolute second difference relative to range.
* ``seed_dispersion.csv`` -- hyperparameters selected by the original Mango
  RMSE optimizer and by the ladder optimizer under several seeds, with their
  in-window objective and achieved log posterior.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from glp_hyperparameter_optimization.config import GLP_LAGS, GLP_REALTIME_PANEL_PATH, MODEL_SIZES, resolve_project_path  # noqa: E402
from glp_hyperparameter_optimization.data_utils import build_glp_estimation_matrix, load_glp_realtime_panel  # noqa: E402
from glp_hyperparameter_optimization.glp_model import (  # noqa: E402
    _build_rmse_origins,
    _rmse_eval_origins,
    _rmse_objective,
    glp_find_mode,
    hyper_to_natural_vector,
    natural_vector_to_hyper,
    prepare_glp_context,
    update_hyperparameters_mango_rmse,
)
from glp_hyperparameter_optimization.objectives import (  # noqa: E402
    FastGLP,
    LadderSpec,
    build_loss,
    glp_log_posterior_unbounded,
    select_ladder_hyperparameters,
)

warnings.filterwarnings("ignore")

PROFILE_SPECS = [
    LadderSpec("logscore_joint_1", 1, 12),
    LadderSpec("logscore_joint", 4, 12),
    LadderSpec("logscore_marginal", 4, 12),
    LadderSpec("rmse", 4, 12),
    LadderSpec("rmse", 4, None),
]
GRIDS = {
    "lambda": np.geomspace(0.02, 5.0, 25),
    "theta": np.geomspace(0.01, 50.0, 25),
    "miu": np.geomspace(0.01, 50.0, 25),
    "psi_scale": np.geomspace(0.1, 10.0, 25),
}


def _with(vec: np.ndarray, ctx, parameter: str, value: float) -> np.ndarray:
    hyper = natural_vector_to_hyper(vec, ctx)
    if parameter == "psi_scale":
        hyper["psi"] = (np.asarray(hyper["psi"]) * value).tolist()
    else:
        hyper[parameter] = value
    return hyper_to_natural_vector(hyper, ctx)


def profiles(y, ctx, ctx0, mode_vec, origin, n_draws, target) -> pd.DataFrame:
    fast = FastGLP.from_context(y, ctx0)
    losses = {"mdd": lambda v: -glp_log_posterior_unbounded(y, ctx, v)}
    for spec in PROFILE_SPECS:
        spec = LadderSpec(spec.objective, spec.horizon, spec.window, (target,), n_draws, 0)
        losses[spec.label()] = build_loss(fast, spec)
    rows = []
    for parameter, grid in GRIDS.items():
        for value in grid:
            vec = _with(mode_vec, ctx, parameter, float(value))
            for label, loss in losses.items():
                rows.append({"origin": origin, "parameter": parameter, "value": value, "objective": label, "loss": loss(vec)})
    frame = pd.DataFrame(rows)
    at_mode = {label: loss(mode_vec) for label, loss in losses.items()}
    frame["loss_minus_mode"] = frame["loss"] - frame["objective"].map(at_mode)
    return frame


def plot_profiles(frame: pd.DataFrame, mode_hyper: dict, path: Path) -> None:
    objectives = list(dict.fromkeys(frame["objective"]))
    fig, axes = plt.subplots(len(objectives), len(GRIDS), figsize=(16, 2.4 * len(objectives)), sharex="col")
    for i, objective in enumerate(objectives):
        for j, parameter in enumerate(GRIDS):
            ax = axes[i, j]
            sub = frame[(frame.objective == objective) & (frame.parameter == parameter)]
            ax.plot(sub["value"], sub["loss_minus_mode"], color="#1f618d", lw=1.6)
            ax.axhline(0, color="#999", lw=0.8, ls="--")
            ref = 1.0 if parameter == "psi_scale" else mode_hyper[parameter]
            ax.axvline(ref, color="#c0392b", lw=0.8)
            ax.set_xscale("log")
            if i == 0:
                ax.set_title(parameter)
            if j == 0:
                ax.set_ylabel(objective, fontsize=8)
    fig.suptitle("Objective minus value at the marginal-likelihood mode (red line = mode); lower is better")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def mc_noise(y, ctx0, mode_vec, origin, n_draws, target, prof: pd.DataFrame) -> pd.DataFrame:
    fast = FastGLP.from_context(y, ctx0)
    rows = []
    for spec in PROFILE_SPECS[1:4]:
        values = [
            build_loss(fast, LadderSpec(spec.objective, spec.horizon, spec.window, (target,), n_draws, seed))(mode_vec)
            for seed in range(10)
        ]
        label = LadderSpec(spec.objective, spec.horizon, spec.window).label()
        span = prof[prof.objective == label].groupby("parameter")["loss"].agg(lambda s: s.max() - s.min())
        rows.append(
            {
                "origin": origin,
                "objective": label,
                "n_draws": n_draws,
                "mc_sd": float(np.std(values, ddof=1)),
                "range_theta": float(span.get("theta", np.nan)),
                "range_miu": float(span.get("miu", np.nan)),
                "range_lambda": float(span.get("lambda", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def roughness(y, ctx, ctx0, mode_vec, origin, target, lags, n_obj_draws) -> pd.DataFrame:
    """Original objective (covbayesvar draws, per-origin fixed seed) vs ladder RMSE."""
    ks = _rmse_eval_origins(y.shape[0], 4, 12, random=False, min_t=None, random_seed=None)
    old_origins = _build_rmse_origins(y, lags, ks, 4, {"hyperpriors": 1})
    old = _rmse_objective(old_origins, [target], 4, 4, ctx, n_obj_draws=n_obj_draws, seed_base=0)
    new = build_loss(FastGLP.from_context(y, ctx0), LadderSpec("rmse", 4, 12, (target,), n_obj_draws, 0))
    hyper = natural_vector_to_hyper(mode_vec, ctx)
    grid = np.geomspace(max(0.05, hyper["theta"] / 4), min(50.0, hyper["theta"] * 4), 40)
    rows = []
    for value in grid:
        vec = _with(mode_vec, ctx, "theta", float(value))
        params = {"lam": vec[0], "theta": vec[-2], "miu": vec[-1], **{f"psi_{i}": vec[1 + i] for i in range(ctx.n)}}
        rows.append({"origin": origin, "theta": value, "old_objective": old(**params), "new_objective": new(vec)})
    frame = pd.DataFrame(rows)
    summary = []
    for column in ("old_objective", "new_objective"):
        values = frame[column].to_numpy()
        rough = np.mean(np.abs(np.diff(values, 2))) / max(values.max() - values.min(), 1e-12)
        summary.append({"origin": origin, "objective": column, "n_obj_draws": n_obj_draws, "roughness": rough,
                        "range": values.max() - values.min()})
    return frame, pd.DataFrame(summary)


def seed_dispersion(y, ctx, mode, origin, codes, target_code, lags, seeds, n_draws) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        old = update_hyperparameters_mango_rmse(
            y, lags, model_codes=codes, var_of_interest=[target_code], H=4, h_eval=4, n_eval=12,
            n_obj_draws=1, init_points=10, n_iter=20, njobs=1, opt_seed=seed,
        )
        new = select_ladder_hyperparameters(
            y, lags, LadderSpec("rmse", 4, 12, (codes.index(target_code),), n_draws, seed), mdd_mode=mode, n_starts=4
        )
        for method, hyper in (("mango_rmse (mode forecast)", old), ("ladder rmse_h4_w12", new)):
            vec = hyper_to_natural_vector(hyper, ctx)
            rows.append(
                {
                    "origin": origin,
                    "method": method,
                    "seed": seed,
                    "lambda": hyper["lambda"],
                    "theta": hyper["theta"],
                    "miu": hyper["miu"],
                    "psi_geo_mean_ratio": float(np.exp(np.mean(np.log(np.asarray(hyper["psi"]) / np.asarray(mode["psi"]))))),
                    "log_posterior": glp_log_posterior_unbounded(y, ctx, vec),
                    "objective_value": hyper.get("objective_value", np.nan),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-size", default="small", choices=MODEL_SIZES)
    parser.add_argument("--origins", default="2002-03-31,2008-03-31,2014-03-31,2019-03-31")
    parser.add_argument("--target", default="GDP")
    parser.add_argument("--n-draws", type=int, default=200)
    parser.add_argument("--seeds", default="1,2,3,4,5,6")
    parser.add_argument("--skip-seeds", action="store_true", help="Skip the (slow) optimizer seed-dispersion study.")
    parser.add_argument("--panel-path", type=Path, default=GLP_REALTIME_PANEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/glp/diagnostics"))
    args = parser.parse_args()

    out = resolve_project_path(args.output_dir) / args.model_size
    out.mkdir(parents=True, exist_ok=True)
    panel = load_glp_realtime_panel(resolve_project_path(args.panel_path))
    seeds = [int(s) for s in args.seeds.split(",") if s]
    collected: dict[str, list[pd.DataFrame]] = {k: [] for k in ("profiles", "mc_noise", "roughness", "roughness_grid", "seeds")}
    for origin in [o.strip() for o in args.origins.split(",") if o.strip()]:
        y, codes, _ = build_glp_estimation_matrix(panel, pd.Timestamp(origin), args.model_size)
        target = codes.index(args.target)
        ctx = prepare_glp_context(y, GLP_LAGS)
        ctx0 = prepare_glp_context(y, GLP_LAGS, hyperpriors=0)
        mode = glp_find_mode(ctx)
        mode_vec = hyper_to_natural_vector(mode, ctx)
        prof = profiles(y, ctx, ctx0, mode_vec, origin, args.n_draws, target)
        plot_profiles(prof, mode, out / f"profiles_{origin}.png")
        collected["profiles"].append(prof)
        collected["mc_noise"].append(mc_noise(y, ctx0, mode_vec, origin, args.n_draws, target, prof))
        grid, summary = roughness(y, ctx, ctx0, mode_vec, origin, target, GLP_LAGS, 50)
        collected["roughness_grid"].append(grid)
        collected["roughness"].append(summary)
        if not args.skip_seeds:
            collected["seeds"].append(seed_dispersion(y, ctx, mode, origin, codes, args.target, GLP_LAGS, seeds, args.n_draws))
        print(f"[diagnostics] {origin} done", flush=True)
    for name, frames in collected.items():
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(out / f"{name}.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
