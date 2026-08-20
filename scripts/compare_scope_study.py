"""Assemble a model-independent scope-study comparison from canonical panels.

This CLI reads a JSON manifest describing the forecast panels to compare (each
tagged with its family, selection scope, selection method, forecast method, and
optional model size), loads and aligns them through the explicit adapters in
:mod:`common_hpo.reporting`, and writes the authoritative comparison tables:

    rmse_by_target.csv        mae_by_target.csv        relative_rmse.csv
    average_ranks.csv         scope_gains.csv          scope_gain_summary.csv
    hyperparameter_summary.csv selection_stability.csv failure_summary.csv
    computational_cost.csv    dm_tests.csv             bootstrap_intervals.csv
    common_sample.csv         comparison_summary.md

Every table is built from a *single*, globally restricted frame: the panels are
first reduced to the cell-wise common sample (see
``common_hpo.reporting.restrict_to_common_sample``) and that one frame feeds the
loss tables, the ranks, the scope gains and the paired DM/bootstrap inference.
The "same sample basis" guarantee is therefore mechanical rather than
aspirational -- the per-table restrictions downstream receive the same
``--coverage-policy``/``--min-coverage`` and are therefore exact no-ops.

Under ``--coverage-policy advisory`` the frame is deliberately left
unrestricted: the coverage shortfall is reported in ``common_sample.csv`` but no
table drops observations, so each model is evaluated on its own rows and the
average ranks relax their common-sample precondition accordingly.

The main paper question is answerable directly from ``scope_gains.csv`` without
manually stitching directories.

Manifest format (JSON)::

    {
      "baseline_model": "ridge_pooled",
      "panels": [
        {
          "model": "ridge_pooled",
          "family": "ridge",
          "scope": "pooled",
          "selection": "forecast_loss",
          "forecast_method": "iterated",
          "size": null,
          "native_method": null,
          "dir": "outputs/regularized/scope_pooled"
        },
        ...
      ]
    }

Each panel ``dir`` is expected to contain ``forecast_panel.csv`` and, where
available, ``selected_hyperparameters.csv``, ``failed_origins.csv`` and
``run_metadata.json``. Missing optional files are skipped, not fabricated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd  # noqa: E402

from common_hpo.reporting import (  # noqa: E402
    PanelSpec,
    average_ranks,
    bootstrap_intervals,
    check_origin_alignment,
    combine_panels,
    computational_cost,
    dm_tests,
    failure_summary,
    hyperparameter_summary,
    load_failed_origins,
    load_forecast_panel,
    load_run_metadata,
    load_selected_hyperparameters,
    mae_by_target,
    relative_rmse,
    restrict_to_common_sample,
    rmse_by_target,
    scope_gain_summary,
    scope_gains,
    selection_stability,
    standard_scope_contrasts,
    write_comparison_summary,
)


def _spec_from_entry(entry: dict[str, Any]) -> PanelSpec:
    return PanelSpec(
        model=entry["model"],
        family=entry["family"],
        scope=entry.get("scope", "native"),
        selection=entry.get("selection", "native"),
        forecast_method=entry.get("forecast_method", "native"),
        size=entry.get("size"),
        native_method=entry.get("native_method"),
        path=Path(entry["dir"]) if "dir" in entry else None,
    )


def load_study(manifest: dict[str, Any], *, root: Path) -> dict[str, Any]:
    """Load every panel in the manifest into combined/aux frames."""

    panels: list[pd.DataFrame] = []
    hyperparameters: list[pd.DataFrame] = []
    failures: list[pd.DataFrame] = []
    metadata_by_model: dict[str, dict[str, Any]] = {}

    for entry in manifest["panels"]:
        spec = _spec_from_entry(entry)
        panel_dir = (root / spec.path) if spec.path is not None else root
        panel_csv = panel_dir / "forecast_panel.csv"
        if not panel_csv.exists():
            raise FileNotFoundError(f"missing forecast_panel.csv for {spec.model!r}: {panel_csv}")
        panels.append(load_forecast_panel(panel_csv, spec))

        hp_csv = panel_dir / "selected_hyperparameters.csv"
        if hp_csv.exists():
            hyperparameters.append(load_selected_hyperparameters(hp_csv, spec))

        fail_csv = panel_dir / "failed_origins.csv"
        if fail_csv.exists():
            try:
                failures.append(load_failed_origins(fail_csv, spec))
            except Exception:  # noqa: BLE001 - empty/degenerate failure files
                pass

        meta_json = panel_dir / "run_metadata.json"
        if meta_json.exists():
            metadata_by_model[spec.model] = load_run_metadata(meta_json)

    combined = combine_panels(panels)
    return {
        "combined": combined,
        "hyperparameters": pd.concat(hyperparameters, ignore_index=True) if hyperparameters else pd.DataFrame(),
        "failures": pd.concat(failures, ignore_index=True) if failures else pd.DataFrame(),
        "metadata_by_model": metadata_by_model,
    }


def build_tables(
    study: dict[str, Any],
    *,
    baseline_model: str,
    bootstrap_seed: int = 0,
    coverage_policy: str = "restrict",
    min_coverage: float = 0.0,
) -> dict[str, pd.DataFrame]:
    raw = study["combined"]
    alignment = check_origin_alignment(raw, policy=coverage_policy, min_coverage=min_coverage)

    # Restrict ONCE, globally: every table below -- point estimates and paired
    # inference alike -- is computed from this single frame, so they cannot drift
    # onto different samples.
    combined, sample_report = restrict_to_common_sample(
        raw, policy=coverage_policy, min_coverage=min_coverage
    )

    # The policy must be threaded into *every* builder. Under "restrict" the
    # frame is already restricted, so the downstream restriction is a genuine
    # no-op; under "advisory" nothing below may silently re-restrict, which is
    # exactly what passing the policy through guarantees.
    coverage_kwargs = {"policy": coverage_policy, "min_coverage": min_coverage}

    rmse_tbl = rmse_by_target(combined, **coverage_kwargs)
    mae_tbl = mae_by_target(combined, **coverage_kwargs)
    rel_tbl = relative_rmse(
        rmse_tbl, baseline_model=baseline_model, combined=combined, **coverage_kwargs
    )
    # Under "advisory" the models legitimately sit on different samples, so the
    # common-sample precondition of the ranks is relaxed (and recorded in
    # common_sample.csv / comparison_summary.md) instead of raising.
    ranks_tbl = average_ranks(rmse_tbl, require_common_sample=coverage_policy != "advisory")
    gains_tbl = scope_gains(combined, **coverage_kwargs)
    gain_summary_tbl = scope_gain_summary(gains_tbl)

    contrasts = standard_scope_contrasts(combined)
    dm_tbl = dm_tests(combined, contrasts)
    boot_tbl = bootstrap_intervals(combined, contrasts, seed=bootstrap_seed)

    hp = study["hyperparameters"]
    hp_summary = hyperparameter_summary(hp) if not hp.empty else pd.DataFrame()
    stability = selection_stability(hp) if not hp.empty else pd.DataFrame()
    fail = study["failures"]
    fail_summary = failure_summary(fail) if not fail.empty else pd.DataFrame(
        columns=["model", "stage", "n_failures"]
    )
    cost = computational_cost(study["metadata_by_model"])

    return {
        "rmse_by_target": rmse_tbl,
        "mae_by_target": mae_tbl,
        "relative_rmse": rel_tbl,
        "average_ranks": ranks_tbl,
        "scope_gains": gains_tbl,
        "scope_gain_summary": gain_summary_tbl,
        "hyperparameter_summary": hp_summary,
        "selection_stability": stability,
        "failure_summary": fail_summary,
        "computational_cost": cost,
        "dm_tests": dm_tbl,
        "bootstrap_intervals": boot_tbl,
        "common_sample": sample_report.to_frame(),
        "_alignment": alignment,
    }


def write_tables(tables: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        if name.startswith("_"):
            continue
        table.to_csv(output_dir / f"{name}.csv", index=False)

    write_comparison_summary(
        output_dir / "comparison_summary.md",
        gains=tables["scope_gains"],
        gain_summary=tables["scope_gain_summary"],
        average_rank_table=tables["average_ranks"],
        dm_table=tables["dm_tests"],
        alignment=tables["_alignment"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assemble a model-independent scope-study comparison from canonical panels."
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Path to the JSON study manifest.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Root for relative panel dirs.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-model", type=str, default=None,
                        help="Model label used as the relative-RMSE baseline (default: manifest).")
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--coverage-policy", type=str, default="restrict",
        choices=["restrict", "raise", "advisory"],
        help="How to handle observations not shared by every model in a cell: "
             "restrict (drop them, default), raise (fail), advisory (report only).",
    )
    parser.add_argument(
        "--min-coverage", type=float, default=0.0,
        help="Fail if the retained share of observations falls below this value.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    with Path(args.manifest).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    baseline = args.baseline_model or manifest.get("baseline_model")
    if not baseline:
        print("error: a baseline model must be given via --baseline-model or the manifest.",
              file=sys.stderr)
        return 2

    try:
        study = load_study(manifest, root=Path(args.root))
        tables = build_tables(
            study,
            baseline_model=baseline,
            bootstrap_seed=args.bootstrap_seed,
            coverage_policy=args.coverage_policy,
            min_coverage=args.min_coverage,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    write_tables(tables, Path(args.output_dir))
    print(f"[done] wrote comparison tables to {args.output_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
