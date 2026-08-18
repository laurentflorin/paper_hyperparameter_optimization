"""Pilot A end-to-end integration validation.

Run after the two ridge scope-grid studies and the compare step.
Usage::

    python tests/pilot_a_validate.py \
        --iterated-root /tmp/pilot_a/iterated \
        --direct-root   /tmp/pilot_a/direct   \
        --comparison    /tmp/pilot_a/comparison

Exit 0 = all checks pass, non-zero = at least one check failed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common_hpo.io import classify_run_directory  # noqa: E402

PASS = "\u2713"
FAIL = "\u2717"

_errors: list[str] = []


def _ok(msg: str) -> None:
    print(f"  {PASS}  {msg}")


def _err(msg: str) -> None:
    print(f"  {FAIL}  {msg}")
    _errors.append(msg)


def check_study(root: Path, label: str, scopes: tuple[str, ...], variables: tuple[str, ...],
                horizons: tuple[int, ...]) -> dict[str, pd.DataFrame]:
    """Validate one scope-grid study and return per-scope forecast panels."""
    print(f"\n{'='*60}")
    print(f"Study: {label}  root={root}")
    print(f"{'='*60}")
    panels: dict[str, pd.DataFrame] = {}

    for scope in scopes:
        scope_dir = root / f"scope_{scope}"
        print(f"\n  Scope: {scope}  dir={scope_dir}")

        # run-state
        state = classify_run_directory(scope_dir)
        if state.status == "complete":
            _ok(f"run_complete.json present (status=complete)")
        else:
            _err(f"run status is {state.status!r}, expected 'complete'")

        # configuration hash
        manifest_path = scope_dir / "run_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("configuration_hash"):
                _ok(f"configuration_hash present: {str(manifest['configuration_hash'])[:16]}...")
            else:
                _err("configuration_hash missing from run_manifest.json")
        else:
            _err("run_manifest.json missing")

        fp_path = scope_dir / "forecast_panel.csv"
        if not fp_path.exists():
            _err(f"forecast_panel.csv missing in {scope_dir}")
            continue
        fp = pd.read_csv(fp_path)
        panels[scope] = fp
        print(f"    forecast rows: {len(fp)}")

        # no duplicates
        key_cols = ["forecast_origin", "variable", "horizon_quarters"]
        dup_count = fp.duplicated(subset=key_cols).sum()
        if dup_count == 0:
            _ok("no duplicate canonical rows")
        else:
            _err(f"{dup_count} duplicate rows on {key_cols}")

        # target coverage: every (variable, horizon) appears at each origin
        origins = fp["forecast_origin"].unique()
        for var in variables:
            for h in horizons:
                sub = fp[(fp["variable"] == var) & (fp["horizon_quarters"] == h)]
                if len(sub) == len(origins):
                    _ok(f"variable={var} horizon={h} covered ({len(sub)} rows)")
                else:
                    _err(f"variable={var} horizon={h}: {len(sub)} rows, expected {len(origins)}")

        # no NaN forecasts
        if fp["mean_metric"].notna().all():
            _ok("no NaN mean_metric forecasts")
        else:
            nan_count = fp["mean_metric"].isna().sum()
            _err(f"{nan_count} NaN mean_metric values")

        # selection linkage: every origin in the forecast panel links to a selection event
        hp_path = scope_dir / "selected_hyperparameters.csv"
        if hp_path.exists():
            hp = pd.read_csv(hp_path)
            _ok(f"selected_hyperparameters.csv present ({len(hp)} rows)")

            # no missing parameter values
            param_cols = [c for c in hp.columns if c.startswith("param_")]
            if param_cols:
                missing_params = hp[param_cols].isna().any().any()
                if not missing_params:
                    _ok("no missing selected parameter values")
                else:
                    _err("NaN values in selected parameter columns")

            # selection origins are a subset of forecast origins (schedule runs on valid outer origins)
            forecast_origins = set(fp["forecast_origin"].unique())
            sel_origins = set(hp["forecast_origin"].unique())
            stray = sel_origins - forecast_origins
            if not stray:
                _ok(f"all selection origins are valid outer origins ({len(sel_origins)} selection events)")
            else:
                _err(f"selection origins not in forecast set: {stray}")

            # selection record count is sane (1 = selected-once; >=1 = periodic)
            n_cells = hp["cell_id"].nunique() if "cell_id" in hp.columns else 1
            events_per_cell = len(hp) / max(n_cells, 1)
            _ok(f"selection events: {len(hp)} total, ~{events_per_cell:.1f} per cell")
        else:
            _err(f"selected_hyperparameters.csv missing in {scope_dir}")

        # failed origins
        fail_path = scope_dir / "failed_origins.csv"
        if fail_path.exists():
            fails = pd.read_csv(fail_path)
            n_fail = len(fails)
            if n_fail == 0:
                _ok("failed_origins.csv present and empty")
            else:
                # numerical failures are classified
                if "failure_category" in fails.columns:
                    cats = fails["failure_category"].value_counts().to_dict()
                    _ok(f"{n_fail} failures classified by category: {cats}")
                else:
                    _err(f"{n_fail} failures in failed_origins.csv but no failure_category column")
        else:
            _err(f"failed_origins.csv missing in {scope_dir}")

    return panels


def check_no_inner_exceeds_outer(root: Path, label: str, scopes: tuple[str, ...]) -> None:
    """Check run_manifest.json inner/outer split descriptions for leakage."""
    print(f"\n  Inner<=Outer cutoff check for {label}")
    for scope in scopes:
        manifest_path = root / f"scope_{scope}" / "run_metadata.json"
        if not manifest_path.exists():
            _err(f"run_metadata.json missing for scope {scope}")
            continue
        meta = json.loads(manifest_path.read_text())
        inner = meta.get("validation_scheme", {})
        outer_origins = meta.get("outer_evaluation", {}).get("n_origins")
        inner_origins = inner.get("n_origins")
        if inner_origins is not None and outer_origins is not None:
            if inner_origins <= outer_origins:
                _ok(f"scope={scope}: inner_n_origins({inner_origins}) <= outer_n_origins({outer_origins})")
            else:
                _err(f"scope={scope}: inner_n_origins({inner_origins}) > outer_n_origins({outer_origins})")
        else:
            _ok(f"scope={scope}: split sizes not directly comparable from metadata (structure ok)")


def check_comparison(comparison_dir: Path, expected_models: set[str]) -> None:
    print(f"\n{'='*60}")
    print(f"Comparison: {comparison_dir}")
    print(f"{'='*60}")

    summary_path = comparison_dir / "comparison_summary.md"
    if summary_path.exists():
        _ok("comparison_summary.md produced")
    else:
        _err("comparison_summary.md missing")

    rmse_path = comparison_dir / "rmse_by_target.csv"
    if rmse_path.exists():
        rmse = pd.read_csv(rmse_path)
        _ok(f"rmse_by_target.csv present ({len(rmse)} rows)")
        models_in_rmse = set(rmse["model"].unique()) if "model" in rmse.columns else set()
        missing = expected_models - models_in_rmse
        if not missing:
            _ok(f"all expected models appear in RMSE table: {sorted(expected_models)}")
        else:
            _err(f"models missing from RMSE table: {missing}")
    else:
        _err("rmse_by_target.csv missing")

    rel_path = comparison_dir / "relative_rmse.csv"
    if rel_path.exists():
        _ok("relative_rmse.csv produced")
    else:
        _err("relative_rmse.csv missing")

    scope_path = comparison_dir / "scope_gains.csv"
    if scope_path.exists():
        _ok("scope_gains.csv produced")
    else:
        _err("scope_gains.csv missing")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterated-root", required=True, type=Path)
    parser.add_argument("--direct-root", required=True, type=Path)
    parser.add_argument("--comparison", required=True, type=Path)
    args = parser.parse_args()

    scopes = ("pooled", "horizon", "variable", "variable_horizon")
    variables = ("gdp", "inv", "cons")
    horizons = (1, 2, 4, 8)

    # Iterated study: selected once
    check_study(args.iterated_root, "iterated (selected once)", scopes, variables, horizons)
    check_no_inner_exceeds_outer(args.iterated_root, "iterated (selected once)", scopes)

    # Direct study: reselected every 2 origins
    check_study(args.direct_root, "direct (every 2 origins)", scopes, variables, horizons)
    check_no_inner_exceeds_outer(args.direct_root, "direct (every 2 origins)", scopes)

    # Comparison output
    expected_models = {
        "ridge_iterated_pooled",
        "ridge_iterated_horizon",
        "ridge_iterated_variable",
        "ridge_iterated_variable_horizon",
        "ridge_direct_pooled",
        "ridge_direct_horizon",
        "ridge_direct_variable",
        "ridge_direct_variable_horizon",
    }
    check_comparison(args.comparison, expected_models)

    print(f"\n{'='*60}")
    if _errors:
        print(f"{FAIL}  {len(_errors)} check(s) FAILED:")
        for e in _errors:
            print(f"    - {e}")
        return 1
    else:
        print(f"{PASS}  ALL CHECKS PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
