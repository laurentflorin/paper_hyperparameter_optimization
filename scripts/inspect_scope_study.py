"""Inspect the current state of a scope-study experiment.

Usage examples::

    # Full study status report
    python scripts/inspect_scope_study.py \
        --config configs/paper_experiment.json \
        --output-root outputs/scope_study

    # Summary only
    python scripts/inspect_scope_study.py \
        --config configs/paper_experiment.json \
        --output-root outputs/scope_study --summary

    # Show only failed jobs
    python scripts/inspect_scope_study.py \
        --config configs/paper_experiment.json \
        --output-root outputs/scope_study --filter-status failed

    # Emit JSON instead of human-readable text
    python scripts/inspect_scope_study.py \
        --config configs/paper_experiment.json \
        --output-root outputs/scope_study --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

# Import orchestrator helpers without re-importing the heavy scientific stack
import importlib.util as _ilu

_runner_spec = _ilu.spec_from_file_location(
    "run_scope_study_mod",
    str(PROJECT_ROOT / "scripts" / "run_scope_study.py"),
)
_runner_mod = _ilu.module_from_spec(_runner_spec)
_runner_spec.loader.exec_module(_runner_mod)

JOB_MANIFEST_FILENAME = _runner_mod.JOB_MANIFEST_FILENAME
STUDY_STATUS_FILENAME = _runner_mod.STUDY_STATUS_FILENAME
JobSpec = _runner_mod.JobSpec
expand_jobs = _runner_mod.expand_jobs
load_config = _runner_mod.load_config
_overall_job_status = _runner_mod._overall_job_status
_read_job_manifest = _runner_mod._read_job_manifest
validate_config = _runner_mod.validate_config

from common_hpo.io import classify_run_directory  # noqa: E402

import pandas as pd  # noqa: E402


# --------------------------------------------------------------------------- #
# Per-job inspection
# --------------------------------------------------------------------------- #


def _scope_detail(spec: JobSpec) -> dict[str, Any]:
    """Return per-scope state, row counts, and schema issues for one job."""
    scopes: dict[str, Any] = {}
    for scope in spec.scopes:
        sd = spec.scope_output_dir(scope)
        if not sd.exists():
            scopes[scope] = {"status": "missing", "forecast_rows": None, "selection_rows": None, "issues": []}
            continue
        state = classify_run_directory(sd)
        issues = []
        forecast_rows = None
        selection_rows = None
        boundary_hits = None

        fp_path = sd / "forecast_panel.csv"
        hp_path = sd / "selected_hyperparameters.csv"
        fail_path = sd / "failed_origins.csv"

        if fp_path.exists():
            try:
                fp = pd.read_csv(fp_path)
                forecast_rows = len(fp)
                if fp.duplicated(subset=["forecast_origin", "variable", "horizon_quarters"]).any():
                    issues.append("duplicate_forecast_rows")
                if fp["mean_metric"].isna().any():
                    issues.append("nan_forecasts")
            except Exception as exc:
                issues.append(f"forecast_panel_unreadable:{exc}")
        else:
            if state.status not in ("missing", "partial"):
                issues.append("forecast_panel_missing")

        if hp_path.exists():
            try:
                hp = pd.read_csv(hp_path)
                selection_rows = len(hp)
                param_cols = [c for c in hp.columns if c.startswith("param_")]
                if param_cols and hp[param_cols].isna().any().any():
                    issues.append("missing_selected_params")
                # Boundary hit detection: flag if any param column is at grid extremes
                # We can only heuristically detect this by checking min/max against
                # the known grid extent — that information is not available here,
                # so we just report min/max for human review.
                if param_cols:
                    boundary_hits = {
                        c: {"min": float(hp[c].min()), "max": float(hp[c].max())}
                        for c in param_cols
                        if hp[c].dtype.kind in ("f", "i")
                    }
            except Exception as exc:
                issues.append(f"hyperparameters_unreadable:{exc}")

        fail_count = 0
        failure_categories: dict[str, int] = {}
        if fail_path.exists():
            try:
                fails = pd.read_csv(fail_path)
                fail_count = len(fails)
                if "failure_category" in fails.columns and fail_count > 0:
                    failure_categories = fails["failure_category"].value_counts().to_dict()
            except Exception:
                issues.append("failed_origins_unreadable")

        scopes[scope] = {
            "status": state.status,
            "configuration_hash": state.configuration_hash,
            "forecast_rows": forecast_rows,
            "selection_rows": selection_rows,
            "failed_origins": fail_count,
            "failure_categories": failure_categories or None,
            "boundary_hits": boundary_hits,
            "issues": issues,
        }
    return scopes


def inspect_job(spec: JobSpec) -> dict[str, Any]:
    """Full inspection of one job directory."""
    manifest = _read_job_manifest(spec.output_dir)
    overall = _overall_job_status(spec)
    scope_detail = _scope_detail(spec)

    log_path = None
    for candidate in (
        spec.output_dir.parent.parent / "logs" / f"{spec.job_id}.log",
        spec.output_dir / "job.log",
    ):
        if candidate.exists():
            log_path = str(candidate)
            break

    # Storage estimate
    storage_bytes = sum(
        f.stat().st_size
        for f in spec.output_dir.rglob("*")
        if f.is_file()
    ) if spec.output_dir.exists() else 0

    return {
        "job_id": spec.job_id,
        "family": spec.family,
        "variant": spec.variant,
        "size": spec.size,
        "forecast_method": spec.forecast_method,
        "seed": spec.seed,
        "status": overall,
        "configuration_hash": spec.configuration_hash[:16],
        "manifest_hash": (manifest or {}).get("configuration_hash", "")[:16],
        "hash_compatible": (
            not manifest
            or manifest.get("configuration_hash") == spec.configuration_hash
        ),
        "output_dir": str(spec.output_dir),
        "log_path": log_path,
        "scopes": scope_detail,
        "storage_bytes": storage_bytes,
    }


# --------------------------------------------------------------------------- #
# Study-level report
# --------------------------------------------------------------------------- #

def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n //= 1024
    return f"{n:.0f} TB"


def inspect_study(
    config: dict[str, Any],
    output_root: Path,
    filter_status: list[str] | None = None,
) -> dict[str, Any]:
    jobs = expand_jobs(config, output_root)
    inspected = [inspect_job(j) for j in jobs]

    if filter_status:
        inspected = [j for j in inspected if j["status"] in filter_status]

    # Summary counters
    from collections import Counter
    status_counts: Counter = Counter(j["status"] for j in inspected)
    total_storage = sum(j["storage_bytes"] for j in inspected)
    incompatible = [j for j in inspected if not j["hash_compatible"]]
    all_issues: list[str] = []
    for j in inspected:
        for scope_data in j["scopes"].values():
            all_issues.extend(scope_data.get("issues", []))

    return {
        "output_root": str(output_root),
        "total_jobs": len(inspected),
        "status_counts": dict(status_counts),
        "total_storage": total_storage,
        "total_storage_human": _fmt_bytes(total_storage),
        "incompatible_jobs": len(incompatible),
        "schema_issues": len(all_issues),
        "jobs": inspected,
    }


# --------------------------------------------------------------------------- #
# Human-readable printing
# --------------------------------------------------------------------------- #

_STATUS_ICON = {
    "complete":   "✓",
    "partial":    "~",
    "failed":     "✗",
    "cancelled":  "∅",
    "missing":    "·",
    "incompatible": "!",
    "unknown":    "?",
}


def print_report(study: dict[str, Any], *, verbose: bool = False) -> None:
    root = study["output_root"]
    total = study["total_jobs"]
    counts = study["status_counts"]
    print(f"\nScope-study inspection — {root}")
    print(f"Total jobs: {total}  |  Storage: {study['total_storage_human']}")
    if study.get("incompatible_jobs"):
        print(f"  WARNING: {study['incompatible_jobs']} jobs have incompatible config hashes")
    print()

    # Status bar
    for status, n in sorted(counts.items(), key=lambda x: x[0]):
        icon = _STATUS_ICON.get(status, "?")
        print(f"  {icon} {status:<14}: {n}")
    print()

    if not verbose:
        # Table of non-complete jobs
        interesting = [j for j in study["jobs"] if j["status"] != "complete"]
        if not interesting:
            print("All jobs complete.")
            return
        print(f"Non-complete jobs ({len(interesting)}):\n")
        print(f"  {'status':<12}  {'job_id':<50}  {'scopes'}")
        print("  " + "-" * 90)
        for j in interesting:
            scope_summary = ", ".join(
                f"{s}:{d['status'][0]}" for s, d in j["scopes"].items()
            )
            icon = _STATUS_ICON.get(j["status"], "?")
            print(f"  {icon} {j['status']:<10}  {j['job_id']:<50}  {scope_summary}")
    else:
        # Full detail
        for j in study["jobs"]:
            icon = _STATUS_ICON.get(j["status"], "?")
            compat = "" if j["hash_compatible"] else " [HASH MISMATCH]"
            print(f"{icon} {j['job_id']}{compat}")
            print(f"    family={j['family']}  variant={j['variant']}  "
                  f"size={j['size']}  method={j['forecast_method']}")
            print(f"    output_dir: {j['output_dir']}")
            for scope, detail in j["scopes"].items():
                issues_str = f"  issues: {detail['issues']}" if detail["issues"] else ""
                fr = detail["forecast_rows"]
                hr = detail["selection_rows"]
                bh = detail.get("boundary_hits")
                bh_str = ""
                if bh:
                    bh_str = "  boundaries: " + " ".join(
                        f"{p}=[{v['min']:.3g},{v['max']:.3g}]" for p, v in bh.items()
                    )
                print(f"    {scope}: {detail['status']:<12}  "
                      f"forecast={fr}  selection={hr}{issues_str}{bh_str}")
            print()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Inspect scope-study experiment state.")
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "paper_experiment.json"))
    p.add_argument("--output-root", default=None,
                   help="Override the output root from the config.")
    p.add_argument("--filter-status", default=None,
                   help="Comma-separated statuses to show (complete,partial,failed,missing,incompatible).")
    p.add_argument("--summary", action="store_true", help="Print summary only, no per-job detail.")
    p.add_argument("--verbose", action="store_true", help="Print full per-scope detail.")
    p.add_argument("--json", action="store_true", help="Emit JSON instead of human text.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    errs = validate_config(config)
    if errs:
        for e in errs:
            print(f"error: {e}", file=sys.stderr)
        return 1

    output_root = Path(args.output_root) if args.output_root else Path(config["study"]["output_root"])

    filter_status = (
        [s.strip() for s in args.filter_status.split(",")]
        if args.filter_status else None
    )
    study = inspect_study(config, output_root, filter_status=filter_status)

    if args.json:
        print(json.dumps(study, indent=2, default=str))
        return 0

    if args.summary:
        print(f"Jobs: {study['total_jobs']}  "
              f"Storage: {study['total_storage_human']}  "
              f"Status: {study['status_counts']}")
        return 0

    print_report(study, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
