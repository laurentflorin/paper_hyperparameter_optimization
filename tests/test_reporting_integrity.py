import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from forecast_comparison import discover_run_directories
from paper_hyperparameter_optimization import reporting as R


def _paper_row(
    model: str,
    origin: str,
    target: str,
    error: float | None,
    *,
    actual: float = 1.0,
) -> dict[str, object]:
    return {
        "model": model,
        "optimization_horizon": None,
        "group": "+0 months",
        "forecast_origin": origin,
        "target_quarter": target,
        "variable": "GDP",
        "horizon_quarters": 1,
        "actual_metric": actual,
        "error_metric": error,
    }


def _strict_metadata(strategy: str, *, actual_vintage: str = "2012-01-31") -> dict[str, object]:
    return {
        "strategy": strategy,
        "actual_vintage": actual_vintage,
        "forecast_origin_start": "2000-01-31",
        "forecast_origin_end": "2000-04-30",
        "horizon_semantics": "calendar_quarters_from_nominal_origin",
        "forecast_horizon_months": 24,
        "temp_agg": "mean",
        "evaluation_transforms": {"GDP": "100*diff(log(level))"},
        "model_universe": ["GDP"],
        "selection_schedule": "first_origin",
        "data_fingerprint": "abc123",
        "dependency_revision": {"MBFVAR": "reviewed-commit"},
        "repository_commit": "deadbeef",
        "optimization_eval_horizon_quarters": None,
    }


def _write_run(
    directory: Path,
    strategy: str,
    rows: list[dict[str, object]],
    *,
    metadata: dict[str, object] | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).drop(columns=["model", "optimization_horizon"], errors="ignore")
    frame.insert(0, "strategy", strategy)
    frame.to_csv(directory / "forecast_panel.csv", index=False)
    (directory / "run_metadata.json").write_text(
        json.dumps(metadata or _strict_metadata(strategy)),
        encoding="utf-8",
    )


def test_pairing_recomputes_baseline_and_exposes_rank_reversal():
    rows = [
        _paper_row("paper", "2000-01-31", "2000Q1", 1.0),
        _paper_row("paper", "2000-04-30", "2000Q2", 100.0),
        {
            **_paper_row("mango_rmse", "2000-01-31", "2000Q1", 2.0),
            "optimization_horizon": "h1q",
        },
    ]
    scores, audit = R.compute_rmse_table(
        pd.DataFrame(rows),
        minimum_common_coverage=0.0,
        return_audit=True,
    )
    competitor = scores[scores["model"] == "mango_rmse"].iloc[0]
    assert competitor["rmse"] == 2.0
    assert competitor["baseline_rmse"] == 1.0
    assert competitor["n_model"] == 1
    assert competitor["n_baseline"] == 2
    assert competitor["n_common"] == 1
    assert set(audit["status"]) == {"missing_model_key"}

    relative = R.compute_relative_rmse(scores)
    competitor_relative = relative[relative["model"] == "mango_rmse"].iloc[0]
    assert competitor_relative["relative_rmse_pct"] == 100.0
    assert np.allclose(
        relative.loc[relative["model"] == "paper", "relative_rmse_pct"],
        0.0,
    )


def test_optimization_horizons_are_distinct_model_variants():
    baseline = _paper_row("paper", "2000-01-31", "2000Q1", 1.0)
    h1 = {
        **_paper_row("mango_rmse", "2000-01-31", "2000Q1", 2.0),
        "optimization_horizon": "h1q",
    }
    h4 = {
        **_paper_row("mango_rmse", "2000-01-31", "2000Q1", 4.0),
        "optimization_horizon": "h4q",
    }
    scores = R.compute_rmse_table(pd.DataFrame([baseline, h1, h4]))
    variants = scores[scores["model"] == "mango_rmse"].sort_values(
        "optimization_horizon"
    )
    assert list(variants["optimization_horizon"]) == ["h1q", "h4q"]
    assert list(variants["rmse"]) == [2.0, 4.0]


def test_duplicate_keys_and_mismatched_actuals_fail_closed():
    baseline = _paper_row("paper", "2000-01-31", "2000Q1", 1.0)
    duplicate = {
        **_paper_row("mango_rmse", "2000-01-31", "2000Q1", 2.0),
        "optimization_horizon": "h1q",
    }
    with pytest.raises(ValueError, match="Duplicate forecast keys"):
        R.compute_rmse_table(pd.DataFrame([baseline, duplicate, duplicate]))

    mismatched = {**duplicate, "actual_metric": 9.0}
    with pytest.raises(ValueError, match="mismatched actual"):
        R.compute_rmse_table(pd.DataFrame([baseline, mismatched]))


def test_horizon_metadata_drives_loading_and_legacy_path_is_validated(tmp_path: Path):
    baseline = _paper_row("paper", "2000-01-31", "2000Q1", 1.0)
    competitor = _paper_row("mango_rmse", "2000-01-31", "2000Q1", 2.0)
    _write_run(tmp_path / "paper", "paper", [baseline])

    for horizon in (1, 4):
        metadata = _strict_metadata("mango_rmse")
        metadata["optimization_eval_horizon_quarters"] = horizon
        _write_run(
            tmp_path / "rmse" / f"h{horizon}q",
            "mango_rmse",
            [competitor],
            metadata=metadata,
        )

    loaded = R.load_forecast_panels(
        {"paper": tmp_path / "paper", "mango_rmse": tmp_path / "rmse"}
    )
    assert set(loaded.loc[loaded["model"] == "mango_rmse", "optimization_horizon"]) == {
        "h1q",
        "h4q",
    }

    bad_metadata = _strict_metadata("mango_rmse")
    bad_metadata["optimization_eval_horizon_quarters"] = 2
    _write_run(
        tmp_path / "bad" / "h1q",
        "mango_rmse",
        [competitor],
        metadata=bad_metadata,
    )
    with pytest.raises(ValueError, match="horizon mismatch"):
        R.load_forecast_panels({"mango_rmse": tmp_path / "bad"})


def test_explicit_empty_panel_and_partial_horizon_batch_are_rejected(tmp_path: Path):
    direct = tmp_path / "direct"
    direct.mkdir()
    (direct / "forecast_panel.csv").write_text("\n", encoding="utf-8")
    (direct / "run_metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        discover_run_directories(direct)

    complete = _paper_row("mango_rmse", "2000-01-31", "2000Q1", 2.0)
    _write_run(tmp_path / "batch" / "h1q", "mango_rmse", [complete])
    incomplete = tmp_path / "batch" / "h8q"
    incomplete.mkdir()
    (incomplete / "forecast_panel.csv").write_text("\n", encoding="utf-8")
    (incomplete / "run_metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete child run"):
        discover_run_directories(tmp_path / "batch")


def test_report_writes_manifest_coverage_and_exclusion_audit(tmp_path: Path):
    baseline_rows = [
        _paper_row("paper", "2000-01-31", "2000Q1", 1.0),
        _paper_row("paper", "2000-04-30", "2000Q2", 3.0),
    ]
    competitor_rows = [
        _paper_row("mango_mdd", "2000-01-31", "2000Q1", 2.0),
    ]
    _write_run(tmp_path / "paper", "paper", baseline_rows)
    _write_run(tmp_path / "mdd", "mango_mdd", competitor_rows)

    output = R.create_comparison_report(
        {"paper": tmp_path / "paper", "mango_mdd": tmp_path / "mdd"},
        tmp_path / "comparison",
        minimum_common_coverage=0.0,
    )
    manifest = json.loads((output / "comparison_manifest.json").read_text(encoding="utf-8"))
    assert manifest["pairing_policy"].startswith("pairwise inner join")
    assert manifest["relative_rmse_definition"].startswith("100 * (model_rmse")
    assert len(manifest["inputs"]) == 2
    assert all(item["directory_input_sha256"] for item in manifest["inputs"])

    scores = pd.read_csv(output / "rmse_all_variables.csv")
    competitor = scores[scores["model"] == "mango_mdd"].iloc[0]
    assert competitor["n_common"] == 1
    assert competitor["n_baseline"] == 2
    audit = pd.read_csv(output / "comparison_exclusion_audit.csv")
    assert "missing_model_key" in set(audit["status"])


def test_report_rejects_incompatible_provenance(tmp_path: Path):
    row = _paper_row("paper", "2000-01-31", "2000Q1", 1.0)
    _write_run(tmp_path / "paper", "paper", [row])
    bad_metadata = _strict_metadata("mango_mdd", actual_vintage="2013-01-31")
    _write_run(
        tmp_path / "mdd",
        "mango_mdd",
        [{**row, "model": "mango_mdd"}],
        metadata=bad_metadata,
    )
    with pytest.raises(ValueError, match="actual_vintage"):
        R.create_comparison_report(
            {"paper": tmp_path / "paper", "mango_mdd": tmp_path / "mdd"},
            tmp_path / "comparison",
        )
