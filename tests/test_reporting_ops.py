import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_forecasts import resolve_experiment_dir


def _dry_run(cwd: Path, output_root: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "SKIP_MODULES": "1",
            "OUTPUT_ROOT": str(output_root),
            "STAGES": "download,paper,mango_mdd,mango_rmse,mango_rmse_random,compare",
        }
    )
    return subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "run_everything_euler.sh")],
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("different_cwd", [False, True])
def test_euler_script_uses_absolute_repo_entry_points(
    tmp_path: Path, different_cwd: bool
):
    cwd = tmp_path if different_cwd else REPO_ROOT
    output_root = tmp_path / ("away" if different_cwd else "root")
    completed = _dry_run(cwd, output_root)
    expected_scripts = [
        "download_data.py",
        "run_paper_hyperparameters.py",
        "run_mango_mdd.py",
        "run_mango_rmse.py",
        "run_mango_rmse_random.py",
        "compare_forecasts.py",
    ]
    for script_name in expected_scripts:
        assert str(REPO_ROOT / "scripts" / script_name) in completed.stdout
    assert "DRY RUN [paper]" in completed.stdout
    status = (output_root / "stage_status.tsv").read_text(encoding="utf-8")
    assert "paper\tplanned" in status
    assert "compare\tplanned" in status


def test_explicit_comparison_path_never_uses_a_fallback(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_experiment_dir(tmp_path / "typo", "paper_hyperparameters")

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "forecast_panel.csv").write_text("\n", encoding="utf-8")
    (incomplete / "run_metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        resolve_experiment_dir(incomplete, "paper_hyperparameters")
