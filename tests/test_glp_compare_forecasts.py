import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts" / "glp"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import compare_glp_forecasts as C


def _write_metadata(directory: Path, *, strategy: str, model_size: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run_metadata.json").write_text(
        json.dumps({"strategy": strategy, "model_size": model_size}), encoding="utf-8"
    )


def test_discover_strategy_dir_finds_direct_and_horizon_layouts(tmp_path: Path):
    _write_metadata(tmp_path / "paper_small", strategy="paper", model_size="small")
    _write_metadata(tmp_path / "mango_small", strategy="mango_mdd", model_size="small")
    _write_metadata(tmp_path / "mango_rmse_small" / "h1q", strategy="mango_rmse", model_size="small")
    _write_metadata(
        tmp_path / "mango_rmse_random_small" / "h1q", strategy="mango_rmse_random", model_size="small"
    )

    assert C._discover_strategy_dir(tmp_path, "paper", "small") == tmp_path / "paper_small"
    assert C._discover_strategy_dir(tmp_path, "mango_mdd", "small") == tmp_path / "mango_small"
    assert C._discover_strategy_dir(tmp_path, "mango_rmse", "small") == tmp_path / "mango_rmse_small"
    assert C._discover_strategy_dir(tmp_path, "mango_rmse_random", "small") == tmp_path / "mango_rmse_random_small"


def test_discover_strategy_dir_requires_disambiguation_if_multiple_match(tmp_path: Path):
    _write_metadata(tmp_path / "paper_small", strategy="paper", model_size="small")
    _write_metadata(tmp_path / "paper_medium", strategy="paper", model_size="medium")

    with pytest.raises(ValueError):
        C._discover_strategy_dir(tmp_path, "paper", None)
    assert C._discover_strategy_dir(tmp_path, "paper", "small") == tmp_path / "paper_small"


def test_resolve_strategy_dir_honors_explicit_override(tmp_path: Path):
    explicit = tmp_path / "paper_custom"
    explicit.mkdir()
    resolved = C._resolve_strategy_dir(explicit, strategy="paper", root_dir=tmp_path, model_size=None)
    assert resolved == explicit
