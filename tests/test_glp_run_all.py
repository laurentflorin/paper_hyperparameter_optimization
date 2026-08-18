import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts" / "glp"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

try:
    import run_glp_all as run_all
except ImportError as exc:  # pragma: no cover - optional integration surface
    pytest.skip(f"optional GLP runner integration unavailable: {exc}", allow_module_level=True)


def test_parse_stage_list_defaults_to_all_run_scripts():
    assert run_all.parse_stage_list(None) == list(run_all.DEFAULT_STAGES)


def test_parse_stage_list_deduplicates_preserving_order():
    assert run_all.parse_stage_list("paper,mango_mdd,paper,compare") == ["paper", "mango_mdd", "compare"]


def test_parse_stage_list_rejects_unknown_stage():
    with pytest.raises(ValueError):
        run_all.parse_stage_list("paper,unknown")


def test_validate_eval_horizons_uniquifies_and_checks_bounds():
    assert run_all.validate_eval_horizons([1, 2, 2, 8]) == [1, 2, 8]
    with pytest.raises(ValueError):
        run_all.validate_eval_horizons([0])


def test_resolve_stage_output_dir_uses_root_or_override(tmp_path: Path):
    output_root = tmp_path / "outputs"
    assert run_all.resolve_stage_output_dir(output_root, None, "paper") == output_root / "paper"
    custom = tmp_path / "custom" / "paper_run"
    assert run_all.resolve_stage_output_dir(output_root, custom, "paper") == custom


def test_build_parser_accepts_optimization_n_obj_draws():
    parser = run_all.build_parser()
    args = parser.parse_args(["--optimization-n-obj-draws", "48"])
    assert args.optimization_n_obj_draws == 48
