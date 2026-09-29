import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts" / "glp"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import run_glp_all as run_all


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


def test_parse_ladder_rungs_and_labels():
    rungs = run_all.parse_ladder_rungs("mdd,lsj1:w12,lsm:h4:w12,rmse:h8:wall")
    assert [r["objective"] for r in rungs] == ["mdd", "logscore_joint_1", "logscore_marginal", "rmse"]
    assert rungs[3]["window"] is None and rungs[3]["horizon"] == 8
    assert [run_all.ladder_label(r) for r in rungs] == ["mdd", "lsj1_w12", "lsm_h4_w12", "rmse_h8_all"]
    with pytest.raises(ValueError):
        run_all.parse_ladder_rungs("crps:h4")


def test_build_parser_accepts_selection_flags():
    args = run_all.build_parser().parse_args(["--optimization-seed", "5", "--local-polish", "--reselect-every", "4"])
    assert args.optimization_seed == 5 and args.local_polish and args.reselect_every == 4
