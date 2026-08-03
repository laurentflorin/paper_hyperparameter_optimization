import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_forecasts import build_parser


def test_parser_has_defaults_for_standard_output_layout():
    parser = build_parser()
    args = parser.parse_args([])

    assert args.paper_dir == Path("outputs/paper_hyperparameters")
    assert args.mango_mdd_dir == Path("outputs/mango_mdd")
    assert args.mango_rmse_dir == Path("outputs/mango_rmse")
    assert args.mango_rmse_random_dir == Path("outputs/mango_rmse_random")
    assert args.output_dir == Path("outputs/comparison")
