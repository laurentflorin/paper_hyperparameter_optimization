import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common_hpo import SelectionPlan, TargetCell, TargetKey, build_selection_plan


@pytest.mark.parametrize(
    ("scope", "expected_cells"),
    [
        (
            "pooled",
            (
                TargetCell(
                    cell_id="pooled",
                    variables=("GDP", "CPI"),
                    horizons=(1, 4),
                ),
            ),
        ),
        (
            "horizon",
            (
                TargetCell("horizon-h1", ("GDP", "CPI"), (1,)),
                TargetCell("horizon-h4", ("GDP", "CPI"), (4,)),
            ),
        ),
        (
            "variable",
            (
                TargetCell("variable-gdp", ("GDP",), (1, 4)),
                TargetCell("variable-cpi", ("CPI",), (1, 4)),
            ),
        ),
        (
            "variable_horizon",
            (
                TargetCell("variable-gdp-h1", ("GDP",), (1,)),
                TargetCell("variable-gdp-h4", ("GDP",), (4,)),
                TargetCell("variable-cpi-h1", ("CPI",), (1,)),
                TargetCell("variable-cpi-h4", ("CPI",), (4,)),
            ),
        ),
    ],
)
def test_build_selection_plan_core_scopes(scope: str, expected_cells: tuple[TargetCell, ...]):
    plan = build_selection_plan(scope, ["GDP", "CPI"], [1, 4])

    assert plan.cells == expected_cells
    assert plan.target_variables == ("GDP", "CPI")
    assert plan.target_horizons == (1, 4)


def test_group_selection_plan_constructs_expected_cells():
    plan = build_selection_plan(
        "group",
        ["GDP", "CPI", "UNR"],
        [1, 4],
        variable_groups=[
            ("Real Activity", ["GDP"]),
            ("Prices", ["CPI"]),
        ],
        separate_group_horizons=True,
        residual_group_name="Other",
    )

    assert plan.cells == (
        TargetCell("group-real-activity-h1", ("GDP",), (1,), "Real Activity"),
        TargetCell("group-real-activity-h4", ("GDP",), (4,), "Real Activity"),
        TargetCell("group-prices-h1", ("CPI",), (1,), "Prices"),
        TargetCell("group-prices-h4", ("CPI",), (4,), "Prices"),
        TargetCell("group-other-h1", ("UNR",), (1,), "Other"),
        TargetCell("group-other-h4", ("UNR",), (4,), "Other"),
    )
    assert plan.cell_for("UNR", 4).group_name == "Other"


def test_group_selection_plan_can_pool_horizons():
    plan = build_selection_plan(
        "group",
        ["GDP", "CPI"],
        [1, 4],
        variable_groups={"All targets": ["GDP", "CPI"]},
    )

    assert plan.cells == (
        TargetCell("group-all-targets", ("GDP", "CPI"), (1, 4), "All targets"),
    )


def test_selection_plan_ids_are_deterministic():
    first = build_selection_plan(
        "group",
        ["GDP", "CPI"],
        [4, 1],
        variable_groups=[("Real Activity", ["GDP"]), ("Prices", ["CPI"])],
        separate_group_horizons=True,
    )
    second = build_selection_plan(
        "group",
        ["GDP", "CPI"],
        [4, 1],
        variable_groups=[("Real Activity", ["GDP"]), ("Prices", ["CPI"])],
        separate_group_horizons=True,
    )

    assert [cell.cell_id for cell in first.cells] == [
        "group-real-activity-h4",
        "group-real-activity-h1",
        "group-prices-h4",
        "group-prices-h1",
    ]
    assert [cell.cell_id for cell in first.cells] == [cell.cell_id for cell in second.cells]


def test_selection_plan_round_trips_through_dict():
    plan = build_selection_plan(
        "group",
        ["GDP", "CPI", "UNR"],
        [1, 4],
        variable_groups=[("Macro", ["GDP", "CPI"])],
        residual_group_name="Residual",
    )

    serialized = plan.to_dict()
    restored = SelectionPlan.from_dict(serialized)

    assert restored == plan
    assert restored.to_dict() == serialized


def test_duplicate_target_variables_are_rejected():
    with pytest.raises(ValueError, match="duplicate 'GDP'"):
        build_selection_plan("pooled", ["GDP", "GDP"], [1, 4])


def test_duplicate_target_horizons_are_rejected():
    with pytest.raises(ValueError, match="duplicate 1"):
        build_selection_plan("pooled", ["GDP", "CPI"], [1, 1])


def test_overlapping_group_definitions_are_rejected():
    with pytest.raises(ValueError, match="may not overlap"):
        build_selection_plan(
            "group",
            ["GDP", "CPI", "UNR"],
            [1],
            variable_groups=[
                ("Macro", ["GDP", "CPI"]),
                ("Inflation", ["CPI", "UNR"]),
            ],
        )


def test_group_scope_rejects_missing_targets_without_residual_group():
    with pytest.raises(ValueError, match="must cover every requested variable"):
        build_selection_plan(
            "group",
            ["GDP", "CPI"],
            [1, 4],
            variable_groups=[("Real Activity", ["GDP"])],
        )


def test_unknown_variable_and_horizon_lookup_raise_key_error():
    plan = build_selection_plan("variable_horizon", ["GDP", "CPI"], [1, 4])

    with pytest.raises(KeyError, match="Unknown variable"):
        plan.cell_for("UNR", 1)
    with pytest.raises(KeyError, match="Unknown horizon"):
        plan.cell_for("GDP", 8)
    with pytest.raises(KeyError, match="Unknown cell_id"):
        plan.targets_for("missing-cell")


def test_selection_plan_enforces_exact_target_coverage():
    with pytest.raises(ValueError, match="missing target mappings"):
        SelectionPlan(
            scope="pooled",
            target_variables=("GDP", "CPI"),
            target_horizons=(1, 4),
            cells=(
                TargetCell("partial-gdp", ("GDP",), (1, 4)),
                TargetCell("partial-cpi", ("CPI",), (1,)),
            ),
        )

    with pytest.raises(ValueError, match="exactly one cell"):
        SelectionPlan(
            scope="pooled",
            target_variables=("GDP",),
            target_horizons=(1,),
            cells=(
                TargetCell("first", ("GDP",), (1,)),
                TargetCell("second", ("GDP",), (1,)),
            ),
        )


def test_every_requested_target_pair_maps_to_exactly_one_cell():
    plan = build_selection_plan("variable_horizon", ["GDP", "CPI"], [1, 4])

    expected_targets = (
        TargetKey("GDP", 1),
        TargetKey("GDP", 4),
        TargetKey("CPI", 1),
        TargetKey("CPI", 4),
    )
    mapped_targets = tuple(
        target
        for cell in plan.cells
        for target in plan.targets_for(cell.cell_id)
    )

    assert mapped_targets == expected_targets
    for target in expected_targets:
        assert plan.cell_for(target.variable, target.horizon).cell_id in {
            "variable-gdp-h1",
            "variable-gdp-h4",
            "variable-cpi-h1",
            "variable-cpi-h4",
        }