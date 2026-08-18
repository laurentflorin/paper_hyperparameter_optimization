import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common_hpo.splits import (
    InfeasibleValidationDesign,
    ValidationScheme,
    ValidationSplit,
    VintagePolicy,
    assert_rolling_window_length,
    assert_sorted_deterministically,
    assert_split_is_leakage_safe,
    build_validation_splits,
    resolve_horizon_offsets,
    verify_validation_splits,
)


# A fake model adapter: canonical horizons are quarters, but the data matrix is
# monthly, so each quarter maps to three data rows. The split engine never needs
# to know this.
def monthly_to_quarterly_offsets(horizon_in_quarters: int) -> int:
    return 3 * horizon_in_quarters


def _same_frequency_offsets(horizons):
    return {h: h for h in horizons}


def test_hand_calculated_multistep_positions_same_frequency():
    scheme = ValidationScheme(
        training_window="expanding",
        origin_selection="most_recent",
        n_origins=1,
        horizons=(1, 2, 4, 8),
        min_train_length=4,
    )
    # n_positions large enough for the 8-step target; last origin.
    splits = build_validation_splits(
        n_positions=20,
        scheme=scheme,
        horizon_row_offsets=_same_frequency_offsets(scheme.horizons),
    )
    assert len(splits) == 1
    split = splits[0]
    # info_cutoff = 19, max offset 8 -> origin_max = 11.
    assert split.origin == 11
    assert split.train_start == 0
    assert split.train_end == 11
    assert split.target_for(1) == 12
    assert split.target_for(2) == 13
    assert split.target_for(4) == 15
    assert split.target_for(8) == 19  # exactly the outer boundary
    assert split.info_cutoff == 19


def test_hand_calculated_positions_monthly_to_quarterly_adapter():
    scheme = ValidationScheme(
        training_window="expanding",
        origin_selection="most_recent",
        n_origins=1,
        horizons=(1, 2, 4, 8),
        min_train_length=12,
    )
    # 8 quarters -> 24 rows. Choose info_cutoff so origin lands on a round number.
    splits = build_validation_splits(
        n_positions=50,
        scheme=scheme,
        horizon_row_offsets=monthly_to_quarterly_offsets,
    )
    split = splits[0]
    # info_cutoff = 49, max offset = 24 -> origin_max = 25.
    assert split.origin == 25
    assert split.target_for(1) == 25 + 3
    assert split.target_for(2) == 25 + 6
    assert split.target_for(4) == 25 + 12
    assert split.target_for(8) == 25 + 24
    assert split.target_for(8) == 49  # exact outer boundary


def test_expanding_windows_grow_from_zero():
    scheme = ValidationScheme(
        training_window="expanding",
        origin_selection="most_recent",
        n_origins=3,
        horizons=(1,),
        min_train_length=5,
    )
    splits = build_validation_splits(
        n_positions=20,
        scheme=scheme,
        horizon_row_offsets={1: 1},
    )
    assert [s.origin for s in splits] == [16, 17, 18]
    assert all(s.train_start == 0 for s in splits)
    assert [s.train_end for s in splits] == [16, 17, 18]


def test_rolling_windows_have_requested_length():
    scheme = ValidationScheme(
        training_window="rolling",
        origin_selection="most_recent",
        n_origins=4,
        horizons=(1, 2),
        min_train_length=6,
        rolling_window_length=6,
    )
    splits = build_validation_splits(
        n_positions=20,
        scheme=scheme,
        horizon_row_offsets={1: 1, 2: 2},
    )
    assert_rolling_window_length(splits, 6)
    for split in splits:
        assert split.train_end - split.train_start + 1 == 6
        assert split.train_start == split.origin - 5


def test_origin_stride_controls_spacing():
    scheme = ValidationScheme(
        training_window="expanding",
        origin_selection="most_recent",
        n_origins=3,
        horizons=(1,),
        min_train_length=4,
        origin_stride=3,
    )
    splits = build_validation_splits(
        n_positions=20,
        scheme=scheme,
        horizon_row_offsets={1: 1},
    )
    origins = [s.origin for s in splits]
    # origin_max = 18, stride 3 -> grid ..., 12, 15, 18; last three selected.
    assert origins == [12, 15, 18]
    diffs = {b - a for a, b in zip(origins, origins[1:])}
    assert diffs == {3}


def test_evenly_spaced_origins_cover_the_range():
    scheme = ValidationScheme(
        training_window="expanding",
        origin_selection="evenly_spaced",
        n_origins=3,
        horizons=(1,),
        min_train_length=4,
    )
    splits = build_validation_splits(
        n_positions=20,
        scheme=scheme,
        horizon_row_offsets={1: 1},
    )
    origins = [s.origin for s in splits]
    # feasible origins 3..18; evenly spaced endpoints included.
    assert origins[0] == 3
    assert origins[-1] == 18
    assert origins == sorted(origins)


def test_infeasible_early_origin_raises():
    scheme = ValidationScheme(
        training_window="expanding",
        origin_selection="most_recent",
        n_origins=1,
        horizons=(8,),
        min_train_length=20,
    )
    # Need origin >= 19 (min train) but origin_max = (12-1) - 8 = 3.
    with pytest.raises(InfeasibleValidationDesign, match="no feasible validation origin"):
        build_validation_splits(
            n_positions=12,
            scheme=scheme,
            horizon_row_offsets={8: 8},
        )


def test_requesting_too_many_origins_raises():
    scheme = ValidationScheme(
        training_window="expanding",
        origin_selection="most_recent",
        n_origins=50,
        horizons=(1,),
        min_train_length=4,
    )
    with pytest.raises(InfeasibleValidationDesign, match="more validation origins"):
        build_validation_splits(
            n_positions=20,
            scheme=scheme,
            horizon_row_offsets={1: 1},
        )


def test_exact_outer_boundary_is_allowed_but_one_past_is_not():
    scheme = ValidationScheme(
        training_window="expanding",
        origin_selection="most_recent",
        n_origins=1,
        horizons=(4,),
        min_train_length=4,
    )
    # info_cutoff explicitly at the boundary: origin_max = cutoff - 4.
    splits = build_validation_splits(
        n_positions=30,
        scheme=scheme,
        horizon_row_offsets={4: 4},
        outer_info_cutoff=10,
    )
    split = splits[0]
    assert split.origin == 6
    assert split.target_for(4) == 10  # exactly at the cutoff
    assert split.info_cutoff == 10

    # Shrinking the cutoff by one shifts the last origin down by one.
    tighter = build_validation_splits(
        n_positions=30,
        scheme=scheme,
        horizon_row_offsets={4: 4},
        outer_info_cutoff=9,
    )
    assert tighter[0].origin == 5
    assert tighter[0].target_for(4) == 9


def test_random_origin_selection_is_reproducible_for_fixed_seed():
    scheme = ValidationScheme(
        training_window="expanding",
        origin_selection="random",
        n_origins=4,
        horizons=(1,),
        min_train_length=4,
        random_seed=1234,
    )
    first = build_validation_splits(30, scheme, {1: 1})
    second = build_validation_splits(30, scheme, {1: 1})
    assert [s.origin for s in first] == [s.origin for s in second]
    # Deterministically sorted output.
    assert [s.origin for s in first] == sorted(s.origin for s in first)

    other_seed = ValidationScheme(
        training_window="expanding",
        origin_selection="random",
        n_origins=4,
        horizons=(1,),
        min_train_length=4,
        random_seed=9999,
    )
    third = build_validation_splits(30, other_seed, {1: 1})
    # Very likely different selection with a different seed over a wide range.
    assert [s.origin for s in third] != [s.origin for s in first]


def test_random_selection_does_not_mutate_supplied_generator_or_global_state():
    scheme = ValidationScheme(
        training_window="expanding",
        origin_selection="random",
        n_origins=3,
        horizons=(1,),
        min_train_length=4,
        random_seed=None,
    )
    generator = np.random.default_rng(2024)
    state_before = generator.bit_generator.state

    global_state_before = np.random.get_state()

    first = build_validation_splits(30, scheme, {1: 1}, rng=generator)
    second = build_validation_splits(30, scheme, {1: 1}, rng=generator)

    # Supplied generator was read, not advanced.
    assert generator.bit_generator.state == state_before
    # Reproducible for the same generator state.
    assert [s.origin for s in first] == [s.origin for s in second]
    # Global NumPy random state untouched.
    global_state_after = np.random.get_state()
    assert global_state_before[0] == global_state_after[0]
    assert np.array_equal(global_state_before[1], global_state_after[1])


def test_exponentially_decaying_recency_weights():
    scheme = ValidationScheme(
        training_window="expanding",
        origin_selection="most_recent",
        n_origins=3,
        horizons=(1,),
        min_train_length=4,
        recency_decay=0.5,
    )
    splits = build_validation_splits(20, scheme, {1: 1})
    weights = [s.origin_weight for s in splits]
    assert all(w is not None for w in weights)
    assert pytest.approx(sum(weights), rel=1e-12) == 1.0
    # Sorted ascending by origin, so the last (most recent) has the largest weight.
    assert weights[0] < weights[1] < weights[2]
    # Ratios reflect a decay factor of 0.5.
    assert pytest.approx(weights[2] / weights[1], rel=1e-9) == 2.0
    assert pytest.approx(weights[1] / weights[0], rel=1e-9) == 2.0


def test_no_recency_weight_when_disabled():
    scheme = ValidationScheme(
        training_window="expanding",
        origin_selection="most_recent",
        n_origins=2,
        horizons=(1,),
        min_train_length=4,
    )
    splits = build_validation_splits(20, scheme, {1: 1})
    assert all(s.origin_weight is None for s in splits)


def test_date_labels_cross_calendar_year_boundary():
    # Monthly labels spanning 2019-11 .. 2020-06.
    months = [
        "2019-11", "2019-12", "2020-01", "2020-02",
        "2020-03", "2020-04", "2020-05", "2020-06",
    ]
    scheme = ValidationScheme(
        training_window="expanding",
        origin_selection="most_recent",
        n_origins=1,
        horizons=(1, 2),
        min_train_length=4,
    )
    splits = build_validation_splits(
        n_positions=len(months),
        scheme=scheme,
        horizon_row_offsets={1: 1, 2: 2},
        date_labels=months,
    )
    split = splits[0]
    # info_cutoff = 7, max offset 2 -> origin = 5 (2020-04).
    assert split.origin == 5
    assert split.date_labels["origin"] == "2020-04"
    assert split.date_labels["train_start"] == "2019-11"
    assert split.date_labels["info_cutoff"] == "2020-06"
    assert split.date_labels["targets"][1] == "2020-05"
    assert split.date_labels["targets"][2] == "2020-06"


def test_strict_inner_real_time_policy_is_not_implemented():
    scheme = ValidationScheme(
        training_window="expanding",
        origin_selection="most_recent",
        n_origins=1,
        horizons=(1,),
        min_train_length=4,
        vintage_policy=VintagePolicy.STRICT_INNER_REAL_TIME,
    )
    with pytest.raises(NotImplementedError, match="strict inner real-time"):
        build_validation_splits(20, scheme, {1: 1})


def test_outer_vintage_consistent_policy_metadata():
    policy = VintagePolicy.OUTER_VINTAGE_CONSISTENT
    metadata = policy.metadata()
    assert metadata["policy"] == "outer_vintage_consistent"
    assert metadata["implemented"] is True
    assert VintagePolicy.STRICT_INNER_REAL_TIME.metadata()["implemented"] is False


def test_verify_validation_splits_passes_for_valid_designs():
    scheme = ValidationScheme(
        training_window="rolling",
        origin_selection="evenly_spaced",
        n_origins=3,
        horizons=(1, 2, 4),
        min_train_length=8,
        rolling_window_length=8,
    )
    offsets = {1: 1, 2: 2, 4: 4}
    splits = build_validation_splits(40, scheme, offsets)
    verify_validation_splits(splits, scheme, offsets)
    assert_sorted_deterministically(splits)


def test_assert_split_is_leakage_safe_detects_target_in_training_window():
    bad = ValidationSplit(
        split_id="bad",
        train_start=0,
        train_end=10,
        origin=10,
        targets=((1, 5),),  # target inside the training window
        info_cutoff=20,
    )
    with pytest.raises(AssertionError):
        assert_split_is_leakage_safe(bad, {1: 1})


def test_resolve_horizon_offsets_rejects_nonpositive_offsets():
    with pytest.raises(ValueError):
        resolve_horizon_offsets((1, 2), {1: 1, 2: 0})


def test_scheme_rejects_rolling_without_length():
    with pytest.raises(ValueError, match="rolling_window_length is required"):
        ValidationScheme(
            training_window="rolling",
            origin_selection="most_recent",
            n_origins=1,
            horizons=(1,),
            min_train_length=4,
        )


def test_random_selection_requires_seed_or_generator():
    scheme = ValidationScheme(
        training_window="expanding",
        origin_selection="random",
        n_origins=1,
        horizons=(1,),
        min_train_length=4,
    )
    with pytest.raises(ValueError, match="requires a seed or a NumPy generator"):
        build_validation_splits(20, scheme, {1: 1})
