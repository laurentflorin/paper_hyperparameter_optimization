import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common_hpo.losses import (
    DuplicateErrorRecord,
    ForecastErrorRecord,
    LossConfig,
    LossConfigurationError,
    MissingCellError,
    ScaleConfig,
    absolute_error,
    attach_benchmark_errors,
    compute_outer_report_metrics,
    evaluate_selection_loss,
    squared_error,
)


def _record(origin, variable, horizon, forecast, realization, **kwargs):
    return ForecastErrorRecord(
        origin=origin,
        variable=variable,
        horizon=horizon,
        forecast=forecast,
        realization=realization,
        **kwargs,
    )


def test_point_metric_helpers():
    assert squared_error(2.0, 5.0) == 9.0
    assert absolute_error(2.0, 5.0) == 3.0


def test_hand_calculated_rmse_single_cell():
    # errors: 1, -2, 3 -> squared 1,4,9 -> mean 14/3 -> rmse sqrt(14/3)
    records = [
        _record("o1", "GDP", 1, 0.0, 1.0),
        _record("o2", "GDP", 1, 0.0, -2.0),
        _record("o3", "GDP", 1, 0.0, 3.0),
    ]
    result = evaluate_selection_loss(records, LossConfig(aggregation="rmse"))
    assert result.value == pytest.approx((14.0 / 3.0) ** 0.5)
    assert result.n_cells == 1
    assert result.n_observations == 3


def test_hand_calculated_mae_single_cell():
    records = [
        _record("o1", "GDP", 1, 0.0, 1.0),
        _record("o2", "GDP", 1, 0.0, -2.0),
        _record("o3", "GDP", 1, 0.0, 3.0),
    ]
    result = evaluate_selection_loss(records, LossConfig(aggregation="mae"))
    # |1|+|2|+|3| = 6 -> mean 2.0
    assert result.value == pytest.approx(2.0)
    assert result.point_metric == "absolute_error"


def test_equal_cell_versus_equal_observation_aggregation():
    # Cell A (GDP,1): one obs with error 2 -> sq 4
    # Cell B (CPI,1): three obs with error 0 -> sq 0
    records = [
        _record("o1", "GDP", 1, 0.0, 2.0),
        _record("o1", "CPI", 1, 0.0, 0.0),
        _record("o2", "CPI", 1, 0.0, 0.0),
        _record("o3", "CPI", 1, 0.0, 0.0),
    ]
    equal_cell = evaluate_selection_loss(
        records, LossConfig(aggregation="mse", cell_aggregation="equal_cell")
    )
    # equal_cell: (4 + 0) / 2 = 2.0
    assert equal_cell.value == pytest.approx(2.0)

    equal_obs = evaluate_selection_loss(
        records, LossConfig(aggregation="mse", cell_aggregation="equal_observation")
    )
    # equal_observation: (1*4 + 3*0) / 4 = 1.0
    assert equal_obs.value == pytest.approx(1.0)


def test_legacy_raw_rmse_equivalence():
    # equal_observation + scale none + uniform weights reproduces plain RMSE.
    errors = [1.0, -2.0, 3.0, 0.5, -1.5]
    records = [
        _record(f"o{i}", "GDP", 1 if i % 2 == 0 else 2, 0.0, e)
        for i, e in enumerate(errors)
    ]
    result = evaluate_selection_loss(
        records, LossConfig(aggregation="rmse", cell_aggregation="equal_observation")
    )
    expected = (sum(e * e for e in errors) / len(errors)) ** 0.5
    assert result.value == pytest.approx(expected)


def test_target_standardization_divides_by_sample_std():
    samples = {"GDP": [10.0, 12.0, 14.0, 16.0]}  # sample std (ddof=1)
    import statistics

    std = statistics.stdev(samples["GDP"])
    records = [
        _record("o1", "GDP", 1, 0.0, std),   # standardized error 1
        _record("o2", "GDP", 1, 0.0, -std),  # standardized error -1
    ]
    scale = ScaleConfig(method="target_std", target_samples=samples)
    result = evaluate_selection_loss(records, LossConfig(aggregation="rmse", scale=scale))
    assert result.value == pytest.approx(1.0)
    assert result.cells[0].scale == pytest.approx(std)


def test_benchmark_scaling_uses_only_supplied_records():
    # benchmark errors 3 and 4 -> benchmark rmse = sqrt((9+16)/2) = 3.5355...
    records = [
        _record("o1", "GDP", 1, 0.0, 3.5355339059327378, benchmark_error=3.0),
        _record("o2", "GDP", 1, 0.0, -3.5355339059327378, benchmark_error=4.0),
    ]
    scale = ScaleConfig(method="benchmark_rmse")
    result = evaluate_selection_loss(records, LossConfig(aggregation="rmse", scale=scale))
    bench_rmse = ((9.0 + 16.0) / 2.0) ** 0.5
    assert result.cells[0].scale == pytest.approx(bench_rmse)
    assert result.value == pytest.approx(1.0)


def test_variable_horizon_and_origin_weights():
    records = [
        _record("o1", "GDP", 1, 0.0, 2.0),
        _record("o1", "CPI", 2, 0.0, 4.0),
    ]
    config = LossConfig(
        aggregation="mse",
        variable_weights={"GDP": 3.0, "CPI": 1.0},
        horizon_weights={1: 1.0, 2: 1.0},
    )
    result = evaluate_selection_loss(records, config)
    # cell GDP,1 value 4 weight 3; cell CPI,2 value 16 weight 1
    # normalized: (3*4 + 1*16)/(3+1) = 28/4 = 7
    assert result.value == pytest.approx(7.0)


def test_origin_weights_affect_within_cell_average():
    records = [
        _record("o1", "GDP", 1, 0.0, 2.0),  # sq 4
        _record("o2", "GDP", 1, 0.0, 4.0),  # sq 16
    ]
    config = LossConfig(aggregation="mse", origin_weights={"o1": 3.0, "o2": 1.0})
    result = evaluate_selection_loss(records, config)
    # weighted mean: (3*4 + 1*16)/4 = 7
    assert result.value == pytest.approx(7.0)


def test_unnormalized_weighting_is_a_weighted_sum():
    records = [
        _record("o1", "GDP", 1, 0.0, 2.0),
        _record("o1", "CPI", 1, 0.0, 2.0),
    ]
    normalized = evaluate_selection_loss(records, LossConfig(aggregation="mse"))
    unnormalized = evaluate_selection_loss(
        records, LossConfig(aggregation="mse", normalized=False)
    )
    # normalized: (4+4)/2 = 4 ; unnormalized: 4+4 = 8
    assert normalized.value == pytest.approx(4.0)
    assert unnormalized.value == pytest.approx(8.0)


def test_near_zero_scale_is_floored_and_recorded():
    records = [
        _record("o1", "GDP", 1, 0.0, 1.0),
        _record("o2", "GDP", 1, 0.0, 1.0),
    ]
    scale = ScaleConfig(
        method="supplied",
        supplied_scales={("GDP", 1): 1e-20},
        min_scale=1e-6,
    )
    result = evaluate_selection_loss(records, LossConfig(aggregation="rmse", scale=scale))
    assert result.cells[0].scale == pytest.approx(1e-6)
    assert result.cells[0].scale_floored is True
    import math

    assert math.isfinite(result.value)


def test_missing_supplied_scale_raises():
    records = [_record("o1", "GDP", 1, 0.0, 1.0)]
    scale = ScaleConfig(method="supplied", supplied_scales={("CPI", 1): 2.0})
    with pytest.raises(MissingCellError, match="no supplied scale"):
        evaluate_selection_loss(records, LossConfig(scale=scale))


def test_missing_benchmark_error_raises():
    records = [_record("o1", "GDP", 1, 0.0, 1.0)]  # no benchmark error
    scale = ScaleConfig(method="benchmark_rmse")
    with pytest.raises(MissingCellError, match="benchmark error"):
        evaluate_selection_loss(records, LossConfig(scale=scale))


def test_duplicate_error_records_raise():
    records = [
        _record("o1", "GDP", 1, 0.0, 1.0),
        _record("o1", "GDP", 1, 0.0, 2.0),
    ]
    with pytest.raises(DuplicateErrorRecord, match="duplicate"):
        evaluate_selection_loss(records)


def test_scale_invariance_under_common_positive_constant():
    samples = {"GDP": [1.0, 2.0, 3.0, 4.0]}
    base_records = [
        _record("o1", "GDP", 1, 1.0, 1.5),
        _record("o2", "GDP", 1, 2.0, 1.0),
    ]
    scale = ScaleConfig(method="target_std", target_samples=samples)
    base = evaluate_selection_loss(base_records, LossConfig(scale=scale))

    c = 7.5
    scaled_records = [
        _record("o1", "GDP", 1, 1.0 * c, 1.5 * c),
        _record("o2", "GDP", 1, 2.0 * c, 1.0 * c),
    ]
    scaled_scale = ScaleConfig(
        method="target_std",
        target_samples={"GDP": [v * c for v in samples["GDP"]]},
    )
    scaled = evaluate_selection_loss(scaled_records, LossConfig(scale=scaled_scale))
    assert scaled.value == pytest.approx(base.value)


def test_deterministic_diagnostics_ordering():
    records = [
        _record("o1", "GDP", 4, 0.0, 1.0),
        _record("o1", "CPI", 2, 0.0, 1.0),
        _record("o1", "GDP", 1, 0.0, 1.0),
        _record("o1", "CPI", 1, 0.0, 1.0),
    ]
    result = evaluate_selection_loss(records, LossConfig(aggregation="mse"))
    ordering = [(c.variable, c.horizon) for c in result.cells]
    assert ordering == sorted(ordering)
    assert ordering == [("CPI", 1), ("CPI", 2), ("GDP", 1), ("GDP", 4)]


def test_results_and_records_are_serializable():
    records = [_record("o1", "GDP", 1, 0.0, 2.0)]
    result = evaluate_selection_loss(records, LossConfig(aggregation="rmse"))
    payload = {
        "config": LossConfig(aggregation="rmse").to_dict(),
        "records": [r.to_dict() for r in records],
        "result": result.to_dict(),
    }
    encoded = json.dumps(payload)
    assert "GDP" in encoded


def test_benchmark_callback_attaches_errors():
    records = [
        _record("o1", "GDP", 1, 0.0, 5.0),
        _record("o2", "GDP", 1, 0.0, 7.0),
    ]

    def no_change_benchmark(*, variable, horizon, origin):
        return 4.0

    enriched = attach_benchmark_errors(records, no_change_benchmark)
    assert enriched[0].benchmark_forecast == 4.0
    assert enriched[0].benchmark_error == pytest.approx(1.0)
    assert enriched[1].benchmark_error == pytest.approx(3.0)


def test_outer_report_metrics_are_distinct_from_selection_loss():
    records = [
        _record("o1", "GDP", 1, 0.0, 2.0),
        _record("o2", "GDP", 1, 0.0, 0.0),
        _record("o1", "CPI", 1, 0.0, 4.0),
    ]
    report = compute_outer_report_metrics(records, aggregation="rmse")
    assert report["purpose"] == "outer_report_only"
    # pooled raw rmse over all residuals: sqrt((4+0+16)/3)
    assert report["pooled_value"] == pytest.approx(((4 + 0 + 16) / 3) ** 0.5)
    cell_order = [(c["variable"], c["horizon"]) for c in report["cells"]]
    assert cell_order == sorted(cell_order)


def test_empty_records_raise():
    with pytest.raises(LossConfigurationError):
        evaluate_selection_loss([])
