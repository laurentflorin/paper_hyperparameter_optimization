from __future__ import annotations

from pathlib import Path

import pytest


OPTIONAL_MARKERS = {"integration", "slow", "requires_covbayesvar", "requires_mbfvar"}
FILE_MARKERS = {
    "test_glp_compare_forecasts.py": ("integration",),
    "test_glp_data_utils.py": ("integration",),
    "test_glp_forecasting.py": ("integration",),
    "test_glp_model.py": ("integration", "requires_covbayesvar"),
    "test_glp_run_all.py": ("integration",),
    "test_optimizer_variable_resolution.py": ("integration", "requires_mbfvar"),
    "test_reporting_ops.py": ("integration",),
}
NODE_MARKERS = {
    "test_mfvar_scope_grid.py::test_existing_scripts_still_import": ("integration", "requires_mbfvar"),
}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--skip-optional",
        action="store_true",
        default=False,
        help="Skip integration, slow, and optional-dependency tests.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_optional = config.getoption("--skip-optional")
    skip_marker = pytest.mark.skip(reason="optional integration/dependency test skipped")

    for item in items:
        filename = Path(str(item.fspath)).name
        for marker_name in FILE_MARKERS.get(filename, ()):
            item.add_marker(getattr(pytest.mark, marker_name))
        for suffix, marker_names in NODE_MARKERS.items():
            if item.nodeid.endswith(suffix):
                for marker_name in marker_names:
                    item.add_marker(getattr(pytest.mark, marker_name))

        has_optional_marker = any(item.get_closest_marker(name) for name in OPTIONAL_MARKERS)
        if not has_optional_marker:
            item.add_marker(pytest.mark.unit)
        elif item.get_closest_marker("integration") is None:
            item.add_marker(pytest.mark.integration)

        if skip_optional and has_optional_marker:
            item.add_marker(skip_marker)