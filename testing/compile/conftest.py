# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import os
import sys
from typing import List, Optional, Sequence

import pytest

_compile_root = os.path.dirname(os.path.abspath(__file__))
if _compile_root not in sys.path:
    sys.path.insert(0, _compile_root)

from testcommon import ascend_mode


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers to avoid PytestUnknownMarkWarning."""
    config.addinivalue_line(
        "markers",
        "mode(name): set TILELANG_ASCEND_MODE for the test (e.g. Expert, Developer).",
    )


def _parse_csv(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _marker_value(item: pytest.Item, marker_name: str) -> Optional[str]:
    marker = item.get_closest_marker(marker_name)
    if marker is None:
        return None

    values = [str(arg) for arg in marker.args]
    values.extend(str(val) for val in marker.kwargs.values())

    if len(values) != 1:
        raise pytest.UsageError(
            f"{item.nodeid}: @{marker_name} must define exactly one value."
        )
    return values[0]


def _matches_filter(item: pytest.Item, marker_name: str, selected: Sequence[str]) -> bool:
    if not selected:
        return True
    marker_value = _marker_value(item, marker_name)
    return marker_value in selected if marker_value is not None else False


def pytest_addoption(parser):
    parser.addoption(
        "--mode",
        action="store",
        default="",
        help="Run only tests for specific ASCEND mode(s), comma-separated (Expert, Developer).",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: List[pytest.Item]) -> None:
    selected_modes = _parse_csv(config.getoption("--mode"))
    if not selected_modes:
        return

    selected_items: List[pytest.Item] = []
    deselected_items: List[pytest.Item] = []

    for item in items:
        keep = _matches_filter(item, "mode", selected_modes)
        if keep:
            selected_items.append(item)
        else:
            deselected_items.append(item)

    if deselected_items:
        config.hook.pytest_deselected(items=deselected_items)
    items[:] = selected_items


@pytest.fixture(autouse=True)
def _apply_mode_marker(request: pytest.FixtureRequest):
    """Apply TILELANG_ASCEND_MODE from pytest.mark.mode so batch runs don't mix modes."""
    mode = _marker_value(request.node, "mode")
    if mode is None:
        # Default to Expert when no marker (e.g. legacy tests).
        os.environ["TILELANG_ASCEND_MODE"] = "Expert"
        yield
        return

    with ascend_mode(mode):
        yield
