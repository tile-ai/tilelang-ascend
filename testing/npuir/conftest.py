import os
import warnings
from typing import List, Optional, Sequence

import pytest

from testcommon import (
    ascend_mode,
    clear_tilelang_cache,
    resolve_npu_device_id,
    set_npu_device,
    set_seed,
)


def _get_npu_device_id(config: pytest.Config) -> tuple[int, Optional[str]]:
    """Resolve xdist worker index or --npu-device to a visible runtime device id."""
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if worker and worker.startswith("gw"):
        return resolve_npu_device_id(int(worker[2:]))
    return resolve_npu_device_id(config.getoption("--npu-device"))


def pytest_addoption(parser):
    parser.addoption(
        "--op",
        action="store",
        default="",
        help="Run only tests for specific op(s), comma-separated.",
    )
    parser.addoption(
        "--mode",
        action="store",
        default="",
        help="Run only tests for specific ASCEND mode(s), comma-separated.",
    )
    parser.addoption(
        "--npu-device", action="store", type=int, default=0, help="NPU device id."
    )
    parser.addoption(
        "--seed", action="store", type=int, default=42, help="Random seed for tests."
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


def _matches_filter(
    item: pytest.Item, marker_name: str, selected: Sequence[str]
) -> bool:
    if not selected:
        return True
    marker_value = _marker_value(item, marker_name)
    return marker_value in selected if marker_value is not None else False


def pytest_collection_modifyitems(
    config: pytest.Config, items: List[pytest.Item]
) -> None:
    selected_ops = _parse_csv(config.getoption("--op"))
    selected_modes = _parse_csv(config.getoption("--mode"))

    if not (selected_ops or selected_modes):
        return

    selected_items: List[pytest.Item] = []
    deselected_items: List[pytest.Item] = []

    for item in items:
        keep = _matches_filter(item, "op", selected_ops) and _matches_filter(
            item, "mode", selected_modes
        )
        if keep:
            selected_items.append(item)
        else:
            deselected_items.append(item)

    if deselected_items:
        config.hook.pytest_deselected(items=deselected_items)
    items[:] = selected_items


@pytest.fixture(scope="session", autouse=True)
def _setup_npu_session(pytestconfig: pytest.Config):
    seed = pytestconfig.getoption("--seed")
    device_id, warning_message = _get_npu_device_id(pytestconfig)
    if warning_message is not None:
        warnings.warn(pytest.PytestWarning(warning_message), stacklevel=2)
    set_seed(seed)
    set_npu_device(device_id)
    clear_tilelang_cache()


@pytest.fixture(autouse=True)
def _apply_mode_marker(request: pytest.FixtureRequest):
    mode = _marker_value(request.node, "mode")
    if mode is None:
        yield
        return

    with ascend_mode(mode):
        yield
