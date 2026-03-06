import re
from typing import List, Sequence

import pytest

from testcommon import clear_tilelang_cache, set_npu_device, set_seed


def pytest_addoption(parser):
    parser.addoption("--op", action="store", default="", help="Run only tests for specific op(s), comma-separated.")
    parser.addoption(
        "--dtype",
        action="store",
        default="",
        help="Run only tests for specific dtype(s), comma-separated.",
    )
    parser.addoption(
        "--mode",
        action="store",
        default="",
        help="Run only tests for specific ASCEND mode(s), comma-separated.",
    )
    parser.addoption("--npu-device", action="store", type=int, default=0, help="NPU device id.")
    parser.addoption("--seed", action="store", type=int, default=42, help="Random seed for tests.")


def _parse_csv(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _marker_values(item: pytest.Item, marker_name: str) -> List[str]:
    values: List[str] = []
    for marker in item.iter_markers(marker_name):
        values.extend(str(arg) for arg in marker.args)
        values.extend(str(val) for val in marker.kwargs.values())
    return values


def _matches_filter(item: pytest.Item, marker_name: str, selected: Sequence[str]) -> bool:
    if not selected:
        return True
    marker_values = set(_marker_values(item, marker_name))
    if marker_values:
        return any(v in marker_values for v in selected)
    if marker_name == "op":
        test_name = item.name.split("[", 1)[0]
        match = re.match(r"^test_([^_]+)_", test_name)
        if match:
            inferred_op = match.group(1)
            return inferred_op in selected
    return any(v in item.nodeid for v in selected)


def pytest_collection_modifyitems(config: pytest.Config, items: List[pytest.Item]) -> None:
    selected_ops = _parse_csv(config.getoption("--op"))
    selected_dtypes = _parse_csv(config.getoption("--dtype"))
    selected_modes = _parse_csv(config.getoption("--mode"))

    if not (selected_ops or selected_dtypes or selected_modes):
        return

    selected_items: List[pytest.Item] = []
    deselected_items: List[pytest.Item] = []

    for item in items:
        keep = (
            _matches_filter(item, "op", selected_ops)
            and _matches_filter(item, "dtype", selected_dtypes)
            and _matches_filter(item, "mode", selected_modes)
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
    device_id = pytestconfig.getoption("--npu-device")
    set_seed(seed)
    set_npu_device(device_id)
    clear_tilelang_cache()
