"""
Pytest configuration and shared fixtures for language API tests.

Fixtures are available under testing/ascend_a2a3/python/language/ without
imports. Seed/cache fixtures remain opt-in through ``usefixtures``. The
private source environment applies only to the adopted generated max/min
classes; other tests retain their own seed/cache/platform behavior.

Tests use the project-standard markers only (low_priority / ci_skip,
see docs/pytest_marker_guide.md); Gate/functional cases run by default,
boundary cases are marked low_priority.
"""

import pytest
import torch
from pathlib import Path
from tilelang.env import CacheState


_GENERATED_BINARY_MODULES = {"test_tile_max.py", "test_tile_min.py"}


def _generated_binary_case(item):
    return (
        getattr(item, "cls", None) is not None and item.path.parent == Path(__file__).parent and item.path.name in _GENERATED_BINARY_MODULES
    )


def pytest_collection_modifyitems(items):
    # Preserve the original registered classes and dtype/priority collector.
    # Their compile and trace-time negative cases need no NPU execution.
    for item in items:
        if not _generated_binary_case(item):
            continue
        frontend_negative = item.name.split("[", 1)[0] in ("test_size_mismatch_src0", "test_size_mismatch_region")
        native = item.cls.__name__.endswith("Compile") or frontend_negative
        item.add_marker(pytest.mark.a2a3_native if native else pytest.mark.a2a3_device)


@pytest.fixture(autouse=True)
def _generated_binary_source_environment(request, monkeypatch):
    if not _generated_binary_case(request.node) or request.node.get_closest_marker("a2a3_native") is None:
        return
    monkeypatch.setenv("TL_PLATFORM", "A3")
    if request.node.cls.__name__.endswith("Compile"):
        # Public compile constructs the original Cython adapter after SDK load;
        # torch's NPU device type must be registered even without NPU execution.
        pytest.importorskip("torch_npu")


# ---------------------------------------------------------------------------
# Fixtures (opt-in, not autouse)
# ---------------------------------------------------------------------------


@pytest.fixture
def disable_tilelang_cache(monkeypatch):
    """Disable JIT cache for this requesting case and restore the caller bit."""
    monkeypatch.setattr(CacheState, "_enabled", False)


@pytest.fixture
def random_seed():
    """Set deterministic random seed for reproducibility."""
    torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Marker registration (moved to pyproject.toml [tool.pytest.ini_options])
# ---------------------------------------------------------------------------


def pytest_generate_tests(metafunc):
    """Parametrize 'dtype' from the test class's op spec.

    Each base test class declares a _dtype_source attribute (e.g.
    "supported_dtypes" or "boundary_dtypes") that points to a list on
    the BinaryOpSpec.  This hook reads that list and feeds it to pytest.

    Dtypes listed in ``op.low_priority_dtypes`` are marked with the
    project ``low_priority`` marker (skipped on PR, run in full tests).
    """
    if "dtype" not in metafunc.fixturenames or metafunc.cls is None:
        return
    op = getattr(metafunc.cls, "op", None)
    if op is None:
        return
    source = getattr(metafunc.cls, "_dtype_source", "supported_dtypes")
    dtypes = getattr(op, source, None)
    if not dtypes:
        return
    low_priority = set(getattr(op, "low_priority_dtypes", []) or [])
    params = []
    for d in dtypes:
        if d in low_priority:
            params.append(pytest.param(d, marks=pytest.mark.low_priority))
        else:
            params.append(d)
    metafunc.parametrize("dtype", params)
