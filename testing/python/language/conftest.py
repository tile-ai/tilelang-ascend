"""
Pytest configuration and shared fixtures for language API tests.

Fixtures defined here are automatically available to all test files
in testing/python/language/ and its subdirectories — no import needed.
They are NOT autouse: test suites opt in via ``usefixtures`` so that
pre-existing tests (e.g. select/elementwise) keep their own seed/cache
behavior untouched.

Tests use the project-standard markers only (low_priority / ci_skip,
see docs/pytest_marker_guide.md); Gate/functional cases run by default,
boundary cases are marked low_priority.
"""

import pytest
import torch
import tilelang


# ---------------------------------------------------------------------------
# Fixtures (opt-in, not autouse)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def disable_tilelang_cache():
    """Disable tilelang JIT cache for the entire test session."""
    tilelang.disable_cache()


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
