import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


BLOCKSPARSE_GEMM_CASES = [
    pytest.param("basic", id="basic"),
    pytest.param("typical", id="typical"),
    pytest.param("boundary_dense", id="boundary_dense"),
    pytest.param("boundary_sparse", id="boundary_sparse"),
]


def _load_blocksparse_gemm_example() -> ModuleType:
    source = Path(__file__).with_name("example_blocksparse_gemm.py")
    spec = importlib.util.spec_from_file_location("_blocksparse_gemm_example_for_test", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load example module: {source}")

    module = importlib.util.module_from_spec(spec)
    original_argv = sys.argv
    try:
        sys.argv = [str(source)]
        spec.loader.exec_module(module)
    finally:
        sys.argv = original_argv
    return module


@pytest.mark.parametrize("case_name", BLOCKSPARSE_GEMM_CASES)
def test_blocksparse_gemm_accuracy(case_name: str) -> None:
    example = _load_blocksparse_gemm_example()

    if case_name == "basic":
        example.test_basic()
    elif case_name == "typical":
        example.test_typical()
    elif case_name == "boundary_dense":
        example.test_boundary_dense()
    elif case_name == "boundary_sparse":
        example.test_boundary_sparse()
    else:
        raise AssertionError(f"Unknown blocksparse_gemm case: {case_name}")
