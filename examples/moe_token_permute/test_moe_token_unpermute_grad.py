import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_example() -> ModuleType:
    source = Path(__file__).with_name("moe_token_unpermute_grad.py")
    spec = importlib.util.spec_from_file_location("_moe_token_unpermute_grad_example_for_test", source)
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


# The example compares against a torch implementation inside
# test_unpermute_grad_parameterized and returns whether it matched; its driver prints the
# outcome and returns nothing, so the value has to be asserted here or a
# mismatch would pass. The three dtypes are the ones the example itself walks.
@pytest.mark.parametrize(
    "torch_dtype_name, tilelang_dtype",
    [
        ("float16", "float16"),
        ("bfloat16", "bfloat16"),
        ("float32", "float32"),
    ],
)
def test_unpermute_grad_accuracy(torch_dtype_name, tilelang_dtype) -> None:
    import torch

    example = _load_example()

    passed = example.test_unpermute_grad_parameterized(pt_dtype=getattr(torch, torch_dtype_name), tl_dtype_str=tilelang_dtype)

    assert passed, f"unpermute_grad mismatched the torch reference for {tilelang_dtype}"
