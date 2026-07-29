import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


DEQUANT_GEMM_FP16_CASES = [
    pytest.param(256, 256, 256, "float16", "float16", id="fp16_256"),
    pytest.param(512, 512, 512, "float16", "float16", id="fp16_512"),
    pytest.param(1024, 1024, 1024, "float16", "float16", id="fp16_1024"),
    pytest.param(512, 512, 512, "bfloat16", "bfloat16", id="bf16_512"),
]

DEQUANT_GEMM_INT8_CASES = [
    pytest.param(256, 256, 256, "int32", id="int8_256_int32"),
    pytest.param(512, 512, 512, "int32", id="int8_512_int32"),
    pytest.param(1024, 1024, 1024, "int32", id="int8_1024_int32"),
    pytest.param(512, 512, 512, "float16", id="int8_512_float16"),
]


def _load_dequant_gemm_fine_grained_example() -> ModuleType:
    source = Path(__file__).with_name("example_dequant_gemm_fine_grained.py")
    spec = importlib.util.spec_from_file_location("_dequant_gemm_fine_grained_example_for_test", source)
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


def test_dequant_gemm_fine_grained_unpack_accuracy() -> None:
    example = _load_dequant_gemm_fine_grained_example()
    example.check_unpack_functions()


@pytest.mark.parametrize(("m", "n", "k", "input_dtype", "output_dtype"), DEQUANT_GEMM_FP16_CASES)
def test_dequant_gemm_fine_grained_fp16_accuracy(
    m: int,
    n: int,
    k: int,
    input_dtype: str,
    output_dtype: str,
) -> None:
    import torch

    example = _load_dequant_gemm_fine_grained_example()

    torch.manual_seed(0)
    example.check_case_fp16(
        m,
        n,
        k,
        block_M=128,
        block_N=256,
        block_K=64,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
    )


@pytest.mark.parametrize(("m", "n", "k", "output_dtype"), DEQUANT_GEMM_INT8_CASES)
def test_dequant_gemm_fine_grained_int8_accuracy(
    m: int,
    n: int,
    k: int,
    output_dtype: str,
) -> None:
    import torch

    example = _load_dequant_gemm_fine_grained_example()

    torch.manual_seed(0)
    example.check_case_int8(
        m,
        n,
        k,
        block_M=128,
        block_N=128,
        block_K=64,
        output_dtype=output_dtype,
    )
