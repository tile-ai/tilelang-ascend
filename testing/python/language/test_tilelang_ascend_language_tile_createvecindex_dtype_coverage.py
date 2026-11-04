"""
T.tile.createvecindex dtype coverage tests.

Existing coverage in test_tilelang_ascend_language_elementwise.py:
- int16/int32/float16/float32 x ascendc x 2D (M=1, N=1024, firstValue=0)
- int16/int32/uint16/uint32 x pto x 2D (M=1, N=1024, firstValue=0)

This file supplements with:
1. 1D shape tests (float32/int16/int32 x ascendc/pto, firstValue=10)
   - Also covers float32 x pto (not in existing tests)
   - Non-zero firstValue (existing tests only use 0)
2. Exception boundary tests (unsupported dtype compilation failure)

dtype x backend support matrix (verified on real hardware):
| dtype    | ascendc | pto  |
|----------|---------|------|
| float16  | PASS    | FAIL |
| float32  | PASS    | PASS |
| int16    | PASS    | PASS |
| int32    | PASS    | PASS |
| uint16   | FAIL    | PASS |
| uint32   | FAIL    | PASS |
| bfloat16 | FAIL    | FAIL |
| int8     | FAIL    | FAIL |
| uint8    | FAIL    | FAIL |

Main table (both backends): float32, int16, int32
Single-backend: float16 (ascendc only), uint16/uint32 (pto only)
"""

import pytest
import tilelang
import tilelang.language as T
import torch


@pytest.fixture(scope="module", autouse=True)
def _disable_cache():
    tilelang.disable_cache()
    yield
    tilelang.enable_cache()


PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "int16": torch.int16,
    "int32": torch.int32,
}

N = 128
FIRST_VALUE = 10


def make_kernel_1d(dtype, n, first_value):
    @T.prim_func
    def main(
        C: T.Tensor((n,), dtype),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            c_ub = T.alloc_ub((n,), dtype)
            T.tile.createvecindex(c_ub, first_value)
            T.copy(c_ub, C[0])

    return main


def make_ref(dtype_str, first_value, count):
    torch_dtype = TORCH_DTYPE_MAP[dtype_str]
    if dtype_str in ("int16",):
        ref = torch.arange(start=first_value, end=first_value + count, dtype=torch.int32).to(torch_dtype)
    else:
        ref = torch.arange(start=first_value, end=first_value + count, dtype=torch_dtype)
    return ref


@pytest.mark.parametrize(
    "dtype",
    [
        "float32",
        pytest.param("int16", marks=pytest.mark.low_priority),
        pytest.param("int32", marks=pytest.mark.low_priority),
    ],
)
@pytest.mark.parametrize(
    "target",
    [
        "ascendc",
        pytest.param("pto", marks=pytest.mark.low_priority),
    ],
)
def test_createvecindex_1d(dtype, target):
    kernel = make_kernel_1d(dtype, N, FIRST_VALUE)
    func = tilelang.compile(kernel, out_idx=[-1], pass_configs=PASS_CONFIGS, target=target)

    torch.npu.synchronize()
    c = func()
    torch.npu.synchronize()

    ref = make_ref(dtype, FIRST_VALUE, N).npu()
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "dtype,target",
    [
        ("float16", "pto"),
        ("uint16", "ascendc"),
    ],
)
def test_createvecindex_unsupported_dtype_raises(dtype, target):
    """Single-backend dtype should fail on the other backend at compile time.

    - float16@pto: ascendc-only dtype, pto fails
    - uint16@ascendc: pto-only dtype, ascendc fails (same for uint32)
    """

    @T.prim_func
    def main(
        C: T.Tensor((N,), dtype),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            c_ub = T.alloc_ub((N,), dtype)
            T.tile.createvecindex(c_ub, 0)
            T.copy(c_ub, C[0])

    with pytest.raises(RuntimeError, match="Compilation Failed"):  # noqa: B017
        tilelang.compile(main, out_idx=[-1], pass_configs=PASS_CONFIGS, target=target)


def test_create_vec_index_snake_case_alias():
    """snake_case alias must accept first_value= keyword argument."""

    @T.prim_func
    def main(
        C: T.Tensor((N,), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            c_ub = T.alloc_ub((N,), "float32")
            T.tile.create_vec_index(c_ub, first_value=FIRST_VALUE)
            T.copy(c_ub, C[0])

    func = tilelang.compile(main, out_idx=[-1], pass_configs=PASS_CONFIGS, target="ascendc")

    torch.npu.synchronize()
    c = func()
    torch.npu.synchronize()

    ref = make_ref("float32", FIRST_VALUE, N).npu()
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-n", "8"])
