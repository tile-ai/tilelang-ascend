"""
T.tile.gather dtype coverage tests.

Existing coverage in test_tilelang_ascend_language_elementwise.py:
- int16/int32/uint16/uint32/float/float16/bfloat16 x ascendc/pto x 2D (128, 1024),
  src_base_addr=0 (test_gather)
- 2D BufferRegion (slice) path x ascendc (test_gather_slice)
- 2D larger src x ascendc (test_gather_larger_src)
- tmp parameter (test_tilelang_ascend_language_explicit_tmp.py)

This file supplements with:
1. Non-zero src_base_addr test (float32 x ascendc, base=64)
   - ftcheck example 2; pto ignores base_addr (documented limitation)
2. Exception boundary tests (constraint violations)
   - dst != src dtype (constraint 1)
   - src_offset != uint32 (constraint 2, ascendc enforces)

dtype x backend support matrix (verified on real hardware):
| dtype    | ascendc | pto  |
|----------|---------|------|
| float16  | PASS    | PASS |
| float32  | PASS    | PASS |
| bfloat16 | PASS    | PASS |
| int16    | PASS    | PASS |
| uint16   | PASS    | PASS |
| int32    | PASS    | PASS |
| uint32   | PASS    | PASS |
| int8     | FAIL*   | FAIL |
| uint8    | FAIL*   | FAIL |

* int8/uint8 on ascendc: compile passes but result incorrect (silent failure)

Main table (both backends): float16, float32, bfloat16, int16, uint16, int32, uint32
No single-backend dtypes.

Known PTO limitations (documented in docs/api_docs/T.tile.gather.md):
- src_base_addr is ignored (codegen assumes 0)
- Minimum 512 elements required (PTO backend bug, not a design constraint)
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


def make_gather_baseaddr_kernel(n_src, n, dtype, base_addr):
    @T.prim_func
    def main(
        A: T.Tensor((n_src,), dtype),  # type: ignore
        B: T.Tensor((n,), "uint32"),  # type: ignore
        C: T.Tensor((n,), dtype),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a_ub = T.alloc_ub((n_src,), dtype)
            b_ub = T.alloc_ub((n,), "uint32")
            c_ub = T.alloc_ub((n,), dtype)
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            T.tile.gather(c_ub, a_ub, b_ub, base_addr)
            T.copy(c_ub, C)

    return main


def golden_gather(a_cpu, b_int32, elem_size, base_addr=0):
    # gather semantics: dst[i] = src[(base_addr + b[i]) / elem_size]  (flat indexing)
    idx = (b_int32 + base_addr) // elem_size
    return a_cpu[idx.long()]


def test_gather_base_addr():
    """Non-zero src_base_addr (ftcheck example 2).

    ascendc applies base_addr; pto ignores it (documented limitation), so only
    tested on ascendc.
    """
    dtype = "float32"
    elem_size = 4
    n_src = 1024
    n = 512
    base_addr = 64  # 64 bytes = 16 float32 elements

    a = torch.arange(n_src, dtype=torch.float32).npu()
    offsets_i32 = torch.arange(0, elem_size * n, elem_size, dtype=torch.int32)
    perm = torch.randperm(n)
    b = offsets_i32[perm].to(torch.uint32).npu()

    ref = golden_gather(a.cpu(), offsets_i32[perm], elem_size, base_addr).npu()

    kernel = make_gather_baseaddr_kernel(n_src, n, dtype, base_addr)
    func = tilelang.compile(kernel, out_idx=[-1], pass_configs=PASS_CONFIGS, target="ascendc")

    torch.npu.synchronize()
    c = func(a, b)
    torch.npu.synchronize()

    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)


def test_gather_dtype_mismatch_raises():
    """Constraint 1: dst and src must have the same dtype."""

    @T.prim_func
    def main(
        A: T.Tensor((128,), "float32"),  # type: ignore
        B: T.Tensor((128,), "uint32"),  # type: ignore
        C: T.Tensor((128,), "float16"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a_ub = T.alloc_ub((128,), "float32")
            b_ub = T.alloc_ub((128,), "uint32")
            c_ub = T.alloc_ub((128,), "float16")
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            T.tile.gather(c_ub, a_ub, b_ub, 0)
            T.copy(c_ub, C)

    with pytest.raises(RuntimeError, match="Compilation Failed"):  # noqa: B017
        tilelang.compile(main, out_idx=[-1], pass_configs=PASS_CONFIGS, target="ascendc")


def test_gather_offset_dtype_raises():
    """Constraint 2: src_offset must be uint32 (enforced on ascendc)."""

    @T.prim_func
    def main(
        A: T.Tensor((128,), "float32"),  # type: ignore
        B: T.Tensor((128,), "int32"),  # type: ignore
        C: T.Tensor((128,), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a_ub = T.alloc_ub((128,), "float32")
            b_ub = T.alloc_ub((128,), "int32")
            c_ub = T.alloc_ub((128,), "float32")
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            T.tile.gather(c_ub, a_ub, b_ub, 0)
            T.copy(c_ub, C)

    with pytest.raises(RuntimeError, match="Compilation Failed"):  # noqa: B017
        tilelang.compile(main, out_idx=[-1], pass_configs=PASS_CONFIGS, target="ascendc")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-n", "8"])
