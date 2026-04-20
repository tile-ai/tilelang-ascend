# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""
Test case for NPU memory allocation ops in Developer mode:
  - T.alloc_var   (local.var scope)
  - T.alloc_L0A   (wmma.matrix_a scope)
  - T.alloc_L0B   (wmma.matrix_b scope)
  - T.alloc_L0C   (wmma.accumulator scope)
  - T.alloc_L1    (shared.dyn scope)
  - T.alloc_ub    (shared / UB scope)

The kernel allocates buffers in every memory scope, performs a simple
computation that touches each buffer, and writes the result to global memory
so we can verify correctness.
"""

import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("alloc"),
    pytest.mark.mode("Developer"),
]


def alloc_all_scopes_kernel(M, N, dtype):
    """Kernel that exercises all six allocation helpers.

    Flow:
      1. Copy input from GM → UB  (alloc_ub)
      2. Copy UB → L1             (alloc_L1)
      3. Copy L1 → UB back
      4. Use alloc_var to hold a scalar iteration variable
      5. Write result from UB → GM

    L0A / L0B / L0C are allocated to verify they compile, but are not
    used in the data path because a full matmul is out of scope for this
    test.
    """
    BLOCK_SIZE = 1

    @T.prim_func
    def allocKernel(
        src: T.Tensor((M, N), dtype),
        dst: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            # UB (Unified Buffer) allocation
            src_ub = T.alloc_ub((M, N), dtype)
            dst_ub = T.alloc_ub((M, N), dtype)

            # L1 allocation
            l1_buf = T.alloc_L1((M, N), dtype)

            # L0A / L0B / L0C allocations (compile-check)
            l0a_buf = T.alloc_L0A((M, N), dtype)
            l0b_buf = T.alloc_L0B((M, N), dtype)
            l0c_buf = T.alloc_L0C((M, N), "float32")

            # Scalar variable allocation
            loop_var = T.alloc_var("int32")

            # GM -> UB
            T.copy(src, src_ub)

            # UB -> L1
            T.copy(src_ub, l1_buf)

            # L1 -> UB
            T.copy(l1_buf, dst_ub)

            # UB -> GM
            T.copy(dst_ub, dst)

    return allocKernel


@pytest.mark.parametrize("dtype", ["float16"])
def test_alloc_all_scopes(dtype):
    M, N = 32, 32
    src = gen_tensor((M, N), dtype, kind="randn")
    dst = gen_tensor((M, N), dtype, kind="zeros")

    program = alloc_all_scopes_kernel(M, N, dtype)
    mod = tilelang.compile(program, target="npuir", execution_backend="npu")
    mod(src, dst)

    assert_close(dst, src, rtol=1e-3, atol=1e-3)
    print("PASSED: test_alloc_all_scopes")


if __name__ == "__main__":
    test_alloc_all_scopes("float16")
