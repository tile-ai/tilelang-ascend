# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

# Regression for using an element of a 2D UB tile as a scalar tile operand.

import pytest
import torch

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


def _run_scalar_from_2d_operand(target, M=8, N=8):
    """Accumulate vectors scaled by elements loaded from a 2D UB buffer."""

    @tilelang.jit(out_idx=[2], pass_configs=pass_configs, target=target)
    def kernel():
        @T.prim_func
        def main(n2d: T.Tensor([M, N], "float32"), x: T.Tensor([N], "float32"), out: T.Tensor([N], "float32")):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                n_ub = T.alloc_ub([M, N], "float32")
                x_ub = T.alloc_ub([N], "float32")
                o_ub = T.alloc_ub([N], "float32")
                row_ub = T.alloc_ub([N], "float32")
                with T.Scope("V"):
                    T.copy(n2d, n_ub)
                    T.copy(x, x_ub)
                    T.tile.fill(o_ub, 0.0)
                    for i in range(M):
                        for j in range(i):
                            T.tile.mul(row_ub, x_ub, n_ub[i, j])
                            T.tile.add(o_ub, o_ub, row_ub)
                    T.copy(o_ub, out)

        return main

    torch.manual_seed(0)
    n2d = torch.randn(M, N, dtype=torch.float32).npu()
    x = torch.randn(N, dtype=torch.float32).npu()
    got = kernel()(n2d, x)

    n_ref = n2d.cpu()
    x_ref = x.cpu()
    expect = torch.zeros(N)
    for i in range(M):
        for j in range(i):
            expect += n_ref[i, j] * x_ref
    torch.testing.assert_close(got.cpu(), expect, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_scalar_operand_from_2d_ub(target):
    _run_scalar_from_2d_operand(target)


if __name__ == "__main__":
    test_scalar_operand_from_2d_ub("ascendc")
    print("ascendc PASS")
