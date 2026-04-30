#!/usr/bin/env python3
"""
TileLang-Ascend simple test script
Used to verify if the environment is correctly configured
"""

import torch
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def simple_add(M=16, N=16, dtype="float"):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a_ub = T.alloc_ub((M, N), dtype)
            b_ub = T.alloc_ub((M, N), dtype)
            c_ub = T.alloc_ub((M, N), dtype)

            with T.Scope("V"):
                T.copy(A[0, 0], a_ub)
                T.copy(B[0, 0], b_ub)
                T.barrier_all()
                T.tile.add(c_ub, a_ub, b_ub)
                T.barrier_all()
                T.copy(c_ub, C[0, 0])

    return main


def main():
    tilelang.cache.clear_cache()

    M, N = 16, 16
    func = simple_add(M, N)

    a = torch.randn(M, N).npu()
    b = torch.randn(M, N).npu()
    c = func(a, b)

    torch.testing.assert_close(c, a + b, rtol=1e-2, atol=1e-2)
    print("✓ TileLang 环境验证通过!")


if __name__ == "__main__":
    main()