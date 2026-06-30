"""Minimal tail-block row-reduction example for the AscendC backend.

N is not a multiple of block_N, so each row's last column block is a tail. The
reduce must therefore sum only over the valid columns -- which is exactly what
the tail-aware reduce helper does, without any ``pad_value``. M is also not a
multiple of block_M to exercise the row tail.
"""

import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def tail_reduce_sum(M, N, block_M, dtype="float"):
    # One block row per core; the whole row (N columns) is reduced in tiles.
    m_num = T.ceildiv(M, block_M)
    block_N = 128
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        Out: T.Tensor((M, 1), dtype),
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, _):
            bx = cid
            a_ub = T.alloc_ub((block_M, block_N), dtype)
            part = T.alloc_ub((block_M, 1), dtype)
            acc = T.alloc_ub((block_M, 1), dtype)
            T.tile.fill(acc, 0.0)
            for by in T.serial(n_num):
                T.copy(A[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], a_ub)
                T.reduce_sum(a_ub, part, dim=-1)
                T.tile.add(acc, acc, part)
            T.copy(acc, Out[bx * block_M : (bx + 1) * block_M, 0:1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    for M, N, block_M, dtype in [
        (34, 130, 32, "float"),
        (34, 200, 16, "float"),
        (100, 300, 64, "float"),
    ]:
        print(f"tail_reduce_sum M={M} N={N} block_M={block_M} dtype={dtype}")
        func = tail_reduce_sum(M, N, block_M, dtype=dtype)
        a = torch.randn(M, N, dtype=torch.float32).npu()
        out = func(a)
        ref = a.sum(dim=1, keepdim=True)
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
        print("  pass")
    print("Kernel Output Match!")
