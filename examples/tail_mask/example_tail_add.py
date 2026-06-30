"""Minimal tail-block elementwise-add example for the AscendC backend.

M and N are deliberately *not* multiples of the block, so the last row/column
blocks are tails. With the tail-mask scheme there is no front-end ``pad_value``:
``T.copy`` loads only the valid region and ``T.tile.add`` is rewritten to a
tail-aware helper that computes only over the valid rectangle.
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


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def tail_add(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((block_M, block_N), dtype)
            b_ub = T.alloc_ub((block_M, block_N), dtype)
            c_ub = T.alloc_ub((block_M, block_N), dtype)
            T.copy(A[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], a_ub)
            T.copy(B[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], b_ub)
            T.tile.add(c_ub, a_ub, b_ub)
            T.copy(c_ub, C[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    for M, N, block_M, block_N, dtype in [
        (34, 130, 32, 32, "float"),
        (34, 130, 32, 32, "float16"),
        (100, 200, 64, 128, "float"),
    ]:
        print(f"tail_add M={M} N={N} block=({block_M},{block_N}) dtype={dtype}")
        func = tail_add(M, N, block_M, block_N, dtype=dtype)
        torch_dtype = torch.float32 if dtype == "float" else torch.float16
        a = torch.randn(M, N, dtype=torch_dtype).npu()
        b = torch.randn(M, N, dtype=torch_dtype).npu()
        c = func(a, b)
        torch.testing.assert_close(c, a + b, rtol=1e-2, atol=1e-2)
        print("  pass")
    print("Kernel Output Match!")
