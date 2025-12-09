import tilelang
from tilelang import language as T
import torch

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def sin(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(
        A: T.Tensor([M, N], dtype),
        B: T.Tensor([M, N], dtype),
    ):
        T.func_attr({"enable_auto_sync": True})
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            a = T.alloc_ub([sub_block_M, block_N], dtype)
            b = T.alloc_ub([sub_block_M, block_N], dtype)
            tmp = T.alloc_ub([2 * sub_block_M * block_N], "uint8")
            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * sub_block_M: bx * block_M + (vid + 1) * sub_block_M,
                         by * block_N: (by + 1) * block_N], a) # Load input
                T.tile.sin(b, a, tmp)  # Compute sin
                T.copy(b, B[bx * block_M + vid * sub_block_M: bx * block_M + (vid + 1) * sub_block_M,
                            by * block_N: (by + 1) * block_N]) # Store output
    return main

torch.manual_seed(0)
test_configs = [
    (1024, 1024, 128, 128),
]

for M, N, block_M, block_N in test_configs:
    func = sin(M, N, block_M, block_N, dtype="float")
    print("Init successful!")
    a = torch.randn(M, N).npu()
    b = func(a)
    ref_b = torch.sin(a)
    torch.testing.assert_close(b, ref_b, rtol=1e-4, atol=1e-4)
    print("Test passed!")
print("Kernel Output Match!")