import tilelang
from tilelang import DataType, language as T
import torch

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def online_softmax(M, N, block_M, block_N, dtype="float"):
    """
    Safe softmax with online normalizer.

    Algorithm:
        m_0 = -inf
        s_0 = 0
        for j in [1, N]:
            m_j = max(m_{j-1}, x_j)
            s_j = s_{j-1} * exp(m_{j-1} - m_j) + exp(x_j - m_j)
        for j in [1, N]:
            y_j = exp(x_j - m_N) / s_N
    """
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
        # One core process one block row
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            bx = cid
            a = T.alloc_ub([sub_block_M, block_N], dtype)
            tile_max = T.alloc_ub([sub_block_M], dtype)
            prev_max = T.alloc_ub([sub_block_M], dtype)
            tile_sum = T.alloc_ub([sub_block_M], dtype)
            prev_sum = T.alloc_ub([sub_block_M], dtype)
            tmp_exp = T.alloc_ub([sub_block_M], dtype)
            tmp = T.alloc_ub([2 * sub_block_M * block_N], "uint8")
            T.tile.fill(prev_max, -1e38)
            T.tile.fill(prev_sum, 0.0)
            with T.Scope("V"):
                # First pass: compute max and sum
                for by in T.serial(n_num):
                    T.copy(A[bx * block_M + vid * sub_block_M: bx * block_M + (vid + 1) * sub_block_M,
                             by * block_N: (by + 1) * block_N], a) # Load input
                    T.tile.reduce_max(tile_max, a, tmp, dim=-1)  # Compute tile max
                    T.tile.max(tile_max, prev_max, tile_max) # m_j = max(m_{j-1}, x_j)
                    T.tile.sub(tmp_exp, prev_max, tile_max) # m_{j-1} - m_j
                    T.tile.exp(tmp_exp, tmp_exp) # exp(m_{j-1} - m_j)
                    T.tile.mul(tmp_exp, prev_sum, tmp_exp) # s_{j-1} * exp(m_{j-1} - m_j)
                    for i in range(sub_block_M):
                        T.tile.sub(a[i, :], a[i, :], tile_max[i]) # x_j - m_j
                    T.tile.exp(a, a) # exp(x_j - m_j)
                    T.tile.reduce_sum(tile_sum, a, tmp, dim=-1) # sum_j exp(x_j - m_j)
                    T.tile.add(prev_sum, tile_sum, tmp_exp) # s_j = s_{j-1} * exp(m_{j-1} - m_j) + exp(x_j - m_j)
                    T.copy(tile_max, prev_max)
                
                # Second pass: compute final output
                # After first pass, prev_max holds m_N, prev_sum holds s_N
                for by in T.serial(n_num):
                    T.copy(A[bx * block_M + vid * sub_block_M: bx * block_M + (vid + 1) * sub_block_M,
                             by * block_N: (by + 1) * block_N], a) # Load input
                    for i in range(sub_block_M):
                        T.tile.sub(a[i, :], a[i, :], prev_max[i]) # x_j - m_N
                    T.tile.exp(a, a) # exp(x_j - m_N)
                    for i in range(sub_block_M):
                        T.tile.div(a[i, :], a[i, :], prev_sum[i]) # y_j = exp(x_j - m_N) / s_N
                    T.copy(a, B[bx * block_M + vid * sub_block_M: bx * block_M + (vid + 1) * sub_block_M,
                                 by * block_N: (by + 1) * block_N]) # Store output
    
    return main

torch.manual_seed(0)
test_configs = [
    (1024, 51200, 128, 128),
]

for M, N, block_M, block_N in test_configs:
    func = online_softmax(M, N, block_M, block_N, dtype="float")
    print("Init successful!")
    a = torch.randn(M, N).npu()
    b = func(a)
    ref_b = torch.nn.functional.softmax(a, dim=1)
    torch.testing.assert_close(b, ref_b, rtol=1e-4, atol=1e-4)
    print("Test passed!")
print("Kernel Output Match!")