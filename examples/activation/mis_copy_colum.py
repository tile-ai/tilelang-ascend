import tilelang
import tilelang.language as T
import torch
import torch.nn as nn

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

@tilelang.jit(out_idx=[-1], target="pto", pass_configs=pass_configs)
def column_parallel_buffer_scalar_mul_kernel(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype),
            B: T.Tensor((M,), dtype),
            C: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM,), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                T.copy(B[bx * block_M + vid * block_M // VEC_NUM], b_ub)
                    
                for (i, j) in T.Parallel(block_M // VEC_NUM, block_N):
                    # c_ub[i, j] = a_ub[i, j] + 5
                    c_ub[i, j] = b_ub[i] + 5
                    # c_ub[i, j] = a_ub[i, j] + b_ub[i]
                    # c_ub[i, j] = b_ub[i]
                    # c_ub[i, j] = a_ub[i, j] + 5

                T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


torch.manual_seed(0)
# Tests
test_configs = [
    (256, 256, 64, 64),
    # (1024, 1024, 128, 128),
]

for M, N, block_M, block_N in test_configs:
    print(f"Testing gelu_mul with M={M}, N={N}, block_M={block_M}, block_N={block_N}")
    func = column_parallel_buffer_scalar_mul_kernel(M, N, block_M, block_N)
    print("Init successful!")
    a = torch.randn(M, N).npu()
    b = torch.randn(M).npu()
    # print("b", b)
    # print("b.unsqueeze", b.unsqueeze(1))
    torch.npu.synchronize()
    c = func(a, b)
    # print("c", c)
    # print("c.shape", c.shape)
    ref_c = torch.broadcast_to(b.unsqueeze(1) + 5, a.shape)
    # ref_c = torch.broadcast_to(b.unsqueeze(1), a.shape)
    # ref_c = a + b.unsqueeze(1)
    # ref_c = a + 5
    # print("ref_c", ref_c)
    # print("ref_c.shape", ref_c.shape)
    torch.testing.assert_close(c.cpu(), ref_c.cpu(), rtol=1e-2, atol=1e-2)
    print("Test passed!")

print("Kernel Output Match!")