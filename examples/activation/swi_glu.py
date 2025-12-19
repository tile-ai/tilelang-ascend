import tilelang
import tilelang.language as T
import torch
import torch.nn as nn

tilelang.cache.clear_cache()

# pass_configs = {
#     tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
#     tilelang.PassConfigKey.TIR_MERGE_STATIC_SMEM: True,
#     tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
# }

@tilelang.jit(out_idx=[1])
def swi_glu(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    # The `swi_glu` operator splits the input tensor into two tensors, x1 and x2, based on the last dimension. 
    # It performs a GELU operation on x1 and multiplies the result by x2. Therefore, the kernel splitting is only relative to the dimension of x1.
    n_num = T.ceildiv(N // 2, block_N)
    
    VEC_NUM = 2


    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N // 2), dtype)
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            a0_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            a1_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            zero_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            temp_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a0_ub)
                T.barrier_all()
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N + N // 2], a1_ub)
                T.barrier_all()
                T.tile.fill(zero_ub, 0.0)
                T.barrier_all()
                T.tile.sub(temp_ub, zero_ub, a0_ub)
                T.barrier_all()
                T.tile.exp(temp_ub, temp_ub)
                T.barrier_all()
                T.tile.add(temp_ub, temp_ub, 1.0)
                T.barrier_all()
                T.tile.div(temp_ub, a0_ub, temp_ub)
                T.barrier_all()
                T.tile.mul(b_ub, temp_ub, a1_ub)
                T.barrier_all()
                T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


torch.manual_seed(0)
# Tests
test_configs = [
    (256, 256, 64, 64),
    # (1024, 1024, 128, 128),
]

for M, N, block_M, block_N in test_configs:
    print(f"Testing swi_gul with M={M}, N={N}, block_M={block_M}, block_N={block_N}")
    func = swi_glu(M, N, block_M, block_N)
    print("Init successful!")
    a = torch.randn(M, N, dtype=torch.float).npu()
    b = func(a)
    print(func.get_kernel_source())
    a1, a2 = torch.split(a, N // 2, dim=1)
    silu = nn.SiLU()
    ref_b = silu(a1) * a2
    torch.testing.assert_close(b.cpu(), ref_b.cpu(), rtol=1e-2, atol=1e-2)
    print("Test passed!")

print("Kernel Output Match!")