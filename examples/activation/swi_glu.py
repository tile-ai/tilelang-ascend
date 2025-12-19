import tilelang
import tilelang.language as T
import torch
import torch.nn as nn

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}

@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def swi_glu(M, N, block_M, block_N, split_dim, dtype="float"):
    # The `swi_glu` operator splits the input tensor into two tensors, x1 and x2, based on the split dimension. 
    # It performs a Swish operation on x1 and multiplies the result by x2. 
    m_div = 1
    n_div = 2
    m_offset = 0
    n_offset = N // 2
    if split_dim == split_dim == 0 or split_dim == -2:
        m_div = 2
        n_div = 1
        m_offset = M // 2
        n_offset = 0
    m_num = T.ceildiv(M // m_div, block_M)
    n_num = T.ceildiv(N // n_div, block_N)
    
    VEC_NUM = 2


    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M // m_div, N // n_div), dtype)
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            a0_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)    
            a1_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            zero_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            temp_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            ## [In vector]
            # The first half is cached using a0_ub
            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a0_ub)
            # The second half is cached using a1_ub
            T.copy(A[bx * block_M + vid * block_M // VEC_NUM + m_offset, by * block_N + n_offset], a1_ub)
            T.tile.fill(zero_ub, 0.0)
            # Calculation formula:-x
            T.tile.sub(temp_ub, zero_ub, a0_ub)
            # Calculation formula:exp(-x)
            T.tile.exp(temp_ub, temp_ub)
            # Calculation formula:1 + exp(-x)
            T.tile.add(temp_ub, temp_ub, 1.0)
            # Calculation formula:x / (1 + exp(-x))
            T.tile.div(temp_ub, a0_ub, temp_ub)
            # Multiply the result of the first half by the second half
            T.tile.mul(b_ub, temp_ub, a1_ub)
            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


torch.manual_seed(0)
# Tests
test_configs = [
    (256, 256, 64, 64, 0),
    (256, 256, 64, 64, 1),
    (1024, 1024, 128, 128, -2),
    (1024, 1024, 128, 128, -1),    
]

for M, N, block_M, block_N, split_dim in test_configs:
    print(f"Testing swi_gul with M={M}, N={N}, block_M={block_M}, block_N={block_N}, split_dim={split_dim}")
    func = swi_glu(M, N, block_M, block_N, split_dim)
    print("Init successful!")
    a = torch.randn(M, N, dtype=torch.float).npu()
    b = func(a)
    print(func.get_kernel_source())
    split_size = N // 2
    if split_dim == 0 or split_dim == -2:
        split_size = M // 2
    a1, a2 = torch.split(a, split_size, dim=split_dim)
    silu = nn.SiLU()
    ref_b = silu(a1) * a2
    torch.testing.assert_close(b.cpu(), ref_b.cpu(), rtol=1e-2, atol=1e-2)
    print("Test passed!")

print("Kernel Output Match!")