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
def gelu_mul(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    # The `gelu_mul` operator splits the input tensor into two tensors, x1 and x2, based on the last dimension. 
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
            a1_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            a2_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            T.printf("-----outer cid:%d-------------vid:%d--------------------------------------\n", cid, vid)
            temp_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            ## [In vector]
            # The left half is cached using a1_ub
            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a1_ub)
            # The right half is cached using a2_ub
            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N + N // 2], a2_ub)
            # Calculation formula:x^2
            T.tile.mul(temp_ub, a1_ub, a1_ub)
            # Calculation formula:x^3
            T.tile.mul(temp_ub, a1_ub, temp_ub)
            # Calculation formula:0.044715 * x^3
            T.tile.mul(temp_ub, temp_ub, 0.044715)
            # Calculation formula:x + 0.044715 * x^3
            T.tile.add(temp_ub, a1_ub, temp_ub)
            # Calculation formula:-sqrt(8/pi)(x + 0.044715 * x^3)
            T.tile.mul(temp_ub, temp_ub, -1.5957691)
            # Calculation formula:exp(-sqrt(8/pi)(x + 0.044715 * x^3))
            T.tile.exp(temp_ub, temp_ub)
            # Calculation formula:1 + exp(-sqrt(8/pi)(x + 0.044715 * x^3))
            T.tile.add(temp_ub, temp_ub, 1.0)
            # Calculation formula:x / (1 + exp(-sqrt(8/pi)(x + 0.044715 * x^3)))
            T.tile.div(temp_ub, a1_ub, temp_ub)
            # Multiply the result of the left half by the right half
            T.tile.mul(b_ub, temp_ub, a2_ub)
            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


torch.manual_seed(0)
# Tests
test_configs = [
    (256, 256, 64, 64),
    (1024, 1024, 128, 128),
]

for M, N, block_M, block_N in test_configs:
    print(f"Testing gelu_mul with M={M}, N={N}, block_M={block_M}, block_N={block_N}")
    func = gelu_mul(M, N, block_M, block_N)
    print("Init successful!")
    a = torch.randn(M, N, dtype=torch.float).npu()
    b = func(a)
    gelu = nn.GELU(approximate='tanh')
    a1, a2 = torch.split(a, N // 2, dim=1)
    ref_b = gelu(a1) * a2
    torch.testing.assert_close(b.cpu(), ref_b.cpu(), rtol=1e-2, atol=1e-2)
    print("Test passed!")

print("Kernel Output Match!")