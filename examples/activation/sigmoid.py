import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def sigmoid(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    VEC_NUM = 2

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            zero_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.tile.fill(zero_ub, 0.0)
            T.tile.sub(a_ub, zero_ub, a_ub)
            T.tile.exp(a_ub, a_ub)
            T.tile.add(a_ub, a_ub, 1.0)
            T.tile.reciprocal(b_ub, a_ub)
            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


torch.manual_seed(0)
# Tests
test_configs = [
    (256, 256, 64, 64),
    (300, 300, 64, 64),
    (1100, 50000, 128, 128),
]

for M, N, block_M, block_N in test_configs:
    print(f"Testing sigmoid with M={M}, N={N}, block_M={block_M}, block_N={block_N}")
    func = sigmoid(M, N, block_M, block_N)
    print("Init successful!")
    a = torch.randn(M, N).npu()
    b = func(a)
    ref_b = torch.sigmoid(a)
    torch.testing.assert_close(b.cpu(), ref_b.cpu(), rtol=1e-2, atol=1e-2)
    print("Test passed!")

print("Kernel Output Match!")
