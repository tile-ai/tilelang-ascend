import tilelang
import tilelang.language as T
import torch
import torch.nn as nn

tilelang.cache.clear_cache()

@tilelang.jit(out_idx=[1])
def gelu_mul(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    
    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype)
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            
            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            temp_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                T.barrier_all()
                T.tile.mul(temp_ub, a_ub, a_ub)
                T.barrier_all()
                T.tile.mul(temp_ub, a_ub, temp_ub)
                T.barrier_all()
                T.tile.mul(temp_ub, temp_ub, 0.044715)
                T.barrier_all()
                T.tile.add(temp_ub, a_ub, temp_ub)
                T.barrier_all()
                T.tile.mul(temp_ub, temp_ub, -1.5957691)
                T.barrier_all()
                T.tile.exp(temp_ub, temp_ub)
                T.barrier_all()
                T.tile.add(temp_ub, temp_ub, 1.0)
                T.barrier_all()
                T.tile.div(b_ub, a_ub, temp_ub)
                T.barrier_all()
                T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


torch.manual_seed(0)
# Tests
test_configs = [
    # (16, 16, 4, 4),
    (256, 128, 64, 64),
    (300, 300, 64, 64),
    (1100, 50000, 128, 128),
]

for M, N, block_M, block_N in test_configs:
    print(f"Testing gelu_mul with M={M}, N={N}, block_M={block_M}, block_N={block_N}")
    func = gelu_mul(M, N, block_M, block_N)
    print("Init successful!")
    a = torch.randn(M, N, dtype=torch.float).npu()
    b = func(a)
    gelu = nn.GELU(approximate='tanh')
    ref_b = gelu(a)
    torch.testing.assert_close(b.cpu(), ref_b.cpu(), rtol=1e-2, atol=1e-2)
    print("Test passed!")

print("Kernel Output Match!")