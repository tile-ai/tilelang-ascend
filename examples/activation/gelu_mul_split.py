import tilelang
import tilelang.language as T
import torch
import torch.nn as nn

tilelang.cache.clear_cache()

@tilelang.jit(out_idx=[1])
def gelu_mul(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
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
            T.printf("-----cid:%d-------------vid:%d--------------------------------------\n", cid, vid)
            T.printf("-----m_num:%d-------------n_num:%d--------------------------------------\n", m_num, n_num)
            a1_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            a2_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            temp_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            testa1_ub = T.alloc_ub((block_M // VEC_NUM, 36), dtype)
            testa2_ub = T.alloc_ub((block_M // VEC_NUM, 36), dtype)
            testb_ub = T.alloc_ub((block_M // VEC_NUM, 36), dtype)
            testtemp_ub = T.alloc_ub((block_M // VEC_NUM, 36), dtype)
            with T.Scope("V"):
                if vid == n_num - 1:
                    T.printf("inner-----cid:%d-------------vid:%d--------------------------------------\n", cid, vid)
                    T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], testa1_ub)
                    T.barrier_all()
                    # T.printf("===========a1_ub before copy:\n")
                    # T.dump_tensor(a1_ub, 222, block_M // VEC_NUM * block_N, (block_M // VEC_NUM, block_N))
                    # T.barrier_all()
                    T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N + N // 2], testa2_ub)
                    T.barrier_all()
                    T.tile.mul(testtemp_ub, testa1_ub, testa1_ub)
                    T.barrier_all()
                    T.tile.mul(testtemp_ub, testa1_ub, testtemp_ub)
                    T.barrier_all()
                    T.tile.mul(testtemp_ub, testtemp_ub, 0.044715)
                    T.barrier_all()
                    T.tile.add(testtemp_ub, testa1_ub, testtemp_ub)
                    T.barrier_all()
                    T.tile.mul(testtemp_ub, testtemp_ub, -1.5957691)
                    T.barrier_all()
                    T.tile.exp(testtemp_ub, testtemp_ub)
                    T.barrier_all()
                    T.tile.add(testtemp_ub, testtemp_ub, 1.0)
                    T.barrier_all()
                    T.tile.div(testb_ub, testa1_ub, testtemp_ub)
                    T.barrier_all()
                    # T.printf("===========b_ub after copy:\n")
                    # T.dump_tensor(b_ub, 222, block_M // VEC_NUM * block_N, (block_M // VEC_NUM, block_N))
                    # T.barrier_all()
                    T.tile.mul(testb_ub, testb_ub, testa2_ub)
                    T.barrier_all()
                    T.copy(testb_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])
                else:
                    T.printf("inner-----cid:%d-------------vid:%d--------------------------------------\n", cid, vid)
                    T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a1_ub)
                    T.barrier_all()
                    # T.printf("===========a1_ub before copy:\n")
                    # T.dump_tensor(a1_ub, 222, block_M // VEC_NUM * block_N, (block_M // VEC_NUM, block_N))
                    # T.barrier_all()
                    T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N + N // 2], a2_ub)
                    T.barrier_all()
                    T.tile.mul(temp_ub, a1_ub, a1_ub)
                    T.barrier_all()
                    T.tile.mul(temp_ub, a1_ub, temp_ub)
                    T.barrier_all()
                    T.tile.mul(temp_ub, temp_ub, 0.044715)
                    T.barrier_all()
                    T.tile.add(temp_ub, a1_ub, temp_ub)
                    T.barrier_all()
                    T.tile.mul(temp_ub, temp_ub, -1.5957691)
                    T.barrier_all()
                    T.tile.exp(temp_ub, temp_ub)
                    T.barrier_all()
                    T.tile.add(temp_ub, temp_ub, 1.0)
                    T.barrier_all()
                    T.tile.div(b_ub, a1_ub, temp_ub)
                    T.barrier_all()
                    # T.printf("===========b_ub after copy:\n")
                    # T.dump_tensor(b_ub, 222, block_M // VEC_NUM * block_N, (block_M // VEC_NUM, block_N))
                    # T.barrier_all()
                    T.tile.mul(b_ub, b_ub, a2_ub)
                    T.barrier_all()
                    T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


torch.manual_seed(0)
# Tests
test_configs = [
    # (256, 256, 64, 64),
    (64, 200, 64, 64),
    # (1100, 50000, 128, 128),
]

for M, N, block_M, block_N in test_configs:
    print(f"Testing gelu_mul with M={M}, N={N}, block_M={block_M}, block_N={block_N}")
    func = gelu_mul(M, N, block_M, block_N)
    print("Init successful!")
    a = torch.randn(M, N, dtype=torch.float).npu()
    b = func(a)
    print(func.get_kernel_source())
    gelu = nn.GELU(approximate='tanh')
    a1, a2 = torch.split(a, N // 2, dim=1)
    # ref_b = gelu(a1)
    ref_b = gelu(a1) * a2
    # print("ref_b", ref_b)
    torch.testing.assert_close(b.cpu(), ref_b.cpu(), rtol=1e-2, atol=1e-2)
    print("Test passed!")

print("Kernel Output Match!")