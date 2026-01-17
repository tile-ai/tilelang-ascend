# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import torch
import tilelang
import tilelang.language as T

torch.npu.set_device(0)
tilelang.cache.clear_cache()

@tilelang.jit(target="npuir")
def vec_add(block_M, block_N, dtype = "float16"):
    M = T.symbolic("M")
    N = T.symbolic("N")
    BLOCK_SIZE = 20

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype)
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            B_VEC = T.alloc_ub((block_M, block_N), dtype)
            C_VEC = T.alloc_ub((block_M, block_N), dtype)

            m_num = T.ceildiv(M, block_M)
            n_num = T.ceildiv(N, block_N)

            start_block_id = cid * T.ceildiv(m_num * n_num, BLOCK_SIZE)
            for i in T.serial(T.ceildiv(m_num * n_num, BLOCK_SIZE)):
                block_id = start_block_id + i
                if block_id < m_num * n_num:
                    block_id_m = block_id // n_num
                    block_id_n = block_id % n_num
                    bx = block_id_m * block_M
                    by = block_id_n * block_N
                    remain_block_M = T.min(M - bx, block_M)
                    remain_block_N = T.min(N - by, block_N)
                    T.copy(A[bx, by], A_VEC, size=[remain_block_M, remain_block_N])
                    T.copy(B[bx, by], B_VEC, size=[remain_block_M, remain_block_N])
                    T.npuir_add(A_VEC, B_VEC, C_VEC)
                    T.copy(C_VEC, C[bx, by], size=[remain_block_M, remain_block_N])

    return main


def run_test():
    func = vec_add(32, 32)

    # shape 1
    M, N = 128, 256
    a = torch.randn([M, N], dtype=torch.float16).npu()
    b = torch.randn([M, N], dtype=torch.float16).npu()
    c = torch.randn([M, N], dtype=torch.float16).npu()
    ref_c = a + b
    func(a, b, c)
    print("Kernel Output:")
    print(c)
    print("Ref Output:")
    print(ref_c)
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    # shape 2
    M, N = 256, 128
    a = torch.randn([M, N], dtype=torch.float16).npu()
    b = torch.randn([M, N], dtype=torch.float16).npu()
    c = torch.randn([M, N], dtype=torch.float16).npu()
    ref_c = a + b
    func(a, b, c)
    print("Kernel Output:")
    print(c)
    print("Ref Output:")
    print(ref_c)
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    # shape 3
    M, N = 77, 88
    a = torch.randn([M, N], dtype=torch.float16).npu()
    b = torch.randn([M, N], dtype=torch.float16).npu()
    c = torch.randn([M, N], dtype=torch.float16).npu()
    ref_c = a + b
    func(a, b, c)
    print("Kernel Output:")
    print(c)
    print("Ref Output:")
    print(ref_c)
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    print("\033[92mAll check passed!\033[0m")

if __name__ == "__main__":
    run_test()