# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os
import torch
import tilelang
import tilelang.language as T

M = 1024
N = 1024
K = 512
dtype="float16"
inner_dtype="float32"

@tilelang.jit(target="npuir")
def minicv(M, N, K, block_M, block_N):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def minicv(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), inner_dtype),
            D: T.Tensor((M, N), inner_dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            blockx = cid // n_num
            bx = blockx * block_M
            blocky = cid % n_num
            by = blocky * block_N

            A_BUF = T.alloc_shared((block_M, K), dtype)
            B_BUF = T.alloc_shared((K, block_N), dtype)
            C_BUF = T.alloc_fragment((block_M, block_N), inner_dtype)
            D_BUF = T.alloc_fragment((block_M, block_N), inner_dtype)

            T.copy(A[bx, 0], A_BUF, [block_M, K])
            T.copy(B[0, by], B_BUF, [K, block_N])

            T.gemm(A_BUF, B_BUF, C_BUF, [block_M, K, block_N], initC = True)
            T.vexp(C_BUF, D_BUF)

            T.copy(C_BUF, C[bx, by], [block_M, block_N])
            T.copy(D_BUF, D[bx, by], [block_M, block_N])

    return minicv

def test_minicv():
    # In the futrue, Developer mode and Expert Mode will transition smoothly without
    # requiring explicit declarations.
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'
    # In the futrue, it will be optimized to automatically derive the workspace size.
    os.environ['TILELANG_ASCEND_WORKSPACE_SIZE'] = str(M * N)
    func = minicv(M, N, K, 128, 256)

    v1 = torch.randn(size=[M, K], dtype=eval("torch." + dtype)).npu()
    v2 = torch.randn(size=[K, N], dtype=eval("torch." + dtype)).npu()
    v3 = torch.zeros(size=[M, N], dtype=eval("torch." + inner_dtype)).npu()
    v4 = torch.zeros(size=[M, N], dtype=eval("torch." + inner_dtype)).npu()

    y_ref = torch.exp(v1.to(torch.float32) @ v2.to(torch.float32))
    func(v1, v2, v3, v4)

    print(y_ref)
    print(v4)
    torch.testing.assert_close(v4, y_ref, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")

if __name__ == "__main__":
    test_minicv()