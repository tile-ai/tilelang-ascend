import torch
import tilelang
import tilelang.language as T

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}

def _compile(program, target="pto", expert=False):
    """Compile program with target and developing mode."""
    pass_config = None if expert else PASS_CONFIGS
    return tilelang.compile(program, pass_configs=pass_config, target=target)


def _torch_dtype(dtype):
    torch_dtype = getattr(torch, dtype)
    assert isinstance(torch_dtype, torch.dtype), f"Unsupported dtype: {dtype!r}"
    return torch_dtype


def _torch_rand_tensor(shape, dtype: torch.dtype, device="npu"):
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device=device)
    return torch.randint(0, 127, shape, dtype=dtype, device=device)


def _multi_task_kernel(M=256, N=256, K=32, block_M=32, block_N=32, block_K=32, dtype="float16", accum_dtype="float"):
    """Test kernel: Combined UB↔L1 and L0C↔UB paths in single kernel"""
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    k_num = T.ceildiv(K, block_K)

    VEC_NUM = 2
    block_M_half = T.ceildiv(block_M, VEC_NUM)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype), # type: ignore
        B: T.Tensor((K, N), dtype), # type: ignore
        C: T.Tensor((M, N), accum_dtype), # type: ignore
        workspace_5: T.Tensor((M, K), dtype), # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            A_ub = T.alloc_ub((block_M_half, block_K), dtype)

            A_L1 = T.alloc_L1((block_M, block_K), dtype)
            B_L1 = T.alloc_L1((block_K, block_N), dtype)

            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            C_ub = T.alloc_ub((block_M_half, block_N), accum_dtype)

            for bk in T.serial(k_num):
                T.copy(A[bx * block_M + vid * block_M_half, bk * block_K], A_ub)
                T.tile.mul(A_ub, A_ub, -1.0)

                # 重点在这里，当前 workspace 消除逻辑未能把 A_ub => A_L1 还原成与注释掉语句相近的形式
                # T.copy(A_ub, workspace_5[bx * block_M + vid * block_M_half, bk * block_K])
                # T.copy(workspace_5[bx * block_M, bk * block_K], A_L1)
                T.copy(A_ub, A_L1)

                T.copy(B[bk * block_K, by * block_N], B_L1)

                T.gemm_v0(A_L1, B_L1, C_L0, init=(bk == 0))
            T.copy(C_L0, C_ub)

            T.tile.mul(C_ub, C_ub, -1.0)

            T.copy(C_ub, C[bx * block_M + vid * block_M_half, by * block_N])

    return main


def _multi_task_case(kernel_func, M=256, N=256, K=32, target="pto"):
    dtype = "float16"
    accum_dtype = "float"
    program = kernel_func(M=M, N=N, K=K, dtype=dtype, accum_dtype=accum_dtype)
    kernel = _compile(program, target=target)
    print(kernel.get_kernel_source())

    torch_dtype = _torch_dtype(dtype)
    torch_accum_dtype = _torch_dtype(accum_dtype)

    a = _torch_rand_tensor((M, K), dtype=torch_dtype, device="npu")
    b = _torch_rand_tensor((K, N), dtype=torch_dtype, device="npu")
    c = torch.empty((M, N), dtype=torch_accum_dtype, device="npu")
    ws = torch.empty((M, K), dtype=torch_dtype, device="npu")
    torch.npu.synchronize()

    kernel(a, b, c, ws)
    torch.npu.synchronize()

    # Verify result
    ref_c = -((-a) @ b)
    ref_c = ref_c.to(torch_accum_dtype)

    print("c = ")
    print(c)
    print("ref_c = ")
    print(ref_c)
    torch.testing.assert_close(c, ref_c, rtol=1e-3, atol=1e-3)


# M, N, K = 256, 256, 32
M, N, K = 64, 64, 64
target = "ascendc"
tilelang.disable_cache()

_multi_task_case(_multi_task_kernel, M=M, N=N, K=K, target=target)

print("Kernel Output Match!")