"""
T.gemm_v0 补充测试：全量 dtype 精度 + 异常边界

补充现有测试缺失的组合：
1. bfloat16 精度测试（现有只测 float16）
2. float32 精度测试（小 shape，避免 L0A 溢出）
3. int8 transpose 全覆盖（现有只测 2 种组合）
4. 异常边界：A/B dtype 不匹配
5. 异常边界：K 维度不一致
6. 异常边界：init=True vs init=False 累加验证
"""

import pytest
import tilelang
import tilelang.language as T
import torch

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

SHAPE_CONFIGS = {
    "A": (1024, 1024, 1024, 128, 256, 128),
    "B": (1024, 1024, 1024, 128, 256, 64),
    "C": (256, 256, 192, 128, 128, 192),
    "D": (1024, 1024, 1024, 128, 128, 256),
}


def _compile(program, target):
    return tilelang.compile(program, pass_configs=PASS_CONFIGS, target=target)


def _gemm_v0_kernel(
    M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float", kL0Size=128, transpose_A=False, transpose_B=False
):
    m_num = M // block_M
    n_num = N // block_N
    a_gm_shape = (K, M) if transpose_A else (M, K)
    b_gm_shape = (N, K) if transpose_B else (K, N)
    a_l1_shape = (K_L1, block_M) if transpose_A else (block_M, K_L1)
    b_l1_shape = (block_N, K_L1) if transpose_B else (K_L1, block_N)

    @T.prim_func
    def main(
        A: T.Tensor(a_gm_shape, dtype),
        B: T.Tensor(b_gm_shape, dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num
            A_L1 = T.alloc_L1(a_l1_shape, dtype)
            B_L1 = T.alloc_L1(b_l1_shape, dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)
            with T.Scope("C"):
                loop_k = T.ceildiv(K, K_L1)
                for k in T.serial(loop_k):
                    if transpose_A:
                        T.copy(A[k * K_L1, bx * block_M], A_L1)
                    else:
                        T.copy(A[bx * block_M, k * K_L1], A_L1)
                    if transpose_B:
                        T.copy(B[by * block_N, k * K_L1], B_L1)
                    else:
                        T.copy(B[k * K_L1, by * block_N], B_L1)
                    T.barrier_all()
                    T.gemm_v0(A_L1, B_L1, C_L0, transpose_A=transpose_A, transpose_B=transpose_B, init=(k == 0), kL0Size=kL0Size)
                    T.barrier_all()
                for i, j in T.Parallel(block_M, block_N):
                    C[bx * block_M + i, by * block_N + j] = C_L0[i, j]

    return main


def _run_dtype_precision(dtype, accum_dtype, target, shape_group="A"):
    full_M, full_N, full_K, block_M, block_N, K_L1 = SHAPE_CONFIGS[shape_group]
    kL0Size = 128 if dtype != "float32" else 64
    K_L1_adj = K_L1 if dtype != "float32" else 64

    program = _gemm_v0_kernel(full_M, full_N, full_K, block_M, block_N, K_L1_adj, dtype, accum_dtype, kL0Size=kL0Size)
    kernel = _compile(program, target)

    torch_dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
    }[dtype]

    a = torch.randn(full_M, full_K, dtype=torch_dtype, device="npu")
    b = torch.randn(full_K, full_N, dtype=torch_dtype, device="npu")
    c = torch.zeros(full_M, full_N, dtype=torch_dtype, device="npu")
    torch.npu.synchronize()
    kernel(a, b, c)
    torch.npu.synchronize()
    ref_c = a @ b
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.low_priority
@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="gemm_v0 correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_gemm_v0_bfloat16_precision(target):
    _run_dtype_precision("bfloat16", "float", target)


@pytest.mark.low_priority
@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="gemm_v0 correctness requires an Ascend NPU runtime",
)
@pytest.mark.xfail(reason="float32 gemm_v0 codegen bug: Find undefined Variable _")
@pytest.mark.parametrize("target", ["ascendc"])
def test_gemm_v0_float32_precision(target):
    M, N, K = 128, 128, 64
    block_M, block_N, K_L1 = 128, 128, 64
    program = _gemm_v0_kernel(M, N, K, block_M, block_N, K_L1, "float32", "float", kL0Size=64)
    kernel = _compile(program, target)
    a = torch.randn(M, K, dtype=torch.float32, device="npu")
    b = torch.randn(K, N, dtype=torch.float32, device="npu")
    c = torch.zeros(M, N, dtype=torch.float32, device="npu")
    torch.npu.synchronize()
    kernel(a, b, c)
    torch.npu.synchronize()
    ref_c = a @ b
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.low_priority
@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="gemm_v0 correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("transpose_A,transpose_B", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("target", ["ascendc"])
def test_gemm_v0_int8_transpose_full(target, transpose_A, transpose_B):
    M, N, K = 128, 256, 128
    block_M, block_N, K_L1 = 128, 256, 128
    kL0Size = 128
    program = _gemm_v0_int8_kernel(M, N, K, block_M, block_N, K_L1, transpose_A, transpose_B, kL0Size)
    kernel = _compile(program, target)

    a_gm_shape = (K, M) if transpose_A else (M, K)
    b_gm_shape = (N, K) if transpose_B else (K, N)
    torch.manual_seed(0)
    a = torch.randint(-5, 6, a_gm_shape, dtype=torch.int8, device="npu")
    b = torch.randint(-5, 6, b_gm_shape, dtype=torch.int8, device="npu")
    c = torch.zeros(M, N, dtype=torch.int32, device="npu")
    torch.npu.synchronize()
    kernel(a, b, c)
    torch.npu.synchronize()

    a_ref = a.cpu().T if transpose_A else a.cpu()
    b_ref = b.cpu().T if transpose_B else b.cpu()
    ref_c = a_ref.to(torch.int32) @ b_ref.to(torch.int32)
    torch.testing.assert_close(c.cpu(), ref_c, rtol=0, atol=0)


def _gemm_v0_int8_kernel(M, N, K, block_M, block_N, K_L1, transpose_A, transpose_B, kL0Size):
    m_num = M // block_M
    n_num = N // block_N
    a_gm_shape = (K, M) if transpose_A else (M, K)
    b_gm_shape = (N, K) if transpose_B else (K, N)
    a_l1_shape = (K_L1, block_M) if transpose_A else (block_M, K_L1)
    b_l1_shape = (block_N, K_L1) if transpose_B else (K_L1, block_N)

    @T.prim_func
    def main(
        A: T.Tensor(a_gm_shape, "int8"),
        B: T.Tensor(b_gm_shape, "int8"),
        C: T.Tensor((M, N), "int32"),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num
            A_L1 = T.alloc_L1(a_l1_shape, "int8")
            B_L1 = T.alloc_L1(b_l1_shape, "int8")
            C_L0 = T.alloc_L0C((block_M, block_N), "int32")
            with T.Scope("C"):
                loop_k = T.ceildiv(K, K_L1)
                for k in T.serial(loop_k):
                    if transpose_A:
                        T.copy(A[k * K_L1, bx * block_M], A_L1)
                    else:
                        T.copy(A[bx * block_M, k * K_L1], A_L1)
                    if transpose_B:
                        T.copy(B[by * block_N, k * K_L1], B_L1)
                    else:
                        T.copy(B[k * K_L1, by * block_N], B_L1)
                    T.barrier_all()
                    T.gemm_v0(A_L1, B_L1, C_L0, transpose_A=transpose_A, transpose_B=transpose_B, init=(k == 0), kL0Size=kL0Size)
                    T.barrier_all()
                T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


# ============================================================
# 异常边界测试
# ============================================================


@pytest.mark.low_priority
def test_gemm_v0_dtype_mismatch():
    """A/B dtype 不一致应编译失败"""
    M, N, K = 128, 128, 128
    block_M, block_N, K_L1 = 128, 128, 128

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "int8"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            A_L1 = T.alloc_L1((block_M, K_L1), "float16")
            B_L1 = T.alloc_L1((K_L1, block_N), "int8")
            C_L0 = T.alloc_L0C((block_M, block_N), "float")
            with T.Scope("C"):
                T.copy(A[0, 0], A_L1)
                T.copy(B[0, 0], B_L1)
                T.barrier_all()
                T.gemm_v0(A_L1, B_L1, C_L0, init=True, kL0Size=K_L1)
                T.barrier_all()
                for i, j in T.Parallel(block_M, block_N):
                    C[i, j] = C_L0[i, j]

    with pytest.raises(Exception):  # noqa: B017
        _compile(main, "ascendc")


@pytest.mark.low_priority
@pytest.mark.xfail(reason="框架问题：K 不一致时未报错（ascend.py 的 assert 被注释掉）")
def test_gemm_v0_k_mismatch():
    """A 的 K 维度与 B 的 K 维度不一致应报错"""
    M, N = 128, 128
    K_A, K_B = 64, 128
    block_M, block_N, K_L1 = 128, 128, 64

    @T.prim_func
    def main(
        A: T.Tensor((M, K_A), "float16"),
        B: T.Tensor((K_B, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            A_L1 = T.alloc_L1((block_M, K_L1), "float16")
            B_L1 = T.alloc_L1((K_L1, block_N), "float16")
            C_L0 = T.alloc_L0C((block_M, block_N), "float")
            with T.Scope("C"):
                T.copy(A[0, 0], A_L1)
                T.copy(B[0, 0], B_L1)
                T.barrier_all()
                T.gemm_v0(A_L1, B_L1, C_L0, init=True, kL0Size=K_L1)
                T.barrier_all()
                for i, j in T.Parallel(block_M, block_N):
                    C[i, j] = C_L0[i, j]

    with pytest.raises(Exception):  # noqa: B017
        _compile(main, "ascendc")


@pytest.mark.low_priority
@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="gemm_v0 correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("target", ["ascendc"])
def test_gemm_v0_init_false_accumulation(target):
    """init=False 累加模式：两次 gemm_v0 结果应等于一次性大 K 的结果"""
    M, N = 128, 128
    K_total = 128
    K_L1 = 64
    block_M, block_N = 128, 128
    kL0Size = 64

    @T.prim_func
    def main(
        A: T.Tensor((M, K_total), "float16"),
        B: T.Tensor((K_total, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            A_L1 = T.alloc_L1((block_M, K_L1), "float16")
            B_L1 = T.alloc_L1((K_L1, block_N), "float16")
            C_L0 = T.alloc_L0C((block_M, block_N), "float")
            with T.Scope("C"):
                loop_k = T.ceildiv(K_total, K_L1)
                for k in T.serial(loop_k):
                    T.copy(A[0, k * K_L1], A_L1)
                    T.copy(B[k * K_L1, 0], B_L1)
                    T.barrier_all()
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0), kL0Size=kL0Size)
                    T.barrier_all()
                for i, j in T.Parallel(block_M, block_N):
                    C[i, j] = C_L0[i, j]

    kernel = _compile(main, target)
    a = torch.randn(M, K_total, dtype=torch.float16, device="npu")
    b = torch.randn(K_total, N, dtype=torch.float16, device="npu")
    c = torch.zeros(M, N, dtype=torch.float16, device="npu")
    torch.npu.synchronize()
    kernel(a, b, c)
    torch.npu.synchronize()
    ref_c = a @ b
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
