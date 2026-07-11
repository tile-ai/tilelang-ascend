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


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    tilelang.disable_cache()
    yield


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


def _run_precision_case(target, kL0Size, shape_group):
    full_M, full_N, full_K, block_M, block_N, K_L1 = SHAPE_CONFIGS[shape_group]
    dtype = "float16"
    accum_dtype = "float"

    program = _gemm_v0_kernel(full_M, full_N, full_K, block_M, block_N, K_L1, dtype, accum_dtype, kL0Size=kL0Size)
    kernel = _compile(program, target)

    a = torch.randn(full_M, full_K, dtype=torch.float16, device="npu")
    b = torch.randn(full_K, full_N, dtype=torch.float16, device="npu")
    c = torch.zeros(full_M, full_N, dtype=torch.float16, device="npu")

    torch.npu.synchronize()
    kernel(a, b, c)
    torch.npu.synchronize()

    ref_c = a @ b
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="gemm_v0 correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("kL0Size", [128, 64])
@pytest.mark.parametrize("shape_group", ["A", "B", "C", "D"])
def test_gemm_v0_precision(target, kL0Size, shape_group):
    _run_precision_case(target, kL0Size, shape_group)


def _run_transpose_case(target, transpose_A, transpose_B):
    M, N, K = 128, 256, 128
    block_M, block_N, K_L1 = 128, 256, 128
    dtype = "float16"
    accum_dtype = "float"
    kL0Size = 64

    program = _gemm_v0_kernel(
        M, N, K, block_M, block_N, K_L1, dtype, accum_dtype, kL0Size=kL0Size, transpose_A=transpose_A, transpose_B=transpose_B
    )
    kernel = _compile(program, target)

    a_gm_shape = (K, M) if transpose_A else (M, K)
    b_gm_shape = (N, K) if transpose_B else (K, N)

    a = torch.randn(*a_gm_shape, dtype=torch.float16, device="npu")
    b = torch.randn(*b_gm_shape, dtype=torch.float16, device="npu")
    c = torch.zeros(M, N, dtype=torch.float16, device="npu")

    torch.npu.synchronize()
    kernel(a, b, c)
    torch.npu.synchronize()

    a_ref = a.T if transpose_A else a
    b_ref = b.T if transpose_B else b
    ref_c = a_ref @ b_ref

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="gemm_v0 correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("transpose_A,transpose_B", [(False, True), (True, False)])
def test_gemm_v0_transpose(target, transpose_A, transpose_B):
    _run_transpose_case(target, transpose_A, transpose_B)


def _run_default_case(target):
    full_M, full_N, full_K, block_M, block_N, K_L1 = SHAPE_CONFIGS["A"]
    dtype = "float16"
    accum_dtype = "float"

    program = _gemm_v0_kernel(full_M, full_N, full_K, block_M, block_N, K_L1, dtype, accum_dtype)
    kernel = _compile(program, target)

    a = torch.randn(full_M, full_K, dtype=torch.float16, device="npu")
    b = torch.randn(full_K, full_N, dtype=torch.float16, device="npu")
    c = torch.zeros(full_M, full_N, dtype=torch.float16, device="npu")

    torch.npu.synchronize()
    kernel(a, b, c)
    torch.npu.synchronize()

    ref_c = a @ b
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="gemm_v0 correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_gemm_v0_default_kL0Size(target):
    _run_default_case(target)


def test_gemm_v0_invalid_kL0Size_zero(capfd):
    with pytest.raises(Exception):  # noqa: B017
        _gemm_v0_kernel(128, 128, 256, 128, 128, 256, "float16", "float", kL0Size=0)
    captured = capfd.readouterr()
    assert "must be a positive integer" in captured.err


def test_gemm_v0_invalid_kL0Size_negative(capfd):
    with pytest.raises(Exception):  # noqa: B017
        _gemm_v0_kernel(128, 128, 256, 128, 128, 256, "float16", "float", kL0Size=-16)
    captured = capfd.readouterr()
    assert "must be a positive integer" in captured.err


def test_gemm_v0_invalid_kL0Size_not_aligned(capfd):
    with pytest.raises(Exception):  # noqa: B017
        _gemm_v0_kernel(128, 128, 256, 128, 128, 256, "float16", "float", kL0Size=100)
    captured = capfd.readouterr()
    assert "multiple of 16" in captured.err


def test_gemm_v0_invalid_kL0Size_exceeds_mmad_limit(capfd):
    with pytest.raises(Exception):  # noqa: B017
        _gemm_v0_kernel(128, 128, 256, 128, 128, 256, "float16", "float", kL0Size=4096)
    captured = capfd.readouterr()
    assert "4095" in captured.err


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-n", "8"])
