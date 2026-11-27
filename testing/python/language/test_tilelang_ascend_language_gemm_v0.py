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


# =============================================================================
# Regression: int8 gemm_v0 with kL0split > 1 (issue #1395)
# -----------------------------------------------------------------------------
# gemm_v0 in src/tl_templates/ascend/common.h previously hardcoded ``16`` in
# the L1->L0 K-tile offset for the transpose_A and non-transpose_B branches.
# ``16`` equals ``ELE_NUM_PER_C0`` only for half (32B/sizeof(half)=16); for
# int8 it must be 32, so once kL0split > 1 (K_L1 > kL0Size) the second K-tile
# read a wrong offset and produced garbage. K_L1 <= kL0Size (kL0split == 1)
# masked the bug because the offset was always 0.
#
# These cases target the two fixed branches on the ascendc backend (PTO uses a
# separate, dtype-aware offset path and is unaffected):
#   * transpose_A=True  -> A[kL0Idx * ELE_NUM_PER_C0 * kL0Size]
#   * transpose_B=False -> B[... + kL0Idx * ELE_NUM_PER_C0 * kL0Size]
# =============================================================================
def _gemm_v0_int8_kernel(M, N, K, block_M, block_N, K_L1, transpose_A=False, transpose_B=False, kL0Size=128):
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
                    T.gemm_v0(
                        A_L1,
                        B_L1,
                        C_L0,
                        transpose_A=transpose_A,
                        transpose_B=transpose_B,
                        init=(k == 0),
                        kL0Size=kL0Size,
                    )
                    T.barrier_all()

                # NOTE: use T.copy (not T.Parallel element-wise store) for the
                # int32 L0C -> GM write: the Parallel path hits an unrelated
                # codegen InternalError on int32 GM stores, out of scope here.
                T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


def _run_int8_kl0split_case(transpose_A, transpose_B, kL0Size, K_L1):
    # int8 fractal needs M/N >= 16 and K >= 32. K_L1 > kL0Size forces kL0split
    # >= 2, which is the trigger condition for issue #1395.
    M, N, K = 256, 256, 512
    block_M, block_N = 128, 128

    program = _gemm_v0_int8_kernel(
        M,
        N,
        K,
        block_M,
        block_N,
        K_L1,
        transpose_A=transpose_A,
        transpose_B=transpose_B,
        kL0Size=kL0Size,
    )
    kernel = _compile(program, "ascendc")

    a_gm_shape = (K, M) if transpose_A else (M, K)
    b_gm_shape = (N, K) if transpose_B else (K, N)
    torch.manual_seed(0)
    a = torch.randint(-5, 6, a_gm_shape, dtype=torch.int8, device="npu")
    b = torch.randint(-5, 6, b_gm_shape, dtype=torch.int8, device="npu")
    c = torch.zeros(M, N, dtype=torch.int32, device="npu")

    torch.npu.synchronize()
    kernel(a, b, c)
    torch.npu.synchronize()

    # int32 matmul is unsupported on NPU; compute the reference on CPU.
    a_ref = a.cpu().T if transpose_A else a.cpu()
    b_ref = b.cpu().T if transpose_B else b.cpu()
    ref_c = a_ref.to(torch.int32) @ b_ref.to(torch.int32)

    # int8 -> int32 matmul is exact; assert a tight tolerance to catch any
    # offset-misalignment regression (the original bug gave max diff ~1649).
    torch.testing.assert_close(c.cpu(), ref_c, rtol=0, atol=0)


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="gemm_v0 correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("transpose_A,transpose_B", [(True, False), (False, False)])
def test_gemm_v0_int8_kl0split_gt1(transpose_A, transpose_B):
    # K_L1=256, kL0Size=128 -> kL0split=2: triggers the offset bug pre-fix.
    _run_int8_kl0split_case(transpose_A, transpose_B, kL0Size=128, K_L1=256)


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="gemm_v0 correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("transpose_A,transpose_B", [(True, False), (False, False)])
def test_gemm_v0_int8_kl0split_eq1(transpose_A, transpose_B):
    # K_L1 == kL0Size -> kL0split=1: the masked case, must remain correct.
    _run_int8_kl0split_case(transpose_A, transpose_B, kL0Size=128, K_L1=128)


# ============================================================
# Supplementary dtype coverage tests
# ============================================================


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


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="gemm_v0 correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("target", ["ascendc", pytest.param("pto", marks=pytest.mark.low_priority)])
def test_gemm_v0_bfloat16_precision(target):
    """bfloat16 precision test (existing tests only cover float16)."""
    _run_dtype_precision("bfloat16", "float", target)


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="gemm_v0 correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize(
    "transpose_A,transpose_B",
    [
        (False, False),
        pytest.param(False, True, marks=pytest.mark.low_priority),
        pytest.param(True, False, marks=pytest.mark.low_priority),
        (True, True),
    ],
)
@pytest.mark.parametrize("target", ["ascendc", pytest.param("pto", marks=pytest.mark.low_priority)])
def test_gemm_v0_int8_transpose_full(target, transpose_A, transpose_B):
    """int8 transpose full coverage (existing tests only cover 2 combinations)."""
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


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_gemm_v0_dtype_mismatch(target):
    """A/B dtype mismatch should fail to compile."""
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
        _compile(main, target)


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="gemm_v0 correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_gemm_v0_init_false_accumulation(target):
    """init=False accumulation: two gemm_v0 calls should equal one large-K call."""
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


# ---------------------------------------------------------------------------
# fp32 matrix: the fp32 copy path is fixed by the catlass v2.2.0 bump; these
# cases pin the corrected behaviour on every branch shape that historically
# mattered (fp32 operands are 4 bytes, so kL0Size is sized to fit the L0A
# ping-pong budget where the fp16-typical 128 would trip the compile-time
# static_assert).
# ---------------------------------------------------------------------------


def _gemm_v0_fp32_kernel(
    M,
    N,
    K,
    block_M,
    block_N,
    K_L1,
    dtype="float32",
    out_dtype="float32",
    accum_dtype="float32",
    kL0Size=64,
    transpose_A=False,
    transpose_B=False,
    use_hf32=False,
):
    # A dedicated staging for fp32: the shared _gemm_v0_kernel writes the
    # L0C result back element-wise, which the combine-CV pass rejects for
    # fp32 (Vector op outside T.Scope("V")); the tile copy below stages the
    # same result through the L0C->GM path instead.
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
        C: T.Tensor((M, N), out_dtype),
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
                    T.gemm_v0(
                        A_L1,
                        B_L1,
                        C_L0,
                        transpose_A=transpose_A,
                        transpose_B=transpose_B,
                        init=(k == 0),
                        kL0Size=kL0Size,
                        use_hf32=use_hf32,
                    )
                    T.barrier_all()

                T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


_TORCH_FP32_DTYPE = {"float32": torch.float32, "float16": torch.float16}


def _run_fp32_case(
    target,
    M,
    N,
    K,
    block_M,
    block_N,
    K_L1,
    kL0Size,
    dtype="float32",
    out_dtype="float32",
    accum_dtype="float32",
    rtol=1e-4,
    atol=1e-4,
    transpose_A=False,
    transpose_B=False,
):
    """Run one fp32 gemm_v0 case and check against the torch fp32 reference.

    Tolerance bands are measured: the tight 1e-4 band holds for K<=256-level
    accumulation; the relaxed 1e-3 band matches the measured 1.37e-4 max
    deviation of the large-K shape below (plus headroom).
    """
    program = _gemm_v0_fp32_kernel(
        M,
        N,
        K,
        block_M,
        block_N,
        K_L1,
        dtype,
        out_dtype,
        accum_dtype,
        kL0Size,
        transpose_A,
        transpose_B,
    )
    kernel = _compile(program, target)

    in_t = _TORCH_FP32_DTYPE[dtype]
    out_t = _TORCH_FP32_DTYPE[out_dtype]
    if transpose_A:
        a = torch.randn(K, M, dtype=in_t, device="npu")
    else:
        a = torch.randn(M, K, dtype=in_t, device="npu")
    if transpose_B:
        b = torch.randn(N, K, dtype=in_t, device="npu")
    else:
        b = torch.randn(K, N, dtype=in_t, device="npu")
    c = torch.zeros(M, N, dtype=out_t, device="npu")

    torch.npu.synchronize()
    kernel(a, b, c)
    torch.npu.synchronize()

    a_ref = a.float().T if transpose_A else a.float()
    b_ref = b.float().T if transpose_B else b.float()
    ref = a_ref @ b_ref
    torch.testing.assert_close(c.float(), ref, rtol=rtol, atol=atol)


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="gemm_v0 fp32 correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_gemm_v0_fp32_basic(target):
    _run_fp32_case(target, 256, 256, 256, 128, 128, 64, kL0Size=64, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="gemm_v0 fp32 correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_gemm_v0_fp32_out_fp16(target):
    # L0C(fp32) -> GM(fp16) rides the on-the-fly cast; fp16 output precision.
    _run_fp32_case(
        target,
        256,
        256,
        256,
        128,
        128,
        64,
        kL0Size=64,
        out_dtype="float16",
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="gemm_v0 fp32 correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("transpose_A,transpose_B", [(False, False), (False, True), (True, False), (True, True)])
def test_gemm_v0_fp32_transpose(target, transpose_A, transpose_B):
    _run_fp32_case(
        target,
        256,
        256,
        256,
        128,
        128,
        64,
        kL0Size=64,
        rtol=1e-3,
        atol=1e-3,
        transpose_A=transpose_A,
        transpose_B=transpose_B,
    )


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="gemm_v0 fp32 correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_gemm_v0_fp32_large_shape(target):
    # The large transpose_B=False shape: the historical failing
    # configuration; measured max deviation on the fixed build is 1.37e-4.
    _run_fp32_case(target, 1024, 1024, 1024, 128, 128, 64, kL0Size=64, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="gemm_v0 fp32 correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize(
    "target",
    [
        "ascendc",
        pytest.param(
            "pto",
            marks=pytest.mark.skip(
                reason="pto backend rejects this K-tail shape at compile time; "
                "tracked as a pto-side limitation, out of scope for the AscendC fp32 guard"
            ),
        ),
    ],
)
def test_gemm_v0_fp32_kl0size_tail(target):
    # K=200 with kL0Size=64: the L0-level K tail (200 = 3*64 + 8) exercises
    # the gemm_v0 tail-mma path (whole-C0-block rounding, zN hole hygiene).
    _run_fp32_case(target, 256, 128, 200, 128, 128, 200, kL0Size=64, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="gemm_v0 fp32 correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_gemm_v0_fp32_kl0size_boundary_ok(target):
    # A legal fp32 tile: must compile and be correct.
    _run_fp32_case(target, 128, 128, 64, 128, 128, 64, kL0Size=64, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="gemm_v0 fp32 correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("target", ["ascendc"])
def test_gemm_v0_fp32_kl0size_boundary_reject(target):
    # An oversized fp32 tile must be rejected at compile time by the L0A
    # budget static_assert -- clearly, not silently. AscendC-only: the check
    # lives in the AscendC common.h path; the pto backend compiles this
    # shape without that gate.
    with pytest.raises(RuntimeError, match=r".*Compilation Failed.*"):
        _run_fp32_case(
            target,
            128,
            128,
            128,
            128,
            128,
            128,
            kL0Size=128,
            rtol=1e-4,
            atol=1e-4,
        )


# ---------------------------------------------------------------------------
# HF32 (tf32-class) mode: T.gemm_v0(..., use_hf32=True) rounds FP32 operands
# in L0A/L0B to the 10-bit-mantissa format before the cube multiply,
# trading precision for throughput. Default off; non-fp32 inputs silently
# ignore the switch. Tolerance bands are measured on 910B3/CANN 9.2.0:
# the tf32 band starts at 1e-2 for K<=256 shapes; the large-K band
# (2e-2/5e-2) carries 2x headroom over the measured 2.36e-2 max deviation.
#
# CI tiering: the switch contract cases (off baseline, on basic, fp16 mix)
# run in the PR gate; the large shape and transpose-combination variants
# are low_priority (daily full runs).
# ---------------------------------------------------------------------------


def _run_hf32_case(
    target,
    use_hf32,
    dtype="float32",
    rtol=1e-2,
    atol=1e-2,
    M=256,
    N=256,
    K=256,
    transpose_A=False,
    transpose_B=False,
):
    program = _gemm_v0_fp32_kernel(
        M,
        N,
        K,
        128,
        128,
        64,
        dtype,
        "float32",
        "float32",
        64,
        transpose_A,
        transpose_B,
        use_hf32,
    )
    kernel = _compile(program, target)

    in_t = _TORCH_FP32_DTYPE[dtype]
    if transpose_A:
        a = torch.randn(K, M, dtype=in_t, device="npu")
    else:
        a = torch.randn(M, K, dtype=in_t, device="npu")
    if transpose_B:
        b = torch.randn(N, K, dtype=in_t, device="npu")
    else:
        b = torch.randn(K, N, dtype=in_t, device="npu")
    c = torch.zeros(M, N, dtype=torch.float32, device="npu")

    torch.npu.synchronize()
    kernel(a, b, c)
    torch.npu.synchronize()

    a_ref = a.float().T if transpose_A else a.float()
    b_ref = b.float().T if transpose_B else b.float()
    ref = a_ref @ b_ref
    torch.testing.assert_close(c, ref, rtol=rtol, atol=atol)


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="gemm_v0 hf32 correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_gemm_v0_hf32_off_baseline(target):
    # The switch present but off: must match the fp32 tight tolerance.
    _run_hf32_case(target, use_hf32=False, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="gemm_v0 hf32 correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_gemm_v0_hf32_on_basic(target):
    # The switch on at the tf32 band (small shape, PR-gate tier).
    _run_hf32_case(target, use_hf32=True, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="gemm_v0 hf32 correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_gemm_v0_hf32_fp16_ignores_switch(target):
    # fp16 inputs with the switch on: silently ignored, fp16-grade result
    # against the fp32 reference.
    _run_hf32_case(target, use_hf32=True, dtype="float16", rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="gemm_v0 hf32 correctness requires an Ascend NPU runtime",
)
@pytest.mark.low_priority
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_gemm_v0_hf32_on_large_shape(target):
    # Large shape under HF32 at the measured large-K band.
    _run_hf32_case(target, use_hf32=True, rtol=2e-2, atol=5e-2, M=1024, N=1024, K=1024)


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="gemm_v0 hf32 correctness requires an Ascend NPU runtime",
)
@pytest.mark.low_priority
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("transpose_A,transpose_B", [(False, True), (True, False), (True, True)])
def test_gemm_v0_hf32_transpose_on(target, transpose_A, transpose_B):
    # HF32 + transpose combinations: SetHF32Mode is set around each Mmad and
    # must hold regardless of the tile-copy branch taken around it.
    _run_hf32_case(
        target,
        use_hf32=True,
        rtol=2e-2,
        atol=5e-2,
        transpose_A=transpose_A,
        transpose_B=transpose_B,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-n", "8"])
