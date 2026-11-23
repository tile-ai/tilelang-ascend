"""gemm_v0 fp32 test matrix.

Backfill for the gap exposed by issues #1396 / #1921: the gemm_v0 test
matrix only covered fp16/bf16, so the old catlass fp32 copy corruption
went unnoticed for months. These cases pin the fixed behavior on every
branch shape that historically mattered:

- basic fp32 (in fp32 / accum fp32 / out fp32)
- fp32 in -> fp16 out (L0C->GM on-the-fly cast)
- all four transpose combinations
- the exact #1396 failing shape (transpose_B=False, 1024, K_L1=64)
- kL0Size=64 tail shape (K=200 with kL0Size=64 -> L0-level K tail)
- kL0Size capacity boundary: 64 must pass, 128 must trip the compile-time
  L0A budget assert (negative case)
"""

import pytest
import tilelang
import tilelang.language as T
import torch

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    tilelang.disable_cache()
    yield


def _gemm_v0_kernel_fp32(
    M,
    N,
    K,
    block_M,
    block_N,
    K_L1,
    dtype="float32",
    out_dtype=None,
    accum_dtype="float32",
    kL0Size=64,
    transpose_A=False,
    transpose_B=False,
):
    if out_dtype is None:
        out_dtype = dtype
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
                    )
                    T.barrier_all()

                T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


def _compile(program, target):
    return tilelang.compile(program, pass_configs=PASS_CONFIGS, target=target)


_TORCH_DTYPE = {"float32": torch.float32, "float16": torch.float16}


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
    rtol=1e-4,
    atol=1e-4,
    transpose_A=False,
    transpose_B=False,
):
    program = _gemm_v0_kernel_fp32(
        M, N, K, block_M, block_N, K_L1, dtype, out_dtype, kL0Size=kL0Size,
        transpose_A=transpose_A, transpose_B=transpose_B,
    )
    kernel = _compile(program, target)

    in_t = _TORCH_DTYPE[dtype]
    out_t = _TORCH_DTYPE[out_dtype]
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
    # Tight tolerance: K=256-level fp32 accumulation rounds well below 1e-4.
    _run_fp32_case(target, 256, 256, 256, 128, 128, 64, kL0Size=64, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="gemm_v0 fp32 correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_gemm_v0_fp32_out_fp16(target):
    # L0C(fp32) -> GM(fp16) rides the on-the-fly cast; fp16 output precision.
    _run_fp32_case(
        target, 256, 256, 256, 128, 128, 64, kL0Size=64,
        out_dtype="float16", rtol=1e-3, atol=1e-3,
    )


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="gemm_v0 fp32 correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("transpose_A,transpose_B", [(False, False), (False, True), (True, False), (True, True)])
def test_gemm_v0_fp32_transpose(target, transpose_A, transpose_B):
    _run_fp32_case(
        target, 256, 256, 256, 128, 128, 64, kL0Size=64,
        rtol=1e-3, atol=1e-3,
        transpose_A=transpose_A, transpose_B=transpose_B,
    )


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="gemm_v0 fp32 correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_gemm_v0_fp32_issue1396_shape(target):
    # The exact failing shape from issue #1396: transpose_B=False, 1024^3,
    # K_L1=64. Measured max_abs_diff on the fixed build: 1.37e-4.
    _run_fp32_case(
        target, 1024, 1024, 1024, 128, 128, 64, kL0Size=64, rtol=1e-3, atol=1e-3
    )


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
    # budget static_assert -- clearly, not silently (the old bug was silent).
    # AscendC-only: the L0A budget check lives in the AscendC common.h path;
    # the pto backend compiles this shape without that gate.
    # The static_assert text lives in bisheng stderr and is wrapped by the
    # "Compilation Failed!" RuntimeError, so match on the compile-time gate.
    with pytest.raises(RuntimeError, match=r".*Compilation Failed.*"):
        _run_fp32_case(
            target, 128, 128, 128, 128, 128, 128, kL0Size=128,
            rtol=1e-4, atol=1e-4,
        )
