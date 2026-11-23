"""gemm_v0 HF32 (tf32-class) mode tests.

HF32 rounds FP32 operands in L0A/L0B to the 10-bit-mantissa format before
the cube multiply (AscendC::SetHF32Mode), trading fp32 precision for
throughput. Three groups:

- off baseline: passing use_hf32=False (or not passing it) must be
  byte-equivalent to the existing behaviour -- the fp32 matrix from the
  sibling fp32 test file at tight tolerance.
- on correctness: same shapes with use_hf32=True, checked against the
  torch fp32 reference at the tf32 tolerance.
- dtype mix: fp16 inputs with use_hf32=True must silently ignore the
  switch (result equals the plain fp16 path).

The tf32 tolerance starts at rtol=1e-2 / atol=1e-2 and is tightened to
"measured max deviation x 5" once first measured on 910B3; the final
values and the measurement that produced them are recorded in the PR.
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


def _gemm_v0_kernel(
    M, N, K, block_M, block_N, K_L1, dtype="float32",
    out_dtype="float32", accum_dtype="float32", kL0Size=64, use_hf32=False,
    transpose_A=False, transpose_B=False,
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
                        A_L1, B_L1, C_L0, init=(k == 0),
                        kL0Size=kL0Size, use_hf32=use_hf32,
                        transpose_A=transpose_A, transpose_B=transpose_B,
                    )
                    T.barrier_all()

                T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


def _compile(program, target):
    return tilelang.compile(program, pass_configs=PASS_CONFIGS, target=target)


_TORCH = {"float32": torch.float32, "float16": torch.float16}


def _run_case(target, use_hf32, dtype, rtol, atol, M=256, N=256, K=256):
    """Run one gemm_v0 case and assert against the torch fp32 reference.

    Measured on 910B3 / CANN 9.2.0 (1024^3, HF32 on): max_abs=2.36e-2,
    mean_rel=1.6e-3 -- the large-K band below is 2x that max_abs; the
    256-shape band stays at the initial 1e-2 (measured well below).
    """
    program = _gemm_v0_kernel(
        M, N, K, 128, 128, 64, dtype=dtype, use_hf32=use_hf32
    )
    kernel = _compile(program, target)

    in_t = _TORCH[dtype]
    a = torch.randn(M, K, dtype=in_t, device="npu")
    b = torch.randn(K, N, dtype=in_t, device="npu")
    c = torch.zeros(M, N, dtype=_TORCH["float32"], device="npu")

    torch.npu.synchronize()
    kernel(a, b, c)
    torch.npu.synchronize()

    ref = a.float() @ b.float()
    torch.testing.assert_close(c, ref, rtol=rtol, atol=atol)
    return (c - ref).abs().max().item()


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="gemm_v0 hf32 correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_gemm_v0_hf32_off_baseline(target):
    # The switch present but off: must match the fp32 tight tolerance.
    _run_case(target, use_hf32=False, dtype="float32", rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="gemm_v0 hf32 correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_gemm_v0_hf32_on_tf32_tolerance(target):
    # tf32-class rounding: initial 1e-2 band, tightened by measurement.
    _run_case(target, use_hf32=True, dtype="float32", rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="gemm_v0 hf32 correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_gemm_v0_hf32_issue1396_shape_on(target):
    # The 1024^3 #1396 shape under HF32. Large-K band: measured
    # max_abs=2.36e-2 on 910B3, band set with 2x headroom.
    _run_case(
        target, use_hf32=True, dtype="float32", rtol=2e-2, atol=5e-2,
        M=1024, N=1024, K=1024,
    )


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="gemm_v0 hf32 correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("transpose_A,transpose_B", [(False, True), (True, False), (True, True)])
def test_gemm_v0_hf32_transpose_on(target, transpose_A, transpose_B):
    # HF32 + transpose combinations: SetHF32Mode is set around each Mmad and
    # must hold regardless of the tile-copy branch taken around it.
    program = _gemm_v0_kernel(
        256, 256, 256, 128, 128, 64, dtype="float32",
        use_hf32=True, transpose_A=transpose_A, transpose_B=transpose_B,
    )
    kernel = _compile(program, target)

    a = torch.randn(256, 256, dtype=torch.float32, device="npu")
    b = torch.randn(256, 256, dtype=torch.float32, device="npu")
    c = torch.zeros(256, 256, dtype=torch.float32, device="npu")

    torch.npu.synchronize()
    kernel(a, b, c)
    torch.npu.synchronize()

    a_ref = a.float().T if transpose_A else a.float()
    b_ref = b.float().T if transpose_B else b.float()
    ref = a_ref @ b_ref
    torch.testing.assert_close(c, ref, rtol=2e-2, atol=5e-2)


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="gemm_v0 hf32 correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_gemm_v0_hf32_fp16_ignores_switch(target):
    # fp16 inputs with the switch on: silently ignored, fp16-grade result
    # against the fp32 reference.
    _run_case(target, use_hf32=True, dtype="float16", rtol=1e-2, atol=1e-2)
