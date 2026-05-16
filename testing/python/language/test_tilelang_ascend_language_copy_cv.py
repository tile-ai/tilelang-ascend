"""
PTO RED-phase test file for UB↔L1 and L0C↔UB copy paths.

This is a TDD test file - tests are expected to FAIL initially because:
- PTO backend does not yet support UB↔L1 copy (TPUSH)
- PTO backend does not yet support L0C↔UB copy

These tests establish the baseline for GREEN phase validation.
"""

import pytest
import torch

import tilelang
import tilelang.language as T


PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    tilelang.disable_cache()
    yield


def _compile(program):
    """Compile program with PTO target (NOT ascendc)."""
    return tilelang.compile(program, pass_configs=PASS_CONFIGS, target="pto")


def _torch_dtype(dtype):
    if dtype == "float16":
        return torch.float16
    return torch.float32


def _ub_to_l1_kernel(M=128, N=128, K=128, dtype="float16"):
    """Test kernel: UB→L1→GM round-trip (Vector to Cube communication via TPUSH)."""

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            A_ub = T.alloc_ub((M, K), dtype)
            A_l1 = T.alloc_L1((M, K), dtype)
            B_l1 = T.alloc_L1((K, N), dtype)
            C_l0c = T.alloc_L0C((M, N), "float")

            # GM → UB
            T.copy(A, A_ub)

            # UB → L1 (TPUSH): Expected to fail in RED phase
            T.copy(A_ub, A_l1)
            T.copy(B, B_l1)

            # L1 → GEMM → L0C (Cube computation)
            T.gemm_v0(A_l1, B_l1, C_l0c, init=True)

            # L0C → GM
            T.copy(C_l0c, C)

    return main


def _l0c_to_ub_kernel(M=128, N=128, K=128, dtype="float16"):
    """Test kernel: GM→L1→GEMM→L0C→UB→GM (Cube to Vector communication)."""

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), "float"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            A_l1 = T.alloc_L1((M, K), dtype)
            B_l1 = T.alloc_L1((K, N), dtype)
            C_l0c = T.alloc_L0C((M, N), "float")
            C_ub = T.alloc_ub((M, N), "float")

            # GM → L1
            T.copy(A, A_l1)
            T.copy(B, B_l1)

            # L1 → GEMM → L0C (Cube computation)
            T.gemm_v0(A_l1, B_l1, C_l0c, init=True)

            # L0C → UB (Cube to Vector): Expected to fail in RED phase
            T.copy(C_l0c, C_ub)

            # UB → GM
            T.copy(C_ub, C)

    return main


def _combined_kernel(M=128, N=128, K=128, dtype="float16"):
    """Test kernel: Combined UB↔L1 and L0C↔UB paths in single kernel."""

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
        D: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            # UB↔L1 path buffers
            A_ub = T.alloc_ub((M, K), dtype)
            A_l1 = T.alloc_L1((M, K), dtype)

            # L0C↔UB path buffers
            B_l1 = T.alloc_L1((K, N), dtype)
            C_l0c = T.alloc_L0C((M, N), "float")
            C_ub = T.alloc_ub((M, N), dtype)

            # Path 1: GM → UB → L1 (TPUSH)
            T.copy(A, A_ub)
            T.copy(A_ub, A_l1)

            # Path 2: GM → L1 → GEMM → L0C → UB
            T.copy(B, B_l1)
            T.gemm_v0(A_l1, B_l1, C_l0c, init=True)
            T.copy(C_l0c, C_ub)

            # Output: UB → GM
            T.copy(C_ub, C)

            # Output: L0C → GM
            T.copy(C_l0c, D)

    return main


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="PTO UB→L1 copy test requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("dtype", ["float16"])
def test_ub_to_l1_pto(dtype):
    """
    RED phase test for UB→L1 copy path with PTO target.

    This test should FAIL because:
    - PTO backend does not support UB→L1 copy (TPUSH) yet
    - The test validates the test infrastructure exists
    - Establishes baseline for GREEN phase validation

    Test flow:
    1. Create input tensor (128x128, float16)
    2. Compile kernel with UB→L1 copy using target="pto"
    3. Execute kernel (expect compilation error)
    4. Verify data correctness (should not reach this point)
    """
    M, N, K = 128, 128, 128

    with open("ub_to_l1.cpp", "w", encoding="utf-8") as f:

        program = _ub_to_l1_kernel(M=M, N=N, K=K, dtype=dtype)

        kernel = _compile(program)

        print("source code dumped to: ub_to_l1.cpp")
        f.write(kernel.get_kernel_source())

    torch_dtype = _torch_dtype(dtype)

    a = torch.randn((M, K), dtype=torch_dtype, device="npu")
    b = torch.randn((K, N), dtype=torch_dtype, device="npu")
    c = torch.empty((M, N), dtype=torch_dtype, device="npu")
    torch.npu.synchronize()

    kernel(a, b, c)
    torch.npu.synchronize()

    torch.testing.assert_close(c, a @ b, rtol=1e-3, atol=1e-3)


# @pytest.mark.xfail(reason="PTO L0C↔UB copy not yet implemented")
@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="PTO L0C→UB copy test requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("dtype", ["float16"])
def test_l0c_to_ub_pto(dtype):
    """
    RED phase test for L0C→UB copy path with PTO target.

    This test should FAIL because:
    - PTO backend does not support L0C→UB copy yet
    - The test validates the test infrastructure exists
    - Establishes baseline for GREEN phase validation

    Test flow:
    1. Create input tensors (128x128, float16)
    2. Compile kernel with L0C→UB copy using target="pto"
    3. Execute kernel (expect compilation error)
    4. Verify GEMM result correctness (should not reach this point)
    """
    M, N, K = 128, 128, 128

    with open("l0c_to_ub.cpp", "w", encoding="utf-8") as f:

        program = _l0c_to_ub_kernel(M=M, N=N, K=K, dtype=dtype)

        kernel = _compile(program)

        print("source code dumped to: l0c_to_ub.cpp")
        f.write(kernel.get_kernel_source())

    torch_dtype = _torch_dtype(dtype)

    a = torch.randn((M, K), dtype=torch_dtype, device="npu")
    b = torch.randn((K, N), dtype=torch_dtype, device="npu")
    c = torch.empty((M, N), dtype=torch.float32, device="npu")
    torch.npu.synchronize()

    kernel(a, b, c)
    torch.npu.synchronize()

    ref_c = a @ b
    torch.testing.assert_close(c, ref_c.to(torch.float32), rtol=1e-2, atol=1e-2)


# @pytest.mark.xfail(reason="PTO UB↔L1 and L0C↔UB copy not yet implemented")
@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="PTO combined copy test requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("dtype", ["float16"])
def test_combined_pto(dtype):
    """
    RED phase test for combined UB↔L1 and L0C↔UB paths with PTO target.

    This test should FAIL because:
    - PTO backend does not support UB↔L1 copy (TPUSH) yet
    - PTO backend does not support L0C→UB copy yet
    - The test validates the test infrastructure exists
    - Establishes baseline for GREEN phase validation

    Test flow:
    1. Create input tensors (128x128, float16)
    2. Compile kernel with both copy paths using target="pto"
    3. Execute kernel (expect compilation error)
    4. Verify both paths correctness (should not reach this point)
    """
    M, N, K = 128, 128, 128

    with open("combined.cpp", "w", encoding="utf-8") as f:

        program = _combined_kernel(M=M, N=N, K=K, dtype=dtype)

        kernel = _compile(program)

        print("source code dumped to: combined.cpp")
        f.write(kernel.get_kernel_source())

    torch_dtype = _torch_dtype(dtype)

    a = torch.randn((M, K), dtype=torch_dtype, device="npu")
    b = torch.randn((K, N), dtype=torch_dtype, device="npu")
    c = torch.empty((M, N), dtype=torch_dtype, device="npu")
    d = torch.empty((M, K), dtype=torch_dtype, device="npu")
    torch.npu.synchronize()

    kernel(a, b, c, d)
    torch.npu.synchronize()

    # Verify GEMM result (C = A @ B)
    ref_c = a @ b
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    # Verify L0C→UB round-trip (D = C)
    torch.testing.assert_close(d, c, rtol=1e-3, atol=1e-3)
