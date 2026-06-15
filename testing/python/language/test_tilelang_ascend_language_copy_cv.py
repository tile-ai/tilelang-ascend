"""
Test file for UB↔L1 and L0C↔UB copy paths
"""

import pytest
import torch

import tilelang
import tilelang.language as T
from tilelang.utils.target import determine_platform


PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}


@pytest.fixture(scope="session", autouse=True)
def disable_cache():
    tilelang.disable_cache()
    yield


@pytest.fixture(scope="session")
def platform():
    return determine_platform()


skip_A5 = pytest.mark.skipif(determine_platform() == "A5", reason="Only test for non-A5 platform")
only_A5 = pytest.mark.skipif(determine_platform() != "A5", reason="Only test for A5 platform")


def _compile(program, target="pto", expert=False):
    """Compile program with target and developing mode."""
    pass_config = None if expert else PASS_CONFIGS
    return tilelang.compile(program, pass_configs=pass_config, target=target)


def _torch_dtype(dtype):
    if dtype == "float16":
        return torch.float16
    return torch.float32


def _ub_to_l1_kernel(M=128, N=128, K=128, dtype="float16", accum_dtype="float"):
    """Test kernel: GM→UB→L1→GM"""
    VEC_NUM = 2
    M_half = M // VEC_NUM

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype), # type: ignore
        B: T.Tensor((K, N), dtype), # type: ignore
        C: T.Tensor((M, N), dtype), # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            A_ub = T.alloc_ub((M_half, K), dtype)
            A_ub_Nz = T.alloc_ub((M_half, K), dtype)
            A_l1 = T.alloc_L1((M, K), dtype)
            B_l1 = T.alloc_L1((K, N), dtype)
            C_l0c = T.alloc_L0C((M, N), accum_dtype)

            # GM → UB
            T.copy(A[vid * M_half: (vid + 1) * M_half, :], A_ub)

            # UB → L1
            T.copy(A_ub, A_l1, tmp=A_ub_Nz)

            # GM → L1
            T.copy(B, B_l1)

            # L1 → GEMM → L0C
            T.gemm_v0(A_l1, B_l1, C_l0c, init=True)

            # L0C → GM
            T.copy(C_l0c, C)

    return main


def _ub_to_l1_kernel_drop_vid(M=128, N=128, K=128, dtype="float16", accum_dtype="float"):
    """Test kernel: GM→UB→L1→GM with vid reduction"""

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype), # type: ignore
        B: T.Tensor((K, N), dtype), # type: ignore
        C: T.Tensor((M, N), dtype), # type: ignore
    ):
        with T.Kernel(1, threads=2, is_npu=True) as _cid:
            A_ub = T.alloc_ub((M, K), dtype)
            A_ub_Nz = T.alloc_ub((M, K), dtype)
            A_l1 = T.alloc_L1((M, K), dtype)
            B_l1 = T.alloc_L1((K, N), dtype)
            C_l0c = T.alloc_L0C((M, N), accum_dtype)

            # GM → UB
            T.copy(A, A_ub)

            # UB → L1
            T.copy(A_ub, A_l1, tmp=A_ub_Nz)

            # GM → L1
            T.copy(B, B_l1)

            # L1 → GEMM → L0C
            T.gemm_v0(A_l1, B_l1, C_l0c, init=True)

            # L0C → GM
            T.copy(C_l0c, C)

    return main


def _ub_to_l1_kernel_lr(M=128, N=128, K=128, dtype="float16", accum_dtype="float"):
    """Test kernel: GM→UB→L1→GM"""
    VEC_NUM = 2
    K_half = K // VEC_NUM

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype), # type: ignore
        B: T.Tensor((K, N), dtype), # type: ignore
        C: T.Tensor((M, N), dtype), # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            A_ub = T.alloc_ub((M, K_half), dtype)
            A_ub_Nz = T.alloc_ub((M, K_half), dtype)
            A_l1 = T.alloc_L1((M, K), dtype)
            B_l1 = T.alloc_L1((K, N), dtype)
            C_l0c = T.alloc_L0C((M, N), accum_dtype)

            # GM → UB
            T.copy(A[:, vid * K_half: (vid + 1) * K_half], A_ub)

            # UB → L1
            T.copy(A_ub, A_l1, tmp=A_ub_Nz)

            # GM → L1
            T.copy(B, B_l1)

            # L1 → GEMM → L0C
            T.gemm_v0(A_l1, B_l1, C_l0c, init=True)

            # L0C → GM
            T.copy(C_l0c, C)

    return main


def _ub_to_l1_kernel_expert_A2(M=128, N=128, K=128, dtype="float16", accum_dtype="float"):
    """Test kernel: GM→UB→L1→GM with hand-crafted copy_ub_to_pipe & copy_pipe_to_l1 calls"""
    VEC_NUM = 2
    M_half = M // VEC_NUM

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype), # type: ignore
        B: T.Tensor((K, N), dtype), # type: ignore
        C: T.Tensor((M, N), dtype), # type: ignore
        workspace_1: T.Tensor((M * K * 2 ), dtype) # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            T._srcCode("using Pipe_0_V2C = TPipe<0, pto::Direction::DIR_V2C, 32768, 2>;")
            T._srcCode("Pipe_0_V2C pipe_0_V2C(workspace_1_handle, 0, 32768);")

            A_ub = T.alloc_ub((M_half, K), dtype)
            A_l1 = T.alloc_L1((M, K), dtype)
            B_l1 = T.alloc_L1((K, N), dtype)
            C_l0c = T.alloc_L0C((M, N), accum_dtype)

            with T.Scope("C"):
                # GM → L1
                T.copy(B, B_l1)

                # UB → L1
                T._srcCode("tl::ascend_pto::copy_pipe_to_l1<pto::TileSplitAxis::TILE_UP_DOWN>(pipe_0_V2C, A_l1);")
                T.set_flag("mte2", "m", 1)
                T.wait_flag("mte2", "m", 1)

                # L1 → GEMM → L0C
                T.gemm_v0(A_l1, B_l1, C_l0c, init=True)
                T.set_flag("m", "fix", 2)
                T.wait_flag("m", "fix", 2)

                # L0C → GM
                T.copy(C_l0c, C)

            with T.Scope("V"):
                # GM → UB
                T.copy(A[vid * M_half: (vid + 1) * M_half, :], A_ub)

                # UB → L1
                # T.copy(A_ub, A_l1)
                T.set_flag("mte2", "mte3", 3)
                T.wait_flag("mte2", "mte3", 3)
                T._srcCode("tl::ascend_pto::copy_ub_to_pipe<pto::TileSplitAxis::TILE_UP_DOWN>(pipe_0_V2C, A_ub);")

    return main


def _ub_to_l1_kernel_expert_A5(M=128, N=128, K=128, dtype="float16", accum_dtype="float"):
    """Test kernel: GM→UB→L1→GM with hand-crafted copy_ub_to_pipe & copy_pipe_to_l1 calls"""
    VEC_NUM = 2
    M_half = M // VEC_NUM

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype), # type: ignore
        B: T.Tensor((K, N), dtype), # type: ignore
        C: T.Tensor((M, N), dtype), # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            T._srcCode("using Pipe_0_V2C = TPipe<0, pto::Direction::DIR_V2C, 32768, 2>;")
            T._srcCode("Pipe_0_V2C pipe_0_V2C(nullptr, 0, 32768);")

            A_ub = T.alloc_ub((M_half, K), dtype)
            A_ub_Nz = T.alloc_ub((M_half, K), dtype)
            A_l1 = T.alloc_L1((M, K), dtype)
            B_l1 = T.alloc_L1((K, N), dtype)
            C_l0c = T.alloc_L0C((M, N), accum_dtype)

            with T.Scope("C"):
                # GM → L1
                T.copy(B, B_l1)

                # UB → L1
                T._srcCode("tl::ascend_pto::copy_pipe_to_l1<pto::TileSplitAxis::TILE_UP_DOWN>(pipe_0_V2C, A_l1);")
                T.set_flag("mte2", "m", 1)
                T.wait_flag("mte2", "m", 1)

                # L1 → GEMM → L0C
                T.gemm_v0(A_l1, B_l1, C_l0c, init=True)
                T.set_flag("m", "fix", 2)
                T.wait_flag("m", "fix", 2)

                # L0C → GM
                T.copy(C_l0c, C)

            with T.Scope("V"):
                # GM → UB
                T.copy(A[vid * M_half: (vid + 1) * M_half, :], A_ub)

                # UB → L1
                # T.copy(A_ub, A_l1)
                T.set_flag("mte2", "v", 3)
                T.wait_flag("mte2", "v", 3)
                T._srcCode("tl::ascend_pto::copy_ub_to_ub_Nz(A_ub, A_ub_Nz);", A_ub_Nz)
                T.set_flag("v", "mte3", 4)
                T.wait_flag("v", "mte3", 4)
                T._srcCode("tl::ascend_pto::copy_ub_to_pipe<pto::TileSplitAxis::TILE_UP_DOWN>(pipe_0_V2C, A_ub, A_ub_Nz);")

    return main


def _l0c_to_ub_kernel(M=128, N=128, K=128, dtype="float16", accum_dtype="float"):
    """Test kernel: GM→L1→GEMM→L0C→UB→GM"""
    VEC_NUM = 2
    M_half = M // VEC_NUM

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype), # type: ignore
        B: T.Tensor((K, N), dtype), # type: ignore
        C: T.Tensor((M, N), accum_dtype), # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            A_l1 = T.alloc_L1((M, K), dtype)
            B_l1 = T.alloc_L1((K, N), dtype)
            C_l0c = T.alloc_L0C((M, N), accum_dtype)
            C_ub = T.alloc_ub((M_half, N), accum_dtype)

            # GM → L1
            T.copy(A, A_l1)
            T.copy(B, B_l1)

            # L1 → GEMM → L0C
            T.gemm_v0(A_l1, B_l1, C_l0c, init=True)

            # L0C → UB
            T.copy(C_l0c, C_ub)

            # UB → GM
            T.copy(C_ub, C[vid * M_half: (vid + 1) * M_half, :])

    return main


def _l0c_to_ub_kernel_drop_vid(M=128, N=128, K=128, dtype="float16", accum_dtype="float"):
    """Test kernel: GM→L1→GEMM→L0C→UB→GM with vid reduction"""

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype), # type: ignore
        B: T.Tensor((K, N), dtype), # type: ignore
        C: T.Tensor((M, N), accum_dtype), # type: ignore
    ):
        with T.Kernel(1, threads=2, is_npu=True) as _cid:
            A_l1 = T.alloc_L1((M, K), dtype)
            B_l1 = T.alloc_L1((K, N), dtype)
            C_l0c = T.alloc_L0C((M, N), accum_dtype)
            C_ub = T.alloc_ub((M, N), accum_dtype)

            # GM → L1
            T.copy(A, A_l1)
            T.copy(B, B_l1)

            # L1 → GEMM → L0C
            T.gemm_v0(A_l1, B_l1, C_l0c, init=True)

            # L0C → UB
            T.copy(C_l0c, C_ub)

            # UB → GM
            T.copy(C_ub, C)

    return main


def _l0c_to_ub_kernel_lr(M=128, N=128, K=128, dtype="float16", accum_dtype="float"):
    """Test kernel: GM→L1→GEMM→L0C→UB→GM"""
    VEC_NUM = 2
    N_half = N // VEC_NUM

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype), # type: ignore
        B: T.Tensor((K, N), dtype), # type: ignore
        C: T.Tensor((M, N), accum_dtype), # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            A_l1 = T.alloc_L1((M, K), dtype)
            B_l1 = T.alloc_L1((K, N), dtype)
            C_l0c = T.alloc_L0C((M, N), accum_dtype)
            C_ub = T.alloc_ub((M, N_half), accum_dtype)

            # GM → L1
            T.copy(A, A_l1)
            T.copy(B, B_l1)

            # L1 → GEMM → L0C
            T.gemm_v0(A_l1, B_l1, C_l0c, init=True)

            # L0C → UB
            T.copy(C_l0c, C_ub)

            # UB → GM
            T.copy(C_ub, C[:, vid * N_half: (vid + 1) * N_half])

    return main


def _combined_kernel(M=128, N=128, K=128, dtype="float16", accum_dtype="float"):
    """Test kernel: Combined UB↔L1 and L0C↔UB paths in single kernel"""
    VEC_NUM = 2
    M_half = M // VEC_NUM

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype), # type: ignore
        B: T.Tensor((K, N), dtype), # type: ignore
        C: T.Tensor((M, N), accum_dtype), # type: ignore
        D: T.Tensor((M, N), dtype), # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            # UB↔L1 path buffers
            A_ub = T.alloc_ub((M_half, K), dtype)
            A_ub_Nz = T.alloc_ub((M_half, K), dtype)
            A_l1 = T.alloc_L1((M, K), dtype)

            # L0C↔UB path buffers
            B_l1 = T.alloc_L1((K, N), dtype)
            C_l0c = T.alloc_L0C((M, N), accum_dtype)
            C_ub = T.alloc_ub((M_half, N), accum_dtype)

            # Path A: GM → UB → L1
            T.copy(A[vid * M_half: (vid + 1) * M_half, :], A_ub)
            T.copy(A_ub, A_l1, tmp=A_ub_Nz)

            # Path B: GM → L1 → GEMM → L0C → UB
            T.copy(B, B_l1)
            T.gemm_v0(A_l1, B_l1, C_l0c, init=True)
            T.copy(C_l0c, C_ub)

            # Output: UB → GM
            T.copy(C_ub, C[vid * M_half: (vid + 1) * M_half, :])

            # Output: L0C → GM
            T.copy(C_l0c, D)

    return main


def _ub_to_l1_case(kernel_func, M=128, N=128, K=128, target="pto", expert=False, manual_workspace=False):
    dtype = "float16"
    program = kernel_func(M=M, N=N, K=K, dtype=dtype)
    kernel = _compile(program, target=target, expert=expert)

    torch_dtype = _torch_dtype(dtype)

    a = torch.randn((M, K), dtype=torch_dtype, device="npu")
    b = torch.randn((K, N), dtype=torch_dtype, device="npu")
    c = torch.empty((M, N), dtype=torch_dtype, device="npu")
    torch.npu.synchronize()

    if not manual_workspace:
        kernel(a, b, c)
    else:
        workspace_1 = torch.empty((M * K * 2,), dtype=torch_dtype, device="npu")
        kernel(a, b, c, workspace_1)
    torch.npu.synchronize()

    ref_c = a @ b
    torch.testing.assert_close(c, ref_c, rtol=1e-3, atol=1e-3)


def _l0c_to_ub_case(kernel_func, M=128, N=128, K=128, target="pto", expert=False, manual_workspace=False):
    dtype = "float16"
    accum_dtype = "float"
    program = kernel_func(M=M, N=N, K=K, dtype=dtype, accum_dtype=accum_dtype)
    kernel = _compile(program, target=target, expert=expert)

    torch_dtype = _torch_dtype(dtype)
    torch_accum_dtype = _torch_dtype(accum_dtype)

    a = torch.randn((M, K), dtype=torch_dtype, device="npu")
    b = torch.randn((K, N), dtype=torch_dtype, device="npu")
    c = torch.empty((M, N), dtype=torch_accum_dtype, device="npu")
    torch.npu.synchronize()

    if not manual_workspace:
        kernel(a, b, c)
    else:
        workspace_1 = torch.empty((M * K * 2,), dtype=torch_dtype, device="npu")
        kernel(a, b, c, workspace_1)
    torch.npu.synchronize()

    ref_c = (a @ b).to(torch_accum_dtype)
    torch.testing.assert_close(c, ref_c, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_ub_to_l1(target):
    M, N, K = 128, 128, 128
    _ub_to_l1_case(_ub_to_l1_kernel, M=M, N=N, K=K, target=target, expert=False)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_ub_to_l1_drop_vid(target):
    M, N, K = 128, 128, 128
    _ub_to_l1_case(_ub_to_l1_kernel_drop_vid, M=M, N=N, K=K, target=target, expert=False)


@pytest.mark.parametrize("target", ["pto"])
def test_ub_to_l1_lr(target):
    M, N, K = 128, 128, 128
    _ub_to_l1_case(_ub_to_l1_kernel_lr, M=M, N=N, K=K, target=target, expert=False)


@skip_A5
@pytest.mark.parametrize("target", ["pto"])
def test_ub_to_l1_expert_A2(target):
    M, N, K = 128, 128, 128
    _ub_to_l1_case(_ub_to_l1_kernel_expert_A2, M=M, N=N, K=K, target=target, expert=True, manual_workspace=True)


@only_A5
@pytest.mark.parametrize("target", ["pto"])
def test_ub_to_l1_expert_A5(target):
    M, N, K = 128, 128, 128
    _ub_to_l1_case(_ub_to_l1_kernel_expert_A5, M=M, N=N, K=K, target=target, expert=True)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_l0c_to_ub(target):
    M, N, K = 128, 128, 128
    _l0c_to_ub_case(_l0c_to_ub_kernel, M=M, N=N, K=K, target=target, expert=False)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_l0c_to_ub_drop_vid(target):
    M, N, K = 128, 128, 128
    _l0c_to_ub_case(_l0c_to_ub_kernel_drop_vid, M=M, N=N, K=K, target=target, expert=False)


@pytest.mark.parametrize("target", ["pto"])
def test_l0c_to_ub_lr(target):
    M, N, K = 128, 128, 128
    _l0c_to_ub_case(_l0c_to_ub_kernel_lr, M=M, N=N, K=K, target=target, expert=False)


def _combined_case(kernel_func, M=128, N=128, K=128, target="pto"):
    dtype = "float16"
    accum_dtype = "float"
    program = kernel_func(M=M, N=N, K=K, dtype=dtype, accum_dtype=accum_dtype)
    kernel = _compile(program, target=target)

    torch_dtype = _torch_dtype(dtype)
    torch_accum_dtype = _torch_dtype(accum_dtype)

    a = torch.randn((M, K), dtype=torch_dtype, device="npu")
    b = torch.randn((K, N), dtype=torch_dtype, device="npu")
    c = torch.empty((M, N), dtype=torch_accum_dtype, device="npu")
    d = torch.empty((M, N), dtype=torch_dtype, device="npu")
    torch.npu.synchronize()

    kernel(a, b, c, d)
    torch.npu.synchronize()

    # Verify GEMM result (C = A @ B)
    ref_d = a @ b
    ref_c = ref_d.to(torch_accum_dtype)
    torch.testing.assert_close(c, ref_c, rtol=1e-3, atol=1e-3)

    # Verify L0C→UB round-trip (D = C)
    torch.testing.assert_close(d, ref_d, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_combined(target):
    M, N, K = 128, 128, 128
    _combined_case(_combined_kernel, M=M, N=N, K=K, target=target)


def _copy_vc_experiment_kernel(M=128, N=128, K=128, dtype="float16", accum_dtype="float"):
    """Test kernel: GM→UB→L1→GM with copy_vc_experiment (TINSERT)"""
    M_half = T.ceildiv(M, 2)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype), # type: ignore
        B: T.Tensor((K, N), dtype), # type: ignore
        C: T.Tensor((M, N), dtype), # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            A_ub = T.alloc_ub((M_half, K), dtype)
            A_ub_nz_tmp = T.alloc_ub((M_half, K), dtype)
            A_l1 = T.alloc_L1((M, K), dtype)
            B_l1 = T.alloc_L1((K, N), dtype)
            C_l0c = T.alloc_L0C((M, N), accum_dtype)

            with T.Scope("C"):
                # UB → L1
                T.wait_cross_flag(0, "M")
                # GM → L1
                T.copy(B, B_l1)
                T.set_flag("mte2", "m", 1)
                T.wait_flag("mte2", "m", 1)
                # L1 → GEMM → L0C (Cube computation)
                T.gemm_v0(A_l1, B_l1, C_l0c, init=True)
                T.set_flag("m", "fix", 2)
                T.wait_flag("m", "fix", 2)
                # L0C → GM
                T.copy(C_l0c, C)

            with T.Scope("V"):
                # GM → UB
                T.copy(A[vid * M_half: (vid + 1) * M_half, :], A_ub)
                # UB → L1 (TINSERT)
                T.set_flag("mte2", "v", 3)
                T.wait_flag("mte2", "v", 3)
                T.copy_op.copy_vc_experiment(A_ub, A_l1[vid * M_half, 0], A_ub_nz_tmp)
                T.set_cross_flag("MTE3", 0)
    return main


def _copy_cv_experiment_kernel(M=128, N=128, K=128, dtype="float16", accum_dtype="float"):
    """Test kernel: GM→L1→GEMM→L0C→UB→GM with copy_cv_experiment (TMOV)"""
    M_half = T.ceildiv(M, 2)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype), # type: ignore
        B: T.Tensor((K, N), dtype), # type: ignore
        C: T.Tensor((M, N), accum_dtype), # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            A_l1 = T.alloc_L1((M, K), dtype)
            B_l1 = T.alloc_L1((K, N), dtype)
            C_l0c = T.alloc_L0C((M, N), accum_dtype)
            C_ub = T.alloc_ub((M_half, N), accum_dtype)

            with T.Scope("C"):
                # GM → L1
                T.copy(A, A_l1)
                T.copy(B, B_l1)
                T.set_flag("mte2", "m", 1)
                T.wait_flag("mte2", "m", 1)
                # L1 → GEMM → L0C (Cube computation)
                T.gemm_v0(A_l1, B_l1, C_l0c, init=True)
                T.set_flag("m", "fix", 2)
                T.wait_flag("m", "fix", 2)
                # L0C → UB (TMOV)
                T.copy_op.copy_cv_experiment(C_l0c, C_ub, T.copy_op.CopyCVMode.DualSplitM)
                T.set_cross_flag("FIX", 0)

            with T.Scope("V"):
                # L0C → UB
                T.wait_cross_flag(0, "MTE3")
                # UB → GM
                T.copy(C_ub, C[vid * M_half: (vid + 1) * M_half, :])
    return main


@only_A5
@pytest.mark.parametrize("target", ["pto"])
def test_copy_vc_experiment(target):
    M, N, K = 128, 128, 128
    _ub_to_l1_case(_copy_vc_experiment_kernel, M=M, N=N, K=K, target=target, expert=True)


@only_A5
@pytest.mark.parametrize("target", ["pto"])
def test_copy_cv_experiment(target):
    M, N, K = 128, 128, 128
    _l0c_to_ub_case(_copy_cv_experiment_kernel, M=M, N=N, K=K, target=target, expert=True)
