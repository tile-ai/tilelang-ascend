"""T.Parallel first-axis (row) broadcast regression tests.

A 1D UB operand indexed by the first (outer) T.Parallel axis used to be lowered
to a direct VEC operand whose UB address advances at element granularity
(sub-32B for fp32) between rows. AscendC vector instructions require 32B-aligned
UB addresses, so such a kernel trapped at first launch with a VEC UB alignment
aicore exception.

The failing family reduces to a first-Parallel-axis broadcast contraction
``dk_i = sum_j v[j] * A[j, i]`` regardless of how the broadcast is expressed
(bare Parallel loop, serial-row accumulation) and of the reduce direction.
The lowering now routes the row-broadcast 1D operand through a scalar read
(GetValue) + Muls/Addss (or falls back to a scalar serial loop for
non-commutative ops), which has no block-alignment requirement.

NOTE: these kernels target Ascend NPU and must be run on NPU hardware.
"""

import pytest
import tilelang
import tilelang.language as T
import torch

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_TAIL_MASK: True,
}


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    """Clear tilelang cache before tests."""
    tilelang.cache.clear_cache()
    yield


@pytest.fixture
def setup_random_seed():
    torch.manual_seed(0)
    yield


# ---------------------------------------------------------------------------
# dk_i = sum_j v[j] * A[j, i]: bare first-axis broadcast inside a 2D
# T.Parallel loop, contracted along the broadcast axis (reduce dim=0).
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def first_axis_bcast_mul_reduce0(DV: int, DK: int, dtype="float"):

    @T.prim_func
    def main(
        A: T.Tensor((DV, DK), dtype),
        V: T.Tensor((DV,), dtype),
        DK_OUT: T.Tensor((DK,), dtype),
    ):
        with T.Kernel(1, threads=1, is_npu=True):
            a_ub = T.alloc_ub((DV, DK), dtype)
            v_ub = T.alloc_ub((DV,), dtype)
            buf2_ub = T.alloc_ub((DV, DK), dtype)
            dk_ub = T.alloc_ub((DK,), dtype)
            T.copy(A[0, 0], a_ub)
            T.copy(V[0], v_ub)
            for j, i in T.Parallel(DV, DK):
                buf2_ub[j, i] = v_ub[j] * a_ub[j, i]
            T.reduce_sum(buf2_ub, dk_ub, dim=0)
            T.copy(dk_ub, DK_OUT[0])

    return main


# ---------------------------------------------------------------------------
# Same first-axis broadcast operand, but this time the 1D operand sits on the
# right and the contraction runs along the inner axis (reduce dim=1).
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def first_axis_bcast_mul_reduce1(DV: int, DK: int, dtype="float"):

    @T.prim_func
    def main(
        A: T.Tensor((DV, DK), dtype),
        V: T.Tensor((DV,), dtype),
        DK_OUT: T.Tensor((DV,), dtype),
    ):
        with T.Kernel(1, threads=1, is_npu=True):
            a_ub = T.alloc_ub((DV, DK), dtype)
            v_ub = T.alloc_ub((DV,), dtype)
            buf2_ub = T.alloc_ub((DV, DK), dtype)
            dk_ub = T.alloc_ub((DV,), dtype)
            T.copy(A[0, 0], a_ub)
            T.copy(V[0], v_ub)
            for j, i in T.Parallel(DV, DK):
                buf2_ub[j, i] = v_ub[j] * a_ub[j, i]
            T.reduce_sum(buf2_ub, dk_ub, dim=1)
            T.copy(dk_ub, DK_OUT[0])

    return main


# ---------------------------------------------------------------------------
# First-axis broadcast inside a serial outer loop
# (serial-row accumulation). The 1D operand is indexed by the serial var.
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def first_axis_bcast_serial_accum(DV: int, DK: int, dtype="float"):

    @T.prim_func
    def main(
        A: T.Tensor((DV, DK), dtype),
        V: T.Tensor((DV,), dtype),
        DK_OUT: T.Tensor((DK,), dtype),
    ):
        with T.Kernel(1, threads=1, is_npu=True):
            a_ub = T.alloc_ub((DV, DK), dtype)
            v_ub = T.alloc_ub((DV,), dtype)
            dk_ub = T.alloc_ub((DK,), dtype)
            T.copy(A[0, 0], a_ub)
            T.copy(V[0], v_ub)
            for i in T.Parallel(DK):
                dk_ub[i] = 0.0
            for jj in T.serial(DV):
                for i in T.Parallel(DK):
                    dk_ub[i] += v_ub[jj] * a_ub[jj, i]
            T.copy(dk_ub, DK_OUT[0])

    return main


# ---------------------------------------------------------------------------
# Non-commutative first-axis broadcast (Sub with the 1D operand on the left):
# must not trap either; the vectorizer falls back to a scalar serial loop and
# the result must stay correct.
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def first_axis_bcast_sub_reduce0(DV: int, DK: int, dtype="float"):

    @T.prim_func
    def main(
        A: T.Tensor((DV, DK), dtype),
        V: T.Tensor((DV,), dtype),
        DK_OUT: T.Tensor((DK,), dtype),
    ):
        with T.Kernel(1, threads=1, is_npu=True):
            a_ub = T.alloc_ub((DV, DK), dtype)
            v_ub = T.alloc_ub((DV,), dtype)
            buf2_ub = T.alloc_ub((DV, DK), dtype)
            dk_ub = T.alloc_ub((DK,), dtype)
            T.copy(A[0, 0], a_ub)
            T.copy(V[0], v_ub)
            for j, i in T.Parallel(DV, DK):
                buf2_ub[j, i] = v_ub[j] - a_ub[j, i]
            T.reduce_sum(buf2_ub, dk_ub, dim=0)
            T.copy(dk_ub, DK_OUT[0])

    return main


def _make_inputs(DV, DK, dtype):
    torch_dtype = torch.float16 if dtype == "float16" else torch.float32
    a = torch.randn(DV, DK).to(torch_dtype).npu()
    v = torch.randn(DV).to(torch_dtype).npu()
    return a, v


def _assert_close(out, ref):
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)


def _ref_mul_reduce0(a, v):
    return (v.unsqueeze(-1).float() * a.float()).sum(0)


def _ref_mul_reduce1(a, v):
    return (v.unsqueeze(-1).float() * a.float()).sum(1)


def _ref_sub_reduce0(a, v):
    return (v.unsqueeze(-1).float() - a.float()).sum(0)


@pytest.mark.parametrize("dtype", ["float", "float16"])
def test_parallel_first_axis_bcast_mul_reduce0(setup_random_seed, dtype):
    DV, DK = 16, 32
    func = first_axis_bcast_mul_reduce0(DV, DK, dtype)
    a, v = _make_inputs(DV, DK, dtype)
    torch.npu.synchronize()
    out = func(a, v)
    ref = _ref_mul_reduce0(a, v)
    _assert_close(out, ref)


@pytest.mark.parametrize("dtype", ["float", "float16"])
def test_parallel_first_axis_bcast_mul_reduce1(setup_random_seed, dtype):
    DV, DK = 16, 32
    func = first_axis_bcast_mul_reduce1(DV, DK, dtype)
    a, v = _make_inputs(DV, DK, dtype)
    torch.npu.synchronize()
    out = func(a, v)
    ref = _ref_mul_reduce1(a, v)
    _assert_close(out, ref)


@pytest.mark.parametrize("dtype", ["float", "float16"])
def test_parallel_first_axis_bcast_serial_accum(setup_random_seed, dtype):
    DV, DK = 16, 32
    func = first_axis_bcast_serial_accum(DV, DK, dtype)
    a, v = _make_inputs(DV, DK, dtype)
    torch.npu.synchronize()
    out = func(a, v)
    ref = _ref_mul_reduce0(a, v)
    _assert_close(out, ref)


@pytest.mark.parametrize("dtype", ["float", "float16"])
def test_parallel_first_axis_bcast_sub_reduce0(setup_random_seed, dtype):
    if dtype == "float16":
        pytest.skip(
            "fp16 Sub with a first-axis-broadcast operand takes the scalar "
            "serial fallback, and bisheng rejects scalar half arithmetic in "
            "aicore functions ('half precision operation is not allowed')"
        )
    DV, DK = 16, 32
    func = first_axis_bcast_sub_reduce0(DV, DK, dtype)
    a, v = _make_inputs(DV, DK, dtype)
    torch.npu.synchronize()
    out = func(a, v)
    ref = _ref_sub_reduce0(a, v)
    _assert_close(out, ref)
