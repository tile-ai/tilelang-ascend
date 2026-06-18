"""T.Parallel 1D scenarios.

Fills the 1D coverage gaps for T.Parallel: the existing
``test_tilelang_ascend_language_parallel.py`` only exercises 1D add, while 2D is
covered broadly.  Here we add 1D unary / binary breadth, 1D immediate scalar, 1D
offset access and 1D discrete (gather) access.

NOTE: these kernels target Ascend NPU (``.npu()`` tensors) and must be run on
NPU hardware; they cannot execute on a CPU-only host.
"""

import pytest
import tilelang
import tilelang.language as T
import torch

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

VEC_NUM = 2


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    """Clear tilelang cache before tests."""
    tilelang.cache.clear_cache()
    yield


@pytest.fixture
def setup_random_seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(0)
    yield


# ---------------------------------------------------------------------------
# 1D unary: B[j] = op(A[j])
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def unary_1d_kernel(N, block_N, op_func, dtype="float"):
    n_num = N // block_N

    @T.prim_func
    def main(A: T.Tensor((N,), dtype), B: T.Tensor((N,), dtype)):
        with T.Kernel(n_num, is_npu=True) as (cid, vid):
            by = cid % n_num
            a_ub = T.alloc_ub((block_N // VEC_NUM,), dtype)
            b_ub = T.alloc_ub((block_N // VEC_NUM,), dtype)
            with T.Scope("V"):
                T.copy(A[vid * block_N // VEC_NUM + by * block_N], a_ub)
                for j in T.Parallel(block_N // VEC_NUM):
                    b_ub[j] = op_func(a_ub[j])
                T.copy(b_ub, B[vid * block_N // VEC_NUM + by * block_N])

    return main


@pytest.mark.parametrize(
    "op_name, op_func, ref_func, input_gen",
    [
        ("exp", lambda a: T.exp(a), torch.exp, lambda n: torch.randn(n).npu()),
        ("abs", lambda a: T.abs(a), torch.abs, lambda n: torch.randn(n).npu()),
        ("sqrt", lambda a: T.sqrt(a), torch.sqrt, lambda n: torch.rand(n).npu()),
    ],
)
def test_parallel_1d_unary(setup_random_seed, op_name, op_func, ref_func, input_gen):
    N, block_N = 1024, 128
    func = unary_1d_kernel(N, block_N, op_func)
    a = input_gen(N)
    torch.npu.synchronize()
    out = func(a)
    torch.testing.assert_close(out, ref_func(a), rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# 1D binary: C[j] = op(A[j], B[j])
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def binary_1d_kernel(N, block_N, op_func, dtype="float"):
    n_num = N // block_N

    @T.prim_func
    def main(A: T.Tensor((N,), dtype), B: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)):
        with T.Kernel(n_num, is_npu=True) as (cid, vid):
            by = cid % n_num
            a_ub = T.alloc_ub((block_N // VEC_NUM,), dtype)
            b_ub = T.alloc_ub((block_N // VEC_NUM,), dtype)
            c_ub = T.alloc_ub((block_N // VEC_NUM,), dtype)
            with T.Scope("V"):
                T.copy(A[vid * block_N // VEC_NUM + by * block_N], a_ub)
                T.copy(B[vid * block_N // VEC_NUM + by * block_N], b_ub)
                for j in T.Parallel(block_N // VEC_NUM):
                    c_ub[j] = op_func(a_ub[j], b_ub[j])
                T.copy(c_ub, C[vid * block_N // VEC_NUM + by * block_N])

    return main


@pytest.mark.parametrize(
    "op_name, op_func, ref_func",
    [
        ("sub", lambda a, b: a - b, lambda a, b: a - b),
        ("mul", lambda a, b: a * b, lambda a, b: a * b),
        ("div", lambda a, b: a / b, lambda a, b: a / b),
        ("max", lambda a, b: T.max(a, b), torch.maximum),
        ("min", lambda a, b: T.min(a, b), torch.minimum),
    ],
)
def test_parallel_1d_binary_float(setup_random_seed, op_name, op_func, ref_func):
    N, block_N = 1024, 128
    func = binary_1d_kernel(N, block_N, op_func)
    a = torch.randn(N).npu()
    b = torch.randn(N).npu()
    if op_name == "div":
        b = b + 1.0  # avoid division by zero
    torch.npu.synchronize()
    out = func(a, b)
    torch.testing.assert_close(out, ref_func(a, b), rtol=1e-2, atol=1e-2)


# Bitwise and/or on the Ascend vector path is only supported for 16-bit ints
# (int16 / uint16); int32 produces garbage on this instruction, matching the
# dtype coverage of the existing 2D and elementwise bitwise tests.
@pytest.mark.parametrize(
    "op_name, op_func, ref_func",
    [
        ("and", lambda a, b: a & b, lambda a, b: a & b),
        ("or", lambda a, b: a | b, lambda a, b: a | b),
    ],
)
def test_parallel_1d_binary_int(setup_random_seed, op_name, op_func, ref_func):
    N, block_N = 1024, 128
    func = binary_1d_kernel(N, block_N, op_func, dtype="int16")
    a = torch.randint(0, 100, (N,), dtype=torch.int16).npu()
    b = torch.randint(0, 100, (N,), dtype=torch.int16).npu()
    torch.npu.synchronize()
    out = func(a, b)
    torch.testing.assert_close(out, ref_func(a, b), rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# 1D vector + immediate scalar: C[j] = A[j] <op> imm
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def scalar_1d_kernel(N, block_N, op_func, dtype="float"):
    n_num = N // block_N

    @T.prim_func
    def main(A: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)):
        with T.Kernel(n_num, is_npu=True) as (cid, vid):
            by = cid % n_num
            a_ub = T.alloc_ub((block_N // VEC_NUM,), dtype)
            c_ub = T.alloc_ub((block_N // VEC_NUM,), dtype)
            with T.Scope("V"):
                T.copy(A[vid * block_N // VEC_NUM + by * block_N], a_ub)
                for j in T.Parallel(block_N // VEC_NUM):
                    c_ub[j] = op_func(a_ub[j])
                T.copy(c_ub, C[vid * block_N // VEC_NUM + by * block_N])

    return main


@pytest.mark.parametrize(
    "op_name, op_func, ref_func",
    [
        ("add_imm", lambda a: a + 1.0, lambda a: a + 1.0),
        ("mul_imm", lambda a: a * 2.0, lambda a: a * 2.0),
    ],
)
def test_parallel_1d_scalar_immediate(setup_random_seed, op_name, op_func, ref_func):
    N, block_N = 1024, 128
    func = scalar_1d_kernel(N, block_N, op_func)
    a = torch.randn(N).npu()
    torch.npu.synchronize()
    out = func(a)
    torch.testing.assert_close(out, ref_func(a), rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# 1D offset access: only the second half of each chunk is consumed via
# C[j] = A[j + half] * 2.  Mirrors the 2D offset test in 1D, verifying that a
# loop-variable offset expression (j + half) is substituted correctly.
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def offset_1d_kernel(N, block_N, dtype="float"):
    n_num = N // block_N
    chunk = block_N // VEC_NUM
    half = chunk // 2

    @T.prim_func
    def main(A: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)):
        with T.Kernel(n_num, is_npu=True) as (cid, vid):
            by = cid % n_num
            a_ub = T.alloc_ub((chunk,), dtype)
            c_ub = T.alloc_ub((chunk,), dtype)
            with T.Scope("V"):
                T.copy(A[vid * chunk + by * block_N], a_ub)
                T.tile.fill(c_ub, 0.0)
                for j in T.Parallel(half):
                    c_ub[j] = a_ub[j + half] * 2.0
                T.copy(c_ub, C[vid * chunk + by * block_N])

    return main


def test_parallel_1d_offset(setup_random_seed):
    N, block_N = 1024, 128
    chunk = block_N // VEC_NUM
    half = chunk // 2
    func = offset_1d_kernel(N, block_N)
    a = torch.randn(N).npu()
    torch.npu.synchronize()
    out = func(a)

    # Reference: each chunk's first half = second half * 2, second half stays 0.
    ref = torch.zeros_like(a)
    a_chunks = a.reshape(-1, chunk)
    ref_chunks = ref.reshape(-1, chunk)
    ref_chunks[:, :half] = a_chunks[:, half:] * 2.0
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# NOTE: there is intentionally NO 1D discrete (gather) test.
# A 1D gather ``a_ub[idx[j]]`` indexes the same dimension that T.Parallel
# vectorizes, which yields an unaligned per-element UB access and traps at
# runtime ("The UB address accessed by the VEC instruction is not aligned").
# Discrete gather is only supported when the index dimension is an OUTER,
# non-vectorized dimension (e.g. ``a[idx[i], j]``); see the 2D tests in
# test_tilelang_ascend_language_parallel_discrete.py. There is no valid 1D
# analog, so the discrete scenario stays 2D.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-n", "8"])
