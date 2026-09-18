"""T.tile.pow test suite.

Element-wise power: dst[i] = src0[i] ** src1[i] (tensor-tensor only).

Developer mode via DEFAULT_PASS_CONFIGS (auto sync + memory planning).

Verified behavior on A2:
- Dtype support differs by backend: ascendc accepts float16/float32/int32
  (C++ check in allocate_tmp_buffer.cc: "Pow only supports
  float16/float32/int32"); pto's log/exp path accepts float16/float32 only
  (enable_if<is_float_or_half>). int16 fails on both.
- AscendC converts float operands to float32 internally: exact for float32,
  float16 matches torch within 1 ulp (max_rel = 0 measured). int32 gives
  exact integer results for non-negative bases with powers representable in
  int32.
- PTO computes via log2(x) * y, exp2: float32 exact, float16 approximate
  (max_rel ~3.6e-3), and it modifies src0 in place (its log is written back),
  so src0 must not be reused after a pto call.
- PTO log/exp path yields nan for 0^0, negative bases, and base=1 with
  exponent nan, while AscendC follows IEEE semantics (0^0 = 1, etc.).
- No size assertion: mismatched dst/src0/src1 sizes run silently on ascendc
  (undefined behavior) and fail to compile on pto (tile-shape mismatch).
  Callers must guarantee equal element counts.

The generic binary framework uses randn data which trips the pto nan cases,
so this file is self-contained instead of registered via BinaryOpSpec.

Process isolation: kernels are tiled so per-core UB stays below capacity;
repeated ascendc pow compiles in one process can still crash the
AscendMemoryPlanning pass (pre-existing compiler bug), so every test runs in
its own forked process.
"""

import argparse

import pytest
import torch

import tilelang
import tilelang.language as T

from base import DEFAULT_PASS_CONFIGS
from base.common import DTYPE_MAP, assert_close_npu

pytestmark = pytest.mark.forked

POW_TOL = {"rtol": 1e-2, "atol": 1e-2}

DTYPE_PARAMS = ["float16", pytest.param("float32", marks=pytest.mark.low_priority)]
TARGET_PARAMS = ["ascendc", pytest.param("pto", marks=pytest.mark.low_priority)]


def make_pow_kernel(dtype, M, N, block_M=64, block_N=64):
    """Tiled tensor-tensor power kernel (whole-row blocks, VEC_NUM=2)."""

    def kernel(M, N, block_M, block_N, dtype="float"):
        m_num = M // block_M
        n_num = N // block_N
        VEC_NUM = 2

        @T.prim_func
        def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
                bx = cid // n_num
                by = cid % n_num
                a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
                b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
                c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)
                T.tile.pow(c_ub, a_ub, b_ub)
                T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

        return main

    return kernel(M, N, block_M, block_N, dtype)


def make_pow_1d_kernel(dtype, N):
    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((N,), dtype)
            b_ub = T.alloc_ub((N,), dtype)
            c_ub = T.alloc_ub((N,), dtype)
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            T.tile.pow(c_ub, a_ub, b_ub)
            T.copy(c_ub, C)

    return main


def make_pow_inplace_kernel(dtype, M, N, block_M=64, block_N=64):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)
            T.tile.pow(a_ub, a_ub, b_ub)  # dst == src0
            T.copy(a_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def make_pow_src0_preserved_kernel(dtype, M, N, block_M=64, block_N=64):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
        D: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)
            T.tile.pow(c_ub, a_ub, b_ub)
            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])
            T.copy(a_ub, D[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def positive_data(shape, dtype, device="npu"):
    """Bases in [0.5, 8], exponents in [-4, 4]: keeps both backends finite."""
    tdtype = DTYPE_MAP[dtype]
    a = (torch.rand(shape, dtype=tdtype, device=device) * 7.5 + 0.5).to(tdtype)
    b = (torch.rand(shape, dtype=tdtype, device=device) * 8.0 - 4.0).to(tdtype)
    return a, b


def run_pow(kernel, M, N, dtype, target, **tol):
    compiled = tilelang.compile(kernel, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target)
    a, b = positive_data((M, N), dtype)
    torch.npu.synchronize()
    c = compiled(a, b)
    merged = {**POW_TOL, **tol}
    assert_close_npu(c, torch.pow(a, b), dtype, **merged)
    return c


@pytest.mark.compile_time
@pytest.mark.usefixtures("disable_tilelang_cache", "random_seed")
class TestPowCompile:
    @pytest.mark.l0
    @pytest.mark.parametrize("dtype", DTYPE_PARAMS)
    @pytest.mark.parametrize("target", TARGET_PARAMS)
    def test_compiles(self, dtype, target):
        compiled = tilelang.compile(
            make_pow_kernel(dtype, 128, 128, 64, 64),
            out_idx=[-1],
            pass_configs=DEFAULT_PASS_CONFIGS,
            target=target,
        )
        assert callable(compiled)

    @pytest.mark.l0
    @pytest.mark.low_priority
    @pytest.mark.parametrize("dtype", ["int16", "int32"])
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_int_dtype_compile_fails(self, dtype, target):
        """int16 never compiles; int32 compiles on ascendc only.

        ascendc accepts float16/float32/int32 (checker in
        allocate_tmp_buffer.cc); pto's enable_if<is_float_or_half> rejects
        every integer dtype. The int32-ascendc combination therefore asserts
        a successful compile, every other combination asserts failure.
        """

        @T.prim_func
        def int_kernel(
            A: T.Tensor((64, 64), dtype),
            B: T.Tensor((64, 64), dtype),
            C: T.Tensor((64, 64), dtype),
        ):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                a_ub = T.alloc_ub((64, 64), dtype)
                b_ub = T.alloc_ub((64, 64), dtype)
                c_ub = T.alloc_ub((64, 64), dtype)
                T.copy(A, a_ub)
                T.copy(B, b_ub)
                T.tile.pow(c_ub, a_ub, b_ub)
                T.copy(c_ub, C)

        if dtype == "int32" and target == "ascendc":
            compiled = tilelang.compile(int_kernel, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target)
            assert callable(compiled)
        else:
            with pytest.raises(Exception, match=r"Compilation Failed|Pow only supports"):
                tilelang.compile(int_kernel, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target)

    @pytest.mark.l0
    def test_int32_ascendc(self):
        """int32 tensor-tensor power works on ascendc (positive bases).

        The int32 path uses a float32 log/exp computation, so results may be
        off by 1 for powers not exactly representable (e.g. 7^3); accept
        atol=1 and additionally require a correct sign/shape.
        """

        compiled = tilelang.compile(
            make_pow_kernel("int32", 128, 128, 64, 64),
            out_idx=[-1],
            pass_configs=DEFAULT_PASS_CONFIGS,
            target="ascendc",
        )
        a = torch.randint(1, 9, (128, 128), dtype=torch.int32, device="npu")
        b = torch.randint(0, 4, (128, 128), dtype=torch.int32, device="npu")
        c = compiled(a, b)
        torch.npu.synchronize()
        golden = torch.pow(a.to(torch.float64), b.to(torch.float64)).to(torch.int32)
        torch.testing.assert_close(c, golden, rtol=0, atol=1)
        assert torch.all(c >= 0)


@pytest.mark.usefixtures("disable_tilelang_cache", "random_seed")
class TestPowE2E:
    @pytest.mark.l0
    @pytest.mark.parametrize("dtype", DTYPE_PARAMS)
    @pytest.mark.parametrize("target", TARGET_PARAMS)
    def test_basic_1024x1024(self, dtype, target):
        run_pow(make_pow_kernel(dtype, 1024, 1024, 128, 128), 1024, 1024, dtype, target)

    @pytest.mark.l1
    @pytest.mark.parametrize("M,N", [(512, 1024)])
    @pytest.mark.parametrize("dtype", DTYPE_PARAMS)
    @pytest.mark.parametrize("target", TARGET_PARAMS)
    @pytest.mark.low_priority
    def test_various_shapes(self, M, N, dtype, target):
        run_pow(make_pow_kernel(dtype, M, N, 64, 64), M, N, dtype, target)

    @pytest.mark.l1
    @pytest.mark.parametrize("dtype", DTYPE_PARAMS)
    @pytest.mark.parametrize("target", TARGET_PARAMS)
    @pytest.mark.low_priority
    def test_1d(self, dtype, target):
        a, b = positive_data((256,), dtype)
        compiled = tilelang.compile(make_pow_1d_kernel(dtype, 256), out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target)
        c = compiled(a, b)
        torch.npu.synchronize()
        assert_close_npu(c, torch.pow(a, b), dtype, **POW_TOL)

    @pytest.mark.l1
    @pytest.mark.parametrize("dtype", DTYPE_PARAMS)
    @pytest.mark.parametrize("target", TARGET_PARAMS)
    @pytest.mark.low_priority
    def test_row_slice_2d(self, dtype, target):
        """Whole-row slices (BufferRegion) participate in the power.

        Only the sliced rows are computed; the rest of the destination buffer
        is left untouched, so the assertion covers the computed rows only.
        """

        @T.prim_func
        def slice_kernel(
            A: T.Tensor((128, 256), dtype),
            B: T.Tensor((128, 256), dtype),
            C: T.Tensor((128, 256), dtype),
        ):
            with T.Kernel(2, is_npu=True) as (cid, vid):
                by = cid
                a_ub = T.alloc_ub((32, 256), dtype)
                b_ub = T.alloc_ub((32, 256), dtype)
                c_ub = T.alloc_ub((32, 256), dtype)
                T.copy(A[by * 32, 0], a_ub)
                T.copy(B[by * 32, 0], b_ub)
                T.tile.pow(c_ub[0:16, :], a_ub[0:16, :], b_ub[0:16, :])
                T.copy(c_ub, C[by * 32, 0])

        compiled = tilelang.compile(slice_kernel, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target)
        a, b = positive_data((128, 256), dtype)
        c = compiled(a, b)
        torch.npu.synchronize()
        g = torch.pow(a, b)
        assert_close_npu(c[0:16, :], g[0:16, :], dtype, **POW_TOL)
        assert_close_npu(c[32:48, :], g[32:48, :], dtype, **POW_TOL)


@pytest.mark.usefixtures("disable_tilelang_cache", "random_seed")
class TestPowSpecialValues:
    @pytest.mark.l2
    @pytest.mark.parametrize("dtype", ["float16", "float32"])
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    @pytest.mark.low_priority
    def test_integer_powers(self, dtype, target):
        """2^3 is exactly representable; both backends match torch."""

        @T.prim_func
        def kernel(
            A: T.Tensor((64, 64), dtype),
            B: T.Tensor((64, 64), dtype),
            C: T.Tensor((64, 64), dtype),
        ):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                a_ub = T.alloc_ub((64, 64), dtype)
                b_ub = T.alloc_ub((64, 64), dtype)
                c_ub = T.alloc_ub((64, 64), dtype)
                T.copy(A, a_ub)
                T.copy(B, b_ub)
                T.tile.pow(c_ub, a_ub, b_ub)
                T.copy(c_ub, C)

        compiled = tilelang.compile(kernel, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target)
        a = torch.full((64, 64), 2.0, dtype=DTYPE_MAP[dtype], device="npu")
        b = torch.full((64, 64), 3.0, dtype=DTYPE_MAP[dtype], device="npu")
        c = compiled(a, b)
        torch.npu.synchronize()
        assert_close_npu(c, torch.pow(a, b), dtype, rtol=1e-2, atol=1e-2)

    def _constant_kernel(self, dtype):
        @T.prim_func
        def kernel(
            A: T.Tensor((64, 64), dtype),
            B: T.Tensor((64, 64), dtype),
            C: T.Tensor((64, 64), dtype),
        ):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                a_ub = T.alloc_ub((64, 64), dtype)
                b_ub = T.alloc_ub((64, 64), dtype)
                c_ub = T.alloc_ub((64, 64), dtype)
                T.copy(A, a_ub)
                T.copy(B, b_ub)
                T.tile.pow(c_ub, a_ub, b_ub)
                T.copy(c_ub, C)

        return kernel

    @pytest.mark.l2
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    @pytest.mark.low_priority
    def test_zero_base_zero_exponent(self, target):
        """0^0: ascendc follows IEEE (1.0); pto's log path yields nan."""

        compiled = tilelang.compile(self._constant_kernel("float16"), out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target)
        a = torch.zeros(64, 64, dtype=torch.float16, device="npu")
        b = torch.zeros(64, 64, dtype=torch.float16, device="npu")
        c = compiled(a, b)
        torch.npu.synchronize()
        if target == "ascendc":
            assert_close_npu(c, torch.pow(a, b), "float16")
        else:
            assert torch.all(torch.isnan(c))

    @pytest.mark.l2
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    @pytest.mark.low_priority
    def test_negative_base(self, target):
        """(-2)^3 = -8 on ascendc; pto's log path yields nan."""

        compiled = tilelang.compile(self._constant_kernel("float16"), out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target)
        a = torch.full((64, 64), -2.0, dtype=torch.float16, device="npu")
        b = torch.full((64, 64), 3.0, dtype=torch.float16, device="npu")
        c = compiled(a, b)
        torch.npu.synchronize()
        if target == "ascendc":
            assert_close_npu(c, torch.pow(a, b), "float16")
        else:
            assert torch.all(torch.isnan(c))


@pytest.mark.usefixtures("disable_tilelang_cache", "random_seed")
class TestPowAliasing:
    @pytest.mark.l2
    @pytest.mark.parametrize("dtype", ["float16", "float32"])
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    @pytest.mark.low_priority
    def test_inplace_dst_is_src0(self, dtype, target):
        run_pow(make_pow_inplace_kernel(dtype, 256, 256, 64, 64), 256, 256, dtype, target)

    @pytest.mark.l2
    @pytest.mark.parametrize(
        "target",
        [
            "ascendc",
            pytest.param("pto", marks=pytest.mark.xfail(strict=False, reason="pto pow modifies src0 in place")),
        ],
    )
    @pytest.mark.low_priority
    def test_src0_preserved(self, target):
        """AscendC never touches src0 (isReuseSource=false); pto writes its
        log back into src0, so a post-call read of src0 differs."""

        compiled = tilelang.compile(
            make_pow_src0_preserved_kernel("float16", 128, 128, 64, 64),
            out_idx=[2, 3],
            pass_configs=DEFAULT_PASS_CONFIGS,
            target=target,
        )
        a, b = positive_data((128, 128), "float16")
        c, d = compiled(a, b)
        torch.npu.synchronize()
        assert_close_npu(c, torch.pow(a, b), "float16", **POW_TOL)
        torch.testing.assert_close(d, a, rtol=0, atol=0)


@pytest.mark.usefixtures("disable_tilelang_cache", "random_seed")
class TestPowSizeConstraint:
    @pytest.mark.l2
    @pytest.mark.low_priority
    def test_ascendc_silently_runs_on_mismatched_sizes(self):
        """No front-end size assertion: ascendc computes with no error."""

        @T.prim_func
        def mm(
            A: T.Tensor((64,), "float16"),
            B: T.Tensor((128,), "float16"),
            C: T.Tensor((128,), "float16"),
        ):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                a_ub = T.alloc_ub((64,), "float16")
                b_ub = T.alloc_ub((128,), "float16")
                c_ub = T.alloc_ub((128,), "float16")
                T.copy(A, a_ub)
                T.copy(B, b_ub)
                T.tile.pow(c_ub, a_ub, b_ub)
                T.copy(c_ub, C)

        compiled = tilelang.compile(mm, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target="ascendc")
        a = torch.randn(64, dtype=torch.float16, device="npu").abs() + 1
        b = torch.randn(128, dtype=torch.float16, device="npu").abs() + 1
        compiled(a, b)
        torch.npu.synchronize()

    @pytest.mark.l2
    @pytest.mark.low_priority
    def test_pto_compile_fails_on_mismatched_sizes(self):
        """PTO tile shape must match; mismatched extents fail to compile."""

        @T.prim_func
        def mm(
            A: T.Tensor((64,), "float16"),
            B: T.Tensor((128,), "float16"),
            C: T.Tensor((128,), "float16"),
        ):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                a_ub = T.alloc_ub((64,), "float16")
                b_ub = T.alloc_ub((128,), "float16")
                c_ub = T.alloc_ub((128,), "float16")
                T.copy(A, a_ub)
                T.copy(B, b_ub)
                T.tile.pow(c_ub, a_ub, b_ub)
                T.copy(c_ub, C)

        with pytest.raises(Exception, match="Compilation Failed"):
            tilelang.compile(mm, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target="pto")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run T.tile.pow tests.")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--target", default="ascendc", choices=["ascendc", "pto"])
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=1024)
    args = parser.parse_args()
    run_pow(make_pow_kernel(args.dtype, args.M, args.N, 128, 128), args.M, args.N, args.dtype, args.target)
