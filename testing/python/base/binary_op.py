"""Binary operator test infrastructure (T.tile.add, .sub, .mul, ...)."""

import inspect
from typing import NamedTuple

import pytest
import torch

import tilelang
import tilelang.language as T

from base.common import DTYPE_MAP, DEFAULT_PASS_CONFIGS, assert_close_npu, make_test_data, skip_if_missing


def make_binary_kernel(tile_op):
    """Factory: tensor op tensor -> tensor."""

    def kernel(M, N, block_M, block_N, dtype="float"):
        m_num = M // block_M
        n_num = N // block_N
        VEC_NUM = 2

        @T.prim_func
        def main(
            A: T.Tensor((M, N), dtype),  # type: ignore
            B: T.Tensor((M, N), dtype),  # type: ignore
            C: T.Tensor((M, N), dtype),  # type: ignore
        ):
            with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
                bx = cid // n_num
                by = cid % n_num
                a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
                b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
                c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)
                tile_op(c_ub, a_ub, b_ub)
                T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

        return main

    return kernel


def make_1d_kernel(tile_op):
    """Factory: tensor op tensor -> tensor on 1D buffers."""

    def kernel(N, dtype="float"):
        @T.prim_func
        def main(
            A: T.Tensor((N,), dtype),  # type: ignore
            B: T.Tensor((N,), dtype),  # type: ignore
            C: T.Tensor((N,), dtype),  # type: ignore
        ):
            with T.Kernel(1, is_npu=True) as (cid, _):
                a_ub = T.alloc_ub((N,), dtype)
                b_ub = T.alloc_ub((N,), dtype)
                c_ub = T.alloc_ub((N,), dtype)
                T.copy(A, a_ub)
                T.copy(B, b_ub)
                tile_op(c_ub, a_ub, b_ub)
                T.copy(c_ub, C)

        return main

    return kernel


def make_scalar_kernel(tile_op):
    """Factory: tensor op scalar -> tensor."""

    def kernel(M, N, block_M, block_N, scalar, dtype="float"):
        m_num = M // block_M
        n_num = N // block_N

        @T.prim_func
        def main(
            A: T.Tensor((M, N), dtype),  # type: ignore
            B: T.Tensor((M, N), dtype),  # type: ignore
        ):
            with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
                bx = cid // n_num
                by = cid % n_num
                a_ub = T.alloc_ub((block_M, block_N), dtype)
                b_ub = T.alloc_ub((block_M, block_N), dtype)
                T.copy(A[bx * block_M, by * block_N], a_ub)
                tile_op(b_ub, a_ub, scalar)
                T.copy(b_ub, B[bx * block_M, by * block_N])

        return main

    return kernel


def make_buffload_kernel(tile_op):
    """Factory: tensor op 1D-buffer-element (BufferLoad) -> tensor."""

    def kernel(M, N, m, dtype="float"):
        @T.prim_func
        def main(
            A: T.Tensor((M, N), dtype),  # type: ignore
            S: T.Tensor((m,), dtype),  # type: ignore
            C: T.Tensor((M, N), dtype),  # type: ignore
        ):
            with T.Kernel(1, is_npu=True) as (cid, _):
                a_ub = T.alloc_ub((M, N), dtype)
                s_ub = T.alloc_ub((m,), dtype)
                c_ub = T.alloc_ub((M, N), dtype)
                T.copy(A, a_ub)
                T.copy(S, s_ub)
                tile_op(c_ub, a_ub, s_ub[3])
                T.copy(c_ub, C)

        return main

    return kernel


def make_row_slice_kernel(tile_op):
    """Factory: 2D whole-row slice regions with a per-row BufferLoad scalar."""

    def kernel(M, N, m, dtype="float"):
        @T.prim_func
        def main(
            A: T.Tensor((M, N), dtype),  # type: ignore
            S: T.Tensor((m,), dtype),  # type: ignore
            C: T.Tensor((M, N), dtype),  # type: ignore
        ):
            with T.Kernel(1, is_npu=True) as (cid, _):
                a_ub = T.alloc_ub((M, N), dtype)
                s_ub = T.alloc_ub((m,), dtype)
                c_ub = T.alloc_ub((M, N), dtype)
                T.copy(A, a_ub)
                T.copy(S, s_ub)
                for h_i in range(M):
                    tile_op(c_ub[h_i, :], a_ub[h_i, :], s_ub[h_i % m])
                T.copy(c_ub, C)

        return main

    return kernel


def make_inplace_kernel(tile_op):
    """Factory: dst = dst op src (in-place on UB buffer)."""

    def kernel(M, N, block_M, block_N, dtype="float"):
        m_num = M // block_M
        n_num = N // block_N
        VEC_NUM = 2

        @T.prim_func
        def main(
            A: T.Tensor((M, N), dtype),  # type: ignore
            B: T.Tensor((M, N), dtype),  # type: ignore
            C: T.Tensor((M, N), dtype),  # type: ignore
        ):
            with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
                bx = cid // n_num
                by = cid % n_num
                a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
                b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)
                tile_op(a_ub, a_ub, b_ub)
                T.copy(a_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

        return main

    return kernel


def make_src0_mismatch_kernel(tile_op):
    """dst(64) vs src0(128) -> traced-time Python assert (constraint 1)."""

    @T.prim_func
    def main(A: T.Tensor((128,), "float16"), B: T.Tensor((64,), "float16"), C: T.Tensor((64,), "float16")):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a_ub = T.alloc_ub((128,), "float16")
            b_ub = T.alloc_ub((64,), "float16")
            c_ub = T.alloc_ub((64,), "float16")
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            tile_op(c_ub, a_ub, b_ub)
            T.copy(c_ub, C)

    return main


def make_region_mismatch_kernel(tile_op):
    """src1 BufferRegion(32) vs dst(64) -> traced-time assert (constraint 2)."""

    @T.prim_func
    def main(A: T.Tensor((64,), "float16"), B: T.Tensor((128,), "float16"), C: T.Tensor((64,), "float16")):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a_ub = T.alloc_ub((64,), "float16")
            b_ub = T.alloc_ub((128,), "float16")
            c_ub = T.alloc_ub((64,), "float16")
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            tile_op(c_ub, a_ub, b_ub[0:32])
            T.copy(c_ub, C)

    return main


def make_buffer_mismatch_kernel(tile_op):
    """src1 Buffer(128) vs dst(64) -> NOT validated (constraint 3): only the
    first dst.size elements participate."""

    @T.prim_func
    def main(A: T.Tensor((64,), "float16"), B: T.Tensor((128,), "float16"), C: T.Tensor((64,), "float16")):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a_ub = T.alloc_ub((64,), "float16")
            b_ub = T.alloc_ub((128,), "float16")
            c_ub = T.alloc_ub((64,), "float16")
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            tile_op(c_ub, a_ub, b_ub)
            T.copy(c_ub, C)

    return main


def run_binary_op(kernel_factory, M, N, block_M, block_N, dtype, target, golden_fn):
    func = kernel_factory(M, N, block_M, block_N, dtype=dtype)
    compiled = tilelang.compile(func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target)
    a = make_test_data((M, N), dtype)
    b = make_test_data((M, N), dtype)
    torch.npu.synchronize()
    c = compiled(a, b)
    assert_close_npu(c, golden_fn(a, b), dtype)


def run_1d_op(kernel_factory, N, dtype, target, golden_fn):
    func = kernel_factory(N, dtype=dtype)
    compiled = tilelang.compile(func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target)
    a = make_test_data((N,), dtype)
    b = make_test_data((N,), dtype)
    torch.npu.synchronize()
    c = compiled(a, b)
    assert_close_npu(c, golden_fn(a, b), dtype)


def run_scalar_op(kernel_factory, M, N, block_M, block_N, scalar, dtype, target, golden_fn):
    func = kernel_factory(M, N, block_M, block_N, scalar, dtype)
    compiled = tilelang.compile(func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target)
    a = make_test_data((M, N), dtype)
    torch.npu.synchronize()
    b = compiled(a)
    expected = golden_fn(a, scalar, dtype)
    assert_close_npu(b, expected, dtype)


def run_buffload_op(kernel_factory, M, N, m, dtype, target, golden_fn):
    func = kernel_factory(M, N, m, dtype=dtype)
    compiled = tilelang.compile(func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target)
    torch.manual_seed(0)
    a = make_test_data((M, N), dtype, device="cpu")
    if dtype.startswith("float"):
        values = [1.5, 2.0, 0.5, 3.0, 4.0, -1.0, 0.25, 0.75]
    else:
        values = [1, 2, 3, 4, 5, 6, 7, 8]
    s = torch.tensor(values, dtype=DTYPE_MAP[dtype])[:m]
    torch.npu.synchronize()
    b = compiled(a.npu(), s.npu())
    torch.npu.synchronize()
    assert_close_npu(b.cpu(), golden_fn(a, s[3]), dtype)


def run_row_slice_op(kernel_factory, M, N, m, dtype, target, golden_fn):
    func = kernel_factory(M, N, m, dtype=dtype)
    compiled = tilelang.compile(func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target)
    torch.manual_seed(0)
    a = make_test_data((M, N), dtype, device="cpu")
    if dtype.startswith("float"):
        values = [1.5, 2.0, 0.5, 3.0, 4.0, -1.0, 0.25, 0.75]
    else:
        values = [1, 2, 3, 4, 5, 6, 7, 8]
    s = torch.tensor(values, dtype=DTYPE_MAP[dtype])[:m]
    torch.npu.synchronize()
    b = compiled(a.npu(), s.npu())
    torch.npu.synchronize()
    ref = torch.stack([golden_fn(a[h], s[h % m]) for h in range(M)])
    assert_close_npu(b.cpu(), ref, dtype)


class BinaryOpSpec:
    def __init__(
        self,
        name,
        tile_op,
        golden,
        supported_dtypes,
        boundary_dtypes=None,
        kernel_tensor=None,
        kernel_scalar=None,
        kernel_inplace=None,
        kernel_1d=None,
        kernel_buffload=None,
        kernel_row_slice=None,
        mismatch_kernels=None,
        low_priority_dtypes=None,
    ):
        self.name = name
        self.tile_op = tile_op
        self.golden = golden
        self.supported_dtypes = supported_dtypes
        self.boundary_dtypes = boundary_dtypes or ["float16", "float32"]
        self.low_priority_dtypes = list(low_priority_dtypes or [])
        # Mandatory variants: None -> default factory (all binary ops share
        # the same binary_op front-end).
        self.kernel_tensor = kernel_tensor or make_binary_kernel(self.tile_op)
        self.kernel_scalar = kernel_scalar or make_scalar_kernel(self.tile_op)
        self.kernel_inplace = kernel_inplace or make_inplace_kernel(self.tile_op)
        # Optional variants: None -> disabled (skip_if_missing skips them);
        # specs opt in by passing the factory explicitly.
        self.kernel_1d = kernel_1d or None
        self.kernel_buffload = kernel_buffload or None
        self.kernel_row_slice = kernel_row_slice or None
        # Size-validation kernels (constraint 1/2/3 checks); None -> disabled.
        self.mismatch_kernels = mismatch_kernels or None

    def scalar_golden(self, a, scalar, dtype):
        if dtype in ("int16", "int32"):
            scalar = int(scalar)
        return self.golden(a, torch.full_like(a, scalar))


@pytest.mark.usefixtures("disable_tilelang_cache", "random_seed")
class _BinaryOpCompile:
    op = None
    _dtype_source = "supported_dtypes"

    def test_compiles(self, dtype):
        skip_if_missing(self.op, "kernel_tensor")
        func = self.op.kernel_tensor(128, 128, 64, 64, dtype)
        compiled = tilelang.compile(func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target="ascendc")
        assert callable(compiled)

    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_compiles_both_targets(self, target):
        skip_if_missing(self.op, "kernel_tensor")
        func = self.op.kernel_tensor(128, 128, 64, 64, self.op.supported_dtypes[0])
        compiled = tilelang.compile(func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target)
        assert callable(compiled)

    def test_scalar_variant_compiles(self):
        skip_if_missing(self.op, "kernel_scalar")
        func = self.op.kernel_scalar(128, 128, 64, 64, 1.0, self.op.supported_dtypes[0])
        compiled = tilelang.compile(func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target="ascendc")
        assert callable(compiled)

    @pytest.mark.parametrize("shape", [(64, 64), (128, 256), (256, 128)])
    def test_various_shapes_compile(self, shape):
        skip_if_missing(self.op, "kernel_tensor")
        M, N = shape
        func = self.op.kernel_tensor(M, N, 64, 64, self.op.supported_dtypes[0])
        compiled = tilelang.compile(func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target="ascendc")
        assert callable(compiled)


@pytest.mark.usefixtures("disable_tilelang_cache", "random_seed")
class _BinaryOpE2E:
    op = None
    _dtype_source = "supported_dtypes"

    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_basic_1024x1024(self, dtype, target):
        skip_if_missing(self.op, "kernel_tensor")
        run_binary_op(
            self.op.kernel_tensor,
            1024,
            1024,
            128,
            128,
            dtype,
            target,
            golden_fn=self.op.golden,
        )

    @pytest.mark.parametrize(
        "M,N,block_M,block_N",
        [
            (256, 256, 64, 64),
            (512, 1024, 64, 128),
            (1024, 512, 128, 64),
            (2048, 2048, 128, 128),
        ],
    )
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_various_shapes(self, M, N, block_M, block_N, dtype, target):
        skip_if_missing(self.op, "kernel_tensor")
        run_binary_op(
            self.op.kernel_tensor,
            M,
            N,
            block_M,
            block_N,
            dtype,
            target,
            golden_fn=self.op.golden,
        )

    @pytest.mark.parametrize("M,N", [(100, 200), (107, 145), (255, 513)])
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_tail_blocks(self, M, N, dtype, target):
        """Non-multiple shapes: only full blocks participate (tail dropped).

        M/N are trimmed down to block multiples before launch, so this
        verifies the aligned prefix stays correct; the ragged tail is
        intentionally not covered by the grid.
        """
        skip_if_missing(self.op, "kernel_tensor")
        block_M = 64 if M >= 64 else 32
        block_N = 64 if N >= 64 else 32
        M_aligned = (M // block_M) * block_M
        N_aligned = (N // block_N) * block_N
        run_binary_op(
            self.op.kernel_tensor,
            M_aligned,
            N_aligned,
            block_M,
            block_N,
            dtype,
            target,
            golden_fn=self.op.golden,
        )

    @pytest.mark.parametrize("N", [256])
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_1d(self, N, dtype, target):
        skip_if_missing(self.op, "kernel_1d")
        run_1d_op(self.op.kernel_1d, N, dtype, target, golden_fn=self.op.golden)

    @pytest.mark.parametrize("scalar_val", [2.0])
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_tensor_op_scalar(self, scalar_val, dtype, target):
        skip_if_missing(self.op, "kernel_scalar")
        scalar = scalar_val if dtype not in ("int16", "int32") else int(scalar_val)
        run_scalar_op(
            self.op.kernel_scalar,
            1024,
            1024,
            128,
            128,
            scalar,
            dtype,
            target,
            golden_fn=self.op.scalar_golden,
        )

    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_buffload(self, dtype, target):
        skip_if_missing(self.op, "kernel_buffload")
        run_buffload_op(self.op.kernel_buffload, 64, 128, 8, dtype, target, golden_fn=self.op.golden)

    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_row_slice_2d(self, dtype, target):
        skip_if_missing(self.op, "kernel_row_slice")
        run_row_slice_op(self.op.kernel_row_slice, 4, 128, 4, dtype, target, golden_fn=self.op.golden)

    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_inplace(self, dtype, target):
        skip_if_missing(self.op, "kernel_inplace")
        func = self.op.kernel_inplace(1024, 1024, 128, 128, dtype)
        compiled = tilelang.compile(func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target)
        a = make_test_data((1024, 1024), dtype)
        b = make_test_data((1024, 1024), dtype)
        torch.npu.synchronize()
        result = compiled(a, b)
        assert_close_npu(result, self.op.golden(a, b), dtype)


@pytest.mark.usefixtures("disable_tilelang_cache", "random_seed")
class _BinaryOpBoundary:
    op = None
    _dtype_source = "boundary_dtypes"

    @pytest.mark.low_priority
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_large_values(self, dtype, target):
        skip_if_missing(self.op, "kernel_tensor")
        torch_dtype = DTYPE_MAP[dtype]
        if dtype == "float32":
            a = torch.full((256, 256), 1e30, dtype=torch_dtype, device="npu")
            b = torch.full((256, 256), 1e30, dtype=torch_dtype, device="npu")
        else:
            a = torch.full((256, 256), 60000.0, dtype=torch_dtype, device="npu")
            b = torch.full((256, 256), 1.0, dtype=torch_dtype, device="npu")
        func = self.op.kernel_tensor(256, 256, 64, 64, dtype)
        compiled = tilelang.compile(func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target)
        torch.npu.synchronize()
        c = compiled(a, b)
        assert c.shape == (256, 256)

    @pytest.mark.low_priority
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_zeros(self, dtype, target):
        skip_if_missing(self.op, "kernel_tensor")
        torch_dtype = DTYPE_MAP[dtype]
        a = torch.zeros(256, 256, dtype=torch_dtype, device="npu")
        b = torch.zeros(256, 256, dtype=torch_dtype, device="npu")
        func = self.op.kernel_tensor(256, 256, 64, 64, dtype)
        compiled = tilelang.compile(func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target)
        torch.npu.synchronize()
        c = compiled(a, b)
        assert_close_npu(c, torch.zeros_like(a), dtype)

    @pytest.mark.low_priority
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_negative_values(self, dtype, target):
        skip_if_missing(self.op, "kernel_tensor")
        torch_dtype = DTYPE_MAP[dtype]
        a = torch.full((256, 256), -5.0, dtype=torch_dtype, device="npu")
        b = torch.full((256, 256), 3.0, dtype=torch_dtype, device="npu")
        func = self.op.kernel_tensor(256, 256, 64, 64, dtype)
        compiled = tilelang.compile(func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target)
        torch.npu.synchronize()
        c = compiled(a, b)
        assert_close_npu(c, self.op.golden(a, b), dtype)

    @pytest.mark.low_priority
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_inf_input(self, target):
        skip_if_missing(self.op, "kernel_tensor")
        if "float16" not in self.op.boundary_dtypes:
            pytest.skip("float16 not in boundary_dtypes")
        a = torch.full((256, 256), float("inf"), dtype=torch.float16, device="npu")
        b = torch.ones(256, 256, dtype=torch.float16, device="npu")
        func = self.op.kernel_tensor(256, 256, 64, 64, "float16")
        compiled = tilelang.compile(func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target)
        torch.npu.synchronize()
        c = compiled(a, b)
        assert_close_npu(c, self.op.golden(a, b), "float16")

    @pytest.mark.low_priority
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_nan_input(self, target):
        skip_if_missing(self.op, "kernel_tensor")
        if "float16" not in self.op.boundary_dtypes:
            pytest.skip("float16 not in boundary_dtypes")
        a = torch.full((256, 256), float("nan"), dtype=torch.float16, device="npu")
        b = torch.ones(256, 256, dtype=torch.float16, device="npu")
        func = self.op.kernel_tensor(256, 256, 64, 64, "float16")
        compiled = tilelang.compile(func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target)
        torch.npu.synchronize()
        c = compiled(a, b)
        assert c.shape == (256, 256)
        assert torch.all(torch.isnan(c))

    @pytest.mark.low_priority
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_minimum_shape(self, dtype, target):
        skip_if_missing(self.op, "kernel_tensor")
        run_binary_op(
            self.op.kernel_tensor,
            64,
            64,
            64,
            64,
            dtype,
            target,
            golden_fn=self.op.golden,
        )

    @pytest.mark.low_priority
    def test_size_mismatch_src0(self):
        """dst vs src0 size mismatch raises at trace time (constraint 1)."""
        skip_if_missing(self.op, "mismatch_kernels")
        with pytest.raises((RuntimeError, AssertionError)):
            self.op.mismatch_kernels[0]()

    @pytest.mark.low_priority
    def test_size_mismatch_region(self):
        """src1 BufferRegion size mismatch raises at trace time (constraint 2)."""
        skip_if_missing(self.op, "mismatch_kernels")
        with pytest.raises((RuntimeError, AssertionError)):
            self.op.mismatch_kernels[1]()

    @pytest.mark.low_priority
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_buffer_size_unvalidated(self, target):
        """src1 Buffer size is NOT validated (constraint 3): only the first
        dst.size elements participate."""
        skip_if_missing(self.op, "mismatch_kernels")
        torch.manual_seed(0)
        a = torch.randn(64, dtype=torch.float16)
        b = torch.rand(128, dtype=torch.float16)
        func = self.op.mismatch_kernels[2]()
        compiled = tilelang.compile(func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target)
        out = compiled(a.npu(), b.npu())
        torch.npu.synchronize()
        torch.testing.assert_close(out.cpu().float(), self.op.golden(a, b[:64]).float(), rtol=1e-2, atol=1e-2)


class BinaryOpTestClasses(NamedTuple):
    Compile: type[_BinaryOpCompile]
    E2E: type[_BinaryOpE2E]
    Boundary: type[_BinaryOpBoundary]


def register_binary_op_tests(spec) -> BinaryOpTestClasses:
    """Create TestTile{Name}Compile / E2E / Boundary classes in the caller's module.

    Also creates ``_``-prefixed base classes and returns them as a
    :class:`BinaryOpTestClasses` tuple so callers can subclass them for
    override scenarios (e.g. extra parametrize decorators).
    """
    caller_globals = inspect.currentframe().f_back.f_globals
    prefix = f"TestTile{spec.name.capitalize()}"
    base_classes = []
    for suffix, base in [
        ("Compile", _BinaryOpCompile),
        ("E2E", _BinaryOpE2E),
        ("Boundary", _BinaryOpBoundary),
    ]:
        # Inject into globals (existing behavior, backward compatible)
        cls = type(f"{prefix}{suffix}", (base,), {"op": spec})
        caller_globals[cls.__name__] = cls
        # Create _-prefixed version for override scenarios
        base_cls = type(f"_{prefix}{suffix}", (base,), {"op": spec})
        base_classes.append(base_cls)
    return BinaryOpTestClasses(*base_classes)
