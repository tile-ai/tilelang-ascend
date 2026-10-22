"""Unary operator test infrastructure (T.tile.exp, .abs, .sqrt, ...)."""

import inspect
from functools import cached_property
from typing import NamedTuple

import pytest
import torch

import tilelang
import tilelang.language as T

from base.common import DTYPE_MAP, DEFAULT_PASS_CONFIGS, assert_close_npu, make_test_data, skip_if_missing


def make_unary_kernel(tile_op):
    """Factory: op(src) -> dst."""

    def kernel(M, N, block_M, block_N, dtype="float"):
        m_num = M // block_M
        n_num = N // block_N
        VEC_NUM = 2

        @T.prim_func
        def main(
            A: T.Tensor((M, N), dtype),  # type: ignore
            B: T.Tensor((M, N), dtype),  # type: ignore
        ):
            with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
                bx = cid // n_num
                by = cid % n_num
                a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
                b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                tile_op(b_ub, a_ub)
                T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

        return main

    return kernel


def run_unary_op(kernel_factory, M, N, block_M, block_N, dtype, target, golden_fn):
    func = kernel_factory(M, N, block_M, block_N, dtype=dtype)
    compiled = tilelang.compile(func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target)
    a = make_test_data((M, N), dtype)
    torch.npu.synchronize()
    b = compiled(a)
    assert_close_npu(b, golden_fn(a), dtype)


class UnaryOpSpec:
    def __init__(self, name, tile_op, golden, supported_dtypes, boundary_dtypes=None, kernel_tensor=None):
        self.name = name
        self.tile_op = tile_op
        self.golden = golden
        self.supported_dtypes = supported_dtypes
        self.boundary_dtypes = boundary_dtypes or ["float16", "float32"]
        if kernel_tensor is not None:
            self.__dict__["kernel_tensor"] = kernel_tensor

    @cached_property
    def kernel_tensor(self):
        return make_unary_kernel(self.tile_op)


@pytest.mark.usefixtures("disable_tilelang_cache", "random_seed")
class _UnaryOpCompile:
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

    @pytest.mark.parametrize("shape", [(64, 64), (128, 256), (256, 128)])
    def test_various_shapes_compile(self, shape):
        skip_if_missing(self.op, "kernel_tensor")
        M, N = shape
        func = self.op.kernel_tensor(M, N, 64, 64, self.op.supported_dtypes[0])
        compiled = tilelang.compile(func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target="ascendc")
        assert callable(compiled)


@pytest.mark.usefixtures("disable_tilelang_cache", "random_seed")
class _UnaryOpE2E:
    op = None
    _dtype_source = "supported_dtypes"

    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_basic_1024x1024(self, dtype, target):
        skip_if_missing(self.op, "kernel_tensor")
        run_unary_op(
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
        run_unary_op(
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
    def test_non_aligned_shapes(self, M, N, dtype, target):
        skip_if_missing(self.op, "kernel_tensor")
        block_M = 64 if M >= 64 else 32
        block_N = 64 if N >= 64 else 32
        M_aligned = (M // block_M) * block_M
        N_aligned = (N // block_N) * block_N
        run_unary_op(
            self.op.kernel_tensor,
            M_aligned,
            N_aligned,
            block_M,
            block_N,
            dtype,
            target,
            golden_fn=self.op.golden,
        )


@pytest.mark.usefixtures("disable_tilelang_cache", "random_seed")
class _UnaryOpBoundary:
    op = None
    _dtype_source = "boundary_dtypes"

    @pytest.mark.low_priority
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_large_values(self, dtype, target):
        skip_if_missing(self.op, "kernel_tensor")
        torch_dtype = DTYPE_MAP[dtype]
        if dtype == "float32":
            a = torch.full((256, 256), 1e30, dtype=torch_dtype, device="npu")
        else:
            a = torch.full((256, 256), 60000.0, dtype=torch_dtype, device="npu")
        func = self.op.kernel_tensor(256, 256, 64, 64, dtype)
        compiled = tilelang.compile(func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target)
        torch.npu.synchronize()
        b = compiled(a)
        assert b.shape == (256, 256)

    @pytest.mark.low_priority
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_zeros(self, dtype, target):
        skip_if_missing(self.op, "kernel_tensor")
        torch_dtype = DTYPE_MAP[dtype]
        a = torch.zeros(256, 256, dtype=torch_dtype, device="npu")
        func = self.op.kernel_tensor(256, 256, 64, 64, dtype)
        compiled = tilelang.compile(func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target)
        torch.npu.synchronize()
        b = compiled(a)
        assert_close_npu(b, self.op.golden(a), dtype)

    @pytest.mark.low_priority
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_negative_values(self, dtype, target):
        skip_if_missing(self.op, "kernel_tensor")
        torch_dtype = DTYPE_MAP[dtype]
        a = torch.full((256, 256), -5.0, dtype=torch_dtype, device="npu")
        func = self.op.kernel_tensor(256, 256, 64, 64, dtype)
        compiled = tilelang.compile(func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target)
        torch.npu.synchronize()
        b = compiled(a)
        assert_close_npu(b, self.op.golden(a), dtype)

    @pytest.mark.low_priority
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_inf_input(self, target):
        skip_if_missing(self.op, "kernel_tensor")
        if "float16" not in self.op.boundary_dtypes:
            pytest.skip("float16 not in boundary_dtypes")
        a = torch.full((256, 256), float("inf"), dtype=torch.float16, device="npu")
        func = self.op.kernel_tensor(256, 256, 64, 64, "float16")
        compiled = tilelang.compile(func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target)
        torch.npu.synchronize()
        b = compiled(a)
        assert b.shape == (256, 256)
        assert torch.all(torch.isinf(b))

    @pytest.mark.low_priority
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_nan_input(self, target):
        skip_if_missing(self.op, "kernel_tensor")
        if "float16" not in self.op.boundary_dtypes:
            pytest.skip("float16 not in boundary_dtypes")
        a = torch.full((256, 256), float("nan"), dtype=torch.float16, device="npu")
        func = self.op.kernel_tensor(256, 256, 64, 64, "float16")
        compiled = tilelang.compile(func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target)
        torch.npu.synchronize()
        b = compiled(a)
        assert b.shape == (256, 256)
        assert torch.all(torch.isnan(b))

    @pytest.mark.low_priority
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_minimum_shape(self, dtype, target):
        skip_if_missing(self.op, "kernel_tensor")
        run_unary_op(
            self.op.kernel_tensor,
            64,
            64,
            64,
            64,
            dtype,
            target,
            golden_fn=self.op.golden,
        )


class UnaryOpTestClasses(NamedTuple):
    Compile: type[_UnaryOpCompile]
    E2E: type[_UnaryOpE2E]
    Boundary: type[_UnaryOpBoundary]


def register_unary_op_tests(spec) -> UnaryOpTestClasses:
    """Create TestTile{Name}Compile / E2E / Boundary classes in the caller's module.

    Also creates ``_``-prefixed base classes and returns them as a
    :class:`UnaryOpTestClasses` tuple so callers can subclass them for
    override scenarios (e.g. extra parametrize decorators).
    """
    caller_globals = inspect.currentframe().f_back.f_globals
    prefix = f"TestTile{spec.name.capitalize()}"
    base_classes = []
    for suffix, base in [
        ("Compile", _UnaryOpCompile),
        ("E2E", _UnaryOpE2E),
        ("Boundary", _UnaryOpBoundary),
    ]:
        cls = type(f"{prefix}{suffix}", (base,), {"op": spec})
        caller_globals[cls.__name__] = cls
        base_cls = type(f"_{prefix}{suffix}", (base,), {"op": spec})
        base_classes.append(base_cls)
    return UnaryOpTestClasses(*base_classes)
