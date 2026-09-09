"""
T.tile.broadcast dtype coverage tests.

Existing coverage in test_tilelang_ascend_language_explicit_tmp.py:
- float32 x ascendc (2D axis=0, explicit tmp runtime)
- uint8 x ascendc (workspace policy, codegen only)
- float32 x ascendc (zero workspace codegen)

This file supplements with:
1. Full dtype coverage for 2D axis=0 (src=(1,N) -> dst=(M,N))
2. 1D-to-2D auto-infer (axis=None)
3. 1D axis=0 (ascendc only; pto not supported)
4. Exception boundary tests (dtype mismatch, shape mismatch, invalid axis, 1D uninferable)

Known limitations (documented in docs/api_docs/T.tile.broadcast.md):
- 2D axis=1 (src=(M,1) -> dst=(M,N)) when M>=2: not supported (CANN BrcLast bug)
- 1D axis=0 on pto: not supported (pto codegen bug)
"""

import pytest
import tilelang
import tilelang.language as T
import torch


@pytest.fixture(scope="module", autouse=True)
def _disable_cache():
    tilelang.disable_cache()
    yield
    tilelang.enable_cache()


PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "int16": torch.int16,
    "uint16": torch.uint16,
    "int32": torch.int32,
    "uint32": torch.uint32,
}


def _gen_input(dtype, shape):
    torch_dtype = TORCH_DTYPE_MAP[dtype]
    n = 1
    for s in shape:
        n *= s
    if dtype in ("float16", "float32", "bfloat16"):
        a = torch.arange(n, dtype=torch.float32).to(torch_dtype)
    else:
        high = min(n, 100)
        if dtype in ("uint16", "uint32"):
            a = torch.randint(0, high, shape, dtype=torch.int32).to(torch_dtype)
        else:
            a = torch.randint(0, high, shape, dtype=torch_dtype)
    return a.npu()


# ---------------------------------------------------------------------------
# Functional: 2D axis=0 (src=(1,N) -> dst=(M,N))
# ---------------------------------------------------------------------------


def _make_2d_axis0(dtype, M=8, N=64):
    @T.prim_func
    def main(
        A: T.Tensor((1, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (_, vid):
            src_ub = T.alloc_ub((1, N), dtype)
            dst_ub = T.alloc_ub((M, N), dtype)
            if vid == 0:
                T.copy(A, src_ub)
                T.tile.broadcast(dst_ub, src_ub, axis=0)
                T.copy(dst_ub, B)

    return main


@pytest.mark.parametrize(
    "dtype",
    [
        "float32",
        pytest.param("float16", marks=pytest.mark.low_priority),
        pytest.param("int8", marks=pytest.mark.low_priority),
        pytest.param("uint8", marks=pytest.mark.low_priority),
        pytest.param("int16", marks=pytest.mark.low_priority),
        pytest.param("uint16", marks=pytest.mark.low_priority),
        pytest.param("bfloat16", marks=pytest.mark.low_priority),
        pytest.param("int32", marks=pytest.mark.low_priority),
        pytest.param("uint32", marks=pytest.mark.low_priority),
    ],
)
@pytest.mark.parametrize(
    "target",
    [
        "ascendc",
        pytest.param("pto", marks=pytest.mark.low_priority),
    ],
)
def test_broadcast_2d_axis0(dtype, target):
    M, N = 8, 64
    program = _make_2d_axis0(dtype, M, N)
    kernel = tilelang.compile(program, out_idx=[-1], pass_configs=PASS_CONFIGS, target=target)
    a = _gen_input(dtype, (1, N))
    b = kernel(a)
    torch.npu.synchronize()
    ref = a.expand(M, N)
    if dtype in ("uint16", "uint32"):
        torch.testing.assert_close(b.cpu(), ref.cpu(), rtol=0, atol=0)
    else:
        rtol = 1e-2 if dtype in ("float16", "float32", "bfloat16") else 0
        atol = 1e-2 if dtype in ("float16", "float32", "bfloat16") else 0
        torch.testing.assert_close(b, ref, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Functional: 1D-to-2D auto-infer (axis=None, src=(N,) -> dst=(M,N))
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "target",
    [
        "ascendc",
        pytest.param("pto", marks=pytest.mark.low_priority),
    ],
)
def test_broadcast_1d_to_2d_auto_infer(target):
    M, N = 8, 64

    @T.prim_func
    def main(
        A: T.Tensor((N,), "float32"),  # type: ignore
        B: T.Tensor((M, N), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (_, vid):
            src_ub = T.alloc_ub((N,), "float32")
            dst_ub = T.alloc_ub((M, N), "float32")
            if vid == 0:
                T.copy(A, src_ub)
                T.tile.broadcast(dst_ub, src_ub)
                T.copy(dst_ub, B)

    kernel = tilelang.compile(main, out_idx=[-1], pass_configs=PASS_CONFIGS, target=target)
    a = torch.arange(N, dtype=torch.float32).npu()
    b = kernel(a)
    torch.npu.synchronize()
    ref = a.expand(M, N)
    torch.testing.assert_close(b, ref, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# Functional: 1D axis=0 (src=(1,) -> dst=(N,))
# ---------------------------------------------------------------------------


def test_broadcast_1d_axis0_ascendc():
    N = 64

    @T.prim_func
    def main(
        A: T.Tensor((1,), "float32"),  # type: ignore
        B: T.Tensor((N,), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (_, vid):
            src_ub = T.alloc_ub((1,), "float32")
            dst_ub = T.alloc_ub((N,), "float32")
            if vid == 0:
                T.copy(A, src_ub)
                T.tile.broadcast(dst_ub, src_ub, axis=0)
                T.copy(dst_ub, B)

    kernel = tilelang.compile(main, out_idx=[-1], pass_configs=PASS_CONFIGS, target="ascendc")
    a = torch.tensor([42.0], dtype=torch.float32).npu()
    b = kernel(a)
    torch.npu.synchronize()
    ref = a.expand(N)
    torch.testing.assert_close(b, ref, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# Exception boundary: dtype mismatch (default — no LP)
# ---------------------------------------------------------------------------


def test_broadcast_dtype_mismatch_raises():
    """dst and src dtype must match; mismatch should fail at compile time."""

    @T.prim_func
    def main(
        A: T.Tensor((1, 64), "float16"),  # type: ignore
        B: T.Tensor((8, 64), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (_, vid):
            src_ub = T.alloc_ub((1, 64), "float16")
            dst_ub = T.alloc_ub((8, 64), "float32")
            if vid == 0:
                T.copy(A, src_ub)
                T.tile.broadcast(dst_ub, src_ub, axis=0)
                T.copy(dst_ub, B)

    with pytest.raises(RuntimeError, match="Compilation Failed"):  # noqa: B017
        tilelang.compile(main, out_idx=[-1], pass_configs=PASS_CONFIGS, target="ascendc")


# ---------------------------------------------------------------------------
# Exception boundary: shape mismatch (default — no LP)
# ---------------------------------------------------------------------------


def test_broadcast_shape_mismatch_axis0_raises():
    """src[0] != 1 and shapes don't match on axis=0 should raise during TIR construction."""

    with pytest.raises(RuntimeError, match="DiagnosticError"):

        @T.prim_func
        def main(
            A: T.Tensor((2, 64), "float32"),  # type: ignore
            B: T.Tensor((8, 64), "float32"),  # type: ignore
        ):
            with T.Kernel(1, is_npu=True) as (_, vid):
                src_ub = T.alloc_ub((2, 64), "float32")
                dst_ub = T.alloc_ub((8, 64), "float32")
                if vid == 0:
                    T.copy(A, src_ub)
                    T.tile.broadcast(dst_ub, src_ub, axis=0)
                    T.copy(dst_ub, B)


def test_broadcast_shape_mismatch_axis1_raises():
    """src[1] != 1 and shapes don't match on axis=1 should raise during TIR construction."""

    with pytest.raises(RuntimeError, match="DiagnosticError"):

        @T.prim_func
        def main(
            A: T.Tensor((8, 2), "float32"),  # type: ignore
            B: T.Tensor((8, 64), "float32"),  # type: ignore
        ):
            with T.Kernel(1, is_npu=True) as (_, vid):
                src_ub = T.alloc_ub((8, 2), "float32")
                dst_ub = T.alloc_ub((8, 64), "float32")
                if vid == 0:
                    T.copy(A, src_ub)
                    T.tile.broadcast(dst_ub, src_ub, axis=1)
                    T.copy(dst_ub, B)


# ---------------------------------------------------------------------------
# Exception boundary: invalid axis (default — no LP)
# ---------------------------------------------------------------------------


def test_broadcast_invalid_axis_raises():
    """axis must be 0 or 1; axis=2 should raise during TIR construction."""

    with pytest.raises(RuntimeError, match="DiagnosticError"):

        @T.prim_func
        def main(
            A: T.Tensor((1, 64), "float32"),  # type: ignore
            B: T.Tensor((8, 64), "float32"),  # type: ignore
        ):
            with T.Kernel(1, is_npu=True) as (_, vid):
                src_ub = T.alloc_ub((1, 64), "float32")
                dst_ub = T.alloc_ub((8, 64), "float32")
                if vid == 0:
                    T.copy(A, src_ub)
                    T.tile.broadcast(dst_ub, src_ub, axis=2)
                    T.copy(dst_ub, B)


# ---------------------------------------------------------------------------
# Exception boundary: 1D uninferable (default — no LP)
# ---------------------------------------------------------------------------


def test_broadcast_1d_uninferable_raises():
    """1D src that can't be inferred to 2D dst should raise during TIR construction."""

    with pytest.raises(RuntimeError, match="DiagnosticError"):

        @T.prim_func
        def main(
            A: T.Tensor((4,), "float32"),  # type: ignore
            B: T.Tensor((8, 16), "float32"),  # type: ignore
        ):
            with T.Kernel(1, is_npu=True) as (_, vid):
                src_ub = T.alloc_ub((4,), "float32")
                dst_ub = T.alloc_ub((8, 16), "float32")
                if vid == 0:
                    T.copy(A, src_ub)
                    T.tile.broadcast(dst_ub, src_ub)
                    T.copy(dst_ub, B)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-n", "8"])
