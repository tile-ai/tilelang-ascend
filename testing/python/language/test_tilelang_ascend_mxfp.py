"""Smoke-test for the new MXFP GEMM / quant / dequant Python APIs.

This file only verifies that the tile-lang DSL accepts the new primitives
(`T.mma_mx`, `T.tile.tquant_mxfp8`, `T.tile.tdequant`) and produces IR
without crashing. It does **not** execute on NPU hardware.

Run with:
    pytest testing/python/language/test_tilelang_ascend_mxfp.py -v
"""

import pytest  # noqa: F401

import tilelang  # noqa: F401
import tilelang.language as T

FP8 = "e4m3_float8"
SCALE = "uint8"


# ---------------------------------------------------------------------------
# T.mma_mx
# ---------------------------------------------------------------------------


def test_mma_mx_ir_generation():
    """Calling T.mma_mx inside a prim_func should not raise."""
    M, N, K = 64, 64, 64
    sa_cols = K // 32
    sb_rows = K // 32

    @T.prim_func
    def main(
        A: T.Tensor((M, K), FP8),
        B: T.Tensor((K, N), FP8),
        Sa: T.Tensor((M, sa_cols), SCALE),
        Sb: T.Tensor((sb_rows, N), SCALE),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _), T.Scope("C"):
            A_l0a = T.alloc_L0A((M, K), FP8)
            B_l0b = T.alloc_L0B((K, N), FP8)
            C_l0c = T.alloc_L0C((M, N), "float32")
            Sa_ub = T.alloc_shared((M, sa_cols), SCALE)
            Sb_ub = T.alloc_shared((sb_rows, N), SCALE)
            T.mma_mx(
                A_l0a,
                B_l0b,
                C_l0c,
                Sa_ub,
                Sb_ub,
                init=True,
                scale_dtype=SCALE,
            )

    # Basic IR sanity: the function is a tvm.tir.PrimFunc
    from tvm.tir import PrimFunc  # local import to avoid import-time cost

    assert isinstance(main, PrimFunc)


def test_mma_mx_k_not_multiple_of_64_rejected():
    """K % 64 != 0 must raise AssertionError at npu_gemm_mx() call time."""
    import tilelang.language.customize as cz
    import tvm.tir as tir

    M, N, K = 64, 64, 65
    sa_cols = K // 32
    sb_rows = K // 32

    a_buf = tir.decl_buffer((M, K), "e4m3_float8", name="a")
    b_buf = tir.decl_buffer((K, N), "e4m3_float8", name="b")
    c_buf = tir.decl_buffer((M, N), "float32", name="c")
    sa_buf = tir.decl_buffer((M, sa_cols), "uint8", name="sa")
    sb_buf = tir.decl_buffer((sb_rows, N), "uint8", name="sb")

    with pytest.raises(AssertionError):
        cz.npu_gemm_mx(a_buf, b_buf, c_buf, sa_buf, sb_buf, init=False)


def test_mma_mx_scale_dtype_validation():
    """Unsupported scale_dtype should raise ValueError."""
    import tilelang.language.customize as cz
    import tvm.tir as tir

    M, N, K = 64, 64, 64
    sa_cols = K // 32
    sb_rows = K // 32

    a_buf = tir.decl_buffer((M, K), "e4m3_float8", name="a")
    b_buf = tir.decl_buffer((K, N), "e4m3_float8", name="b")
    c_buf = tir.decl_buffer((M, N), "float32", name="c")
    sa_buf = tir.decl_buffer((M, sa_cols), "uint8", name="sa")
    sb_buf = tir.decl_buffer((sb_rows, N), "uint8", name="sb")

    with pytest.raises(ValueError):
        cz.npu_gemm_mx(a_buf, b_buf, c_buf, sa_buf, sb_buf, init=False, scale_dtype="bogus_scale_dtype")


def test_mma_mx_happy_path_ir():
    """Valid input should produce a tir.call_intrin (not raise)."""
    import tilelang.language.customize as cz
    import tvm.tir as tir
    from tvm.tir import Call

    M, N, K = 64, 64, 64
    sa_cols = K // 32
    sb_rows = K // 32

    a_buf = tir.decl_buffer((M, K), "e4m3_float8", name="a")
    b_buf = tir.decl_buffer((K, N), "e4m3_float8", name="b")
    c_buf = tir.decl_buffer((M, N), "float32", name="c")
    sa_buf = tir.decl_buffer((M, sa_cols), "uint8", name="sa")
    sb_buf = tir.decl_buffer((sb_rows, N), "uint8", name="sb")

    call = cz.npu_gemm_mx(a_buf, b_buf, c_buf, sa_buf, sb_buf, init=False)
    assert isinstance(call, Call)
    # The template string should include "mma_mxfp<...>"
    template = str(call.args[0])
    assert "mma_mxfp<" in template
    assert "float8_e4m3_t" in template
    assert "float, " in template


# ---------------------------------------------------------------------------
# T.tile.tquant_mxfp8 / T.tile.tdequant
# (Pure Python/IR-level tests: we call the DSL helpers on hand-crafted
# Buffer objects rather than going through the TVM @prim_func decorator,
# which would trigger the ScriptCompleter and fail on opaque access_ptr
# calls.)
# ---------------------------------------------------------------------------


def test_tile_tquant_mxfp8_call():
    """tquant_mxfp8 on plain Buffers returns a tir.Call intrinsic."""
    import tvm.tir as tir
    from tvm.tir import Call

    M, K = 32, 64
    sa_cols = K // 32

    src = tir.decl_buffer((M, K), "float32", name="src")
    dst = tir.decl_buffer((M, K), "uint8", name="dst")
    exp = tir.decl_buffer((M, sa_cols), "uint8", name="exp")
    max_b = tir.decl_buffer((M, K), "float32", name="max_b")
    sca_b = tir.decl_buffer((M, K), "float32", name="sca_b")

    call = T.tile.tquant_mxfp8(dst, src, exp, max_b, sca_b)
    assert isinstance(call, Call)
    assert "tquant_mxfp8<" in str(call.args[0])


def test_tile_tdequant_call_with_offset():
    """tdequant with explicit offset buffer returns a tir.Call."""
    import tvm.tir as tir
    from tvm.tir import Call

    M, N = 16, 16
    src = tir.decl_buffer((M, N), "int8", name="src")
    scale = tir.decl_buffer((M, N), "float32", name="scale")
    off = tir.decl_buffer((M, N), "float32", name="off")
    dst = tir.decl_buffer((M, N), "float32", name="dst")

    call = T.tile.tdequant(dst, src, scale, offset=off)
    assert isinstance(call, Call)
    assert "tdequant<" in str(call.args[0])


def test_tile_tdequant_call_without_offset():
    """tdequant with offset=None should still produce a tir.Call."""
    import tvm.tir as tir
    from tvm.tir import Call

    M, N = 16, 16
    src = tir.decl_buffer((M, N), "int16", name="src")
    scale = tir.decl_buffer((M, N), "float32", name="scale")
    dst = tir.decl_buffer((M, N), "float32", name="dst")

    call = T.tile.tdequant(dst, src, scale, offset=None)
    assert isinstance(call, Call)
    assert "tdequant<" in str(call.args[0])
