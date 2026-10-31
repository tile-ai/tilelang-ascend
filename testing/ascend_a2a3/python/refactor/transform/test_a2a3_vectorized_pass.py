"""TA vectorization planning, including its original fallback boundaries."""

import pytest

from tilelang import tvm
from tilelang.ascend.a2a3 import transform

tirx = tvm.tirx


def _lower(extent=16, stride=1, external=False, dynamic=False, kind=tirx.ForKind.VECTORIZED, minimum=0, dtype="float32"):
    index = tirx.Var("i", "int32")
    n = tirx.Var("n", "int32")
    src = tirx.decl_buffer((n if dynamic else extent * stride,), dtype, name="src")
    dst = tirx.decl_buffer((extent * stride,), dtype, name="dst")
    value = tirx.call_extern("float32", "user_fn", index) if external else tirx.BufferLoad(src, [index * stride])
    body = tirx.For(index, minimum, extent, kind, tirx.BufferStore(dst, value, [index * stride]))
    func = tirx.PrimFunc([src.data, dst.data, n], body, buffer_map={src.data: src, dst.data: dst}).with_attr("global_symbol", "main")
    after = transform.LegalizeVectorizedLoop()(tvm.IRModule.from_expr(func))["main"]
    loops = []
    tirx.stmt_functor.post_order_visit(after.body, lambda node: loops.append(node) if isinstance(node, tirx.For) else None)
    return body, after, loops


@pytest.mark.parametrize("extent,width", [(16, 4), (10, 2)])
def test_vectorized_loop_uses_original_128_bit_access_plan(extent, width):
    _, after, loops = _lower(extent=extent)
    assert len(loops) == 2
    inner, outer = loops
    assert inner.kind == tirx.ForKind.VECTORIZED
    assert int(inner.extent) == width
    assert outer.kind == tirx.ForKind.SERIAL
    assert int(outer.extent) == extent // width
    assert isinstance(after.body, tirx.For)


@pytest.mark.parametrize("options", [{"stride": 2}, {"external": True}, {"dynamic": True}])
def test_original_serial_fallbacks(options):
    _, _, loops = _lower(**options)
    assert len(loops) == 1
    assert loops[0].kind == tirx.ForKind.SERIAL
    assert int(loops[0].extent) == 16


def test_serial_loop_is_not_reclassified():
    original, after, _ = _lower(kind=tirx.ForKind.SERIAL)
    tvm.ir.assert_structural_equal(after.body, original)


def test_original_nonzero_vectorized_minimum_diagnostic():
    with pytest.raises(tvm.error.InternalError, match="is_zero"):
        _lower(minimum=1)


def test_bool_buffer_uses_current_ir_physical_width():
    # TA's UInt1 representation produced width 128 here. TIRX uses Bool8;
    # retain the original 128-bit access limit using its actual element size.
    _, _, loops = _lower(extent=128, dtype="bool")
    inner, outer = loops
    assert inner.kind == tirx.ForKind.VECTORIZED
    assert int(inner.extent) == 16
    assert outer.kind == tirx.ForKind.SERIAL
    assert int(outer.extent) == 8
