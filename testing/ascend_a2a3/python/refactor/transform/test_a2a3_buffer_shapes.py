"""TA shape collectors through the TIRX allocation representation."""

import pytest

from tilelang import tvm
from tilelang.ascend.a2a3 import transform

tirx = tvm.tirx


def _program():
    n = tirx.Var("n", "int32")
    gm = tirx.decl_buffer((n, 9), "float16", name="gm")
    ub = tirx.decl_buffer((2, 3, 9), "float16", name="ub", scope="shared.ub")
    l1 = tirx.decl_buffer((64, 200), "float16", name="l1", scope="shared.l1")
    scalar = tirx.decl_buffer((), "int32", name="scalar", scope="local")
    # A constant-sized Buffer allocated explicitly exercises the metadata
    # carried by AllocBuffer after removal of old Allocate/AllocateConst.
    explicit = tirx.decl_buffer((7, 5), "int8", name="explicit", scope="shared.ub")
    block = tirx.SBlock([], [], [], "root", tirx.SeqStmt([tirx.AllocBuffer(explicit), tirx.Evaluate(1)]), alloc_buffers=[ub, l1, scalar])
    func = tirx.PrimFunc([gm.data, n], tirx.SBlockRealize([], True, block), buffer_map={gm.data: gm})
    return func.with_attr({"global_symbol": "main", "keep_metadata": "keep"}), (gm, ub, l1, scalar, explicit)


@pytest.mark.parametrize(
    "factory,attr", [(transform.BufferShapeCollector, "initial_buffer_shapes"), (transform.CollectBufferShapes, "buffer_shapess")]
)
def test_collectors_preserve_shapes_identity_and_body(factory, attr):
    original, buffers = _program()
    after = factory()(tvm.IRModule.from_expr(original))["main"]
    shapes = after.attrs[attr]
    for buffer in buffers:
        tvm.ir.assert_structural_equal(shapes[buffer.data], buffer.shape)
    tvm.ir.assert_structural_equal(after.body, original.body)
    assert after.attrs["keep_metadata"] == "keep"
    assert attr not in original.attrs


def test_flatten2d_alignment_l1_tail_and_initial_inner_dimension():
    original, buffers = _program()
    initial = transform.BufferShapeCollector()(tvm.IRModule.from_expr(original))
    after = transform.Flatten2DBuffer()(initial)["main"]
    gm, ub, l1, scalar, explicit = buffers
    shapes = after.attrs["logic_buffer_shapes"]
    tvm.ir.assert_structural_equal(shapes[gm.data], gm.shape)
    assert [int(x) for x in shapes[ub.data]] == [6, 16, 6, 9]
    # Full zN allocation 13*64*16=13312; ceil-div by original N=200
    # must cover it, rather than truncating to 66 rows.
    assert [int(x) for x in shapes[l1.data]] == [67, 200, 67, 200]
    assert len(shapes[scalar.data]) == 0
    assert [int(x) for x in shapes[explicit.data]] == [7, 32, 7, 5]
    tvm.ir.assert_structural_equal(after.body, original.body)
    assert after.attrs["keep_metadata"] == "keep"
    # TA WithAttrs merges keys: despite the source's removal comment, the
    # initial shape attribute survives. Retain that observed source behavior.
    assert "initial_buffer_shapes" in after.attrs


def test_flatten2d_ub_vector_axis_and_missing_initial_shapes():
    buffer = tirx.decl_buffer((2, 5, 1), "float32", scope="shared.ub")
    func = tirx.PrimFunc([], tirx.AllocBuffer(buffer)).with_attr("global_symbol", "main")
    after = transform.Flatten2DBuffer()(tvm.IRModule.from_expr(func))["main"]
    assert [int(x) for x in after.attrs["logic_buffer_shapes"][buffer.data]] == [1, 16, 1, 10]
