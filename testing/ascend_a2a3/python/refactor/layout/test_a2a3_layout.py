"""TA layout tags, physical offsets and whole-fractal allocation geometry."""

import math

import pytest

from tilelang import tvm
from tilelang.ascend.a2a3.layout import (
    AscendLayout,
    Layout,
    layout_str,
    make_col_major_layout,
    make_nz_layout,
    make_zn_layout,
    physical_allocation_shape,
    wrap_layout,
)
from tilelang.layout import Layout as SharedLayout
from tilelang.layout import Fragment


@pytest.mark.parametrize("tag,spelling", list(enumerate(["RowMajor", "ColumnMajor", "zN", "zZ", "nZ"])))
def test_layout_tag_and_explicit_bridge(tag, spelling):
    layout = Layout((8, 9), lambda i, j: [i, j], layout_tag=tag)
    assert layout_str(layout) == "layout::" + spelling
    shared = SharedLayout((8, 9), lambda i, j: [i, j])
    with pytest.raises(tvm.error.InternalError, match="metadata is missing"):
        layout_str(shared)
    wrapped = wrap_layout(shared, tag)
    assert layout_str(wrapped) == layout_str(layout)
    tvm.ir.assert_structural_equal(wrapped.index, shared.index)


@pytest.mark.parametrize("dtype", ["float16", "int8", "float32"])
@pytest.mark.parametrize("shape", [(7, 9), (64, 200), (200, 128), (2, 17, 33)])
@pytest.mark.parametrize("transposed", [False, True])
def test_fractal_offsets_and_full_tail_capacity(dtype, shape, transposed):
    buffer = tvm.tirx.decl_buffer(shape, dtype, scope="shared.l1")
    layout = (make_nz_layout if transposed else make_zn_layout)(buffer)
    rows, cols = shape[-2:]
    c0 = 256 // tvm.DataType(dtype).bits
    # Independent physical grid oracle: rows of fractals, then the two
    # lanes within each fractal. This includes padding never reached by a
    # valid logical coordinate.
    if transposed:
        physical = ((rows + c0 - 1) // c0, (cols + 15) // 16, 16, c0)
    else:
        physical = ((cols + c0 - 1) // c0, (rows + 15) // 16, 16, c0)
    capacity = math.prod(physical)
    assert [int(x) for x in physical_allocation_shape(layout, dtype)] == [*shape[:-2], capacity]
    for i, j in [(0, 0), (rows - 1, cols - 1), (rows // 2, cols // 2)]:
        coord = (i // c0, j // 16, j % 16, i % c0) if transposed else (j // c0, i // 16, i % 16, j % c0)
        offset = 0
        for index, extent in zip(coord, physical):
            offset = offset * extent + index
        actual = layout.map_forward_index([*[0] * len(shape[:-2]), i, j])
        assert int(tvm.arith.Analyzer().simplify(actual[-1])) == offset
        assert 0 <= offset < capacity
    assert layout_str(layout) == ("layout::nZ" if transposed else "layout::zN")


def test_column_major_and_default_layout_preserve_ta_tags():
    buffer = tvm.tirx.decl_buffer((3, 5), "float16")
    layout = make_col_major_layout(buffer)
    assert layout_str(layout) == "layout::ColumnMajor"
    assert [int(x) for x in layout.map_forward_index([2, 4])] == [4, 2]
    assert layout_str(Layout((3, 5), lambda i, j: [i, j])) == "layout::RowMajor"
    # TA inverse construction resets the metadata to its default tag 0.
    assert layout_str(layout.inverse()) == "layout::RowMajor"


def test_symbolic_shape_retains_ta_static_override_boundary():
    m = tvm.tirx.Var("m", "int32")
    layout = make_zn_layout(tvm.tirx.decl_buffer((m, 33), "float16", scope="shared.l1"))
    assert layout_str(layout) == "layout::zN"
    tvm.ir.assert_structural_equal(physical_allocation_shape(layout, "float16"), layout.get_output_shape())


def test_zz_tag_uses_ta_allocation_rule():
    layout = Layout((17, 33), lambda i, j: [i * 33 + j], layout_tag=AscendLayout.kzZ.value)
    assert [int(x) for x in physical_allocation_shape(layout, "float16")] == [1536]


def test_bridge_cannot_slice_fragment_state():
    fragment = Fragment((8,), forward_thread_fn=lambda i: i, forward_index_fn=lambda i: 0)
    with pytest.raises(tvm.error.InternalError, match="cannot discard additional layout state"):
        wrap_layout(fragment, AscendLayout.kRowMajor)


@pytest.mark.parametrize(
    "operation", [lambda layout: layout.expand([2]), lambda layout: layout.repeat(0, 2), lambda layout: layout.reshape((15,))]
)
def test_new_shared_layout_transformations_require_migration(operation):
    layout = Layout((3, 5), lambda i, j: [i, j], layout_tag=1)
    with pytest.raises(tvm.error.InternalError, match="is not migrated"):
        operation(layout)
