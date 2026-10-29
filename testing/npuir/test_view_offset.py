"""A view keeps its source address when shape or element type changes."""

import pytest
import tilelang.language as T
import tvm


@pytest.mark.parametrize(
    "old,new,offset,expected",
    [
        ("float32", "float32", 64, 64),
        ("float32", "float16", 32, 64),
        ("float16", "float32", 64, 32),
        ("float32", "float16", 0, 0),
    ],
)
def test_view_preserves_byte_offset(old, new, offset, expected):
    src = tvm.tir.decl_buffer((4, 16), old, elem_offset=offset, scope="shared")
    dst = T.view(src, (8, 8), new)
    assert dst.data.same_as(src.data)
    assert int(dst.elem_offset) == expected
    assert [int(x) for x in dst.shape] == [8, 8]


def test_default_view_preserves_offset():
    src = tvm.tir.decl_buffer((4, 16), "float32", elem_offset=64, scope="shared")
    dst = T.view(src)
    assert int(dst.elem_offset) == 64
    assert dst.data.same_as(src.data)


def test_symbolic_aligned_offset():
    i = tvm.tir.Var("i", "int32")
    src = tvm.tir.decl_buffer((4, 16), "float16", elem_offset=8 * i, scope="shared")
    dst = T.view(src, (8, 8), "float32")
    assert tvm.arith.Analyzer().can_prove_equal(dst.elem_offset, 4 * i)


def test_misaligned_offset_is_rejected():
    src = tvm.tir.decl_buffer((4, 16), "float16", elem_offset=1, scope="shared")
    with pytest.raises(ValueError, match="offset.*align|offset.*divis"):
        T.view(src, (8, 8), "float32")
