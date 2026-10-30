"""CPU-only lowering checks for the physical UB row pitch of copy/atomic DMA."""

import re

import pytest
import tvm
from tvm import ir, tir

import tilelang
import tilelang.language as T


def _lower_copy(src, extents, *, atomic=False, layout=None, mins=None):
    """Exercise LowerTileOp directly, without device compilation or allocation."""
    mins = mins if mins is not None else [0] * len(extents)
    dst = tir.decl_buffer(extents, "float32", name="output")
    src_region = tir.BufferRegion(src, [ir.Range.from_min_extent(start, size) for start, size in zip(mins, extents)])
    if atomic:
        call = T.tile.atomic_add(dst, src_region)
    else:
        # Ascend's earlier legalization selects this intrinsic from T.copy.
        call = tir.call_intrin("handle", ir.Op.get("tl.ascend_copy"), *T.copy(src_region, dst).args)
    block = tir.Block(
        [],
        [],
        [],
        "root",
        tir.Evaluate(call),
        alloc_buffers=[src],
        annotations={"layout_map": {src: layout}} if layout is not None else {},
    )
    func = tir.PrimFunc([dst.data], tir.BlockRealize([], True, block), buffer_map={dst.data: dst}).with_attr(
        "target", tvm.target.Target({"kind": "llvm", "model": "ascendc"})
    )
    return tilelang.transform.LowerTileOp()(tvm.IRModule({"main": func}))["main"].body


def _assert_pitch(body, expected, *, atomic=False):
    helper = "atomic_add_ub_to_gm" if atomic else "copy_ub_to_gm"
    names = []

    def collect(node):
        if isinstance(node, tir.StringImm) and node.value.startswith(f"tl::ascend::{helper}<"):
            names.append(node.value)

    tir.stmt_functor.post_order_visit(body, collect)
    assert len(names) == 1, names
    match = re.match(rf"tl::ascend::{helper}<float,\s*(\d+)(?:,|>)", names[0])
    assert match is not None, names[0]
    assert int(match.group(1)) == expected


@pytest.mark.parametrize("atomic", [False, True])
@pytest.mark.parametrize("column_start", [0, 4])
def test_sliced_ub_uses_backing_row_pitch(atomic, column_start):
    src = tir.decl_buffer((256, 8), "float32", name="ub", scope="shared.ub")
    body = _lower_copy(src, (256, 4), atomic=atomic, mins=[0, column_start])
    _assert_pitch(body, 8, atomic=atomic)


@pytest.mark.parametrize("atomic", [False, True])
def test_rank_one_source(atomic):
    src = tir.decl_buffer((16,), "float32", name="ub", scope="shared.ub")
    _assert_pitch(_lower_copy(src, (8,), atomic=atomic), 16, atomic=atomic)


@pytest.mark.parametrize("atomic", [False, True])
def test_singleton_dimension_between_rows_and_columns(atomic):
    src = tir.decl_buffer((4, 3, 8), "float32", name="ub", scope="shared.ub")
    body = _lower_copy(src, (4, 1, 4), atomic=atomic, mins=[0, 1, 0])
    _assert_pitch(body, 24, atomic=atomic)


@pytest.mark.parametrize("atomic", [False, True])
def test_explicit_source_strides(atomic):
    src = tir.decl_buffer((4, 8), "float32", name="ub", scope="shared.ub", strides=(16, 1))
    _assert_pitch(_lower_copy(src, (4, 4), atomic=atomic), 16, atomic=atomic)


@pytest.mark.parametrize("atomic", [False, True])
@pytest.mark.parametrize("strides", [None, (32, 1)])
def test_affine_layout_uses_remapped_storage(atomic, strides):
    src = tir.decl_buffer((4, 8), "float32", name="ub", scope="shared.ub", strides=strides)
    layout = tilelang.Layout((4, 8), lambda i, j: i * 16 + j)
    body = _lower_copy(src, (4, 4), atomic=atomic, layout=layout)
    _assert_pitch(body, 16, atomic=atomic)


@pytest.mark.parametrize("atomic", [False, True])
def test_noncontiguous_columns_are_rejected(atomic):
    src = tir.decl_buffer((4, 8), "float32", name="ub", scope="shared.ub", strides=(16, 2))
    with pytest.raises(tvm.error.TVMError, match="UB copy requires contiguous columns"):
        _lower_copy(src, (4, 4), atomic=atomic)


@pytest.mark.parametrize("atomic", [False, True])
def test_nonuniform_layout_row_pitch_is_rejected(atomic):
    src = tir.decl_buffer((4, 8), "float32", name="ub", scope="shared.ub")
    # Rows 0->1 advance 8, but rows 1->2 advance 16. Checking only the origin
    # would incorrectly accept this layout as a uniform-pitch DMA source.
    layout = tilelang.Layout((4, 8), lambda i, j: i * 8 + (i // 2) * 8 + j)
    with pytest.raises(tvm.error.TVMError, match="UB copy requires a positive compile-time row pitch"):
        _lower_copy(src, (4, 4), atomic=atomic, layout=layout)
