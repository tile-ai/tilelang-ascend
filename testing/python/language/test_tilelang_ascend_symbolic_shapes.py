"""Frontend proofs: acceptance does not imply dynamic backend support."""

import pytest
from tilelang import tvm
from tilelang.language import ascend_tile

tir = tvm.tir


def _broadcast(src_shape, dst_shape, axis):
    src = tir.decl_buffer(src_shape, "float32", scope="shared.ub")
    dst = tir.decl_buffer(dst_shape, "float32", scope="shared.ub")
    return ascend_tile.broadcast(dst, src, axis=axis)


@pytest.mark.parametrize("kind", ["independent_expr", "algebraic", "constant", "no_broadcast", "one_dim", "infer_axis"])
def test_broadcast_proven_equal_shapes(kind):
    k = tir.Var("k", "int32")
    lhs, rhs = 32 - k, 32 - k
    if kind == "algebraic":
        rhs = 31 - k + 1
    if kind == "constant":
        lhs, rhs = 16, 16
    if kind == "no_broadcast":
        call = _broadcast((lhs, 32), (rhs, 32), 0)
    elif kind == "one_dim":
        call = _broadcast((lhs,), (rhs,), 0)
    elif kind == "infer_axis":
        call = _broadcast((lhs,), (8, rhs), None)
    else:
        call = _broadcast((lhs, 1), (rhs, 32), 1)
    assert isinstance(call, tir.Call)


@pytest.mark.parametrize("kind", ["unequal", "unknown", "invalid_axis"])
def test_broadcast_unproven_shapes_rejected(kind):
    k = tir.Var("k", "int32")
    rhs = 31 - k if kind == "unequal" else tir.Var("m", "int32")
    axis = 2 if kind == "invalid_axis" else 1
    with pytest.raises(ValueError):
        _broadcast((32 - k, 1), (rhs, 32), axis)


@pytest.mark.parametrize("source", [0, 1])
@pytest.mark.parametrize("kind", ["independent_expr", "algebraic", "constant", "unequal", "unknown"])
def test_mul_add_symbolic_size_compatibility(kind, source):
    k = tir.Var("k", "int32")
    lhs, rhs = 32 - k, 32 - k
    if kind == "algebraic":
        rhs = 31 - k + 1
    elif kind == "constant":
        lhs, rhs = 16, 16
    elif kind == "unequal":
        rhs = 31 - k
    elif kind == "unknown":
        rhs = tir.Var("m", "int32")
    dst = tir.decl_buffer((lhs, 32), "float32", scope="shared.ub")
    src0 = tir.decl_buffer((rhs, 32), "float32", scope="shared.ub")
    src1 = tir.decl_buffer((32 - k if kind != "constant" else 16, 32), "float32", scope="shared.ub")
    if source == 1:
        src0, src1 = src1, src0
    if kind in ("unequal", "unknown"):
        with pytest.raises(AssertionError, match="same size"):
            ascend_tile.mul_add_dst(dst, src0, src1)
    else:
        assert isinstance(ascend_tile.mul_add_dst(dst, src0, src1), tir.Call)


def test_broadcast_buffer_region():
    k = tir.Var("k", "int32")
    src = tir.decl_buffer((64, 1), "float32", scope="shared.ub")
    dst = tir.decl_buffer((64, 32), "float32", scope="shared.ub")
    src_region = tir.BufferRegion(src, [tvm.ir.Range(k, 64), tvm.ir.Range(0, 1)])
    dst_region = tir.BufferRegion(dst, [tvm.ir.Range(k, 63 + 1), tvm.ir.Range(0, 32)])
    assert isinstance(ascend_tile.broadcast(dst_region, src_region, axis=1), tir.Call)


@pytest.mark.parametrize("operand", [0, 1, 2])
@pytest.mark.parametrize("symbolic", [False, True])
def test_mul_add_noncontiguous_region_rejected(operand, symbolic):
    k = tir.Var("k", "int32")
    extent = 32 - k if symbolic else 32
    buf = tir.decl_buffer((2, 64), "float32", scope="shared.ub")
    narrow = tir.BufferRegion(buf, [tvm.ir.Range(0, 2), tvm.ir.Range(0, extent)])
    # Other operands are compact buffers with the same logical shape.
    operands = [tir.decl_buffer((2, extent), "float32", scope="shared.ub") for _ in range(3)]
    operands[operand] = narrow
    with pytest.raises(ValueError, match="contiguous when flattened"):
        ascend_tile.mul_add_dst(*operands)


def test_mul_add_symbolic_whole_rows():
    k = tir.Var("k", "int32")
    buffers = [tir.decl_buffer((4, 64), "float32", scope="shared.ub") for _ in range(3)]
    regions = [tir.BufferRegion(buf, [tvm.ir.Range(0, rows), tvm.ir.Range(0, 64)]) for buf, rows in zip(buffers, [4 - k, 3 - k + 1, 4 - k])]
    assert isinstance(ascend_tile.mul_add_dst(*regions), tir.Call)


@pytest.mark.parametrize("operand", [0, 1])
@pytest.mark.parametrize("symbolic", [False, True])
def test_broadcast_noncontiguous_region_rejected(operand, symbolic):
    k = tir.Var("k", "int32")
    extent = 32 - k if symbolic else 32
    buf = tir.decl_buffer((2, 64), "float32", scope="shared.ub")
    narrow = tir.BufferRegion(buf, [tvm.ir.Range(0, 2), tvm.ir.Range(0, extent)])
    operands = [tir.decl_buffer((2, extent), "float32", scope="shared.ub") for _ in range(2)]
    operands[operand] = narrow
    with pytest.raises(ValueError, match="contiguous when flattened"):
        ascend_tile.broadcast(*operands, axis=0)


@pytest.mark.parametrize("kind", ["single_row", "whole_rows"])
def test_broadcast_contiguous_symbolic_regions(kind):
    k = tir.Var("k", "int32")
    src = tir.decl_buffer((1, 64), "float32", scope="shared.ub")
    dst = tir.decl_buffer((4, 64), "float32", scope="shared.ub")
    if kind == "single_row":
        src_region = tir.BufferRegion(src, [tvm.ir.Range(0, 1), tvm.ir.Range(0, 64 - k)])
        dst_region = tir.BufferRegion(dst, [tvm.ir.Range(1, 2), tvm.ir.Range(0, 63 - k + 1)])
    else:
        src_region = tir.BufferRegion(src, [tvm.ir.Range(0, 1), tvm.ir.Range(0, 64)])
        dst_region = tir.BufferRegion(dst, [tvm.ir.Range(0, 4 - k), tvm.ir.Range(0, 64)])
    assert isinstance(ascend_tile.broadcast(dst_region, src_region, axis=0), tir.Call)
