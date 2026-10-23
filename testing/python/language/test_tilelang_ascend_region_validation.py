"""CPU-only regressions for flat vector access and broadcast dimensionality."""

import pytest
import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.language import ascend_tile


def _buffer(shape, strides=None, elem_offset=0):
    return tvm.tir.decl_buffer(shape, "float32", scope="shared.ub", strides=strides, elem_offset=elem_offset)


def _region(buffer, bounds=None):
    bounds = bounds or [(0, extent) for extent in buffer.shape]
    return tvm.tir.BufferRegion(buffer, [tvm.ir.Range.from_min_extent(start, extent) for start, extent in bounds])


def _call(op, operands):
    if op == "broadcast":
        return ascend_tile.broadcast(*operands, axis=0)
    return ascend_tile.mul_add_dst(*operands)


@pytest.mark.parametrize("op,operand", [("broadcast", i) for i in range(2)] + [("mul_add_dst", i) for i in range(3)])
@pytest.mark.parametrize("as_region", [False, True])
@pytest.mark.parametrize("shape,strides", [((2, 32), (64, 1)), ((32,), (2,)), ((2, 1), (4, 1))])
def test_flat_ops_reject_strided_operands(op, operand, as_region, shape, strides):
    operands = [_buffer(shape) for _ in range(2 if op == "broadcast" else 3)]
    bad = _buffer(shape, strides)
    operands[operand] = _region(bad) if as_region else bad
    with pytest.raises(ValueError, match="contiguous when flattened"):
        _call(op, operands)


@pytest.mark.parametrize("op", ["broadcast", "mul_add_dst"])
@pytest.mark.parametrize(
    "shape,strides,bounds,count,offset",
    [
        ((2, 32), (32, 1), None, 64, 0),
        ((2, 32), (64, 1), [(1, 1), (8, 16)], 16, 72),
        ((1, 32), (999, 1), None, 32, 0),
        ((2, 1), (1, 999), None, 2, 0),
        ((32,), (1,), [(4, 16)], 16, 4),
        ((1,), (9,), None, 1, 0),
    ],
)
def test_flat_ops_accept_physically_contiguous_regions(op, shape, strides, bounds, count, offset):
    operands = [_region(_buffer(shape, strides), bounds) for _ in range(2 if op == "broadcast" else 3)]
    call = _call(op, operands)
    pointer = call.args[1] if op == "broadcast" else call.args[0]
    assert int(pointer.args[2]) == offset
    assert int(pointer.args[3]) == count


@pytest.mark.parametrize("op", ["broadcast", "mul_add_dst"])
@pytest.mark.parametrize("as_region", [False, True])
def test_flat_ops_preserve_elem_offset_once(op, as_region):
    operands = [_buffer((32,), (1,), elem_offset=16) for _ in range(2 if op == "broadcast" else 3)]
    if as_region:
        operands = [_region(buf, [(8, 16)]) for buf in operands]
    call = _call(op, operands)
    pointer = call.args[1] if op == "broadcast" else call.args[0]
    assert int(pointer.args[2]) == (24 if as_region else 16)


@pytest.mark.parametrize("operand", [0, 1])
@pytest.mark.parametrize("as_region", [False, True])
@pytest.mark.parametrize("leading", [2, "symbolic"])
def test_broadcast_rejects_nonunit_leading_axis(operand, as_region, leading):
    leading = tvm.tir.Var("batch", "int32") if leading == "symbolic" else leading
    operands = [_buffer((4, 32)), _buffer((4, 32))]
    bad = _buffer((leading, 4, 32))
    operands[operand] = _region(bad) if as_region else bad
    with pytest.raises(ValueError, match="only supports 1D or 2D"):
        ascend_tile.broadcast(*operands, axis=0)


@pytest.mark.parametrize(
    "shape,bounds,offset", [((1, 4, 32), None, 0), ((1, 1, 4, 32), None, 0), ((3, 4, 32), [(1, 1), (0, 4), (0, 32)], 128)]
)
def test_broadcast_accepts_unit_leading_axes(shape, bounds, offset):
    dst = _region(_buffer(shape), bounds)
    call = ascend_tile.broadcast(dst, _buffer((1, 32)), axis=0)
    assert int(call.args[1].args[2]) == offset
    assert [int(x) for x in call.args[3:]] == [2, 4, 32, 1, 32]


def _broadcast_slice():
    # Same T.Kernel/T.alloc_ub/T.copy pattern as existing broadcast coverage.
    @T.prim_func
    def main(A: T.Tensor((1, 32), "float32"), O: T.Tensor((4, 32), "float32")):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            src = T.alloc_ub((1, 32), "float32")
            dst = T.alloc_ub((2, 4, 32), "float32")
            T.copy(A, src)
            T.tile.broadcast(dst[1, :, :], src, axis=0)
            T.copy(dst[1, :, :], O)

    return main


def test_broadcast_single_plane_lowering_keeps_offset():
    with tvm.transform.PassContext(config={tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True}):
        source = tilelang.lower(_broadcast_slice(), target="ascendc", platform="A2").kernel_source
    assert "Broadcast<float, 2, 0, false>" in source
    assert "128" in source


def _offset_mul(dst, src0, src1, base):
    views = [tvm.tir.decl_buffer((32,), "float32", data=buf.data, elem_offset=base, scope="shared.ub") for buf in (dst, src0, src1)]
    return ascend_tile.mul_add_dst(*[_region(buf, [(8, 16)]) for buf in views])


def _mul_offset_kernel(base):
    @T.prim_func
    def main(A: T.Tensor((64,), "float32"), O: T.Tensor((64,), "float32")):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            src0 = T.alloc_ub((64,), "float32")
            src1 = T.alloc_ub((64,), "float32")
            dst = T.alloc_ub((64,), "float32")
            T.copy(A, src0)
            T.copy(A, src1)
            T.copy(A, dst)
            T.evaluate(_offset_mul(dst, src0, src1, base))
            T.copy(dst, O)

    return main


@pytest.mark.parametrize("base", [0, 16])
def test_mul_elem_offset_lowering(base):
    with tvm.transform.PassContext(config={tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True}):
        source = tilelang.lower(_mul_offset_kernel(base), target="ascendc", platform="A2").kernel_source
    mul_line = next(line for line in source.splitlines() if "AscendC::MulAddDst(" in line)
    assert f"dst[{base + 8}]" in mul_line, mul_line
    assert f"src0[{base + 8}]" in mul_line, mul_line
    assert f"src1[{base + 8}]" in mul_line, mul_line


@pytest.mark.parametrize("op", ["broadcast", "mul_add_dst"])
@pytest.mark.parametrize("as_region", [False, True])
@pytest.mark.parametrize("proven", [False, True])
def test_flat_ops_symbolic_explicit_stride(op, as_region, proven):
    n = tvm.tir.Var("n", "int32")
    stride = n + 1 - 1 if proven else tvm.tir.Var("pitch", "int32")
    operands = [_buffer((2, n), (stride, 1)) for _ in range(2 if op == "broadcast" else 3)]
    if as_region:
        operands = [_region(buf) for buf in operands]
    if proven:
        assert isinstance(_call(op, operands), tvm.tir.Call)
    else:
        with pytest.raises(ValueError, match="contiguous when flattened"):
            _call(op, operands)


@pytest.mark.parametrize("base", [0, 16])
def test_row_expand_region_preserves_elem_offset_once(base):
    dst, src = [_region(_buffer((4, 32), elem_offset=base), [(1, 2), (0, 32)]) for _ in range(2)]
    call = ascend_tile.row_expand_mul(dst, src, _buffer((2,)))
    assert int(call.args[1].args[2]) == base + 32
    assert int(call.args[2].args[2]) == base + 32


@pytest.mark.parametrize("op", ["broadcast", "mul_add_dst"])
@pytest.mark.parametrize("shape,strides,count", [((1, 32), (999, 1), 32), ((1,), (9,), 1)])
def test_flat_buffer_pointer_extent_ignores_singleton_stride(op, shape, strides, count):
    call = _call(op, [_buffer(shape, strides) for _ in range(2 if op == "broadcast" else 3)])
    pointer = call.args[1] if op == "broadcast" else call.args[0]
    assert int(pointer.args[3]) == count
