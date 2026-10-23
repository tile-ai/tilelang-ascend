"""Flat fill rejects padded row windows before they become contiguous writes."""

import pytest

import tilelang
from tilelang import tvm
import tilelang.language as T
from tilelang.language import ascend_tile


def _region(buffer, starts, extents):
    return tvm.tir.BufferRegion(
        buffer,
        [tvm.ir.Range.from_min_extent(start, extent) for start, extent in zip(starts, extents)],
    )


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("column", [0, 8])
def test_fill_rejects_padded_multirow_window(dtype, column):
    buffer = tvm.tir.decl_buffer((64, 40), dtype, scope="shared.ub")
    with pytest.raises(ValueError, match="contiguous when flattened"):
        ascend_tile.fill(_region(buffer, (0, column), (64, 32)), 1)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize(
    "shape,starts,extents,offset,count",
    [
        ((64,), (8,), (24,), 8, 24),
        ((64, 40), (3, 0), (1, 32), 120, 32),
        ((64, 40), (3, 8), (1, 32), 128, 32),
        ((64, 40), (3, 0), (8, 40), 120, 320),
    ],
)
def test_fill_contiguous_region_keeps_offset_and_count(dtype, shape, starts, extents, offset, count):
    buffer = tvm.tir.decl_buffer(shape, dtype, scope="shared.ub")
    call = ascend_tile.fill(_region(buffer, starts, extents), 1)
    assert int(call.args[1].args[2]) == offset
    assert int(call.args[1].args[3]) == count
    assert int(call.args[3]) == count


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_fill_dense_buffer_keeps_count(dtype):
    buffer = tvm.tir.decl_buffer((64, 40), dtype, scope="shared.ub")
    call = ascend_tile.fill(buffer, 1)
    assert int(call.args[1].args[2]) == 0
    assert int(call.args[1].args[3]) == 2560
    assert int(call.args[3]) == 2560


def _fill_window_kernel(dtype="float32", rowwise=False):
    @T.prim_func
    def main(O: T.Tensor((64, 40), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            dst = T.alloc_ub((64, 40), dtype)
            T.tile.fill(dst, 7)
            if rowwise:
                for row in T.serial(64):
                    T.tile.fill(dst[row : row + 1, 0:32], 1)
            else:
                T.tile.fill(dst[0:64, 0:32], 1)
            T.copy(dst, O)

    return main


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_fill_padded_window_fails_before_lowering(dtype, capfd):
    with pytest.raises(tvm.error.DiagnosticError), tvm.transform.PassContext(config={tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True}):
        tilelang.lower(_fill_window_kernel(dtype), target="ascendc", platform="A2")
    assert "contiguous when flattened" in capfd.readouterr().err


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_fill_rowwise_window_lowers(dtype):
    with tvm.transform.PassContext(config={tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True}):
        source = tilelang.lower(_fill_window_kernel(dtype, rowwise=True), target="ascendc", platform="A2").kernel_source
    assert "Duplicate" in source
    assert "40" in source
