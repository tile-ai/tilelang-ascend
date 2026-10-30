"""Ascend UB planning must reject excess demand without crashing the compiler."""

import pytest
import tilelang
import tilelang.language as T
import tvm
from tvm import tir

# ASCEND_SHARED_MEM_SIZE reserves 256 bytes from the 192 KiB UB.
# Keep the boundary explicit: these tests exercise this pass's current limit,
# not a runtime device capacity query.
UB_CAPACITY_BYTES = 192 * 1024 - 256
FLOAT32_BYTES = 4
CAPACITY_ELEMENTS = UB_CAPACITY_BYTES // FLOAT32_BYTES


def _plan(sizes, simultaneous=False):
    buffers = []
    for index, size in enumerate(sizes):
        pointer = tir.Var(f"b{index}", tvm.ir.PointerType(tvm.ir.PrimType("float32"), "shared.ub"))
        buffers.append(tir.decl_buffer((size,), "float32", data=pointer))
    stores = [tir.BufferStore(buffer, tir.const(1, "float32"), [0]) for buffer in buffers]
    if simultaneous:
        reads = [tir.Evaluate(tir.BufferLoad(buffer, [0])) for buffer in buffers]
        body = tir.SeqStmt(stores + reads)
        for buffer, size in reversed(list(zip(buffers, sizes))):
            body = tir.Allocate(buffer.data, "float32", [size], tir.const(True, "bool"), body)
    else:
        allocations = [
            tir.Allocate(buffer.data, "float32", [size], tir.const(True, "bool"), store)
            for buffer, size, store in zip(buffers, sizes, stores)
        ]
        body = allocations[0] if len(allocations) == 1 else tir.SeqStmt(allocations)
    function = tir.PrimFunc([], body).with_attr("global_symbol", "main")
    with tvm.transform.PassContext(config={"tl.ascend_memory_planning": True}):
        planned = tilelang.transform.AscendMemoryPlanning()(tvm.IRModule({"main": function}))
    return planned["main"]


def _ranges(function, expected_sizes):
    addresses = function.attrs["address_map"]
    sizes = function.attrs["size_map"]
    assert len(addresses) == len(expected_sizes)
    result = {}
    for buffer, address in addresses.items():
        offset, size = int(address), int(sizes[buffer])
        assert offset >= 0 and offset % 32 == 0
        assert size == expected_sizes[int(buffer.name[1:])] * FLOAT32_BYTES
        assert offset + size <= UB_CAPACITY_BYTES
        result[buffer.name] = (offset, offset + size)
    return result


@pytest.mark.parametrize("size", [16, CAPACITY_ELEMENTS])
def test_allocation_within_capacity_has_valid_address(size):
    _ranges(_plan([size]), [size])


@pytest.mark.parametrize("size", [CAPACITY_ELEMENTS + 1, 65536])
def test_allocation_exceeding_capacity_reports_error(size):
    with pytest.raises(tvm.error.TVMError, match="Memory allocation failed"):
        _plan([size])


def test_live_allocations_exceeding_capacity_report_error():
    with pytest.raises(tvm.error.TVMError, match="Memory allocation failed"):
        _plan([30000, 30000], simultaneous=True)


def test_live_allocations_do_not_overlap():
    ranges = _ranges(_plan([16000, 16000], simultaneous=True), [16000, 16000])
    a, b = ranges["b0"], ranges["b1"]
    assert a[1] <= b[0] or b[1] <= a[0]


def test_expired_allocation_space_is_reused():
    ranges = _ranges(_plan([40000, 40000]), [40000, 40000])
    a, b = ranges["b0"], ranges["b1"]
    assert max(a[0], b[0]) < min(a[1], b[1])


def test_free_tail_can_expand_within_capacity():
    ranges = _ranges(_plan([30000, 45000]), [30000, 45000])
    a, b = ranges["b0"], ranges["b1"]
    assert max(a[0], b[0]) < min(a[1], b[1])
    assert b[1] > a[1]


@pytest.mark.parametrize("target", [None, "ascendc", "pto"], ids=["default", "ascendc", "pto"])
@pytest.mark.parametrize("size", [16, 65536], ids=["within_capacity", "over_capacity"])
def test_capacity_check_in_lowering_pipeline(target, size):
    @T.prim_func
    def main(A: T.Tensor((size,), "float32"), B: T.Tensor((size,), "float32")):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((size,), "float32")
            T.copy(A[:], a_ub)
            T.copy(a_ub, B[:])

    target_kwargs = {} if target is None else {"target": target}
    with tvm.transform.PassContext(config={"tl.ascend_memory_planning": True}):
        if size * FLOAT32_BYTES > UB_CAPACITY_BYTES:
            with pytest.raises(tvm.error.TVMError, match="Memory allocation failed"):
                tilelang.lower(main, **target_kwargs)
        else:
            artifact = tilelang.lower(main, **target_kwargs)
            assert "a_ub" in artifact.kernel_source
