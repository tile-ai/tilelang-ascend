"""Native S1 regressions for legacy Ascend offsets and physical buffer contracts."""

import os
import re

import pytest

import tilelang
from tilelang import language as T
from tilelang import tvm
from tilelang.language import ascend_tile


PLATFORMS = [os.environ.get("TL_ASCEND_MIGRATION_PLATFORM", "A3")]


def make_copy_offset_kernel(
    src_offset: int = 32,
    dst_offset: int = 16,
    src_start: int = 3,
    dst_start: int = 5,
    size: int = 64,
) -> tvm.tir.PrimFunc:
    """Copy through UB while keeping nonzero GM buffer-view bases explicit."""
    source = f"""@T.prim_func
def main(A: T.handle, O: T.handle):
    a = T.match_buffer(A, (128,), "float32", elem_offset={src_offset})
    o = T.match_buffer(O, (128,), "float32", elem_offset={dst_offset})
    with T.Kernel(1, is_npu=True) as (cid, vid):
        u = T.alloc_ub(({size},), "float32")
        with T.Scope("V"):
            T.copy(a[{src_start}:{src_start + size}], u)
            T.copy(u, o[{dst_start}:{dst_start + size}])
"""
    return tvm.script.from_source(source, extra_vars={"T": T})


def make_atomic_offset_kernel(
    dst_offset: int = 16,
    src_offset: int = 16,
    src_start: int = 0,
    dst_start: int = 0,
    size: int = 64,
) -> tvm.tir.PrimFunc:
    """Atomic UB-to-GM addition from an explicitly offset UB view."""
    source = f"""@T.prim_func
def main(A: T.Tensor((128,), "float32"), O: T.handle):
    o = T.match_buffer(O, (128,), "float32", elem_offset={dst_offset})
    with T.Kernel(1, is_npu=True) as (cid, vid):
        u = T.alloc_ub((128,), "float32")
        v = T.decl_buffer(({size},), "float32", data=u.data,
                          elem_offset={src_offset + src_start}, scope="shared.ub")
        with T.Scope("V"):
            T.copy(A, u)
            T.barrier_all()
            T.tile.atomic_add(o[{dst_start}], v)
"""
    return tvm.script.from_source(source, extra_vars={"T": T})


def _lower_source(program: tvm.tir.PrimFunc, platform: str) -> str:
    return tilelang.lower(program, target="ascendc", platform=platform).kernel_source


@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize("src_offset,dst_offset", [(0, 0), (32, 0), (0, 16), (32, 16)])
@pytest.mark.parametrize("src_start,dst_start", [(0, 0), (3, 5)])
def test_copy_elem_offset(platform, src_offset, dst_offset, src_start, dst_start):
    source = _lower_source(make_copy_offset_kernel(src_offset, dst_offset, src_start, dst_start), platform)
    load = re.search(r"copy_gm_to_ub<[^>]+>\(u\[0\], a\[(\d+)\]", source)
    store = re.search(r"copy_ub_to_gm<[^>]+>\(o\[(\d+)\], u\[0\]", source)
    assert load and store, source
    assert int(load[1]) == src_offset + src_start
    assert int(store[1]) == dst_offset + dst_start


@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize("src_offset,dst_offset", [(0, 0), (16, 0), (0, 16), (16, 16)])
@pytest.mark.parametrize("src_start,dst_start", [(0, 0), (3, 5)])
def test_atomic_elem_offset(platform, src_offset, dst_offset, src_start, dst_start):
    source = _lower_source(make_atomic_offset_kernel(dst_offset, src_offset, src_start, dst_start), platform)
    call = re.search(r"atomic_add_ub_to_gm<[^>]+>\(o\[(\d+)\], v\[(\d+)\]", source)
    assert call, source
    assert int(call[1]) == dst_offset + dst_start
    assert int(call[2]) == src_offset + src_start


@pytest.mark.parametrize("platform", PLATFORMS)
def test_contiguous_view_offsets(platform):
    """Nested contiguous views retain their accumulated base and final slice."""
    source = """@T.prim_func
def main(A: T.handle, O: T.Tensor((64,), "float32")):
    a = T.match_buffer(A, (128,), "float32", elem_offset=16)
    with T.Kernel(1, is_npu=True) as (cid, vid):
        first = T.decl_buffer((112,), "float32", data=a.data,
                              elem_offset=a.elem_offset + 8)
        second = T.decl_buffer((104,), "float32", data=first.data,
                               elem_offset=first.elem_offset + 4)
        u = T.alloc_ub((64,), "float32")
        with T.Scope("V"):
            T.copy(second[3:67], u)
            T.copy(u, O)
"""
    program = tvm.script.from_source(source, extra_vars={"T": T})
    code = _lower_source(program, platform)
    call = re.search(r"copy_gm_to_ub<[^>]+>\(u\[0\], \w+\[(\d+)\]", code)
    assert call, code
    assert int(call[1]) == 16 + 8 + 4 + 3


@pytest.mark.parametrize("elem_offset", [0, 16])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("helper", ["_get_buffer_info", "_handle_buffer_region", "_handle_buffer_region_2d"])
def test_tile_region_elem_offset(elem_offset, strided, helper):
    buffer = tvm.tir.decl_buffer(
        (8, 16) if strided else (128,),
        "float32",
        scope="shared.ub",
        elem_offset=elem_offset,
        strides=[32, 1] if strided else None,
    )
    ranges = [(1, 2), (3, 8)] if strided else [(3, 64)]
    region = tvm.tir.BufferRegion(buffer, [tvm.ir.Range.from_min_extent(start, extent) for start, extent in ranges])
    pointer, _ = getattr(ascend_tile, helper)(region, "r")
    assert int(tvm.arith.Analyzer().simplify(pointer.args[2])) == elem_offset + (35 if strided else 3)


def make_tile_fill_offset_kernel(region_start: int = 3) -> tvm.tir.PrimFunc:
    """Fill an offset UB region and preserve sentinels outside the region."""
    source = f"""@T.prim_func
def main(O: T.Tensor((128,), "float32")):
    with T.Kernel(1, is_npu=True) as (cid, vid):
        u = T.alloc_ub((128,), "float32")
        v = T.decl_buffer((112,), "float32", data=u.data,
                          elem_offset=16, scope="shared.ub")
        with T.Scope("V"):
            T.tile.fill(u, -99.0)
            T.tile.fill(v[{region_start}:{region_start + 64}], 1.0)
            T.barrier_all()
            T.copy(u, O)
"""
    return tvm.script.from_source(source, extra_vars={"T": T})


@pytest.mark.parametrize("platform", PLATFORMS)
def test_tile_fill_view_elem_offset(platform):
    code = _lower_source(make_tile_fill_offset_kernel(), platform)
    calls = re.findall(r"AscendC::Duplicate<[^>]+>\(v\[(\d+)\]", code)
    assert [int(offset) for offset in calls] == [0, 19], code


@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize("k_l1", [200, 152])
def test_l1_pipeline_physical_versions(platform, k_l1):
    """Lower the existing K-tail/Parallel regression without invoking Bisheng."""
    source = f"""@T.prim_func
def main(A: T.Tensor((256, {2 * k_l1}), "float16"),
         B: T.Tensor(({2 * k_l1}, 256), "float16"),
         C: T.Tensor((256, 256), "float16")):
    with T.Kernel(4, is_npu=True) as (cid, vid):
        bx = cid // 2
        by = cid % 2
        a_l1 = T.alloc_shared((128, {k_l1}), "float16")
        b_l1 = T.alloc_shared(({k_l1}, 128), "float16")
        c_l0 = T.alloc_L0C((128, 128), "float32")
        with T.Scope("C"):
            for k in T.Pipelined(2, num_stages=2):
                T.barrier_all()
                T.copy(A[bx * 128, k * {k_l1}], a_l1)
                T.copy(B[k * {k_l1}, by * 128], b_l1)
                T.gemm_v0(a_l1, b_l1, c_l0, init=(k == 0))
                T.barrier_all()
            for i, j in T.Parallel(128, 128):
                C[bx * 128 + i, by * 128 + j] = c_l0[i, j]
"""
    program = tvm.script.from_source(source, extra_vars={"T": T})
    with tvm.transform.PassContext(config={"tl.ascend_auto_sync": True, "tl.ascend_memory_planning": True}):
        artifact = tilelang.lower(program, target="ascendc", platform=platform)
    physical_version = 128 * ((k_l1 + 15) // 16 * 16)
    allocations = {}

    def collect(node):
        if isinstance(node, tvm.tir.Allocate) and node.buffer_var.name in ("a_l1", "b_l1"):
            allocations[node.buffer_var.name] = int(tvm.arith.Analyzer().simplify(node.extents[0]))

    for function in artifact.device_mod.functions.values():
        tvm.tir.stmt_functor.post_order_visit(function.body, collect)
    assert allocations == {"a_l1": 2 * physical_version, "b_l1": 2 * physical_version}
    # Each independent version must be clearable at its padded physical base.
    assert f"a_l1[{physical_version}]" in artifact.kernel_source
    assert f"b_l1[{physical_version}]" in artifact.kernel_source


@pytest.mark.parametrize("auto", [False, True])
@pytest.mark.parametrize("preallocated", [False, True])
def test_legacy_memory_maps(auto, preallocated):
    """Inspect the legacy planner directly, including deliberate byte aliases."""
    buffers = [tvm.tir.decl_buffer((64,), "float32", name=name, scope="shared.ub") for name in ("a", "b")]
    body = tvm.tir.SeqStmt([tvm.tir.Evaluate(tvm.tir.call_extern("handle", "AscendC::Duplicate", b.access_ptr("w"))) for b in buffers])
    for buffer in buffers:
        body = tvm.tir.Allocate(buffer.data, buffer.dtype, [64], tvm.tir.const(True, "bool"), body)
    function = tvm.tir.PrimFunc([], body).with_attr("global_symbol", "main")
    function = function.with_attr("logic_buffer_shapes", {buffer.data: [64] for buffer in buffers})
    # User-specified byte bases remain a supported legacy alias contract.
    if preallocated:
        function = function.with_attr("address_map", {buffer.data: 0 for buffer in buffers})
    with tvm.transform.PassContext(config={"tl.ascend_memory_planning": True} if auto else {}):
        result = tilelang.transform.AscendMemoryPlanning()(tvm.IRModule({"main": function}))["main"]
    for buffer in buffers:
        if preallocated:
            assert int(result.attrs["address_map"][buffer.data]) == 0
        assert int(result.attrs["size_map"][buffer.data]) == 64 * 4
    if not auto and not preallocated:
        offsets = [int(result.attrs["address_map"][buffer.data]) for buffer in buffers]
        assert sorted(offsets) == [0, 256]
