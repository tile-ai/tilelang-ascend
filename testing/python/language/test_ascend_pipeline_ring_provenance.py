# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""CPU checks for injector physical rings across the real lowering pipeline.

These tests run IR transformations only. They do not compile AscendC or execute
a kernel, and do not require the optional Z3 proof service.
"""

from unittest.mock import patch

import pytest

import tilelang
from tilelang import language as T
from tilelang import tvm
from tilelang.engine.phase import LowerAndLegalize, OptimizeForTarget

tir = tvm.tir
_PROVENANCE = "tl.ascend_physical_ring_provenance"
_CLOCK = "tl.ascend_physical_ring_clock"


def _pipelined_abs(trip_count: int, num_stages: int = 2):
    # Same complete-tile Vector pattern as examples/mhc_post/example_mhc_post.py
    # and the verified vec_abs factory in the elementwise language tests.
    width = 64

    @T.prim_func
    def main(
        A: T.Tensor((trip_count, width), "float32"),
        O: T.Tensor((2, trip_count, width), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (_cid, vid):
            source = T.alloc_ub((width,), "float32")
            result = T.alloc_ub((width,), "float32")
            with T.Scope("V"):
                for tile in T.Pipelined(trip_count, num_stages=num_stages):
                    T.copy(A[tile, :], source)
                    T.tile.abs(result, source)
                    T.copy(result, O[vid, tile, :])

    return main


def _nodes(statement, node_type):
    result = []
    tir.stmt_functor.post_order_visit(statement, lambda node: result.append(node) if isinstance(node, node_type) else None)
    return result


def _clocks(function):
    return [node for node in _nodes(function.body, tir.AttrStmt) if node.attr_key == _CLOCK]


def _pre_sync(program, *, backend="upstream", memory_planning=False):
    snapshots = {}

    @tvm.instrument.pass_instrument
    class CaptureProvenance:
        def run_after_pass(self, mod, info):
            if _PROVENANCE in mod["main"].attrs:
                snapshots[info.name] = mod["main"]

    captured = []

    @tvm.transform.module_pass(opt_level=0)
    def capture(mod, _context):
        captured.append(mod["main"])
        return mod

    configs = {
        "tl.ascend_sync_backend": backend,
        "tl.ascend_auto_sync": True,
        "tl.ascend_auto_sync_vs": True,
        "tl.ascend_memory_planning": memory_planning,
        "tl.ascend_auto_cv_combine": False,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    }
    # Match engine/lower.py: the host target kind is LLVM; the model selects
    # the Ascend device code generator. A synthetic C target hid ABI gates.
    target = tvm.target.Target({"kind": "llvm", "model": "ascendc"})
    with tvm.transform.PassContext(config=configs, instruments=[CaptureProvenance()]), target:
        module = LowerAndLegalize(tvm.IRModule({"main": program}), target)
        with (
            patch.object(tilelang.transform, "AscendInsertSync", return_value=capture),
            patch.object(tilelang.transform, "AscendSyncInsert", return_value=capture),
            patch.object(tilelang.transform, "AscendSyncInsertVS", return_value=capture),
            patch("tilelang.utils.target.check_npu_availability", return_value=True),
        ):
            OptimizeForTarget(module, target, "A3")
    return captured[-1], snapshots


@pytest.mark.parametrize("num_stages,trip_count", [(2, 4), (2, 5), (3, 7), (3, 8)])
@pytest.mark.parametrize("memory_planning", [False, True])
def test_real_pipelined_vector_ring_survives_lowering(num_stages, trip_count, memory_planning):
    function, snapshots = _pre_sync(_pipelined_abs(trip_count, num_stages), memory_planning=memory_planning)
    assert {
        "tl.InjectSoftwarePipeline",
        "tl.AscendLowerOpaqueBlock",
        "tl.Flatten2DBuffer",
        "tl.FlattenBuffer",
        "tl.AscendStorageRewrite",
        "tl.AscendMemoryPlanning",
    }.issubset(snapshots)
    for stage, snapshot in snapshots.items():
        records = snapshot.attrs[_PROVENANCE]
        assert len(records) == 1, stage
        for logical_id, record in records.items():
            assert len(record) == 7, stage
            assert [int(record[index]) for index in (0, 1, 3, 4, 5, 6)] == [
                1,
                num_stages,
                0,
                2048,
                32,
                0,
            ], stage
            assert int(snapshot.attrs["buffer_versions"][logical_id]) == num_stages, stage
            assert all(clock.node[0].same_as(logical_id) for clock in _clocks(snapshot)), stage

    assert not _nodes(function.body, tir.IfThenElse)
    assert len(_clocks(function)) >= 4
    logical_id, record = next(iter(function.attrs[_PROVENANCE].items()))
    allocations = _nodes(function.body, tir.Allocate)
    allocation = next(node for node in allocations if node.buffer_var.same_as(record[2]))
    assert int(allocation.extents[0]) == num_stages * 64
    assert int(function.attrs["size_map"][record[2]]) == num_stages * 256
    assert sum(int(value) == 1 for value in function.attrs["buffer_versions"].values()) == 1


def test_default_legacy_pipeline_does_not_produce_new_metadata():
    function, snapshots = _pre_sync(_pipelined_abs(4), backend="legacy")
    assert _PROVENANCE not in function.attrs
    assert not snapshots
    assert not _clocks(function)
    assert sorted(int(value) for value in function.attrs["buffer_versions"].values()) == [1, 2]


def _record(root, *, base_bits=0):
    return [
        tir.IntImm("int32", 1),
        tir.IntImm("int32", 2),
        root,
        tir.IntImm("int64", base_bits),
        tir.IntImm("int64", 2048),
        tir.IntImm("int64", 32),
        tir.IntImm("int32", 0),
    ]


def _inject_l1_pipeline(loop_min=0):
    ring = tir.decl_buffer((128, 200), "float16", name="ring", scope="shared.l1")
    elements = 128 * 200
    source = tir.decl_buffer(((loop_min + 4) * elements,), "float16", name="input")
    output = tir.decl_buffer(((loop_min + 4) * elements,), "float16", name="output")
    index = tir.Var("tile", "int32")
    statements = tir.SeqStmt(
        [
            tir.Evaluate(
                tir.call_extern(
                    "handle",
                    "copy_gm_to_l1",
                    source.access_ptr("r", offset=index * elements, extent=elements),
                    ring.access_ptr("w"),
                )
            ),
            tir.Evaluate(
                tir.call_extern(
                    "handle",
                    "copy_l1_to_gm",
                    ring.access_ptr("r"),
                    output.access_ptr("w", offset=index * elements, extent=elements),
                )
            ),
        ]
    )
    body = tir.BlockRealize(
        [],
        tir.const(True, "bool"),
        tir.Block([], [], [], "ring_body", statements, alloc_buffers=[ring]),
    )
    loop = tir.For(
        index,
        loop_min,
        4,
        tir.ForKind.SERIAL,
        body,
        annotations={
            "software_pipeline_stage": [0, 1],
            "software_pipeline_order": [0, 1],
        },
    )
    function = tir.PrimFunc(
        [source.data, output.data],
        loop,
        buffer_map={source.data: source, output.data: output},
    ).with_attr("global_symbol", "main")
    with tvm.transform.PassContext(config={"tl.ascend_sync_backend": "upstream"}):
        result = tilelang.transform.InjectSoftwarePipeline()(tvm.IRModule({"main": function}))
    return result["main"], ring


def test_injector_records_padded_physical_stride():
    function, ring = _inject_l1_pipeline()
    record = function.attrs[_PROVENANCE][ring.data]
    # The existing L1 rewrite pads 200 fp16 columns to 208 before adding B.
    assert int(record[1]) == int(function.attrs["buffer_versions"][ring.data]) == 2
    assert int(record[4]) == 128 * 208 * 16
    assert int(record[4]) > 128 * 200 * 16
    allocation = next(
        buffer for block in _nodes(function.body, tir.Block) for buffer in block.alloc_buffers if buffer.data.same_as(record[2])
    )
    assert [int(extent) for extent in allocation.shape] == [2, 128, 208]


def test_nonzero_min_keeps_real_version_count_without_false_provenance():
    function, ring = _inject_l1_pipeline(loop_min=3)
    assert int(function.attrs["buffer_versions"][ring.data]) == 2
    assert _PROVENANCE not in function.attrs
    assert not _clocks(function)


def test_storage_merge_preserves_logical_ids_and_real_bit_offsets():
    first = tir.decl_buffer((128,), "float32", name="first", scope="shared.l0a")
    second = tir.decl_buffer((128,), "float32", name="second", scope="shared.l0a")
    touch = tir.Evaluate(tir.call_extern("handle", "touch_both", first.access_ptr("rw"), second.access_ptr("rw")))
    body = tir.AttrStmt(tvm.runtime.convert([second.data]), _CLOCK, 0, touch)
    for buffer in (second, first):
        body = tir.Allocate(buffer.data, buffer.dtype, [128], tir.const(True, "bool"), body)
    body = tir.AttrStmt(tir.IterVar(tvm.ir.Range(0, 1), "block", 1, "blockIdx.x"), "thread_extent", 1, body)
    function = tir.PrimFunc([], body).with_attr(_PROVENANCE, {first.data: _record(first.data), second.data: _record(second.data)})
    result = tilelang.transform.AscendStorageRewrite(is_npu=True)(tvm.IRModule({"main": function}))["main"]

    records = result.attrs[_PROVENANCE]
    assert set(records.keys()) == {first.data, second.data}
    roots = [record[2] for record in records.values()]
    assert roots[0].same_as(roots[1])
    assert sorted(int(record[3]) for record in records.values()) == [0, 4096]
    assert all(int(record[4]) == 2048 for record in records.values())
    assert _clocks(result)[0].node[0].same_as(second.data)
    pointers = [node for node in _nodes(result.body, tir.Call) if node.op.same_as(tvm.ir.Op.get("tir.tvm_access_ptr"))]
    assert sorted(int(pointer.args[2]) * 32 for pointer in pointers) == [0, 4096]


def test_pointer_retyping_updates_physical_root_without_scaling_bits():
    backing = tir.decl_buffer((128,), "int32", name="backing", scope="shared.ub")
    alias = tir.Var("alias", tvm.ir.PointerType(tvm.ir.PrimType("int32"), "shared.ub"))
    view = tir.decl_buffer((128,), "float32", name="view", data=alias, scope="shared.ub")
    body = tir.AttrStmt(
        tvm.runtime.convert([alias]),
        _CLOCK,
        0,
        tir.Evaluate(tir.call_extern("handle", "touch", view.access_ptr("rw"))),
    )
    body = tir.LetStmt(alias, backing.data, body)
    body = tir.Allocate(backing.data, backing.dtype, [128], tir.const(True, "bool"), body)
    function = tir.PrimFunc([], body).with_attr(_PROVENANCE, {alias: _record(alias, base_bits=32)})

    result = tilelang.transform.AscendStorageRewrite(is_npu=True)(tvm.IRModule({"main": function}))["main"]

    record = result.attrs[_PROVENANCE][alias]
    assert not record[2].same_as(alias)
    assert str(record[2].type_annotation.element_type.dtype) == "float32"
    assert [int(record[index]) for index in (3, 4, 5)] == [32, 2048, 32]
    assert _clocks(result)[0].node[0].same_as(alias)


def test_clock_shell_is_transparent_to_memory_planning():
    program, _ = _pre_sync(_pipelined_abs(5))
    stripped_body = tir.stmt_functor.ir_transform(
        program.body,
        lambda node: node.body if isinstance(node, tir.AttrStmt) and node.attr_key == _CLOCK else None,
        None,
        ["tir.AttrStmt"],
    )
    attrs = {key: value for key, value in program.attrs.items() if key not in ("address_map", "size_map")}
    functions = [
        tir.PrimFunc(
            program.params,
            body,
            program.ret_type,
            program.buffer_map,
            tvm.ir.make_node("DictAttrs", **attrs),
        )
        for body in (program.body, stripped_body)
    ]
    with tvm.transform.PassContext(config={"tl.ascend_memory_planning": True}):
        results = [tilelang.transform.AscendMemoryPlanning()(tvm.IRModule({"main": function}))["main"] for function in functions]

    def allocation_plan(function):
        # IRModule construction may alpha-rename the two functions' pointer
        # identities. This fixture has unique allocation names, so compare
        # each named allocation's actual address and size after planning.
        addresses = function.attrs["address_map"]
        sizes = function.attrs["size_map"]
        plan = {variable.name: (int(address), int(sizes[variable])) for variable, address in addresses.items()}
        assert len(plan) == len(addresses)
        return plan

    assert allocation_plan(results[0]) == allocation_plan(results[1])
