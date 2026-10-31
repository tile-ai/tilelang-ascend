"""Scope inference retains TA decisions before workspace/vid passes."""

import pytest

from tilelang import tvm
from tilelang.ascend.a2a3.layout import Layout, layout_str


tir = tvm.tirx


def _function(scopes, operation="unused", *, flat=False, explicit_layout=False):
    buffers = [tir.decl_buffer((16, 32), "float16", name=f"buffer{i}", scope=scope) for i, scope in enumerate(scopes)]
    if operation == "copy":
        regions = [tir.call_intrin("handle", tvm.ir.Op.get("tl.region"), buffer[0, 0], 1, 16, 32) for buffer in buffers]
        call = tir.call_intrin("handle", tvm.ir.Op.get("tl.ascend_copy"), *regions)
    elif operation == "unused":
        call = tir.const(0, "int32")
    else:
        name = {"vector": "vector_add", "cube": "gemm", "upper_cube": "MATMUL"}[operation]
        call = tir.call_extern("handle", name, *[buffer.access_ptr("rw") for buffer in buffers])
    body = tir.SeqStmt([tir.Evaluate(call), tir.BufferStore(buffers[0], buffers[0][0, 0], [0, 0])])
    annotations = {}
    if explicit_layout:
        annotations["layout_map"] = {buffers[0].data: Layout((16, 32), lambda i, j: [j, i], layout_tag=1)}
    if flat:
        # Allocate/body becomes a flat declaration followed by its lexical tail.
        body = tir.SeqStmt([*[tir.AllocBuffer(buffer) for buffer in buffers], body])
        allocations = []
    else:
        allocations = buffers
    block = tir.SBlock([], [], [], "compute", body, alloc_buffers=allocations, annotations=annotations)
    return tir.PrimFunc([], tir.SBlockRealize([], True, block)), buffers


def _run(func):
    factory = tvm.ffi.get_global_func("tl.transform.A2A3InferAllocScope")
    return factory()(tvm.IRModule({"main": func}))["main"]


def _allocations(func):
    found = []

    def visit(node):
        if isinstance(node, tir.SBlock):
            found.extend(node.alloc_buffers)
        elif isinstance(node, tir.AllocBuffer):
            found.append(node.buffer)

    tir.stmt_functor.post_order_visit(func.body, visit)
    return found


@pytest.mark.parametrize(
    ("scopes", "operation", "expected"),
    [
        (["local.fragment"], "unused", ["wmma.accumulator"]),
        (["shared"], "unused", ["shared"]),
        (["shared", "shared"], "vector", ["shared.ub", "shared.ub"]),
        (["local.fragment"] * 3, "cube", ["wmma.matrix_a", "wmma.matrix_b", "wmma.accumulator"]),
        (["local.fragment"] * 3, "upper_cube", ["wmma.accumulator"] * 3),
        (["shared", "wmma.matrix_a"], "copy", ["shared.l1", "wmma.matrix_a"]),
        (["shared", "wmma.matrix_b"], "copy", ["shared.l1", "wmma.matrix_b"]),
        (["shared", "shared"], "copy", ["shared.ub", "shared.ub"]),
        (["shared.ub", "shared"], "copy", ["shared.ub", "shared.ub"]),
        (["shared.ub", "shared.l1"], "copy", ["shared.ub", "shared.l1"]),
        (["shared", "shared", "shared"], "cube", ["shared.l1"] * 3),
        (["global", "local", "shared.ub"], "vector", ["global", "local", "shared.ub"]),
    ],
)
def test_source_scope_decisions(scopes, operation, expected):
    func, _ = _function(scopes, operation)
    assert [buffer.scope() for buffer in _allocations(_run(func))] == expected


@pytest.mark.parametrize("flat", [False, True])
def test_rewritten_allocation_load_store_and_access_pointer_agree(flat):
    func, original = _function(["shared"], "vector", flat=flat)
    result = _run(func)
    allocated = _allocations(result)[0]
    assert allocated.scope() == "shared.ub"
    assert not allocated.data.same_as(original[0].data)
    handles = []

    def visit(node):
        if isinstance(node, (tir.BufferLoad, tir.BufferStore)):
            handles.append(node.buffer.data)
        if isinstance(node, tir.Call) and node.op.name == "tirx.tvm_access_ptr":
            handles.append(node.args[1])

    tir.stmt_functor.post_order_visit(result.body, visit)
    assert len(handles) == 3
    assert all(handle.same_as(allocated.data) for handle in handles)


def test_default_l1_layout_uses_local_tag_bridge():
    func, _ = _function(["shared.l1"])
    result = _run(func)
    block = result.body.block
    allocated = block.alloc_buffers[0]
    assert layout_str(block.annotations["layout_map"][allocated.data]) == "layout::zN"


def test_explicit_layout_retains_the_original_ta_replacement_behavior():
    func, original = _function(["shared.l1"], explicit_layout=True)
    result = _run(func)
    block = result.body.block
    layouts = block.annotations["layout_map"]
    # TA rebuilds every L1 handle even if its scope stays L1. It retains the
    # user's old key, then inserts a default for the new key; do not silently
    # fix that original algorithm during the TIRX port.
    assert len(layouts) == 2
    assert layout_str(layouts[original[0].data]) == "layout::ColumnMajor"
    assert layout_str(layouts[block.alloc_buffers[0].data]) == "layout::zN"


def test_unchanged_non_l1_explicit_layout_remains_the_only_entry():
    func, original = _function(["shared.ub"], explicit_layout=True)
    result = _run(func)
    layouts = result.body.block.annotations["layout_map"]
    assert len(layouts) == 1
    assert layout_str(layouts[original[0].data]) == "layout::ColumnMajor"


@pytest.mark.parametrize("valid_first", [True, False])
def test_partial_region_copy_keeps_unknown_scope_inference_conservative(valid_first):
    first = tir.decl_buffer((16, 32), "float16", name="first", scope="shared")
    second = tir.decl_buffer((16, 32), "float16", name="second", scope="shared")
    region = tir.call_intrin("handle", tvm.ir.Op.get("tl.region"), (first if valid_first else second)[0, 0], 1, 16, 32)
    first_arg = region if valid_first else first.access_ptr("r")
    second_arg = second.access_ptr("w") if valid_first else region
    call = tir.call_intrin("handle", tvm.ir.Op.get("tl.ascend_copy"), first_arg, second_arg)
    block = tir.SBlock([], [], [], "compute", tir.Evaluate(call), alloc_buffers=[first, second])
    func = tir.PrimFunc([], tir.SBlockRealize([], True, block))
    # Source sees no complete region pair and cannot infer the first buffer's
    # role. A default-constructed C++ Var is a real v, not a missing pointer.
    assert [buffer.scope() for buffer in _allocations(_run(func))] == ["shared", "shared"]
