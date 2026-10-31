"""Focused contracts for the original TA opaque pass."""

from tilelang import tvm
from tilelang.ascend.a2a3 import transform

tirx = tvm.tirx


def _run(factory, body, buffers=(), attrs=None, config=None):
    func = tirx.PrimFunc([buffer.data for buffer in buffers], body, buffer_map={b.data: b for b in buffers})
    func = func.with_attr({"global_symbol": "main", **(attrs or {})})
    with tvm.transform.PassContext(config=config or {}):
        return factory()(tvm.IRModule.from_expr(func))["main"]


def _nodes(body, node_type):
    found = []
    tirx.stmt_functor.post_order_visit(body, lambda node: found.append(node) if isinstance(node, node_type) else None)
    return found


def test_opaque_block_allocation_predicate_pragmas_and_local_initializer():
    buffer = tirx.decl_buffer((1,), "int32", name="value", scope="local.var")
    pred = tirx.Var("pred", "bool")
    block = tirx.SBlock(
        [],
        [],
        [],
        "root",
        tirx.BufferStore(buffer, 7, [0]),
        alloc_buffers=[buffer],
        annotations={"pragma_z": 9, "pragma_a": "first", "tl.local_var_init": tirx.IntImm("int32", 3), "discard": 1},
    )
    after = _run(transform.AscendLowerOpaqueBlock, tirx.SBlockRealize([], pred, block))
    assert not _nodes(after.body, tirx.SBlock)
    allocations = _nodes(after.body, tirx.AllocBuffer)
    assert len(allocations) == 1
    assert allocations[0].buffer.data.same_as(buffer.data)
    assert int(allocations[0].annotations["tl.local_var_init"]) == 3
    assert int(after.attrs["tl.local_var_init"][buffer.data]) == 3
    assert _nodes(after.body, tirx.DeclBuffer)[0].buffer.same_as(buffer)
    condition = _nodes(after.body, tirx.IfThenElse)[0].condition
    assert condition.same_as(pred)
    assert after.body.attr_key == "pragma_a"
    assert after.body.body.attr_key == "pragma_z"


def test_opaque_unit_loop_substitution_and_thread_binding():
    index = tirx.Var("i", "int32")
    unit = tirx.For(index, 3, 1, tirx.ForKind.SERIAL, tirx.Evaluate(index))
    after = _run(transform.AscendLowerOpaqueBlock, unit)
    assert isinstance(after.body, tirx.Evaluate)
    assert int(after.body.value) == 3
    thread = tirx.IterVar(tvm.ir.Range(0, 4), index, tirx.IterVar.ThreadIndex, "blockIdx.x")
    launch = tirx.For(index, 0, 4, tirx.ForKind.THREAD_BINDING, tirx.Evaluate(index), thread_binding=thread)
    after = _run(transform.AscendLowerOpaqueBlock, launch)
    assert after.body.attr_key == "thread_extent"
    assert after.body.node.thread_tag == "blockIdx.x"
    assert int(after.body.value) == 4


def test_opaque_loop_annotations_and_existing_initializers():
    index = tirx.Var("i", "int32")
    local = tirx.decl_buffer((1,), "int32", scope="local.var")
    body = tirx.For(index, 0, 2, tirx.ForKind.SERIAL, tirx.Evaluate(index), annotations={"pragma_unroll_explicit": 0, "keep": 17})
    after = _run(transform.AscendLowerOpaqueBlock, body, attrs={"tl.local_var_init": {local.data: tirx.IntImm("int32", 5)}})
    assert after.body.attr_key == "pragma_unroll_explicit"
    assert int(after.body.body.annotations["keep"]) == 17
    assert int(after.attrs["tl.local_var_init"][local.data]) == 5


def test_opaque_preserves_storage_alignment_and_strided_allocation():
    buffer = tirx.decl_buffer((2, 3), "float32", scope="shared", strides=[8, 1])
    region = tirx.BufferRegion(buffer, [tvm.ir.Range(0, 2), tvm.ir.Range(0, 3)])
    inner = tirx.SBlock([], [], [region], "work", tirx.BufferStore(buffer, tirx.FloatImm("float32", 0), [0, 0]))
    root = tirx.SBlock([], [], [], "root", tirx.SBlockRealize([], True, inner), alloc_buffers=[buffer])
    func = tirx.PrimFunc([], tirx.SBlockRealize([], True, root)).with_attr("global_symbol", "main")
    schedule = tvm.s_tir.Schedule(tvm.IRModule.from_expr(func))
    schedule.storage_align(schedule.get_sblock("work"), 0, 0, 16, 1)
    after = transform.AscendLowerOpaqueBlock()(schedule.mod)["main"]
    allocation = _nodes(after.body, tirx.AllocBuffer)[0]
    # Preserve the physical allocation extent separately from the logical
    # declaration view, as old Allocate + DeclBuffer did.
    assert [int(v) for v in allocation.buffer.shape] == [2, 8]
    declared = _nodes(after.body, tirx.DeclBuffer)[0].buffer
    assert [int(v) for v in declared.shape] == [2, 3]
    alignment = allocation.annotations["buffer_dim_align"][0]
    assert tuple(int(v) for v in alignment) == (-1, 0, 16, 1)
