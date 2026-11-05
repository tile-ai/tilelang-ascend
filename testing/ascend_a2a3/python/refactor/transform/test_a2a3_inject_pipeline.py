"""Complete TA software-pipeline rewriting, allocation and rejection boundaries."""

import math
import operator

import pytest
from tilelang import tvm


tir = tvm.tirx


def _injection_case(
    extent=4,
    start=0,
    stages=(0, 1),
    order=(0, 1),
    *,
    dtype="float32",
    shape=(16,),
    scope="shared.ub",
    kind="stores",
    strides=None,
    annotations=None,
):
    source = tir.decl_buffer((1024,), dtype, name="source")
    output = tir.decl_buffer((1024,), dtype, name="output")
    local = tir.decl_buffer(shape, dtype, name="local", scope=scope, strides=strides)
    k = tir.Var("k", "int32")
    indices = [0] * len(shape)
    copy = tir.BufferStore(local, source[k], indices)
    use = tir.BufferStore(output, local[tuple(indices)] + tir.const(1, dtype), [k])
    if kind in ("access_ptr", "async"):
        copy = tir.Evaluate(tir.call_extern("handle", "load", source.access_ptr("r", offset=k), local.access_ptr("w")))
        use = tir.Evaluate(tir.call_extern("handle", "use", local.access_ptr("r"), output.access_ptr("w", offset=k)))
    statements = [copy, use]
    if kind == "conditional":
        statements[1] = tir.IfThenElse(k < start + extent - 1, use, None)
    if kind == "nested":
        statements = [
            tir.SBlockRealize([], True, tir.SBlock([], [], [], "nested", tir.SeqStmt(statements))),
            tir.BufferStore(output, tir.const(9, dtype), [k + 32]),
        ]
    body = statements[0] if kind == "single" else tir.SeqStmt(statements)
    if kind == "predicate":
        body = tir.IfThenElse(k < start + extent - 1, body, None)
    block = tir.SBlock([], [], [], "pipeline_body", body, alloc_buffers=[] if kind == "external" else [local])
    attrs = {"software_pipeline_stage": list(stages), "software_pipeline_order": list(order), "user_marker": 7}
    if kind == "nested":
        attrs["software_pipeline_stage"].append(stages[-1])
        attrs["software_pipeline_order"].append(2)
    if kind == "async":
        attrs["software_pipeline_async_stages"] = [0]
    if annotations is not None:
        attrs = annotations
    loop = tir.For(k, start, extent, tir.ForKind.SERIAL, tir.SBlockRealize([], True, block), annotations=attrs)
    if kind == "external":
        loop = tir.SBlockRealize([], True, tir.SBlock([], [], [], "root", loop, alloc_buffers=[local]))
    return tir.PrimFunc([source.data, output.data], loop, buffer_map={source.data: source, output.data: output}).with_attr(
        "global_symbol", "pipeline_case"
    )


def _run(func):
    factory = tvm.ffi.get_global_func("tl.transform.A2A3InjectSoftwarePipeline")
    return factory()(tvm.IRModule({"main": func}))["main"]


def _allocations(func):
    result = []
    tir.stmt_functor.post_order_visit(func.body, lambda node: result.extend(node.alloc_buffers) if isinstance(node, tir.SBlock) else None)
    return result


def _execute_stores(func):
    """Interpret this bounded scalar-store fixture, without hardware or codegen."""
    env = {}
    memory = {}
    for buffer in func.buffer_map.values():
        memory[buffer.data] = list(range(1024)) if buffer.name == "source" else [-1] * 1024

    def expr(node):
        if isinstance(node, (tir.IntImm, tir.FloatImm)):
            return node.value
        if isinstance(node, tir.Var):
            return env[node]
        if isinstance(node, tir.BufferLoad):
            return memory[node.buffer.data][offset(node.buffer, node.indices)]
        if isinstance(node, tir.Cast):
            return expr(node.value)
        if isinstance(node, tir.Not):
            return not expr(node.a)
        if isinstance(node, tir.Select):
            return expr(node.true_value) if expr(node.condition) else expr(node.false_value)
        operations = {
            tir.Add: operator.add,
            tir.Sub: operator.sub,
            tir.Mul: operator.mul,
            tir.FloorDiv: operator.floordiv,
            tir.FloorMod: operator.mod,
            tir.LT: operator.lt,
            tir.LE: operator.le,
            tir.GT: operator.gt,
            tir.GE: operator.ge,
            tir.EQ: operator.eq,
            tir.NE: operator.ne,
            tir.And: lambda a, b: a and b,
            tir.Or: lambda a, b: a or b,
        }
        if type(node) in operations:
            return operations[type(node)](expr(node.a), expr(node.b))
        raise AssertionError(type(node).__name__)

    def offset(buffer, indices):
        shape = [expr(item) for item in buffer.shape]
        values = [expr(item) for item in indices]
        if buffer.strides:
            return sum(i * expr(s) for i, s in zip(values, buffer.strides))
        return sum(i * math.prod(shape[d + 1 :]) for d, i in enumerate(values))

    def stmt(node):
        if isinstance(node, tir.SeqStmt):
            for item in node.seq:
                stmt(item)
        elif isinstance(node, tir.For):
            for value in range(expr(node.min), expr(node.min) + expr(node.extent)):
                env[node.loop_var] = value
                stmt(node.body)
        elif isinstance(node, tir.SBlockRealize):
            if expr(node.predicate):
                stmt(node.block)
        elif isinstance(node, tir.SBlock):
            for buffer in node.alloc_buffers:
                size = math.prod(expr(item) for item in buffer.shape)
                if buffer.strides:
                    size = 1 + sum((expr(dim) - 1) * expr(stride) for dim, stride in zip(buffer.shape, buffer.strides))
                memory[buffer.data] = [0] * size
            stmt(node.body)
        elif isinstance(node, tir.IfThenElse):
            if expr(node.condition):
                stmt(node.then_case)
            elif node.else_case:
                stmt(node.else_case)
        elif isinstance(node, tir.BufferStore):
            memory[node.buffer.data][offset(node.buffer, node.indices)] = expr(node.value)
        elif isinstance(node, tir.Evaluate):
            assert expr(node.value) == 0
        elif isinstance(node, tir.AttrStmt):
            stmt(node.body)
        else:
            raise AssertionError(type(node).__name__)

    stmt(func.body)
    return memory[next(buffer.data for buffer in func.buffer_map.values() if buffer.name == "output")]


@pytest.mark.parametrize("extent", [1, 2, 3, 7])
@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("start", [0, 3])
def test_pipeline_matches_original_scalar_output(extent, depth, start):
    before = _injection_case(extent, start, (0, depth))
    after = _run(before)
    assert _execute_stores(before) == _execute_stores(after)
    local = next(buffer for buffer in _allocations(after) if buffer.name == "local")
    assert list(local.shape) == [depth + 1, 16]
    assert "buffer_versions" in after.attrs
    assert sorted(int(value) for value in after.attrs["buffer_versions"].values()) == [depth + 1]


@pytest.mark.parametrize("kind", ["conditional", "predicate", "nested", "external"])
def test_source_wrappers_and_outside_allocations(kind):
    before = _injection_case(kind=kind, stages=(0, 0) if kind == "external" else (0, 1))
    after = _run(before)
    assert _execute_stores(before) == _execute_stores(after)
    if kind == "external":
        assert list(next(buffer for buffer in _allocations(after) if buffer.name == "local").shape) == [16]


def test_source_external_allocation_overlap_is_not_silently_fixed():
    before = _injection_case(kind="external")
    after = _run(before)
    assert _execute_stores(before)[:4] == [1, 2, 3, 4]
    # TA does not version allocations outside the pipeline scope. Its raw
    # pass accepts this overlapping schedule and changes the scalar output.
    assert _execute_stores(after)[:4] == [2, 3, 4, 4]
    assert list(next(buffer for buffer in _allocations(after) if buffer.name == "local").shape) == [16]


@pytest.mark.parametrize(
    "dtype,shape,padded", [("float16", (128, 200), (128, 208)), ("float32", (17, 9), (32, 16)), ("int8", (17, 33), (32, 64))]
)
def test_l1_multiversion_allocations_use_source_fractal_padding(dtype, shape, padded):
    after = _run(_injection_case(dtype=dtype, shape=shape, scope="shared.l1", kind="access_ptr"))
    local = next(buffer for buffer in _allocations(after) if buffer.name == "local")
    assert list(local.shape) == [2, *padded]
    offsets = []

    def collect(node):
        if isinstance(node, tir.Call) and node.op.name.endswith("tvm_access_ptr") and node.args[1].same_as(local.data):
            offsets.append(str(node.args[2]))

    tir.stmt_functor.post_order_visit(after.body, collect)
    assert any(str(math.prod(padded)) in value for value in offsets)


def test_explicit_strides_add_a_version_stride():
    after = _run(_injection_case(shape=(16, 8), strides=(16, 1)))
    local = next(buffer for buffer in _allocations(after) if buffer.name == "local")
    assert list(local.shape) == [2, 16, 8]
    assert list(local.strides) == [256, 16, 1]


def test_async_commit_wait_and_scope_paths_are_retained():
    after = _run(_injection_case(kind="async"))
    attrs = []
    tir.stmt_functor.post_order_visit(after.body, lambda node: attrs.append(node.attr_key) if isinstance(node, tir.AttrStmt) else None)
    assert {"async_scope", "async_commit_queue_scope", "async_wait_queue_scope", "async_wait_inflight_count"} <= set(attrs)


@pytest.mark.parametrize(
    "annotations,pattern",
    [
        ({"software_pipeline_stage": [0, 1]}, "Stage"),
        ({"software_pipeline_order": [0, 1]}, "Order"),
        ({"software_pipeline_stage": [0], "software_pipeline_order": [0]}, "different size"),
        ({"software_pipeline_stage": [0, 1], "software_pipeline_order": [0, 0]}, "same order"),
        ({"software_pipeline_stage": [1, 0], "software_pipeline_order": [0, 1]}, "later stage"),
        ({"software_pipeline_stage": [0, 0], "software_pipeline_order": [1, 0]}, "cannot be reordered"),
    ],
)
def test_invalid_schedule_keeps_source_rejection(annotations, pattern):
    with pytest.raises(Exception, match=pattern):
        _run(_injection_case(annotations=annotations))


def test_unannotated_loop_is_unchanged():
    before = _injection_case(annotations={"user_marker": 7})
    after = _run(before)
    tvm.ir.assert_structural_equal(before.body, after.body)


def test_single_statement_pipeline_is_rejected():
    with pytest.raises(Exception, match="SeqStmt"):
        _run(_injection_case(kind="single"))


@pytest.mark.parametrize("start", [0, 3])
def test_dynamic_extent_and_user_loop_annotations(start):
    extent = tir.Var("N", "int32")
    after = _run(_injection_case(extent, start))
    variables, loops = [], []

    def collect(node):
        if isinstance(node, tir.Var) and node.same_as(extent):
            variables.append(node)
        if isinstance(node, tir.For):
            loops.append(node)

    tir.stmt_functor.post_order_visit(after.body, collect)
    assert variables
    assert loops
    assert all(int(loop.annotations["user_marker"]) == 7 for loop in loops)
    assert all("software_pipeline_stage" not in loop.annotations for loop in loops)
    assert list(next(buffer for buffer in _allocations(after) if buffer.name == "local").shape) == [2, 16]
