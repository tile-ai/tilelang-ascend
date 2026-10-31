"""Complete TA parallel expression lowering, without device execution."""

import ctypes
import os

import pytest
from tilelang import tvm
from tilelang.ascend.a2a3.transform import AscendLowerParallelToVector

if os.environ.get("A2A3_TEST_LOWER_PARALLEL_LIBRARY"):
    _LIBRARY = ctypes.CDLL(os.environ["A2A3_TEST_LOWER_PARALLEL_LIBRARY"], mode=ctypes.RTLD_GLOBAL)


def _nodes(stmt, cls):
    result = []
    tvm.tirx.stmt_functor.post_order_visit(stmt, lambda n: result.append(n) if isinstance(n, cls) else None)
    return result


def _lower(func):
    return AscendLowerParallelToVector()(tvm.IRModule.from_expr(func))["main"]


def _calls(func):
    return [n for n in _nodes(func.body, tvm.tirx.Call) if isinstance(n.op, tvm.ir.Op) and n.op.name.startswith("tl.ascend_")]


def _function(value, dtype="float32", shape=(64,), extent=None, gm=False, annotations=None):
    t = tvm.tirx
    a, b, c = [t.decl_buffer(shape, dtype, name=name, scope="shared.ub") for name in ("a", "b", "c")]
    d = t.decl_buffer(shape, dtype, name="d", scope="global" if gm else "shared.ub")
    indices = [t.Var("i" + str(i), "int32") for i in range(len(shape))]
    loads = [t.BufferLoad(buf, indices) for buf in (a, b, c)]
    body = t.BufferStore(d, value(*loads), indices)
    for index, size in reversed(list(zip(indices, extent or shape))):
        body = t.For(index, 0, size, t.ForKind.PARALLEL, body)
    allocs = [a, b, c] if gm else [a, b, c, d]
    block = t.SBlock([], [], [], "root", body, alloc_buffers=allocs, annotations=annotations or {})
    params = [t.Var("output", "handle")] if gm else []
    return t.PrimFunc(params, t.SBlockRealize([], True, block), buffer_map={params[0]: d} if gm else {})


@pytest.mark.parametrize(
    "build,expected",
    [
        (lambda a, b, c: a + b, "tl.ascend_add"),
        (lambda a, b, c: a - b, "tl.ascend_sub"),
        (lambda a, b, c: a * b, "tl.ascend_mul"),
        (lambda a, b, c: a / b, "tl.ascend_div"),
        (lambda a, b, c: tvm.tirx.Min(a, b), "tl.ascend_min"),
        (lambda a, b, c: tvm.tirx.Max(a, b), "tl.ascend_max"),
        (lambda a, b, c: a + 2, "tl.ascend_adds"),
        (lambda a, b, c: a - 2, "tl.ascend_subs"),
        (lambda a, b, c: a * 2, "tl.ascend_muls"),
        (lambda a, b, c: a / 2, "tl.ascend_divs"),
    ],
)
def test_all_arithmetic_and_scalar_paths(build, expected):
    result = _lower(_function(build))
    calls = _calls(result)
    assert [call.op.name for call in calls] == [expected]
    assert int(calls[0].args[-1]) == 64
    assert not _nodes(result.body, tvm.tirx.For)


@pytest.mark.parametrize("name,expected", [("exp", "exp"), ("log", "ln"), ("sqrt", "sqrt"), ("rsqrt", "rsqrt"), ("fabs", "abs")])
def test_unary_tirx_op_names_match_the_original_map(name, expected):
    result = _lower(_function(lambda a, b, c: tvm.tirx.call_intrin("float32", "tirx." + name, a)))
    assert [call.op.name for call in _calls(result)] == ["tl.ascend_" + expected]


@pytest.mark.parametrize("zero_first", [False, True])
def test_relu_recognizes_both_operand_orders(zero_first):
    result = _lower(
        _function(
            lambda a, b, c: tvm.tirx.Max(tvm.tirx.const(0, "float32"), a) if zero_first else tvm.tirx.Max(a, tvm.tirx.const(0, "float32"))
        )
    )
    assert [call.op.name for call in _calls(result)] == ["tl.ascend_relu"]


@pytest.mark.parametrize(
    "operation,expected",
    [("bitwise_and", "and"), ("bitwise_or", "or"), ("shift_left", "lshift"), ("shift_right", "rshift"), ("bitwise_not", "not")],
)
def test_bitwise_builtin_paths(operation, expected):
    result = _lower(
        _function(
            lambda a, b, c: getattr(tvm.tirx, operation)(a) if operation == "bitwise_not" else getattr(tvm.tirx, operation)(a, b),
            dtype="int32",
        )
    )
    assert [call.op.name for call in _calls(result)] == ["tl.ascend_bitwise_" + expected]


def test_complex_expression_retains_two_temporaries_and_all_operations():
    result = _lower(_function(lambda a, b, c: (a + b) * (a - c)))
    assert [call.op.name for call in _calls(result)] == ["tl.ascend_add", "tl.ascend_sub", "tl.ascend_mul"]
    temps = [buf for buf in result.body.block.alloc_buffers if "_tmp_" in buf.name]
    assert len(temps) == 2
    assert all([int(n) for n in buf.shape] == [64] for buf in temps)


def test_nested_sblocks_keep_temporary_buffers_in_their_original_scope():
    t = tvm.tirx
    outer = _function(lambda a, b, c: (a + b) * (a - c)).body.block
    inner = _function(lambda a, b, c: (a + b) * (a - c)).body.block
    nested = t.SBlock([], [], [], "inner", inner.body, alloc_buffers=inner.alloc_buffers)
    body = t.SeqStmt([outer.body, t.SBlockRealize([], True, nested)])
    root = t.SBlock([], [], [], "root", body, alloc_buffers=outer.alloc_buffers)
    result = _lower(t.PrimFunc([], t.SBlockRealize([], True, root)))
    blocks = {block.name_hint: block for block in _nodes(result.body, t.SBlock)}
    outer_temps = [buf for buf in blocks["root"].alloc_buffers if "_tmp_" in buf.name]
    inner_temps = [buf for buf in blocks["inner"].alloc_buffers if "_tmp_" in buf.name]
    assert len(outer_temps) == len(inner_temps) == 2
    assert all(not a.data.same_as(b.data) for a in outer_temps for b in inner_temps)


@pytest.mark.parametrize("extent", [13, 17])
def test_tail_element_counts_are_not_rounded_to_vector_width(extent):
    result = _lower(_function(lambda a, b, c: a + b, extent=(extent,)))
    assert int(_calls(result)[0].args[-1]) == extent


def test_global_output_keeps_temporary_ub_and_copyback():
    result = _lower(_function(lambda a, b, c: a + b, gm=True))
    assert [call.op.name for call in _calls(result)] == ["tl.ascend_add", "tl.ascend_copy"]
    temps = [buf for buf in result.body.block.alloc_buffers if "_tmp_" in buf.name]
    assert len(temps) == 1 and temps[0].scope() == "shared.ub"


def test_scalar_left_expression_retains_the_original_serial_fallback():
    result = _lower(_function(lambda a, b, c: 2 - a))
    assert not _calls(result)
    assert all(loop.kind == tvm.tirx.ForKind.SERIAL for loop in _nodes(result.body, tvm.tirx.For))


def test_data_dependent_condition_keeps_serial_fallback():
    t = tvm.tirx
    a, d = [t.decl_buffer((64,), "float32", name=name, scope="shared.ub") for name in ("a", "d")]
    i = t.Var("i", "int32")
    value = t.BufferLoad(a, [i])
    conditional = t.IfThenElse(value > 0, t.BufferStore(d, value, [i]), t.BufferStore(d, t.const(0, "float32"), [i]))
    loop = t.For(i, 0, 64, t.ForKind.PARALLEL, conditional)
    result = _lower(t.PrimFunc([], t.SBlockRealize([], True, t.SBlock([], [], [], "root", loop, alloc_buffers=[a, d]))))
    assert not _calls(result)
    assert len(_nodes(result.body, t.IfThenElse)) == 1
    assert _nodes(result.body, t.For)[0].kind == t.ForKind.SERIAL


def _broadcast(dtype):
    t = tvm.tirx
    a = t.decl_buffer((16,), dtype, name="a", scope="shared.ub")
    b = t.decl_buffer((4,), dtype, name="b", scope="shared.ub")
    d = t.decl_buffer((4, 16), dtype, name="d", scope="shared.ub")
    i, j = t.Var("i", "int32"), t.Var("j", "int32")
    body = t.For(
        i,
        0,
        4,
        t.ForKind.PARALLEL,
        t.For(j, 0, 16, t.ForKind.PARALLEL, t.BufferStore(d, t.BufferLoad(a, [j]) + t.BufferLoad(b, [i]), [i, j])),
    )
    return t.PrimFunc([], t.SBlockRealize([], True, t.SBlock([], [], [], "root", body, alloc_buffers=[a, b, d])))


def test_broadcast_keeps_both_axes_and_workspace_allocations():
    result = _lower(_broadcast("float32"))
    calls = _calls(result)
    assert [call.op.name for call in calls] == ["tl.ascend_broadcast", "tl.ascend_broadcast", "tl.ascend_adds"]
    assert str(calls[0].args[0].value) == "Broadcast<float, 2, 0, false>"
    assert str(calls[1].args[0].value) == "Broadcast<float, 2, 1, false>"
    workspace = [buf for buf in result.body.block.alloc_buffers if buf.name.startswith("broadcast_workspace_")]
    assert len(workspace) == 2
    assert all(buf.dtype == "uint8" and int(buf.shape[0]) == 128 for buf in workspace)


def test_bfloat_broadcast_rejects_only_the_original_empty_cstring_mapping():
    with pytest.raises(tvm.error.InternalError, match="parallel broadcast does not support dtype bfloat16"):
        _lower(_broadcast("bfloat16"))
    result = _lower(_function(lambda a, b, c: a + b, dtype="bfloat16"))
    assert [call.op.name for call in _calls(result)] == ["tl.ascend_add"]
    assert _calls(result)[0].args[0].args[0].dtype == "bfloat16"


def test_compact_to_aligned_ub_copy_keeps_serial_fallback():
    t = tvm.tirx
    a = t.decl_buffer((4, 8), "int32", name="a", scope="shared.ub")
    d = t.decl_buffer((4, 16), "int32", name="d", scope="shared.ub")
    i, j = t.Var("i", "int32"), t.Var("j", "int32")
    body = t.For(i, 0, 4, t.ForKind.PARALLEL, t.For(j, 0, 8, t.ForKind.PARALLEL, t.BufferStore(d, t.BufferLoad(a, [i, j]), [i, j])))
    result = _lower(t.PrimFunc([], t.SBlockRealize([], True, t.SBlock([], [], [], "root", body, alloc_buffers=[a, d]))))
    assert not _calls(result)
    assert len(_nodes(result.body, t.For)) == 2
    assert all(loop.kind == t.ForKind.SERIAL for loop in _nodes(result.body, t.For))


def test_original_int64_offset_substitution_error_is_not_silently_coerced():
    t = tvm.tirx
    a, d = [t.decl_buffer((128,), "float32", name=name, scope="shared.ub") for name in ("a", "d")]
    i = t.Var("i64", "int64")
    index = i + t.IntImm("int64", 2)
    loop = t.For(
        i, t.IntImm("int64", 0), t.IntImm("int64", 64), t.ForKind.PARALLEL, t.BufferStore(d, t.BufferLoad(a, [index]) + 1, [index])
    )
    func = t.PrimFunc([], t.SBlockRealize([], True, t.SBlock([], [], [], "root", loop, alloc_buffers=[a, d])))
    # The pinned TA substitution also replaces the loop Var by int32 zero;
    # changing that pre-existing dtype error is outside this migration.
    with pytest.raises(TypeError, match="mismatched types.*int32 vs. int64"):
        _lower(func)


def test_three_parallel_dimensions_keep_the_original_strict_error():
    with pytest.raises(tvm.error.InternalError, match="3D or higher dimensional parallel loops"):
        _lower(_function(lambda a, b, c: a + b, shape=(2, 4, 16)))


@pytest.mark.parametrize("developer_mode", [0, 1])
def test_rebuilt_sblock_preserves_only_the_two_required_launch_markers(developer_mode):
    t = tvm.tirx
    keys = ["tilelang.is_npu_kernel_frame", "tilelang.is_npu_kernel_frame_dev_mode"]
    func = _function(lambda a, b, c: a + b, annotations={keys[0]: 1, keys[1]: developer_mode, "layout_map": {}, "source_metadata": 9})
    before = func.body.block
    source = before.alloc_buffers[0]
    view = t.decl_buffer((64,), "float32", name="view", scope="shared.ub")
    match = t.MatchBufferRegion(view, t.BufferRegion(source, [tvm.ir.Range(0, 64)]))
    annotated = t.SBlock(
        before.iter_vars,
        before.reads,
        before.writes,
        before.name_hint,
        before.body,
        alloc_buffers=before.alloc_buffers,
        match_buffers=[match],
        annotations=before.annotations,
        span=tvm.ir.Span(tvm.ir.SourceName("parallel_metadata.py"), 1, 1, 1, 10),
    )
    func = t.PrimFunc([], t.SBlockRealize([], True, annotated))
    result = _lower(func).body.block
    assert set(result.annotations.keys()) == set(keys)
    assert int(result.annotations[keys[0]]) == 1
    assert int(result.annotations[keys[1]]) == developer_mode
    # Keep the pinned TA block-rebuild behavior; unrelated metadata is a
    # separate source bug, not an implicit feature change in this migration.
    assert "layout_map" not in result.annotations and "source_metadata" not in result.annotations
    assert not result.match_buffers
    unchanged = t.SBlock([], [], [], "unchanged", t.Evaluate(0), annotations={"source_metadata": 9})
    preserved = _lower(t.PrimFunc([], t.SBlockRealize([], True, unchanged))).body.block
    assert preserved.same_as(unchanged)
    assert not result.same_as(before)


@pytest.mark.parametrize("launch", ["none", "attr", "thread_for"])
def test_flat_temporaries_are_declared_inside_launch_before_all_uses(launch):
    t = tvm.tirx
    a, b = [t.decl_buffer((64,), "float32", name=name, scope="shared.ub") for name in ("a", "b")]
    d = t.decl_buffer((128,), "float32", name="d", scope="global")
    i, vid = t.Var("i", "int32"), t.Var("vid", "int32")
    offset = 0 if launch == "none" else vid * 64
    body = t.SeqStmt(
        [
            t.AllocBuffer(a),
            t.AllocBuffer(b),
            t.For(i, 0, 64, t.ForKind.PARALLEL, t.BufferStore(d, t.BufferLoad(a, [i]) + t.BufferLoad(b, [i]), [offset + i])),
        ]
    )
    binding = t.IterVar(tvm.ir.Range(0, 2), vid, t.IterVar.ThreadIndex, "cthread")
    if launch == "attr":
        body = t.AttrStmt(binding, "thread_extent", 2, t.AttrStmt(t.IntImm("int32", 0), "resource_scope", 1, body))
    elif launch == "thread_for":
        body = t.For(vid, 0, 2, t.ForKind.THREAD_BINDING, body, thread_binding=binding)
    param = t.Var("output", "handle")
    result = _lower(t.PrimFunc([param], body, buffer_map={param: d}))
    root = result.body
    if launch == "attr":
        assert isinstance(root, t.AttrStmt) and root.attr_key == "thread_extent"
        root = root.body
        assert isinstance(root, t.AttrStmt) and root.attr_key == "resource_scope"
        root = root.body
    elif launch == "thread_for":
        assert isinstance(root, t.For) and root.kind == t.ForKind.THREAD_BINDING
        root = root.body
    assert isinstance(root, t.SeqStmt)
    assert isinstance(root.seq[0], t.AllocBuffer) and "_tmp_" in root.seq[0].buffer.name
    assert not tvm.tirx.analysis.undefined_vars(result.body, [param, d.data])
    calls = _calls(result)
    assert [call.op.name for call in calls] == ["tl.ascend_add", "tl.ascend_copy"]
    dst_index = calls[-1].args[1].args[0].indices[0]
    assert tvm.arith.Analyzer().can_prove_equal(dst_index, offset)


def test_flat_temporaries_do_not_escape_nonleading_kernel_launches():
    t = tvm.tirx
    a, b = [t.decl_buffer((64,), "float32", name=name, scope="shared.ub") for name in ("a", "b")]
    d = t.decl_buffer((128,), "float32", name="d", scope="global")
    i, vid = t.Var("i", "int32"), t.Var("vid", "int32")
    loop = t.For(i, 0, 64, t.ForKind.PARALLEL, t.BufferStore(d, t.BufferLoad(a, [i]) + t.BufferLoad(b, [i]), [vid * 64 + i]))
    binding = t.IterVar(tvm.ir.Range(0, 2), vid, t.IterVar.ThreadIndex, "cthread")
    launch = t.AttrStmt(binding, "thread_extent", 2, loop)
    body = t.SeqStmt([t.AllocBuffer(a), t.AllocBuffer(b), launch])
    param = t.Var("output", "handle")
    with pytest.raises(tvm.error.InternalError, match="must not escape a nested kernel launch"):
        _lower(t.PrimFunc([param], body, buffer_map={param: d}))
