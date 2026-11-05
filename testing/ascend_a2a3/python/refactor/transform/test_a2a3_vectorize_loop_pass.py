"""Original TA TLVectorizer contracts, distinct from the planning helper."""

import pytest

from tilelang import tvm

tirx = tvm.tirx


def _run(body, buffers=(), enable=True):
    func = tirx.PrimFunc([b.data for b in buffers], body, buffer_map={b.data: b for b in buffers})
    func = func.with_attr({"global_symbol": "main", "target": tvm.target.Target("llvm")})
    with tvm.target.Target("llvm"):
        return tvm.ffi.get_global_func("tl.transform.A2A3VectorizeLoop")(enable)(tvm.IRModule.from_expr(func))["main"]


def _nodes(body, typ):
    out = []
    tirx.stmt_functor.post_order_visit(body, lambda x: out.append(x) if isinstance(x, typ) else None)
    return out


def _buffers():
    return tirx.decl_buffer((16,), "float32", name="A"), tirx.decl_buffer((16,), "float32", name="B")


def test_vectorize_loop_legalizes_load_dtype_and_vector_store():
    a, b = _buffers()
    i = tirx.Var("i", "int32")
    body = tirx.For(i, 0, 4, tirx.ForKind.VECTORIZED, tirx.BufferStore(b, tirx.BufferLoad(a, [i]) + 1, [i]))
    after = _run(body, [a, b])
    assert not _nodes(after.body, tirx.For)
    assert str(_nodes(after.body, tirx.BufferLoad)[0].dtype) == "float32x4"
    store = _nodes(after.body, tirx.BufferStore)[0]
    assert str(store.value.dtype) == "float32x4"
    assert isinstance(store.indices[0], tirx.Ramp)


def test_vectorize_loop_disable_preserves_min_extent_and_drops_source_annotations():
    i = tirx.Var("i", "int32")
    body = tirx.For(i, 3, 4, tirx.ForKind.VECTORIZED, tirx.Evaluate(i), annotations={"keep": 7})
    after = _run(body, enable=False)
    assert after.body.kind == tirx.ForKind.SERIAL
    assert int(after.body.min) == 3
    assert int(after.body.extent) == 4
    assert not after.body.annotations


def test_vectorize_loop_bind_and_expression_let_keep_vector_values():
    a, b = _buffers()
    i, value, inner = tirx.Var("i", "int32"), tirx.Var("value", "float32"), tirx.Var("inner", "float32")
    expr = tirx.Let(inner, value + 1, inner * 2)
    body = tirx.SeqStmt([tirx.Bind(value, tirx.BufferLoad(a, [i])), tirx.BufferStore(b, expr, [i])])
    after = _run(tirx.For(i, 0, 4, tirx.ForKind.VECTORIZED, body), [a, b])
    assert str(_nodes(after.body, tirx.Bind)[0].var.dtype) == "float32x4"
    assert str(_nodes(after.body, tirx.Let)[0].var.dtype) == "float32x4"
    assert str(_nodes(after.body, tirx.BufferStore)[0].value.dtype) == "float32x4"


def test_vectorize_loop_select_preserves_current_boolean_representation():
    a, b = _buffers()
    i = tirx.Var("i", "int32")
    value = tirx.Select(i < 2, tirx.BufferLoad(a, [i]), tirx.FloatImm("float32", 0))
    body = tirx.For(i, 0, 4, tirx.ForKind.VECTORIZED, tirx.BufferStore(b, value, [i]))
    after = _run(body, [a, b])
    select = _nodes(after.body, tirx.Select)[0]
    assert str(select.condition.dtype) == "boolx4"
    assert str(select.dtype) == "float32x4"


@pytest.mark.parametrize("kind", ["if", "opaque_call"])
def test_vectorize_loop_scalarizes_lane_dependent_control_or_opaque_call(kind):
    a, b = _buffers()
    i = tirx.Var("i", "int32")
    load = tirx.BufferLoad(a, [i])
    body = tirx.BufferStore(b, load, [i])
    if kind == "if":
        body = tirx.IfThenElse(i < 2, body, None)
    else:
        body = tirx.BufferStore(b, tirx.call_extern("float32", "opaque", load), [i])
    after = _run(tirx.For(i, 0, 4, tirx.ForKind.VECTORIZED, body), [a, b])
    loops = _nodes(after.body, tirx.For)
    assert len(loops) == 1
    assert loops[0].kind == tirx.ForKind.SERIAL
    assert int(loops[0].extent) == 4
    assert str(_nodes(after.body, tirx.BufferLoad)[0].dtype) == "float32"


def test_vectorize_loop_preserves_ta_workspace_access_behavior():
    a, _ = _buffers()
    tmp = tirx.decl_buffer((1,), "float32", name="tmp", scope="local")
    i = tirx.Var("i", "int32")
    body = tirx.SeqStmt([tirx.AllocBuffer(tmp), tirx.BufferStore(tmp, tirx.BufferLoad(a, [i]), [0])])
    after = _run(tirx.For(i, 0, 4, tirx.ForKind.VECTORIZED, body), [a])
    allocation = _nodes(after.body, tirx.AllocBuffer)[0]
    store = _nodes(after.body, tirx.BufferStore)[0]
    assert [int(tvm.arith.Analyzer().simplify(x)) for x in allocation.buffer.shape] == [4]
    # TLVecAllocAccess in this original component returns the original access.
    # The similarly named common helper instead remaps its buffer/indices.
    assert [int(x) for x in store.buffer.shape] == [1]
    assert isinstance(store.indices[0], tirx.Broadcast)
    assert int(store.indices[0].value) == 0


def test_vectorize_loop_scalarizes_dynamic_allocation_with_its_lexical_tail():
    a, _ = _buffers()
    i = tirx.Var("i", "int32")
    tmp = tirx.decl_buffer((i + 1,), "float32", name="tmp", scope="local")
    body = tirx.SeqStmt([tirx.AllocBuffer(tmp), tirx.BufferStore(tmp, tirx.BufferLoad(a, [i]), [0])])
    after = _run(tirx.For(i, 0, 4, tirx.ForKind.VECTORIZED, body), [a])
    assert isinstance(after.body, tirx.For)
    assert after.body.kind == tirx.ForKind.SERIAL
    assert _nodes(after.body.body, tirx.AllocBuffer)
    assert _nodes(after.body.body, tirx.BufferStore)


@pytest.mark.parametrize("bad", ["while", "nonzero_min", "symbolic_extent"])
def test_vectorize_loop_retains_original_invalid_input_diagnostics(bad):
    i = tirx.Var("i", "int32")
    body = tirx.While(tirx.const(True, "bool"), tirx.Evaluate(i)) if bad == "while" else tirx.Evaluate(i)
    extent = tirx.Var("n", "int32") if bad == "symbolic_extent" else 4
    minimum = 1 if bad == "nonzero_min" else 0
    loop = tirx.For(i, minimum, extent, tirx.ForKind.VECTORIZED, body)
    with pytest.raises(tvm.error.InternalError):
        _run(loop)
