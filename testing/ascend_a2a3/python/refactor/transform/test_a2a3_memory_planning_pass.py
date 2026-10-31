"""Original TA Ascend memory allocation metadata and liveness contracts."""

import pytest

from tilelang import tvm

tirx = tvm.tirx


def _buffer(name, shape=(16,), dtype="float32", scope="shared.ub"):
    return tirx.decl_buffer(shape, dtype, name=name, scope=scope)


def _shape_map(buffers):
    return {b.data: list(b.shape) for b in buffers}


def _run(body, buffers=(), auto=None, attrs=None):
    func = tirx.PrimFunc([], body).with_attr({"global_symbol": "main", **(attrs or {})})
    config = {} if auto is None else {"tl.ascend_memory_planning": auto}
    with tvm.transform.PassContext(config=config):
        return tvm.ffi.get_global_func("tl.transform.A2A3AscendMemoryPlanning")()(tvm.IRModule.from_expr(func))["main"]


def _body(buffers):
    return tirx.SeqStmt(
        [*[tirx.AllocBuffer(b) for b in buffers], *[tirx.BufferStore(b, tirx.const(1, b.dtype), [0] * len(b.shape)) for b in buffers]]
    )


def _named(attribute):
    return {var.name: int(value) for var, value in attribute.items()}


def test_memory_planning_default_keeps_source_ub_shape_map_requirement():
    buffer = _buffer("a")
    body = _body([buffer])
    after = _run(body)
    assert not after.attrs["address_map"]
    assert _named(after.attrs["size_map"]) == {"a": 64}
    assert after.body.same_as(body)


@pytest.mark.parametrize("auto", [False, True])
def test_memory_planning_shape_map_addresses_and_32byte_sizes(auto):
    buffers = [_buffer("a", (3,)), _buffer("b", (9,))]
    after = _run(_body(buffers), auto=auto, attrs={"logic_buffer_shapes": _shape_map(buffers)})
    assert _named(after.attrs["size_map"]) == {"a": 32, "b": 64}
    assert _named(after.attrs["address_map"]) == {"a": 0, "b": 32}


@pytest.mark.parametrize("scope", ["shared.l1", "wmma.matrix_a", "wmma.matrix_b", "wmma.accumulator"])
def test_memory_planning_retains_all_original_npu_scope_groups(scope):
    buffers = [_buffer("a", (3,), scope=scope), _buffer("b", (9,), scope=scope)]
    after = _run(_body(buffers), auto=False)
    assert _named(after.attrs["size_map"]) == {"a": 32, "b": 64}
    assert _named(after.attrs["address_map"]) == {"a": 0, "b": 32}


@pytest.mark.parametrize("auto,expected", [(False, 45), (True, 64)])
def test_memory_planning_preserves_source_manual_offset_mode_difference(auto, expected):
    buffer = _buffer("a")
    after = _run(
        _body([buffer]),
        auto=auto,
        attrs={"address_map": {buffer.data: tirx.IntImm("int32", 45)}, "logic_buffer_shapes": _shape_map([buffer])},
    )
    assert _named(after.attrs["address_map"]) == {"a": expected}


@pytest.mark.parametrize("key", ["logic_buffer_shapes", "buffer_shapess"])
def test_memory_planning_pto_physical_shape_and_legacy_attribute_key(key):
    buffer = _buffer("tile", (4,), dtype="float16")
    physical = [tirx.IntImm("int32", x) for x in (16, 32, 2, 3)]
    after = _run(_body([buffer]), attrs={key: {buffer.data: physical}})
    assert _named(after.attrs["size_map"]) == {"tile": 1024}
    assert _named(after.attrs["address_map"]) == {"tile": 0}


def test_memory_planning_ignores_non_npu_memory():
    buffer = _buffer("local", scope="local")
    after = _run(_body([buffer]), auto=True)
    assert not after.attrs["address_map"]
    assert not after.attrs["size_map"]


def test_memory_planning_rejects_duplicate_buffer_names_and_dynamic_extents():
    a, b = _buffer("duplicate"), _buffer("duplicate")
    with pytest.raises(tvm.error.InternalError, match="Duplicate buffer name"):
        _run(_body([a, b]))
    dynamic = _buffer("dynamic", (tirx.Var("n", "int32"),))
    with pytest.raises(tvm.error.InternalError, match="Extent must be an integer constant"):
        _run(_body([dynamic]))


def test_memory_planning_first_oversized_allocation_has_original_oom_diagnostic():
    huge = _buffer("huge", (65536,))  # 262144 > original 196352-byte UB limit.
    with pytest.raises(tvm.error.InternalError, match="Memory allocation failed for"):
        _run(_body([huge]), auto=True)


def test_memory_planning_loop_live_buffer_does_not_alias_later_iteration_scratch():
    a = _buffer("persistent", (16384,))
    expired = _buffer("expired", (16384,))
    scratch = _buffer("scratch", (25000,))
    i = tirx.Var("segment", "int32")
    body = tirx.SeqStmt(
        [
            tirx.AllocBuffer(a),
            tirx.AllocBuffer(expired),
            tirx.BufferStore(a, tirx.FloatImm("float32", 1), [0]),
            tirx.BufferStore(expired, tirx.FloatImm("float32", 2), [0]),
            tirx.Evaluate(tirx.BufferLoad(expired, [0])),
            tirx.For(
                i,
                0,
                4,
                tirx.ForKind.SERIAL,
                tirx.SeqStmt(
                    [
                        tirx.Evaluate(tirx.BufferLoad(a, [0])),
                        tirx.AllocBuffer(scratch),
                        tirx.BufferStore(scratch, tirx.FloatImm("float32", 3), [0]),
                        tirx.Evaluate(tirx.BufferLoad(scratch, [0])),
                    ]
                ),
            ),
        ]
    )
    after = _run(body, auto=True)
    offsets = _named(after.attrs["address_map"])
    assert offsets["persistent"] == 0
    assert offsets["expired"] == 65536
    assert offsets["scratch"] == 65536
    assert offsets["persistent"] != offsets["scratch"]


def test_memory_planning_bind_tail_retains_data_touch_and_address_metadata():
    buffer = _buffer("a")
    value = tirx.Var("value", "float32")
    body = tirx.SeqStmt(
        [
            tirx.AllocBuffer(buffer),
            tirx.BufferStore(buffer, tirx.FloatImm("float32", 1), [0]),
            tirx.Bind(value, tirx.BufferLoad(buffer, [0])),
            tirx.Evaluate(value),
        ]
    )
    after = _run(body, auto=True)
    assert _named(after.attrs["address_map"]) == {"a": 0}
    assert after.body.same_as(body)


@pytest.mark.parametrize("scalar", [False, True])
def test_memory_planning_packed_compare_backing_in_allocation_tail(scalar):
    dst = _buffer("packed", (4, 1), dtype="uint8")
    source = _buffer("source", (4, 8), dtype="float16")
    other = tirx.FloatImm("float16", 1) if scalar else source.access_ptr("r")
    call = tirx.call_intrin(
        "handle",
        tvm.ir.Op.get("tl.ascend_tail_compare_scalar" if scalar else "tl.ascend_tail_compare"),
        dst.access_ptr("w"),
        source.access_ptr("r"),
        other,
        "GE",
        3,
        7,
        4,
        8,
        1,
    )
    body = tirx.SeqStmt([tirx.AllocBuffer(dst), tirx.AllocBuffer(source), tirx.Evaluate(call)])
    after = _run(body, auto=False, attrs={"logic_buffer_shapes": _shape_map([dst, source])})
    assert _named(after.attrs["size_map"])["packed"] == 128
    assert _named(after.attrs["size_map"])["source"] == 64


@pytest.mark.parametrize("with_tmp", [False, True])
def test_memory_planning_broadcast_source_preserves_physical_row_backing(with_tmp):
    source = _buffer("source", (4, 1), dtype="float16")
    dst = _buffer("destination", (4, 16), dtype="float16")
    tmp = _buffer("tmp", (16,), dtype="float16")
    args = [tirx.StringImm("row"), dst.access_ptr("w"), source.access_ptr("r")]
    if with_tmp:
        args.append(tmp.access_ptr("rw"))
    args += [tirx.IntImm("int32", x) for x in [1, 4, 16, 4, 1, 3, 15, 3, 1]]
    call = tirx.call_intrin("handle", tvm.ir.Op.get("tl.ascend_tail_broadcast"), *args)
    buffers = [source, dst, *([tmp] if with_tmp else [])]
    body = tirx.SeqStmt([*[tirx.AllocBuffer(b) for b in buffers], tirx.Evaluate(call)])
    after = _run(body, attrs={"logic_buffer_shapes": _shape_map(buffers)})
    assert _named(after.attrs["size_map"])["source"] == 128
    assert _named(after.attrs["size_map"])["destination"] == 128


def test_memory_planning_assert_tail_and_conditional_scope_keep_body():
    a, b = _buffer("a"), _buffer("b")
    condition = tirx.Var("condition", "bool")
    body = tirx.SeqStmt(
        [
            tirx.AllocBuffer(a),
            tirx.AllocBuffer(b),
            tirx.BufferStore(a, tirx.FloatImm("float32", 1), [0]),
            tirx.AssertStmt(tirx.BufferLoad(a, [0]) > 0, tirx.StringImm("RuntimeError"), [tirx.StringImm("positive")]),
            tirx.IfThenElse(
                condition,
                tirx.BufferStore(b, tirx.BufferLoad(a, [0]), [0]),
                tirx.BufferStore(b, tirx.FloatImm("float32", 2), [0]),
            ),
            tirx.Evaluate(tirx.BufferLoad(b, [0])),
        ]
    )
    after = _run(body, auto=True)
    assert _named(after.attrs["address_map"]) == {"a": 0, "b": 64}
    assert after.body.same_as(body)


@pytest.mark.parametrize("auto", [False, True])
def test_memory_planning_nested_bind_assert_tails_preserve_original_offsets(auto):
    a, b, c = [_buffer(name) for name in ("a", "b", "c")]
    outer = tirx.Var("outer", "float32")
    inner = tirx.Var("inner", "float32")
    body = tirx.SeqStmt(
        [
            tirx.AllocBuffer(a),
            tirx.AllocBuffer(b),
            tirx.AllocBuffer(c),
            tirx.BufferStore(a, tirx.FloatImm("float32", 1), [0]),
            tirx.Bind(outer, tirx.BufferLoad(a, [0])),
            tirx.SeqStmt(
                [
                    tirx.AssertStmt(tirx.BufferLoad(a, [0]) > 0, tirx.StringImm("RuntimeError"), [tirx.StringImm("positive")]),
                    tirx.BufferStore(b, outer, [0]),
                    tirx.Bind(inner, tirx.BufferLoad(b, [0])),
                    tirx.Evaluate(outer + inner),
                ]
            ),
            tirx.BufferStore(c, outer, [0]),
            tirx.Evaluate(outer + tirx.BufferLoad(c, [0])),
        ]
    )
    after = _run(body, auto=auto, attrs={"logic_buffer_shapes": _shape_map([a, b, c])})
    assert _named(after.attrs["address_map"]) == {"a": 0, "b": 64, "c": 128}
    assert _named(after.attrs["size_map"]) == {"a": 64, "b": 64, "c": 64}
    assert after.body.same_as(body)


def test_memory_planning_wide_flat_bind_assert_tail_does_not_exhaust_stack():
    # The original NSA varlen request has 11,881 flat bindings after unrolling.
    # These statements stay flat; their lexical tails must not become native
    # recursion, even when no simplification removes unused bindings.
    buffer = _buffer("wide")
    statements = [tirx.AllocBuffer(buffer), tirx.BufferStore(buffer, tirx.FloatImm("float32", 1), [0])]
    for i in range(12000):
        if i % 2:
            statements.append(tirx.AssertStmt(tirx.const(True, "bool"), tirx.StringImm("RuntimeError"), []))
        else:
            statements.append(tirx.Bind(tirx.Var(f"value_{i}", "float32"), tirx.BufferLoad(buffer, [0])))
    statements.append(tirx.Evaluate(tirx.BufferLoad(buffer, [0])))
    body = tirx.SeqStmt(statements)
    after = _run(body, auto=True, attrs={"logic_buffer_shapes": _shape_map([buffer])})
    assert _named(after.attrs["address_map"]) == {"wide": 0}
    assert _named(after.attrs["size_map"]) == {"wide": 64}
    assert after.body.same_as(body)
