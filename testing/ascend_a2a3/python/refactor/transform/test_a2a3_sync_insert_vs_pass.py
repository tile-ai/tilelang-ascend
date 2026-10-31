"""CPU contracts for the complete TA 0deb8b93 SyncInsertVS algorithm."""

import pytest

from tilelang import tvm

ir = tvm.tirx


def _buffers():
    return {
        name: ir.decl_buffer((64,), "float32", name=name, scope=scope)
        for name, scope in [("a", "shared.ub"), ("b", "shared.ub"), ("input", "global"), ("output", "global")]
    }


def _seq(statements):
    return statements[0] if len(statements) == 1 else ir.SeqStmt(statements)


def _call(name, *args):
    return ir.Evaluate(ir.call_extern("handle", name, *args))


def _intrin(name, *args):
    return ir.Evaluate(ir.call_intrin("handle", tvm.ir.Op.get("tl." + name), *args))


def _access(pipeline, buffers, name="a"):
    a = buffers[name]
    if pipeline == "MTE2":
        return _call("copy_gm_to_ub", buffers["input"].access_ptr("r"), a.access_ptr("w"))
    if pipeline == "MTE3":
        return _call("copy_ub_to_gm", a.access_ptr("r"), buffers["output"].access_ptr("w"))
    if pipeline == "V":
        return _call("AscendC::Duplicate", a.access_ptr("w"))
    assert pipeline == "S"
    return ir.BufferStore(a, ir.FloatImm("float32", 1), [0])


def _run(body, buffers, enabled=True, platform="A3", dialect="ascendc", offsets=None, sizes=None):
    if isinstance(body, list):
        body = _seq(body)
    func = ir.PrimFunc([], body)
    if offsets is not None:
        func = func.with_attr("address_map", {buffers[n].data: ir.IntImm("int32", v) for n, v in offsets.items()})
    if sizes is not None:
        func = func.with_attr("size_map", {buffers[n].data: ir.IntImm("int32", v) for n, v in sizes.items()})
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "keys": [dialect]})
    config = {} if enabled is None else {"tl.ascend_auto_sync_vs": enabled}
    with tvm.transform.PassContext(config=config):
        return tvm.ffi.get_global_func("tl.transform.A2A3AscendSyncInsertVS")(target, platform)(tvm.IRModule.from_expr(func))["main"]


def _syncs(function):
    result = []

    def visit(node):
        if isinstance(node, ir.Evaluate) and isinstance(node.value, ir.Call):
            name = node.value.op.name
            if name in ("tl.ascend_auto_barrier", "tl.ascend_auto_set_flag", "tl.ascend_auto_wait_flag"):
                result.append((name.removeprefix("tl.ascend_auto_"), *[a.value for a in node.value.args]))

    ir.stmt_functor.post_order_visit(function.body, visit)
    return result


def _events(function):
    pending = {}
    events = []
    for record in _syncs(function):
        if record[0] == "set_flag":
            assert len(record) == 3
            key = record[1:]
            assert key not in pending
            pending[key] = True
        elif record[0] == "wait_flag":
            assert len(record) == 3
            key = record[1:]
            assert pending.pop(key)
            events.append(record[1])
    assert not pending
    return events


@pytest.mark.parametrize("producer", ["V", "S", "MTE2", "MTE3"])
@pytest.mark.parametrize("consumer", ["V", "S", "MTE2", "MTE3"])
def test_sync_vs_complete_supported_pipeline_hazard_matrix(producer, consumer):
    buffers = _buffers()
    result = _run([_access(producer, buffers), _access(consumer, buffers)], buffers)
    events = _events(result)
    if producer != consumer and "S" in (producer, consumer):
        assert events == [producer + "_" + consumer]
    else:
        assert not events
    assert (("barrier", "PIPE_V") in _syncs(result)) == (producer == consumer == "V")


@pytest.mark.parametrize("enabled", [None, False])
def test_sync_vs_default_and_disabled_do_not_change_body(enabled):
    buffers = _buffers()
    body = _seq([_access("V", buffers), _access("V", buffers)])
    assert _run(body, buffers, enabled=enabled).body.same_as(body)


@pytest.mark.parametrize("platform", ["A2", "A3", "A5", "auto", "custom"])
@pytest.mark.parametrize("dialect", ["ascendc", "pto"])
def test_sync_vs_raw_platform_v_barrier_and_a5_event_fallback(platform, dialect):
    buffers = _buffers()
    result = _run([_access("V", buffers), _access("V", buffers)], buffers, platform=platform, dialect=dialect)
    if platform == "A5":
        assert _events(result) == ["V_V"]
        assert not any(s[0] == "barrier" for s in _syncs(result))
    else:
        assert _syncs(result) == [("barrier", "PIPE_V")]


@pytest.mark.parametrize("offset,size,expected", [(0, 256, True), (128, 256, True), (256, 256, False), (0, None, True), (128, None, False)])
def test_sync_vs_alias_range_overlap_and_exact_address_fallback(offset, size, expected):
    buffers = _buffers()
    sizes = {"a": size, "b": size} if size else None
    result = _run([_access("V", buffers), _access("S", buffers, "b")], buffers, offsets={"a": 0, "b": offset}, sizes=sizes)
    assert ("V_S" in _events(result)) == expected


def test_sync_vs_alias_different_memory_scopes_do_not_overlap():
    buffers = _buffers()
    buffers["b"] = ir.decl_buffer((64,), "float32", name="b", scope="shared.l1")
    result = _run([_access("V", buffers), _access("S", buffers, "b")], buffers, offsets={"a": 0, "b": 0}, sizes={"a": 256, "b": 256})
    assert not _events(result)


@pytest.mark.parametrize("producer", ["V", "MTE2"])
def test_sync_vs_last_write_survives_a_non_syncing_reader(producer):
    buffers = _buffers()
    result = _run([_access(producer, buffers), _access("MTE3", buffers), ir.Evaluate(ir.BufferLoad(buffers["a"], [0]))], buffers)
    assert _events(result) == [producer + "_S"]


@pytest.mark.parametrize("expression", ["add", "cast", "select"])
def test_sync_vs_finds_buffer_loads_inside_scalar_expression_trees(expression):
    buffers = _buffers()
    load = ir.BufferLoad(buffers["a"], [0])
    expr = {"add": load * 2 + 1, "cast": ir.Cast("float16", load), "select": ir.Select(load > 0, load, -load)}[expression]
    result = _run([_access("V", buffers), ir.Evaluate(expr)], buffers)
    assert _events(result) == ["V_S"]


@pytest.mark.parametrize("scope", ["local.var", "n"])
def test_sync_vs_scalar_local_write_still_scans_ub_reads(scope):
    buffers = _buffers()
    local = ir.decl_buffer((1,), "float32", name="local", scope=scope)
    result = _run([_access("V", buffers), ir.BufferStore(local, ir.BufferLoad(buffers["a"], [0]), [0])], buffers)
    assert _events(result) == ["V_S"]


@pytest.mark.parametrize("sides,expected", [((1, 1), True), ((0, 0), True), ((0, 1), False)])
def test_sync_vs_resource_scope_same_side_keeps_outstanding_writer(sides, expected):
    buffers = _buffers()
    first = ir.AttrStmt(ir.IntImm("int32", 0), "resource_scope", sides[0], _access("V", buffers))
    second = ir.AttrStmt(ir.IntImm("int32", 0), "resource_scope", sides[1], _access("S", buffers))
    body = ir.AttrStmt(ir.IntImm("int32", 0), "resource_scope", sides[0], _seq([first, second]))
    assert ("V_S" in _events(_run(body, buffers))) == expected


def test_sync_vs_existing_auto_barrier_prevents_duplicate_for_prior_writer():
    buffers = _buffers()
    result = _run([_access("V", buffers), _intrin("ascend_auto_barrier", "PIPE_V"), _access("V", buffers)], buffers)
    assert _syncs(result) == [("barrier", "PIPE_V")]


def test_sync_vs_existing_auto_event_pair_prevents_duplicate_for_prior_writer():
    buffers = _buffers()
    result = _run(
        [
            _access("V", buffers),
            _intrin("ascend_auto_set_flag", "V_S", 5),
            _intrin("ascend_auto_wait_flag", "V_S", 5),
            _access("S", buffers),
        ],
        buffers,
    )
    assert _events(result) == ["V_S"]
    assert [s[2] for s in _syncs(result)] == [5, 5]


def test_sync_vs_event_ids_rotate_modulo_eight_and_pairs_match():
    buffers = _buffers()
    result = _run([_access(p, buffers) for _ in range(6) for p in ["S", "V"]], buffers)
    assert len(_events(result)) == 11
    assert [s[2] for s in _syncs(result) if s[0] == "set_flag"] == [(i + 1) % 8 for i in range(11)]


@pytest.mark.parametrize("prefix", [None, "MTE2"])
def test_sync_vs_loop_revisit_keeps_vector_back_edge(prefix):
    buffers = _buffers()
    loop = ir.For(ir.Var("i", "int32"), 0, 3, ir.ForKind.SERIAL, _access("V", buffers))
    statements = ([] if prefix is None else [_access(prefix, buffers)]) + [loop]
    result = _run(statements, buffers)
    assert ("barrier", "PIPE_V") in _syncs(result)
    assert not _events(result)


def test_sync_vs_if_branches_analyze_from_same_entry_state():
    buffers = _buffers()
    branch = ir.IfThenElse(ir.Var("condition", "bool"), _access("V", buffers), _access("V", buffers))
    result = _run([branch, _access("V", buffers)], buffers)
    assert _syncs(result) == [("barrier", "PIPE_V")]


def test_sync_vs_unsupported_matrix_pipeline_does_not_create_events():
    buffers = _buffers()
    matrix = _call("mma", buffers["input"].access_ptr("r"), buffers["b"].access_ptr("r"), buffers["a"].access_ptr("w"))
    result = _run([matrix, _access("S", buffers)], buffers)
    assert not _syncs(result)


def test_sync_vs_flat_bind_tail_keeps_scalar_read_and_writer_history():
    buffers = _buffers()
    value = ir.Var("value", "float32")
    body = _seq([_access("V", buffers), ir.Bind(value, ir.BufferLoad(buffers["a"], [0])), ir.Evaluate(value)])
    result = _run(body, buffers)
    assert _events(result) == ["V_S"]


def test_sync_vs_flat_allocation_tail_and_loop_revisit_keep_accesses():
    buffers = _buffers()
    body = _seq([ir.AllocBuffer(buffers["a"]), _access("V", buffers), _access("S", buffers)])
    result = _run(ir.For(ir.Var("i", "int32"), 0, 3, ir.ForKind.SERIAL, body), buffers)
    assert "V_S" in _events(result)
    assert "S_V" in _events(result)


@pytest.mark.parametrize("with_tmp", [False, True])
def test_sync_vs_real_language_optional_workspace_operand_contract(with_tmp):
    from tilelang.ascend.a2a3.language import tile

    buffers = _buffers()
    temporary = ir.decl_buffer((64,), "float32", name="temporary", scope="shared.ub")
    producer = ir.BufferStore(temporary if with_tmp else buffers["b"], ir.FloatImm("float32", 1), [0])
    call = tile.clamp(buffers["a"], buffers["b"], -1, 1, 64, tmp=temporary if with_tmp else None)
    result = _run([producer, ir.Evaluate(call)], buffers)
    assert _events(result) == ["S_V"]


def test_sync_vs_keeps_shared_manual_protocol_without_treating_it_as_auto():
    buffers = _buffers()
    manual = [
        _intrin("ascend_set_flag", "V_S", 3),
        _intrin("ascend_wait_flag", "V_S", 3),
        _intrin("ascend_pipe_barrier", "PIPE_V"),
    ]
    result = _run([_access("V", buffers), *manual, _access("S", buffers)], buffers)
    assert _events(result) == ["V_S"]  # Original pass only recognizes auto calls.
    leaves = []
    ir.stmt_functor.post_order_visit(result.body, lambda n: leaves.append(n) if isinstance(n, ir.Evaluate) else None)
    for original in manual:
        assert any(n.value.same_as(original.value) for n in leaves)


def test_sync_vs_flat_allocation_retains_source_annotation_reset():
    buffers = _buffers()
    body = _seq([ir.AllocBuffer(buffers["a"], {"double_buffer_scope": 1}), _access("V", buffers), _access("S", buffers)])
    result = _run(body, buffers)
    allocations = []
    ir.stmt_functor.post_order_visit(result.body, lambda n: allocations.append(n) if isinstance(n, ir.AllocBuffer) else None)
    assert len(allocations) == 1
    assert not allocations[0].annotations
    assert _events(result) == ["V_S"]
