"""TA resource ownership and CV composition, including the shared sync ABI."""

import pytest

from tilelang import tvm
from tvm import s_tir
from tilelang.ascend.a2a3 import transform


tir = tvm.tirx


def _seq(*stmts):
    return tir.SeqStmt(list(stmts)) if len(stmts) > 1 else stmts[0]


def _scope(owner, body):
    return tir.AttrStmt(tir.const(0), "resource_scope", owner, body)


def _call(name, *args):
    return tir.Evaluate(tir.call_intrin("handle", tvm.ir.Op.get(name), *args))


def _root(body, buffers=(), name="tilelang_root", params=()):
    block = tir.SBlock([], [], [], name, body, alloc_buffers=list(buffers))
    return tir.PrimFunc([buffer.data for buffer in params], tir.SBlockRealize([], True, block), buffer_map={b.data: b for b in params})


def _apply(func, *, verify=False, combine=True, sync=False):
    factory = transform.AscendResourceScopeVerify if verify else transform.CombineCV
    with tvm.transform.PassContext(config={"tl.ascend_auto_cv_combine": combine, "tl.ascend_auto_cross_core_sync": sync}):
        return factory()(tvm.IRModule({"main": func}))["main"]


def _at_noop_phase(func):
    # Unit-test the later compatibility pass independently. Production keeps
    # its original full pipeline ordering; this does not move NoOp earlier.
    return transform.RemoveNoOp()(tvm.IRModule({"main": func}))["main"]


def _nodes(body, cls):
    result = []
    tir.stmt_functor.post_order_visit(body, lambda node: result.append(node) if isinstance(node, cls) else None)
    return result


def _owner_nodes(func, owner, cls):
    # Later NoOp may remove an entirely empty resource_scope. Do not require
    # the original two-element Combine Seq to survive that separate phase.
    parts = [node.body for node in _nodes(func.body, tir.AttrStmt) if node.attr_key == "resource_scope" and int(node.value) == owner]
    return [node for part in parts for node in _nodes(part, cls)]


def _parts(func):
    return func.body.block.body.seq


def _stores(body):
    return [store.buffer.name for store in _nodes(body, tir.BufferStore)]


@pytest.mark.parametrize(
    "scope,owner", [("shared.ub", 1), ("shared.l1", 0), ("wmma.matrix_a", 0), ("wmma.matrix_b", 0), ("wmma.accumulator", 0)]
)
def test_local_buffer_owner_is_preserved(scope, owner):
    buffer = tir.decl_buffer((16,), "float32", name="data", scope=scope)
    result = _apply(_root(tir.BufferStore(buffer, tir.const(3, "float32"), [0]), [buffer]))
    assert _stores(_parts(result)[owner]) == ["data"]
    assert _stores(_parts(result)[1 - owner]) == []
    _apply(result, verify=True)


@pytest.mark.parametrize("name", ["tl.ascend_set_flag", "tl.ascend_wait_flag", "tl.ascend_auto_set_flag", "tl.ascend_auto_wait_flag"])
@pytest.mark.parametrize("event,owner", [("MTE2_V", 1), ("V_MTE3", 1), ("MTE2_MTE1", 0), ("M_FIX", 0)])
def test_sync_event_uses_two_argument_hard_event(name, event, owner):
    result = _apply(_root(_call(name, event, 3)))
    for index, part in enumerate(_parts(result)):
        calls = [c for c in _nodes(part, tir.Call) if c.op.name == name]
        assert len(calls) == (1 if index == owner else 0)
        if calls:
            assert len(calls[0].args) == 2
            assert calls[0].args[0].value == event
            assert int(calls[0].args[1]) == 3


@pytest.mark.parametrize("pipe,owner", [("PIPE_V", 1), ("PIPE_M", 0), ("PIPE_FIX", 0), ("PIPE_MTE1", 0)])
def test_pipe_owner(pipe, owner):
    result = _apply(_root(_call("tl.ascend_pipe_barrier", pipe)))
    assert len(_nodes(_parts(result)[owner], tir.Call)) == 1
    assert not _nodes(_parts(result)[1 - owner], tir.Call)


@pytest.mark.parametrize("scope,owner", [("shared.ub", 1), ("shared.l1", 0)])
def test_contextual_barrier_uses_unambiguous_neighbor_work(scope, owner):
    buffer = tir.decl_buffer((16,), "float32", name="local", scope=scope)
    work = tir.BufferStore(buffer, tir.const(1, "float32"), [0])
    result = _apply(_root(_seq(work, _call("tl.ascend_pipe_barrier", "PIPE_ALL"), work), [buffer]))
    assert len(_nodes(_parts(result)[owner], tir.Call)) == 1
    assert not _nodes(_parts(result)[1 - owner], tir.Call)


@pytest.mark.parametrize("owner", [0, 1])
def test_source_call_requires_and_preserves_explicit_scope(owner):
    call = _call("tl.ascend_src_code", "custom();")
    with pytest.raises(tvm.error.InternalError, match="explicit|Unscoped"):
        _apply(_root(call))
    result = _apply(_root(_scope(owner, call)))
    assert len(_nodes(_parts(result)[owner], tir.Call)) == 1
    assert not _nodes(_parts(result)[1 - owner], tir.Call)
    _apply(result, verify=True)


def test_scope_conflict_and_unowned_global_store_are_rejected():
    ub = tir.decl_buffer((16,), "float32", scope="shared.ub")
    gm = tir.decl_buffer((16,), "float32")
    with pytest.raises(tvm.error.InternalError, match="conflict|Vector|resource_scope"):
        _apply(_root(_scope(0, tir.BufferStore(ub, tir.const(1, "float32"), [0])), [ub]))
    with pytest.raises(tvm.error.InternalError, match="GM|explicit"):
        _apply(_root(tir.BufferStore(gm, tir.const(1, "float32"), [0]), params=[gm]))
    result = _apply(_root(_scope(1, tir.BufferStore(gm, tir.const(1, "float32"), [0])), params=[gm]))
    assert _stores(_parts(result)[1]) == [gm.name]
    assert not _stores(_parts(result)[0])


def test_shared_global_bind_prevents_implicit_cache_owner_and_survives_multiple_uses():
    gm = tir.decl_buffer((16,), "float32", name="gm")
    ub = tir.decl_buffer((16,), "float32", name="ub", scope="shared.ub")
    scalar = tir.Var("scalar", "float32")
    read = tir.Bind(scalar, tir.BufferLoad(gm, [0]))
    work = tir.BufferStore(ub, scalar + scalar, [0])
    result = _apply(_root(_seq(read, work), [ub], params=[gm]))
    for part in _parts(result):
        binds = _nodes(part, tir.Bind)
        assert len(binds) == 1
        assert binds[0].var.same_as(scalar)
        assert binds[0].value.buffer.same_as(gm)
    cache = _call("tl.ascend_datacachecleanandinvalid_experiment", 0, gm.access_ptr("r"))
    with pytest.raises(tvm.error.InternalError, match="explicit|Unscoped"):
        _apply(_root(_seq(read, work, cache), [ub], params=[gm]))


def test_disabled_combine_and_non_root_block_preserve_input():
    func = _root(_call("tl.ascend_src_code", "custom();"))
    assert _apply(func, combine=False).same_as(func)
    renamed = _root(tir.Evaluate(0), name="other")
    assert tvm.ir.structural_equal(_apply(renamed), renamed)


def test_verifier_requires_explicit_owner_after_splitting():
    buffer = tir.decl_buffer((16,), "float32", scope="shared.ub")
    func = _root(tir.BufferStore(buffer, tir.const(1, "float32"), [0]), [buffer])
    with pytest.raises(tvm.error.InternalError, match="resource_scope|scope|Unscoped"):
        _apply(func, verify=True)
    _apply(_apply(func), verify=True)


def _workspace_func(staged=False, interval=1, primitive=False):
    cube = tir.decl_buffer((16,), "float32", name="cube", scope="wmma.accumulator")
    vector = tir.decl_buffer((16,), "float32", name="vector", scope="shared.ub")
    workspace = tir.decl_buffer((16,), "float32", name="workspace_cv")
    write = tir.Evaluate(
        tir.call_extern("handle", "tl::ascend::copy_l0c_to_gm<float, float>", cube.access_ptr("r"), workspace.access_ptr("w"))
    )
    read = tir.Evaluate(tir.call_extern("handle", "tl::ascend::copy_gm_to_ub<float>", workspace.access_ptr("r"), vector.access_ptr("w")))
    if staged:

        def loop(body):
            return tir.For(
                tir.Var("stage", "int32"),
                0,
                4,
                tir.ForKind.SERIAL,
                body,
                annotations={"stage_loop": True, "tl_cross_interval": interval if primitive else tir.const(interval)},
            )

        write, read = loop(write), loop(read)
    return _root(_seq(write, read), [cube, vector], params=[workspace])


@pytest.mark.parametrize("staged,interval", [(False, 1), (True, 1), (True, 2)])
def test_auto_cross_sync_preserves_copy_and_gates_only_flags(staged, interval):
    func = _workspace_func(staged, interval)
    assert not [c for c in _nodes(_apply(func).body, tir.Call) if "cross_flag" in c.op.name]
    result = _apply(func, sync=True)
    flags = [c for c in _nodes(result.body, tir.Call) if "cross_flag" in c.op.name]
    assert sorted(c.op.name for c in flags) == ["tl.ascend_auto_set_cross_flag", "tl.ascend_auto_wait_cross_flag"]
    copies = [c for c in _nodes(result.body, tir.Call) if c.op.name == "tirx.call_extern"]
    assert len(copies) == 2
    assert len(flags[0].args) == 3  # Auto cross-core flags retain their separate source ABI.
    assert len(flags[1].args) == 2
    if staged and interval > 1:
        assert len(_nodes(result.body, tir.IfThenElse)) == 2
    _apply(result, verify=True)


@pytest.mark.parametrize("name,args", [("tl.ascend_set_mask_mode", [0]), ("tl.ascend_set_mask_payload", [0, 0, 0])])
def test_mask_catalog_setters_remain_vector_owned(name, args):
    result = _apply(_root(_call(name, *args)))
    assert [call.op.name for call in _nodes(_parts(result)[1], tir.Call)] == [name]
    assert not _nodes(_parts(result)[0], tir.Call)


@pytest.mark.parametrize("owner,label", [(0, "free_pipe_C"), (1, "free_pipe_V")])
def test_free_pipe_call_keeps_source_resource_owner(owner, label):
    result = _apply(_root(_call("tl.ascend_free_pipe", label)))
    assert len(_nodes(_parts(result)[owner], tir.Call)) == 1
    assert not _nodes(_parts(result)[1 - owner], tir.Call)


def test_common_sync_is_duplicated_without_picking_one_core():
    result = _apply(_root(_call("tl.ascend_sync_all")))
    for part in _parts(result):
        assert [call.op.name for call in _nodes(part, tir.Call)] == ["tl.ascend_sync_all"]


def test_copy_resource_contract_rejects_incorrect_operand_scope():
    func = _workspace_func()
    bad = tir.Evaluate(
        tir.call_extern(
            "handle",
            "tl::ascend::copy_l0c_to_gm<float, float>",
            func.body.block.alloc_buffers[0].access_ptr("r"),
            func.body.block.alloc_buffers[1].access_ptr("w"),
        )
    )
    with pytest.raises(tvm.error.InternalError, match="Invalid copy operand storage scope"):
        _apply(_root(bad, func.body.block.alloc_buffers))


def test_auto_cross_sync_mismatch_keeps_source_rejection():
    func = _workspace_func()
    write, read = list(func.body.block.body.seq)
    body = _seq(write, write, read)
    with pytest.raises(tvm.error.TVMError, match="Mismatch in sync points"):
        _apply(_root(body, func.body.block.alloc_buffers, params=list(func.buffer_map.values())), sync=True)


@pytest.mark.parametrize("interval", [1, 2])
def test_cross_sync_integer_annotation_representations_match(interval):
    boxed = _apply(_workspace_func(True, interval), sync=True)
    primitive = _apply(_workspace_func(True, interval, primitive=True), sync=True)
    # The annotation itself intentionally keeps its TIRX representation; only
    # the generated flag conditions must agree.
    left = _nodes(boxed.body, tir.IfThenElse)
    right = _nodes(primitive.body, tir.IfThenElse)
    assert len(left) == len(right)
    assert all(tvm.ir.structural_equal(a.condition, b.condition, map_free_vars=True) for a, b in zip(left, right))


@pytest.mark.parametrize("chain", [False, True])
def test_split_preserves_foreign_read_until_original_noop_phase(chain):
    ub = tir.decl_buffer((16,), "int32", name="ub", scope="shared.ub")
    scalar, derived, unused = (tir.Var(name, "int32") for name in ("scalar", "derived", "unused"))
    body = _seq(
        tir.Bind(scalar, tir.BufferLoad(ub, [0])),
        tir.Bind(derived, scalar + 1),
        tir.IfThenElse(derived > 0, tir.Bind(unused, derived * 2), None),
        tir.BufferStore(ub, tir.const(1), [0]),
    )
    if not chain:
        body = _seq(
            tir.Bind(scalar, tir.BufferLoad(ub, [0])),
            tir.IfThenElse(scalar > 0, tir.Evaluate(0), None),
            tir.BufferStore(ub, tir.const(1), [0]),
        )
    result = _apply(_root(body, [ub]))
    assert len(_nodes(_parts(result)[0], tir.BufferLoad)) == 1
    assert _stores(_parts(result)[1]) == ["ub"]
    cleaned = _at_noop_phase(result)
    assert not _owner_nodes(cleaned, 0, tir.BufferLoad)
    assert [node.buffer.name for node in _owner_nodes(cleaned, 1, tir.BufferStore)] == ["ub"]
    _apply(cleaned, verify=True)


def test_live_cube_consumer_of_ub_bind_remains_invalid():
    ub = tir.decl_buffer((16,), "float32", scope="shared.ub")
    cube = tir.decl_buffer((16,), "float32", scope="shared.l1")
    scalar = tir.Var("scalar", "float32")
    read = tir.Bind(scalar, tir.BufferLoad(ub, [0]))
    result = _apply(_root(_seq(read, tir.BufferStore(cube, scalar, [0])), [ub, cube]))
    assert _nodes(_parts(result)[0], tir.Bind)[0].var.same_as(scalar)
    with pytest.raises(tvm.error.InternalError, match="Vector operation.*Scope"):
        _apply(result, verify=True)
    cleaned = _at_noop_phase(result)
    assert _owner_nodes(cleaned, 0, tir.Bind)[0].var.same_as(scalar)
    with pytest.raises(tvm.error.InternalError, match="Vector operation.*Scope"):
        _apply(cleaned, verify=True)
    with pytest.raises(tvm.error.InternalError, match="Vector operation.*Scope"):
        _apply(_root(_scope(0, read), [ub]))


def test_foreign_read_cleanup_preserves_shared_gm_scalar_consumed_by_both_sides():
    gm = tir.decl_buffer((16,), "float32", name="gm")
    ub = tir.decl_buffer((16,), "float32", name="ub", scope="shared.ub")
    cube = tir.decl_buffer((16,), "float32", name="cube", scope="shared.l1")
    common, dead = (tir.Var("same_hint", "float32") for _ in range(2))
    result = _apply(
        _root(
            _seq(
                tir.Bind(common, tir.BufferLoad(gm, [0])),
                tir.Bind(dead, tir.BufferLoad(ub, [0])),
                tir.BufferStore(cube, common, [0]),
                tir.BufferStore(ub, common, [0]),
            ),
            [ub, cube],
            params=[gm],
        )
    )
    for part in _parts(result):
        assert any(bind.var.same_as(common) for bind in _nodes(part, tir.Bind))
        assert any(bind.var.same_as(dead) for bind in _nodes(part, tir.Bind))
    cleaned = _at_noop_phase(result)
    for part in _parts(cleaned):
        assert any(bind.var.same_as(common) for bind in _nodes(part, tir.Bind))
        assert not any(bind.var.same_as(dead) for bind in _nodes(part, tir.Bind))
    _apply(cleaned, verify=True)


def test_foreign_read_cleanup_keeps_effectful_binding():
    ub = tir.decl_buffer((16,), "int32", scope="shared.ub")
    from tilelang.ascend.a2a3.language.constant import allocate_const

    effect, dead = (tir.Var(name, "int32") for name in ("effect", "dead"))
    literal = tir.Var("literal", "handle")
    result = _apply(
        _root(
            _seq(
                _scope(0, tir.Bind(effect, tir.call_extern("int32", "side_effect"))),
                tir.Bind(literal, allocate_const([7, 11], "int32", [2])),
                tir.Bind(dead, tir.BufferLoad(ub, [0])),
            ),
            [ub],
        )
    )
    calls = _nodes(_parts(result)[0], tir.Call)
    assert any(call.op.name == "tirx.call_extern" and call.args[0].value == "side_effect" for call in calls)
    assert any(call.op.name == "tl.a2a3_const_data" for call in calls)
    assert len(_nodes(_parts(result)[0], tir.BufferLoad)) == 1
    cleaned = _at_noop_phase(result)
    calls = _owner_nodes(cleaned, 0, tir.Call)
    assert any(call.op.name == "tirx.call_extern" and call.args[0].value == "side_effect" for call in calls)
    assert any(call.op.name == "tl.a2a3_const_data" for call in calls)
    assert not _owner_nodes(cleaned, 0, tir.BufferLoad)


@pytest.mark.parametrize("volatile", [False, True])
def test_raw_nested_consumer_or_volatile_read_blocks_visible_ancestor_cleanup(volatile):
    ub = tir.decl_buffer((16,), "int32", scope="shared.ub")
    gm = tir.decl_buffer((16,), "int32", name="gm")
    named, dead = (tir.Var(name, "int32") for name in ("named", "dead"))
    tail = (
        tir.AttrStmt(ub.data, "volatile_scope", 1, tir.Evaluate(0))
        if volatile
        else _scope(0, tir.IfThenElse(tir.BufferLoad(gm, [1]) > 0, _call("tl.ascend_src_code", "consume(named);"), None))
    )
    result = _apply(_root(_seq(tir.Bind(named, tir.BufferLoad(gm, [0])), tir.Bind(dead, tir.BufferLoad(ub, [0])), tail), [ub], params=[gm]))
    binds = _nodes(_parts(result)[0], tir.Bind)
    assert any(bind.var.same_as(named) for bind in binds)
    assert any(bind.var.same_as(dead) for bind in binds)
    # Conservative barriers preserve the original invalid read for the checker;
    # raw C++ names must never be inferred from an absent explicit Var use.
    with pytest.raises(tvm.error.InternalError, match="Vector operation.*Scope"):
        _apply(result, verify=True)


def test_second_allocation_plan_sees_all_four_source_scalar_reads_in_both_domains():
    # Source-derived VL1 schema: two scalar UB views in work_i, then two in t.
    # Early C-read removal makes these shared descriptors appear Vector-only.
    # This is an IR phase test, not an initialized tensor/numerical execution.
    names = ["scalar_fp32", "softplus_val", "scalar2_fp32", "norm_val"]
    scalars = [tir.decl_buffer((1,), "float32", name=name, scope="shared.ub") for name in names]
    destination = tir.decl_buffer((1,), "float32", name="destination", scope="shared.ub")
    values = [tir.Var("value_" + name, "float32") for name in names]
    work_i, t = tir.Var("work_i", "int32"), tir.Var("t", "int32")
    inner = tir.SBlock(
        [],
        [],
        [],
        "",
        _seq(tir.Bind(values[2], scalars[2][0]), tir.Bind(values[3], scalars[3][0]), tir.BufferStore(destination, sum(values), [0])),
        alloc_buffers=scalars[2:],
    )
    outer = tir.SBlock(
        [],
        [],
        [],
        "",
        _seq(
            tir.Bind(values[0], scalars[0][0]),
            tir.Bind(values[1], scalars[1][0]),
            tir.For(t, 0, 2, tir.ForKind.SERIAL, tir.SBlockRealize([], True, inner)),
        ),
        alloc_buffers=scalars[:2],
    )
    source = _root(tir.For(work_i, 0, 3, tir.ForKind.SERIAL, tir.SBlockRealize([], True, outer)), [destination])
    combined = _apply(source)
    for part in _parts(combined):
        loads = _nodes(part, tir.BufferLoad)
        assert {load.buffer.name for load in loads} == set(names)
        assert all(any(load.buffer.same_as(buffer) for load in loads) for buffer in scalars)
    planned = s_tir.transform.PlanAndUpdateBufferAllocationLocation()(tvm.IRModule({"main": combined}))["main"]
    allocations = {}

    def walk(stmt, owner=-1):
        if isinstance(stmt, tir.AttrStmt):
            if stmt.attr_key == "resource_scope":
                owner = int(stmt.value)
            walk(stmt.body, owner)
        elif isinstance(stmt, tir.SeqStmt):
            for child in stmt.seq:
                walk(child, owner)
        elif isinstance(stmt, tir.SBlockRealize):
            walk(stmt.block, owner)
        elif isinstance(stmt, tir.SBlock):
            for buffer in stmt.alloc_buffers:
                if buffer.name in names:
                    allocations.setdefault(buffer.name, []).append((owner, buffer))
            if stmt.init is not None:
                walk(stmt.init, owner)
            walk(stmt.body, owner)
        elif isinstance(stmt, tir.AllocBuffer):
            if stmt.buffer.name in names:
                allocations.setdefault(stmt.buffer.name, []).append((owner, stmt.buffer))
        elif hasattr(stmt, "body"):
            walk(stmt.body, owner)

    walk(planned.body)
    assert set(allocations) == set(names)
    for buffer in scalars:
        assert allocations[buffer.name]
        assert all(owner == -1 and actual.same_as(buffer) for owner, actual in allocations[buffer.name])
