"""Combine preserves source bindings; original later phases own cleanup."""

import pytest
from tilelang import tvm
from tvm import tirx as T


def _seq(*body):
    return T.SeqStmt(list(body))


def _nodes(stmt, cls):
    found = []
    T.stmt_functor.post_order_visit(stmt, lambda node: found.append(node) if isinstance(node, cls) else None)
    return found


def _work(gm, local, owner, offset=0, annotations=None):
    name = "copy_gm_to_ub" if owner else "copy_gm_to_l1"
    call = T.call_extern(
        "handle", "tl::ascend::" + name + "<int32_t, 16>", gm.access_ptr("r", offset=offset), local.access_ptr("w"), 16, 1, 1
    )
    if annotations:
        call = T.Call(call.dtype, call.op, call.args, annotations=annotations)
    return T.Evaluate(call)


def _case(owner=1, extent=227350, nested=False, body_kind="pure", loop_annotations=None):
    gm = T.decl_buffer((16,), "int32", name="gm")
    local = T.decl_buffer((16,), "int32", name="local", scope="shared.ub" if owner else "shared.l1")
    t = T.Var("t", "int32")
    first, second, third = (T.Var(name, "int32") for name in ("task", "src_off", "dst_off"))
    buffers = [local]
    work = _work(gm, local, owner)
    value = first + 1
    if body_kind == "load":
        value = T.BufferLoad(gm, [0])
    elif body_kind == "call":
        value = T.call_pure_extern("int32", "read_unknown", t)
    tail = _seq(T.Bind(second, value), T.Bind(third, second + 1), work)
    if body_kind == "store":
        cube = T.decl_buffer((16,), "int32", name="cube", scope="shared.l1")
        buffers.append(cube)
        tail = _seq(T.Bind(second, value), T.Bind(third, second + 1), T.BufferStore(cube, t, [0]), work)
    if nested:
        inner = T.Var("inner", "int32")
        tail = T.For(inner, 0, 3, T.ForKind.SERIAL, tail)
    condition = t < extent - 1
    loop = T.For(
        t, 0, extent, T.ForKind.SERIAL, _seq(T.Bind(first, t + 1), T.IfThenElse(condition, tail, None)), annotations=loop_annotations
    )
    block = T.SBlock([], [], [], "tilelang_root", loop, alloc_buffers=buffers)
    func = T.PrimFunc([gm.data], T.SBlockRealize([], True, block), buffer_map={gm.data: gm})
    return func, (first, second, third), loop, gm, local, condition


def _native_factory():
    factory = tvm.ffi.get_global_func("tl.transform.A2A3CombineCV", allow_missing=True)
    if factory is None:
        pytest.skip("requires USE_ASCEND=ON A2/A3 CombineCV")
    return factory


def _apply(func, combine=True):
    factory = _native_factory()
    with tvm.transform.PassContext(config={"tl.ascend_auto_cv_combine": combine}):
        return factory()(tvm.IRModule({"main": func}))["main"]


def _later_phase(func, name):
    # Direct pass contract tests, not a substitute production pipeline.
    # In production Simplify remains before Unroll and NoOp remains after it.
    factory = tvm.ffi.get_global_func("tl.transform." + name, allow_missing=True)
    if factory is None:
        pytest.skip("requires rebuilt A2/A3 compatibility component " + name)
    return factory()(tvm.IRModule({"main": func}))["main"]


def _parts(func):
    return func.body.block.body.seq


def _append(func, stmt):
    block = func.body.block
    new = T.SBlock(
        block.iter_vars,
        block.reads,
        block.writes,
        block.name_hint,
        _seq(block.body, stmt),
        alloc_buffers=block.alloc_buffers,
        annotations=block.annotations,
    )
    return func.with_body(T.SBlockRealize([], True, new))


@pytest.mark.parametrize("owner", [0, 1])
@pytest.mark.parametrize("nested", [False, True])
def test_combine_keeps_original_bindings_until_pre_unroll_simplify(owner, nested):
    func, bindings, original_loop, _, _, condition = _case(owner=owner, nested=nested)
    result = _apply(func)
    dead, live = _parts(result)[1 - owner], _parts(result)[owner]
    assert len(_nodes(dead, T.Bind)) == 3
    assert all(any(node.var.same_as(var) for node in _nodes(dead, T.Bind)) for var in bindings)
    assert len(_nodes(live, T.Bind)) == 3
    assert all(any(node.var.same_as(var) for node in _nodes(live, T.Bind)) for var in bindings)
    assert _nodes(dead, T.For)[-1].loop_var.same_as(original_loop.loop_var)
    assert _nodes(dead, T.IfThenElse)[0].condition.same_as(condition)
    assert len(_nodes(dead, T.For)) == (2 if nested else 1)
    assert all(loop.kind == T.ForKind.SERIAL for loop in _nodes(dead, T.For))
    assert len(_nodes(live, T.Call)) > 0
    # The original later Simplify point, not Combine, removes eligible pure
    # integer bindings before real UnrollLoop sees this full 227350 extent.
    simplified = _later_phase(result, "A2A3Simplify")
    assert not _nodes(_parts(simplified)[1 - owner], T.Bind)
    assert len(_nodes(_parts(simplified)[owner], T.Call)) == len(_nodes(live, T.Call))
    unrolled = T.transform.UnrollLoop()(tvm.IRModule({"main": simplified}))["main"]
    assert len(_nodes(unrolled.body, T.For)) == len(_nodes(simplified.body, T.For))
    assert max(len(node.seq) for node in _nodes(unrolled.body, T.SeqStmt)) < 20


@pytest.mark.parametrize("body_kind", ["load", "call", "store"])
def test_candidate_with_memory_or_any_call_keeps_bindings(body_kind):
    func, bindings, _, _, _, _ = _case(body_kind=body_kind, extent=5)
    result = _apply(func)
    assert any(node.var.same_as(bindings[1]) for node in _nodes(_parts(result)[0], T.Bind))


@pytest.mark.parametrize("where", ["function", "buffer_shape", "buffer_stride", "buffer_offset", "for", "block", "call", "value"])
def test_external_identity_and_metadata_use_prevents_new_cleanup(where):
    func, bindings, _, gm, local, _ = _case(extent=5)
    variable = bindings[2]
    if where == "function":
        func = func.with_attr("keep_nested", {variable: [{"value": variable}]})
    elif where.startswith("buffer_"):
        desc = T.decl_buffer(
            (variable + 1,) if where == "buffer_shape" else (16,),
            "int32",
            name="shape_carrier",
            strides=[variable] if where == "buffer_stride" else None,
            elem_offset=variable if where == "buffer_offset" else 0,
        )
        func = T.PrimFunc([gm.data, desc.data], func.body, buffer_map={gm.data: gm, desc.data: desc})
    elif where == "for":
        other = T.Var("other", "int32")
        func = _append(func, T.For(other, 0, 1, T.ForKind.SERIAL, T.Evaluate(0), annotations={"nested": {variable: [1]}}))
    elif where == "block":
        block = T.SBlock([], [], [], "other", T.Evaluate(0), annotations={"nested": {variable: [1]}})
        func = _append(func, T.SBlockRealize([], True, block))
    elif where == "call":
        func = _append(func, _work(gm, local, 1, annotations={"nested": {variable: [1]}}))
    else:
        func = _append(func, _work(gm, local, 1, offset=variable))
    result = _apply(func)
    assert any(node.var.same_as(variable) for node in _nodes(_parts(result)[0], T.Bind))


def test_same_name_other_var_is_not_an_external_use_of_the_dead_binding():
    func, bindings, _, _, _, _ = _case(extent=5)
    different = T.Var("dst_off", "int32")
    func = _append(func, T.Bind(different, 3)).with_attr("kept", {different: [different]})
    result = _apply(func)
    assert any(node.var.same_as(bindings[2]) for node in _nodes(_parts(result)[0], T.Bind))
    assert any(node.var.same_as(different) for node in _nodes(_parts(result)[0], T.Bind))
    cleaned = _later_phase(result, "A2A3RemoveNoOp")
    assert not any(node.var.same_as(bindings[2]) for node in _nodes(cleaned.body, T.Bind))
    assert any(node.var.same_as(different) for node in _nodes(cleaned.body, T.Bind))


@pytest.mark.parametrize("kind", ["raw", "volatile", "volatile_metadata", "const_data", "unknown_extern", "unknown_pure"])
def test_whole_kernel_unproven_names_and_effects_block_new_cleanup(kind):
    _native_factory()  # Private Op construction also requires this native backend.
    func, bindings, _, _, local, _ = _case(extent=5)
    if kind == "raw":
        stmt = T.Evaluate(T.call_intrin("handle", tvm.ir.Op.get("tl.ascend_src_code"), "consume(task);"))
    elif kind == "const_data":
        from tilelang.ascend.a2a3.language.constant import allocate_const

        stmt = T.Bind(T.Var("literal", "handle"), allocate_const([7], "int32", [1]))
    elif kind == "volatile_metadata":
        func = func.with_attr("source_annotations", {"volatile": True})
        stmt = T.Evaluate(0)
    elif kind == "volatile":
        stmt = T.AttrStmt(local.data, "volatile_scope", 1, T.Evaluate(0))
    elif kind == "unknown_extern":
        stmt = T.Evaluate(T.call_extern("int32", "unknown_even_with_UB", local.access_ptr("r")))
    else:
        stmt = T.Evaluate(T.call_pure_extern("int32", "unknown_pure", 1))
    func = _append(func, T.AttrStmt(T.const(0), "resource_scope", 1, stmt))
    result = _apply(func)
    assert any(node.var.same_as(bindings[0]) for node in _nodes(_parts(result)[0], T.Bind))


def test_annotations_disabled_split_and_neutral_program_do_not_enter_new_gate():
    func, _, _, _, _, _ = _case(extent=5, loop_annotations={"keep": True})
    result = _apply(func)
    assert len(_nodes(_parts(result)[0], T.Bind)) == 3
    assert _apply(func, combine=False).same_as(func)
    var, index = T.Var("plain", "int32"), T.Var("i", "int32")
    loop = T.For(index, 0, 5, T.ForKind.SERIAL, T.Bind(var, index + 1))
    block = T.SBlock([], [], [], "tilelang_root", loop)
    neutral = T.PrimFunc([], T.SBlockRealize([], True, block))
    neutral_result = _apply(neutral)
    assert all(any(node.var.same_as(var) for node in _nodes(part, T.Bind)) for part in _parts(neutral_result))
    other = T.SBlock([], [], [], "not_the_kernel_root", loop)
    untouched = T.PrimFunc([], T.SBlockRealize([], True, other))
    assert tvm.ir.structural_equal(_apply(untouched), untouched)


def test_binding_consumed_by_retained_control_is_not_partially_cleaned():
    func, bindings, loop, _, _, _ = _case(extent=5)
    branch = loop.body.seq[1]
    loop = T.For(
        loop.loop_var, loop.min, loop.extent, loop.kind, _seq(loop.body.seq[0], T.IfThenElse(bindings[0] > 0, branch.then_case, None))
    )
    block = func.body.block
    body = T.SBlock(
        block.iter_vars, block.reads, block.writes, block.name_hint, loop, alloc_buffers=block.alloc_buffers, annotations=block.annotations
    )
    result = _apply(func.with_body(T.SBlockRealize([], True, body)))
    assert all(any(node.var.same_as(var) for node in _nodes(_parts(result)[0], T.Bind)) for var in bindings)


def test_typed_call_metadata_does_not_turn_a_neutral_program_into_hardware_work():
    gm = T.decl_buffer((16,), "int32", name="gm")
    local = T.decl_buffer((16,), "int32", scope="shared.ub")
    scalar, index = T.Var("scalar", "int32"), T.Var("i", "int32")
    loop = T.For(index, 0, 5, T.ForKind.SERIAL, T.Bind(scalar, index + 1))
    block = T.SBlock([], [], [], "tilelang_root", loop)
    func = T.PrimFunc([gm.data], T.SBlockRealize([], True, block), buffer_map={gm.data: gm})
    func = func.with_attr("typed_metadata", _work(gm, local, 1).value)
    result = _apply(func)
    assert all(any(node.var.same_as(scalar) for node in _nodes(part, T.Bind)) for part in _parts(result))


def test_raw_function_metadata_is_conservatively_preserved():
    _native_factory()
    func, bindings, _, _, _, _ = _case(extent=5)
    raw = T.call_intrin("handle", tvm.ir.Op.get("tl.ascend_src_code"), "consume(task);")
    result = _apply(func.with_attr("raw_metadata", {"nested": [raw]}))
    assert any(node.var.same_as(bindings[0]) for node in _nodes(_parts(result)[0], T.Bind))


def test_legal_outer_header_binding_is_distinct_from_same_named_inner_binding():
    func, bindings, loop, _, _, _ = _case(extent=5)
    outer = T.Var("task", "int32")
    loop = T.For(loop.loop_var, loop.min, outer, loop.kind, loop.body)
    block = func.body.block
    body = T.SBlock(
        block.iter_vars, block.reads, block.writes, block.name_hint, _seq(T.Bind(outer, 5), loop), alloc_buffers=block.alloc_buffers
    )
    func = func.with_body(T.SBlockRealize([], True, body))
    assert T.analysis.verify_well_formed(func, assert_mode=False)
    result = _apply(func)
    dead = _parts(result)[0]
    assert any(node.var.same_as(outer) for node in _nodes(dead, T.Bind))
    assert any(node.var.same_as(bindings[0]) for node in _nodes(dead, T.Bind))
    assert _nodes(dead, T.For)[0].extent.same_as(outer)


def test_body_definition_is_not_a_legal_for_header_carrier():
    func, bindings, loop, _, _, _ = _case(extent=5)
    loop = T.For(loop.loop_var, loop.min, bindings[0], loop.kind, loop.body)
    block = func.body.block
    body = T.SBlock(block.iter_vars, block.reads, block.writes, block.name_hint, loop, alloc_buffers=block.alloc_buffers)
    assert not T.analysis.verify_well_formed(func.with_body(T.SBlockRealize([], True, body)), assert_mode=False)


@pytest.mark.parametrize("key", ["tirx.volatile", "volatile_scope"])
def test_canonical_and_legacy_volatile_metadata_keys_block_new_cleanup(key):
    func, bindings, _, _, _, _ = _case(extent=5)
    result = _apply(func.with_attr("source_annotations", {key: True}))
    assert any(node.var.same_as(bindings[0]) for node in _nodes(_parts(result)[0], T.Bind))


def _lexical_block_case(kind="plain", owner=1):
    # Source-derived from the original 5D transpose before CombineCV: the
    # serial builder wraps its three arithmetic binds and copies in an
    # anonymous block with read/write regions and a one-element UB allocation.
    func, bindings, loop, gm, local, condition = _case(owner=owner)
    alloc = T.decl_buffer((1,), "int32", name="row", scope="shared.ub")
    region = T.BufferRegion(gm, [tvm.ir.Range.from_min_extent(0, 16)])
    fields = dict(alloc_buffers=[alloc])
    reads, writes, body = [region], [region], loop.body
    predicate, values, variables, name = True, [], [], ""
    carrier = bindings[2]
    if kind == "predicate_false":
        predicate = False
    elif kind == "predicate_dynamic":
        predicate = loop.loop_var > 0
    elif kind == "iter_binding":
        values = [0]
        variables = [T.IterVar(tvm.ir.Range.from_min_extent(0, 1), T.Var("axis", "int32"), 0)]
    elif kind == "init":
        fields["init"] = T.Evaluate(0)
    elif kind == "match":
        alias = T.decl_buffer((16,), "int32", name="alias")
        fields["match_buffers"] = [T.MatchBufferRegion(alias, region)]
    elif kind == "annotations":
        fields["annotations"] = {"keep": True}
    elif kind == "named":
        name = "user_block"
    elif kind == "region":
        reads = [T.BufferRegion(gm, [tvm.ir.Range.from_min_extent(carrier, 16)])]
    elif kind == "write_region":
        writes = [T.BufferRegion(gm, [tvm.ir.Range.from_min_extent(carrier, 16)])]
    elif kind == "descriptor_data":
        alias = T.Var("local_alias", local.data.type_annotation)
        body = _seq(body, T.Bind(alias, local.data))
        fields["alloc_buffers"] = [T.decl_buffer((1,), "int32", data=alias, scope="shared.ub")]
    elif kind.startswith("descriptor_"):
        alloc = T.decl_buffer(
            (carrier + 1,) if kind == "descriptor_shape" else (1,),
            "int32",
            name="row",
            scope="shared.ub",
            strides=[carrier] if kind == "descriptor_stride" else None,
            elem_offset=carrier if kind == "descriptor_offset" else 0,
        )
        fields["alloc_buffers"] = [alloc]
    elif kind == "distinct_same_name":
        other = T.Var(carrier.name, "int32")
        fields["alloc_buffers"] = [T.decl_buffer((other + 1,), "int32", scope="shared.ub")]
    elif kind in ("load", "call", "store"):
        if kind == "load":
            extra = T.Bind(T.Var("read", "int32"), T.BufferLoad(gm, [0]))
        elif kind == "call":
            extra = T.Bind(T.Var("called", "int32"), T.call_pure_extern("int32", "unknown", 0))
        else:
            cube = T.decl_buffer((1,), "int32", scope="shared.l1")
            fields["alloc_buffers"] = [alloc, cube]
            extra = T.BufferStore(cube, 1, [0])
        body = _seq(body, extra)
    block = T.SBlock(variables, reads, writes, name, body, **fields)
    block_body = block if kind == "bare_block" else T.SBlockRealize(values, predicate, block)
    wrapped = T.For(loop.loop_var, loop.min, loop.extent, loop.kind, block_body)
    root = func.body.block
    root = T.SBlock(root.iter_vars, root.reads, root.writes, root.name_hint, wrapped, alloc_buffers=root.alloc_buffers)
    return func.with_body(T.SBlockRealize([], True, root)), bindings, loop, block, condition


@pytest.mark.parametrize("owner", [0, 1])
@pytest.mark.parametrize("kind", ["plain", "distinct_same_name"])
def test_actual_serial_block_shell_keeps_descriptors_and_vector_payload(owner, kind):
    func, bindings, loop, block, condition = _lexical_block_case(kind, owner)
    result = _apply(func)
    dead, live = _parts(result)[1 - owner], _parts(result)[owner]
    assert all(any(node.var.same_as(var) for node in _nodes(dead, T.Bind)) for var in bindings)
    assert all(any(node.var.same_as(var) for node in _nodes(live, T.Bind)) for var in bindings)
    for part in (dead, live):
        shell = _nodes(part, T.SBlock)[0]
        assert shell.reads[0].same_as(block.reads[0])
        assert shell.writes[0].same_as(block.writes[0])
        assert shell.alloc_buffers[0].same_as(block.alloc_buffers[0])
        assert _nodes(part, T.For)[0].loop_var.same_as(loop.loop_var)
        assert _nodes(part, T.IfThenElse)[0].condition.same_as(condition)
    before_calls = _nodes(func.body, T.Call)
    after_calls = _nodes(live, T.Call)
    assert len(before_calls) == len(after_calls)
    assert all(a.same_as(b) for a, b in zip(before_calls, after_calls))
    simplified = _later_phase(result, "A2A3Simplify")
    assert not _nodes(_parts(simplified)[1 - owner], T.Bind)
    unrolled = T.transform.UnrollLoop()(tvm.IRModule({"main": simplified}))["main"]
    assert len(_nodes(unrolled.body, T.For)) == 2
    assert max(len(node.seq) for node in _nodes(unrolled.body, T.SeqStmt)) < 20


@pytest.mark.parametrize(
    "kind",
    [
        "predicate_false",
        "predicate_dynamic",
        "iter_binding",
        "init",
        "match",
        "annotations",
        "named",
        "bare_block",
        "region",
        "write_region",
        "descriptor_data",
        "descriptor_shape",
        "descriptor_stride",
        "descriptor_offset",
        "load",
        "call",
        "store",
    ],
)
def test_nontrivial_or_live_serial_block_shell_keeps_bindings(kind):
    func, bindings, _, _, _ = _lexical_block_case(kind)
    result = _apply(func)
    assert all(any(node.var.same_as(var) for node in _nodes(_parts(result)[0], T.Bind)) for var in bindings)
