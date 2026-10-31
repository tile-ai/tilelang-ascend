"""Pinned TA dead-Let behavior for backend-local flat Bind cleanup."""

import importlib.util
from pathlib import Path

import pytest
import tilelang
from tilelang import tvm
from tvm import tirx


@pytest.fixture
def cleanup():
    factory = tvm.ffi.get_global_func("tl.transform.A2A3RemoveNoOp", allow_missing=True)
    if factory is None:
        pytest.skip("requires USE_ASCEND=ON with rebuilt A2A3RemoveNoOp")
    return factory()


def _run(cleanup, body, buffers=(), params=(), attrs=None):
    params = list(params) + [buffer.data for buffer in buffers]
    func = tirx.PrimFunc(params, body, buffer_map={buffer.data: buffer for buffer in buffers})
    func = func.with_attr({"global_symbol": "main", **(attrs or {})})
    return cleanup(tvm.IRModule({"main": func}))["main"]


def _nodes(body, kind):
    result = []
    tirx.stmt_functor.post_order_visit(body, lambda node: result.append(node) if isinstance(node, kind) else None)
    return result


@pytest.mark.parametrize("scope", ["global", "shared", "shared.ub"])
def test_unused_read_and_dependent_binding_chain_disappear(cleanup, scope):
    source = tirx.decl_buffer((4,), "int32", name="source", scope=scope)
    x, y = tirx.Var("x", "int32"), tirx.Var("y", "int32")
    func = _run(cleanup, tirx.SeqStmt([tirx.Bind(x, source[0]), tirx.Bind(y, x + 1), tirx.Evaluate(0)]), [source])
    assert not _nodes(func.body, tirx.Bind)
    assert not _nodes(func.body, tirx.BufferLoad)


def test_live_common_global_read_keeps_identity_and_consumer(cleanup):
    source = tirx.decl_buffer((4,), "int32", name="source")
    output = tirx.decl_buffer((4,), "int32", name="output")
    value = tirx.Var("value", "int32")
    func = _run(cleanup, tirx.SeqStmt([tirx.Bind(value, source[0]), tirx.BufferStore(output, value, [0])]), [source, output])
    bindings = _nodes(func.body, tirx.Bind)
    assert len(bindings) == 1 and bindings[0].var.same_as(value)
    assert _nodes(func.body, tirx.BufferStore)[0].value.same_as(value)


@pytest.mark.parametrize("control", ["for", "if", "while"])
def test_dead_read_inside_control_cleans_shell_without_losing_live_store(cleanup, control):
    source = tirx.decl_buffer((4,), "int32", name="source")
    output = tirx.decl_buffer((4,), "int32", name="output")
    value = tirx.Var("value", "int32")
    predicate = tirx.Var("condition", "int32")
    body = tirx.Bind(value, source[0])
    if control == "for":
        body = tirx.For(tirx.Var("i", "int32"), 0, 4, tirx.ForKind.SERIAL, body)
    elif control == "if":
        body = tirx.IfThenElse(predicate > 0, body, None)
    else:
        body = tirx.While(predicate > 0, body)
    func = _run(cleanup, tirx.SeqStmt([body, tirx.BufferStore(output, 7, [0])]), [source, output], [predicate])
    assert not _nodes(func.body, tirx.Bind) and not _nodes(func.body, tirx.BufferLoad)
    stores = _nodes(func.body, tirx.BufferStore)
    assert len(stores) == 1 and int(stores[0].value) == 7


def test_same_name_distinct_var_identities_are_not_merged(cleanup):
    source = tirx.decl_buffer((4,), "int32", name="source")
    output = tirx.decl_buffer((4,), "int32", name="output")
    dead, live = tirx.Var("same", "int32"), tirx.Var("same", "int32")
    body = tirx.SeqStmt([tirx.Bind(dead, source[0]), tirx.Bind(live, source[1]), tirx.BufferStore(output, live, [0])])
    func = _run(cleanup, body, [source, output])
    bindings = _nodes(func.body, tirx.Bind)
    assert len(bindings) == 1 and bindings[0].var.same_as(live) and not bindings[0].var.same_as(dead)


def test_unused_impure_result_becomes_evaluate_preserving_call_order(cleanup):
    value = tirx.Var("unused", "int32")
    first = tirx.call_extern("int32", "first_effect", 1)
    second = tirx.call_extern("int32", "second_effect", 2)
    body = tirx.SeqStmt([tirx.Bind(value, first), tirx.Evaluate(second)])
    func = _run(cleanup, body)
    assert not _nodes(func.body, tirx.Bind)
    calls = [node for node in _nodes(func.body, tirx.Call) if node.op.same_as(tvm.ir.Op.get("tirx.call_extern"))]
    assert [node.args[0].value for node in calls] == ["first_effect", "second_effect"]


def test_ordinary_extern_only_keeps_explicit_operands_not_all_bindings(cleanup):
    source = tirx.decl_buffer((4,), "int32", name="source")
    dead, live = tirx.Var("dead", "int32"), tirx.Var("live", "int32")
    body = tirx.SeqStmt(
        [tirx.Bind(dead, source[0]), tirx.Bind(live, source[1]), tirx.Evaluate(tirx.call_extern("int32", "unknown_external", live))]
    )
    func = _run(cleanup, body, [source])
    bindings = _nodes(func.body, tirx.Bind)
    assert len(bindings) == 1 and bindings[0].var.same_as(live)


def test_effectful_buffer_predicate_is_preserved_across_fixed_point(cleanup):
    source = tirx.decl_buffer((4,), "int32", name="source")
    dead = tirx.Var("dead", "int32")
    predicate = tirx.call_extern("bool", "predicate_effect")
    load = tirx.BufferLoad(source, [0], predicate=predicate)
    func = _run(cleanup, tirx.Bind(dead, load), [source])
    bindings = _nodes(func.body, tirx.Bind)
    assert len(bindings) == 1 and bindings[0].var.same_as(dead)
    assert bindings[0].value.predicate.same_as(predicate)


@pytest.mark.parametrize("kind", ["function", "for", "call", "attr_node", "descriptor"])
def test_real_metadata_and_descriptor_uses_keep_binding(cleanup, kind):
    source = tirx.decl_buffer((4,), "int32", name="source")
    value = tirx.Var("meta", "int32")
    tail, attrs = tirx.Evaluate(0), {}
    if kind == "function":
        attrs = {"metadata": {"nested": [value]}}
    elif kind == "for":
        tail = tirx.For(
            tirx.Var("i", "int32"),
            0,
            1,
            tirx.ForKind.SERIAL,
            tirx.Evaluate(tirx.call_extern("int32", "effect")),
            annotations={"metadata": value},
        )
    elif kind == "call":
        tail = tirx.Evaluate(tirx.Call("int32", "tirx.call_extern", [tirx.StringImm("effect")], annotations={"metadata": value}))
    elif kind == "attr_node":
        tail = tirx.AttrStmt(value, "metadata", 1, tirx.Evaluate(tirx.call_extern("int32", "effect")))
    else:
        view = tirx.decl_buffer((4,), "int32", name="view", data=source.data, strides=[value], elem_offset=value)
        tail = tirx.Evaluate(tirx.call_extern("int32", "consume", view[0]))
    func = _run(cleanup, tirx.SeqStmt([tirx.Bind(value, source[0]), tail]), [source], attrs=attrs)
    assert any(node.var.same_as(value) for node in _nodes(func.body, tirx.Bind))


def test_raw_text_preserves_visible_bindings_without_protecting_later_dead_read(cleanup):
    source = tirx.decl_buffer((4,), "int32", name="source")
    visible, later = tirx.Var("raw_name", "int32"), tirx.Var("later", "int32")
    raw = tirx.call_intrin("handle", tvm.ir.Op.get("tl.ascend_src_code"), tirx.StringImm("int copied = raw_name;"))
    body = tirx.SeqStmt([tirx.Bind(visible, source[0]), tirx.Evaluate(raw), tirx.Bind(later, source[1])])
    func = _run(cleanup, body, [source])
    bindings = _nodes(func.body, tirx.Bind)
    assert len(bindings) == 1 and bindings[0].var.same_as(visible)


@pytest.mark.parametrize("scope", ["attr", "for", "sblock", "if", "while"])
def test_volatile_inside_scope_kept_outside_plain_dead_read_removed(cleanup, scope):
    source = tirx.decl_buffer((4,), "int32", name="source")
    inside, outside = tirx.Var("inside", "int32"), tirx.Var("outside", "int32")
    condition = tirx.Var("condition", "int32")
    volatile = tirx.AttrStmt(source.data, "volatile_scope", 1, tirx.Bind(inside, source[0]))
    if scope == "for":
        volatile = tirx.For(tirx.Var("i", "int32"), 0, 2, tirx.ForKind.SERIAL, volatile)
    elif scope == "sblock":
        volatile = tirx.SBlockRealize([], True, tirx.SBlock([], [], [], "scope", volatile))
    elif scope == "if":
        volatile = tirx.IfThenElse(condition > 0, volatile, None)
    elif scope == "while":
        volatile = tirx.While(condition > 0, volatile)
    func = _run(cleanup, tirx.SeqStmt([volatile, tirx.Bind(outside, source[1])]), [source], [condition])
    bindings = _nodes(func.body, tirx.Bind)
    assert any(node.var.same_as(inside) for node in bindings)
    assert not any(node.var.same_as(outside) for node in bindings)


@pytest.mark.parametrize("dialect", ["ascendc", "pto"])
def test_original_full_paged_flash_factory_has_no_unused_vector_metadata_read(cleanup, dialect):
    from tilelang.ascend.a2a3.pass_config import normalize_a2a3_pass_configs

    path = Path(__file__).parents[5] / "examples/ascend_a2a3/flash_attention/paged_flash_attn_bhsd.py"
    spec = importlib.util.spec_from_file_location("original_full_paged_regression", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    keys = ["ascend"] if dialect == "ascendc" else ["pto", "ascend"]
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": keys})
    factory = module.paged_flash_attention_fwd
    func = factory._get_tir_with_target(target, 1, 16, 128, 128, 1, 1)
    config = normalize_a2a3_pass_configs(target, factory.pass_configs)
    with tvm.transform.PassContext(opt_level=3, config=config):
        artifact = tilelang.lower(func, target=target)
    reads = {"C": 0, "V": 0}

    def walk(stmt, owner=None):
        if isinstance(stmt, tirx.AttrStmt):
            if stmt.attr_key == "resource_scope":
                owner = "C" if int(stmt.value) == 0 else "V"
            walk(stmt.body, owner)
        elif isinstance(stmt, tirx.SeqStmt):
            for child in stmt.seq:
                walk(child, owner)
        elif isinstance(stmt, tirx.Bind):
            for load in _nodes(tirx.Evaluate(stmt.value), tirx.BufferLoad):
                if "block_table" in load.buffer.name and owner is not None:
                    reads[owner] += 1
        elif hasattr(stmt, "block"):
            walk(stmt.block.body, owner)
        elif hasattr(stmt, "body"):
            walk(stmt.body, owner)

    lowered = next(iter(artifact.device_mod.functions.values()))
    walk(lowered.body)
    assert reads == {"C": 1, "V": 0}
    assert len(artifact.params) == 8
    assert "void call(" in artifact.kernel_source


def test_shared_statement_DAG_occurrences_can_be_removed_on_only_one_branch(cleanup):
    source = tirx.decl_buffer((4,), "int32", name="source")
    output = tirx.decl_buffer((4,), "int32", name="output")
    condition, value = tirx.Var("condition", "int32"), tirx.Var("value", "int32")
    shared = tirx.Bind(value, source[0])
    branch = tirx.IfThenElse(condition > 0, tirx.SeqStmt([shared, tirx.BufferStore(output, value, [0])]), shared)
    func = _run(cleanup, branch, [source, output], [condition])
    bindings = _nodes(func.body, tirx.Bind)
    assert len(bindings) == 1 and bindings[0].var.same_as(value)
    assert len(_nodes(func.body, tirx.BufferStore)) == 1


def test_binding_used_only_by_alloc_annotation_is_live(cleanup):
    source = tirx.decl_buffer((4,), "int32", name="source")
    scratch = tirx.decl_buffer((4,), "int32", name="scratch", scope="shared")
    value = tirx.Var("annotation_value", "int32")
    allocation = tirx.AllocBuffer(scratch, annotations={"metadata": value})
    body = tirx.SeqStmt([tirx.Bind(value, source[0]), allocation, tirx.Evaluate(tirx.call_extern("int32", "effect"))])
    func = _run(cleanup, body, [source])
    assert any(node.var.same_as(value) for node in _nodes(func.body, tirx.Bind))
    assert any(node.annotations["metadata"].same_as(value) for node in _nodes(func.body, tirx.AllocBuffer))


@pytest.mark.parametrize("dataflow", [False, True])
def test_nondefault_shared_noop_configuration_is_preserved_for_live_program(cleanup, dataflow):
    source = tirx.decl_buffer((4,), "int32", name="source")
    output = tirx.decl_buffer((4,), "int32", name="output")
    value = tirx.Var("live", "int32")
    body = tirx.SeqStmt([tirx.Bind(value, source[0]), tirx.BufferStore(output, value, [0])])
    func = tirx.PrimFunc([source.data, output.data], body, buffer_map={source.data: source, output.data: output}).with_attr(
        "global_symbol", "main"
    )
    module = tvm.IRModule({"main": func})
    config = {"tirx.RemoveNoOp": {"use_dataflow_analysis": dataflow, "max_simplification_steps": 17}}
    with tvm.transform.PassContext(config=config):
        expected = tirx.transform.RemoveNoOp()(module)
        actual = cleanup(module)
    assert tvm.ir.structural_equal(expected, actual)


@pytest.mark.parametrize("dynamic", [False, True])
def test_nonunit_and_dynamic_loop_step_is_a_live_header_use(cleanup, dynamic):
    source = tirx.decl_buffer((4,), "int32", name="source")
    output = tirx.decl_buffer((8,), "int32", name="output")
    step = tirx.Var("step", "int32")
    index = tirx.Var("index", "int32")
    actual_step = step if dynamic else tirx.IntImm("int32", 2)
    loop = tirx.For(index, 0, 4, tirx.ForKind.SERIAL, tirx.BufferStore(output, 3, [index]), step=actual_step)
    func = _run(cleanup, tirx.SeqStmt([tirx.Bind(step, source[0]), loop]), [source, output])
    loops = _nodes(func.body, tirx.For)
    assert len(loops) == 1 and loops[0].step.same_as(actual_step)
    bindings = _nodes(func.body, tirx.Bind)
    assert any(node.var.same_as(step) for node in bindings) is dynamic
