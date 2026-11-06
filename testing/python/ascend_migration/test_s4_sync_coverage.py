"""Verify emitted events and original logical/final physical dependencies."""

import math

import pytest

import tilelang
from tilelang import tvm

from testing.python.ascend_migration.test_s2_task_ir import (
    _add,
    _buffer,
    _dma,
    _fill,
    _function,
)

tir = tvm.tir
TARGET = tvm.target.Target("c -keys=ascend -model=ascendc")


def _flag(kind, domain="MTE2_V", slot=0):
    return tir.Evaluate(
        tir.Call(
            "handle",
            tvm.ir.Op.get(f"tl.ascend_auto_{kind}_flag"),
            [tir.StringImm(domain), tir.const(slot, "int32")],
        )
    )


def _barrier(pipe="PIPE_V"):
    return tir.Evaluate(tir.Call("handle", tvm.ir.Op.get("tl.ascend_auto_barrier"), [tir.StringImm(pipe)]))


def _pair(domain, slot=0):
    return [_flag("set", domain, slot), _flag("wait", domain, slot)]


def _fixture(buffers, statements, emitted, *, addresses=None, noalias=True):
    source = _function(buffers, statements, noalias=noalias)
    graph = tilelang.transform.AnalyzeAscendSchedule(source, "A3")
    marked = [
        tir.AttrStmt(tir.const(index, "int64"), "tl.ascend_constraint_task_id", 1, statement) for index, statement in enumerate(statements)
    ]
    source = _function(buffers, marked, noalias=noalias)
    body = [marked[item] if isinstance(item, int) else item for item in emitted]
    final = _function(buffers, body, noalias=noalias)
    local = [buffer for buffer in buffers if buffer.scope() != "global"]
    size_map = {buffer.data: math.prod(map(int, buffer.shape)) * tvm.DataType(buffer.dtype).bits // 8 for buffer in local}
    offset = 0
    address_map = {}
    for buffer in local:
        address_map[buffer.data] = offset
        offset += size_map[buffer.data]
    if addresses:
        address_map.update({buffer.data: offset for buffer, offset in addresses.items()})
    final = final.with_attr("address_map", address_map)
    final = final.with_attr("size_map", size_map)
    return final.with_attr(
        "tl.ascend_constraint_plan",
        {
            "schema_version": 1,
            "source_graph": graph,
            "source_function": source,
            "candidates": [],
        },
    )


def _verify(function, *, strategy="constraint", target=TARGET, platform="A3"):
    module = tvm.IRModule({"main": function})
    before = tvm.ir.save_json(module)
    with tvm.transform.PassContext(config={"tl.ascend_pipeline_planner": strategy}):
        result = tilelang.transform.AscendVerifyConstraintSync(target, platform)(module)
    assert tvm.ir.save_json(module) == before
    assert result["main"].same_as(function)
    return result


def _copy_add_store(emitted=None):
    gm = _buffer("gm", "global")
    output = _buffer("output", "global")
    a, b = [_buffer(name) for name in "ab"]
    ops = [_dma(gm, a), _add(b, a, a), _dma(b, output)]
    if emitted is None:
        emitted = [0, *_pair("MTE2_V"), 1, *_pair("V_MTE3"), 2]
    return _fixture([gm, output, a, b], ops, emitted)


def test_audited_copy_add_store_is_verified_without_mutation():
    _verify(_copy_add_store())


def test_read_read_does_not_require_a_redundant_wait():
    a = _buffer("a")
    output, other = [_buffer(name, "global") for name in ("output", "other")]
    _verify(_fixture([a, output, other], [_dma(a, output), _dma(a, other)], [0, 1]))


@pytest.mark.parametrize("barrier", ["PIPE_V", "PIPE_ALL"])
def test_same_pipe_hazards_require_a_real_completion_boundary(barrier):
    a = _buffer("a")
    ops = [_fill(a), _add(a, a, a), _fill(a)]
    _verify(_fixture([a], ops, [0, _barrier(barrier), 1, _barrier(barrier), 2]))
    with pytest.raises(tvm.TVMError, match="Uncovered.*RAW/WAR/WAW"):
        _verify(_fixture([a], ops, [0, _barrier(barrier), 1, 2]))


@pytest.mark.parametrize("offset,overlap", [(0, True), (128, True), (256, False)])
@pytest.mark.parametrize("reordered", [False, True])
def test_physical_reuse_follows_final_order_not_source_rank(offset, overlap, reordered):
    a, b = [_buffer(name) for name in "ab"]
    ops = [_fill(a), _fill(b)]
    order = [1, 0] if reordered else [0, 1]
    addresses = {a: 0, b: offset}
    if overlap:
        with pytest.raises(tvm.TVMError, match="Uncovered.*RAW/WAR/WAW"):
            _verify(_fixture([a, b], ops, order, addresses=addresses))
        order = [order[0], _barrier(), order[1]]
    _verify(_fixture([a, b], ops, order, addresses=addresses))


def test_original_logical_waw_direction_survives_reordering():
    a = _buffer("a")
    with pytest.raises(tvm.TVMError, match="Uncovered.*RAW/WAR/WAW"):
        _verify(_fixture([a], [_fill(a), _fill(a)], [1, _barrier(), 0]))


@pytest.mark.parametrize("payload", ["fill_value", "dtype_and_count"])
def test_byte_effects_alone_do_not_prove_stable_call_identity(payload):
    a = _buffer("a")
    if payload == "fill_value":
        original = _fill(a)
        arguments = list(original.value.args)
        arguments[2] = tir.const(2.0, "float32")
        changed = tir.Evaluate(tir.Call("handle", original.value.op, arguments))
    else:
        original = _add(a, a, a)
        half_view = _buffer("half_view", dtype="float16", shape=(256,), data=a.data)
        changed = _add(half_view, half_view, half_view, count=128)
    final = tir.AttrStmt(tir.const(0, "int64"), "tl.ascend_constraint_task_id", 1, changed)
    with pytest.raises(tvm.TVMError, match="payload changed"):
        _verify(_fixture([a], [original], [final]))


@pytest.mark.parametrize("forward", [False, True])
def test_transitive_completion_respects_emitted_event_order(forward):
    gm, output = [_buffer(name, "global") for name in ("gm", "output")]
    a, b = [_buffer(name) for name in "ab"]
    ops = [_dma(gm, a), _fill(b), _dma(a, output)]
    emitted = [0, 1]
    if forward:
        emitted += [*_pair("MTE2_V"), *_pair("V_MTE3"), 2]
        _verify(_fixture([gm, output, a, b], ops, emitted))
    else:
        emitted += [*_pair("V_MTE3"), *_pair("MTE2_V"), 2]
        with pytest.raises(tvm.TVMError, match="Uncovered.*RAW/WAR/WAW"):
            _verify(_fixture([gm, output, a, b], ops, emitted))


@pytest.mark.parametrize(
    "prefix,suffix,diagnostic",
    [
        ([_flag("wait")], [], "wait without matching"),
        ([_flag("set"), _flag("set")], [], "Duplicate set"),
        ([_flag("set")], [], "Undrained"),
        ([_flag("set")], [_flag("wait", "V_MTE2")], "wait without matching"),
        ([_flag("set", slot=8)], [], "event slot"),
        ([_flag("set", "MTE2_UNKNOWN")], [], "Unsupported automatic event"),
    ],
)
def test_corrupt_event_protocol_is_rejected(prefix, suffix, diagnostic):
    a = _buffer("a")
    function = _fixture([a], [_fill(a)], [*prefix, 0, *suffix])
    with pytest.raises(tvm.TVMError, match=diagnostic):
        _verify(function)


def test_opposite_domains_and_consumed_slot_reuse_are_independent():
    a = _buffer("a")
    protocol = [
        _flag("set", "MTE2_V", 0),
        _flag("set", "V_MTE2", 0),
        _flag("wait", "V_MTE2", 0),
        _flag("wait", "MTE2_V", 0),
        *_pair("MTE2_V", 0),
        0,
    ]
    _verify(_fixture([a], [_fill(a)], protocol))


def test_pairing_alone_does_not_protect_a_later_producer():
    with pytest.raises(tvm.TVMError, match="Uncovered.*RAW/WAR/WAW"):
        _verify(_copy_add_store([_flag("set"), 0, _flag("wait"), 1, *_pair("V_MTE3"), 2]))


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "unmarked", "wrong_id"])
def test_stable_task_coverage_is_required(corruption):
    a, b = [_buffer(name) for name in "ab"]
    ops = [_fill(a), _fill(b)]
    emitted = {
        "missing": [0],
        "duplicate": [0, 0],
        "unmarked": [0, ops[1]],
        "wrong_id": [0, tir.AttrStmt(tir.const(9, "int64"), "tl.ascend_constraint_task_id", 1, ops[1])],
    }[corruption]
    with pytest.raises(tvm.TVMError, match="[Cc]onstraint|Unmarked"):
        _verify(_fixture([a, b], ops, emitted))


@pytest.mark.parametrize("boolean", [False, True])
def test_corrupt_iteration_identity_is_not_ignored(boolean):
    function = _copy_add_store()

    def insert(node):
        if isinstance(node, tir.AttrStmt) and node.attr_key == "tl.ascend_constraint_task_id":
            value = tir.const(False, "bool") if boolean else tir.const(0, "int32")
            body = tir.AttrStmt(tir.const(99, "int64"), "tl.ascend_constraint_iteration", value, node.body)
            return tir.AttrStmt(node.node, node.attr_key, node.value, body)
        return None

    body = tir.stmt_functor.ir_transform(function.body, None, insert, ["tir.AttrStmt"])
    with pytest.raises(tvm.TVMError, match="iteration"):
        _verify(function.with_body(body))


@pytest.mark.parametrize("corruption", ["missing_address", "missing_size", "unaligned", "too_small"])
def test_final_physical_metadata_is_validated(corruption):
    function = _copy_add_store()
    addresses = dict(function.attrs["address_map"].items())
    sizes = dict(function.attrs["size_map"].items())
    first = next(iter(addresses))
    if corruption == "missing_address":
        del addresses[first]
    elif corruption == "missing_size":
        del sizes[first]
    elif corruption == "unaligned":
        addresses[first] = 1
    else:
        sizes[first] = 32
    function = function.with_attr("address_map", addresses).with_attr("size_map", sizes)
    with pytest.raises(tvm.TVMError, match="physical"):
        _verify(function)


def test_final_physical_interval_cannot_exceed_native_ub_budget():
    # Deliberately corrupt final metadata; never invoke the old over-budget allocator.
    function = _copy_add_store()
    addresses = dict(function.attrs["address_map"].items())
    first = next(iter(addresses))
    addresses[first] = 196352 - 128
    with pytest.raises(tvm.TVMError, match="physical allocation interval"):
        _verify(function.with_attr("address_map", addresses))


@pytest.mark.parametrize("limit", ["tasks", "empty_control", "index_update"])
def test_final_expansion_limits_and_actual_loop_dtype_are_checked(limit):
    a = _buffer("a")
    if limit == "tasks":
        function = _fixture([a], [_fill(a)], [0] * 4097)
        diagnostic = "instance limit"
    else:
        i = tir.Var("i", "int32")
        minimum, extent = (0, 65537) if limit == "empty_control" else (2**31 - 1, 1)
        loop = tir.For(i, minimum, extent, tir.ForKind.SERIAL, tir.Evaluate(0))
        function = _fixture([a], [_fill(a)], [loop, 0])
        diagnostic = "control limit" if limit == "empty_control" else "endpoint/update"
    with pytest.raises(tvm.TVMError, match=diagnostic):
        _verify(function)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("total", [192, 160])
@pytest.mark.parametrize("inplace", [False, True])
def test_real_loop_and_tail_have_verified_single_version_coverage(dtype, total, inplace):
    from testing.python.ascend_migration.test_s4_kernels import (
        make_vector_kernel,
        vector_pass_configs,
    )

    with tvm.transform.PassContext(config=vector_pass_configs("constraint")):
        artifact = tilelang.lower(make_vector_kernel(total, dtype, inplace=inplace), target="ascendc", platform="A3")
    assert "AscendC::SetFlag" in artifact.kernel_source
    assert "AscendC::WaitFlag" in artifact.kernel_source


def test_unknown_gm_aliases_across_loop_iterations_are_not_waived():
    from testing.python.ascend_migration.test_s4_kernels import (
        make_vector_kernel,
        vector_pass_configs,
    )

    with (
        tvm.transform.PassContext(config=vector_pass_configs("constraint")),
        pytest.raises(tvm.TVMError, match="Uncovered.*RAW/WAR/WAW"),
    ):
        tilelang.lower(make_vector_kernel(noalias=False), target="ascendc", platform="A3")


def test_legacy_verifier_is_identity_without_new_rejections():
    raw = tir.PrimFunc([], tir.Evaluate(tir.call_extern("handle", "legacy_raw")))
    _verify(raw, strategy="legacy", target=tvm.target.Target("c -model=pto"), platform="A5")


@pytest.mark.parametrize("model,platform", [("pto", "A3"), ("ascendc", "A5")])
def test_constraint_verifier_target_gate(model, platform):
    with pytest.raises(tvm.TVMError, match="requires A3 AscendC/auto"):
        _verify(_copy_add_store(), target=tvm.target.Target(f"c -model={model}"), platform=platform)


def test_constraint_verifier_requires_manifest():
    with pytest.raises(tvm.TVMError, match="Missing constraint source manifest"):
        _verify(tir.PrimFunc([], tir.Evaluate(0)))
