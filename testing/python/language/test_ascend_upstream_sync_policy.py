# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""CPU IR tests for sync configuration masks and legacy target policies.

These fixtures call the migrated pass directly. They do not compile a kernel
or execute on an NPU, and their expected events come from each access pair.
"""

import pytest

import tilelang
from tilelang import tvm


tir = tvm.tir
_CONFIGURATIONS = [(False, False), (False, True), (True, False), (True, True)]
_SYNC_OPS = {
    "tl.ascend_auto_barrier",
    "tl.ascend_auto_set_flag",
    "tl.ascend_auto_wait_flag",
}


@pytest.fixture(autouse=True)
def _requires_proof_service() -> None:
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")


def _buffers() -> dict[str, tir.Buffer]:
    return {
        name: tir.decl_buffer((64,), "float32", name=name, scope=scope)
        for name, scope in (
            ("a", "shared.ub"),
            ("b", "shared.ub"),
            ("input", "global"),
            ("output", "global"),
        )
    }


def _copy(name: str, source: tir.Buffer, destination: tir.Buffer) -> tir.Evaluate:
    return tir.Evaluate(tir.call_extern("handle", name, source.access_ptr("r"), destination.access_ptr("w")))


def _fill(destination: tir.Buffer) -> tir.Evaluate:
    return tir.Evaluate(tir.call_extern("handle", "AscendC::Duplicate", destination.access_ptr("w")))


def _store(destination: tir.Buffer, value: tir.PrimExpr | None = None) -> tir.BufferStore:
    if value is None:
        value = tir.FloatImm("float32", 2.0)
    return tir.BufferStore(destination, value, [tir.IntImm("int32", 0)])


def _function(buffers: dict[str, tir.Buffer], statements: list[tir.Stmt]) -> tir.PrimFunc:
    body = tir.AttrStmt(tir.IntImm("int32", 0), "resource_scope", 1, tir.SeqStmt(statements))
    for name in ("a", "b"):
        buffer = buffers[name]
        body = tir.Allocate(buffer.data, buffer.dtype, [64], tir.const(True, "bool"), body)
    parameters = [buffers["input"], buffers["output"]]
    function = tir.PrimFunc(
        [buffer.data for buffer in parameters],
        body,
        buffer_map={buffer.data: buffer for buffer in parameters},
    )
    return function.with_attr(
        {
            "address_map": {buffers["a"].data: 0, buffers["b"].data: 1024},
            "size_map": {buffers["a"].data: 256, buffers["b"].data: 256},
            "tir.noalias": True,
        }
    )


def _run(function: tir.PrimFunc, *, ordinary: bool, vs: bool, platform: str = "A3") -> tir.PrimFunc:
    target = tvm.target.Target("c -keys=ascend -model=ascendc")
    with tvm.transform.PassContext(config={"tl.ascend_auto_sync": ordinary, "tl.ascend_auto_sync_vs": vs}):
        result = tilelang.transform.AscendInsertSync(target, platform)(tvm.IRModule({"main": function}))
    return result["main"]


def _sync_calls(function: tir.PrimFunc) -> list[tir.Call]:
    result = []

    def collect(node: object) -> None:
        if isinstance(node, tir.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name in _SYNC_OPS:
            result.append(node)

    tir.stmt_functor.post_order_visit(function.body, collect)
    return result


def _assert_pair(calls: list[tir.Call], event: str) -> None:
    matching = [call for call in calls if call.args[0].value == event]
    assert [call.op.name for call in matching] == [
        "tl.ascend_auto_set_flag",
        "tl.ascend_auto_wait_flag",
    ]
    assert tvm.ir.structural_equal(matching[0].args[1], matching[1].args[1])


@pytest.mark.parametrize("ordinary,vs", _CONFIGURATIONS)
@pytest.mark.parametrize("direction", ["MTE2_V", "V_MTE2", "MTE3_V", "V_MTE3", "MTE2_MTE3"])
def test_dma_handoffs_require_ordinary_sync(ordinary: bool, vs: bool, direction: str) -> None:
    buffers = _buffers()
    load = _copy("copy_gm_to_ub", buffers["input"], buffers["a"])
    store = _copy("copy_ub_to_gm", buffers["a"], buffers["output"])
    vector_read = _copy("copy_ub_to_ub", buffers["a"], buffers["b"])
    vector_write = _fill(buffers["a"])
    statements = {
        "MTE2_V": [load, vector_read],
        "V_MTE2": [vector_write, load],
        "MTE3_V": [store, vector_write],
        "V_MTE3": [vector_write, store],
        "MTE2_MTE3": [load, store],
    }[direction]
    original = _function(buffers, statements)
    result = _run(original, ordinary=ordinary, vs=vs)
    calls = _sync_calls(result)
    if ordinary:
        _assert_pair(calls, direction)
    else:
        assert not calls
    if not ordinary and not vs:
        assert tvm.ir.structural_equal(result, original)
        assert result.attrs.get("tl.ascend_sync_proof") is None


@pytest.mark.parametrize("ordinary,vs", _CONFIGURATIONS)
@pytest.mark.parametrize("platform", ["A2", "A3", "A5"])
def test_vector_self_handoff_obeys_platform_and_configuration(ordinary: bool, vs: bool, platform: str) -> None:
    buffers = _buffers()
    original = _function(
        buffers,
        [_fill(buffers["a"]), _copy("copy_ub_to_ub", buffers["a"], buffers["b"])],
    )
    calls = _sync_calls(_run(original, ordinary=ordinary, vs=vs, platform=platform))
    if not ordinary and not vs:
        assert not calls
    elif platform == "A5":
        _assert_pair(calls, "V_V")
    else:
        assert len(calls) == 1
        assert calls[0].op.name == "tl.ascend_auto_barrier"
        assert calls[0].args[0].value == "PIPE_V"


@pytest.mark.parametrize("ordinary,vs", _CONFIGURATIONS)
@pytest.mark.parametrize("direction", ["S_V", "V_S", "S_MTE2", "MTE2_S", "S_MTE3", "MTE3_S"])
def test_scalar_handoffs_obey_both_sync_options(ordinary: bool, vs: bool, direction: str) -> None:
    buffers = _buffers()
    scalar_write = _store(buffers["a"])
    scalar_read = _store(buffers["b"], tir.BufferLoad(buffers["a"], [0]))
    vector_read = _copy("copy_ub_to_ub", buffers["a"], buffers["b"])
    vector_write = _fill(buffers["a"])
    load = _copy("copy_gm_to_ub", buffers["input"], buffers["a"])
    store = _copy("copy_ub_to_gm", buffers["a"], buffers["output"])
    statements = {
        "S_V": [scalar_write, vector_read],
        "V_S": [vector_write, scalar_read],
        "S_MTE2": [scalar_write, load],
        "MTE2_S": [load, scalar_read],
        "S_MTE3": [scalar_write, store],
        "MTE3_S": [store, scalar_write],
    }[direction]
    calls = _sync_calls(_run(_function(buffers, statements), ordinary=ordinary, vs=vs))
    if ordinary or vs:
        _assert_pair(calls, direction)
    else:
        assert not calls


def test_scalar_self_dependency_needs_no_hardware_event() -> None:
    buffers = _buffers()
    original = _function(buffers, [_store(buffers["a"]), _store(buffers["b"], tir.BufferLoad(buffers["a"], [0]))])
    assert not _sync_calls(_run(original, ordinary=False, vs=True))


def _manual_flag(kind: str, event_id: int) -> tir.Evaluate:
    return tir.Evaluate(
        tir.Call(
            "handle",
            tvm.ir.Op.get(f"tl.ascend_{kind}_flag"),
            [tir.StringImm("MTE2"), tir.StringImm("V"), tir.IntImm("int32", event_id)],
        )
    )


def test_all_reserved_event_slots_reject_a_real_automatic_dependency() -> None:
    buffers = _buffers()
    dependency = [
        _copy("copy_gm_to_ub", buffers["input"], buffers["a"]),
        _fill(buffers["a"]),
    ]
    # Independently establish that this access pair actually needs MTE2_V,
    # rather than obtaining a capacity error from an invalid manual protocol.
    baseline = _run(_function(buffers, dependency), ordinary=True, vs=True)
    _assert_pair(_sync_calls(baseline), "MTE2_V")
    reserved = [_manual_flag(kind, event_id) for event_id in range(8) for kind in ("set", "wait")]
    _run(_function(buffers, reserved), ordinary=True, vs=True)
    with pytest.raises(
        tvm.error.TVMError,
        match=r"capacity exhausted for MTE2_V \(8 slots, including explicit reservations\)",
    ):
        _run(_function(buffers, reserved + dependency), ordinary=True, vs=True)


@pytest.mark.parametrize("event_id", [-1, 8])
def test_manual_event_id_outside_the_hardware_range_is_rejected(event_id: int) -> None:
    buffers = _buffers()
    statements = [_manual_flag("set", event_id), _manual_flag("wait", event_id)]
    with pytest.raises(tvm.error.TVMError, match=r"literal manual event ID in \[0, 8\)"):
        _run(_function(buffers, statements), ordinary=True, vs=True)
