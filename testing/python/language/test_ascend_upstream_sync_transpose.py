# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Host-only synchronization tests for the actual AscendC transpose helper."""

import pytest

import tilelang
from tilelang import tvm

tir = tvm.tir

_DISPATCH_CASES = [
    pytest.param("float16", (16, 16), "V", id="b16-instruction"),
    pytest.param("float16", (32, 64), "V", id="b16-block"),
    pytest.param("float32", (32, 32), "V", id="b32-block"),
    pytest.param("float32", (8, 16), "S", id="b32-eight-scalar"),
    pytest.param("int32", (24, 24), "S", id="b32-twenty-four-scalar"),
    pytest.param("int8", (32, 32), "S", id="int8-scalar"),
    pytest.param("bfloat16", (16, 32), "S", id="bfloat16-scalar"),
    pytest.param("int64", (64, 64), "S", id="int64-scalar"),
]


def _function(dtype: str, shape: tuple[int, int]) -> tir.PrimFunc:
    rows, columns = shape
    elements = rows * columns
    element_bytes = tvm.DataType(dtype).bits // 8
    source = tir.decl_buffer(shape, dtype, name="source")
    output = tir.decl_buffer((columns, rows), dtype, name="output")
    first = tir.decl_buffer(shape, dtype, name="first", scope="shared.ub")
    second = tir.decl_buffer((columns, rows), dtype, name="second", scope="shared.ub")
    body = tir.SeqStmt(
        [
            tir.Evaluate(tir.call_extern("handle", "copy_gm_to_ub", source.access_ptr("r"), first.access_ptr("w"))),
            tir.Evaluate(_transpose(second.access_ptr("w"), first.access_ptr("r"))),
            tir.Evaluate(tir.call_extern("handle", "copy_ub_to_gm", second.access_ptr("r"), output.access_ptr("w"))),
        ]
    )
    # No resource_scope: scalar transpose must still be assigned to AIV.
    for buffer in (first, second):
        body = tir.Allocate(buffer.data, buffer.dtype, [elements], tir.const(True), body)
    return tir.PrimFunc(
        [source.data, output.data],
        body,
        buffer_map={source.data: source, output.data: output},
    ).with_attr(
        {
            "address_map": {first.data: 0, second.data: elements * element_bytes},
            "size_map": {first.data: elements * element_bytes, second.data: elements * element_bytes},
            "logic_buffer_shapes": {first.data: list(shape), second.data: [columns, rows]},
            "tir.noalias": True,
        }
    )


def _transpose(destination: tir.Call, source: tir.Call) -> tir.Call:
    return tir.Call("handle", tvm.ir.Op.get("tl.ascend_transpose"), [destination, source])


def _run(function: tir.PrimFunc, backend: str, model: str = "ascendc") -> tir.PrimFunc:
    target = tvm.target.Target(f"c -keys=ascend -model={model}")
    module = tvm.IRModule({"main": function})
    with tvm.transform.PassContext(config={"tl.ascend_auto_sync": True, "tl.ascend_auto_sync_vs": True}):
        if backend == "upstream":
            if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
                pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")
            module = tilelang.transform.AscendInsertSync(target, "A3")(module)
        else:
            module = tilelang.transform.AscendSyncInsert(target, "A3")(module)
            module = tilelang.transform.AscendSyncInsertVS(target, "A3")(module)
    return module["main"]


def _events(function: tir.PrimFunc) -> set[str]:
    events = set()

    def visit(node: object) -> None:
        if (
            isinstance(node, tir.Call)
            and isinstance(node.op, tvm.ir.Op)
            and node.op.name in {"tl.ascend_auto_set_flag", "tl.ascend_auto_wait_flag"}
        ):
            events.add(node.args[0].value)

    tir.stmt_functor.post_order_visit(function.body, visit)
    return events


@pytest.mark.parametrize("backend", ["upstream", "legacy"])
@pytest.mark.parametrize("dtype,shape,pipe", _DISPATCH_CASES)
def test_transpose_handoffs_match_actual_ascendc_helper(
    backend: str,
    dtype: str,
    shape: tuple[int, int],
    pipe: str,
) -> None:
    events = _events(_run(_function(dtype, shape), backend))
    assert f"MTE2_{pipe}" in events
    assert f"{pipe}_MTE3" in events
    other = "V" if pipe == "S" else "S"
    assert f"MTE2_{other}" not in events
    assert f"{other}_MTE3" not in events


@pytest.mark.parametrize("backend", ["upstream", "legacy"])
def test_pto_transpose_keeps_ttrans_vector_handoffs(backend: str) -> None:
    events = _events(_run(_function("float32", (8, 24)), backend, "pto"))
    assert "MTE2_V" in events
    assert "V_MTE3" in events
    assert "MTE2_S" not in events
    assert "S_MTE3" not in events


@pytest.mark.parametrize("backend", ["upstream", "legacy"])
@pytest.mark.parametrize("shape", [None, (0, 16), (-1, 16), ("dynamic", 16)])
def test_transpose_rejects_unavailable_or_invalid_helper_shape(
    backend: str,
    shape: tuple[int | str, int] | None,
) -> None:
    function = _function("float32", (16, 16))
    if shape is None:
        function = function.without_attr("logic_buffer_shapes")
    else:
        original = function.attrs["logic_buffer_shapes"]
        source = next(variable for variable in original if variable.name == "first")
        rows = tir.Var("dynamic", "int32") if shape[0] == "dynamic" else shape[0]
        changed = dict(original.items())
        changed[source] = [rows, shape[1]]
        function = function.with_attr("logic_buffer_shapes", changed)
    with pytest.raises(tvm.error.InternalError, match="Transpose synchronization requires"):
        _run(function, backend)


@pytest.mark.parametrize("backend", ["upstream", "legacy"])
def test_transpose_rejects_dtype_width_not_preserved_by_the_emitter(backend: str) -> None:
    with pytest.raises(tvm.error.InternalError, match="audited element type"):
        _run(_function("float64", (16, 16)), backend)
