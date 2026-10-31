"""Native packed-int4 contracts; no SDK compilation or device execution."""

import pytest
from tilelang import tvm

from test_a2a3_codegen import _allocate, _make_func, _source

X = tvm.tirx


@pytest.mark.parametrize("location", ["global", "shared.ub"])
def test_odd_int4_row_is_rejected_before_source_emission(location):
    buffer = X.decl_buffer((4, 17), "int4", name="odd", scope=location)
    body = X.Evaluate(0) if location == "global" else _allocate([buffer], [X.Evaluate(0)])
    buffers = (buffer,) if location == "global" else ()
    with pytest.raises(tvm.error.InternalError, match="odd contiguous dimension"):
        _source(_make_func(body, buffers=buffers))


@pytest.mark.parametrize("dtype,width", [("int4", 18), ("float16", 17), ("float32", 17)])
def test_packed_even_rows_and_non_int4_odd_rows_remain_accepted(dtype, width):
    buffer = X.decl_buffer((4, width), dtype, name="input", scope="global")
    assert "main_kernel" in _source(_make_func(X.Evaluate(0), buffers=(buffer,)))


def test_dynamic_int4_row_retains_upstream_deferred_validation():
    width = X.Var("width", "int32")
    buffer = X.decl_buffer((4, width), "int4", name="dynamic", scope="global")
    assert "int64_t width" in _source(_make_func(X.Evaluate(0), buffers=(buffer,)))


@pytest.mark.parametrize(
    "dtype,elements,expected_bytes",
    [("int4", 64, 32), ("int4", 65, 64), ("int4", 4096, 2048), ("float16", 4096, 8192), ("float32", 4096, 16384)],
)
def test_planner_uses_packed_bits_before_byte_and_allocation_rounding(dtype, elements, expected_bytes):
    # The planner is independently callable: its ceil(bits / 8) contract also
    # covers odd totals rejected later by the AscendC contiguous-row validator.
    buffers = [X.decl_buffer((elements,), dtype, name=name, scope="shared.ub") for name in ("a", "b")]
    body = X.SeqStmt(
        [
            *(X.AllocBuffer(buffer) for buffer in buffers),
            *(X.BufferStore(buffer, X.const(1, dtype), [0]) for buffer in buffers),
        ]
    )
    func = X.PrimFunc([], body).with_attr(
        {"global_symbol": "main", "logic_buffer_shapes": {buffer.data: list(buffer.shape) for buffer in buffers}}
    )
    with tvm.transform.PassContext(config={"tl.ascend_memory_planning": False}):
        after = tvm.ffi.get_global_func("tl.transform.A2A3AscendMemoryPlanning")()(tvm.IRModule.from_expr(func))["main"]
    assert {var.name: int(value) for var, value in after.attrs["size_map"].items()} == {"a": expected_bytes, "b": expected_bytes}
    assert {var.name: int(value) for var, value in after.attrs["address_map"].items()} == {"a": 0, "b": expected_bytes}


def test_pto_packed_int4_rejection_is_actionable_without_a_device():
    source = X.decl_buffer((4, 64), "float16", name="source", scope="shared.ub")
    quantized = X.decl_buffer((4, 64), "int4", name="quantized", scope="shared.ub")
    cast = X.Evaluate(
        X.call_intrin("handle", tvm.ir.Op.get("tl.ascend_cast"), quantized.access_ptr("w"), source.access_ptr("r"), "CAST_RINT", 256)
    )
    func = _make_func(
        _allocate([source, quantized], [cast]),
        attrs={"logic_buffer_shapes": {buffer.data: [4, 64, 4, 64] for buffer in (source, quantized)}},
    )
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "keys": ["pto", "ascend"]})
    with pytest.raises(tvm.error.InternalError, match="int4 is not supported by the pto target"):
        tvm.ffi.get_global_func("target.build.tilelang_pto_a2a3")(tvm.IRModule.from_expr(func), target, "A3")
