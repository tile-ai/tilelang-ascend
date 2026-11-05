"""TA's infinity is a native FloatImm, independent of the shared intrinsic."""

import math

import pytest

from tilelang import tvm
import tilelang.language as shared_language
import tilelang.ascend.a2a3.language as T
from tilelang.ascend.a2a3 import pipeline
from tilelang.ascend.a2a3.pass_config import normalize_a2a3_pass_configs


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_native_infinity_preserves_dtype_span_sign_and_json(dtype):
    span = tvm.ir.Span(tvm.ir.SourceName("infinity.py"), 3, 3, 2, 7)
    value = T.infinity(dtype, span)
    assert isinstance(value, tvm.tirx.FloatImm)
    assert value.dtype == dtype and math.isinf(value.value) and value.value > 0
    assert value.span.same_as(span)
    negative = -value
    assert negative.dtype == dtype
    if dtype == "float16":
        # The old and current native expression APIs both retain half negation
        # as Mul(FloatImm(inf), -1); do not impose float32 constant folding.
        assert isinstance(negative, tvm.tirx.Mul)
        assert math.isinf(negative.a.value) and float(negative.b.value) == -1
    else:
        assert isinstance(negative, tvm.tirx.FloatImm)
        assert math.isinf(negative.value) and negative.value < 0
    restored = tvm.ir.load_json(tvm.ir.save_json(negative))
    assert tvm.ir.structural_equal(restored, negative)
    assert isinstance(shared_language.infinity(dtype), tvm.tirx.Call)


@pytest.mark.parametrize("dtype", ["bfloat16", "int8", "int32", "int64", "uint32", "bool"])
def test_non_float_infinity_keeps_native_rejection(dtype):
    with pytest.raises((tvm.error.InternalError, ValueError), match="infinity"):
        T.infinity(dtype)


@pytest.mark.parametrize("dialect", ["ascendc", "pto"])
def test_negative_infinity_reaches_the_complete_native_pipeline(dialect):
    native = "target.build.tilelang_pto_a2a3" if dialect == "pto" else "target.build.tilelang_ascend_a2a3"
    builder = tvm.get_global_func(native, allow_missing=True)
    if builder is None:
        pytest.skip("A2/A3 source codegen requires USE_ASCEND=ON")

    @T.prim_func
    def kernel(A: T.Tensor((1,), "float32"), B: T.Tensor((1,), "float32")):
        with T.Kernel(1, is_npu=True), T.Scope("C"):
            B[0] = -T.infinity("float32")

    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": [dialect, "ascend"]})
    with tvm.transform.PassContext(config=normalize_a2a3_pass_configs(target)):
        mod = (
            pipeline._lower_pto(tvm.IRModule.from_expr(kernel), target)
            if dialect == "pto"
            else pipeline._lower_ascendc(tvm.IRModule.from_expr(kernel), target)
        )
        source = builder(mod, target, "A3").inspect_source()
    assert "CUDART_INF_F" in source
    calls = []
    tvm.tirx.stmt_functor.post_order_visit(
        next(iter(mod.functions.values())).body,
        lambda node: calls.append(node.op.name) if isinstance(node, tvm.tirx.Call) and isinstance(node.op, tvm.ir.Op) else None,
    )
    assert "tl.infinity" not in calls
