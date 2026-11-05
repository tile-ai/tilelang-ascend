"""TA annotation conversion contracts that need no Ascend native library."""

import numpy as np
import pytest

from tilelang import tvm
import tilelang.ascend.a2a3.language as T
from tvm.script.ir_builder import IRBuilder
from tvm.tirx.script.builder import ir as B


def _annotated(value):
    buffer = tvm.tirx.decl_buffer((16,), name="manual", scope="shared.ub")
    with IRBuilder() as builder, B.prim_func():
        B.func_name("annotated")
        with B.sblock("root"):
            T.annotate_address({buffer: value})
            B.evaluate(0)
    blocks = []
    tvm.tirx.stmt_functor.post_order_visit(
        builder.get().body,
        lambda node: blocks.append(node) if isinstance(node, tvm.tirx.SBlock) and "address_map" in node.annotations else None,
    )
    entries = list(blocks[0].annotations["address_map"].items())
    assert len(entries) == 1 and entries[0][0].same_as(buffer.data)
    return entries[0][1]


@pytest.mark.parametrize(
    "value,dtype",
    [
        (0, "int32"),
        (2**31 - 1, "int32"),
        (2**31, "int64"),
        (-(2**31), "int32"),
        (-(2**31) - 1, "int64"),
        (2**63 - 1, "int64"),
        (-(2**63), "int64"),
        (np.int32(65536), "int32"),
        (np.int64(2**32), "int64"),
        (True, "bool"),
        (False, "bool"),
    ],
)
def test_source_integer_and_bool_literal_types(value, dtype):
    result = _annotated(value)
    assert isinstance(result, tvm.tirx.IntImm)
    assert str(result.dtype) == dtype and result.value == value


@pytest.mark.parametrize("value,dtype", [(1.5, "float32"), (1e100, "float64"), (np.float32(1.5), "float32")])
def test_source_float_frontend_acceptance_is_distinct_from_planner_legality(value, dtype):
    # TA's planner assumes IntImm without checking. Do not send these source
    # frontend values to that unsafe native branch or claim valid addresses.
    result = _annotated(value)
    assert isinstance(result, tvm.tirx.FloatImm)
    assert str(result.dtype) == dtype and result.value == value


@pytest.mark.parametrize("expr", [tvm.tirx.IntImm("int64", 45), tvm.tirx.Var("offset", "int32")])
def test_existing_expression_identity_survives(expr):
    assert _annotated(expr).same_as(expr)


@pytest.mark.parametrize("value", [2**63, -(2**63) - 1, 2**100 + 7])
def test_checked_new_ffi_does_not_reproduce_legacy_integer_overflow(value):
    with pytest.raises(OverflowError):
        _annotated(value)


@pytest.mark.parametrize("value", [object(), np.bool_(True)])
def test_source_unknown_python_objects_are_rejected_in_frontend(value):
    with pytest.raises(ValueError, match="don't know how to convert type"):
        _annotated(value)


def test_source_complex_inference_error_is_preserved():
    with pytest.raises(NotImplementedError, match="Cannot automatically inference"):
        _annotated(1 + 2j)


@pytest.mark.parametrize("value", [None, "0"])
def test_source_nonexpression_frontend_values_are_not_coerced_to_addresses(value):
    assert _annotated(value) == value


def test_nonbuffer_key_and_nonmapping_keep_source_attribute_errors():
    with pytest.raises(AttributeError, match="data"):
        T.annotate_address({object(): 0})
    with pytest.raises(AttributeError, match="items"):
        T.annotate_address([])
