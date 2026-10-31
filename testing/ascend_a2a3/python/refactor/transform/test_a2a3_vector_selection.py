"""TA compiler-managed vector regressions, using raw pass input contracts."""

import pytest
from tilelang.ascend.a2a3 import transform
from tvm import IRModule
from tvm import tirx as tir
from tvm.ir import Op, assert_structural_equal
from tvm.target import Target

from _a2a3_vector_mask_cases import (
    ASCENDC,
    UINT64_MASK,
    _access,
    _add_fp32,
    _add_uint32,
    _call,
    _int,
    _name,
    _names,
    _select,
    _selected_add,
    _selected_call,
    _with_body,
    get_global_func,
)


def test_catalog_is_closed_and_all_terminals_are_registered():
    names = list(get_global_func("tl.transform.A2A3AscendVectorTerminalCatalog")())
    assert names
    assert len(names) == len(set(names))
    assert not any("datacachecleanandinvalid" in name for name in names)
    for name in names:
        assert Op.get(name).name == name


def test_selection_uses_normal_and_counter_without_dtype_fallback():
    normal = _selected_add(64)
    counter = _selected_add(96)
    assert _name(normal) == _name(counter) == "tl.ascend_add_raw"
    assert [arg.value for arg in normal.args[-4:-2]] == [0, 1]
    assert [arg.value for arg in counter.args[-4:-2]] == [1, 1]
    assert counter.args[-2].dtype == "uint64" and counter.args[-2].value == 96

    wrong_mask_args = list(normal.args)
    wrong_mask_args[-2] = tir.IntImm("uint64", 0x7FFFFFFF)
    wrong_mask = tir.Call(normal.dtype, normal.op, wrong_mask_args, normal.span)
    with pytest.raises(Exception, match="payload does not match"):
        _select(_with_body(_add_fp32, tir.Evaluate(wrong_mask)))

    wrong_count_args = list(counter.args)
    wrong_count_args[-2] = tir.IntImm("uint64", 64)
    wrong_count = tir.Call(counter.dtype, counter.op, wrong_count_args, counter.span)
    with pytest.raises(Exception, match="payload does not match"):
        _select(_with_body(_add_fp32, tir.Evaluate(wrong_count)))

    with pytest.raises(Exception, match=r"must be in \[0, 4294967295\]"):
        _selected_add(-1)

    with pytest.raises(Exception, match="Unsupported AscendC Vector dtype uint32.*no fallback"):
        _select(_with_body(_add_uint32, _add_uint32.body))


@pytest.mark.parametrize("operation", ["add", "cast"])
def test_counter_range_accepts_runtime_counts_and_checks_large_literals(operation):
    def selected_count(count):
        if operation == "add":
            selected = _selected_add(count)
            assert selected.args[-4].value == 1  # COUNTER mode
            return selected.args[-2]
        call = _call(
            "tl.ascend_cast",
            _access("float32", "cast_dst", access_mask=2),
            _access("float16", "cast_src", access_mask=1),
            tir.StringImm("CAST_NONE"),
            count,
        )
        return _selected_call(call, "tl.ascend_cast_raw_counter").args[-1]

    for count in [tir.const(0, "uint64"), tir.const((1 << 32) - 1, "uint64"), tir.Var("count", "uint64")]:
        assert_structural_equal(selected_count(count), count)

    for value in [-1, 1 << 32, 1 << 63, UINT64_MASK]:
        count = tir.const(value, "int64" if value < 0 else "uint64")
        with pytest.raises(Exception, match=r"must be in \[0, 4294967295\]"):
            selected_count(count)


def test_non_natural_capabilities_are_checked_before_selection():
    supported_cast = _call(
        "tl.ascend_cast",
        _access("float32", "cast_dst", access_mask=2),
        _access("float16", "cast_src", access_mask=1),
        tir.StringImm("CAST_NONE"),
        _int(64),
    )
    assert _name(_selected_call(supported_cast, "tl.ascend_cast_raw_counter")) == ("tl.ascend_cast_raw_counter")

    unsupported_mode = tir.Call(
        supported_cast.dtype,
        supported_cast.op,
        [*supported_cast.args[:2], tir.StringImm("CAST_RINT"), supported_cast.args[3]],
    )
    with pytest.raises(Exception, match="Unsupported AscendC Cast tuple"):
        _select(_with_body(_add_fp32, tir.Evaluate(unsupported_mode)))

    def gather(dtype, count):
        return _call(
            "tl.ascend_gather",
            _access(dtype, "gather_dst", access_mask=2),
            _access(dtype, "gather_src", access_mask=1),
            _access("uint32", "gather_offsets", access_mask=1),
            _int(0),
            _int(count),
        )

    with pytest.raises(Exception, match="Gather supports only 16-bit and 32-bit"):
        _selected_call(gather("uint8", 8), "tl.ascend_gather_count")

    assert _name(_selected_call(gather("float32", 16383), "tl.ascend_gather_count")) == ("tl.ascend_gather_count")
    with pytest.raises(Exception, match=r"Gather count must be in \[0, 16383\]"):
        _selected_call(gather("float32", 16384), "tl.ascend_gather_count")

    def block_reduce(repeat, mask):
        return _call(
            "tl.ascend_block_reduce_max",
            _access("float32", "reduce_dst", access_mask=2),
            _access("float32", "reduce_src", access_mask=1),
            *[value if isinstance(value, tir.PrimExpr) else _int(value) for value in (repeat, mask, 1, 1, 8)],
        )

    for call, message in [
        (block_reduce(1, 65), r"float32.*\[0, 64\]"),
        (block_reduce(256, 64), r"repeat must be in \[0, 255\]"),
        (block_reduce(-1, 64), r"repeat must be in \[0, 255\]"),
        (block_reduce(tir.const(UINT64_MASK, "uint64"), 64), r"repeat must be in \[0, 255\]"),
        (block_reduce(1, tir.const(UINT64_MASK, "uint64")), r"float32.*\[0, 64\]"),
    ]:
        with pytest.raises(Exception, match=message):
            _selected_call(call, "tl.ascend_block_reduce_max_raw_normal")


def test_mixed_scalar_buffers_are_specific_to_floating_divs_and_subs():
    def scalar_call(operation, vector_dtype="float16"):
        return _call(
            f"tl.ascend_{operation}",
            _access(vector_dtype, "scalar_dst", access_mask=2),
            _access(vector_dtype, "scalar_src", access_mask=1),
            _access("float32", "scalar_value", access_mask=1),
            _int(0),
            _int(64),
        )

    for operation in ["divs", "subs"]:
        selected = _selected_call(scalar_call(operation), f"tl.ascend_{operation}_raw")
        assert selected.args[2].args[0].value == "float32"
    for call in [scalar_call("adds"), scalar_call("subs", "int16")]:
        with pytest.raises(Exception, match="Unsupported AscendC Vector dtype tuple"):
            _select(_with_body(_add_fp32, tir.Evaluate(call)))


@pytest.mark.parametrize("platform", ["A2", "A3"])
def test_frozen_hardware_model_does_not_replace_ascendc_dialect(platform):
    target = Target({"kind": "ascend", "arch": "dav-2201", "model": platform, "keys": ["ascendc", "ascend"]})
    func = _with_body(_add_fp32, _add_fp32.body)
    result = transform.AscendVectorInstructionSelection(target, platform)(IRModule.from_expr(func))["main"]
    assert "tl.ascend_add_raw" in _names(result.body)


@pytest.mark.parametrize(
    "target,platform",
    [
        (Target("llvm"), "A2"),
        (Target({"kind": "ascend", "arch": "dav-3510"}), "A3"),
        (Target({"kind": "ascend", "arch": "dav-2201", "keys": ["pto", "ascend"]}), "A2"),
        (ASCENDC, "A5"),
    ],
)
def test_selection_preserves_source_backend_platform_gate(target, platform):
    func = _with_body(_add_fp32, _add_fp32.body)
    result = transform.AscendVectorInstructionSelection(target, platform)(IRModule.from_expr(func))["main"]
    assert_structural_equal(result, func)


def test_selection_refuses_a_preexisting_internal_mask_setter():
    setter = _call("tl.ascend_set_mask_mode", _int(1))
    with pytest.raises(Exception, match="before selection"):
        _select(_with_body(_add_fp32, tir.Evaluate(setter)))
