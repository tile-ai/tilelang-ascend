"""Native A2/A3 Op contracts and TA copy/atomic lowering without an NPU."""

import json
from pathlib import Path

import pytest
import tilelang
from tilelang import tvm


def _buffer(name, scope, shape=(16, 32), dtype="float16"):
    return tvm.tirx.decl_buffer(shape, dtype, name=name, scope=scope)


def _region(buffer, mask, mins=None, extents=None):
    mins = [0] * len(buffer.shape) if mins is None else mins
    extents = list(buffer.shape) if extents is None else extents
    return tvm.tirx.Call("handle", tvm.ir.Op.get("tl.region"), [tvm.tirx.BufferLoad(buffer, mins), mask, *extents])


def _lower(name, args, buffers, parameters=(), layout_map=None):
    tirx = tvm.tirx
    body = tirx.Evaluate(tirx.Call("handle", tvm.ir.Op.get("tl." + name), args))
    if layout_map is not None:
        block = tirx.SBlock([], [], [], "copy", body, annotations={"layout_map": layout_map})
        body = tirx.SBlockRealize([], True, block)
    func = tirx.PrimFunc([buf.data for buf in buffers] + list(parameters), body)
    func = func.with_attr("target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))
    lowered = tilelang.transform.LowerTileOp()(tvm.IRModule.from_expr(func))["main"]
    external = []
    tirx.stmt_functor.post_order_visit(
        lowered.body,
        lambda node: external.append(node) if isinstance(node, tirx.Call) and node.op.same_as(tvm.ir.Op.get("tirx.call_extern")) else None,
    )
    assert len(external) == 1
    return external[0]


def test_a2a3_complete_op_schema_and_shared_sync_abi():
    schema = json.loads(Path(__file__).with_name("a2a3_op_schema.json").read_text())
    registered = list(tvm.ir.Op.list_op_names())
    for name, arity in schema["num_inputs"].items():
        full_name = "tl." + name
        assert registered.count(full_name) == 1, full_name
        op = tvm.ir.Op.get(full_name)
        assert op.num_inputs == arity, full_name
        assert int(op.get_attr("TCallEffectKind")) == tvm.tirx.CallEffectKind.Opaque.value, full_name
    # The original three-argument TA flag registration must not overwrite A5.
    assert tvm.ir.Op.get("tl.ascend_set_flag").num_inputs == 2
    assert tvm.ir.Op.get("tl.ascend_wait_flag").num_inputs == 2
    assert tvm.ir.Op.get("tl.ascend_pipe_barrier").num_inputs == 1


@pytest.mark.parametrize(
    "src_scope,dst_scope,helper",
    [
        ("global", "shared.l1", "copy_gm_to_l1<"),
        ("shared.l1", "wmma.matrix_a", "copy_l1_to_l0a<"),
        ("shared.l1", "wmma.matrix_b", "copy_l1_to_l0b<"),
        ("wmma.accumulator", "global", "copy_l0c_to_gm<"),
        ("global", "shared.ub", "copy_gm_to_ub<"),
        ("shared.ub", "global", "copy_ub_to_gm<"),
        ("shared.ub", "shared.l1", "copy_ub_to_l1<"),
        ("wmma.accumulator", "shared.ub", "copy_l0c_to_ub<"),
        ("shared.ub", "shared.ub", "copy_ub_to_ub<"),
    ],
)
def test_copy_preserves_all_ta_transfer_branches(src_scope, dst_scope, helper):
    src, dst = _buffer("src", src_scope), _buffer("dst", dst_scope)
    result = _lower("ascend_copy", [_region(src, 1), _region(dst, 2)], [src, dst])
    assert result.args[0].value.startswith("tl::ascend::" + helper)
    if src_scope == "wmma.accumulator":
        assert ", 0>" in result.args[0].value  # omitted enable_relu is False


@pytest.mark.parametrize("count", [0, 1, 10])
def test_copy_rejects_arguments_outside_real_ta_abi(count):
    builder = tvm.ir.Op.get("tl.ascend_copy").get_attr("TLOpBuilder")
    with pytest.raises(tvm.error.InternalError, match="expects source|at most 9"):
        builder([tvm.tirx.IntImm("int32", 0)] * count, {})


def test_copy_keeps_runtime_cube_arguments_and_fixpipe_flag():
    src, dst = _buffer("src", "shared.l1"), _buffer("dst", "wmma.matrix_b")
    real_k, real_n = tvm.tirx.Var("real_k", "int32"), tvm.tirx.Var("real_n", "int32")
    args = [_region(src, 1), _region(dst, 2), False, True, 0, 0, 3, real_k, real_n]
    result = _lower("ascend_copy", args, [src, dst], [real_k, real_n])
    assert result.args[0].value.endswith(", true>")
    tvm.ir.assert_structural_equal(result.args[3], real_k)
    tvm.ir.assert_structural_equal(result.args[4], real_n)
    src, dst = _buffer("src", "wmma.accumulator"), _buffer("dst", "global")
    result = _lower("ascend_copy", [_region(src, 1), _region(dst, 2), True, False, 0, 0, 3], [src, dst])
    assert ", 1>" in result.args[0].value
    assert int(result.args[-1]) == 3


@pytest.mark.parametrize("name", ["ascend_copy", "ascend_atomic_add"])
def test_runtime_inner_extent_stays_a_function_argument(name):
    src, dst = _buffer("src", "shared.ub"), _buffer("dst", "global")
    width = tvm.tirx.Var("width", "int32")
    src_region, dst_region = _region(src, 1, extents=[16, width]), _region(dst, 2, extents=[16, width])
    args = [src_region, dst_region] if name == "ascend_copy" else [dst_region, src_region]
    result = _lower(name, args, [src, dst], [width])
    assert "<half, 32, 16>" in result.args[0].value
    assert "width" not in result.args[0].value
    for value in (16, 32, 48):
        valid = tvm.tirx.stmt_functor.substitute(result.args[5], {width: tvm.tirx.IntImm("int32", value)})
        assert int(tvm.arith.Analyzer().simplify(valid)) == min(value, 32)


@pytest.mark.parametrize("name", ["ascend_copy", "ascend_atomic_add"])
def test_row_slice_stride_is_last_buffer_dimension(name):
    src, dst = _buffer("src", "shared.ub", (1, 32)), _buffer("dst", "global", (128, 32))
    row = tvm.tirx.Var("row", "int32")
    sr, dr = _region(src, 1), _region(dst, 2, mins=[row, 0], extents=[1, 32])
    args = [sr, dr] if name == "ascend_copy" else [dr, sr]
    result = _lower(name, args, [src, dst], [row])
    assert int(result.args[3]) == 32


@pytest.mark.parametrize("source_scope", ["shared.ub", "wmma.accumulator"])
def test_atomic_add_preserves_ub_and_l0c_paths(source_scope):
    src, dst = _buffer("src", source_scope), _buffer("dst", "global")
    result = _lower("ascend_atomic_add", [_region(dst, 2), _region(src, 1)], [src, dst])
    helper = "atomic_add_ub_to_gm" if source_scope == "shared.ub" else "atomic_add_l0c_to_gm"
    assert result.args[0].value.startswith("tl::ascend::" + helper + "<")


@pytest.mark.parametrize("dst_mask,src_mask", [(1, 1), (2, 2)])
def test_atomic_add_preserves_region_access_validation(dst_mask, src_mask):
    src, dst = _buffer("src", "shared.ub"), _buffer("dst", "global")
    builder = tvm.ir.Op.get("tl.ascend_atomic_add").get_attr("TLOpBuilder")
    with pytest.raises(tvm.error.InternalError, match="writable|readable"):
        builder([_region(dst, dst_mask), _region(src, src_mask)], {})


def test_atomic_add_builder_keeps_ramp_region_minima():
    src, dst = _buffer("src", "shared.ub"), _buffer("dst", "global")
    ramp = tvm.tirx.Ramp(0, 1, 16)
    builder = tvm.ir.Op.get("tl.ascend_atomic_add").get_attr("TLOpBuilder")
    node = builder([_region(dst, 2, mins=[ramp, 0], extents=[1, 32]), _region(src, 1)], {"origin": tvm.tirx.IntImm("int32", 7)})
    tvm.ir.assert_structural_equal(node.dst_range[0].min, ramp)
    assert int(node.dst_range[0].extent) == 1
    assert int(node.annotations["origin"]) == 7


@pytest.mark.parametrize("tag,spelling", [(0, "RowMajor"), (1, "ColumnMajor"), (2, "zN"), (3, "zZ"), (4, "nZ")])
@pytest.mark.parametrize("name", ["ascend_copy", "ascend_atomic_add"])
def test_l0c_lowering_retains_explicit_ta_layout_tags(name, tag, spelling):
    src, dst = _buffer("src", "wmma.accumulator"), _buffer("dst", "global")
    identity = tilelang.Layout([16, 32], lambda i, j: (i, j))
    tagged = tvm.ffi.get_global_func("tl.ascend.a2a3.WrapLayout")(identity, tag)
    sr, dr = _region(src, 1), _region(dst, 2)
    args = [sr, dr] if name == "ascend_copy" else [dr, sr]
    result = _lower(name, args, [src, dst], layout_map={src: tagged})
    assert "layout::" + spelling in result.args[0].value
