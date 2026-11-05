"""Pinned TA PTO convenience API contracts; no SDK or device is required."""

from types import SimpleNamespace

import pytest

from tilelang import tvm
from tilelang.ascend.a2a3.language import pto
from tilelang.language.frame import _get_let_stack

T = tvm.tirx
pytestmark = pytest.mark.skipif(
    tvm.get_global_func("tl.A2A3KernelLaunch", allow_missing=True) is None,
    reason="A2/A3 native dialect requires USE_ASCEND=ON",
)


def _buffer(name="A", shape=(4, 8), scope="shared.ub", dtype="float16"):
    return T.decl_buffer(shape, dtype, name=name, scope=scope)


def _ptr(call, buffer, mask, offset=0):
    assert call.op.name == "tirx.tvm_access_ptr"
    assert call.args[1].same_as(buffer.data)
    assert int(tvm.arith.Analyzer().simplify(call.args[2])) == offset
    assert int(call.args[4]) == mask


@pytest.mark.parametrize("name", ["add", "sub", "mul", "div", "max", "min", "bitwise_and", "bitwise_or"])
@pytest.mark.parametrize("variant", ["buffer", "region", "scalar", "load"])
def test_binary_contract(name, variant):
    dst, src, other = _buffer("C"), _buffer("A"), _buffer("B")
    right = other
    offset, size = 0, 32
    if variant == "region":
        region = [tvm.ir.Range.from_min_extent(1, 2), tvm.ir.Range.from_min_extent(2, 4)]
        dst_arg, src_arg, right = [T.BufferRegion(buffer, region) for buffer in (dst, src, other)]
        offset, size = 10, 8
    else:
        dst_arg, src_arg = dst, src
    if variant == "scalar":
        right = 2
    elif variant == "load":
        right = T.BufferLoad(_buffer("scalar", (4,)), [1])
    if name in ("bitwise_and", "bitwise_or") and variant in ("scalar", "load"):
        # The source emits an unregistered scalar spelling; retain its failure.
        with pytest.raises(AttributeError, match="tl.ascend_" + name + "s"):
            getattr(pto, name)(dst_arg, src_arg, right)
        return
    call = getattr(pto, name)(dst_arg, src_arg, right)
    assert call.op.name == "tl.ascend_" + name + ("s" if variant in ("scalar", "load") else "")
    _ptr(call.args[0], dst, 2, offset)
    _ptr(call.args[1], src, 1, offset)
    assert int(call.args[-1]) == size
    if variant in ("buffer", "region"):
        _ptr(call.args[2], other, 1, offset)
    elif variant == "scalar":
        assert int(call.args[2]) == 2
    else:
        _ptr(call.args[2], right.buffer, 1)
        assert int(call.args[3]) == 1


@pytest.mark.parametrize(
    "name,opcode",
    [
        ("exp", "exp"),
        ("abs", "abs"),
        ("reciprocal", "reciprocal"),
        ("sqrt", "sqrt"),
        ("rsqrt", "rsqrt"),
        ("relu", "relu"),
        ("not_tl", "bitwise_not"),
    ],
)
def test_unary_contract(name, opcode):
    dst, src = _buffer("C"), _buffer("A")
    call = getattr(pto, name)(dst, src)
    assert call.op.name == "tl.ascend_" + opcode
    _ptr(call.args[0], dst, 2)
    _ptr(call.args[1], src, 1)
    assert int(call.args[2]) == 32
    assert tvm.ir.structural_equal(call, pto.unary_op(dst, src, opcode))


def test_source_log_spelling_still_fails():
    with pytest.raises(AttributeError, match="tl.ascend_log"):
        pto.ln(_buffer("C"), _buffer("A"))


@pytest.mark.parametrize("name", ["leaky_relu", "axpy"])
def test_scalar_contract(name):
    dst, src = _buffer("C"), _buffer("A")
    call = getattr(pto, name)(dst, src, 2)
    assert call.op.name == "tl.ascend_" + name
    _ptr(call.args[0], dst, 2)
    _ptr(call.args[1], src, 1)
    assert [int(arg) for arg in call.args[2:]] == [2, 32]
    assert tvm.ir.structural_equal(call, pto.scalar_op(dst, src, 2, name))


def test_fill_and_direct_binary_helper():
    dst, src, other = _buffer("C"), _buffer("A"), _buffer("B")
    call = pto.fill(dst, T.const(1, "float16"))
    assert call.op.name == "tl.ascend_fill"
    assert call.args[0].value == "tl::ascend::Fill<half>"
    _ptr(call.args[1], dst, 2)
    assert float(call.args[2]) == 1
    assert int(call.args[3]) == 32
    assert tvm.ir.structural_equal(pto.binary_op(dst, src, other, "add"), pto.add(dst, src, other))


@pytest.mark.parametrize("name", ["reduce_sum", "reduce_max"])
def test_reduce_contract(name):
    out, src, tmp = _buffer("C"), _buffer("A"), _buffer("tmp")
    call = getattr(pto, name)(out, src, tmp, 1)
    assert call.op.name == "tl.ascend_reduce"
    assert call.args[0].value == name + "<half, 4, 8, 1>"
    for ptr, buffer, mask in zip(call.args[1:], (out, src, tmp), (2, 1, 1)):
        _ptr(ptr, buffer, mask)
    assert tvm.ir.structural_equal(call, pto.reduce(out, src, tmp, name, 1))


@pytest.mark.parametrize("ta,tb", [(False, False), (True, False), (False, True), (True, True)])
def test_gemm_transpose_and_init(ta, tb):
    a = _buffer("A", (32, 16) if ta else (16, 32), "shared.l1")
    b = _buffer("B", (16, 32) if tb else (32, 16), "shared.l1")
    c = _buffer("C", (16, 16), "wmma.accumulator", "float32")
    call = pto.gemm_v0(a, b, c, ta, tb, True, 64)
    assert call.args[0].value == f"gemm_v0<half, float, 16, 16, 32, {str(ta).lower()}, {str(tb).lower()}, 64>"
    for ptr, buffer, mask in zip(call.args[1:4], (a, b, c), (1, 1, 2)):
        _ptr(ptr, buffer, mask)
    assert int(call.args[4]) == 1
    with pytest.raises(AssertionError, match="multiple of 16"):
        pto.gemm_v0(a, b, c, ta, tb, kL0Size=15)


def test_private_helpers_and_active_binding():
    a = _buffer()
    region = T.BufferRegion(a, [tvm.ir.Range.from_min_extent(1, 2), tvm.ir.Range.from_min_extent(2, 4)])
    assert pto._dtype(a) == pto._dtype(region) == "half"
    assert [int(x) for x in pto._retrieve_shape(a)] == [4, 8]
    assert [int(x) for x in pto._retrieve_shape(region)] == [2, 4]
    _ptr(pto._retrieve_ptr(a), a, 1)
    _ptr(pto._retrieve_ptr(region, "w"), a, 2, 10)
    assert pto._legalize_arguments(a).same_as(a)
    bound = T.Var("bound", "handle")
    assert pto._legalize_arguments(bound).same_as(bound)
    stack = _get_let_stack()
    stack.push(SimpleNamespace(var=bound, value=region))
    try:
        assert pto._legalize_arguments(bound).same_as(a)
    finally:
        stack.pop()
    for helper in (pto._retrieve_shape, pto._retrieve_ptr):
        with pytest.raises(ValueError, match="Unsupported argument type"):
            helper(1)


def test_shared_sync_schema_and_preserved_cross_flag_arity():
    assert pto.barrier_all().args[0].value == "PIPE_ALL"
    assert pto.pipe_barrier("v").args[0].value == "PIPE_V"
    for helper, opcode in ((pto.set_flag, "tl.ascend_set_flag"), (pto.wait_flag, "tl.ascend_wait_flag")):
        call = helper("mte2", "v", 2)
        assert call.op.name == opcode
        assert len(call.args) == 2
        assert call.args[0].value == "MTE2_V"
        assert int(call.args[1]) == 2
    call = pto.set_cross_flag("mte3", 3)
    assert call.args[0].value == "MTE3"
    assert [int(x) for x in call.args[1:]] == [3, 2]
    # Preserve the original producer even though the original native reader
    # rejected it; callers can use the primary dialect's two-argument helper.
    assert len(pto.wait_cross_flag(3).args) == 1
