"""TA compiler-managed vector contracts (source 0deb8b93), without a pipeline."""

from tilelang import tvm
from tilelang.ascend.a2a3 import transform
from tvm import DataType, IRModule
from tvm import tirx as tir
from tvm.ir import Op
from tvm.target import Target

get_global_func = tvm.ffi.get_global_func
ASCENDC = Target({"kind": "ascend", "arch": "dav-2201", "keys": ["ascendc", "ascend"]})
UINT64_MASK = (1 << 64) - 1


def _calls(node) -> list[tir.Call]:
    result = []

    def visit(current):
        if isinstance(current, tir.Call):
            result.append(current)

    tir.stmt_functor.post_order_visit(node, visit)
    return result


def _name(call: tir.Call) -> str | None:
    return call.op.name if isinstance(call.op, Op) else None


def _integer_value(expr: tir.PrimExpr) -> int:
    if isinstance(expr, tir.IntImm):
        return int(expr.value)
    if isinstance(expr, tir.Call) and _name(expr) == "tirx.large_uint_imm":
        return int(expr.args[0].value) | (int(expr.args[1].value) << 32)
    raise TypeError(f"Expected an integer literal, got {expr}")


def _evaluate_mask_word(expr: tir.PrimExpr) -> int:
    """Evaluate the constant integer AST left by TVM's partial simplifier."""
    if isinstance(expr, tir.Cast):
        dtype = DataType(expr.dtype)
        assert dtype.type_code in [0, 1] and dtype.lanes == 1
        value = _evaluate_mask_word(expr.value) & ((1 << dtype.bits) - 1)
        if dtype.type_code == 0 and value >= 1 << (dtype.bits - 1):
            value -= 1 << dtype.bits
        return value
    if isinstance(expr, tir.Sub):
        return _evaluate_mask_word(expr.a) - _evaluate_mask_word(expr.b)
    if isinstance(expr, tir.Call) and _name(expr) in ["tirx.bitwise_and", "tirx.shift_right"]:
        lhs, rhs = [_evaluate_mask_word(arg) for arg in expr.args]
        if _name(expr) == "tirx.bitwise_and":
            return lhs & rhs
        assert 0 <= rhs < DataType(expr.args[0].dtype).bits
        return lhs >> rhs
    return _integer_value(expr)


def _names(node) -> list[str | None]:
    return [_name(call) for call in _calls(node)]


def _first_call(func: tir.PrimFunc, name: str) -> tir.Call:
    return next(call for call in _calls(func.body) if _name(call) == name)


def _selected_call(call: tir.Call, name: str) -> tir.Call:
    selected = _select(_with_body(_add_fp32, tir.Evaluate(call)))
    return _first_call(selected, name)


def _call(name: str, *args: tir.PrimExpr, dtype: str = "handle") -> tir.Call:
    return tir.Call(dtype, Op.get(name), list(args))


def _extern(name: str, *args: tir.PrimExpr, dtype: str = "handle") -> tir.Call:
    return _call("tirx.call_extern", tir.StringImm(name), *args, dtype=dtype)


def _int(value: int, dtype: str = "int32") -> tir.IntImm:
    return tir.IntImm(dtype, value)


def _vector_scope(body: tir.Stmt) -> tir.Stmt:
    return tir.AttrStmt(tir.IntImm("int32", 0), "resource_scope", 1, body)


def _with_body(template: tir.PrimFunc, body: tir.Stmt, *, scoped: bool = True) -> tir.PrimFunc:
    if scoped:
        body = _vector_scope(body)
    return tir.PrimFunc(
        template.params,
        body,
        template.ret_type,
        template.buffer_map,
        template.attrs,
        template.span,
    )


def _select(func: tir.PrimFunc, platform: str = "A2") -> tir.PrimFunc:
    native_pass = transform.AscendVectorInstructionSelection(ASCENDC, platform)
    return native_pass(IRModule({"main": func}))["main"]


def _legalize(func: tir.PrimFunc) -> tir.PrimFunc:
    native_pass = transform.AscendVectorMaskLegalize(ASCENDC, "A2")
    return native_pass(IRModule({"main": func}))["main"]


def _selected_add_for(template: tir.PrimFunc, length) -> tir.Call:
    base = _first_call(template, "tl.ascend_add")
    args = list(base.args)
    args[-1] = length if isinstance(length, tir.PrimExpr) else tir.IntImm("int32", length)
    selected = _select(_with_body(template, tir.Evaluate(tir.Call(base.dtype, base.op, args))))
    return _first_call(selected, "tl.ascend_add_raw")


def _selected_add(length) -> tir.Call:
    return _selected_add_for(_add_fp32, length)


def _legalize_calls(*calls: tir.Call) -> tir.PrimFunc:
    statements = [tir.Evaluate(call) for call in calls]
    body = statements[0] if len(statements) == 1 else tir.SeqStmt(statements)
    return _legalize(_with_body(_add_fp32, body))


def _setter_counts(func: tir.PrimFunc) -> tuple[int, int]:
    names = _names(func.body)
    return names.count("tl.ascend_set_mask_mode"), names.count("tl.ascend_set_mask_payload")


def _access(dtype: str, name: str, extent: int = 64, access_mask: int = 3) -> tir.Call:
    return tir.Call(
        "handle",
        Op.get("tirx.tvm_access_ptr"),
        [
            tir.StringImm(dtype),
            tir.Var(f"{name}_{dtype}", "handle"),
            tir.IntImm("int32", 0),
            tir.IntImm("int32", extent),
            tir.IntImm("int32", access_mask),
        ],
    )


def _gather_with_count(count, dtype="float32"):
    return _call(
        "tl.ascend_gather",
        _access(dtype, "gather_dst", access_mask=2),
        _access(dtype, "gather_src", access_mask=1),
        _access("uint32", "gather_offsets", access_mask=1),
        _int(0),
        count,
    )


def _bilinear_with_mask(mask):
    return _call(
        "tl.ascend_bilinear_interpolation",
        _access("float16", "bilinear_dst", 256, 2),
        _access("float16", "bilinear_src", 512, 1),
        _access("uint32", "bilinear_offsets", 32, 1),
        _access("float16", "bilinear_weights", 16, 1),
        mask,
        _int(2),
        _int(0, "bool"),
        _int(1),
        _int(128),
        _int(2),
        _access("float16", "bilinear_tmp", 512),
    )


def _template(dtype="float32"):
    args = [_access(dtype, "dst", 128, 2), _access(dtype, "src0", 128, 1), _access(dtype, "src1", 128, 1), _int(128)]
    return tir.PrimFunc([], tir.Evaluate(_call("tl.ascend_add", *args)))


_add_fp32 = _template()
_add_uint32 = _template("uint32")


def _let(var, value, body):
    return tir.SeqStmt([tir.Bind(var, value), body])


def _binding_of(body):
    assert isinstance(body, tir.SeqStmt) and isinstance(body.seq[0], tir.Bind)
    return body.seq[0], body.seq[1:]
