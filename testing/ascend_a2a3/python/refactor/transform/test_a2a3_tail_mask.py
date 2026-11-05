"""Tail-state propagation keeps every original rewrite and fallback boundary."""

import pytest
from tilelang import tvm


tir = tvm.tirx
_BINARY = ["add", "sub", "mul", "div", "max", "min"]
_UNARY = ["exp", "ln", "abs", "reciprocal", "sqrt", "rsqrt", "relu"]
_SCALAR = ["adds", "muls", "maxs", "mins"]


def _case(kind="exp", dtype="float32", *, full=False, explicit_tmp=False, clear=True, dim=0, dynamic=False):
    spelling = {"float16": "half", "float32": "float", "bfloat16": "bfloat16_t", "int16": "int16_t"}[dtype]
    shape = (8, 32)
    if kind in ("broadcast_row", "broadcast_scalar"):
        shape = (8, 1) if kind == "broadcast_row" else (1, 1)
    elif kind == "broadcast_col":
        shape = (1, 32)

    def buffer(name, shape=(8, 32), dtype=dtype, scope="shared.ub"):
        return tir.decl_buffer(shape, dtype, name=name, scope=scope)

    gm = buffer("gm", shape, scope="global")
    src, other, dst = buffer("src", shape), buffer("other"), buffer("dst")
    mask, arena = buffer("mask", (8, 4), "uint8"), buffer("arena", (4096,), "uint8")
    out = buffer("out", (32,) if dim in (0, -2) else (8,))
    rows = tir.Var("valid_rows", "int32") if dynamic else (shape[0] if full else min(5, shape[0]))
    cols = tir.Var("valid_cols", "int32") if dynamic else (shape[1] if full else min(29, shape[1]))

    def op(name, *args):
        return tir.call_intrin("handle", tvm.ir.Op.get("tl.ascend_" + name), *args)

    def seed(target, valid_rows=rows, valid_cols=cols, physical=shape):
        return tir.call_extern(
            "handle",
            f"tl::ascend::copy_gm_to_ub<{spelling}, {physical[1]}, {physical[0]}>",
            gm.access_ptr("r"),
            target.access_ptr("w"),
            physical[1],
            valid_rows,
            valid_cols,
            0,
            physical[0],
            physical[1],
        )

    statements = [tir.Evaluate(seed(src))]
    pointers = (dst.access_ptr("w"), src.access_ptr("r"), other.access_ptr("r"))
    count = 256
    if kind in _BINARY:
        call = op(kind, *pointers, count)
    elif kind in _UNARY or kind == "bitwise_not":
        call = op(kind, *pointers[:2], count)
    elif kind in _SCALAR:
        call = op(kind, *pointers[:2], tir.const(1, dtype), count)
    elif kind in ("bitwise_and", "bitwise_or"):
        call = op(kind, *pointers, count)
    elif kind in ("copy_then_exp", "cast_then_exp"):
        if kind == "copy_then_exp":
            propagated = tir.call_extern(
                "handle", f"tl::ascend::copy_ub_to_ub<{spelling}, {spelling}, 256>", src.access_ptr("r"), other.access_ptr("w")
            )
        else:
            propagated = op("cast", other.access_ptr("w"), src.access_ptr("r"), "CAST_NONE", count)
        statements.append(tir.Evaluate(propagated))
        call = op("exp", dst.access_ptr("w"), other.access_ptr("r"), count)
    elif kind.startswith("compare") or kind.startswith("select"):
        statements.append(tir.Evaluate(seed(other)))
        scalar = kind in ("compare_scalar", "select_scalar")
        compare = op(
            "compare_scalar" if scalar else "compare",
            mask.access_ptr("w"),
            src.access_ptr("r"),
            tir.const(1, dtype) if scalar else other.access_ptr("r"),
            "LT",
            count,
        )
        if kind.startswith("compare"):
            call = compare
        else:
            statements.append(tir.Evaluate(compare))
            if kind in ("select_invalidated", "select_bufferload_invalidated"):
                invalid = [mask.access_ptr("w"), src.access_ptr("r"), src.access_ptr("r")]
                if kind == "select_bufferload_invalidated":
                    invalid.append(0)
                invalid.extend(["LT", count])
                statements.append(tir.Evaluate(op("compare_scalar", *invalid)))
            args = [dst.access_ptr("w"), mask.access_ptr("r"), src.access_ptr("r")]
            if explicit_tmp:
                args.append(arena.access_ptr("w"))
            args.extend(
                [
                    1 if scalar else 2,
                    tir.const(1, dtype) if scalar else other.access_ptr("r"),
                    "VSEL_TENSOR_SCALAR_MODE" if scalar else "VSEL_TENSOR_TENSOR_MODE",
                    count,
                ]
            )
            call = op("select", *args)
    elif kind == "reduce":
        args = [f"reduce_sum<{spelling}, 8, 32, {dim}>", out.access_ptr("w"), src.access_ptr("r")]
        if explicit_tmp:
            args.append(arena.access_ptr("w"))
        args.append(tir.const(clear, "bool"))
        call = op("reduce", *args)
    elif kind.startswith("broadcast"):
        args = [f"Broadcast<{spelling}, 2, {0 if kind == 'broadcast_col' else 1}, false>", dst.access_ptr("w"), src.access_ptr("r")]
        if explicit_tmp:
            args.append(arena.access_ptr("w"))
        args.extend([2, 8, 32, shape[0], shape[1]])
        call = op("broadcast", *args)
    else:
        raise ValueError(kind)
    statements.append(tir.Evaluate(call))
    block = tir.SBlock([], [], [], "tilelang_root", tir.SeqStmt(statements), alloc_buffers=[src, other, dst, mask, arena, out])
    handle = tir.Var("gm_handle", "handle")
    params = [handle] + ([rows, cols] if dynamic else [])
    return tir.PrimFunc(params, tir.SBlockRealize([], True, block), buffer_map={handle: gm})


def _run(func, enabled=True, rewrite_reduce=True):
    factory = tvm.ffi.get_global_func("tl.transform.A2A3AscendTailMaskPropagation")
    with tvm.transform.PassContext(config={"tl.ascend_tail_mask": enabled}):
        return factory(rewrite_reduce)(tvm.IRModule({"main": func}))["main"]


def _ops(func):
    found = []
    tir.stmt_functor.post_order_visit(func.body, lambda n: found.append(n.op.name) if isinstance(n, tir.Call) else None)
    return found


@pytest.mark.parametrize(("names", "expected"), [(_BINARY, "binary"), (_UNARY, "unary"), (_SCALAR, "scalar")])
@pytest.mark.parametrize("dtype", ["float16", "float32", "bfloat16"])
def test_all_plain_float_operation_tags(names, expected, dtype):
    for name in names:
        assert "tl.ascend_tail_" + expected in _ops(_run(_case(name, dtype)))


@pytest.mark.parametrize("name", ["add", "exp", "adds", "copy_then_exp", "cast_then_exp"])
def test_full_tiles_and_disabled_pass_are_unchanged(name):
    func = _case(name)
    tvm.ir.assert_structural_equal(_run(func, enabled=False), func)
    full = _case(name, full=True)
    tvm.ir.assert_structural_equal(_run(full), full)


@pytest.mark.parametrize("name", ["add", "exp", "adds", "copy_then_exp", "cast_then_exp"])
def test_dynamic_rectangles_propagate_without_static_shape_narrowing(name):
    assert any(op.startswith("tl.ascend_tail_") for op in _ops(_run(_case(name, dynamic=True))))


@pytest.mark.parametrize("name", ["bitwise_and", "bitwise_or", "bitwise_not", "add", "exp"])
def test_unvalidated_integer_contracts_stay_native(name):
    assert not any(op.startswith("tl.ascend_tail_") for op in _ops(_run(_case(name, "int16"))))


@pytest.mark.parametrize("kind", ["compare", "compare_scalar", "select_tensor", "select_scalar"])
@pytest.mark.parametrize("explicit_tmp", [False, True])
def test_packed_compare_and_select_contracts(kind, explicit_tmp):
    ops = _ops(_run(_case(kind, explicit_tmp=explicit_tmp)))
    assert "tl.ascend_tail_compare" in ops or "tl.ascend_tail_compare_scalar" in ops
    if kind.startswith("select"):
        assert "tl.ascend_tail_select" in ops


@pytest.mark.parametrize("kind", ["select_invalidated", "select_bufferload_invalidated"])
def test_unsupported_compare_overwrite_invalidates_packed_provenance(kind):
    assert "tl.ascend_tail_select" not in _ops(_run(_case(kind)))


@pytest.mark.parametrize("kind", ["broadcast_row", "broadcast_col", "broadcast_scalar"])
@pytest.mark.parametrize("explicit_tmp", [False, True])
def test_broadcast_preserves_scalar_fallback_and_axis_expansion(kind, explicit_tmp):
    ops = _ops(_run(_case(kind, explicit_tmp=explicit_tmp)))
    assert ("tl.ascend_tail_broadcast" in ops) == (kind != "broadcast_scalar")


@pytest.mark.parametrize("explicit_tmp", [False, True])
@pytest.mark.parametrize("dim", [0, -2, -1])
@pytest.mark.parametrize("clear", [False, True])
def test_reduce_allow_list_and_rewrite_flag(explicit_tmp, dim, clear):
    func = _case("reduce", explicit_tmp=explicit_tmp, dim=dim, clear=clear)
    assert ("tl.ascend_tail_reduce" in _ops(_run(func))) == (dim in (0, -2) and clear)
    assert "tl.ascend_tail_reduce" not in _ops(_run(func, rewrite_reduce=False))


def test_default_bool_option_stays_disabled():
    func = _case()
    factory = tvm.ffi.get_global_func("tl.transform.A2A3AscendTailMaskPropagation")
    tvm.ir.assert_structural_equal(factory(True)(tvm.IRModule({"main": func}))["main"], func)


def _loop_case(outside=True):
    func = _case("exp", dynamic=True)
    block = func.body.block
    copy, compute = block.body.seq
    index = tir.Var("by", "int32")
    args = list(copy.value.args)
    args[4] = index + 1
    seeded = tir.Evaluate(tir.Call(copy.value.dtype, copy.value.op, args))
    if outside:
        body = tir.SeqStmt([tir.For(index, 0, 8, tir.ForKind.SERIAL, seeded), compute])
    else:
        body = tir.For(index, 0, 8, tir.ForKind.SERIAL, tir.SeqStmt([seeded, compute]))
    block = tir.SBlock([], [], [], "tilelang_root", body, alloc_buffers=block.alloc_buffers)
    return tir.PrimFunc(func.params, tir.SBlockRealize([], True, block), buffer_map=func.buffer_map)


@pytest.mark.parametrize("outside", [False, True])
def test_loop_dependent_rectangles_may_not_escape_scope(outside):
    assert ("tl.ascend_tail_unary" in _ops(_run(_loop_case(outside)))) == (not outside)


@pytest.mark.parametrize("kind", ["compare", "compare_scalar", "select_tensor", "select_scalar"])
def test_bfloat16_packed_contracts_stay_on_the_native_path(kind):
    assert not any(op.startswith("tl.ascend_tail_") for op in _ops(_run(_case(kind, "bfloat16"))))


def test_lower_precision_reduce_stays_on_the_native_path():
    assert "tl.ascend_tail_reduce" not in _ops(_run(_case("reduce", "float16")))


def _hint_case():
    func = _case("broadcast_col")
    block = func.body.block
    dst = next(buffer for buffer in block.alloc_buffers if buffer.name == "dst")
    output = tir.decl_buffer((8, 32), "float32", name="output", scope="global")
    handle = tir.Var("output_handle", "handle")
    sink = tir.Evaluate(
        tir.call_extern("handle", "tl::ascend::copy_ub_to_gm<float, 32, 8>", dst.access_ptr("r"), output.access_ptr("w"), 32, 5, 27, 8, 32)
    )
    body = tir.SeqStmt([*block.body.seq, sink])
    block = tir.SBlock([], [], [], "tilelang_root", body, alloc_buffers=block.alloc_buffers)
    buffers = dict(func.buffer_map.items())
    buffers[handle] = output
    return tir.PrimFunc([*func.params, handle], tir.SBlockRealize([], True, block), buffer_map=buffers)


def test_broadcast_intersects_direct_output_sink_hints():
    result = _run(_hint_case())
    calls = []
    tir.stmt_functor.post_order_visit(
        result.body, lambda node: calls.append(node) if isinstance(node, tir.Call) and node.op.name == "tl.ascend_tail_broadcast" else None
    )
    assert len(calls) == 1
    analyzer = tvm.arith.Analyzer()
    assert [int(analyzer.simplify(value)) for value in calls[0].args[-4:]] == [5, 27, 1, 29]
