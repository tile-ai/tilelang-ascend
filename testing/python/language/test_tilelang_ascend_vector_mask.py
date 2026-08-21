"""Focused regressions for compiler-managed Ascend Vector masks."""

import pytest

import tilelang
import tilelang.language as T
from tilelang.engine.phase import LowerAndLegalize, OptimizeForTarget
from tvm import IRModule, get_global_func, tir
from tvm.ir import Op
from tvm.target import Target


ASCENDC = Target({"kind": "llvm", "model": "ascendc"})
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
    if isinstance(expr, tir.Call) and _name(expr) == "tir.large_uint_imm":
        return int(expr.args[0].value) | (int(expr.args[1].value) << 32)
    raise TypeError(f"Expected an integer literal, got {expr}")


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
    return _call("tir.call_extern", tir.StringImm(name), *args, dtype=dtype)


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
    transform = tilelang.transform.AscendVectorInstructionSelection(ASCENDC, platform)
    return transform(IRModule({"main": func}))["main"]


def _legalize(func: tir.PrimFunc) -> tir.PrimFunc:
    transform = tilelang.transform.AscendVectorMaskLegalize(ASCENDC, "A2")
    return transform(IRModule({"main": func}))["main"]


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
        Op.get("tir.tvm_access_ptr"),
        [
            tir.StringImm(dtype),
            tir.Var(f"{name}_{dtype}", "handle"),
            tir.IntImm("int32", 0),
            tir.IntImm("int32", extent),
            tir.IntImm("int32", access_mask),
        ],
    )


def _add_program(length: int, dtype: str = "float32", add_count: int = 1):
    @T.prim_func
    def main(
        a: T.Tensor((length,), dtype),
        b: T.Tensor((length,), dtype),
        c: T.Tensor((length,), dtype),
    ):
        with T.Kernel(1, threads=1, is_npu=True):
            a_ub = T.alloc_ub((length,), dtype)
            b_ub = T.alloc_ub((length,), dtype)
            c_ub = T.alloc_ub((length,), dtype)
            with T.Scope("V"):
                T.copy(a, a_ub)
                T.copy(b, b_ub)
                T.tile.add(c_ub, a_ub, b_ub)
                if add_count == 2:
                    T.tile.add(c_ub, a_ub, b_ub)
                T.copy(c_ub, c)

    return main


def _uint8_bitwise_program(operation: str):
    @T.prim_func
    def main(
        a: T.Tensor((8,), "uint8"),
        b: T.Tensor((8,), "uint8"),
        c: T.Tensor((8,), "uint8"),
    ):
        with T.Kernel(1, threads=1, is_npu=True):
            a_ub = T.alloc_ub((8,), "uint8")
            b_ub = T.alloc_ub((8,), "uint8")
            c_ub = T.alloc_ub((8,), "uint8")
            with T.Scope("V"):
                T.copy(a, a_ub)
                T.copy(b, b_ub)
                if operation == "and":
                    T.tile.bitwise_and(c_ub, a_ub, b_ub)
                else:
                    T.tile.bitwise_or(c_ub, a_ub, b_ub)
                T.copy(c_ub, c)

    return main


_add_fp32 = _add_program(128)
_add_uint32 = _add_program(64, "uint32")
_two_adds = _add_program(64, add_count=2)
_counter_add = _add_program(96)
_bitwise_and_uint8 = _uint8_bitwise_program("and")
_bitwise_or_uint8 = _uint8_bitwise_program("or")


@T.prim_func
def _mixed_axpy(
    x: T.Tensor((128,), "float16"),
    y: T.Tensor((128,), "float32"),
    out: T.Tensor((128,), "float32"),
):
    with T.Kernel(1, threads=1, is_npu=True):
        x_ub = T.alloc_ub((128,), "float16")
        y_ub = T.alloc_ub((128,), "float32")
        with T.Scope("V"):
            T.copy(x, x_ub)
            T.copy(y, y_ub)
            T.tile.axpy(y_ub, x_ub, 2.0)
            T.copy(y_ub, out)


def _exp_experiment(rows: int, cols: int):
    @T.prim_func
    def main(a: T.Tensor((rows, cols), "float32")):
        with T.Kernel(1, threads=1, is_npu=True):
            a_ub = T.alloc_ub((rows, cols), "float32")
            with T.Scope("V"):
                T.tile.exp_experiment(a_ub[:, 0:64], a_ub[:, 0:64])

    return main


def test_catalog_is_closed_and_all_terminals_are_registered():
    names = list(get_global_func("tl.transform.AscendVectorTerminalCatalog")())
    assert names
    assert len(names) == len(set(names))
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

    with pytest.raises(Exception, match="must not be a negative constant"):
        _selected_add(-1)

    lowered = LowerAndLegalize(IRModule({"main": _add_uint32}), ASCENDC)
    with pytest.raises(Exception, match="Unsupported AscendC Vector dtype uint32.*no fallback"):
        OptimizeForTarget(lowered, ASCENDC, "A2")


@pytest.mark.parametrize(
    ("program", "terminal", "intrinsic"),
    [
        (_bitwise_and_uint8, "tl.ascend_bitwise_and_uint8_legacy_count", "And"),
        (_bitwise_or_uint8, "tl.ascend_bitwise_or_uint8_legacy_count", "Or"),
    ],
)
def test_uint8_bitwise_preserves_legacy_count_form(program, terminal, intrinsic):
    selected = _select(program)
    assert _names(selected.body).count(terminal) == 1

    tilelang.disable_cache()
    source = tilelang.compile(program, target="ascendc", platform="A2", out_idx=[2]).get_kernel_source()
    assert f"AscendC::{intrinsic}(c_ub[0], a_ub[0], b_ub[0], 8);" in source
    assert f"AscendC::{intrinsic}<uint8_t, false>" not in source


def test_effect_only_variants_share_terminals_and_compute_contextual_contracts():
    def selected_copy(src_type, tag, *extra):
        call = _extern(
            f"copy_ub_to_ub<float,{tag}>",
            _access("float32", "copy_dst", access_mask=2),
            _access(src_type, "copy_src", access_mask=1),
            *extra,
        )
        return _selected_call(call, "tl.ascend_copy_ub_to_ub_selected")

    copy_same = selected_copy("float32", "float")
    copy_cast = selected_copy("float16", "half")
    assert len(copy_same.args) == len(copy_cast.args) == 3
    assert _setter_counts(_legalize_calls(copy_same, _selected_add(64))) == (1, 1)
    assert _setter_counts(_legalize_calls(copy_cast, _selected_add(64))) == (0, 0)

    strided_tail = tuple(_int(value) for value in (32, 64, 1, 32, 64))
    for rows, next_length, expected in [
        (_int(0), 96, (1, 1)),
        (_int(1), 64, (1, 1)),
        (tir.Var("runtime_rows", "int32"), 64, (2, 2)),
    ]:
        strided = selected_copy("float16", "half,64", rows, *strided_tail)
        assert _setter_counts(_legalize_calls(_selected_add(96), strided, _selected_add(next_length))) == expected

    common = [
        tir.StringImm("GatherMask<float>"),
        _access("float32", "gather_mask_dst", access_mask=2),
        _access("float32", "gather_mask_src", access_mask=1),
    ]
    fixed = _selected_call(
        _call("tl.ascend_gather_mask", *common, tir.StringImm("P0101")),
        "tl.ascend_gather_mask_selected",
    )
    custom = _selected_call(
        _call("tl.ascend_gather_mask", *common, _access("uint32", "pattern")),
        "tl.ascend_gather_mask_selected",
    )
    assert len(fixed.args) == len(custom.args) == 4
    assert _setter_counts(_legalize_calls(fixed, _selected_add(64))) == (0, 1)
    assert _setter_counts(_legalize_calls(custom, _selected_add(64))) == (2, 2)

    loop = tir.For(tir.Var("i", "int32"), 0, 4, tir.ForKind.SERIAL, tir.Evaluate(fixed))
    around_fixed_loop = _with_body(
        _add_fp32,
        tir.SeqStmt(
            [
                tir.Evaluate(_selected_add(64)),
                loop,
                tir.Evaluate(_selected_add(64)),
            ]
        ),
    )
    assert _setter_counts(_legalize(around_fixed_loop)) == (2, 2)

    gather = _selected_call(
        _call(
            "tl.ascend_gather",
            _access("float32", "gather_dst", access_mask=2),
            _access("float32", "gather_src", access_mask=1),
            _access("uint32", "gather_offset", access_mask=1),
            _int(0),
            _int(32),
        ),
        "tl.ascend_gather_count",
    )
    assert _setter_counts(_legalize_calls(_selected_add(96), gather, _selected_add(32))) == (2, 1)

    fill = _selected_call(
        _call(
            "tl.ascend_fill_experiment",
            tir.StringImm("Fill_experiment<float>"),
            _access("float32", "fill_dst", access_mask=2),
            tir.FloatImm("float32", 1.0),
            _int(0xFFFFFFFF, "uint64"),
            *[_int(value) for value in (1, 1, 8)],
        ),
        "tl.ascend_fill_experiment_explicit_mask",
    )
    assert _setter_counts(_legalize_calls(_selected_add(96), fill, _selected_add(32))) == (2, 1)

    gather_mask_experiment = _selected_call(
        _call(
            "tl.ascend_gather_mask_experiment",
            tir.StringImm("GatherMask_experiment<float>"),
            _access("float32", "gm_exp_dst", access_mask=2),
            _access("float32", "gm_exp_src", access_mask=1),
            _access("uint32", "gm_exp_pattern", access_mask=1),
            _int(0),
            tir.const(UINT64_MASK, "uint64"),
            *[_int(value) for value in (1, 1, 8, 1)],
            _int(0, "uint64"),
        ),
        "tl.ascend_gather_mask_experiment_self_contained",
    )
    fp16_add = _call(
        "tl.ascend_add",
        _access("float16", "fp16_dst", 128, 2),
        _access("float16", "fp16_src0", 128, 1),
        _access("float16", "fp16_src1", 128, 1),
        _int(128),
    )
    fp16_full = _selected_call(fp16_add, "tl.ascend_add_raw")
    cross_dtype = _legalize_calls(fp16_full, gather_mask_experiment, fp16_full)
    assert _setter_counts(cross_dtype) == (1, 2)


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
    with pytest.raises(Exception, match="compile-time constant in.*16383"):
        _selected_call(gather("float32", 16384), "tl.ascend_gather_count")

    def block_reduce(repeat, mask):
        return _call(
            "tl.ascend_block_reduce_max",
            _access("float32", "reduce_dst", access_mask=2),
            _access("float32", "reduce_src", access_mask=1),
            *[_int(value) for value in (repeat, mask, 1, 1, 8)],
        )

    dynamic_repeat = block_reduce(1, 64)
    dynamic_repeat = tir.Call(
        dynamic_repeat.dtype,
        dynamic_repeat.op,
        [*dynamic_repeat.args[:2], tir.Var("repeat", "int32"), *dynamic_repeat.args[3:]],
    )
    for call, message in [
        (block_reduce(1, 65), r"float32.*\[0, 64\]"),
        (block_reduce(256, 64), r"repeat must be in \[0, 255\]"),
        (dynamic_repeat, r"repeat must be a compile-time constant"),
    ]:
        with pytest.raises(Exception, match=message):
            _selected_call(call, "tl.ascend_block_reduce_max_raw_normal")


def test_legalizer_reuses_state_and_repairs_transitions():
    same = _legalize_calls(_selected_add(64), _selected_add(64))
    changed_payload = _legalize_calls(_selected_add(32), _selected_add(63))
    changed_mode = _legalize_calls(_selected_add(64), _selected_add(96), _selected_add(64))
    assert _setter_counts(same) == (1, 1)
    assert _setter_counts(changed_payload) == (1, 2)
    assert _setter_counts(changed_mode) == (3, 3)

    for name in [
        "opaque_user_vector_helper",
        "copy_user_vector_helper",
        "evil::copy_ub_to_gm<float>",
    ]:
        opaque = _extern(name)
        assert _setter_counts(_legalize_calls(_selected_add(64), opaque, _selected_add(64))) == (
            2,
            2,
        )
    known_dma = _extern("tl::ascend::copy_ub_to_gm<float>")
    assert _setter_counts(_legalize_calls(_selected_add(64), known_dma, _selected_add(64))) == (1, 1)

    gm_to_ub = tir.Call(
        "handle",
        Op.get("tir.call_extern"),
        [tir.StringImm("copy_gm_to_ub<float, 96>")],
    )
    padded_copy = _legalize_calls(_selected_add(96), gm_to_ub, _selected_add(96))
    assert _setter_counts(padded_copy) == (2, 2)


def test_control_flow_keeps_only_must_facts():
    cond = tir.Var("cond", "bool")
    full = _selected_add(64)
    partial = _selected_add(32)
    body = tir.SeqStmt(
        [
            tir.IfThenElse(cond, tir.Evaluate(full), tir.Evaluate(partial)),
            tir.Evaluate(full),
        ]
    )
    function = tir.PrimFunc(
        [*list(_add_fp32.params), cond],
        _vector_scope(body),
        _add_fp32.ret_type,
        _add_fp32.buffer_map,
        _add_fp32.attrs,
        _add_fp32.span,
    )
    assert _setter_counts(_legalize(function)) == (2, 3)

    loop = tir.For(
        tir.Var("i", "int32"),
        0,
        4,
        tir.ForKind.SERIAL,
        tir.SeqStmt([tir.Evaluate(full), tir.Evaluate(full)]),
    )
    result = _legalize(_with_body(_add_fp32, tir.SeqStmt([loop, tir.Evaluate(full)])))
    assert _setter_counts(result) == (2, 2)

    effectful_condition = _extern("opaque_mask_condition", dtype="bool")
    nested_call = tir.IfThenElse(effectful_condition, tir.Evaluate(full), None)
    with pytest.raises(Exception, match="top-level Evaluate"):
        _legalize(_with_body(_add_fp32, tir.SeqStmt([tir.Evaluate(full), nested_call])))


@pytest.mark.parametrize(
    ("op_name", "args"),
    [
        (
            "gatherb",
            [
                tir.StringImm("Gatherb<float>"),
                _access("float32", "gatherb_dst", access_mask=2),
                _access("float32", "gatherb_src", access_mask=1),
                _access("uint32", "gatherb_offset", access_mask=1),
                *[tir.IntImm("int32", value) for value in (1, 1, 8)],
            ],
        ),
        (
            "brcb_experiment",
            [
                tir.StringImm("brcb<float>"),
                _access("float32", "brcb_dst", access_mask=2),
                _access("float32", "brcb_src", access_mask=1),
                *[tir.IntImm("int32", value) for value in (1, 1, 8)],
            ],
        ),
    ],
)
def test_reset_mask_helpers_publish_full_payload(op_name, args):
    helper = tir.Call("handle", Op.get(f"tl.ascend_{op_name}"), args)
    selected = _select(_with_body(_add_fp32, tir.Evaluate(helper)))
    selected_helper = next(call for call in _calls(selected.body) if (_name(call) or "").endswith("_payload_full"))
    result = _legalize_calls(_selected_add(96), selected_helper, _selected_add(64))
    assert _setter_counts(result) == (2, 1)


def test_malformed_selected_payload_is_rejected():
    base = _first_call(_add_fp32, "tl.ascend_add")
    malformed = tir.Call("handle", Op.get("tl.ascend_add_raw"), list(base.args))
    with pytest.raises(Exception, match="Malformed|ABI|payload"):
        _legalize(_with_body(_add_fp32, tir.Evaluate(malformed)))


def test_selection_runs_after_phase_two_and_legalizer_runs_last(monkeypatch):
    lowered = LowerAndLegalize(IRModule({"main": _two_adds}), ASCENDC)
    real_selection = tilelang.transform.AscendVectorInstructionSelection
    real_legalize = tilelang.transform.AscendVectorMaskLegalize
    events = []
    observed = {}

    def record(label, factory):
        def make_pass(*args):
            pass_object = factory(*args)

            def apply(module):
                events.append(label)
                observed[label] = _names(module["main"].body)
                return pass_object(module)

            return apply

        return make_pass

    monkeypatch.setattr(
        tilelang.transform,
        "AscendVectorInstructionSelection",
        record("selection", real_selection),
    )
    monkeypatch.setattr(
        tilelang.transform,
        "AscendVectorMaskLegalize",
        record("legalize", real_legalize),
    )
    result = OptimizeForTarget(lowered, ASCENDC, "A2")["main"]

    assert events[-2:] == ["selection", "legalize"]
    assert observed["selection"].count("tl.ascend_add") == 2
    assert "tl.ascend_add_raw" not in observed["selection"]
    assert observed["legalize"].count("tl.ascend_add_raw") == 2
    assert not any(name and name.startswith("tl.ascend_set_mask_") for name in observed["legalize"])
    assert _setter_counts(result) == (1, 1)


def test_resource_scope_is_explicit_nested_and_fail_closed():
    semantic_add = _first_call(_add_fp32, "tl.ascend_add")
    vector_scope = tir.AttrStmt(
        tir.IntImm("int32", 0),
        "resource_scope",
        tir.IntImm("int32", 1),
        tir.Evaluate(semantic_add),
    )
    mixed_scope = _with_body(
        _add_fp32,
        tir.SeqStmt([vector_scope, tir.Evaluate(semantic_add)]),
        scoped=False,
    )

    selected = _select(mixed_scope)
    assert _names(selected.body).count("tl.ascend_add_raw") == 1
    assert _names(selected.body).count("tl.ascend_add") == 1
    with pytest.raises(Exception, match="must be inside T.Scope"):
        tilelang.transform.AscendResourceScopeVerify()(IRModule({"main": selected}))

    nested = _vector_scope(
        tir.SeqStmt(
            [
                tir.Evaluate(semantic_add),
                _vector_scope(tir.Evaluate(semantic_add)),
                tir.Evaluate(semantic_add),
            ]
        )
    )
    legalized = _legalize(_select(_with_body(_add_fp32, nested, scoped=False)))
    assert _names(legalized.body).count("tl.ascend_add_raw") == 3
    assert _setter_counts(legalized) == (1, 1)

    calls = [
        tir.Call("handle", Op.get("tl.ascend_src_code"), [tir.StringImm("int value = 0;")]),
        _extern("opaque_ascend_helper"),
    ]
    for call in calls:
        function = _with_body(_add_fp32, tir.Evaluate(call), scoped=False)
        with pytest.raises(Exception, match="must be inside T.Scope"):
            tilelang.transform.AscendResourceScopeVerify()(IRModule({"main": function}))

    with (
        tilelang.transform.PassContext(config={"tl.ascend_auto_cv_combine": True}),
        pytest.raises(Exception, match="must be inside T.Scope"),
    ):
        tilelang.transform.CombineCV()(IRModule({"main": _with_body(_add_fp32, tir.Evaluate(calls[0]), scoped=False)}))


def test_resource_scope_normalizes_pipe_case_but_rejects_ambiguous_sync():
    @T.prim_func
    def vector_context():
        with T.Kernel(1, threads=1, is_npu=True):
            data = T.alloc_ub((64,), "float32")
            T.tile.fill(data, 0.0)
            T.set_flag("mte2", "v", 0)
            T.tile.fill(data, 1.0)
            T.barrier_all()
            for i in T.serial(1):
                if i == 0:
                    T.wait_flag("mte3", "mte2", 0)
                T.tile.fill(data, 2.0)
            T.tile.fill(data, 3.0)

    @T.prim_func
    def cube_context(a: T.Tensor((16, 16), "float16")):
        with T.Kernel(1, threads=1, is_npu=True):
            data = T.alloc_L1((16, 16), "float16")
            T.copy(a, data)
            T.barrier_all()
            T.copy(a, data)

    @T.prim_func
    def cv_boundary(a: T.Tensor((16, 16), "float16")):
        with T.Kernel(1, threads=1, is_npu=True):
            data_l1 = T.alloc_L1((16, 16), "float16")
            data_ub = T.alloc_ub((16, 16), "float16")
            T.copy(a, data_l1)
            T.barrier_all()
            T.tile.fill(data_ub, 0.0)

    def combine(program):
        module = LowerAndLegalize(IRModule({"main": program}), ASCENDC)
        with tilelang.transform.PassContext(config={"tl.ascend_auto_cv_combine": True}):
            return tilelang.transform.CombineCV()(module)["main"]

    for program in [vector_context, cube_context]:
        combined = combine(program)
        tilelang.transform.AscendResourceScopeVerify()(IRModule({"main": combined}))

    with pytest.raises(Exception, match="must be inside T.Scope"):
        combine(cv_boundary)


def test_codegen_uses_raw_false_overloads_and_reuses_mask_state():
    tilelang.disable_cache()
    normal = tilelang.compile(_two_adds, target="ascendc", platform="A2", out_idx=[2])
    normal_source = normal.get_kernel_source()
    assert normal_source.count("AscendC::SetMaskNorm();") == 1
    assert normal_source.count("AscendC::SetVectorMask<uint8_t>") == 1
    assert normal_source.count("AscendC::Add<float, false>") == 2

    counter = tilelang.compile(_counter_add, target="ascendc", platform="A2", out_idx=[2])
    counter_source = counter.get_kernel_source()
    assert counter_source.count("AscendC::SetMaskCount();") == 1
    assert counter_source.count("AscendC::SetVectorMask<uint8_t>((uint64_t)0, (uint64_t)96);") == 1
    assert counter_source.count("AscendC::Add<float, false>") == 1
    assert "AscendC::Add(" not in counter_source

    conservative = tilelang.compile(
        _two_adds,
        target="ascendc",
        platform="A2",
        out_idx=[2],
        pass_configs={tilelang.PassConfigKey.TL_ASCEND_VECTOR_MASK_REUSE: False},
    )
    conservative_source = conservative.get_kernel_source()
    assert conservative_source.count("AscendC::SetMaskNorm();") == 2
    assert conservative_source.count("AscendC::SetVectorMask<uint8_t>") == 2
    assert conservative_source.count("AscendC::Add<float, false>") == 2

    for program, message in [
        (_exp_experiment(256, 64), "repeat_time=256 must fit uint8_t"),
        (_exp_experiment(1, 2048), "repeat stride=256 must fit uint8_t"),
    ]:
        with pytest.raises(Exception, match=message):
            tilelang.lower(program, target="ascendc", platform="A2")


def test_mixed_axpy_is_selected_but_reverse_mixed_has_no_fallback():
    tilelang.disable_cache()
    kernel = tilelang.compile(_mixed_axpy, target="ascendc", platform="A2", out_idx=[2])
    source = kernel.get_kernel_source()
    assert source.count("AscendC::Axpy<float, half, false>") == 1

    base = _first_call(_mixed_axpy, "tl.ascend_axpy")
    reverse = tir.Call("handle", base.op, [base.args[1], base.args[0], base.args[2], base.args[3]])
    with pytest.raises(Exception, match="Unsupported AscendC Axpy dtype tuple"):
        _select(_with_body(_mixed_axpy, tir.Evaluate(reverse)))


def test_target_scope_keeps_a5_and_pto_unchanged():
    a5 = _select(_add_fp32, platform="A5")
    assert "tl.ascend_add" in _names(a5.body)
    assert "tl.ascend_add_raw" not in _names(a5.body)

    tilelang.disable_cache()
    pto_source = tilelang.lower(_two_adds, target="pto", platform="A2").kernel_source
    assert "tl.ascend_add_raw" not in pto_source
