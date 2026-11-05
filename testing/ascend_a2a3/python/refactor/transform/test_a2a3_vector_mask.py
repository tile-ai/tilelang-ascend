"""TA compiler-managed vector regressions, using raw pass input contracts."""

import pytest
from tilelang import tvm
from tilelang.ascend.a2a3 import transform
from tvm import IRModule, arith
from tvm import tirx as tir
from tvm.ir import Op, assert_structural_equal
from tvm.target import Target

from _a2a3_vector_mask_cases import (
    ASCENDC,
    UINT64_MASK,
    _access,
    _add_fp32,
    _bilinear_with_mask,
    _binding_of,
    _call,
    _calls,
    _evaluate_mask_word,
    _extern,
    _first_call,
    _gather_with_count,
    _int,
    _legalize,
    _legalize_calls,
    _let,
    _name,
    _names,
    _select,
    _selected_add,
    _selected_add_for,
    _selected_call,
    _setter_counts,
    _template,
    _vector_scope,
    _with_body,
)


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
    for mask in [tir.const(UINT64_MASK, "uint64"), tir.const(0x1FFFFFFFF, "uint64")]:
        args = list(gather_mask_experiment.args)
        args[5] = mask
        narrowed = tir.Call(gather_mask_experiment.dtype, gather_mask_experiment.op, args)
        assert _setter_counts(_legalize_calls(narrowed, _selected_add(32))) == (0, 0)
        assert _setter_counts(_legalize_calls(narrowed, _selected_add(64))) == (0, 1)
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


def test_bilinear_dynamic_mask_is_tracked_exactly():
    mask = tir.Var("mask", "int32")
    bilinear = _selected_call(_bilinear_with_mask(mask), "tl.ascend_bilinear_interpolation_self_contained")
    reduce = _selected_call(
        _call(
            "tl.ascend_block_reduce_max",
            _access("float16", "dst", access_mask=2),
            _access("float16", "src", access_mask=1),
            _int(1),
            mask,
            _int(1),
            _int(1),
            _int(8),
        ),
        "tl.ascend_block_reduce_max_raw_normal",
    )
    # The helper already established both words required by the next operation.
    assert _setter_counts(_legalize_calls(bilinear, reduce)) == (0, 0)


@pytest.mark.parametrize("dtype,lanes", [("float16", 128), ("float32", 64)])
def test_gather_preserves_zero_count_and_tracks_tail(dtype, lanes):
    full_half = _selected_add_for(_template("float16"), 128)
    zero = _selected_call(_gather_with_count(_int(0), dtype), "tl.ascend_gather_count")
    # Even a float32 Gather must preserve the nonzero high word when count=0.
    assert _setter_counts(_legalize_calls(full_half, zero, full_half)) == (1, 1)
    for count, tail in [(lanes, lanes), (lanes + 1, 1)]:
        gather = _selected_call(_gather_with_count(_int(count), dtype), "tl.ascend_gather_count")
        following = _selected_add_for(_template(dtype), tail)
        assert _setter_counts(_legalize_calls(gather, following)) == (1, 0)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("operation", ["block_reduce_max", "wholereducesum"])
def test_dynamic_normal_mask_and_repeat_preserve_exact_state(dtype, operation):
    length = tir.Var("length", "int32")
    repeat = tir.Var("repeat", "int32")
    mask_index = 3 if operation.startswith("block") else 2
    repeat_index = 2 if mask_index == 3 else 3

    def selected(mask):
        shape = [repeat, mask] if mask_index == 3 else [mask, repeat]
        call = _call(
            f"tl.ascend_{operation}",
            _access(dtype, "reduce_dst", access_mask=2),
            _access(dtype, "reduce_src", access_mask=1),
            *shape,
            *[_int(value) for value in (1, 1, 8)],
        )
        return _selected_call(call, f"tl.ascend_{operation}_raw_normal")

    first = selected(length)
    assert first.args[repeat_index].same_as(repeat)
    # Equivalent pure expressions retain one exact setup across both calls.
    result = _legalize_calls(first, selected(length + 0))
    assert _setter_counts(result) == (1, 1)
    analyzer = arith.Analyzer()
    max_lanes = 128 if dtype == "float16" else 64
    for lanes in [0, 1, 63, 64, 65, 127, 128]:
        if lanes > max_lanes:
            continue
        words = [analyzer.simplify(tir.stmt_functor.substitute(word, {length: _int(lanes)})) for word in first.args[-2:]]
        assert [_evaluate_mask_word(word) for word in words] == [
            (1 << min(lanes, 64)) - 1,
            (1 << max(lanes - 64, 0)) - 1,
        ]

    # The TA test also calls the complete lowering pipeline and source printer.
    # That seam is covered when the public A2/A3 pipeline is assembled; this
    # component checks the exact selected mask words and repairs above.


def test_buffer_loaded_normal_mask_is_sampled_once():
    counts = tir.decl_buffer((1,), "int32", name="counts")
    load = tir.BufferLoad(counts, [0])
    call = _call(
        "tl.ascend_block_reduce_max",
        _access("float16", "reduce_dst", access_mask=2),
        _access("float16", "reduce_src", access_mask=1),
        _int(1),
        load,
        *[_int(value) for value in (1, 1, 8)],
    )
    selected = _select(_with_body(_add_fp32, tir.Evaluate(call)))
    binding, tail = _binding_of(selected.body.body)
    assert isinstance(binding, tir.Bind) and binding.value.same_as(load)
    terminal = tail[0].value
    assert terminal.args[3].same_as(binding.var)
    for word in terminal.args[-2:]:
        variables = []
        tir.stmt_functor.post_order_visit(
            word, lambda node, variables=variables: variables.append(node) if isinstance(node, tir.Var) else None
        )
        assert any(variable.same_as(binding.var) for variable in variables)
    result = _legalize(selected)
    loads = []
    tir.stmt_functor.post_order_visit(result.body, lambda node: loads.append(node) if isinstance(node, tir.BufferLoad) else None)
    assert len(loads) == 1 and loads[0].same_as(load)
    assert _setter_counts(result) == (1, 1)


def test_createvecindex_requires_normal_and_discards_vector_post_state():
    for count, expected in [(_int(8), (1, 1)), (_int(64), (3, 2)), (tir.Var("count", "int32"), (3, 2))]:
        helper = _selected_call(
            _call(
                "tl.ascend_createvecindex",
                tir.StringImm("CreateVecIndex<float>"),
                _access("float32", "index_dst", access_mask=2),
                tir.FloatImm("float32", 0.0),
                count,
            ),
            "tl.ascend_createvecindex_unknown",
        )
        following = _selected_add(96 if isinstance(count, tir.IntImm) and count.value == 8 else 64)
        assert _setter_counts(_legalize_calls(_selected_add(96), helper, following)) == expected


def test_dcci_stays_unselected_and_mask_neutral():
    call = _call(
        "tl.ascend_datacachecleanandinvalid_experiment",
        tir.StringImm("AscendC::DataCacheCleanAndInvalid"),
        _access("float32", "cache"),
    )
    selected = _select(_with_body(_add_fp32, tir.Evaluate(call)))
    assert _names(selected.body) == _names(call)
    assert _setter_counts(_legalize_calls(_selected_add(64), call, _selected_add(64))) == (1, 1)


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
        Op.get("tirx.call_extern"),
        [tir.StringImm("copy_gm_to_ub<float, 96>")],
    )
    padded_copy = _legalize_calls(_selected_add(96), gm_to_ub, _selected_add(96))
    assert _setter_counts(padded_copy) == (2, 2)


@pytest.mark.parametrize("source", ["buffer", "buffer_expr", "loop_expr", "let_expr"])
def test_mask_payload_reuse_requires_stable_values(source):
    index = tir.Var("i", "int32")
    derived = tir.Var("length", "int32")
    counts = tir.decl_buffer((1,), "int32", name="counts")
    load = tir.BufferLoad(counts, [0])
    expression = 96 + 32 * index
    length = {
        "buffer": load,
        "buffer_expr": expression + load,
        "loop_expr": expression,
        "let_expr": derived,
    }[source]
    add = _selected_add(length)
    body = tir.SeqStmt([tir.Evaluate(add), tir.BufferStore(counts, load + 1, [0]), tir.Evaluate(add)])
    if source == "let_expr":
        body = _let(derived, expression, body)
    loop = tir.For(index, 0, 2, tir.ForKind.SERIAL, body)
    result = _legalize(_with_body(_add_fp32, loop))
    # A memory write must not reuse an earlier sampled count. Pure loop/Let
    # expressions still share one setter within each iteration; mode survives.
    assert _setter_counts(result) == (1, 2 if source.startswith("buffer") else 1)
    surrounding = tir.Evaluate(_selected_add(96))
    result = _legalize(_with_body(_add_fp32, tir.SeqStmt([surrounding, loop, surrounding])))
    assert _setter_counts(result) == (3, 4 if source.startswith("buffer") else 3)


def test_helper_ensures_does_not_cache_memory_reads():
    counts = tir.decl_buffer((1,), "int32", name="counts")
    count = tir.BufferLoad(counts, [0])
    fill = _selected_call(
        _call(
            "tl.ascend_fill_experiment",
            tir.StringImm("Fill_experiment<float>"),
            _access("float32", "fill_dst", access_mask=2),
            tir.FloatImm("float32", 1.0),
            count.astype("uint64"),
            *[_int(value) for value in (1, 1, 8)],
        ),
        "tl.ascend_fill_experiment_explicit_mask",
    )
    body = tir.SeqStmt([tir.Evaluate(fill), tir.BufferStore(counts, count + 1, [0]), tir.Evaluate(_selected_add(count))])
    # Fill establishes NORMAL itself; Add needs COUNTER and a freshly read
    # payload even though its expression matches the helper's earlier ensures.
    assert _setter_counts(_legalize(_with_body(_add_fp32, body))) == (2, 1)


@pytest.mark.parametrize("kind", ["src_code", "extern"])
def test_opaque_symbol_changes_invalidate_mask_facts(kind):
    value = tir.Var("value", "int32")
    add = tir.Evaluate(_selected_add(96 + 32 * value))
    opaque = {
        "src_code": tir.Evaluate(_call("tl.ascend_src_code", tir.StringImm("value += 1;"))),
        "extern": tir.Evaluate(_extern("opaque_symbol_update")),
    }[kind]
    # A loop containing only opaque code must not take the mask-neutral fast
    # path: that code can change either an input symbol or the mask itself.
    for middle in [
        opaque,
        tir.For(tir.Var("i", "int32"), 0, 2, tir.ForKind.SERIAL, opaque),
        tir.While(tir.Var("condition", "bool"), opaque),
    ]:
        body = _with_body(_add_fp32, tir.SeqStmt([add, middle, add]))
        assert _setter_counts(_legalize(body)) == (2, 2)


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

    value = tir.Var("length", "int32")
    bound_add = tir.Evaluate(_selected_add(value))
    binding = _let(value, 96, tir.SeqStmt([bound_add, bound_add]))
    result = _legalize(_with_body(_add_fp32, tir.SeqStmt([binding, tir.Evaluate(_selected_add(96))])))
    assert _setter_counts(result) == (1, 2)

    effectful_condition = _extern("opaque_mask_condition", dtype="bool")
    effectful_bound = _extern("opaque_mask_bound", dtype="int32")
    for statement in [
        tir.IfThenElse(effectful_condition, tir.Evaluate(full), None),
        tir.While(effectful_condition, tir.Evaluate(0)),
        tir.For(tir.Var("i", "int32"), effectful_bound, 1, tir.ForKind.SERIAL, tir.Evaluate(0)),
        tir.For(tir.Var("i", "int32"), 0, effectful_bound, tir.ForKind.SERIAL, tir.Evaluate(0)),
    ]:
        with pytest.raises(Exception, match="top-level Evaluate"):
            _legalize(_with_body(_add_fp32, tir.SeqStmt([tir.Evaluate(full), statement])))


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


@pytest.mark.parametrize("reuse,expected", [(True, (1, 1)), (False, (2, 2))])
def test_mask_reuse_config_preserves_source_default_and_disabled_behavior(reuse, expected):
    call = _selected_add(96)
    func = _with_body(_add_fp32, tir.SeqStmt([tir.Evaluate(call), tir.Evaluate(call)]))
    with tvm.transform.PassContext(config={"tl.ascend_vector_mask_reuse": reuse}):
        result = _legalize(func)
    assert _setter_counts(result) == expected


@pytest.mark.parametrize(
    "target,platform",
    [
        (Target("llvm"), "A2"),
        (Target({"kind": "ascend", "arch": "dav-3510"}), "A3"),
        (Target({"kind": "ascend", "arch": "dav-2201", "keys": ["pto", "ascend"]}), "A2"),
        (ASCENDC, "A5"),
    ],
)
def test_mask_legalization_preserves_source_backend_platform_gate(target, platform):
    func = _with_body(_add_fp32, tir.Evaluate(_selected_add(96)))
    result = transform.AscendVectorMaskLegalize(target, platform)(IRModule.from_expr(func))["main"]
    assert_structural_equal(result, func)


def test_mask_legalizer_refuses_unselected_semantic_and_preexisting_setters():
    for body in [_add_fp32.body, tir.Evaluate(_call("tl.ascend_set_mask_mode", _int(1)))]:
        with pytest.raises(Exception, match="Unselected|before legalization"):
            _legalize(_with_body(_add_fp32, body))


@pytest.mark.parametrize(
    "operation,make_call,index,maximum",
    [
        ("gather", _gather_with_count, 4, 16383),
        ("bilinear_interpolation", _bilinear_with_mask, 4, 128),
    ],
)
def test_helper_dynamic_parameters_and_constant_bounds(operation, make_call, index, maximum):
    value = tir.Var("value", "uint64")
    for expr in [_int(0), _int(maximum), value]:
        selected = _select(_with_body(_add_fp32, tir.Evaluate(make_call(expr))))
        terminal = next(c for c in _calls(selected.body) if operation in (_name(c) or ""))
        assert_structural_equal(terminal.args[index], expr)
    for invalid in [_int(-1), _int(maximum + 1), tir.const(UINT64_MASK, "uint64")]:
        with pytest.raises(Exception, match="must be in"):
            _select(_with_body(_add_fp32, tir.Evaluate(make_call(invalid))))

    values = tir.decl_buffer((1,), "int32", name="values")
    load = tir.BufferLoad(values, [0])
    selected = _select(_with_body(_add_fp32, tir.Evaluate(make_call(load))))
    binding, tail = _binding_of(selected.body.body)
    assert isinstance(binding, tir.Bind)
    assert_structural_equal(binding.value, load)
    assert tail[0].value.args[index].same_as(binding.var)


def test_malformed_selected_payload_is_rejected():
    base = _first_call(_add_fp32, "tl.ascend_add")
    malformed = tir.Call("handle", Op.get("tl.ascend_add_raw"), list(base.args))
    with pytest.raises(Exception, match="Malformed|ABI|payload"):
        _legalize(_with_body(_add_fp32, tir.Evaluate(malformed)))
