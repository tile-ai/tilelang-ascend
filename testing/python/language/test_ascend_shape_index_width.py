# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Keep Ascend shape-derived geometry wide without changing its input ABI.

Boundary values are evaluated as scalar TIR on the host. The integration case
runs real lowering and code generation with device compilation disabled; none
of these tests allocate or launch an NPU tensor.
"""

from unittest.mock import patch

import pytest

import tilelang
from tilelang import language as T
from tilelang import tvm
from tilelang.jit.ascend_runtime_contract import CYTHON_INPUT_SIGNATURE_ATTR

tir = tvm.tir
_INT32_MAX = (1 << 31) - 1
_UPSTREAM_CONFIG = {
    "tl.ascend_sync_backend": "upstream",
    "tl.ascend_auto_sync": True,
    "tl.ascend_auto_sync_vs": True,
}


def _function(body, *buffers, scalars=(), platform="A3"):
    params = [buffer.data for buffer in buffers] + list(scalars)
    return tir.PrimFunc(
        params,
        body,
        buffer_map={buffer.data: buffer for buffer in buffers},
    ).with_attr(
        "npu_platform",
        platform,
    )


def _run(function, *, backend="upstream", model="ascendc"):
    target = tvm.target.Target({"kind": "c", "model": model})
    with tvm.transform.PassContext(config={"tl.ascend_sync_backend": backend}):
        return tilelang.transform.AscendLegalizeShapeIndices(target)(
            tvm.IRModule({"main": function}),
        )["main"]


def _nodes(body, node_type):
    nodes = []
    tir.stmt_functor.post_order_visit(
        body,
        lambda node: nodes.append(node) if isinstance(node, node_type) else None,
    )
    return nodes


def _evaluate(expression, values):
    substitutions = {var: tir.IntImm(var.dtype, value) for var, value in values.items()}
    value = tvm.arith.Analyzer().simplify(tir.stmt_functor.substitute(expression, substitutions))
    assert isinstance(value, tir.IntImm), value
    return int(value)


def _probe(*expressions):
    return tir.Evaluate(tir.call_extern("handle", "inspect_geometry", *expressions))


@pytest.mark.parametrize("rows", [1, 64, _INT32_MAX - 62, _INT32_MAX])
def test_ceildiv_addition_uses_64_bits_before_the_addition(rows):
    m = tir.Var("M", "int32")
    source = tir.decl_buffer((m, 16), "float32", name="source")
    iteration = tir.Var("iteration", "int32")
    extent = tir.floordiv(m + 63, 64)
    function = _function(
        tir.For(iteration, 0, extent, tir.ForKind.SERIAL, _probe(iteration)),
        source,
    )
    rewritten = _run(function)
    loop = _nodes(rewritten.body, tir.For)[0]
    assert str(loop.loop_var.dtype) == str(loop.extent.dtype) == "int64"
    assert _evaluate(loop.extent, {m: rows}) == (rows + 63) // 64
    assert rewritten.buffer_map[source.data].shape[0].same_as(m)


def test_two_dimensional_shape_arithmetic_and_derived_lets_do_not_wrap():
    m, n = tir.Var("M", "int32"), tir.Var("N", "int32")
    source = tir.decl_buffer((m, n), "float32", name="source")
    i, j, flat = (tir.Var(name, "int32") for name in ("i", "j", "flat"))
    body = tir.For(
        i,
        0,
        m,
        tir.ForKind.SERIAL,
        tir.For(
            j,
            0,
            n,
            tir.ForKind.SERIAL,
            tir.LetStmt(
                flat,
                i * n + j,
                tir.BufferStore(
                    source,
                    tir.FloatImm("float32", 0),
                    [tir.floordiv(flat, n), tir.floormod(flat, n)],
                ),
            ),
        ),
    )
    rewritten = _run(_function(body, source))
    loops = _nodes(rewritten.body, tir.For)
    assert len(loops) == 2
    assert all(str(loop.loop_var.dtype) == "int64" for loop in loops)
    binding = _nodes(rewritten.body, tir.LetStmt)[0]
    assert str(binding.var.dtype) == str(binding.value.dtype) == "int64"
    loop_vars = {str(loop.loop_var.name): loop.loop_var for loop in loops}
    actual = _evaluate(
        binding.value,
        {m: 65536, n: 65536, loop_vars["i"]: 65535, loop_vars["j"]: 65535},
    )
    assert actual == (1 << 32) - 1


def test_block_axis_bindings_and_body_indices_share_the_wide_loop_domain():
    m, n = tir.Var("M", "int32"), tir.Var("N", "int32")
    output = tir.decl_buffer((m, n), "int32", name="output")
    i, j, vi, vj, flat = (tir.Var(name, "int32") for name in ("i", "j", "vi", "vj", "flat"))
    axes = [
        tir.IterVar(tvm.ir.Range.from_min_extent(0, m), vi, tir.IterVar.DataPar),
        tir.IterVar(tvm.ir.Range.from_min_extent(0, n), vj, tir.IterVar.DataPar),
    ]
    region = tir.BufferRegion(
        output,
        [tvm.ir.Range.from_min_extent(vi, 1), tvm.ir.Range.from_min_extent(vj, 1)],
    )
    block = tir.Block(
        axes,
        [],
        [region],
        "flattened_store",
        tir.LetStmt(
            flat,
            vi * n + vj,
            tir.BufferStore(
                output,
                tir.IntImm("int32", 0),
                [tir.floordiv(flat, n), tir.floormod(flat, n)],
            ),
        ),
    )
    body = tir.For(
        i,
        0,
        m,
        tir.ForKind.SERIAL,
        tir.For(j, 0, n, tir.ForKind.SERIAL, tir.BlockRealize([i, j], True, block)),
    )
    rewritten = _run(_function(body, output))
    realize = _nodes(rewritten.body, tir.BlockRealize)[0]
    assert all(str(value.dtype) == "int64" for value in realize.iter_values)
    assert all(str(axis.var.dtype) == "int64" for axis in realize.block.iter_vars)
    binding = _nodes(realize.block.body, tir.LetStmt)[0]
    assert str(binding.var.dtype) == str(binding.value.dtype) == "int64"
    values = {m: 65536, n: 65536}
    values.update({axis.var: 65535 for axis in realize.block.iter_vars})
    assert _evaluate(binding.value, values) == (1 << 32) - 1
    store = _nodes(realize.block.body, tir.BufferStore)[0]
    assert all(str(index.dtype) == "int64" for index in store.indices)
    assert all(str(range_.min.dtype) == "int64" for range_ in realize.block.writes[0].region)
    assert realize.block.writes[0].buffer.same_as(output)


@pytest.mark.parametrize("geometry", ["elem_offset", "strides", "shape"])
def test_local_buffer_geometry_and_accesses_use_the_new_bound_loop_variable(geometry):
    m = tir.Var("M", "int32")
    source = tir.decl_buffer((m, 16), "float32", name="source")
    index = tir.Var("index", "int32")
    shape = (index + 1,) if geometry == "shape" else (16,)
    strides = (index + 1,) if geometry == "strides" else ()
    offset = index * 16 if geometry == "elem_offset" else 0
    view = tir.decl_buffer(
        shape,
        "float32",
        name="view",
        data=source.data,
        strides=strides,
        elem_offset=offset,
    )
    region = tir.BufferRegion(view, [tvm.ir.Range.from_min_extent(0, 1)])
    update = tir.Block(
        [],
        [region],
        [region],
        "update_view",
        tir.BufferStore(view, tir.BufferLoad(view, [0]), [0]),
    )
    body = tir.For(
        index,
        0,
        m,
        tir.ForKind.SERIAL,
        tir.DeclBuffer(view, tir.BlockRealize([], True, update)),
    )
    rewritten = _run(_function(body, source))
    loop = _nodes(rewritten.body, tir.For)[0]
    declaration = _nodes(loop.body, tir.DeclBuffer)[0]
    new_view = declaration.buffer
    assert not new_view.same_as(view)
    # undefined_vars(body) does not visit Buffer metadata. Check each field
    # explicitly so an unbound original int32 index cannot hide in a view.
    metadata = [*new_view.shape, *new_view.strides, new_view.elem_offset]
    variables = [var for expression in metadata for var in _nodes(expression, tir.Var)]
    assert not any(var.same_as(index) for var in variables)
    assert any(var.same_as(loop.loop_var) for var in variables)
    assert str(loop.loop_var.dtype) == "int64"
    for access in [*_nodes(loop.body, tir.BufferLoad), *_nodes(loop.body, tir.BufferStore)]:
        assert access.buffer.same_as(new_view)
    block = _nodes(loop.body, tir.Block)[0]
    assert all(region.buffer.same_as(new_view) for region in [*block.reads, *block.writes])
    assert rewritten.buffer_map[source.data].same_as(source)
    if geometry == "elem_offset":
        value = (1 << 28) + 1
        assert str(new_view.elem_offset.dtype) == "int64"
        assert _evaluate(new_view.elem_offset, {loop.loop_var: value}) == value * 16


def test_match_buffer_source_and_body_share_the_remapped_local_view():
    m = tir.Var("M", "int32")
    source = tir.decl_buffer((m, 16), "float32", name="source")
    index = tir.Var("index", "int32")
    view = tir.decl_buffer((16,), "float32", name="view", data=source.data, elem_offset=index * 16)
    alias = tir.decl_buffer((1,), "float32", name="alias")
    match = tir.MatchBufferRegion(
        alias,
        tir.BufferRegion(view, [tvm.ir.Range.from_min_extent(0, 1)]),
    )
    alias_region = tir.BufferRegion(alias, [tvm.ir.Range.from_min_extent(0, 1)])
    block = tir.Block(
        [],
        [alias_region],
        [alias_region],
        "matched_update",
        tir.BufferStore(alias, tir.BufferLoad(alias, [0]), [0]),
        match_buffers=[match],
    )
    body = tir.For(
        index,
        0,
        m,
        tir.ForKind.SERIAL,
        tir.DeclBuffer(view, tir.BlockRealize([], True, block)),
    )
    rewritten = _run(_function(body, source))
    loop = _nodes(rewritten.body, tir.For)[0]
    new_view = _nodes(loop.body, tir.DeclBuffer)[0].buffer
    new_block = _nodes(loop.body, tir.Block)[0]
    new_match = new_block.match_buffers[0]
    assert new_match.source.buffer.same_as(new_view)
    assert not new_match.source.buffer.same_as(view)
    assert not any(var.same_as(index) for var in _nodes(new_view.elem_offset, tir.Var))
    assert any(var.same_as(loop.loop_var) for var in _nodes(new_view.elem_offset, tir.Var))
    for access in [
        *_nodes(new_block.body, tir.BufferLoad),
        *_nodes(new_block.body, tir.BufferStore),
    ]:
        assert access.buffer.same_as(new_match.buffer)
    assert all(
        (
            region.buffer.same_as(new_match.buffer)
            for region in [
                *new_block.reads,
                *new_block.writes,
            ]
        ),
    )
    assert rewritten.buffer_map[source.data].same_as(source)


@pytest.mark.parametrize("binding_kind", ["for", "attr"])
def test_thread_binding_domain_matches_the_promoted_thread_variable(binding_kind):
    m = tir.Var("M", "int32")
    source = tir.decl_buffer((m,), "float32", name="source")
    index = tir.Var("thread", "int32")
    iteration = tir.IterVar(
        tvm.ir.Range.from_min_extent(0, m),
        index,
        tir.IterVar.ThreadIndex,
        "blockIdx.x",
    )
    store = tir.BufferStore(source, tir.FloatImm("float32", 0), [index])
    if binding_kind == "for":
        body = tir.For(index, 0, m, tir.ForKind.THREAD_BINDING, store, thread_binding=iteration)
    else:
        body = tir.AttrStmt(iteration, "thread_extent", m, store)
    rewritten = _run(_function(body, source))
    if binding_kind == "for":
        loop = _nodes(rewritten.body, tir.For)[0]
        binding = loop.thread_binding
        assert binding.var.same_as(loop.loop_var)
    else:
        binding = rewritten.body.node
    assert str(binding.var.dtype) == "int64"
    assert str(binding.dom.min.dtype) == str(binding.dom.extent.dtype) == "int64"
    assert _evaluate(binding.dom.extent, {m: _INT32_MAX}) == _INT32_MAX
    access = _nodes(rewritten.body, tir.BufferStore)[0]
    assert access.indices[0].same_as(binding.var)
    assert rewritten.buffer_map[source.data].same_as(source)


def test_nested_loop_and_block_annotations_use_their_current_bound_variables():
    m = tir.Var("M", "int32")
    output = tir.decl_buffer((m,), "int32", name="output")
    index, axis = tir.Var("index", "int32"), tir.Var("axis", "int32")
    sentinel = tir.IntImm("int32", 7)
    axis_view = tir.decl_buffer(
        (1,),
        "int32",
        name="axis_view",
        data=output.data,
        elem_offset=axis * 16,
    )
    axis_metadata = {
        axis: [
            axis * 16,
            [{axis: axis + 3}, sentinel],
            tvm.ir.Range.from_min_extent(axis, 1),
            axis_view,
        ],
        m: sentinel,
    }
    block = tir.Block(
        [tir.IterVar(tvm.ir.Range.from_min_extent(0, m), axis, tir.IterVar.DataPar)],
        [],
        [tir.BufferRegion(output, [tvm.ir.Range.from_min_extent(axis, 1)])],
        "annotated_store",
        tir.BufferStore(output, tir.IntImm("int32", 0), [axis]),
        annotations={"index_metadata": axis_metadata},
    )
    loop_view = tir.decl_buffer(
        (1,),
        "int32",
        name="loop_view",
        data=output.data,
        elem_offset=index * 16,
    )
    loop_metadata = {
        index: [
            index * 16,
            [{index: index + 3}, sentinel],
            tvm.ir.Range.from_min_extent(index, 1),
            loop_view,
        ],
        m: sentinel,
    }
    body = tir.For(
        index,
        0,
        m,
        tir.ForKind.SERIAL,
        tir.BlockRealize([index], True, block),
        annotations={"index_metadata": loop_metadata},
    )
    function = _function(body, output)
    rewritten = _run(function)
    assert rewritten.attrs.same_as(function.attrs)
    loop = _nodes(rewritten.body, tir.For)[0]
    new_block = _nodes(loop.body, tir.Block)[0]
    pairs = [
        (loop.annotations["index_metadata"], index, loop.loop_var),
        (new_block.annotations["index_metadata"], axis, new_block.iter_vars[0].var),
    ]
    for metadata, old_variable, bound_variable in pairs:
        keys = list(metadata.keys())
        assert len(keys) == 2
        bound_key = next(key for key in keys if key.same_as(bound_variable))
        shape_key = next(key for key in keys if key.same_as(m))
        assert str(shape_key.dtype) == "int32" and metadata[shape_key].same_as(sentinel)
        values = metadata[bound_key]
        nested_map = values[1][0]
        nested_keys = list(nested_map.keys())
        assert len(nested_keys) == 1 and nested_keys[0].same_as(bound_variable)
        expressions = [values[0], nested_map[nested_keys[0]], values[2].min, values[3].elem_offset]
        assert not any(
            (
                variable.same_as(old_variable)
                for expression in expressions
                for variable in _nodes(
                    expression,
                    tir.Var,
                )
            ),
        )
        assert _evaluate(values[0], {bound_variable: (1 << 28) + 1}) == ((1 << 28) + 1) * 16
        assert _evaluate(expressions[1], {bound_variable: 13}) == 16
        assert values[1][1].same_as(sentinel)
        assert _evaluate(values[2].min, {bound_variable: 13}) == 13
        assert (
            _evaluate(
                values[3].elem_offset,
                {bound_variable: (1 << 28) + 1},
            )
            == ((1 << 28) + 1) * 16
        )


def test_assumption_attribute_node_uses_the_current_bound_loop_variable():
    m = tir.Var("M", "int32")
    output = tir.decl_buffer((m,), "int32", name="output")
    index = tir.Var("index", "int32")
    predicate = index * 16 < tir.Cast("int64", m) * 16
    sentinel = tir.IntImm("int32", 1)
    assumption = tir.AttrStmt(
        predicate,
        "tl.assume",
        sentinel,
        tir.BufferStore(output, tir.IntImm("int32", 0), [index]),
    )
    function = _function(tir.For(index, 0, m, tir.ForKind.SERIAL, assumption), output)
    rewritten = _run(function)
    loop = _nodes(rewritten.body, tir.For)[0]
    attribute = _nodes(loop.body, tir.AttrStmt)[0]
    variables = _nodes(attribute.node, tir.Var)
    assert not any(variable.same_as(index) for variable in variables)
    assert any(variable.same_as(loop.loop_var) for variable in variables)
    assert any(variable.same_as(m) for variable in variables)
    assert _evaluate(attribute.node, {m: 65536, loop.loop_var: 65535}) == 1
    assert attribute.value.same_as(sentinel)
    assert _nodes(attribute.body, tir.BufferStore)[0].indices[0].same_as(loop.loop_var)
    assert rewritten.attrs.same_as(function.attrs)
    assert rewritten.buffer_map[output.data].same_as(output)


def test_a_hoisted_index_let_uses_the_same_binder_as_its_later_loop():
    rows = tir.Var("rows", "int32")
    output = tir.decl_buffer((rows, 16), "int32", name="output")
    wave, worker, row = (tir.Var(name, "int32") for name in ("wave", "worker", "row"))
    loop = tir.For(
        wave,
        0,
        tir.floordiv(rows + 3, 4),
        tir.ForKind.SERIAL,
        tir.IfThenElse(
            wave * 4 + worker < rows,
            tir.BufferStore(output, tir.IntImm("int32", 0), [row, 0]),
            None,
        ),
    )
    # T.Persistent's actual DSL output hoists this pure index definition before
    # the For while retaining the exact same wave Var object in both places.
    body = tir.LetStmt(row, tir.truncmod(wave * 4 + worker, rows), loop)
    body = tir.AttrStmt(
        tir.IterVar(None, worker, tir.IterVar.ThreadIndex, "blockIdx.x"),
        "thread_extent",
        4,
        body,
    )
    rewritten = _run(_function(body, output))
    new_loop = _nodes(rewritten.body, tir.For)[0]
    binding = _nodes(rewritten.body, tir.LetStmt)[0]
    assert str(new_loop.loop_var.dtype) == "int64"
    assert not any(variable.same_as(wave) for variable in _nodes(binding.value, tir.Var))
    assert any(variable.same_as(new_loop.loop_var) for variable in _nodes(binding.value, tir.Var))
    simplified = tir.transform.Simplify()(tvm.IRModule({"main": rewritten}))["main"]
    assert not any(variable.same_as(wave) for variable in _nodes(simplified.body, tir.Var))
    simplified_loop = _nodes(simplified.body, tir.For)[0]
    store = _nodes(simplified_loop.body, tir.BufferStore)[0]
    assert (
        _evaluate(
            store.indices[0],
            {rows: 5, simplified_loop.loop_var: 1, worker: 0},
        )
        == 4
    )


def test_a_globally_lossless_index_cast_can_expose_a_wide_loop_count():
    rows = tir.Var("rows", "int32")
    output = tir.decl_buffer((rows,), "int32", name="output")
    wave = tir.Var("wave", "int32")
    mathematical_count = tir.floordiv(tir.Cast("int64", rows) + 3, 4)
    count = tir.Cast("int32", mathematical_count)
    body = tir.For(
        wave,
        0,
        count,
        tir.ForKind.SERIAL,
        tir.BufferStore(output, tir.IntImm("int32", 0), [wave]),
    )
    rewritten = _run(_function(body, output))
    loop = _nodes(rewritten.body, tir.For)[0]
    assert str(loop.loop_var.dtype) == str(loop.extent.dtype) == "int64"
    assert _evaluate(loop.extent, {rows: _INT32_MAX}) == (_INT32_MAX + 3) // 4
    tvm.ir.assert_structural_equal(loop.extent, mathematical_count)


def test_a_nested_cast_does_not_hide_an_original_width_overflow():
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")
    from test_ascend_upstream_sync_tail_copy import (
        _buffers,
        _copy,
        _function as tail_function,
        _run as run_sync,
    )

    rows = tir.Var("rows", "int32")
    narrowed = tir.Cast("int32", tir.Cast("int64", rows + 63))
    output = tir.decl_buffer((rows,), "int32", name="output")
    wave = tir.Var("wave", "int32")
    function = _function(
        tir.For(
            wave,
            0,
            narrowed,
            tir.ForKind.SERIAL,
            tir.BufferStore(output, tir.IntImm("int32", 0), [wave]),
        ),
        output,
    )
    rewritten = _run(function)
    extent = _nodes(rewritten.body, tir.For)[0].extent
    assert any(str(addition.dtype) == "int32" for addition in _nodes(extent, tir.Add))
    assert _evaluate(extent, {rows: _INT32_MAX}) == -(1 << 31) + 62
    buffers = _buffers()
    buffers["source"] = tir.decl_buffer((rows, 16), "float32", name="source")
    mask = tir.min(tir.max(narrowed, 0), 1)
    unsafe = tail_function(
        buffers,
        [_copy(buffers, columns=16, rows=mask, stride=16)],
    ).with_attr("npu_platform", "A3")
    with pytest.raises(tvm.error.TVMError, match="nonwrapping signed index arithmetic"):
        run_sync(_run(unsafe))


def test_a_cancelling_expression_cannot_justify_removing_an_unsafe_index_cast():
    rows = tir.Var("rows", "int32")
    output = tir.decl_buffer((rows,), "int32", name="output")
    # A mathematical simplifier could cancel rows and claim that the result is
    # 63. The original rows + 63 is int32 and may wrap before either int64 cast.
    cancelled = tir.Cast(
        "int32",
        tir.Cast("int64", rows + 63) - tir.Cast("int64", rows),
    )
    function = _function(
        tir.BufferStore(output, tir.IntImm("int32", 0), [cancelled]),
        output,
    )
    rewritten = _run(function)
    index = _nodes(rewritten.body, tir.BufferStore)[0].indices[0]
    tvm.ir.assert_structural_equal(index, cancelled)
    assert str(index.dtype) == "int32"
    assert any(str(addition.dtype) == "int32" for addition in _nodes(index, tir.Add))


def test_negative_floor_division_cannot_make_an_out_of_range_cast_look_lossless():
    rows = tir.Var("rows", "int32")
    output = tir.decl_buffer((rows,), "int32", name="output")
    mathematical_index = tir.floordiv(-(tir.Cast("int64", rows) * 4 + 5), 4)
    narrowed = tir.Cast("int32", mathematical_index)
    function = _function(
        tir.BufferStore(output, tir.IntImm("int32", 0), [narrowed]),
        output,
    )
    rewritten = _run(function)
    index = _nodes(rewritten.body, tir.BufferStore)[0].indices[0]
    # Truncating the negative bound toward zero would incorrectly claim that
    # this endpoint fits int32. The original FloorDiv result is INT32_MIN - 1.
    assert _evaluate(mathematical_index, {rows: _INT32_MAX}) == -(1 << 31) - 1
    tvm.ir.assert_structural_equal(index, narrowed)


def test_a_safe_cast_in_an_unrelated_scalar_index_does_not_widen_its_parent():
    rows = tir.Var("rows", "int32")
    scalar = tir.Var("user_scalar", "int32")
    output = tir.decl_buffer((rows,), "int32", name="output")
    converted = tir.Cast("int32", tir.Cast("int64", tir.floordiv(scalar, 4)))
    index = converted + tir.IntImm("int32", _INT32_MAX)
    function = _function(
        tir.BufferStore(output, tir.IntImm("int32", 0), [index]),
        output,
        scalars=(scalar,),
    )
    rewritten = _run(function)
    result = _nodes(rewritten.body, tir.BufferStore)[0].indices[0]
    # The buffer has a dynamic shape, but this index has no dependence on it.
    # A lossless inner conversion must not change this unrelated Add's width.
    tvm.ir.assert_structural_equal(result, index)
    assert str(result.dtype) == "int32"
    assert _evaluate(result, {scalar: 4}) == -(1 << 31)


def test_shape_parameter_buffers_and_signature_attributes_keep_their_identities():
    m, n = tir.Var("M", "int32"), tir.Var("N", "int32")
    scalar = tir.Var("user_scalar", "int32")
    first = tir.decl_buffer((m, n), "float32", name="first")
    second = tir.decl_buffer((m, n), "float32", name="second")
    index = tir.Var("index", "int32")
    function = _function(
        tir.For(
            index,
            0,
            m,
            tir.ForKind.SERIAL,
            tir.BufferStore(first, tir.FloatImm("float32", 0), [index, n - 1]),
        ),
        first,
        second,
        scalars=(scalar,),
    )
    function = function.with_attr(CYTHON_INPUT_SIGNATURE_ATTR, function.params).with_attr(
        "initial_buffer_shapes",
        {first.data: [m, n], second.data: [m, n]},
    )
    rewritten = _run(function)
    assert all(before.same_as(after) for before, after in zip(function.params, rewritten.params))
    assert str(rewritten.params[-1].dtype) == "int32"
    for parameter, buffer in function.buffer_map.items():
        assert rewritten.buffer_map[parameter].same_as(buffer)
    assert rewritten.attrs.same_as(function.attrs)
    for old, new in zip(
        function.attrs[CYTHON_INPUT_SIGNATURE_ATTR],
        rewritten.attrs[CYTHON_INPUT_SIGNATURE_ATTR],
    ):
        assert old.same_as(new)
    body_vars = _nodes(rewritten.body, tir.Var)
    assert any(var.same_as(m) for var in body_vars)
    assert any(var.same_as(n) for var in body_vars)
    assert not any(str(var.name) in ("M", "N") and str(var.dtype) == "int64" for var in body_vars)


def test_explicit_narrowing_cast_is_a_boundary_not_removed_by_shape_promotion():
    m = tir.Var("M", "int32")
    source = tir.decl_buffer((m, 16), "float32", name="source")
    narrowed = tir.Cast("int32", tir.Cast("int64", m) + tir.IntImm("int64", 1 << 31))
    function = _function(tir.BufferStore(source, tir.FloatImm("float32", 0), [narrowed, 0]), source)
    rewritten = _run(function)
    casts = _nodes(rewritten.body, tir.Cast)
    assert any(str(node.dtype) == "int32" and str(node.value.dtype) == "int64" for node in casts)
    conversion = next(
        (node for node in casts if str(node.dtype) == "int32" and str(node.value.dtype) == "int64"),
    )
    tvm.ir.assert_structural_equal(conversion, narrowed)
    assert _evaluate(conversion, {m: -(1 << 31)}) == 0


def test_lossy_shape_cast_still_fails_the_tail_copy_safety_audit():
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")
    from test_ascend_upstream_sync_tail_copy import (
        _buffers,
        _copy,
        _function as tail_function,
        _run as run_sync,
    )

    m = tir.Var("M", "int32")
    buffers = _buffers()
    buffers["source"] = tir.decl_buffer((m, 16), "float32", name="source")
    narrowed = tir.Cast("int32", tir.Cast("int64", m) + tir.IntImm("int64", 1 << 31))
    mask = tir.min(tir.max(narrowed, 0), 1)
    function = tail_function(buffers, [_copy(buffers, columns=16, rows=mask, stride=16)]).with_attr(
        "npu_platform",
        "A3",
    )
    rewritten = _run(function)
    with pytest.raises(tvm.error.TVMError, match="lossy index casts are unsupported"):
        run_sync(rewritten)


def test_int32_buffer_values_and_unsigned_scalar_arithmetic_keep_their_widths():
    m = tir.Var("M", "int32")
    source = tir.decl_buffer((m,), "int32", name="source")
    output = tir.decl_buffer((m,), "int32", name="output")
    unsigned = tir.Var("unsigned_data", "uint32")
    index = tir.Var("index", "int32")
    value = tir.BufferLoad(source, [index]) + tir.IntImm("int32", 1)
    body = tir.For(
        index,
        0,
        m,
        tir.ForKind.SERIAL,
        tir.SeqStmt([tir.BufferStore(output, value, [index]), _probe(unsigned + 1)]),
    )
    rewritten = _run(_function(body, source, output, scalars=(unsigned,)))
    store = _nodes(rewritten.body, tir.BufferStore)[0]
    assert str(store.indices[0].dtype) == "int64"
    assert str(store.value.dtype) == "int32"
    assert all(str(load.dtype) == "int32" for load in _nodes(rewritten.body, tir.BufferLoad))
    unsigned_adds = [
        node
        for node in _nodes(
            rewritten.body,
            tir.Add,
        )
        if str(node.dtype) == "uint32"
    ]
    assert len(unsigned_adds) == 1
    assert rewritten.params[-1].same_as(unsigned)


def test_data_only_shape_arithmetic_retains_its_original_integer_semantics():
    m = tir.Var("M", "int32")
    output = tir.decl_buffer((m, 16), "int32", name="output")
    function = _function(tir.BufferStore(output, m + 63, [0, 0]), output)
    rewritten = _run(function)
    tvm.ir.assert_structural_equal(rewritten.body, function.body)
    store = _nodes(rewritten.body, tir.BufferStore)[0]
    assert str(store.value.dtype) == "int32"
    assert _evaluate(store.value, {m: _INT32_MAX}) == -(1 << 31) + 62


def test_promoted_loop_index_is_cast_back_at_integer_data_uses():
    m = tir.Var("M", "int32")
    output = tir.decl_buffer((m,), "int32", name="output")
    index = tir.Var("index", "int32")
    function = _function(
        tir.For(
            index,
            0,
            m,
            tir.ForKind.SERIAL,
            tir.BufferStore(output, index + 1, [index]),
        ),
        output,
    )
    rewritten = _run(function)
    loop = _nodes(rewritten.body, tir.For)[0]
    store = _nodes(rewritten.body, tir.BufferStore)[0]
    assert str(loop.loop_var.dtype) == "int64"
    assert str(store.indices[0].dtype) == "int64"
    assert str(store.value.dtype) == "int32"
    assert any(str(node.dtype) == "int32" for node in _nodes(store.value, tir.Cast))
    assert _evaluate(store.value, {loop.loop_var: _INT32_MAX - 1}) == _INT32_MAX


def test_explicit_widening_cast_preserves_its_original_width_child_arithmetic():
    m = tir.Var("M", "int32")
    source = tir.decl_buffer((m, 16), "float32", name="source")
    index = tir.Var("index", "int64")
    explicit = tir.Cast("int64", m + 63)
    function = _function(
        tir.For(index, tir.IntImm("int64", 0), explicit, tir.ForKind.SERIAL, _probe(index)),
        source,
    )
    rewritten = _run(function)
    loop = _nodes(rewritten.body, tir.For)[0]
    assert any(str(node.dtype) == "int32" for node in _nodes(loop.extent, tir.Add))
    assert _evaluate(loop.extent, {m: _INT32_MAX}) == -(1 << 31) + 62


@pytest.mark.parametrize("platform", ["A2", "A3"])
def test_supported_ascend_platforms_promote_shape_derived_loop_geometry(platform):
    m = tir.Var("M", "int32")
    source = tir.decl_buffer((m,), "float32", name="source")
    index = tir.Var("index", "int32")
    function = _function(
        tir.For(index, 0, m, tir.ForKind.SERIAL, _probe(index)),
        source,
        platform=platform,
    )
    assert str(_nodes(_run(function).body, tir.For)[0].loop_var.dtype) == "int64"


@pytest.mark.parametrize("backend,platform", [("legacy", "A3"), ("upstream", "A5")])
def test_unrelated_sync_profiles_keep_the_original_function(backend, platform):
    m = tir.Var("M", "int32")
    source = tir.decl_buffer((m,), "float32", name="source")
    index = tir.Var("index", "int32")
    function = _function(
        tir.For(index, 0, m, tir.ForKind.SERIAL, _probe(index)),
        source,
        platform=platform,
    )
    assert _run(function, backend=backend).same_as(function)


def test_static_shapes_do_not_change_unrelated_int32_loops_or_values():
    source = tir.decl_buffer((64,), "int32", name="source")
    index = tir.Var("index", "int32")
    function = _function(
        tir.For(index, 0, 64, tir.ForKind.SERIAL, tir.BufferStore(source, index, [index])),
        source,
    )
    assert _run(function).same_as(function)


def test_equal_shape_variable_names_do_not_merge_distinct_input_dimensions():
    first_dim, second_dim = tir.Var("dimension", "int32"), tir.Var("dimension", "int32")
    source = tir.decl_buffer((first_dim, second_dim), "float32", name="source")
    index = tir.Var("index", "int32")
    function = _function(
        tir.For(index, 0, first_dim + second_dim, tir.ForKind.SERIAL, _probe(index)),
        source,
    )
    loop = _nodes(_run(function).body, tir.For)[0]
    assert _evaluate(loop.extent, {first_dim: _INT32_MAX, second_dim: 17}) == _INT32_MAX + 17


@pytest.mark.parametrize("model", ["pto", "cuda"])
def test_other_device_models_keep_the_original_function(model):
    m = tir.Var("M", "int32")
    source = tir.decl_buffer((m,), "float32", name="source")
    index = tir.Var("index", "int32")
    function = _function(tir.For(index, 0, m, tir.ForKind.SERIAL, _probe(index)), source)
    assert _run(function, model=model).same_as(function)


def test_explicit_scalar_without_a_shape_buffer_does_not_gain_a_different_abi():
    scalar = tir.Var("user_scalar", "int32")
    index = tir.Var("index", "int32")
    function = _function(
        tir.For(index, 0, scalar, tir.ForKind.SERIAL, _probe(index)),
        scalars=(scalar,),
    )
    assert _run(function).same_as(function)


def test_unsigned_shape_geometry_keeps_its_original_wrapping_semantics():
    size = tir.Var("size", "uint32")
    source = tir.decl_buffer((size,), "int32", name="source")
    function = _function(
        tir.BufferStore(source, tir.IntImm("int32", 0), [size + tir.IntImm("uint32", 1)]),
        source,
    )
    assert _run(function).same_as(function)


def test_narrow_datatype_keeps_ceildiv_intermediates_wide_when_the_loop_count_fits():
    m = tir.Var("M", "int32")
    source = tir.decl_buffer((m, 16), "float32", name="source")
    index = tir.Var("index", "int32")
    function = _function(
        tir.For(
            index,
            0,
            tir.floordiv(m + 63, 64),
            tir.ForKind.SERIAL,
            tir.BufferStore(source, tir.FloatImm("float32", 0), [index * 64, 0]),
        ),
        source,
    )
    wide = _run(function)
    narrowed = tir.transform.NarrowDataType(32)(tvm.IRModule({"main": wide}))["main"]
    restored = _run(narrowed)
    loop = _nodes(restored.body, tir.For)[0]
    # The count is at most 2**25 and the last row start fits int32. Narrowing
    # those values is sound; the M + 63 intermediate must remain wide.
    assert str(loop.loop_var.dtype) == str(loop.extent.dtype)
    additions = _nodes(loop.extent, tir.Add)
    assert additions and all(str(addition.dtype) == "int64" for addition in additions)
    expected_count = (_INT32_MAX + 63) // 64
    assert _evaluate(loop.extent, {m: _INT32_MAX}) == expected_count
    row = _nodes(restored.body, tir.BufferStore)[0].indices[0]
    assert _evaluate(row, {loop.loop_var: expected_count - 1}) == (expected_count - 1) * 64


def test_narrow_datatype_keeps_a_large_two_dimensional_flat_offset_wide():
    m = tir.Var("M", "int32")
    source = tir.decl_buffer((m, 65536), "float32", name="source")
    index, flat = tir.Var("index", "int32"), tir.Var("flat", "int32")
    function = _function(
        tir.For(
            index,
            0,
            m,
            tir.ForKind.SERIAL,
            tir.LetStmt(
                flat,
                index * 65536 + 65535,
                tir.BufferStore(
                    source,
                    tir.FloatImm("float32", 0),
                    [tir.floordiv(flat, 65536), tir.floormod(flat, 65536)],
                ),
            ),
        ),
        source,
    )
    wide = _run(function)
    narrowed = tir.transform.NarrowDataType(32)(tvm.IRModule({"main": wide}))["main"]
    restored = _run(narrowed)
    loop = _nodes(restored.body, tir.For)[0]
    binding = _nodes(restored.body, tir.LetStmt)[0]
    assert str(binding.var.dtype) == str(binding.value.dtype) == "int64"
    assert _evaluate(binding.value, {m: 65536, loop.loop_var: 65535}) == (1 << 32) - 1
    products = _nodes(binding.value, tir.Mul)
    assert products and all(str(product.dtype) == "int64" for product in products)


def _dynamic_copy_program():
    # Match the original SwiGLU dynamic row tiling without executing sample code.
    m = T.symbolic("M")

    @T.prim_func
    def main(source: T.Tensor((m, 16), "float32"), output: T.Tensor((m, 16), "float32")):
        with T.Kernel(48, is_npu=True) as (cid, vid):
            local = T.alloc_ub((32, 16), "float32")
            for step in T.serial(T.ceildiv(T.ceildiv(m, 64), 48)):
                tile = cid + step * 48
                if tile < T.ceildiv(m, 64):
                    row = tile * 64 + vid * 32
                    T.copy(source[row, 0], local)
                    T.copy(local, output[row, 0])

    return main


def _dynamic_two_dimensional_copy_program(dtype):
    # Use the same flattened row/column grid as the original SwiGLU samples.
    m, n = T.symbolic("M"), T.symbolic("N")
    row_tiles = T.ceildiv(m, 64)
    column_tiles = T.ceildiv(n, 64)
    tiles = row_tiles * column_tiles

    @T.prim_func
    def main(source: T.Tensor((m, n), dtype), output: T.Tensor((m, n), dtype)):
        with T.Kernel(48, is_npu=True) as (cid, vid):
            local = T.alloc_ub((32, 64), dtype)
            for step in T.serial(T.ceildiv(tiles, 48)):
                tile = cid + step * 48
                if tile < tiles:
                    row = tile // column_tiles * 64 + vid * 32
                    column = tile % column_tiles * 64
                    T.copy(source[row, column], local)
                    T.copy(local, output[row, column])

    return main


@pytest.mark.parametrize("dtype,accepted", [("bfloat16", True), ("float32", False)])
def test_real_two_dimensional_copy_audits_the_complete_dynamic_strided_gm_read(dtype, accepted):
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")
    with (
        tvm.transform.PassContext(config=_UPSTREAM_CONFIG),
        patch("tilelang.utils.target.check_npu_availability", return_value=True),
    ):
        program = _dynamic_two_dimensional_copy_program(dtype)
        if not accepted:
            # N is any nonnegative source int32 dimension. Four-byte rows can
            # exceed the actual uint32 byte-stride ABI; wider arithmetic or a
            # larger proof budget must not grant this physical safety fact.
            with pytest.raises(tvm.error.TVMError, match="nonwrapping DMA stride"):
                tilelang.lower(
                    program,
                    target="ascendc",
                    platform="A3",
                    enable_host_codegen=False,
                    enable_device_compile=False,
                )
            return
        artifact = tilelang.lower(
            program,
            target="ascendc",
            platform="A3",
            enable_host_codegen=False,
            enable_device_compile=False,
        )
    source = artifact.kernel_source
    assert "copy_gm_to_ub" in source
    assert "int64_t M" in source and "int64_t N" in source
    assert "for (int64_t" in source


@pytest.mark.parametrize("nested", [False, True])
def test_real_dynamic_persistent_lowering_preserves_bound_geometry(nested):
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")
    from test_ascend_upstream_sync_persistent import (
        _PASS_CONFIGS as persistent_configs,
        _persistent_abs,
    )

    captured = {}

    @tvm.instrument.pass_instrument
    class Capture:
        def run_before_pass(self, module, info):
            if info.name == "tl.AscendInsertSync":
                functions = [f for f in module.functions.values() if isinstance(f, tir.PrimFunc)]
                assert len(functions) == 1
                captured["before_sync"] = functions[0]

    program = _persistent_abs(None, nested=nested).with_attr("tir.noalias", True)
    with (
        tvm.transform.PassContext(config=persistent_configs, instruments=[Capture()]),
        patch("tilelang.utils.target.check_npu_availability", return_value=True),
    ):
        artifact = tilelang.lower(
            program,
            target="ascendc",
            platform="A3",
            enable_host_codegen=False,
            enable_device_compile=False,
        )
    function = captured["before_sync"]
    definitions = list(function.params)
    for buffer in function.buffer_map.values():
        definitions.append(buffer.data)
        for dimension in buffer.shape:
            definitions.extend(_nodes(dimension, tir.Var))
    assert not tir.analysis.undefined_vars(function.body, definitions)
    assert "copy_gm_to_ub" in artifact.kernel_source
    assert "int64_t rows" in artifact.kernel_source


def test_real_dynamic_copy_lowering_keeps_wide_geometry_after_narrow_datatype():
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")
    captured = {}

    @tvm.instrument.pass_instrument
    class Capture:
        def run_before_pass(self, module, info):
            if info.name == "tl.AscendInsertSync":
                functions = [f for f in module.functions.values() if isinstance(f, tir.PrimFunc)]
                assert len(functions) == 1
                captured["before_sync"] = functions[0]

    with (
        tvm.transform.PassContext(config=_UPSTREAM_CONFIG, instruments=[Capture()]),
        patch("tilelang.utils.target.check_npu_availability", return_value=True),
    ):
        artifact = tilelang.lower(
            _dynamic_copy_program(),
            target="ascendc",
            platform="A3",
            enable_host_codegen=False,
            enable_device_compile=False,
        )
    function = captured["before_sync"]
    loops = _nodes(function.body, tir.For)
    assert loops and all(str(loop.loop_var.dtype) == "int64" for loop in loops)
    dynamic_loop = next(loop for loop in loops if _nodes(loop.extent, tir.Var))
    m = next(var for var in _nodes(dynamic_loop.extent, tir.Var) if str(var.name) == "M")
    expected = ((_INT32_MAX + 63) // 64 + 47) // 48
    assert _evaluate(dynamic_loop.extent, {m: _INT32_MAX}) == expected
    assert "int64_t M" in artifact.kernel_source
    assert "for (int64_t" in artifact.kernel_source
