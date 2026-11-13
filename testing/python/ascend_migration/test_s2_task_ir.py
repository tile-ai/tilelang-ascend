"""Read-only native schedule analysis; no device compilation or launch."""

import os
from pathlib import Path

import pytest

import tilelang
from tilelang import tvm
from tilelang import language as T

tir = tvm.tir


def _buffer(name, scope="shared.ub", dtype="float32", shape=(128,), **kwargs):
    return tir.decl_buffer(shape, dtype, name=name, scope=scope, **kwargs)


def _pointer(buffer, mask, offset=0):
    return buffer.access_ptr(mask, offset=offset)


def _fill(buffer, offset=0, count=64):
    return tir.Evaluate(
        tir.Call(
            "handle",
            tvm.ir.Op.get("tl.ascend_fill"),
            [
                tir.StringImm("Fill<float>"),
                _pointer(buffer, "w", offset),
                tir.const(1.0, buffer.dtype),
                tir.const(count, "int32") if isinstance(count, int) else count,
            ],
        )
    )


def _add(dst, a, b, count=64):
    return tir.Evaluate(
        tir.Call(
            "handle",
            tvm.ir.Op.get("tl.ascend_add"),
            [
                dst.access_ptr("w"),
                a.access_ptr("r"),
                b.access_ptr("r"),
                tir.const(count, "int32") if isinstance(count, int) else count,
            ],
        )
    )


def _dma(source, destination, *, count=64, valid=None, offset=0):
    load = source.scope() == "global"
    name = "copy_gm_to_ub" if load else "copy_ub_to_gm"
    typename = "half" if source.dtype == "float16" else "float"
    args = [
        source.access_ptr("r", offset=offset),
        destination.access_ptr("w"),
        tir.const(count, "int32"),
        tir.const(1, "int32"),
        tir.const(count if valid is None else valid, "int32") if valid is None or isinstance(valid, int) else valid,
    ]
    if load:
        args.append(tir.const(0.0, source.dtype))
    args.append(tir.const(count, "int32"))
    return tir.Evaluate(tir.call_extern("handle", f"{name}<{typename},{count}>", *args))


def _function(buffers, statements, *, scalars=(), noalias=False, scope=1, addresses=None):
    body = statements if isinstance(statements, tir.Stmt) else statements[0] if len(statements) == 1 else tir.SeqStmt(statements)
    if scope is not None:
        body = tir.AttrStmt(tir.const(0, "int32"), "resource_scope", scope, body)
    local = [b for b in buffers if b.scope() != "global"]
    external = [b for b in buffers if b.scope() == "global"]
    # Multiple views of one data Var are accesses to one allocation.
    local = list({b.data: b for b in local}.values())
    root = tir.Block([], [], [], "tilelang_root", body, alloc_buffers=local)
    function = tir.PrimFunc(
        [b.data for b in external] + list(scalars),
        tir.BlockRealize([], True, root),
        buffer_map={b.data: b for b in external},
    )
    if noalias:
        function = function.with_attr("tir.noalias", True)
    if addresses:
        function = function.with_attr("address_map", addresses)
    return function


def _analyze(function):
    return tilelang.transform.AnalyzeAscendSchedule(function, "A3")


def _edges(graph):
    return {(int(e["src"]), int(e["dst"]), str(e["kind"])) for e in graph["dependencies"]}


def test_native_registration_is_read_only_and_uses_verified_libraries():
    a, b, c = [_buffer(n) for n in "abc"]
    function = _function([a, b, c], [_fill(a), _add(b, a, c), _fill(a), _fill(b)])
    original = tvm.ir.save_json(function)
    assert tvm.get_global_func("tl.transform.AnalyzeAscendSchedule")
    graph = _analyze(function)
    assert tvm.ir.save_json(function) == original
    assert int(graph["schema_version"]) == 1 and graph["interval_unit"] == "bytes"
    assert len(graph["tasks"]) == len(graph["groups"]) == 4
    assert {(0, 1, "RAW"), (1, 2, "WAR"), (0, 2, "WAW"), (1, 3, "WAW")} <= _edges(graph)
    if os.environ.get("TL_ASCEND_MIGRATION_BUILD_MANIFEST"):
        import json

        manifest = json.loads(Path(os.environ["TL_ASCEND_MIGRATION_BUILD_MANIFEST"]).read_text())
        libraries = {entry["path"] for entry in manifest["libraries"]}
        assert str(Path(tilelang._LIB_PATH).resolve()) in libraries
        assert str(Path(tvm._ffi.base._LIB._name).resolve()) in libraries
        assert Path(tvm.__file__).resolve().is_relative_to(Path(manifest["tvm_source_root"]) / "python")


@pytest.mark.parametrize("dtype,bytes_per_element", [("float16", 2), ("float32", 4)])
def test_full_copy_vector_copy_slice(dtype, bytes_per_element):
    source = _buffer("source", "global", dtype)
    output = _buffer("output", "global", dtype)
    a, b = [_buffer(n, dtype=dtype) for n in "ab"]
    function = _function([source, output, a, b], [_dma(source, a), _add(b, a, a), _dma(b, output)])
    graph = _analyze(function)
    assert [str(t["pipe"]) for t in graph["tasks"]] == ["PIPE_MTE2", "PIPE_V", "PIPE_MTE3"]
    assert {(0, 1, "RAW"), (1, 2, "RAW")} <= _edges(graph)
    assert all(bool(g["schedulable"]) for g in graph["groups"])
    assert int(graph["tasks"][0]["writes"][0]["end"]) == 64 * bytes_per_element


def test_real_lowered_copy_add_copy_is_accepted_before_planning():
    from tilelang.engine.phase import LowerAndLegalize

    @T.prim_func
    def program(source: T.Tensor((64,), "float32"), output: T.Tensor((64,), "float32")):
        with T.Kernel(1, threads=1, is_npu=True):
            a = T.alloc_ub((64,), "float32")
            b = T.alloc_ub((64,), "float32")
            with T.Scope("V"):
                T.copy(source, a)
                T.tile.add(b, a, a)
                T.copy(b, output)

    target = tvm.target.Target("c -keys=ascend -model=ascendc")
    module = LowerAndLegalize(tvm.IRModule({"main": program}), target)
    module = tir.transform.PlanAndUpdateBufferAllocationLocation()(module)
    module = tilelang.transform.CrossCorePipeline()(module)
    module = tilelang.transform.CombineCV()(module)
    module = tir.transform.PlanAndUpdateBufferAllocationLocation()(module)
    original = tvm.ir.save_json(module)
    graph = _analyze(module["main"])
    assert tvm.ir.save_json(module) == original
    assert len(graph["tasks"]) == 3
    assert [str(task["pipe"]) for task in graph["tasks"]] == ["PIPE_MTE2", "PIPE_V", "PIPE_MTE3"]
    assert {(0, 1, "RAW"), (1, 2, "RAW")} <= _edges(graph)
    eligible = [group for group in graph["groups"] if bool(group["schedulable"])]
    assert len(eligible) == 3
    for group in eligible:
        parent = int(group["parent_group_id"])
        assert parent >= 0
        assert not bool(graph["groups"][parent]["schedulable"])
        assert isinstance(graph["groups"][parent]["statement"], tir.AttrStmt)
    identities = {
        (
            tuple(map(int, group["loop_ids"])),
            int(group["parent_group_id"]),
            tuple(map(int, group["source_path"])),
        )
        for group in graph["groups"]
    }
    assert len(identities) == len(graph["groups"])


def test_same_data_var_and_different_dtype_views_alias_in_bytes():
    a = _buffer("a")
    view = _buffer("view", dtype="float16", shape=(256,), data=a.data)
    graph = _analyze(_function([a, view], [_fill(a, 16, 8), _fill(view, 32, 16)]))
    assert len(graph["storages"]) == 1
    assert (0, 1, "WAW") in _edges(graph)
    accesses = [task["writes"][0] for task in graph["tasks"]]
    assert [int(x["start"]) for x in accesses] == [64, 64]
    assert [int(x["end"]) for x in accesses] == [96, 96]


def test_equal_names_do_not_merge_distinct_local_allocations():
    a, b = _buffer("same_name"), _buffer("same_name")
    graph = _analyze(_function([a, b], [_fill(a), _fill(b)]))
    assert len(graph["storages"]) == 2
    assert not _edges(graph)


@pytest.mark.parametrize("noalias", [False, True])
def test_distinct_gm_parameters_may_alias_without_noalias(noalias):
    source, output = [_buffer(n, "global", shape=(256,)) for n in ("source", "output")]
    a, b = [_buffer(n) for n in "ab"]
    function = _function([source, output, a, b], [_dma(a, output), _dma(source, b, offset=128)], noalias=noalias)
    assert ((0, 1, "RAW") in _edges(_analyze(function))) is not noalias


@pytest.mark.parametrize("address", [0, 32, 1024])
def test_explicit_addresses_are_conservative_before_final_planning(address):
    a, b = [_buffer(n) for n in "ab"]
    graph = _analyze(_function([a, b], [_fill(a), _fill(b)], addresses={a.data: 0, b.data: address}))
    assert (0, 1, "WAW") in _edges(graph)


def test_nested_block_groups_follow_injector_flattening():
    a, b = [_buffer(n) for n in "ab"]
    nested = tir.BlockRealize([], True, tir.Block([], [], [], "nested", tir.SeqStmt([_fill(a), _fill(b)])))
    graph = _analyze(_function([a, b], [_fill(a), nested]))
    assert [list(map(int, g["source_path"])) for g in graph["groups"]] == [[0], [1, 0], [1, 1]]
    assert [list(map(int, g["task_ids"])) for g in graph["groups"]] == [[0], [1], [2]]


def test_complex_group_stays_indivisible_and_records_guard():
    a, b = [_buffer(n) for n in "ab"]
    n = tir.Var("n", "int32")
    branch = tir.IfThenElse(n > 0, tir.SeqStmt([_fill(a), _fill(b)]), None)
    graph = _analyze(_function([a, b], [branch], scalars=[n]))
    assert len(graph["groups"]) == 1
    assert not bool(graph["groups"][0]["schedulable"])
    assert (0, 1, "GROUP_ORDER") in _edges(graph)
    assert not tvm.tir.analysis.undefined_vars(graph["tasks"][0]["guard"], [n])


def test_nested_loop_and_symbolic_vector_bounds_remain_symbolic():
    a, b, c = [_buffer(n) for n in "abc"]
    i, j, n = [tir.Var(x, "int32") for x in ("i", "j", "n")]
    body = tir.SeqStmt([_fill(a), _add(b, a, c, tir.Min(n, 64))])
    loop = tir.For(i, 0, 2, tir.ForKind.SERIAL, tir.For(j, 0, n, tir.ForKind.SERIAL, body))
    graph = _analyze(_function([a, b, c], [loop], scalars=[n]))
    assert len(graph["loops"]) == 2
    assert list(map(int, graph["tasks"][1]["loop_ids"])) == [0, 1]
    assert graph["loops"][1]["extent"].dtype == "int64"
    assert graph["tasks"][1]["writes"][0]["end"].dtype == "int64"
    assert not bool(graph["tasks"][1]["writes"][0]["exact"])
    assert (0, 1, "RAW") in _edges(graph)


def test_index_arithmetic_is_widened_before_endpoint_construction():
    source = _buffer("source", "global", shape=(1 << 35,))
    a = _buffer("a")
    index = tir.Var("index", "int32")
    offset = tir.Mul(index, tir.IntImm("int32", 1 << 30))
    graph = _analyze(_function([source, a], [_dma(source, a, offset=offset)], scalars=[index]))
    start = graph["tasks"][0]["reads"][0]["start"]
    value = tvm.arith.Analyzer().simplify(tir.stmt_functor.substitute(start, {index: tir.IntImm("int32", 4)}))
    assert int(value) == (4 * (1 << 30) * 4)


@pytest.mark.parametrize("kind", ["raw", "flag", "mask", "compound", "unknown"])
def test_unknown_manual_state_and_compound_effects_are_rejected(kind):
    a = _buffer("a")
    calls = {
        "raw": tir.Call("handle", tvm.ir.Op.get("tl.ascend_src_code"), [tir.StringImm("int x=0;")]),
        "flag": tir.Call(
            "handle",
            tvm.ir.Op.get("tl.ascend_set_flag"),
            [tir.StringImm("V"), tir.StringImm("MTE3"), tir.IntImm("int32", 0)],
        ),
        "mask": tir.Call("handle", tvm.ir.Op.get("tl.ascend_set_mask_mode"), [tir.IntImm("int32", 0)]),
        "compound": tir.call_extern("handle", "atomic_add_ub_to_gm<float,64>", a.access_ptr("r"), a.access_ptr("w")),
        "unknown": tir.call_extern("handle", "opaque_helper", a.access_ptr("w")),
    }
    with pytest.raises(tvm.TVMError, match="Unsupported schedule"):
        _analyze(_function([a], [tir.Evaluate(calls[kind])]))


@pytest.mark.parametrize("valid", [32, "dynamic"])
def test_partial_or_dynamic_dma_padding_is_rejected(valid):
    source = _buffer("source", "global")
    a = _buffer("a")
    n = tir.Var("n", "int32")
    with pytest.raises(
        tvm.TVMError,
        match="Unsupported dynamic DMA padding|Unsupported schedule DMA template/physical footprint",
    ):
        _analyze(
            _function(
                [source, a],
                [_dma(source, a, valid=n if valid == "dynamic" else valid)],
                scalars=[n],
            )
        )


@pytest.mark.parametrize("scope", [None, 0])
def test_resource_ownership_is_required(scope):
    a = _buffer("a")
    with pytest.raises(tvm.TVMError, match="explicit V owner"):
        _analyze(_function([a], [_fill(a)], scope=scope))


def test_loop_reuse_does_not_claim_loop_carried_coverage():
    a, b = [_buffer(n) for n in "ab"]
    i = tir.Var("i", "int32")
    loop = tir.For(i, 0, 3, tir.ForKind.SERIAL, tir.SeqStmt([_fill(a), _add(b, a, a)]))
    graph = _analyze(_function([a, b], [loop]))
    assert graph["dependency_scope"] == "single_iteration"
    assert (0, 1, "RAW") in _edges(graph)
    relevant = [group for group in graph["groups"] if len(group["task_ids"])]
    assert all(bool(group["requires_sequential_iterations"]) for group in relevant)
    assert not any(bool(group["schedulable"]) for group in relevant)


def test_managed_vectors_reject_gm_operands():
    gm = _buffer("gm", "global")
    with pytest.raises(tvm.TVMError, match="must be UB"):
        _analyze(_function([gm], [_fill(gm)]))


def test_unregistered_local_pointer_cannot_prove_independence():
    raw = _buffer("raw")
    with pytest.raises(tvm.TVMError, match="Unregistered schedule storage"):
        _analyze(_function([], [_fill(raw)]))


def test_view_cannot_expand_physical_allocation():
    a = _buffer("a", shape=(64,))
    view = _buffer("view", shape=(256,), data=a.data)
    with pytest.raises(tvm.TVMError, match="physical allocation"):
        _analyze(_function([a], [_fill(view, offset=64)]))


def test_pointer_extent_does_not_override_physical_capacity():
    a = _buffer("a")
    call = _fill(a).value
    args = list(call.args)
    args[1] = a.access_ptr("w", offset=120, extent=64)
    with pytest.raises(tvm.TVMError, match="physical allocation"):
        _analyze(_function([a], [tir.Evaluate(tir.Call(call.dtype, call.op, args))]))


@pytest.mark.parametrize("dtype,source_dtype", [("int32", "int64"), ("uint64", "int32")])
def test_lossy_or_uint64_index_casts_are_rejected(dtype, source_dtype):
    source = _buffer("source", "global", shape=(1 << 35,))
    a = _buffer("a")
    n = tir.Var("n", source_dtype)
    with pytest.raises(tvm.TVMError, match="schedule.*cast"):
        _analyze(_function([source, a], [_dma(source, a, offset=tir.Cast(dtype, n))], scalars=[n]))


@pytest.mark.parametrize("argument", [3, 6])
def test_dma_scalar_arguments_cannot_hide_reads(argument):
    source = _buffer("source", "global")
    a = _buffer("a")
    call = _dma(source, a).value
    args = list(call.args)
    args[argument] = tir.BufferLoad(source, [0])
    with pytest.raises(tvm.TVMError, match="memory-dependent/opaque DMA"):
        _analyze(_function([source, a], [tir.Evaluate(tir.Call(call.dtype, call.op, args))]))


def test_unknown_semantic_attribute_is_rejected():
    a = _buffer("a")
    body = tir.AttrStmt(tir.const(0, "int32"), "unknown_memory_semantics", 1, _fill(a))
    with pytest.raises(tvm.TVMError, match="Unsupported schedule attribute"):
        _analyze(_function([a], [body]))


def test_conditional_allocation_is_rejected():
    a = _buffer("a")
    n = tir.Var("n", "int32")
    body = tir.Allocate(a.data, a.dtype, [128], n > 0, _fill(a))
    with pytest.raises(tvm.TVMError, match="conditional schedule allocation"):
        _analyze(_function([a], [body], scalars=[n]))


def test_equivalent_new_functions_have_the_same_stable_ids():
    def make():
        source, output = [_buffer(n, "global") for n in ("source", "output")]
        a = _buffer("a")
        return _analyze(_function([source, output, a], [_dma(source, a), _dma(a, output)]))

    left, right = make(), make()
    for graph in (left, right):
        assert [int(storage["id"]) for storage in graph["storages"]] == [0, 1, 2]
        assert [str(storage["data"].name) for storage in graph["storages"]] == [
            "source",
            "output",
            "a",
        ]
        assert [int(task["id"]) for task in graph["tasks"]] == [0, 1]


def test_unproven_dynamic_bounds_cannot_prove_disjoint_accesses():
    a = _buffer("a")
    n = tir.Var("n", "int32")
    graph = _analyze(_function([a], [_fill(a, count=tir.Min(n, 64)), _fill(a, offset=64)], scalars=[n]))
    assert not bool(graph["tasks"][0]["writes"][0]["bounded"])
    assert (0, 1, "WAW") in _edges(graph)


def test_nested_flattening_does_not_drop_block_initialization_effects():
    a, b = [_buffer(n) for n in "ab"]
    nested = tir.BlockRealize(
        [],
        True,
        tir.Block([], [], [], "nested", tir.SeqStmt([_fill(a), _fill(b)]), init=_fill(a)),
    )
    with pytest.raises(tvm.TVMError, match="unlowered schedule block"):
        _analyze(_function([a, b], [_fill(a), nested]))


def test_nested_flattening_keeps_explicit_address_alias_metadata():
    a, b = [_buffer(n) for n in "ab"]
    nested = tir.BlockRealize(
        [],
        True,
        tir.Block(
            [],
            [],
            [],
            "nested",
            tir.SeqStmt([_fill(a), _fill(b)]),
            annotations={"address_map": {a.data: 0, b.data: 0}},
        ),
    )
    graph = _analyze(_function([a, b], [_fill(a), nested]))
    assert (1, 2, "WAW") in _edges(graph)


@pytest.mark.parametrize("count", [tir.FloatImm("float32", 64.0), tir.const(True, "bool")])
def test_managed_count_must_be_an_integer_scalar(count):
    a = _buffer("a")
    with pytest.raises(tvm.TVMError, match="schedule integer index/count"):
        _analyze(_function([a], [_fill(a, count=count)]))
