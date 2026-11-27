"""Native canonical storage identity and raw-pointer contracts; no device execution."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

import tilelang
from tilelang import tvm

from test_s2_task_ir import _buffer, _fill, _function

tir = tvm.tir


def owners(function):
    before = tvm.ir.save_json(function)
    result = tilelang.transform.AnalyzeAscendStorageOwners(function, "A3")
    assert tvm.ir.save_json(function) == before
    assert isinstance(result, tvm.ir.Map)
    assert int(result["schema_version"]) == 1
    return result


def canonical(function, strategy="constraint"):
    before = tvm.ir.save_json(function)
    with tvm.transform.PassContext(config={"tl.ascend_pipeline_planner": strategy}):
        first = tilelang.transform.AscendCanonicalizeStorageAliases()(tvm.IRModule({"main": function}))["main"]
        first_json = tvm.ir.save_json(first)
        second = tilelang.transform.AscendCanonicalizeStorageAliases()(tvm.IRModule({"main": first}))["main"]
    assert tvm.ir.save_json(function) == before
    assert tvm.ir.save_json(second) == first_json
    assert second.same_as(first)
    return first


def pointers(function):
    found = []

    def visit(node):
        if isinstance(node, tir.Call) and getattr(node.op, "name", None) == "tir.tvm_access_ptr":
            found.append(node)

    tir.stmt_functor.post_order_visit(function.body if isinstance(function, tir.PrimFunc) else function, visit)
    return found


def integer(expression):
    value = tvm.arith.Analyzer().simplify(expression)
    assert isinstance(value, tir.IntImm), value
    return int(value)


def test_public_owner_registration_and_source_readonly_contract():
    assert tvm.get_global_func("tl.transform.AnalyzeAscendStorageOwners")
    assert tvm.get_global_func("tl.transform.AscendCanonicalizeStorageAliases")
    a = _buffer("a", shape=(64,))
    function = _function([a], _fill(a))
    report = owners(function)
    storage = report["storages"][0]
    assert len(report["storages"]) == 1
    assert storage["data"].same_as(a.data) and storage["source_data"].same_as(a.data)
    assert str(storage["scope"]) == "shared.ub"
    assert integer(storage["capacity_bytes"]) == 256
    assert bool(storage["physical_allocation"])
    assert str(storage["origin"]) == "allocation"
    assert int(storage["allocation_site_id"]) >= 0


def test_cold_owner_analysis_and_canonicalization_do_not_request_solver():
    source = """import sys
class RejectSolver:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0]=='z3' or 'ascend_z3_scheduler' in fullname:
            raise AssertionError('readonly owner analysis requested solver: '+fullname)
sys.meta_path.insert(0,RejectSolver())
import tilelang
from tilelang import tvm
from test_s2_task_ir import _buffer,_fill,_function
root=_buffer('root',shape=(64,))
function=_function([root],_fill(root))
before=tvm.ir.save_json(function)
report=tilelang.transform.AnalyzeAscendStorageOwners(function,'A3')
assert int(report['schema_version'])==1
with tvm.transform.PassContext(config={'tl.ascend_pipeline_planner':'constraint'}):
    result=tilelang.transform.AscendCanonicalizeStorageAliases()(tvm.IRModule({'main':function}))['main']
assert result.same_as(function) and tvm.ir.save_json(function)==before
assert tvm.get_global_func('tl.ascend.solve_schedule',allow_missing=True) is None
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parent), env.get("PYTHONPATH", "")])
    result = subprocess.run([sys.executable, "-c", source], env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("strategy", ["legacy", "constraint"])
def test_no_alias_canonicalization_preserves_function_identity(strategy):
    a = _buffer("a", shape=(64,))
    function = _function([a], _fill(a))
    assert canonical(function, strategy).same_as(function)


@pytest.mark.parametrize(
    "dtype,shape,offset,count",
    [
        ("float32", (64,), 16, 16),
        ("float16", (128,), 32, 32),
    ],
)
def test_same_data_offset_and_dtype_views_have_one_byte_owner(dtype, shape, offset, count):
    root = _buffer("root", shape=(64,))
    view = _buffer("view", dtype=dtype, shape=shape, data=root.data)
    function = _function([root], [_fill(root, 16, 16), _fill(view, offset, count)])
    report = owners(function)
    assert len(report["storages"]) == 1
    graph = tilelang.transform.AnalyzeAscendSchedule(canonical(function), "A3")
    first, second = [task["writes"][0] for task in graph["tasks"]]
    assert int(first["storage_id"]) == int(second["storage_id"])
    assert (integer(first["start"]), integer(first["end"])) == (64, 128)
    assert (integer(second["start"]), integer(second["end"])) == (64, 128)


def test_equal_names_remain_distinct_allocations_and_metadata_is_renamed():
    a, b = [_buffer("same_name", shape=(64,)) for _ in range(2)]
    function = _function([a, b], [_fill(a), _fill(b)], addresses={a.data: 0, b.data: 512})
    for key in ("initial_buffer_shapes", "buffer_shapess", "logic_buffer_shapes"):
        function = function.with_attr(key, {a.data: [64], b.data: [64]})
    function = function.with_attr("size_map", {a.data: 256, b.data: 256})
    result = canonical(function)
    report = owners(result)
    rows = list(report["storages"])
    assert len(rows) == 2
    assert len({int(row["allocation_site_id"]) for row in rows}) == 2
    assert len({row["data"].name for row in rows}) == 2
    assert all(any(row["source_data"].same_as(source.data) for source in (a, b)) for row in rows)
    for row in rows:
        for key in (
            "initial_buffer_shapes",
            "buffer_shapess",
            "logic_buffer_shapes",
            "address_map",
            "size_map",
        ):
            assert row["data"] in result.attrs[key]
        expected = 0 if row["source_data"].same_as(a.data) else 512
        assert integer(result.attrs["address_map"][row["data"]]) == expected
        assert integer(result.attrs["size_map"][row["data"]]) == 256


@pytest.mark.parametrize("binding_offset,consumer_offset", [(0, 0), (8, 0), (8, 16)])
def test_pointer_binding_offsets_are_added_once(binding_offset, consumer_offset):
    root = _buffer("root", shape=(64,))
    alias = tir.Var("alias", root.data.type_annotation)
    view = _buffer("alias_view", shape=(64,), data=alias)
    binding = root.data if not binding_offset else root.access_ptr("rw", offset=binding_offset)
    function = _function([root], tir.LetStmt(alias, binding, _fill(view, consumer_offset, 32)))
    result = canonical(function)
    actual = pointers(result)
    assert len(actual) == 1 and actual[0].args[1].same_as(root.data)
    assert integer(actual[0].args[2]) == binding_offset + consumer_offset
    report = owners(result)
    assert len(report["storages"]) == 1 and len(report["aliases"]) == 1
    record = report["aliases"][0]
    assert record["data"].same_as(alias) and record["root"].same_as(root.data)
    assert integer(record["byte_offset"]) == binding_offset * 4
    assert int(record["storage_id"]) == int(report["storages"][0]["id"])


def test_nested_pointer_alias_chain_uses_one_canonical_root():
    root = _buffer("root", shape=(64,))
    first, second = [tir.Var(name, root.data.type_annotation) for name in ("first", "second")]
    first_view = _buffer("first_view", shape=(64,), data=first)
    second_view = _buffer("second_view", shape=(64,), data=second)
    statement = tir.LetStmt(second, first_view.access_ptr("rw", offset=8), _fill(second_view, 8, 32))
    function = _function([root], tir.LetStmt(first, root.access_ptr("rw", offset=8), statement))
    result = canonical(function)
    pointer = pointers(result)[0]
    assert pointer.args[1].same_as(root.data) and integer(pointer.args[2]) == 24
    assert len(owners(result)["aliases"]) == 2


@pytest.mark.parametrize("dtype,elements,expected", [("float16", 64, 32), ("float32", 32, 16)])
def test_raw_pointer_bytes_are_converted_to_the_consumer_dtype(dtype, elements, expected):
    root = _buffer("root", shape=(64,))
    alias = tir.Var("alias", root.data.type_annotation)
    view = _buffer("typed_view", dtype=dtype, shape=(elements,), data=alias)
    function = _function([root], tir.LetStmt(alias, root.access_ptr("rw", offset=16), _fill(view, 0, 32)))
    result = canonical(function)
    pointer = pointers(result)[0]
    assert pointer.args[1].same_as(root.data)
    assert str(pointer.args[0].dtype) == dtype and integer(pointer.args[2]) == expected
    assert len(owners(result)["storages"]) == 1


def test_buffer_load_store_and_declared_view_base_follow_the_binding():
    root = _buffer("root", shape=(64,))
    alias = tir.Var("alias", root.data.type_annotation)
    view = _buffer("view", shape=(32,), elem_offset=8, data=alias)
    body = tir.BufferStore(view, tir.BufferLoad(view, [0]), [1])
    function = _function([root], tir.LetStmt(alias, root.access_ptr("rw", offset=8), body))
    result = canonical(function)
    stores, loads = [], []

    def visit(node):
        if isinstance(node, tir.BufferStore):
            stores.append(node)
        if isinstance(node, tir.BufferLoad):
            loads.append(node)

    tir.stmt_functor.post_order_visit(result.body, visit)
    assert len(stores) == len(loads) == 1
    for node in (stores[0], loads[0]):
        assert node.buffer.data.same_as(root.data)
        assert integer(node.buffer.elem_offset) == 16


@pytest.mark.parametrize("escape", [False, True])
def test_opaque_alias_expression_and_pointer_escape_are_rejected(escape):
    root = _buffer("root", shape=(64,))
    alias = tir.Var("alias", root.data.type_annotation)
    if escape:
        binding = root.data
        body = tir.Evaluate(tir.call_extern("handle", "opaque_escape", alias))
    else:
        binding = tir.call_extern("handle", "opaque_pointer", root.data)
        body = _fill(_buffer("view", shape=(64,), data=alias))
    function = _function([root], tir.LetStmt(alias, binding, body))
    with pytest.raises(tvm.TVMError):
        canonical(function)
    assert canonical(function, "legacy").same_as(function)


def test_unregistered_raw_pointer_cannot_acquire_an_owner_from_its_name():
    root, unrelated = [_buffer("same_name", shape=(64,)) for _ in range(2)]
    with pytest.raises(tvm.TVMError):
        owners(_function([root], _fill(unrelated)))


def test_byte_alias_must_be_divisible_by_the_consuming_dtype():
    root = _buffer("root", dtype="float16", shape=(128,))
    alias = tir.Var("alias", root.data.type_annotation)
    view = _buffer("wide", dtype="float32", shape=(32,), data=alias)
    function = _function([root], tir.LetStmt(alias, root.access_ptr("rw", offset=1), _fill(view)))
    with pytest.raises(tvm.TVMError):
        canonical(function)


def test_raw_pointer_byte_scaling_cannot_wrap_a_huge_element_offset_to_zero():
    root = _buffer("root", shape=(64,))
    alias = tir.Var("alias", root.data.type_annotation)
    view = _buffer("view", shape=(64,), data=alias)
    binding = root.access_ptr("rw", offset=tir.IntImm("int64", 1 << 62))
    function = _function([root], tir.LetStmt(alias, binding, _fill(view)))
    with pytest.raises(tvm.TVMError, match="int64|extent|overflow"):
        canonical(function)
    with pytest.raises(tvm.TVMError, match="int64|extent|overflow"):
        owners(function)


@pytest.mark.parametrize(
    "mask,dtype",
    [(0, "int32"), (-1, "int32"), (4, "int32"), (3, "uint32"), (1.0, "float32"), (True, "bool")],
)
def test_pointer_binding_requires_a_signed_access_mask_from_one_to_three(mask, dtype):
    root = _buffer("root", shape=(64,))
    alias = tir.Var("alias", root.data.type_annotation)
    view = _buffer("view", shape=(64,), data=alias)
    binding = root.access_ptr("rw")
    args = list(binding.args)
    args[4] = tir.const(mask, dtype)
    function = _function([root], tir.LetStmt(alias, tir.Call(binding.dtype, binding.op, args), _fill(view)))
    with pytest.raises(tvm.TVMError):
        canonical(function)


@pytest.mark.parametrize("kind", ["buffer_load", "opaque_call"])
def test_removing_a_pointer_binding_cannot_drop_extent_side_effects(kind):
    root = _buffer("root", shape=(64,))
    external = _buffer("extent", "global", "int32", shape=(1,))
    alias = tir.Var("alias", root.data.type_annotation)
    view = _buffer("view", shape=(64,), data=alias)
    binding = root.access_ptr("rw")
    args = list(binding.args)
    args[3] = tir.BufferLoad(external, [0]) if kind == "buffer_load" else tir.call_extern("int32", "opaque_extent")
    function = _function([external, root], tir.LetStmt(alias, tir.Call(binding.dtype, binding.op, args), _fill(view)))
    with pytest.raises(tvm.TVMError):
        canonical(function)


@pytest.mark.parametrize("offset", [0, 8])
def test_address_and_size_metadata_follow_only_a_zero_offset_known_pointer_alias(offset):
    root = _buffer("root", shape=(64,))
    alias = tir.Var("alias", root.data.type_annotation)
    view = _buffer("view", shape=(32,), data=alias)
    function = _function([root], tir.LetStmt(alias, root.access_ptr("rw", offset=offset), _fill(view, count=32)))
    function = function.with_attr("address_map", {alias: 32}).with_attr("size_map", {alias: 256})
    if offset:
        with pytest.raises(tvm.TVMError):
            canonical(function)
    else:
        result = canonical(function)
        assert integer(result.attrs["address_map"][root.data]) == 32
        assert integer(result.attrs["size_map"][root.data]) == 256
        assert alias not in result.attrs["address_map"] and alias not in result.attrs["size_map"]


@pytest.mark.parametrize("count,offset", [(64, 32), (32, 64)])
def test_alias_view_cannot_expand_the_physical_owner(count, offset):
    root = _buffer("root", shape=(64,))
    view = _buffer("large_view", shape=(128,), data=root.data)
    with pytest.raises(tvm.TVMError):
        owners(_function([root], _fill(view, offset, count)))


@pytest.mark.parametrize(
    "key",
    ["address_map", "size_map", "initial_buffer_shapes", "buffer_shapess", "logic_buffer_shapes"],
)
def test_unknown_or_dead_metadata_owner_is_rejected(key):
    root, unknown = [_buffer(name, shape=(64,)) for name in ("root", "unregistered")]
    value = [64] if "shape" in key else 0 if key == "address_map" else 256
    function = _function([root], _fill(root)).with_attr(key, {unknown.data: value})
    with pytest.raises(tvm.TVMError):
        owners(function)


@pytest.mark.parametrize("scope", [None, 0])
def test_owner_catalog_cannot_hide_wrong_or_missing_vector_execution_owner(scope):
    root = _buffer("root", shape=(64,))
    with pytest.raises(tvm.TVMError):
        owners(_function([root], _fill(root), scope=scope))


def test_allocation_site_records_lexical_parent_loop_identity():
    root = _buffer("root", shape=(64,))
    i = tir.Var("i", "int32")
    block = tir.Block([], [], [], "local_activation", _fill(root), alloc_buffers=[root])
    body = tir.For(i, 0, 3, tir.ForKind.SERIAL, tir.BlockRealize([], True, block))
    function = _function([], body)
    row = owners(function)["storages"][0]
    graph = tilelang.transform.AnalyzeAscendSchedule(canonical(function), "A3")
    assert [int(value) for value in row["parent_loop_ids"]] == [int(graph["loops"][0]["id"])]


def test_explicit_storage_analysis_rejects_overflow_before_a_folded_capacity_can_hide_it():
    shape = [
        tir.IntImm("int64", value)
        for value in (
            (1 << 32) + 1,
            (1 << 32) - 1,
            (1 << 62) - 64,
            64,
        )
    ]
    root = _buffer("overflowing_owner", shape=shape)
    function = _function([root], _fill(root))
    for analyze in (
        tilelang.transform.AnalyzeAscendStorageOwners,
        tilelang.transform.AnalyzeAscendLifetimes,
    ):
        before = tvm.ir.save_json(function)
        with pytest.raises(tvm.TVMError):
            analyze(function, "A3")
        assert tvm.ir.save_json(function) == before


def test_explicit_owner_analysis_rejects_other_platforms():
    root = _buffer("root", shape=(64,))
    with pytest.raises(tvm.TVMError):
        tilelang.transform.AnalyzeAscendStorageOwners(_function([root], _fill(root)), "A5")


def test_dense_match_buffer_keeps_the_one_owner_and_view_byte_range():
    root = _buffer("root", shape=(64,))
    view = _buffer("matched", shape=(32,))
    region = tir.BufferRegion(root, [tvm.ir.Range.from_min_extent(16, 32)])
    block = tir.Block(
        [],
        [],
        [],
        "matched_scope",
        _fill(view, count=32),
        match_buffers=[tir.MatchBufferRegion(view, region)],
    )
    function = _function([root], tir.BlockRealize([], True, block))
    result = canonical(function)
    pointer = pointers(result)[0]
    assert pointer.args[1].same_as(root.data) and integer(pointer.args[2]) == 16
    report = owners(function)
    assert len(report["storages"]) == 1
    matched = [row for row in report["views"] if str(row["dtype"]) == "float32" and [integer(value) for value in row["shape"]] == [32]]
    assert matched
    assert any(integer(row["byte_begin"]) == 64 and integer(row["byte_end"]) == 192 for row in matched)


def test_generic_scope_rewrite_updates_block_view_declarations_without_mutating_source():
    root = _buffer("root", scope="shared", shape=(32,))
    half = _buffer("half", scope="shared", dtype="float16", shape=(64,), data=root.data)
    matched = _buffer("matched", scope="shared", dtype="float16", shape=(64,))

    def full(buffer):
        return tir.BufferRegion(buffer, [tvm.ir.Range.from_min_extent(0, buffer.shape[0])])

    block = tir.Block(
        [],
        [full(root)],
        [full(half)],
        "declared_views",
        _fill(root, count=32),
        match_buffers=[tir.MatchBufferRegion(matched, full(half))],
    )
    function = _function([root], tir.BlockRealize([], True, block))
    before = tvm.ir.save_json(function)
    early = canonical(function)
    early_json = tvm.ir.save_json(early)
    with tvm.transform.PassContext(config={"tl.ascend_pipeline_planner": "constraint"}):
        result = tilelang.transform.AscendInferBufferScope()(tvm.IRModule({"main": early}))["main"]
    assert tvm.ir.save_json(function) == before
    assert tvm.ir.save_json(early) == early_json
    allocations, declarations = [], []

    def visit(node):
        if isinstance(node, tir.Block):
            allocations.extend(node.alloc_buffers)
            declarations.extend(region.buffer for region in [*node.reads, *node.writes])
            for match in node.match_buffers:
                declarations.extend((match.buffer, match.source.buffer))

    tir.stmt_functor.post_order_visit(result.body, visit)
    assert len(allocations) == 1 and declarations
    owner = allocations[0]
    assert owner.scope() == "shared.ub" and not owner.data.same_as(root.data)
    assert {str(buffer.dtype) for buffer in declarations} == {"float16", "float32"}
    for buffer in declarations:
        assert buffer.scope() == "shared.ub" and buffer.data.same_as(owner.data)
        assert integer(buffer.elem_offset) == 0
        assert integer(buffer.shape[0]) * tvm.DataType(buffer.dtype).bits // 8 == 128


@pytest.mark.parametrize("kind,dtype,tile", [("views", "float32", 64), ("dtype_views", "float32", 32)])
def test_public_alias_fixture_has_complete_schema3_lowering(kind, dtype, tile):
    from test_s5_versions_ir import configs
    from test_s6_kernels import make_reuse_kernel

    with tvm.transform.PassContext(config=configs()):
        artifact = tilelang.lower(make_reuse_kernel(kind, dtype=dtype, tile=tile), target="ascendc", platform="A3")
    for function in artifact.device_mod.functions.values():
        manifest = function.attrs["tl.ascend_constraint_plan"]
        assert int(manifest["schema_version"]) == 3
        assert manifest["lifetime_contract"]
    assert "copy_gm_to_ub" in artifact.kernel_source and "copy_ub_to_gm" in artifact.kernel_source


@pytest.mark.parametrize("developer", [False, True])
def test_private_mixed_dtype_versions_share_one_actual_128_byte_pitch(developer):
    from test_s5_versions_ir import configs
    from test_s6_kernels import make_dtype_version_kernel

    with tvm.transform.PassContext(config=configs()):
        artifact = tilelang.lower(make_dtype_version_kernel(developer=developer), target="ascendc", platform="A3")
    analyzer = tvm.arith.Analyzer()
    for function in artifact.device_mod.functions.values():
        lifetime = function.attrs["tl.ascend_constraint_plan"]["lifetime_contract"]
        half_roots = []
        for task in lifetime["source_graph"]["tasks"]:
            for pointer in pointers(task["statement"]):
                if str(pointer.args[0].dtype) == "float16" and not any(pointer.args[1].same_as(value) for value in half_roots):
                    half_roots.append(pointer.args[1])
        assert len(half_roots) == 1
        owners_ = [
            row
            for row in lifetime["storage_catalog"]["storages"]
            if bool(row["physical_allocation"]) and integer(row["capacity_bytes"]) == 128 and row["data"].same_as(half_roots[0])
        ]
        assert len(owners_) == 1
        owner = owners_[0]
        assert str(owner["scope"]) == "shared.ub"
        assert str(owner["allocation"].dtype) == "float32"
        source_map = function.attrs.get("tl.ascend_storage_source_data", {})

        def original(variable, source_map=source_map):
            for _ in range(len(source_map) + 1):
                if variable not in source_map or source_map[variable].same_as(variable):
                    return variable
                variable = source_map[variable]
            raise AssertionError("Cyclic owner provenance")

        remaps = [
            row
            for row in function.attrs["tl.ascend_pipeline_buffer_remap"]
            if row["logical_data"].same_as(owner["data"]) or original(row["logical_data"]).same_as(owner["source_data"])
        ]
        assert len(remaps) == 1
        remap = remaps[0]
        actual, pitch = integer(remap["versions"]), integer(remap["version_stride_bytes"])
        assert 2 <= actual <= 3 and pitch == integer(remap["version_span_bytes"]) == 128
        assert integer(remap["total_bytes"]) == actual * pitch
        typed = set()
        for pointer in pointers(function):
            if pointer.args[1].same_as(remap["physical_data"]):
                dtype = str(pointer.args[0].dtype)
                typed.add(dtype)
                width = tvm.DataType(dtype).bits // 8
                assert analyzer.can_prove(tir.floormod(pointer.args[2] * width, pitch) == 0)
                assert integer(pointer.args[3]) * width == 128
        assert typed == {"float16", "float32"}
        allocations = []

        def visit(node, remap=remap, allocations=allocations):
            if isinstance(node, tir.Allocate) and node.buffer_var.same_as(remap["physical_data"]):
                size = tvm.DataType(node.dtype).bits // 8
                for extent in node.extents:
                    size *= integer(extent)
                allocations.append(size)

        tir.stmt_functor.post_order_visit(function.body, visit)
        assert allocations == [actual * pitch]


@pytest.mark.parametrize("base,valid", [(32, True), (196224, False)])
def test_explicit_base_must_fit_the_actual_multi_version_owner_not_just_n1(base, valid):
    from test_s5_versions_ir import configs
    from test_s6_kernels import make_dtype_version_kernel

    program = make_dtype_version_kernel()
    half_roots = []
    for pointer in pointers(program):
        if str(pointer.args[0].dtype) == "float16" and not any(pointer.args[1].same_as(value) for value in half_roots):
            half_roots.append(pointer.args[1])
    assert len(half_roots) == 1
    modified = []

    def annotate(block):
        if any(buffer.data.same_as(half_roots[0]) for buffer in block.alloc_buffers):
            attributes = dict(block.annotations)
            attributes["address_map"] = {half_roots[0]: base}
            modified.append(block)
            return tir.Block(
                block.iter_vars,
                block.reads,
                block.writes,
                block.name_hint,
                block.body,
                block.init,
                block.alloc_buffers,
                block.match_buffers,
                attributes,
            )
        return block

    program = program.with_body(tir.stmt_functor.ir_transform(program.body, None, annotate, ["tir.Block"]))
    assert len(modified) == 1
    with tvm.transform.PassContext(config=configs()):
        if valid:
            artifact = tilelang.lower(program, target="ascendc", platform="A3")
            assert any(
                base in [integer(value) for value in function.attrs["address_map"].values()]
                for function in artifact.device_mod.functions.values()
            )
        else:
            with pytest.raises(tvm.TVMError, match="actual versions|UB capacity"):
                tilelang.lower(program, target="ascendc", platform="A3")
