# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Host-only schema and reconstruction checks for the Cython allocation promise."""

import hashlib
from types import SimpleNamespace

import pytest

from tilelang import tvm
from tilelang.jit.ascend_runtime_contract import (
    AUTO_GM_FRESH_CONTRACT_ATTR,
    AUTO_GM_FRESH_CONTRACT_VERSION,
    AUTO_GM_MANIFEST_PATH,
    build_cached_auto_gm_manifest,
    prepare_auto_gm_contract,
    requires_fresh_auto_gm,
    validate_cached_auto_gm_contract,
    validate_cached_auto_gm_manifest,
)


def _function() -> tvm.tir.PrimFunc:
    return tvm.tir.PrimFunc([], tvm.tir.Evaluate(0)).with_attr("existing_marker", 7)


def _prepared() -> tvm.tir.PrimFunc:
    return prepare_auto_gm_contract(_function(), "cython", "ascendc")


def test_prepare_clones_attributes_without_changing_the_input_or_noalias() -> None:
    original = _function()
    prepared = prepare_auto_gm_contract(original, "cython", "ascendc")

    assert not prepared.same_as(original)
    assert prepared.body.same_as(original.body)
    assert AUTO_GM_FRESH_CONTRACT_ATTR not in original.attrs
    assert "tir.noalias" not in original.attrs
    assert "tir.noalias" not in prepared.attrs
    assert int(prepared.attrs["existing_marker"]) == 7
    marker = prepared.attrs[AUTO_GM_FRESH_CONTRACT_ATTR]
    assert isinstance(marker, tvm.tir.StringImm)
    assert marker.value == AUTO_GM_FRESH_CONTRACT_VERSION
    assert requires_fresh_auto_gm(prepared)


@pytest.mark.parametrize("noalias", [False, True])
def test_existing_noalias_contract_is_preserved(noalias: bool) -> None:
    original = _function().with_attr("tir.noalias", noalias)
    prepared = prepare_auto_gm_contract(original, "cython", "ascendc")
    assert int(prepared.attrs["tir.noalias"]) == int(noalias)
    assert int(original.attrs["tir.noalias"]) == int(noalias)


@pytest.mark.parametrize(
    "target",
    [
        "ascendc",
        "auto",
        tvm.target.Target({"kind": "llvm", "model": "ascendc"}),
        tvm.target.Target({"kind": "llvm", "model": "auto"}),
        tvm.target.Target({"kind": "c", "model": "ascendc"}),
    ],
)
def test_native_target_kind_does_not_hide_the_ascend_model(target: object) -> None:
    assert requires_fresh_auto_gm(prepare_auto_gm_contract(_function(), "cython", target))


@pytest.mark.parametrize("backend", ["ctypes", "dlpack", "unknown"])
def test_other_adapters_cannot_reuse_an_old_allocation_promise(backend: str) -> None:
    original = _prepared()
    result = prepare_auto_gm_contract(original, backend, "ascendc")
    assert not requires_fresh_auto_gm(result)
    assert requires_fresh_auto_gm(original)


@pytest.mark.parametrize("target", ["llvm", "pto", "llvm -model=ascendc", tvm.target.Target("llvm")])
def test_unrecognized_targets_do_not_acquire_the_contract(target: object) -> None:
    original = _prepared()
    result = prepare_auto_gm_contract(original, "cython", target)
    assert not requires_fresh_auto_gm(result)
    assert requires_fresh_auto_gm(original)


def test_new_compilation_replaces_an_older_private_schema() -> None:
    original = _function().with_attr(AUTO_GM_FRESH_CONTRACT_ATTR, tvm.tir.StringImm("old"))
    prepared = prepare_auto_gm_contract(original, "cython", "ascendc")
    assert requires_fresh_auto_gm(prepared)
    assert original.attrs[AUTO_GM_FRESH_CONTRACT_ATTR].value == "old"


@pytest.mark.parametrize("function", [None, object(), SimpleNamespace(attrs={})])
def test_unmarked_cache_callers_keep_the_legacy_path(function: object) -> None:
    assert not requires_fresh_auto_gm(function)
    # Legacy cache validation must not start imposing allocation promises.
    validate_cached_auto_gm_contract(function, "dlpack", None, -1)


@pytest.mark.parametrize(
    "marker",
    [
        tvm.tir.StringImm("cython-auto-gm-v2"),
        tvm.tir.IntImm("int32", 1),
        AUTO_GM_FRESH_CONTRACT_VERSION,
    ],
)
def test_unknown_or_malformed_schema_is_not_a_legacy_cache_entry(marker: object) -> None:
    function = _function().with_attr(AUTO_GM_FRESH_CONTRACT_ATTR, marker)
    with pytest.raises(ValueError, match="Invalid Ascend auto-GM fresh contract"):
        requires_fresh_auto_gm(function)
    with pytest.raises(ValueError, match="Invalid Ascend auto-GM fresh contract"):
        validate_cached_auto_gm_contract(function, "cython", [], 0)


def test_a_present_null_marker_is_not_an_absent_contract() -> None:
    function = SimpleNamespace(attrs={AUTO_GM_FRESH_CONTRACT_ATTR: None})
    with pytest.raises(ValueError, match="Invalid Ascend auto-GM fresh contract"):
        requires_fresh_auto_gm(function)


@pytest.mark.parametrize("indices,param_count", [([], 0), ([], 3), ([0, 2], 3)])
def test_valid_lowered_workspace_indices_are_preserved(indices: list[int], param_count: int) -> None:
    before = indices.copy()
    validate_cached_auto_gm_contract(_prepared(), "cython", indices, param_count)
    assert indices == before


@pytest.mark.parametrize("backend", ["ctypes", "dlpack", "unknown"])
def test_cached_fresh_binary_requires_the_same_allocator(backend: str) -> None:
    with pytest.raises(ValueError, match="requires the Cython allocator"):
        validate_cached_auto_gm_contract(_prepared(), backend, [], 0)


@pytest.mark.parametrize("indices", [None, (), (0,), "[]", {0}])
def test_missing_or_nonlist_workspace_metadata_fails_closed(indices: object) -> None:
    with pytest.raises(ValueError, match="requires an explicit index list"):
        validate_cached_auto_gm_contract(_prepared(), "cython", indices, 3)


@pytest.mark.parametrize("indices", [[-1], [3], [True], [False], ["0"], [tvm.tir.IntImm("int32", 0)]])
def test_workspace_indices_are_strict_integers_in_the_lowered_parameter_range(
    indices: list[object],
) -> None:
    with pytest.raises(ValueError, match="invalid workspace index"):
        validate_cached_auto_gm_contract(_prepared(), "cython", indices, 3)


def test_duplicate_workspace_indices_fail_closed() -> None:
    with pytest.raises(ValueError, match="duplicate workspace index"):
        validate_cached_auto_gm_contract(_prepared(), "cython", [1, 1], 3)


@pytest.mark.parametrize("param_count", [-1, True, 1.0, None])
def test_invalid_lowered_parameter_counts_fail_closed(param_count: object) -> None:
    with pytest.raises(ValueError, match="valid parameter count"):
        validate_cached_auto_gm_contract(_prepared(), "cython", [], param_count)


def test_manifest_binds_a_copied_index_table_to_the_marked_source() -> None:
    function = _prepared()
    indices = [0, 2]
    manifest = build_cached_auto_gm_manifest(function, "cython", indices, 3)
    assert AUTO_GM_MANIFEST_PATH == "ascend_auto_gm_contract.json"
    assert manifest == {
        "schema": 1,
        "param_count": 3,
        "auto_gm_indices": [0, 2],
        "source_hash": hashlib.sha256(function.script().encode("utf-8")).hexdigest(),
    }
    assert manifest["auto_gm_indices"] is not indices
    indices.append(1)
    assert manifest["auto_gm_indices"] == [0, 2]


@pytest.mark.parametrize("function", [None, object(), SimpleNamespace(attrs={})])
def test_unmarked_functions_do_not_require_a_manifest(function: object) -> None:
    assert build_cached_auto_gm_manifest(function, "dlpack", None, -1) is None
    validate_cached_auto_gm_manifest(function, "dlpack", None, -1, None)


@pytest.mark.parametrize("indices,param_count", [([], 0), ([], 3), ([0, 2], 3)])
def test_valid_manifests_accept_empty_or_populated_workspace_tables(indices: list[int], param_count: int) -> None:
    function = _prepared()
    manifest = build_cached_auto_gm_manifest(function, "cython", indices, param_count)
    validate_cached_auto_gm_manifest(function, "cython", indices, param_count, manifest)


def test_manifest_count_describes_lowered_parameters_including_added_workspace() -> None:
    original = tvm.tir.PrimFunc([tvm.tir.Var(f"arg{i}", "handle") for i in range(3)], tvm.tir.Evaluate(0))
    function = prepare_auto_gm_contract(original, "cython", "ascendc")
    manifest = build_cached_auto_gm_manifest(function, "cython", [3], 4)
    validate_cached_auto_gm_manifest(function, "cython", [3], 4, manifest)
    assert len(function.params) == 3
    assert manifest["param_count"] == 4


@pytest.mark.parametrize("manifest", [None, [], {}, "{}", False])
def test_missing_or_nonobject_manifests_fail_closed(manifest: object) -> None:
    with pytest.raises(ValueError, match="complete cache manifest"):
        validate_cached_auto_gm_manifest(_prepared(), "cython", [], 0, manifest)


@pytest.mark.parametrize("change", ["missing_field", "extra_field"])
def test_manifest_fields_are_versioned_as_a_complete_schema(change: str) -> None:
    function = _prepared()
    manifest = build_cached_auto_gm_manifest(function, "cython", [], 0)
    if change == "missing_field":
        del manifest["source_hash"]
    else:
        manifest["unknown_field"] = 1
    with pytest.raises(ValueError, match="complete cache manifest"):
        validate_cached_auto_gm_manifest(function, "cython", [], 0, manifest)


@pytest.mark.parametrize("schema", [True, 1.0, 0, 2, tvm.tir.IntImm("int32", 1)])
def test_manifest_schema_is_an_exact_supported_integer(schema: object) -> None:
    function = _prepared()
    manifest = build_cached_auto_gm_manifest(function, "cython", [], 0)
    manifest["schema"] = schema
    with pytest.raises(ValueError, match="unsupported schema"):
        validate_cached_auto_gm_manifest(function, "cython", [], 0, manifest)


@pytest.mark.parametrize("param_count", [-1, True, 1.0, None])
def test_manifest_parameter_count_is_independently_validated(param_count: object) -> None:
    function = _prepared()
    manifest = build_cached_auto_gm_manifest(function, "cython", [1], 2)
    manifest["param_count"] = param_count
    with pytest.raises(ValueError, match="valid parameter count"):
        validate_cached_auto_gm_manifest(function, "cython", [1], 2, manifest)


def test_manifest_cannot_change_the_lowered_parameter_count() -> None:
    function = _prepared()
    manifest = build_cached_auto_gm_manifest(function, "cython", [1], 3)
    with pytest.raises(ValueError, match="parameter count does not match"):
        validate_cached_auto_gm_manifest(function, "cython", [1], 2, manifest)


@pytest.mark.parametrize("indices", [None, {}, [True], [False], ["1"], [2], [1, 1]])
def test_manifest_index_table_is_validated_before_python_equality(indices: object) -> None:
    function = _prepared()
    manifest = build_cached_auto_gm_manifest(function, "cython", [1], 2)
    manifest["auto_gm_indices"] = indices
    with pytest.raises(ValueError):
        validate_cached_auto_gm_manifest(function, "cython", [1], 2, manifest)


@pytest.mark.parametrize("indices", [[], [0]])
def test_manifest_cannot_replace_the_actual_workspace_table(indices: list[int]) -> None:
    function = _prepared()
    manifest = build_cached_auto_gm_manifest(function, "cython", [1], 2)
    with pytest.raises(ValueError, match="workspace indices do not match"):
        validate_cached_auto_gm_manifest(function, "cython", indices, 2, manifest)


@pytest.mark.parametrize("source_hash", [None, False, 123, "", "0" * 64])
def test_manifest_requires_the_matching_source_hash(source_hash: object) -> None:
    function = _prepared()
    manifest = build_cached_auto_gm_manifest(function, "cython", [], 0)
    manifest["source_hash"] = source_hash
    with pytest.raises(ValueError, match="source hash does not match"):
        validate_cached_auto_gm_manifest(function, "cython", [], 0, manifest)


def test_another_marked_source_cannot_reuse_the_manifest() -> None:
    function = _prepared()
    manifest = build_cached_auto_gm_manifest(function, "cython", [], 0)
    changed = function.with_attr("existing_marker", 8)
    with pytest.raises(ValueError, match="source hash does not match"):
        validate_cached_auto_gm_manifest(changed, "cython", [], 0, manifest)


def test_manifest_does_not_relax_backend_or_actual_metadata_validation() -> None:
    function = _prepared()
    manifest = build_cached_auto_gm_manifest(function, "cython", [1], 2)
    with pytest.raises(ValueError, match="requires the Cython allocator"):
        validate_cached_auto_gm_manifest(function, "ctypes", [1], 2, manifest)
    with pytest.raises(ValueError, match="invalid workspace index"):
        validate_cached_auto_gm_manifest(function, "cython", [True], 2, manifest)
    with pytest.raises(ValueError, match="requires an explicit index list"):
        build_cached_auto_gm_manifest(function, "cython", None, 2)


def _allocator_function():
    buffers = [tvm.tir.decl_buffer((4 + i,), name=f"arg{i}", dtype="float32") for i in range(3)]
    params = [tvm.tir.Var(f"arg{i}_handle", "handle") for i in range(3)]
    return tvm.tir.PrimFunc(
        params,
        tvm.tir.Evaluate(0),
        buffer_map=dict(zip(params, buffers)),
    ).with_attr("global_symbol", "allocator_contract")


def _append_workspace(function):
    param = tvm.tir.Var("auto_workspace", "handle")
    buffer = tvm.tir.decl_buffer((7,), name="auto_workspace", dtype="float32")
    return tvm.tir.PrimFunc(
        list(function.params) + [param],
        function.body,
        buffer_map={**dict(function.buffer_map), param: buffer},
        attrs=function.attrs,
    ).with_attr("auto_gm_indices", [3])


def _runtime_params(function):
    from tilelang.engine.param import KernelParam

    return [KernelParam.from_buffer(function.buffer_map[param]) for param in function.params]


def test_v2_snapshots_raw_roles_and_normalizes_against_appended_final_parameters():
    from tilelang.jit.ascend_runtime_contract import (
        CYTHON_RESULT_INDICES_ATTR,
        bind_final_auto_gm_contract,
        has_cython_allocator_fresh_contract,
        validate_fresh_runtime_signature,
    )

    results, workspaces = [-2, -2, True], [-1, True]
    original = _allocator_function()
    prepared = prepare_auto_gm_contract(original, "cython", "ascendc", results, workspaces)
    results.clear()
    workspaces.clear()
    assert has_cython_allocator_fresh_contract(prepared)
    assert [int(index) for index in prepared.attrs[CYTHON_RESULT_INDICES_ATTR]] == [-2, -2, 1]
    assert "tir.noalias" not in prepared.attrs
    assert AUTO_GM_FRESH_CONTRACT_ATTR not in original.attrs
    lowered = bind_final_auto_gm_contract(_append_workspace(prepared))
    validate_cached_auto_gm_contract(prepared, "cython", [3], 4, [2, 2, 1], [3, 1])
    validate_fresh_runtime_signature(prepared, lowered, _runtime_params(lowered))
    with pytest.raises(ValueError, match="actual allocator roles"):
        validate_cached_auto_gm_contract(prepared, "cython", [3], 4, [1, 1, 1], [2, 1])


@pytest.mark.parametrize("change", ["reorder", "insert", "remove", "replace"])
def test_v2_refuses_original_parameter_identity_changes_before_binding(change):
    from tilelang.jit.ascend_runtime_contract import bind_final_auto_gm_contract

    prepared = prepare_auto_gm_contract(_allocator_function(), "cython", "ascendc", [-1])
    params = list(prepared.params)
    if change == "reorder":
        params[0], params[1] = params[1], params[0]
    elif change == "insert":
        params.insert(0, tvm.tir.Var("inserted", "handle"))
    elif change == "remove":
        params.pop(0)
    else:
        params[0] = tvm.tir.Var(params[0].name, "handle")
    changed = tvm.tir.PrimFunc(params, prepared.body, attrs=prepared.attrs)
    with pytest.raises(ValueError, match="signature changed"):
        bind_final_auto_gm_contract(changed)


@pytest.mark.parametrize("change", ["append", "reorder", "remove"])
def test_v2_refuses_final_signature_changes_after_binding(change):
    from tilelang.jit.ascend_runtime_contract import bind_final_auto_gm_contract

    prepared = prepare_auto_gm_contract(_allocator_function(), "cython", "ascendc", [-1])
    bound = bind_final_auto_gm_contract(_append_workspace(prepared))
    params = list(bound.params)
    if change == "append":
        params.append(tvm.tir.Var("later_workspace", "handle"))
    elif change == "reorder":
        params[2], params[3] = params[3], params[2]
    else:
        params.pop()
    changed = tvm.tir.PrimFunc(params, bound.body, attrs=bound.attrs)
    with pytest.raises(ValueError, match="signature changed"):
        bind_final_auto_gm_contract(changed)


@pytest.mark.parametrize("change", ["unbound", "shape", "count", "recipe", "source"])
def test_v2_checks_runtime_params_and_source_provenance_after_lowering(change):
    from tilelang.jit.ascend_runtime_contract import (
        CYTHON_FINAL_SIGNATURE_ATTR,
        CYTHON_INPUT_SIGNATURE_ATTR,
        CYTHON_RESULT_INDICES_ATTR,
        bind_final_auto_gm_contract,
        validate_fresh_runtime_signature,
    )

    prepared = prepare_auto_gm_contract(_allocator_function(), "cython", "ascendc", [-1])
    lowered = bind_final_auto_gm_contract(_append_workspace(prepared))
    params = _runtime_params(lowered)
    if change == "unbound":
        lowered = lowered.without_attr(CYTHON_FINAL_SIGNATURE_ATTR)
    elif change == "shape":
        params[0].shape = [999]
    elif change == "count":
        params.pop()
    elif change == "recipe":
        lowered = lowered.with_attr(CYTHON_RESULT_INDICES_ATTR, [tvm.tir.IntImm("int64", 2)])
    else:
        # A cloned source with same textual names has different Var provenance.
        other = prepare_auto_gm_contract(_allocator_function(), "cython", "ascendc", [-1])
        prepared = prepared.with_attr(CYTHON_INPUT_SIGNATURE_ATTR, other.params)
    with pytest.raises(ValueError):
        validate_fresh_runtime_signature(prepared, lowered, params)


@pytest.mark.parametrize(
    "change",
    [None, "result", "workspace", "union", "boolean_union", "shape", "dtype", "count"],
)
def test_v2_cache_manifest_binds_actual_roles_union_and_runtime_parameters(change):
    from copy import deepcopy
    from tilelang.jit.ascend_runtime_contract import bind_final_auto_gm_contract

    prepared = prepare_auto_gm_contract(
        _allocator_function(),
        "cython",
        "ascendc",
        [-2, -2],
        [-1, -2],
    )
    lowered = bind_final_auto_gm_contract(_append_workspace(prepared))
    params = _runtime_params(lowered)
    manifest = build_cached_auto_gm_manifest(
        prepared,
        "cython",
        [3],
        4,
        out_idx=[2, 2],
        workspace_idx=[3, 2],
        runtime_params=params,
    )
    assert manifest["schema"] == 3
    assert manifest["allocated_indices"] == [2, 3]
    manifest = deepcopy(manifest)
    if change == "result":
        manifest["result_indices"] = [1, 1]
    elif change == "workspace":
        manifest["workspace_indices"] = [1, 2]
    elif change == "union":
        manifest["allocated_indices"] = [1, 2, 3]
    elif change == "boolean_union":
        manifest["allocated_indices"] = [True, 3]
    elif change == "shape":
        manifest["runtime_signature"][0]["shape"] = ["999"]
    elif change == "dtype":
        manifest["runtime_signature"][0]["dtype"] = "torch.float16"
    elif change == "count":
        params.pop()
    if change is None:
        validate_cached_auto_gm_manifest(
            prepared,
            "cython",
            [3],
            4,
            manifest,
            out_idx=[2, 2],
            workspace_idx=[3, 2],
            runtime_params=params,
        )
    else:
        with pytest.raises(ValueError):
            validate_cached_auto_gm_manifest(
                prepared,
                "cython",
                [3],
                4,
                manifest,
                out_idx=[2, 2],
                workspace_idx=[3, 2],
                runtime_params=params,
            )


def test_cython_wrapper_allocates_fresh_union_on_every_call_even_with_full_args(monkeypatch):
    import torch
    from tilelang.engine.param import KernelParam
    from tilelang.jit.adapter.cython.adapter import CythonKernelWrapper

    calls = []
    allocations = []
    real_empty = torch.empty

    def allocate(*args, **kwargs):
        tensor = real_empty(*args, **kwargs)
        allocations.append(tensor)
        return tensor

    monkeypatch.setattr(torch, "empty", allocate)
    # Cython never invokes this captured getter when CPU tensors supply a device.
    monkeypatch.setattr(torch.npu, "current_device", lambda: 0)
    results, workspaces, auto = [3, 3], [2, 3], [2]
    params = [KernelParam(torch.float32, [4])] * 3 + [KernelParam(torch.float16, [2])]
    wrapper = CythonKernelWrapper(
        results,
        workspaces,
        auto,
        params,
        SimpleNamespace(call=lambda *args: calls.append(args)),
    )
    for setter in (
        wrapper.set_dynamic_symbolic_map,
        wrapper.set_buffer_dtype_map,
        wrapper.set_static_shape_map,
        wrapper.set_buffer_device_map,
        wrapper.set_ptr_map,
    ):
        setter({})
    results.clear()
    workspaces.clear()
    auto.clear()
    source = torch.ones(4)
    supplied = torch.ones(2, dtype=torch.float16)
    first = wrapper.forward([source, source, source, supplied], stream=0)
    second = wrapper.forward([source, source, source, supplied], stream=0)
    assert first[0] is first[1]
    assert second[0] is second[1]
    assert first[0].dtype == torch.float16
    assert first[0].data_ptr() != second[0].data_ptr()
    assert len(allocations) == 4  # Two distinct fresh slots, each allocated once per call.
    for call, allocated in zip(calls, (allocations[:2], allocations[2:])):
        assert call[0].value == call[1].value == source.data_ptr()
        assert call[2].value == allocated[0].data_ptr()
        assert call[3].value == allocated[1].data_ptr()
        assert call[3].value != supplied.data_ptr()


def test_cython_database_adapter_snapshots_caller_roles_without_inplace_normalization(monkeypatch):
    import torch
    from tilelang.jit.adapter.cython import adapter as cython_adapter
    from tilelang.jit.ascend_runtime_contract import bind_final_auto_gm_contract

    function = prepare_auto_gm_contract(_allocator_function(), "cython", "ascendc", [-1, -1], [-1])
    lowered = bind_final_auto_gm_contract(_append_workspace(function))
    params = _runtime_params(lowered)
    monkeypatch.setattr(torch.npu, "current_device", lambda: 0)
    for name in ("dynamic_symbolic", "buffer_dtype", "static_shape", "ptr_map", "buffer_device"):
        monkeypatch.setattr(cython_adapter.CythonKernelAdapter, f"_process_{name}", lambda self: {})
    library = SimpleNamespace(call=lambda *args: None)
    monkeypatch.setattr(
        cython_adapter,
        "LibraryGenerator",
        lambda *args: SimpleNamespace(load_lib=lambda **kwargs: library),
    )
    results, workspaces, auto = [-1, -1], [-1], [3]
    adapter = cython_adapter.CythonKernelAdapter.from_database(
        params,
        results,
        workspaces,
        auto,
        "ascendc",
        "A3",
        function,
        "",
        "/unused.so",
    )
    assert results == [-1, -1] and workspaces == [-1] and auto == [3]
    assert adapter.result_idx == [3, 3] and adapter.workspace_idx == [3]
    results.clear()
    workspaces.clear()
    auto.clear()
    outputs = adapter(torch.ones(4), torch.ones(5), torch.ones(6), stream=0)
    assert outputs[0] is outputs[1]
    assert outputs[0].shape == (7,)


@pytest.mark.parametrize("target", ["pto", "llvm", tvm.target.Target("llvm")])
def test_v2_database_adapter_refuses_unsupported_target_before_loading_library(monkeypatch, target):
    from tilelang.jit.adapter.cython import adapter as cython_adapter
    from tilelang.jit.ascend_runtime_contract import bind_final_auto_gm_contract

    function = prepare_auto_gm_contract(_allocator_function(), "cython", "ascendc", [-1])
    lowered = bind_final_auto_gm_contract(_append_workspace(function))

    def unexpected_load(*args, **kwargs):
        pytest.fail("incompatible allocator target reached the library loader")

    monkeypatch.setattr(cython_adapter, "LibraryGenerator", unexpected_load)
    with pytest.raises(ValueError, match="AscendC target"):
        cython_adapter.CythonKernelAdapter.from_database(
            _runtime_params(lowered),
            [-1],
            None,
            [3],
            target,
            "A3",
            function,
            "",
            "/unused.so",
        )


class _MembershipIndex(int):
    """An integer index whose equality does not describe its integer value."""

    def __eq__(self, other):
        return other == 0


@pytest.mark.parametrize("role_name", ["result", "workspace", "auto_gm"])
@pytest.mark.parametrize("kind", ["custom_equality", "mutable_scalar"])
def test_cython_wrapper_freezes_integer_value_instead_of_mutable_membership(
    monkeypatch,
    role_name,
    kind,
):
    import torch
    from tilelang.jit.adapter.cython.adapter import CythonKernelWrapper

    monkeypatch.setattr(torch.npu, "current_device", lambda: 0)
    role = _MembershipIndex(2) if kind == "custom_equality" else torch.tensor(2)
    recipe = [role]
    raw_roles = {"result": [], "workspace": [], "auto_gm": []}
    raw_roles[role_name] = recipe
    function = prepare_auto_gm_contract(
        _allocator_function(),
        "cython",
        "ascendc",
        raw_roles["result"],
        raw_roles["workspace"],
    )
    canonical = {"result": [], "workspace": [], "auto_gm": []}
    canonical[role_name] = [2]
    validate_cached_auto_gm_contract(
        function,
        "cython",
        canonical["auto_gm"],
        3,
        canonical["result"],
        canonical["workspace"],
    )
    calls = []
    wrapper = CythonKernelWrapper(
        raw_roles["result"],
        raw_roles["workspace"],
        raw_roles["auto_gm"],
        _runtime_params(function),
        SimpleNamespace(call=lambda *args: calls.append(args)),
    )
    for setter in (
        wrapper.set_dynamic_symbolic_map,
        wrapper.set_buffer_dtype_map,
        wrapper.set_static_shape_map,
        wrapper.set_buffer_device_map,
        wrapper.set_ptr_map,
    ):
        setter({})
    if kind == "mutable_scalar":
        role.fill_(0)
    recipe[:] = [0]
    source = torch.ones(4)
    other = torch.ones(5)
    supplied = torch.ones(6)
    output = wrapper.forward([source, other, supplied], stream=0)
    assert calls[0][0].value == source.data_ptr()
    assert calls[0][1].value == other.data_ptr()
    assert calls[0][2].value not in (source.data_ptr(), other.data_ptr(), supplied.data_ptr())
    if role_name == "result":
        assert output.data_ptr() == calls[0][2].value


@pytest.mark.parametrize("kind", ["custom_equality", "mutable_scalar"])
def test_cython_database_adapter_uses_same_canonical_integer_recipe_as_compiler(
    monkeypatch,
    kind,
):
    import torch
    from tilelang.jit.adapter.cython import adapter as cython_adapter

    monkeypatch.setattr(torch.npu, "current_device", lambda: 0)
    role = _MembershipIndex(2) if kind == "custom_equality" else torch.tensor(2)
    results = [role, role]
    workspaces = [role]
    auto = [role]
    function = prepare_auto_gm_contract(
        _allocator_function(),
        "cython",
        "ascendc",
        results,
        workspaces,
    )
    for name in ("dynamic_symbolic", "buffer_dtype", "static_shape", "ptr_map", "buffer_device"):
        monkeypatch.setattr(cython_adapter.CythonKernelAdapter, f"_process_{name}", lambda self: {})
    calls = []
    library = SimpleNamespace(call=lambda *args: calls.append(args))
    monkeypatch.setattr(
        cython_adapter,
        "LibraryGenerator",
        lambda *args: SimpleNamespace(load_lib=lambda **kwargs: library),
    )
    adapter = cython_adapter.CythonKernelAdapter.from_database(
        _runtime_params(function),
        results,
        workspaces,
        auto,
        "ascendc",
        "A3",
        function,
        "",
        "/unused.so",
    )
    assert adapter.result_idx == [2, 2]
    assert adapter.workspace_idx == adapter.auto_gm_idx == [2]
    assert all(type(index) is int for index in adapter.result_idx + adapter.workspace_idx)
    if kind == "mutable_scalar":
        role.fill_(0)
    results[:] = workspaces[:] = auto[:] = [0]
    source, other, supplied = torch.ones(4), torch.ones(5), torch.ones(6)
    output = adapter(source, other, supplied, stream=0)
    assert output[0] is output[1]
    assert calls[0][0].value == source.data_ptr()
    assert calls[0][1].value == other.data_ptr()
    assert calls[0][2].value == output[0].data_ptr() != supplied.data_ptr()


def test_cython_database_adapter_rejects_index_mutation_since_compilation(monkeypatch):
    import torch
    from tilelang.jit.adapter.cython import adapter as cython_adapter

    role = torch.tensor(2)
    results = [role]
    function = prepare_auto_gm_contract(
        _allocator_function(),
        "cython",
        "ascendc",
        results,
    )
    role.fill_(0)

    def unexpected_load(*args, **kwargs):
        pytest.fail("mutated allocator recipe reached the library loader")

    monkeypatch.setattr(cython_adapter, "LibraryGenerator", unexpected_load)
    with pytest.raises(ValueError, match="actual allocator roles"):
        cython_adapter.CythonKernelAdapter.from_database(
            _runtime_params(function),
            results,
            None,
            [],
            "ascendc",
            "A3",
            function,
            "",
            "/unused.so",
        )
