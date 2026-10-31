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
