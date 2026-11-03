# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Private allocation promises made by the Ascend Cython JIT entry.

This contract describes real Cython allocations: compiler-added workspaces
and, in v2, explicit result/workspace slots in the final lowered signature.
It does not declare user parameters mutually disjoint or add ``tir.noalias``.
Direct lowering and other adapters do not acquire this promise implicitly.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tvm.target import Target
    from tvm.tir import PrimFunc


AUTO_GM_FRESH_CONTRACT_ATTR = "tl.ascend_auto_gm_fresh_contract"
AUTO_GM_FRESH_CONTRACT_VERSION = "cython-auto-gm-v1"
CYTHON_ALLOCATOR_FRESH_CONTRACT_VERSION = "cython-auto-gm-v2"
CYTHON_RESULT_INDICES_ATTR = "tl.ascend_jit_result_indices"
CYTHON_WORKSPACE_INDICES_ATTR = "tl.ascend_jit_workspace_indices"
CYTHON_INPUT_SIGNATURE_ATTR = "tl.ascend_jit_input_signature"
CYTHON_FINAL_SIGNATURE_ATTR = "tl.ascend_jit_final_signature"
AUTO_GM_MANIFEST_PATH = "ascend_auto_gm_contract.json"


def _allocation_recipe(indices: Any) -> list[int]:
    """Snapshot the index membership used by the existing Cython allocator.

    Negative indices remain relative to the final, lowered parameter list.
    Duplicates, overlapping roles and bool's integer-index meaning are kept.
    Every value becomes a built-in int, so later equality overrides or mutation
    of an input index object cannot change the executing allocation membership.
    """
    import operator

    if indices is None:
        return []
    if isinstance(indices, int):
        values = [indices]
    elif isinstance(indices, list):
        values = indices
    else:
        raise TypeError("allocation indices require an integer or list")
    result = [int(operator.index(value)) for value in values]
    if any(value < -(1 << 63) or value >= 1 << 63 for value in result):
        raise OverflowError("allocation index cannot be represented in the private contract")
    return result


def _normalized_recipe(indices: Any, param_count: int) -> list[int]:
    values = _allocation_recipe(indices)
    if any(value < -param_count or value >= param_count for value in values):
        raise ValueError(
            "Ascend Cython allocation contract has an index outside the final signature",
        )
    return [value + param_count if value < 0 else value for value in values]


def _v2_recipe(func: Any) -> tuple[list[int], list[int]]:
    from tilelang import tvm

    attrs = func.attrs
    result = []
    for name in (CYTHON_RESULT_INDICES_ATTR, CYTHON_WORKSPACE_INDICES_ATTR):
        values = attrs.get(name, None)
        if not isinstance(values, tvm.ir.container.Array) or any(
            not isinstance(value, tvm.tir.IntImm) or value.dtype != "int64" for value in values
        ):
            raise ValueError(f"Invalid Ascend Cython allocation recipe: {name}")
        result.append([int(value) for value in values])
    signature = attrs.get(CYTHON_INPUT_SIGNATURE_ATTR, None)
    params = getattr(func, "params", ())
    if (
        not isinstance(signature, tvm.ir.container.Array)
        or len(signature) > len(params)
        or any(not isinstance(param, tvm.tir.Var) for param in signature)
        or any(not param.same_as(params[i]) for i, param in enumerate(signature))
    ):
        raise ValueError(
            "Ascend Cython allocation contract input signature changed before lowering",
        )
    return result[0], result[1]


def has_cython_allocator_fresh_contract(func: Any) -> bool:
    if not requires_fresh_auto_gm(func):
        return False
    return func.attrs[AUTO_GM_FRESH_CONTRACT_ATTR].value == CYTHON_ALLOCATOR_FRESH_CONTRACT_VERSION


def bind_final_auto_gm_contract(func: Any) -> Any:
    """Bind fresh roles to the exact signature seen by synchronization.

    Lowering may append compiler workspaces, but may not insert, delete or
    reorder the original ABI parameters behind the allocator's back.
    """
    if not has_cython_allocator_fresh_contract(func):
        return func
    from tilelang import tvm

    _v2_recipe(func)
    if CYTHON_FINAL_SIGNATURE_ATTR in func.attrs:
        signature = func.attrs[CYTHON_FINAL_SIGNATURE_ATTR]
        if (
            not isinstance(signature, tvm.ir.container.Array)
            or len(signature) != len(func.params)
            or any(not isinstance(param, tvm.tir.Var) for param in signature)
            or any(not param.same_as(func.params[i]) for i, param in enumerate(signature))
        ):
            raise ValueError(
                "Ascend Cython allocation contract final signature changed after binding",
            )
        return func
    return func.with_attr(CYTHON_FINAL_SIGNATURE_ATTR, func.params)


def bind_final_auto_gm_contracts(mod: Any) -> Any:
    from tilelang import tvm

    changed = {}
    for name, func in mod.functions.items():
        if isinstance(func, tvm.tir.PrimFunc):
            bound = bind_final_auto_gm_contract(func)
            if not bound.same_as(func):
                changed[name] = bound
    if not changed:
        return mod
    result = tvm.IRModule(mod.functions, mod.type_definitions, mod.attrs)
    for name, func in changed.items():
        result.update_func(name, func)
    return result


def sync_engine_cache_abi(
    pass_configs: dict[str, Any] | None = None,
    target: str | Target | None = None,
) -> str | None:
    """Return the loaded synchronization ABI for an actual Ascend target.

    The native version is authoritative: package/HEAD versions do not change
    after an uncommitted compiler rebuild. Every semantic engine, importer or
    shared operation-policy change must bump it before reusing cached code.
    Callers without a target retain the historical explicit-upstream behavior;
    all compiler cache and database paths provide their actual target.
    """
    if target is None:
        enabled = bool(pass_configs and pass_configs.get("tl.ascend_sync_backend") == "upstream")
    elif isinstance(target, str):
        # In this compiler, auto selects AscendC. Host-side Target objects use
        # kind=llvm with model=ascendc/pto, so do not gate on kind alone.
        enabled = target in ("auto", "ascendc", "pto")
        if not enabled and "-model" in target:
            from tvm.target import Target

            enabled = getattr(Target(target), "model", None) in ("auto", "ascendc", "pto")
    else:
        enabled = getattr(target, "model", None) in ("auto", "ascendc", "pto")
    if not enabled:
        return None
    from tilelang import tvm

    query = tvm.get_global_func("tl.ascend_sync.EngineCacheABI", allow_missing=True)
    if query is None:
        raise RuntimeError(
            "The Ascend synchronization cache ABI is unavailable; "
            "rebuild the native TileLang libraries before compiling or loading a kernel"
        )
    version = query()
    if not isinstance(version, str) or not version:
        raise RuntimeError("Invalid Ascend synchronization cache ABI; rebuild TileLang")
    return version


def _ascend_model(target: str | Target) -> bool:
    # Host-side lowering uses kind=llvm with an Ascend model. The kind cannot
    # be used to decide whether this JIT invocation has an Ascend allocator.
    model = target if isinstance(target, str) else getattr(target, "model", None)
    return model in ("ascendc", "auto")


def prepare_auto_gm_contract(
    func: PrimFunc,
    execution_backend: str,
    target: str | Target,
    out_idx: Any = None,
    workspace_idx: Any = None,
) -> PrimFunc:
    """Return an attributed copy for a new compilation, keeping ``func`` intact.

    Recompilation first discards a previous private allocation promise. Only
    the selected Cython Ascend JIT path can replace it with the current schema.
    This helper must run before cache-key generation as well as direct JIT
    compilation, so the contract version participates in artifact identity.
    """
    from tilelang import tvm

    if not isinstance(func, tvm.tir.PrimFunc):
        raise TypeError("Ascend allocation contracts require a PrimFunc")
    prepared = func
    for name in (
        AUTO_GM_FRESH_CONTRACT_ATTR,
        CYTHON_RESULT_INDICES_ATTR,
        CYTHON_WORKSPACE_INDICES_ATTR,
        CYTHON_INPUT_SIGNATURE_ATTR,
        CYTHON_FINAL_SIGNATURE_ATTR,
    ):
        prepared = prepared.without_attr(name)
    if execution_backend == "cython" and _ascend_model(target):
        version = AUTO_GM_FRESH_CONTRACT_VERSION
        try:
            results, workspaces = _allocation_recipe(out_idx), _allocation_recipe(workspace_idx)
        except (TypeError, OverflowError):
            # Unsupported public recipes retain their existing adapter errors;
            # they do not acquire additional compiler alias capabilities.
            results, workspaces = [], []
        if results or workspaces:
            version = CYTHON_ALLOCATOR_FRESH_CONTRACT_VERSION
            prepared = (
                prepared.with_attr(
                    CYTHON_RESULT_INDICES_ATTR,
                    [tvm.tir.IntImm("int64", value) for value in results],
                )
                .with_attr(
                    CYTHON_WORKSPACE_INDICES_ATTR,
                    [tvm.tir.IntImm("int64", value) for value in workspaces],
                )
                .with_attr(CYTHON_INPUT_SIGNATURE_ATTR, func.params)
            )
        prepared = prepared.with_attr(AUTO_GM_FRESH_CONTRACT_ATTR, tvm.tir.StringImm(version))
    return prepared


def requires_fresh_auto_gm(func: Any) -> bool:
    """Validate the private schema, returning false for an unmarked function.

    Cache callers may have no function or a legacy mock without attributes.
    A present marker always requires validation; unknown or malformed markers
    must not silently become the old, unconstrained allocation path.
    """
    attrs = getattr(func, "attrs", None)
    if attrs is None or AUTO_GM_FRESH_CONTRACT_ATTR not in attrs:
        return False
    from tilelang import tvm

    contract = attrs[AUTO_GM_FRESH_CONTRACT_ATTR]
    if not isinstance(contract, tvm.tir.StringImm) or contract.value not in (
        AUTO_GM_FRESH_CONTRACT_VERSION,
        CYTHON_ALLOCATOR_FRESH_CONTRACT_VERSION,
    ):
        raise ValueError("Invalid Ascend auto-GM fresh contract schema")
    if contract.value == CYTHON_ALLOCATOR_FRESH_CONTRACT_VERSION:
        try:
            _v2_recipe(func)
        except (ValueError, AttributeError) as error:
            raise ValueError(f"Invalid Ascend auto-GM fresh contract: {error}") from error
    return True


def fresh_allocation_contract_matches(func: Any, compiled_func: Any) -> bool:
    """Do not publish a newer allocator capability under an unprepared key."""
    try:
        if not requires_fresh_auto_gm(compiled_func):
            return True
        if not requires_fresh_auto_gm(func):
            return False
        source_version = func.attrs[AUTO_GM_FRESH_CONTRACT_ATTR].value
        compiled_version = compiled_func.attrs[AUTO_GM_FRESH_CONTRACT_ATTR].value
        if source_version != compiled_version:
            return False
        if has_cython_allocator_fresh_contract(func):
            original = func.attrs[CYTHON_INPUT_SIGNATURE_ATTR]
            compiled = compiled_func.attrs[CYTHON_INPUT_SIGNATURE_ATTR]
            return (
                _v2_recipe(func) == _v2_recipe(compiled_func)
                and len(original) == len(compiled)
                and all(param.same_as(compiled[i]) for i, param in enumerate(original))
            )
        return True
    except (ValueError, AttributeError):
        return False


def validate_cached_auto_gm_contract(
    func: Any,
    execution_backend: str,
    auto_gm_idx: Any,
    param_count: int,
    out_idx: Any = None,
    workspace_idx: Any = None,
    target: str | Target | None = None,
) -> None:
    """Reject cache/database reconstruction that cannot honor the promise.

    ``param_count`` describes the lowered artifact, whose compiler-generated
    workspace parameters may not exist in the original input PrimFunc. An
    empty list is valid; missing metadata is not equivalent to that list.
    Caller code must turn cache validation failure into a cache miss/rebuild,
    or reject an incompatible explicit database load before kernel execution.
    """
    if not requires_fresh_auto_gm(func):
        return
    if execution_backend != "cython":
        raise ValueError("Ascend auto-GM fresh contract requires the Cython allocator")
    _validate_indices(auto_gm_idx, param_count)
    if has_cython_allocator_fresh_contract(func):
        if target is not None and not _ascend_model(target):
            raise ValueError("Ascend Cython allocator contract requires an AscendC target")
        results, workspaces = _v2_recipe(func)
        for expected, actual in ((results, out_idx), (workspaces, workspace_idx)):
            if _normalized_recipe(expected, param_count) != _normalized_recipe(actual, param_count):
                raise ValueError(
                    "Ascend Cython allocation recipe disagrees with the actual allocator roles",
                )


def _validate_indices(auto_gm_idx: Any, param_count: int) -> None:
    if type(param_count) is not int or param_count < 0:
        raise ValueError("Ascend auto-GM fresh contract requires a valid parameter count")
    if not isinstance(auto_gm_idx, list):
        raise ValueError("Ascend auto-GM fresh contract requires an explicit index list")
    seen = set()
    for index in auto_gm_idx:
        if type(index) is not int or index < 0 or index >= param_count:
            raise ValueError("Ascend auto-GM fresh contract has an invalid workspace index")
        if index in seen:
            raise ValueError("Ascend auto-GM fresh contract has a duplicate workspace index")
        seen.add(index)


def _source_hash(func: Any) -> str:
    script = getattr(func, "script", None)
    if not callable(script):
        raise ValueError("Ascend cache manifest requires its original function source")
    source = script()
    if not isinstance(source, str):
        raise ValueError("Ascend cache manifest requires a textual function source")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _runtime_param_signature(params: Any) -> list[dict[str, Any]]:
    if not isinstance(params, list):
        raise ValueError("Cython fresh allocator requires its complete runtime parameter metadata")
    signature = []
    for param in params:
        if not hasattr(param, "dtype") or not isinstance(getattr(param, "shape", None), list):
            raise ValueError("Cython fresh allocator has malformed runtime parameter metadata")
        signature.append({"dtype": str(param.dtype), "shape": [str(dim) for dim in param.shape]})
    return signature


def validate_fresh_runtime_signature(func: Any, lowered_func: Any, params: Any) -> None:
    """Verify that compiler fresh slots and the executing adapter share an ABI."""
    if not has_cython_allocator_fresh_contract(func):
        return
    from tilelang import tvm
    from tilelang.engine.param import KernelParam

    is_lowered_func = isinstance(lowered_func, tvm.tir.PrimFunc)
    if not is_lowered_func or not has_cython_allocator_fresh_contract(lowered_func):
        raise ValueError("Cython fresh allocator lost its lowered signature provenance")
    bound = bind_final_auto_gm_contract(lowered_func)
    if CYTHON_FINAL_SIGNATURE_ATTR not in lowered_func.attrs or not bound.same_as(lowered_func):
        raise ValueError("Cython fresh allocator has no pre-sync final signature binding")
    source_signature = func.attrs[CYTHON_INPUT_SIGNATURE_ATTR]
    lowered_signature = lowered_func.attrs[CYTHON_INPUT_SIGNATURE_ATTR]
    if (
        len(source_signature) != len(lowered_signature)
        or any(not param.same_as(lowered_signature[i]) for i, param in enumerate(source_signature))
        or _v2_recipe(func) != _v2_recipe(lowered_func)
    ):
        raise ValueError("Cython fresh allocator lowered a different source signature or recipe")
    if not isinstance(params, list) or len(params) != len(lowered_func.params):
        raise ValueError(
            "Cython fresh allocator runtime parameter count changed after synchronization",
        )
    expected = []
    for param in lowered_func.params:
        buffer = lowered_func.buffer_map.get(param)
        expected.append(
            KernelParam.from_buffer(buffer) if buffer is not None else KernelParam.from_var(param),
        )
    if _runtime_param_signature(expected) != _runtime_param_signature(params):
        raise ValueError(
            "Cython fresh allocator runtime parameter order or shape differs from the compiled ABI",
        )


def build_cached_auto_gm_manifest(
    func: Any,
    execution_backend: str,
    auto_gm_idx: Any,
    param_count: int,
    pass_configs: dict[str, Any] | None = None,
    target: str | Target | None = None,
    out_idx: Any = None,
    workspace_idx: Any = None,
    runtime_params: Any = None,
) -> dict[str, Any] | None:
    """Bind workspace metadata to the marked source and lowered signature.

    The manifest does not authenticate arbitrary external cache contents. It
    prevents metadata from an incomplete or different cached compilation from
    silently changing the allocator contract during reconstruction.
    """
    abi = sync_engine_cache_abi(pass_configs, target)
    fresh = requires_fresh_auto_gm(func)
    if not fresh and abi is None:
        return None
    validate_cached_auto_gm_contract(
        func,
        execution_backend,
        auto_gm_idx,
        param_count,
        out_idx,
        workspace_idx,
        target,
    )
    if abi is not None:
        _validate_indices(auto_gm_idx, param_count)
    manifest = {
        "schema": 2 if abi is not None else 1,
        "param_count": param_count,
        "auto_gm_indices": auto_gm_idx.copy(),
        "source_hash": _source_hash(func),
    }
    if abi is not None:
        manifest["sync_engine_abi"] = abi
    if has_cython_allocator_fresh_contract(func):
        manifest["schema"] = 3
        manifest["result_indices"] = _normalized_recipe(out_idx, param_count)
        manifest["workspace_indices"] = _normalized_recipe(workspace_idx, param_count)
        manifest["allocated_indices"] = sorted(
            set(manifest["result_indices"] + manifest["workspace_indices"] + auto_gm_idx),
        )
        manifest["runtime_signature"] = _runtime_param_signature(runtime_params)
        if len(manifest["runtime_signature"]) != param_count:
            raise ValueError("Cython fresh allocator manifest parameter count mismatch")
    return manifest


def validate_cached_auto_gm_manifest(
    func: Any,
    execution_backend: str,
    auto_gm_idx: Any,
    param_count: int,
    manifest: Any,
    pass_configs: dict[str, Any] | None = None,
    target: str | Target | None = None,
    out_idx: Any = None,
    workspace_idx: Any = None,
    runtime_params: Any = None,
) -> None:
    """Require the complete persisted allocation promise for marked artifacts.

    Parameter counts describe the lowered artifact, not ``func.params``:
    lowering can append automatic workspace parameters to the input function.
    Every index table is validated before equality, since Python considers
    boolean values equal to some integer indices.
    """
    abi = sync_engine_cache_abi(pass_configs, target)
    fresh = requires_fresh_auto_gm(func)
    if not fresh and abi is None:
        return
    validate_cached_auto_gm_contract(
        func,
        execution_backend,
        auto_gm_idx,
        param_count,
        out_idx,
        workspace_idx,
        target,
    )
    if abi is not None:
        _validate_indices(auto_gm_idx, param_count)
    fields = {"schema", "param_count", "auto_gm_indices", "source_hash"}
    if abi is not None:
        fields.add("sync_engine_abi")
    v2 = has_cython_allocator_fresh_contract(func)
    if v2:
        fields.update(
            {"result_indices", "workspace_indices", "allocated_indices", "runtime_signature"},
        )
    if type(manifest) is not dict or set(manifest) != fields:
        raise ValueError("Ascend auto-GM fresh contract requires a complete cache manifest")
    expected_schema = 3 if v2 else (2 if abi is not None else 1)
    if type(manifest["schema"]) is not int or manifest["schema"] != expected_schema:
        raise ValueError("Ascend auto-GM cache manifest has an unsupported schema")
    validate_cached_auto_gm_contract(
        func,
        execution_backend,
        manifest["auto_gm_indices"],
        manifest["param_count"],
        out_idx,
        workspace_idx,
        target,
    )
    if abi is not None:
        _validate_indices(manifest["auto_gm_indices"], manifest["param_count"])
        if not isinstance(manifest["sync_engine_abi"], str) or manifest["sync_engine_abi"] != abi:
            raise ValueError("Ascend synchronization engine cache ABI does not match")
    if manifest["param_count"] != param_count:
        raise ValueError("Ascend auto-GM cache manifest parameter count does not match")
    if manifest["auto_gm_indices"] != auto_gm_idx:
        raise ValueError("Ascend auto-GM cache manifest workspace indices do not match")
    if v2:
        message = "Cython fresh allocator manifest role indices do not match the actual allocator"
        for key, recipe in (("result_indices", out_idx), ("workspace_indices", workspace_idx)):
            expected = _normalized_recipe(recipe, param_count)
            actual = manifest[key]
            malformed = type(actual) is not list or any(type(value) is not int for value in actual)
            if malformed or actual != expected:
                raise ValueError(message)
        results = _normalized_recipe(out_idx, param_count)
        workspaces = _normalized_recipe(workspace_idx, param_count)
        union = sorted(set(results + workspaces + auto_gm_idx))
        signature = _runtime_param_signature(runtime_params)
        if (
            type(manifest["allocated_indices"]) is not list
            or any(type(value) is not int for value in manifest["allocated_indices"])
            or manifest["allocated_indices"] != union
            or len(signature) != param_count
            or manifest["runtime_signature"] != signature
        ):
            raise ValueError(
                "Cython fresh allocator manifest allocation union or runtime signature mismatch",
            )
    if type(manifest["source_hash"]) is not str or manifest["source_hash"] != _source_hash(func):
        raise ValueError("Ascend auto-GM cache manifest source hash does not match")
