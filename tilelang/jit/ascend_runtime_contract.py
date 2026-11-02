# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Private allocation promises made by the Ascend Cython JIT entry.

This contract describes allocation of compiler-added ``auto_gm_indices``.
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
AUTO_GM_MANIFEST_PATH = "ascend_auto_gm_contract.json"


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


def prepare_auto_gm_contract(func: PrimFunc, execution_backend: str, target: str | Target) -> PrimFunc:
    """Return an attributed copy for a new compilation, keeping ``func`` intact.

    Recompilation first discards a previous private allocation promise. Only
    the selected Cython Ascend JIT path can replace it with the current schema.
    This helper must run before cache-key generation as well as direct JIT
    compilation, so the contract version participates in artifact identity.
    """
    from tilelang import tvm

    if not isinstance(func, tvm.tir.PrimFunc):
        raise TypeError("Ascend allocation contracts require a PrimFunc")
    prepared = func.without_attr(AUTO_GM_FRESH_CONTRACT_ATTR)
    if execution_backend == "cython" and _ascend_model(target):
        prepared = prepared.with_attr(AUTO_GM_FRESH_CONTRACT_ATTR, tvm.tir.StringImm(AUTO_GM_FRESH_CONTRACT_VERSION))
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
    if not isinstance(contract, tvm.tir.StringImm) or contract.value != AUTO_GM_FRESH_CONTRACT_VERSION:
        raise ValueError(f"Invalid Ascend auto-GM fresh contract; expected StringImm({AUTO_GM_FRESH_CONTRACT_VERSION!r})")
    return True


def validate_cached_auto_gm_contract(
    func: Any,
    execution_backend: str,
    auto_gm_idx: Any,
    param_count: int,
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


def build_cached_auto_gm_manifest(
    func: Any,
    execution_backend: str,
    auto_gm_idx: Any,
    param_count: int,
    pass_configs: dict[str, Any] | None = None,
    target: str | Target | None = None,
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
    validate_cached_auto_gm_contract(func, execution_backend, auto_gm_idx, param_count)
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
    return manifest


def validate_cached_auto_gm_manifest(
    func: Any,
    execution_backend: str,
    auto_gm_idx: Any,
    param_count: int,
    manifest: Any,
    pass_configs: dict[str, Any] | None = None,
    target: str | Target | None = None,
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
    validate_cached_auto_gm_contract(func, execution_backend, auto_gm_idx, param_count)
    if abi is not None:
        _validate_indices(auto_gm_idx, param_count)
    fields = {"schema", "param_count", "auto_gm_indices", "source_hash"}
    if abi is not None:
        fields.add("sync_engine_abi")
    if type(manifest) is not dict or set(manifest) != fields:
        raise ValueError("Ascend auto-GM fresh contract requires a complete cache manifest")
    expected_schema = 2 if abi is not None else 1
    if type(manifest["schema"]) is not int or manifest["schema"] != expected_schema:
        raise ValueError("Ascend auto-GM cache manifest has an unsupported schema")
    validate_cached_auto_gm_contract(func, execution_backend, manifest["auto_gm_indices"], manifest["param_count"])
    if abi is not None:
        _validate_indices(manifest["auto_gm_indices"], manifest["param_count"])
        if not isinstance(manifest["sync_engine_abi"], str) or manifest["sync_engine_abi"] != abi:
            raise ValueError("Ascend synchronization engine cache ABI does not match")
    if manifest["param_count"] != param_count:
        raise ValueError("Ascend auto-GM cache manifest parameter count does not match")
    if manifest["auto_gm_indices"] != auto_gm_idx:
        raise ValueError("Ascend auto-GM cache manifest workspace indices do not match")
    if type(manifest["source_hash"]) is not str or manifest["source_hash"] != _source_hash(func):
        raise ValueError("Ascend auto-GM cache manifest source hash does not match")
