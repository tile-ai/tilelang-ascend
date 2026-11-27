"""Actual JIT cache routes isolate the new synchronization compiler contract."""

import copy
import hashlib
import importlib
import json

import pytest

import tilelang
from tilelang import tvm
from test_s5_cache import _arguments
from test_s5_kernels import make_pipeline_kernel
from test_s5_versions_ir import fixture_profile
from test_s6_cache import cache

cache_module = importlib.import_module("tilelang.cache.kernel_cache")
SYNC_ABI = "ascend_sync_compiler_abi"


def keys(instance, arguments, monkeypatch):
    real_dumps = json.dumps
    payload = {}

    def capture(value, *args, **kwargs):
        payload.update(copy.deepcopy(value))
        return real_dumps(value, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(cache_module.json, "dumps", capture)
        current = instance._generate_key(**arguments)
    previous = {key: value for key, value in payload.items() if key != SYNC_ABI}
    return (
        current,
        hashlib.sha256(real_dumps(previous, sort_keys=True).encode()).hexdigest(),
        payload,
    )


@pytest.mark.parametrize("planner", ["legacy", "constraint"])
def test_new_sync_abi_is_internal_and_independent_of_the_planner(tmp_path, monkeypatch, planner):
    args = _arguments(make_pipeline_kernel(), fixture_profile())
    args["pass_configs"].update({"tl.ascend_pipeline_planner": planner, "tl.ascend_sync_strategy": "a3"})
    current, previous, payload = keys(cache(tmp_path), args, monkeypatch)
    assert current != previous and payload[SYNC_ABI] == 2
    assert (payload.get("ascend_constraint_compiler_abi") == 5) == (planner == "constraint")


@pytest.mark.parametrize("explicit", [False, True])
def test_default_and_explicit_old_sync_keep_their_existing_key_bytes(tmp_path, monkeypatch, explicit):
    args = _arguments(make_pipeline_kernel(), fixture_profile())
    if explicit:
        args["pass_configs"]["tl.ascend_sync_strategy"] = "legacy"
    current, previous, payload = keys(cache(tmp_path), args, monkeypatch)
    assert SYNC_ABI not in payload and current == previous
    assert payload["ascend_constraint_compiler_abi"] == 5


def test_user_configuration_does_not_override_the_fixed_sync_contract(tmp_path, monkeypatch):
    args = _arguments(make_pipeline_kernel(), fixture_profile())
    args["pass_configs"].update({"tl.ascend_sync_strategy": "a3", SYNC_ABI: 0})
    _, _, payload = keys(cache(tmp_path), args, monkeypatch)
    assert payload[SYNC_ABI] == 2 and payload["pass_configs"][SYNC_ABI] == 0


@pytest.mark.parametrize("layer", ["memory", "disk"])
def test_previous_sync_artifacts_do_not_bypass_new_compilation(tmp_path, monkeypatch, layer):
    instance = cache(tmp_path)
    args = _arguments(make_pipeline_kernel(), fixture_profile())
    args["pass_configs"]["tl.ascend_sync_strategy"] = "a3"
    current, previous, _ = keys(instance, args, monkeypatch)
    stale = {"sync": "previous unvalidated compiler contract"}
    disk = {previous: stale} if layer == "disk" else {}
    if layer == "memory":
        instance._memory_cache[previous] = stale
    constructed, loaded = [], []

    def build(*pargs, **kwargs):
        constructed.append(kwargs)
        return {"sync": "new compiler contract"}

    def load(key, *pargs):
        loaded.append(key)
        return disk.get(key)

    monkeypatch.setattr(cache_module, "is_cache_enabled", lambda: True)
    monkeypatch.setattr(cache_module, "JITKernel", build)
    monkeypatch.setattr(instance, "_load_kernel_from_disk", load)
    monkeypatch.setattr(instance, "_save_kernel_to_disk", lambda key, value, *rest: disk.update({key: value}))
    args.pop("args")
    assert instance.cached(**args) == {"sync": "new compiler contract"}
    assert len(constructed) == 1 and loaded == [current]
    assert previous not in loaded


@pytest.mark.parametrize("explicit", [None, "legacy", "a3"])
def test_real_jit_inner_context_and_key_agree_on_effective_sync(tmp_path, monkeypatch, explicit):
    instance = cache(tmp_path)
    monkeypatch.setattr(importlib.import_module("tilelang.cache"), "_kernel_cache_instance", instance)
    monkeypatch.setattr(cache_module, "is_cache_enabled", lambda: True)
    monkeypatch.setattr(instance, "_load_kernel_from_disk", lambda *args: None)
    captured, observed = {}, []
    real_key = instance._generate_key

    def generate(**kwargs):
        captured.update(kwargs)
        return real_key(**kwargs)

    class StopBeforeLower(Exception):
        pass

    def lower(*args, **kwargs):
        observed.append(str(tvm.transform.PassContext.current().config.get("tl.ascend_sync_strategy", "legacy")))
        raise StopBeforeLower

    monkeypatch.setattr(instance, "_generate_key", generate)
    monkeypatch.setattr(tilelang, "lower", lower)
    configs = None if explicit is None else {"tl.ascend_sync_strategy": explicit}
    with (
        tvm.transform.PassContext(config={"tl.ascend_sync_strategy": "a3"}),
        pytest.raises(StopBeforeLower),
    ):
        tilelang.compile(
            make_pipeline_kernel(),
            out_idx=[],
            target="ascendc",
            platform="A3",
            pass_configs=configs,
        )
    assert observed == [explicit or "legacy"]
    current, previous, payload = keys(instance, captured, monkeypatch)
    assert (SYNC_ABI in payload) == (explicit == "a3")
    assert (current != previous) == (explicit == "a3")


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("transpose", [False, True])
def test_gm_byte_hash_matches_the_existing_dtype_shape_contract(dtype, transpose, monkeypatch):
    import torch
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts/ascend_migration"))
    from _s5_npu import tensor_digest as previous_digest
    from _s7_npu import tensor_digest

    value = torch.arange(32, dtype=getattr(torch, dtype)).reshape(4, 8)
    value[1, 1] = float("nan")
    if transpose:
        value = value.transpose(0, 1)
    assert tensor_digest(value, torch) == previous_digest(value, torch)
