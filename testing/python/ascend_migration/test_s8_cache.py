"""Actual cache routing isolates the complete auto compiler contract."""

import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys

import pytest

from test_s6_cache import cache
from test_s8_kernels import make_vector_kernel

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/ascend_migration"))
from _s8_npu import COMPILE_FLAGS, pass_configs  # noqa: E402

cache_module = importlib.import_module("tilelang.cache.kernel_cache")
ABI = "ascend_auto_compiler_abi"


def arguments(strategy="auto"):
    return {
        "func": make_vector_kernel(3),
        "out_idx": [],
        "workspace_idx": [],
        "auto_gm_idx": [],
        "target": "ascendc",
        "platform": "A3",
        "args": (),
        "pass_configs": pass_configs(strategy),
        "compile_flags": list(COMPILE_FLAGS),
    }


def keys(instance, args, monkeypatch):
    real_dumps = json.dumps
    payload = {}

    def capture(value, *positional, **keywords):
        payload.update(copy.deepcopy(value))
        return real_dumps(value, *positional, **keywords)

    with monkeypatch.context() as context:
        context.setattr(cache_module.json, "dumps", capture)
        current = instance._generate_key(**args)
    previous = {key: value for key, value in payload.items() if key != ABI}
    old = hashlib.sha256(real_dumps(previous, sort_keys=True).encode()).hexdigest()
    return current, old, payload


def test_auto_has_fixed_internal_abi_without_a_nanosecond_cost_model(tmp_path, monkeypatch):
    current, old, payload = keys(cache(tmp_path), arguments(), monkeypatch)
    assert current != old and payload[ABI] == 3
    assert payload["ascend_sync_compiler_abi"] == 2
    assert "tl.ascend_pipeline_cost_model" not in payload["pass_configs"]


@pytest.mark.parametrize("explicit", [False, True])
def test_existing_legacy_key_bytes_do_not_gain_the_auto_salt(tmp_path, monkeypatch, explicit):
    args = arguments("legacy")
    if not explicit:
        args["pass_configs"].pop("tl.ascend_pipeline_planner")
    current, old, payload = keys(cache(tmp_path), args, monkeypatch)
    assert current == old and ABI not in payload


def test_user_field_cannot_override_internal_auto_abi(tmp_path, monkeypatch):
    args = arguments()
    args["pass_configs"][ABI] = 0
    _, _, payload = keys(cache(tmp_path), args, monkeypatch)
    assert payload[ABI] == 3 and payload["pass_configs"][ABI] == 0


@pytest.mark.parametrize("layer", ["memory", "disk"])
def test_an_unverified_old_key_does_not_restore_an_auto_kernel(tmp_path, monkeypatch, layer):
    instance, args = cache(tmp_path), arguments()
    current, old, _ = keys(instance, args, monkeypatch)
    stale = {"compiler": "old unverified auto contract"}
    disk = {old: stale} if layer == "disk" else {}
    if layer == "memory":
        instance._memory_cache[old] = stale
    constructed, loaded = [], []

    def construct(*positional, **keywords):
        constructed.append(keywords)
        return {"compiler": "new complete auto contract"}

    def load(key, *rest):
        loaded.append(key)
        return disk.get(key)

    monkeypatch.setattr(cache_module, "is_cache_enabled", lambda: True)
    monkeypatch.setattr(cache_module, "JITKernel", construct)
    monkeypatch.setattr(instance, "_load_kernel_from_disk", load)
    monkeypatch.setattr(instance, "_save_kernel_to_disk", lambda key, value, *rest: disk.update({key: value}))
    args.pop("args")
    assert instance.cached(**args) == {"compiler": "new complete auto contract"}
    assert len(constructed) == 1 and loaded == [current] and old not in loaded


def test_scoped_flags_and_actual_source_domain_remain_cache_inputs(tmp_path):
    instance, args = cache(tmp_path), arguments()
    base = instance._generate_key(**args)
    args["compile_flags"] = ["-O1"]
    assert instance._generate_key(**args) != base
    args["compile_flags"] = list(COMPILE_FLAGS)
    args["func"] = make_vector_kernel(1)
    assert instance._generate_key(**args) != base
