"""Real cache-key/routing isolation from S5 artifacts, without compiling device kernels."""

import copy
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

import pytest

import tilelang
from tilelang import tvm
from test_s5_cache import _arguments
from test_s5_kernels import make_pipeline_kernel
from test_s5_versions_ir import fixture_profile

cache_module = importlib.import_module("tilelang.cache.kernel_cache")
ABI_FIELD = "ascend_constraint_compiler_abi"
S5_FIELDS = {
    "version",
    "func",
    "out_idx",
    "workspace_idx",
    "auto_gm_idx",
    "args_repr",
    "target",
    "target_host",
    "platform",
    "execution_backend",
    "pass_configs",
    "compile_flags",
}


def cache(tmp_path):
    result = object.__new__(cache_module.KernelCache)
    result.cache_dir = tmp_path
    result._memory_cache = {}
    result.logger = logging.getLogger(__name__)
    return result


def keys_and_payload(instance, arguments, monkeypatch):
    """Reconstruct the old persisted key by removing only the new ABI field."""
    real_dumps = json.dumps
    captured = {}

    def record(value, *args, **kwargs):
        captured.update(copy.deepcopy(value))
        return real_dumps(value, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(cache_module.json, "dumps", record)
        current = instance._generate_key(**arguments)
    previous = {key: value for key, value in captured.items() if key != ABI_FIELD}
    assert set(previous) == S5_FIELDS
    old = hashlib.sha256(real_dumps(previous, sort_keys=True).encode()).hexdigest()
    return current, old, captured


def test_constraint_abi_separates_the_same_head_program_profile_and_flags(tmp_path, monkeypatch):
    arguments = _arguments(make_pipeline_kernel(), fixture_profile())
    new, old, payload = keys_and_payload(cache(tmp_path), arguments, monkeypatch)
    assert new != old and payload[ABI_FIELD] == 5
    assert payload["version"] == cache_module.__version__


@pytest.mark.parametrize("explicit", [False, True])
def test_legacy_cache_key_bytes_remain_in_the_s5_format(tmp_path, monkeypatch, explicit):
    arguments = _arguments(make_pipeline_kernel(), fixture_profile())
    if explicit:
        arguments["pass_configs"]["tl.ascend_pipeline_planner"] = "legacy"
    else:
        arguments["pass_configs"].pop("tl.ascend_pipeline_planner")
    new, old, payload = keys_and_payload(cache(tmp_path), arguments, monkeypatch)
    assert ABI_FIELD not in payload and new == old


def test_user_pass_configuration_cannot_override_the_internal_compiler_abi(tmp_path, monkeypatch):
    arguments = _arguments(make_pipeline_kernel(), fixture_profile())
    arguments["pass_configs"][ABI_FIELD] = 2
    _, _, payload = keys_and_payload(cache(tmp_path), arguments, monkeypatch)
    assert payload["pass_configs"][ABI_FIELD] == 2
    assert payload[ABI_FIELD] == 5


@pytest.mark.parametrize("layer", ["memory", "disk"])
@pytest.mark.parametrize("reject", [False, True])
def test_old_artifact_cannot_bypass_new_compilation_or_readiness(tmp_path, monkeypatch, layer, reject):
    instance = cache(tmp_path)
    arguments = _arguments(make_pipeline_kernel(), fixture_profile())
    new, old, _ = keys_and_payload(instance, arguments, monkeypatch)
    previous = {"schema": 2, "validated_by_s6": False}
    disk = {old: previous} if layer == "disk" else {}
    if layer == "memory":
        instance._memory_cache[old] = previous
    constructed, loaded = [], []

    def construct(*args, **kwargs):
        constructed.append(kwargs)
        if reject:
            raise ValueError("S6 readiness rejected the synthetic compiler input")
        return {"schema": 3, "validated_by_s6": True}

    def load(key, *args):
        loaded.append(key)
        return disk.get(key)

    monkeypatch.setattr(cache_module, "is_cache_enabled", lambda: True)
    monkeypatch.setattr(cache_module, "JITKernel", construct)
    monkeypatch.setattr(instance, "_load_kernel_from_disk", load)
    monkeypatch.setattr(instance, "_save_kernel_to_disk", lambda key, value, *args: disk.update({key: value}))
    arguments.pop("args")
    if reject:
        with pytest.raises(ValueError, match="S6 readiness"):
            instance.cached(**arguments)
    else:
        assert instance.cached(**arguments) == {"schema": 3, "validated_by_s6": True}
        if layer == "disk":
            instance._memory_cache.clear()
        assert instance.cached(**arguments) == {"schema": 3, "validated_by_s6": True}
    assert len(constructed) == 1 and old not in loaded and set(loaded) == {new}


@pytest.mark.parametrize("explicit", [None, "legacy", "constraint"])
def test_jit_inner_context_matches_the_strategy_used_by_the_cache_key(tmp_path, monkeypatch, explicit):
    """Exercise the real JIT context boundary, stopping before any lowering/codegen."""
    instance = cache(tmp_path)
    cache_package = importlib.import_module("tilelang.cache")
    monkeypatch.setattr(cache_package, "_kernel_cache_instance", instance)
    monkeypatch.setattr(cache_module, "is_cache_enabled", lambda: True)
    monkeypatch.setattr(instance, "_load_kernel_from_disk", lambda *args: None)
    observed = []
    captured = {}
    real_key = instance._generate_key

    def generate(**kwargs):
        captured.update(kwargs)
        return real_key(**kwargs)

    monkeypatch.setattr(instance, "_generate_key", generate)

    class StopBeforeLower(Exception):
        pass

    def lower(*args, **kwargs):
        observed.append(str(tvm.transform.PassContext.current().config.get("tl.ascend_pipeline_planner", "legacy")))
        raise StopBeforeLower

    monkeypatch.setattr(tilelang, "lower", lower)
    configs = None if explicit is None else {"tl.ascend_pipeline_planner": explicit}
    with (
        tvm.transform.PassContext(config={"tl.ascend_pipeline_planner": "constraint"}),
        pytest.raises(StopBeforeLower),
    ):
        tilelang.compile(
            make_pipeline_kernel(),
            out_idx=[],
            target="ascendc",
            platform="A3",
            execution_backend="ctypes",
            pass_configs=configs,
        )
    assert observed == [explicit or "legacy"]
    _, _, payload = keys_and_payload(instance, captured, monkeypatch)
    assert (ABI_FIELD in payload) is (observed[0] == "constraint")


def test_cold_legacy_cache_key_does_not_request_solver():
    source = """import sys,importlib
class RejectSolver:
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0]=='z3' or 'ascend_z3_scheduler' in fullname:
   raise AssertionError('legacy cache requested solver: '+fullname)
sys.meta_path.insert(0,RejectSolver())
import tilelang
from test_s5_kernels import make_pipeline_kernel
m=importlib.import_module('tilelang.cache.kernel_cache')
c=object.__new__(m.KernelCache)
key=c._generate_key(make_pipeline_kernel(),[],[],[],args=(),target='ascendc',platform='A3',pass_configs={'tl.ascend_pipeline_planner':'legacy'},compile_flags=['-O2'])
assert len(key)==64
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parent), env.get("PYTHONPATH", "")])
    result = subprocess.run([sys.executable, "-c", source], env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
