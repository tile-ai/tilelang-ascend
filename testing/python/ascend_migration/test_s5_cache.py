"""Native cache routing with explicitly synthetic protocol-cost fixtures; no NPU compile."""

import importlib
import logging
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_s5_cost_model import seal
from test_s5_kernels import make_pipeline_kernel
from test_s5_versions_ir import configs, fixture_profile

cache_module = importlib.import_module("tilelang.cache.kernel_cache")


def _cache(tmp_path):
    cache = object.__new__(cache_module.KernelCache)
    cache.cache_dir = tmp_path
    cache._memory_cache = {}
    cache.logger = logging.getLogger(__name__)
    return cache


def _arguments(program, profile):
    from tilelang.transform.pass_config import process_default_pass_config

    return {
        "func": program,
        "out_idx": [],
        "workspace_idx": [],
        "auto_gm_idx": [],
        "target": "ascendc",
        "platform": "A3",
        "args": (),
        "pass_configs": process_default_pass_config("ascendc", configs(profile)),
        "compile_flags": ["-O2"],
    }


def _profiles():
    left = fixture_profile()
    right = fixture_profile()
    right["records"][0]["latency"] += 1
    right["records"][0]["latency_effective_ns"] += 1
    right["evidence"]["raw_samples"] = ["different explicitly synthetic measurement fixture"]
    return left, seal(right)


def test_same_profile_filename_changed_contents_change_real_cache_key(tmp_path):
    import json
    from tilelang.transform.ascend_pipeline_cost import load_cost_model

    cache = _cache(tmp_path)
    program = make_pipeline_kernel()
    path = tmp_path / "synthetic-cost.json"
    keys = []
    for profile in _profiles():
        path.write_text(json.dumps(profile))
        loaded = load_cost_model(path)
        keys.append(cache._generate_key(**_arguments(program, loaded)))
    assert keys[0] != keys[1]


@pytest.mark.parametrize("layer", ["memory", "disk"])
def test_cost_contents_cannot_restore_a_kernel_for_the_other_profile(tmp_path, monkeypatch, layer):
    cache = _cache(tmp_path)
    program = make_pipeline_kernel()
    disk, constructed = {}, []

    def construct(*args, **kwargs):
        value = {"profile_sha256": kwargs["pass_configs"]["tl.ascend_pipeline_cost_model"]["profile_sha256"]}
        constructed.append(value)
        return value

    monkeypatch.setattr(cache_module, "is_cache_enabled", lambda: True)
    monkeypatch.setattr(cache_module, "JITKernel", construct)
    monkeypatch.setattr(cache, "_load_kernel_from_disk", lambda key, *args: disk.get(key))
    monkeypatch.setattr(cache, "_save_kernel_to_disk", lambda key, kernel, *args: disk.update({key: kernel}))
    profiles = _profiles()
    for profile in (*profiles, *profiles):
        if layer == "disk":
            cache._memory_cache.clear()
        arguments = _arguments(program, profile)
        arguments.pop("args")
        result = cache.cached(**arguments)
        assert result["profile_sha256"] == profile["profile_sha256"]
    assert len(constructed) == len(disk) == 2


def test_cold_legacy_pipeline_lowering_never_requests_optional_solver():
    source = """import sys
class RejectSolver:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0]=='z3' or 'ascend_z3_scheduler' in fullname:
            raise AssertionError('legacy requested optional solver: '+fullname)
sys.meta_path.insert(0,RejectSolver())
import tilelang
from tilelang import tvm
from test_s5_kernels import make_pipeline_kernel
with tvm.transform.PassContext(config={'tl.ascend_pipeline_planner':'legacy','tl.ascend_auto_sync':True,'tl.ascend_memory_planning':True}):
    tilelang.lower(make_pipeline_kernel(),target='ascendc',platform='A3')
assert tvm.get_global_func('tl.ascend.solve_schedule',allow_missing=True) is None
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parent), env.get("PYTHONPATH", "")])
    result = subprocess.run([sys.executable, "-c", source], env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
