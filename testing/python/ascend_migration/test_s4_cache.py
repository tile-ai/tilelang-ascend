"""Exercise real cache identities and routing without invoking a native compiler."""

import importlib
import logging

import pytest

from test_s4_kernels import make_vector_kernel, vector_pass_configs

cache_module = importlib.import_module("tilelang.cache.kernel_cache")


def _cache(tmp_path):
    # Keep the production process singleton and its files untouched.
    cache = object.__new__(cache_module.KernelCache)
    cache.cache_dir = tmp_path
    cache._memory_cache = {}
    cache.logger = logging.getLogger(__name__)
    return cache


def _arguments(program, configs):
    return {
        "func": program,
        "out_idx": [],
        "workspace_idx": [],
        "auto_gm_idx": [],
        "target": "ascendc",
        "platform": "A3",
        "args": (),
        "pass_configs": configs,
        "compile_flags": ["-O2"],
    }


def test_actual_cache_key_partitions_constraint_from_both_legacy_forms(tmp_path):
    cache = _cache(tmp_path)
    program = make_vector_kernel()
    explicit = vector_pass_configs("legacy")
    default = {key: value for key, value in explicit.items() if key != "tl.ascend_pipeline_planner"}
    constraint = vector_pass_configs("constraint")
    keys = [cache._generate_key(**_arguments(program, configs)) for configs in (default, explicit, constraint)]
    # Default and explicit legacy have the same behavior, but may have distinct
    # cache keys because the optional default is not materialized by Python.
    assert keys[2] not in keys[:2]
    assert keys[2] == cache._generate_key(**_arguments(program, dict(reversed(constraint.items()))))
    assert explicit["tl.ascend_pipeline_planner"] == "legacy"
    assert "tl.ascend_pipeline_planner" not in default


@pytest.mark.parametrize("layer", ["memory", "disk"])
def test_cache_hit_routing_cannot_restore_the_other_strategy(tmp_path, monkeypatch, layer):
    cache = _cache(tmp_path)
    program = make_vector_kernel()
    constructed = []
    disk = {}

    def construct(*args, **kwargs):
        result = {"strategy": kwargs["pass_configs"]["tl.ascend_pipeline_planner"]}
        constructed.append(result)
        return result

    monkeypatch.setattr(cache_module, "is_cache_enabled", lambda: True)
    monkeypatch.setattr(cache_module, "JITKernel", construct)
    monkeypatch.setattr(cache, "_load_kernel_from_disk", lambda key, *args: disk.get(key))
    monkeypatch.setattr(cache, "_save_kernel_to_disk", lambda key, kernel, *args: disk.update({key: kernel}))
    restored = {}
    for strategy in ("legacy", "constraint", "legacy", "constraint"):
        if layer == "disk":
            cache._memory_cache.clear()
        arguments = _arguments(program, vector_pass_configs(strategy))
        # cached() accepts Python *args; _generate_key receives its tuple.
        arguments.pop("args")
        result = cache.cached(**arguments)
        assert result["strategy"] == strategy
        if strategy in restored:
            assert result is restored[strategy]
        restored[strategy] = result
    assert [kernel["strategy"] for kernel in constructed] == ["legacy", "constraint"]
    assert len(disk) == 2
