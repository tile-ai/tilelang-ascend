"""Autotune cache identity and measurement consume the same profiling settings."""

from types import SimpleNamespace

import pytest

import tilelang
from tilelang.autotuner import tuner as tuner_module


def _factory(block=32):
    return block


def _reference():
    pass


@pytest.fixture
def tuner(monkeypatch):
    calls = SimpleNamespace(profiles=[], benchmarks=[], timeouts=[], saved_keys=[])
    autotuner = tuner_module.AutoTuner(_factory, configs=[{"block": 32}])
    autotuner.set_compile_args(target="ascendc", platform="A3")
    autotuner.set_profile_args(
        warmup=7,
        rep=11,
        timeout=13,
        ref_prog=_reference,
        skip_check=True,
    )

    class Profiler:
        def _get_inputs(self, **kwargs):
            return []

        def do_bench(self, *args, **kwargs):
            if args:
                assert args == (_reference,)
                values = ("reference", kwargs["n_warmup"], kwargs["n_repeat"])
            else:
                values = ("kernel", kwargs["warmup"], kwargs["rep"])
            calls.benchmarks.append(values)
            return 1.0

    class Kernel:
        prim_func = None

        def get_profiler(self, **kwargs):
            return Profiler()

        def get_kernel_source(self):
            return "// mock kernel"

        def update_tuner_result(self, **kwargs):
            return self

    generate_key = autotuner.generate_cache_key

    def record_key(parameters, profile_args=None):
        calls.profiles.append(profile_args)
        return generate_key(parameters, profile_args=profile_args)

    def run_with_timeout(fn, timeout, kernel):
        calls.timeouts.append(timeout)
        return fn(kernel)

    autotuner.jit_compile = lambda **kwargs: Kernel()
    monkeypatch.setattr(autotuner, "generate_cache_key", record_key)
    monkeypatch.setattr(autotuner, "_load_result_from_disk", lambda key: None)
    monkeypatch.setattr(
        autotuner,
        "_save_result_to_disk",
        lambda key, result: calls.saved_keys.append(key),
    )
    monkeypatch.setattr(tuner_module.AutoTuner, "_memory_cache", {})
    monkeypatch.setattr(tuner_module, "_init_logger_handlers", lambda: None)
    monkeypatch.setattr(tuner_module, "get_available_cpu_count", lambda: 1)
    monkeypatch.setattr(tuner_module, "run_with_timeout", run_with_timeout)
    monkeypatch.setattr(tuner_module.env, "is_cache_enabled", lambda: True)
    monkeypatch.setattr(tuner_module.torch.cuda, "is_available", lambda: False)
    if hasattr(tuner_module.torch, "npu"):
        monkeypatch.setattr(tuner_module.torch.npu, "is_available", lambda: False)
    return autotuner, calls


@pytest.mark.parametrize("field,value", [("warmup", 17), ("rep", 19), ("timeout", 23)])
def test_effective_profile_controls_cache_and_measurement(tuner, field, value):
    autotuner, calls = tuner
    configured_profile = autotuner.profile_args
    first = autotuner.run()
    assert autotuner.run() is first
    assert calls.benchmarks == [("kernel", 7, 11), ("reference", 7, 11)]
    assert calls.timeouts == [13]

    second = autotuner.run(**{field: value})
    assert second is not first
    effective = calls.profiles[-1]
    assert getattr(effective, field) == value
    assert autotuner.run(**{field: value}) is second
    assert autotuner.run() is first
    assert autotuner.profile_args is configured_profile
    assert calls.benchmarks[-2:] == [
        ("kernel", effective.warmup, effective.rep),
        ("reference", effective.warmup, effective.rep),
    ]
    assert calls.timeouts == [13, effective.timeout]
    assert len(set(calls.saved_keys)) == 2


def test_run_overrides_apply_only_to_current_call(tuner):
    autotuner, calls = tuner
    configured_profile = autotuner.profile_args
    autotuner.run(3, 20, 17)
    partial = autotuner.run(rep=23)
    effective = calls.profiles[-1]
    assert (effective.warmup, effective.rep, effective.timeout) == (7, 23, 13)
    assert autotuner.run(None, 23, None) is partial
    autotuner.run()
    assert autotuner.profile_args is configured_profile
    assert calls.benchmarks == [
        ("kernel", 3, 20),
        ("reference", 3, 20),
        ("kernel", 7, 23),
        ("reference", 7, 23),
        ("kernel", 7, 11),
        ("reference", 7, 11),
    ]
    assert calls.timeouts == [17, 13, 13]
    assert len(set(calls.saved_keys)) == 3


def test_decorator_populates_profile_args():
    @tilelang.autotune(configs=[{"block": 32}], warmup=3, rep=20, timeout=17)
    @tilelang.jit(target="ascendc", platform="A3")
    def factory(block=32):
        return block

    autotuner = factory.get_tunner()
    profile = autotuner.profile_args
    assert (profile.warmup, profile.rep, profile.timeout) == (3, 20, 17)
    assert autotuner.run.__func__ is tuner_module.AutoTuner.run
