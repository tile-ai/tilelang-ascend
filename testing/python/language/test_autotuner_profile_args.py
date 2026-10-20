"""Autotune cache identity and measurement consume the same profiling settings."""

from pathlib import Path
from types import SimpleNamespace

import pytest

import tilelang
from tilelang.autotuner import tuner as tuner_module
from tilelang.engine.param import KernelParam


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

        update_tuner_result = tilelang.JITKernel.update_tuner_result

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


def test_profile_cache_round_trips_through_disk(tuner, monkeypatch, tmp_path):
    autotuner, calls = tuner
    monkeypatch.delattr(autotuner, "_load_result_from_disk")
    monkeypatch.delattr(autotuner, "_save_result_to_disk")
    monkeypatch.setattr(tuner_module.AutoTuner, "cache_dir", tmp_path / "cache")
    library = tmp_path / "compiled.so"
    library.write_bytes(b"mock compiled library")
    create_kernel = autotuner.jit_compile
    compiles = []
    restores = []

    def compile_kernel(**config):
        compiles.append(config)
        kernel = create_kernel(**config)
        kernel.prim_func = _factory
        kernel.kernel_source = kernel.get_kernel_source()
        kernel.params = [KernelParam(tuner_module.torch.float32, [1])]
        kernel.auto_gm_idx = []
        kernel.adapter = SimpleNamespace(libpath=str(library))
        return kernel

    def restore_kernel(**kwargs):
        # Keep the real cache readers; replace only loading the compiled library.
        library_path = Path(kwargs["kernel_lib_path"])
        assert library_path.read_bytes() == library.read_bytes()
        assert kwargs["kernel_global_source"] == "// mock kernel"
        assert kwargs["params"] == [KernelParam(tuner_module.torch.float32, [1])]
        restores.append(library_path)
        kernel = create_kernel()
        kernel.prim_func = kwargs["func"]
        return kernel

    monkeypatch.setattr(tilelang.JITKernel, "from_database", staticmethod(restore_kernel))
    autotuner.jit_compile = compile_kernel
    first = autotuner.run()
    first_benchmarks = list(calls.benchmarks)
    first_entries = set(autotuner.cache_dir.iterdir())
    assert len(first_entries) == 1
    assert compiles == [{"block": 32}]
    assert first_benchmarks == [("kernel", 7, 11), ("reference", 7, 11)]

    tuner_module.AutoTuner._memory_cache.clear()
    fresh = tuner_module.AutoTuner(_factory, configs=[{"block": 32}])
    fresh.set_compile_args(target="ascendc", platform="A3")
    fresh.set_profile_args(warmup=7, rep=11, timeout=13, ref_prog=_reference, skip_check=True)
    fresh.jit_compile = compile_kernel
    restored = fresh.run()
    assert restored is not first
    assert restored.kernel is not first.kernel
    assert (restored.config, restored.latency, restored.ref_latency, restored.libcode) == (
        first.config,
        first.latency,
        first.ref_latency,
        first.libcode,
    )
    assert restored.func(64) == 64
    assert restored.kernel.latency == restored.latency
    assert restored.kernel.config == restored.config
    assert restored.kernel.ref_latency == restored.ref_latency
    assert len(restores) == 1
    assert calls.benchmarks == first_benchmarks
    assert compiles == [{"block": 32}]
    assert calls.timeouts == [13]

    tuner_module.AutoTuner._memory_cache.clear()
    changed = fresh.run(timeout=23)
    assert changed is not restored
    assert len(restores) == 1
    assert calls.benchmarks == first_benchmarks * 2
    assert compiles == [{"block": 32}, {"block": 32}]
    assert calls.timeouts == [13, 23]
    assert len(set(autotuner.cache_dir.iterdir()) - first_entries) == 1


def test_decorator_populates_profile_args():
    @tilelang.autotune(configs=[{"block": 32}], warmup=3, rep=20, timeout=17)
    @tilelang.jit(target="ascendc", platform="A3")
    def factory(block=32):
        return block

    autotuner = factory.get_tunner()
    profile = autotuner.profile_args
    assert (profile.warmup, profile.rep, profile.timeout) == (3, 20, 17)
    assert autotuner.run.__func__ is tuner_module.AutoTuner.run
