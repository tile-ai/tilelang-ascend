"""Ascend cache entries, linker commands and restored libraries share one run mode."""

import importlib
import inspect
import json
from types import SimpleNamespace
from unittest.mock import Mock

import cloudpickle
import pytest

import tilelang
from tilelang.autotuner import param as autotune_param
from tilelang.autotuner import AutoTuner
from tilelang.cache import kernel_cache as cache_module
from tilelang.jit import kernel as kernel_module
from tilelang.jit.adapter import libgen


class Program:
    def script(self):
        return "run_mode_cache_regression"


def _write_kernel_files(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / cache_module.WRAPPED_KERNEL_PATH).write_text("// wrapped kernel")
    (path / cache_module.KERNEL_LIB_PATH).write_bytes(b"")
    (path / cache_module.PARAMS_PATH).write_bytes(cloudpickle.dumps(["parameter"]))
    (path / cache_module.AUTO_GM_IDX_PATH).write_text("[]")


@pytest.fixture
def cache(monkeypatch, tmp_path):
    monkeypatch.setattr(cache_module.KernelCache, "_instance", None)
    instance = cache_module.KernelCache(tmp_path / "cache")
    monkeypatch.setattr(cache_module, "is_cache_enabled", lambda: True)
    return instance


def test_run_mode_partitions_memory_disk_and_uncached_compilation(cache, monkeypatch):
    compiled = []
    restored = []

    class Kernel:
        def __init__(self, func, **kwargs):
            self.run_mode = kwargs.get("run_mode")
            compiled.append(self.run_mode)

        @classmethod
        def from_database(cls, **kwargs):
            restored.append(kwargs["run_mode"])
            return SimpleNamespace(run_mode=kwargs["run_mode"])

    monkeypatch.setattr(cache_module, "JITKernel", Kernel)
    monkeypatch.setattr(
        cache,
        "_save_kernel_to_disk",
        lambda key, kernel, func: _write_kernel_files(cache.cache_dir / key),
    )

    def compile_kernel():
        return cache.cached(Program(), target="ascendc", platform="A3")

    monkeypatch.delenv("TL_RUN_MODE", raising=False)
    npu = compile_kernel()
    monkeypatch.setenv("TL_RUN_MODE", "sim")
    sim = compile_kernel()
    monkeypatch.setenv("TL_RUN_MODE", "npu")
    assert compile_kernel() is npu
    assert sim is not npu
    assert compiled == ["npu", "sim"]

    cache._memory_cache.clear()
    monkeypatch.setenv("TL_RUN_MODE", "sim")
    result = compile_kernel()
    assert result.run_mode == "sim"
    assert restored == ["sim"]
    assert compiled == ["npu", "sim"]

    monkeypatch.setattr(cache_module, "is_cache_enabled", lambda: False)
    result = compile_kernel()
    assert result.run_mode == "sim"
    assert compiled == ["npu", "sim", "sim"]


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("mode", ["npu", "sim"])
def test_cache_linker_and_loader_consume_resolved_mode(cache, monkeypatch, tmp_path, target, mode):
    seen = {}
    generate_key = cache._generate_key

    def record_key(**kwargs):
        seen["key_mode"] = kwargs["run_mode"]
        return generate_key(**kwargs)

    def create_adapter(**kwargs):
        seen["adapter_mode"] = kwargs["run_mode"]
        # A later environment change must not change this build or its loader.
        monkeypatch.setenv("TL_RUN_MODE", "npu" if mode == "sim" else "sim")
        gen = libgen.LibraryGenerator(
            target,
            "A3",
            kwargs["compile_flags"],
            run_mode=kwargs["run_mode"],
        )
        gen.update_lib_code("// test kernel")
        gen.compile_lib()
        gen.load_lib()
        return SimpleNamespace(func=lambda: None)

    def compile_command(command, **kwargs):
        seen["command"] = command
        return SimpleNamespace(returncode=0)

    monkeypatch.setenv("TL_RUN_MODE", mode)
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    monkeypatch.setattr(tilelang.cache, "_kernel_cache_instance", cache)
    monkeypatch.setattr(cache, "_generate_key", record_key)
    monkeypatch.setattr(cache, "_load_kernel_from_disk", lambda *args: None)
    monkeypatch.setattr(cache, "_save_kernel_to_disk", lambda *args: None)
    monkeypatch.setattr(
        tilelang,
        "lower",
        lambda *args, **kwargs: SimpleNamespace(
            params=[],
            host_mod=None,
            device_mod=None,
            kernel_source="// kernel",
        ),
    )
    monkeypatch.setattr(kernel_module, "CythonKernelAdapter", create_adapter)
    monkeypatch.setattr(libgen, "_get_ascend_home_path", lambda: "/ascend")
    monkeypatch.setattr(libgen, "_get_tl_root", lambda: "/tilelang")
    monkeypatch.setattr(libgen, "_get_simulator_lib_path", lambda *args: "/simulator/lib")
    monkeypatch.setattr(libgen.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(libgen.subprocess, "run", compile_command)
    monkeypatch.setattr(libgen.ctypes, "CDLL", lambda path: object())

    kernel = tilelang.compile(Program(), target=target, platform="A3")
    assert seen["key_mode"] == seen["adapter_mode"] == kernel.run_mode == mode
    command = seen["command"]
    if mode == "sim":
        assert "-lruntime_camodel" in command and "-lruntime" not in command
        assert "-L/simulator/lib" in command
        assert "-Wl,-rpath,/simulator/lib" in command
        assert libgen.os.environ["LD_LIBRARY_PATH"].split(":")[0] == "/simulator/lib"
    else:
        assert "-lruntime" in command and "-lruntime_camodel" not in command
        assert "-L/simulator/lib" not in command
        assert libgen.os.environ["LD_LIBRARY_PATH"] == ""


@pytest.mark.parametrize("from_database", [False, True])
def test_cython_adapter_threads_mode_on_compile_and_restore(monkeypatch, from_database):
    adapter_module = importlib.import_module("tilelang.jit.adapter.cython.adapter")
    adapter_class = adapter_module.CythonKernelAdapter
    for name in (
        "_process_dynamic_symbolic",
        "_process_buffer_dtype",
        "_process_ptr_map",
        "_process_static_shape",
        "_process_buffer_device",
    ):
        monkeypatch.setattr(adapter_class, name, lambda self: {})
    monkeypatch.setattr(adapter_class, "_post_init", lambda self: None)
    monkeypatch.setattr(adapter_module, "TLWrapper", lambda *args: Mock())
    monkeypatch.setattr(adapter_module, "CythonKernelWrapper", Mock())
    monkeypatch.setattr(libgen.LibraryGenerator, "compile_lib", lambda self: None)
    monkeypatch.setattr(libgen.LibraryGenerator, "load_lib", lambda self, **kwargs: object())
    monkeypatch.setenv("TL_RUN_MODE", "npu")
    kwargs = dict(
        params=[],
        result_idx=[],
        workspace_idx=[],
        auto_gm_idx=[],
        target="ascendc",
        platform="A3",
        func_or_mod=tilelang.tvm.IRModule(),
        kernel_global_source="// kernel",
        run_mode="sim",
    )
    if from_database:
        adapter = adapter_class.from_database(**kwargs, kernel_lib_path="unused.so")
    else:
        adapter = adapter_class(**kwargs)
    assert adapter.lib_generator.run_mode == "sim"


def test_jit_local_cache_scopes_mode_and_preserves_user_kwargs(monkeypatch):
    jit_module = importlib.import_module("tilelang.jit")
    compiled = []

    def compile_program(program, **kwargs):
        assert "run_mode" not in kwargs
        compiled.append(libgen.resolve_run_mode())
        return object()

    monkeypatch.setattr(jit_module, "compile", compile_program)

    @tilelang.jit(target="ascendc", platform="A3")
    def factory(n, __run_mode=None):
        assert __run_mode == "user-value"
        return n

    monkeypatch.setenv("TL_RUN_MODE", "npu")
    npu = factory(16, __run_mode="user-value")
    monkeypatch.setenv("TL_RUN_MODE", "sim")
    sim = factory(16, __run_mode="user-value")
    monkeypatch.setenv("TL_RUN_MODE", "npu")
    assert factory(16, __run_mode="user-value") is npu
    assert sim is not npu
    assert compiled == ["npu", "sim"]


def test_autotune_local_cache_and_jit_share_mode(monkeypatch):
    jit_module = importlib.import_module("tilelang.jit")
    compiled = []
    tuner_modes = []

    def compile_program(program, **kwargs):
        assert "run_mode" not in kwargs
        compiled.append(libgen.resolve_run_mode())
        return object()

    monkeypatch.setattr(jit_module, "compile", compile_program)

    @tilelang.autotune(configs=[{"block": 32}])
    @tilelang.jit(target="ascendc", platform="A3")
    def factory(n, block=32):
        return n, block

    class Tuner:
        def set_kernel_parameters(self, key, parameters):
            assert key == ((16,), ())

        def run(self):
            return SimpleNamespace(kernel=self.jit_compile(block=32))

    def get_tuner():
        tuner_modes.append(libgen.resolve_run_mode())
        return Tuner()

    monkeypatch.setattr(factory, "get_tunner", get_tuner)
    monkeypatch.setenv("TL_RUN_MODE", "npu")
    npu = factory(16)
    monkeypatch.setenv("TL_RUN_MODE", "sim")
    sim = factory(16)
    monkeypatch.setenv("TL_RUN_MODE", "npu")
    assert factory(16) is npu
    assert sim is not npu
    assert compiled == tuner_modes == ["npu", "sim"]


def test_autotune_key_and_disk_restore_resolve_environment_on_call(monkeypatch, tmp_path):
    def factory(block=32):
        return Program()

    tuner = AutoTuner(factory, configs=[{"block": 32}])
    args = autotune_param.CompileArgs(target="ascendc", platform="A3")
    tuner.compile_args = args
    parameters = inspect.signature(factory).parameters
    monkeypatch.setenv("TL_RUN_MODE", "npu")
    npu_key = tuner.generate_cache_key(parameters)
    monkeypatch.setenv("TL_RUN_MODE", "sim")
    sim_key = tuner.generate_cache_key(parameters)
    assert sim_key != npu_key
    assert tuner.generate_cache_key(parameters) == sim_key

    _write_kernel_files(tmp_path)
    (tmp_path / autotune_param.BEST_CONFIG_PATH).write_text('{"block": 32}')
    (tmp_path / autotune_param.FUNCTION_PATH).write_bytes(cloudpickle.dumps(Program()))
    latency = json.dumps({"latency": 1.0, "ref_latency": None})
    (tmp_path / autotune_param.LATENCY_PATH).write_text(latency)
    restored = {}

    def restore(**kwargs):
        restored.update(kwargs)
        return SimpleNamespace(
            update_tuner_result=lambda **kwargs: None,
            get_kernel_source=lambda: "// kernel",
        )

    monkeypatch.setattr(kernel_module.JITKernel, "from_database", restore)
    result = autotune_param.AutotuneResult.load_from_disk(tmp_path, args)
    assert result is not None
    assert restored["run_mode"] == "sim"
