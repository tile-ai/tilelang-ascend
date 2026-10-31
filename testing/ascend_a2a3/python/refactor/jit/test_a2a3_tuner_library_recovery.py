"""A2/A3 tuner restoration recovers only disappearance of its own native SO."""

import errno
import hashlib
import json
from pathlib import Path

import pytest
import torch

from tilelang.ascend.a2a3 import compiler
from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache
from tilelang.autotuner import param as param_module
from tilelang.autotuner.param import AutotuneResult
from tilelang.autotuner import AutoTuner
from tilelang.backend import create_backend_context
from tilelang.engine.param import KernelParam, dump_kernel_params

from test_a2a3_autotuner_contract import actual_toolchain as actual_toolchain
from test_a2a3_autotuner_contract import snapshot_paths as snapshot_paths
from test_a2a3_autotuner_contract import _args, _factories
from test_a2a3_public_execution import _small_func


def _entry(path, context, func):
    path.mkdir()
    (path / param_module.DEVICE_KERNEL_PATH).write_text("// unit restore source")
    (path / param_module.HOST_KERNEL_PATH).write_text("// unit restore wrapper")
    params = [KernelParam.from_buffer(func.buffer_map[param]) for param in func.params]
    (path / param_module.PARAMS_PATH).write_text(dump_kernel_params(params))
    library = path / AutotuneResult._get_kernel_lib_file(context.execution_backend.name)
    library.write_bytes(b"unit loader boundary; not a compiled kernel")
    (path / A2A3CythonKernelCache.runtime_metadata_path).write_text("{}")
    return library


@pytest.mark.parametrize("dialect", ["ascendc", "pto"])
@pytest.mark.parametrize("kind", ["value", "metadata", "permission", "wrong_elf", "truncated", "dependency", "recreated"])
def test_a2a3_restore_preserves_non_removal_failures(snapshot_paths, monkeypatch, tmp_path, dialect, kind):
    target = {"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": [dialect, "ascend"]}
    args = _args(target=target).freeze()
    context = args._frozen.context
    func = _small_func()
    entry = tmp_path / "result"
    library = _entry(entry, context, func)
    failures = {
        "value": ValueError("inconsistent runtime signature"),
        "metadata": FileNotFoundError(errno.ENOENT, "No such file or directory", str(entry / "a2a3_runtime.json")),
        "permission": PermissionError(errno.EACCES, "Permission denied", str(library)),
        "wrong_elf": OSError(f"{library}: wrong ELF class: ELFCLASS32"),
        "truncated": OSError(f"{library}: file too short"),
        "dependency": OSError("libdependency.so: cannot open shared object file: No such file or directory"),
        "recreated": OSError(f"{library}: cannot open shared object file: No such file or directory"),
    }
    expected = failures[kind]

    def fail(*positional, **keyword):
        if kind == "metadata":
            (entry / "a2a3_runtime.json").unlink()
        elif kind != "recreated":
            library.unlink()
        raise expected

    monkeypatch.setattr(A2A3CythonKernelCache, "_build_kernel", fail)
    with pytest.raises(type(expected)) as failure:
        AutotuneResult()._load_kernel_from_disk(entry, context, func=func, compile_args=args)
    assert failure.value is expected


@pytest.mark.parametrize("backend", ["ascend", "cuda"])
def test_existing_backends_keep_the_original_load_error(monkeypatch, tmp_path, backend):
    target = {"kind": "ascend", "arch": "dav-3510", "model": "A5"} if backend == "ascend" else {"kind": "cuda", "arch": "sm_80"}
    context = create_backend_context(target, execution_backend="cython")
    assert context.module.name == backend
    func = _small_func()
    library = _entry(tmp_path / "result", context, func)
    expected = OSError(f"{library}: cannot open shared object file: No such file or directory")

    def fail(cls, **kwargs):
        library.unlink()
        raise expected

    monkeypatch.setattr(param_module.JITKernel, "from_database", classmethod(fail))
    with pytest.raises(OSError) as failure:
        AutotuneResult()._load_kernel_from_disk(library.parent, context, func=func)
    assert failure.value is expected


def test_actual_tuner_rebuilds_after_its_library_disappears(actual_toolchain, monkeypatch, tmp_path):
    config = {"M": 256, "N": 256, "K": 128, "block_M": 128, "block_N": 128, "block_K": 64}
    tuner = (
        AutoTuner(_factories().matmul_add, [config])
        .set_compile_args(
            target={"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["ascendc", "ascend"]},
            execution_backend="cython",
            out_idx=[2],
            workspace_idx=-1,
            pass_configs={
                "tl.ascend_auto_sync": True,
                "tl.ascend_auto_cv_combine": True,
                "tl.ascend_memory_planning": True,
                "tl.ascend_auto_cross_core_sync": True,
                "tl.ascend_auto_sync_vs": False,
            },
        )
        .set_profile_args(skip_check=True, warmup=1, rep=1, timeout=0)
    )
    monkeypatch.setattr(torch.npu, "is_available", lambda: False)
    monkeypatch.setattr(torch.npu, "current_device", lambda: pytest.fail("IO recovery must not query a device"))
    monkeypatch.setattr(tuner, "_resolve_num_compile_workers", lambda: 1)
    benchmarks = []
    compiles = []
    original_compile = compiler.LibraryGenerator.compile_lib

    def compile(generator, *args, **kwargs):
        result = original_compile(generator, *args, **kwargs)
        compiles.append(generator.config.to_metadata())
        return result

    def benchmark(**kwargs):
        # Exercise real orchestration/ABI/cache repair, not numerical timing.
        benchmarks.append(kwargs["jit_kernel"])
        return 1.0, None

    monkeypatch.setattr(compiler.LibraryGenerator, "compile_lib", compile)
    monkeypatch.setattr(tuner, "_benchmark_target", benchmark)
    cold = tuner.run()
    assert cold.kernel.workspace_idx == [4]
    [runtime_path] = list(tuner.cache_dir.rglob("a2a3_runtime.json"))
    library = runtime_path.parent / "kernel_lib.so"
    before_hash = hashlib.sha256(library.read_bytes()).hexdigest()
    frozen_config = cold.kernel.compile_config.cache_identity
    A2A3CythonKernelCache().clear_cache()
    tuner._memory_cache.clear()
    assert library.exists()
    monkeypatch.setenv("TL_CCE_OPT_LEVEL", "invalid-after-cold")
    visits = []
    original_load = compiler.LibraryGenerator.load_lib

    def delete_once(generator, lib_path=None):
        path = Path(lib_path or generator.libpath)
        if not visits and path.resolve() == library.resolve():
            path.unlink()
            visits.append(str(path))
        return original_load(generator, lib_path)

    monkeypatch.setattr(compiler.LibraryGenerator, "load_lib", delete_once)
    repaired = tuner.run()
    assert len(visits) == 1 and len(benchmarks) == 2 and len(compiles) == 2
    assert repaired.kernel.workspace_idx == [4]
    assert repaired.kernel.compile_config.cache_identity == frozen_config
    assert library.exists() and runtime_path.exists()
    tuner._memory_cache.clear()
    restored = tuner.run()
    assert restored.kernel.artifact is None and restored.kernel.workspace_idx == [4]
    assert len(benchmarks) == 2 and len(compiles) == 2
    assert restored.kernel.compile_config.cache_identity == frozen_config
    (tmp_path / "recovery.json").write_text(
        json.dumps(
            {
                "library_before_sha256": before_hash,
                "library_after_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                "compile_configs": compiles,
                "deleted_library": visits,
                "workspace_idx": restored.kernel.workspace_idx,
                "benchmark": "explicit constant stub; no NPU execution",
            },
            indent=2,
        )
    )
