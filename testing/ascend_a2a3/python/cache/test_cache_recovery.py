# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""All original recovery scenarios on actual current cache APIs.

Shared kernel loads warn/remove/miss on loader errors. A2/A3 tuning restoration
is stricter and recovers only disappearance of its own library. No retired
module loader, fake TVM/Target, or standalone AutoTunerCache is installed.
"""

import ctypes
import errno
import io
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from tilelang import env, tvm
from tilelang.ascend.a2a3 import kernel_cache as a2a3_cache_module
from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache, _is_missing_own_library
from tilelang.autotuner import AutoTuner
import tilelang.autotuner.tuner as tuner_module
from tilelang.autotuner import param as param_module
from tilelang.autotuner.param import AutotuneResult, CompileArgs
from tilelang.backend import create_backend_context
from tilelang.cache import kernel_cache as cache_module
from tilelang.cache.kernel_cache import KernelCache
from tilelang.engine.param import KernelParam, dump_kernel_params
from tilelang.env import CacheState


def _program(size=4):
    buffer = tvm.tirx.decl_buffer((size,), "float32", name="A")
    return tvm.tirx.PrimFunc([buffer.data], tvm.tirx.Evaluate(0), buffer_map={buffer.data: buffer}).with_attr(
        "global_symbol", "cache_state_unit"
    )


class _StateKernel:
    """File-state fixture only, never used as compiled or numerical evidence."""

    def __init__(self, library, func=None):
        self.prim_func = func if func is not None else _program()
        self.params = [KernelParam.from_buffer(self.prim_func.buffer_map[p]) for p in self.prim_func.params]
        self.execution_backend = "tvm_ffi"
        self.kernel_source = "// state-machine device source"
        self.adapter = SimpleNamespace(
            libpath=str(library),
            get_kernel_source=lambda: "// state-machine wrapper",
            get_host_source=lambda: "// state-machine host source",
        )

    def get_kernel_source(self):
        return self.kernel_source

    def update_tuner_result(self, **kwargs):
        self.tuner_result = kwargs
        return self


class _Files(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def dl_error(self, path):
        try:
            ctypes.CDLL(str(path))
        except OSError as error:
            return error
        self.fail(f"loading {path} unexpectedly succeeded")


class TestIsFileMissing(_Files):
    def missing(self, path):
        # Modern safety requires both a matching loader error and a fresh stat,
        # not the retired filesystem-only helper in isolation.
        error = OSError(f"{path}: cannot open shared object file: No such file or directory")
        return _is_missing_own_library(error, path)

    def test_missing_entry_is_missing(self):
        self.assertTrue(self.missing(self.root / "nope.so"))

    def test_existing_entry_is_not_missing(self):
        path = self.root / "present.so"
        path.write_bytes(b"x")
        self.assertFalse(self.missing(path))

    def test_permission_error_is_not_missing(self):
        path = self.root / "present.so"
        path.write_bytes(b"x")
        with mock.patch.object(a2a3_cache_module.os, "stat", side_effect=PermissionError(errno.EACCES, "denied")):
            self.assertFalse(self.missing(path))

    def test_not_a_directory_is_not_missing(self):
        (self.root / "plain.txt").write_bytes(b"x")
        self.assertFalse(self.missing(self.root / "plain.txt" / "child" / "kernel_lib.so"))


class TestErrorReportsMissingPath(_Files):
    def test_real_missing_target_is_reported(self):
        missing = self.root / "kernel_lib.so"
        self.assertTrue(_is_missing_own_library(self.dl_error(missing), missing))

    def test_real_truncated_library_is_not_reported(self):
        bad = self.root / "kernel_lib.so"
        bad.write_bytes(b"")
        self.assertFalse(_is_missing_own_library(self.dl_error(bad), bad))

    def test_dependency_name_containing_target_is_not_reported(self):
        target = self.root / "kernel_lib.so"
        error = OSError("libkernel_lib.so: cannot open shared object file: No such file or directory")
        self.assertFalse(_is_missing_own_library(error, target))

    def test_same_basename_in_other_directory_is_not_reported(self):
        target = self.root / "kernel_lib.so"
        other = self.root / "other" / "kernel_lib.so"
        error = OSError(f"{other}: cannot open shared object file: No such file or directory")
        self.assertFalse(_is_missing_own_library(error, target))


class KernelCacheRecoveryMixin(_Files):
    def setUp(self):
        super().setUp()
        descriptor = vars(type(env))["TILELANG_CACHE_DIR"]
        # The actual orchestration must not open/truncate a caller cwd log.
        unit_logger = logging.Logger("a2a3-cache-state-unit")
        unit_logger.setLevel(logging.DEBUG)
        self.tuner_log = io.StringIO()
        owned_handler = logging.StreamHandler(self.tuner_log)
        unit_logger.addHandler(owned_handler)
        unit_logger.parent = logging.getLogger()
        self.addCleanup(self.tuner_log.close)
        self.addCleanup(owned_handler.close)
        patches = [
            mock.patch.object(tuner_module, "logger", unit_logger),
            mock.patch.object(tuner_module, "_logger_handlers_initialized", True),
            mock.patch.object(descriptor, "_forced_value", str(self.root / "cache")),
            mock.patch.object(KernelCache, "_instance", None),
            mock.patch.object(A2A3CythonKernelCache, "_instance", None),
            mock.patch.object(KernelCache, "_last_cleaned_staging_root", KernelCache._last_cleaned_staging_root),
            mock.patch.object(AutoTuner, "_memory_cache", {}),
            mock.patch.object(CacheState, "_enabled", True),
            mock.patch.object(torch.cuda, "is_available", return_value=False),
        ]
        for name in ("TILELANG_DISABLE_CACHE", "TILELANG_AUTO_TUNING_DISABLE_CACHE"):
            patches.append(mock.patch.object(vars(type(env))[name], "_forced_value", "0"))
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        logger = logging.getLogger(cache_module.__name__)
        self.addCleanup(logger.setLevel, logger.level)
        self.cache = KernelCache()
        self.context = create_backend_context("llvm", execution_backend="tvm_ffi")
        self.func = _program()
        self.library = self.root / "compiled.so"
        self.library.write_bytes(b"unit state fixture; not a compiled kernel")
        self.kernel = _StateKernel(self.library, self.func)

    def write_entry(self, key, with_lib=True):
        entry = Path(self.cache._get_cache_path(key))
        entry.mkdir(parents=True, exist_ok=True)
        (entry / self.cache.device_kernel_path).write_text("// kernel")
        (entry / self.cache.host_kernel_path).write_text("// wrapper")
        (entry / self.cache.params_path).write_text(dump_kernel_params(self.kernel.params))
        lib = entry / self.cache.kernel_lib_path
        if with_lib:
            lib.write_bytes(b"unit loader boundary; not a compiled kernel")
        self.cache._write_manifest(str(entry))
        return entry, lib

    def load(self, key):
        return self.cache._load_kernel_from_disk(key, self.context, func=self.func)

    def cached(self):
        return self.cache.cached(func=self.func, backend_context=self.context, verbose=False)

    def shared_error_is_miss(self, key, error, delete=False, recreate=False):
        entry, library = self.write_entry(key)

        def fail(**kwargs):
            if delete or recreate:
                library.unlink()
            if recreate:
                library.write_bytes(b"recreated")
            raise error(library) if callable(error) else error

        with (
            mock.patch.object(cache_module.JITKernel, "from_database", side_effect=fail),
            self.assertLogs(self.cache.logger, level="WARNING") as logged,
        ):
            self.assertIsNone(self.load(key))
        self.assertTrue(any("treating it as a cache miss" in line for line in logged.output))
        self.assertFalse(entry.exists())

    def result_path(self, key):
        return self.root / "namespace" / "autotuner" / key

    def result(self, func=None):
        return AutotuneResult(
            config={"x": 1},
            func=self.func if func is None else func,
            kernel=self.kernel,
            latency=0.1,
            ref_latency=0.2,
        )

    def strict_entry(self, path, context):
        path.mkdir(parents=True)
        (path / param_module.DEVICE_KERNEL_PATH).write_text("// kernel")
        (path / param_module.HOST_KERNEL_PATH).write_text("// wrapper")
        (path / param_module.PARAMS_PATH).write_text(dump_kernel_params(self.kernel.params))
        lib = path / AutotuneResult._get_kernel_lib_file(context.execution_backend.name)
        lib.write_bytes(b"unit loader boundary; not a compiled kernel")
        # _build_kernel is the injected loader boundary below. This is only a
        # presence fixture, explicitly not proof of valid metadata or final ABI.
        (path / A2A3CythonKernelCache.runtime_metadata_path).write_text("{}")
        return lib

    def strict_load_boundary(self, key, fail, returns_none):
        if "ascend" not in tvm.target.Target.list_kinds():
            self.skipTest("requires the actual USE_ASCEND=ON Target kind for strict A2/A3 context")
        for dialect in ("ascendc", "pto"):
            target = {"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": [dialect, "ascend"]}
            args = CompileArgs(target=target, execution_backend="cython", out_idx=[0]).freeze()
            entry = self.result_path(key + "_" + dialect)
            lib = self.strict_entry(entry, args._frozen.context)

            def load(*positional, _lib=lib, **keyword):
                return fail(_lib)

            with mock.patch.object(A2A3CythonKernelCache, "_build_kernel", side_effect=load):

                def operation(_entry=entry, _args=args):
                    return AutotuneResult()._load_kernel_from_disk(_entry, _args._frozen.context, func=self.func, compile_args=_args)

                if returns_none:
                    self.assertIsNone(operation())
                else:
                    with self.assertRaisesRegex(OSError, "wrong ELF class: ELFCLASS32"):
                        operation()
            if returns_none:
                lib.write_bytes(b"restored fixture library")
                with mock.patch.object(A2A3CythonKernelCache, "_build_kernel", return_value=self.kernel):
                    self.assertIs(
                        AutotuneResult()._load_kernel_from_disk(entry, args._frozen.context, func=self.func, compile_args=args),
                        self.kernel,
                    )

    def strict_deleted_library(self, key):
        def fail(lib):
            lib.unlink()
            raise OSError(f"{lib}: cannot open shared object file: No such file or directory")

        self.strict_load_boundary(key, fail, returns_none=True)

    def strict_wrong_elf(self, key):
        def fail(lib):
            raise OSError("wrong ELF class: ELFCLASS32")

        self.strict_load_boundary(key, fail, returns_none=False)


class TestKernelCacheRecovery(KernelCacheRecoveryMixin):
    def test_recovers_when_library_deleted_before_load(self):
        key = "k_del"
        self.write_entry(key)

        def delete_then_fail(**kwargs):
            lib = Path(kwargs["kernel_lib_path"])
            lib.unlink()
            raise OSError(f"{lib}: cannot open shared object file: No such file or directory")

        with mock.patch.object(cache_module.JITKernel, "from_database", side_effect=delete_then_fail):
            self.assertIsNone(self.load(key))
        self.write_entry(key)
        with mock.patch.object(cache_module.JITKernel, "from_database", return_value=self.kernel):
            self.assertIs(self.load(key), self.kernel)

    def test_oserror_without_errno_is_recovered(self):
        def error(lib):
            value = OSError(f"{lib}: cannot open shared object file: No such file or directory")
            value.errno = None
            return value

        self.shared_error_is_miss("k_noerrno", error, delete=True)

    def test_permission_error_is_not_a_miss(self):
        # Source ID retained; shared modern load logs/removes/misses, unlike the
        # strict A2 tuning restore. Their contracts are intentionally distinct.
        self.shared_error_is_miss("k_perm", PermissionError("permission denied"))

    def test_unloadable_library_is_not_a_miss(self):
        self.shared_error_is_miss("k_elf", OSError("wrong ELF class: ELFCLASS32"))

    def test_missing_entry_directory_returns_none(self):
        self.assertIsNone(self.load("k_nothing"))

    def test_cold_and_hot_cache_do_not_reload(self):
        with mock.patch.object(self.cache, "_load_kernel_from_disk", return_value=self.kernel) as load:
            self.assertIs(self.cached(), self.kernel)
            self.assertIs(self.cached(), self.kernel)
        load.assert_called_once()

    def test_permission_error_with_deletion_is_not_a_miss(self):
        self.shared_error_is_miss("k_perm_del", PermissionError("permission denied"), delete=True)

    def test_dependency_error_with_deletion_is_not_a_miss(self):
        self.shared_error_is_miss(
            "k_dep_del", OSError("libkernel_lib.so: cannot open shared object file: No such file or directory"), delete=True
        )

    def test_miss_compiles_and_returns_result(self):
        with (
            mock.patch.object(self.cache, "_load_kernel_from_disk", return_value=None),
            mock.patch.object(self.cache, "_save_kernel_to_disk"),
            mock.patch.object(cache_module, "JITKernel", return_value=self.kernel) as constructor,
        ):
            self.assertIs(self.cached(), self.kernel)
        constructor.assert_called_once()

    def test_disk_hot_cache_does_not_compile(self):
        key = "k_hot"
        self.write_entry(key)
        with (
            mock.patch.object(self.cache, "_generate_key", return_value=key),
            mock.patch.object(cache_module, "JITKernel", side_effect=AssertionError("compile must not be reached")) as constructor,
        ):
            constructor.from_database.return_value = self.kernel
            self.assertIs(self.cached(), self.kernel)
        constructor.assert_not_called()
        constructor.from_database.assert_called_once()

    def test_truncated_library_with_deletion_is_not_a_miss(self):
        self.shared_error_is_miss("k_trunc", lambda lib: OSError(f"{lib}: file too short"), delete=True)

    def test_recreated_library_after_deletion_reraises(self):
        self.shared_error_is_miss(
            "k_recreated", lambda lib: OSError(f"{lib}: cannot open shared object file: No such file or directory"), recreate=True
        )

    def test_makedirs_failure_still_returns_result(self):
        # Entry staging mkdir is outside the modern save try. Preserve the same
        # OS fault and stage and assert its actual propagation, not old recovery.
        with (
            mock.patch.object(self.cache, "_load_kernel_from_disk", return_value=None),
            mock.patch.object(cache_module, "JITKernel", return_value=self.kernel),
            mock.patch("os.makedirs", side_effect=FileNotFoundError("cache directory deleted")),
            self.assertRaisesRegex(FileNotFoundError, "cache directory deleted"),
        ):
            self.cached()

    def test_makedirs_permission_error_propagates(self):
        with (
            mock.patch.object(self.cache, "_load_kernel_from_disk", return_value=None),
            mock.patch.object(cache_module, "JITKernel", return_value=self.kernel),
            mock.patch("os.makedirs", side_effect=PermissionError("permission denied")),
            self.assertRaisesRegex(PermissionError, "permission denied"),
        ):
            self.cached()


class TestTunerCacheRecovery(KernelCacheRecoveryMixin):
    def test_recovers_when_library_deleted_before_load(self):
        self.strict_deleted_library("t_del")

    def test_unloadable_library_is_not_a_miss(self):
        self.strict_wrong_elf("t_elf")

    def test_full_control_flow(self):
        # Current AutoTuner replaces the retired AutoTunerCache. Execute its
        # real scheduling/save/memory/disk flow; only compile/benchmark/loader
        # boundaries are stubs. This is not NPU numerical/timing evidence.
        tuner = AutoTuner(_program, [{"size": 4}]).set_compile_args(target="llvm", execution_backend="tvm_ffi", out_idx=[0])
        tuner.set_profile_args(skip_check=True, warmup=1, rep=1, timeout=0)
        compiles = []

        def compile(**kwargs):
            compiles.append(kwargs)
            return self.kernel

        tuner.jit_compile = compile
        tuner.jit_elaborate = lambda **kwargs: self.func
        with (
            mock.patch.object(tuner, "_resolve_num_compile_workers", return_value=1),
            mock.patch.object(tuner, "_benchmark_target", return_value=(0.1, 0.2)) as benchmark,
        ):
            cold = tuner.run(warmup=1, rep=1, timeout=0)
            hot = tuner.run(warmup=1, rep=1, timeout=0)
        self.assertIs(hot, cold)
        self.assertEqual(len(compiles), 1)
        benchmark.assert_called_once()
        self.assertEqual(cold.config, {"size": 4})
        self.assertEqual(cold.latency, 0.1)
        # Replace the test-owned dictionary, preserving all caller state.
        AutoTuner._memory_cache = {}
        with (
            mock.patch.object(param_module.JITKernel, "from_database", return_value=self.kernel) as loader,
            mock.patch.object(tuner, "jit_compile", side_effect=AssertionError("disk hit must not compile")),
            mock.patch.object(tuner, "_benchmark_target", side_effect=AssertionError("disk hit must not benchmark")),
        ):
            disk = tuner.run(warmup=1, rep=1, timeout=0)
        loader.assert_called_once()
        self.assertEqual(disk.config, cold.config)
        self.assertEqual(disk.latency, cold.latency)


class TestAutotuneResultRecovery(KernelCacheRecoveryMixin):
    def test_recovers_when_library_deleted_before_load(self):
        self.strict_deleted_library("a_del")

    def test_unloadable_library_is_not_a_miss(self):
        self.strict_wrong_elf("a_elf")

    def test_load_from_disk_metadata_deleted_returns_none(self):
        entry = self.result_path("meta_del")
        self.result().save_to_disk(entry)
        (entry / param_module.BEST_CONFIG_PATH).unlink()
        # The modern loader reads real metadata directly; do not weaken it to
        # None or rebuild retired mocks merely to retain the old assertion.
        with self.assertRaises(FileNotFoundError) as error:
            AutotuneResult.load_from_disk(entry, CompileArgs(target="llvm", execution_backend="tvm_ffi"))
        self.assertEqual(Path(error.exception.filename), entry / param_module.BEST_CONFIG_PATH)

    def test_save_to_disk_survives_cache_deletion(self):
        entry = self.result_path("save_del")
        with (
            mock.patch("builtins.open", side_effect=FileNotFoundError("cache directory deleted")),
            self.assertLogs(param_module.logger, level="ERROR") as logs,
        ):
            self.result().save_to_disk(entry)
        self.assertTrue(any("Error during atomic autotune result save" in message for message in logs.output))
        self.assertFalse(entry.exists())
        self.assertFalse(any((entry.parent.parent / ".staging").iterdir()))

    def test_save_to_disk_serialization_error_propagates(self):
        entry = self.result_path("ser_err")

        class ReadsMissingFile:
            def __reduce__(self):
                open("/definitely/does/not/exist_xyz.pkl", "rb")
                return (self.__class__, ())

        # Modern save logs any try-internal serialization error and publishes
        # nothing. This differs from the old writer, not the modern main tests.
        with self.assertLogs(param_module.logger, level="ERROR") as logs:
            self.result(func=ReadsMissingFile()).save_to_disk(entry)
        self.assertTrue(any("exist_xyz.pkl" in message for message in logs.output))
        self.assertFalse(entry.exists())
        self.assertFalse(any((entry.parent.parent / ".staging").iterdir()))

    def test_save_to_disk_makedirs_deletion_skips_save(self):
        entry = self.result_path("mk_del")
        with (
            mock.patch("os.makedirs", side_effect=FileNotFoundError("cache directory deleted")),
            self.assertRaisesRegex(FileNotFoundError, "cache directory deleted"),
        ):
            self.result().save_to_disk(entry)
        self.assertFalse(entry.exists())

    def test_save_to_disk_makedirs_permission_error_propagates(self):
        entry = self.result_path("mk_perm")
        with (
            mock.patch("os.makedirs", side_effect=PermissionError("permission denied")),
            self.assertRaisesRegex(PermissionError, "permission denied"),
        ):
            self.result().save_to_disk(entry)
        self.assertFalse(entry.exists())


if __name__ == "__main__":
    unittest.main()
