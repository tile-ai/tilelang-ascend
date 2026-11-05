# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Host-only cache-load recovery tests; no kernel execution is required.

These tests verify that a cache entry whose ``kernel_lib.so`` disappears while it
is being loaded (for example, by a concurrent ``clear_cache`` in another process)
falls back to a cache miss instead of raising. They are deterministic: a mocked
loader deletes the library and raises ``OSError`` at the exact point the real
``ctypes.CDLL`` would fail.
"""

import ctypes
import logging
import pickle
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from tilelang import tvm
from tilelang.ascend.a2a3 import kernel_cache as a2a3_cache
from tilelang.autotuner import param as param_module
from tilelang.backend.module import BackendContext, create_backend_context, get_backend
from tilelang.cache import kernel_cache as cache_module
from tilelang.engine.param import KernelParam, dump_kernel_params


class TestIsFileMissing(unittest.TestCase):
    """Filesystem inputs for the combined missing-own-library predicate."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.helper = a2a3_cache

    def test_missing_entry_is_missing(self):
        path = self.root / "nope.so"
        error = OSError(f"{path}: cannot open shared object file: No such file or directory")
        self.assertTrue(self.helper._is_missing_own_library(error, path))

    def test_existing_entry_is_not_missing(self):
        path = self.root / "present.so"
        path.write_bytes(b"x")
        error = OSError(f"{path}: cannot open shared object file: No such file or directory")
        self.assertFalse(self.helper._is_missing_own_library(error, path))

    def test_permission_error_is_not_missing(self):
        path = self.root / "present.so"
        path.write_bytes(b"x")
        error = OSError(f"{path}: cannot open shared object file: No such file or directory")
        with mock.patch.object(self.helper.os, "stat", side_effect=PermissionError("denied")):
            self.assertFalse(self.helper._is_missing_own_library(error, path))

    def test_not_a_directory_is_not_missing(self):
        (self.root / "plain.txt").write_bytes(b"x")
        path = self.root / "plain.txt" / "child" / "kernel_lib.so"
        error = OSError(f"{path}: cannot open shared object file: No such file or directory")
        self.assertFalse(self.helper._is_missing_own_library(error, path))


class TestErrorReportsMissingPath(unittest.TestCase):
    """The loader-error classifier distinguishes target-missing from bad/foreign files."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.helper = a2a3_cache

    def _dl_error(self, path):
        try:
            ctypes.CDLL(str(path))
        except OSError as error:
            return error
        self.fail(f"loading {path} unexpectedly succeeded")

    def test_real_missing_target_is_reported(self):
        missing = self.root / "kernel_lib.so"
        self.assertTrue(self.helper._is_missing_own_library(self._dl_error(missing), missing))

    def test_real_truncated_library_is_not_reported(self):
        bad = self.root / "kernel_lib.so"
        bad.write_bytes(b"")  # dlopen reports "file too short"
        self.assertFalse(self.helper._is_missing_own_library(self._dl_error(bad), bad))

    def test_dependency_name_containing_target_is_not_reported(self):
        target = self.root / "kernel_lib.so"
        error = OSError("libkernel_lib.so: cannot open shared object file: No such file or directory")
        self.assertFalse(self.helper._is_missing_own_library(error, target))

    def test_same_basename_in_other_directory_is_not_reported(self):
        target = self.root / "kernel_lib.so"
        other = self.root / "other" / "kernel_lib.so"
        error = OSError(f"{other}: cannot open shared object file: No such file or directory")
        self.assertFalse(self.helper._is_missing_own_library(error, target))


class KernelCacheRecoveryMixin:
    """Shared fixtures and assertions for the cache-load entry points."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.jit_kernel = mock.MagicMock()
        cache_env = types.SimpleNamespace(
            TILELANG_CACHE_DIR=self.tmp.name,
            is_cache_enabled=lambda: True,
            should_use_kernel_cache_lib_stamp=lambda: False,
        )
        for owner, name, value in (
            (cache_module, "env", cache_env),
            (cache_module.KernelCache, "_last_cleaned_staging_root", cache_module.KernelCache._last_cleaned_staging_root),
        ):
            patch = mock.patch.object(owner, name, value)
            patch.start()
            self.addCleanup(patch.stop)
        logger = logging.getLogger(cache_module.__name__)
        self.addCleanup(logger.setLevel, logger.level)

    def write_entry(self, cache, key, with_lib=True):
        """Write a minimal cache entry and return ``(entry_dir, lib_path)``."""
        entry = Path(cache._get_cache_path(key))
        entry.mkdir(parents=True, exist_ok=True)
        (entry / cache.device_kernel_path).write_text("// kernel")
        (entry / cache.host_kernel_path).write_text("// wrapper")
        (entry / cache.params_path).write_text(dump_kernel_params([KernelParam(tvm.DataType("float32"), [1])]))
        lib = entry / "kernel_lib.so"
        if with_lib:
            lib.write_bytes(b"fake library")
        cache._write_manifest(str(entry))
        return entry, lib

    def assert_recovers_to_miss(self, cache, key):
        fake = object()

        def delete_then_fail(**kwargs):
            lib_path = kwargs["kernel_lib_path"]
            Path(lib_path).unlink()
            raise OSError(f"{lib_path}: cannot open shared object file: No such file or directory")

        with mock.patch.object(self.jit_kernel, "from_database", side_effect=delete_then_fail) as load:
            self.assertIsNone(cache._load_kernel_from_disk(key, self.context))

        load.assert_called_once()

        # A retry with the library restored does not hit the deleted inode.
        entry, lib = self.write_entry(cache, key, with_lib=True)
        self.jit_kernel.from_database.return_value = fake
        self.assertIs(cache._load_kernel_from_disk(key, self.context), fake)
        self.jit_kernel.from_database.assert_called_once()

    def assert_unloadable_library_is_a_miss(self, cache, key):
        entry, lib = self.write_entry(cache, key, with_lib=True)

        def fail_without_deleting(**kwargs):
            raise OSError("wrong ELF class: ELFCLASS32")

        with mock.patch.object(self.jit_kernel, "from_database", side_effect=fail_without_deleting) as load:
            self.assertIsNone(cache._load_kernel_from_disk(key, self.context))
        load.assert_called_once()


class TestKernelCacheRecovery(KernelCacheRecoveryMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.kernel_cache = cache_module
        for owner, name, value in (
            (self.kernel_cache.KernelCache, "_instance", None),
            (self.kernel_cache, "JITKernel", self.jit_kernel),
        ):
            patch = mock.patch.object(owner, name, value)
            patch.start()
            self.addCleanup(patch.stop)
        self.cache = self.kernel_cache.KernelCache()
        self.context = create_backend_context("llvm", execution_backend="tvm_ffi")

    def test_recovers_when_library_deleted_before_load(self):
        key = "k_del"
        self.write_entry(self.cache, key, with_lib=True)
        self.assert_recovers_to_miss(self.cache, key)

    def test_oserror_without_errno_is_recovered(self):
        key = "k_noerrno"
        self.write_entry(self.cache, key, with_lib=True)

        def delete_then_fail(**kwargs):
            lib_path = kwargs["kernel_lib_path"]
            Path(lib_path).unlink()
            error = OSError(f"{lib_path}: cannot open shared object file: No such file or directory")
            error.errno = None
            raise error

        with mock.patch.object(self.jit_kernel, "from_database", side_effect=delete_then_fail) as load:
            self.assertIsNone(self.cache._load_kernel_from_disk(key, self.context))
        load.assert_called_once()

    def test_permission_error_is_a_cache_miss(self):
        key = "k_perm"
        self.write_entry(self.cache, key, with_lib=True)

        def fail_permission(**kwargs):
            raise PermissionError("permission denied")

        with mock.patch.object(self.jit_kernel, "from_database", side_effect=fail_permission) as load:
            self.assertIsNone(self.cache._load_kernel_from_disk(key, self.context))
        load.assert_called_once()

    def test_unloadable_library_is_a_cache_miss(self):
        self.assert_unloadable_library_is_a_miss(self.cache, "k_elf")

    def test_missing_entry_directory_returns_none(self):
        self.assertIsNone(self.cache._load_kernel_from_disk("k_nothing", self.context))

    def test_cold_and_hot_cache_do_not_reload(self):
        fake = object()
        func = mock.MagicMock()
        func.script.return_value = "script"

        with mock.patch.object(self.cache, "_load_kernel_from_disk", return_value=fake) as load:
            first = self.cache.cached(func=func, backend_context=self.context, verbose=False)
            second = self.cache.cached(func=func, backend_context=self.context, verbose=False)
        self.assertIs(first, fake)
        self.assertIs(second, fake)
        load.assert_called_once()

    def test_permission_error_with_deletion_is_a_cache_miss(self):
        key = "k_perm_del"
        self.write_entry(self.cache, key, with_lib=True)

        def fail_permission_and_delete(**kwargs):
            Path(kwargs["kernel_lib_path"]).unlink()
            raise PermissionError("permission denied")

        with mock.patch.object(self.jit_kernel, "from_database", side_effect=fail_permission_and_delete) as load:
            self.assertIsNone(self.cache._load_kernel_from_disk(key, self.context))
        load.assert_called_once()

    def test_dependency_error_with_deletion_is_a_cache_miss(self):
        key = "k_dep_del"
        self.write_entry(self.cache, key, with_lib=True)

        def fail_dependency_and_delete(**kwargs):
            Path(kwargs["kernel_lib_path"]).unlink()
            # The dependency's name contains the target basename as a substring.
            error = OSError("libkernel_lib.so: cannot open shared object file: No such file or directory")
            error.errno = None
            raise error

        with mock.patch.object(self.jit_kernel, "from_database", side_effect=fail_dependency_and_delete) as load:
            self.assertIsNone(self.cache._load_kernel_from_disk(key, self.context))
        load.assert_called_once()

    def test_miss_compiles_and_returns_result(self):
        func = mock.MagicMock()
        func.script.return_value = "script"
        with (
            mock.patch.object(self.cache, "_load_kernel_from_disk", return_value=None),
            mock.patch.object(self.cache, "_save_kernel_to_disk"),
        ):
            result = self.cache.cached(func=func, backend_context=self.context, verbose=False)
        self.jit_kernel.assert_called_once()
        self.assertIs(result, self.jit_kernel.return_value)

    def test_disk_hot_cache_does_not_compile(self):
        key = "k_hot"
        self.write_entry(self.cache, key, with_lib=True)
        fake_kernel = object()
        func = mock.MagicMock()
        func.script.return_value = "script"

        # Compiling must fail if reached, proving the disk entry is what loads.
        self.jit_kernel.side_effect = RuntimeError("compile must not be reached")
        with (
            mock.patch.object(self.cache, "_generate_key", return_value=key),
            mock.patch.object(self.jit_kernel, "from_database", return_value=fake_kernel) as load,
        ):
            result = self.cache.cached(func=func, backend_context=self.context, verbose=False)
        self.assertIs(result, fake_kernel)
        load.assert_called_once()

    def test_truncated_library_with_deletion_is_a_cache_miss(self):
        key = "k_trunc"
        self.write_entry(self.cache, key, with_lib=True)

        def fail_truncated_and_delete(**kwargs):
            lib_path = kwargs["kernel_lib_path"]
            Path(lib_path).unlink()
            raise OSError(f"{lib_path}: file too short")

        with mock.patch.object(self.jit_kernel, "from_database", side_effect=fail_truncated_and_delete) as load:
            self.assertIsNone(self.cache._load_kernel_from_disk(key, self.context))
        load.assert_called_once()

    def test_recreated_library_loader_error_is_a_cache_miss(self):
        key = "k_recreated"
        self.write_entry(self.cache, key, with_lib=True)

        def delete_recreate_and_fail(**kwargs):
            lib_path = Path(kwargs["kernel_lib_path"])
            lib_path.unlink()
            lib_path.write_bytes(b"recreated")  # reappears before the filesystem check
            raise OSError(f"{lib_path}: cannot open shared object file: No such file or directory")

        with mock.patch.object(self.jit_kernel, "from_database", side_effect=delete_recreate_and_fail) as load:
            self.assertIsNone(self.cache._load_kernel_from_disk(key, self.context))
        load.assert_called_once()

    def test_makedirs_failure_propagates(self):
        func = mock.MagicMock()
        func.script.return_value = "script"
        with (
            mock.patch.object(self.cache, "_load_kernel_from_disk", return_value=None),
            mock.patch("os.makedirs", side_effect=FileNotFoundError("cache directory deleted")),
            self.assertRaises(FileNotFoundError),
        ):
            self.cache.cached(func=func, backend_context=self.context, verbose=False)

    def test_makedirs_permission_error_propagates(self):
        func = mock.MagicMock()
        func.script.return_value = "script"
        with (
            mock.patch.object(self.cache, "_load_kernel_from_disk", return_value=None),
            mock.patch("os.makedirs", side_effect=PermissionError("permission denied")),
            self.assertRaises(PermissionError),
        ):
            self.cache.cached(func=func, backend_context=self.context, verbose=False)


class TestAutotuneResultRecovery(KernelCacheRecoveryMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.param = param_module
        self.result_cls = self.param.AutotuneResult
        patch = mock.patch.object(a2a3_cache.A2A3CythonKernelCache, "_instance", None)
        patch.start()
        self.addCleanup(patch.stop)

    def _load(self, cache_path, **kwargs):
        if "ascend" not in tvm.target.Target.list_kinds():
            self.skipTest("requires USE_ASCEND=ON Target registration")
        backend = get_backend("ascend_a2a3")
        host = tvm.target.Target("c")
        target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}, host=host)
        context = BackendContext(backend, target, host, backend.execution_backends[0])
        args = self.param.CompileArgs(target=target, execution_backend="cython", _frozen=self.param._FrozenCompileRequest(context))
        func = mock.MagicMock()
        # Preparation/sidecar data only; the injected build does not validate ABI.
        prepared = (None, {}, None, types.SimpleNamespace(to_metadata=lambda: {}))
        with mock.patch.object(self.param.CompileArgs, "_prepare_a2a3_program", return_value=prepared):
            return self.result_cls._load_kernel_from_disk(self.result_cls, cache_path, context, func=func, compile_args=args, **kwargs)

    def test_recovers_when_library_deleted_before_load(self):
        entry = Path(self.tmp.name) / "a_del"
        entry.mkdir(parents=True, exist_ok=True)
        (entry / self.param.DEVICE_KERNEL_PATH).write_text("// kernel")
        (entry / self.param.HOST_KERNEL_PATH).write_text("// wrapper")
        (entry / self.param.PARAMS_PATH).write_text(dump_kernel_params([KernelParam(tvm.DataType("float32"), [1])]))
        (entry / a2a3_cache.A2A3CythonKernelCache.runtime_metadata_path).write_text("{}")
        (entry / "kernel_lib.so").write_bytes(b"fake library")

        fake = object()

        def delete_then_fail(*args, **kwargs):
            lib_path = args[3]
            Path(lib_path).unlink()
            raise OSError(f"{lib_path}: cannot open shared object file: No such file or directory")

        with mock.patch.object(a2a3_cache.A2A3CythonKernelCache, "_build_kernel", side_effect=delete_then_fail) as load:
            self.assertIsNone(self._load(entry))

        load.assert_called_once()
        (entry / "kernel_lib.so").write_bytes(b"fake library")
        with mock.patch.object(a2a3_cache.A2A3CythonKernelCache, "_build_kernel", return_value=fake) as load:
            self.assertIs(self._load(entry), fake)
        load.assert_called_once()

    def test_unloadable_library_is_not_a_miss(self):
        entry = Path(self.tmp.name) / "a_elf"
        entry.mkdir(parents=True, exist_ok=True)
        (entry / self.param.DEVICE_KERNEL_PATH).write_text("// kernel")
        (entry / self.param.HOST_KERNEL_PATH).write_text("// wrapper")
        (entry / self.param.PARAMS_PATH).write_text(dump_kernel_params([KernelParam(tvm.DataType("float32"), [1])]))
        (entry / a2a3_cache.A2A3CythonKernelCache.runtime_metadata_path).write_text("{}")
        (entry / "kernel_lib.so").write_bytes(b"fake library")

        def fail_without_deleting(*args, **kwargs):
            raise OSError("wrong ELF class: ELFCLASS32")

        with (
            mock.patch.object(a2a3_cache.A2A3CythonKernelCache, "_build_kernel", side_effect=fail_without_deleting) as load,
            self.assertRaises(OSError),
        ):
            self._load(entry)
        load.assert_called_once()

    def test_missing_metadata_propagates(self):
        entry = Path(self.tmp.name) / "meta_del"
        entry.mkdir(parents=True, exist_ok=True)
        (entry / "best_config.json").write_text('{"x": 1}')
        with open(entry / "function.pkl", "wb") as f:
            pickle.dump("fn", f)
        (entry / "latency.json").write_text('{"latency": 0.1, "ref_latency": 0.2}')
        (entry / "wrapped_kernel.cu").write_text("// kernel")
        with open(entry / "params.pkl", "wb") as f:
            pickle.dump([1], f)
        (entry / "auto_gm_idx.pkl").write_text("[]")
        (entry / "kernel_lib.so").write_bytes(b"fake library")
        # Simulate the deletion window: the dir passed the existence check, but
        # best_config.json is gone by the time it is opened.
        (entry / "best_config.json").unlink()

        with self.assertRaises(FileNotFoundError) as error:
            self.result_cls.load_from_disk(entry, self.param.CompileArgs(target="llvm", execution_backend="tvm_ffi"))
        self.assertEqual(Path(error.exception.filename), entry / "best_config.json")

    def test_save_to_disk_survives_cache_deletion(self):
        entry = Path(self.tmp.name) / "autotuner" / "save_del"
        entry.mkdir(parents=True, exist_ok=True)
        result = self.result_cls(
            config={"x": 1},
            func="fn",
            kernel=mock.MagicMock(),
            latency=0.1,
            ref_latency=0.2,
        )

        def fail_open(path, *args, **kwargs):
            raise FileNotFoundError(f"{path}: No such file or directory")

        with mock.patch("builtins.open", side_effect=fail_open):
            result.save_to_disk(entry)

    def test_save_to_disk_serialization_error_is_logged(self):
        entry = Path(self.tmp.name) / "autotuner" / "ser_err"
        entry.mkdir(parents=True, exist_ok=True)

        class ReadsMissingFile:
            def __reduce__(self):
                open("/definitely/does/not/exist_xyz.pkl", "rb")
                return (self.__class__, ())

        result = self.result_cls(
            config={"x": 1},
            func=ReadsMissingFile(),
            kernel=mock.MagicMock(),
            latency=0.1,
            ref_latency=0.2,
        )
        # The modern staged save logs serialization errors instead of propagating them.
        with self.assertLogs(self.param.logger, level="ERROR") as logs:
            result.save_to_disk(entry)
        self.assertTrue(any("exist_xyz.pkl" in message for message in logs.output))

    def test_save_to_disk_makedirs_deletion_propagates(self):
        entry = Path(self.tmp.name) / "mk_del"
        result = self.result_cls(
            config={"x": 1},
            func="fn",
            kernel=mock.MagicMock(),
            latency=0.1,
            ref_latency=0.2,
        )
        with mock.patch("os.makedirs", side_effect=FileNotFoundError("cache directory deleted")), self.assertRaises(FileNotFoundError):
            result.save_to_disk(entry)

    def test_save_to_disk_makedirs_permission_error_propagates(self):
        entry = Path(self.tmp.name) / "mk_perm"
        result = self.result_cls(
            config={"x": 1},
            func="fn",
            kernel=mock.MagicMock(),
            latency=0.1,
            ref_latency=0.2,
        )
        with mock.patch("os.makedirs", side_effect=PermissionError("permission denied")), self.assertRaises(PermissionError):
            result.save_to_disk(entry)


if __name__ == "__main__":
    unittest.main()
