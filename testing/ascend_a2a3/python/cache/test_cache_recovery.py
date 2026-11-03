# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Host-only cache recovery scenarios on the current cache APIs."""

import ctypes
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from tilelang import tvm
from tilelang.ascend.a2a3 import kernel_cache as a2a3_cache_module
from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache, _is_missing_own_library
from tilelang.autotuner import param as param_module
from tilelang.autotuner.param import AutotuneResult, CompileArgs, _FrozenCompileRequest
from tilelang.backend.module import BackendContext, create_backend_context, get_backend
from tilelang.cache import kernel_cache as cache_module
from tilelang.cache.kernel_cache import KernelCache
from tilelang.engine.param import KernelParam, dump_kernel_params


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


class TestMissingOwnLibraryStat(_Files):
    """Original filesystem examples on the combined own-library predicate."""

    def missing(self, path):
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
        with mock.patch.object(a2a3_cache_module.os, "stat", side_effect=PermissionError("denied")):
            self.assertFalse(self.missing(path))

    def test_not_a_directory_is_not_missing(self):
        (self.root / "plain.txt").write_bytes(b"x")
        self.assertFalse(self.missing(self.root / "plain.txt" / "child" / "kernel_lib.so"))


class TestMissingOwnLibraryError(_Files):
    """Original loader-message examples on the combined own-library predicate."""

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
        logger = logging.getLogger(cache_module.__name__)
        self.addCleanup(logger.setLevel, logger.level)
        patches = [
            mock.patch.object(
                cache_module,
                "env",
                SimpleNamespace(
                    TILELANG_CACHE_DIR=str(self.root / "cache"),
                    is_cache_enabled=lambda: True,
                    should_use_kernel_cache_lib_stamp=lambda: False,
                ),
            ),
            mock.patch.object(KernelCache, "_instance", None),
            mock.patch.object(A2A3CythonKernelCache, "_instance", None),
            mock.patch.object(KernelCache, "_last_cleaned_staging_root", KernelCache._last_cleaned_staging_root),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.cache = KernelCache()
        self.context = create_backend_context("llvm", execution_backend="tvm_ffi")
        buffer = tvm.tirx.decl_buffer((4,), "float32", name="A")
        self.func = tvm.tirx.PrimFunc([buffer.data], tvm.tirx.Evaluate(0), buffer_map={buffer.data: buffer}).with_attr(
            "global_symbol", "cache_state_unit"
        )
        self.library = self.root / "compiled.so"
        self.library.write_bytes(b"unit state fixture; not a compiled kernel")
        self.kernel = SimpleNamespace(
            params=[KernelParam.from_buffer(buffer)],
            execution_backend="tvm_ffi",
            kernel_source="// kernel",
            adapter=SimpleNamespace(
                libpath=str(self.library), get_kernel_source=lambda: "// wrapper", get_host_source=lambda: "// wrapper"
            ),
        )

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
            mock.patch.object(cache_module.JITKernel, "from_database", side_effect=fail) as loader,
            self.assertLogs(self.cache.logger, level="WARNING") as logged,
        ):
            self.assertIsNone(self.load(key))
        loader.assert_called_once()
        self.assertEqual(Path(loader.call_args.kwargs["kernel_lib_path"]), library)
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
        backend = get_backend("ascend_a2a3")
        host = tvm.target.Target("c")
        target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}, host=host)
        context = BackendContext(backend, target, host, backend.execution_backends[0])
        self.assertEqual(context.module.name, "ascend_a2a3")
        args = CompileArgs(target=target, execution_backend="cython", out_idx=[0], _frozen=_FrozenCompileRequest(context))
        entry = self.result_path(key)
        lib = self.strict_entry(entry, context)

        def operation():
            return AutotuneResult()._load_kernel_from_disk(entry, context, func=self.func, compile_args=args)

        # Preparation data only: the build boundary is mocked, so this does not
        # certify compiler metadata or ABI, and never discovers a toolchain.
        prepared = (args.out_idx, {}, None, SimpleNamespace(to_metadata=lambda: {}))
        with mock.patch.object(CompileArgs, "_prepare_a2a3_program", return_value=prepared) as prepare:
            with mock.patch.object(A2A3CythonKernelCache, "_build_kernel", side_effect=lambda *a, **kw: fail(lib)) as loader:
                if returns_none:
                    self.assertIsNone(operation())
                else:
                    with self.assertRaisesRegex(OSError, "wrong ELF class: ELFCLASS32"):
                        operation()
            loader.assert_called_once()
            self.assertEqual(Path(loader.call_args.args[3]), lib)
            prepare.assert_called_once_with(self.func)
            if returns_none:
                lib.write_bytes(b"restored fixture library")
                prepare.reset_mock()
                with mock.patch.object(A2A3CythonKernelCache, "_build_kernel", return_value=self.kernel) as loader:
                    self.assertIs(operation(), self.kernel)
                loader.assert_called_once()
                self.assertEqual(Path(loader.call_args.args[3]), lib)
                prepare.assert_called_once_with(self.func)

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

        with mock.patch.object(cache_module.JITKernel, "from_database", side_effect=delete_then_fail) as loader:
            self.assertIsNone(self.load(key))
        loader.assert_called_once()
        self.write_entry(key)
        with mock.patch.object(cache_module.JITKernel, "from_database", return_value=self.kernel) as loader:
            self.assertIs(self.load(key), self.kernel)
        loader.assert_called_once()

    def test_oserror_without_errno_is_recovered(self):
        def error(lib):
            value = OSError(f"{lib}: cannot open shared object file: No such file or directory")
            value.errno = None
            return value

        self.shared_error_is_miss("k_noerrno", error, delete=True)

    def test_permission_error_is_a_cache_miss(self):
        self.shared_error_is_miss("k_perm", PermissionError("permission denied"))

    def test_unloadable_library_is_a_cache_miss(self):
        self.shared_error_is_miss("k_elf", OSError("wrong ELF class: ELFCLASS32"))

    def test_missing_entry_directory_returns_none(self):
        self.assertIsNone(self.load("k_nothing"))

    def test_cold_and_hot_cache_do_not_reload(self):
        with mock.patch.object(self.cache, "_load_kernel_from_disk", return_value=self.kernel) as load:
            self.assertIs(self.cached(), self.kernel)
            self.assertIs(self.cached(), self.kernel)
        load.assert_called_once()

    def test_permission_error_with_deletion_is_a_cache_miss(self):
        self.shared_error_is_miss("k_perm_del", PermissionError("permission denied"), delete=True)

    def test_dependency_error_with_deletion_is_a_cache_miss(self):
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

    def test_truncated_library_with_deletion_is_a_cache_miss(self):
        self.shared_error_is_miss("k_trunc", lambda lib: OSError(f"{lib}: file too short"), delete=True)

    def test_recreated_library_loader_error_is_a_cache_miss(self):
        self.shared_error_is_miss(
            "k_recreated", lambda lib: OSError(f"{lib}: cannot open shared object file: No such file or directory"), recreate=True
        )

    def test_makedirs_failure_propagates(self):
        with (
            mock.patch.object(self.cache, "_load_kernel_from_disk", return_value=None),
            mock.patch.object(cache_module, "JITKernel", return_value=self.kernel) as constructor,
            mock.patch("os.makedirs", side_effect=FileNotFoundError("cache directory deleted")),
            self.assertRaisesRegex(FileNotFoundError, "cache directory deleted"),
        ):
            self.cached()
        constructor.assert_called_once()

    def test_makedirs_permission_error_propagates(self):
        with (
            mock.patch.object(self.cache, "_load_kernel_from_disk", return_value=None),
            mock.patch.object(cache_module, "JITKernel", return_value=self.kernel) as constructor,
            mock.patch("os.makedirs", side_effect=PermissionError("permission denied")),
            self.assertRaisesRegex(PermissionError, "permission denied"),
        ):
            self.cached()
        constructor.assert_called_once()


class TestAutotuneResultRecovery(KernelCacheRecoveryMixin):
    def test_recovers_when_library_deleted_before_load(self):
        self.strict_deleted_library("a_del")

    def test_unloadable_library_is_not_a_miss(self):
        self.strict_wrong_elf("a_elf")

    def test_missing_metadata_propagates(self):
        entry = self.result_path("meta_del")
        self.result().save_to_disk(entry)
        (entry / param_module.BEST_CONFIG_PATH).unlink()
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

    def test_save_to_disk_serialization_error_logs_and_discards_staging(self):
        entry = self.result_path("ser_err")

        class ReadsMissingFile:
            def __reduce__(self):
                open("/definitely/does/not/exist_xyz.pkl", "rb")
                return (self.__class__, ())

        with self.assertLogs(param_module.logger, level="ERROR") as logs:
            self.result(func=ReadsMissingFile()).save_to_disk(entry)
        self.assertTrue(any("exist_xyz.pkl" in message for message in logs.output))
        self.assertFalse(entry.exists())
        self.assertFalse(any((entry.parent.parent / ".staging").iterdir()))

    def test_save_to_disk_makedirs_deletion_propagates(self):
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
