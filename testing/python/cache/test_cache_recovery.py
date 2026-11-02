# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Host-only cache-load recovery tests; no compiler runtime or device is required.

These tests verify that a cache entry whose ``kernel_lib.so`` disappears while it
is being loaded (for example, by a concurrent ``clear_cache`` in another process)
falls back to a cache miss instead of raising. They are deterministic: a mocked
``JITKernel.from_database`` deletes the library and raises ``OSError`` at the
exact point the real ``ctypes.CDLL`` would fail.
"""

import ctypes
import importlib.util
import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[3]
FILE_HELPER_PATH = REPO_ROOT / "tilelang" / "utils" / "file.py"
KERNEL_CACHE_PATH = REPO_ROOT / "tilelang" / "cache" / "kernel_cache.py"
TUNER_CACHE_PATH = REPO_ROOT / "tilelang" / "cache" / "tuner_cache.py"
AUTOTUNER_PARAM_PATH = REPO_ROOT / "tilelang" / "autotuner" / "param.py"


def _load_standalone(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass processing (``sys.modules[name]``) works.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _make_module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def install_tilelang_mocks(tmp_cache_dir: str):
    """Install minimal tilelang/tvm/cloudpickle modules into ``sys.modules``.

    The cache modules are loaded directly from source with
    ``spec_from_file_location``; their imports resolve against these stand-ins.
    ``JITKernel`` is a ``MagicMock`` so ``JITKernel.from_database`` (the real
    point at which the shared library is loaded) can be patched per test.
    """
    saved = {name: sys.modules.get(name) for name in []}

    file_module = _load_standalone(FILE_HELPER_PATH, "tilelang_cache_file_helpers")

    tvm = _make_module(
        "tvm",
        get_global_func=lambda name, allow_missing=False: (
            (lambda: "test-engine-default") if name == "tl.ascend_sync.EngineCacheABI" else None
        ),
    )
    tvm_target = _make_module("tvm.target", Target=object)
    tvm_tir = _make_module("tvm.tir", PrimFunc=object)
    tvm.target = tvm_target
    tvm.tir = tvm_tir

    jit_kernel = mock.MagicMock()
    tilelang_jit = _make_module("tilelang.jit", JITKernel=jit_kernel)
    # File-race recovery tests isolate synchronization semantics. The dedicated
    # Ascend contract fixtures replace this stub with the real ABI validator.
    tilelang_contract = _make_module("tilelang.jit.ascend_runtime_contract", sync_engine_cache_abi=lambda *args, **kwargs: None)
    tilelang_engine = _make_module("tilelang.engine")
    tilelang_engine_param = _make_module("tilelang.engine.param", KernelParam=object)
    tilelang_engine.param = tilelang_engine_param
    tilelang_env = _make_module("tilelang.env", TILELANG_CACHE_DIR=tmp_cache_dir, is_cache_enabled=lambda: True)
    tilelang_version = _make_module("tilelang.version", __version__="test")
    tilelang_utils = _make_module("tilelang.utils")
    tilelang_utils_target = _make_module("tilelang.utils.target", determine_platform=lambda platform: platform or "A3")
    tilelang_utils.target = tilelang_utils_target
    tilelang_utils.file = file_module

    tilelang = _make_module(
        "tilelang",
        jit=tilelang_jit,
        engine=tilelang_engine,
        env=tilelang_env,
        version=tilelang_version,
        utils=tilelang_utils,
        tvm=tvm,
        logger=mock.MagicMock(),
        TensorSupplyType=types.SimpleNamespace(Auto="auto"),
    )

    cloudpickle = _make_module(
        "cloudpickle",
        load=pickle.load,
        dump=pickle.dump,
        dumps=pickle.dumps,
    )

    modules = {
        "tvm": tvm,
        "tvm.target": tvm_target,
        "tvm.tir": tvm_tir,
        "tilelang": tilelang,
        "tilelang.jit": tilelang_jit,
        "tilelang.jit.ascend_runtime_contract": tilelang_contract,
        "tilelang.engine": tilelang_engine,
        "tilelang.engine.param": tilelang_engine_param,
        "tilelang.env": tilelang_env,
        "tilelang.version": tilelang_version,
        "tilelang.utils": tilelang_utils,
        "tilelang.utils.target": tilelang_utils_target,
        "tilelang.utils.file": file_module,
        "cloudpickle": cloudpickle,
    }
    saved = {name: sys.modules.get(name) for name in modules}
    for name, module in modules.items():
        sys.modules[name] = module
    return modules, saved, file_module, jit_kernel


def uninstall_tilelang_mocks(saved):
    for name, previous in saved.items():
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


class TestIsFileMissing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.helper = _load_standalone(FILE_HELPER_PATH, "tilelang_is_file_missing")

    def test_missing_entry_is_missing(self):
        self.assertTrue(self.helper.is_file_missing(self.root / "nope.so"))

    def test_existing_entry_is_not_missing(self):
        path = self.root / "present.so"
        path.write_bytes(b"x")
        self.assertFalse(self.helper.is_file_missing(path))

    def test_permission_error_is_not_missing(self):
        path = self.root / "present.so"
        path.write_bytes(b"x")
        with mock.patch.object(self.helper.os, "stat", side_effect=PermissionError("denied")):
            self.assertFalse(self.helper.is_file_missing(path))

    def test_not_a_directory_is_not_missing(self):
        (self.root / "plain.txt").write_bytes(b"x")
        path = self.root / "plain.txt" / "child" / "kernel_lib.so"
        self.assertFalse(self.helper.is_file_missing(path))


class TestErrorReportsMissingPath(unittest.TestCase):
    """The loader-error classifier distinguishes target-missing from bad/foreign files."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.helper = _load_standalone(FILE_HELPER_PATH, "tilelang_error_reports_missing_path")

    def _dl_error(self, path):
        try:
            ctypes.CDLL(str(path))
        except OSError as error:
            return error
        self.fail(f"loading {path} unexpectedly succeeded")

    def test_real_missing_target_is_reported(self):
        missing = self.root / "kernel_lib.so"
        self.assertTrue(self.helper.error_reports_missing_path(self._dl_error(missing), missing))

    def test_real_truncated_library_is_not_reported(self):
        bad = self.root / "kernel_lib.so"
        bad.write_bytes(b"")  # dlopen reports "file too short"
        self.assertFalse(self.helper.error_reports_missing_path(self._dl_error(bad), bad))

    def test_dependency_name_containing_target_is_not_reported(self):
        target = self.root / "kernel_lib.so"
        error = OSError("libkernel_lib.so: cannot open shared object file: No such file or directory")
        self.assertFalse(self.helper.error_reports_missing_path(error, target))

    def test_same_basename_in_other_directory_is_not_reported(self):
        target = self.root / "kernel_lib.so"
        other = self.root / "other" / "kernel_lib.so"
        error = OSError(f"{other}: cannot open shared object file: No such file or directory")
        self.assertFalse(self.helper.error_reports_missing_path(error, target))


class KernelCacheRecoveryMixin:
    """Shared fixtures and assertions for the three cache-load entry points."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.modules, self.saved, self.file_helper, self.jit_kernel = install_tilelang_mocks(self.tmp.name)
        self.addCleanup(uninstall_tilelang_mocks, self.saved)

    def write_entry(self, cache, key, with_lib=True):
        """Write a minimal cache entry and return ``(entry_dir, lib_path)``."""
        entry = Path(cache._get_cache_path(key))
        entry.mkdir(parents=True, exist_ok=True)
        (entry / "wrapped_kernel.cu").write_text("// kernel")
        with open(entry / "params.pkl", "wb") as f:
            pickle.dump([1], f)
        (entry / "auto_gm_idx.pkl").write_text("[]")
        lib = entry / "kernel_lib.so"
        if with_lib:
            lib.write_bytes(b"fake library")
        return entry, lib

    def assert_recovers_to_miss(self, cache, key):
        fake = object()

        def delete_then_fail(**kwargs):
            lib_path = kwargs["kernel_lib_path"]
            Path(lib_path).unlink()
            raise OSError(f"{lib_path}: cannot open shared object file: No such file or directory")

        with mock.patch.object(self.jit_kernel, "from_database", side_effect=delete_then_fail):
            self.assertIsNone(cache._load_kernel_from_disk(key))

        # A retry with the library restored does not hit the deleted inode.
        entry, lib = self.write_entry(cache, key, with_lib=True)
        self.jit_kernel.from_database.return_value = fake
        self.assertIs(cache._load_kernel_from_disk(key), fake)

    def assert_unloadable_library_reraises(self, cache, key):
        entry, lib = self.write_entry(cache, key, with_lib=True)

        def fail_without_deleting(**kwargs):
            raise OSError("wrong ELF class: ELFCLASS32")

        with mock.patch.object(self.jit_kernel, "from_database", side_effect=fail_without_deleting), self.assertRaises(OSError):
            cache._load_kernel_from_disk(key)


class TestKernelCacheRecovery(KernelCacheRecoveryMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        # Cache recovery uses mock functions; allocator contract semantics are
        # covered with the real helper in test_ascend_runtime_contract_cache.py.
        contract_module = "tilelang.jit.ascend_runtime_contract"
        self.modules[contract_module].prepare_auto_gm_contract = lambda func, *_args: func
        self.kernel_cache = _load_standalone(KERNEL_CACHE_PATH, "kernel_cache_under_test")
        self.kernel_cache.KernelCache._instance = None
        self.cache = self.kernel_cache.KernelCache(cache_dir=self.tmp.name)

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

        with mock.patch.object(self.jit_kernel, "from_database", side_effect=delete_then_fail):
            self.assertIsNone(self.cache._load_kernel_from_disk(key))

    def test_permission_error_is_not_a_miss(self):
        key = "k_perm"
        self.write_entry(self.cache, key, with_lib=True)

        def fail_permission(**kwargs):
            raise PermissionError("permission denied")

        with mock.patch.object(self.jit_kernel, "from_database", side_effect=fail_permission), self.assertRaises(PermissionError):
            self.cache._load_kernel_from_disk(key)

    def test_unloadable_library_is_not_a_miss(self):
        self.assert_unloadable_library_reraises(self.cache, "k_elf")

    def test_missing_entry_directory_returns_none(self):
        self.assertIsNone(self.cache._load_kernel_from_disk("k_nothing"))

    def test_cold_and_hot_cache_do_not_reload(self):
        fake = object()
        func = mock.MagicMock()
        func.script.return_value = "script"

        with mock.patch.object(self.cache, "_load_kernel_from_disk", return_value=fake) as load:
            first = self.cache.cached(func=func)
            second = self.cache.cached(func=func)
        self.assertIs(first, fake)
        self.assertIs(second, fake)
        load.assert_called_once()

    def test_permission_error_with_deletion_is_not_a_miss(self):
        key = "k_perm_del"
        self.write_entry(self.cache, key, with_lib=True)

        def fail_permission_and_delete(**kwargs):
            Path(kwargs["kernel_lib_path"]).unlink()
            raise PermissionError("permission denied")

        with (
            mock.patch.object(self.jit_kernel, "from_database", side_effect=fail_permission_and_delete),
            self.assertRaises(PermissionError),
        ):
            self.cache._load_kernel_from_disk(key)

    def test_dependency_error_with_deletion_is_not_a_miss(self):
        key = "k_dep_del"
        self.write_entry(self.cache, key, with_lib=True)

        def fail_dependency_and_delete(**kwargs):
            Path(kwargs["kernel_lib_path"]).unlink()
            # The dependency's name contains the target basename as a substring.
            error = OSError("libkernel_lib.so: cannot open shared object file: No such file or directory")
            error.errno = None
            raise error

        with mock.patch.object(self.jit_kernel, "from_database", side_effect=fail_dependency_and_delete), self.assertRaises(OSError):
            self.cache._load_kernel_from_disk(key)

    def test_miss_compiles_and_returns_result(self):
        func = mock.MagicMock()
        func.script.return_value = "script"
        with (
            mock.patch.object(self.cache, "_load_kernel_from_disk", return_value=None),
            mock.patch.object(self.cache, "_save_kernel_to_disk"),
        ):
            result = self.cache.cached(func=func)
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
            mock.patch.object(self.jit_kernel, "from_database", return_value=fake_kernel),
        ):
            result = self.cache.cached(func=func)
        self.assertIs(result, fake_kernel)

    def test_truncated_library_with_deletion_is_not_a_miss(self):
        key = "k_trunc"
        self.write_entry(self.cache, key, with_lib=True)

        def fail_truncated_and_delete(**kwargs):
            lib_path = kwargs["kernel_lib_path"]
            Path(lib_path).unlink()
            raise OSError(f"{lib_path}: file too short")

        with mock.patch.object(self.jit_kernel, "from_database", side_effect=fail_truncated_and_delete), self.assertRaises(OSError):
            self.cache._load_kernel_from_disk(key)

    def test_recreated_library_after_deletion_reraises(self):
        key = "k_recreated"
        self.write_entry(self.cache, key, with_lib=True)

        def delete_recreate_and_fail(**kwargs):
            lib_path = Path(kwargs["kernel_lib_path"])
            lib_path.unlink()
            lib_path.write_bytes(b"recreated")  # reappears before the filesystem check
            raise OSError(f"{lib_path}: cannot open shared object file: No such file or directory")

        with mock.patch.object(self.jit_kernel, "from_database", side_effect=delete_recreate_and_fail), self.assertRaises(OSError):
            self.cache._load_kernel_from_disk(key)

    def test_makedirs_failure_still_returns_result(self):
        func = mock.MagicMock()
        func.script.return_value = "script"
        with (
            mock.patch.object(self.cache, "_load_kernel_from_disk", return_value=None),
            mock.patch("os.makedirs", side_effect=FileNotFoundError("cache directory deleted")),
        ):
            result = self.cache.cached(func=func)
        self.assertIs(result, self.jit_kernel.return_value)

    def test_makedirs_permission_error_propagates(self):
        func = mock.MagicMock()
        func.script.return_value = "script"
        with (
            mock.patch.object(self.cache, "_load_kernel_from_disk", return_value=None),
            mock.patch("os.makedirs", side_effect=PermissionError("permission denied")),
            self.assertRaises(PermissionError),
        ):
            self.cache.cached(func=func)


class TestTunerCacheRecovery(KernelCacheRecoveryMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.tuner_cache = _load_standalone(TUNER_CACHE_PATH, "tuner_cache_under_test")
        self.tuner_cache.AutoTunerCache._instance = None
        self.cache = self.tuner_cache.AutoTunerCache(cache_dir=self.tmp.name)

    def test_recovers_when_library_deleted_before_load(self):
        key = "t_del"
        self.write_entry(self.cache, key, with_lib=True)
        self.assert_recovers_to_miss(self.cache, key)

    def test_unloadable_library_is_not_a_miss(self):
        self.assert_unloadable_library_reraises(self.cache, "t_elf")

    def test_full_control_flow(self):
        func = mock.MagicMock()
        func.script.return_value = "script"
        with (
            mock.patch.object(self.cache, "_load_kernel_from_disk", return_value=None) as load,
            mock.patch.object(self.cache, "_save_kernel_to_disk") as save,
        ):
            result = self.cache.cached(func=func)
        self.assertIs(result, self.jit_kernel.return_value)
        self.assertEqual(load.call_count, 2)
        save.assert_called_once()


class TestAutotuneResultRecovery(KernelCacheRecoveryMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.param = _load_standalone(AUTOTUNER_PARAM_PATH, "autotuner_param_under_test")
        self.result_cls = self.param.AutotuneResult

    def _load(self, cache_path, **kwargs):
        return self.result_cls._load_kernel_from_disk(self.result_cls, cache_path, **kwargs)

    def test_recovers_when_library_deleted_before_load(self):
        entry = Path(self.tmp.name) / "a_del"
        entry.mkdir(parents=True, exist_ok=True)
        (entry / "wrapped_kernel.cu").write_text("// kernel")
        with open(entry / "params.pkl", "wb") as f:
            pickle.dump([1], f)
        (entry / "kernel_lib.so").write_bytes(b"fake library")

        fake = object()

        def delete_then_fail(**kwargs):
            lib_path = kwargs["kernel_lib_path"]
            Path(lib_path).unlink()
            raise OSError(f"{lib_path}: cannot open shared object file: No such file or directory")

        with mock.patch.object(self.jit_kernel, "from_database", side_effect=delete_then_fail):
            self.assertIsNone(self._load(entry))

        self.jit_kernel.from_database.return_value = fake
        (entry / "kernel_lib.so").write_bytes(b"fake library")
        self.assertIs(self._load(entry), fake)

    def test_unloadable_library_is_not_a_miss(self):
        entry = Path(self.tmp.name) / "a_elf"
        entry.mkdir(parents=True, exist_ok=True)
        (entry / "wrapped_kernel.cu").write_text("// kernel")
        with open(entry / "params.pkl", "wb") as f:
            pickle.dump([1], f)
        (entry / "kernel_lib.so").write_bytes(b"fake library")

        def fail_without_deleting(**kwargs):
            raise OSError("wrong ELF class: ELFCLASS32")

        with mock.patch.object(self.jit_kernel, "from_database", side_effect=fail_without_deleting), self.assertRaises(OSError):
            self._load(entry)

    def test_load_from_disk_metadata_deleted_returns_none(self):
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

        self.assertIsNone(self.result_cls.load_from_disk(entry, self.param.CompileArgs()))

    def test_save_to_disk_survives_cache_deletion(self):
        entry = Path(self.tmp.name) / "save_del"
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

    def test_save_to_disk_serialization_error_propagates(self):
        entry = Path(self.tmp.name) / "ser_err"
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
        # Serialization-internal FileNotFoundError must propagate, not be swallowed
        # by the cache-open recovery.
        with self.assertRaises(FileNotFoundError):
            result.save_to_disk(entry)

    def test_save_to_disk_makedirs_deletion_skips_save(self):
        entry = Path(self.tmp.name) / "mk_del"
        result = self.result_cls(
            config={"x": 1},
            func="fn",
            kernel=mock.MagicMock(),
            latency=0.1,
            ref_latency=0.2,
        )
        with mock.patch("os.makedirs", side_effect=FileNotFoundError("cache directory deleted")):
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
