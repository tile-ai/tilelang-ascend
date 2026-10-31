# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Cache reconstruction must validate the Ascend allocator promise before loading.

Use the real cache loader and contract validator with the existing host-only
cache fixtures. Only the TVM StringImm type and database adapter are stand-ins;
no compiler runtime, device, or shared-library loader is needed.
"""

import json
import pickle
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from test_cache_recovery import (
    KERNEL_CACHE_PATH,
    REPO_ROOT,
    KernelCacheRecoveryMixin,
    _load_standalone,
)

CONTRACT_MODULE = "tilelang.jit.ascend_runtime_contract"
CONTRACT_PATH = REPO_ROOT / "tilelang" / "jit" / "ascend_runtime_contract.py"


class _StringImm:
    def __init__(self, value: str):
        self.value = value


class _ManifestCacheChecks:
    def test_upstream_writer_and_loader_bind_the_native_engine_abi(self):
        configs = {"tl.ascend_sync_backend": "upstream"}
        version = ["test-engine-v1"]
        with mock.patch.object(
            self.modules["tvm"],
            "get_global_func",
            create=True,
            return_value=lambda: version[0],
        ):
            kernel = self._compiled_kernel()
            kernel.pass_configs = configs
            self._save(kernel)
            manifest = json.loads((self.entry / self.contract.AUTO_GM_MANIFEST_PATH).read_text())
            self.assertEqual(manifest["schema"], 2)
            self.assertEqual(manifest["sync_engine_abi"], "test-engine-v1")
            self.assertIs(self._load(pass_configs=configs), self.kernel)
            self.assertEqual(self.jit_kernel.from_database.call_count, 1)
            version[0] = "test-engine-v2"
            self.assertIsNone(self._load(pass_configs=configs))
            self.assertEqual(self.jit_kernel.from_database.call_count, 1)

    def test_unmarked_upstream_artifact_still_requires_matching_engine_abi(self):
        configs = {"tl.ascend_sync_backend": "upstream"}
        self.func.attrs = {}
        version = ["test-engine-v1"]
        with mock.patch.object(
            self.modules["tvm"],
            "get_global_func",
            create=True,
            return_value=lambda: version[0],
        ):
            self._write_indices([])
            manifest = self.contract.build_cached_auto_gm_manifest(self.func, "ctypes", [], 1, configs)
            (self.entry / self.contract.AUTO_GM_MANIFEST_PATH).write_text(json.dumps(manifest))
            self.assertIs(self._load(pass_configs=configs, execution_backend="ctypes"), self.kernel)
            version[0] = "test-engine-v2"
            self.assertIsNone(self._load(pass_configs=configs, execution_backend="ctypes"))
            self.assertEqual(self.jit_kernel.from_database.call_count, 1)

    def test_missing_upstream_native_abi_requires_rebuild_instead_of_a_saltless_load(self):
        with (
            mock.patch.object(self.modules["tvm"], "get_global_func", create=True, return_value=None),
            self.assertRaisesRegex(RuntimeError, "rebuild the native TileLang"),
        ):
            self._load(pass_configs={"tl.ascend_sync_backend": "upstream"})
        self.jit_kernel.from_database.assert_not_called()

    def test_legacy_ascend_writer_and_loader_bind_the_shared_native_abi(self):
        version = ["legacy-policy-v1"]
        with mock.patch.object(self.modules["tvm"], "get_global_func", create=True, return_value=lambda: version[0]):
            for target in ("ascendc", "pto"):
                for configs in (None, {"tl.ascend_sync_backend": "legacy"}):
                    with self.subTest(target=target, configs=configs):
                        version[0] = "legacy-policy-v1"
                        self.jit_kernel.from_database.reset_mock()
                        kernel = self._compiled_kernel()
                        kernel.target = target
                        kernel.pass_configs = configs
                        self._save(kernel)
                        manifest = json.loads((self.entry / self.contract.AUTO_GM_MANIFEST_PATH).read_text())
                        self.assertEqual(manifest["schema"], 2)
                        self.assertEqual(manifest["sync_engine_abi"], "legacy-policy-v1")
                        self.assertIs(self._load(target=target, pass_configs=configs), self.kernel)
                        version[0] = "legacy-policy-v2"
                        self.assertIsNone(self._load(target=target, pass_configs=configs))
                        self.assertEqual(self.jit_kernel.from_database.call_count, 1)

    def test_unmarked_legacy_ascend_artifact_requires_an_engine_manifest(self):
        self.func.attrs = {}
        self._write_indices([])
        (self.entry / self.contract.AUTO_GM_MANIFEST_PATH).unlink()
        self.assertIsNone(self._load(execution_backend="ctypes"))
        self.jit_kernel.from_database.assert_not_called()

    def test_old_legacy_schema_one_is_a_cache_miss(self):
        old = self.contract.build_cached_auto_gm_manifest(self.func, "cython", [], 1)
        self.assertEqual(old["schema"], 1)
        (self.entry / self.contract.AUTO_GM_MANIFEST_PATH).write_text(json.dumps(old))
        self.assertIsNone(self._load())
        self.jit_kernel.from_database.assert_not_called()

    def _compiled_kernel(self):
        library = Path(self.tmp.name) / "compiled_source.so"
        library.write_bytes(b"new compiled library")
        source = "// new wrapped kernel"
        return SimpleNamespace(
            artifact=SimpleNamespace(kernel_source="// new kernel"),
            adapter=SimpleNamespace(libpath=str(library), get_kernel_source=lambda: source),
            kernel_source="// new kernel",
            get_kernel_source=lambda: source,
            prim_func=self.func,
            execution_backend="cython",
            target="auto",
            params=[1, 2, 3, 4],
            auto_gm_idx=[3],
        )

    def _write_manifest(self, indices, param_count=1, **changes):
        manifest = self.contract.build_cached_auto_gm_manifest(self.func, "cython", indices, param_count, target="auto")
        manifest.update(changes)
        (self.entry / self.contract.AUTO_GM_MANIFEST_PATH).write_text(json.dumps(manifest))

    def test_missing_manifest_is_a_miss(self):
        (self.entry / self.contract.AUTO_GM_MANIFEST_PATH).unlink()
        self.assertIsNone(self._load())
        self.jit_kernel.from_database.assert_not_called()

    def test_unparseable_manifest_is_a_miss(self):
        (self.entry / self.contract.AUTO_GM_MANIFEST_PATH).write_text("{not valid JSON")
        self.assertIsNone(self._load())
        self.jit_kernel.from_database.assert_not_called()

    def test_null_manifest_is_a_miss(self):
        (self.entry / self.contract.AUTO_GM_MANIFEST_PATH).write_text("null")
        self.assertIsNone(self._load())
        self.jit_kernel.from_database.assert_not_called()

    def test_metadata_cannot_silently_remove_a_manifest_workspace(self):
        self._write_indices([])
        self._write_manifest([0])
        self.assertIsNone(self._load())
        self.jit_kernel.from_database.assert_not_called()

    def test_manifest_from_another_source_is_a_miss(self):
        self._write_manifest([], source_hash="0" * 64)
        self.assertIsNone(self._load())
        self.jit_kernel.from_database.assert_not_called()

    def test_manifest_for_another_lowered_signature_is_a_miss(self):
        self._write_manifest([], param_count=2)
        self.assertIsNone(self._load())
        self.jit_kernel.from_database.assert_not_called()

    def test_unknown_or_boolean_manifest_schema_is_a_miss(self):
        for schema in (1, True):
            with self.subTest(schema=schema):
                self._write_manifest([], schema=schema)
                self.assertIsNone(self._load())
        self.jit_kernel.from_database.assert_not_called()

    def test_manifest_table_cannot_use_a_boolean_equal_to_an_integer_index(self):
        self._write_indices([0])
        self._write_manifest([0], auto_gm_indices=[False])
        self.assertIsNone(self._load())
        self.jit_kernel.from_database.assert_not_called()

    def test_lowered_workspace_parameter_is_not_limited_to_original_parameter_count(self):
        # Lowering appends the automatic workspace to the original three inputs.
        self.assertEqual(len(self.func.params), 3)
        with open(self.entry / "params.pkl", "wb") as stream:
            pickle.dump([1, 2, 3, 4], stream)
        self._write_indices([3])
        self._write_manifest([3], param_count=4)
        self.assertIs(self._load(), self.kernel)
        self.jit_kernel.from_database.assert_called_once()
        kwargs = self.jit_kernel.from_database.call_args.kwargs
        self.assertEqual(kwargs["auto_gm_idx"], [3])
        self.assertEqual(len(kwargs["params"]), 4)

    def test_writer_persists_the_manifest_for_a_complete_cache_roundtrip(self):
        manifest_path = self.entry / self.contract.AUTO_GM_MANIFEST_PATH
        manifest_path.unlink()
        (self.entry / "auto_gm_idx.pkl").unlink()
        kernel = self._compiled_kernel()
        self._save(kernel)

        manifest = json.loads(manifest_path.read_text())
        expected = self.contract.build_cached_auto_gm_manifest(self.func, "cython", [3], 4, target="auto")
        self.assertEqual(manifest, expected)
        self.assertEqual(json.loads((self.entry / "auto_gm_idx.pkl").read_text()), [3])
        self.assertEqual((self.entry / "kernel_lib.so").read_bytes(), b"new compiled library")
        self.assertIs(self._load(), self.kernel)
        self.jit_kernel.from_database.assert_called_once()
        kwargs = self.jit_kernel.from_database.call_args.kwargs
        self.assertEqual(kwargs["auto_gm_manifest"], expected)
        self.assertEqual(kwargs["auto_gm_idx"], [3])
        self.assertEqual(len(kwargs["params"]), 4)


class _KeyedContractWriterChecks:
    def test_writer_skips_marked_artifact_under_an_unprepared_legacy_key(self):
        key = "unprepared_legacy_key"
        entry = Path(self.cache._get_cache_path(key))
        self.assertFalse(entry.exists())
        self.cache._save_kernel_to_disk(key, self._compiled_kernel(), SimpleNamespace(attrs={}))
        self.assertFalse(entry.exists())


class TestAscendRuntimeContractCache(_KeyedContractWriterChecks, _ManifestCacheChecks, KernelCacheRecoveryMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.modules["tvm.tir"].StringImm = _StringImm
        # The recovery fixture deliberately makes tilelang.jit a non-package.
        # Register only this real helper explicitly, preserving that isolation.
        self.saved.setdefault(CONTRACT_MODULE, sys.modules.get(CONTRACT_MODULE))
        self.contract = _load_standalone(CONTRACT_PATH, CONTRACT_MODULE)
        kernel_cache = _load_standalone(KERNEL_CACHE_PATH, "ascend_contract_cache_under_test")
        kernel_cache.KernelCache._instance = None
        self.cache = kernel_cache.KernelCache(cache_dir=self.tmp.name)
        self.key = "ascend_contract"
        self.entry, _ = self.write_entry(self.cache, self.key)
        self.kernel = object()
        self.jit_kernel.from_database.return_value = self.kernel
        self.func = SimpleNamespace(
            params=["A", "B", "C"],
            script=lambda: "marked Ascend source(A, B, C): cython-auto-gm-v1",
            attrs={self.contract.AUTO_GM_FRESH_CONTRACT_ATTR: _StringImm(self.contract.AUTO_GM_FRESH_CONTRACT_VERSION)},
        )
        self._write_manifest([])

    def _write_indices(self, indices):
        (self.entry / "auto_gm_idx.pkl").write_text(json.dumps(indices))

    def _load(self, **kwargs):
        return self.cache._load_kernel_from_disk(self.key, func=self.func, **kwargs)

    def _save(self, kernel):
        self.cache._save_kernel_to_disk(self.key, kernel, self.func)

    def test_missing_metadata_is_a_miss_even_with_caller_fallback(self):
        (self.entry / "auto_gm_idx.pkl").unlink()
        # A caller-supplied list does not establish what the cached artifact used.
        self.assertIsNone(self._load(auto_gm_idx=[0]))
        self.jit_kernel.from_database.assert_not_called()

    def test_null_metadata_is_a_miss(self):
        self._write_indices(None)
        self.assertIsNone(self._load(auto_gm_idx=[]))
        self.jit_kernel.from_database.assert_not_called()

    def test_unreadable_metadata_is_a_miss(self):
        (self.entry / "auto_gm_idx.pkl").write_text("{not valid JSON")
        self.assertIsNone(self._load(auto_gm_idx=[]))
        self.jit_kernel.from_database.assert_not_called()

    def test_contract_for_non_cython_backend_is_a_miss(self):
        for backend in ("ctypes", "dlpack"):
            with self.subTest(backend=backend):
                self.assertIsNone(self._load(execution_backend=backend))
        self.jit_kernel.from_database.assert_not_called()

    def test_unknown_schema_is_a_miss(self):
        self.func.attrs[self.contract.AUTO_GM_FRESH_CONTRACT_ATTR] = _StringImm("cython-auto-gm-v2")
        self.assertIsNone(self._load())
        self.jit_kernel.from_database.assert_not_called()

    def test_wrong_schema_type_is_a_miss(self):
        self.func.attrs[self.contract.AUTO_GM_FRESH_CONTRACT_ATTR] = self.contract.AUTO_GM_FRESH_CONTRACT_VERSION
        self.assertIsNone(self._load())
        self.jit_kernel.from_database.assert_not_called()

    def test_invalid_indices_are_a_miss_before_adapter_reconstruction(self):
        # The fixture contains one lowered parameter, hence only index 0 is valid.
        for indices in ({}, [True], [-1], [1], [0, 0], ["0"]):
            with self.subTest(indices=indices):
                self._write_indices(indices)
                self.assertIsNone(self._load())
        self.jit_kernel.from_database.assert_not_called()

    def test_valid_empty_list_loads_without_fabricating_workspace(self):
        self._write_indices([])
        self.assertIs(self._load(), self.kernel)
        self.jit_kernel.from_database.assert_called_once()
        kwargs = self.jit_kernel.from_database.call_args.kwargs
        self.assertEqual(kwargs["auto_gm_idx"], [])
        self.assertIs(kwargs["func"], self.func)

    def test_valid_list_uses_cached_metadata_instead_of_caller_fallback(self):
        self._write_indices([0])
        self._write_manifest([0])
        self.assertIs(self._load(auto_gm_idx=[]), self.kernel)
        self.jit_kernel.from_database.assert_called_once()
        self.assertEqual(self.jit_kernel.from_database.call_args.kwargs["auto_gm_idx"], [0])

    def test_unmarked_nonascend_cache_loads_without_manifest(self):
        (self.entry / "auto_gm_idx.pkl").unlink()
        self.func = SimpleNamespace(attrs={})
        self.assertIs(self._load(target="cuda", execution_backend="ctypes"), self.kernel)
        self.jit_kernel.from_database.assert_called_once()
        kwargs = self.jit_kernel.from_database.call_args.kwargs
        self.assertEqual(kwargs["auto_gm_idx"], [])
        self.assertEqual(kwargs["execution_backend"], "ctypes")

    def test_nonascend_cache_without_function_loads_without_manifest(self):
        self.func = None
        self.assertIs(self._load(target="cuda"), self.kernel)
        self.jit_kernel.from_database.assert_called_once()


class _AdditionalContractLoaderMixin(_ManifestCacheChecks, KernelCacheRecoveryMixin):
    def setUp(self):
        super().setUp()
        self.modules["tvm.tir"].StringImm = _StringImm
        self.saved.setdefault(CONTRACT_MODULE, sys.modules.get(CONTRACT_MODULE))
        self.contract = _load_standalone(CONTRACT_PATH, CONTRACT_MODULE)
        self.func = SimpleNamespace(
            params=["A", "B", "C"],
            script=lambda: "marked Ascend source(A, B, C): cython-auto-gm-v1",
            attrs={self.contract.AUTO_GM_FRESH_CONTRACT_ATTR: _StringImm(self.contract.AUTO_GM_FRESH_CONTRACT_VERSION)},
        )
        self.kernel = object()
        self.jit_kernel.from_database.return_value = self.kernel

    def _write_indices(self, indices):
        (self.entry / "auto_gm_idx.pkl").write_text(json.dumps(indices))


class TestAscendAutotuneResultContractCache(_AdditionalContractLoaderMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        param = _load_standalone(
            REPO_ROOT / "tilelang" / "autotuner" / "param.py",
            "ascend_contract_autotuner_param_under_test",
        )
        self.result_cls = param.AutotuneResult
        paths = SimpleNamespace(_get_cache_path=lambda key: self.tmp.name)
        self.entry, _ = self.write_entry(paths, "artifact")
        self._write_manifest([])

    def _load(self, **kwargs):
        return self.result_cls._load_kernel_from_disk(self.result_cls, self.entry, func=self.func, **kwargs)

    def _save(self, kernel):
        self.result_cls._save_kernel_to_disk(self.result_cls, self.entry, kernel)

    def test_missing_metadata_is_a_miss(self):
        (self.entry / "auto_gm_idx.pkl").unlink()
        self.assertIsNone(self._load())
        self.jit_kernel.from_database.assert_not_called()

    def test_parse_failure_is_a_miss(self):
        (self.entry / "auto_gm_idx.pkl").write_text("{not valid JSON")
        self.assertIsNone(self._load())
        self.jit_kernel.from_database.assert_not_called()

    def test_null_metadata_is_a_miss(self):
        self._write_indices(None)
        self.assertIsNone(self._load())
        self.jit_kernel.from_database.assert_not_called()

    def test_invalid_raw_metadata_cannot_be_normalized_into_an_empty_list(self):
        for indices in ({}, False, 0, "", [True], [-1], [1], [0, 0]):
            with self.subTest(indices=indices):
                self._write_indices(indices)
                self.assertIsNone(self._load())
        self.jit_kernel.from_database.assert_not_called()

    def test_valid_empty_list_loads(self):
        self._write_indices([])
        self.assertIs(self._load(), self.kernel)
        self.jit_kernel.from_database.assert_called_once()
        self.assertEqual(self.jit_kernel.from_database.call_args.kwargs["auto_gm_idx"], [])

    def test_valid_workspace_list_is_preserved(self):
        self._write_indices([0])
        self._write_manifest([0])
        self.assertIs(self._load(), self.kernel)
        self.jit_kernel.from_database.assert_called_once()
        self.assertEqual(self.jit_kernel.from_database.call_args.kwargs["auto_gm_idx"], [0])

    def test_contract_for_non_cython_backend_is_a_miss(self):
        for backend in ("ctypes", "dlpack"):
            with self.subTest(backend=backend):
                self.assertIsNone(self._load(execution_backend=backend))
        self.jit_kernel.from_database.assert_not_called()

    def test_unknown_schema_is_a_miss(self):
        self.func.attrs[self.contract.AUTO_GM_FRESH_CONTRACT_ATTR] = _StringImm("cython-auto-gm-v2")
        self.assertIsNone(self._load())
        self.jit_kernel.from_database.assert_not_called()

    def test_unmarked_nonascend_cache_loads_without_metadata(self):
        (self.entry / "auto_gm_idx.pkl").unlink()
        self.func = SimpleNamespace(attrs={})
        self.assertIs(self._load(target="cuda", execution_backend="ctypes"), self.kernel)
        self.jit_kernel.from_database.assert_called_once()
        kwargs = self.jit_kernel.from_database.call_args.kwargs
        self.assertEqual(kwargs["auto_gm_idx"], [])
        self.assertEqual(kwargs["execution_backend"], "ctypes")

    def test_unmarked_nonascend_cache_keeps_parse_failure_fallback(self):
        (self.entry / "auto_gm_idx.pkl").write_text("{not valid JSON")
        self.func = SimpleNamespace(attrs={})
        self.assertIs(self._load(target="cuda"), self.kernel)
        self.jit_kernel.from_database.assert_called_once()
        self.assertEqual(self.jit_kernel.from_database.call_args.kwargs["auto_gm_idx"], [])


class TestAscendLegacyTunerContractCache(_KeyedContractWriterChecks, _AdditionalContractLoaderMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        tuner_cache = _load_standalone(
            REPO_ROOT / "tilelang" / "cache" / "tuner_cache.py",
            "ascend_contract_tuner_cache_under_test",
        )
        tuner_cache.AutoTunerCache._instance = None
        self.cache = tuner_cache.AutoTunerCache(cache_dir=self.tmp.name)
        self.key = "ascend_tuner_contract"
        self.entry, _ = self.write_entry(self.cache, self.key)
        self._write_manifest([])

    def _load(self, **kwargs):
        return self.cache._load_kernel_from_disk(self.key, func=self.func, **kwargs)

    def _save(self, kernel):
        self.cache._save_kernel_to_disk(self.key, kernel, self.func)

    def test_contract_metadata_is_a_miss_without_a_manifest(self):
        path = self.entry / "auto_gm_idx.pkl"
        (self.entry / self.contract.AUTO_GM_MANIFEST_PATH).unlink()
        # Workspace metadata alone cannot bind this allocation promise to the
        # cached source and lowered signature, even when the list is empty.
        for contents in (None, "[]", "null", "{not valid JSON"):
            with self.subTest(contents=contents):
                if contents is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_text(contents)
                self.assertIsNone(self._load())
        self.jit_kernel.from_database.assert_not_called()

    def test_valid_empty_table_and_manifest_loads(self):
        self._write_indices([])
        self._write_manifest([])
        self.assertIs(self._load(), self.kernel)
        self.jit_kernel.from_database.assert_called_once()
        self.assertEqual(self.jit_kernel.from_database.call_args.kwargs["auto_gm_idx"], [])

    def test_missing_metadata_is_a_miss_even_with_a_valid_manifest(self):
        (self.entry / "auto_gm_idx.pkl").unlink()
        self.assertIsNone(self._load())
        self.jit_kernel.from_database.assert_not_called()

    def test_contract_for_non_cython_backend_is_a_miss(self):
        for backend in ("ctypes", "dlpack"):
            with self.subTest(backend=backend):
                self.assertIsNone(self._load(execution_backend=backend))
        self.jit_kernel.from_database.assert_not_called()

    def test_unknown_schema_is_a_miss(self):
        self.func.attrs[self.contract.AUTO_GM_FRESH_CONTRACT_ATTR] = _StringImm("cython-auto-gm-v2")
        self.assertIsNone(self._load())
        self.jit_kernel.from_database.assert_not_called()

    def test_unmarked_nonascend_cache_loads_without_manifest(self):
        (self.entry / "auto_gm_idx.pkl").unlink()
        self.func = SimpleNamespace(attrs={})
        self.assertIs(self._load(target="cuda", execution_backend="ctypes"), self.kernel)
        self.jit_kernel.from_database.assert_called_once()
        kwargs = self.jit_kernel.from_database.call_args.kwargs
        self.assertNotIn("auto_gm_idx", kwargs)
        self.assertEqual(kwargs["execution_backend"], "ctypes")

    def test_nonascend_cache_without_function_loads_without_manifest(self):
        self.func = None
        self.assertIs(self._load(target="cuda"), self.kernel)
        self.jit_kernel.from_database.assert_called_once()
