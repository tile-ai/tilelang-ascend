# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Real Bisheng/cache reconstruction acceptance; no NPU kernel is launched.

Run independently of device tests: the first call builds an actual Cython
kernel library. Later disk loads must honor its fresh-workspace allocation
contract without recompiling; missing or unreadable metadata forces a miss.
"""

import json
from pathlib import Path

import pytest

import tilelang
import tilelang.cache as kernel_cache
from tilelang import tvm
from tilelang.jit.ascend_runtime_contract import (
    AUTO_GM_FRESH_CONTRACT_ATTR,
    AUTO_GM_MANIFEST_PATH,
    sync_engine_cache_abi,
)
from test_ascend_upstream_sync_cross_core_device import (
    _PASS_CONFIGS,
    _flat_developer_gemm_exp,
)


@pytest.mark.parametrize("backend", ["upstream", "legacy"])
def test_real_cython_store_disk_hit_and_damaged_metadata_miss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    if backend == "upstream" and not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON and Bisheng compilation")
    configs = {**_PASS_CONFIGS, "tl.ascend_sync_backend": backend}
    cache = kernel_cache._kernel_cache_instance
    previous_directory = cache.get_cache_dir()
    previous_memory = cache._memory_cache
    previous_enabled = tilelang.is_cache_enabled()
    cache.set_cache_dir(str(tmp_path))
    cache._memory_cache = {}
    tilelang.enable_cache()
    try:
        program = _flat_developer_gemm_exp(explicit_noalias=False)
        assert len(program.params) == 3

        def compile_kernel():
            return tilelang.compile(
                program,
                out_idx=[2],
                target="ascendc",
                platform="A3",
                execution_backend="cython",
                pass_configs=configs,
                compile_flags=["--cce-auto-sync=off"],
            )

        built = compile_kernel()
        assert len(built.prim_func.params) == 3
        assert AUTO_GM_FRESH_CONTRACT_ATTR in built.prim_func.attrs
        assert len(built.params) == 4
        assert built.auto_gm_idx == [3]
        metadata_paths = list(tmp_path.glob("*/auto_gm_idx.pkl"))
        assert len(metadata_paths) == 1
        metadata = metadata_paths[0]
        assert json.loads(metadata.read_text()) == [3]
        manifest_path = metadata.parent / AUTO_GM_MANIFEST_PATH
        manifest = json.loads(manifest_path.read_text())
        assert manifest["schema"] == 2
        assert manifest["sync_engine_abi"] == sync_engine_cache_abi(configs, "ascendc")
        assert manifest["param_count"] == 4
        assert manifest["auto_gm_indices"] == [3]

        class CompileAttempted(RuntimeError):
            pass

        def reject_recompile(*args: object, **kwargs: object) -> None:
            raise CompileAttempted("cache miss reached real lowering")

        # This patches only the compiler boundary. The hit still reconstructs
        # the real adapter from the actual saved params and shared library.
        monkeypatch.setattr(tilelang, "lower", reject_recompile)
        cache._memory_cache.clear()
        loaded = compile_kernel()
        assert loaded is not built
        assert len(loaded.prim_func.params) == 3
        assert len(loaded.params) == 4
        assert loaded.auto_gm_idx == [3]

        # Simulate the next native engine ABI while retaining the real saved
        # artifact. Its previous key must miss, even with the same source and
        # package/HEAD version. This does not claim a second native rebuild.
        original_lookup = tvm.get_global_func

        def changed_engine(name, allow_missing=False):
            if name == "tl.ascend_sync.EngineCacheABI":
                return lambda: "simulated-next-engine"
            return original_lookup(name, allow_missing=allow_missing)

        with monkeypatch.context() as different_engine:
            different_engine.setattr(tvm, "get_global_func", changed_engine)
            cache._memory_cache.clear()
            with pytest.raises(CompileAttempted, match="cache miss reached real lowering"):
                compile_kernel()

        metadata.unlink()
        cache._memory_cache.clear()
        with pytest.raises(CompileAttempted, match="cache miss reached real lowering"):
            compile_kernel()

        metadata.write_text("{invalid json")
        cache._memory_cache.clear()
        with pytest.raises(CompileAttempted, match="cache miss reached real lowering"):
            compile_kernel()

        for wrong_indices in ([], [2]):
            metadata.write_text(json.dumps(wrong_indices))
            cache._memory_cache.clear()
            with pytest.raises(CompileAttempted, match="cache miss reached real lowering"):
                compile_kernel()

        metadata.write_text("[3]")
        manifest_path.write_text(json.dumps({**manifest, "sync_engine_abi": "another-engine"}))
        cache._memory_cache.clear()
        with pytest.raises(CompileAttempted, match="cache miss reached real lowering"):
            compile_kernel()

        manifest_path.write_text(json.dumps({**manifest, "source_hash": "different source"}))
        cache._memory_cache.clear()
        with pytest.raises(CompileAttempted, match="cache miss reached real lowering"):
            compile_kernel()

        manifest_path.write_text(json.dumps(manifest))
        cache._memory_cache.clear()
        restored = compile_kernel()
        assert restored.auto_gm_idx == [3]
        assert len(restored.params) == 4
    finally:
        cache.set_cache_dir(str(previous_directory))
        cache._memory_cache = previous_memory
        if not previous_enabled:
            tilelang.disable_cache()
