"""Public A2A3 JIT/cache contracts, including the real source compiler chain.

Source: TA 0deb8b93da11b7f0becbcded732d9633b7ca4e34 workspace/JIT ABI.
The toolchain cases compile and load real AscendC libraries without launching NPU
kernels; numerical device checks belong to the paired example harness.
"""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import tilelang
from tilelang import tvm, env
from tilelang.ascend.a2a3 import compiler
from tilelang.ascend.a2a3.execution_backend import COMPILER_CONFIG_ATTR, WORKSPACE_IDX_ATTR
from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache
from tilelang.backend import create_backend_context
from tilelang.engine.lower import lower_with_context
from tilelang.jit.adapter.cython import adapter as adapter_module
from tilelang.jit.adapter.cython.kernel_cache import CythonKernelCache


TARGET = {"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["ascendc"]}


def _factories():
    path = Path(__file__).parents[1] / "language" / "_a2a3_source_kernels.py"
    spec = importlib.util.spec_from_file_location("a2a3_public_execution_source", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _small_func():
    buffers = [tvm.tirx.decl_buffer((16,), "float16", name=name) for name in ["input", "workspace", "output"]]
    return tvm.tirx.PrimFunc([b.data for b in buffers], tvm.tirx.Evaluate(0), buffer_map={b.data: b for b in buffers}).with_attr(
        "global_symbol", "unit"
    )


@pytest.fixture
def isolated_cache(monkeypatch, tmp_path):
    previous = env.is_cache_enabled()
    env.enable_cache()
    monkeypatch.setattr(env, "TILELANG_CACHE_DIR", str(tmp_path / "cache"))
    cache = A2A3CythonKernelCache()
    cache._memory_cache.clear()
    yield cache
    cache._memory_cache.clear()
    if not previous:
        env.disable_cache()


def _fake_snapshot(monkeypatch, tmp_path):
    monkeypatch.setattr(compiler, "_get_ascend_home_path", lambda **kwargs: str(tmp_path / "sdk"))
    monkeypatch.setattr(compiler, "_get_tl_root", lambda **kwargs: str(Path(__file__).parents[5]))
    monkeypatch.setattr(compiler.shutil, "which", lambda *args, **kwargs: "/unused/bisheng")
    monkeypatch.setenv("TL_RUN_MODE", "npu")
    monkeypatch.delenv("TL_CCE_OPT_LEVEL", raising=False)


def test_request_identity_is_derived_before_lookup(monkeypatch, tmp_path, isolated_cache):
    _fake_snapshot(monkeypatch, tmp_path)
    original = _small_func()
    original_json = tvm.ir.save_json(original)
    captures = []

    def lookup(_self, func, out_idx, *args, **kwargs):
        captures.append((func, kwargs))
        return func

    monkeypatch.setattr(CythonKernelCache, "cached", lookup)
    first = isolated_cache.cached(original, [-1], workspace_idx=[-2], target=TARGET, verbose=False)
    same = isolated_cache.cached(original, [-1], workspace_idx=[-2], target=TARGET, verbose=False)
    monkeypatch.setenv("TL_CCE_OPT_LEVEL", "0")
    changed = isolated_cache.cached(original, [-1], workspace_idx=[-2], target=TARGET, verbose=False)
    other_workspace = isolated_cache.cached(original, [-1], workspace_idx=[1], target=TARGET, verbose=False)
    assert tvm.ir.save_json(original) == original_json
    assert original.body.same_as(first.body)
    assert WORKSPACE_IDX_ATTR not in original.attrs and COMPILER_CONFIG_ATTR not in original.attrs
    assert first.script(show_meta=True) == same.script(show_meta=True)
    assert str(first.attrs[COMPILER_CONFIG_ATTR]) != str(changed.attrs[COMPILER_CONFIG_ATTR])
    assert list(changed.attrs[WORKSPACE_IDX_ATTR]) != list(other_workspace.attrs[WORKSPACE_IDX_ATTR])
    assert captures[0][1]["backend_context"].target.attrs["model"] == "A3"
    # Direct restore consumes the derived snapshot, even if process options
    # change after lookup. The shared JIT must not resolve the driver twice.
    monkeypatch.setenv("TL_PLATFORM", "A2")
    monkeypatch.setenv("TL_RUN_MODE", "sim")
    monkeypatch.setattr(compiler, "resolve_compile_config", lambda *args, **kwargs: pytest.fail("snapshot was resolved twice"))
    kernel = tilelang.JITKernel(first, target=TARGET, execution_backend="cython", from_database=True)
    assert kernel.compile_config.flags == ("-O2",)
    assert kernel.compile_config.platform == "A3" and kernel.compile_config.run_mode == "npu"
    assert kernel._workspace_idx_spec == [-2]


@pytest.mark.parametrize("workspace", [(), ["1"], "1"])
def test_invalid_workspace_request_rejected_before_compile(monkeypatch, tmp_path, isolated_cache, workspace):
    _fake_snapshot(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="workspace_idx"):
        isolated_cache.cached(_small_func(), [-1], workspace_idx=workspace, target=TARGET, verbose=False)


def test_non_a2a3_dispatch_and_key_remain_unchanged():
    from tilelang.cache import _resolve_cache_dispatch, cached

    cache, context, _ = _resolve_cache_dispatch({"kind": "ascend", "arch": "dav-3510"}, None, "cython", False)
    from tilelang.ascend.kernel_cache import AscendCythonKernelCache

    assert type(cache) is AscendCythonKernelCache
    assert context.module.name == "ascend"
    with pytest.raises(ValueError, match="only by the A2A3"):
        cached(_small_func(), target={"kind": "ascend", "arch": "dav-3510"}, execution_backend="cython", workspace_idx=[1])


def test_compile_decorator_and_parallel_forward_original_workspace_option(monkeypatch):
    import importlib

    jit_module = importlib.import_module("tilelang.jit")
    calls = []
    sentinel = object()

    def cached(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(jit_module, "cached", cached)
    func = _small_func()
    assert tilelang.compile(func, workspace_idx=[-2]) is sentinel
    assert tilelang.par_compile([func], workspace_idx=[-2], num_workers=1) == [sentinel]

    @tilelang.jit(workspace_idx=[-2], out_idx=[-1])
    def factory(size: int):
        return _small_func()

    assert factory(256) is sentinel
    assert factory(__return_compile_arguments=True)["workspace_idx"] == [-2]
    assert len(calls) == 3
    assert all(call["workspace_idx"] == [-2] for call in calls)


def test_profiler_workspace_exclusion_preserves_constructor_and_output_switch():
    import inspect
    from tilelang.profiler import Profiler, TensorSupplyType
    from tilelang.engine.param import KernelParam

    assert list(inspect.signature(Profiler).parameters) == ["params", "result_idx", "supply_type", "adapter"]
    params = [KernelParam("float32", [i]) for i in [2, 3, 4]]
    profiler = Profiler(params, [2], TensorSupplyType.Auto).with_workspace_idx([-2])
    profiler.supply = lambda param: param.shape
    assert profiler._get_inputs() == [[2]]
    assert profiler._get_inputs(with_output=True) == [[2], [4]]
    assert profiler._get_inputs(with_workspace=True) == [[2], [3]]
    assert profiler._get_params() == params[:1]
    assert profiler._get_params(with_output=True, with_workspace=True) == params


@pytest.fixture
def actual_toolchain(monkeypatch, isolated_cache):
    if tvm.ffi.get_global_func("tl.transform.A2A3AscendVectorMaskLegalize", allow_missing=True) is None:
        pytest.skip("requires a fresh USE_ASCEND native library")
    monkeypatch.setenv("TL_RUN_MODE", "npu")
    monkeypatch.delenv("TL_CCE_OPT_LEVEL", raising=False)
    monkeypatch.delenv("TL_CCE_AUTO_SYNC", raising=False)
    try:
        compiler.resolve_compile_config(tvm.target.Target(TARGET))
    except (ValueError, FileNotFoundError) as error:
        pytest.skip(f"requires actual AscendC compiler/resources: {error}")
    # Only launch-time providers/validation are isolated. No lowering, codegen,
    # compiler, cache, serialization or load function is replaced.
    monkeypatch.setattr(adapter_module.CythonKernelAdapter, "_process_buffer_device", lambda self: {})
    monkeypatch.setattr(adapter_module, "_device_providers", lambda: (lambda: torch.device("cpu"), lambda device: 0))
    return isolated_cache


def _compile(func, **kwargs):
    return tilelang.compile(func, target=TARGET, execution_backend="cython", verbose=False, **kwargs)


def test_real_pipeline_cache_disabled_cold_memory_disk_and_environment(monkeypatch, actual_toolchain):
    calls = []
    original_compile = compiler.LibraryGenerator.compile_lib

    def compile_lib(self, timeout=None):
        calls.append(self.config.to_metadata())
        return original_compile(self, timeout)

    monkeypatch.setattr(compiler.LibraryGenerator, "compile_lib", compile_lib)
    func = _factories().matmul_add(256, 256, 128, 128, 128, 64)
    before = tvm.ir.save_json(func)
    configs = {
        "tl.ascend_auto_sync": True,
        "tl.ascend_auto_cross_core_sync": True,
        "tl.ascend_auto_cv_combine": True,
        "tl.ascend_memory_planning": True,
    }
    options = dict(out_idx=[2], workspace_idx=[-1], pass_configs=configs)
    cold = _compile(func, **options)
    assert len(calls) == 1 and cold.workspace_idx == [4] and cold.out_idx == [2]
    assert len(cold.params) == 5
    assert COMPILER_CONFIG_ATTR in cold.artifact.device_mod["main"].attrs
    assert WORKSPACE_IDX_ATTR in cold.artifact.device_mod["main"].attrs
    # The request identity must not change emitted code or parameter semantics.
    ctx = create_backend_context(TARGET, "c", "cython")
    with tvm.transform.PassContext(opt_level=3, config=configs):
        plain = lower_with_context(func, ctx)
    assert cold.get_kernel_source() == plain.kernel_source
    assert [(str(p.dtype), [str(s) for s in p.shape]) for p in cold.params] == [
        (str(p.dtype), [str(s) for s in p.shape]) for p in plain.params
    ]
    memory = _compile(_factories().matmul_add(256, 256, 128, 128, 128, 64), **options)
    assert memory is cold and len(calls) == 1
    cache_dir = next(Path(env.TILELANG_CACHE_DIR).rglob("a2a3_runtime.json")).parent
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    assert "a2a3_runtime.json" in str(manifest)
    sidecar = cache_dir / "a2a3_runtime.json"
    valid_bytes = sidecar.read_bytes()
    sidecar.write_bytes(b"broken metadata")
    assert not actual_toolchain._verify_cache_dir_integrity(str(cache_dir))
    sidecar.write_bytes(valid_bytes)
    assert actual_toolchain._verify_cache_dir_integrity(str(cache_dir))
    actual_toolchain._memory_cache.clear()
    disk = _compile(func, **options)
    assert disk is not cold and disk.artifact is None and len(calls) == 1
    assert disk.workspace_idx == [4] and disk.out_idx == [2]
    assert disk.compile_config.cache_identity == cold.compile_config.cache_identity
    env.disable_cache()
    try:
        disabled = _compile(func, **options)
    finally:
        env.enable_cache()
    assert len(calls) == 2 and disabled.workspace_idx == [4]
    monkeypatch.setenv("TL_CCE_OPT_LEVEL", "0")
    different = _compile(func, **options)
    assert len(calls) == 3 and different is not cold
    assert different.compile_config.flags == ("-O0",)
    assert len(list(Path(env.TILELANG_CACHE_DIR).rglob("a2a3_runtime.json"))) == 2
    assert tvm.ir.save_json(func) == before
    print("actual workspace cache: cold=1, memory=0, disk=0, disabled=1, changed-env=1 Bisheng calls")


def test_real_workspace_reduction_gm_survives_disk_restore(monkeypatch, actual_toolchain):
    import tilelang.ascend.a2a3.language as T

    @T.prim_func
    def automatic(A: T.Tensor((16, 16), "float16"), B: T.Tensor((16, 16), "float16"), C: T.Tensor((16, 16), "float16")):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a = T.alloc_L1((16, 16), "float16")
            b = T.alloc_L1((16, 16), "float16")
            c0 = T.alloc_L0C((16, 16), "float")
            ub = T.alloc_ub((16, 16), "float16")
            with T.Scope("C"):
                T.copy(A, a)
                T.copy(B, b)
                T.gemm_v0(a, b, c0, init=True)
                T.copy(c0, ub)
            with T.Scope("V"):
                T.copy(ub, C)

    cold = _compile(automatic, out_idx=[2])
    assert len(automatic.params) == 3 and len(cold.params) == 4
    assert cold.auto_gm_idx == [3] and cold.workspace_idx == []
    assert [int(dim) for dim in cold.params[3].shape] == [1, 16, 16]
    actual_toolchain._memory_cache.clear()
    monkeypatch.setattr(compiler.LibraryGenerator, "compile_lib", lambda *args, **kwargs: pytest.fail("disk restore must not compile"))
    restored = _compile(automatic, out_idx=[2])
    assert restored.artifact is None and restored.auto_gm_idx == [3] and len(restored.params) == 4
    assert [int(dim) for dim in restored.params[3].shape] == [1, 16, 16]


def test_real_dynamic_source_new_process_restore(monkeypatch, actual_toolchain, tmp_path):
    func = _factories().indexer_dynamic(1, 32, 64, 1024, 512, 4, 128, 256, 64)
    configs = {"tl.ascend_memory_planning": True}
    cold = _compile(func, out_idx=[-1], workspace_idx=[-3], pass_configs=configs)
    assert cold.workspace_idx == [2]
    assert [str(var) for var in cold.adapter.dynamic_symbolic_map] == ["B", "S1", "S2"]
    host = cold.get_host_source()
    declaration = next(line for line in host.splitlines() if 'extern "C" void call(' in line)
    assert declaration.index("int64_t B") < declaration.index("int64_t S1") < declaration.index("int64_t S2")
    source_file = Path(__file__).parents[1] / "language" / "_a2a3_source_kernels.py"
    script = tmp_path / "restore.py"
    script.write_text("""import importlib.util,json,sys,os
from pathlib import Path
# The subprocess may use a newly rebuilt, isolated Cython module in development.
if os.environ.get("A2A3_TEST_CYTHON_MODULE"):
 s=importlib.util.spec_from_file_location("tilelang_cython_wrapper",os.environ["A2A3_TEST_CYTHON_MODULE"]);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
import tilelang
from tilelang.jit.adapter.cython import adapter
from tilelang.ascend.a2a3 import compiler
adapter.CythonKernelAdapter._process_buffer_device=lambda self:{}
compiler.LibraryGenerator.compile_lib=lambda *a,**kw:(_ for _ in ()).throw(AssertionError("new process restore compiled"))
s=importlib.util.spec_from_file_location("source_fresh",sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
k=tilelang.compile(m.indexer_dynamic(1,32,64,1024,512,4,128,256,64),out_idx=[-1],workspace_idx=[-3],pass_configs={"tl.ascend_memory_planning":True},target={"kind":"ascend","arch":"dav-2201","model":"A3","keys":["ascendc"]},execution_backend="cython")
assert k.artifact is None
print("RESTORE_JSON="+json.dumps({"workspace":k.workspace_idx,"auto_gm":k.auto_gm_idx,"symbols":[str(v)for v in k.adapter.dynamic_symbolic_map],"slots":list(k.adapter.dynamic_symbolic_map.values()),"config":k.compile_config.to_metadata()}))
""")
    child_env = dict(os.environ, TILELANG_CACHE_DIR=str(env.TILELANG_CACHE_DIR))
    result = subprocess.run([sys.executable, str(script), str(source_file)], env=child_env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    metadata = json.loads(
        next(line.removeprefix("RESTORE_JSON=") for line in result.stdout.splitlines() if line.startswith("RESTORE_JSON="))
    )
    assert metadata["symbols"] == ["B", "S1", "S2"]
    assert metadata["slots"] == [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1]]
    assert metadata["workspace"] == [2]
    assert metadata["config"] == cold.compile_config.to_metadata()
    print("actual source declaration:", declaration)
    print("fresh process dynamic ABI:", json.dumps(metadata, sort_keys=True))
