"""Actual PTO source/compiler/cache path; device numerics live in paired examples."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import tilelang
from tilelang import tvm
from tilelang.ascend.a2a3 import compiler
from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache
from tilelang.ascend.a2a3.pass_config import normalize_a2a3_pass_configs
from tilelang.jit.adapter.cython import adapter as adapter_module
from testing.ascend_a2a3.python.refactor.jit.test_a2a3_pto_execution import TARGET


def _dynamic_workspace_kernel():
    """Keep the source virtual-C/V-copy body; add logical shape carriers."""
    import tilelang.ascend.a2a3.language as T

    m, n = T.dynamic("M"), T.dynamic("N")

    @T.prim_func
    def automatic(
        composite: T.Tensor((m * 2 + n,), "float32"),
        m_carrier: T.Tensor((m,), "float32"),
        n_carrier: T.Tensor((n,), "float32"),
        A: T.Tensor((16, 16), "float16"),
        B: T.Tensor((16, 16), "float16"),
        C: T.Tensor((16, 16), "float16"),
        workspace: T.Tensor((m + n,), "float32"),
    ):
        with T.Kernel(1, is_npu=True):
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

    return automatic


@pytest.fixture
def actual_pto_toolchain(monkeypatch, tmp_path):
    if tvm.ffi.get_global_func("target.build.tilelang_pto_a2a3", allow_missing=True) is None:
        pytest.skip("requires a fresh USE_ASCEND PTO native builder")
    target = tvm.target.Target(TARGET)
    monkeypatch.setenv("TL_RUN_MODE", "npu")
    monkeypatch.delenv("TL_CCE_OPT_LEVEL", raising=False)
    monkeypatch.delenv("TL_CCE_AUTO_SYNC", raising=False)
    try:
        compiler.resolve_compile_config(target, normalize_a2a3_pass_configs(target))
    except (ValueError, FileNotFoundError) as error:
        pytest.skip(f"requires actual PTO compiler/resources: {error}")
    from tilelang import env

    previous = env.is_cache_enabled()
    env.enable_cache()
    monkeypatch.setattr(env, "TILELANG_CACHE_DIR", str(tmp_path / "cache"))
    cache = A2A3CythonKernelCache()
    cache._memory_cache.clear()
    # Compilation/codegen/cache/load stay real. This host-only suite does not
    # launch device kernels, so provider/device validation is isolated to CPU.
    monkeypatch.setattr(adapter_module, "_device_providers", lambda: (lambda: torch.device("cpu"), lambda _: 0))
    monkeypatch.setattr(adapter_module.CythonKernelAdapter, "_process_buffer_device", lambda _: {})
    yield cache
    cache._memory_cache.clear()
    if not previous:
        env.disable_cache()


def _public_compile(func, **kwargs):
    return tilelang.compile(func, target=TARGET, execution_backend="cython", verbose=False, **kwargs)


def test_public_pto_real_pipeline_cache_disabled_cold_memory_disk_and_environment(monkeypatch, actual_pto_toolchain):
    from tilelang import env
    from testing.ascend_a2a3.python.refactor.jit.test_a2a3_public_execution import _factories

    calls = []
    original = compiler.LibraryGenerator.compile_lib

    def compile_lib(self, timeout=None):
        calls.append(self.config.to_metadata())
        return original(self, timeout)

    monkeypatch.setattr(compiler.LibraryGenerator, "compile_lib", compile_lib)
    options = dict(
        out_idx=2,
        workspace_idx=-1,
        pass_configs={
            "tl.ascend_auto_sync": True,
            "tl.ascend_auto_cross_core_sync": True,
            "tl.ascend_auto_cv_combine": True,
            "tl.ascend_memory_planning": True,
        },
    )
    factory = _factories().matmul_add
    cold = _public_compile(factory(256, 256, 128, 128, 128, 64), **options)
    assert len(calls) == 1 and cold.backend.name == "pto_a2a3" and cold.execution_backend == "cython"
    assert cold.workspace_idx == [4] and cold.out_idx == [2] and cold.compile_config.dialect == "pto"
    assert cold.compile_config.arch == "dav-c220" and cold.target.attrs["arch"] == "dav-2201"
    assert _public_compile(factory(256, 256, 128, 128, 128, 64), **options) is cold and len(calls) == 1
    sidecar = next(Path(env.TILELANG_CACHE_DIR).rglob("a2a3_runtime.json"))
    metadata = json.loads(sidecar.read_text())
    assert metadata["compiler_config"]["dialect"] == "pto"
    assert actual_pto_toolchain._verify_cache_dir_integrity(str(sidecar.parent))
    original_bytes = sidecar.read_bytes()
    sidecar.write_bytes(b"corrupted snapshot")
    assert not actual_pto_toolchain._verify_cache_dir_integrity(str(sidecar.parent))
    sidecar.write_bytes(original_bytes)
    actual_pto_toolchain._memory_cache.clear()
    disk = _public_compile(factory(256, 256, 128, 128, 128, 64), **options)
    assert disk is not cold and disk.artifact is None and len(calls) == 1
    assert disk.compile_config.cache_identity == cold.compile_config.cache_identity
    assert disk.get_kernel_source() == cold.get_kernel_source() and disk.workspace_idx == [4]
    env.disable_cache()
    try:
        disabled = _public_compile(factory(256, 256, 128, 128, 128, 64), **options)
    finally:
        env.enable_cache()
    assert len(calls) == 2 and disabled.backend.name == "pto_a2a3"
    monkeypatch.setenv("TL_CCE_OPT_LEVEL", "0")
    different = _public_compile(factory(256, 256, 128, 128, 128, 64), **options)
    assert len(calls) == 3 and different.compile_config.flags == ("-O0",)
    assert different.compile_config.cache_identity != cold.compile_config.cache_identity


def test_public_pto_composite_shape_real_workspace_and_independent_cache_restore(monkeypatch, actual_pto_toolchain, tmp_path):
    from tilelang import env

    func = _dynamic_workspace_kernel()
    cold = _public_compile(func, out_idx=5, workspace_idx=6)
    assert len(func.params) == 7 and len(cold.params) == 8
    assert cold.workspace_idx == [6] and cold.auto_gm_idx == [7]
    # The original PTO pipe uses a flat FIFO with two slots; AscendC's
    # [core, M, N] workspace is a different source branch.
    assert [int(x) for x in cold.params[7].shape] == [512]
    assert str(cold.params[7].dtype) == "float16"
    symbols = [str(v) for v in cold.adapter.dynamic_symbolic_map]
    assert symbols == ["M", "N"]
    assert list(cold.adapter.dynamic_symbolic_map.values()) == [(0, 1, 0, 1), (0, 2, 0, 1)]
    assert "int64_t M, int64_t N, void *stream)" in cold.get_kernel_source()
    script = tmp_path / "public_restore.py"
    script.write_text(f"""import importlib.util,json
import tilelang
from tilelang import env
from pathlib import Path
spec=importlib.util.spec_from_file_location("pto_execution_fixture",{str(Path(__file__))!r})
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
fixture.adapter_module._device_providers=lambda:(lambda:fixture.torch.device("cpu"),lambda _:0)
fixture.adapter_module.CythonKernelAdapter._process_buffer_device=lambda _:{{}}
env.TILELANG_CACHE_DIR={str(env.TILELANG_CACHE_DIR)!r};env.enable_cache()
fixture.compiler.LibraryGenerator.compile_lib=lambda *a,**kw:(_ for _ in ()).throw(AssertionError("disk restore compiled"))
kernel=fixture._public_compile(fixture._dynamic_workspace_kernel(),out_idx=5,workspace_idx=6)
assert kernel.artifact is None
print("PTO_PUBLIC_RESTORE="+json.dumps({{"symbols":[str(v)for v in kernel.adapter.dynamic_symbolic_map],"slots":list(kernel.adapter.dynamic_symbolic_map.values()),"workspace":kernel.workspace_idx,"gm":kernel.auto_gm_idx,"config":kernel.compile_config.to_metadata(),"source":kernel.get_kernel_source()}}))
""")
    result = subprocess.run(
        [sys.executable, str(script)], cwd=Path.cwd(), env=os.environ.copy(), text=True, capture_output=True, check=True
    )
    row = json.loads(
        next(line.removeprefix("PTO_PUBLIC_RESTORE=") for line in result.stdout.splitlines() if line.startswith("PTO_PUBLIC_RESTORE="))
    )
    assert row == {
        "symbols": symbols,
        "slots": [[0, 1, 0, 1], [0, 2, 0, 1]],
        "workspace": [6],
        "gm": [7],
        "config": cold.compile_config.to_metadata(),
        "source": cold.get_kernel_source(),
    }
