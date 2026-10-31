"""PTO identity and shared-Cython restore, before public compiler activation.

CPU C launchers isolate the real argument/allocation ABI, not NPU numerics.
"""

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
from tilelang.ascend.a2a3.execution_backend import COMPILER_CONFIG_ATTR
from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache
from tilelang.ascend.a2a3.pass_config import normalize_a2a3_pass_configs
from tilelang.ascend.a2a3.runtime import dump_runtime_signature, load_runtime_signature
from tilelang.backend import create_backend_context
from tilelang.cache import _resolve_cache_dispatch
from tilelang.engine.lower import extrac_params
from tilelang.jit.adapter.base import CachedTextSource
from tilelang.jit.adapter.cython import adapter as adapter_module
from tilelang.jit.adapter.cython.kernel_cache import CythonKernelCache
from testing.ascend_a2a3.python.refactor.jit.test_a2a3_pto_shape_abi import _signature

TARGET = {"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["pto", "ascend"]}


def _snapshot(monkeypatch, tmp_path, dialect="pto"):
    # Configuration-only fixture; these paths are not advertised as a real
    # Bisheng toolchain. CPU C code below is compiled with the host compiler.
    monkeypatch.setattr(compiler, "_get_ascend_home_path", lambda **_: str(tmp_path / "sdk"))
    monkeypatch.setattr(compiler, "_get_tl_root", lambda **_: str(Path(__file__).parents[5]))
    monkeypatch.setattr(compiler.shutil, "which", lambda *a, **kw: "/unused/frozen/bisheng")
    monkeypatch.setenv("TL_RUN_MODE", "npu")
    monkeypatch.delenv("TL_CCE_OPT_LEVEL", raising=False)
    target = tvm.target.Target({**TARGET, "keys": [dialect, "ascend"]})
    configs = normalize_a2a3_pass_configs(target)
    return target, configs, compiler.resolve_compile_config(target, configs)


def test_pto_dispatch_reuses_cython_cache_and_keeps_target_dialect_identity():
    cache, context, _ = _resolve_cache_dispatch(TARGET, "c", "auto", False)
    assert type(cache) is A2A3CythonKernelCache
    assert context.name == "pto_a2a3" and context.execution_backend.name == "cython"
    other, ascendc, _ = _resolve_cache_dispatch({**TARGET, "keys": ["ascendc", "ascend"]}, "c", "auto", False)
    assert other is cache
    func = _signature()
    assert cache._generate_key(func, [4], "cython", (), target=context.target) != cache._generate_key(
        func, [4], "cython", (), target=ascendc.target
    )


def test_pto_request_snapshot_freezes_once_before_lookup(monkeypatch, tmp_path):
    _snapshot(monkeypatch, tmp_path)
    cache = A2A3CythonKernelCache()
    func = _signature()
    original = tvm.ir.save_json(func)
    captures = []

    def capture(_self, derived, *args, **kwargs):
        captures.append(kwargs)
        return derived

    monkeypatch.setattr(CythonKernelCache, "cached", capture)
    derived = cache.cached(func, [4], workspace_idx=[1], target=TARGET, verbose=False)
    assert tvm.ir.save_json(func) == original
    metadata = json.loads(str(derived.attrs[COMPILER_CONFIG_ATTR]))
    assert metadata["dialect"] == "pto" and metadata["platform"] == "A3" and metadata["arch"] == "dav-c220"
    assert metadata["flags"] == ["-O3"]
    assert captures[0]["pass_configs"]["tl.ascend_auto_sync_vs"] is True
    monkeypatch.setenv("TL_PLATFORM", "A2")
    monkeypatch.setenv("TL_RUN_MODE", "sim")
    monkeypatch.setenv("TL_CCE_OPT_LEVEL", "0")
    monkeypatch.setattr(compiler, "resolve_compile_config", lambda *a, **kw: pytest.fail("frozen request was resolved twice"))
    restored = tilelang.JITKernel(derived, target=TARGET, execution_backend="cython", from_database=True)
    assert restored.backend.name == "pto_a2a3"
    assert restored.compile_config.to_metadata() == metadata
    assert restored._workspace_idx_spec == [1]


def test_jit_refuses_an_ascendc_snapshot_for_pto(monkeypatch, tmp_path):
    _, _, config = _snapshot(monkeypatch, tmp_path, "ascendc")
    with pytest.raises(ValueError, match="same target dialect keys"):
        tilelang.JITKernel(_signature(), target=TARGET, execution_backend="cython", from_database=True, compile_config=config)


def _restore_kernel(folder, *, target=None, backend_context=None):
    func = load_runtime_signature((folder / "signature.json").read_text())
    config = compiler.CompileConfig.from_metadata(json.loads((folder / "config.json").read_text()))
    return tilelang.JITKernel.from_database(
        func=func,
        runtime_func=func,
        params=extrac_params(func),
        target=tvm.target.Target(TARGET) if target is None else target,
        target_host="c",
        out_idx=4,
        backend_context=backend_context,
        workspace_idx=1,
        execution_backend="cython",
        compile_config=config,
        host_kernel_source=CachedTextSource(path=str(folder / "host.cc")),
        device_kernel_source=CachedTextSource(path=str(folder / "host.cc")),
        kernel_lib_path=str(folder / "host.so"),
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_pto_database_restore_preserves_native_slots_workspace_and_gm(monkeypatch, tmp_path, reverse):
    _, _, config = _snapshot(monkeypatch, tmp_path)
    signature = _signature((lambda m, n: (n + m,)) if reverse else None)
    (tmp_path / "signature.json").write_text(dump_runtime_signature(signature))
    (tmp_path / "config.json").write_text(json.dumps(config.to_metadata()))
    shapes = "int64_t N, int64_t M" if reverse else "int64_t M, int64_t N"
    source = f"""#include <stdint.h>
    extern "C" void call(float *composite, float *workspace, float *m, float *n,
                         float *output, float *gm, {shapes}, void *stream) {{
      workspace[0] = M + N; gm[0] = 16; output[0] = M; output[1] = N;
    }}"""
    (tmp_path / "host.cc").write_text(source)
    subprocess.run(["c++", "-shared", "-fPIC", str(tmp_path / "host.cc"), "-o", str(tmp_path / "host.so")], capture_output=True, check=True)
    monkeypatch.setattr(adapter_module, "_device_providers", lambda: (lambda: torch.device("cpu"), lambda _: 0))
    monkeypatch.setattr(adapter_module.CythonKernelAdapter, "_process_buffer_device", lambda _: {})
    monkeypatch.setenv("TL_PLATFORM", "A2")
    monkeypatch.setenv("TL_RUN_MODE", "sim")
    monkeypatch.setenv("TL_CCE_OPT_LEVEL", "0")
    monkeypatch.setattr(compiler, "resolve_compile_config", lambda *a, **kw: pytest.fail("restore re-resolved compiler inputs"))
    context = create_backend_context(TARGET, "c", "cython")
    # The raw alias would now resolve to A2 if it leaked into the adapter.
    # The already-resolved context is authoritative throughout restoration.
    kernel = _restore_kernel(tmp_path, target="pto", backend_context=context)
    composite = torch.zeros(9 if reverse else 13)
    assert kernel(composite, torch.zeros(4), torch.zeros(5), stream=0).tolist() == [4.0, 5.0]
    assert kernel.workspace_idx == [1] and kernel.auto_gm_idx == [5] and kernel.out_idx == [4]
    assert kernel.adapter.lib.call.restype is None
    assert kernel.get_profiler().workspace_idx == [1]
    expected = ["N", "M"] if reverse else ["M", "N"]
    assert [str(v) for v in kernel.adapter.dynamic_symbolic_map] == expected
    assert kernel.compile_config.to_metadata() == config.to_metadata()
    assert kernel.adapter.target.attrs["model"] == "A3"
    child = tmp_path / "restore.py"
    child.write_text(f"""import importlib.util,json
from pathlib import Path
import torch
spec=importlib.util.spec_from_file_location("pto_execution_fixture",{str(Path(__file__))!r})
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
fixture.adapter_module._device_providers=lambda:(lambda:torch.device("cpu"),lambda _:0)
fixture.adapter_module.CythonKernelAdapter._process_buffer_device=lambda _:{{}}
fixture.compiler.resolve_compile_config=lambda *a,**kw:(_ for _ in ()).throw(AssertionError("restore re-resolved"))
kernel=fixture._restore_kernel(Path({str(tmp_path)!r}))
output=kernel(torch.zeros({9 if reverse else 13}),torch.zeros(4),torch.zeros(5),stream=0)
print("PTO_JIT_RESTORE="+json.dumps({{"actual":output.tolist(),"symbols":[str(v)for v in kernel.adapter.dynamic_symbolic_map],"workspace":kernel.workspace_idx,"gm":kernel.auto_gm_idx,"config":kernel.compile_config.to_metadata()}}))
""")
    result = subprocess.run([sys.executable, str(child)], env=os.environ.copy(), cwd=tmp_path, text=True, capture_output=True, check=True)
    row = json.loads(
        next(line.removeprefix("PTO_JIT_RESTORE=") for line in result.stdout.splitlines() if line.startswith("PTO_JIT_RESTORE="))
    )
    assert row == {"actual": [4.0, 5.0], "symbols": expected, "workspace": [1], "gm": [5], "config": config.to_metadata()}
