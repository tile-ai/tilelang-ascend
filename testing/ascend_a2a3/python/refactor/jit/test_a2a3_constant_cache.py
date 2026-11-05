"""Real public compile and fresh-process cache restore for embedded data."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import tilelang
from tilelang import env, tvm
import tilelang.ascend.a2a3.language as T
from tilelang.ascend.a2a3 import compiler

from test_a2a3_public_execution import actual_toolchain, isolated_cache  # noqa: F401

TARGET = {"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["ascendc"]}


def constant_kernel(last=37):
    @T.prim_func
    def constant(Input: T.Tensor((1,), "int32"), Output: T.Tensor((1,), "int32")):
        pointer = T.allocate_const([7, 11, 23, last], "int32", [4])
        table = T.decl_buffer((4,), "int32", data=pointer)  # noqa: F841 - raw source name
        with T.Scope("C"):
            T._src_code("Output.SetValue(0, table[Input.GetValue(0) & 3]);", pointer)

    return constant


def test_public_compile_literal_identity_and_new_process_disk_restore(monkeypatch, actual_toolchain, tmp_path):  # noqa: F811
    calls = []
    original_compile = compiler.LibraryGenerator.compile_lib

    def compile_lib(self, timeout=None):
        calls.append(self.config.cache_identity)
        return original_compile(self, timeout)

    monkeypatch.setattr(compiler.LibraryGenerator, "compile_lib", compile_lib)
    func = constant_kernel()
    before = tvm.ir.save_json(func)
    options = dict(target=TARGET, out_idx=[1], execution_backend="cython", verbose=False)
    cold = tilelang.compile(func, **options)
    assert len(calls) == 1 and len(cold.params) == 2
    source = cold.get_kernel_source()
    assert source.count('section(".rodata.tvm")') == 1
    assert "table[Input.GetValue(0) & 3]" in source
    assert "call(uint8_t* Input_handle, uint8_t* Output_handle, aclrtStream stream)" in source
    assert tilelang.compile(constant_kernel(), **options) is cold
    assert len(calls) == 1
    actual_toolchain._memory_cache.clear()
    restored = tilelang.compile(constant_kernel(), **options)
    assert restored.artifact is None and restored.get_kernel_source() == source
    assert len(calls) == 1 and len(restored.params) == 2
    child = tmp_path / "restore.py"
    child.write_text("""import hashlib,importlib.util,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[1]).parent))
import torch,tilelang
from tilelang.jit.adapter.cython import adapter
from tilelang.ascend.a2a3 import compiler
adapter.CythonKernelAdapter._process_buffer_device=lambda self:{}
adapter._device_providers=lambda:(lambda:torch.device("cpu"),lambda device:0)
compiler.LibraryGenerator.compile_lib=lambda *a,**kw:(_ for _ in ()).throw(AssertionError("restore compiled"))
s=importlib.util.spec_from_file_location("constant_source",sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
k=tilelang.compile(m.constant_kernel(),target=m.TARGET,out_idx=[1],execution_backend="cython",verbose=False)
assert k.artifact is None
print("LITERAL_RESTORE="+json.dumps({"source_hash":hashlib.sha256(k.get_kernel_source().encode()).hexdigest(),"params":len(k.params),"auto_gm":k.auto_gm_idx,"workspace":k.workspace_idx}))
""")
    process = subprocess.run(
        [sys.executable, str(child), str(Path(__file__).resolve())],
        env=dict(os.environ, TILELANG_CACHE_DIR=str(env.TILELANG_CACHE_DIR)),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    metadata = json.loads(next(line.split("=", 1)[1] for line in process.stdout.splitlines() if line.startswith("LITERAL_RESTORE=")))
    assert metadata == {"source_hash": hashlib.sha256(source.encode()).hexdigest(), "params": 2, "auto_gm": [], "workspace": []}
    changed = tilelang.compile(constant_kernel(41), **options)
    assert len(calls) == 2 and changed.get_kernel_source() != source
    assert tvm.ir.save_json(func) == before
    print("literal real compile: cold=1 memory=0 disk=0 fresh-process=0 changed-payload=1; params=2")
