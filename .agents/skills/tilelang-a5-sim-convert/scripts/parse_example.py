#!/usr/bin/env python3
"""Parse a tilelang example script to extract kernel info for A5 sim conversion.

Usage: python parse_example.py <script_path>
Output: JSON with kernel shapes, dtypes, and the kernel function name.
"""

import contextlib
import sys
import os
import json
import importlib.util


def parse(script_path):
    script_path = os.path.abspath(script_path)
    script_dir = os.path.dirname(script_path)
    sys.path.insert(0, script_dir)

    # --- Mock torch.npu before import ---
    import torch as _torch

    _orig_tensor_npu = getattr(_torch.Tensor, "npu", None)
    _torch.Tensor.npu = lambda self, *a, **kw: self
    with contextlib.suppress(Exception):
        _torch.nn.Module.npu = lambda self, *a, **kw: self

    # --- Protect sys.argv ---
    _saved_argv = sys.argv
    sys.argv = [script_path]

    spec = importlib.util.spec_from_file_location("_target", script_path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        # Non-fatal: script's torch.npu calls failed but kernel was defined
        pass
    finally:
        sys.argv = _saved_argv
        if _orig_tensor_npu is not None:
            _torch.Tensor.npu = _orig_tensor_npu

    # --- Find kernel ---
    prim_func = None
    kernel_name = None

    # Try make_kernel()
    if hasattr(mod, "make_kernel"):
        prim_func = mod.make_kernel()
        kernel_name = "make_kernel"
    # Try common names
    for name in ["matmul", "kernel", "main"]:
        if hasattr(mod, name):
            obj = getattr(mod, name)
            if hasattr(obj, "__wrapped__"):
                kernel_name = name
                # Try calling with minimal args
                import inspect

                sig = inspect.signature(obj.__wrapped__)
                params = list(sig.parameters.keys())
                # Provide sensible defaults based on param count
                n_params = len(params)
                if n_params >= 6:
                    prim_func = obj.__wrapped__(1024, 512, 256, 128, 256, 64)
                elif n_params >= 3:
                    prim_func = obj.__wrapped__(1024, 512, 256)
                else:
                    prim_func = obj.__wrapped__()
                break
            elif isinstance(obj, __import__("tilelang").tvm.tir.PrimFunc):
                prim_func = obj
                kernel_name = name
                break

    # Fallback: search all module attributes
    if prim_func is None:
        import tilelang

        for name in dir(mod):
            obj = getattr(mod, name)
            if hasattr(obj, "__wrapped__"):
                try:
                    prim_func = obj.__wrapped__(1024, 512, 256, 128, 256, 64)
                    kernel_name = name
                    break
                except Exception:
                    pass
            if isinstance(obj, tilelang.tvm.tir.PrimFunc):
                prim_func = obj
                kernel_name = name
                break

    if prim_func is None:
        raise RuntimeError(f"No kernel found in {script_path}")

    # --- Extract shapes & dtypes ---
    buffers = []
    for p in prim_func.params:
        buf = prim_func.buffer_map.get(p)
        if buf is not None:
            buffers.append(
                {
                    "shape": [int(s) for s in buf.shape],
                    "dtype": str(buf.dtype),
                }
            )

    result = {
        "script_path": script_path,
        "kernel_name": kernel_name,
        "buffers": buffers,
        "num_buffers": len(buffers),
    }

    # Read original source code to extract the kernel definition text
    with open(script_path) as f:
        original_source = f.read()

    result["original_lines"] = len(original_source.splitlines())

    return result


if __name__ == "__main__":
    result = parse(sys.argv[1])
    print(json.dumps(result, indent=2))
