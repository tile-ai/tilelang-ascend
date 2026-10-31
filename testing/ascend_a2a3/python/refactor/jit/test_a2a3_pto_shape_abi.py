"""PTO recursive shape order and direct-carrier limits, without an NPU."""

import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from tilelang import tvm
from tilelang.ascend.a2a3.runtime import dump_runtime_signature, load_runtime_signature, order_pto_shape_symbols
from tilelang.engine.lower import extrac_params
from tilelang.jit.adapter.cython.adapter import CythonKernelAdapter
from tilelang_cython_wrapper import CythonKernelWrapper

tir = tvm.tirx


def _signature(first_shape=None):
    m, n = tir.Var("M", "int32"), tir.Var("N", "int32")
    first_shape = (m * 2 + n,) if first_shape is None else first_shape(m, n)
    buffers = [
        tir.decl_buffer(shape, "float32", name=name)
        for name, shape in [
            ("composite", first_shape),
            ("workspace", (m + n,)),
            ("m_carrier", (m,)),
            ("n_carrier", (n,)),
            ("output", (2,)),
            ("auto_gm", (16,)),
        ]
    ]
    params = [tir.Var(b.name + "_handle", "handle") for b in buffers]
    return tir.PrimFunc(params, tir.Evaluate(0), buffer_map=dict(zip(params, buffers))).with_attr(
        {"global_symbol": "main", "calling_conv": tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH, "auto_gm_indices": [5]}
    )


def _adapter(func, pto=True):
    # Isolate the runtime shape-map component while public PTO guards remain.
    adapter = object.__new__(CythonKernelAdapter)
    adapter.ir_module = tvm.IRModule.from_expr(func)
    adapter.target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["pto"] if pto else ["ascend"]})
    adapter._a2a3_runtime = True
    adapter._automatic_indices = {1, 4, 5}
    return adapter


def _capture(func, library, composite_length=13):
    adapter = _adapter(func)
    symbols = adapter._process_dynamic_symbolic()
    wrapper = CythonKernelWrapper([4], extrac_params(func), library, lambda: torch.device("cpu"), lambda _: 0)
    wrapper.set_dynamic_symbolic_map(symbols)
    wrapper.set_dynamic_symbolic_sources(adapter._process_dynamic_symbolic_sources())
    wrapper.set_buffer_dtype_map({})
    wrapper.set_buffer_device_map({})
    wrapper.set_static_shape_map({})
    wrapper.set_static_strides_map({})
    wrapper.set_static_contiguous_list([])
    wrapper.set_ptr_map({})
    wrapper.set_a2a3_runtime([1], [5])
    output = wrapper.forward([torch.zeros(composite_length), torch.zeros(4), torch.zeros(5)], stream=17)
    return output, symbols


@pytest.mark.parametrize("reverse", [False, True])
def test_recursive_first_occurrence_is_independent_of_carrier_order_and_json_identity(reverse):
    func = _signature(lambda m, n: (n + m,) if reverse else (m * 2 + n,))
    expected = ["N", "M"] if reverse else ["M", "N"]
    for candidate in [func, load_runtime_signature(dump_runtime_signature(func))]:
        symbols = _adapter(candidate)._process_dynamic_symbolic()
        assert [str(var) for var in symbols] == expected
        assert list(symbols.values()) == ([(0, 3, 0, 1), (0, 2, 0, 1)] if reverse else [(0, 2, 0, 1), (0, 3, 0, 1)])
    # AscendC remains the original direct-carrier traversal; this change does
    # not silently change its already validated shape ABI.
    assert [str(var) for var in _adapter(func, pto=False)._process_dynamic_symbolic()] == ["M", "N"]


def test_runtime_values_and_automatic_allocation_follow_the_recursive_host_slots():
    received = []
    output, symbols = _capture(_signature(lambda m, n: (n + m,)), SimpleNamespace(call=lambda *args: received.append(args)), 9)
    assert output.shape == (2,)
    assert [str(var) for var in symbols] == ["N", "M"]
    assert len(received[0]) == 9  # six pointer slots, two int64 shapes, stream.
    assert [arg.value for arg in received[0][-3:]] == [5, 4, 17]
    assert all(isinstance(arg, ctypes.c_int64) for arg in received[0][-3:-1])


def test_composite_only_shape_never_invents_an_inverse():
    m = tir.Var("M", "int32")
    b = tir.decl_buffer((m * 2 + 1,), "float32", name="composite")
    func = tir.PrimFunc([b.data], tir.Evaluate(0), buffer_map={b.data: b})
    with pytest.raises(ValueError, match="M requires a direct Var dimension.*inversion is unsupported"):
        order_pto_shape_symbols(func, {})


def test_explicit_shape_parameter_keeps_the_source_duplicate_signature_boundary():
    m = tir.Var("M", "int32")
    b = tir.decl_buffer((m,), "float32", name="input")
    func = tir.PrimFunc([b.data, m], tir.Evaluate(0), buffer_map={b.data: b})
    with pytest.raises(ValueError, match="overlaps an explicit parameter"):
        order_pto_shape_symbols(func, {})


def test_distinct_vars_with_the_same_literal_name_are_not_conflated():
    left, right = tir.Var("M", "int32"), tir.Var("M", "int32")
    b = tir.decl_buffer((left + right,), "float32", name="input")
    func = tir.PrimFunc([b.data], tir.Evaluate(0), buffer_map={b.data: b})
    with pytest.raises(ValueError, match="name M is ambiguous"):
        order_pto_shape_symbols(func, {left: (0, 0, 0, 1), right: (0, 0, 0, 1)})


def test_independent_process_restores_recursive_signature_and_native_int64_slots(tmp_path):
    # This CPU C launcher isolates the original PTO host ABI. Full Bisheng/PTO
    # source and public cache tests follow native+driver component integration.
    library = tmp_path / "host.so"
    source = """#include <stdint.h>
    extern "C" void call(float *composite, float *workspace, float *m, float *n,
                         float *output, float *gm, int64_t M, int64_t N, void *stream) {
      workspace[0] = M + N; gm[0] = 16; output[0] = M; output[1] = N;
    }"""
    subprocess.run(
        ["c++", "-shared", "-fPIC", "-x", "c++", "-", "-o", str(library)], input=source, text=True, check=True, capture_output=True
    )
    signature = tmp_path / "signature.json"
    signature.write_text(dump_runtime_signature(_signature()))
    child = tmp_path / "restore.py"
    child.write_text(f"""import ctypes,json,importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location("pto_abi_fixture",{str(Path(__file__))!r})
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
func=fixture.load_runtime_signature(Path({str(signature)!r}).read_text())
library=ctypes.CDLL({str(library)!r});library.call.restype=None
output,symbols=fixture._capture(func,library)
print("PTO_RESTORE="+json.dumps({{"order":[str(v)for v in symbols],"slots":list(symbols.values()),"actual":output.tolist()}}))
""")
    result = subprocess.run([sys.executable, str(child)], cwd=tmp_path, env=os.environ.copy(), text=True, capture_output=True, check=True)
    restored = json.loads(next(line.removeprefix("PTO_RESTORE=") for line in result.stdout.splitlines() if line.startswith("PTO_RESTORE=")))
    assert restored == {"order": ["M", "N"], "slots": [[0, 2, 0, 1], [0, 3, 0, 1]], "actual": [4.0, 5.0]}
