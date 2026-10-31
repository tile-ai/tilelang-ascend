"""Run the A2A3 automatic-memory/void-launch ABI with CPU tensors and a C library."""

import ctypes
import subprocess
from types import SimpleNamespace

import pytest
import torch
from tilelang import tvm
from tilelang.engine.param import KernelParam, dump_kernel_params, load_kernel_params
from tilelang.ascend.a2a3.runtime import configure_void_call, normalize_memory_indices, resolve_allocation_shape
from tilelang_cython_wrapper import CythonKernelWrapper


tir = tvm.tirx


def _wrapper(params, result_idx, workspace_idx=(), auto_gm_idx=(), symbolic=None, *, a2a3=True, library=None, providers=None):
    if library is None:
        library = SimpleNamespace(call=lambda *args: None if a2a3 else 0)
    if providers is None:
        providers = (lambda: torch.device("cpu"), lambda device: 0)
    wrapper = CythonKernelWrapper(result_idx, params, library, *providers)
    wrapper.set_dynamic_symbolic_map(symbolic or {})
    wrapper.set_dynamic_symbolic_sources({})
    wrapper.set_buffer_device_map({})
    wrapper.set_buffer_dtype_map({})
    wrapper.set_static_shape_map({})
    wrapper.set_static_strides_map({})
    wrapper.set_static_contiguous_list([])
    wrapper.set_ptr_map({})
    if a2a3:
        wrapper.set_a2a3_runtime(workspace_idx, auto_gm_idx)
    return wrapper


def _params(*shapes, dtype="float32"):
    return [KernelParam(tvm.DataType(dtype), list(shape)) for shape in shapes]


@pytest.mark.parametrize("value,expected", [(None, []), (0, [0]), (-1, [3]), (-4, [0]), ([0, -1, -3], [0, 3, 1]), ([1, 1], [1, 1])])
def test_source_index_forms_and_negative_indices(value, expected):
    before = value.copy() if isinstance(value, list) else value
    assert normalize_memory_indices(value, 4, "workspace_idx") == expected
    assert value == before


@pytest.mark.parametrize("value", [4, -5, [4], [-5], (0,), "0", ["0"]])
def test_source_index_rejections(value):
    with pytest.raises(ValueError, match="workspace_idx"):
        normalize_memory_indices(value, 4, "workspace_idx")


def test_dynamic_workspace_expressions_and_parameter_reflection():
    n = tir.Var("N", "int32")
    params = load_kernel_params(dump_kernel_params(_params((n * 2 + 1,), (n,), (tir.ceildiv(n, 4),))))
    # Cache params and the frontend own separate Var handles, so resolve by name.
    assert resolve_allocation_shape(params[0].shape, {n: (0, 1, 0, 1)}, {}, [None, torch.zeros(9)]) == [19]
    assert resolve_allocation_shape(params[2].shape, {n: (0, 1, 0, 1)}, {}, [None, torch.zeros(9)]) == [3]


def test_shape_can_use_a_later_available_carrier():
    n = tir.Var("N", "int32")
    assert resolve_allocation_shape([n * 2], {n: (0, 0, 0, 1)}, {"N": [(0, 0, 0, 1), (0, 1, 0, 1)]}, [None, torch.zeros(7)]) == [14]


def test_missing_symbol_keeps_source_error_boundary():
    with pytest.raises(KeyError, match="Unfounded symbolic var"):
        resolve_allocation_shape([tir.Var("missing", "int32") * 2], {}, {}, [])


def test_compiler_gm_dynamic_shapes_remain_explicitly_unsupported():
    n = tir.Var("N", "int32")
    with pytest.raises(NotImplementedError, match="dynamic auto_gm_idx"):
        resolve_allocation_shape([n * 2], {n: (0, 0, 0, 1)}, {}, [torch.zeros(7)], True)


def test_interleaved_auto_allocations_and_implicit_native_shape_order(monkeypatch):
    n, m = tir.Var("N", "int32"), tir.Var("M", "int32")
    source, other = torch.zeros(7), torch.zeros(3)
    calls, allocations = [], []
    empty = torch.empty

    def allocate(*shape, **kwargs):
        allocations.append(tuple(shape))
        return empty(*shape, **kwargs)

    monkeypatch.setattr(torch, "empty", allocate)
    params = _params((n * 2,), (n,), (n + 1,), (m,), (16,))
    wrapper = _wrapper(
        params, [2], [0], [4], {n: (0, 1, 0, 1), m: (0, 3, 0, 1)}, library=SimpleNamespace(call=lambda *args: calls.append(args))
    )
    output = wrapper.forward([source, other], stream=42)
    assert tuple(output.shape) == (8,)
    assert allocations == [(14,), (8,), (16,)]
    assert len(calls[0]) == 8
    assert calls[0][1].value == source.data_ptr()
    assert calls[0][3].value == other.data_ptr()
    assert [value.value for value in calls[0][-3:]] == [7, 3, 42]


def test_output_workspace_overlap_is_allocated_once(monkeypatch):
    allocations = []
    empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *shape, **kwargs: (allocations.append(shape), empty(*shape, **kwargs))[1])
    wrapper = _wrapper(_params((4,), (4,)), [1], [1], [1])
    output = wrapper.forward([torch.zeros(4)], stream=0)
    assert output.shape == (4,)
    assert len(allocations) == 1


def test_workspace_without_output_returns_source_empty_list():
    wrapper = _wrapper(_params((4,), (4,)), [], [1])
    assert wrapper.forward([torch.zeros(4)], stream=0) == []


def test_raw_source_extra_python_shape_arguments_are_ignored():
    wrapper = _wrapper(_params((4,), (4,)), [1])
    assert wrapper.forward([torch.zeros(4), 4], stream=0).shape == (4,)


def test_shared_default_abi_still_checks_arity_and_static_shapes():
    wrapper = _wrapper(_params((4,), (4,)), [1], a2a3=False)
    with pytest.raises(ValueError, match="Expected"):
        wrapper.forward([torch.zeros(4), 4], stream=0)
    wrapper.set_static_shape_map({"source": (0, [(0, 4)])})
    with pytest.raises(ValueError, match="Static shape mismatch"):
        wrapper.forward([torch.zeros(3)], stream=0)
    source_wrapper = _wrapper(_params((4,), (4,)), [1])
    source_wrapper.set_static_shape_map({"source": (0, [(0, 4)])})
    assert source_wrapper.forward([torch.zeros(3)], stream=0).shape == (4,)


def test_a2a3_still_validates_dtype_device_and_contiguity():
    wrapper = _wrapper(_params((4,), (4,)), [1])
    wrapper.set_buffer_dtype_map({"source": (0, torch.float32)})
    with pytest.raises(ValueError, match="dtype mismatch"):
        wrapper.forward([torch.zeros(4, dtype=torch.float64)], stream=0)
    wrapper.set_buffer_dtype_map({})
    wrapper.set_buffer_device_map({"source": (0, torch.device("cuda"))})
    with pytest.raises(ValueError, match="device mismatch"):
        wrapper.forward([torch.zeros(4)], stream=0)
    wrapper.set_buffer_device_map({})
    wrapper.set_static_contiguous_list([(0, "source")])
    with pytest.raises(ValueError, match="contiguous"):
        wrapper.forward([torch.zeros(8)[::2]], stream=0)


def test_native_void_call_with_dynamic_workspace(tmp_path):
    path = tmp_path / "void_launcher.so"
    source = """
    #include <stdint.h>
    extern "C" void call(float *workspace, float *input, float *output, int64_t n, void *stream) {
      for (int64_t i = 0; i < n; ++i) {
        workspace[2*i] = input[i] * 2;
        output[i] = workspace[2*i] + 1;
      }
    }
    """
    subprocess.run(["c++", "-shared", "-fPIC", "-x", "c++", "-", "-o", str(path)], input=source, text=True, capture_output=True, check=True)
    library = ctypes.CDLL(str(path))
    configure_void_call(library)
    assert library.call.restype is None
    assert not hasattr(library, "init")
    assert not hasattr(library, "get_last_error")
    n = tir.Var("N", "int32")
    wrapper = _wrapper(_params((n * 2,), (n,), (n,)), [2], [0], symbolic={n: (0, 1, 0, 1)}, library=library)
    source = torch.arange(13, dtype=torch.float32)
    output = wrapper.forward([source], stream=0)
    torch.testing.assert_close(output, source * 2 + 1)


def test_shared_default_status_return_errors_remain_visible():
    library = SimpleNamespace(call=lambda *args: 7, get_last_error=lambda: b"original status failure")
    wrapper = _wrapper(_params((4,), (4,)), [1], a2a3=False, library=library)
    with pytest.raises(RuntimeError, match="original status failure"):
        wrapper.forward([torch.zeros(4)], stream=0)


def test_real_derived_workspace_parameter_extractor_and_signature_roundtrip():
    import importlib.util
    from pathlib import Path
    from tvm.ir import CallingConv
    from tilelang.ascend.a2a3.pipeline import ASCENDC_PIPELINE, get_device_kernel
    from tilelang.ascend.a2a3.runtime import dump_runtime_signature, load_runtime_signature

    if tvm.ffi.get_global_func("tl.transform.A2A3AscendWorkspaceReduction", allow_missing=True) is None:
        pytest.skip("requires the real A2A3 workspace pass in a USE_ASCEND build")
    path = Path(__file__).parents[1] / "transform" / "test_a2a3_workspace_reduction.py"
    spec = importlib.util.spec_from_file_location("a2a3_workspace_source_fixture", path)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    # A real migrated source pass adds the GM argument. Do not fake the public
    # pipeline or replace the native workspace algorithm to manufacture it.
    before = source._kernel("v2c", threads=2)
    assert len(before.params) == 0
    after = source._run(before).with_attr("calling_conv", CallingConv.DEVICE_KERNEL_LAUNCH)
    module = tvm.IRModule({"main": after})
    selected = get_device_kernel(module)
    params = ASCENDC_PIPELINE.extract_parameters(module)
    assert len(params) == len(selected.params) == 1
    assert list(params[0].shape) == [4, 16, 16]
    assert [int(value) for value in selected.attrs["auto_gm_indices"]] == [0]
    restored = load_runtime_signature(dump_runtime_signature(selected))
    assert len(restored.params) == 1
    assert list(restored.buffer_map[restored.params[0]].shape) == [4, 16, 16]
    assert restored.buffer_map[restored.params[0]].dtype == params[0].dtype
    assert [int(value) for value in restored.attrs["auto_gm_indices"]] == [0]
    # Compare under the PrimFunc parameter binders; a bare identity-keyed map
    # has no binder establishing correspondence for restored handle keys.
    tvm.ir.assert_structural_equal(
        tir.PrimFunc(selected.params, tir.Evaluate(0), buffer_map=selected.buffer_map),
        tir.PrimFunc(restored.params, tir.Evaluate(0), buffer_map=restored.buffer_map),
    )


@pytest.mark.parametrize("integer", [17, 2**40 + 7])
def test_a2a3_preserves_source_scalar_argument_encoding(integer):
    calls = []
    params = [KernelParam(tvm.DataType(dtype), shape) for dtype, shape in [("float32", [4]), ("int64", []), ("float64", []), ("bool", [])]]
    wrapper = _wrapper(params, [], library=SimpleNamespace(call=lambda *args: calls.append(args)))
    wrapper.forward([torch.zeros(4), integer, 1.25, True], stream=0)
    assert calls[0][1] is integer
    # This deliberately preserves TA's c_float encoding, including its old
    # double/big-integer limitation. No native numeric result is claimed.
    assert isinstance(calls[0][2], ctypes.c_float)
    assert calls[0][2].value == 1.25
    assert calls[0][3] is True


def test_a2a3_none_rejection_does_not_change_shared_nullable_inputs():
    wrapper = _wrapper(_params((4,)), [])
    with pytest.raises(ValueError, match="Unsupported tensor type"):
        wrapper.forward([None], stream=0)
    shared = _wrapper(_params((4,)), [], a2a3=False)
    assert shared.forward([None], stream=0) == []


def test_a2a3_first_input_device_rule_keeps_source_scalar_first_boundary():
    params = [KernelParam(tvm.DataType("int32"), []), *_params((4,), (4,))]
    wrapper = _wrapper(params, [2])
    with pytest.raises(AttributeError, match="device"):
        wrapper.forward([3, torch.zeros(4)], stream=0)
    shared = _wrapper(params, [2], a2a3=False)
    assert shared.forward([3, torch.zeros(4)], stream=0).shape == (4,)


def test_source_device_validation_does_not_accept_scalar_first_tensor_kernels():
    params = [KernelParam(tvm.DataType("int32"), []), *_params((4,))]
    wrapper = _wrapper(params, [])
    wrapper.set_buffer_device_map({"input": (1, torch.device("cpu"))})
    with pytest.raises(ValueError, match="first parameter slot"):
        wrapper.forward([3, torch.zeros(4)], stream=0)
    shared = _wrapper(params, [], a2a3=False)
    shared.set_buffer_device_map({"input": (1, torch.device("cpu"))})
    assert shared.forward([3, torch.zeros(4)], stream=0) == []


def test_a2a3_stream_uses_current_device_and_shared_uses_input_device():
    observed = []
    providers = (lambda: torch.device("cpu", 7), lambda device: (observed.append(device), 17)[1])
    source = torch.zeros(4)
    wrapper = _wrapper(_params((4,), (4,)), [1], providers=providers)
    assert wrapper.forward([source]).device == source.device
    assert observed == [torch.device("cpu", 7)]
    observed.clear()
    shared = _wrapper(_params((4,), (4,)), [1], providers=providers, a2a3=False)
    assert shared.forward([source]).device == source.device
    assert observed == [source.device]


def test_shared_default_int4_output_still_uses_int8_storage():
    # The shared constructor retains its storage policy without A2A3 opt-in.
    wrapper = _wrapper(_params((2, 18), (2, 18), dtype="int4"), [1], a2a3=False)
    output = wrapper.forward([torch.empty((2, 18), dtype=torch.int8)], stream=0)
    assert output.dtype == torch.int8
    assert output.shape == (2, 18) and output.stride() == (18, 1)
