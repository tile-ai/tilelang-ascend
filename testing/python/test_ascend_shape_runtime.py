# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Exercise the actual Cython shape ABI on CPU without allocating large tensors."""

from types import SimpleNamespace

import pytest
import torch

from tilelang import tvm
from tilelang.engine.param import KernelParam
from tilelang.jit.adapter.cython.adapter import CythonKernelWrapper
from tilelang.jit.adapter.cython.shape import checked_shape_value


class _ShapeTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, dimensions):
        return torch.Tensor._make_subclass(cls, torch.ones(1), False)

    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.shape_reads = 0

    @property
    def shape(self):
        index = min(self.shape_reads, len(self.dimensions) - 1)
        self.shape_reads += 1
        return (self.dimensions[index],)


def _wrapper(monkeypatch, symbol, calls, *, allocate_output=False):
    monkeypatch.setattr(torch.npu, "current_device", lambda: 0)
    params = [KernelParam(torch.float32, [symbol])]
    if allocate_output:
        params.append(KernelParam(torch.float32, [symbol]))
    wrapper = CythonKernelWrapper(
        [1] if allocate_output else [],
        [],
        [],
        params,
        SimpleNamespace(call=lambda *args: calls.append(args)),
    )
    wrapper.set_dynamic_symbolic_map({symbol: (0, 0)})
    for setter in (
        wrapper.set_buffer_dtype_map,
        wrapper.set_static_shape_map,
        wrapper.set_buffer_device_map,
        wrapper.set_ptr_map,
    ):
        setter({})
    return wrapper


@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64", "uint8", "uint32", "uint64"])
def test_shape_domain_matches_declared_integer_type_and_signed_launch_abi(dtype):
    symbol = tvm.tir.Var("size", dtype)
    bits = int(dtype.removeprefix("uint").removeprefix("int"))
    maximum = min((1 << (bits - dtype.startswith("int"))) - 1, (1 << 63) - 1)
    assert checked_shape_value(symbol, 0) == 0
    assert checked_shape_value(symbol, maximum) == maximum
    for value in (-1, maximum + 1):
        with pytest.raises(ValueError, match="dimension range"):
            checked_shape_value(symbol, value)


@pytest.mark.parametrize("dtype", ["bool", "float32", "int32x2"])
def test_shape_domain_rejects_non_scalar_integer_declarations(dtype):
    with pytest.raises(TypeError, match="scalar integer"):
        checked_shape_value(tvm.tir.Var("size", dtype), 1)


@pytest.mark.parametrize("value", [0, (1 << 31) - 1])
def test_cython_wrapper_passes_declared_shape_boundary_to_int64_abi(monkeypatch, value):
    calls = []
    wrapper = _wrapper(monkeypatch, tvm.tir.Var("size", "int32"), calls)
    tensor = _ShapeTensor([value])
    wrapper.forward([tensor], stream=0)
    assert tensor.shape_reads == 1
    assert len(calls) == 1
    assert calls[0][0].value == tensor.data_ptr()
    assert calls[0][1].value == value


@pytest.mark.parametrize("value", [-1, 1 << 31, 1 << 63])
def test_cython_wrapper_rejects_shape_before_result_allocation_or_launch(monkeypatch, value):
    calls = []
    wrapper = _wrapper(monkeypatch, tvm.tir.Var("size", "int32"), calls, allocate_output=True)
    tensor = _ShapeTensor([value])

    def unexpected_allocation(*args, **kwargs):
        pytest.fail("out-of-domain shape reached result allocation")

    monkeypatch.setattr(torch, "empty", unexpected_allocation)
    with pytest.raises(ValueError, match="int32 dimension range"):
        wrapper.forward([tensor], stream=0)
    assert not calls


def test_cython_wrapper_allocates_and_launches_from_the_same_shape_snapshot(monkeypatch):
    calls = []
    wrapper = _wrapper(monkeypatch, tvm.tir.Var("size", "int32"), calls, allocate_output=True)
    tensor = _ShapeTensor([7, 1 << 31])
    result = wrapper.forward([tensor], stream=0)
    assert tensor.shape_reads == 1
    assert result.shape == (7,)
    assert len(calls) == 1
    assert calls[0][2].value == 7


def test_database_adapter_validates_the_original_shape_domain_before_launch(monkeypatch):
    from tilelang.jit.adapter.cython import adapter as cython_adapter

    symbol = tvm.tir.Var("size", "int32")
    buffer = tvm.tir.decl_buffer((symbol,), "float32", name="source")
    function = tvm.tir.PrimFunc(
        [buffer.data],
        tvm.tir.Evaluate(0),
        buffer_map={buffer.data: buffer},
    ).with_attr("global_symbol", "shape_test")
    calls = []
    monkeypatch.setattr(torch.npu, "current_device", lambda: 0)
    monkeypatch.setattr(
        cython_adapter,
        "LibraryGenerator",
        lambda *args: SimpleNamespace(
            load_lib=lambda **kwargs: SimpleNamespace(call=lambda *args: calls.append(args)),
        ),
    )
    for name in ("buffer_dtype", "static_shape", "buffer_device", "ptr_map"):
        monkeypatch.setattr(cython_adapter.CythonKernelAdapter, f"_process_{name}", lambda self: {})
    adapter = cython_adapter.CythonKernelAdapter.from_database(
        [KernelParam(torch.float32, [symbol])],
        [],
        [],
        [],
        "ascendc",
        "A3",
        function,
        "",
        "/unused.so",
    )
    assert next(iter(adapter.dynamic_symbolic_map)).same_as(symbol)
    with pytest.raises(ValueError, match="int32 dimension range"):
        adapter(_ShapeTensor([1 << 31]), stream=0)
    assert not calls
