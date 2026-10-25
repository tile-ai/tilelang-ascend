# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""CPU tensor regressions for Profiler.assert_consistent; no TVM or NPU required."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest import mock

import pytest
import torch


@pytest.fixture(scope="module")
def profiler_class():
    """Load the actual profiler, stubbing only its unused native dependencies."""
    path = Path(__file__).resolve().parents[3] / "tilelang" / "profiler" / "__init__.py"
    name = "_tilelang_profiler_consistency_test"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)

    tensor_utils = ModuleType("tilelang.utils.tensor")
    tensor_utils.TensorSupplyType = object
    tensor_utils.get_tensor_supply = mock.Mock()
    tensor_utils.torch_assert_close = mock.Mock()
    tensor_utils.adapt_torch2tvm = mock.Mock()
    param = ModuleType("tilelang.engine.param")
    param.KernelParam = object
    adapter = ModuleType("tilelang.jit.adapter")
    adapter.BaseKernelAdapter = object
    bench = ModuleType("tilelang.profiler.bench")
    bench.do_bench = mock.Mock()

    with mock.patch.dict(
        sys.modules,
        {
            name: module,
            "tvm": ModuleType("tvm"),
            "tilelang.utils.tensor": tensor_utils,
            "tilelang.engine.param": param,
            "tilelang.jit.adapter": adapter,
            "tilelang.profiler.bench": bench,
        },
    ):
        spec.loader.exec_module(module)
    return module.Profiler


def make_profiler(profiler_class, adapter):
    return profiler_class(params=[], result_idx=[], workspace_idx=[], supply_type=object(), adapter=adapter)


@pytest.mark.parametrize("container", [lambda x: x, lambda x: [x], lambda x: (x,)], ids=["tensor", "list", "tuple"])
def test_stable_reused_buffer(profiler_class, container):
    output = torch.ones(2, 3)
    adapter = mock.Mock(side_effect=lambda: container(output))

    make_profiler(profiler_class, adapter).assert_consistent(repeat=3)

    assert adapter.call_count == 4


@pytest.mark.parametrize("container", [lambda x: x, lambda x: [x], lambda x: (x,)], ids=["tensor", "list", "tuple"])
def test_changing_reused_buffer_is_rejected(profiler_class, container):
    output = torch.zeros(2, 3)

    def adapter():
        output.add_(1)
        return container(output)

    with pytest.raises(AssertionError, match="result is not consistent"):
        make_profiler(profiler_class, adapter).assert_consistent(repeat=2)


def test_reused_output_list_is_snapshotted(profiler_class):
    outputs = [torch.zeros(2)]

    def adapter():
        outputs[0] = outputs[0] + 1
        return outputs

    with pytest.raises(AssertionError, match="result is not consistent"):
        make_profiler(profiler_class, adapter).assert_consistent(repeat=2)


@pytest.mark.parametrize("counts", [(1, 2), (2, 1), (0, 1), (1, 0)])
def test_output_count_change_is_rejected(profiler_class, counts):
    adapter = mock.Mock(side_effect=[[torch.ones(2) for _ in range(n)] for n in counts])

    with pytest.raises(AssertionError, match="output tensors"):
        make_profiler(profiler_class, adapter).assert_consistent(repeat=1)


def test_changing_fresh_buffer_is_rejected(profiler_class):
    adapter = mock.Mock(side_effect=[[torch.zeros(2)], [torch.ones(2)]])

    with pytest.raises(AssertionError, match="result is not consistent"):
        make_profiler(profiler_class, adapter).assert_consistent(repeat=1)


@pytest.mark.parametrize("changing", [False, True])
def test_scalar_tensor(profiler_class, changing):
    output = torch.tensor(1.0)

    def adapter():
        if changing:
            output.add_(1)
        return output

    profiler = make_profiler(profiler_class, adapter)
    if changing:
        with pytest.raises(AssertionError, match="result is not consistent"):
            profiler.assert_consistent(repeat=2)
    else:
        profiler.assert_consistent(repeat=2)


def test_changing_second_output_is_rejected(profiler_class):
    stable = torch.ones(2)
    changing = torch.zeros(2)

    def adapter():
        changing.add_(1)
        return stable, changing

    with pytest.raises(AssertionError, match="result is not consistent"):
        make_profiler(profiler_class, adapter).assert_consistent(repeat=1)


def test_broadcastable_shape_change_is_rejected(profiler_class):
    adapter = mock.Mock(side_effect=[[torch.ones(2, 1)], [torch.ones(2, 3)]])

    with pytest.raises(AssertionError, match="shape"):
        make_profiler(profiler_class, adapter).assert_consistent(repeat=1)


def test_empty_outputs(profiler_class):
    make_profiler(profiler_class, lambda: []).assert_consistent(repeat=2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
