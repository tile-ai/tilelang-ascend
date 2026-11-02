# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Real Persistent execution with the upstream synchronization engine enabled."""

import pytest
import torch

from tilelang import tvm

from test_ascend_upstream_sync_device import _compile_device_kernel
from test_ascend_upstream_sync_persistent import _persistent_abs


pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="requires Ascend NPU execution",
)


@pytest.fixture(scope="module", autouse=True)
def _requires_proof_service():
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")


def _check(kernel, rows):
    for invocation in range(3):
        values = torch.arange(2 * rows * 64, dtype=torch.float32).reshape(2, rows, 64)
        values -= (invocation + 1) * 97
        output = torch.full((2, rows, 64), -999.0, dtype=torch.float32, device="npu")
        kernel(values.npu(), output)
        torch.testing.assert_close(output.cpu(), values.abs(), rtol=0, atol=0)


@pytest.mark.parametrize(
    "rows,nested",
    [(1, False), (4, False), (5, False), (6, False), (8, False), (5, True), (6, True)],
)
def test_persistent_full_and_partial_waves(rows, nested):
    program = _persistent_abs(rows, nested=nested).with_attr("tir.noalias", True)
    _check(_compile_device_kernel(program), rows)


@pytest.mark.parametrize("nested", [False, True])
def test_one_dynamic_persistent_kernel_handles_empty_and_partial_waves(nested):
    program = _persistent_abs(None, nested=nested).with_attr("tir.noalias", True)
    kernel = _compile_device_kernel(program)
    for rows in (0, 1, 4, 5, 8, 17, 1, 0):
        _check(kernel, rows)
