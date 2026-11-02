# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Persistent waves containing a real B2 software pipeline, CCE sync disabled."""

import pytest
import torch

import tilelang
from tilelang import tvm

from test_ascend_upstream_sync_device import _PASS_CONFIGS
from test_ascend_upstream_sync_persistent_pipeline import _persistent_pipelined_abs


pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="requires Ascend NPU execution",
)


@pytest.fixture(scope="module", autouse=True)
def _requires_proof_service():
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")


@pytest.mark.parametrize("rows,tiles", [(9, 3), (9, 5), (10, 5), (17, 3)])
@pytest.mark.parametrize("memory_planning", [False, True])
def test_persistent_pipeline_wrap_partial_wave_and_repeated_invocation(rows, tiles, memory_planning):
    tilelang.disable_cache()
    kernel = tilelang.compile(
        _persistent_pipelined_abs(rows, tiles),
        out_idx=[],
        target="ascendc",
        platform="A3",
        pass_configs={
            **_PASS_CONFIGS,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: memory_planning,
        },
        compile_flags=["--cce-auto-sync=off"],
    )
    source = kernel.get_kernel_source()
    assert "PIPE_ALL" not in source
    for event in ("S_V", "S_MTE2", "S_MTE3", "V_S", "MTE2_S", "MTE3_S"):
        assert f"HardEvent::{event}" in source
    for invocation in range(3):
        values = torch.arange(2 * rows * tiles * 64, dtype=torch.float32)
        values = values.reshape(2, rows, tiles, 64) - (invocation + 1) * 397
        output = torch.full(values.shape, -999.0, dtype=torch.float32, device="npu")
        kernel(values.npu(), output)
        torch.npu.synchronize()
        torch.testing.assert_close(output.cpu(), values.abs(), rtol=0, atol=0)
