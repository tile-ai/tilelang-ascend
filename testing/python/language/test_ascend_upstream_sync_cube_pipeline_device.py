# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Static Cube SIP with real L1 rings and CCE automatic sync disabled."""

import pytest
import torch

import tilelang
from tilelang import tvm

from test_ascend_upstream_sync_cube_pipeline import _pipelined_k_gemm
from test_ascend_upstream_sync_device import _PASS_CONFIGS


pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="requires Ascend NPU execution",
)


@pytest.fixture(scope="module", autouse=True)
def _requires_proof_service():
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")


@pytest.mark.parametrize("trip_count", [3, 5, 7])
@pytest.mark.parametrize("memory_planning", [False, True])
def test_cube_pipeline_slot_wrap_and_repeated_invocation(trip_count, memory_planning):
    tilelang.disable_cache()
    kernel = tilelang.compile(
        _pipelined_k_gemm(trip_count),
        out_idx=[2],
        target="ascendc",
        platform="A3",
        pass_configs={
            **_PASS_CONFIGS,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: memory_planning,
        },
        compile_flags=["--cce-auto-sync=off"],
    )
    generator = torch.Generator().manual_seed(20261030 + trip_count)
    for _invocation in range(3):
        first = torch.randn((16, trip_count * 64), generator=generator, dtype=torch.float16)
        second = torch.randn((trip_count * 64, 16), generator=generator, dtype=torch.float16)
        expected = (first.float() @ second.float()).half()
        actual = kernel(first.npu(), second.npu())
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-2, atol=1e-2)
