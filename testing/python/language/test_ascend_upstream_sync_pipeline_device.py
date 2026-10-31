# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Device checks for physical rings emitted by the existing pipeline injector."""

import pytest
import torch

import tilelang
from tilelang import language as T
from tilelang import tvm

from test_ascend_pipeline_ring_provenance import _pipelined_abs
from test_ascend_upstream_sync_device import _PASS_CONFIGS, _compile_device_kernel


pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="requires Ascend NPU execution",
)


@pytest.fixture(scope="module", autouse=True)
def _requires_proof_service() -> None:
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")


@pytest.mark.parametrize("num_stages,trip_count", [(2, 4), (2, 5), (3, 7), (3, 8)])
def test_upstream_sync_injected_ring_wrap_and_repeated_invocation(num_stages: int, trip_count: int) -> None:
    kernel = _compile_device_kernel(_pipelined_abs(trip_count, num_stages))
    source = torch.arange(trip_count * 64, dtype=torch.float32).reshape(trip_count, 64)
    output = torch.empty((2, trip_count, 64), dtype=torch.float32, device="npu")
    for iteration in range(3):
        values = source - (iteration + 1) * 97
        output.fill_(-999)
        kernel(values.npu(), output)
        expected = values.abs().unsqueeze(0).expand(2, -1, -1)
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@pytest.fixture(
    scope="module",
    params=[(2, False), (2, True), (3, False), (3, True)],
    ids=["b2-unplanned", "b2-planned", "b3-unplanned", "b3-planned"],
)
def dynamic_pipeline_kernel(request):
    # One compiled kernel must traverse the zero, short, zero-steady and long
    # cases. Recompiling for each n would hide errors in the runtime partition.
    stages, planning = request.param
    program = _pipelined_abs(T.symbolic("n"), stages).with_attr("tir.noalias", True)
    tilelang.disable_cache()
    return tilelang.compile(
        program,
        out_idx=[],
        target="ascendc",
        platform="A3",
        pass_configs={
            **_PASS_CONFIGS,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: planning,
        },
        compile_flags=["--cce-auto-sync=off"],
    )


def test_dynamic_pipeline_short_cases_wrap_and_repeated_invocation(dynamic_pipeline_kernel) -> None:
    # Alternate case lengths across invocations to expose unconsumed event
    # tokens as well as wraparound and a steady loop whose extent is zero.
    for trip_count in (0, 1, 2, 3, 5, 33, 1, 0, 5):
        source = torch.arange(trip_count * 64, dtype=torch.float32).reshape(trip_count, 64)
        output = torch.full((2, trip_count, 64), -999.0, dtype=torch.float32, device="npu")
        for invocation in range(2):
            values = source - (invocation + 1) * 97
            dynamic_pipeline_kernel(values.npu(), output)
            expected = values.abs().unsqueeze(0).expand(2, -1, -1)
            torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
