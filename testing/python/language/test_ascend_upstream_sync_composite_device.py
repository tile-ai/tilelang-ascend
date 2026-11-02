# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Device regressions for the audited flat AscendC gemm_v0 helper contract."""

import pytest
import torch

import tilelang
from tilelang import tvm

from test_ascend_upstream_sync_device import _PASS_CONFIGS
from test_ascend_upstream_sync_composite_loop import _ordinary_k_gemm
from test_tilelang_ascend_language_gemm_v0_n_tiling import gemm_nt


pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="requires Ascend NPU execution",
)


@pytest.fixture(scope="module", autouse=True)
def _requires_proof_service() -> None:
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")


@pytest.mark.parametrize("k", [64, 256])
def test_flat_gemm_internal_event_pools_are_drained_between_invocations(k: int) -> None:
    # The existing factory contains one flat gemm_v0 call, with no manual
    # barriers or flags. K=256 also exercises its internal L0 ping-pong.
    tilelang.disable_cache()
    kernel = tilelang.compile(
        gemm_nt(16, 16, k, "float16", "float"),
        out_idx=[-1],
        target="ascendc",
        platform="A3",
        pass_configs=_PASS_CONFIGS,
        compile_flags=["--cce-auto-sync=off"],
    )
    generator = torch.Generator().manual_seed(20261030 + k)
    for _invocation in range(3):
        first = torch.randn((16, k), generator=generator, dtype=torch.float16)
        second = torch.randn((k, 16), generator=generator, dtype=torch.float16)
        expected = (first.float() @ second.float()).half()
        actual = kernel(first.npu(), second.npu())
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("trip_count", [2, 3])
def test_ordinary_k_loop_without_manual_sync(trip_count: int) -> None:
    tilelang.disable_cache()
    kernel = tilelang.compile(
        _ordinary_k_gemm(trip_count),
        out_idx=[-1],
        target="ascendc",
        platform="A3",
        pass_configs=_PASS_CONFIGS,
        compile_flags=["--cce-auto-sync=off"],
    )
    full_k = trip_count * 64
    generator = torch.Generator().manual_seed(20261031 + trip_count)
    for _invocation in range(3):
        first = torch.randn((16, full_k), generator=generator, dtype=torch.float16)
        second = torch.randn((full_k, 16), generator=generator, dtype=torch.float16)
        expected = (first.float() @ second.float()).half()
        actual = kernel(first.npu(), second.npu())
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-2, atol=1e-2)
