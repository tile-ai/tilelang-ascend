# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""K-loop C prefix publishes once before a flat V epilogue, with CCE sync off."""

import pytest
import torch

import tilelang
from tilelang import tvm

from test_ascend_upstream_sync_cross_core_loop import (
    _M,
    _N,
    _K_TILE,
    _PASS_CONFIGS,
    _developer_kloop_gemm_exp,
)


pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="requires Ascend NPU execution",
)


@pytest.fixture(scope="module", autouse=True)
def _requires_proof_service():
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")


@pytest.mark.parametrize("k_tiles", [2, 3])
@pytest.mark.parametrize("alias", ["noalias", "separate", "A", "B"])
def test_c_prefix_completes_every_k_iteration_before_publication(k_tiles, alias):
    program = _developer_kloop_gemm_exp(k_tiles, explicit_noalias=(alias == "noalias"))
    tilelang.disable_cache()
    kernel = tilelang.compile(
        program,
        out_idx=[],
        execution_backend="cython",
        target="ascendc",
        platform="A3",
        pass_configs=_PASS_CONFIGS,
        compile_flags=["--cce-auto-sync=off"],
    )
    total_k = k_tiles * _K_TILE
    generator = torch.Generator().manual_seed(20261031 + k_tiles)
    for _invocation in range(3):
        first = (torch.randn((_M, total_k), generator=generator) * 0.125).half()
        second = (torch.randn((total_k, _N), generator=generator) * 0.125).half()
        expected = (first.float() @ second.float()).half().exp().half()
        first_device, second_device = first.npu(), second.npu()
        if alias in ("A", "B"):
            owner = first_device if alias == "A" else second_device
            output = owner.view(-1)[: _M * _N].view(_M, _N)
            assert output.data_ptr() == owner.data_ptr()
        else:
            output = torch.empty((_M, _N), dtype=torch.float16, device=first_device.device)
        kernel(first_device, second_device, output)
        torch.npu.synchronize()
        torch.testing.assert_close(output.cpu(), expected, rtol=1e-2, atol=1e-2)
