# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Keep external-output C/V publication safe across equivalent allocator roles."""

from __future__ import annotations

import pytest
import torch

import tilelang
from tilelang import tvm
from tilelang.jit.ascend_runtime_contract import (
    AUTO_GM_FRESH_CONTRACT_ATTR,
    AUTO_GM_FRESH_CONTRACT_VERSION,
    CYTHON_ALLOCATOR_FRESH_CONTRACT_VERSION,
)
from test_ascend_upstream_sync_cross_core_device import (
    _K,
    _M,
    _N,
    _PASS_CONFIGS,
    _flat_developer_gemm_exp,
)

pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="requires Ascend NPU execution",
)


@pytest.fixture(autouse=True)
def _requires_proof_service_and_disables_cache():
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")
    previous = tilelang.is_cache_enabled()
    tilelang.disable_cache()
    yield
    if previous:
        tilelang.enable_cache()


@pytest.mark.parametrize(
    "workspace_idx",
    [None, [-1], [3], [-1, 3, -1]],
    ids=["auto", "negative", "positive", "duplicate-overlap"],
)
@pytest.mark.parametrize("output_alias", [None, "A", "B"], ids=["separate", "alias-A", "alias-B"])
def test_external_output_alias_is_safe_with_explicit_auto_workspace_roles(
    workspace_idx: list[int] | None,
    output_alias: str | None,
) -> None:
    program = _flat_developer_gemm_exp(explicit_noalias=False)
    recipe = None if workspace_idx is None else workspace_idx.copy()
    kernel = tilelang.compile(
        program,
        out_idx=[],
        workspace_idx=workspace_idx,
        execution_backend="cython",
        target="ascendc",
        platform="A3",
        pass_configs=_PASS_CONFIGS,
        compile_flags=["--cce-auto-sync=off"],
    )
    assert workspace_idx == recipe
    assert "tir.noalias" not in kernel.prim_func.attrs
    version = kernel.prim_func.attrs[AUTO_GM_FRESH_CONTRACT_ATTR].value
    expected_version = AUTO_GM_FRESH_CONTRACT_VERSION
    if workspace_idx is not None:
        expected_version = CYTHON_ALLOCATOR_FRESH_CONTRACT_VERSION
    assert version == expected_version
    assert len(kernel.params) == 4 and kernel.auto_gm_idx == [3]
    source = kernel.get_kernel_source()
    assert source.count("CrossCoreSetFlag<2, PIPE_FIX>") == 1
    assert source.count("CrossCoreWaitFlag(") == 1
    assert "HardEvent::S_MTE2" in source

    generator = torch.Generator().manual_seed(20261103)
    for _invocation in range(3):
        first = (torch.randn((_M, _K), generator=generator) * 0.125).half()
        second = (torch.randn((_K, _N), generator=generator) * 0.125).half()
        expected = (first.float() @ second.float()).half().exp().half()
        first_device, second_device = first.npu(), second.npu()
        if output_alias is None:
            output = torch.empty((_M, _N), dtype=torch.float16, device=first_device.device)
        else:
            owner = first_device if output_alias == "A" else second_device
            output = owner.view(-1)[: _M * _N].view(_M, _N)
            assert output.data_ptr() == owner.data_ptr()
        assert kernel(first_device, second_device, output) == []
        torch.npu.synchronize()
        torch.testing.assert_close(output.cpu(), expected, rtol=1e-2, atol=1e-2)
