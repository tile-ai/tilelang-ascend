# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""A3 PTO execution for the audited Vector subset, with CCE sync disabled."""

import pytest
import torch

import tilelang
from tilelang import language as T
from tilelang import tvm

from test_ascend_pipeline_ring_provenance import _pipelined_abs
from test_ascend_upstream_sync_codegen_matrix import _same_pipe_abs
from test_ascend_upstream_sync_device import _PASS_CONFIGS
from test_tilelang_ascend_language_gemm_v0_n_tiling import gemm_nt


pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="requires Ascend NPU execution",
)


@pytest.fixture(scope="module", autouse=True)
def _requires_proof_service():
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")


@pytest.mark.parametrize("kind", ["same_pipe", "static_b2", "static_b3", "dynamic_b2", "dynamic_b3"])
def test_a3_pto_vector_and_physical_pipeline_protocol(kind):
    if kind == "same_pipe":
        program, lengths = _same_pipe_abs(), (1,)
    else:
        stages = int(kind[-1])
        count = T.symbolic("n") if kind.startswith("dynamic") else 8
        program = _pipelined_abs(count, stages)
        lengths = (0, 1, 2, 3, 5, 33, 1, 0) if kind.startswith("dynamic") else (8,)
    tilelang.disable_cache()
    kernel = tilelang.compile(
        program.with_attr("tir.noalias", True),
        out_idx=[],
        target="pto",
        platform="A3",
        pass_configs=_PASS_CONFIGS,
        compile_flags=["--cce-auto-sync=off"],
    )
    for length in lengths:
        shape = (64,) if kind == "same_pipe" else (length, 64)
        for invocation in range(3):
            values = torch.arange(length * 64, dtype=torch.float32).reshape(shape)
            values -= (invocation + 1) * 97
            expected = values.abs().unsqueeze(0).expand(2, *shape)
            output = torch.full(expected.shape, -999.0, dtype=torch.float32, device="npu")
            kernel(values.npu(), output)
            torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("k", [128, 256])
def test_a3_pto_gemm_single_scratch_slot_is_drained_between_calls(k):
    tilelang.disable_cache()
    kernel = tilelang.compile(
        gemm_nt(16, 16, k, "float16", "float").with_attr("tir.noalias", True),
        out_idx=[2],
        target="pto",
        platform="A3",
        pass_configs=_PASS_CONFIGS,
        compile_flags=["--cce-auto-sync=off"],
    )
    generator = torch.Generator().manual_seed(20261030 + k)
    for _invocation in range(3):
        first = (torch.randn((16, k), generator=generator) * 0.25).half()
        second = (torch.randn((k, 16), generator=generator) * 0.25).half()
        expected = (first.float() @ second.float()).half()
        actual = kernel(first.npu(), second.npu())
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-2, atol=1e-2)
