# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""A3 execution of repeatedly reused bidirectional C/V GM channels."""

import pytest
import torch

import tilelang
from tilelang import language as T

from test_ascend_pipeline_ring_provenance import _nodes, _pre_sync
from test_ascend_upstream_sync_device import _PASS_CONFIGS, _compile_device_kernel


pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="requires Ascend NPU execution",
)


def _two_way_gemm_program(groups, epochs, *, preload_inputs=False, omit_final_drain=False):
    # Retain the ready/free and local init/reuse/drain structure of the
    # Expert FA schedules; small dyadic values make every round checkable.
    rows, columns, width = 32, 16, 16

    @T.prim_func
    def main(
        Workspace: T.Tensor((groups * rows, columns), "float16"),
        Identity: T.Tensor((width, columns), "float16"),
        Output: T.Tensor((groups * rows, columns), "float16"),
    ):
        with T.Kernel(groups, is_npu=True) as (cid, vid):
            left = T.alloc_L1((rows, width), "float16")
            right = T.alloc_L1((width, columns), "float16")
            accumulator = T.alloc_L0C((rows, columns), "float32")
            value = T.alloc_ub((rows // 2, columns), "float16")
            with T.Scope("C"):
                if preload_inputs:
                    T.copy(Workspace[cid * rows, 0], left)
                    T.copy(Identity[:, :], right)
                T.set_flag("S", "MTE2", 6)
                for _epoch in T.serial(epochs):
                    T.wait_cross_flag(0)
                    T.wait_flag("S", "MTE2", 6)
                    if not preload_inputs:
                        T.copy(Workspace[cid * rows, 0], left)
                        T.copy(Identity[:, :], right)
                    T.gemm_v0(left, right, accumulator, init=True, kL0Size=16)
                    T.copy(accumulator, Workspace[cid * rows, 0])
                    T.set_flag("S", "MTE2", 6)
                    T.set_cross_flag("FIX", 1)
                if not omit_final_drain:
                    T.wait_cross_flag(0)
                T.wait_flag("S", "MTE2", 6)
            with T.Scope("V"):
                T.set_flag("S", "MTE2", 6)
                T.set_cross_flag("MTE3", 0)
                for _epoch in T.serial(epochs):
                    T.wait_cross_flag(1)
                    T.wait_flag("S", "MTE2", 6)
                    T.copy(Workspace[cid * rows + vid * (rows // 2), 0], value)
                    T.tile.add(value, value, 0.5)
                    T.copy(value, Workspace[cid * rows + vid * (rows // 2), 0])
                    T.copy(value, Output[cid * rows + vid * (rows // 2), 0])
                    T.set_flag("S", "MTE2", 6)
                    T.set_cross_flag("MTE3", 0)
                T.wait_flag("S", "MTE2", 6)

    return main.with_attr("tir.noalias", True)


@pytest.mark.parametrize(
    "groups,epochs,preload_inputs,omit_final_drain",
    [(1, 2, False, False), (2, 3, False, False), (2, 3, True, False), (2, 3, False, True)],
)
def test_bidirectional_gm_reuse_drains_across_repeated_invocations(
    groups,
    epochs,
    preload_inputs,
    omit_final_drain,
):
    # _compile_device_kernel disables TileLang caching and Bisheng automatic sync.
    assert _PASS_CONFIGS[tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC]
    program = _two_way_gemm_program(
        groups,
        epochs,
        preload_inputs=preload_inputs,
        omit_final_drain=omit_final_drain,
    )
    if preload_inputs:
        before, _ = _pre_sync(program, memory_planning=False)
        gemm_loops = []
        for loop in _nodes(before.body, tilelang.tvm.tir.For):
            calls = _nodes(loop.body, tilelang.tvm.tir.Call)
            if any(call.op.name == "tl.ascend_gemm_v0" for call in calls):
                gemm_loops.append(calls)
        assert len(gemm_loops) == 1
        copies = [call for call in gemm_loops[0] if call.op.name == "tir.call_extern"]
        inner_gm_copies = []
        for call in copies:
            if "copy_gm_to_l1" in call.args[0].value:
                inner_gm_copies.append(call.args[0].value)
        expected_inner_copies = []
        assert inner_gm_copies == expected_inner_copies
    kernel = _compile_device_kernel(program)
    if omit_final_drain:
        generated = kernel.get_kernel_source()
        assert generated.count("CrossCoreSetFlag<") == 3
        assert generated.count("CrossCoreWaitFlag(") == 3
    identity = torch.eye(16, dtype=torch.float16, device="npu")
    workspace = torch.empty((groups * 32, 16), dtype=torch.float16, device="npu")
    output = torch.empty_like(workspace)
    for invocation in range(3):
        source = (torch.arange(groups * 32 * 16).reshape(groups * 32, 16) % 13).half()
        source = source / 8 - invocation
        workspace.copy_(source)
        output.fill_(-456)
        kernel(workspace, identity, output)
        torch.npu.synchronize()
        expected = source + (1 if preload_inputs else epochs) * 0.5
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(workspace.cpu(), expected, rtol=0, atol=0)
