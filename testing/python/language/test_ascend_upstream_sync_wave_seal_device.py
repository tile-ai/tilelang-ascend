# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Positive A3 device acceptance for six explicit scalar wave seals.

One compiled kernel alternates runtime wave counts. Each wave owns the same UB
until all Vector, MTE2 and MTE3 work has returned to the scalar pipeline. Both
TileLang synchronization passes and Bisheng automatic synchronization are off.
"""

from typing import Any
import re

import pytest
import torch

import tilelang
from tilelang import language as T
from tilelang import tvm


pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="requires Ascend NPU execution",
)

_ROWS = 33
_WIDTH = 64
_VECTOR_CORES = 2
_SENTINEL = -987654.0
_RUNTIME_COUNTS = (0, 1, 3, 33, 1, 0, 33, 3)
_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_SYNC_BACKEND: "upstream",
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC_VS: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
}


def _wave_seal_program() -> tvm.tir.PrimFunc:
    # Match optional_load_kernel's runtime Count.shape binding. Its contents
    # never participate in control flow or data movement.
    n = T.symbolic("n")

    @T.prim_func
    def main(
        A: T.Tensor((_VECTOR_CORES, _ROWS, _WIDTH), "float32"),
        O: T.Tensor((_VECTOR_CORES, _ROWS, _WIDTH), "float32"),
        Count: T.Tensor((n,), "int32"),
    ):
        with T.Kernel(1, is_npu=True) as (_cid, vid):
            value = T.alloc_ub((_WIDTH,), "float32")
            with T.Scope("V"):
                if n <= _ROWS:
                    for wave in T.serial(n):
                        # Entry: every participating asynchronous pipeline
                        # acquires this wave's scalar execution context.
                        T.set_flag("s", "v", 0)
                        T.wait_flag("s", "v", 0)
                        T.set_flag("s", "mte2", 0)
                        T.wait_flag("s", "mte2", 0)
                        T.set_flag("s", "mte3", 0)
                        T.wait_flag("s", "mte3", 0)

                        T.copy(A[vid, wave, :], value)
                        T.set_flag("mte2", "v", 1)
                        T.wait_flag("mte2", "v", 1)
                        T.tile.abs(value, value)
                        T.set_flag("v", "mte3", 1)
                        T.wait_flag("v", "mte3", 1)
                        T.copy(value, O[vid, wave, :])

                        # Exit: scalar progress into the next wave waits for
                        # all work that can still touch the shared UB.
                        T.set_flag("v", "s", 0)
                        T.wait_flag("v", "s", 0)
                        T.set_flag("mte2", "s", 0)
                        T.wait_flag("mte2", "s", 0)
                        T.set_flag("mte3", "s", 0)
                        T.wait_flag("mte3", "s", 0)

    return main.with_attr("tir.noalias", True)


def _nodes(statement, node_type):
    result = []
    tvm.tir.stmt_functor.post_order_visit(
        statement,
        lambda node: result.append(node) if isinstance(node, node_type) else None,
    )
    return result


def _assert_wave_seal_structure(program: tvm.tir.PrimFunc) -> None:
    """Inspect the original PrimFunc without lowering or compiling it."""
    loops = _nodes(program.body, tvm.tir.For)
    assert len(loops) == 1 and loops[0].kind == tvm.tir.ForKind.SERIAL
    assert isinstance(loops[0].extent, tvm.tir.Var), "Count.shape must remain runtime-bound"
    allocations = [buffer for block in _nodes(program.body, tvm.tir.Block) for buffer in block.alloc_buffers]
    assert len(allocations) == 1 and allocations[0].scope() == "shared.ub"
    assert [int(extent) for extent in allocations[0].shape] == [_WIDTH]
    count = program.buffer_map[program.params[2]]
    assert not any(load.buffer.data.same_as(count.data) for load in _nodes(program.body, tvm.tir.BufferLoad)), (
        "Count supplies its shape, not runtime scalar values"
    )
    calls = _nodes(program.body, tvm.tir.Call)
    names = [getattr(call.op, "name", "") for call in calls]
    assert not any("barrier" in name for name in names)
    assert not any(name.startswith("tl.ascend_auto_") for name in names)
    flags = [
        (
            call.op.name.removeprefix("tl.ascend_"),
            call.args[0].value,
            call.args[1].value,
            int(call.args[2]),
        )
        for call in calls
        if getattr(call.op, "name", "") in ("tl.ascend_set_flag", "tl.ascend_wait_flag")
    ]
    expected = []
    for source, destination, event_id in (
        ("S", "V", 0),
        ("S", "MTE2", 0),
        ("S", "MTE3", 0),
        ("MTE2", "V", 1),
        ("V", "MTE3", 1),
        ("V", "S", 0),
        ("MTE2", "S", 0),
        ("MTE3", "S", 0),
    ):
        expected.extend(
            [
                ("set_flag", source, destination, event_id),
                ("wait_flag", source, destination, event_id),
            ]
        )
    assert flags == expected, "each wave requires exactly its six seals and two data handoffs"


@pytest.fixture(scope="module")
def wave_seal_kernel() -> Any:
    program = _wave_seal_program()
    _assert_wave_seal_structure(program)
    tilelang.disable_cache()
    # Both sync options are false, so backend=upstream requires no Z3 service.
    kernel = tilelang.compile(
        program,
        out_idx=[],
        target="ascendc",
        platform="A3",
        pass_configs=_PASS_CONFIGS,
        compile_flags=["--cce-auto-sync=off"],
    )
    source = kernel.get_kernel_source()
    emitted = re.findall(
        r"AscendC::(SetFlag|WaitFlag)<AscendC::HardEvent::([A-Z0-9_]+)>\((\d+)\)",
        source,
    )
    expected = []
    for event, event_id in (
        ("S_V", 0),
        ("S_MTE2", 0),
        ("S_MTE3", 0),
        ("MTE2_V", 1),
        ("V_MTE3", 1),
        ("V_S", 0),
        ("MTE2_S", 0),
        ("MTE3_S", 0),
    ):
        expected.extend([("SetFlag", event, str(event_id)), ("WaitFlag", event, str(event_id))])
    assert emitted == expected, "lowering must preserve exactly the explicit wave protocol"
    assert "PipeBarrier<PIPE_ALL>" not in source, "a blanket barrier would mask this litmus"
    return kernel


def test_scalar_wave_seals_preserve_optional_work_and_drain_repeated_calls(
    wave_seal_kernel: Any,
) -> None:
    shape = (_VECTOR_CORES, _ROWS, _WIDTH)
    base = torch.arange(_VECTOR_CORES * _ROWS * _WIDTH, dtype=torch.float32).reshape(shape) + 1
    output = torch.full(shape, _SENTINEL, dtype=torch.float32, device="npu")
    for path, trip_count in enumerate(_RUNTIME_COUNTS):
        count = torch.empty((trip_count,), dtype=torch.int32, device="npu")
        for invocation in range(3):
            magnitudes = base + (path * 3 + invocation + 1) * 8191
            signs = torch.where((base + path + invocation) % 2 == 0, -1.0, 1.0)
            source = magnitudes * signs
            count.fill_(-111 - invocation)
            output.fill_(_SENTINEL)
            expected = torch.full(shape, _SENTINEL, dtype=torch.float32)
            expected[:, :trip_count, :] = source[:, :trip_count, :].abs()
            if trip_count:
                assert not torch.equal(expected[0, :trip_count], expected[1, :trip_count])

            wave_seal_kernel(source.npu(), output, count)

            torch.npu.synchronize()
            torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
