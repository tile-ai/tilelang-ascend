# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Execute normalized manual event protocols with Bisheng automatic sync off."""

from typing import Any

import pytest
import torch

import tilelang
from tilelang import language as T
from tilelang import tvm

from test_ascend_pipeline_ring_provenance import _nodes, _pre_sync
from test_ascend_upstream_sync_device import _PASS_CONFIGS, _compile_device_kernel


pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="requires Ascend NPU execution",
)

_WIDTH = 64
_ROWS = 5
_VECTOR_CORES = 2
_SENTINEL = -987654.0
_MANUAL_FLAGS = {"tl.ascend_set_flag", "tl.ascend_wait_flag"}
_AUTO_SET = "tl.ascend_auto_set_flag"
_AUTO_WAIT = "tl.ascend_auto_wait_flag"


@pytest.fixture(scope="module", autouse=True)
def _requires_proof_service() -> None:
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")


def _two_stage_prefetch_program(trip_count: int) -> tvm.tir.PrimFunc:
    # Keep the init/prefetch/cur+nxt/drain protocol from the original
    # examples/elementwise/elementwise_add_pipeline.py. Abs makes every
    # output independently checkable without changing the event schedule.
    @T.prim_func
    def main(
        A: T.Tensor((_VECTOR_CORES, trip_count, _WIDTH), "float32"),
        O: T.Tensor((_VECTOR_CORES, trip_count, _WIDTH), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (_cid, vid):
            source = T.alloc_ub((2, _WIDTH), "float32")
            result = T.alloc_ub((2, _WIDTH), "float32")
            with T.Scope("V"):
                T.set_flag("mte3", "mte2", 0)
                T.set_flag("mte3", "mte2", 1)
                T.wait_flag("mte3", "mte2", 0)
                T.copy(A[vid, 0, :], source[0, :])
                T.set_flag("mte2", "v", 0)

                for tile in T.serial(trip_count):
                    cur = tile % 2
                    nxt = (tile + 1) % 2
                    if tile < trip_count - 1:
                        T.wait_flag("mte3", "mte2", nxt)
                        T.copy(A[vid, tile + 1, :], source[nxt, :])
                        T.set_flag("mte2", "v", nxt)

                    T.wait_flag("mte2", "v", cur)
                    T.tile.abs(result[cur, :], source[cur, :])
                    T.set_flag("v", "mte3", cur)
                    T.wait_flag("v", "mte3", cur)
                    T.copy(result[cur, :], O[vid, tile, :])
                    T.set_flag("mte3", "mte2", cur)

                T.wait_flag("mte3", "mte2", 0)
                T.wait_flag("mte3", "mte2", 1)

    return main.with_attr("tir.noalias", True)


def _runtime_serial_program() -> tvm.tir.PrimFunc:
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
                    T.set_flag("mte3", "mte2", 3)
                    for tile in T.serial(n):
                        T.wait_flag("mte3", "mte2", 3)
                        T.copy(A[vid, tile, :], value)
                        T.set_flag("mte2", "v", 4)
                        T.wait_flag("mte2", "v", 4)
                        T.tile.abs(value, value)
                        T.set_flag("v", "mte3", 5)
                        T.wait_flag("v", "mte3", 5)
                        T.copy(value, O[vid, tile, :])
                        T.set_flag("mte3", "mte2", 3)
                    T.wait_flag("mte3", "mte2", 3)

    # Count supplies only a runtime shape. A/O/Count are separate allocations.
    return main.with_attr("tir.noalias", True)


def _assert_manual_events_replaced(program: tvm.tir.PrimFunc, *, runtime: bool = False) -> None:
    before, _ = _pre_sync(program, memory_planning=False)
    before_calls = _nodes(before.body, tvm.tir.Call)
    assert any(getattr(call.op, "name", "") in _MANUAL_FLAGS for call in before_calls)
    if runtime:
        loops = _nodes(before.body, tvm.tir.For)
        assert len(loops) == 1 and isinstance(loops[0].extent, tvm.tir.Var)

    target = tvm.target.Target({"kind": "llvm", "model": "ascendc"})
    with tvm.transform.PassContext(config=_PASS_CONFIGS):
        after = tilelang.transform.AscendInsertSync(target, "A3")(
            tvm.IRModule({"main": before}),
        )["main"]
    calls = _nodes(after.body, tvm.tir.Call)
    assert not any(getattr(call.op, "name", "") in _MANUAL_FLAGS for call in calls), (
        "the compiled schedule must use the upstream engine's replacement handoffs"
    )
    for operation in (_AUTO_SET, _AUTO_WAIT):
        events = {call.args[0].value for call in calls if getattr(call.op, "name", "") == operation}
        assert {"MTE2_V", "V_MTE3"}.issubset(
            events,
        ), "DMA/Vector data dependencies must survive manual protocol normalization"
    if runtime:
        loops = _nodes(after.body, tvm.tir.For)
        assert len(loops) == 1 and isinstance(
            loops[0].extent,
            tvm.tir.Var,
        ), "one kernel must retain the runtime Count.shape loop"


@pytest.mark.parametrize("trip_count", [2, 3, 5])
def test_two_stage_manual_prefetch_reuses_slots_and_drains_repeated_calls(
    trip_count: int,
) -> None:
    program = _two_stage_prefetch_program(trip_count)
    _assert_manual_events_replaced(program)
    kernel = _compile_device_kernel(program)
    shape = (_VECTOR_CORES, trip_count, _WIDTH)
    base = torch.arange(_VECTOR_CORES * trip_count * _WIDTH, dtype=torch.float32).reshape(shape)
    output = torch.full(shape, _SENTINEL, dtype=torch.float32, device="npu")
    for invocation in range(3):
        source = base - (invocation + 1) * 137
        output.fill_(_SENTINEL)
        kernel(source.npu(), output)
        torch.npu.synchronize()
        torch.testing.assert_close(output.cpu(), source.abs(), rtol=0, atol=0)


@pytest.fixture(scope="module")
def runtime_serial_kernel() -> Any:
    program = _runtime_serial_program()
    _assert_manual_events_replaced(program, runtime=True)
    return _compile_device_kernel(program)


@pytest.mark.parametrize("trip_count", [0, 1, 3, 5])
def test_runtime_manual_serial_loop_keeps_untouched_rows_and_drains_repeated_calls(
    runtime_serial_kernel: Any,
    trip_count: int,
) -> None:
    shape = (_VECTOR_CORES, _ROWS, _WIDTH)
    base = torch.arange(_VECTOR_CORES * _ROWS * _WIDTH, dtype=torch.float32).reshape(shape)
    count = torch.empty((trip_count,), dtype=torch.int32, device="npu")
    output = torch.full(shape, _SENTINEL, dtype=torch.float32, device="npu")
    for invocation in range(3):
        source = base - (invocation + 1) * 137
        output.fill_(_SENTINEL)
        expected = torch.full(shape, _SENTINEL, dtype=torch.float32)
        expected[:, :trip_count, :] = source[:, :trip_count, :].abs()
        runtime_serial_kernel(source.npu(), output, count)
        torch.npu.synchronize()
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
