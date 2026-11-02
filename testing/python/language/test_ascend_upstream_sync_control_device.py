# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""NPU regression for a runtime loop containing a real If/else decision.

The module fixture compiles once for all runtime lengths. No manual flags or
barriers are present, and Bisheng automatic synchronization stays disabled.
Host-only lowering checks precede compilation so unrolling or branch removal
cannot silently replace the intended conditional synchronization coverage.
"""

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

_ROWS = 5
_WIDTH = 64
_VECTOR_CORES = 2
_SENTINEL = -99.0


@pytest.fixture(scope="module", autouse=True)
def _requires_proof_service() -> None:
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")


def _if_else_loop_program() -> tvm.tir.PrimFunc:
    # Reuse optional_load_kernel's runtime Count.shape[0] binding and explicit
    # Vector scope; only the branch payload and output rows differ.
    n = T.symbolic("n")

    @T.prim_func
    def main(
        A: T.Tensor((_ROWS, _WIDTH), "float32"),
        O: T.Tensor((_VECTOR_CORES, _ROWS, _WIDTH), "float32"),
        Count: T.Tensor((n,), "int32"),
    ):
        with T.Kernel(1, is_npu=True) as (_cid, vid):
            value = T.alloc_ub((_WIDTH,), "float32")
            with T.Scope("V"):
                # Bound the real fixed-height A/O views without specializing
                # n. In every tested path this guard enters; the inner copy
                # is therefore provably full-mask and uses only audited MTE2.
                if n <= _ROWS:
                    for index in T.serial(n):
                        if index % 2 == 0:
                            T.copy(A[index, :], value)
                        else:
                            T.tile.fill(value, 3)
                        T.copy(value, O[vid, index, :])

    # Every invocation below uses independent A/O/Count allocations. Count
    # provides only a runtime shape; the kernel never reads its data values.
    return main.with_attr("tir.noalias", True)


def _assert_host_control_structure(program: tvm.tir.PrimFunc) -> None:
    before, _ = _pre_sync(program, memory_planning=False)
    branches = [node for node in _nodes(before.body, tvm.tir.IfThenElse) if node.else_case is not None]
    assert len(branches) == 1, "pre-sync IR must retain the real If/else"
    loops = _nodes(before.body, tvm.tir.For)
    assert len(loops) == 1
    assert isinstance(loops[0].extent, tvm.tir.Var), "Count must keep the loop runtime-dependent"

    target = tvm.target.Target({"kind": "llvm", "model": "ascendc"})
    with tvm.transform.PassContext(config=_PASS_CONFIGS):
        after = tilelang.transform.AscendInsertSync(target, "A3")(tvm.IRModule({"main": before}))["main"]
    captures = [
        node
        for node in _nodes(after.body, tvm.tir.LetStmt)
        if node.var.dtype == "bool" and tvm.ir.structural_equal(node.value, branches[0].condition)
    ]
    assert len(captures) == 1, "one pure Let snapshot must capture the original branch decision"
    assert not _nodes(tvm.tir.Evaluate(captures[0].value), tvm.tir.BufferLoad)
    assert len(_nodes(after.body, tvm.tir.For)) == 1
    assert all(node.else_case is None for node in _nodes(after.body, tvm.tir.IfThenElse))
    assert not any(node.name.startswith("__ascend_if_proxy_") for node in _nodes(after.body, tvm.tir.Var))


@pytest.fixture(scope="module")
def if_else_loop_kernel() -> Any:
    program = _if_else_loop_program()
    _assert_host_control_structure(program)
    # This helper disables TileLang's cache and passes --cce-auto-sync=off.
    return _compile_device_kernel(program)


@pytest.mark.parametrize("trip_count", [0, 1, 3, 5])
def test_runtime_if_else_keeps_untouched_rows_and_drains_repeated_calls(if_else_loop_kernel: Any, trip_count: int) -> None:
    output = torch.full((_VECTOR_CORES, _ROWS, _WIDTH), _SENTINEL, dtype=torch.float32, device="npu")
    count = torch.empty((trip_count,), dtype=torch.int32, device="npu")
    base = torch.arange(_ROWS * _WIDTH, dtype=torch.float32).reshape(_ROWS, _WIDTH)
    for invocation in range(2):
        source = base + invocation * 701 - 137
        output.fill_(_SENTINEL)
        expected = torch.full((_VECTOR_CORES, _ROWS, _WIDTH), _SENTINEL, dtype=torch.float32)
        for row in range(trip_count):
            expected[:, row, :] = source[row] if row % 2 == 0 else 3.0

        if_else_loop_kernel(source.npu(), output, count)

        torch.npu.synchronize()
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
