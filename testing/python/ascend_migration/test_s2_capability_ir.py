"""Check A3 model budgets against native legacy passes, without device compilation.

The isolated planner fixtures exercise allocation metadata only. Their synthetic
calls are not examples of hardware operations on every tested storage scope.
Legacy over-budget error paths are deliberately not invoked here.
"""

import re

import pytest

import tilelang
from tilelang import language as T
from tilelang import tvm
from tilelang.transform.ascend_schedule_capabilities import get_capabilities


PROFILE = get_capabilities()
SCOPES = [pool.scope for pool in PROFILE.pools]
tir = tvm.tir


def _plan(scope: str, size_bytes: int, overlapping: bool = False):
    buffers = [tir.decl_buffer((size_bytes // 2,), "float16", name=name, scope=scope) for name in ("a", "b")]
    statements = [tir.Evaluate(tir.call_extern("handle", "AscendC::Duplicate", buffer.access_ptr("w"))) for buffer in buffers]
    if overlapping:
        statements.append(
            tir.Evaluate(
                tir.call_extern(
                    "handle",
                    "AscendC::Add",
                    buffers[0].access_ptr("w"),
                    buffers[0].access_ptr("r"),
                    buffers[1].access_ptr("r"),
                )
            )
        )
    body = tir.SeqStmt(statements)
    for buffer in buffers:
        body = tir.Allocate(buffer.data, buffer.dtype, [size_bytes // 2], tir.const(True, "bool"), body)
    function = tir.PrimFunc([], body).with_attr("global_symbol", "main")
    function = function.with_attr("logic_buffer_shapes", {buffer.data: [size_bytes // 2] for buffer in buffers})
    with tvm.transform.PassContext(config={"tl.ascend_memory_planning": True}):
        result = tilelang.transform.AscendMemoryPlanning()(tvm.IRModule({"main": function}))["main"]
    return [(int(result.attrs["address_map"][buffer.data]), int(result.attrs["size_map"][buffer.data])) for buffer in buffers]


@pytest.mark.parametrize("scope", SCOPES)
@pytest.mark.parametrize("margin", [0, 32])
def test_capability_budget_matches_native_accepted_boundary(scope, margin):
    pool = PROFILE.pool(scope)
    size = pool.budget_bytes - margin
    assert PROFILE.validate_allocation(scope, size) == size
    for offset, physical_bytes in _plan(scope, size):
        assert physical_bytes == size
        assert offset >= 0
        assert offset + physical_bytes <= pool.budget_bytes


@pytest.mark.parametrize("scope", SCOPES)
def test_capability_alignment_matches_native_live_allocations(scope):
    # Six logical bytes become one 32-byte physical allocation in each pool.
    allocations = _plan(scope, 6, overlapping=True)
    assert sorted(allocations) == [(0, 32), (32, 32)]
    for offset, physical_bytes in allocations:
        assert PROFILE.validate_allocation(scope, 6, offset) == physical_bytes


def test_capability_budgets_match_ascendc_codegen_buffers():
    @T.prim_func
    def vector(A: T.Tensor((64,), "float32"), O: T.Tensor((64,), "float32")):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            u = T.alloc_ub((64,), "float32")
            with T.Scope("V"):
                T.copy(A, u)
                T.tile.fill(u, 1.0)
                T.copy(u, O)

    source = tilelang.lower(vector, target="ascendc", platform="A3").kernel_source
    observed = {name: int(size) for name, size in re.findall(r"pipe\.InitBuffer\(ascend_(\w+),\s*(\d+)\)", source)}
    scopes = {
        "ub": "shared.ub",
        "l1": "shared.l1",
        "l0a": "wmma.matrix_a",
        "l0b": "wmma.matrix_b",
        "l0c": "wmma.accumulator",
    }
    assert observed == {name: PROFILE.pool(scope).budget_bytes for name, scope in scopes.items()}
