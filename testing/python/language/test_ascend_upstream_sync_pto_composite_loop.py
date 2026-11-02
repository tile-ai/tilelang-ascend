# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""PTO external K-loop admission and asynchronous completion regressions."""

from itertools import permutations

import pytest

from tilelang import language as T
from tilelang import tvm

from test_ascend_pipeline_ring_provenance import _nodes, _pre_sync
from test_ascend_upstream_sync_composite import _events, _gemms, _run
from test_ascend_upstream_sync_pto_composite import _assert_hidden_ids_are_reserved
from test_ascend_upstream_sync_pto_composite_protocol import _execute, _program

tir = tvm.tir


def _pto_k_loop(m=16, n=16, k_l1=64, trips=3, k_l0=128):
    # Same explicit Cube staging as example_gemm_pto_developer.py. Parameters
    # remain dimension-derived; the original large tile is one tested point.
    @T.prim_func
    def main(
        A: T.Tensor((m, trips * k_l1), "float16"),
        B: T.Tensor((trips * k_l1, n), "float16"),
        C: T.Tensor((m, n), "float16"),
    ):
        with T.Kernel(1, is_npu=True) as (_cid, _vid):
            a = T.alloc_L1((m, k_l1), "float16")
            b = T.alloc_L1((k_l1, n), "float16")
            c = T.alloc_L0C((m, n), "float32")
            with T.Scope("C"):
                for k in T.serial(trips):
                    T.copy(A[0, k * k_l1], a)
                    T.copy(B[k * k_l1, 0], b)
                    T.gemm_v0(a, b, c, init=(k == 0), kL0Size=k_l0)
                T.copy(c, C)

    return main


@pytest.fixture(scope="module", autouse=True)
def _requires_proof_service():
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")


@pytest.mark.parametrize("geometry", [(16, 16, 64, 3, 128), (128, 256, 64, 16, 128)])
def test_pto_k_loop_preserves_geometry_clear_and_calculation(geometry):
    original = _pre_sync(_pto_k_loop(*geometry))[0]
    after = _run(original, model="pto", target_kind="llvm")
    assert len(_nodes(original.body, tir.For)) == len(_nodes(after.body, tir.For)) == 1
    assert len(_gemms(original)) == len(_gemms(after)) == 1
    tvm.ir.assert_structural_equal(_gemms(original)[0], _gemms(after)[0])
    assert int(_gemms(after)[0].args[3].args[4]) == 3
    _assert_hidden_ids_are_reserved(after)
    tvm.ir.assert_structural_equal(after, _run(after, model="pto", target_kind="llvm"))


@pytest.mark.parametrize("trips", [2, 3])
def test_pto_tail_chunk_external_accumulation_on_all_pipeline_orderings(trips):
    original = _pre_sync(_pto_k_loop(trips=trips))[0]
    result = _run(original, model="pto", target_kind="llvm")
    queues = _program(result, 64, outer_trip_count=trips)
    for priority in permutations(range(4)):
        _execute(queues, priority)


def test_pto_k_loop_does_not_borrow_the_helper_event_pool():
    original = _pre_sync(_pto_k_loop())[0]
    result = _run(original, model="pto", target_kind="llvm")
    _assert_hidden_ids_are_reserved(result)
    assert all(event_id != 1 for _, _, event_id in _events(result))
