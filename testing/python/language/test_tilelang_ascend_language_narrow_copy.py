# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

"""
GM -> UB narrow-row copy + broadcast correctness suite.

Feature under test
------------------
``T.copy(src_gm, dst_ub)`` for narrow (sub-32B inner dim) tensors followed by
V-pipe ops (broadcast, add).  The MTE2/MTE3 DMA engines require 32B-granular
bursts; a 2D ``DataCopyPad`` with sub-32B blockLen writes a corrupted layout
(elements dropped / shifted).  The fix (``common.h``) flattens contiguous
narrow rows to a single 1D burst; strided narrow rows (e.g. column slices) are
rejected at compile time.

Test design
-----------
Groups 1-3 combine multiple V-pipe readers (Broadcast, Add, identity copy-out)
into a single kernel invocation so that one set of GM->UB copy and NPU launch
exercises all three at once, reducing total compile + run time.

Strived narrow copies test only the compile-time rejection path (no NPU launch).
Sync-chain test validates the sync-pass fix (MTE2 write -> MTE3 read -> V read).

NOTE: these cases execute on real NPU hardware (``.npu()``); they cannot run
in a CPU-only environment.
"""

import pytest

import torch
import tilelang
import tilelang.language as T
import tvm


VEC_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

TARGETS = ["ascendc"]


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    tilelang.cache.clear_cache()
    yield


# =============================================================================
# Group 1-3 combined: one kernel tests broadcast + add + identity
# =============================================================================


def combined_kernel(M, N=32, axis=1):
    src_shape = (M, 1) if axis == 1 else (1, N)

    @T.prim_func
    def main(
        A: T.Tensor(src_shape, "float32"),
        B_bcast: T.Tensor((M, N), "float32"),
        B_add: T.Tensor(src_shape, "float32"),
        B_ident: T.Tensor(src_shape, "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            src_ub = T.alloc_ub(src_shape, "float32")
            aux_ub = T.alloc_ub(src_shape, "float32")
            add_ub = T.alloc_ub(src_shape, "float32")
            bcast_ub = T.alloc_ub((M, N), "float32")
            T.tile.fill(aux_ub, 0.0)
            T.copy(A, src_ub)
            T.tile.broadcast(bcast_ub, src_ub, axis=axis)
            T.copy(bcast_ub, B_bcast)
            T.copy(src_ub, B_ident)
            T.tile.add(add_ub, src_ub, aux_ub)
            T.copy(add_ub, B_add)

    return main


@pytest.mark.parametrize(
    "M,axis",
    [
        (32, 1),
        pytest.param(32, 0, marks=pytest.mark.low_priority),
        pytest.param(1, 1, marks=pytest.mark.low_priority),
    ],
)
def test_narrow_broadcast(M, axis):
    """Narrow copy broadcast — (M,1) GM -> UB -> broadcast -> GM."""
    N = 32
    func = combined_kernel(M, N, axis)
    func = tilelang.compile(func, out_idx=[-3, -2, -1], pass_configs=VEC_PASS_CONFIGS, target="ascendc")
    if axis == 1:
        src = torch.arange(M, dtype=torch.float32).reshape(M, 1).npu()
    else:
        src = torch.arange(N, dtype=torch.float32).reshape(1, N).npu()
    ref_bcast = src.cpu().expand(M, N).contiguous()
    ref_add = src.cpu()
    ref_ident = src.cpu()
    b_bcast, b_add, b_ident = func(src)
    torch.npu.synchronize()
    torch.testing.assert_close(b_bcast.cpu(), ref_bcast, atol=0, rtol=0)
    torch.testing.assert_close(b_add.cpu(), ref_add, atol=0, rtol=0)
    torch.testing.assert_close(b_ident.cpu(), ref_ident, atol=0, rtol=0)


# =============================================================================
# Group 2 — ub->gm narrow round-trip identity (core case M=32)
# =============================================================================


def narrow_identity_kernel(M):
    @T.prim_func
    def main(A: T.Tensor((M, 1), "float32"), B: T.Tensor((M, 1), "float32")):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            src_ub = T.alloc_ub((M, 1), "float32")
            T.copy(A, src_ub)
            T.copy(src_ub, B)

    return main


@pytest.mark.parametrize(
    "M",
    [
        32,
        pytest.param(20, marks=pytest.mark.low_priority),
        pytest.param(1, marks=pytest.mark.low_priority),
    ],
)
def test_narrow_identity(M):
    """Narrow (M,1) GM -> UB -> GM round-trip identity."""
    func = narrow_identity_kernel(M)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target="ascendc")
    src = (torch.arange(M, dtype=torch.float32) + 100).reshape(M, 1).npu()
    dst = func(src)
    torch.npu.synchronize()
    torch.testing.assert_close(dst.cpu(), src.cpu(), atol=0, rtol=0)


# =============================================================================
# Group 3 — Strided narrow copy (compile-time rejection)
# =============================================================================


def strided_gm2ub_kernel(M, W):
    @T.prim_func
    def main(A: T.Tensor((M, W), "float32"), B: T.Tensor((M, 1), "float32")):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            src_ub = T.alloc_ub((M, 1), "float32")
            aux_ub = T.alloc_ub((M, 1), "float32")
            add_ub = T.alloc_ub((M, 1), "float32")
            T.tile.fill(aux_ub, 0.0)
            T.copy(A[:, 0:1], src_ub)
            T.tile.add(add_ub, src_ub, aux_ub)
            T.copy(add_ub, B)

    return main


def strided_ub2gm_kernel(M, W):
    @T.prim_func
    def main(A: T.Tensor((M, 1), "float32"), B: T.Tensor((M, W), "float32")):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            src_ub = T.alloc_ub((M, 1), "float32")
            T.copy(A, src_ub)
            T.copy(src_ub, B[:, 0:1])

    return main


@pytest.mark.parametrize("kernel_fn", [strided_gm2ub_kernel, strided_ub2gm_kernel])
def test_strided_narrow_rejected(kernel_fn):
    """Strided sub-32B copies must be rejected at compile time."""
    M, W = 32, 64
    with pytest.raises(tvm.TVMError, match="Unsupported strided sub-32B"):
        tilelang.compile(kernel_fn(M, W), out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target="ascendc")


# =============================================================================
# Group 4 — Sync pass: MTE2 write -> MTE3 read -> V read
# =============================================================================


def narrow_sync_chain_kernel(M):
    @T.prim_func
    def main(
        A: T.Tensor((M, 1), "float32"),
        B_mte3: T.Tensor((M, 1), "float32"),
        B_v: T.Tensor((M, 1), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            src_ub = T.alloc_ub((M, 1), "float32")
            aux_ub = T.alloc_ub((M, 1), "float32")
            add_ub = T.alloc_ub((M, 1), "float32")
            T.tile.fill(aux_ub, 0.0)
            T.copy(A, src_ub)
            T.copy(src_ub, B_mte3)
            T.tile.add(add_ub, src_ub, aux_ub)
            T.copy(add_ub, B_v)

    return main


@pytest.mark.parametrize(
    "M",
    [
        32,
        pytest.param(16, marks=pytest.mark.low_priority),
    ],
)
def test_narrow_sync_chain(M):
    """MTE2 write -> MTE3 read -> V read: sync-pass fix validation."""
    func = narrow_sync_chain_kernel(M)
    func = tilelang.compile(func, out_idx=[-2, -1], pass_configs=VEC_PASS_CONFIGS, target="ascendc")
    src = (torch.arange(M, dtype=torch.float32) + 100).reshape(M, 1).npu()
    b_mte3, b_v = func(src)
    torch.npu.synchronize()
    torch.testing.assert_close(b_mte3.cpu(), src.cpu(), atol=0, rtol=0)
    torch.testing.assert_close(b_v.cpu(), src.cpu(), atol=0, rtol=0)
