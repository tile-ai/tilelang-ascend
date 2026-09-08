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

Memory hierarchy recap
----------------------
    GM (HBM, no alignment)
      |
      +-- UB (Unified Buffer, 192 KB, 32-Byte alignment)   <-- this file
      |     feeds the Vector unit
      |
      +-- L1 (512 KB) -> L0A/L0B -> L0C (128 KB)           <-- cube path

Scenarios covered
-----------------
  Group 1 - Contiguous narrow copy + broadcast (core fix):
      (M, 1) f32 GM -> UB -> broadcast -> GM.  M = 8/16/32/64, axis = 0/1.
      Verifies that a V-pipe Broadcast reads the correct row values.

  Group 2 - Contiguous narrow copy + V-pipe add:
      (M, 1) f32 GM -> UB -> Add -> GM.  Broadens the reader check beyond
      Broadcast (the issue was not BRC-specific).

  Group 3 - ub->gm narrow round-trip:
      (M, 1) f32 GM -> UB -> GM identity.  Ensures the mirror fix in
      ``copy_ub_to_gm`` is correct.

  Group 4 - M = 1 boundary (scalar path):
      Single-row narrow copy.  This degenerates to a 1D scalar burst and
      should always work.

  Group 5 - Non-32B-multiple total length:
      M = 20 (80 B, unaligned tail).  Ensures the flattened 1D burst handles
      sub-32B-aligned total lengths inside the 32B-rounded allocation slot.

  Group 6 - Strided narrow copy (compile-time rejection):
      ``T.copy(A[:, 0:1], ub)`` with a wide GM source (pitch > 1).  The
      lowering must emit a ``TVMError`` at compile time for static shapes.

  Group 7 - Sync pass: MTE2 write -> MTE3 read -> V read:
      Ensures the ``AscendSyncInsert`` pass inserts the missing ``MTE2_V``
      sync when an intermediate MTE3 read has hidden the MTE2 writer.

NOTE: these cases execute on real NPU hardware (``.npu()``); they cannot run
in a CPU-only environment.
"""

import pytest

import torch
import tilelang
import tilelang.language as T
import tvm

# Vector-scope config: auto in-core sync + memory planning.
VEC_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

# Only the AscendC backend triggers the narrow-copy bug; the PTO backend uses
# TLOAD (no DataCopyPad) and is unaffected.
TARGETS = ["ascendc"]


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    """Clear tilelang cache before the session."""
    tilelang.cache.clear_cache()
    yield


# =============================================================================
# Helper: build a narrow-copy + broadcast kernel
# =============================================================================


def narrow_broadcast_kernel(M, axis, N=32):
    @T.prim_func
    def main(
        A: T.Tensor((M, 1) if axis == 1 else (1, N), "float32"),
        B: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            src_ub = T.alloc_ub((M, 1) if axis == 1 else (1, N), "float32")
            dst_ub = T.alloc_ub((M, N), "float32")
            T.copy(A, src_ub)
            T.tile.broadcast(dst_ub, src_ub, axis=axis)
            T.copy(dst_ub, B)

    return main


def narrow_add_kernel(M):
    @T.prim_func
    def main(
        A: T.Tensor((M, 1), "float32"),
        B: T.Tensor((M, 1), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            src_ub = T.alloc_ub((M, 1), "float32")
            aux_ub = T.alloc_ub((M, 1), "float32")
            add_ub = T.alloc_ub((M, 1), "float32")
            T.tile.fill(aux_ub, 0.0)
            T.copy(A, src_ub)
            T.tile.add(add_ub, src_ub, aux_ub)
            T.copy(add_ub, B)

    return main


def narrow_identity_kernel(M):
    @T.prim_func
    def main(
        A: T.Tensor((M, 1), "float32"),
        B: T.Tensor((M, 1), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            src_ub = T.alloc_ub((M, 1), "float32")
            T.copy(A, src_ub)
            T.copy(src_ub, B)

    return main


def narrow_sync_chain_kernel(M):
    """Kernel that does MTE2 write -> MTE3 read -> V read on the same buffer.
    Without the sync-pass fix the V read races the MTE2 write."""

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
            T.copy(A, src_ub)  # MTE2 write
            T.copy(src_ub, B_mte3)  # MTE3 read (intermediate)
            T.tile.add(add_ub, src_ub, aux_ub)  # V  read (must sync MTE2)
            T.copy(add_ub, B_v)

    return main


# =============================================================================
# Group 1 — Contiguous narrow copy + broadcast (core fix)
# =============================================================================


@pytest.mark.parametrize("M", [8, 16, 32, 64])
@pytest.mark.parametrize("axis", [0, 1])
def test_narrow_broadcast(M, axis):
    """Contiguous (M,1) GM -> UB -> broadcast -> GM matches torch.expand."""
    N = 32
    func = narrow_broadcast_kernel(M, axis)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target="ascendc")
    if axis == 1:
        src = torch.arange(M, dtype=torch.float32).reshape(M, 1).npu()
        ref = src.expand(M, N).contiguous()
    else:
        src = torch.arange(N, dtype=torch.float32).reshape(1, N).npu()
        ref = src.expand(M, N).contiguous()
    dst = func(src)
    torch.npu.synchronize()
    torch.testing.assert_close(dst.cpu(), ref.cpu(), atol=0, rtol=0)


# =============================================================================
# Group 2 — Contiguous narrow copy + V-pipe add
# =============================================================================


@pytest.mark.parametrize("M", [8, 16, 32, 64])
def test_narrow_add(M):
    """Contiguous (M,1) GM -> UB -> Add -> GM matches input + 0 = identity."""
    func = narrow_add_kernel(M)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target="ascendc")
    src = (torch.arange(M, dtype=torch.float32) + 100).reshape(M, 1).npu()
    dst = func(src)
    torch.npu.synchronize()
    torch.testing.assert_close(dst.cpu(), src.cpu(), atol=0, rtol=0)


# =============================================================================
# Group 3 — ub->gm narrow round-trip identity
# =============================================================================


@pytest.mark.parametrize("M", [8, 16, 32, 64])
def test_narrow_identity(M):
    """Contiguous (M,1) GM -> UB -> GM round-trip identity."""
    func = narrow_identity_kernel(M)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target="ascendc")
    src = (torch.arange(M, dtype=torch.float32) + 100).reshape(M, 1).npu()
    dst = func(src)
    torch.npu.synchronize()
    torch.testing.assert_close(dst.cpu(), src.cpu(), atol=0, rtol=0)


# =============================================================================
# Group 4 — M = 1 boundary (scalar path)
# =============================================================================


@pytest.mark.parametrize("axis", [0, 1])
def test_narrow_broadcast_m1(axis):
    """M=1 narrow copy + broadcast.  Degenerates to scalar Duplicate path."""
    M, N = 1, 32
    func = narrow_broadcast_kernel(M, axis)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target="ascendc")
    if axis == 1:
        src = torch.tensor([[42.0]], dtype=torch.float32).npu()
        ref = src.expand(M, N).contiguous()
    else:
        src = torch.arange(N, dtype=torch.float32).reshape(1, N).npu()
        ref = src.expand(M, N).contiguous()
    dst = func(src)
    torch.npu.synchronize()
    torch.testing.assert_close(dst.cpu(), ref.cpu(), atol=0, rtol=0)


# =============================================================================
# Group 5 — Non-32B-multiple total length
# =============================================================================


@pytest.mark.parametrize("M", [20, 21, 36])
def test_narrow_unaligned_tail(M):
    """Narrow copy whose total bytes are not a multiple of 32.  The flattened
    1D burst must handle the unaligned tail inside the slot padding."""
    N = 32
    axis = 1
    func = narrow_broadcast_kernel(M, axis)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target="ascendc")
    src = torch.arange(M, dtype=torch.float32).reshape(M, 1).npu()
    ref = src.expand(M, N).contiguous()
    dst = func(src)
    torch.npu.synchronize()
    torch.testing.assert_close(dst.cpu(), ref.cpu(), atol=0, rtol=0)


# =============================================================================
# Group 6 — Strided narrow copy (compile-time rejection)
# =============================================================================


def strided_gm2ub_kernel(M, W):
    @T.prim_func
    def main(
        A: T.Tensor((M, W), "float32"),
        B: T.Tensor((M, 1), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            src_ub = T.alloc_ub((M, 1), "float32")
            aux_ub = T.alloc_ub((M, 1), "float32")
            add_ub = T.alloc_ub((M, 1), "float32")
            T.tile.fill(aux_ub, 0.0)
            T.copy(A[:, 0:1], src_ub)  # strided narrow -> rejected
            T.tile.add(add_ub, src_ub, aux_ub)
            T.copy(add_ub, B)

    return main


def strided_ub2gm_kernel(M, W):
    @T.prim_func
    def main(
        A: T.Tensor((M, 1), "float32"),
        B: T.Tensor((M, W), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            src_ub = T.alloc_ub((M, 1), "float32")
            T.copy(A, src_ub)
            T.copy(src_ub, B[:, 0:1])  # strided narrow ub->gm -> rejected

    return main


@pytest.mark.parametrize("kernel_fn", [strided_gm2ub_kernel, strided_ub2gm_kernel])
def test_strided_narrow_rejected(kernel_fn):
    """Strided sub-32B copies must be rejected at compile time."""
    M, W = 32, 64
    func = kernel_fn(M, W)
    with pytest.raises(tvm.TVMError, match="Unsupported strided sub-32B"):
        tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target="ascendc")


# =============================================================================
# Group 7 — Sync pass: MTE2 write -> MTE3 read -> V read
# =============================================================================


@pytest.mark.parametrize("M", [8, 16, 32])
def test_narrow_sync_chain(M):
    """MTE2 write -> MTE3 read -> V read: the V read must get the MTE2 data,
    not stale UB.  Requires the sync-pass fix (current_write_history_)."""
    func = narrow_sync_chain_kernel(M)
    func = tilelang.compile(func, out_idx=[-1, -2], pass_configs=VEC_PASS_CONFIGS, target="ascendc")
    src = (torch.arange(M, dtype=torch.float32) + 100).reshape(M, 1).npu()
    b_mte3, b_v = func(src)
    torch.npu.synchronize()
    # Both MTE3 and V readers must see the correct data.
    torch.testing.assert_close(b_mte3.cpu(), src.cpu(), atol=0, rtol=0)
    torch.testing.assert_close(b_v.cpu(), src.cpu(), atol=0, rtol=0)
