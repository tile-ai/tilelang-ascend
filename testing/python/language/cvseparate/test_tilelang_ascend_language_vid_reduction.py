"""Tests for the AscendVidReduction Pass.

Tests ``src/transform/ascend_vid_reduction.cc`` which handles the C:V=1:2
(``threads=2``) scenario in Developer mode.

Each test compiles a kernel with ``threads=2`` (or ``threads=1`` as
baseline), runs on NPU, and compares against a torch reference.  If the
Pass transforms the IR incorrectly, the kernel produces wrong results
or crashes.

Scenarios covered (mirroring patterns from
``examples/developer_mode/sparse_flash_attn_developer_only_vid_reduce.py``,
``gelu_mul_developer.py``, ``flash_attn_bshd_developer.py``):

1. **UB allocation + UB copy** — UB first dim halved, GM vid offset.
2. **T.Parallel loop (1D)** — tile op size and loop extent halved.
3. **Skip-set + workspace loop offset** — ``T.copy(KV[b_i, idx_ub[bi_i],
   g_i, :D], kv_ub)`` where GM index contains UB BufferLoad, triggering
   ``buffers_skip_vid_reduction`` so ``kv_ub`` shape is preserved, and
   ``T.copy(kv_ub, ws[cid, bi_i, :])`` where kv_ub is in skip set,
   loop-var vid offset (``!ub_was_vid_reduced`` path).
4. **T.tile ops** — ``T.tile.mul/add/exp`` size halved (``IsTileOp``).
5. **T.tile + T.Parallel** — ``for h_i, j in T.Parallel(v_block, D):
   acc_o[h_i, j] = acc_o[h_i, j] * m_i_prev[h_i]``
   (``LoopVarUsedInVidReducedUbFirstDim``).
6. **T.tile + for range** — ``for h_i in range(block_M):
   T.tile.div(acc_o[h_i, :], ...)`` (``LoopVarUsedInVidReducedUbFirstDim``).
7. **BufferRegion** — ``T.copy(ws[cid, 0:v_block, :], acc_s_ub_)`` GM slice
   → UB (``ModifyBufferRegions``).
8. **Baseline threads=1** — Pass is a no-op.
"""

import pytest
import tilelang
import tilelang.language as T
import torch


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    tilelang.cache.clear_cache()
    yield


@pytest.fixture
def setup_random_seed():
    torch.manual_seed(0)
    yield


# ============================================================
# 1) GM -> UB -> GM identity (UB allocation + UB copy)
#
# Mirrors: T.copy(Q[..., :D], q_l1) / T.copy(acc_o_half, Output[...])
# Pass: a_ub first dim halved, GM indices get vid offset.
# ============================================================

def gm_ub_gm_identity(M, N, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, threads=2, is_npu=True) as (cid):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_shared((block_M, block_N), dtype)
            T.copy(A[bx * block_M, by * block_N], a_ub)
            T.copy(a_ub, C[bx * block_M, by * block_N])

    return main


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_vid_reduction_gm_ub_gm_identity(setup_random_seed, target):
    M, N, block_M, block_N = 1024, 128, 128, 128
    func = tilelang.compile(gm_ub_gm_identity(M, N, block_M, block_N), out_idx=[1], pass_configs=pass_configs, target=target)
    a = torch.randn(M, N, dtype=torch.float16).npu()
    torch.npu.synchronize()
    out = func(a)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a, rtol=1e-2, atol=1e-2)


# ============================================================
# 2) 1D T.Parallel elementwise (for i in T.Parallel(v_block):)
#
# Mirrors: for i in T.Parallel(v_block):
#              m_i[i] = T.max(m_i[i], m_i_prev[i])
# Pass: a_ub/c_ub first (only) dim halved, tile op size halved.
# ============================================================

def parallel_elementwise_1d(M, block_M, dtype="float16"):
    m_num = M // block_M

    @T.prim_func
    def main(A: T.Tensor((M,), dtype), C: T.Tensor((M,), dtype)):
        with T.Kernel(m_num, threads=2, is_npu=True) as (cid):
            bx = cid % m_num
            a_ub = T.alloc_shared((block_M,), dtype)
            c_ub = T.alloc_shared((block_M,), dtype)
            T.copy(A[bx * block_M], a_ub)
            for i in T.Parallel(block_M):
                c_ub[i] = a_ub[i] * 2.0 + 1.0
            T.copy(c_ub, C[bx * block_M])

    return main


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_vid_reduction_parallel_elementwise_1d(setup_random_seed, target):
    M, block_M = 1024, 128
    func = tilelang.compile(parallel_elementwise_1d(M, block_M), out_idx=[1], pass_configs=pass_configs, target=target)
    a = torch.randn(M, dtype=torch.float16).npu()
    torch.npu.synchronize()
    out = func(a)
    torch.npu.synchronize()
    ref = a * 2.0 + 1.0
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


# ============================================================
# 3) Skip-set + workspace loop offset (dual buffer: kv_ub + kv_tail_ub)
#
# Mirrors the reference pattern:
#   T.copy(KV[b_i, indices_ub_[bi_i], g_i, :D], kv_ub)       # GM->UB, skip!
#   T.copy(KV[b_i, indices_ub_[bi_i], g_i, D:], kv_tail_ub)  # GM->UB, skip!
#   T.copy(kv_ub, workspace_1[cid, bi_i, :])                  # UB->GM ws (output)
#   T.copy(kv_tail_ub, workspace_2[cid, bi_i, :])             # UB->GM ws (output)
#
# Pass (IndicesContainUbBufferLoad + !ub_was_vid_reduced in
# ascend_vid_reduction.cc):
#   * GM index `idx_ub[bi_i]` contains UB BufferLoad
#     -> kv_ub AND kv_tail_ub added to buffers_skip_vid_reduction,
#        shape NOT halved (skip-set)
#   * idx_ub (normal UB) IS halved: BI -> BI // 2
#   * for bi_i loop halved (bi_i indexes vid-reduced idx_ub)
#   * vid offset injected into ws[cid, bi_i + vid*(BI//2), :] via the
#     !ub_was_vid_reduced loop-var path: since kv_ub is in skip set (not
#     vid-reduced), the GM index gets vid offset through current_loops_
#     + GmDimNeedsVidOffset instead of the standard vid*ub_shape[0] path
#
# Uses 4D GM index (KV[b_i, idx_ub[bi_i], g_i, :D]) matching the
# reference example so the codegen generates a constant mask=1.
# ws1/ws2 are declared as outputs (out_idx) to avoid GM->GM copy.
# ============================================================

def gm_gather_skip_set_dual(B, SKV, G, D_full, D_main, D_tail, BI, dtype="float16"):
    block_num = B * 4

    @T.prim_func
    def main(
        KV: T.Tensor((B, SKV, G, D_full), dtype),
        IDX: T.Tensor((B, 4, G, BI), "int32"),
        ws1: T.Tensor((block_num, BI, D_main), dtype),
        ws2: T.Tensor((block_num, BI, D_tail), dtype),
    ):
        with T.Kernel(block_num, threads=2, is_npu=True) as (cid):
            b_i = cid // 4
            s_i = cid % 4
            g_i = 0

            idx_ub = T.alloc_shared((BI,), "int32")
            kv_ub = T.alloc_shared((D_main,), dtype)
            kv_tail_ub = T.alloc_shared((D_tail,), dtype)

            T.copy(IDX[b_i, s_i, g_i, 0:BI], idx_ub)

            for bi_i in range(BI):
                T.copy(KV[b_i, idx_ub[bi_i], g_i, :D_main], kv_ub)
                T.copy(KV[b_i, idx_ub[bi_i], g_i, D_main:], kv_tail_ub)
                T.copy(kv_ub, ws1[cid, bi_i, :])
                T.copy(kv_tail_ub, ws2[cid, bi_i, :])

    return main


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_vid_reduction_skip_set_dual_buffer(setup_random_seed, target):
    """kv_ub/kv_tail_ub in skip set; gather via 4D GM index; ws loop offset."""
    B, SKV, G, D_full, D_main, D_tail, BI = 1, 256, 1, 128, 64, 64, 64
    block_num = B * 4

    func = tilelang.compile(gm_gather_skip_set_dual(B, SKV, G, D_full, D_main, D_tail, BI), out_idx=[2, 3], pass_configs=pass_configs, target=target)
    kv = torch.randn(B, SKV, G, D_full, dtype=torch.float16).npu()
    idx = torch.arange(BI, dtype=torch.int32).npu().repeat(B * 4 * G).reshape(B, 4, G, BI)
    torch.npu.synchronize()

    ws1, ws2 = func(kv, idx)
    torch.npu.synchronize()

    ref_ws1 = torch.empty_like(ws1)
    ref_ws2 = torch.empty_like(ws2)
    for cid in range(block_num):
        b_i = cid // 4
        s_i = cid % 4
        for bi_i in range(BI):
            kv_idx = idx[b_i, s_i, 0, bi_i].item()
            ref_ws1[cid, bi_i, :] = kv[b_i, kv_idx, 0, :D_main]
            ref_ws2[cid, bi_i, :] = kv[b_i, kv_idx, 0, D_main:]

    torch.testing.assert_close(ws1, ref_ws1, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(ws2, ref_ws2, rtol=1e-2, atol=1e-2)


# ============================================================
# 4) T.tile ops (IsTileOp: binary/unary/scalar/compare/sin-cos)
#
# Mirrors: T.tile.mul(temp_ub, a1_ub, temp_ub) / T.tile.exp(temp_ub, temp_ub)
#          (from gelu_mul_developer.py)
# Pass: tile op last size arg halved (ModifyTileOpSize in ascend_vid_reduction.cc).
# Covers all tile op categories in IsTileOp() as a single chained pipeline
# where each step's output feeds the next, so any incorrect size halving
# propagates to the final result sin(a + sqrt(e)) + cos(a):
#   binary_op: add/sub/max/min/mul/div — chain reduces to temp=1
#   unary_op:  exp/sqrt/relu — temp = sqrt(e)
#   scalar_op: leaky_relu/axpy — temp = a + sqrt(e)
#   sin/cos:   sin(temp) + cos(a) — final output
#   compare:   compare(GT) — Pass processes size, result not in output chain
# ============================================================

def tile_ops_kernel(M, N, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, threads=2, is_npu=True) as (cid):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_shared((block_M, block_N), dtype)
            b_ub = T.alloc_shared((block_M, block_N), dtype)
            temp_ub = T.alloc_shared((block_M, block_N), dtype)

            T.copy(A[bx * block_M, by * block_N], a_ub)

            # binary_op: add/sub/max/min/mul/div — chain: temp = 1
            T.tile.add(temp_ub, a_ub, a_ub)       # temp = 2a
            T.tile.sub(temp_ub, temp_ub, a_ub)    # temp = a
            T.tile.max(temp_ub, temp_ub, a_ub)    # temp = a
            T.tile.min(temp_ub, temp_ub, a_ub)    # temp = a
            T.tile.mul(temp_ub, temp_ub, a_ub)    # temp = a^2
            T.tile.add(temp_ub, temp_ub, 1.0)     # temp = a^2 + 1
            T.tile.div(temp_ub, temp_ub, temp_ub) # temp = 1

            # unary_op: exp/sqrt/relu — temp = sqrt(e)
            T.tile.exp(temp_ub, temp_ub)           # temp = e
            T.tile.sqrt(temp_ub, temp_ub)          # temp = sqrt(e)
            T.tile.relu(temp_ub, temp_ub)          # temp = sqrt(e)

            # scalar_op: leaky_relu/axpy — temp = a + sqrt(e)
            T.tile.leaky_relu(temp_ub, temp_ub, 0.1)  # temp = sqrt(e)
            T.tile.axpy(temp_ub, a_ub, 1.0)            # temp = a + sqrt(e)

            # sin/cos: b = sin(temp) + cos(a) = sin(a + sqrt(e)) + cos(a)
            T.tile.sin(b_ub, temp_ub)               # b = sin(a + sqrt(e))
            T.tile.cos(temp_ub, a_ub)               # temp = cos(a)
            T.tile.add(b_ub, b_ub, temp_ub)         # b = sin(a + sqrt(e)) + cos(a)

            # compare (Pass processes its size; result not in output chain)
            T.tile.compare(temp_ub, a_ub, 0.0, "GT")

            T.copy(b_ub, B[bx * block_M, by * block_N])

    return main


@pytest.mark.parametrize("target", ["ascendc"])
def test_vid_reduction_tile_ops(setup_random_seed, target):
    M, N, block_M, block_N = 256, 128, 64, 128
    func = tilelang.compile(tile_ops_kernel(M, N, block_M, block_N), out_idx=[1], pass_configs=pass_configs, target=target)
    a = torch.randn(M, N, dtype=torch.float16).npu()
    torch.npu.synchronize()
    b = func(a)
    torch.npu.synchronize()
    af = a.float()
    sqrt_e = torch.sqrt(torch.exp(torch.tensor(1.0)))
    ref = (torch.sin(af + sqrt_e) + torch.cos(af)).half()
    torch.testing.assert_close(b, ref, rtol=1e-2, atol=1e-2)


# ============================================================
# 5) T.tile + T.Parallel (LoopVarUsedInVidReducedUbFirstDim)
#
# Mirrors: for h_i, j in T.Parallel(v_block, D):
#              acc_o[h_i, j] = acc_o[h_i, j] * m_i_prev[h_i]
# Pass: UB first dim halved, T.Parallel extent halved.
# ============================================================

def tile_parallel_kernel(M, N, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, threads=2, is_npu=True) as (cid):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_shared((block_M, block_N), dtype)
            scale_ub = T.alloc_shared((block_M,), dtype)
            c_ub = T.alloc_shared((block_M, block_N), dtype)

            T.copy(A[bx * block_M, by * block_N], a_ub)
            T.tile.fill(scale_ub, 2.0)
            for h_i, j in T.Parallel(block_M, block_N):
                c_ub[h_i, j] = a_ub[h_i, j] * scale_ub[h_i]
            T.copy(c_ub, C[bx * block_M, by * block_N])

    return main


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_vid_reduction_tile_parallel(setup_random_seed, target):
    M, N, block_M, block_N = 256, 128, 128, 128
    func = tilelang.compile(tile_parallel_kernel(M, N, block_M, block_N), out_idx=[1], pass_configs=pass_configs, target=target)
    a = torch.randn(M, N, dtype=torch.float16).npu()
    torch.npu.synchronize()
    c = func(a)
    torch.npu.synchronize()
    ref = a * 2.0
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)


# ============================================================
# 6) T.tile + for range (LoopVarUsedInVidReducedUbFirstDim)
#
# Mirrors: for h_i in range(block_M):
#              T.tile.div(acc_o[h_i, :], acc_o[h_i, :], sumexp[h_i])
# Pass: UB first dim halved, for range extent halved.
# ============================================================

def tile_range_kernel(M, N, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, threads=2, is_npu=True) as (cid):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_shared((block_M, block_N), dtype)
            scale_ub = T.alloc_shared((block_M,), dtype)
            c_ub = T.alloc_shared((block_M, block_N), dtype)

            T.copy(A[bx * block_M, by * block_N], a_ub)
            T.tile.fill(scale_ub, 3.0)
            T.copy(a_ub, c_ub)
            for h_i in range(block_M):
                T.tile.mul(c_ub[h_i, :], c_ub[h_i, :], scale_ub[h_i])
            T.copy(c_ub, C[bx * block_M, by * block_N])

    return main


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_vid_reduction_tile_range(setup_random_seed, target):
    M, N, block_M, block_N = 256, 128, 128, 128
    func = tilelang.compile(tile_range_kernel(M, N, block_M, block_N), out_idx=[1], pass_configs=pass_configs, target=target)
    a = torch.randn(M, N, dtype=torch.float16).npu()
    torch.npu.synchronize()
    c = func(a)
    torch.npu.synchronize()
    ref = a * 3.0
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)


# ============================================================
# 7) BufferRegion (ModifyBufferRegions: GM slice -> UB)
#
# Mirrors: T.copy(workspace_3[cid, 0:v_block, :], acc_s_ub_)
# Pass: GM BufferRegion Range.min += vid * ub_new_shape[0],
#       region extent halved.
# ============================================================

def buffer_region_kernel(M, N, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
        ws: T.Tensor((m_num * n_num, block_M, block_N), dtype),
    ):
        with T.Kernel(m_num * n_num, threads=2, is_npu=True) as (cid):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_shared((block_M, block_N), dtype)
            out_ub = T.alloc_shared((block_M, block_N), dtype)

            T.copy(A[bx * block_M, by * block_N], a_ub)
            T.copy(a_ub, ws[cid, 0:block_M, :])
            T.copy(ws[cid, 0:block_M, :], out_ub)
            T.copy(out_ub, C[bx * block_M, by * block_N])

    return main


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_vid_reduction_buffer_region(setup_random_seed, target):
    M, N, block_M, block_N = 256, 128, 128, 128
    func = tilelang.compile(buffer_region_kernel(M, N, block_M, block_N), out_idx=[1], workspace_idx=[2], pass_configs=pass_configs, target=target)
    a = torch.randn(M, N, dtype=torch.float16).npu()
    torch.npu.synchronize()
    c = func(a)
    torch.npu.synchronize()
    torch.testing.assert_close(c, a, rtol=1e-2, atol=1e-2)


# ============================================================
# 8) Baseline: threads=1 (no vid reduction)
#
# With threads=1 the Pass is a no-op.  The kernel should produce
# correct results without any UB halving or vid offset injection.
# ============================================================

def gm_ub_gm_identity_t1(M, N, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, threads=1, is_npu=True) as (cid):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_shared((block_M, block_N), dtype)
            T.copy(A[bx * block_M, by * block_N], a_ub)
            T.copy(a_ub, C[bx * block_M, by * block_N])

    return main


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_no_vid_reduction_threads1_identity(setup_random_seed, target):
    M, N, block_M, block_N = 1024, 128, 128, 128
    func = tilelang.compile(gm_ub_gm_identity_t1(M, N, block_M, block_N), out_idx=[1], pass_configs=pass_configs, target=target)
    a = torch.randn(M, N).half().npu()
    torch.npu.synchronize()
    out = func(a)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a, rtol=1e-2, atol=1e-2)


def parallel_elementwise_1d_t1(M, block_M, dtype="float16"):
    m_num = M // block_M

    @T.prim_func
    def main(A: T.Tensor((M,), dtype), C: T.Tensor((M,), dtype)):
        with T.Kernel(m_num, threads=1, is_npu=True) as (cid):
            bx = cid % m_num
            a_ub = T.alloc_shared((block_M,), dtype)
            c_ub = T.alloc_shared((block_M,), dtype)
            T.copy(A[bx * block_M], a_ub)
            for i in T.Parallel(block_M):
                c_ub[i] = a_ub[i] * 2.0 + 1.0
            T.copy(c_ub, C[bx * block_M])

    return main


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_no_vid_reduction_threads1_elementwise(setup_random_seed, target):
    M, block_M = 1024, 128
    func = tilelang.compile(parallel_elementwise_1d_t1(M, block_M), out_idx=[1], pass_configs=pass_configs, target=target)
    a = torch.randn(M).half().npu()
    torch.npu.synchronize()
    out = func(a)
    torch.npu.synchronize()
    ref = a * 2.0 + 1.0
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
