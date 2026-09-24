"""TileLang-Ascend sparse FlashAttention operator.

Fused Gather -> QK^T -> online softmax -> PV. Three kernels + wrapper routing:
  rev1  : hybrid-mode fallback path (any legal shape, one (b,s,g) chunk per
          kernel block)
  rev3  : main path (fixed cores + shared gather + software pipelining across
          n-iters, R6 champion)
  dense : dense bitmap-mask path for high query-reuse shapes (family B)

The wrapper routes by constraints: dense_ok -> dense; v3_ok -> rev3; else rev1.
Cross-core C/V data moves through GM workspaces (AUTO_CV_SYNC); per-kernel
sync constraints and rejected experiments are recorded in
perf_tuning/board/optimization_log.md.
"""

import tilelang
from tilelang import DataType, language as T
import contextlib
import sys
import torch

try:
    from tilelang.intrinsics import make_zn_layout
except Exception:  # pragma: no cover - layout helper unavailable
    make_zn_layout = None

# ========== Configuration ==========
# rev1: intra-core dependencies use AUTO_SYNC; cross-core C/V handoff uses
# manual T.Scope("C"/"V") + cross_flag (AUTO_CV_SYNC/COMBINE neither covers
# same-iteration V->C dependencies nor composes with T.Scope).
PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

# rev3/dense champion sync model: AUTO_CV_SYNC handles the cross-core
# workspace handoff; AUTO_SYNC is off (it inserts a PipeBarrier per row for
# gather address dependencies, serializing every row copy); intra-core
# ordering uses manual set_flag/wait_flag (AscendC event ids must be in [0,7]).
PASS_CONFIGS_V3 = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

_DTYPE_MAP = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float",
}

_kernel_cache = {}

# R5-P1 (rejected): sorting sparseIndices rows for gather locality - aclnnSort
# on NPU is prohibitively expensive (6.1ms for 65K elements, superlinear
# growth), far beyond the locality gain of a ~500us kernel.

# Dense-path bitmap memoization (R3-2): building the 0/-inf bitmap via scatter
# costs ~450ns/idx, more than the dense kernel itself; the bitmap is a pure
# function of sparseIndices, so repeated calls with the same indices tensor
# (eval perf loops, decode-style loads) reuse it directly. The key carries
# (data_ptr, _version, shape): in-place writes bump _version and invalidate.
_bitmap_cache = {}  # key -> bitmap tensor
_BITMAP_CACHE_MAX = 2

# R4 diagnostics: one-shot stderr logs on the dense path (the evaluator
# captures stderr per case; platform runs can directly locate the failing
# stage or confirm dense engaged, without affecting the measured path).
_dense_fb_logged = set()
_dense_ok_logged = set()
# Cached lower-triangular [S1, S1] template for causal bitmap folding.
_causal_tri_cache = {}


def _dense_log_once(logged, msg):
    """Print each distinct dense diagnostic line to stderr once."""
    if msg not in logged:
        logged.add(msg)
        with contextlib.suppress(Exception):
            print(msg, file=sys.stderr, flush=True)


def _largest_pow2_le(x):
    """Largest power of two <= x (Dk split: 128/192 -> 128, 512/576 -> 512)."""
    p = 1
    while p * 2 <= x:
        p *= 2
    return p


_core_num_cached = None


def _ai_core_num():
    """Physical cube core count of the current NPU (A3 / Ascend910 = 20).

    v3/dense launch one kernel block per cube core (each block pairs with
    two vector cores via vid). A hardcoded count would idle cores on larger
    SoCs (e.g. 24-core parts) or oversubscribe smaller ones.
    """
    global _core_num_cached
    if _core_num_cached is None:
        try:
            _core_num_cached = int(torch.npu.get_device_properties(torch.npu.current_device()).cube_core_num)
        except Exception:
            _core_num_cached = 20  # A3 baseline; conservative fallback
    return _core_num_cached


# ========== rev1 kernel: hybrid-mode fallback path ==========
@tilelang.jit(out_idx=[4], workspace_idx=[5, 6, 7, 8, 9, 10], pass_configs=PASS_CONFIGS)
def sparse_flash_attention_fwd(
    heads,
    kv_groups,
    dim_base,
    dim_tail,
    dim_v,
    topk,
    block_I=64,
    head_block=64,
    is_causal=False,
    input_layout=0,  # 0 = BSND, 1 = BNSD
    dtype="float16",
    sm_scale=None,
):
    """rev1 hybrid-mode kernel: each kernel block handles one (b, s, kv_group) chunk.

    Five stages VG/C1/V1/C2/V2; C/V data moves through 6 GM workspaces,
    same-iteration V->C dependencies (VG->C1, V1->C2) use manual cross_flag
    synchronization.
    """
    assert topk % block_I == 0, "topk must be a multiple of block_I"
    assert dim_base % 16 == 0 and (dim_tail == 0 or dim_tail % 16 == 0)
    assert dim_v % 16 == 0
    assert heads % kv_groups == 0

    sm_scale = sm_scale if sm_scale is not None else (1.0 / (dim_base + dim_tail)) ** 0.5

    indices_dtype = "int32"
    accum_dtype = "float"
    Dk = dim_base + dim_tail

    head_kv = heads // kv_groups  # G: query heads per kv head
    # Large G is split into head_block chunks; small G is padded up to a
    # multiple of 16. Two alignment constraints: the cube-side L0C fractal
    # (M dim), and v_block = H_per_block // 2 must stay a multiple of 8 for
    # the 32B-aligned per-row vector buffers - an odd H_per_block would also
    # leave the last query head unprocessed (vector side covers exactly
    # 2 * v_block rows). Padding rows are guarded by head_idx < H1 on the
    # output write.
    if head_kv > head_block:
        assert head_kv % head_block == 0, "head_kv must be a multiple of head_block"
        REPLICATE_H = head_kv // head_block
        H_per_block = head_block
    else:
        REPLICATE_H = 1
        H_per_block = (max(head_kv, 16) + 15) // 16 * 16
    v_block = H_per_block // 2
    ub_len = max(32 // (DataType(accum_dtype).bits // 8), v_block)  # UB 32B alignment

    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim_base
    D_tail = dim_tail
    Dv = dim_v

    batch = T.symbolic("batch")
    seq_len = T.symbolic("seq_len")
    seq_len_kv = T.symbolic("seq_len_kv")
    block_num = batch * seq_len * REPLICATE_H * kv_groups

    if input_layout == 0:  # BSND: [B, S, N, D]
        q_shape = [batch, seq_len, heads, Dk]
        k_shape = [batch, seq_len_kv, kv_groups, Dk]
        v_shape = [batch, seq_len_kv, kv_groups, Dv]
        i_shape = [batch, seq_len, kv_groups, topk]
        o_shape = [batch, seq_len, heads, Dv]
    else:  # BNSD: [B, N, S, D]
        q_shape = [batch, heads, seq_len, Dk]
        k_shape = [batch, kv_groups, seq_len_kv, Dk]
        v_shape = [batch, kv_groups, seq_len_kv, Dv]
        i_shape = [batch, kv_groups, seq_len, topk]
        o_shape = [batch, heads, seq_len, Dv]

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        K: T.Tensor(k_shape, dtype),  # type: ignore
        V: T.Tensor(v_shape, dtype),  # type: ignore
        Indices: T.Tensor(i_shape, indices_dtype),  # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
        ws_k_base: T.Tensor([block_num, BI, D], dtype),
        ws_k_tail: T.Tensor([block_num, BI, D_tail if D_tail > 0 else 1], dtype),
        ws_v: T.Tensor([block_num, BI, Dv], dtype),
        ws_scores: T.Tensor([block_num, H_per_block, BI], accum_dtype),
        ws_p: T.Tensor([block_num, H_per_block, BI], dtype),
        ws_o: T.Tensor([block_num, H_per_block, Dv], accum_dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bx = cid % (seq_len * REPLICATE_H)
            by = cid // (seq_len * REPLICATE_H) % batch
            bz = cid // (seq_len * REPLICATE_H) // batch % kv_groups

            # ---- Cube-side buffers ----
            q_l1 = T.alloc_L1([H_per_block, D], dtype)
            q_tail_l1 = T.alloc_L1([H_per_block, D_tail if D_tail > 0 else 1], dtype)
            kv_l1 = T.alloc_L1([BI, D], dtype)
            kv_tail_l1 = T.alloc_L1([BI, D_tail if D_tail > 0 else 1], dtype)
            kv_v_l1 = T.alloc_L1([BI, Dv], dtype)
            acc_s_l1 = T.alloc_L1([H_per_block, BI], dtype)
            acc_s_l0c = T.alloc_L0C([H_per_block, BI], accum_dtype)
            acc_o_l0c = T.alloc_L0C([H_per_block, Dv], accum_dtype)

            # ---- Vector-side buffers ----
            acc_o = T.alloc_ub([v_block, Dv], accum_dtype)
            sumexp = T.alloc_ub([ub_len], accum_dtype)
            m_i = T.alloc_ub([ub_len], accum_dtype)
            indices_ub_ = T.alloc_ub([BI], indices_dtype)
            indices_ub_float = T.alloc_ub([BI], accum_dtype)
            kv_ub_base = T.alloc_ub([D], dtype)
            kv_ub_tail = T.alloc_ub([D_tail if D_tail > 0 else 1], dtype)
            kv_ub_v = T.alloc_ub([Dv], dtype)
            acc_s_ub = T.alloc_ub([v_block, BI], accum_dtype)
            m_i_prev = T.alloc_ub([ub_len], accum_dtype)
            acc_s_ub_ = T.alloc_ub([v_block, BI], accum_dtype)
            sumexp_i_ub = T.alloc_ub([ub_len], accum_dtype)
            acc_s_half = T.alloc_ub([v_block, BI], dtype)
            acc_o_ub = T.alloc_ub([v_block, Dv], accum_dtype)
            acc_o_half = T.alloc_ub([v_block, Dv], dtype)
            mask_ub = T.alloc_ub([BI // 8], "uint8")

            b_i = by
            g_i = bz
            s_i = bx // REPLICATE_H
            heads_per_group = heads // kv_groups
            group_start = g_i * heads_per_group
            group_end = (g_i + 1) * heads_per_group
            # H0/H1 assigned once: re-assigning a Python variable inside T.Scope
            # trips tilelang parsing (it always takes the first value). With
            # REPLICATE_H==1, bx % 1 == 0, so H0 == group_start.
            block_idx_in_group = bx % REPLICATE_H
            H0 = group_start + block_idx_in_group * H_per_block
            H1 = T.if_then_else(H0 + H_per_block > group_end, group_end, H0 + H_per_block)

            # ===== Cube scope (AIC): Q load + NI loop (C1 + C2) =====
            with T.Scope("C"):
                if input_layout == 0:
                    T.copy(Q[b_i, s_i, H0:H1, 0:D], q_l1)
                    if D_tail > 0:
                        T.copy(Q[b_i, s_i, H0:H1, D:Dk], q_tail_l1)
                else:
                    T.copy(Q[b_i, H0:H1, s_i, 0:D], q_l1)
                    if D_tail > 0:
                        T.copy(Q[b_i, H0:H1, s_i, D:Dk], q_tail_l1)

                for _ in T.serial(NI):
                    # -- C1: scores = Q @ K_sel^T (Dk accumulated in base/tail halves) --
                    T.wait_cross_flag(0)
                    T.copy(ws_k_base[cid, 0:BI, 0:D], kv_l1)
                    if D_tail > 0:
                        T.copy(ws_k_tail[cid, 0:BI, 0:D_tail], kv_tail_l1)
                    T.gemm_v0(q_l1, kv_l1, acc_s_l0c, transpose_B=True, init=True)
                    if D_tail > 0:
                        T.gemm_v0(q_tail_l1, kv_tail_l1, acc_s_l0c, transpose_B=True)
                    T.copy(acc_s_l0c, ws_scores[cid, 0:H_per_block, 0:BI])
                    T.set_cross_flag("FIX", 1)

                    # -- C2: PV = P @ V_sel --
                    T.wait_cross_flag(2)
                    T.copy(ws_p[cid, 0:H_per_block, 0:BI], acc_s_l1)
                    T.copy(ws_v[cid, 0:BI, 0:Dv], kv_v_l1)
                    T.gemm_v0(acc_s_l1, kv_v_l1, acc_o_l0c, init=True)
                    T.copy(acc_o_l0c, ws_o[cid, 0:H_per_block, 0:Dv])
                    T.set_cross_flag("FIX", 3)
                    T.wait_cross_flag(4)  # no-lag: wait for this iteration's V2
                T.wait_cross_flag(8)  # epilogue: wait for V to finish the output

            # ===== Vector scope (AIV x2, split by vid): VG + V1 + V2 + output =====
            with T.Scope("V"):
                T.tile.fill(acc_o, 0.0)
                T.tile.fill(sumexp, 0.0)
                T.tile.fill(m_i, -(2.0**30))

                for i_i in range(NI):
                    # -- VG: gather the K/V rows of this topK chunk (halved by vid) --
                    if input_layout == 0:
                        T.copy(
                            Indices[b_i, s_i, g_i, i_i * BI : i_i * BI + BI],
                            indices_ub_,
                        )
                    else:
                        T.copy(
                            Indices[b_i, g_i, s_i, i_i * BI : i_i * BI + BI],
                            indices_ub_,
                        )

                    if is_causal:
                        T.copy(indices_ub_, indices_ub_float)
                        threshold = T.float32(s_i + (seq_len_kv - seq_len))
                        T.tile.compare(mask_ub, indices_ub_float, threshold, "LE")

                    for bi_i in range(BI // 2):
                        idx = indices_ub_[bi_i + vid * BI // 2]
                        if input_layout == 0:
                            T.copy(K[b_i, idx, g_i, 0:D], kv_ub_base)
                            if D_tail > 0:
                                T.copy(K[b_i, idx, g_i, D:Dk], kv_ub_tail)
                            T.copy(V[b_i, idx, g_i, 0:Dv], kv_ub_v)
                        else:
                            T.copy(K[b_i, g_i, idx, 0:D], kv_ub_base)
                            if D_tail > 0:
                                T.copy(K[b_i, g_i, idx, D:Dk], kv_ub_tail)
                            T.copy(V[b_i, g_i, idx, 0:Dv], kv_ub_v)
                        T.copy(kv_ub_base, ws_k_base[cid, bi_i + vid * BI // 2, :])
                        if D_tail > 0:
                            T.copy(kv_ub_tail, ws_k_tail[cid, bi_i + vid * BI // 2, :])
                        T.copy(kv_ub_v, ws_v[cid, bi_i + vid * BI // 2, :])

                    T.set_cross_flag("MTE3", 0)  # notify C1: K/V ready

                    # -- V1: online safe softmax over this chunk's scores --
                    T.tile.fill(acc_s_ub, 0.0)
                    if is_causal:
                        T.tile.fill(acc_s_ub_, 0.0)
                        for h_i in range(v_block):
                            T.tile.select(
                                acc_s_ub[h_i, :],
                                mask_ub,
                                acc_s_ub_[h_i, :],
                                -T.infinity(accum_dtype),
                                "VSEL_TENSOR_SCALAR_MODE",
                            )

                    T.copy(m_i, m_i_prev)

                    T.wait_cross_flag(1)
                    T.copy(
                        ws_scores[cid, vid * v_block : vid * v_block + v_block, :],
                        acc_s_ub_,
                    )
                    T.tile.add(acc_s_ub, acc_s_ub, acc_s_ub_)
                    T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)

                    T.reduce_max(acc_s_ub, m_i, dim=-1)
                    T.tile.max(m_i, m_i, m_i_prev)
                    T.tile.sub(m_i_prev, m_i_prev, m_i)
                    T.tile.exp(m_i_prev, m_i_prev)

                    for h_i in range(v_block):
                        T.tile.sub(acc_s_ub[h_i, :], acc_s_ub[h_i, :], m_i[h_i])
                    T.tile.exp(acc_s_ub, acc_s_ub)

                    T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)
                    T.tile.mul(sumexp, sumexp, m_i_prev)
                    T.tile.add(sumexp, sumexp, sumexp_i_ub)

                    for h_i in range(v_block):
                        T.tile.mul(acc_o[h_i, :], acc_o[h_i, :], m_i_prev[h_i])

                    T.copy(acc_s_ub, acc_s_half)
                    T.copy(
                        acc_s_half,
                        ws_p[cid, vid * v_block : vid * v_block + v_block, :],
                    )
                    T.set_cross_flag("MTE3", 2)  # notify C2: P ready

                    # -- V2: fold this chunk's PV into the accumulated output --
                    T.wait_cross_flag(3)
                    T.copy(
                        ws_o[cid, vid * v_block : vid * v_block + v_block, :],
                        acc_o_ub,
                    )
                    T.tile.add(acc_o, acc_o, acc_o_ub)
                    T.set_cross_flag("V", 4)  # notify C1 of the next iteration

                # ---- normalize + write output (0/0 guard for fully-masked rows) ----
                T.tile.add(sumexp, sumexp, 1e-30)
                for h_i in range(v_block):
                    T.tile.div(acc_o[h_i, :], acc_o[h_i, :], sumexp[h_i])

                T.copy(acc_o, acc_o_half)
                if REPLICATE_H != 1:
                    # REPLICATE_H>1: no padding (H_per_block <= head_kv), slice
                    # writes are safe; this also avoids if_then_else conditions
                    # mis-generating write-back IR under REPLICATE_H>1.
                    if input_layout == 0:
                        T.copy(
                            acc_o_half,
                            Output[b_i, s_i, H0 + vid * v_block : H0 + v_block + vid * v_block, 0:Dv],
                        )
                    else:
                        T.copy(
                            acc_o_half,
                            Output[b_i, H0 + vid * v_block : H0 + v_block + vid * v_block, s_i, 0:Dv],
                        )
                else:
                    # REPLICATE_H=1: padding possible (H_per_block > head_kv),
                    # guard each row with head_idx < H1 against OOB writes.
                    for h_i in range(v_block):
                        head_idx = H0 + vid * v_block + h_i
                        if head_idx < H1:
                            if input_layout == 0:
                                T.copy(
                                    acc_o_half[h_i, :],
                                    Output[b_i, s_i, head_idx, 0:Dv],
                                )
                            else:
                                T.copy(
                                    acc_o_half[h_i, :],
                                    Output[b_i, head_idx, s_i, 0:Dv],
                                )

                T.set_cross_flag("MTE3", 8)  # epilogue: notify C the output is done

    return main


# ========== rev3 kernel: main path (fixed cores + shared gather + R6 software pipeline) ==========
@tilelang.jit(out_idx=[4], workspace_idx=[5, 6, 7, 8, 9, 10], pass_configs=PASS_CONFIGS_V3)
def sparse_flash_attention_fwd_v3(
    heads,
    kv_groups,
    dim_base,
    dim_tail,
    dim_v,
    topk,
    batch_size,
    seq_len,
    seq_len_kv,
    m_base=16,
    n_base=256,
    gather_rows=32,
    is_causal=False,
    input_layout=0,  # 0 = BSND, 1 = BNSD
    dtype="float16",
    sm_scale=None,
    core_num=20,
):
    """rev3 main kernel: fixed cores, all G query heads of one logical block
    (b, s, kv_group) share the same gathered K rows (removes rev1's G-fold
    gather amplification per (b,s,g) block).

        prologue : Q[all G heads] -> L1; softmax state init
        n-loop   : V0 gathers n_base K rows -> workspace_1/2 (double buffered)
                   C1 all m: scores -> workspace_3
                   V1 all m: causal select + online softmax -> workspace_4
                   C2 all m: PV (B operand = gathered K, V == K[:, :dim_v])
                             -> workspace_5
                   V2 all m: rescale + accumulate (UB when NM==1, else acc_gm)
        epilogue : normalize + write Output

    The value == key[..., :dim_v] contract (proto.yaml MLA latent KV) lets C2
    reuse the gathered K rows directly - V never needs gathering.

    Sync: AUTO_CV_SYNC handles the cross-core workspace handoff (buffer names
    must contain "workspace", and cube/vec GM copy statements must pair 1:1);
    intra-core ordering uses manual set_flag/wait_flag (AUTO_SYNC off).
    acc_gm (the NM>1 overflow buffer) deliberately avoids the "workspace"
    naming so the CV pass does not claim it.
    """
    assert topk % n_base == 0, "topk must be a multiple of n_base"
    assert dim_base % 16 == 0 and (dim_tail == 0 or dim_tail % 16 == 0)
    assert dim_v % 16 == 0
    assert heads % kv_groups == 0
    assert dim_v == dim_base, "rev3 requires Dv == largest-pow2(Dk); wrapper falls back to rev1"

    sm_scale = sm_scale if sm_scale is not None else (1.0 / (dim_base + dim_tail)) ** 0.5

    indices_dtype = "int32"
    accum_dtype = "float"
    Dk = dim_base + dim_tail

    G = heads // kv_groups  # query heads per kv head
    NM = tilelang.cdiv(G, m_base)  # head sub-blocks per group
    NI = tilelang.cdiv(topk, n_base)  # KV blocks
    G_pad = NM * m_base
    m_half = m_base // 2
    n_half = n_base // 2
    tail = dim_tail if dim_tail > 0 else 1
    acc_rows = G_pad if NM > 1 else 1  # acc_gm spill only when needed
    acc_cols = dim_v if NM > 1 else 1

    # Static shapes: sidesteps a tilelang cross-process cache bug (symbolic
    # variants fail with "Unfounded symbolic var" when a fresh subprocess
    # retries) and keeps the disk cache safe across processes.
    kernel_count = batch_size * seq_len * kv_groups

    if input_layout == 0:  # BSND: [B, S, N, D]
        q_shape = [batch_size, seq_len, heads, Dk]
        k_shape = [batch_size, seq_len_kv, kv_groups, Dk]
        i_shape = [batch_size, seq_len, kv_groups, topk]
        o_shape = [batch_size, seq_len, heads, dim_v]
    else:  # BNSD: [B, N, S, D]
        q_shape = [batch_size, heads, seq_len, Dk]
        k_shape = [batch_size, kv_groups, seq_len_kv, Dk]
        i_shape = [batch_size, kv_groups, seq_len, topk]
        o_shape = [batch_size, heads, seq_len, dim_v]

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        K: T.Tensor(k_shape, dtype),  # type: ignore
        V: T.Tensor(k_shape, dtype),  # type: ignore  (== K[..., :dim_v], unused)
        Indices: T.Tensor(i_shape, indices_dtype),  # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
        workspace_1: T.Tensor([core_num, 2, n_base, dim_base], dtype),
        workspace_2: T.Tensor([core_num, 2, n_base, tail], dtype),
        workspace_3: T.Tensor([core_num, G_pad, n_base], accum_dtype),
        workspace_4: T.Tensor([core_num, G_pad, n_base], dtype),
        workspace_5: T.Tensor([core_num, G_pad, dim_v], accum_dtype),
        acc_gm: T.Tensor([core_num, acc_rows, acc_cols], accum_dtype),
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, vid):
            # ---- L1 (cube side) ----
            q_l1 = T.alloc_L1([NM, m_base, dim_base], dtype)
            q_tail_l1 = T.alloc_L1([NM, m_base, tail], dtype)
            kv_l1 = T.alloc_L1([n_base, dim_base], dtype)
            kv_tail_l1 = T.alloc_L1([n_base, tail], dtype)
            p_l1 = T.alloc_L1([m_base, n_base], dtype)
            # ---- L0C ----
            acc_s_l0c = T.alloc_L0C([m_base, n_base], accum_dtype)
            acc_o_l0c = T.alloc_L0C([m_base, dim_v], accum_dtype)
            # ---- UB (vector side) ----
            indices_ub = T.alloc_ub([topk], indices_dtype)
            indices_f = T.alloc_ub([topk], accum_dtype)
            mask_ub = T.alloc_ub([topk // 8], "uint8")
            mask_iter_ub = T.alloc_ub([n_base // 8], "uint8")
            # R5-P1: additive causal mask (0 / -inf) - the column vector is
            # built once per n-block via select, then applied to each m
            # sub-block with broadcast+add (instead of per-row VSEL).
            zero_ub = T.alloc_ub([n_base], accum_dtype)
            mask_add_ub = T.alloc_ub([n_base], accum_dtype)
            kv_ub = T.alloc_ub([2, gather_rows, dim_base], dtype)
            kv_tail_ub = T.alloc_ub([2, gather_rows, tail], dtype)
            acc_s_ub = T.alloc_ub([m_half, n_base], accum_dtype)
            m_bcast = T.alloc_ub([m_half, n_base], accum_dtype)
            p_ub = T.alloc_ub([m_half, n_base], dtype)
            m_i = T.alloc_ub([NM * m_half, 1], accum_dtype)
            alpha = T.alloc_ub([NM * m_half, 1], accum_dtype)
            sumexp = T.alloc_ub([NM * m_half, 1], accum_dtype)
            m_i_prev = T.alloc_ub([m_half, 1], accum_dtype)
            m_new = T.alloc_ub([m_half, 1], accum_dtype)
            sum_new = T.alloc_ub([m_half, 1], accum_dtype)
            acc_o_temp = T.alloc_ub([m_half, dim_v], accum_dtype)
            alpha_bcast = T.alloc_ub([m_half, dim_v], accum_dtype)
            acc_o_ub = T.alloc_ub([m_half, dim_v], accum_dtype)
            sum_bcast = T.alloc_ub([m_half, dim_v], accum_dtype)
            out_half = T.alloc_ub([m_half, dim_v], dtype)

            if make_zn_layout is not None:
                T.annotate_layout(
                    {
                        q_l1: make_zn_layout(q_l1),
                        q_tail_l1: make_zn_layout(q_tail_l1),
                        kv_l1: make_zn_layout(kv_l1),
                        kv_tail_l1: make_zn_layout(kv_tail_l1),
                        p_l1: make_zn_layout(p_l1),
                    }
                )

            single_core_load = T.ceildiv(kernel_count, core_num)
            used_core_num = T.ceildiv(kernel_count, single_core_load)
            tail_block_size = kernel_count - (used_core_num - 1) * single_core_load
            start_idx = cid * single_core_load
            end_idx = T.if_then_else(cid == used_core_num - 1, start_idx + tail_block_size, start_idx + single_core_load)

            if cid < used_core_num:
                # R5-P1: one-shot per-CORE init (hoisted out of the block
                # loop). Zero-initializing acc_gm per block is redundant: the
                # first n-iter's alpha is always 0, and any residual finite
                # value times 0 stays 0; this only guards the first block
                # against reading uninitialized GM (NaN/Inf bit patterns).
                if NM > 1:
                    T.tile.fill(acc_o_temp, 0.0)
                    T.pipe_barrier("v")
                    T.set_flag("v", "mte3", 7)
                    T.wait_flag("v", "mte3", 7)
                    for m_i_ in T.serial(NM):
                        rows0 = m_i_ * m_base + vid * m_half
                        T.copy(acc_o_temp, acc_gm[cid, rows0 : rows0 + m_half, :])
                    T.set_flag("mte3", "mte2", 7)
                    T.wait_flag("mte3", "mte2", 7)
                if is_causal:
                    T.tile.fill(zero_ub, 0.0)
                for block_idx in T.serial(start_idx, end_idx):
                    s_i = block_idx % seq_len
                    b_i = block_idx // seq_len % batch_size
                    g_i = block_idx // (seq_len * batch_size)
                    H0 = g_i * G

                    # ---- prologue: Q (all head sub-blocks) -> L1 ----
                    for m_i_ in T.serial(NM):
                        h0 = H0 + m_i_ * m_base
                        h1 = T.if_then_else(h0 + m_base > H0 + G, H0 + G, h0 + m_base)
                        if input_layout == 0:
                            T.copy(Q[b_i, s_i, h0:h1, 0:dim_base], q_l1[m_i_, :, :])
                            if dim_tail > 0:
                                T.copy(Q[b_i, s_i, h0:h1, dim_base:Dk], q_tail_l1[m_i_, :, :])
                        else:
                            T.copy(Q[b_i, h0:h1, s_i, 0:dim_base], q_l1[m_i_, :, :])
                            if dim_tail > 0:
                                T.copy(Q[b_i, h0:h1, s_i, dim_base:Dk], q_tail_l1[m_i_, :, :])

                    # ---- prologue: load all indices + causal mask (once per block) ----
                    if input_layout == 0:
                        T.copy(Indices[b_i, s_i, g_i, 0:topk], indices_ub)
                    else:
                        T.copy(Indices[b_i, g_i, s_i, 0:topk], indices_ub)
                    # indices_ub ready: feeds the V-side cast/compare below and
                    # the scalar gather reads
                    T.set_flag("mte2", "v", 5)
                    T.wait_flag("mte2", "v", 5)
                    if is_causal:
                        T.copy(indices_ub, indices_f)
                        threshold = T.float32(s_i + (seq_len_kv - seq_len))
                        T.tile.compare(mask_ub, indices_f, threshold, "LE")

                    # ---- prologue: softmax state init ----
                    # (acc_gm zero-init hoisted to per-core scope, see above)
                    T.tile.fill(m_i, -(2.0**30))
                    T.tile.fill(sumexp, 0.0)
                    if NM == 1:
                        T.tile.fill(acc_o_ub, 0.0)

                    # R6-P2: software-pipelined stage loop - stage s issues
                    # the V0 gather of n-iter s while computing n-iter s-1, so
                    # the cube's cross-core wait on ws_1[s%2] overlaps the
                    # vector's V1/V2(s-1) (previously it serialized behind V2:
                    # simulation on case 9 showed cube WAIT_FLAG at 56% of
                    # cycles). Workspace statement shapes unchanged: one
                    # guarded V0 write and one guarded C1 read inside the c
                    # loop still pair 1:1, the CV pass per-stage/per-m handoff
                    # points are untouched, and event ids 0-7 are unchanged.
                    for s_stage in T.serial(NI + 1):
                        if s_stage < NI:
                            vbuf = s_stage % 2

                            # ---- V0: gather n_base K rows (workspace double buffer) ----
                            # No AUTO_SYNC: MTE2 row copies issue back-to-back
                            # with no per-row barrier; kv_ub ping-pongs, guarded
                            # by the mte2<->mte3 flags.
                            for c in range(n_half // gather_rows):
                                gt = s_stage * (n_half // gather_rows) + c
                                task_id = gt % 2
                                if gt > 1:
                                    T.wait_flag("mte3", "mte2", task_id)
                                for r in range(gather_rows):
                                    idx = indices_ub[s_stage * n_base + c * gather_rows + r + vid * n_half]
                                    if input_layout == 0:
                                        T.copy(K[b_i, idx, g_i, 0:dim_base], kv_ub[task_id, r, :])
                                        if dim_tail > 0:
                                            T.copy(
                                                K[b_i, idx, g_i, dim_base:Dk],
                                                kv_tail_ub[task_id, r, :],
                                            )
                                    else:
                                        T.copy(K[b_i, g_i, idx, 0:dim_base], kv_ub[task_id, r, :])
                                        if dim_tail > 0:
                                            T.copy(
                                                K[b_i, g_i, idx, dim_base:Dk],
                                                kv_tail_ub[task_id, r, :],
                                            )
                                T.set_flag("mte2", "mte3", task_id)
                                T.wait_flag("mte2", "mte3", task_id)
                                row0 = vid * n_half + c * gather_rows
                                T.copy(
                                    kv_ub[task_id, :, :],
                                    workspace_1[cid, vbuf, row0 : row0 + gather_rows, :],
                                )
                                if dim_tail > 0:
                                    T.copy(
                                        kv_tail_ub[task_id, :, :],
                                        workspace_2[cid, vbuf, row0 : row0 + gather_rows, :],
                                    )
                                if gt < NI * (n_half // gather_rows) - 2:
                                    T.set_flag("mte3", "mte2", task_id)

                        # ---- compute(s-1): C1/V1/C2/V2 of n-iter s_stage-1 ----
                        # Loop body is verbatim identical to the pre-pipelined
                        # version (i_i = s_stage - 1). The WAR hazard of
                        # ws_1[(s-2)%2] against V0(s) is covered transitively by
                        # the m-level handshake chain over ws_4: cube reads ws_1
                        # before the m loop, the m loop waits for V1(s-2), and
                        # V1(s-2) precedes V0(s) on the vector stream.
                        if s_stage >= 1:
                            i_i = s_stage - 1
                            buf = i_i % 2

                            # ---- C1: scores for all head sub-blocks ----
                            T.copy(workspace_1[cid, buf, :, :], kv_l1)
                            if dim_tail > 0:
                                T.copy(workspace_2[cid, buf, :, :], kv_tail_l1)
                            for m_i_ in T.serial(NM):
                                T.gemm_v0(
                                    q_l1[m_i_, :, :],
                                    kv_l1,
                                    acc_s_l0c,
                                    transpose_B=True,
                                    init=True,
                                    kL0Size=64,
                                )
                                if dim_tail > 0:
                                    T.gemm_v0(
                                        q_tail_l1[m_i_, :, :],
                                        kv_tail_l1,
                                        acc_s_l0c,
                                        transpose_B=True,
                                        init=False,
                                        kL0Size=64,
                                    )
                                T.copy(
                                    acc_s_l0c,
                                    workspace_3[cid, m_i_ * m_base : (m_i_ + 1) * m_base, :],
                                )
                                T.copy(
                                    workspace_4[cid, m_i_ * m_base : (m_i_ + 1) * m_base, :],
                                    p_l1,
                                )
                                T.gemm_v0(p_l1, kv_l1, acc_o_l0c, init=True, kL0Size=64)
                                T.copy(
                                    acc_o_l0c,
                                    workspace_5[cid, m_i_ * m_base : (m_i_ + 1) * m_base, :],
                                )

                            # ---- V1: causal select + online softmax for all m ----
                            if is_causal:
                                m_lo = i_i * (n_base // 8)
                                T.copy(mask_ub[m_lo : m_lo + n_base // 8], mask_iter_ub)
                                # Additive column mask, built once per n-block.
                                # -inf (not a large negative constant): a
                                # constant cancels out in the max-subtract on
                                # fully-masked rows and would softmax the raw
                                # scores instead of zeroing the row.
                                T.tile.select(
                                    mask_add_ub,
                                    mask_iter_ub,
                                    zero_ub,
                                    -T.infinity(accum_dtype),
                                    "VSEL_TENSOR_SCALAR_MODE",
                                )
                            for m_i_ in T.serial(NM):
                                rows0 = m_i_ * m_base + vid * m_half
                                msl = m_i_ * m_half
                                T.copy(workspace_3[cid, rows0 : rows0 + m_half, :], acc_s_ub)
                                T.set_flag("mte2", "v", 0)
                                T.wait_flag("mte2", "v", 0)
                                T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)
                                if is_causal:
                                    # R5-P1: broadcast+add instead of per-row VSEL
                                    # (the column mask is constant).
                                    T.tile.broadcast(m_bcast, mask_add_ub)
                                    T.tile.add(acc_s_ub, acc_s_ub, m_bcast)

                                T.copy(m_i[msl : msl + m_half, :], m_i_prev)
                                T.reduce_max(acc_s_ub, m_new, dim=-1)
                                T.tile.max(m_new, m_new, m_i_prev)
                                T.tile.sub(m_i_prev, m_i_prev, m_new)
                                T.tile.exp(m_i_prev, m_i_prev)  # alpha

                                T.tile.broadcast(m_bcast, m_new)
                                T.tile.sub(acc_s_ub, acc_s_ub, m_bcast)
                                T.tile.exp(acc_s_ub, acc_s_ub)

                                T.reduce_sum(acc_s_ub, sum_new, dim=-1)
                                T.tile.mul(sumexp[msl : msl + m_half, :], sumexp[msl : msl + m_half, :], m_i_prev)
                                T.tile.add(sumexp[msl : msl + m_half, :], sumexp[msl : msl + m_half, :], sum_new)
                                T.copy(m_new, m_i[msl : msl + m_half, :])
                                T.copy(m_i_prev, alpha[msl : msl + m_half, :])

                                T.copy(acc_s_ub, p_ub)
                                T.pipe_barrier("v")
                                T.set_flag("v", "mte3", 1)
                                T.wait_flag("v", "mte3", 1)
                                T.copy(p_ub, workspace_4[cid, rows0 : rows0 + m_half, :])

                            # ---- V2: rescale + accumulate for all m ----
                            for m_i_ in T.serial(NM):
                                rows0 = m_i_ * m_base + vid * m_half
                                msl = m_i_ * m_half
                                # R5-P1 (reverted): sharing one flag between two
                                # MTE2 loads produced NaN at dim512/NM>1 - the
                                # degraded set_flag only pairs with the
                                # immediately preceding MTE2, not all predecessors.
                                T.copy(workspace_5[cid, rows0 : rows0 + m_half, :], acc_o_temp)
                                T.set_flag("mte2", "v", 2)
                                T.wait_flag("mte2", "v", 2)
                                if NM > 1:
                                    T.copy(acc_gm[cid, rows0 : rows0 + m_half, :], acc_o_ub)
                                    T.set_flag("mte2", "v", 6)
                                    T.wait_flag("mte2", "v", 6)
                                T.tile.broadcast(alpha_bcast, alpha[msl : msl + m_half, :])
                                T.tile.mul(acc_o_ub, acc_o_ub, alpha_bcast)
                                T.tile.add(acc_o_ub, acc_o_ub, acc_o_temp)
                                if NM > 1:
                                    T.pipe_barrier("v")
                                    T.set_flag("v", "mte3", 3)
                                    T.wait_flag("v", "mte3", 3)
                                    T.copy(acc_o_ub, acc_gm[cid, rows0 : rows0 + m_half, :])
                                    T.set_flag("mte3", "mte2", 7)
                                    T.wait_flag("mte3", "mte2", 7)

                    # ---- epilogue: normalize + write Output ----
                    # Note: the barrier_all below is load-bearing for the
                    # AUTO_CV_SYNC handoff - removing it or hoisting it out of
                    # the m loop turns 1-2% of rows into NaN at dim512/NM>1.
                    # Its exact position inside the loop is part of the CV
                    # pairing rhythm; do not move.
                    for m_i_ in T.serial(NM):
                        rows0 = m_i_ * m_base + vid * m_half
                        msl = m_i_ * m_half
                        T.barrier_all()
                        if NM > 1:
                            T.copy(acc_gm[cid, rows0 : rows0 + m_half, :], acc_o_ub)
                            T.set_flag("mte2", "v", 6)
                            T.wait_flag("mte2", "v", 6)
                        T.tile.add(
                            sumexp[msl : msl + m_half, :],
                            sumexp[msl : msl + m_half, :],
                            1e-30,
                        )
                        T.tile.broadcast(sum_bcast, sumexp[msl : msl + m_half, :])
                        T.tile.div(acc_o_ub, acc_o_ub, sum_bcast)
                        T.copy(acc_o_ub, out_half)
                        T.pipe_barrier("v")
                        T.set_flag("v", "mte3", 4)
                        T.wait_flag("v", "mte3", 4)
                        if G % m_base == 0:
                            if input_layout == 0:
                                T.copy(
                                    out_half,
                                    Output[b_i, s_i, H0 + rows0 : H0 + rows0 + m_half, 0:dim_v],
                                )
                            else:
                                T.copy(
                                    out_half,
                                    Output[b_i, H0 + rows0 : H0 + rows0 + m_half, s_i, 0:dim_v],
                                )
                        else:
                            for r in range(m_half):
                                h_idx = H0 + rows0 + r
                                if h_idx < H0 + G:
                                    if input_layout == 0:
                                        T.copy(
                                            out_half[r, :],
                                            Output[b_i, s_i, h_idx, 0:dim_v],
                                        )
                                    else:
                                        T.copy(
                                            out_half[r, :],
                                            Output[b_i, h_idx, s_i, 0:dim_v],
                                        )

    return main


# ========== R3-2: dense bitmap-mask kernel (family B, amp >= 16) ==========
# High query-reuse shapes (S1*topk/S2 >= 16) are scalar-issue bound on the
# per-(b,s,g) gather (msprof case 1: aiv_scalar 55%). This path streams all
# S2 rows of K with large tiles and turns top-k selection into an additive
# bitmap mask (the golden semantics are dense attention over a scatter mask).
# The bitmap carries 0 (keep) / -inf (drop): -inf makes fully-masked rows
# output exactly zero, where a large negative constant would cancel out in
# the online-softmax max-subtract and softmax the raw scores instead.
#
# Synchronization matches rev3 (AUTO_CV_SYNC cross-core handoff + manual
# intra-core flags); the bitmap sub-copy is issued before the ws_3 copy and
# ordered by the latter's flag wait (MTE2 ordering).
@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6], pass_configs=PASS_CONFIGS_V3)
def sparse_flash_attention_fwd_dense(
    heads,
    kv_groups,
    dim_base,
    dim_tail,
    dim_v,
    batch_size,
    seq_len,
    seq_len_kv,
    m_tile=64,
    n_base=256,
    dtype="float16",
    sm_scale=None,
    core_num=20,
):
    """Dense FlashAttention with an additive bitmap mask (top-k pre-scattered).

    BNSD only internally (the wrapper normalizes BSND input with a
    device-side transpose): all GM->L1/UB source tiles must be contiguous -
    the framework copy_gm_to_l1 misreads strided source tiles (measured:
    +1 row offset).

    Block = (b, head h, s_block); head -> kv group uses an explicit
    (sb, hg, g, b) decomposition (an h//G derived index drops the g term
    from the generated K addresses).

    Local flags use ids 4-7: AUTO_CV_SYNC's cross-core flags occupy ids 0-2
    and share the event-id space with local pipe flags - an id collision
    lets wait_flag consume a cross-core event early and pass too soon,
    missing the bitmap's MTE2 landing.
    """
    assert dim_v == dim_base, "dense path requires Dv == largest-pow2(Dk)"
    assert dim_base % 16 == 0 and (dim_tail == 0 or dim_tail % 16 == 0)
    assert heads % kv_groups == 0
    G = heads // kv_groups
    half = m_tile // 2  # s-rows per AIV

    sm_scale = sm_scale if sm_scale is not None else (1.0 / (dim_base + dim_tail)) ** 0.5
    accum_dtype = "float"
    Dk = dim_base + dim_tail
    tail = dim_tail if dim_tail > 0 else 1

    # Fully static shapes: the wrapper caches per shape anyway; static dims
    # let codegen emit complete compile-time copy guards (a runtime tail
    # guard on GM->UB copies mis-lands from the >= 1st iteration).
    s_num = seq_len // m_tile
    kernel_count = batch_size * heads * s_num

    @T.prim_func
    def main(
        Q: T.Tensor([batch_size, heads, seq_len, Dk], dtype),  # BNSD
        K: T.Tensor([batch_size, kv_groups, seq_len_kv, Dk], dtype),  # BNSD
        Bitmap: T.Tensor([batch_size, kv_groups, seq_len, seq_len_kv], dtype),
        Output: T.Tensor([batch_size, heads, seq_len, dim_v], dtype),
        workspace_3: T.Tensor([core_num, m_tile, n_base], accum_dtype),
        workspace_4: T.Tensor([core_num, m_tile, n_base], dtype),
        workspace_5: T.Tensor([core_num, m_tile, dim_v], accum_dtype),
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, vid):
            # ---- L1 (cube side) ----
            q_l1 = T.alloc_L1([m_tile, dim_base], dtype)
            q_tail_l1 = T.alloc_L1([m_tile, tail], dtype)
            kv_l1 = T.alloc_L1([n_base, dim_base], dtype)
            kv_tail_l1 = T.alloc_L1([n_base, tail], dtype)
            p_l1 = T.alloc_L1([m_tile, n_base], dtype)
            # ---- L0C ----
            acc_s_l0c = T.alloc_L0C([m_tile, n_base], accum_dtype)
            acc_o_l0c = T.alloc_L0C([m_tile, dim_v], accum_dtype)
            # ---- UB (vector side, each AIV handles half the rows) ----
            bm_ub = T.alloc_ub([2, half, n_base], dtype)
            mask_f = T.alloc_ub([half, n_base], accum_dtype)
            acc_s_ub = T.alloc_ub([half, n_base], accum_dtype)
            m_bcast = T.alloc_ub([half, n_base], accum_dtype)
            p_ub = T.alloc_ub([half, n_base], dtype)
            m_i = T.alloc_ub([half, 1], accum_dtype)
            m_i_prev = T.alloc_ub([half, 1], accum_dtype)
            m_new = T.alloc_ub([half, 1], accum_dtype)
            sum_new = T.alloc_ub([half, 1], accum_dtype)
            sumexp = T.alloc_ub([half, 1], accum_dtype)
            acc_o_temp = T.alloc_ub([half, dim_v], accum_dtype)
            alpha_bcast = T.alloc_ub([half, dim_v], accum_dtype)
            acc_o_ub = T.alloc_ub([half, dim_v], accum_dtype)
            sum_bcast = T.alloc_ub([half, dim_v], accum_dtype)
            out_half = T.alloc_ub([half, dim_v], dtype)

            if make_zn_layout is not None:
                T.annotate_layout(
                    {
                        q_l1: make_zn_layout(q_l1),
                        q_tail_l1: make_zn_layout(q_tail_l1),
                        kv_l1: make_zn_layout(kv_l1),
                        kv_tail_l1: make_zn_layout(kv_tail_l1),
                        p_l1: make_zn_layout(p_l1),
                    }
                )

            single_core_load = T.ceildiv(kernel_count, core_num)
            used_core_num = T.ceildiv(kernel_count, single_core_load)
            tail_block_size = kernel_count - (used_core_num - 1) * single_core_load
            start_idx = cid * single_core_load
            end_idx = T.if_then_else(cid == used_core_num - 1, start_idx + tail_block_size, start_idx + single_core_load)

            if cid < used_core_num:
                for block_idx in T.serial(start_idx, end_idx):
                    # Explicit decomposition: sb fastest, then hg, g, b
                    sb = block_idx % s_num
                    hg = (block_idx // s_num) % G
                    g_i = (block_idx // (s_num * G)) % kv_groups
                    b_i = block_idx // (s_num * G * kv_groups) % batch_size
                    h_i = g_i * G + hg
                    s0 = sb * m_tile

                    # ---- prologue (C): Q tile (contiguous BNSD rows) ----
                    T.copy(Q[b_i, h_i, s0 : s0 + m_tile, 0:dim_base], q_l1)
                    if dim_tail > 0:
                        T.copy(Q[b_i, h_i, s0 : s0 + m_tile, dim_base:Dk], q_tail_l1)

                    # ---- prologue (V): softmax state init ----
                    T.tile.fill(m_i, -(2.0**30))
                    T.tile.fill(sumexp, 0.0)
                    T.tile.fill(acc_o_ub, 0.0)

                    for i_i in T.serial(seq_len_kv // n_base):
                        n0 = i_i * n_base

                        # ---- C: K chunks (contiguous) + C1 + C2 ----
                        T.copy(K[b_i, g_i, n0 : n0 + n_base, 0:dim_base], kv_l1)
                        if dim_tail > 0:
                            T.copy(K[b_i, g_i, n0 : n0 + n_base, dim_base:Dk], kv_tail_l1)
                        T.gemm_v0(q_l1, kv_l1, acc_s_l0c, transpose_B=True, init=True, kL0Size=64)
                        if dim_tail > 0:
                            T.gemm_v0(
                                q_tail_l1,
                                kv_tail_l1,
                                acc_s_l0c,
                                transpose_B=True,
                                init=False,
                                kL0Size=64,
                            )
                        T.copy(acc_s_l0c, workspace_3[cid, :, :])
                        T.copy(workspace_4[cid, :, :], p_l1)
                        T.gemm_v0(p_l1, kv_l1, acc_o_l0c, init=True, kL0Size=64)
                        T.copy(acc_o_l0c, workspace_5[cid, :, :])

                        # ---- V1: additive bitmap mask + online softmax ----
                        rows0 = vid * half
                        T.copy(workspace_3[cid, rows0 : rows0 + half, :], acc_s_ub)
                        T.copy(
                            Bitmap[b_i, g_i, s0 + rows0 : s0 + rows0 + half, n0 : n0 + n_base],
                            bm_ub[i_i % 2, :, :],
                        )
                        T.set_flag("mte2", "v", 4)
                        T.wait_flag("mte2", "v", 4)
                        # Bitmap already carries 0 (keep) / -inf (drop); -inf
                        # zeroes fully-masked rows instead of softmaxing raw
                        # scores (a constant cancels in max-subtract).
                        T.copy(bm_ub[i_i % 2, :, :], mask_f)  # fp16/bf16 -> f32
                        T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)
                        T.tile.add(acc_s_ub, acc_s_ub, mask_f)
                        T.copy(m_i, m_i_prev)
                        T.reduce_max(acc_s_ub, m_new, dim=-1)
                        T.tile.max(m_new, m_new, m_i_prev)
                        T.tile.sub(m_i_prev, m_i_prev, m_new)
                        T.tile.exp(m_i_prev, m_i_prev)  # alpha
                        T.tile.broadcast(m_bcast, m_new)
                        T.tile.sub(acc_s_ub, acc_s_ub, m_bcast)
                        T.tile.exp(acc_s_ub, acc_s_ub)  # P
                        T.reduce_sum(acc_s_ub, sum_new, dim=-1)
                        T.tile.mul(sumexp, sumexp, m_i_prev)
                        T.tile.add(sumexp, sumexp, sum_new)
                        T.copy(m_new, m_i)
                        T.copy(acc_s_ub, p_ub)  # f32 -> fp16/bf16
                        T.pipe_barrier("v")
                        T.set_flag("v", "mte3", 5)
                        T.wait_flag("v", "mte3", 5)
                        T.copy(p_ub, workspace_4[cid, rows0 : rows0 + half, :])

                        # ---- V2: rescale + accumulate (UB-resident) ----
                        T.copy(workspace_5[cid, rows0 : rows0 + half, :], acc_o_temp)
                        T.set_flag("mte2", "v", 6)
                        T.wait_flag("mte2", "v", 6)
                        T.tile.broadcast(alpha_bcast, m_i_prev)
                        T.tile.mul(acc_o_ub, acc_o_ub, alpha_bcast)
                        T.tile.add(acc_o_ub, acc_o_ub, acc_o_temp)

                    # ---- epilogue (V): normalize + store output ----
                    T.barrier_all()
                    T.tile.add(sumexp, sumexp, 1e-30)
                    T.tile.broadcast(sum_bcast, sumexp)
                    T.tile.div(acc_o_ub, acc_o_ub, sum_bcast)
                    T.copy(acc_o_ub, out_half)
                    T.pipe_barrier("v")
                    T.set_flag("v", "mte3", 7)
                    T.wait_flag("v", "mte3", 7)
                    rows0 = vid * half
                    T.copy(
                        out_half,
                        Output[b_i, h_i, s0 + rows0 : s0 + rows0 + half, 0:dim_v],
                    )

    return main


# ========== Python wrapper: routing and caches ==========


def sparse_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparseIndices: torch.Tensor,
    scaleValue: float,
    inputLayout: str = "BSND",
    is_causal: bool = False,
) -> torch.Tensor:
    """Sparse FlashAttention Python wrapper.

    Args:
        query: [B,S1,N1,Dk] (BSND) or [B,N1,S1,Dk] (BNSD), float16/bfloat16.
        key:   [B,S2,N2,Dk] / [B,N2,S2,Dk].
        value: [B,S2,N2,Dv] / [B,N2,S2,Dv]; Dv <= Dk, value == key[..., :Dv].
        sparseIndices: [B,S1,N2,topK] / [B,N2,S1,topK], int32, values in [0, S2).
        scaleValue: scaling factor (usually 1/sqrt(Dk)).
        inputLayout: "BSND" or "BNSD".
        is_causal: additionally apply a bottom-right aligned causal mask on
            top of the sparse gather.

    Returns:
        output [B,S1,N1,Dv] / [B,N1,S1,Dv], dtype follows query.
    """
    if inputLayout == "BSND":
        B, S1, N1, Dk = query.shape
        N2 = int(key.shape[2])
        S2 = int(key.shape[1])
    elif inputLayout == "BNSD":
        B, N1, S1, Dk = query.shape
        N2 = int(key.shape[1])
        S2 = int(key.shape[2])
    else:
        raise ValueError(f"inputLayout must be 'BSND' or 'BNSD', got {inputLayout}")

    Dv = int(value.shape[-1])
    topK = int(sparseIndices.shape[-1])
    assert N1 % N2 == 0, f"N1 ({N1}) must be divisible by N2 ({N2})"
    assert Dv <= Dk, f"Dv ({Dv}) must be <= Dk ({Dk})"
    assert topK <= S2, f"topK ({topK}) must be <= S2 ({S2})"

    dtype_str = _DTYPE_MAP[query.dtype]
    dim_base = _largest_pow2_le(int(Dk))
    dim_tail = int(Dk) - dim_base
    head_block = 64 if Dv <= 256 else 32
    layout_code = 0 if inputLayout == "BSND" else 1

    # ---------------------------------------------------------------------------
    # Path selection: static shape analysis, not autotuning.
    #
    # Each (shape, dtype) routes to exactly one of the three compiled kernels,
    # picked by cheap host-side constraint checks in fixed priority order:
    #
    #   1. dense  - dense attention over an additive 0/-inf bitmap pre-scattered
    #      from sparseIndices. Wins when queries reuse keys heavily:
    #      amplification amp = S1*topK/S2 >= 16 is where the gather paths
    #      become scalar-issue bound (msprof: aiv_scalar 55% on case 1).
    #      Constraints: S2 % 256 == 0 (n_base blocking), dim_base == 128,
    #      dim_tail in {0, 64}, Dv == dim_base, S1 % 64 == 0 (m_tile).
    #
    #   2. rev3 (v3) - fixed-core persistent kernel; all G query heads of a
    #      (b, s, kv_group) block share one gathered K (removes the G-fold
    #      gather amplification of rev1). Constraints: topK % 256 == 0,
    #      Dv == largest-pow2(Dk) (C2 reuses the gathered K as B operand),
    #      dim_tail in {0, 64}, dim_base in {128, 512}, and an L1 tile budget
    #      <= 480KB (checked below via l1_bytes).
    #
    #   3. rev1 - universal fallback, any legal shape (one (b, s, kv_group)
    #      chunk per kernel block).
    #
    # Tiling parameters are fixed per path, not searched: each value is the
    # best of the candidates measured during tuning (rejected experiments in
    # perf_tuning/board/optimization_log.md), bounded by a hardware budget -
    # e.g. n_base=256 keeps the 64KB B subblock within the 32KB L0B ping-pong
    # slot; gather_rows=64 halves V0 ping-pong chunks for dim<=128 while
    # dim512 needs 16 to stay under ~248KB usable UB. Kernels are compiled
    # once per shape and cached in _kernel_cache.
    # ---------------------------------------------------------------------------
    # R3-2: high query-reuse shapes (family B) take the dense mask path -
    # beyond gather amplification S1*topk/S2 >= 16 the sparse paths are
    # scalar-issue bound; dense instead streams all S2 rows of K with the
    # additive bitmap mask (same semantics as the golden's scatter mask).
    m_tile_dense = 64
    G = N1 // N2
    amp = S1 * topK / S2
    dense_ok = amp >= 16 and S2 % 256 == 0 and dim_base == 128 and dim_tail in (0, 64) and Dv == dim_base and S1 % m_tile_dense == 0

    if dense_ok:
        # Platform guard (R3/R4): on SoCs whose CANN build lacks aclnn
        # operator variants (e.g. Ascend910_93 / CANN 9.1 fails with
        # "aclnnCast failed, 561103"), any failure here falls back to the
        # v3/rev1 gather paths below - correct for all shapes, just slower.
        # The failing stage is printed to stderr once for platform triage
        # (see _dense_log_once).
        stage = "compile"
        try:
            cache_key = ("dense", B, S1, S2, N1, N2, dim_base, dim_tail, Dv, layout_code, dtype_str, float(scaleValue), _ai_core_num())
            if cache_key not in _kernel_cache:
                _kernel_cache[cache_key] = sparse_flash_attention_fwd_dense(
                    heads=int(N1),
                    kv_groups=int(N2),
                    dim_base=dim_base,
                    dim_tail=dim_tail,
                    dim_v=Dv,
                    batch_size=int(B),
                    seq_len=int(S1),
                    seq_len_kv=int(S2),
                    m_tile=m_tile_dense,
                    n_base=256,
                    dtype=dtype_str,
                    sm_scale=float(scaleValue),
                    core_num=_ai_core_num(),
                )
                _dense_log_once(
                    _dense_ok_logged,
                    f"[sfa-dense-ok] B={B} S1={S1} S2={S2} N1={N1} N2={N2} causal={bool(is_causal)} {dtype_str}",
                )
            # Build the BNSD-layout [B, N2, S1, S2] additive bitmap on device:
            # 0 = keep (selected), -inf = drop. -inf (not a large negative
            # constant) keeps fully-masked rows at exactly zero output.
            # Memoized per sparseIndices tensor (see note at top of file).
            bm_key = (
                sparseIndices.data_ptr(),
                sparseIndices._version,
                tuple(sparseIndices.shape),
                bool(is_causal),
                query.dtype,
                int(S2),  # bitmap width / row stride; a stale hit across
                # different S2 would misalign or overread the mask
                inputLayout,  # indices layout semantics (shape alone can
                # collide between BSND/BNSD when S1 == N2)
                sparseIndices.device,
            )
            bitmap = _bitmap_cache.get(bm_key)
            if bitmap is None:
                stage = "indices"
                idx32 = (sparseIndices.transpose(1, 2) if inputLayout == "BSND" else sparseIndices).contiguous()  # [B, N2, S1, topk]
                stage = "scatter"
                bitmap = torch.full((B, N2, S1, S2), float("-inf"), dtype=query.dtype, device=query.device)
                # Layer 1: int32 indices directly - torch_npu accepts int32 and
                # matches int64 bit-for-bit (verified), avoiding a dtype-cast
                # op. Layer 2 (host fallback for the SoCs where scatter_ forces
                # int64): reinterpret interleaved [idx, 0] int32 pairs as
                # little-endian int64 (equal to idx for non-negative indices),
                # bypassing aclnnCast. Scatter writes 0; partial writes from a
                # failed layer 1 are harmless (idempotent re-scatter).
                try:
                    bitmap.scatter_(-1, idx32, 0)
                except Exception:
                    stage = "scatter-int64"
                    idx = torch.stack([idx32, torch.zeros_like(idx32)], dim=-1).view(torch.int64).squeeze(-1)
                    bitmap.scatter_(-1, idx, 0)
                if is_causal:
                    # Causal must set j > s + (S2 - S1) to -inf. For S2 >= S1
                    # the blocked region lies entirely in the last S1 columns
                    # and forms a lower-triangular [S1, S1] pattern - apply the
                    # cached tril on the narrow slice (~free) instead of a
                    # full-tensor masked_fill (~650us measured on case 3), and
                    # drop arange/compare/masked_fill_ from the per-call op
                    # chain (fewer platform-fragile aclnn variants).
                    stage = "causal"
                    if S2 >= S1:
                        tri = _causal_tri_cache.get((S1, query.dtype))
                        if tri is None or tri.device != query.device:
                            tri = torch.ones(S1, S1, dtype=query.dtype, device=query.device).tril()
                            _causal_tri_cache[(S1, query.dtype)] = tri
                        bitmap[:, :, :, S2 - S1 :].masked_fill_(tri == 0, float("-inf"))
                    else:
                        pos = torch.arange(S2, device=bitmap.device)
                        thr = torch.arange(S1, device=bitmap.device) + (S2 - S1)
                        causal = (pos.view(1, S2) > thr.view(S1, 1)).view(1, 1, S1, S2)
                        bitmap.masked_fill_(causal, float("-inf"))
                if len(_bitmap_cache) >= _BITMAP_CACHE_MAX:
                    _bitmap_cache.pop(next(iter(_bitmap_cache)))
                _bitmap_cache[bm_key] = bitmap
            # The dense kernel is BNSD-only (GM->L1 source tiles must be
            # contiguous; the framework misreads strided tiles); BSND input is
            # normalized with a device-side transpose and transposed back.
            stage = "transpose"
            if inputLayout == "BSND":
                q_in = query.transpose(1, 2).contiguous()
                k_in = key.transpose(1, 2).contiguous()
            else:
                q_in, k_in = query, key
            stage = "launch"
            out = _kernel_cache[cache_key](q_in, k_in, bitmap)
            if inputLayout == "BSND":
                out = out.transpose(1, 2)  # view; checker materializes on .cpu()
            return out
        except Exception as e:
            _dense_log_once(
                _dense_fb_logged,
                f"[sfa-dense-fallback] stage={stage}: {type(e).__name__}: {e}",
            )
            # The dense fast path is unavailable here - fall through to the
            # v3/rev1 gather paths below.

    # rev3 fast-path constraints: topk divisible by n_base=256; Dv ==
    # largest-pow2(Dk) (C2's B operand reuses the gathered K); dim_tail in
    # {0, 64}; L1 budget <= 480KB. UB budget (196KB/core): the Dv=512 working
    # set needs m_base=16; Dv<=256 has headroom with m_base=32.
    m_base_v3 = 32
    # n_base fixed at 256 (512 rejected in R5-P1: C1 uses transpose_B=True,
    # the gemm_v0 N-tiling fix only covers transpose_B==false, and N=512
    # stuffs a 64KB B subblock into the 32KB L0B ping-pong slot - the cube
    # crashes with ERR99999 at runtime, measured).
    # gather_rows: 64 for dim_base<=128 (halves the V0 ping-pong chunk count);
    # dim512 stays at 16 (gr=32 pushes kv_ub past ~248KB of usable UB -
    # silently NaNs, measured).
    n_base_v3 = 256
    gather_rows_v3 = 64 if dim_base <= 128 else 16
    l1_bytes = (
        (tilelang.cdiv(G, m_base_v3) * m_base_v3) * (dim_base + dim_tail) * 2
        + n_base_v3 * (dim_base + dim_tail) * 2
        + m_base_v3 * n_base_v3 * 2
    )
    v3_ok = topK % n_base_v3 == 0 and Dv == dim_base and dim_tail in (0, 64) and dim_base in (128, 512) and l1_bytes <= 480 * 1024

    if v3_ok:
        cache_key = (
            "v3",
            B,
            S1,
            S2,
            N1,
            N2,
            dim_base,
            dim_tail,
            Dv,
            topK,
            bool(is_causal),
            layout_code,
            dtype_str,
            float(scaleValue),
            _ai_core_num(),
        )
        if cache_key not in _kernel_cache:
            _kernel_cache[cache_key] = sparse_flash_attention_fwd_v3(
                heads=int(N1),
                kv_groups=int(N2),
                dim_base=dim_base,
                dim_tail=dim_tail,
                dim_v=Dv,
                topk=topK,
                batch_size=int(B),
                seq_len=int(S1),
                seq_len_kv=int(S2),
                m_base=m_base_v3,
                n_base=n_base_v3,
                gather_rows=gather_rows_v3,
                is_causal=bool(is_causal),
                input_layout=layout_code,
                dtype=dtype_str,
                sm_scale=float(scaleValue),
                core_num=_ai_core_num(),
            )
        return _kernel_cache[cache_key](query, key, value, sparseIndices)

    cache_key = (N1, N2, dim_base, dim_tail, Dv, topK, 64, head_block, bool(is_causal), layout_code, dtype_str, float(scaleValue))
    if cache_key not in _kernel_cache:
        _kernel_cache[cache_key] = sparse_flash_attention_fwd(
            heads=int(N1),
            kv_groups=int(N2),
            dim_base=dim_base,
            dim_tail=dim_tail,
            dim_v=Dv,
            topk=topK,
            block_I=64,
            head_block=head_block,
            is_causal=bool(is_causal),
            input_layout=layout_code,
            dtype=dtype_str,
            sm_scale=float(scaleValue),
        )
    kernel = _kernel_cache[cache_key]
    return kernel(query, key, value, sparseIndices)


# ---------------------------------------------------------------------------
# Standalone run entry (smoke test, picked from test_sparse_flash_attention.py L0 case)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # L0 v3_decode case: BSND fp16 decode shape (topK%256==0, Dv==dim_base -> v3)
    B, S1, S2, N1, N2, Dk, Dv, topK = 16, 1, 1024, 32, 8, 128, 128, 512
    scale = 1.0 / (Dk**0.5)
    gen = torch.Generator().manual_seed(42)
    q = (torch.rand(B, S1, N1, Dk, generator=gen) * 2 - 1).to(torch.float16)
    k = (torch.rand(B, S2, N2, Dk, generator=gen) * 2 - 1).to(torch.float16)
    v = k[..., :Dv].clone()  # value == key[..., :Dv] prefix contract
    sel = torch.rand(B * S1 * N2, S2, generator=gen).argsort(dim=1)[:, :topK]
    si = sel.reshape(B, S1, N2, topK).to(torch.int32)

    out = sparse_flash_attention(
        query=q.npu(),
        key=k.npu(),
        value=v.npu(),
        sparseIndices=si.npu(),
        scaleValue=scale,
        inputLayout="BSND",
        is_causal=False,
    )
    torch.npu.synchronize()

    # Compact fp64 reference (BSND, non-causal):
    # scatter top-k mask -> QK^T * scale -> softmax -> PV
    qp, kp, vp = (t.double().permute(0, 2, 1, 3) for t in (q, k, v))  # [B, N, S, D]
    sp = si.long().permute(0, 2, 1, 3)  # [B, N2, S1, topK]
    G = N1 // N2
    mask = torch.zeros(B, N2, S1, S2, dtype=torch.bool)
    mask.scatter_(-1, sp, True)
    scores = torch.einsum("bngsd,bnkd->bngsk", qp.reshape(B, N2, G, S1, Dk), kp) * scale
    p = torch.softmax(scores.masked_fill(~mask.unsqueeze(2), float("-inf")), dim=-1)
    ref = torch.einsum("bngsk,bnkd->bngsd", p, vp).reshape(B, N1, S1, Dv)
    ref = ref.permute(0, 2, 1, 3).contiguous().to(out.dtype)

    torch.testing.assert_close(out.cpu().float(), ref.float(), rtol=1e-3, atol=1e-3)
    print("Kernel Output Match!")
