"""WY Fast Backward Split for Ascend NPU.

Implements the backward pass of a WY-style chunked attention mechanism using
@tilelang.jit kernels (bf16 I/O, T.tile.cast for bf16<->fp32) with a pure
PyTorch golden and smoke test.

Mathematical overview — see golden_wy_fast_bwd_split below (two phases:
per-chunk GEMMs + elementwise merges, then refine + gate + combine producing
dA, dk, dv, dbeta, dg, dbeta_k, dg_A_positive, dg_A_negative).

Golden config: B=1, S=32768, H=8, DK=DV=128, chunk_size=64, bf16 inputs.

Outputs (all fp32): dA, dk, dv, dbeta, dg, dbeta_k, dg_A_positive, dg_A_negative.

Architecture (3 kernel definitions, 2 launches; every supported shape takes
one of two fused paths — guard shapes take the 2-slot fast path, the rest
the 1-chunk universal):
  k1b_gemm -> [k1cv_k2cd_fused_2s | k1cv_k2cd_fused]
    K1b    k1b_gemm         (Cube) bf16 GEMMs with T.mma + explicit L0 staging
    Fused2s k1cv_k2cd_fused_2s (CV)  the 2-slot hand-unrolled fast path of the
                           fused body (guard shapes only)
    Fused  k1cv_k2cd_fused  (CV)   k1c + k2a_pre + refine + k2b + k2c + k2d
                             in ONE 1-chunk launch (block = chunk):
                             intermediates that a multi-launch pipeline
                             would round-trip through GM are internalized
                             as in-kernel relays / UB values; K_T is
                             produced by an in-kernel rider; dk crosses
                             C->V per-iteration via an explicit
                             "workspace"-named GM relay (GQA pattern,
                             per-ik disjoint slices)

Programming mode: Developer (alloc_shared/fragment + auto sync) with
pass_configs = {AUTO_CV_COMBINE, AUTO_CV_SYNC, AUTO_SYNC, MEMORY_PLANNING: True}.
"""

import tilelang
import torch
import torch.nn.functional as F
from tilelang import language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

CAST_LOW2HIGH = "CAST_NONE"
CAST_HIGH2LOW = "CAST_RINT"


def compute_masks(G, BS):
    """Host-side mask precomputation. G is [BH, S] fp32.

    The combined gate is never materialized — k2b recomputes it in-kernel
    from G and the sign_flip constant. This function returns only the two
    small constants every consumer needs:
      lower_tri_mask = tril(ones(BS, BS), -1)   (k2a_pre's dA_masked gate)
      sign_flip      = -tril(ones(BS, BS), -1)  (k2b's in-kernel gate sign)
    """
    lower_tri_mask = torch.tril(torch.ones(BS, BS, dtype=torch.float32), diagonal=-1)
    sign_flip = -lower_tri_mask
    return lower_tri_mask, sign_flip


def compute_masks_npu(G, BS):
    """On-device mask precomputation (no gate tensor).

    Same products as the CPU path (see compute_masks): just lower_tri_mask
    and sign_flip, two small [BS, BS] constants. The combined gate is not
    materialized — k2b recomputes it inside the kernel.
    """
    dev = G.device
    ones = torch.ones(BS, BS, dtype=torch.float32, device=dev)
    lower_tri_mask = torch.tril(ones, diagonal=-1)
    sign_flip = -lower_tri_mask
    return lower_tri_mask, sign_flip


def golden_wy_fast_bwd_split(K, V, Beta, G, A, dw, du, chunk_size, block_DK=64, block_DV=64):
    """Pure PyTorch golden. Tensors use [BH, S, D] / [BH, S] shapes."""
    BH, S, DK = K.shape
    _, _, DV = V.shape
    BS = chunk_size
    K_f, V_f = K.float(), V.float()
    Beta_f, G_f = Beta.float(), G.float()
    A_f, dw_f, du_f = A.float(), dw.float(), du.float()
    G_exp = torch.exp(G_f)
    dA_out = torch.zeros(BH, S, BS, dtype=torch.float32, device=K.device)
    dk_out = torch.zeros(BH, S, DK, dtype=torch.float32, device=K.device)
    dv_out = torch.zeros(BH, S, DV, dtype=torch.float32, device=K.device)
    dbeta_inter = torch.zeros(BH, S, dtype=torch.float32, device=K.device)
    dg_inter = torch.zeros(BH, S, dtype=torch.float32, device=K.device)
    dbeta_k_out = torch.zeros(BH, S, dtype=torch.float32, device=K.device)
    dg_A_pos = torch.zeros(BH, S, BS, dtype=torch.float32, device=K.device)
    dg_A_neg = torch.zeros(BH, S, BS, dtype=torch.float32, device=K.device)
    for bh in range(BH):
        for cs in range(0, S, BS):
            ce = min(cs + BS, S)
            A_chunk = A_f[bh, cs:ce, :]
            Beta_chunk = Beta_f[bh, cs:ce]
            G_chunk = G_f[bh, cs:ce]
            G_exp_chunk = G_exp[bh, cs:ce]
            dA_frag = torch.zeros(BS, BS, dtype=torch.float32, device=K.device)
            for ik in range(0, DK, block_DK):
                ke = min(ik + block_DK, DK)
                K_chunk = K_f[bh, cs:ce, ik:ke]
                dw_chunk = dw_f[bh, cs:ce, ik:ke]
                kbg = K_chunk * Beta_chunk.unsqueeze(-1) * G_exp_chunk.unsqueeze(-1)
                dA_frag += dw_chunk @ kbg.T
                dkbg = A_chunk.T @ dw_chunk
                dk_out[bh, cs:ce, ik:ke] = dkbg * Beta_chunk.unsqueeze(-1) * G_exp_chunk.unsqueeze(-1)
                dbeta_inter[bh, cs:ce] += (dkbg * K_chunk * G_exp_chunk.unsqueeze(-1)).sum(-1)
                dg_inter[bh, cs:ce] += (dkbg * K_chunk * G_exp_chunk.unsqueeze(-1) * Beta_chunk.unsqueeze(-1)).sum(-1)
            for iv in range(0, DV, block_DV):
                ve = min(iv + block_DV, DV)
                V_chunk = V_f[bh, cs:ce, iv:ve]
                du_chunk = du_f[bh, cs:ce, iv:ve]
                vb = V_chunk * Beta_chunk.unsqueeze(-1)
                dA_frag += du_chunk @ vb.T
                dvb = A_chunk.T @ du_chunk
                dv_out[bh, cs:ce, iv:ve] = dvb * Beta_chunk.unsqueeze(-1)
                dbeta_inter[bh, cs:ce] += (dvb * V_chunk).sum(-1)
            mask_upper = torch.triu(torch.ones(BS, BS, device=K.device), diagonal=0).bool()
            dA_frag[mask_upper] = 0
            dA_frag = dA_frag @ A_chunk.T
            dA_frag = A_chunk.T @ dA_frag
            G_diff = G_chunk.unsqueeze(-1) - G_chunk.unsqueeze(-2)
            sign_flip = -torch.tril(torch.ones(BS, BS, device=K.device), diagonal=-1)
            exp_gate = torch.where(G_diff <= 0, torch.exp(G_diff), torch.zeros_like(G_diff))
            dA_frag = dA_frag * (sign_flip * exp_gate)
            A_frag = torch.zeros(BS, BS, dtype=torch.float32, device=K.device)
            for ik in range(0, DK, block_DK):
                ke = min(ik + block_DK, DK)
                K_chunk = K_f[bh, cs:ce, ik:ke]
                K_beta = K_chunk * Beta_chunk.unsqueeze(-1)
                A_frag += K_beta @ K_chunk.T
                dk_beta = dA_frag @ K_chunk
                dbeta_k_out[bh, cs:ce] += (dk_beta * K_chunk).sum(-1)
                dk_out[bh, cs:ce, ik:ke] += dA_frag.T @ K_beta
                dk_out[bh, cs:ce, ik:ke] += dk_beta * Beta_chunk.unsqueeze(-1)
            dA_A = dA_frag * A_frag
            dg_A_pos[bh, cs:ce, :] = dA_A
            dg_A_neg[bh, cs:ce, :] = dA_A.T
            dA_out[bh, cs:ce, :] = dA_frag
    dbeta_out = dbeta_inter + dbeta_k_out
    dg_out = dg_inter + dg_A_pos.sum(-1) - dg_A_neg.sum(-1)
    # Outputs stay fp32 (bf16 ULP at large values, e.g. 4.0 for dA ~512, exceeds the precision cap).
    return (dA_out, dk_out, dv_out, dbeta_out, dg_out, dbeta_k_out, dg_A_pos, dg_A_neg)


def get_precision(dtype_str):
    table = {
        "bfloat16": (2**-10, 2**-6, 1e0, 0.99),
        "float16": (2**-14, 2**-9, 1e-1, 0.99),
        "float32": (2**-16, 2**-10, 1e-2, 0.99),
    }
    return table.get(dtype_str, (2**-14, 2**-9, 1e-1, 0.99))


def check_precision(actual, golden, dtype_str):
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype_str)
    a = actual.detach().cpu().float()
    g = golden.detach().cpu().float()
    abs_err = (a - g).abs()
    threshold = atol + rtol * g.abs()
    matched_ratio = (abs_err <= threshold).float().mean().item()
    max_abs_error = abs_err.max().item()
    passed = (matched_ratio >= required_ratio) and (max_abs_error <= max_abs_limit)
    return passed, matched_ratio, max_abs_error


# K1b: Pure Cube — bf16-input GEMMs with fp32 L0C accumulators.
#   GEMM1: dw @ K^T -> dA_k_raw   (column scale beta*exp(G) folded into
#                                 k2a_pre: (dw@K^T)*beta[n]*e^G[n] =
#                                 dw@(K*beta*e^G)^T)
#   GEMM2: A^T @ dw -> dk_beta_g  (transpose_A=True on the original
#                                 layouts; bf16 + transpose_A is safe —
#                                 fp32 + transpose_A hits a codegen NaN bug)
#   GEMM3: du @ V^T -> dA_v_raw   (raw V; beta column scale in k2a_pre —
#                                 the two column scales differ)
#   GEMM4: A^T @ du -> dv_beta    (transpose_A=True, du_l1 already loaded)
#   GEMM5: K @ K^T -> A_frag_raw  (reuses GEMM1's K staging)
# All operands are the user's original bf16 tensors: Ascend Mmul computes
# exact bf16xbf16 products (8+8 mantissa bits < 24) with fp32 accumulation,
# so bf16 feeding is value-identical to a lossless fp32 cast apart from
# L0C accumulation order. The fused kernel's dk GEMMs take fp32 operands
# (mixed-dtype GEMM unsupported) — its K_T rider casts K to fp32 in-kernel.
# T.mma with EXPLICIT L0 staging, three mechanisms:
#   1. Dedicated L0A/L0B buffer names cut auto_sync's name-granularity
#      false WAR deps (implicit staging would make the next GM->L1 prefetch wait on the whole mma).
#   2. Shared L0 staging: A^T staged once per wave (GEMM2/4), K^T once
#      per K-iteration (GEMM1/5).
#   3. Interleaved staging->mma->staging order overlaps the next GEMM's
#      staging with the current mma on the MTE1 pipe.
# L1 operands use alloc_L1 (NOT alloc_shared): a shared-class buffer may
# be reassigned to UB by the memory planner, silently breaking the L1->L0
# staging (no UB->L0A path exists). L0 budgets are asserted host-side in
# _validate_inputs.


@tilelang.jit(pass_configs=pass_configs)
def k1b_gemm(BH, S, DK, DV, BS, block_DK=64, block_DV=64, core_num=20):
    num_chunks = S // BS
    total_tiles = num_chunks * BH
    waves = (total_tiles + core_num - 1) // core_num

    @T.prim_func
    def main(
        dw_bf16: T.Tensor((BH, S, DK), "bfloat16"),
        K_bf16: T.Tensor((BH, S, DK), "bfloat16"),
        A_bf16: T.Tensor((BH, S, BS), "bfloat16"),
        du_bf16: T.Tensor((BH, S, DV), "bfloat16"),
        V_bf16: T.Tensor((BH, S, DV), "bfloat16"),
        dA_k_raw: T.Tensor((BH, S, BS), "float"),
        dA_v_raw: T.Tensor((BH, S, BS), "float"),
        dk_beta_g: T.Tensor((BH, S, DK), "float"),
        dv_beta: T.Tensor((BH, S, DV), "float"),
        A_frag_raw: T.Tensor((BH, S, BS), "float"),
    ):
        with T.Kernel(core_num, threads=1, is_npu=True) as (cid):
            # L1 operands: explicit alloc_L1
            A_l1 = T.alloc_L1((BS, BS), "bfloat16")
            dw_l1 = T.alloc_L1((BS, block_DK), "bfloat16")
            K_l1 = T.alloc_L1((BS, block_DK), "bfloat16")
            du_l1 = T.alloc_L1((BS, block_DV), "bfloat16")
            V_l1 = T.alloc_L1((BS, block_DV), "bfloat16")
            # L0 staging buffers (bf16): A_T_l0a shared GEMM2/GEMM4,
            # K_l0b_T shared GEMM1/GEMM5
            A_T_l0a = T.alloc_L0A((BS, BS), "bfloat16")
            dw_l0a = T.alloc_L0A((BS, block_DK), "bfloat16")
            K_l0a = T.alloc_L0A((BS, block_DK), "bfloat16")
            du_l0a = T.alloc_L0A((BS, block_DV), "bfloat16")
            K_l0b_T = T.alloc_L0B((block_DK, BS), "bfloat16")
            dw_l0b = T.alloc_L0B((BS, block_DK), "bfloat16")
            V_l0b_T = T.alloc_L0B((block_DV, BS), "bfloat16")
            du_l0b = T.alloc_L0B((BS, block_DV), "bfloat16")
            # L0C accumulators (fp32)
            dA_k_frag = T.alloc_L0C((BS, BS), "float")
            dA_v_frag = T.alloc_L0C((BS, BS), "float")
            dk_beta_g_frag = T.alloc_L0C((BS, block_DK), "float")
            dv_beta_frag = T.alloc_L0C((BS, block_DV), "float")
            # A_frag accumulator (K@K^T)
            A_frag_acc = T.alloc_L0C((BS, BS), "float")
            # wave loop (num_stages=1 — memory-throughput-bound, prefetch depth irrelevant)
            for wave in T.Pipelined(waves, num_stages=1):
                pid = wave * core_num + cid
                if pid < total_tiles:
                    chunk_idx = pid // BH
                    bh = pid % BH
                    cs = chunk_idx * BS
                    T.copy(A_bf16[bh, cs : cs + BS, :], A_l1)
                    # A^T staged to L0A ONCE per wave — shared by GEMM2/GEMM4
                    T.copy(A_l1, A_T_l0a, transpose=True)
                    for ik in T.serial(T.ceildiv(DK, block_DK)):
                        ik_st = ik * block_DK
                        T.copy(K_bf16[bh, cs : cs + BS, ik_st : ik_st + block_DK], K_l1)
                        T.copy(dw_bf16[bh, cs : cs + BS, ik_st : ik_st + block_DK], dw_l1)
                        # GEMM1: dw @ K^T — stage both operands, then mma
                        T.copy(dw_l1, dw_l0a)
                        T.copy(K_l1, K_l0b_T, transpose=True)
                        T.mma(dw_l0a, K_l0b_T, dA_k_frag, init=(ik == 0))
                        # GEMM2: A^T @ dw — only B needs staging
                        # (A_T_l0a still live)
                        T.copy(dw_l1, dw_l0b)
                        T.mma(A_T_l0a, dw_l0b, dk_beta_g_frag, init=True)
                        # GEMM5 K@K^T reuses K_l0b_T (staged for GEMM1)
                        # — only A needs staging
                        T.copy(K_l1, K_l0a)
                        T.mma(K_l0a, K_l0b_T, A_frag_acc, init=(ik == 0))
                        T.copy(dk_beta_g_frag, dk_beta_g[bh, cs : cs + BS, ik_st : ik_st + block_DK])
                    T.copy(A_frag_acc, A_frag_raw[bh, cs : cs + BS, :])
                    for iv in T.serial(T.ceildiv(DV, block_DV)):
                        iv_st = iv * block_DV
                        T.copy(V_bf16[bh, cs : cs + BS, iv_st : iv_st + block_DV], V_l1)
                        T.copy(du_bf16[bh, cs : cs + BS, iv_st : iv_st + block_DV], du_l1)
                        # GEMM3: du @ V^T
                        T.copy(du_l1, du_l0a)
                        T.copy(V_l1, V_l0b_T, transpose=True)
                        T.mma(du_l0a, V_l0b_T, dA_v_frag, init=(iv == 0))
                        # GEMM4: A^T @ du — only B needs staging
                        # (A_T_l0a still live)
                        T.copy(du_l1, du_l0b)
                        T.mma(A_T_l0a, du_l0b, dv_beta_frag, init=True)
                        T.copy(dv_beta_frag, dv_beta[bh, cs : cs + BS, iv_st : iv_st + block_DV])
                    T.copy(dA_k_frag, dA_k_raw[bh, cs : cs + BS, :])
                    T.copy(dA_v_frag, dA_v_raw[bh, cs : cs + BS, :])

    return main


# K1cv_k2cd_fused: k1c + k2a_pre + refine + k2b + k2c + k2d fused into ONE
# 1-chunk universal CV kernel (block = chunk — serves EVERY shape, no
# guards). Per-block segments:
#   [S1 V1b]    k2a_pre body: dA_k/dA_v column-scale merge + lower-tri
#               mask + A/A^T lossless casts + colsums -> dbeta_inter_ub /
#               dg_inter_ub stay in UB (Phase B consumes them — never GM)
#   [S2 relays] V->C x3 (result_ub->dm_l1, A_ub->A_l1, A_T_ub->AT_l1),
#               each + mul-by-0 drain; sources have DISTINCT names
#               (ws map keys by SOURCE name)
#   [S3]        sign_flip -> sign_ub at the C-segment head (NOT kernel
#               head: a head-hoisted sign_ub stays live across V1a+V1b
#               and overflows the UB budget at large BS)
#   [S4 C]      refine chain: mma1 (A @ dm^T) -> tmp_frag -> dA_tmp_T GM
#               relay (C->C, same engine) -> dtt_l1 -> mma2 (A^T @ dtt^T)
#               -> ref_frag. L0B staging is SINGLE-NAME: mma2's B operand
#               re-stages into dm_l0b_T (same-name WAR is sync-visible);
#               a separate dtt_l0b_T would exceed L0B once K_l0b joins
#   [S5]        C->V relay: ref_frag -> dA_ub + sign pre-mul anchor
#               (value-exact: sign_flip in {+0, -0, -1})
#   [S6 V2]     k2b gate chain -> dA_final_fp32 GM write (pipeline
#               output) -> beta mul into the FRESH dA_beta_ub ->
#               ONE UB->L1 relay onto the FRESH dA_l1 name
#               (dA_final_beta never touches GM; relay dst names must
#               be UNIQUE per relay site) -> drains (dA_ub lives on
#               to S9 holding dA_final)
#   [S6.5 V1a]  dv = dv_beta * beta_row, LATE placement (out of the
#               relay-production path; shortens the AIC's first wait),
#               in-place src_ub mul + per-store drain
#   [S7 C]      GEMM6/7 persistent A operands: dA_l1 -> A_l0a and
#               dA_T_l1 -> AT_l0a (L0A names reused from the refine
#               mmas; the fp32 transpose stays V-side — fp32 L0A
#               transpose staging hits a codegen NaN bug)
#   [S8 ik]     per-iteration C-V alternation: inline K_T rider (K bf16
#               -> cast -> transpose -> kt_workspace GM relay -> K_T_l1
#               -> K_l0b) + mma6 (init) + mma7 (accumulate) -> dk_frag
#               -> dk_workspace relay (GQA pattern: the tensor NAME must
#               contain "workspace" — AscendCombineCV only creates
#               cross-core sync points for GM copies on workspace-named
#               buffers; per-ik slices are DISJOINT so no loop-carried
#               ws WAR) + anchor -> Phase A (dk = dk_beta_g *
#               (e^G*beta)_row + update). V setup reuses S1's Beta_ub /
#               G_exp_ub (no reload, no exp recompute)
#   [S9 V]      Phase B k2d combine: A_frag_raw loaded FIRST (MTE3->MTE2
#               load-order defense), dg_inter / dbeta_inter consumed from
#               UB, dbeta_k / dg_A_pos / dg_A_neg / dbeta / dg writes,
#               tail liveness dummy on the relay dst
# L0/L1 name reuse across disjoint live ranges: A_l0a / AT_l0a (refine
# mmas -> GEMM6/7 A operands), A_l1 / AT_l1 (V1b relay dsts ->
# dA_final_beta relays), dm_l0b_T (mma1 B-op -> mma2 B-op). Same-name
# reuse relies on same-name WAR being sync-visible.


@tilelang.jit(pass_configs=pass_configs, workspace_idx=[20, 21])
def k1cv_k2cd_fused(BH, S, DK, DV, BS, block_DK, block_DV):
    num_chunks = S // BS
    block_num = num_chunks * BH
    nK = (DK + block_DK - 1) // block_DK

    @T.prim_func
    def main(
        # k1b outputs (GM, cross-launch boundary)
        dA_k_raw: T.Tensor((BH, S, BS), "float"),
        dA_v_raw: T.Tensor((BH, S, BS), "float"),
        dk_beta_g: T.Tensor((BH, S, DK), "float"),
        dv_beta: T.Tensor((BH, S, DV), "float"),
        A_frag_raw: T.Tensor((BH, S, BS), "float"),
        # original inputs
        Beta: T.Tensor((BH, S), "bfloat16"),
        G: T.Tensor((BH, S), "float"),
        A: T.Tensor((BH, S, BS), "bfloat16"),
        K_bf16: T.Tensor((BH, S, DK), "bfloat16"),
        lower_tri_mask: T.Tensor((BS, BS), "float"),
        sign_flip: T.Tensor((BS, BS), "float"),
        # GM relay (S4, C->C same engine)
        dA_tmp_T: T.Tensor((BH, S, BS), "float"),
        # pipeline outputs
        dA_final_fp32: T.Tensor((BH, S, BS), "float"),
        dv: T.Tensor((BH, S, DV), "float"),
        dk_final: T.Tensor((BH, S, DK), "float"),
        dbeta_final: T.Tensor((BH, S), "float"),
        dg_final: T.Tensor((BH, S), "float"),
        dbeta_k: T.Tensor((BH, S), "float"),
        dg_A_positive: T.Tensor((BH, S, BS), "float"),
        dg_A_negative: T.Tensor((BH, S, BS), "float"),
        # JIT-allocated GM workspace (index 20): per-ik disjoint slices
        # carry dk_frag to UB; name must contain "workspace"
        dk_workspace: T.Tensor((BH, S, DK), "float"),
        # JIT-allocated GM workspace (index 21): the K_T rider's V->C relay
        # (GQA workspace_1 pattern — V writes, C reads; per-ik disjoint slices)
        kt_workspace: T.Tensor((BH, num_chunks * block_DK, nK * BS), "float"),
    ):
        with T.Kernel(block_num, threads=1, is_npu=True) as (cid):
            chunk_idx = cid // BH
            bh = cid % BH
            cs = chunk_idx * BS
            # ---- C-segment buffers (L1/L0; names reused across disjoint live ranges) ----
            A_l1 = T.alloc_L1((BS, BS), "float")  # S2 relay dst -> S4 mma1 A-op source
            AT_l1 = T.alloc_L1((BS, BS), "float")  # S2 relay dst -> S4 mma2 A-op source
            dm_l1 = T.alloc_L1((BS, BS), "float")  # S2 relay dst -> S4 mma1 B-op source
            dtt_l1 = T.alloc_L1((BS, BS), "float")  # S4 dA_tmp_T readback -> mma2 B-op source
            dA_l1 = T.alloc_L1((BS, BS), "float")  # S6 relay dst (dA_final_beta) -> S7 GEMM6 A-op source
            dA_T_l1 = T.alloc_L1((BS, BS), "float")  # S6 relay dst (dA_final_beta_T) -> S7 GEMM7 A-op source
            K_T_l1 = T.alloc_L1((block_DK, BS), "float")  # S8 rider relay dst (via kt_workspace)
            A_l0a = T.alloc_L0A((BS, BS), "float")  # S4 mma1 A-op; REUSED S7 GEMM6 A-op
            AT_l0a = T.alloc_L0A((BS, BS), "float")  # S4 mma2 A-op; REUSED S7 GEMM7 A-op
            dm_l0b_T = T.alloc_L0B((BS, BS), "float")  # S4 mma1 B-op staging; re-staged SAME-NAME for mma2
            K_l0b = T.alloc_L0B((BS, block_DK), "float")  # S8 GEMM6/7 B-op
            tmp_frag = T.alloc_L0C((BS, BS), "float")  # S4 mma1 accum -> GM relay
            ref_frag = T.alloc_L0C((BS, BS), "float")  # S4 mma2 accum -> C->V relay
            dk_frag = T.alloc_L0C((BS, block_DK), "float")  # S8 GEMM6+7 accum -> ws relay
            # ---- V1b buffers (k2a_pre; result_ub / A_ub / A_T_ub are the relay sources) ----
            dA_k_ub = T.alloc_shared((BS, BS), "float")
            dA_v_ub = T.alloc_shared((BS, BS), "float")
            Beta_half = T.alloc_shared((BS,), "bfloat16")
            Beta_ub = T.alloc_shared((BS,), "float")
            G_ub = T.alloc_shared((BS,), "float")
            G_exp_ub = T.alloc_shared((BS,), "float")
            Beta_col = T.alloc_shared((BS, BS), "float")
            G_exp_col = T.alloc_shared((BS, BS), "float")
            scale_k_col = T.alloc_shared((BS, BS), "float")
            t_v_ub = T.alloc_shared((BS, BS), "float")
            mask_ub = T.alloc_shared((BS, BS), "float")
            result_ub = T.alloc_shared((BS, BS), "float")  # t_k chain (in-place)
            A_half = T.alloc_shared((BS, BS), "bfloat16")
            A_ub = T.alloc_shared((BS, BS), "float")
            A_T_ub = T.alloc_shared((BS, BS), "float")
            colsum_k_ub = T.alloc_shared((BS,), "float")
            colsum_v_ub = T.alloc_shared((BS,), "float")
            scale_tmp_ub = T.alloc_shared((BS,), "float")
            dbeta_inter_ub = T.alloc_shared((BS,), "float")  # live until S9 (Phase B consumes from UB)
            dg_inter_ub = T.alloc_shared((BS,), "float")  # live until S9
            # ---- V2 buffers (k2b; dA_ub = C->V relay dst) ----
            dA_ub = T.alloc_shared((BS, BS), "float")
            dA_beta_ub = T.alloc_shared((BS, BS), "float")  # dA_final_beta (plain buffer; S6 relay source)
            sign_ub = T.alloc_shared((BS, BS), "float")
            G_half = T.alloc_shared((BS,), "float")
            G_row = T.alloc_shared((BS, BS), "float")
            G_col = T.alloc_shared((BS, BS), "float")
            bmask = T.alloc_shared((BS * BS // 8,), "uint8")
            Beta_2d = T.alloc_shared((BS, BS), "float")  # S6 beta mul; re-broadcast (idempotent) in S9
            tr_ub = T.alloc_shared((BS, BS), "float")
            # ---- V1a buffers (dv body; LATE placement, _dv suffix disjoint from V1b) ----
            src_ub = T.alloc_shared((BS, block_DV), "float")  # in-place mul
            Beta_half_dv = T.alloc_shared((BS,), "bfloat16")
            Beta_ub_dv = T.alloc_shared((BS,), "float")
            Beta_2d_dv = T.alloc_shared((BS, block_DV), "float")
            # ---- S8 buffers (ik loop; dk_upd_ub = C->V ws relay dst, live to S9 tail) ----
            scale_ub = T.alloc_shared((BS,), "float")  # s = e^G * beta (from S1's G_exp_ub/Beta_ub)
            K_in_ub = T.alloc_shared((BS, block_DK), "bfloat16")
            K_fp32_ub = T.alloc_shared((BS, block_DK), "float")
            K_tr_ub = T.alloc_shared((block_DK, BS), "float")
            dk_upd_ub = T.alloc_shared((BS, block_DK), "float")
            dk_beta_g_ub = T.alloc_shared((BS, block_DK), "float")
            scale_2d = T.alloc_shared((BS, block_DK), "float")
            # ---- S9 Phase B buffers (dbeta_inter_ub / dg_inter_ub reused from V1b) ----
            A_frag_ub = T.alloc_shared((BS, BS), "float")
            dbeta_k_ub = T.alloc_shared((BS,), "float")
            col_sum_ub = T.alloc_shared((BS,), "float")
            row_sum_ub = T.alloc_shared((BS,), "float")
            dg_final_ub = T.alloc_shared((BS,), "float")
            # ===== S1: k2a_pre body =====
            T.copy(dA_k_raw[bh, cs : cs + BS, :], dA_k_ub)
            T.copy(dA_v_raw[bh, cs : cs + BS, :], dA_v_ub)
            T.copy(Beta[bh, cs : cs + BS], Beta_half)
            T.tile.cast(Beta_ub, Beta_half, CAST_LOW2HIGH, BS)
            T.copy(G[bh, cs : cs + BS], G_ub)
            T.tile.exp(G_exp_ub, G_ub)
            T.tile.broadcast(Beta_col, Beta_ub, axis=0)
            T.tile.broadcast(G_exp_col, G_exp_ub, axis=0)
            T.tile.mul(scale_k_col, G_exp_col, Beta_col)
            T.tile.mul(result_ub, dA_k_ub, scale_k_col)
            T.tile.mul(t_v_ub, dA_v_ub, Beta_col)
            T.tile.add(result_ub, result_ub, t_v_ub)
            T.copy(lower_tri_mask[:, :], mask_ub)
            T.tile.mul(result_ub, result_ub, mask_ub)
            T.copy(A[bh, cs : cs + BS, :], A_half)
            T.tile.cast(A_ub, A_half, CAST_LOW2HIGH, BS * BS)
            T.tile.transpose(A_T_ub, A_ub)
            T.tile.mul(dA_k_ub, dA_k_ub, A_ub)
            T.reduce_sum(dA_k_ub, colsum_k_ub, dim=0)
            T.tile.mul(dA_v_ub, dA_v_ub, A_ub)
            T.reduce_sum(dA_v_ub, colsum_v_ub, dim=0)
            T.tile.mul(scale_tmp_ub, G_exp_ub, colsum_k_ub)
            T.tile.add(dbeta_inter_ub, scale_tmp_ub, colsum_v_ub)
            T.tile.mul(dg_inter_ub, scale_tmp_ub, Beta_ub)
            # ===== S2: V->C relays x3 + drains =====
            T.copy(result_ub, dm_l1)
            T.tile.mul(result_ub, result_ub, 0.0)
            T.copy(A_ub, A_l1)
            T.tile.mul(A_ub, A_ub, 0.0)
            T.copy(A_T_ub, AT_l1)
            T.tile.mul(A_T_ub, A_T_ub, 0.0)
            # ===== S3: hoisted constant at the C-segment head =====
            T.copy(sign_flip[:, :], sign_ub)
            # ===== S4: C refine GEMM chain (L0B staging single-name reuse) =====
            T.copy(A_l1, A_l0a)
            T.copy(dm_l1, dm_l0b_T, transpose=True)
            T.mma(A_l0a, dm_l0b_T, tmp_frag, init=True)
            T.copy(tmp_frag, dA_tmp_T[bh, cs : cs + BS, :])
            T.copy(dA_tmp_T[bh, cs : cs + BS, :], dtt_l1)
            T.copy(AT_l1, AT_l0a)
            T.copy(dtt_l1, dm_l0b_T, transpose=True)
            T.mma(AT_l0a, dm_l0b_T, ref_frag, init=True)
            # ===== S5: C->V relay + anchor (sign pre-mul) =====
            T.copy(ref_frag, dA_ub)
            T.tile.mul(dA_ub, dA_ub, sign_ub)
            # ===== S6: V seg 2 — k2b gate chain =====
            # gate = sign_flip (pre-mul anchor above) * masked_exp(G_diff)
            T.copy(G[bh, cs : cs + BS], G_half)
            T.tile.broadcast(G_row, G_half, axis=1)
            T.tile.broadcast(G_col, G_half, axis=0)
            T.tile.sub(G_row, G_row, G_col)
            T.tile.compare(bmask, G_row, T.float32(0.0), "LE")
            T.tile.exp(G_row, G_row)
            T.tile.select(G_row, bmask, G_row, T.float32(0.0), "VSEL_TENSOR_SCALAR_MODE")
            T.tile.mul(dA_ub, dA_ub, G_row)
            T.copy(dA_ub, dA_final_fp32[bh, cs : cs + BS, :])
            # beta mul into the FRESH dA_beta_ub — relaying FROM a relay
            # dst is unsupported by the ws machinery
            T.tile.broadcast(Beta_2d, Beta_ub, axis=1)
            T.tile.mul(dA_beta_ub, dA_ub, Beta_2d)
            # relay to a FRESH dA_l1 name — a reused dst overwrites
            # dst_to_workspace_map_ and breaks the CV sync-point pairing
            T.copy(dA_beta_ub, dA_l1)
            # x1.0 drain anchor — forces the relay's MTE3 read to drain (x*1.0 is IEEE value-exact)
            T.tile.mul(dA_beta_ub, dA_beta_ub, T.float32(1.0))
            T.tile.transpose(tr_ub, dA_beta_ub)
            # fp32 transpose stays V-side (fp32 L0A transpose staging hits a
            # codegen NaN bug — see the k1b GEMM2 comment) + relay + drains
            T.copy(tr_ub, dA_T_l1)
            T.tile.mul(tr_ub, tr_ub, 0.0)
            # end-of-life drain AFTER the transpose's read
            T.tile.mul(dA_beta_ub, dA_beta_ub, 0.0)
            # (no drain on dA_ub here — it holds dA_final until S9; its drain lives at the kernel tail)
            # ===== S6.5: dv body (LATE placement) =====
            T.copy(Beta[bh, cs : cs + BS], Beta_half_dv)
            T.tile.cast(Beta_ub_dv, Beta_half_dv, CAST_LOW2HIGH, BS)
            T.tile.broadcast(Beta_2d_dv, Beta_ub_dv, axis=1)
            for iv in T.serial(T.ceildiv(DV, block_DV)):
                iv_st = iv * block_DV
                T.copy(dv_beta[bh, cs : cs + BS, iv_st : iv_st + block_DV], src_ub)
                T.tile.mul(src_ub, src_ub, Beta_2d_dv)
                T.copy(src_ub, dv[bh, cs : cs + BS, iv_st : iv_st + block_DV])
                # per-store drain: the MTE3 read must drain before the next MTE2 reload reuses src_ub
                T.tile.mul(src_ub, src_ub, 0.0)
            # ===== S7: C staging — GEMM6/7 persistent A operands (L0A names reused) =====
            T.copy(dA_l1, A_l0a)
            T.copy(dA_T_l1, AT_l0a)
            # ===== S8: V setup (reuses S1's Beta_ub / G_exp_ub) =====
            T.tile.mul(scale_ub, G_exp_ub, Beta_ub)
            T.tile.broadcast(scale_2d, scale_ub, axis=1)
            # ===== S8: ik loop — per-iteration C-V alternation =====
            for ik in T.serial(T.ceildiv(DK, block_DK)):
                ik_st = ik * block_DK
                # inline K_T rider
                T.copy(K_bf16[bh, cs : cs + BS, ik_st : ik_st + block_DK], K_in_ub)
                T.tile.cast(K_fp32_ub, K_in_ub, CAST_LOW2HIGH, BS * block_DK)
                T.tile.transpose(K_tr_ub, K_fp32_ub)
                # V->C relay via explicit GM ws, per-ik DISJOINT slices (an
                # auto-relay inside a loop reuses one ws slot and races for nK >= 2)
                T.copy(
                    K_tr_ub,
                    kt_workspace[
                        bh,
                        chunk_idx * block_DK : chunk_idx * block_DK + block_DK,
                        ik * BS : ik * BS + BS,
                    ],
                )
                T.copy(
                    kt_workspace[
                        bh,
                        chunk_idx * block_DK : chunk_idx * block_DK + block_DK,
                        ik * BS : ik * BS + BS,
                    ],
                    K_T_l1,
                )
                T.copy(K_T_l1, K_l0b, transpose=True)
                # GEMM6: dk_beta_scaled = dA_final_beta @ K (beta folded into left operand)
                T.mma(A_l0a, K_l0b, dk_frag, init=True)
                # GEMM7 accumulated in place: += dA_final_beta^T @ K
                T.mma(AT_l0a, K_l0b, dk_frag, init=False)
                # C->V relay via explicit GM ws (per-ik disjoint slice) + x1.0 anchor
                T.copy(dk_frag, dk_workspace[bh, cs : cs + BS, ik_st : ik_st + block_DK])
                T.copy(dk_workspace[bh, cs : cs + BS, ik_st : ik_st + block_DK], dk_upd_ub)
                T.tile.mul(dk_upd_ub, dk_upd_ub, T.float32(1.0))
                # V Phase A (this K-block; in-place on the relay dst)
                T.copy(dk_beta_g[bh, cs : cs + BS, ik_st : ik_st + block_DK], dk_beta_g_ub)
                T.tile.mul(dk_beta_g_ub, dk_beta_g_ub, scale_2d)
                T.tile.add(dk_upd_ub, dk_beta_g_ub, dk_upd_ub)
                T.copy(dk_upd_ub, dk_final[bh, cs : cs + BS, ik_st : ik_st + block_DK])
            # ===== S9: V Phase B — k2d combine (dA_ub still holds dA_final) =====
            # A_frag loaded FIRST (MTE3->MTE2 load-order defense)
            T.copy(A_frag_raw[bh, cs : cs + BS, :], A_frag_ub)
            T.tile.broadcast(Beta_2d, Beta_ub, axis=1)
            # P = dA_final * A_frag_raw (in-place on dA_ub)
            T.tile.mul(dA_ub, dA_ub, A_frag_ub)
            # dbeta_k = rowsum(P) — BEFORE the beta mul scales the rows
            T.reduce_sum(dA_ub, dbeta_k_ub, dim=-1)
            # dg_A_positive = P * beta_row (in-place)
            T.tile.mul(dA_ub, dA_ub, Beta_2d)
            T.copy(dA_ub, dg_A_positive[bh, cs : cs + BS, :])
            # dg = dg_inter + beta*dbeta_k - colsum(dg_A_positive)
            # (dg_inter consumed from UB — S1's value, never went through GM)
            T.reduce_sum(dA_ub, col_sum_ub, dim=0)
            T.tile.mul(row_sum_ub, dbeta_k_ub, Beta_ub)
            T.tile.add(dg_final_ub, dg_inter_ub, row_sum_ub)
            T.tile.sub(dg_final_ub, dg_final_ub, col_sum_ub)
            T.copy(dg_final_ub, dg_final[bh, cs : cs + BS])
            # dbeta_final = dbeta_inter + dbeta_k (dbeta_inter from UB)
            T.tile.add(dbeta_inter_ub, dbeta_inter_ub, dbeta_k_ub)
            T.copy(dbeta_inter_ub, dbeta_final[bh, cs : cs + BS])
            T.copy(dbeta_k_ub, dbeta_k[bh, cs : cs + BS])
            # dg_A_negative = transpose of dg_A_positive (dst reuses the
            # dead A_frag_ub — same-name WAR is sync-visible)
            T.tile.transpose(A_frag_ub, dA_ub)
            T.copy(A_frag_ub, dg_A_negative[bh, cs : cs + BS, :])
            # tail liveness dummy (planner-alias defense: Phase B's GM->UB loads
            # must not reuse dk_upd_ub's UB space while the dk_final MTE3 read flies)
            T.tile.mul(dk_upd_ub, dk_upd_ub, 0.0)
            # dA_ub end-of-life drain (last read: the transpose above)
            T.tile.mul(dA_ub, dA_ub, 0.0)

    return main


# K1cv_k2cd_fused_2s: the 2-slot hand-unrolled fast path of the fused CV
# kernel (block = chunk PAIR; guard shapes only: num_chunks even, BS<=64,
# DK==block_DK (nK==1, flat body), DV<=128). Non-guard shapes take the
# 1-chunk universal k1cv_k2cd_fused. Same relay protocol as the 1-chunk
# kernel: relay dst names UNIQUE per site; never relay FROM a relay dst
# (fresh dA_beta intermediates); value-preserving x1.0 drains vs
# end-of-life mul-by-0; fp32 transposes stay V-side (fp32 L0A transpose
# staging hits a codegen NaN bug). The body is FLAT (nK==1): the dk C->V
# relay is a direct L0C->UB copy and the K_T rider relay a direct V->C
# copy — no GM workspaces needed (per-slot src/dst names).


@tilelang.jit(pass_configs=pass_configs)
def k1cv_k2cd_fused_2s(BH, S, DK, DV, BS, num_chunks, block_DK):
    half_chunks = num_chunks // 2
    block_num = half_chunks * BH

    @T.prim_func
    def main(
        # k1b outputs (GM, cross-launch boundary)
        dA_k_raw: T.Tensor((BH, S, BS), "float"),
        dA_v_raw: T.Tensor((BH, S, BS), "float"),
        dk_beta_g: T.Tensor((BH, S, DK), "float"),
        dv_beta: T.Tensor((BH, S, DV), "float"),
        A_frag_raw: T.Tensor((BH, S, BS), "float"),
        # original inputs
        Beta: T.Tensor((BH, S), "bfloat16"),
        G: T.Tensor((BH, S), "float"),
        A: T.Tensor((BH, S, BS), "bfloat16"),
        K_bf16: T.Tensor((BH, S, DK), "bfloat16"),
        lower_tri_mask: T.Tensor((BS, BS), "float"),
        sign_flip: T.Tensor((BS, BS), "float"),
        # GM relay (C->C same engine)
        dA_tmp_T: T.Tensor((BH, S, BS), "float"),
        # pipeline outputs
        dA_final_fp32: T.Tensor((BH, S, BS), "float"),
        dv: T.Tensor((BH, S, DV), "float"),
        dk_final: T.Tensor((BH, S, DK), "float"),
        dbeta_final: T.Tensor((BH, S), "float"),
        dg_final: T.Tensor((BH, S), "float"),
        dbeta_k: T.Tensor((BH, S), "float"),
        dg_A_positive: T.Tensor((BH, S, BS), "float"),
        dg_A_negative: T.Tensor((BH, S, BS), "float"),
    ):
        with T.Kernel(block_num, threads=1, is_npu=True) as (cid):
            pair_idx = cid // BH
            bh = cid % BH
            cs = pair_idx * 2 * BS
            cs0 = cs
            cs1 = cs + BS
            # ---- L1: V1b relay dsts + refine sources (per-slot) ----
            A_l1f_0 = T.alloc_L1((BS, BS), "float")  # V1b_0 relay dst -> refine mma1
            AT_l1_0 = T.alloc_L1((BS, BS), "float")  # V1b_0 relay dst -> refine mma2
            dm_l1_0 = T.alloc_L1((BS, BS), "float")  # V1b_0 relay dst -> mma1 B-op
            dtt_l1_0 = T.alloc_L1((BS, BS), "float")  # dA_tmp_T readback_0 -> mma2 B-op
            A_l1f_1 = T.alloc_L1((BS, BS), "float")
            AT_l1_1 = T.alloc_L1((BS, BS), "float")
            dm_l1_1 = T.alloc_L1((BS, BS), "float")
            dtt_l1_1 = T.alloc_L1((BS, BS), "float")
            # ---- L1: V2 relay dsts -> GEMM6/7 A-op sources (per-slot; names
            # DISTINCT from the V1b dsts above — relay dsts must be unique per site) ----
            dA_l1_0 = T.alloc_L1((BS, BS), "float")
            dA_T_l1_0 = T.alloc_L1((BS, BS), "float")
            dA_l1_1 = T.alloc_L1((BS, BS), "float")
            dA_T_l1_1 = T.alloc_L1((BS, BS), "float")
            K_T_l1_0 = T.alloc_L1((block_DK, BS), "float")  # rider_0 relay dst (direct, flat)
            K_T_l1_1 = T.alloc_L1((block_DK, BS), "float")  # rider_1 relay dst (direct, flat)
            # ---- L0A: refine mmas, REUSED per-slot by GEMM6/7 (same-engine) ----
            A_l0a_0 = T.alloc_L0A((BS, BS), "float")
            AT_l0a_0 = T.alloc_L0A((BS, BS), "float")
            A_l0a_1 = T.alloc_L0A((BS, BS), "float")
            AT_l0a_1 = T.alloc_L0A((BS, BS), "float")
            # ---- L0B: refine staging single-name per slot + ONE shared K_l0b ----
            dm_l0b_T_0 = T.alloc_L0B((BS, BS), "float")  # mma1_0 + re-staged mma2_0 B-op
            dm_l0b_T_1 = T.alloc_L0B((BS, BS), "float")  # mma1_1 + re-staged mma2_1 B-op
            K_l0b = T.alloc_L0B((BS, block_DK), "float")  # re-staged per slot (same name, same engine)
            # ---- L0C ----
            tmp_frag_0 = T.alloc_L0C((BS, BS), "float")
            ref_frag_0 = T.alloc_L0C((BS, BS), "float")
            tmp_frag_1 = T.alloc_L0C((BS, BS), "float")
            ref_frag_1 = T.alloc_L0C((BS, BS), "float")
            dk_frag_0 = T.alloc_L0C((BS, block_DK), "float")
            dk_frag_1 = T.alloc_L0C((BS, block_DK), "float")
            # ---- V1b shared working buffers ----
            dA_k_ub = T.alloc_shared((BS, BS), "float")
            dA_v_ub = T.alloc_shared((BS, BS), "float")
            Beta_half = T.alloc_shared((BS,), "bfloat16")
            Beta_col = T.alloc_shared((BS, BS), "float")
            G_exp_col = T.alloc_shared((BS, BS), "float")
            t_v_ub = T.alloc_shared((BS, BS), "float")
            A_half = T.alloc_shared((BS, BS), "bfloat16")
            colsum_k_ub = T.alloc_shared((BS,), "float")
            colsum_v_ub = T.alloc_shared((BS,), "float")
            scale_tmp_ub = T.alloc_shared((BS,), "float")
            # ---- V1b per-slot vectors (live to PhaseB; never overwritten by the other slot) ----
            Beta_ub_0 = T.alloc_shared((BS,), "float")
            G_ub_0 = T.alloc_shared((BS,), "float")
            G_exp_ub_0 = T.alloc_shared((BS,), "float")
            dbeta_inter_ub_0 = T.alloc_shared((BS,), "float")
            dg_inter_ub_0 = T.alloc_shared((BS,), "float")
            Beta_ub_1 = T.alloc_shared((BS,), "float")
            G_ub_1 = T.alloc_shared((BS,), "float")
            G_exp_ub_1 = T.alloc_shared((BS,), "float")
            dbeta_inter_ub_1 = T.alloc_shared((BS,), "float")
            dg_inter_ub_1 = T.alloc_shared((BS,), "float")
            # ---- V1b relay sources: per-slot DISTINCT (ws map keys by source) ----
            t_k_ub_0 = T.alloc_shared((BS, BS), "float")
            t_k_ub_1 = T.alloc_shared((BS, BS), "float")
            A_ub_0 = T.alloc_shared((BS, BS), "float")
            A_ub_1 = T.alloc_shared((BS, BS), "float")
            A_T_ub_0 = T.alloc_shared((BS, BS), "float")
            A_T_ub_1 = T.alloc_shared((BS, BS), "float")
            # ---- hoisted constants (BS<=64 guard: head-hoisting safe) ----
            sign_ub = T.alloc_shared((BS, BS), "float")
            mask_ub = T.alloc_shared((BS, BS), "float")
            # ---- V1a (dv; LATE placement, full-width, shared buffers) ----
            Beta_half_dv = T.alloc_shared((BS,), "bfloat16")
            Beta_ub_dv = T.alloc_shared((BS,), "float")
            Beta_2d_dv = T.alloc_shared((BS, DV), "float")
            src_ub = T.alloc_shared((BS, DV), "float")  # in-place mul + per-slot drains
            # ---- V2 shared working ----
            G_row = T.alloc_shared((BS, BS), "float")
            G_col = T.alloc_shared((BS, BS), "float")
            bmask = T.alloc_shared((BS * BS // 8,), "uint8")
            Beta_2d = T.alloc_shared((BS, BS), "float")
            # ---- V2 relay dsts + relay sources (per-slot) ----
            dA_ub_0 = T.alloc_shared((BS, BS), "float")  # CV relay dst 0; holds dA_final_0 to PhaseB_0
            dA_ub_1 = T.alloc_shared((BS, BS), "float")
            dA_beta_0 = T.alloc_shared((BS, BS), "float")  # dA_final_beta_0 (plain buffer, relay source)
            dA_beta_1 = T.alloc_shared((BS, BS), "float")
            tr_ub_0 = T.alloc_shared((BS, BS), "float")  # dA_final_beta_T_0 (V-side fp32 transpose)
            tr_ub_1 = T.alloc_shared((BS, BS), "float")
            # ---- rider (shared, per-slot sequential) ----
            K_in_ub = T.alloc_shared((BS, block_DK), "bfloat16")
            K_fp32_ub = T.alloc_shared((BS, block_DK), "float")
            K_tr_ub_0 = T.alloc_shared((block_DK, BS), "float")  # rider_0 relay source
            K_tr_ub_1 = T.alloc_shared((block_DK, BS), "float")  # rider_1 relay source
            # ---- k2cd V (Phase A/B; scale/scale_2d recomputed per slot) ----
            scale_ub = T.alloc_shared((BS,), "float")
            scale_2d = T.alloc_shared((BS, block_DK), "float")
            dk_upd_ub_0 = T.alloc_shared((BS, block_DK), "float")  # ws relay dst 0
            dk_upd_ub_1 = T.alloc_shared((BS, block_DK), "float")  # ws relay dst 1
            dk_beta_g_ub = T.alloc_shared((BS, block_DK), "float")
            A_frag_ub = T.alloc_shared((BS, BS), "float")
            dbeta_k_ub = T.alloc_shared((BS,), "float")
            col_sum_ub = T.alloc_shared((BS,), "float")
            row_sum_ub = T.alloc_shared((BS,), "float")
            dg_final_ub = T.alloc_shared((BS,), "float")
            dA_final_ub = T.alloc_shared((BS, BS), "float")  # PhaseB GM read-back dst
            # ===== head: hoisted constants (safe under the BS<=64 guard) =====
            T.copy(sign_flip[:, :], sign_ub)
            T.copy(lower_tri_mask[:, :], mask_ub)
            # ===== V1b slot 0: k2a_pre body (shared buffers WAR-reused by
            # slot 1; per-slot vectors live to PhaseB_0) =====
            T.copy(dA_k_raw[bh, cs0 : cs0 + BS, :], dA_k_ub)
            T.copy(dA_v_raw[bh, cs0 : cs0 + BS, :], dA_v_ub)
            T.copy(Beta[bh, cs0 : cs0 + BS], Beta_half)
            T.tile.cast(Beta_ub_0, Beta_half, CAST_LOW2HIGH, BS)
            T.copy(G[bh, cs0 : cs0 + BS], G_ub_0)
            T.tile.exp(G_exp_ub_0, G_ub_0)
            T.tile.broadcast(Beta_col, Beta_ub_0, axis=0)
            T.tile.broadcast(G_exp_col, G_exp_ub_0, axis=0)
            T.tile.mul(G_exp_col, G_exp_col, Beta_col)
            T.tile.mul(t_k_ub_0, dA_k_ub, G_exp_col)
            T.tile.mul(t_v_ub, dA_v_ub, Beta_col)
            T.tile.add(t_k_ub_0, t_k_ub_0, t_v_ub)
            T.tile.mul(t_k_ub_0, t_k_ub_0, mask_ub)
            T.copy(A[bh, cs0 : cs0 + BS, :], A_half)
            T.tile.cast(A_ub_0, A_half, CAST_LOW2HIGH, BS * BS)
            T.tile.transpose(A_T_ub_0, A_ub_0)
            T.tile.mul(dA_k_ub, dA_k_ub, A_ub_0)
            T.reduce_sum(dA_k_ub, colsum_k_ub, dim=0)
            T.tile.mul(dA_v_ub, dA_v_ub, A_ub_0)
            T.reduce_sum(dA_v_ub, colsum_v_ub, dim=0)
            T.tile.mul(scale_tmp_ub, G_exp_ub_0, colsum_k_ub)
            T.tile.add(dbeta_inter_ub_0, scale_tmp_ub, colsum_v_ub)
            T.tile.mul(dg_inter_ub_0, scale_tmp_ub, Beta_ub_0)
            # (per-slot UB vectors — PhaseB_0 consumes; no GM writes)
            # V->C relays slot 0 + drains (early: slot 1's V work overlaps refine_0)
            T.copy(t_k_ub_0, dm_l1_0)
            T.tile.mul(t_k_ub_0, t_k_ub_0, 0.0)
            T.copy(A_ub_0, A_l1f_0)
            T.tile.mul(A_ub_0, A_ub_0, 0.0)
            T.copy(A_T_ub_0, AT_l1_0)
            T.tile.mul(A_T_ub_0, A_T_ub_0, 0.0)
            # ===== V1b slot 1 (isomorphic to slot 0: cs1 + _1 names; shared
            # buffers rewritten in place — slot-0 values already consumed) =====
            T.copy(dA_k_raw[bh, cs1 : cs1 + BS, :], dA_k_ub)
            T.copy(dA_v_raw[bh, cs1 : cs1 + BS, :], dA_v_ub)
            T.copy(Beta[bh, cs1 : cs1 + BS], Beta_half)
            T.tile.cast(Beta_ub_1, Beta_half, CAST_LOW2HIGH, BS)
            T.copy(G[bh, cs1 : cs1 + BS], G_ub_1)
            T.tile.exp(G_exp_ub_1, G_ub_1)
            T.tile.broadcast(Beta_col, Beta_ub_1, axis=0)
            T.tile.broadcast(G_exp_col, G_exp_ub_1, axis=0)
            T.tile.mul(G_exp_col, G_exp_col, Beta_col)
            T.tile.mul(t_k_ub_1, dA_k_ub, G_exp_col)
            T.tile.mul(t_v_ub, dA_v_ub, Beta_col)
            T.tile.add(t_k_ub_1, t_k_ub_1, t_v_ub)
            T.tile.mul(t_k_ub_1, t_k_ub_1, mask_ub)
            T.copy(A[bh, cs1 : cs1 + BS, :], A_half)
            T.tile.cast(A_ub_1, A_half, CAST_LOW2HIGH, BS * BS)
            T.tile.transpose(A_T_ub_1, A_ub_1)
            T.tile.mul(dA_k_ub, dA_k_ub, A_ub_1)
            T.reduce_sum(dA_k_ub, colsum_k_ub, dim=0)
            T.tile.mul(dA_v_ub, dA_v_ub, A_ub_1)
            T.reduce_sum(dA_v_ub, colsum_v_ub, dim=0)
            T.tile.mul(scale_tmp_ub, G_exp_ub_1, colsum_k_ub)
            T.tile.add(dbeta_inter_ub_1, scale_tmp_ub, colsum_v_ub)
            T.tile.mul(dg_inter_ub_1, scale_tmp_ub, Beta_ub_1)
            # (per-slot UB vectors — PhaseB_1 consumes; no GM writes)
            # V->C relays slot 1 + drains
            T.copy(t_k_ub_1, dm_l1_1)
            T.tile.mul(t_k_ub_1, t_k_ub_1, 0.0)
            T.copy(A_ub_1, A_l1f_1)
            T.tile.mul(A_ub_1, A_ub_1, 0.0)
            T.copy(A_T_ub_1, AT_l1_1)
            T.tile.mul(A_T_ub_1, A_T_ub_1, 0.0)
            # ===== V1a slot 0: dv full-width, no loop (DV<=128 guard allows
            # one full-width tile; LATE placement keeps dv out of the
            # relay-production path; shared buffers, slot-overwrite) =====
            T.copy(Beta[bh, cs0 : cs0 + BS], Beta_half_dv)
            T.tile.cast(Beta_ub_dv, Beta_half_dv, CAST_LOW2HIGH, BS)
            # Beta_2d_dv is (BS, DV) — one broadcast covers the full width
            T.tile.broadcast(Beta_2d_dv, Beta_ub_dv, axis=1)
            T.copy(dv_beta[bh, cs0 : cs0 + BS, :], src_ub)
            # in-place mul — src_ub holds dv_beta * Beta_2d on exit
            T.tile.mul(src_ub, src_ub, Beta_2d_dv)
            T.copy(src_ub, dv[bh, cs0 : cs0 + BS, :])
            # dv-store drain: the MTE3 read must drain before slot 1's
            # MTE2 reload reuses src_ub (the same-name V op forces the drain)
            T.tile.mul(src_ub, src_ub, 0.0)
            # ===== V1a slot 1 (shared names redefined, Beta reloaded) =====
            T.copy(Beta[bh, cs1 : cs1 + BS], Beta_half_dv)
            T.tile.cast(Beta_ub_dv, Beta_half_dv, CAST_LOW2HIGH, BS)
            T.tile.broadcast(Beta_2d_dv, Beta_ub_dv, axis=1)
            T.copy(dv_beta[bh, cs1 : cs1 + BS, :], src_ub)
            T.tile.mul(src_ub, src_ub, Beta_2d_dv)
            T.copy(src_ub, dv[bh, cs1 : cs1 + BS, :])
            # dv-store drain (same as slot 0)
            T.tile.mul(src_ub, src_ub, 0.0)
            # ===== C refine slot 0 (dm_l0b_T_0: single-name L0B reuse —
            # mma2's B-op re-staged into the same name) =====
            T.copy(A_l1f_0, A_l0a_0)
            T.copy(dm_l1_0, dm_l0b_T_0, transpose=True)
            T.mma(A_l0a_0, dm_l0b_T_0, tmp_frag_0, init=True)
            T.copy(tmp_frag_0, dA_tmp_T[bh, cs0 : cs0 + BS, :])
            # C->C GM relay, same engine on both sides — L0/L1 ops serialize
            # naturally, no drain dummies needed
            T.copy(dA_tmp_T[bh, cs0 : cs0 + BS, :], dtt_l1_0)
            T.copy(AT_l1_0, AT_l0a_0)
            T.copy(dtt_l1_0, dm_l0b_T_0, transpose=True)
            T.mma(AT_l0a_0, dm_l0b_T_0, ref_frag_0, init=True)
            # ===== C refine slot 1 (same single-name L0B reuse) =====
            T.copy(A_l1f_1, A_l0a_1)
            T.copy(dm_l1_1, dm_l0b_T_1, transpose=True)
            T.mma(A_l0a_1, dm_l0b_T_1, tmp_frag_1, init=True)
            T.copy(tmp_frag_1, dA_tmp_T[bh, cs1 : cs1 + BS, :])
            T.copy(dA_tmp_T[bh, cs1 : cs1 + BS, :], dtt_l1_1)
            T.copy(AT_l1_1, AT_l0a_1)
            T.copy(dtt_l1_1, dm_l0b_T_1, transpose=True)
            T.mma(AT_l0a_1, dm_l0b_T_1, ref_frag_1, init=True)
            # ===== CV relay slot 0 + sign pre-mul anchor =====
            T.copy(ref_frag_0, dA_ub_0)
            # x*(±1.0) is IEEE value-exact — bakes sign_flip in before the gate chain
            T.tile.mul(dA_ub_0, dA_ub_0, sign_ub)
            # ===== CV relay slot 1 + sign pre-mul anchor =====
            T.copy(ref_frag_1, dA_ub_1)
            T.tile.mul(dA_ub_1, dA_ub_1, sign_ub)
            # ===== V2 slot 0: k2b gate chain (broadcasts from the per-slot
            # G_ub_0/Beta_ub_0 loaded in V1b) =====
            # gate = sign_flip (pre-mul anchor in the CV relay) * masked_exp(G_diff)
            T.tile.broadcast(G_row, G_ub_0, axis=1)
            T.tile.broadcast(G_col, G_ub_0, axis=0)
            T.tile.sub(G_row, G_row, G_col)
            T.tile.compare(bmask, G_row, T.float32(0.0), "LE")
            T.tile.exp(G_row, G_row)
            T.tile.select(G_row, bmask, G_row, T.float32(0.0), "VSEL_TENSOR_SCALAR_MODE")
            T.tile.mul(dA_ub_0, dA_ub_0, G_row)
            T.copy(dA_ub_0, dA_final_fp32[bh, cs0 : cs0 + BS, :])
            # beta mul into the FRESH dA_beta_0 (never relay FROM a relay dst);
            # Beta_2d shared, slot-rewritten
            T.tile.broadcast(Beta_2d, Beta_ub_0, axis=1)
            T.tile.mul(dA_beta_0, dA_ub_0, Beta_2d)
            # relays + drains (x1.0 value-preserving after the relay, before
            # the transpose; mul-by-0 end-of-life after the last read)
            T.copy(dA_beta_0, dA_l1_0)
            T.tile.mul(dA_beta_0, dA_beta_0, T.float32(1.0))
            # fp32 transpose stays V-side (codegen NaN bug — see the k1b GEMM2 comment)
            T.tile.transpose(tr_ub_0, dA_beta_0)
            T.copy(tr_ub_0, dA_T_l1_0)
            T.tile.mul(tr_ub_0, tr_ub_0, 0.0)
            # end-of-life drain AFTER the transpose's read
            T.tile.mul(dA_beta_0, dA_beta_0, 0.0)
            # dA_ub_0 end-of-life drain HERE (UB budget: dk_upd_0/1 are both
            # live through Phase A; PhaseB_0 reads dA_final_0 back from GM)
            T.tile.mul(dA_ub_0, dA_ub_0, 0.0)
            # ===== V2 slot 1 (isomorphic to slot 0; shared working buffers
            # safe to rewrite — slot 0's chain consumed them) =====
            # gate on dA_ub_1 (sign anchor already applied above)
            T.tile.broadcast(G_row, G_ub_1, axis=1)
            T.tile.broadcast(G_col, G_ub_1, axis=0)
            T.tile.sub(G_row, G_row, G_col)
            T.tile.compare(bmask, G_row, T.float32(0.0), "LE")
            T.tile.exp(G_row, G_row)
            T.tile.select(G_row, bmask, G_row, T.float32(0.0), "VSEL_TENSOR_SCALAR_MODE")
            T.tile.mul(dA_ub_1, dA_ub_1, G_row)
            T.copy(dA_ub_1, dA_final_fp32[bh, cs1 : cs1 + BS, :])
            # beta mul into the FRESH dA_beta_1 (per-slot Beta_ub_1 source)
            T.tile.broadcast(Beta_2d, Beta_ub_1, axis=1)
            T.tile.mul(dA_beta_1, dA_ub_1, Beta_2d)
            # relay dsts dA_l1_1/dA_T_l1_1 slot-distinct from slot 0's
            T.copy(dA_beta_1, dA_l1_1)
            T.tile.mul(dA_beta_1, dA_beta_1, T.float32(1.0))
            T.tile.transpose(tr_ub_1, dA_beta_1)
            T.copy(tr_ub_1, dA_T_l1_1)
            T.tile.mul(tr_ub_1, tr_ub_1, 0.0)
            # end-of-life drain AFTER the transpose's read
            T.tile.mul(dA_beta_1, dA_beta_1, 0.0)
            # dA_ub_1 end-of-life drain HERE (same UB-budget reason as slot 0)
            T.tile.mul(dA_ub_1, dA_ub_1, 0.0)
            # ===== V rider slot 0: inline K_T (full-width, block_DK==DK —
            # no ik loop, no GM workspaces; shared buffers rewritten by slot 1) =====
            T.copy(K_bf16[bh, cs0 : cs0 + BS, :], K_in_ub)
            T.tile.cast(K_fp32_ub, K_in_ub, CAST_LOW2HIGH, BS * block_DK)
            T.tile.transpose(K_tr_ub_0, K_fp32_ub)
            # direct V->C relay, flat single shot (no GM ws hop)
            T.copy(K_tr_ub_0, K_T_l1_0)
            # end-of-life drain after the relay's read
            T.tile.mul(K_tr_ub_0, K_tr_ub_0, 0.0)
            # ===== V rider slot 1 (isomorphic; shared buffers rewritten in place) =====
            T.copy(K_bf16[bh, cs1 : cs1 + BS, :], K_in_ub)
            T.tile.cast(K_fp32_ub, K_in_ub, CAST_LOW2HIGH, BS * block_DK)
            T.tile.transpose(K_tr_ub_1, K_fp32_ub)
            T.copy(K_tr_ub_1, K_T_l1_1)
            T.tile.mul(K_tr_ub_1, K_tr_ub_1, 0.0)
            # ===== C k2cd: four-way L0A staging front-loaded (persistent A
            # operands for both GEMM chains stage before either mma issues;
            # L0A names reused from the refine mmas) =====
            T.copy(dA_l1_0, A_l0a_0)
            T.copy(dA_T_l1_0, AT_l0a_0)
            T.copy(dA_l1_1, A_l0a_1)
            T.copy(dA_T_l1_1, AT_l0a_1)
            # slot 0 GEMM chain: dk_frag_0 = dA_final_beta_0 @ K_0, then += dA_final_beta_T_0 @ K_0
            T.copy(K_T_l1_0, K_l0b, transpose=True)
            T.mma(A_l0a_0, K_l0b, dk_frag_0, init=True)
            T.mma(AT_l0a_0, K_l0b, dk_frag_0, init=False)
            # ===== dk relay_0 + x1.0 anchor (direct L0C->UB copy) =====
            T.copy(dk_frag_0, dk_upd_ub_0)
            T.tile.mul(dk_upd_ub_0, dk_upd_ub_0, T.float32(1.0))
            # slot 1 GEMM chain (K_l0b re-staged under the same name — same engine)
            T.copy(K_T_l1_1, K_l0b, transpose=True)
            T.mma(A_l0a_1, K_l0b, dk_frag_1, init=True)
            T.mma(AT_l0a_1, K_l0b, dk_frag_1, init=False)
            # ===== dk relay_1 + anchor_1 =====
            T.copy(dk_frag_1, dk_upd_ub_1)
            T.tile.mul(dk_upd_ub_1, dk_upd_ub_1, T.float32(1.0))
            # ===== V PhaseA slot 0: scale from the per-slot G_exp_ub_0/Beta_ub_0
            # held live since V1b (no reload, no exp recompute); shared buffers
            # rewritten by slot 1; full-width load/store =====
            T.tile.mul(scale_ub, G_exp_ub_0, Beta_ub_0)
            T.tile.broadcast(scale_2d, scale_ub, axis=1)
            T.copy(dk_beta_g[bh, cs0 : cs0 + BS, :], dk_beta_g_ub)
            T.tile.mul(dk_beta_g_ub, dk_beta_g_ub, scale_2d)
            # in-place accumulate on the relay dst
            T.tile.add(dk_upd_ub_0, dk_beta_g_ub, dk_upd_ub_0)
            T.copy(dk_upd_ub_0, dk_final[bh, cs0 : cs0 + BS, :])
            # ===== V PhaseA slot 1 (isomorphic; shared buffers rewritten in place) =====
            T.tile.mul(scale_ub, G_exp_ub_1, Beta_ub_1)
            T.tile.broadcast(scale_2d, scale_ub, axis=1)
            T.copy(dk_beta_g[bh, cs1 : cs1 + BS, :], dk_beta_g_ub)
            T.tile.mul(dk_beta_g_ub, dk_beta_g_ub, scale_2d)
            T.tile.add(dk_upd_ub_1, dk_beta_g_ub, dk_upd_ub_1)
            T.copy(dk_upd_ub_1, dk_final[bh, cs1 : cs1 + BS, :])
            # ===== V PhaseB slot 0: k2d combine (dA_final_0 read back from GM
            # into dA_final_ub; dA_ub_0 died at V2 for the UB budget) =====
            # A_frag loaded FIRST (MTE3->MTE2 load-order defense, also covers the read-back)
            T.copy(A_frag_raw[bh, cs0 : cs0 + BS, :], A_frag_ub)
            T.copy(dA_final_fp32[bh, cs0 : cs0 + BS, :], dA_final_ub)
            # Beta_2d re-broadcast from THIS slot's Beta_ub_0 — the shared
            # Beta_2d was last written by V2 slot 1 (slot 1's beta)
            T.tile.broadcast(Beta_2d, Beta_ub_0, axis=1)
            # P = dA_final_0 * A_frag_raw (in-place on dA_final_ub)
            T.tile.mul(dA_final_ub, dA_final_ub, A_frag_ub)
            # dbeta_k = rowsum(P) — BEFORE the beta mul scales the rows
            T.reduce_sum(dA_final_ub, dbeta_k_ub, dim=-1)
            # dg_A_positive = P * beta_row (in-place)
            T.tile.mul(dA_final_ub, dA_final_ub, Beta_2d)
            T.copy(dA_final_ub, dg_A_positive[bh, cs0 : cs0 + BS, :])
            # dg = dg_inter + beta*dbeta_k - colsum(dg_A_positive)
            # (dg_inter_0 consumed from the per-slot UB vector — no GM hop)
            T.reduce_sum(dA_final_ub, col_sum_ub, dim=0)
            T.tile.mul(row_sum_ub, dbeta_k_ub, Beta_ub_0)
            T.tile.add(dg_final_ub, dg_inter_ub_0, row_sum_ub)
            T.tile.sub(dg_final_ub, dg_final_ub, col_sum_ub)
            T.copy(dg_final_ub, dg_final[bh, cs0 : cs0 + BS])
            # dbeta_final = dbeta_inter + dbeta_k (dbeta_inter from UB)
            T.tile.add(dbeta_inter_ub_0, dbeta_inter_ub_0, dbeta_k_ub)
            T.copy(dbeta_inter_ub_0, dbeta_final[bh, cs0 : cs0 + BS])
            T.copy(dbeta_k_ub, dbeta_k[bh, cs0 : cs0 + BS])
            # dg_A_negative = transpose of dg_A_positive (dst reuses dead A_frag_ub)
            T.tile.transpose(A_frag_ub, dA_final_ub)
            T.copy(A_frag_ub, dg_A_negative[bh, cs0 : cs0 + BS, :])
            # ===== V PhaseB slot 1 (isomorphic; dA_final_1 read back from GM;
            # shared working buffers slot-rewritten) =====
            # A_frag loaded FIRST, per slot (MTE3->MTE2 defense)
            T.copy(A_frag_raw[bh, cs1 : cs1 + BS, :], A_frag_ub)
            T.copy(dA_final_fp32[bh, cs1 : cs1 + BS, :], dA_final_ub)
            # Beta_2d restored from THIS slot's Beta_ub_1 — PhaseB_0's own
            # re-broadcast left slot 0's beta in the shared buffer
            T.tile.broadcast(Beta_2d, Beta_ub_1, axis=1)
            T.tile.mul(dA_final_ub, dA_final_ub, A_frag_ub)
            T.reduce_sum(dA_final_ub, dbeta_k_ub, dim=-1)
            T.tile.mul(dA_final_ub, dA_final_ub, Beta_2d)
            T.copy(dA_final_ub, dg_A_positive[bh, cs1 : cs1 + BS, :])
            T.reduce_sum(dA_final_ub, col_sum_ub, dim=0)
            T.tile.mul(row_sum_ub, dbeta_k_ub, Beta_ub_1)
            T.tile.add(dg_final_ub, dg_inter_ub_1, row_sum_ub)
            T.tile.sub(dg_final_ub, dg_final_ub, col_sum_ub)
            T.copy(dg_final_ub, dg_final[bh, cs1 : cs1 + BS])
            T.tile.add(dbeta_inter_ub_1, dbeta_inter_ub_1, dbeta_k_ub)
            T.copy(dbeta_inter_ub_1, dbeta_final[bh, cs1 : cs1 + BS])
            T.copy(dbeta_k_ub, dbeta_k[bh, cs1 : cs1 + BS])
            T.tile.transpose(A_frag_ub, dA_final_ub)
            T.copy(A_frag_ub, dg_A_negative[bh, cs1 : cs1 + BS, :])
            # ===== tail liveness dummies (planner-alias defense: Phase B's
            # loads must not reuse the relay dsts' UB space while the dk_final reads fly) =====
            T.tile.mul(dk_upd_ub_0, dk_upd_ub_0, 0.0)
            T.tile.mul(dk_upd_ub_1, dk_upd_ub_1, 0.0)

    return main


def _prepare(B, S, H, DK, DV, chunk_size, seed=0):
    torch.manual_seed(seed)
    BH = B * H
    BS = chunk_size
    K = torch.randn(BH, S, DK, dtype=torch.bfloat16, device="cpu")
    K = F.normalize(K.float(), dim=-1, p=2).to(torch.bfloat16)
    V = torch.randn(BH, S, DV, dtype=torch.bfloat16, device="cpu")
    V = F.normalize(V.float(), dim=-1, p=2).to(torch.bfloat16)
    Beta = torch.randn(BH, S, dtype=torch.bfloat16, device="cpu")
    G = torch.randn(BH, S, dtype=torch.float32, device="cpu")
    A = torch.randn(BH, S, BS, dtype=torch.bfloat16, device="cpu")
    dw = torch.randn(BH, S, DK, dtype=torch.bfloat16, device="cpu")
    du = torch.randn(BH, S, DV, dtype=torch.bfloat16, device="cpu")
    return K, V, Beta, G, A, dw, du


def _vec_block(dim):
    """Widest Vector-kernel tile width for a K/V dim.

    Tile ops are lowered at the load-region extent, so using the full dim
    (capped at 128) halves the op count vs 64-wide chunks and doubles the MTE
    burst width. Requires dim % block == 0 (all supported shapes satisfy it).
    """
    for b in (128, 64, 32, 16):
        if dim % b == 0:
            return min(b, dim)
    return dim


def _get_core_num(dev):
    """Query the device's cube core count at runtime.

    Falls back to 20 (the 910B3 cube core count) when the property query
    fails. KNOWN LIMITATION: on a device with more
    than 20 cores where the query fails, the fallback under-utilizes the
    device (the count only sizes the launch grid, never correctness).
    """
    try:
        return int(torch.npu.get_device_properties(dev).cube_core_num)
    except Exception:
        return 20  # 910B3 fallback


def _validate_inputs(K, V, Beta, G, A, dw, du, B, S, H, DK, DV, chunk_size, block_DK, block_DV):
    """Host-side input validation.

    All checks are HOST-ONLY (no device work). Catches silently-wrong
    configurations at the wrapper boundary with actionable messages
    instead of cryptic downstream errors:
      * S < chunk_size: num_chunks=0 divides by zero inside kernel
        construction (a TVM DiagnosticError "Divide by zero" — not silent,
        but unreadable); rejected here deterministically.
      * S % chunk_size != 0: the kernels floor-divide and the output tail
        rows would be left UNINITIALIZED (torch.empty) while a reference
        computes them — silently wrong output; rejected.
      * fractal 16 alignment (chunk_size/DK/DV): the Cube GEMM fractal
        layout requirement (a non-multiple would miscompile or garbage).
      * dtype: the pipeline's kernel inputs are bfloat16 (k1b consumes the
        raw bf16 tensors; fp32 inputs are NOT supported by the kernels).
      * device/ndim/shape consistency + contiguity.
      * T.mma L0 budget formulas (fail here instead of deep in the
        backend): k1b's explicit staging (L0A = A_T + dw + K + du,
        L0B = K_T + dw + V_T + du, L0C = 5 fragments) and the fused
        kernel's (k1cv_k2cd_fused: 2 L0A + 2 L0B + 3 L0C),
        conservative no-aliasing sums.
    """
    BS = chunk_size
    BH = B * H
    assert K.dim() == 3 and V.dim() == 3 and A.dim() == 3 and dw.dim() == 3 and du.dim() == 3, (
        f"K/V/A/dw/du must be 3D [BH,S,D]: K.dim={K.dim()}, V.dim={V.dim()}, A.dim={A.dim()}, dw.dim={dw.dim()}, du.dim={du.dim()}"
    )
    assert Beta.dim() == 2 and G.dim() == 2, f"Beta/G must be 2D [BH,S]: Beta.dim={Beta.dim()}, G.dim={G.dim()}"
    assert K.shape == (BH, S, DK) and dw.shape == (BH, S, DK), f"K/dw must be [{BH},{S},{DK}]: K={tuple(K.shape)}, dw={tuple(dw.shape)}"
    assert V.shape == (BH, S, DV) and du.shape == (BH, S, DV), f"V/du must be [{BH},{S},{DV}]: V={tuple(V.shape)}, du={tuple(du.shape)}"
    assert A.shape == (BH, S, BS), f"A must be [{BH},{S},{BS}] (last dim = chunk_size): A={tuple(A.shape)}"
    assert Beta.shape == (BH, S) and G.shape == (BH, S), f"Beta/G must be [{BH},{S}]: Beta={tuple(Beta.shape)}, G={tuple(G.shape)}"
    assert chunk_size <= S, f"S={S} must be >= chunk_size={chunk_size} (num_chunks=0 breaks kernel construction with a divide-by-zero)"
    assert S % chunk_size == 0, (
        f"S={S} must be divisible by chunk_size={chunk_size} (the kernels floor-divide; tail output rows would be uninitialized)"
    )
    assert chunk_size % 16 == 0 and DK % 16 == 0 and DV % 16 == 0, (
        f"fractal 16 alignment required: chunk_size={chunk_size}, DK={DK}, DV={DV}"
    )
    # Explicit UB upper bound — the Vector kernels' (k2a_pre/k2b and the
    # fused variants) UB working sets scale as BS^2 (~7-13*BS^2 fp32 words)
    # and overflow the ~207KB per-core UB budget beyond BS=80.
    assert chunk_size <= 80, (
        f"chunk_size={chunk_size} exceeds the maximum supported value 80: the Vector kernels' "
        f"UB working sets scale as BS^2 and overflow the ~207KB budget beyond BS=80 "
        f"(supported: multiples of 16 in [16, 80])"
    )
    assert DK % block_DK == 0 and DV % block_DV == 0, (
        f"DK/DV must be divisible by their block sizes (partial K/V tiles would "
        f"read out of bounds): DK={DK}%block_DK={block_DK}, DV={DV}%block_DV={block_DV}"
    )
    # The Cube mma tile N/K dims ARE the block sizes (k1b GEMM2's N=block_DK,
    # the dk GEMMs' N=block_DK, ...) — they need the same fractal 16 alignment as
    # DK/DV themselves; a non-16-multiple block (e.g. 24, a legal divisor
    # of DK=48) passes the divisibility checks above but crashes the
    # aicore at runtime.
    assert block_DK % 16 == 0 and block_DV % 16 == 0, (
        f"fractal 16 alignment required for the block sizes: block_DK={block_DK}, "
        f"block_DV={block_DV} (the Cube mma tile N/K dims; a non-16-multiple "
        f"block crashes the aicore at runtime)"
    )
    assert K.dtype == V.dtype == A.dtype == dw.dtype == du.dtype == Beta.dtype == torch.bfloat16, (
        f"K/V/A/dw/du/Beta must be bfloat16: K={K.dtype}, V={V.dtype}, A={A.dtype}, dw={dw.dtype}, du={du.dtype}, Beta={Beta.dtype}"
    )
    assert G.dtype == torch.float32, f"G must be float32: G={G.dtype}"
    assert K.device.type == "npu", f"kernels are NPU-only: K.device={K.device} (move tensors with .npu() first)"
    for t, name in ((K, "K"), (V, "V"), (Beta, "Beta"), (G, "G"), (A, "A"), (dw, "dw"), (du, "du")):
        assert t.is_contiguous(), f"{name} must be contiguous (got strides {t.stride()})"
    # T.mma L0 budgets (bytes, conservative no-aliasing sums)
    l0a = max(
        (BS * BS + 2 * BS * block_DK + BS * block_DV) * 2,  # k1b: A_T+dw+K+du (bf16)
        2 * BS * BS * 4,  # fused: A_l0a+AT_l0a, reused refine->GEMM6/7 (fp32)
    )
    l0b = max(
        2 * (BS * block_DK + BS * block_DV) * 2,  # k1b: K_T+dw+V_T+du (bf16)
        (BS * BS + BS * block_DK) * 4,  # fused: dm_l0b_T (single-name refine staging) + K_l0b (fp32)
    )
    l0c = max(
        (3 * BS * BS + BS * block_DK + BS * block_DV) * 4,  # k1b: 5 fragments (fp32)
        (2 * BS * BS + BS * block_DK) * 4,  # fused: tmp_frag+ref_frag+dk_frag (fp32)
    )
    assert l0a <= 64 * 1024, f"L0A budget exceeded: {l0a}B > 64KB (BS={BS}, block_DK={block_DK}, block_DV={block_DV})"
    assert l0b <= 64 * 1024, f"L0B budget exceeded: {l0b}B > 64KB (BS={BS}, block_DK={block_DK}, block_DV={block_DV})"
    assert l0c <= 128 * 1024, f"L0C budget exceeded: {l0c}B > 128KB (BS={BS}, block_DK={block_DK}, block_DV={block_DV})"


def _run_kernel_pipeline(K, V, Beta, G, A, dw, du, B, S, H, DK, DV, chunk_size, block_DK=64, block_DV=64, cache=None):
    BS = chunk_size
    BH = B * H

    # OPT-IN memoization of the input-derived precompute products. `cache`
    # is a dict the CALLER owns and passes back on every call. A product is
    # reused only when every source tensor is the SAME OBJECT with an
    # unchanged `_version` (in-place mutation guard) and the derived shape
    # parameters match. Contract: repeated calls with IDENTICAL inputs skip
    # the device precompute (an amortized-cost metric, not the per-new-input
    # cost of a training loop); the cache holds STRONG references
    # (memory-for-speed); not thread-safe. Outputs and intermediates are
    # never cached.
    def _cache_get(key, srcs):
        if cache is None:
            return None
        ent = cache.get(key)
        if ent is None:
            return None
        stored_srcs, stored_vers, products = ent
        if len(stored_srcs) != len(srcs):
            return None
        for ss, sv, s in zip(stored_srcs, stored_vers, srcs):
            if ss is not s or sv != s._version:
                return None
        return products

    def _cache_put(key, srcs, products):
        if cache is None:
            return
        cache[key] = (tuple(srcs), tuple(s._version for s in srcs), products)

    # Widen Cube-kernel K/V tiles to a single 128-wide tile when the dim
    # allows it (block loop collapses to 1 iteration); shapes not divisible
    # by 128 keep the caller's blocks. The dv segment keeps _vec_block sizing.
    if DK % 128 == 0:
        block_DK = 128
    if DV % 128 == 0:
        block_DV = 128
    dev = K.device
    # Host-side validation (see _validate_inputs), AFTER the widening so
    # the budget asserts see the actual blocks.
    _validate_inputs(K, V, Beta, G, A, dw, du, B, S, H, DK, DV, chunk_size, block_DK, block_DV)
    # Dynamic core count with fallback (see _get_core_num).
    core_num = _get_core_num(dev)
    # k1b outputs (cross-launch GM boundary; consumed by the fused kernel)
    dA_k_raw = torch.empty(BH, S, BS, dtype=torch.float32, device=dev)
    dA_v_raw = torch.empty(BH, S, BS, dtype=torch.float32, device=dev)
    dk_beta_g = torch.empty(BH, S, DK, dtype=torch.float32, device=dev)
    dv_beta = torch.empty(BH, S, DV, dtype=torch.float32, device=dev)
    A_frag_raw = torch.empty(BH, S, BS, dtype=torch.float32, device=dev)
    dv = torch.empty(BH, S, DV, dtype=torch.float32, device=dev)
    # dA_tmp_T: the fused kernel's S4 C->C GM relay buffer (internal, not an output)
    dA_tmp_T = torch.empty(BH, S, BS, dtype=torch.float32, device=dev)
    dA_final_fp32 = torch.empty(BH, S, BS, dtype=torch.float32, device=dev)
    dk_final = torch.empty(BH, S, DK, dtype=torch.float32, device=dev)
    dbeta_final = torch.empty(BH, S, dtype=torch.float32, device=dev)
    dg_final = torch.empty(BH, S, dtype=torch.float32, device=dev)
    dbeta_k = torch.empty(BH, S, dtype=torch.float32, device=dev)
    dg_A_positive = torch.empty(BH, S, BS, dtype=torch.float32, device=dev)
    dg_A_negative = torch.empty(BH, S, BS, dtype=torch.float32, device=dev)
    # masks reduced to the two [BS, BS] constants (lower_tri for k2a_pre,
    # sign_flip for k2b's in-kernel gate — the combined gate is never
    # materialized, k2b recomputes it bit-identically). Memoized under
    # ("mask", BS): BS keys the [BS, BS] shape; same-tensor identity/
    # version still guards in-place mutation.
    mask_products = _cache_get(("mask", BS), (G,))
    if mask_products is not None:
        lower_tri_mask, sign_flip = mask_products
    else:
        if G.device.type == "npu":
            lower_tri_mask, sign_flip = compute_masks_npu(G, BS)
        else:
            lower_tri_mask, sign_flip = compute_masks(G, BS)
            lower_tri_mask = lower_tri_mask.to(dev)
            sign_flip = sign_flip.to(dev)
        _cache_put(("mask", BS), (G,), (lower_tri_mask, sign_flip))
    # No intermediate torch.npu.synchronize(): everything below is enqueued
    # on the current stream in program order, so cross-kernel GM data
    # dependencies are guaranteed by stream ordering. Callers needing
    # completion synchronize themselves.
    k1b_mod = k1b_gemm(BH, S, DK, DV, BS, block_DK, block_DV, core_num)
    k1b_mod(
        dw,
        K,
        A,
        du,
        V,
        dA_k_raw,
        dA_v_raw,
        dk_beta_g,
        dv_beta,
        A_frag_raw,  # GEMM5 (K@K^T) output
    )
    # The whole backward tail (k1c + k2a_pre + refine + k2b + k2c + k2d)
    # runs in ONE fused CV launch on BOTH paths — intermediates never touch
    # GM; K_T comes from the in-kernel rider (no host prepack). Guard
    # shapes (nc even, BS<=64, DK==block_DK, DV<=128) take the 2-slot
    # hand-unrolled fast path; every other shape the 1-chunk universal.
    num_chunks = S // BS
    fuse_2s = num_chunks % 2 == 0 and BS <= 64 and BS % 8 == 0 and block_DK == DK and DV <= 128 and block_DK % 8 == 0
    if fuse_2s:
        fused_mod = k1cv_k2cd_fused_2s(BH, S, DK, DV, BS, num_chunks, block_DK)
    else:
        fused_mod = k1cv_k2cd_fused(BH, S, DK, DV, BS, block_DK, _vec_block(DV))
    fused_mod(
        dA_k_raw,
        dA_v_raw,
        dk_beta_g,
        dv_beta,
        A_frag_raw,
        Beta,
        G,
        A,
        K,
        lower_tri_mask,
        sign_flip,
        dA_tmp_T,
        dA_final_fp32,
        dv,
        dk_final,
        dbeta_final,
        dg_final,
        dbeta_k,
        dg_A_positive,
        dg_A_negative,
    )
    return (
        dA_final_fp32,
        dk_final,
        dv,
        dbeta_final,
        dg_final,
        dbeta_k,
        dg_A_positive,
        dg_A_negative,
    )


if __name__ == "__main__":
    import sys

    # Repeated direct runs hit the JIT disk cache (cold-compile verification
    # lives in test_wy_fast_bwd_split.py, which calls disable_cache()).
    torch.manual_seed(0)
    B, S, H, DK, DV, chunk_size = 1, 64, 1, 64, 64, 64
    block_DK, block_DV = 64, 64
    print(f"Config: B={B} S={S} H={H} DK={DK} DV={DV} chunk_size={chunk_size}")
    K, V, Beta, G, A, dw, du = _prepare(B, S, H, DK, DV, chunk_size, seed=0)
    print("Computing golden...")
    (g_dA, g_dk, g_dv, g_dbeta, g_dg, g_dbeta_k, g_dg_A_pos, g_dg_A_neg) = golden_wy_fast_bwd_split(
        K, V, Beta, G, A, dw, du, chunk_size, block_DK, block_DV
    )
    K_n, V_n = K.to("npu"), V.to("npu")
    Beta_n, G_n = Beta.to("npu"), G.to("npu")
    A_n, dw_n, du_n = A.to("npu"), dw.to("npu"), du.to("npu")
    print("Running kernel pipeline (2-kernel fused path)...")
    (dA, dk, dv, dbeta, dg, dbeta_k, dg_A_pos, dg_A_neg) = _run_kernel_pipeline(
        K_n, V_n, Beta_n, G_n, A_n, dw_n, du_n, B, S, H, DK, DV, chunk_size, block_DK, block_DV
    )
    print("Checking precision...")
    outputs = [
        ("dA", dA, g_dA, "float32"),
        ("dk", dk, g_dk, "float32"),
        ("dv", dv, g_dv, "float32"),
        ("dbeta", dbeta, g_dbeta, "float32"),
        ("dg", dg, g_dg, "float32"),
        ("dbeta_k", dbeta_k, g_dbeta_k, "float32"),
        ("dg_A_positive", dg_A_pos, g_dg_A_pos, "float32"),
        ("dg_A_negative", dg_A_neg, g_dg_A_neg, "float32"),
    ]
    all_pass = True
    for name, actual, golden, dtype_str in outputs:
        passed, ratio, max_err = check_precision(actual, golden, dtype_str)
        tag = "[PRECISION_PASS]" if passed else "[PRECISION_FAIL]"
        print(f"  {tag} {name}: matched_ratio={ratio:.6f} max_abs_error={max_err:.6e}")
        if not passed:
            all_pass = False
    if all_pass:
        print("Test Passed!")
    else:
        print("Test FAILED - precision check failed")
        sys.exit(1)
