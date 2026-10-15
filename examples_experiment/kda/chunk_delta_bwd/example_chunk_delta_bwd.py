"""chunk_delta_bwd (KDA variant): gated delta rule backward, dhu path.

Per (b, h) and DV-block, an fp32 state [DK, block_DV] resides on-chip (UB)
and is updated by a reverse chunk recurrence (i = BS-1..0, BS = S // C,
C = chunk_size = 64):

    state     = dht[b, h][:, vsl]                        (or zeros)
    dh[:, i]  = bf16(state)                              # truncation point 1
    dv2[:, i] = bf16(K_i @ bf16(state) + dv_i)           # truncation points 2 & 3
    state     = state * exp2(GK_last_i)                  # per-DK-dim gate (fp32)
    state    += (Q_i^T @ dO_i) * scale - W_i^T @ dv2_i   # truncation point 4
    dh0[b, h] = state                                    (fp32 direct store)

Numeric contract, four truncation points:
  1. dh is stored as bf16, and the same bf16 value feeds the dv GEMM;
  2. the dv GEMM state operand is the truncated bf16 value;
  3. dv2 = bf16(fp32 add), a single truncation whose result also feeds
     the W GEMM;
  4. the state recurrence stays fp32 throughout (gate multiply, GEMM
     accumulation, add); only dh0 is stored fp32 at the end.
exp2 is emulated as exp(x * ln2) (Ascend has no exp2 intrinsic).

Kernel structure: a single fused kernel in Developer mode (4-True
pass_configs, no manual sync primitives). GEMM operands are staged
explicitly through the memory hierarchy (GM -> L1 -> L0A/L0B, one L0C
accumulator per GEMM), and the fp32 state is loop-carried in UB - it
never round-trips through GM.

Threads: 1. The 2-thread CV split misaligns the V-side readback of dv2
rows (framework limitation); single-thread execution is bit-exact on
dh/dv2 and within fp32 GEMM summation-order noise on dh0.

Semantics mirror the gated-delta-rule backward reference
implementation (chunk_gated_delta_rule_bwd_dhu, per-dim gk gate, exp2).
"""

import torch
import torch.nn.functional as F

import tilelang
from tilelang import language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

LN2 = 0.6931471805599453  # exp2(x) == exp(x * ln2)


@tilelang.jit(out_idx=[7, 8, 9], pass_configs=pass_configs)
def chunk_delta_bwd(
    B,
    S,
    H,
    DK,
    DV,
    chunk_size,
    scale,
    use_gk=True,
    use_initial_state=True,
    use_final_state_gradient=True,
    block_DV=128,
    threads=1,
    num_stages=1,
):
    """Build the chunk_delta_bwd kernel (JIT-compiled per shape config).

    Returns a kernel taking (Q, K, W, GK, dht, dO, dv) and producing
    (dh, dh0, dv2); outputs are auto-allocated via out_idx=[7, 8, 9]
    (explicit positive indices).
    """
    C = chunk_size
    BS = S // C
    bv_num = (DV + block_DV - 1) // block_DV

    @T.prim_func
    def kernel(
        Q: T.Tensor((B, S, H, DK), "bfloat16"),  # type: ignore
        K: T.Tensor((B, S, H, DK), "bfloat16"),  # type: ignore
        W: T.Tensor((B, S, H, DK), "bfloat16"),  # type: ignore
        GK: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
        dht: T.Tensor((B, H, DK, DV), "bfloat16"),  # type: ignore
        dO: T.Tensor((B, S, H, DV), "bfloat16"),  # type: ignore
        dv: T.Tensor((B, S, H, DV), "bfloat16"),  # type: ignore
        # outputs
        dh: T.Tensor((B, BS, H, DK, DV), "bfloat16"),  # type: ignore
        dh0: T.Tensor((B, H, DK, DV), "float32"),  # type: ignore
        dv2: T.Tensor((B, S, H, DV), "bfloat16"),  # type: ignore
    ):
        with T.Kernel(B * H * bv_num, threads=threads, is_npu=True) as (cid):
            bv = cid % bv_num
            bbh = cid // bv_num
            bh = bbh % H
            bb = bbh // H
            v_lo = bv * block_DV
            v_hi = (bv + 1) * block_DV

            # ---- on-chip buffers ----
            # Vector side (UB): resident fp32 state and elementwise staging.
            state_f32 = T.alloc_ub((DK, block_DV), "float32")
            state_bf16 = T.alloc_ub((DK, block_DV), "bfloat16")
            # gemm1/gemm2 use separate UB staging buffers: relays are routed
            # by source-buffer name; disjoint lifetimes let the planner reuse one slot.
            gemm1_ub = T.alloc_ub((DK, block_DV), "float32")
            gemm2_ub = T.alloc_ub((DK, block_DV), "float32")
            dv_f_ub = T.alloc_ub((C, block_DV), "float32")
            dv_in_ub = T.alloc_ub((C, block_DV), "float32")
            dv2_ub = T.alloc_ub((C, block_DV), "bfloat16")
            gk_last_ub = T.alloc_ub((DK,), "float32")
            gate_ub = T.alloc_ub((DK,), "float32")
            # brcb scratch for the fused row-broadcast multiply: one 32B
            # block (8 fp32 lanes) per state row.
            gate_tmp = T.alloc_ub((DK, 8), "float32")
            # Cube side L1: GM->L1 staging for the GEMM operands (MTE2).
            K_l1 = T.alloc_L1((C, DK), "bfloat16")
            Q_l1 = T.alloc_L1((C, DK), "bfloat16")
            W_l1 = T.alloc_L1((C, DK), "bfloat16")
            dO_l1 = T.alloc_L1((C, block_DV), "bfloat16")
            state_bf16_l1 = T.alloc_L1((DK, block_DV), "bfloat16")
            dv2_l1 = T.alloc_L1((C, block_DV), "bfloat16")
            # Cube side L0A/L0B: named per-GEMM staging - a shared L0 pool
            # would impose name-granularity false dependencies between the
            # three GEMMs; transposed-GEMM A-operands stage through a
            # transposed L1->L0A copy. L0 budget: L0A 48KB, L0B 64KB (exact).
            k_l0a = T.alloc_L0A((C, DK), "bfloat16")
            q_l0a = T.alloc_L0A((DK, C), "bfloat16")
            w_l0a = T.alloc_L0A((DK, C), "bfloat16")
            s_l0b = T.alloc_L0B((DK, block_DV), "bfloat16")
            do_l0b = T.alloc_L0B((C, block_DV), "bfloat16")
            dv2_l0b = T.alloc_L0B((C, block_DV), "bfloat16")
            # L0C accumulators: relay-by-name as above; after liveness
            # reuse the L0C peak is 96KB <= 128KB.
            dv_l0c = T.alloc_L0C((C, block_DV), "float32")
            g1_l0c = T.alloc_L0C((DK, block_DV), "float32")
            g2_l0c = T.alloc_L0C((DK, block_DV), "float32")

            # ---- state init (fp32; bf16 -> fp32 upcast is exact) ----
            # GM->UB copies must be same-dtype (copy_gm_to_ub template is
            # single-dtype); the cast to fp32 happens inside UB.
            if use_final_state_gradient:
                T.copy(dht[bb, bh, 0:DK, v_lo:v_hi], state_bf16)
                T.copy(state_bf16, state_f32)
            else:
                T.tile.fill(state_f32, 0.0)

            # ---- reverse chunk scan (state is loop-carried in UB) ----
            for i_s in T.Pipelined(BS, num_stages=num_stages):
                i = BS - 1 - i_s  # reverse chunk index (affine in i_s)
                s = i * C
                # wrapper guarantees S % C == 0; the golden's min() is redundant here.
                last_idx = (i + 1) * C - 1

                # (1) truncate state to bf16 (tp1); the same bf16 value
                #     is staged to L1 as the dv GEMM operand (tp2).
                T.copy(state_f32, state_bf16)
                T.copy(state_bf16, state_bf16_l1)

                # (2) dv GEMM: K_i @ bf16(state), fp32 accumulate (tp2).
                #     Explicit L0 staging + T.mma (full conversion; T.mma
                #     and gemm_v0 cannot be mixed in sequential
                #     statements).
                T.copy(K[bb, s : s + C, bh, 0:DK], K_l1)
                T.copy(K_l1, k_l0a)
                T.copy(state_bf16_l1, s_l0b)
                T.mma(k_l0a, s_l0b, dv_l0c, init=True)

                # (3) dv2 = bf16(dv_f + dv_i): fp32 add, single truncation
                #     (tp3); the truncated value feeds the W GEMM (tp4).
                T.copy(dv[bb, s : s + C, bh, v_lo:v_hi], dv2_ub)
                T.copy(dv2_ub, dv_in_ub)
                T.copy(dv_l0c, dv_f_ub)
                T.tile.add(dv_f_ub, dv_f_ub, dv_in_ub)
                T.copy(dv_f_ub, dv2_ub)
                T.copy(dv2_ub, dv2_l1)

                # (5)(6) per-DK-dim gating: state *= exp2(GK_last) (fp32).
                # row-expand needs a 256-byte last dim (64 fp32 cols) and a
                # static row count in 8..248 divisible by 8 (DK in {64, 128}
                # qualifies); block_DV == 32 (128B rows) falls back to the
                # scalar T.Parallel path, which serializes the Vector core.
                if use_gk:
                    T.copy(GK[bb, last_idx, bh, 0:DK], gk_last_ub)
                    T.tile.mul(gk_last_ub, gk_last_ub, LN2)
                    T.tile.exp(gate_ub, gk_last_ub)
                    if block_DV % 64 == 0:
                        for w in range(block_DV // 64):
                            T.tile.row_expand_mul_experiment(
                                state_f32[:, w * 64 : (w + 1) * 64],
                                state_f32[:, w * 64 : (w + 1) * 64],
                                gate_ub,
                                gate_tmp,
                            )
                    else:
                        for i_k, i_v in T.Parallel(DK, block_DV):
                            state_f32[i_k, i_v] = state_f32[i_k, i_v] * gate_ub[i_k]

                # (7) gemm1 = Q_i^T @ dO_i; state += scale * gemm1 (fp32).
                T.copy(Q[bb, s : s + C, bh, 0:DK], Q_l1)
                T.copy(dO[bb, s : s + C, bh, v_lo:v_hi], dO_l1)
                T.copy(Q_l1, q_l0a, transpose=True)
                T.copy(dO_l1, do_l0b)
                T.mma(q_l0a, do_l0b, g1_l0c, init=True)
                T.copy(g1_l0c, gemm1_ub)
                T.tile.axpy(state_f32, gemm1_ub, scale)

                # (8) gemm2 = W_i^T @ bf16(dv2); state -= gemm2 (fp32).
                T.copy(W[bb, s : s + C, bh, 0:DK], W_l1)
                T.copy(W_l1, w_l0a, transpose=True)
                T.copy(dv2_l1, dv2_l0b)
                T.mma(w_l0a, dv2_l0b, g2_l0c, init=True)
                T.copy(g2_l0c, gemm2_ub)
                T.tile.sub(state_f32, state_f32, gemm2_ub)

                # (9) tail store burst: the dh (tp1) and dv2 (tp3) GM writes
                #     fire once at the iteration tail - batched stores
                #     avoid mid-phase MTE3 blocking of the Vector pipe and
                #     overlap with the next iteration's head loads. Same
                #     values, same truncation points (state_bf16/dv2_ub
                #     lifetimes extend to here).
                T.copy(state_bf16, dh[bb, i, bh, 0:DK, v_lo:v_hi])
                T.copy(dv2_ub, dv2[bb, s : s + C, bh, v_lo:v_hi])

            # (10) dh0: fp32 direct store (no further truncation).
            if use_initial_state:
                T.copy(state_f32, dh0[bb, bh, 0:DK, v_lo:v_hi])

    return kernel


# ---------------------------------------------------------------------------
# Reference test-data generation (GPU-reference-identical; CPU randomness, single H2D).
# ---------------------------------------------------------------------------
def prepare_input(B, S, H, DK, DV, chunk_size, seed=42):
    """Reference prep (CPU): W=randn unnormalized, Q=randn*0.01,
    K=F.normalize(dim=-1, p=2), GK=logsigmoid(randn fp32) + chunk-local
    cumsum along S, dht=randn, dO=randn*0.01, dv=randn."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    Q = (torch.randn(B, S, H, DK, dtype=torch.float32, generator=gen) * 0.01).to(torch.bfloat16)
    K = F.normalize(torch.randn(B, S, H, DK, dtype=torch.float32, generator=gen), dim=-1, p=2).to(torch.bfloat16)
    W = torch.randn(B, S, H, DK, dtype=torch.float32, generator=gen).to(torch.bfloat16)
    G = torch.randn(B, S, H, DK, dtype=torch.float32, generator=gen)
    G = F.logsigmoid(G)
    # chunk-local cumsum along S (KDA gk semantics: gate at chunk last row
    # holds the full-chunk log decay)
    G = G.view(B, S // chunk_size, chunk_size, H, DK).cumsum(dim=2).view(B, S, H, DK)
    dht = torch.randn(B, H, DK, DV, dtype=torch.float32, generator=gen).to(torch.bfloat16)
    dO = (torch.randn(B, S, H, DV, dtype=torch.float32, generator=gen) * 0.01).to(torch.bfloat16)
    dv = torch.randn(B, S, H, DV, dtype=torch.float32, generator=gen).to(torch.bfloat16)
    return Q, K, W, G, dht, dO, dv


# ---------------------------------------------------------------------------
# Host wrapper: support-domain validation + kernel launch (pure metadata path).
# ---------------------------------------------------------------------------
_KERNEL_CACHE = {}


def _select_block_dv(DV):
    for bd in (128, 64, 32):
        if DV % bd == 0:
            return bd
    raise ValueError(
        f"unsupported DV={DV}: no block_DV in (128, 64, 32) divides DV; supported DV are multiples of 16 in [16, 128] divisible by 32"
    )


def run_chunk_delta_bwd(
    Q,
    K,
    W,
    GK,
    dht,
    dO,
    dv,
    chunk_size=64,
    scale=None,
    use_gk=True,
    use_initial_state=True,
    use_final_state_gradient=True,
    block_DV=None,
    threads=1,
    num_stages=1,
):
    """Validate shapes/dtypes/support domain/device/contiguity, launch the
    kernel, return (dh, dh0, dv2)."""
    if Q.dim() != 4:
        raise ValueError(f"Q must be 4D (B, S, H, DK), got {tuple(Q.shape)}")
    B, S, H, DK = Q.shape
    DV = dO.shape[-1]

    if K.shape != Q.shape:
        raise ValueError(f"K.shape {tuple(K.shape)} != Q.shape {tuple(Q.shape)}")
    if W.shape != Q.shape:
        raise ValueError(f"W.shape {tuple(W.shape)} != Q.shape {tuple(Q.shape)}")
    if GK.shape != Q.shape:
        raise ValueError(f"GK.shape {tuple(GK.shape)} != Q.shape {tuple(Q.shape)}")
    if dO.shape != (B, S, H, DV):
        raise ValueError(f"dO.shape {tuple(dO.shape)} != {(B, S, H, DV)}")
    if dv.shape != (B, S, H, DV):
        raise ValueError(f"dv.shape {tuple(dv.shape)} != {(B, S, H, DV)}")
    if dht.shape != (B, H, DK, DV):
        raise ValueError(f"dht.shape {tuple(dht.shape)} != {(B, H, DK, DV)}")

    expected_dtypes = {
        "Q": (Q, torch.bfloat16),
        "K": (K, torch.bfloat16),
        "W": (W, torch.bfloat16),
        "GK": (GK, torch.float32),
        "dht": (dht, torch.bfloat16),
        "dO": (dO, torch.bfloat16),
        "dv": (dv, torch.bfloat16),
    }
    for name, (t, dt) in expected_dtypes.items():
        if t.dtype != dt:
            raise ValueError(f"{name} must be {dt}, got {t.dtype}")

    if B < 1 or H < 1:
        raise ValueError(f"require B>=1 and H>=1, got B={B}, H={H}")
    if chunk_size != 64:
        raise ValueError(f"unsupported chunk_size={chunk_size} (must be 64)")
    if S % chunk_size != 0 or chunk_size > S:
        raise ValueError(f"require S % {chunk_size} == 0 and S >= {chunk_size}, got S={S}")
    if DK not in (64, 128):
        raise ValueError(f"unsupported DK={DK} (must be 64 or 128)")
    if DV % 16 != 0 or not 1 <= DV <= 128:
        raise ValueError(f"unsupported DV={DV} (must be multiple of 16, in [16, 128])")

    if block_DV is None:
        block_DV = _select_block_dv(DV)
    elif DV % block_DV != 0:
        raise ValueError(f"DV={DV} not divisible by block_DV={block_DV}")

    # Tensor-level checks, after the metadata domain checks above: rejection
    # probes pass CPU tensors carrying illegal metadata and must hit the
    # domain ValueError first. The kernel addresses inputs row-major
    # directly; a non-contiguous tensor would be silently misread.
    for name, t in (
        ("Q", Q),
        ("K", K),
        ("W", W),
        ("GK", GK),
        ("dht", dht),
        ("dO", dO),
        ("dv", dv),
    ):
        if t.device.type != "npu":
            raise ValueError(f"{name} must be on npu, got {t.device}")
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    if scale is None:
        scale = DK**-0.5

    # num_stages clamp: BS < num_stages yields an extent-0 pipeline loop
    # that drops all compute; num_stages is clamped to <= BS (small-S shapes).
    num_stages = min(num_stages, S // chunk_size)

    key = (
        B,
        S,
        H,
        DK,
        DV,
        chunk_size,
        scale,
        use_gk,
        use_initial_state,
        use_final_state_gradient,
        block_DV,
        threads,
        num_stages,
    )
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = chunk_delta_bwd(
            B=B,
            S=S,
            H=H,
            DK=DK,
            DV=DV,
            chunk_size=chunk_size,
            scale=scale,
            use_gk=use_gk,
            use_initial_state=use_initial_state,
            use_final_state_gradient=use_final_state_gradient,
            block_DV=block_DV,
            threads=threads,
            num_stages=num_stages,
        )
        _KERNEL_CACHE[key] = kernel
    dh, dh0, dv2 = kernel(Q, K, W, GK, dht, dO, dv)
    if not use_initial_state:
        # Contract: the kernel skips the dh0 store when use_initial_state
        # is False; zero the out_idx-allocated buffer (torch.empty - dirty
        # memory) so no uninitialized data escapes to callers.
        dh0.zero_()
    return dh, dh0, dv2


# ---------------------------------------------------------------------------
# Smoke run: one shape, shapes/dtypes check; precision lives in test_chunk_delta_bwd.py.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B, S, H, DK, DV, C = 1, 128, 2, 128, 128, 64
    cpu_in = prepare_input(B, S, H, DK, DV, C, seed=42)
    npu_in = tuple(t.npu() for t in cpu_in)
    torch.npu.synchronize()

    dh, dh0, dv2 = run_chunk_delta_bwd(*npu_in, chunk_size=C)
    torch.npu.synchronize()

    assert dh.shape == (B, S // C, H, DK, DV), f"dh shape mismatch: {tuple(dh.shape)} != {(B, S // C, H, DK, DV)}"
    assert dh.dtype == torch.bfloat16, f"dh dtype mismatch: {dh.dtype}"
    assert dh0.shape == (B, H, DK, DV), f"dh0 shape mismatch: {tuple(dh0.shape)} != {(B, H, DK, DV)}"
    assert dh0.dtype == torch.float32, f"dh0 dtype mismatch: {dh0.dtype}"
    assert dv2.shape == (B, S, H, DV), f"dv2 shape mismatch: {tuple(dv2.shape)} != {(B, S, H, DV)}"
    assert dv2.dtype == torch.bfloat16, f"dv2 dtype mismatch: {dv2.dtype}"

    print("Test Passed!")
