"""Indexer topk ReduceSum — DSA lightning indexer forward (topk + reducesum,
varlen causal) for Ascend 910B3.

Semantics (per varlen segment, per query token s1; sm_scale = dim ** -0.5):

    logits[s1, h, s2] = einsum(q[s1, h, :], k[s2, :])
    score[s1, s2]     = (sum_h relu(logits) * w[s1, h])_{fp32} * sm_scale
    score[s1, s2]     = -inf  if s2 > s1                     # causal mask
    topk_vals, topk_idx = topk(score[s1, :], topk)           # descending
    topk_score        = softmax(topk_vals)_{fp32}

Outputs TopkIndices [S, topk] int32 (-1 at the -inf tail positions) and
ReduceSum (= topk_score) [S, topk] float32.

Architecture: dual path, exact (non-heuristic) host dispatch on max_seg_len.
  fast    max_seg_len <= 4096: persistent T.Kernel(grid) over QG-token
          groups; transposed GEMM [BN, D] x [D, QG*H] -> L0C fp32 [BN,
          QG*H] -> GM bf16 transit (relu rides the transit write) -> AIV
          relay -> weighted head reduce -> causal mask -> score_row
          [QG2, CAP] -> per-token hardware topk -> softmax -> index fixup.
  generic any segment length: same GEMM data flow; the score row is
          produced CH columns at a time, each CH segment runs a hardware
          topk whose local indices are corrected to the global column
          frame and merged into a per-token (val, idx) pool by a stable
          descending merge_sort — the pool after the last segment is the
          full-row topk.

Programming model: Developer mode — T.alloc_shared / T.alloc_fragment with
the automatic synchronization passes (TL_ASCEND_AUTO_CV_COMBINE,
TL_ASCEND_AUTO_CV_SYNC, TL_ASCEND_AUTO_SYNC, TL_ASCEND_MEMORY_PLANNING).
Launch form T.Kernel(grid, is_npu=True) as (cid, vid): MIX_AIC_1_2 (1 Cube
+ 2 Vector cores per block). The Cube side owns the Q/K loads, the GEMM and
the full transit write; each AIV-vid owns the contiguous token half
[vid*QG2, (vid+1)*QG2) of every group with private UB buffers, and every
AIV sub-block participates in every transit as the 1_2 cross-core
handshake requires.

Cross-core synchronization (auto CV sync + back-pressure):
  - The Cube side computes each K-block GEMM and writes the fp32->bf16
    (relu) transit tile to GM (``workspace_logits`` — the CV-sync pass
    pairs cross-core GM writes/reads by the literal buffer-name substring
    "workspace", so the name is load-bearing); each AIV relays its
    disjoint half of the tile to UB and consumes it.
  - A V->C ack transit (``workspace_ack``, one heartbeat slot per AIV)
    throttles the Cube side: the fast path runs produce-then-wait over a
    2-slot ping-pong (slot j % 2 is rewritten only after its ack has been
    waited one iteration earlier, so at most one transit is in flight
    beyond the Vector side; the trailing +1 pipeline iteration drains the
    task's last ack); the generic path (not perf-gated) uses a serialized
    single-slot wait-before-produce form. Flag credit is exactly balanced
    per task on both paths.

Implementation notes (each item is a platform constraint):
  1. relu rides the L0C->GM transit leg (enable_relu on the fp32->bf16
     write): GM->UB copies are single-dtype, so a "GM bf16 -> UB fp32 +
     relu" leg is not expressible. Numerically identical — relu commutes
     with the bf16 rounding (relu(x) >= 0 and bf16 preserves sign/zero).
  2. The per-head product is rounded to bf16 before the fp32 head reduce
     (prod = bf16(fp32(relu_logits) * fp32(w)); score = sum_h(prod) *
     sm_scale), replicating the reference's bf16 product round point. The
     scale multiplies the [H, BN] rounded products (>= 256B, satisfying
     the vector-op minimum-size rule); the fast path instead applies a
     positive scale once per token on the [CAP] score row after the fp32
     reduce (the reference's exact "fp32 sum then scale" order — a
     positive scale preserves the descending topk order and the -inf tail
     signs).
  3. Causal masking granularity is per path: the fast path masks
     post-reduce [QG2, BN] tiles on straddle blocks only (blocks fully
     below the group's first-token diagonal skip the compare — every row
     is fully valid there); the generic path masks pre-reduce [H, BN]
     tiles; the pto fast branch masks the final [CAP] score row.
  4. T.tile.topk takes a Buffer (not a BufferRegion row), so topk runs on
     a 1D scratch; the softmax / idx-fixup chain uses TOPK_A =
     max(topk, 64) wide buffers with -inf / -1.0 pad fills (compare needs
     >= 64 fp32 elements; pads are never written out).
  5. tl::ascend::GatherMask quantises its source to 256B units (64 fp32),
     so all gather src/dst buffers are sized to the quantum (P2 =
     max(2*topk, 64), PS = max(topk, 32)) and pre-filled with -inf.
  6. IndexQ / Weights are declared flat ([S*H, D] / [S*H]) and the cached
     launcher applies zero-copy host-side views (T.copy structural
     equality forbids the [S, H, D] row-segment form). Inputs must be
     contiguous: non-contiguous tensors are rejected before the view —
     an implicit reshape copy dies with the marshal frame, so its storage
     could be reused before the (cached) launch reads it.
  7. gemm_v0 is instantiated with a host-computed kL0Size (_pick_kl0_size):
     the default 128 overflows L0A for M = QG*heads = 512 — the A slot
     M * kL0Size * 2B must fit 64KB for a single (N, K) step or 32KB once
     the ping-pong alternates. The K-order fp32 accumulation is unchanged.
  8. pto target branches (the kernel body is shared between backends;
     the branches cover pto codegen constraints):
     (a) bulk Q/K/weights GM->L1 loads only — the pto codegen provides a
         runtime tail clamp + zero pad; out-of-segment rows are causally
         masked (cols >= ln) or belong to unconsumed tokens;
     (b) the consume chain uses half-token [BN, HALFC] wide-row tiles with
         a transposed staging buffer and an in-place add folding tree for
         the head reduce (the pto vector template needs >= 256B inner
         rows, pto tile views cannot address column slices of a
         [BN, QG2H] buffer, and the dim=-1 row-sum reduce is not
         available on the pto path);
     (c) the fast path adds a tail [CAP]-row causal mask after the scale
         (its block-level mask is skipped on this branch);
     (d) scalar T.max/T.min on vid-derived expressions use if_then_else
         ternaries — the CCE max()/min() overload set {(long,long),
         (int,int)} is ambiguous for the (int64, int) pair under -xcce;
         identical integer semantics.
     The pto fast path additionally routes CAP > 2048 to the generic path
     (the pto auto-topk temporary overflows the 192KB UB budget there).

Supported domain (host-validated): q/w/k bfloat16; heads % 8 == 0 and
<= 64; dim % 16 == 0; topk a power of two in [16, 1024] and <= every
segment length; the fast path additionally requires max_seg_len <= 4096;
the generic path covers any segment length (topk <= 1024, CH =
max(1024, topk)). Configurations whose exact UB model exceeds the 196352B
budget or whose Cube-side L1 staging (Q_l1 + K_l1 + ack) exceeds the
524032B L1 pool are rejected with ValueError (see _fast_gate_ok /
_generic_gate_ok); the host first demotes the group tier to fit (fast:
QG 8 -> 4; generic: 8 -> 4 -> 2).
"""

import ctypes

import numpy as np
import torch

import tilelang
from tilelang import language as T

try:  # fast raw-stream query (0.9 us vs 33 us for torch.npu.current_stream)
    import torch_npu  # noqa: F401

    _npu_raw_stream = torch_npu._C._npu_getCurrentRawStream
except Exception:  # pragma: no cover - non-NPU import context
    _npu_raw_stream = None

__all__ = [
    "tl_indexer_topk_reducesum_fast",
    "tl_indexer_topk_reducesum_generic",
    "indexer_topk_reducesum_interface",
    "prepare_token_indices",
    "prepare_group_table",
]

_pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

_UB_CAPACITY = 196352  # A2/A3 UB pool (codegen ASCEND_A2A3_UB_SIZE)
_L1_CAPACITY = 524032  # A2/A3 per-AI-Core L1 pool (codegen ASCEND_A2A3_L1_SIZE)
_L0C_CAPACITY = 128 * 1024
_BN = 32  # K-block width (GEMM N)
_ACK_PER_AIV = 16  # fp32 heartbeat elements per AIV slot (64B, 32B-aligned)
_ACK_TOTAL = 2 * _ACK_PER_AIV


class _CachedLauncher:
    """Low-overhead callable around a compiled tilelang kernel.

    The stock cython launch path re-marshals every argument on each call —
    constructing a fresh TVM ``Analyzer`` alone costs ~0.55 ms. This wrapper
    marshals the low-level ctypes argument vector directly (~tens of us per
    launch, the bare ACL submission cost).

    ``view_spec`` maps argument positions to zero-copy reshape modes for the
    flat-declared params (IndexQ as [S*H, D], Weights as [S*H]): raw
    caller-side tensors are viewed per call.

    Safety model: the single cache slot holds strong references to the exact
    argument objects of the last marshaled call, keyed on their ``id()``
    tuple. While the references are alive those ids cannot be recycled, so a
    key hit guarantees the cached data pointers / dynamic-shape values are
    still the right ones for these very tensors. Any miss (fresh tensors,
    changed arg count, explicit stream) re-marshals through the same cheap
    path; non-tensor args or a missing adapter fall back to the stock call.

    Inputs must be contiguous: the view spec may only produce zero-copy
    views of the caller's storage. A reshape of a non-contiguous tensor
    materializes a temporary copy that dies with the marshal frame (the
    cache slot retains only the original args, so the marshaled pointer
    would dangle once the storage is reused), and a retained copy would go
    stale on in-place mutation — non-contiguous tensors are therefore
    rejected before any view is applied.
    """

    __slots__ = (
        "_kernel",
        "_adapter",
        "_lib",
        "_params_len",
        "_key",
        "_args",
        "_args_refs",
        "_dev",
        "_view_spec",
    )

    def __init__(self, kernel, view_spec=None):
        self._kernel = kernel
        adapter = kernel.adapter
        self._adapter = adapter
        self._lib = adapter.lib
        # fast path only for plain all-tensor signatures (no auto-allocated
        # result/workspace buffers — our kernels pass everything explicitly)
        ok = not getattr(adapter, "result_idx", []) and not getattr(adapter, "workspace_idx", [])
        ok = ok and not getattr(adapter, "auto_gm_idx", None)
        self._params_len = len(adapter.params) if ok else -1
        self._key = None
        self._args = None
        self._dev = None
        self._view_spec = view_spec or {}

    def __getattr__(self, name):  # delegate get_kernel_source()/etc.
        return getattr(self._kernel, name)

    @staticmethod
    def _apply_view(t, mode):
        if mode == "flat2d":
            return t.reshape(-1, t.shape[-1])
        if mode == "flat1d":
            return t.reshape(-1)
        return t

    def _marshal(self, args):
        ad = self._adapter
        # Contiguity is checked on the caller-side tensors BEFORE the view
        # spec: a reshape of a non-contiguous input materializes a temporary
        # copy whose storage dies with this frame, while the cache slot (and
        # this launch) only see the raw pointer — non-contiguous inputs are
        # rejected instead (call .contiguous() at the source).
        for t in args:
            if isinstance(t, torch.Tensor) and not t.is_contiguous():
                raise ValueError("kernel tensors must be contiguous")
        tensors = [self._apply_view(t, self._view_spec.get(i, "")) if isinstance(t, torch.Tensor) else t for i, t in enumerate(args)]
        c_args = []
        for t in tensors:
            if not isinstance(t, torch.Tensor):
                return None
            c_args.append(ctypes.c_void_p(t.data_ptr()))
        for _, (bidx, sidx) in ad.dynamic_symbolic_map.items():
            c_args.append(ctypes.c_int64(int(tensors[bidx].shape[sidx])))
        c_args.append(None)  # stream slot (filled per call)
        return c_args

    def __call__(self, *args, stream=-1):
        plen = self._params_len
        if plen == len(args) and (stream == -1 or isinstance(stream, int)):
            key = tuple(map(id, args))
            if key == self._key:
                dev = self._dev
                if stream == -1:
                    stream = _npu_raw_stream(dev) if _npu_raw_stream is not None else torch.npu.current_stream().npu_stream
                c_args = self._args
                c_args[-1] = ctypes.c_void_p(stream)
                self._lib.call(*c_args)
                return
            c_args = self._marshal(args)
            if c_args is not None:
                dev = args[0].get_device()
                if stream == -1:
                    stream = _npu_raw_stream(dev) if _npu_raw_stream is not None else torch.npu.current_stream().npu_stream
                c_args[-1] = ctypes.c_void_p(stream)
                # commit slot (strong refs via _args_refs keep ids and views alive)
                self._key = key
                self._args = c_args
                self._args_refs = args
                self._dev = dev
                self._lib.call(*c_args)
                return
        # full validating fallback (also applies the view spec; same
        # pre-view contiguity contract as _marshal)
        for t in args:
            if isinstance(t, torch.Tensor) and not t.is_contiguous():
                raise ValueError("kernel tensors must be contiguous")
        args2 = tuple(self._apply_view(t, self._view_spec.get(i, "")) if isinstance(t, torch.Tensor) else t for i, t in enumerate(args))
        self._kernel(*args2)


# ============================================================================
# Fast kernel (persistent grid, score_row [QG, CAP] + single hardware topk)
# ============================================================================

# Per-target jit wrapper cache: one _JitImplementation per backend, each with
# its own (args, kwargs) kernel cache — the builder body below is shared
# verbatim between the backends (same TIR, different codegen path:
# tilelang/jit/adapter/libgen.py forks bisheng -xasc vs -xcce).
_JIT_WRAPPERS: dict = {}


def _jit_wrapper(target):
    w = _JIT_WRAPPERS.get(target)
    if w is None:
        if target not in ("ascendc", "pto"):
            raise ValueError(f"target must be 'ascendc' or 'pto', got {target!r}")
        w = tilelang.jit(pass_configs=_pass_configs, target=target)
        _JIT_WRAPPERS[target] = w
    return w


def tl_indexer_topk_reducesum_fast(
    heads,
    dim,
    topk,
    CAP,
    QG=8,
    BN=32,
    grid_size=20,
    sm_scale=None,
    KL0=128,
    target="ascendc",
):
    """Build the fast-path kernel (Developer mode, MIX_AIC_1_2 persistent grid).

    Args:
        heads: number of heads H (% 8 == 0 and <= 64).
        dim: head dim D (% 16 == 0).
        topk: selected positions (power of two, 16 <= topk <= CAP).
        CAP: score_row width, power of two in {512, 1024, 2048, 4096};
            the host guarantees CAP >= max segment length (dispatch contract).
        QG: tokens per group (GEMM M = QG*heads; host tiering picks 8
            whenever CAP <= 2048 and the UB/L1 budgets fit, else 4).
            Each AIV-vid of the group's block owns the contiguous token half
            [vid*QG//2, (vid+1)*QG//2).
        BN: K-block width (GEMM N).
        grid_size: persistent grid (compile-time constant; 910B3 = 20).
        sm_scale: softmax scale, defaults to dim ** -0.5.
        KL0: gemm_v0 L1->L0 K-axis tile (host-computed by _pick_kl0_size so
            the A operand fits its L0A ping-pong slot).
        target: codegen backend, "ascendc" (default) or "pto". The kernel
            body is shared; only the jit wrapper (bisheng -xasc vs -xcce
            codegen fork) differs.
    """
    return _jit_wrapper(target)(_tl_indexer_topk_reducesum_fast_main)(
        heads=heads,
        dim=dim,
        topk=topk,
        CAP=CAP,
        QG=QG,
        BN=BN,
        grid_size=grid_size,
        sm_scale=sm_scale,
        KL0=KL0,
        target=target,
    )


def _tl_indexer_topk_reducesum_fast_main(
    heads,
    dim,
    topk,
    CAP,
    QG=8,
    BN=32,
    grid_size=20,
    sm_scale=None,
    KL0=128,
    target="ascendc",
):
    """(private) fast-path prim_func builder — see the public wrapper above."""
    assert heads % 8 == 0 and heads <= 64, "heads must be % 8 == 0 and <= 64"
    assert dim % 16 == 0, "dim must be a multiple of 16"
    assert topk == tilelang.math.next_power_of_2(topk) and topk >= 16, f"topk must be a power of two >= 16, got {topk}"
    assert CAP in (512, 1024, 2048, 4096), f"CAP must be pow2 in [512,4096], got {CAP}"
    assert CAP % BN == 0, "CAP must be a multiple of BN"
    assert topk <= CAP, "topk must be <= CAP"
    assert QG % 4 == 0, "QG must be a multiple of 4 (AIV-vid split + half-token chain fusion)"
    assert QG * heads % 16 == 0, "GEMM M must be 16-aligned"
    assert KL0 % 16 == 0 and KL0 <= 4095, "KL0 must be % 16 == 0 and <= 4095"
    if sm_scale is None:
        sm_scale = dim**-0.5
    # A positive scale preserves the descending topk order and the -inf tail
    # signs, so it can be applied once per token row after the fp32 reduce
    # (the reference's exact "fp32 sum then scale" order); non-positive
    # scales keep the per-block product multiply.
    scale_after_reduce = isinstance(sm_scale, (int, float)) and sm_scale > 0

    NBLK = CAP // BN
    TOPK_A = max(topk, 64)  # softmax/idx chain width (compare needs >= 64 fp32)
    P2 = max(2 * topk, 64)  # gather src width: 256B (64-fp32) quantum, -inf padded
    QG2 = QG // 2  # tokens per AIV-vid (contiguous half of the group)
    HALFC = QG2 * heads // 2  # half-token chain width (pto branch)
    HTOK = QG2 // 2  # tokens per pto half-chain

    seq_len = T.symbolic("seq_len")  # total packed length (sum of segment lens)
    batch_plus_one = T.symbolic("batch_plus_one")  # Offsets dim
    num_groups = T.symbolic("num_groups")  # GroupTable rows (persistent tasks)
    q_rows = T.symbolic("q_rows")  # = seq_len * heads (flat-2D IndexQ view)
    w_rows = T.symbolic("w_rows")  # = seq_len * heads (flat-1D Weights view)

    dtype = "bfloat16"
    accum_dtype = "float32"
    indices_dtype = "int32"

    @T.prim_func
    def main(
        IndexQ: T.Tensor([q_rows, dim], dtype),  # type: ignore  # flat [S*H, D]
        Weights: T.Tensor([w_rows], dtype),  # type: ignore  # flat [S*H]
        IndexK: T.Tensor([seq_len, dim], dtype),  # type: ignore
        TopkIndices: T.Tensor([seq_len, topk], indices_dtype),  # type: ignore
        ReduceSum: T.Tensor([seq_len, topk], accum_dtype),  # type: ignore
        Offsets: T.Tensor([batch_plus_one], indices_dtype),  # type: ignore
        TokenIndices: T.Tensor([seq_len, 2], indices_dtype),  # type: ignore  # interface fidelity, not consumed
        GroupTable: T.Tensor([num_groups, 3], indices_dtype),  # type: ignore
        workspace_logits: T.Tensor([grid_size, 2, BN, QG * heads], dtype),  # type: ignore  # C->V transit, 2-slot ping-pong
        workspace_ack: T.Tensor([grid_size, _ACK_TOTAL], accum_dtype),  # type: ignore  # V->C back-pressure
    ):
        with T.Kernel(grid_size, is_npu=True) as (cid, vid):  # MIX_AIC_1_2
            # ---- L1: Cube GEMM operands (alloc_shared maps to L1) ----
            Q_l1 = T.alloc_shared([QG * heads, dim], dtype)
            K_l1 = T.alloc_shared([BN, dim], dtype)
            # explicit 2D L1 (plain "shared" is use-inferred to UB, which
            # reclassifies the ack GM read as a vector op and breaks the CV
            # pairing; 1D L1 breaks the default zN layout injection)
            ack_l1 = T.alloc_shared([1, _ACK_TOTAL], accum_dtype, scope="shared.l1")
            # ---- L0C: GEMM accumulator [BN, QG*H] — the transposed
            #      orientation makes the transit rows 1KB/512B wide instead
            #      of 64B and the head reduce a contiguous dim=-1 form ----
            logits_l0c = T.alloc_fragment([BN, QG * heads], accum_dtype)
            # ---- UB (per AIV-vid): transit relay + score accumulation ----
            relay_ub = T.alloc_shared([BN, QG2 * heads], dtype)  # GM bf16 -> UB bf16 (this AIV's columns)
            score_row = T.alloc_shared([QG2, CAP], accum_dtype)
            w_src = T.alloc_shared([QG2 * heads], dtype)
            w_ub = T.alloc_shared([QG2 * heads], accum_dtype)  # fp32 weights
            ack_ub = T.alloc_shared([_ACK_PER_AIV], accum_dtype)  # ack heartbeat
            # ---- UB: fused full-tile [BN, QG2H] vector chain (ascendc) ----
            logits_f = T.alloc_shared([BN, QG2 * heads], accum_dtype)
            bc_f = T.alloc_shared([BN, QG2 * heads], accum_dtype)  # w broadcast
            prod_f = T.alloc_shared([BN, QG2 * heads], dtype)  # reference product round
            score_2d = T.alloc_shared([QG2, BN], accum_dtype)  # per-block QG2-token scores
            # ---- UB: fused post-reduce causal-validity mask (straddle
            #      blocks; [QG2, BN] tiles) ----
            t_abs_big = T.alloc_shared([64], accum_dtype)  # arith 256B-quantum safe
            colg2 = T.alloc_shared([QG2, BN], accum_dtype)  # col index per column
            tpart = T.alloc_shared([QG2, BN], accum_dtype)  # t_abs per row
            # compare dst: packed uint8 predicate (API contract, N/8 bytes)
            mask2 = T.alloc_shared([QG2, BN // 8], "uint8")
            col_1d = T.alloc_shared([BN], accum_dtype)
            # ---- pto branch: half-token wide-row chain buffers ----
            relay_ub_h = T.alloc_shared([BN, HALFC], dtype)
            logits_f_h = T.alloc_shared([BN, HALFC], accum_dtype)
            bc_f_h = T.alloc_shared([BN, HALFC], accum_dtype)
            prod_f_h = T.alloc_shared([BN, HALFC], dtype)  # reference product round
            seg_t = T.alloc_shared([HALFC, BN], accum_dtype)  # transposed staging
            # ---- pto branch: tail [CAP]-row causal mask buffers ----
            col_cap = T.alloc_shared([CAP], accum_dtype)
            valid_cap = T.alloc_shared([CAP // 8], "uint8")
            # ---- UB: per-token topk + softmax + fixup ----
            topk_src = T.alloc_shared([CAP], accum_dtype)  # topk needs a 1D Buffer
            topk_dst = T.alloc_shared([P2], accum_dtype)  # (val, idx) pairs + -inf pad
            val_ub = T.alloc_shared([TOPK_A], accum_dtype)
            idx_f_ub = T.alloc_shared([TOPK_A], accum_dtype)
            e_ub = T.alloc_shared([TOPK_A], accum_dtype)
            softmax_ub = T.alloc_shared([TOPK_A], accum_dtype)
            valid = T.alloc_shared([TOPK_A // 8], "uint8")  # packed compare dst
            idx_ub = T.alloc_shared([TOPK_A], indices_dtype)
            max_ub = T.alloc_shared([1], accum_dtype)
            sum_ub = T.alloc_shared([1], accum_dtype)

            ntask = T.max(0, T.ceildiv(num_groups - cid, grid_size))
            for k in T.serial(ntask):
                bx = k * grid_size + cid
                bos = GroupTable[bx, 0]
                t0 = GroupTable[bx, 1]
                ln = GroupTable[bx, 2]
                t_max = T.min(t0 + QG, ln) - 1  # group's max causally-valid column
                vh = vid * QG2  # this AIV's contiguous token offset

                # ---- Q load (Cube side, full QG rows) / weights load (V
                #      side, this AIV's half). pto target: bulk loads only —
                #      the pto codegen adds a runtime tail clamp + zero pad;
                #      out-of-segment rows are causally masked (cols >= ln)
                #      or belong to unconsumed tokens. ----
                if target == "pto":
                    T.copy(IndexQ[(bos + t0) * heads : (bos + t0 + QG) * heads, 0:dim], Q_l1)
                else:
                    if t0 + QG <= ln:
                        T.copy(IndexQ[(bos + t0) * heads : (bos + t0 + QG) * heads, 0:dim], Q_l1)
                    else:
                        for u in T.serial(QG):
                            if t0 + u < ln:
                                T.copy(
                                    IndexQ[(bos + t0 + u) * heads : (bos + t0 + u + 1) * heads, 0:dim],
                                    Q_l1[u * heads : (u + 1) * heads, 0:dim],
                                )
                if target == "pto":
                    T.copy(
                        Weights[(bos + t0 + vh) * heads : (bos + t0 + vh + QG2) * heads],
                        w_src,
                    )
                else:
                    if t0 + vh + QG2 <= ln:
                        T.copy(
                            Weights[(bos + t0 + vh) * heads : (bos + t0 + vh + QG2) * heads],
                            w_src,
                        )
                    else:
                        for u in T.serial(QG2):
                            if t0 + vh + u < ln:
                                T.copy(
                                    Weights[(bos + t0 + vh + u) * heads : (bos + t0 + vh + u + 1) * heads],
                                    w_src[u * heads : (u + 1) * heads],
                                )
                T.copy(w_src, w_ub)  # bf16 -> fp32 auto-cast (UB->UB); no sm_scale here
                # this AIV's per-token causal bounds [QG2] (rows of the fused
                # post-reduce mask; [64] buffer absorbs the arith 256B quantum)
                T.tile.arith_progression(t_abs_big, t0 + vh, 1, QG2)
                T.tile.fill(ack_ub, 0.0)  # heartbeat payload
                T.tile.fill(score_row, -T.infinity(accum_dtype))  # topk pad + stale-data guard

                # ---- K-block pipeline: NBLK+1 iterations. The produce block
                #      (j < NBLK) and the ack-wait block (j >= 1, ack of
                #      transit j-1) are separate statements — produce-then-
                #      wait keeps one slot of overlap; the +1 iteration
                #      drains the task's last ack at the task boundary. ----
                for j in T.serial(NBLK + 1):
                    k_st = j * BN
                    if j < NBLK:  # noqa: SIM102 (pipeline block structure)
                        if k_st <= t_max:
                            if target == "pto":
                                # bulk K load: the codegen's runtime tail
                                # clamp + zero pad protect the tensor end;
                                # cols >= ln are causally masked below
                                T.copy(IndexK[bos + k_st : bos + k_st + BN, 0:dim], K_l1)
                            else:
                                if k_st + BN <= ln:
                                    T.copy(IndexK[bos + k_st : bos + k_st + BN, 0:dim], K_l1)
                                else:
                                    # K tail block: per-row guarded load; cols >= ln
                                    # are causally masked to -inf below
                                    for i in T.serial(BN):
                                        if k_st + i < ln:
                                            T.copy(IndexK[bos + k_st + i, 0:dim], K_l1[i, 0:dim])
                            # GEMM (K x Q^T: same dot products and K-chunk
                            # accumulation order as Q x K^T) + CV transit leg
                            # 1 (Cube): L0C fp32 -> GM bf16 + relu — the
                            # reference's bf16 round point; ping-pong slot
                            # j % 2 (1KB rows)
                            T.gemm_v0(K_l1, Q_l1, logits_l0c, transpose_B=True, init=True, kL0Size=KL0)
                            T.copy(
                                logits_l0c,
                                workspace_logits[cid, j % 2, 0:BN, 0 : QG * heads],
                                enable_relu=True,
                            )
                    if j >= 1:  # noqa: SIM102 (pipeline block structure)
                        if k_st - BN <= t_max:  # a transit was produced at j-1
                            # V->C back-pressure: the ack of transit j-1 has
                            # been waited by the time its slot is rewritten
                            # (produce at j+2), so the Cube side never runs
                            # more than one transit ahead of the Vector side
                            T.copy(workspace_ack[cid, 0:_ACK_TOTAL], ack_l1[0, 0:_ACK_TOTAL])
                    if j < NBLK:  # noqa: SIM102 (pipeline block structure)
                        if k_st <= t_max:
                            if target == "pto":
                                # pto consume chain: half-token [BN, HALFC]
                                # wide-row tiles (the pto vector template
                                # needs >= 256B inner rows and its tile views
                                # cannot address column slices of a
                                # [BN, QG2H] buffer): relay read -> fp32
                                # cast -> w-mul -> bf16 round -> transpose
                                # staging -> in-place add folding tree
                                for hh in T.serial(2):
                                    T.copy(
                                        workspace_logits[
                                            cid,
                                            j % 2,
                                            0:BN,
                                            vh * heads + hh * HALFC : vh * heads + (hh + 1) * HALFC,
                                        ],
                                        relay_ub_h,
                                    )
                                    T.tile.broadcast(bc_f_h, w_ub[hh * HALFC : (hh + 1) * HALFC], axis=0)  # [HALFC] -> [BN, HALFC]
                                    T.copy(relay_ub_h, logits_f_h)  # bf16 -> fp32 (dense flat)
                                    T.tile.mul(logits_f_h, logits_f_h, bc_f_h)  # fp32 product p * w
                                    T.copy(logits_f_h, prod_f_h)  # fp32 -> bf16 (reference product round)
                                    T.copy(prod_f_h, logits_f_h)  # bf16 -> fp32
                                    if not scale_after_reduce:
                                        T.tile.mul(logits_f_h, logits_f_h, sm_scale)  # non-positive scale fallback
                                    # transpose to [HALFC, BN] staging, then fold
                                    # the head axis with an in-place add tree
                                    # (the pto dim=-1 row-sum reduce is not
                                    # available; the tree runs the same fp32
                                    # sums on the plain binop path)
                                    T.tile.transpose(seg_t, logits_f_h)
                                    for u in T.serial(HTOK):
                                        r = u * heads
                                        # statically branched per pow2 heads
                                        # (TVM script cannot prove shape
                                        # equality for TIR-expr tree
                                        # strides); non-pow2 heads fall back
                                        # to reduce_sum
                                        if heads == 64:
                                            T.tile.add(seg_t[r : r + 32, 0:BN], seg_t[r : r + 32, 0:BN], seg_t[r + 32 : r + 64, 0:BN])
                                            T.tile.add(seg_t[r : r + 16, 0:BN], seg_t[r : r + 16, 0:BN], seg_t[r + 16 : r + 32, 0:BN])
                                            T.tile.add(seg_t[r : r + 8, 0:BN], seg_t[r : r + 8, 0:BN], seg_t[r + 8 : r + 16, 0:BN])
                                            T.tile.add(seg_t[r : r + 4, 0:BN], seg_t[r : r + 4, 0:BN], seg_t[r + 4 : r + 8, 0:BN])
                                            T.tile.add(seg_t[r : r + 2, 0:BN], seg_t[r : r + 2, 0:BN], seg_t[r + 2 : r + 4, 0:BN])
                                            T.tile.add(seg_t[r : r + 1, 0:BN], seg_t[r : r + 1, 0:BN], seg_t[r + 1 : r + 2, 0:BN])
                                        elif heads == 32:
                                            T.tile.add(seg_t[r : r + 16, 0:BN], seg_t[r : r + 16, 0:BN], seg_t[r + 16 : r + 32, 0:BN])
                                            T.tile.add(seg_t[r : r + 8, 0:BN], seg_t[r : r + 8, 0:BN], seg_t[r + 8 : r + 16, 0:BN])
                                            T.tile.add(seg_t[r : r + 4, 0:BN], seg_t[r : r + 4, 0:BN], seg_t[r + 4 : r + 8, 0:BN])
                                            T.tile.add(seg_t[r : r + 2, 0:BN], seg_t[r : r + 2, 0:BN], seg_t[r + 2 : r + 4, 0:BN])
                                            T.tile.add(seg_t[r : r + 1, 0:BN], seg_t[r : r + 1, 0:BN], seg_t[r + 1 : r + 2, 0:BN])
                                        elif heads == 16:
                                            T.tile.add(seg_t[r : r + 8, 0:BN], seg_t[r : r + 8, 0:BN], seg_t[r + 8 : r + 16, 0:BN])
                                            T.tile.add(seg_t[r : r + 4, 0:BN], seg_t[r : r + 4, 0:BN], seg_t[r + 4 : r + 8, 0:BN])
                                            T.tile.add(seg_t[r : r + 2, 0:BN], seg_t[r : r + 2, 0:BN], seg_t[r + 2 : r + 4, 0:BN])
                                            T.tile.add(seg_t[r : r + 1, 0:BN], seg_t[r : r + 1, 0:BN], seg_t[r + 1 : r + 2, 0:BN])
                                        elif heads == 8:
                                            T.tile.add(seg_t[r : r + 4, 0:BN], seg_t[r : r + 4, 0:BN], seg_t[r + 4 : r + 8, 0:BN])
                                            T.tile.add(seg_t[r : r + 2, 0:BN], seg_t[r : r + 2, 0:BN], seg_t[r + 2 : r + 4, 0:BN])
                                            T.tile.add(seg_t[r : r + 1, 0:BN], seg_t[r : r + 1, 0:BN], seg_t[r + 1 : r + 2, 0:BN])
                                        else:
                                            T.reduce_sum(
                                                seg_t[u * heads : (u + 1) * heads, 0:BN],
                                                score_2d[hh * HTOK + u, 0:BN],
                                                dim=0,
                                                clear=True,
                                            )
                                        if heads in (8, 16, 32, 64):
                                            T.copy(seg_t[r : r + 1, 0:BN], score_2d[hh * HTOK + u, 0:BN])
                                # no block-level causal mask on this branch:
                                # the tail [CAP] row mask below covers
                                # causality (the scale moved off the block
                                # tile)
                            else:
                                # CV transit leg 2 (Vector): this AIV's
                                # contiguous half of the [BN, QG*H] tile,
                                # GM bf16 -> UB bf16 (disjoint column halves;
                                # 32 x 512B wide bursts)
                                T.copy(
                                    workspace_logits[cid, j % 2, 0:BN, vh * heads : (vh + QG2) * heads],
                                    relay_ub,
                                )
                                # fused full-tile chain: dense fp32 cast +
                                # w-mul + the reference's bf16 product
                                # round-trip over all this AIV's tokens at
                                # once
                                T.tile.broadcast(bc_f, w_ub, axis=0)  # [QG2H] -> [BN, QG2H]
                                T.copy(relay_ub, logits_f)  # bf16 -> fp32 (dense flat)
                                T.tile.mul(logits_f, logits_f, bc_f)  # fp32 product p * w
                                T.copy(logits_f, prod_f)  # fp32 -> bf16 (reference product round)
                                T.copy(prod_f, logits_f)  # bf16 -> fp32
                                if not scale_after_reduce:
                                    T.tile.mul(logits_f, logits_f, sm_scale)  # non-positive scale fallback
                                for u in T.serial(QG2):
                                    # unconditional reduce: the post-reduce
                                    # mask below overwrites every causally-
                                    # invalid (token, column) with -inf
                                    # (garbage rows included — never consumed)
                                    T.reduce_sum(
                                        logits_f[0:BN, u * heads : (u + 1) * heads],
                                        score_2d[u, 0:BN],
                                        dim=-1,
                                        clear=True,
                                    )  # [BN, H] -> [BN] (contiguous head reduce)
                                # fused causal-validity mask, straddle blocks
                                # only: blocks fully below the group's first
                                # token's diagonal (k_st + BN <= t0 + vh)
                                # skip it — every row is fully valid there
                                if k_st + BN > t0 + vh:
                                    T.tile.arith_progression(col_1d, k_st, 1, BN)
                                    T.tile.broadcast(colg2, col_1d, axis=0)  # [BN] -> [QG2, BN]
                                    T.tile.broadcast(tpart, t_abs_big[0:QG2], axis=1)  # [QG2] rows
                                    T.tile.compare(mask2, colg2, tpart, "LE")  # col <= t_abs(row)
                                    T.tile.select(
                                        score_2d,
                                        mask2,
                                        score_2d,
                                        -T.infinity(accum_dtype),
                                        "VSEL_TENSOR_SCALAR_MODE",
                                    )
                            # fused QG2-token store (dst rows stride CAP; invalid
                            # positions carry the mask's -inf)
                            T.copy(score_2d, score_row[0:QG2, k_st : k_st + BN])
                            # ack: transit consumed -> the Cube side may reuse
                            # slot j % 2 (after the chain: V->MTE3 ordered)
                            T.copy(ack_ub, workspace_ack[cid, vid * _ACK_PER_AIV : (vid + 1) * _ACK_PER_AIV])

                # ---- per-token topk + softmax + fixup + writeback ----
                T.tile.fill(val_ub, -T.infinity(accum_dtype))  # pads for topk < TOPK_A
                T.tile.fill(idx_f_ub, -1.0)
                for u in T.serial(QG2):
                    t_abs = t0 + vh + u
                    if t_abs < ln:
                        i_u = t_abs  # causal bound = in-segment position
                        # if_then_else instead of T.max: the CCE max()
                        # overload set is ambiguous for the (int64, int)
                        # argument pair under -xcce; identical integer
                        # semantics on both backends
                        n_eff = T.if_then_else(i_u + 1 > topk, i_u + 1, topk)
                        T.copy(score_row[u, 0:CAP], topk_src)
                        if scale_after_reduce:
                            # reference order: (fp32 sum of bf16 products) *
                            # sm_scale — one [CAP] mul per token per task
                            T.tile.mul(topk_src, topk_src, sm_scale)
                            if target == "pto":
                                # pto branch tail causal mask (this branch
                                # skipped the block-level mask); the
                                # reference-exact -inf form
                                T.tile.arith_progression(col_cap, 0, 1, CAP)
                                T.tile.compare(valid_cap, col_cap, T.float32(i_u), "LE")
                                T.tile.select(
                                    topk_src,
                                    valid_cap,
                                    topk_src,
                                    -T.infinity(accum_dtype),
                                    "VSEL_TENSOR_SCALAR_MODE",
                                )
                        # pad [2*topk:P2] with -inf so the 256B-quantised
                        # gather never selects out-of-bounds data into the pads
                        T.tile.fill(topk_dst, -T.infinity(accum_dtype))
                        T.tile.topk(topk_dst, topk_src, topk, n_eff)
                        T.tile.gather_mask(val_ub, topk_dst, "P0101")  # even = vals
                        T.tile.gather_mask(idx_f_ub, topk_dst, "P1010")  # odd = idx
                        # softmax over [TOPK_A] (pads are -inf -> exp 0)
                        T.reduce_max(val_ub, max_ub, dim=-1)
                        T.tile.sub(e_ub, val_ub, max_ub[0])
                        T.tile.exp(e_ub, e_ub)
                        T.reduce_sum(e_ub, sum_ub, dim=-1)
                        T.tile.div(softmax_ub, e_ub, sum_ub[0])
                        # idx fixup: -inf tail positions (idx > i_u) -> -1
                        T.tile.compare(valid, idx_f_ub, T.float32(i_u), "LE")
                        T.tile.select(idx_f_ub, valid, idx_f_ub, -1.0, "VSEL_TENSOR_SCALAR_MODE")
                        T.tile.cast(idx_ub, idx_f_ub, "CAST_ROUND", TOPK_A)
                        T.copy(idx_ub[0:topk], TopkIndices[bos + t_abs, 0:topk])
                        T.copy(softmax_ub[0:topk], ReduceSum[bos + t_abs, 0:topk])

    return main


# ============================================================================
# Generic kernel (CH-streamed score segments + merge_sort pool topk)
# ============================================================================


def tl_indexer_topk_reducesum_generic(
    heads,
    dim,
    topk,
    CH=1024,
    NSEG=5,
    QG=8,
    BN=32,
    grid_size=20,
    sm_scale=None,
    KL0=128,
    target="ascendc",
):
    """Build the generic-path kernel (any segment length; Developer mode).

    Same GEMM data flow as the fast path, but the score row is produced CH
    columns at a time; each CH segment runs a hardware topk whose LOCAL
    indices are corrected to the global column frame (fp32 + seg_lo) and
    merged into a per-token (val, idx) pool via the stable descending
    merge_sort — the pool after the last segment is the full-row topk.

    Args:
        heads / dim / topk / QG / BN / grid_size / sm_scale / KL0: as in the
            fast kernel (topk <= CH; KL0 host-computed by _pick_kl0_size).
        CH: score segment width (max(1024, next_pow2(topk)); % BN == 0).
        NSEG: compile-time upper bound on segments per group
            (= ceildiv(max_seg_len, CH), host-computed; the runtime guard
            ``seg_lo <= t_max`` truncates shorter groups).
        target: codegen backend, "ascendc" (default) or "pto" — same body.
    """
    return _jit_wrapper(target)(_tl_indexer_topk_reducesum_generic_main)(
        heads=heads,
        dim=dim,
        topk=topk,
        CH=CH,
        NSEG=NSEG,
        QG=QG,
        BN=BN,
        grid_size=grid_size,
        sm_scale=sm_scale,
        KL0=KL0,
        target=target,
    )


def _tl_indexer_topk_reducesum_generic_main(
    heads,
    dim,
    topk,
    CH=1024,
    NSEG=5,
    QG=8,
    BN=32,
    grid_size=20,
    sm_scale=None,
    KL0=128,
    target="ascendc",
):
    """(private) generic-path prim_func builder — see the public wrapper above."""
    assert heads % 8 == 0 and heads <= 64, "heads must be % 8 == 0 and <= 64"
    assert dim % 16 == 0, "dim must be a multiple of 16"
    assert topk == tilelang.math.next_power_of_2(topk) and topk >= 16, f"topk must be a power of two >= 16, got {topk}"
    assert CH % BN == 0 and topk <= CH, "CH must be a multiple of BN and >= topk"
    assert NSEG >= 1, "NSEG must be >= 1"
    assert QG % 2 == 0, "QG must be even (AIV-vid token-half split)"
    assert QG * heads * BN * 4 <= _L0C_CAPACITY, "L0C budget exceeded"
    assert KL0 % 16 == 0 and KL0 <= 4095, "KL0 must be % 16 == 0 and <= 4095"
    if sm_scale is None:
        sm_scale = dim**-0.5

    NBLK_SEG = CH // BN
    TOPK_A = max(topk, 64)
    P2 = max(2 * topk, 64)  # gather src width: 256B (64-fp32) quantum
    PS = max(topk, 32)  # gather dst width = selected count = P2 // 2
    QG2 = QG // 2  # tokens per AIV-vid (contiguous half of the group)

    seq_len = T.symbolic("seq_len")
    batch_plus_one = T.symbolic("batch_plus_one")
    num_groups = T.symbolic("num_groups")
    q_rows = T.symbolic("q_rows")
    w_rows = T.symbolic("w_rows")

    dtype = "bfloat16"
    accum_dtype = "float32"
    indices_dtype = "int32"

    @T.prim_func
    def main(
        IndexQ: T.Tensor([q_rows, dim], dtype),  # type: ignore
        Weights: T.Tensor([w_rows], dtype),  # type: ignore
        IndexK: T.Tensor([seq_len, dim], dtype),  # type: ignore
        TopkIndices: T.Tensor([seq_len, topk], indices_dtype),  # type: ignore
        ReduceSum: T.Tensor([seq_len, topk], accum_dtype),  # type: ignore
        Offsets: T.Tensor([batch_plus_one], indices_dtype),  # type: ignore
        TokenIndices: T.Tensor([seq_len, 2], indices_dtype),  # type: ignore  # interface fidelity, not consumed
        GroupTable: T.Tensor([num_groups, 3], indices_dtype),  # type: ignore
        workspace_logits: T.Tensor([grid_size, QG * heads, BN], dtype),  # type: ignore  # C->V transit, single slot
        workspace_ack: T.Tensor([grid_size, _ACK_TOTAL], accum_dtype),  # type: ignore  # V->C back-pressure
    ):
        with T.Kernel(grid_size, is_npu=True) as (cid, vid):  # MIX_AIC_1_2
            # ---- L1 / L0C / K-loop UB chain: identical to the fast path ----
            Q_l1 = T.alloc_shared([QG * heads, dim], dtype)
            K_l1 = T.alloc_shared([BN, dim], dtype)
            # explicit 2D L1 (plain "shared" is use-inferred to UB, which
            # reclassifies the ack GM read as a vector op and breaks the CV
            # pairing; 1D L1 breaks the default zN layout injection)
            ack_l1 = T.alloc_shared([1, _ACK_TOTAL], accum_dtype, scope="shared.l1")
            logits_l0c = T.alloc_fragment([QG * heads, BN], accum_dtype)
            relay_ub = T.alloc_shared([QG2 * heads, BN], dtype)
            w_src = T.alloc_shared([QG2 * heads], dtype)
            w_ub = T.alloc_shared([QG2 * heads], accum_dtype)
            ack_ub = T.alloc_shared([_ACK_PER_AIV], accum_dtype)  # ack heartbeat
            logits_f32 = T.alloc_shared([heads, BN], accum_dtype)
            prod_bf16 = T.alloc_shared([heads, BN], dtype)  # reference product round
            bc_2d = T.alloc_shared([heads, BN], accum_dtype)
            causal = T.alloc_shared([heads, BN // 8], "uint8")  # packed compare dst
            w_1d = T.alloc_shared([heads], accum_dtype)
            col_1d = T.alloc_shared([BN], accum_dtype)
            score_u = T.alloc_shared([BN], accum_dtype)
            # ---- generic topk consumption (per AIV-vid) ----
            score_seg = T.alloc_shared([QG2, CH], accum_dtype)  # local columns [0, CH)
            seg_src = T.alloc_shared([CH], accum_dtype)
            seg_dst = T.alloc_shared([P2], accum_dtype)  # + -inf pad
            seg_val = T.alloc_shared([PS], accum_dtype)  # gather dst width
            seg_idx = T.alloc_shared([PS], accum_dtype)
            seg_pairs = T.alloc_shared([2 * topk], accum_dtype)
            pool = T.alloc_shared([QG2, 2 * topk], accum_dtype)  # (val, idx) pairs, global idx
            pool_row = T.alloc_shared([P2], accum_dtype)  # gather src, 256B-quantised
            merged = T.alloc_shared([4 * topk], accum_dtype)
            # ---- tail chain (identical to the fast path) ----
            val_ub = T.alloc_shared([TOPK_A], accum_dtype)
            idx_f_ub = T.alloc_shared([TOPK_A], accum_dtype)
            e_ub = T.alloc_shared([TOPK_A], accum_dtype)
            softmax_ub = T.alloc_shared([TOPK_A], accum_dtype)
            valid = T.alloc_shared([TOPK_A // 8], "uint8")  # packed compare dst
            idx_ub = T.alloc_shared([TOPK_A], indices_dtype)
            max_ub = T.alloc_shared([1], accum_dtype)
            sum_ub = T.alloc_shared([1], accum_dtype)

            ntask = T.max(0, T.ceildiv(num_groups - cid, grid_size))
            for k in T.serial(ntask):
                bx = k * grid_size + cid
                bos = GroupTable[bx, 0]
                t0 = GroupTable[bx, 1]
                ln = GroupTable[bx, 2]
                t_max = T.min(t0 + QG, ln) - 1
                vh = vid * QG2  # this AIV's contiguous token offset

                # ---- Q load (Cube side, full QG) / weights load (V side,
                #      this AIV's half). pto target: bulk loads only — the
                #      pto codegen adds a runtime tail clamp + zero pad;
                #      out-of-segment rows are masked or unconsumed. ----
                if target == "pto":
                    T.copy(IndexQ[(bos + t0) * heads : (bos + t0 + QG) * heads, 0:dim], Q_l1)
                else:
                    if t0 + QG <= ln:
                        T.copy(IndexQ[(bos + t0) * heads : (bos + t0 + QG) * heads, 0:dim], Q_l1)
                    else:
                        for u in T.serial(QG):
                            if t0 + u < ln:
                                T.copy(
                                    IndexQ[(bos + t0 + u) * heads : (bos + t0 + u + 1) * heads, 0:dim],
                                    Q_l1[u * heads : (u + 1) * heads, 0:dim],
                                )
                if target == "pto":
                    T.copy(
                        Weights[(bos + t0 + vh) * heads : (bos + t0 + vh + QG2) * heads],
                        w_src,
                    )
                else:
                    if t0 + vh + QG2 <= ln:
                        T.copy(
                            Weights[(bos + t0 + vh) * heads : (bos + t0 + vh + QG2) * heads],
                            w_src,
                        )
                    else:
                        for u in T.serial(QG2):
                            if t0 + vh + u < ln:
                                T.copy(
                                    Weights[(bos + t0 + vh + u) * heads : (bos + t0 + vh + u + 1) * heads],
                                    w_src[u * heads : (u + 1) * heads],
                                )
                T.copy(w_src, w_ub)  # bf16 -> fp32; no sm_scale here
                T.tile.fill(ack_ub, 0.0)  # heartbeat payload

                # ---- CH-segment loop (compile-time bound + runtime guard).
                #      Serialized back-pressure (not perf-gated): the ack of
                #      the previous produce (k + sg + j > 0 skips only the
                #      core's very first) is waited BEFORE this produce, so a
                #      single transit slot is safe; the final leftover ack
                #      set is inert at kernel end. ----
                for sg in T.serial(NSEG):
                    seg_lo = sg * CH
                    if seg_lo <= t_max:
                        T.tile.fill(score_seg, -T.infinity(accum_dtype))
                        for j in T.serial(NBLK_SEG):
                            k_st = seg_lo + j * BN  # global column (causal frame)
                            if k_st <= t_max:
                                if k + sg + j > 0:
                                    T.copy(workspace_ack[cid, 0:_ACK_TOTAL], ack_l1[0, 0:_ACK_TOTAL])
                                if target == "pto":
                                    # bulk K load: runtime tail clamp + zero
                                    # pad; cols >= ln causally masked
                                    T.copy(IndexK[bos + k_st : bos + k_st + BN, 0:dim], K_l1)
                                else:
                                    if k_st + BN <= ln:
                                        T.copy(IndexK[bos + k_st : bos + k_st + BN, 0:dim], K_l1)
                                    else:
                                        for i in T.serial(BN):
                                            if k_st + i < ln:
                                                T.copy(IndexK[bos + k_st + i, 0:dim], K_l1[i, 0:dim])
                                T.gemm_v0(
                                    Q_l1,
                                    K_l1,
                                    logits_l0c,
                                    transpose_B=True,
                                    init=True,
                                    kL0Size=KL0,
                                )
                                T.copy(
                                    logits_l0c,
                                    workspace_logits[cid, 0 : QG * heads, 0:BN],
                                    enable_relu=True,
                                )
                                # this AIV's contiguous half (disjoint halves)
                                T.copy(
                                    workspace_logits[cid, vh * heads : (vh + QG2) * heads, 0:BN],
                                    relay_ub,
                                )
                                for u in T.serial(QG2):
                                    t_abs = t0 + vh + u
                                    T.copy(w_ub[u * heads : (u + 1) * heads], w_1d)
                                    T.tile.broadcast(bc_2d, w_1d, axis=1)
                                    T.copy(relay_ub[u * heads : (u + 1) * heads, 0:BN], logits_f32)
                                    T.tile.mul(logits_f32, logits_f32, bc_2d)  # fp32 product p * w
                                    T.copy(logits_f32, prod_bf16)  # reference product round
                                    T.copy(prod_bf16, logits_f32)
                                    T.tile.mul(logits_f32, logits_f32, sm_scale)  # scale after the round
                                    T.tile.arith_progression(col_1d, k_st, 1, BN)
                                    T.tile.broadcast(bc_2d, col_1d, axis=0)
                                    T.tile.compare(causal, bc_2d, T.float32(t_abs), "LE")
                                    T.tile.select(
                                        logits_f32,
                                        causal,
                                        logits_f32,
                                        -T.infinity(accum_dtype),
                                        "VSEL_TENSOR_SCALAR_MODE",
                                    )
                                    T.reduce_sum(logits_f32, score_u, dim=0, clear=True)
                                    # local column inside the CH segment
                                    T.copy(score_u, score_seg[u, j * BN : j * BN + BN])
                                # ack: transit consumed (after the chain)
                                T.copy(
                                    ack_ub,
                                    workspace_ack[cid, vid * _ACK_PER_AIV : (vid + 1) * _ACK_PER_AIV],
                                )
                        # ---- segment topk + global-idx correction + pool merge ----
                        for u in T.serial(QG2):
                            t_abs = t0 + vh + u
                            if t_abs < ln:
                                i_u = t_abs
                                T.copy(score_seg[u, 0:CH], seg_src)
                                # if_then_else instead of T.max/T.min (CCE
                                # overload ambiguity on the vid-derived int64
                                # argument); identical integer semantics
                                in_range = i_u + 1 - seg_lo
                                seg_valid = T.if_then_else(
                                    in_range < CH,
                                    T.if_then_else(in_range > topk, in_range, topk),
                                    T.if_then_else(topk < CH, CH, topk),
                                )
                                T.tile.fill(seg_dst, -T.infinity(accum_dtype))  # pad
                                T.tile.topk(seg_dst, seg_src, topk, seg_valid)
                                T.tile.gather_mask(seg_val, seg_dst, "P0101")
                                T.tile.gather_mask(seg_idx, seg_dst, "P1010")
                                # local -> global column frame
                                T.tile.add(seg_idx, seg_idx, T.float32(seg_lo))
                                # scalar interleave back into (val, idx) pairs
                                for i in T.serial(topk):
                                    seg_pairs[2 * i] = seg_val[i]
                                    seg_pairs[2 * i + 1] = seg_idx[i]
                                if sg == 0:
                                    T.copy(seg_pairs, pool[u, 0 : 2 * topk])
                                else:
                                    T.tile.merge_sort(merged, pool[u, 0 : 2 * topk], seg_pairs)  # stable descending 2-way merge
                                    T.copy(merged[0 : 2 * topk], pool[u, 0 : 2 * topk])

                # ---- tail: softmax + fixup + writeback from the pool ----
                T.tile.fill(val_ub, -T.infinity(accum_dtype))
                T.tile.fill(idx_f_ub, -1.0)
                for u in T.serial(QG2):
                    t_abs = t0 + vh + u
                    if t_abs < ln:
                        i_u = t_abs
                        T.tile.fill(pool_row, -T.infinity(accum_dtype))  # pad
                        T.copy(pool[u, 0 : 2 * topk], pool_row[0 : 2 * topk])
                        T.tile.gather_mask(val_ub, pool_row, "P0101")
                        T.tile.gather_mask(idx_f_ub, pool_row, "P1010")
                        T.reduce_max(val_ub, max_ub, dim=-1)
                        T.tile.sub(e_ub, val_ub, max_ub[0])
                        T.tile.exp(e_ub, e_ub)
                        T.reduce_sum(e_ub, sum_ub, dim=-1)
                        T.tile.div(softmax_ub, e_ub, sum_ub[0])
                        T.tile.compare(valid, idx_f_ub, T.float32(i_u), "LE")
                        T.tile.select(idx_f_ub, valid, idx_f_ub, -1.0, "VSEL_TENSOR_SCALAR_MODE")
                        T.tile.cast(idx_ub, idx_f_ub, "CAST_ROUND", TOPK_A)
                        T.copy(idx_ub[0:topk], TopkIndices[bos + t_abs, 0:topk])
                        T.copy(softmax_ub[0:topk], ReduceSum[bos + t_abs, 0:topk])

    return main


# ============================================================================
# Host side: data preparation + detection + gates + dispatch entry
# ============================================================================


def _next_pow2(n):
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _pick_kl0_size(m_rows, dim):
    """Largest gemm_v0 kL0Size whose big-side tile fits the L0A/L0B ping-pong slot.

    Mirror of the C++ static asserts in the Ascend gemm template: both the
    A slot (M x kL0Size) and — on the transpose_B path — the B slot
    (nTile x kL0Size) must fit 64KB for a single (N-tile, K-tile) step or
    32KB once the ping-pong alternates; ``m_rows`` is the big side
    (QG*heads). Returns 0 when no candidate fits (M > 1024 for dim >= 16 —
    outside the operator domain).
    """
    for kl0 in (128, 64, 32, 16):
        if kl0 > dim:
            continue
        steps = -(-dim // kl0)
        budget = 65536 if steps == 1 else 32768
        if m_rows * kl0 * 2 <= budget:
            return kl0
    return 0


def _off_list(offsets):
    """Fetch the offsets content once per interface call (one synchronizing
    D2H; every consumer reuses the list)."""
    return [int(x) for x in offsets.detach().cpu().tolist()]


def prepare_token_indices(offsets, off_list=None):
    """Build TokenIndices [sum(S), 2] int32 ((batch_id, position)) from cu_seqlens.

    numpy construction + a single H2D copy (interface-fidelity output of the
    reference's prepare_token_indices; the kernels do not consume it).
    """
    if off_list is None:
        off_list = _off_list(offsets)
    off = np.asarray(off_list, dtype=np.int64)
    lens = off[1:] - off[:-1]
    total = int(lens.sum()) if lens.size else 0
    if total == 0:
        return torch.zeros((0, 2), dtype=torch.int32).to(offsets.device)
    batch = np.repeat(np.arange(lens.size, dtype=np.int32), lens)
    starts = np.concatenate(([0], np.cumsum(lens)[:-1]))
    pos = np.arange(total, dtype=np.int64) - starts[batch.astype(np.int64)]
    ti = np.stack((batch, pos.astype(np.int32)), axis=1)
    return torch.from_numpy(ti).to(offsets.device)


def prepare_group_table(offsets, qg, off_list=None):
    """Build GroupTable [num_groups, 3] int32 ((bos, t0_local, seq_len)).

    One row per ceildiv(len_i, qg) group of each varlen segment; numpy
    construction + a single H2D copy (fast/generic share the layout).
    """
    if off_list is None:
        off_list = _off_list(offsets)
    off = np.asarray(off_list, dtype=np.int64)
    lens = off[1:] - off[:-1] if off.size > 1 else np.zeros(0, dtype=np.int64)
    counts = (lens + qg - 1) // qg
    n_groups = int(counts.sum()) if counts.size else 0
    if n_groups == 0:
        return torch.zeros((0, 3), dtype=torch.int32).to(offsets.device)
    seg = np.repeat(np.arange(lens.size, dtype=np.int64), counts)
    base = np.concatenate(([0], np.cumsum(counts)[:-1]))  # first group id per segment
    g = np.arange(n_groups, dtype=np.int64) - base[seg]
    gt = np.stack((off[seg], g * qg, lens[seg]), axis=1).astype(np.int32)
    return torch.from_numpy(gt).to(offsets.device)


def _get_cube_core_num(default=20):
    """Query the device cube-core count (persistent grid constant; 910B3 = 20)."""
    try:
        if torch.npu.is_available():
            n = torch.npu.get_device_properties(0).cube_core_num
            if isinstance(n, int) and n > 0:
                return n
    except Exception:  # pragma: no cover - fall back on probe failure
        pass
    return default


def _max_seg_len(offsets, off_list=None):
    if off_list is None:
        off_list = _off_list(offsets)
    return max((off_list[b + 1] - off_list[b] for b in range(len(off_list) - 1)), default=0)


def _detect_dispatch(offsets, off_list=None):
    """Exact (non-heuristic) host detection: 'fast' | 'generic'."""
    return "fast" if _max_seg_len(offsets, off_list) <= 4096 else "generic"


def _fast_gate_ok(heads, dim, topk, CAP, QG, BN=32):
    """Host mirror of the fast-path domain + exact UB/L1/L0A budget model.

    UB model is liveness-aware and per-AIV (MIX_AIC_1_2: each Vector core
    owns QG//2 tokens of every group): the K-block-phase buffers
    (relay/vector chain) and the topk-phase buffers (topk_src/dst/softmax
    chain/auto tmp) have disjoint live ranges inside one task; score_row
    persists across both. The L0A model mirrors _pick_kl0_size. The L1
    model bounds the Cube-side staging — Q_l1 [QG*heads, dim] + K_l1
    [BN, dim] bf16 + the fp32 ack heartbeat, simultaneously live in the
    K-block loop — by the 524032B L1 pool (the memory planner packs them
    disjointly with no overflow check; heads=64 x dim=512 x QG=8 alone
    needs 544KiB).
    """
    if not (heads % 8 == 0 and heads <= 64):
        return False
    if topk != _next_pow2(topk) or topk < 16 or topk > 1024:
        return False
    if CAP not in (512, 1024, 2048, 4096) or topk > CAP:
        return False
    qg_h = QG * heads
    if qg_h * BN * 4 > _L0C_CAPACITY:
        return False
    if QG % 4 != 0:
        return False  # AIV-vid token-half split + half-token chain fusion
    # (L0A feasibility with the actual dim is enforced in _run_fast via
    #  _pick_kl0_size; here M = QG*heads <= 512 always has a candidate.)
    if (QG * heads + BN) * dim * 2 + _ACK_TOTAL * 4 > _L1_CAPACITY:
        return False
    qg2_h = (QG // 2) * heads
    topk_a = max(topk, 64)
    k_phase = (
        qg2_h * BN * 2  # relay_ub bf16 (per AIV)
        + qg2_h * BN * 4 * 2  # logits_f + bc_f (fused full-tile chain)
        + qg2_h * BN * 2  # prod_f bf16 (reference product round)
        + 64 * 4  # t_abs_big (arith 256B-quantum safe)
        + (QG // 2) * BN * 4 * 3  # colg2 + tpart + mask2 (fused post-reduce mask)
        + qg2_h * 2
        + qg2_h * 4  # w_src bf16 + w_ub fp32 (per AIV)
        + BN * 4  # col_1d
        + (QG // 2) * BN * 4  # score_2d (per-block token scores)
        + 64  # ack_ub
    )
    p2 = max(2 * topk, 64)
    topk_phase = (
        CAP * 4  # topk_src
        + p2 * 4  # topk_dst (256B gather quantum)
        + 4 * topk_a * 4
        + topk_a * 4
        + topk_a  # val/idx_f/e/softmax + idx_ub + valid
        + 8
    )
    # The planner packs every buffer disjointly (no cross-phase liveness
    # sharing); the auto tmp is ONE shared buffer sized by its largest user
    # (topk 16B/elem vs the K-phase reduce's active_rows * padded_row_bytes).
    tmp_shared = max(16 * CAP, (heads // 2) * BN * 4)
    total = (QG // 2) * CAP * 4 + k_phase + topk_phase + tmp_shared + 2 * 1024
    return total < _UB_CAPACITY


def _generic_gate_ok(heads, dim, topk, CH=1024, QG=8, BN=32):
    """Host mirror of the generic-path domain + exact per-AIV UB / Cube L1
    budget model (the L1 staging bound is identical to the fast gate)."""
    if not (heads % 8 == 0 and heads <= 64):
        return False
    if topk != _next_pow2(topk) or topk < 16 or topk > 1024:
        return False
    CH = max(CH, _next_pow2(topk))
    if CH % BN != 0 or topk > CH:
        return False
    qg_h = QG * heads
    if qg_h * BN * 4 > _L0C_CAPACITY:
        return False
    if (QG * heads + BN) * dim * 2 + _ACK_TOTAL * 4 > _L1_CAPACITY:
        return False
    qg2_h = (QG // 2) * heads
    topk_a = max(topk, 64)
    k_phase = (
        qg2_h * BN * 2
        + heads * BN * 4 * 2
        + heads * BN * 2  # prod_bf16 (reference product round)
        + heads * BN * 4
        + qg2_h * 2
        + qg2_h * 4
        + heads * 4
        + BN * 4 * 2
    )
    p2 = max(2 * topk, 64)
    ps = max(topk, 32)
    seg_phase = (
        CH * 4  # seg_src
        + p2 * 4  # seg_dst (256B gather quantum)
        + 2 * ps * 4  # seg_val + seg_idx (gather dst width)
        + 2 * topk * 4  # seg_pairs
        + p2 * 4  # pool_row (256B gather quantum)
        + 4 * topk * 4  # merged
        + 16 * CH  # auto topk tmp (ascendc)
    )
    tail_phase = 4 * topk_a * 4 + topk_a * 4 + topk_a + 8
    total = (
        (QG // 2) * CH * 4  # score_seg (persists across the sg loop, per AIV)
        + (QG // 2) * 2 * topk * 4  # pool (persists, per AIV)
        + max(k_phase, seg_phase, tail_phase)
        + 8 * 1024
    )
    return total < _UB_CAPACITY


_BUILD_CACHE: dict = {}

# content-keyed memo of the interface tables: both tables are pure functions
# of (offsets contents, qg), so the key IS the content — no tensor-identity
# staleness caveat. Two H2D copies at ~85us fixed overhead each dominate the
# build; benchmark/training loops reuse varlen layouts.
_TABLE_CACHE: dict = {}
_TABLE_CACHE_LIMIT = 64


def _cached_token_indices(offsets, off_list):
    key = ("ti", tuple(off_list), str(offsets.device))
    hit = _TABLE_CACHE.get(key)
    if hit is None:
        hit = prepare_token_indices(offsets, off_list)
        if len(_TABLE_CACHE) >= _TABLE_CACHE_LIMIT:
            _TABLE_CACHE.clear()
        _TABLE_CACHE[key] = hit
    return hit


def _cached_group_table(offsets, off_list, qg):
    key = ("gt", tuple(off_list), qg, str(offsets.device))
    hit = _TABLE_CACHE.get(key)
    if hit is None:
        hit = prepare_group_table(offsets, qg, off_list)
        if len(_TABLE_CACHE) >= _TABLE_CACHE_LIMIT:
            _TABLE_CACHE.clear()
        _TABLE_CACHE[key] = hit
    return hit


def _run_fast(q, weights, k, topk, offsets, token_indices, off_list, backend="ascendc"):
    """Launch the fast kernel (max_seg_len <= 4096 contract already validated)."""
    seq_len, heads, dim = q.shape
    device = q.device
    grid_size = _get_cube_core_num()
    max_seg = _max_seg_len(offsets, off_list)
    CAP = max(512, _next_pow2(max_seg))
    # QG tier: 8 tokens/group whenever the UB and L1 models fit, else 4 (the
    # fused [BN, QG2H] fp32 pair overflows the UB model at CAP=2048 x
    # topk=1024 with QG=8; the Q/K L1 staging overflows at heads=64 x
    # dim=512 with QG=8)
    QG = 8 if CAP <= 2048 else 4
    if QG == 8 and not _fast_gate_ok(heads, dim, topk, CAP, QG):
        QG = 4
    if not _fast_gate_ok(heads, dim, topk, CAP, QG):
        raise ValueError(
            f"fast path infeasible (heads={heads}, dim={dim}, topk={topk}, CAP={CAP}, QG={QG}): "
            "no QG tier in (8, 4) fits the UB/L1 budgets — reduce topk, reduce heads*dim, "
            "use a shorter max segment, or dispatch='generic'"
        )
    if backend == "pto" and CAP > 2048:
        # the pto fast path at CAP=4096 exceeds the 192KB UB budget (the
        # pto auto topk tmp is 24B/elem); route to the generic path — both
        # paths are output-equivalent
        return _run_generic(q, weights, k, topk, offsets, token_indices, off_list, backend)

    group_table = _cached_group_table(offsets, off_list, QG)
    kl0 = _pick_kl0_size(QG * heads, dim)
    if kl0 == 0:
        raise ValueError(f"fast path infeasible: no kL0Size fits L0A for M={QG * heads}, dim={dim}")
    topk_indices = torch.zeros((seq_len, topk), dtype=torch.int32, device=device)
    topk_score = torch.zeros((seq_len, topk), dtype=torch.float32, device=device)
    workspace_logits = torch.empty((grid_size, 2, _BN, QG * heads), dtype=torch.bfloat16, device=device)
    workspace_ack = torch.empty((grid_size, 2 * 16), dtype=torch.float32, device=device)

    key = ("fast", backend, heads, dim, topk, CAP, QG, grid_size, kl0)
    func = _BUILD_CACHE.get(key)
    if func is None:
        kernel = tl_indexer_topk_reducesum_fast(
            heads=heads,
            dim=dim,
            topk=topk,
            CAP=CAP,
            QG=QG,
            BN=_BN,
            grid_size=grid_size,
            KL0=kl0,
            target=backend,
        )
        # cached low-overhead launch path: raw [S, H, D] q / [S, H] weights
        # are viewed flat inside
        func = _CachedLauncher(kernel, view_spec={0: "flat2d", 1: "flat1d"})
        _BUILD_CACHE[key] = func

    func(
        q,
        weights,
        k,
        topk_indices,
        topk_score,
        offsets,
        token_indices,
        group_table,
        workspace_logits,
        workspace_ack,
    )
    return topk_indices, topk_score


def _run_generic(q, weights, k, topk, offsets, token_indices, off_list, backend="ascendc"):
    """Launch the generic kernel (any segment length)."""
    seq_len, heads, dim = q.shape
    device = q.device
    grid_size = _get_cube_core_num()
    max_seg = _max_seg_len(offsets, off_list)
    CH = max(1024, _next_pow2(topk))
    NSEG = (max_seg + CH - 1) // CH
    # QG tier: 8 tokens/group whenever the UB and L1 models fit, else 4,
    # else 2 (the Q/K L1 staging overflows at heads=64 x dim=512 with QG=8;
    # the generic body accepts any even QG, so QG=2 also serves as the
    # rescue tier for dims the fast path cannot stage)
    QG = 8
    if not _generic_gate_ok(heads, dim, topk, CH, QG):
        QG = 4
        if not _generic_gate_ok(heads, dim, topk, CH, QG):
            QG = 2
    if not _generic_gate_ok(heads, dim, topk, CH, QG):
        raise ValueError(
            f"generic path infeasible (heads={heads}, dim={dim}, topk={topk}, CH={CH}, QG={QG}): "
            "topk must be a power of two in [16, 1024] and no QG tier in (8, 4, 2) fits "
            "the UB/L1 budgets — reduce topk or heads*dim"
        )

    group_table = _cached_group_table(offsets, off_list, QG)
    kl0 = _pick_kl0_size(QG * heads, dim)
    if kl0 == 0:
        raise ValueError(f"generic path infeasible: no kL0Size fits L0A for M={QG * heads}, dim={dim}")
    topk_indices = torch.zeros((seq_len, topk), dtype=torch.int32, device=device)
    topk_score = torch.zeros((seq_len, topk), dtype=torch.float32, device=device)
    workspace_logits = torch.empty((grid_size, QG * heads, _BN), dtype=torch.bfloat16, device=device)
    workspace_ack = torch.empty((grid_size, 2 * 16), dtype=torch.float32, device=device)

    key = ("generic", backend, heads, dim, topk, CH, NSEG, QG, grid_size, kl0)
    func = _BUILD_CACHE.get(key)
    if func is None:
        kernel = tl_indexer_topk_reducesum_generic(
            heads=heads,
            dim=dim,
            topk=topk,
            CH=CH,
            NSEG=NSEG,
            QG=QG,
            BN=_BN,
            grid_size=grid_size,
            KL0=kl0,
            target=backend,
        )
        func = _CachedLauncher(kernel, view_spec={0: "flat2d", 1: "flat1d"})
        _BUILD_CACHE[key] = func

    func(
        q,
        weights,
        k,
        topk_indices,
        topk_score,
        offsets,
        token_indices,
        group_table,
        workspace_logits,
        workspace_ack,
    )
    return topk_indices, topk_score


def indexer_topk_reducesum_interface(
    q,
    weights,
    k,
    topk,
    offsets,
    dtype="bfloat16",
    dispatch="auto",
    backend="ascendc",
):
    """Host entry: varlen lightning-indexer forward topk + reducesum.

    Signature of the reference implementation (q, weights, k, topk, offsets,
    dtype=BF16) plus tail parameters with defaults.

    Args:
        q: IndexQ [seq_len, heads, dim] bfloat16.
        weights: Weights [seq_len, heads] bfloat16.
        k: IndexK [seq_len, dim] bfloat16.
        topk: selected positions (power of two, 16 <= topk <= 1024, and
            <= every segment length — the reference asserts segment >= topk).
        offsets: cu_seqlens [batch + 1] int32.
        dtype: input dtype (bfloat16, the reference's fixed precision path).
        dispatch: "auto" (exact host detection: max_seg_len <= 4096 -> fast,
            else generic), "fast" (validates max_seg_len <= 4096), "generic".
        backend: codegen backend, "ascendc" (default) or "pto". Backend
            selection is orthogonal to dispatch (detection and gates are
            pure host logic, identical for both backends); the compiled
            kernels are cached per (config, backend).

    Returns:
        (topk_indices [seq_len, topk] int32, topk_score [seq_len, topk] fp32).

    All tensor inputs must be contiguous (call ``.contiguous()`` on sliced
    or permuted views): the cached launch path marshals raw data pointers
    and cannot retain an implicit reshape copy across the launch.
    """
    if q.dim() != 3 or weights.dim() != 2 or k.dim() != 2:
        raise ValueError(f"rank mismatch: q {tuple(q.shape)}, w {tuple(weights.shape)}, k {tuple(k.shape)}")
    seq_len, heads, dim = q.shape
    if weights.shape != (seq_len, heads) or k.shape != (seq_len, dim):
        raise ValueError(f"shape mismatch: q {tuple(q.shape)}, w {tuple(weights.shape)}, k {tuple(k.shape)}")
    if not (q.is_contiguous() and weights.is_contiguous() and k.is_contiguous() and offsets.is_contiguous()):
        raise ValueError(
            "q/weights/k/offsets must be contiguous (call .contiguous() on non-contiguous "
            f"views), got q={q.is_contiguous()}, weights={weights.is_contiguous()}, "
            f"k={k.is_contiguous()}, offsets={offsets.is_contiguous()}"
        )
    if dtype not in ("bfloat16", "bf16", "float32"):
        # the reference passes T.bfloat16 (== "bfloat16"); accept the
        # bfloat16 family only — the precision path is fixed bf16
        raise ValueError(f"only bfloat16 inputs are supported, got {dtype!r}")
    if q.dtype != torch.bfloat16 or weights.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise ValueError(f"q/weights/k must be bfloat16, got {q.dtype}/{weights.dtype}/{k.dtype}")
    if offsets.dtype != torch.int32:
        raise ValueError(f"offsets must be int32, got {offsets.dtype}")
    if heads % 8 != 0 or heads > 64:
        raise ValueError(f"heads must be % 8 == 0 and <= 64, got {heads}")
    if dim % 16 != 0:
        raise ValueError(f"dim must be a multiple of 16, got {dim}")
    if topk != _next_pow2(topk) or topk < 16 or topk > 1024:
        raise ValueError(f"topk must be a power of two in [16, 1024], got {topk}")
    if dispatch not in ("auto", "fast", "generic"):
        raise ValueError(f"dispatch must be auto/fast/generic, got {dispatch!r}")
    if backend not in ("ascendc", "pto"):
        raise ValueError(f"backend must be ascendc/pto, got {backend!r}")

    # single synchronizing D2H per call; every consumer reuses the list
    off_list = _off_list(offsets)
    for b in range(len(off_list) - 1):
        seg_len = off_list[b + 1] - off_list[b]
        if seg_len < topk:
            raise ValueError(f"varlen segment {b} has length {seg_len} < topk {topk}")

    token_indices = _cached_token_indices(offsets, off_list)  # interface fidelity (not consumed)

    if dispatch == "auto":
        dispatch = _detect_dispatch(offsets, off_list)
    if dispatch == "fast":
        if _max_seg_len(offsets, off_list) > 4096:
            raise ValueError("dispatch='fast' requires max segment length <= 4096; use dispatch='generic'")
        return _run_fast(q, weights, k, topk, offsets, token_indices, off_list, backend)
    return _run_generic(q, weights, k, topk, offsets, token_indices, off_list, backend)
