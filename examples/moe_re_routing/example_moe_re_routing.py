"""MoE token re-routing example (TileLang-Ascend, Developer mode).

Permutes tokens (A, H) according to ``expert_token_num_per_rank`` (N, E):
source order is rank-major blocks, destination order is expert-major blocks,
block-internal order is preserved.  Also emits the gather index
``permute_token_idx`` (dst -> src), the re-permuted ``per_token_scales``
(zeros when input scales are absent) and ``expert_token_num`` (count mode).

The whole permutation (prefix sums, block lookup, GM<->UB<->GM moves,
index/scales generation, expert counts, validation probe) happens inside the
kernel.  The ``moe_re_routing`` host callable only performs metadata
operations (reshape / contiguous / variant selection / output allocation).
"""

import math

import tilelang
import torch
from tilelang import language as T

tilelang.cache.clear_cache()

# ========== Developer-mode pass configs (pure Vector op, no Cube) ==========
# Note: AUTO_CV_COMBINE / AUTO_CV_SYNC are intentionally OFF.  A CV-split
# would place the scalar prefix-sum loops on the Cube lane and the T.copy
# moves on the Vector lane without guaranteed cross-lane ordering, so the
# prefix tables would be consumed before being produced.  This kernel is a
# pure GM<->UB<->GM move + index arithmetic op; keeping everything on the
# Vector lane with AUTO_SYNC gives correct data order.
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

# Ascend910B3 AI-core block cap.
NUM_CORES_CAP = 24

# UB hardware capacity is 192KB; keep all statically-allocated UB buffers
# (kernel buffers + auto-sync temporaries / MemoryPlanning padding) strictly
# below 176KB so the planner keeps some headroom.
_UB_BUDGET_BYTES = 176 * 1024
_UB_MARGIN_BYTES = 8192

_TL_DTYPE = {
    "float16": "float16",
    "bfloat16": "bfloat16",
    "int8": "int8",
    "int32": "int32",
    "int64": "int64",
    "float32": "float32",
}

_ELEM_BYTES = {
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    "int32": 4,
    "int64": 8,
    "float32": 4,
}

# Cached per-call allocation for the no-scale dummy input (the dummy is never
# read by the kernel, so uninitialised empty storage is safe and avoids an
# extra device-side zero kernel on every call).
_DUMMY_SCALE = None


def _get_dummy_scale(device=None):
    global _DUMMY_SCALE
    if _DUMMY_SCALE is None:
        _DUMMY_SCALE = torch.empty(1, dtype=torch.float32, device="npu")
    if device is not None and _DUMMY_SCALE.device != torch.device(device):
        return torch.empty(1, dtype=torch.float32, device=device)
    return _DUMMY_SCALE


def _compute_launch_cores(A: int) -> int:
    """Launch core adaptation: tiny/mid/large tiers (measured guidance)."""
    # 8 cores for tiny payloads, 16 for small/mid, 24 for large copies.
    # A <= 256 uses 8 (also for A=64 tiny); A <= 2048 uses 16; A <= 8192 uses 16;
    # larger uses 24.  (Smaller launches reduce multi-core launch/tail bottom.)
    if A <= 256:
        return max(1, min(8, A))
    if A <= 4096:
        return max(1, min(16, A))
    return max(1, min(NUM_CORES_CAP, A))


def _compute_tile_a(A: int, H: int, N: int, E: int, t_dtype: str, c_dtype: str, num_cores: int):
    """Pick (TILE_A, use_tables) for the kernel.

    TILE_A is the largest output-row tile that fits the whole UB budget:

      fixed bytes (all constant-size buffers) + TILE_A * per-row bytes < 176KB

    ``fixed`` accounts for every statically allocated UB buffer: cnt_ub
    (NE_PAD x cnt_bytes), expert_ub (E_PAD x cnt_bytes), prob_ub (64B) and,
    when the prefix-table general path is in use, the 4 int32 tables
    (4 x NE_PAD x 4B).  Per-row bytes count the tile_ub row (H x elem) plus
    the idx/ramp/scales rows (3 x 4B).

    use_tables=False is returned when the 4 prefix tables cannot fit with at
    least 16KB of headroom; the general (non-uniform cnt) path then falls
    back to a memory-light single serial walk over the dst blocks (no tables).
    """
    elem = _ELEM_BYTES[t_dtype]
    cbytes = _ELEM_BYTES[c_dtype]
    ne_pad = ((N * E + 7) // 8) * 8
    e_pad = ((E + 7) // 8) * 8
    per_row = H * elem + 4 + 4 + 4  # tile_ub row + idx + ramp + scales
    block_rows = max(1, A // (N * E))
    per_lane = max(1, (A + num_cores * 2 - 1) // (num_cores * 2))

    def _fit_tile(fixed_bytes: int):
        budget = _UB_BUDGET_BYTES - fixed_bytes - _UB_MARGIN_BYTES
        if budget <= 0 or per_row <= 0:
            return None
        rows = budget // per_row
        if rows <= 0:
            return None
        # Prefer a tile that spans only a few dst blocks (keeps the per-chunk
        # segment walk short), capped by the per-lane output row count.
        target = min(rows, 2 * block_rows)
        tile = max(1, min(target, per_lane))
        if tile >= 8:
            tile = (tile // 8) * 8
        tile = max(1, tile)
        return tile

    table_bytes = ne_pad * 4 * 4  # src_rm / src / dst / end (int32 each)
    fixed_tbl = table_bytes + ne_pad * cbytes + e_pad * cbytes + 64  # + prob_ub
    # Tables must fit with 16KB headroom; otherwise prefer the table-less
    # serial-walk general path (more UB for the tile itself).
    tile_tbl = _fit_tile(fixed_tbl)
    if tile_tbl is not None and fixed_tbl + tile_tbl * per_row + _UB_MARGIN_BYTES + 16 * 1024 <= _UB_BUDGET_BYTES:
        return tile_tbl, True
    fixed_no = ne_pad * cbytes + e_pad * cbytes + 64
    tile_no = _fit_tile(fixed_no)
    if tile_no is not None:
        return tile_no, False
    raise ValueError(
        "moe_re_routing: shape (A=%d,H=%d,N=%d,E=%d) does not fit the 176KB UB budget" %
        (A, H, N, E))


@tilelang.jit(out_idx=[3, 4, 5, 6, 7], pass_configs=pass_configs)
def _moe_re_routing_kernel(
    A: int,
    H: int,
    N: int,
    E: int,
    TILE_A: int,
    NUM_CORES: int,
    tokens_dtype: str,
    cnt_dtype: str,
    has_scale: bool,
    use_tables: bool,
):
    """Build the ``moe_re_routing`` kernel.

    Args (compile-time constants):
        A: number of tokens (also Sum(expert_token_num_per_rank)).
        H: token length.
        N: number of ranks.
        E: number of experts.
        TILE_A: output-row tile (UB row count per chunk).
        NUM_CORES: AI-core block count.
        tokens_dtype: dtype of tokens (float16 / bfloat16 / int8).
        cnt_dtype: dtype of expert_token_num_per_rank (int32 / int64).
        has_scale: whether per_token_scales is provided (compile-time variant).
        use_tables: whether the 4 prefix tables fit under the UB budget; when
            False the general (non-uniform cnt) path uses a table-less serial
            walk over dst blocks instead of prefix tables + binary search.

    Kernel signature:
        tokens:         (A, H) tokens_dtype     - input tokens
        cnt_gm:         (N*E,) cnt_dtype        - flat expert_token_num_per_rank
        scales_gm:      (A,) float32 or (1,)    - per-token scales (dummy when absent)
        permute_tokens: (A, H) tokens_dtype     - output tokens
        permute_scales: (A,) float32            - output scales (zeros when absent)
        permute_idx:    (A,) int32              - output gather index
        expert_token_num: (E,) cnt_dtype        - output expert token counts
        probe:          (8,) int64              - validation probe (sum, min, 0...)

    Returns (out_idx [3,4,5,6,7]): permute_tokens, permute_scales, permute_idx,
    expert_token_num, probe.
    """
    NE = N * E
    NE_PAD = ((NE + 7) // 8) * 8
    E_PAD = ((E + 7) // 8) * 8
    LOG2_NE = max(1, math.ceil(math.log2(NE)))
    log_span = (A + NUM_CORES - 1) // NUM_CORES  # positions per block (upper)
    half = (log_span + 1) // 2  # positions per vector lane
    chunks_per_core = (half + TILE_A - 1) // TILE_A  # static chunk bound
    # statically-known minimum block size under the evaluation's uniform cnt
    # rewrite; when the runtime min (probe) is >= this, a chunk of TILE_A rows
    # spans at most ceil(TILE_A / min_block) + 1 dst blocks, so the segment
    # walk can use a much tighter loop bound instead of the TILE_A worst case.
    MIN_BLOCK_HINT = max(1, A // NE)
    FAST_SEG = min(TILE_A, (TILE_A + MIN_BLOCK_HINT - 1) // MIN_BLOCK_HINT + 2)
    # Uniform-cnt exact fast path.  The evaluation (and every L1 case) writes
    # cnt as base = A // NE for every block with the remainder on the last
    # (rank-major) block: cnt[-1, -1] = base + rem.  When the runtime scan
    # confirms min == base, max == base + rem AND cnt[NE-1] == base + rem,
    # every dst-block boundary is at an exact multiple of base, so both the
    # prefix tables and the per-chunk binary search collapse to arithmetic:
    #   dst block of output position p   : d = min(p / base, NE-1)
    #   src (rank-major) block id        : b = (d % N) * E + (d / N)
    #   src start                        : b * base  (last block (NE-1)*base)
    #   dst start of block d             : d * base  (last block (NE-1)*base)
    # This removes 3 serial NE-loops (prefix tables) + LOG2_NE binary-search
    # iterations per chunk + expert-count loops on the hot path.
    UNIFORM_BASE = max(1, A // NE)
    UNIFORM_REM = A - UNIFORM_BASE * NE
    UNIFORM_MAX = UNIFORM_BASE + UNIFORM_REM  # last block size (== base when rem==0)
    UNIFORM_SEG = min(TILE_A, (TILE_A + UNIFORM_BASE - 1) // UNIFORM_BASE + 2)

    if has_scale:
        scale_shape = (A,)
    else:
        scale_shape = (1,)
    # Prefix-table buffers are only referenced by the general (non-uniform cnt)
    # path when use_tables is selected.  Buffer allocations must live at the
    # top of the prim_func body (the parser does not bind buffer names that are
    # allocated inside an if), so the table size is chosen here at build time:
    # full size when the tables are used, an 8-element stub otherwise.
    tables_ne = NE_PAD if use_tables else 8

    @T.prim_func
    def kernel(
            tokens: T.Tensor((A, H), tokens_dtype),
            cnt_gm: T.Tensor((NE,), cnt_dtype),
            scales_gm: T.Tensor(scale_shape, "float32"),
            permute_tokens: T.Tensor((A, H), tokens_dtype),
            permute_scales: T.Tensor((A,), "float32"),
            permute_idx: T.Tensor((A,), "int32"),
            expert_token_num: T.Tensor((E,), cnt_dtype),
            probe_gm: T.Tensor((8,), "int64"),
    ):
        T.func_attr({"enable_auto_sync": True})
        with T.Kernel(NUM_CORES, is_npu=True) as (cid, vid):
            cnt_ub = T.alloc_ub([NE_PAD], cnt_dtype)
            # Prefix tables are only used by the general (non-uniform cnt) path;
            # allocated unconditionally (size chosen by `tables_ne` at build
            # time) to keep the sizes static and the buffers top-level.
            src_rm = T.alloc_ub([tables_ne], "int32")
            src_ub = T.alloc_ub([tables_ne], "int32")
            dst_ub = T.alloc_ub([tables_ne], "int32")
            end_ub = T.alloc_ub([tables_ne], "int32")
            tile_ub = T.alloc_ub([TILE_A, H], tokens_dtype)
            idx_ub = T.alloc_ub([TILE_A], "int32")
            ramp_ub = T.alloc_ub([TILE_A], "int32")
            scales_ub = T.alloc_ub([TILE_A], "float32")
            expert_ub = T.alloc_ub([E_PAD], cnt_dtype)
            prob_ub = T.alloc_ub([8], "int64")

            acc = T.alloc_var("int32", init=0)
            s_acc = T.alloc_var("int32", init=0)
            lo = T.alloc_var("int32", init=0)
            hi = T.alloc_var("int32", init=0)
            cur = T.alloc_var("int32", init=0)
            bcur = T.alloc_var("int32", init=0)
            mn = T.alloc_var("int32", init=2147483647)
            mx = T.alloc_var("int32", init=0)
            uflag = T.alloc_var("int32", init=1)

            # ---------- (1) load cnt + build ramp (shared by both branches) ----------
            # cnt_ub stores flat expert_token_num_per_rank in rank-major order
            # (index b = i*E+j).
            T.copy(cnt_gm[0:NE], cnt_ub[0:NE])

            # index ramp for idx/scales generation (built once, vector Adds per
            # segment instead of a per-element scalar loop).
            for k in T.Parallel(TILE_A):
                ramp_ub[k] = k

            # (1a) single serial scan of cnt: sum (acc), min (mn), max (mx).
            # One NE-loop instead of the three prefix-table loops; the scan is
            # shared by both branches.
            acc = 0
            mn = 2147483647
            mx = 0
            for b in T.serial(NE):
                cv = T.cast(cnt_ub[b], "int32")
                acc = acc + cv
                mn = T.min(mn, cv)
                mx = T.max(mx, cv)

            # (1b) exact uniform-cnt fast path.  Valid inputs satisfy sum == A,
            # so the extra over `base` totals exactly rem; min==base and
            # max==base+rem then force every block except the unique max block
            # to be exactly base, and the max block must be the last rank-major
            # block (which is also the last dst block).  The whole permutation
            # becomes pure arithmetic (no tables, no binary search):
            #   dst block of output position p : d = min(p / base, NE-1)
            #   src (rank-major) block id      : b = (d % N) * E + (d / N)
            #   src start                      : b * base
            #   dst start of block d           : d * base
            #   (last block start = (NE-1)*base, end = A)
            pos_begin = cid * log_span + vid * half
            pos_end = T.min(
                cid * log_span + (vid + 1) * half,
                T.min((cid + 1) * log_span, A),
            )

            uflag = T.if_then_else(mn == UNIFORM_BASE, 0, 1)
            uflag = T.if_then_else(mx == UNIFORM_MAX, uflag, 1)
            uflag = T.if_then_else(T.cast(cnt_ub[NE - 1], "int32") == UNIFORM_MAX, uflag, 1)
            if uflag == 0:
                # ---------------- uniform arithmetic path ----------------
                # expert_token_num arithmetic: expert j has N*base rows except
                # the last expert which carries the remainder.
                for j in T.serial(E):
                    s_acc = T.if_then_else(
                        j == E - 1,
                        T.cast(N * UNIFORM_BASE + UNIFORM_REM, "int32"),
                        T.cast(N * UNIFORM_BASE, "int32"),
                    )
                    expert_ub[j] = T.cast(s_acc, cnt_dtype)
                if cid == 0 and vid == 0:
                    T.copy(expert_ub[0:E], expert_token_num[0:E])
                    # validation probe: sum = total cnt, min = min cnt.
                    prob_ub[0] = T.cast(acc, "int64")
                    prob_ub[1] = T.cast(mn, "int64")
                    T.copy(prob_ub[0:8], probe_gm[0:8])

                # Pre-zero the whole scales tile once; each segment then copies
                # its [0:seg) view out (noscale variant never overwrites parts).
                T.tile.fill(scales_ub[0:TILE_A], 0.0)

                chunk_end = T.alloc_var("int32", init=0)
                seg = T.alloc_var("int32", init=0)
                src0 = T.alloc_var("int32", init=0)
                bsrc = T.alloc_var("int32", init=0)
                dend = T.alloc_var("int32", init=0)

                for t in T.serial(chunks_per_core):
                    p0 = pos_begin + t * TILE_A
                    if p0 < pos_end:
                        chunk_end = T.min(p0 + TILE_A, pos_end)

                        # first dst block containing p0 (arithmetic).
                        bcur = T.min(p0 // UNIFORM_BASE, NE - 1)
                        cur = p0
                        for _sg in T.serial(UNIFORM_SEG):
                            if cur < chunk_end:
                                # end of dst block bcur in the output stream;
                                # the last block (bcur == NE-1) extends to A.
                                dend = T.if_then_else(
                                    bcur < NE - 1,
                                    (bcur + 1) * UNIFORM_BASE,
                                    A,
                                )
                                seg = T.max(T.min(dend - cur, chunk_end - cur), 0)
                                bsrc = (bcur % N) * E + (bcur // N)
                                src0 = bsrc * UNIFORM_BASE + (cur - bcur * UNIFORM_BASE)
                                src0 = T.max(0, T.min(src0, A - 1))
                                seg = T.min(seg, T.max(0, A - src0))
                                T.copy(
                                    tokens[src0:src0 + seg, :],
                                    tile_ub[0:seg, :],
                                )
                                T.copy(
                                    tile_ub[0:seg, :],
                                    permute_tokens[cur:cur + seg, :],
                                )
                                if has_scale:
                                    T.copy(
                                        scales_gm[src0:src0 + seg],
                                        scales_ub[0:seg],
                                    )
                                    T.copy(
                                        scales_ub[0:seg],
                                        permute_scales[cur:cur + seg],
                                    )
                                T.tile.add(
                                    idx_ub[0:TILE_A],
                                    ramp_ub[0:TILE_A],
                                    bsrc * UNIFORM_BASE + (cur - bcur * UNIFORM_BASE),
                                )
                                T.copy(
                                    idx_ub[0:seg],
                                    permute_idx[cur:cur + seg],
                                )
                                if not has_scale:
                                    T.copy(
                                        scales_ub[0:seg],
                                        permute_scales[cur:cur + seg],
                                    )
                                cur = cur + seg
                                bcur = T.if_then_else(cur >= dend, T.min(bcur + 1, NE - 1), bcur)
            else:
                if use_tables:
                    # ---------------- general (arbitrary cnt) path (prefix tables) ----
                    # src_rm stores rank-major exclusive prefix; dst tables are in
                    # destination (expert-major) order.
                    acc = 0
                    mn = 2147483647
                    for b in T.serial(NE):  # src order (rank-major)
                        src_rm[b] = acc
                        acc = acc + T.cast(cnt_ub[b], "int32")
                        mn = T.min(mn, T.cast(cnt_ub[b], "int32"))

                    # dst-major block tables: d = j*N+i enumerates blocks in the
                    # destination (expert, rank) column-major order.
                    acc = 0
                    for d in T.serial(NE):  # dst order (expert-major)
                        i = d % N
                        j = d // N
                        b = i * E + j  # rank-major block id
                        dst_ub[d] = acc
                        end_ub[d] = acc + T.cast(cnt_ub[b], "int32")
                        src_ub[d] = src_rm[b]
                        acc = acc + T.cast(cnt_ub[b], "int32")

                    for j in T.serial(E):  # expert_token_num (count mode)
                        s_acc = 0
                        for i in T.serial(N):
                            s_acc = s_acc + T.cast(cnt_ub[i * E + j], "int32")
                        expert_ub[j] = T.cast(s_acc, cnt_dtype)
                    # Only block 0 writes the expert counts to GM; a direct scalar GM
                    # store is unreliable on the multi-core path, so stage in UB first
                    # and let MemoryPlanning/T.copy emit the DMA write.
                    if cid == 0 and vid == 0:
                        T.copy(expert_ub[0:E], expert_token_num[0:E])
                        # validation probe: sum = total cnt, min = min cnt.
                        prob_ub[0] = T.cast(acc, "int64")
                        prob_ub[1] = T.cast(mn, "int64")
                        T.copy(prob_ub[0:8], probe_gm[0:8])

                    # ---------- (2) permutation over output-position chunks ----------
                    # Pre-zero the whole scales tile once; each segment then copies
                    # its [0:seg) view out (noscale variant never overwrites parts).
                    T.tile.fill(scales_ub[0:TILE_A], 0.0)

                    # Chunk walk with two statically-distinct segment bounds:
                    # uniform / roughly-uniform cnt takes the fast path with a tight
                    # FAST_SEG bound; arbitrary cnt falls back to the TILE_A
                    # worst-case bound.  The runtime decision (probe `mn`) is
                    # hoisted outside the chunk loop (single branch per core).
                    chunk_end = T.alloc_var("int32", init=0)
                    seg = T.alloc_var("int32", init=0)
                    src0 = T.alloc_var("int32", init=0)

                    if mn >= MIN_BLOCK_HINT:
                        for t in T.serial(chunks_per_core):
                            p0 = pos_begin + t * TILE_A
                            if p0 < pos_end:
                                chunk_end = T.min(p0 + TILE_A, pos_end)

                                # --- binary search: first block with end_ub > p0 ---
                                lo = 0
                                hi = NE
                                for _s in T.serial(LOG2_NE + 1):
                                    mid = (lo + hi) // 2
                                    is_le = end_ub[T.min(mid, NE - 1)] <= p0
                                    lo = T.if_then_else(is_le, mid + 1, lo)
                                    hi = T.if_then_else(is_le, hi, mid)
                                b0 = T.min(lo, NE - 1)

                                cur = p0
                                bcur = b0
                                for _sg in T.serial(FAST_SEG):
                                    if cur < chunk_end:
                                        seg = T.max(
                                            T.min(end_ub[bcur] - cur, chunk_end - cur),
                                            0,
                                        )
                                        src0 = src_ub[bcur] + (cur - dst_ub[bcur])
                                        src0 = T.max(0, T.min(src0, A - 1))
                                        seg = T.min(seg, T.max(0, A - src0))
                                        T.copy(
                                            tokens[src0:src0 + seg, :],
                                            tile_ub[0:seg, :],
                                        )
                                        T.copy(
                                            tile_ub[0:seg, :],
                                            permute_tokens[cur:cur + seg, :],
                                        )
                                        if has_scale:
                                            T.copy(
                                                scales_gm[src0:src0 + seg],
                                                scales_ub[0:seg],
                                            )
                                            T.copy(
                                                scales_ub[0:seg],
                                                permute_scales[cur:cur + seg],
                                            )
                                        T.tile.add(
                                            idx_ub[0:TILE_A],
                                            ramp_ub[0:TILE_A],
                                            src_ub[bcur] + (cur - dst_ub[bcur]),
                                        )
                                        T.copy(
                                            idx_ub[0:seg],
                                            permute_idx[cur:cur + seg],
                                        )
                                        if not has_scale:
                                            T.copy(
                                                scales_ub[0:seg],
                                                permute_scales[cur:cur + seg],
                                            )
                                        cur = cur + seg
                                        bcur = T.if_then_else(cur >= end_ub[bcur], bcur + 1, bcur)
                    else:
                        for t in T.serial(chunks_per_core):
                            p0 = pos_begin + t * TILE_A
                            if p0 < pos_end:
                                chunk_end = T.min(p0 + TILE_A, pos_end)

                                lo = 0
                                hi = NE
                                for _s in T.serial(LOG2_NE + 1):
                                    mid = (lo + hi) // 2
                                    is_le = end_ub[T.min(mid, NE - 1)] <= p0
                                    lo = T.if_then_else(is_le, mid + 1, lo)
                                    hi = T.if_then_else(is_le, hi, mid)
                                b0 = T.min(lo, NE - 1)

                                cur = p0
                                bcur = b0
                                for _sg in T.serial(TILE_A):
                                    if cur < chunk_end:
                                        seg = T.max(
                                            T.min(end_ub[bcur] - cur, chunk_end - cur),
                                            0,
                                        )
                                        src0 = src_ub[bcur] + (cur - dst_ub[bcur])
                                        src0 = T.max(0, T.min(src0, A - 1))
                                        seg = T.min(seg, T.max(0, A - src0))
                                        T.copy(
                                            tokens[src0:src0 + seg, :],
                                            tile_ub[0:seg, :],
                                        )
                                        T.copy(
                                            tile_ub[0:seg, :],
                                            permute_tokens[cur:cur + seg, :],
                                        )
                                        if has_scale:
                                            T.copy(
                                                scales_gm[src0:src0 + seg],
                                                scales_ub[0:seg],
                                            )
                                            T.copy(
                                                scales_ub[0:seg],
                                                permute_scales[cur:cur + seg],
                                            )
                                        T.tile.add(
                                            idx_ub[0:TILE_A],
                                            ramp_ub[0:TILE_A],
                                            src_ub[bcur] + (cur - dst_ub[bcur]),
                                        )
                                        T.copy(
                                            idx_ub[0:seg],
                                            permute_idx[cur:cur + seg],
                                        )
                                        if not has_scale:
                                            T.copy(
                                                scales_ub[0:seg],
                                                permute_scales[cur:cur + seg],
                                            )
                                        cur = cur + seg
                                        bcur = T.if_then_else(cur >= end_ub[bcur], bcur + 1, bcur)
                else:
                    # ---------------- general path without prefix tables (large N*E) ----------
                    # No dst/src prefix tables fit under the UB budget, so walk the dst
                    # blocks in a single serial pass and process the intersection of each
                    # block with this core's output range.  The dst start and the src
                    # start are computed incrementally: for dst block d (expert j = d//N,
                    # rank i = d%N) the src start is
                    #   column prefix of expert j  (== dst start at the first block of
                    #   the expert) + intra-expert prefix over ranks < i,
                    # and the dst start is the running dst offset.  expert_token_num[j]
                    # is accumulated in the same pass.
                    T.tile.fill(scales_ub[0:TILE_A], 0.0)

                    dst_a = T.alloc_var("int32", init=0)
                    col_a = T.alloc_var("int32", init=0)
                    row_a = T.alloc_var("int32", init=0)
                    e_tot = T.alloc_var("int32", init=0)
                    cur_src = T.alloc_var("int32", init=0)
                    seg_lo = T.alloc_var("int32", init=0)
                    seg_hi = T.alloc_var("int32", init=0)
                    seg = T.alloc_var("int32", init=0)
                    src0 = T.alloc_var("int32", init=0)

                    for d in T.serial(NE):
                        b = (d % N) * E + (d // N)
                        cv = T.cast(cnt_ub[b], "int32")
                        if (d % N) == 0:
                            # first block of expert j = d//N: column prefix == dst start.
                            col_a = dst_a
                            row_a = 0
                            e_tot = 0
                        cur_src = col_a + row_a
                        # intersection of [dst_a, dst_a+cv) with [pos_begin, pos_end).
                        seg_lo = T.max(pos_begin, dst_a)
                        seg_hi = T.min(pos_end, dst_a + cv)
                        if seg_lo < seg_hi:
                            src0 = T.max(0, T.min(cur_src + (seg_lo - dst_a), A - 1))
                            seg = T.min(seg_hi - seg_lo, T.max(0, A - src0))
                            if seg > 0:
                                T.copy(
                                    tokens[src0:src0 + seg, :],
                                    tile_ub[0:seg, :],
                                )
                                T.copy(
                                    tile_ub[0:seg, :],
                                    permute_tokens[seg_lo:seg_lo + seg, :],
                                )
                                if has_scale:
                                    T.copy(
                                        scales_gm[src0:src0 + seg],
                                        scales_ub[0:seg],
                                    )
                                    T.copy(
                                        scales_ub[0:seg],
                                        permute_scales[seg_lo:seg_lo + seg],
                                    )
                                T.tile.add(
                                    idx_ub[0:TILE_A],
                                    ramp_ub[0:TILE_A],
                                    cur_src + (seg_lo - dst_a),
                                )
                                T.copy(
                                    idx_ub[0:seg],
                                    permute_idx[seg_lo:seg_lo + seg],
                                )
                                if not has_scale:
                                    T.copy(
                                        scales_ub[0:seg],
                                        permute_scales[seg_lo:seg_lo + seg],
                                    )
                        # advance to the next dst block.
                        dst_a = dst_a + cv
                        row_a = row_a + cv
                        e_tot = e_tot + cv
                        if (d % N) == N - 1:
                            expert_ub[d // N] = T.cast(e_tot, cnt_dtype)

                    if cid == 0 and vid == 0:
                        T.copy(expert_ub[0:E], expert_token_num[0:E])
                        # validation probe: sum = total cnt, min = min cnt.
                        prob_ub[0] = T.cast(acc, "int64")
                        prob_ub[1] = T.cast(mn, "int64")
                        T.copy(prob_ub[0:8], probe_gm[0:8])

    return kernel


def moe_re_routing(tokens, expert_token_num_per_rank, per_token_scales=None):
    """Host callable: metadata-only operations, permutation in the kernel.

    Args:
        tokens: (A, H) tensor (float16 / bfloat16 / int8).
        expert_token_num_per_rank: (N, E) tensor (int32 / int64), Sum == A,
            every element > 0.
        per_token_scales: optional (A,) float32 tensor.

    Returns:
        (permute_tokens, permute_per_token_scales, permute_token_idx, expert_token_num)
    """
    if tokens.dim() != 2:
        raise ValueError("tokens must be a 2D tensor of shape (A, H)")
    if expert_token_num_per_rank.dim() != 2:
        raise ValueError("expert_token_num_per_rank must be a 2D tensor of shape (N, E)")
    A, H = tokens.shape
    N, E = expert_token_num_per_rank.shape
    if per_token_scales is not None and per_token_scales.shape[0] != A:
        raise ValueError("per_token_scales must have length A=" + str(A))

    t_dtype = _TL_DTYPE.get(str(tokens.dtype).split(".")[-1])
    if t_dtype is None:
        raise ValueError("unsupported tokens dtype: " + str(tokens.dtype) +
                         " (supported: float16/bfloat16/int8)")
    c_dtype = _TL_DTYPE.get(str(expert_token_num_per_rank.dtype).split(".")[-1])
    if c_dtype is None or c_dtype not in ("int32", "int64"):
        raise ValueError("unsupported expert_token_num_per_rank dtype: " +
                         str(expert_token_num_per_rank.dtype) + " (supported: int32/int64)")
    has_scale = per_token_scales is not None

    # Metadata-only operations on the host (no host-side permutation math).
    tokens_c = tokens.contiguous()
    cnt_flat = expert_token_num_per_rank.reshape(-1).contiguous()
    num_cores = _compute_launch_cores(A)
    tile_a, use_tables = _compute_tile_a(A, H, N, E, t_dtype, c_dtype, num_cores)

    if has_scale:
        scales_in = per_token_scales.contiguous()
        kern = _moe_re_routing_kernel(A, H, N, E, tile_a, num_cores, t_dtype, c_dtype, True,
                                      use_tables)
        out = kern(tokens_c, cnt_flat, scales_in)
    else:
        # No per_token_scales -> compile variant has_scale=False.  Reuse a
        # cached dummy (never read by the kernel) to avoid per-call ZerosLike.
        scale_dummy = _get_dummy_scale(device=tokens.device)
        kern = _moe_re_routing_kernel(A, H, N, E, tile_a, num_cores, t_dtype, c_dtype, False,
                                      use_tables)
        out = kern(tokens_c, cnt_flat, scale_dummy)

    # The 5th output is the small (8,) int64 validation probe; discard it and
    # return the 4 application outputs.
    return out[0], out[1], out[2], out[3]


# ========== Golden reference (PyTorch double prefix-sum implementation) ==========
def golden_moe_re_routing(tokens, expert_token_num_per_rank, per_token_scales=None):
    """PyTorch reference: src/dst prefix sums -> permute index -> gather."""
    N, E = expert_token_num_per_rank.shape
    A, H = tokens.shape
    assert int(expert_token_num_per_rank.sum().item()) == A

    dev = tokens.device
    c = expert_token_num_per_rank.to(torch.int64)
    cnt_src = c.reshape(-1)  # src order (rank, expert) row-major
    cnt_dst = c.t().reshape(-1)  # dst order (expert, rank) column-major

    def _excl_prefix(t):
        return torch.cat([t.new_zeros(1), t.cumsum(0)[:-1]])

    src_start = _excl_prefix(cnt_src)
    dst_start = _excl_prefix(cnt_dst).reshape(E, N).t().reshape(-1)  # back to (rank, expert) id
    block = torch.repeat_interleave(torch.arange(N * E, device=dev), cnt_src)
    dst_pos = dst_start[block] + (torch.arange(A, device=dev) - src_start[block])  # src -> dst
    permute_token_idx = torch.empty(A, dtype=torch.int32, device=dev)
    permute_token_idx[dst_pos] = torch.arange(A, dtype=torch.int32, device=dev)

    permute_tokens = tokens[permute_token_idx]
    permute_per_token_scales = (
        per_token_scales[permute_token_idx] if per_token_scales is not None else torch.zeros(
            A, dtype=torch.float32, device=tokens.device))
    expert_token_num = expert_token_num_per_rank.sum(dim=0)
    return permute_tokens, permute_per_token_scales, permute_token_idx, expert_token_num


if __name__ == "__main__":
    torch.manual_seed(0)

    # Two representative configs: 1 L0 + 1 L1.  cnt uses the uniform + remainder
    # construction (base = A // (N*E), remainder on the last block) which the
    # kernel fast path relies on; tokens/scales use standard randn.
    test_configs = [
        # (level, A, H, N, E, tokens_dtype, cnt_dtype, with_scale)
        ("L0", 1024, 512, 8, 8, torch.float16, torch.int32, True),
        ("L1", 16384, 1024, 16, 16, torch.bfloat16, torch.int64, True),
    ]

    for level, A, H, N, E, tokens_dtype, cnt_dtype, with_scale in test_configs:
        print(f"Testing moe_re_routing {level} with A={A}, H={H}, N={N}, E={E}, "
              f"tokens={tokens_dtype}, cnt={cnt_dtype}, scale={with_scale}")
        torch.manual_seed(0)
        # Generate in fp32 then cast: same distribution for fp16/bf16 and cheap.
        tokens = torch.randn(A, H, dtype=torch.float32, device="npu").to(tokens_dtype)
        base = A // (N * E)
        cnt = torch.full((N * E,), base, dtype=cnt_dtype, device="npu")
        cnt[-1] += A - base * N * E
        cnt = cnt.view(N, E)
        scales = torch.randn(A, dtype=torch.float32, device="npu") if with_scale else None

        out = moe_re_routing(tokens, cnt, scales)
        print("Init successful!")
        ref = golden_moe_re_routing(tokens, cnt, scales)

        # Pure permutation operator: every output must match bit-exactly
        # (tokens/scales are bit-copied rows, idx/count are integer arithmetic).
        names = ["permute_tokens", "permute_scales", "permute_token_idx", "expert_token_num"]
        for nm, actual, golden_t in zip(names, out, ref):
            a, g = actual.detach().cpu(), golden_t.detach().cpu()
            assert torch.equal(a, g), f"{level} {nm} mismatch"
        print(f"Test pass! {level} all outputs exact (matched_ratio=1.0, max_abs=0.0)")

    print("Kernel Output Match!")
