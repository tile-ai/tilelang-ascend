import math
import torch
import tilelang
from tilelang import language as T
from ._common import PASS_CONFIGS

import linecache
import inspect as _inspect

_src_file = _inspect.getsourcefile(_inspect.currentframe()) or __file__
try:
    with open(_src_file, 'r') as _f:
        _src_lines = _f.readlines()
    linecache.cache[_src_file] = (len(_src_lines), None, _src_lines, _src_file)
except (OSError, IOError):
    pass


def _copy_rows(dst_buf, dst_off, src_buf, src_off, rows, cols, src_stride, dst_stride):
    """One strided UB->UB copy: dst[i*dst_stride+j] = src[i*src_stride+j], i<rows, j<cols (elements).

    Emits tl::ascend::copy_ub_to_ub<dtype, dtype, rows*cols> directly; common.h's
    7-param helper collapses it into a single DataCopyParams burst copy when row
    width and both gaps are whole 32B blocks (V-pipe, same intrinsic family as
    the old per-row loop).
    """
    return T.call_extern(
        "handle",
        f"tl::ascend::copy_ub_to_ub<float, float, {int(rows) * int(cols)}>",
        src_buf.access_ptr("r", offset=src_off),
        dst_buf.access_ptr("w", offset=dst_off),
        rows, cols, src_stride, rows, cols, dst_stride)


_kernel_cache = {}
NUM_CORES = 40   # iter2-adopted baseline
# iter6 (wave model): device has 40 physical AIVs (vector_core_num=40).
#   NUM_CORES=40 x VEC_NUM=2 = 80 AIV sub-blocks = exactly 2 full waves.
#   NUM_CORES=20 x VEC_NUM=2 = 40 sub-blocks  = exactly 1 full wave.
# 1-wave saves the inter-wave ramp/MTE contention (case4 -20.7%, case6 -12.0%),
# but forces outer_iters 1->2 for mid-size B/HC combos (case5 +3.2%, case12
# +108% -- outer penalty dominates). Route: use 20 when outer stays 1 at
# launch=20, or when launch=40 is already multi-pack (1-wave strictly better);
# otherwise keep 40 (outer=1 @ 2 waves).
VEC_NUM = 2      # rev3: keep (AIV hardware parallelism 2.25x, cannot abandon)


def _route_num_cores(B, HC, HC_pad):
    pack_max = _compute_pack(HC, HC_pad)

    def per_vid_of(nc):
        launch = min(nc, B)
        per_core = (B + launch - 1) // launch
        return (per_core + VEC_NUM - 1) // VEC_NUM

    if per_vid_of(20) <= pack_max or per_vid_of(40) > pack_max:
        return 20
    return 40


def _pad_hc(hc):
    return max(((hc + 7) // 8) * 8, 8)


def _compute_pack(HC, HC_pad):
    ub_budget = 170 * 1024
    # 7 PACK-sized buffers: lbuf(2x), out_buf(2x), mat, bcast, col_bcast
    # + row_red(HC), col_sum(HC_pad)
    per_pack = 7 * (HC * HC_pad * 4) + HC * 4 + HC_pad * 4
    pack = ub_budget // per_pack
    # rev3: cap at 255//HC so PACK_HC ≤ 255, avoiding split reduce alignment bug
    return max(1, min(pack, 255 // HC))


@tilelang.jit(out_idx=[1])
def _sinkhorn_kernel(B, HC, HC_pad, PACK, PACK_HC, B_HC, iter_step, eps, num_cores):
    cd = "float32"
    launch = min(num_cores, B)
    per_core = T.ceildiv(B, launch)
    per_vid = T.ceildiv(per_core, VEC_NUM)
    outer_iters = T.ceildiv(per_vid, PACK)

    @T.macro
    def init_flag():
        T.set_flag("mte3", "mte2", 0)
        T.set_flag("mte3", "mte2", 1)

    @T.macro
    def clear_flag():
        T.wait_flag("mte3", "mte2", 0)
        T.wait_flag("mte3", "mte2", 1)

    # Split narrow reduce when PACK_HC > 255 (WholeReduceSum repeatTime limit)
    REDUCE_CHUNK = min(PACK_HC, 255)
    REDUCE_SPLITS = (PACK_HC + REDUCE_CHUNK - 1) // REDUCE_CHUNK
    LAST_CHUNK = PACK_HC - (REDUCE_SPLITS - 1) * REDUCE_CHUNK

    @T.macro
    def row_reduce_sum(buf, out):
        if REDUCE_SPLITS == 1:
            T.reduce_sum(buf, out, dim=-1, real_shape=[PACK_HC, HC])
        else:
            for s in T.serial(REDUCE_SPLITS - 1):
                base = s * REDUCE_CHUNK
                T.reduce_sum(buf[base : base + REDUCE_CHUNK, 0:HC_pad],
                             out[base : base + REDUCE_CHUNK, 0:1],
                             dim=-1, real_shape=[REDUCE_CHUNK, HC])
            base = (REDUCE_SPLITS - 1) * REDUCE_CHUNK
            T.reduce_sum(buf[base : base + LAST_CHUNK, 0:HC_pad],
                         out[base : base + LAST_CHUNK, 0:1],
                         dim=-1, real_shape=[LAST_CHUNK, HC])

    @T.macro
    def row_reduce_max(buf, out):
        if REDUCE_SPLITS == 1:
            T.reduce_max(buf, out, dim=-1, real_shape=[PACK_HC, HC])
        else:
            for s in T.serial(REDUCE_SPLITS - 1):
                base = s * REDUCE_CHUNK
                T.reduce_max(buf[base : base + REDUCE_CHUNK, 0:HC_pad],
                             out[base : base + REDUCE_CHUNK, 0:1],
                             dim=-1, real_shape=[REDUCE_CHUNK, HC])
            base = (REDUCE_SPLITS - 1) * REDUCE_CHUNK
            T.reduce_max(buf[base : base + LAST_CHUNK, 0:HC_pad],
                         out[base : base + LAST_CHUNK, 0:1],
                         dim=-1, real_shape=[LAST_CHUNK, HC])

    @T.prim_func
    def main(
        comb: T.Tensor((B_HC, HC), cd),
        comb_out: T.Tensor((B_HC, HC), cd),
    ):
        T.func_attr({"enable_auto_sync": False})
        with T.Kernel(launch, is_npu=True) as (cid, vid):
            lbuf = T.alloc_ub((2, PACK_HC, HC_pad), cd)
            out_buf = T.alloc_ub((2, PACK_HC, HC_pad), cd)
            mat = T.alloc_ub((PACK_HC, HC_pad), cd)
            row_red = T.alloc_ub((PACK_HC, 1), cd)
            bcast = T.alloc_ub((PACK_HC, HC_pad), cd)
            col_sum = T.alloc_ub((PACK, HC_pad), cd)
            col_bcast = T.alloc_ub((PACK_HC, HC_pad), cd)

            with T.Scope("V"):
                init_flag()

                # === Prefetch: GM → lbuf[0] (merged 2D copy for full packs) ===
                base_0 = cid * per_core + vid * per_vid
                T.wait_flag("mte3", "mte2", 0)
                if base_0 + PACK <= B:
                    T.copy(comb[base_0 * HC : (base_0 + PACK) * HC, 0:HC_pad], lbuf[0, 0:PACK_HC, 0:HC_pad], pad_value=-float("inf"))
                else:
                    for p in T.serial(PACK):
                        bid = base_0 + p
                        safe_bid = T.min(bid, B - 1)
                        T.copy(comb[safe_bid * HC : (safe_bid + 1) * HC, 0:HC_pad], lbuf[0, p * HC : (p + 1) * HC, 0:HC_pad], pad_value=-float("inf"))
                T.set_flag("mte2", "v", 0)

                # === Main body ===
                for i in T.serial(outer_iters - 1):
                    cur = i % 2
                    nxt = 1 - cur
                    base_cur = cid * per_core + vid * per_vid + i * PACK
                    base_nxt = cid * per_core + vid * per_vid + (i + 1) * PACK

                    # 预取下一批 (MTE2, merged for full packs)
                    T.wait_flag("mte3", "mte2", nxt)
                    if base_nxt + PACK <= B:
                        T.copy(comb[base_nxt * HC : (base_nxt + PACK) * HC, 0:HC_pad], lbuf[nxt, 0:PACK_HC, 0:HC_pad], pad_value=-float("inf"))
                    else:
                        for p in T.serial(PACK):
                            bid = base_nxt + p
                            safe_bid = T.min(bid, B - 1)
                            T.copy(comb[safe_bid * HC : (safe_bid + 1) * HC, 0:HC_pad], lbuf[nxt, p * HC : (p + 1) * HC, 0:HC_pad], pad_value=-float("inf"))
                    T.set_flag("mte2", "v", nxt)

                    # 消费当前批: lbuf[cur] → mat (V pipe 转置, 每 r 一条 strided copy)
                    T.wait_flag("mte2", "v", cur)
                    for r in T.serial(HC):
                        _copy_rows(mat, r * PACK * HC_pad, lbuf, (cur * PACK_HC + r) * HC_pad,
                                   PACK, HC_pad, HC * HC_pad, HC_pad)

                    # V pipe: row softmax (iter 1)
                    row_reduce_max(mat, row_red)
                    T.tile.broadcast(bcast, row_red)
                    T.tile.sub(mat, mat, bcast)
                    T.tile.exp(mat, mat)
                    row_reduce_sum(mat, row_red)
                    T.tile.broadcast(bcast, row_red)
                    T.tile.div(mat, mat, bcast)
                    T.tile.add(mat, mat, eps)

                    # V pipe: col normalize (iter 1)
                    T.tile.add(col_sum[0:PACK, 0:HC_pad], mat[0:PACK, 0:HC_pad], mat[PACK : 2 * PACK, 0:HC_pad])
                    for r in T.serial(HC - 2):
                        T.tile.add(col_sum[0:PACK, 0:HC_pad], col_sum[0:PACK, 0:HC_pad], mat[(r + 2) * PACK : (r + 3) * PACK, 0:HC_pad])
                    T.tile.add(col_sum[0:PACK, 0:HC_pad], col_sum[0:PACK, 0:HC_pad], eps)
                    for r in T.serial(HC):
                        T.copy(col_sum[0:PACK, 0:HC_pad], col_bcast[r * PACK : (r + 1) * PACK, 0:HC_pad])
                    T.tile.div(mat, mat, col_bcast)

                    # V pipe: iter 2..iter_step
                    for _ in T.serial(iter_step - 1):
                        row_reduce_sum(mat, row_red)
                        T.tile.add(row_red, row_red, eps)
                        T.tile.broadcast(bcast, row_red)
                        T.tile.div(mat, mat, bcast)
                        T.tile.add(col_sum[0:PACK, 0:HC_pad], mat[0:PACK, 0:HC_pad], mat[PACK : 2 * PACK, 0:HC_pad])
                        for r in T.serial(HC - 2):
                            T.tile.add(col_sum[0:PACK, 0:HC_pad], col_sum[0:PACK, 0:HC_pad], mat[(r + 2) * PACK : (r + 3) * PACK, 0:HC_pad])
                        T.tile.add(col_sum[0:PACK, 0:HC_pad], col_sum[0:PACK, 0:HC_pad], eps)
                        for r in T.serial(HC):
                            T.copy(col_sum[0:PACK, 0:HC_pad], col_bcast[r * PACK : (r + 1) * PACK, 0:HC_pad])
                        T.tile.div(mat, mat, col_bcast)

                    # 写回: mat → out_buf[cur] (V pipe, 每 r 一条 strided copy) → GM (MTE3 pipe, 2D merged)
                    if base_cur + PACK <= B:
                        for r in T.serial(HC):
                            _copy_rows(out_buf, (cur * PACK_HC + r) * HC_pad, mat, r * PACK * HC_pad,
                                       PACK, HC_pad, HC_pad, HC * HC_pad)
                        T.set_flag("v", "mte3", cur)
                        T.wait_flag("v", "mte3", cur)
                        T.copy(out_buf[cur, 0:PACK_HC, 0:HC], comb_out[base_cur * HC : (base_cur + PACK) * HC, 0:HC])
                    else:
                        for p in T.serial(PACK):
                            bid = base_cur + p
                            if bid < B:
                                for r in T.serial(HC):
                                    T.copy(mat[r * PACK + p, 0:HC_pad], out_buf[cur, p * HC + r, 0:HC_pad])
                        T.set_flag("v", "mte3", cur)
                        T.wait_flag("v", "mte3", cur)
                        for p in T.serial(PACK):
                            bid = base_cur + p
                            if bid < B:
                                T.copy(out_buf[cur, p * HC : (p + 1) * HC, 0:HC], comb_out[bid * HC : (bid + 1) * HC, 0:HC])
                    T.set_flag("mte3", "mte2", cur)

                # === Epilogue ===
                last = (outer_iters - 1) % 2
                base_last = cid * per_core + vid * per_vid + (outer_iters - 1) * PACK

                T.wait_flag("mte2", "v", last)
                for r in T.serial(HC):
                    _copy_rows(mat, r * PACK * HC_pad, lbuf, (last * PACK_HC + r) * HC_pad,
                               PACK, HC_pad, HC * HC_pad, HC_pad)

                row_reduce_max(mat, row_red)
                T.tile.broadcast(bcast, row_red)
                T.tile.sub(mat, mat, bcast)
                T.tile.exp(mat, mat)
                row_reduce_sum(mat, row_red)
                T.tile.broadcast(bcast, row_red)
                T.tile.div(mat, mat, bcast)
                T.tile.add(mat, mat, eps)

                T.tile.add(col_sum[0:PACK, 0:HC_pad], mat[0:PACK, 0:HC_pad], mat[PACK : 2 * PACK, 0:HC_pad])
                for r in T.serial(HC - 2):
                    T.tile.add(col_sum[0:PACK, 0:HC_pad], col_sum[0:PACK, 0:HC_pad], mat[(r + 2) * PACK : (r + 3) * PACK, 0:HC_pad])
                T.tile.add(col_sum[0:PACK, 0:HC_pad], col_sum[0:PACK, 0:HC_pad], eps)
                for r in T.serial(HC):
                    T.copy(col_sum[0:PACK, 0:HC_pad], col_bcast[r * PACK : (r + 1) * PACK, 0:HC_pad])
                T.tile.div(mat, mat, col_bcast)

                for _ in T.serial(iter_step - 1):
                    row_reduce_sum(mat, row_red)
                    T.tile.add(row_red, row_red, eps)
                    T.tile.broadcast(bcast, row_red)
                    T.tile.div(mat, mat, bcast)
                    T.tile.add(col_sum[0:PACK, 0:HC_pad], mat[0:PACK, 0:HC_pad], mat[PACK : 2 * PACK, 0:HC_pad])
                    for r in T.serial(HC - 2):
                        T.tile.add(col_sum[0:PACK, 0:HC_pad], col_sum[0:PACK, 0:HC_pad], mat[(r + 2) * PACK : (r + 3) * PACK, 0:HC_pad])
                    T.tile.add(col_sum[0:PACK, 0:HC_pad], col_sum[0:PACK, 0:HC_pad], eps)
                    for r in T.serial(HC):
                        T.copy(col_sum[0:PACK, 0:HC_pad], col_bcast[r * PACK : (r + 1) * PACK, 0:HC_pad])
                    T.tile.div(mat, mat, col_bcast)

                if base_last + PACK <= B:
                    for r in T.serial(HC):
                        _copy_rows(out_buf, (last * PACK_HC + r) * HC_pad, mat, r * PACK * HC_pad,
                                   PACK, HC_pad, HC_pad, HC * HC_pad)
                    T.set_flag("v", "mte3", last)
                    T.wait_flag("v", "mte3", last)
                    T.copy(out_buf[last, 0:PACK_HC, 0:HC], comb_out[base_last * HC : (base_last + PACK) * HC, 0:HC])
                else:
                    for p in T.serial(PACK):
                        bid = base_last + p
                        if bid < B:
                            for r in T.serial(HC):
                                T.copy(mat[r * PACK + p, 0:HC_pad], out_buf[last, p * HC + r, 0:HC_pad])
                    T.set_flag("v", "mte3", last)
                    T.wait_flag("v", "mte3", last)
                    for p in T.serial(PACK):
                        bid = base_last + p
                        if bid < B:
                            T.copy(out_buf[last, p * HC : (p + 1) * HC, 0:HC], comb_out[bid * HC : (bid + 1) * HC, 0:HC])
                T.set_flag("mte3", "mte2", last)

                clear_flag()

    return main


def mhc_sinkhorn(
    comb: torch.Tensor, iter_step: int = 20, eps: float = 1e-6
) -> torch.Tensor:
    B, HC, HC2 = comb.shape
    assert HC == HC2

    orig_dtype = comb.dtype
    if orig_dtype != torch.float32:
        comb = comb.to(torch.float32)
    comb_contig = comb if comb.is_contiguous() else comb.contiguous()
    HC_pad = _pad_hc(HC)
    num_cores = _route_num_cores(B, HC, HC_pad)
    launch = min(num_cores, B)
    per_core = (B + launch - 1) // launch
    per_vid = (per_core + VEC_NUM - 1) // VEC_NUM

    # rev3: PACK = min(255//HC (via _compute_pack cap), per_vid)
    # Maximize PACK to amortize V-pipe per-op startup_overhead (rev3 core strategy)
    PACK = min(_compute_pack(HC, HC_pad), per_vid)
    # iter9: split-batch for the 2-wave outer=1 large-PACK regime (case5-like:
    # B=4096, HC=4 -> PACK 52->26, outer 1->2). At NC=40 routing guarantees
    # outer==1 (per_vid(40) <= pack_max by construction), and each wave's
    # leading MTE2 (PACK*HC*4B per sub-block) is serialized at wave start.
    # Halving PACK activates lbuf[0/1] double-buffering: batch-2 MTE2 hides
    # under batch-1 V compute. Only worthwhile when PACK >= 40 (leading MTE2
    # significant, halved PACK still amortizes per-op overhead); small-PACK
    # outer=1 cases regressed +8..25% (iter9: case4/10 at 1-wave, case11/12
    # at 2-wave but PACK=13 with ~740 steady ops/outer for HC>=12).
    # Measured: case5 98.2->91.6us (-6.7%), bit-exact. No other case affected.
    if num_cores == 40 and PACK >= 40:
        PACK = (PACK + 1) // 2
    PACK_HC = PACK * HC
    B_HC = B * HC

    key = ("sinkhorn_2d_idx", B, HC, HC_pad, PACK, PACK_HC, B_HC, iter_step, eps, num_cores)
    if key not in _kernel_cache:
        _kernel_cache[key] = _sinkhorn_kernel(B, HC, HC_pad, PACK, PACK_HC, B_HC, int(iter_step), float(eps), num_cores)
    kernel = _kernel_cache[key]

    # Reshape comb to 2D [B*HC, HC] (free .view(), no data movement)
    comb_2d = comb_contig.view(B_HC, HC)
    comb_out_2d = kernel(comb_2d)
    # Reshape back to 3D and slice
    result = comb_out_2d.view(B, HC, HC)

    if orig_dtype != torch.float32:
        result = result.to(orig_dtype)
    return result
