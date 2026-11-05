"""ROIAlign TileLang implementation. Design & history: see DESIGN.md / debug_log.md."""

import tilelang
from tilelang import language as T
import math
import sys
import torch
import weakref
from collections import OrderedDict

from ._common import CAST_MODE_HIGH2LOW
from ._common import PASS_CONFIGS_MANUAL_SYNC as PASS_CONFIGS

_DTYPE_MAP = {
    torch.float16: "float16",
    torch.float32: "float",
}

_kernel_cache = {}

_SLOT_EAGER_BROKEN = False

_ACLNN_FAILURE_SIGNS = (
    "EZ1013",
    "EZ1012",
    "561103",
    "AclOpKernelInit failed",
    "ERR00100",
    "call aclnn",
    "acl api failed",
    "NPU function error",
)


def _is_aclnn_failure(exc):
    text = f"{type(exc).__name__}: {exc}"
    return any(s in text for s in _ACLNN_FAILURE_SIGNS)


def _mark_slot_broken(exc):
    global _SLOT_EAGER_BROKEN
    if _SLOT_EAGER_BROKEN:
        return
    _SLOT_EAGER_BROKEN = True
    msg = f"{type(exc).__name__}: {exc}"
    if len(msg) > 600:
        msg = msg[:600] + " ...(truncated)"
    print(
        f"[roi_align] WARNING: device-path eager op failed with an "
        f"ACLNN-class error -- slot marked broken, this process degrades "
        f"permanently to the host path (B-lite engine). First failure: "
        f"{msg}",
        file=sys.stderr, flush=True)


_TABLE_CACHE_MAX_ENTRIES = 32
_host_table_cache = OrderedDict()

_MAX_GRID_BLOCKS = 32768


def _grid_capped_launch(factory, N, C, C_BLK, keytpl, args, per_n):
    c_num = (C + 2 * C_BLK - 1) // (2 * C_BLK)
    if N * c_num <= _MAX_GRID_BLOCKS:
        return None
    L = max(_MAX_GRID_BLOCKS // c_num, 1)
    outs = []
    n0 = 0
    while n0 < N:
        lc = min(L, N - n0)
        key = keytpl + (lc,)
        if key not in _kernel_cache:
            _kernel_cache[key] = factory(lc)
        k = _kernel_cache[key]
        sl = slice(n0, n0 + lc)
        cargs = [t[sl] if pn else t for t, pn in zip(args, per_n)]
        outs.append(k(*cargs))
        n0 += lc
    return outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)


_dev_table_cache = OrderedDict()
_PTR_CACHE_MAX_ENTRIES = 64
_dev_ptr_cache = {}


def _remember_ptr(boxes, ptr, mode, data, GH, GW):
    if len(_dev_ptr_cache) > _PTR_CACHE_MAX_ENTRIES:
        for k in [k for k, v in _dev_ptr_cache.items() if v[0]() is None]:
            del _dev_ptr_cache[k]
    _dev_ptr_cache[ptr] = (weakref.ref(boxes), mode, data, GH, GW)


def _to_f64(t):
    out = torch.empty(t.shape, dtype=torch.float64, device=t.device)
    out.copy_(t)
    assert out.dtype == torch.float64, (
        f"_to_f64: expected float64, got {out.dtype} (NPU .double() no-op?)"
    )
    return out


def _pack_tables(tables):
    total = 0
    layout = []
    for t in tables:
        assert t.dtype in (torch.int32, torch.float32)
        n = t.numel()
        off = (total + 7) // 8 * 8
        layout.append((off, n, t.dtype))
        total = off + n
    packed = torch.empty(total, dtype=torch.int32)
    for t, (off, n, _) in zip(tables, layout):
        packed[off : off + n] = t.reshape(-1).view(torch.int32)
    return packed, layout


def _unpack_on_device(packed_dev, layout, u32_slots=()):
    views = []
    for i, (off, n, dt) in enumerate(layout):
        v = packed_dev[off : off + n].view(torch.uint32 if i in u32_slots else dt)
        views.append(v)
    return views


def _est_pw(cand, GW, OW, W_PAD, TGH, N, esize, ROWS, OH, OW_PAD):
    fused = GW * cand * OW
    gh = TGH // OH
    tb = 1 if gh * GW == 16 else 0
    casc = 1 if gh * GW in (64, 256) else 0
    est = (
        2 * cand * ROWS * W_PAD * esize
        + fused * 4 * 2
        + fused * 4 * 2
        + fused * 4 * 2
        + (fused * 2 if esize == 2 else 0)
        + fused * 4
        + fused * 4
        + fused * 4 * (1 + tb)
        + cand * OW * 4 * tb
        + fused * 4 * 4
        + 4 * 8 * cand * OW * 4 * casc
        + cand * OW * 4
        + cand * OW * esize
        + 2 * cand * OW_PAD * esize
        + cand * OW_PAD * 4
        + 2 * GW * OW * 4
        + 8 * TGH * 4
        + TGH * 4
        + 2 * 4
        + 8192
    )
    if esize == 4 and gh * GW <= 4:
        est += 7 * fused * 4 + 5 * cand * OW * 4
    return est


_PW_BUDGET = 184 * 1024


_PW_LADDER = (128, 96, 64, 48, 32, 16, 8, 4, 2, 1)


def _est_pw_rf(cand, GW, OW, W_PAD, TGH, N, esize, ROWS, OH, OW_PAD):
    fused = GW * cand * OW
    est = (
        2 * cand * ROWS * W_PAD * esize
        + fused * 4 * 2
        + fused * 4 * 2
        + fused * 4 * 2
        + (fused * 2 * 2 if esize == 2 else 0)
        + ROWS * fused * 4 * 2
        + fused * 4 * 2
        + fused * 4 * 2
        + fused * 4
        + 4 * 8 * cand * OW * 4
        + cand * OW * 4
        + cand * OW * esize
        + 2 * cand * OW_PAD * esize
        + cand * OW_PAD * 4
        + 2 * GW * OW * 4
        + 8 * TGH * 4
        + TGH * 4
        + 2 * 4
        + 8192
    )
    return est


_PW_LADDER_RF = (128, 96, 64, 48, 32, 16, 8, 4)


def _select_c_blk_rf(GW, OW, W_PAD, TGH, N, esize, ROWS, OH, OW_PAD, C=-1):
    ladder = _PW_LADDER_RF
    if C > 512:
        ladder = tuple(c for c in ladder if c not in (96, 48))
    for cand in ladder:
        if C > 0:
            vids = (C + cand - 1) // cand
            if C / (vids * cand) < 0.85:
                continue
        if (cand * OW * 4) % 32 != 0:
            continue
        est = _est_pw_rf(cand, GW, OW, W_PAD, TGH, N, esize, ROWS, OH, OW_PAD)
        if est <= 168 * 1024:
            return cand, est
    return None, 0


def _select_c_blk(GW, OW, W_PAD, TGH, N, esize, ROWS, OH, OW_PAD, C=-1,
                  min_cand=1, min_cbow_bytes=0):
    ladder = _PW_LADDER
    if C > 512:
        ladder = tuple(c for c in ladder if c not in (96, 48))
    if min_cand > 1:
        ladder = tuple(c for c in ladder if c >= min_cand)
    if min_cbow_bytes > 0:
        ladder = tuple(
            c for c in ladder
            if c * OW * 4 >= min_cbow_bytes
            and (c * OW * 4) % min_cbow_bytes == 0
        )
    cand, est = _o4_policy_select(
        lambda c: _est_pw(c, GW, OW, W_PAD, TGH, N, esize, ROWS, OH, OW_PAD),
        ladder, C, OW, OH, esize, OW_PAD, esize == 2, budget=_PW_BUDGET,
    )
    if cand is not None:
        return cand, est
    for cand in ladder:
        if C > 0:
            vids = (C + cand - 1) // cand
            if C / (vids * cand) < 0.85:
                continue
        est = _est_pw(cand, GW, OW, W_PAD, TGH, N, esize, ROWS, OH, OW_PAD)
        if est <= _PW_BUDGET:
            return cand, est
    return None, 0


def _align_elems(n, esize):
    nbytes = n * esize
    return (nbytes + 31) // 32 * 32 // esize


def _select_merge_k(est, cand, OW, OH, esize, OW_PAD, is_fp16,
                    budget=168 * 1024):
    legacy_out = (
        2 * cand * OW_PAD * esize
        + cand * OW_PAD * 4
        + (cand * OW * esize if is_fp16 else 0)
    )
    base = est - legacy_out
    if OH == 1:
        cost = _align_elems(cand * OW, esize) * esize + 64
        return 1 if base + cost <= budget else 0
    K = OH
    CP = _align_elems(cand * OW, esize)
    out_all_b = _align_elems(cand * OH * OW, esize) * esize
    map_b = _align_elems(cand * OH * OW, 4) * 4
    cost = K * CP * esize + out_all_b + map_b + 64
    return K if base + cost <= budget else 0


def _o4_policy_select(est_fn, ladder, C, OW, OH, esize, OW_PAD, is_fp16,
                      chan_gate=True, amp_gate=1.4, budget=None):
    if budget is None:
        budget = 168 * 1024
    cands = []
    for cand in ladder:
        if chan_gate and C > 0:
            vids = (C + cand - 1) // cand
            if C / (vids * cand) < 0.85:
                continue
        e = est_fn(cand)
        if e <= budget:
            cands.append((cand, e))
    if not cands:
        return None, 0
    cands.sort(reverse=True)

    def mk_of(est, cand):
        return _select_merge_k(est, cand, OW, OH, esize, OW_PAD, is_fp16,
                               budget=budget)

    def units_of(cand):
        return 2 * ((C + 2 * cand - 1) // (2 * cand)) if C > 0 else 2

    cand0, est0 = cands[0]
    cand_sel, est_sel = cand0, est0
    if mk_of(est0, cand0) == 0 and OH > 1:
        row_b = OW * esize
        amp = ((row_b + 31) // 32 * 32) / row_b
        u0 = units_of(cand0)
        for cand, est in cands:
            if cand >= cand0:
                continue
            if mk_of(est, cand) != OH:
                continue
            if amp >= amp_gate and units_of(cand) <= 1.75 * u0:
                cand_sel, est_sel = cand, est
                break
    return cand_sel, est_sel


def _build_offc_map(C_BLK, OW, OH, K, CP, KP, one_d, esize, dev=None):
    if K <= 1:
        return torch.zeros(1, dtype=torch.int32, device=dev)
    if one_d:
        c_ar = torch.arange(C_BLK, dtype=torch.int64, device=dev).view(C_BLK, 1, 1)
        oh_ar = torch.arange(OH, dtype=torch.int64, device=dev).view(1, OH, 1)
        j_ar = torch.arange(OW, dtype=torch.int64, device=dev).view(1, 1, OW)
        m = (oh_ar * CP + c_ar * OW + j_ar).reshape(C_BLK * OH * OW)
    else:
        c_ar = torch.arange(C_BLK, dtype=torch.int64, device=dev).view(C_BLK, 1, 1)
        oh_ar = torch.arange(K, dtype=torch.int64, device=dev).view(1, K, 1)
        j_ar = torch.arange(OW, dtype=torch.int64, device=dev).view(1, 1, OW)
        m = (oh_ar * CP + c_ar * OW + j_ar).reshape(C_BLK, K * OW)
        if KP > K * OW:
            m = torch.cat(
                [m, m[:, -1:].expand(C_BLK, KP - K * OW)], dim=1
            ).reshape(-1)
    return (m * esize).to(torch.int32).contiguous()


def _est_ts(cand, OW, W_PAD, D3, NC_PAD, TNR, N, esize, ROWS, OH, OW_PAD):
    return (
        2 * cand * ROWS * W_PAD * esize
        + cand * D3 * esize
        + cand * D3 * 4
        + cand * OW * NC_PAD * 4
        + cand * OW * 4 + cand * OW * esize
        + 2 * cand * OW_PAD * esize
        + cand * OW_PAD * 4
        + cand * D3 * 4
        + cand * D3 * 4
        + cand * D3 * 4
        + 2 * D3 * 4
        + 2 * TNR * 4
        + (2 * OH) * 4
        + 4 + 4
        + 8192
    )


def _select_c_blk_ts(OW, W_PAD, D3, NC_PAD, TNR, N, esize, ROWS, OH, OW_PAD):
    for cand in (32, 16, 8):
        est = _est_ts(cand, OW, W_PAD, D3, NC_PAD, TNR, N, esize, ROWS, OH,
                      OW_PAD)
        if est <= 168 * 1024:
            return cand, est
    return None, 0


def _est_tsk(cand, W_PAD, D3K, TNR, N, esize, ROWS, OH, OW, OW_PAD):
    return (
        2 * cand * ROWS * W_PAD * esize
        + (cand * D3K * esize if esize == 2 else 0)
        + cand * D3K * 4
        + cand * D3K * 4
        + cand * D3K * 4
        + cand * D3K * 4
        + cand * D3K * 4
        + cand * OW * 4
        + (cand * OW * esize if esize == 2 else 0)
        + 2 * cand * OW_PAD * esize
        + cand * OW_PAD * 4
        + D3K * 4
        + 2 * TNR * 4
        + 2 * OH * 4
        + 4 + 4
        + 8192
    )


def _select_c_blk_tsk(W_PAD, D3K, TNR, N, esize, ROWS, OH, OW, OW_PAD, C):
    cand, est = _o4_policy_select(
        lambda c: _est_tsk(c, W_PAD, D3K, TNR, N, esize, ROWS, OH, OW,
                           OW_PAD),
        (128, 96, 64, 48, 32, 16, 8), C, OW, OH, esize, OW_PAD, esize == 2,
    )
    if cand is not None:
        return cand, est
    for cand in (128, 96, 64, 48, 32, 16, 8):
        if C > 0:
            vids = (C + cand - 1) // cand
            if C / (vids * cand) < 0.85:
                continue
        est = _est_tsk(cand, W_PAD, D3K, TNR, N, esize, ROWS, OH, OW, OW_PAD)
        if est <= 168 * 1024:
            return cand, est
    return None, 0


US_ELEM_RATIO_THRESH = 0.86


def _est_us(cand, W_PAD, K_U, NCEFF, D3B, TNR, N, esize, ROWS, OH, OW, OW_PAD):
    per_bits = NCEFF * cand * OW
    mask_bytes = (per_bits + 7) // 8
    mask_bytes = (mask_bytes + 31) // 32 * 32
    return (
        2 * cand * ROWS * W_PAD * esize
        + (cand * K_U * esize if esize == 2 else 0)
        + cand * K_U * 4
        + cand * K_U * 4
        + cand * K_U * 4
        + cand * D3B * 4
        + cand * D3B * 4
        + cand * D3B * 4
        + cand * D3B * 4
        + cand * D3B * 4
        + cand * OW * 4
        + (cand * OW * esize if esize == 2 else 0)
        + 2 * cand * OW_PAD * esize
        + cand * OW_PAD * 4
        + D3B * 4
        + 2 * TNR * 4
        + 2 * OH * 4
        + 4 + 4
        + mask_bytes
        + 8192
    )


def _select_c_blk_us(W_PAD, K_U, NCEFF, D3B, TNR, N, esize, ROWS, OH, OW,
                     OW_PAD, C):
    for cand in (128, 96, 64, 48, 32, 16, 8):
        if C > 0:
            vids = (C + cand - 1) // cand
            if C / (vids * cand) < 0.85:
                continue
        est = _est_us(cand, W_PAD, K_U, NCEFF, D3B, TNR, N, esize, ROWS, OH,
                      OW, OW_PAD)
        if est <= 168 * 1024:
            return cand, est
    return None, 0

@tilelang.jit(out_idx=[10], pass_configs=PASS_CONFIGS)
def _roi_align_kernel(
    B, C, H, W, N, OH, OW, GH, GW, C_BLK, W_PAD, OW_PAD, TGH, ROWS, HW,
    MERGE_K=0, dtype="float16", stage_every_it=False, row_fold=False,
):
    c_num = T.ceildiv(C, 2 * C_BLK)
    is_fp16 = dtype == "float16"
    esize = 2 if is_fp16 else 4
    per_it = stage_every_it
    row_rf = row_fold
    FUSED = GW * C_BLK * OW
    CBOW = C_BLK * OW
    PAIR16 = GH * GW == 16
    CASCADE = GH * GW in (64, 256)
    LANE_BLK = 8 * CBOW
    CP = _align_elems(C_BLK * OW, esize) if MERGE_K > 0 else 1
    KP = _align_elems(MERGE_K * OW, esize) if MERGE_K > 1 else 1
    one_d = MERGE_K == OH
    if MERGE_K > 1:
        OFFC_LEN = C_BLK * OH * OW if one_d else C_BLK * KP
    else:
        OFFC_LEN = 1
    MAP_LEN = OFFC_LEN if MERGE_K > 0 else C_BLK * OW_PAD
    TI_LEN = _align_elems(1 + 3 * TGH, 4)
    YL = 1
    YH = 1 + TGH
    Y0 = 1 + 2 * TGH
    precise = dtype == "float" and GH * GW <= 4
    TF_LEN = _align_elems(
        1 + 2 * TGH + 2 * GW * OW
        + (2 * TGH + 2 * GW * OW if precise else 0), 4)
    WL = 1
    WH = 1 + TGH
    PL = 1 + 2 * TGH + 2 * GW * OW
    PH = PL + TGH
    CLAMP_B = 2.0 ** 30

    @T.prim_func
    def main(
        X2: T.Tensor((B, C, HW), dtype),  # type: ignore  (x viewed [B,C,H*W])
        ti_t: T.Tensor((N, TI_LEN), "int32"),  # type: ignore (packed scalars)
        tf_t: T.Tensor((N, TF_LEN), "float32"),  # type: ignore (packed weights)
        offlo_t: T.Tensor((N, FUSED), "uint32"),  # type: ignore (xlo offsets)
        offhi_t: T.Tensor((N, FUSED), "uint32"),  # type: ignore (xhi offsets)
        offmap_wxlo_t: T.Tensor((FUSED,), "uint32"),  # type: ignore (const)
        offmap_wxhi_t: T.Tensor((FUSED,), "uint32"),  # type: ignore (const)
        offmap_wxlop_t: T.Tensor((FUSED,), "uint32"),  # type: ignore (precise)
        offmap_wxhip_t: T.Tensor((FUSED,), "uint32"),  # type: ignore (precise)
        offb_t: T.Tensor((MAP_LEN,), "uint32"),  # type: ignore (output map)
        Y: T.Tensor(
            (N, C, OH, OW) if MERGE_K == 0
            else ((N, C * OH * OW) if one_d else (N, C, OH, OW)),
            dtype,
        ),  # type: ignore
    ):
        with T.Kernel(N * c_num, is_npu=True) as (cid, vid):
            n = cid // c_num
            c_blk = cid % c_num
            cs = c_blk * (2 * C_BLK) + vid * C_BLK
            cv = T.min(C - cs, C_BLK)
            if cv > 0:

                staging = T.alloc_ub((2, C_BLK, ROWS * W_PAD), dtype)
                ti = T.alloc_ub((TI_LEN,), "int32")
                tf = T.alloc_ub((TF_LEN,), "float32")

                offlo = T.alloc_ub((FUSED,), "uint32")
                offhi = T.alloc_ub((FUSED,), "uint32")
                offmap_wxlo = T.alloc_ub((FUSED,), "uint32")
                offmap_wxhi = T.alloc_ub((FUSED,), "uint32")
                wxlo_bc = T.alloc_ub((FUSED,), "float32")
                wxhi_bc = T.alloc_ub((FUSED,), "float32")

                gh = T.alloc_ub((FUSED,) if is_fp16 else (1,), dtype)
                v_f = T.alloc_ub((FUSED,), "float32")
                tmp = T.alloc_ub((FUSED,), "float32")
                w1t = T.alloc_ub((FUSED,), "float32")
                w2t = T.alloc_ub((FUSED,), "float32")
                w3t = T.alloc_ub((FUSED,), "float32")
                w4t = T.alloc_ub((FUSED,), "float32")
                t_a = T.alloc_ub((FUSED,), "float32")
                t_b = T.alloc_ub((FUSED,) if PAIR16 else (1,), "float32")
                pbuf = T.alloc_ub((CBOW,) if PAIR16 else (1,), "float32")
                p_cas = T.alloc_ub(
                    (4, LANE_BLK) if CASCADE else (1, 1), "float32")
                acc = T.alloc_ub((CBOW,), "float32")
                _PB = (FUSED,) if precise else (1,)
                _PC = (CBOW,) if precise else (1,)
                wxlop_bc = T.alloc_ub(_PB, "float32")
                wxhip_bc = T.alloc_ub(_PB, "float32")
                offmap_wxlop = T.alloc_ub(_PB, "uint32")
                offmap_wxhip = T.alloc_ub(_PB, "uint32")
                w_p = T.alloc_ub(_PB, "float32")
                t_p = T.alloc_ub(_PB, "float32")
                _PT = (FUSED,) if precise else (1,)
                t_c = T.alloc_ub(_PT, "float32")
                d_s = T.alloc_ub(_PC, "float32")
                dacc = T.alloc_ub(_PC, "float32")
                hbf = T.alloc_ub(_PC, "float32")
                cbf = T.alloc_ub(_PC, "float32")
                nbf = T.alloc_ub(_PC, "float32")
                _P2 = (2 * FUSED,) if row_rf else (1,)
                _VP = (ROWS, 2 * FUSED) if row_rf else (1, 1)
                _GH2 = (2 * FUSED,) if (row_rf and is_fp16) else (1,)
                off_pair = T.alloc_ub(_P2, "uint32")
                wx_pair_bc = T.alloc_ub(_P2, "float32")
                gh_row = T.alloc_ub(_GH2, dtype)
                V_pair = T.alloc_ub(_VP, "float32")
                w_pair = T.alloc_ub(_P2, "float32")
                t_pair = T.alloc_ub(_P2, "float32")
                rc_all = T.alloc_ub(
                    (MERGE_K, CP) if MERGE_K > 0 else (1, 1), dtype)
                out_all = T.alloc_ub(
                    (C_BLK * OH * OW,) if (MERGE_K > 1 and one_d)
                    else ((C_BLK, KP) if MERGE_K > 1 else (1,)),
                    dtype,
                )
                rc_h = T.alloc_ub(
                    (C_BLK * OW,) if MERGE_K == 0 else (1,), dtype)
                offb_ub = T.alloc_ub((MAP_LEN,), "uint32")
                out_pad = T.alloc_ub(
                    (2, C_BLK, OW_PAD) if MERGE_K == 0 else (1, 1, 1), dtype)

                T.copy(ti_t[n, 0:TI_LEN], ti)
                T.copy(tf_t[n, 0:TF_LEN], tf)
                T.copy(offlo_t[n, 0:FUSED], offlo)
                T.copy(offhi_t[n, 0:FUSED], offhi)
                T.copy(offmap_wxlo_t, offmap_wxlo)
                T.copy(offmap_wxhi_t, offmap_wxhi)
                if precise:
                    T.copy(offmap_wxlop_t, offmap_wxlop)
                    T.copy(offmap_wxhip_t, offmap_wxhip)
                T.copy(offb_t, offb_ub)
                T.barrier_all()

                T.tile.gather(wxlo_bc, tf, offmap_wxlo, 0)
                T.tile.gather(wxhi_bc, tf, offmap_wxhi, 0)
                if row_rf:
                    T.copy(offlo, off_pair[0:FUSED])
                    T.copy(offhi, off_pair[FUSED : 2 * FUSED])
                    T.tile.gather(wx_pair_bc[0:FUSED], tf, offmap_wxlo, 0)
                    T.tile.gather(wx_pair_bc[FUSED : 2 * FUSED], tf,
                                  offmap_wxhi, 0)
                if precise:
                    T.tile.gather(wxlop_bc, tf, offmap_wxlop, 0)
                    T.tile.gather(wxhip_bc, tf, offmap_wxhip, 0)

                b = ti[0]
                cnt = tf[0]

                if per_it:
                    for oh_i in T.serial(OH):
                        ohg = oh_i % (MERGE_K if MERGE_K > 0 else 1)
                        T.tile.fill(acc, 0.0)
                        if CASCADE:
                            for kk in range(4):
                                T.tile.fill(p_cas[kk, 0:LANE_BLK], 0.0)
                        if precise:
                            T.tile.fill(dacc, 0.0)
                        if PAIR16:
                            for pit in T.serial(GH // 2):
                                t_a_i = oh_i * GH + pit
                                t_b_i = oh_i * GH + pit + GH // 2
                                y0cw_a = ti[Y0 + t_a_i]
                                T.copy(
                                    X2[b, cs : cs + cv,
                                       y0cw_a : y0cw_a + ROWS * W_PAD],
                                    staging[0, 0:cv, 0 : ROWS * W_PAD],
                                )
                                T.barrier_all()
                                ylo_off = ti[YL + t_a_i]
                                yhi_off = ti[YH + t_a_i]
                                wylo = tf[WL + t_a_i]
                                wyhi = tf[WH + t_a_i]
                                T.tile.mul(w1t, wxlo_bc, wylo)
                                T.tile.mul(w2t, wxhi_bc, wylo)
                                T.tile.mul(w3t, wxlo_bc, wyhi)
                                T.tile.mul(w4t, wxhi_bc, wyhi)
                                if is_fp16:
                                    T.tile.gather(gh, staging[0, :, :],
                                                  offlo, ylo_off)
                                    T.tile.cast(v_f, gh, "CAST_NONE", FUSED)
                                else:
                                    T.tile.gather(v_f, staging[0, :, :],
                                                  offlo, ylo_off)
                                T.tile.mul(t_a, v_f, w1t)
                                if is_fp16:
                                    T.tile.gather(gh, staging[0, :, :],
                                                  offhi, ylo_off)
                                    T.tile.cast(v_f, gh, "CAST_NONE", FUSED)
                                else:
                                    T.tile.gather(v_f, staging[0, :, :],
                                                  offhi, ylo_off)
                                T.tile.mul(tmp, v_f, w2t)
                                T.tile.add(t_a, t_a, tmp)
                                if is_fp16:
                                    T.tile.gather(gh, staging[0, :, :],
                                                  offlo, yhi_off)
                                    T.tile.cast(v_f, gh, "CAST_NONE", FUSED)
                                else:
                                    T.tile.gather(v_f, staging[0, :, :],
                                                  offlo, yhi_off)
                                T.tile.mul(tmp, v_f, w3t)
                                T.tile.add(t_a, t_a, tmp)
                                if is_fp16:
                                    T.tile.gather(gh, staging[0, :, :],
                                                  offhi, yhi_off)
                                    T.tile.cast(v_f, gh, "CAST_NONE", FUSED)
                                else:
                                    T.tile.gather(v_f, staging[0, :, :],
                                                  offhi, yhi_off)
                                T.tile.mul(tmp, v_f, w4t)
                                T.tile.add(t_a, t_a, tmp)
                                y0cw_b = ti[Y0 + t_b_i]
                                T.copy(
                                    X2[b, cs : cs + cv,
                                       y0cw_b : y0cw_b + ROWS * W_PAD],
                                    staging[0, 0:cv, 0 : ROWS * W_PAD],
                                )
                                T.barrier_all()
                                ylo_off = ti[YL + t_b_i]
                                yhi_off = ti[YH + t_b_i]
                                wylo = tf[WL + t_b_i]
                                wyhi = tf[WH + t_b_i]
                                T.tile.mul(w1t, wxlo_bc, wylo)
                                T.tile.mul(w2t, wxhi_bc, wylo)
                                T.tile.mul(w3t, wxlo_bc, wyhi)
                                T.tile.mul(w4t, wxhi_bc, wyhi)
                                if is_fp16:
                                    T.tile.gather(gh, staging[0, :, :],
                                                  offlo, ylo_off)
                                    T.tile.cast(v_f, gh, "CAST_NONE", FUSED)
                                else:
                                    T.tile.gather(v_f, staging[0, :, :],
                                                  offlo, ylo_off)
                                T.tile.mul(t_b, v_f, w1t)
                                if is_fp16:
                                    T.tile.gather(gh, staging[0, :, :],
                                                  offhi, ylo_off)
                                    T.tile.cast(v_f, gh, "CAST_NONE", FUSED)
                                else:
                                    T.tile.gather(v_f, staging[0, :, :],
                                                  offhi, ylo_off)
                                T.tile.mul(tmp, v_f, w2t)
                                T.tile.add(t_b, t_b, tmp)
                                if is_fp16:
                                    T.tile.gather(gh, staging[0, :, :],
                                                  offlo, yhi_off)
                                    T.tile.cast(v_f, gh, "CAST_NONE", FUSED)
                                else:
                                    T.tile.gather(v_f, staging[0, :, :],
                                                  offlo, yhi_off)
                                T.tile.mul(tmp, v_f, w3t)
                                T.tile.add(t_b, t_b, tmp)
                                if is_fp16:
                                    T.tile.gather(gh, staging[0, :, :],
                                                  offhi, yhi_off)
                                    T.tile.cast(v_f, gh, "CAST_NONE", FUSED)
                                else:
                                    T.tile.gather(v_f, staging[0, :, :],
                                                  offhi, yhi_off)
                                T.tile.mul(tmp, v_f, w4t)
                                T.tile.add(t_b, t_b, tmp)
                                for iz in range(GW):
                                    lo = iz * CBOW
                                    T.tile.add(
                                        pbuf, t_a[lo : lo + CBOW],
                                        t_b[lo : lo + CBOW])
                                    T.tile.add(acc, acc, pbuf)
                        else:
                            for it in T.serial(GH):
                                t = oh_i * GH + it
                                y0cw = ti[Y0 + t]
                                T.copy(
                                    X2[b, cs : cs + cv,
                                       y0cw : y0cw + ROWS * W_PAD],
                                    staging[0, 0:cv, 0 : ROWS * W_PAD],
                                )
                                T.barrier_all()
                                ylo_off = ti[YL + t]
                                yhi_off = ti[YH + t]
                                wylo = tf[WL + t]
                                wyhi = tf[WH + t]
                                T.tile.mul(w1t, wxlo_bc, wylo)
                                T.tile.mul(w2t, wxhi_bc, wylo)
                                T.tile.mul(w3t, wxlo_bc, wyhi)
                                T.tile.mul(w4t, wxhi_bc, wyhi)
                                wylop = tf[(PL + t) if precise else (WL + t)]
                                wyhip = tf[(PH + t) if precise else (WH + t)]
                                if is_fp16:
                                    T.tile.gather(gh, staging[0, :, :],
                                                  offlo, ylo_off)
                                    T.tile.cast(v_f, gh, "CAST_NONE", FUSED)
                                else:
                                    T.tile.gather(v_f, staging[0, :, :],
                                                  offlo, ylo_off)
                                T.tile.mul(t_a, v_f, w1t)
                                if precise:
                                    T.tile.mul(w_p, wxlop_bc, wylop)
                                    T.tile.mul(t_p, v_f, w_p)
                                if is_fp16:
                                    T.tile.gather(gh, staging[0, :, :],
                                                  offhi, ylo_off)
                                    T.tile.cast(v_f, gh, "CAST_NONE", FUSED)
                                else:
                                    T.tile.gather(v_f, staging[0, :, :],
                                                  offhi, ylo_off)
                                T.tile.mul(tmp, v_f, w2t)
                                T.tile.add(t_a, t_a, tmp)
                                if precise:
                                    T.tile.mul(w_p, wxhip_bc, wylop)
                                    T.tile.mul_add_dst(t_p, v_f, w_p)
                                if is_fp16:
                                    T.tile.gather(gh, staging[0, :, :],
                                                  offlo, yhi_off)
                                    T.tile.cast(v_f, gh, "CAST_NONE", FUSED)
                                else:
                                    T.tile.gather(v_f, staging[0, :, :],
                                                  offlo, yhi_off)
                                T.tile.mul(tmp, v_f, w3t)
                                T.tile.add(t_a, t_a, tmp)
                                if precise:
                                    T.tile.mul(w_p, wxlop_bc, wyhip)
                                    T.tile.mul_add_dst(t_p, v_f, w_p)
                                if is_fp16:
                                    T.tile.gather(gh, staging[0, :, :],
                                                  offhi, yhi_off)
                                    T.tile.cast(v_f, gh, "CAST_NONE", FUSED)
                                else:
                                    T.tile.gather(v_f, staging[0, :, :],
                                                  offhi, yhi_off)
                                T.tile.mul(tmp, v_f, w4t)
                                T.tile.add(t_a, t_a, tmp)
                                if precise:
                                    T.tile.mul(w_p, wxhip_bc, wyhip)
                                    T.tile.mul_add_dst(t_p, v_f, w_p)
                                if CASCADE:
                                    if GW == 8:
                                        T.tile.add(
                                            p_cas[it % 4, 0:LANE_BLK],
                                            p_cas[it % 4, 0:LANE_BLK],
                                            t_a[0:LANE_BLK])
                                    else:
                                        T.tile.add(
                                            p_cas[(2 * it) % 4, 0:LANE_BLK],
                                            p_cas[(2 * it) % 4, 0:LANE_BLK],
                                            t_a[0:LANE_BLK])
                                        T.tile.add(
                                            p_cas[(2 * it + 1) % 4,
                                                  0:LANE_BLK],
                                            p_cas[(2 * it + 1) % 4,
                                                  0:LANE_BLK],
                                            t_a[LANE_BLK : 2 * LANE_BLK])
                                else:
                                    if precise:
                                        T.tile.max(t_c, t_a, -CLAMP_B)
                                        T.tile.min(t_c, t_c, CLAMP_B)
                                        T.tile.max(t_p, t_p, -CLAMP_B)
                                        T.tile.min(t_p, t_p, CLAMP_B)
                                    for iz in range(GW):
                                        lo = iz * CBOW
                                        if precise:
                                            T.tile.sub(
                                                d_s, t_p[lo : lo + CBOW],
                                                t_c[lo : lo + CBOW])
                                            T.tile.add(dacc, dacc, d_s)
                                        T.tile.add(
                                            acc, acc, t_a[lo : lo + CBOW])
                        if CASCADE:
                            T.tile.add(t_a[0:LANE_BLK],
                                       p_cas[0, 0:LANE_BLK],
                                       p_cas[1, 0:LANE_BLK])
                            T.tile.add(t_a[0:LANE_BLK], t_a[0:LANE_BLK],
                                       p_cas[2, 0:LANE_BLK])
                            T.tile.add(t_a[0:LANE_BLK], t_a[0:LANE_BLK],
                                       p_cas[3, 0:LANE_BLK])
                            for ln in range(8):
                                lo = ln * CBOW
                                T.tile.add(acc, acc, t_a[lo : lo + CBOW])
                        if MERGE_K > 0:
                            if is_fp16:
                                T.tile.mul(acc, acc, cnt)
                                T.tile.cast(
                                    rc_all[ohg, 0 : C_BLK * OW],
                                    acc,
                                    CAST_MODE_HIGH2LOW, C_BLK * OW)
                            else:
                                if precise:
                                    T.tile.mul(acc, acc, cnt)
                                    T.tile.mul(hbf, dacc, cnt)
                                    T.tile.mul(hbf, hbf, 0.4)
                                    T.tile.abs(cbf, acc)
                                    T.tile.sub(cbf, cbf, 2 ** -11)
                                    T.tile.max(cbf, cbf, 0.0)
                                    T.tile.mul(cbf, cbf, 0.05)
                                    T.tile.mul(nbf, cbf, -1.0)
                                    T.tile.max(hbf, hbf, nbf)
                                    T.tile.min(hbf, hbf, cbf)
                                    T.tile.add(acc, acc, hbf)
                                    T.copy(acc, rc_all[ohg, 0 : C_BLK * OW])
                                else:
                                    T.tile.mul(
                                        rc_all[ohg, 0 : C_BLK * OW],
                                        acc, cnt)
                        else:
                            T.tile.mul(acc, acc, cnt)
                            if is_fp16:
                                T.tile.cast(rc_h, acc,
                                            CAST_MODE_HIGH2LOW, C_BLK * OW)
                                T.tile.gather(out_pad[0, :, :], rc_h,
                                              offb_ub, 0)
                            else:
                                if precise:
                                    T.tile.mul(hbf, dacc, cnt)
                                    T.tile.mul(hbf, hbf, 0.4)
                                    T.tile.abs(cbf, acc)
                                    T.tile.sub(cbf, cbf, 2 ** -11)
                                    T.tile.max(cbf, cbf, 0.0)
                                    T.tile.mul(cbf, cbf, 0.05)
                                    T.tile.mul(nbf, cbf, -1.0)
                                    T.tile.max(hbf, hbf, nbf)
                                    T.tile.min(hbf, hbf, cbf)
                                    T.tile.add(acc, acc, hbf)
                                T.tile.gather(out_pad[0, :, :], acc,
                                              offb_ub, 0)
                        T.barrier_all()
                        if MERGE_K == 0:
                            T.copy(out_pad[0, 0:cv, 0:OW],
                                   Y[n, cs : cs + cv, oh_i, 0:OW])
                        elif MERGE_K == 1:
                            T.copy(rc_all[0, 0 : cv * OW],
                                   Y[n, cs * OW : (cs + cv) * OW])
                        elif one_d:
                            if oh_i + 1 == OH:
                                T.tile.gather(out_all, rc_all, offb_ub, 0)
                                T.barrier_all()
                                T.copy(out_all[0 : cv * OH * OW],
                                       Y[n, cs * OH * OW : (cs + cv) * OH * OW])
                        else:
                            if (oh_i + 1) % MERGE_K == 0 or oh_i + 1 == OH:
                                kr = ohg + 1
                                oh_a = oh_i - ohg
                                T.tile.gather(out_all, rc_all, offb_ub, 0)
                                T.barrier_all()
                                T.copy(out_all[0:cv, 0 : kr * OW],
                                       Y[n, cs : cs + cv,
                                         oh_a * OW : oh_a * OW + kr * OW])

                else:
                    y0cw0 = ti[Y0]
                    T.copy(
                        X2[b, cs : cs + cv, y0cw0 : y0cw0 + ROWS * W_PAD],
                        staging[0, 0:cv, 0 : ROWS * W_PAD],
                    )
                    T.barrier_all()

                    for oh_i in T.serial(OH):
                        cur = oh_i % 2
                        nxt = (oh_i + 1) % 2
                        ohg = oh_i % (MERGE_K if MERGE_K > 0 else 1)
                        if oh_i + 1 < OH:
                            y0cw_p = ti[Y0 + (oh_i + 1) * GH]
                            T.copy(
                                X2[b, cs : cs + cv, y0cw_p : y0cw_p + ROWS * W_PAD],
                                staging[nxt, 0:cv, 0 : ROWS * W_PAD],
                            )
                        T.tile.fill(acc, 0.0)
                        if CASCADE:
                            for kk in range(4):
                                T.tile.fill(p_cas[kk, 0:LANE_BLK], 0.0)
                        if precise:
                            T.tile.fill(dacc, 0.0)
                        if PAIR16:
                            for pit in T.serial(GH // 2):
                                t_a_i = oh_i * GH + pit
                                t_b_i = oh_i * GH + pit + GH // 2
                                ylo_off = ti[YL + t_a_i]
                                yhi_off = ti[YH + t_a_i]
                                wylo = tf[WL + t_a_i]
                                wyhi = tf[WH + t_a_i]
                                T.tile.mul(w1t, wxlo_bc, wylo)
                                T.tile.mul(w2t, wxhi_bc, wylo)
                                T.tile.mul(w3t, wxlo_bc, wyhi)
                                T.tile.mul(w4t, wxhi_bc, wyhi)
                                if is_fp16:
                                    T.tile.gather(gh, staging[cur, :, :],
                                                  offlo, ylo_off)
                                    T.tile.cast(v_f, gh, "CAST_NONE", FUSED)
                                else:
                                    T.tile.gather(v_f, staging[cur, :, :],
                                                  offlo, ylo_off)
                                T.tile.mul(t_a, v_f, w1t)
                                if is_fp16:
                                    T.tile.gather(gh, staging[cur, :, :],
                                                  offhi, ylo_off)
                                    T.tile.cast(v_f, gh, "CAST_NONE", FUSED)
                                else:
                                    T.tile.gather(v_f, staging[cur, :, :],
                                                  offhi, ylo_off)
                                T.tile.mul(tmp, v_f, w2t)
                                T.tile.add(t_a, t_a, tmp)
                                if is_fp16:
                                    T.tile.gather(gh, staging[cur, :, :],
                                                  offlo, yhi_off)
                                    T.tile.cast(v_f, gh, "CAST_NONE", FUSED)
                                else:
                                    T.tile.gather(v_f, staging[cur, :, :],
                                                  offlo, yhi_off)
                                T.tile.mul(tmp, v_f, w3t)
                                T.tile.add(t_a, t_a, tmp)
                                if is_fp16:
                                    T.tile.gather(gh, staging[cur, :, :],
                                                  offhi, yhi_off)
                                    T.tile.cast(v_f, gh, "CAST_NONE", FUSED)
                                else:
                                    T.tile.gather(v_f, staging[cur, :, :],
                                                  offhi, yhi_off)
                                T.tile.mul(tmp, v_f, w4t)
                                T.tile.add(t_a, t_a, tmp)
                                ylo_off = ti[YL + t_b_i]
                                yhi_off = ti[YH + t_b_i]
                                wylo = tf[WL + t_b_i]
                                wyhi = tf[WH + t_b_i]
                                T.tile.mul(w1t, wxlo_bc, wylo)
                                T.tile.mul(w2t, wxhi_bc, wylo)
                                T.tile.mul(w3t, wxlo_bc, wyhi)
                                T.tile.mul(w4t, wxhi_bc, wyhi)
                                if is_fp16:
                                    T.tile.gather(gh, staging[cur, :, :],
                                                  offlo, ylo_off)
                                    T.tile.cast(v_f, gh, "CAST_NONE", FUSED)
                                else:
                                    T.tile.gather(v_f, staging[cur, :, :],
                                                  offlo, ylo_off)
                                T.tile.mul(t_b, v_f, w1t)
                                if is_fp16:
                                    T.tile.gather(gh, staging[cur, :, :],
                                                  offhi, ylo_off)
                                    T.tile.cast(v_f, gh, "CAST_NONE", FUSED)
                                else:
                                    T.tile.gather(v_f, staging[cur, :, :],
                                                  offhi, ylo_off)
                                T.tile.mul(tmp, v_f, w2t)
                                T.tile.add(t_b, t_b, tmp)
                                if is_fp16:
                                    T.tile.gather(gh, staging[cur, :, :],
                                                  offlo, yhi_off)
                                    T.tile.cast(v_f, gh, "CAST_NONE", FUSED)
                                else:
                                    T.tile.gather(v_f, staging[cur, :, :],
                                                  offlo, yhi_off)
                                T.tile.mul(tmp, v_f, w3t)
                                T.tile.add(t_b, t_b, tmp)
                                if is_fp16:
                                    T.tile.gather(gh, staging[cur, :, :],
                                                  offhi, yhi_off)
                                    T.tile.cast(v_f, gh, "CAST_NONE", FUSED)
                                else:
                                    T.tile.gather(v_f, staging[cur, :, :],
                                                  offhi, yhi_off)
                                T.tile.mul(tmp, v_f, w4t)
                                T.tile.add(t_b, t_b, tmp)
                                for iz in range(GW):
                                    lo = iz * CBOW
                                    T.tile.add(
                                        pbuf, t_a[lo : lo + CBOW],
                                        t_b[lo : lo + CBOW])
                                    T.tile.add(acc, acc, pbuf)
                        else:
                            if row_rf:
                                for k in range(ROWS):
                                    if is_fp16:
                                        T.tile.gather(
                                            gh_row, staging[cur, :, :],
                                            off_pair, k * W * esize)
                                        T.tile.cast(
                                            V_pair[k, 0 : 2 * FUSED], gh_row,
                                            "CAST_NONE", 2 * FUSED)
                                    else:
                                        T.tile.gather(
                                            V_pair[k, 0 : 2 * FUSED],
                                            staging[cur, :, :], off_pair,
                                            k * W * esize)
                                T.barrier_all()
                                for it in T.serial(GH):
                                    t = oh_i * GH + it
                                    kl = ti[YL + t]
                                    kh = ti[YH + t]
                                    wylo = tf[WL + t]
                                    wyhi = tf[WH + t]
                                    T.tile.mul(w_pair, wx_pair_bc, wylo)
                                    T.tile.mul(
                                        t_pair,
                                        V_pair[kl, 0 : 2 * FUSED], w_pair)
                                    T.tile.add(
                                        t_a, t_pair[0:FUSED],
                                        t_pair[FUSED : 2 * FUSED])
                                    T.tile.mul(w_pair, wx_pair_bc, wyhi)
                                    T.tile.mul(
                                        t_pair,
                                        V_pair[kh, 0 : 2 * FUSED], w_pair)
                                    T.tile.add(t_a, t_a, t_pair[0:FUSED])
                                    T.tile.add(
                                        t_a, t_a, t_pair[FUSED : 2 * FUSED])
                                    if GW == 8:
                                        T.tile.add(
                                            p_cas[it % 4, 0:LANE_BLK],
                                            p_cas[it % 4, 0:LANE_BLK],
                                            t_a[0:LANE_BLK])
                                    else:
                                        T.tile.add(
                                            p_cas[(2 * it) % 4, 0:LANE_BLK],
                                            p_cas[(2 * it) % 4, 0:LANE_BLK],
                                            t_a[0:LANE_BLK])
                                        T.tile.add(
                                            p_cas[(2 * it + 1) % 4,
                                                  0:LANE_BLK],
                                            p_cas[(2 * it + 1) % 4,
                                                  0:LANE_BLK],
                                            t_a[LANE_BLK : 2 * LANE_BLK])
                            else:
                                for it in T.serial(GH):
                                    t = oh_i * GH + it
                                    ylo_off = ti[YL + t]
                                    yhi_off = ti[YH + t]
                                    wylo = tf[WL + t]
                                    wyhi = tf[WH + t]
                                    T.tile.mul(w1t, wxlo_bc, wylo)
                                    T.tile.mul(w2t, wxhi_bc, wylo)
                                    T.tile.mul(w3t, wxlo_bc, wyhi)
                                    T.tile.mul(w4t, wxhi_bc, wyhi)
                                    wylop = tf[(PL + t) if precise else (WL + t)]
                                    wyhip = tf[(PH + t) if precise else (WH + t)]
                                    if is_fp16:
                                        T.tile.gather(gh, staging[cur, :, :],
                                                      offlo, ylo_off)
                                        T.tile.cast(v_f, gh, "CAST_NONE", FUSED)
                                    else:
                                        T.tile.gather(v_f, staging[cur, :, :],
                                                      offlo, ylo_off)
                                    T.tile.mul(t_a, v_f, w1t)
                                    if precise:
                                        T.tile.mul(w_p, wxlop_bc, wylop)
                                        T.tile.mul(t_p, v_f, w_p)
                                    if is_fp16:
                                        T.tile.gather(gh, staging[cur, :, :],
                                                      offhi, ylo_off)
                                        T.tile.cast(v_f, gh, "CAST_NONE", FUSED)
                                    else:
                                        T.tile.gather(v_f, staging[cur, :, :],
                                                      offhi, ylo_off)
                                    T.tile.mul(tmp, v_f, w2t)
                                    T.tile.add(t_a, t_a, tmp)
                                    if precise:
                                        T.tile.mul(w_p, wxhip_bc, wylop)
                                        T.tile.mul_add_dst(t_p, v_f, w_p)
                                    if is_fp16:
                                        T.tile.gather(gh, staging[cur, :, :],
                                                      offlo, yhi_off)
                                        T.tile.cast(v_f, gh, "CAST_NONE", FUSED)
                                    else:
                                        T.tile.gather(v_f, staging[cur, :, :],
                                                      offlo, yhi_off)
                                    T.tile.mul(tmp, v_f, w3t)
                                    T.tile.add(t_a, t_a, tmp)
                                    if precise:
                                        T.tile.mul(w_p, wxlop_bc, wyhip)
                                        T.tile.mul_add_dst(t_p, v_f, w_p)
                                    if is_fp16:
                                        T.tile.gather(gh, staging[cur, :, :],
                                                      offhi, yhi_off)
                                        T.tile.cast(v_f, gh, "CAST_NONE", FUSED)
                                    else:
                                        T.tile.gather(v_f, staging[cur, :, :],
                                                      offhi, yhi_off)
                                    T.tile.mul(tmp, v_f, w4t)
                                    T.tile.add(t_a, t_a, tmp)
                                    if precise:
                                        T.tile.mul(w_p, wxhip_bc, wyhip)
                                        T.tile.mul_add_dst(t_p, v_f, w_p)
                                    if CASCADE:
                                        if GW == 8:
                                            T.tile.add(
                                                p_cas[it % 4, 0:LANE_BLK],
                                                p_cas[it % 4, 0:LANE_BLK],
                                                t_a[0:LANE_BLK])
                                        else:
                                            T.tile.add(
                                                p_cas[(2 * it) % 4, 0:LANE_BLK],
                                                p_cas[(2 * it) % 4, 0:LANE_BLK],
                                                t_a[0:LANE_BLK])
                                            T.tile.add(
                                                p_cas[(2 * it + 1) % 4,
                                                      0:LANE_BLK],
                                                p_cas[(2 * it + 1) % 4,
                                                      0:LANE_BLK],
                                                t_a[LANE_BLK : 2 * LANE_BLK])
                                    else:
                                        if precise:
                                            T.tile.max(t_c, t_a, -CLAMP_B)
                                            T.tile.min(t_c, t_c, CLAMP_B)
                                            T.tile.max(t_p, t_p, -CLAMP_B)
                                            T.tile.min(t_p, t_p, CLAMP_B)
                                        for iz in range(GW):
                                            lo = iz * CBOW
                                            if precise:
                                                T.tile.sub(
                                                    d_s, t_p[lo : lo + CBOW],
                                                    t_c[lo : lo + CBOW])
                                                T.tile.add(dacc, dacc, d_s)
                                            T.tile.add(
                                                acc, acc, t_a[lo : lo + CBOW])
                        if CASCADE:
                            T.tile.add(t_a[0:LANE_BLK],
                                       p_cas[0, 0:LANE_BLK],
                                       p_cas[1, 0:LANE_BLK])
                            T.tile.add(t_a[0:LANE_BLK], t_a[0:LANE_BLK],
                                       p_cas[2, 0:LANE_BLK])
                            T.tile.add(t_a[0:LANE_BLK], t_a[0:LANE_BLK],
                                       p_cas[3, 0:LANE_BLK])
                            for ln in range(8):
                                lo = ln * CBOW
                                T.tile.add(acc, acc, t_a[lo : lo + CBOW])
                        if MERGE_K > 0:
                            if is_fp16:
                                T.tile.mul(acc, acc, cnt)
                                T.tile.cast(
                                    rc_all[ohg, 0 : C_BLK * OW],
                                    acc,
                                    CAST_MODE_HIGH2LOW, C_BLK * OW)
                            else:
                                if precise:
                                    T.tile.mul(acc, acc, cnt)
                                    T.tile.mul(hbf, dacc, cnt)
                                    T.tile.mul(hbf, hbf, 0.4)
                                    T.tile.abs(cbf, acc)
                                    T.tile.sub(cbf, cbf, 2 ** -11)
                                    T.tile.max(cbf, cbf, 0.0)
                                    T.tile.mul(cbf, cbf, 0.05)
                                    T.tile.mul(nbf, cbf, -1.0)
                                    T.tile.max(hbf, hbf, nbf)
                                    T.tile.min(hbf, hbf, cbf)
                                    T.tile.add(acc, acc, hbf)
                                    T.copy(acc, rc_all[ohg, 0 : C_BLK * OW])
                                else:
                                    T.tile.mul(
                                        rc_all[ohg, 0 : C_BLK * OW],
                                        acc, cnt)
                        else:
                            T.tile.mul(acc, acc, cnt)
                            if is_fp16:
                                T.tile.cast(rc_h, acc,
                                            CAST_MODE_HIGH2LOW, C_BLK * OW)
                                T.tile.gather(out_pad[cur, :, :], rc_h,
                                              offb_ub, 0)
                            else:
                                if precise:
                                    T.tile.mul(hbf, dacc, cnt)
                                    T.tile.mul(hbf, hbf, 0.4)
                                    T.tile.abs(cbf, acc)
                                    T.tile.sub(cbf, cbf, 2 ** -11)
                                    T.tile.max(cbf, cbf, 0.0)
                                    T.tile.mul(cbf, cbf, 0.05)
                                    T.tile.mul(nbf, cbf, -1.0)
                                    T.tile.max(hbf, hbf, nbf)
                                    T.tile.min(hbf, hbf, cbf)
                                    T.tile.add(acc, acc, hbf)
                                T.tile.gather(out_pad[cur, :, :], acc,
                                              offb_ub, 0)
                        T.barrier_all()
                        if MERGE_K == 0:
                            T.copy(out_pad[cur, 0:cv, 0:OW],
                                   Y[n, cs : cs + cv, oh_i, 0:OW])
                        elif MERGE_K == 1:
                            T.copy(rc_all[0, 0 : cv * OW],
                                   Y[n, cs * OW : (cs + cv) * OW])
                        elif one_d:
                            if oh_i + 1 == OH:
                                T.tile.gather(out_all, rc_all, offb_ub, 0)
                                T.barrier_all()
                                T.copy(out_all[0 : cv * OH * OW],
                                       Y[n, cs * OH * OW : (cs + cv) * OH * OW])
                        else:
                            if (oh_i + 1) % MERGE_K == 0 or oh_i + 1 == OH:
                                kr = ohg + 1
                                oh_a = oh_i - ohg
                                T.tile.gather(out_all, rc_all, offb_ub, 0)
                                T.barrier_all()
                                T.copy(out_all[0:cv, 0 : kr * OW],
                                       Y[n, cs : cs + cv,
                                         oh_a * OW : oh_a * OW + kr * OW])

    return main


@tilelang.jit(out_idx=[10], pass_configs=PASS_CONFIGS)
def _roi_align_ts_kernel(
    B, C, H, W, N, OH, OW, C_BLK, W_PAD, OW_PAD, NC_PAD, D3, TNR, NRMAX, ROWS,
    HW, MERGE_K=0, dtype="float16",
):
    c_num = T.ceildiv(C, 2 * C_BLK)
    is_fp16 = dtype == "float16"
    esize = 2 if is_fp16 else 4
    CP = _align_elems(C_BLK * OW, esize) if MERGE_K > 0 else 1
    KP = _align_elems(MERGE_K * OW, esize) if MERGE_K > 1 else 1
    one_d = MERGE_K == OH
    if MERGE_K > 1:
        OFFC_LEN = C_BLK * OH * OW if one_d else C_BLK * KP
    else:
        OFFC_LEN = 1
    MAP_LEN = OFFC_LEN if MERGE_K > 0 else C_BLK * OW_PAD

    @T.prim_func
    def main(
        X2: T.Tensor((B, C, HW), dtype),  # type: ignore  (x viewed [B,C,H*W])
        rw_t: T.Tensor((N, TNR), "float32"),  # type: ignore  (merged row weights)
        rbyte_t: T.Tensor((N, TNR), "int32"),  # type: ignore  (support row byte offs)
        wt_t: T.Tensor((N, D3), "float32"),  # type: ignore  (merged col weights)
        cp_t: T.Tensor((N, D3), "int32"),  # type: ignore  (support col byte offs)
        y0cw_t: T.Tensor((N, OH), "int32"),  # type: ignore  (staging window start)
        bidx_t: T.Tensor((N,), "int32"),  # type: ignore
        cnt_t: T.Tensor((N,), "float32"),  # type: ignore
        nr_t: T.Tensor((N, OH), "int32"),  # type: ignore  (per-(n,oh) support rows)
        offb_t: T.Tensor((MAP_LEN,), "uint32"),  # type: ignore (output map)
        Y: T.Tensor(
            (N, C, OH, OW) if MERGE_K == 0
            else ((N, C * OH * OW) if one_d else (N, C, OH * OW)),
            dtype,
        ),  # type: ignore
    ):
        with T.Kernel(N * c_num, is_npu=True) as (cid, vid):
            n = cid // c_num
            c_blk = cid % c_num
            cs = c_blk * (2 * C_BLK) + vid * C_BLK
            cv = T.min(C - cs, C_BLK)
            if cv > 0:

                staging = T.alloc_ub((2, C_BLK, ROWS * W_PAD), dtype)
                rw_ub = T.alloc_ub((TNR,), "float32")
                rbyte_ub = T.alloc_ub((TNR,), "int32")
                wt_ub = T.alloc_ub((D3,), "float32")
                cp_ub = T.alloc_ub((D3,), "int32")
                y0cw_ub = T.alloc_ub((OH,), "int32")
                bidx_ub = T.alloc_ub((1,), "int32")
                cnt_ub = T.alloc_ub((1,), "float32")
                nr_ub = T.alloc_ub((OH,), "int32")

                idx_ub = T.alloc_ub((C_BLK, 1), "int32")
                rowbase = T.alloc_ub((C_BLK, 1), "int32")
                rowbase2d = T.alloc_ub((C_BLK, D3), "int32")
                off2d = T.alloc_ub((C_BLK, D3), "int32")
                off_u32 = T.alloc_ub((C_BLK, D3), "uint32")
                wt_gw = T.alloc_ub((C_BLK, D3), "float32")

                gh = T.alloc_ub((1, C_BLK, D3), dtype)
                g = T.alloc_ub((1, C_BLK, D3), "float32")
                acc = T.alloc_ub((C_BLK * OW, NC_PAD), "float32")
                rc = T.alloc_ub((C_BLK * OW,), "float32")
                rc_all = T.alloc_ub(
                    (MERGE_K, CP) if MERGE_K > 0 else (1, 1), dtype)
                out_all = T.alloc_ub(
                    (C_BLK * OH * OW,) if (MERGE_K > 1 and one_d)
                    else ((C_BLK, KP) if MERGE_K > 1 else (1,)),
                    dtype,
                )
                rc_h = T.alloc_ub(
                    (C_BLK * OW,) if MERGE_K == 0 else (1,),
                    dtype)
                offb_ub = T.alloc_ub((MAP_LEN,), "uint32")
                out_pad = T.alloc_ub(
                    (2, C_BLK, OW_PAD) if MERGE_K == 0 else (1, 1, 1), dtype)

                T.copy(bidx_t[n : n + 1], bidx_ub)
                T.copy(cnt_t[n : n + 1], cnt_ub)
                T.copy(rw_t[n, 0:TNR], rw_ub)
                T.copy(rbyte_t[n, 0:TNR], rbyte_ub)
                T.copy(wt_t[n, 0:D3], wt_ub)
                T.copy(cp_t[n, 0:D3], cp_ub)
                T.copy(y0cw_t[n, 0:OH], y0cw_ub)
                T.copy(nr_t[n, 0:OH], nr_ub)
                T.copy(offb_t, offb_ub)
                T.barrier_all()

                T.tile.createvecindex(idx_ub, 0)
                T.tile.clamp_max(idx_ub, idx_ub, cv - 1, C_BLK)
                T.tile.mul(rowbase, idx_ub, ROWS * W_PAD * esize)
                T.tile.broadcast(rowbase2d, rowbase, axis=1)

                T.tile.broadcast(off2d, cp_ub, axis=0)
                T.tile.add(off2d, off2d, rowbase2d)
                T.reinterpretcast(off_u32, off2d, "uint32_t")
                T.tile.broadcast(wt_gw, wt_ub, axis=0)

                b = bidx_ub[0]
                cnt = cnt_ub[0]

                y0cw0 = y0cw_ub[0]
                T.copy(
                    X2[b, cs : cs + cv, y0cw0 : y0cw0 + ROWS * W_PAD],
                    staging[0, 0:cv, 0 : ROWS * W_PAD],
                )
                T.barrier_all()

                for oh_i in T.serial(OH):
                    cur = oh_i % 2
                    nxt = (oh_i + 1) % 2
                    ohg = oh_i % (MERGE_K if MERGE_K > 0 else 1)
                    if oh_i + 1 < OH:
                        y0cw_p = y0cw_ub[oh_i + 1]
                        T.copy(
                            X2[b, cs : cs + cv, y0cw_p : y0cw_p + ROWS * W_PAD],
                            staging[nxt, 0:cv, 0 : ROWS * W_PAD],
                        )
                    nr = nr_ub[oh_i]
                    T.tile.fill(acc, 0.0)
                    for r in T.serial(nr):
                        t = oh_i * NRMAX + r
                        rb = rbyte_ub[t]
                        w = rw_ub[t]
                        if is_fp16:
                            T.tile.gather(gh, staging[cur, :, :], off_u32, rb)
                            T.tile.cast(g, gh, "CAST_NONE", C_BLK * D3)
                        else:
                            T.tile.gather(g, staging[cur, :, :], off_u32, rb)
                        T.tile.mul(g, g, w)
                        T.tile.mul_add_dst(acc, g, wt_gw)
                    T.reduce_sum(acc, rc, dim=-1)
                    if MERGE_K > 0:
                        if is_fp16:
                            T.tile.mul(rc, rc, cnt)
                            T.tile.cast(
                                rc_all[ohg, 0 : C_BLK * OW],
                                rc, CAST_MODE_HIGH2LOW, C_BLK * OW)
                        else:
                            T.tile.mul(
                                rc_all[ohg, 0 : C_BLK * OW], rc, cnt)
                    else:
                        T.tile.mul(rc, rc, cnt)
                        if is_fp16:
                            T.tile.cast(rc_h, rc, CAST_MODE_HIGH2LOW,
                                        C_BLK * OW)
                            T.tile.gather(out_pad[cur, :, :], rc_h,
                                          offb_ub, 0)
                        else:
                            T.tile.gather(out_pad[cur, :, :], rc, offb_ub, 0)
                    T.barrier_all()
                    if MERGE_K == 0:
                        T.copy(out_pad[cur, 0:cv, 0:OW],
                               Y[n, cs : cs + cv, oh_i, 0:OW])
                    elif MERGE_K == 1:
                        T.copy(rc_all[0, 0 : cv * OW],
                               Y[n, cs * OW : (cs + cv) * OW])
                    elif one_d:
                        if oh_i + 1 == OH:
                            T.tile.gather(out_all, rc_all, offb_ub, 0)
                            T.barrier_all()
                            T.copy(out_all[0 : cv * OH * OW],
                                   Y[n, cs * OH * OW : (cs + cv) * OH * OW])
                    else:
                        if (oh_i + 1) % MERGE_K == 0 or oh_i + 1 == OH:
                            kr = ohg + 1
                            oh_a = oh_i - ohg
                            T.tile.gather(out_all, rc_all, offb_ub, 0)
                            T.barrier_all()
                            T.copy(out_all[0:cv, 0 : kr * OW],
                                   Y[n, cs : cs + cv,
                                     oh_a * OW : oh_a * OW + kr * OW])

    return main

@tilelang.jit(out_idx=[12], pass_configs=PASS_CONFIGS)
def _roi_align_tsk_kernel(
    B, C, H, W, N, OH, OW, C_BLK, W_PAD, OW_PAD, NCEFF, D3K, TNR, NREFFMAX,
    ROWS, MASK_BYTES, HW, MERGE_K=0, dtype="float16",
):
    c_num = T.ceildiv(C, 2 * C_BLK)
    is_fp16 = dtype == "float16"
    esize = 2 if is_fp16 else 4
    CP = _align_elems(C_BLK * OW, esize) if MERGE_K > 0 else 1
    KP = _align_elems(MERGE_K * OW, esize) if MERGE_K > 1 else 1
    one_d = MERGE_K == OH
    if MERGE_K > 1:
        OFFC_LEN = C_BLK * OH * OW if one_d else C_BLK * KP
    else:
        OFFC_LEN = 1
    MAP_LEN = OFFC_LEN if MERGE_K > 0 else C_BLK * OW_PAD

    @T.prim_func
    def main(
        X2: T.Tensor((B, C, HW), dtype),  # type: ignore  (x viewed [B,C,H*W])
        rw_t: T.Tensor((N, TNR), "float32"),  # type: ignore (kept row weights)
        rbyte_t: T.Tensor((N, TNR), "int32"),  # type: ignore (row byte offs)
        wt_t: T.Tensor((N, D3K), "float32"),  # type: ignore (k-major col w)
        off_t: T.Tensor((N, C_BLK * D3K), "uint32"),  # type: ignore
        selmask_t: T.Tensor((N, MASK_BYTES), "uint8"),  # type: ignore
        offmap_d_t: T.Tensor((C_BLK * D3K,), "uint32"),  # type: ignore
        y0cw_t: T.Tensor((N, OH), "int32"),  # type: ignore
        bidx_t: T.Tensor((N,), "int32"),  # type: ignore
        cnt_t: T.Tensor((N,), "float32"),  # type: ignore
        nreff_t: T.Tensor((N, OH), "int32"),  # type: ignore
        offb_t: T.Tensor((MAP_LEN,), "uint32"),  # type: ignore (output map)
        Y: T.Tensor(
            (N, C, OH, OW) if MERGE_K == 0
            else ((N, C * OH * OW) if one_d else (N, C, OH * OW)),
            dtype,
        ),  # type: ignore
    ):
        with T.Kernel(N * c_num, is_npu=True) as (cid, vid):
            n = cid // c_num
            c_blk = cid % c_num
            cs = c_blk * (2 * C_BLK) + vid * C_BLK
            cv = T.min(C - cs, C_BLK)
            if cv > 0:

                staging = T.alloc_ub((2, C_BLK, ROWS * W_PAD), dtype)
                rw_ub = T.alloc_ub((TNR,), "float32")
                rbyte_ub = T.alloc_ub((TNR,), "int32")
                wt_ub = T.alloc_ub((D3K,), "float32")
                off_ub = T.alloc_ub((C_BLK * D3K,), "uint32")
                selmask_ub = T.alloc_ub((MASK_BYTES,), "uint8")
                offmap_d = T.alloc_ub((C_BLK * D3K,), "uint32")
                y0cw_ub = T.alloc_ub((OH,), "int32")
                bidx_ub = T.alloc_ub((1,), "int32")
                cnt_ub = T.alloc_ub((1,), "float32")
                nreff_ub = T.alloc_ub((OH,), "int32")

                gh = T.alloc_ub((C_BLK * D3K,) if is_fp16 else (1,), dtype)
                g = T.alloc_ub((C_BLK * D3K,), "float32")
                wt_gw = T.alloc_ub((C_BLK * D3K,), "float32")
                acc = T.alloc_ub((NCEFF, C_BLK * OW), "float32")
                out_flat = T.alloc_ub((1, C_BLK * OW), "float32")
                rc_all = T.alloc_ub(
                    (MERGE_K, CP) if MERGE_K > 0 else (1, 1), dtype)
                out_all = T.alloc_ub(
                    (C_BLK * OH * OW,) if (MERGE_K > 1 and one_d)
                    else ((C_BLK, KP) if MERGE_K > 1 else (1,)),
                    dtype,
                )
                rc_h = T.alloc_ub(
                    (C_BLK * OW,) if (MERGE_K == 0 and is_fp16) else (1,),
                    dtype)
                offb_ub = T.alloc_ub((MAP_LEN,), "uint32")
                out_pad = T.alloc_ub(
                    (2, C_BLK, OW_PAD) if MERGE_K == 0 else (1, 1, 1), dtype)

                T.copy(bidx_t[n : n + 1], bidx_ub)
                T.copy(cnt_t[n : n + 1], cnt_ub)
                T.copy(rw_t[n, 0:TNR], rw_ub)
                T.copy(rbyte_t[n, 0:TNR], rbyte_ub)
                T.copy(wt_t[n, 0:D3K], wt_ub)
                T.copy(off_t[n, 0 : C_BLK * D3K], off_ub)
                T.copy(selmask_t[n, 0:MASK_BYTES], selmask_ub)
                T.copy(offmap_d_t, offmap_d)
                T.copy(y0cw_t[n, 0:OH], y0cw_ub)
                T.copy(nreff_t[n, 0:OH], nreff_ub)
                T.copy(offb_t, offb_ub)
                T.barrier_all()

                T.tile.gather(wt_gw, wt_ub, offmap_d, 0)

                b = bidx_ub[0]
                cnt = cnt_ub[0]

                y0cw0 = y0cw_ub[0]
                T.copy(
                    X2[b, cs : cs + cv, y0cw0 : y0cw0 + ROWS * W_PAD],
                    staging[0, 0:cv, 0 : ROWS * W_PAD],
                )
                T.barrier_all()

                for oh_i in T.serial(OH):
                    cur = oh_i % 2
                    nxt = (oh_i + 1) % 2
                    ohg = oh_i % (MERGE_K if MERGE_K > 0 else 1)
                    if oh_i + 1 < OH:
                        y0cw_p = y0cw_ub[oh_i + 1]
                        T.copy(
                            X2[b, cs : cs + cv,
                               y0cw_p : y0cw_p + ROWS * W_PAD],
                            staging[nxt, 0:cv, 0 : ROWS * W_PAD],
                        )
                    nreff = nreff_ub[oh_i]
                    T.tile.fill(acc, 0.0)
                    for r in T.serial(nreff):
                        t = oh_i * NREFFMAX + r
                        rb = rbyte_ub[t]
                        w = rw_ub[t]
                        if is_fp16:
                            T.tile.gather(gh, staging[cur, :, :], off_ub, rb)
                            T.tile.cast(g, gh, "CAST_NONE", C_BLK * D3K)
                        else:
                            T.tile.gather(g, staging[cur, :, :], off_ub, rb)
                        T.tile.mul(g, g, w)
                        T.tile.mul_add_dst(acc, g, wt_gw)
                    T.tile.select(
                        acc, selmask_ub, acc, 0.0,
                        "VSEL_TENSOR_SCALAR_MODE",
                    )
                    T.reduce_sum(acc, out_flat, dim=0)
                    if MERGE_K > 0:
                        if is_fp16:
                            T.tile.mul(out_flat, out_flat, cnt)
                            T.tile.cast(
                                rc_all[ohg, 0 : C_BLK * OW],
                                out_flat[0, 0 : C_BLK * OW],
                                CAST_MODE_HIGH2LOW, C_BLK * OW)
                        else:
                            T.tile.mul(
                                rc_all[ohg, 0 : C_BLK * OW],
                                out_flat[0, 0 : C_BLK * OW], cnt)
                    else:
                        T.tile.mul(out_flat, out_flat, cnt)
                        if is_fp16:
                            T.tile.cast(rc_h, out_flat, CAST_MODE_HIGH2LOW,
                                        C_BLK * OW)
                            T.tile.gather(out_pad[cur, :, :], rc_h,
                                          offb_ub, 0)
                        else:
                            T.tile.gather(out_pad[cur, :, :], out_flat[0, :],
                                          offb_ub, 0)
                    T.barrier_all()
                    if MERGE_K == 0:
                        T.copy(out_pad[cur, 0:cv, 0:OW],
                               Y[n, cs : cs + cv, oh_i, 0:OW])
                    elif MERGE_K == 1:
                        T.copy(rc_all[0, 0 : cv * OW],
                               Y[n, cs * OW : (cs + cv) * OW])
                    elif one_d:
                        if oh_i + 1 == OH:
                            T.tile.gather(out_all, rc_all, offb_ub, 0)
                            T.barrier_all()
                            T.copy(out_all[0 : cv * OH * OW],
                                   Y[n, cs * OH * OW : (cs + cv) * OH * OW])
                    else:
                        if (oh_i + 1) % MERGE_K == 0 or oh_i + 1 == OH:
                            kr = ohg + 1
                            oh_a = oh_i - ohg
                            T.tile.gather(out_all, rc_all, offb_ub, 0)
                            T.barrier_all()
                            T.copy(out_all[0:cv, 0 : kr * OW],
                                   Y[n, cs : cs + cv,
                                     oh_a * OW : oh_a * OW + kr * OW])

    return main


@tilelang.jit(out_idx=[13], pass_configs=PASS_CONFIGS)
def _roi_align_us_kernel(
    B, C, H, W, N, OH, OW, C_BLK, W_PAD, OW_PAD, K_U, NCEFF, D3B, TNR,
    NREFFMAX, ROWS, MASK_BYTES, HW, MERGE_K=0, dtype="float16",
):
    c_num = T.ceildiv(C, 2 * C_BLK)
    is_fp16 = dtype == "float16"
    esize = 2 if is_fp16 else 4
    CP = _align_elems(C_BLK * OW, esize) if MERGE_K > 0 else 1
    KP = _align_elems(MERGE_K * OW, esize) if MERGE_K > 1 else 1
    one_d = MERGE_K == OH
    if MERGE_K > 1:
        OFFC_LEN = C_BLK * OH * OW if one_d else C_BLK * KP
    else:
        OFFC_LEN = 1
    MAP_LEN = OFFC_LEN if MERGE_K > 0 else C_BLK * OW_PAD

    @T.prim_func
    def main(
        X2: T.Tensor((B, C, HW), dtype),  # type: ignore  (x viewed [B,C,H*W])
        rw_t: T.Tensor((N, TNR), "float32"),  # type: ignore (kept row weights)
        rbyte_t: T.Tensor((N, TNR), "int32"),  # type: ignore (row byte offs)
        wt_t: T.Tensor((N, D3B), "float32"),  # type: ignore (k-major col w)
        offA_t: T.Tensor((N, C_BLK * K_U), "uint32"),  # type: ignore
        offB_t: T.Tensor((N, C_BLK * D3B), "uint32"),  # type: ignore
        selmask_t: T.Tensor((N, MASK_BYTES), "uint8"),  # type: ignore
        offmap_d_t: T.Tensor((C_BLK * D3B,), "uint32"),  # type: ignore
        y0cw_t: T.Tensor((N, OH), "int32"),  # type: ignore
        bidx_t: T.Tensor((N,), "int32"),  # type: ignore
        cnt_t: T.Tensor((N,), "float32"),  # type: ignore
        nreff_t: T.Tensor((N, OH), "int32"),  # type: ignore
        offb_t: T.Tensor((MAP_LEN,), "uint32"),  # type: ignore (output map)
        Y: T.Tensor(
            (N, C, OH, OW) if MERGE_K == 0
            else ((N, C * OH * OW) if one_d else (N, C, OH * OW)),
            dtype,
        ),  # type: ignore
    ):
        with T.Kernel(N * c_num, is_npu=True) as (cid, vid):
            n = cid // c_num
            c_blk = cid % c_num
            cs = c_blk * (2 * C_BLK) + vid * C_BLK
            cv = T.min(C - cs, C_BLK)
            if cv > 0:

                staging = T.alloc_ub((2, C_BLK, ROWS * W_PAD), dtype)
                rw_ub = T.alloc_ub((TNR,), "float32")
                rbyte_ub = T.alloc_ub((TNR,), "int32")
                wt_ub = T.alloc_ub((D3B,), "float32")
                offA_ub = T.alloc_ub((C_BLK * K_U,), "uint32")
                offB_ub = T.alloc_ub((C_BLK * D3B,), "uint32")
                selmask_ub = T.alloc_ub((MASK_BYTES,), "uint8")
                offmap_d = T.alloc_ub((C_BLK * D3B,), "uint32")
                y0cw_ub = T.alloc_ub((OH,), "int32")
                bidx_ub = T.alloc_ub((1,), "int32")
                cnt_ub = T.alloc_ub((1,), "float32")
                nreff_ub = T.alloc_ub((OH,), "int32")

                ghA = T.alloc_ub(
                    (C_BLK * K_U,) if is_fp16 else (1,), dtype)
                gA = T.alloc_ub((C_BLK * K_U,), "float32")
                acc_U = T.alloc_ub((C_BLK * K_U,), "float32")
                gB = T.alloc_ub((NCEFF, C_BLK * OW), "float32")
                wt_gw = T.alloc_ub((C_BLK * D3B,), "float32")
                out_flat = T.alloc_ub((1, C_BLK * OW), "float32")
                rc_all = T.alloc_ub(
                    (MERGE_K, CP) if MERGE_K > 0 else (1, 1), dtype)
                out_all = T.alloc_ub(
                    (C_BLK * OH * OW,) if (MERGE_K > 1 and one_d)
                    else ((C_BLK, KP) if MERGE_K > 1 else (1,)),
                    dtype,
                )
                rc_h = T.alloc_ub(
                    (C_BLK * OW,) if (MERGE_K == 0 and is_fp16) else (1,),
                    dtype)
                offb_ub = T.alloc_ub((MAP_LEN,), "uint32")
                out_pad = T.alloc_ub(
                    (2, C_BLK, OW_PAD) if MERGE_K == 0 else (1, 1, 1), dtype)

                T.copy(bidx_t[n : n + 1], bidx_ub)
                T.copy(cnt_t[n : n + 1], cnt_ub)
                T.copy(rw_t[n, 0:TNR], rw_ub)
                T.copy(rbyte_t[n, 0:TNR], rbyte_ub)
                T.copy(wt_t[n, 0:D3B], wt_ub)
                T.copy(offA_t[n, 0 : C_BLK * K_U], offA_ub)
                T.copy(offB_t[n, 0 : C_BLK * D3B], offB_ub)
                T.copy(selmask_t[n, 0:MASK_BYTES], selmask_ub)
                T.copy(offmap_d_t, offmap_d)
                T.copy(y0cw_t[n, 0:OH], y0cw_ub)
                T.copy(nreff_t[n, 0:OH], nreff_ub)
                T.copy(offb_t, offb_ub)
                T.barrier_all()

                T.tile.gather(wt_gw, wt_ub, offmap_d, 0)

                b = bidx_ub[0]
                cnt = cnt_ub[0]

                y0cw0 = y0cw_ub[0]
                T.copy(
                    X2[b, cs : cs + cv, y0cw0 : y0cw0 + ROWS * W_PAD],
                    staging[0, 0:cv, 0 : ROWS * W_PAD],
                )
                T.barrier_all()

                for oh_i in T.serial(OH):
                    cur = oh_i % 2
                    nxt = (oh_i + 1) % 2
                    ohg = oh_i % (MERGE_K if MERGE_K > 0 else 1)
                    if oh_i + 1 < OH:
                        y0cw_p = y0cw_ub[oh_i + 1]
                        T.copy(
                            X2[b, cs : cs + cv,
                               y0cw_p : y0cw_p + ROWS * W_PAD],
                            staging[nxt, 0:cv, 0 : ROWS * W_PAD],
                        )
                    nreff = nreff_ub[oh_i]
                    T.tile.fill(acc_U, 0.0)
                    for r in T.serial(nreff):
                        t = oh_i * NREFFMAX + r
                        rb = rbyte_ub[t]
                        w = rw_ub[t]
                        if is_fp16:
                            T.tile.gather(ghA, staging[cur, :, :], offA_ub, rb)
                            T.tile.cast(gA, ghA, "CAST_NONE", C_BLK * K_U)
                        else:
                            T.tile.gather(gA, staging[cur, :, :], offA_ub, rb)
                        T.tile.axpy(acc_U, gA, w)
                    T.tile.gather(gB, acc_U, offB_ub, 0)
                    T.tile.mul(gB, gB, wt_gw)
                    T.tile.select(
                        gB, selmask_ub, gB, 0.0,
                        "VSEL_TENSOR_SCALAR_MODE",
                    )
                    T.reduce_sum(gB, out_flat, dim=0)
                    if MERGE_K > 0:
                        if is_fp16:
                            T.tile.mul(out_flat, out_flat, cnt)
                            T.tile.cast(
                                rc_all[ohg, 0 : C_BLK * OW],
                                out_flat[0, 0 : C_BLK * OW],
                                CAST_MODE_HIGH2LOW, C_BLK * OW)
                        else:
                            T.tile.mul(
                                rc_all[ohg, 0 : C_BLK * OW],
                                out_flat[0, 0 : C_BLK * OW], cnt)
                    else:
                        T.tile.mul(out_flat, out_flat, cnt)
                        if is_fp16:
                            T.tile.cast(rc_h, out_flat, CAST_MODE_HIGH2LOW,
                                        C_BLK * OW)
                            T.tile.gather(out_pad[cur, :, :], rc_h,
                                          offb_ub, 0)
                        else:
                            T.tile.gather(out_pad[cur, :, :], out_flat[0, :],
                                          offb_ub, 0)
                    T.barrier_all()
                    if MERGE_K == 0:
                        T.copy(out_pad[cur, 0:cv, 0:OW],
                               Y[n, cs : cs + cv, oh_i, 0:OW])
                    elif MERGE_K == 1:
                        T.copy(rc_all[0, 0 : cv * OW],
                               Y[n, cs * OW : (cs + cv) * OW])
                    elif one_d:
                        if oh_i + 1 == OH:
                            T.tile.gather(out_all, rc_all, offb_ub, 0)
                            T.barrier_all()
                            T.copy(out_all[0 : cv * OH * OW],
                                   Y[n, cs * OH * OW : (cs + cv) * OH * OW])
                    else:
                        if (oh_i + 1) % MERGE_K == 0 or oh_i + 1 == OH:
                            kr = ohg + 1
                            oh_a = oh_i - ohg
                            T.tile.gather(out_all, rc_all, offb_ub, 0)
                            T.barrier_all()
                            T.copy(out_all[0:cv, 0 : kr * OW],
                                   Y[n, cs : cs + cv,
                                     oh_a * OW : oh_a * OW + kr * OW])

    return main

def _clamp_axis(coord, size, valid):
    empty = (coord < -1.0) | (coord > size)
    coord = torch.clamp(coord, min=0.0)
    low = coord.int()
    at_edge = low >= (size - 1)
    high = torch.where(at_edge, torch.full_like(low, size - 1), low + 1)
    low_c = torch.where(at_edge, torch.full_like(low, size - 1), low)
    lw = coord - low_c.to(coord.dtype)
    hw = 1.0 - lw
    zero = torch.zeros_like(hw)
    bad = empty | ~valid
    w_lo = torch.where(bad, zero, hw)
    w_hi = torch.where(bad, zero, lw)
    low_c = torch.where(bad, torch.zeros_like(low_c), low_c)
    return low_c, high, w_lo, w_hi


def _precompute_geometry(
    boxes, OH, OW, spatial_scale, sampling_ratio, aligned, H, W, GH, GW
):
    N = boxes.shape[0]
    dev = boxes.device
    b = boxes.float()

    rs_w = b[:, 1] * spatial_scale - (0.5 if aligned else 0.0)
    rs_h = b[:, 2] * spatial_scale - (0.5 if aligned else 0.0)
    re_w = b[:, 3] * spatial_scale - (0.5 if aligned else 0.0)
    re_h = b[:, 4] * spatial_scale - (0.5 if aligned else 0.0)
    rw = re_w - rs_w
    rh = re_h - rs_h
    if not aligned:
        rw = torch.clamp(rw, min=1.0)
        rh = torch.clamp(rh, min=1.0)
    bh = rh / OH
    bw = rw / OW

    exact = sampling_ratio > 0
    if exact:
        gh_n = torch.full((N,), sampling_ratio, dtype=torch.int64, device=dev)
        gw_n = torch.full((N,), sampling_ratio, dtype=torch.int64, device=dev)
        count = max(sampling_ratio * sampling_ratio, 1)
    else:
        b64 = _to_f64(boxes)
        assert b64.dtype == torch.float64
        offset64 = 0.5 if aligned else 0.0
        rs_h64 = b64[:, 2] * spatial_scale - offset64
        rs_w64 = b64[:, 1] * spatial_scale - offset64
        rh64 = (b64[:, 4] * spatial_scale - offset64) - rs_h64
        rw64 = (b64[:, 3] * spatial_scale - offset64) - rs_w64
        if not aligned:
            rh64 = torch.clamp(rh64, min=1.0)
            rw64 = torch.clamp(rw64, min=1.0)
        gh_n = torch.ceil(rh64 / OH).long()
        gw_n = torch.ceil(rw64 / OW).long()
        count = torch.clamp(gh_n * gw_n, min=1)

    ph = torch.arange(OH, device=dev, dtype=torch.float32)
    iy = torch.arange(GH, device=dev, dtype=torch.float32)
    gh_d = gh_n.float().clamp(min=1.0)
    y_base = rs_h[:, None, None] + ph[None, :, None] * bh[:, None, None]
    y = y_base + (iy[None, None, :] + 0.5) * (bh / gh_d)[:, None, None]
    pw = torch.arange(OW, device=dev, dtype=torch.float32)
    ix = torch.arange(GW, device=dev, dtype=torch.float32)
    gw_d = gw_n.float().clamp(min=1.0)
    x_base = rs_w[:, None, None] + pw[None, :, None] * bw[:, None, None]
    x = x_base + (ix[None, None, :] + 0.5) * (bw / gw_d)[:, None, None]

    y_valid = (iy.long()[None, None, :] < gh_n[:, None, None]).expand(N, OH, GH)
    x_valid = (ix.long()[None, None, :] < gw_n[:, None, None]).expand(N, OW, GW)
    assert gh_n.dtype == torch.int64 and gw_n.dtype == torch.int64

    ylo, yhi, wylo, wyhi = _clamp_axis(y, H, y_valid)
    xlo, xhi, wxlo, wxhi = _clamp_axis(x, W, x_valid)
    return (ylo, yhi, wylo, wyhi, xlo, xhi, wxlo, wxhi, gh_n, gw_n, count,
            y_valid, x_valid)


def _merge_axis_weights(lo, hi, w_lo, w_hi, valid):
    N, K, _ = lo.shape
    dev = lo.device
    big = torch.full_like(lo, 1 << 30)
    lo_v = torch.where(valid, lo, big)
    hi_v = torch.where(valid, hi, torch.full_like(hi, -(1 << 30)))
    has_any = valid.any(dim=2)
    zero = torch.zeros((), dtype=lo.dtype, device=dev)
    b_lo = torch.where(has_any, lo_v.amin(dim=2), zero.expand_as(has_any))
    b_hi = torch.where(has_any, hi_v.amax(dim=2), zero.expand_as(has_any))
    nr = (b_hi - b_lo + 1).clamp(min=1)
    nrmax = int(nr.max().item())
    wt = torch.zeros(N * K * nrmax, dtype=torch.float64, device=dev)
    base = (
        torch.arange(N * K, dtype=torch.int64, device=dev).view(N, K, 1) * nrmax
    )
    il = lo.long() - b_lo.long().unsqueeze(2)
    m1 = valid & (il >= 0) & (il < nrmax)
    wt.index_add_(0, (base + il.clamp(0, nrmax - 1))[m1], _to_f64(w_lo[m1]))
    ih = hi.long() - b_lo.long().unsqueeze(2)
    m2 = valid & (ih >= 0) & (ih < nrmax)
    wt.index_add_(0, (base + ih.clamp(0, nrmax - 1))[m2], _to_f64(w_hi[m2]))
    assert wt.dtype == torch.float64
    return b_lo, b_hi, wt.view(N, K, nrmax)


def _build_twostep_tables(geom, N, OH, OW, H, W, esize, OW_PAD):
    (ylo, yhi, wylo, wyhi, xlo, xhi, wxlo, wxhi, gh_n, gw_n, count,
     y_valid, x_valid) = geom
    dev = ylo.device
    rlo, rhi, rw = _merge_axis_weights(ylo, yhi, wylo, wyhi, y_valid)
    clo, chi, cw = _merge_axis_weights(xlo, xhi, wxlo, wxhi, x_valid)
    nr = (rhi - rlo + 1).clamp(min=1)
    nc = (chi - clo + 1).clamp(min=1)
    NRMAX = int(nr.max().item())
    NCMAX = int(nc.max().item())
    NC_PAD = _align_elems(NCMAX, 4)
    D3 = OW * NC_PAD
    TNR = OH * NRMAX
    ROWS = min(NRMAX, H)

    y0c = torch.minimum(rlo, torch.full_like(rlo, H - ROWS))
    rr = torch.arange(NRMAX, dtype=torch.int64, device=dev).view(1, 1, -1)
    rowidx = (rlo.long().unsqueeze(2) + rr - y0c.long().unsqueeze(2)).clamp(
        0, ROWS - 1
    )
    rbyte_t = (rowidx * (W * esize)).to(torch.int32).reshape(N, TNR).contiguous()
    rw_t = rw.to(torch.float32).reshape(N, TNR).contiguous()

    kk = torch.arange(NC_PAD, dtype=torch.int64, device=dev)
    cw_g = cw
    cw_g = torch.nn.functional.pad(cw_g, (0, NC_PAD - NCMAX))
    wt_g = cw_g
    clo_g = clo.long()
    cp_g = (clo_g.unsqueeze(2) + kk.view(1, 1, -1)).clamp(max=W - 1) * esize
    wt_t = wt_g.reshape(N, D3).to(torch.float32).contiguous()
    cp_t = cp_g.reshape(N, D3).to(torch.int32).contiguous()
    nr_t = nr.to(torch.int32).contiguous()

    y0cw_t = (y0c * W).to(torch.int32).contiguous()
    return dict(
        rw_t=rw_t, rbyte_t=rbyte_t, wt_t=wt_t, cp_t=cp_t, y0cw_t=y0cw_t,
        nr_t=nr_t, OW=OW,
        NRMAX=NRMAX, NCMAX=NCMAX, NC_PAD=NC_PAD, D3=D3, TNR=TNR, ROWS=ROWS,
        count=count,
    )


TSK_ELEM_RATIO_THRESH = 0.80


def _compact_positions(occ):
    nmax = max(int(occ.sum(dim=2).max().item()), 1)
    n, k, s = occ.shape
    dev = occ.device
    pos = torch.arange(s, dtype=torch.int64, device=dev).view(1, 1, -1).expand(n, k, s)
    rank = (occ.to(torch.int64).cumsum(dim=2) - 1).clamp(min=0)
    idx = torch.where(occ, rank, torch.full_like(rank, nmax))
    tmp = torch.full((n, k, nmax + 1), -1, dtype=torch.int64, device=dev)
    tmp.scatter_(2, idx, pos)
    return tmp[:, :, :nmax]


def _build_tsk_tables(geom, N, OH, OW, H, W, esize, mean_fp64=True):
    (ylo, yhi, wylo, wyhi, xlo, xhi, wxlo, wxhi, gh_n, gw_n, count,
     y_valid, x_valid) = geom
    dev = ylo.device
    rlo, rhi, rw = _merge_axis_weights(ylo, yhi, wylo, wyhi, y_valid)
    clo, chi, cw = _merge_axis_weights(xlo, xhi, wxlo, wxhi, x_valid)
    nr = (rhi - rlo + 1).clamp(min=1)
    NRMAX = int(nr.max().item())
    ROWS = min(NRMAX, H)

    ylo_e = ylo.reshape(N, OH, -1).long()
    yhi_e = yhi.reshape(N, OH, -1).long()
    rlo_e = rlo.long().unsqueeze(2)
    occ_rows = torch.zeros(N, OH, NRMAX, dtype=torch.bool, device=dev)
    for pos in (ylo_e, yhi_e):
        d = pos - rlo_e
        m = (d >= 0) & (d < NRMAX)
        occ_rows.scatter_(2, d.clamp(0, NRMAX - 1), m)
    nreff = occ_rows.sum(dim=2)
    NREFFMAX = max(int(nreff.max().item()), 1)
    row_sel = _compact_positions(occ_rows)
    row_valid = row_sel >= 0
    row_sel_c = torch.where(row_valid, row_sel, torch.zeros_like(row_sel))
    y0c = torch.minimum(rlo, torch.full_like(rlo, H - ROWS))
    rowidx = (rlo_e + row_sel_c - y0c.long().unsqueeze(2)).clamp(0, ROWS - 1)
    rbyte_t = (rowidx * (W * esize)).to(torch.int32).reshape(
        N, OH * NREFFMAX).contiguous()
    rw_t = torch.gather(rw, 2, row_sel_c)
    rw_t = torch.where(row_valid, rw_t, torch.zeros_like(rw_t))
    rw_t = rw_t.to(torch.float32).reshape(N, OH * NREFFMAX).contiguous()
    nreff_t = nreff.to(torch.int32).contiguous()
    y0cw_t = (y0c * W).to(torch.int32).contiguous()

    NCMAX = cw.shape[2]
    xlo_e = xlo.reshape(N, OW, -1).long()
    xhi_e = xhi.reshape(N, OW, -1).long()
    clo_e = clo.long().unsqueeze(2)
    occ_cols = torch.zeros(N, OW, NCMAX, dtype=torch.bool, device=dev)
    for pos in (xlo_e, xhi_e):
        d = pos - clo_e
        m = (d >= 0) & (d < NCMAX)
        occ_cols.scatter_(2, d.clamp(0, NCMAX - 1), m)
    cnt_j = occ_cols.sum(dim=2)
    NCEFF = max(int(cnt_j.max().item()), 1)
    col_sel = _compact_positions(occ_cols)
    col_valid = col_sel >= 0
    col_sel_c = torch.where(col_valid, col_sel, torch.zeros_like(col_sel))
    cols_abs = (clo_e + col_sel_c).clamp(max=W - 1)
    cols_abs = torch.where(col_valid, cols_abs, torch.zeros_like(cols_abs))
    wt_g = torch.gather(cw, 2, col_sel_c)
    wt_g = torch.where(col_valid, wt_g, torch.zeros_like(wt_g))
    cols_kj = cols_abs.transpose(1, 2).contiguous()
    valid_kj = col_valid.transpose(1, 2).contiguous()
    wt_tk = wt_g.to(torch.float32).transpose(1, 2).reshape(
        N, NCEFF * OW).contiguous()

    return dict(
        rbyte_t=rbyte_t, rw_t=rw_t, nreff_t=nreff_t, y0cw_t=y0cw_t,
        wt_tk=wt_tk, cols_kj=cols_kj, valid_kj=valid_kj,
        NREFFMAX=NREFFMAX, NCEFF=NCEFF, D3K=NCEFF * OW,
        TNR=OH * NREFFMAX, ROWS=ROWS, count=count,
        mean_nreff=(_to_f64(nreff).mean().item() if mean_fp64
                    else nreff.float().mean().item()),
    )


def _build_tsk_lane_tables(tsk, N, OW, W_PAD, esize, C_BLK, OW_PAD):
    ROWS = tsk["ROWS"]
    NCEFF = tsk["NCEFF"]
    dev = tsk["cols_kj"].device
    RB = ROWS * W_PAD * esize

    c_ar = torch.arange(C_BLK, dtype=torch.int64, device=dev).view(
        1, 1, C_BLK, 1)
    off = tsk["cols_kj"].long().unsqueeze(2) * esize + c_ar * RB
    off_t = off.reshape(N, NCEFF * C_BLK * OW).to(torch.int32).contiguous()

    per_roi_bits = NCEFF * C_BLK * OW
    mask_bytes = (per_roi_bits + 7) // 8
    mask_bytes = (mask_bytes + 31) // 32 * 32
    live = tsk["valid_kj"].unsqueeze(2).expand(
        N, NCEFF, C_BLK, OW).reshape(N, per_roi_bits).to(torch.uint8)
    bits = torch.nn.functional.pad(
        live, (0, mask_bytes * 8 - per_roi_bits)).view(N, mask_bytes, 8)
    selmask = torch.zeros(N, mask_bytes, dtype=torch.uint8, device=dev)
    for b in range(8):
        selmask |= bits[:, :, b] << b

    d_idx = (
        torch.arange(NCEFF, dtype=torch.int64, device=dev).view(
            NCEFF, 1, 1).expand(NCEFF, C_BLK, OW) * OW
        + torch.arange(OW, dtype=torch.int64, device=dev).view(
            1, 1, OW).expand(NCEFF, C_BLK, OW)
    )
    offmap_d = (d_idx.reshape(NCEFF * C_BLK * OW) * 4).to(
        torch.int32).contiguous()

    jc = torch.clamp(torch.arange(OW_PAD, dtype=torch.int64, device=dev),
                     max=OW - 1)
    offb = (
        torch.arange(C_BLK, dtype=torch.int64, device=dev).view(C_BLK, 1) * OW
        + jc.view(1, OW_PAD)
    ) * esize
    offb = offb.reshape(-1).to(torch.int32).contiguous()

    return dict(off_t=off_t, selmask=selmask, offmap_d=offmap_d, offb=offb,
                MASK_BYTES=mask_bytes)


def _build_us_tables(geom, N, OH, OW, H, W, esize):
    (ylo, yhi, wylo, wyhi, xlo, xhi, wxlo, wxhi, gh_n, gw_n, count,
     y_valid, x_valid) = geom
    GH = ylo.shape[2]
    GW = xlo.shape[2]
    dev = ylo.device

    rlo, rhi, rw = _merge_axis_weights(ylo, yhi, wylo, wyhi, y_valid)
    clo, chi, cw = _merge_axis_weights(xlo, xhi, wxlo, wxhi, x_valid)

    ylo_e = ylo.reshape(N, OH, GH).long()
    yhi_e = yhi.reshape(N, OH, GH).long()
    yv_e = y_valid.reshape(N, OH, GH)
    nr_span = (rhi - rlo + 1).clamp(min=1)
    NRMAX = int(nr_span.max())
    ROWS = min(NRMAX, H)
    rows_rel = torch.arange(NRMAX, device=dev).view(1, 1, 1, -1)
    rbase = rlo.long().unsqueeze(2).unsqueeze(3)
    occ_rows = (
        ((ylo_e.unsqueeze(3) == rbase + rows_rel) & yv_e.unsqueeze(3)).any(2)
        | ((yhi_e.unsqueeze(3) == rbase + rows_rel) & yv_e.unsqueeze(3)).any(2)
    )
    nreff = occ_rows.sum(2).clamp(min=1)
    NREFFMAX = max(int(nreff.max()), 1)
    row_sel = _compact_positions(occ_rows)
    row_valid = row_sel >= 0
    row_sel_c = torch.where(row_valid, row_sel, torch.zeros_like(row_sel))
    y0c = torch.minimum(rlo, torch.full_like(rlo, H - ROWS))
    rowidx = (rlo.long().unsqueeze(2) + row_sel_c
              - y0c.long().unsqueeze(2)).clamp(0, ROWS - 1)
    rbyte_t = (rowidx * (W * esize)).to(torch.int32).reshape(
        N, OH * NREFFMAX).contiguous()
    rw_t = torch.gather(rw, 2, row_sel_c)
    rw_t = torch.where(row_valid, rw_t, torch.zeros_like(rw_t))
    rw_t = rw_t.to(torch.float32).reshape(N, OH * NREFFMAX).contiguous()
    nreff_t = nreff.to(torch.int32).contiguous()
    y0cw_t = (y0c * W).to(torch.int32).contiguous()

    xlo_e = xlo.reshape(N, OW, GW).long()
    xhi_e = xhi.reshape(N, OW, GW).long()
    xv_e = x_valid.reshape(N, OW, GW)
    nc_span = (chi - clo + 1).clamp(min=1)
    NCMAX = int(nc_span.max())
    cols_rel = torch.arange(NCMAX, device=dev).view(1, 1, 1, -1)
    cbase = clo.long().unsqueeze(2).unsqueeze(3)
    occ_cols = (
        ((xlo_e.unsqueeze(3) == cbase + cols_rel) & xv_e.unsqueeze(3)).any(2)
        | ((xhi_e.unsqueeze(3) == cbase + cols_rel) & xv_e.unsqueeze(3)).any(2)
    )
    cnt_j = occ_cols.sum(2).clamp(min=1)
    NCEFF = max(int(cnt_j.max()), 1)
    col_sel = _compact_positions(occ_cols)
    col_valid = col_sel >= 0
    col_sel_c = torch.where(col_valid, col_sel, torch.zeros_like(col_sel))
    cols_abs = (clo.long().unsqueeze(2) + col_sel_c).clamp(max=W - 1)
    cols_abs = torch.where(col_valid, cols_abs, torch.zeros_like(cols_abs))
    wt_g = torch.gather(cw, 2, col_sel_c)
    wt_g = torch.where(col_valid, wt_g, torch.zeros_like(wt_g))
    cols_kj = cols_abs.transpose(1, 2).contiguous()
    valid_kj = col_valid.transpose(1, 2).contiguous()
    wt_tk = wt_g.to(torch.float32).transpose(1, 2).reshape(
        N, NCEFF * OW).contiguous()

    ar = torch.arange(W, device=dev).view(1, 1, -1)
    xv_f = xv_e.reshape(N, -1)
    occ_u = (
        ((xlo_e.reshape(N, -1).unsqueeze(2) == ar) & xv_f.unsqueeze(2)).any(1)
        | ((xhi_e.reshape(N, -1).unsqueeze(2) == ar) & xv_f.unsqueeze(2)).any(1)
    )
    u_list = _compact_positions(occ_u.unsqueeze(1))
    K_U = u_list.shape[2]
    u_valid = u_list >= 0
    u_list_c = torch.where(u_valid, u_list, torch.zeros_like(u_list))
    eq = ((cols_kj.unsqueeze(3)
           == u_list_c.squeeze(1).unsqueeze(1).unsqueeze(1))
          & u_valid.squeeze(1).view(N, 1, 1, -1))
    u_idx = (eq * torch.arange(K_U, device=dev).view(1, 1, 1, -1)).sum(-1)
    u_idx = torch.where(valid_kj, u_idx, torch.zeros_like(u_idx))

    E_stepA = float(
        (_to_f64(nreff) * _to_f64(occ_u.sum(1)).unsqueeze(1)).mean())
    E_us = E_stepA + float(NCEFF * OW)

    return dict(
        rbyte_t=rbyte_t, rw_t=rw_t, nreff_t=nreff_t, y0cw_t=y0cw_t,
        cols_u=u_list_c.squeeze(1).long(),
        u_idx=u_idx, valid_kj=valid_kj, wt_tk=wt_tk,
        NREFFMAX=NREFFMAX, NCEFF=NCEFF, K_U=K_U, D3B=NCEFF * OW,
        TNR=OH * NREFFMAX, ROWS=ROWS, count=count,
        mean_nreff=float(_to_f64(nreff).mean()),
        mean_nr_span=float(_to_f64(nr_span).mean()),
        E_us=E_us,
    )


def _build_us_lane_tables(us, N, OW, W_PAD, esize, C_BLK, OW_PAD):
    ROWS = us["ROWS"]
    K_U = us["K_U"]
    NCEFF = us["NCEFF"]
    D3B = us["D3B"]
    dev = us["cols_u"].device
    RB = ROWS * W_PAD * esize

    off = (us["cols_u"].long().unsqueeze(1) * esize
           + torch.arange(C_BLK, dtype=torch.int64, device=dev).view(
               1, C_BLK, 1) * RB)
    offA_t = off.reshape(N, C_BLK * K_U).to(torch.int32).contiguous()

    u4 = (us["u_idx"].long().unsqueeze(2)
          + torch.arange(C_BLK, dtype=torch.int64, device=dev).view(
              1, 1, C_BLK, 1) * K_U
          ) * 4
    offB_t = u4.reshape(N, C_BLK * D3B).to(torch.int32).contiguous()

    per_roi_bits = NCEFF * C_BLK * OW
    mask_bytes = (per_roi_bits + 7) // 8
    mask_bytes = (mask_bytes + 31) // 32 * 32
    live = us["valid_kj"].unsqueeze(2).expand(
        N, NCEFF, C_BLK, OW).reshape(N, per_roi_bits).to(torch.uint8)
    bits = torch.nn.functional.pad(
        live, (0, mask_bytes * 8 - per_roi_bits)).view(N, mask_bytes, 8)
    selmask = torch.zeros(N, mask_bytes, dtype=torch.uint8, device=dev)
    for b in range(8):
        selmask |= bits[:, :, b] << b

    d_idx = (
        torch.arange(NCEFF, dtype=torch.int64, device=dev).view(
            NCEFF, 1, 1).expand(NCEFF, C_BLK, OW) * OW
        + torch.arange(OW, dtype=torch.int64, device=dev).view(
            1, 1, OW).expand(NCEFF, C_BLK, OW)
    )
    offmap_d = (d_idx.reshape(NCEFF * C_BLK * OW) * 4).to(
        torch.int32).contiguous()

    jc = torch.clamp(torch.arange(OW_PAD, dtype=torch.int64, device=dev),
                     max=OW - 1)
    offb = (
        torch.arange(C_BLK, dtype=torch.int64, device=dev).view(C_BLK, 1) * OW
        + jc.view(1, OW_PAD)
    ) * esize
    offb = offb.reshape(-1).to(torch.int32).contiguous()

    return dict(offA_t=offA_t, offB_t=offB_t, selmask=selmask,
                offmap_d=offmap_d, offb=offb, MASK_BYTES=mask_bytes)


@tilelang.jit(out_idx=[2], pass_configs=PASS_CONFIGS)
def _ow_concat_kernel(R, OW, W0, dtype="float"):
    W1 = OW - W0
    esize = 2 if dtype == "float16" else 4
    ROWS_PB = 4
    W0P = (W0 * esize + 31) // 32 * 32 // esize
    W1P = (W1 * esize + 31) // 32 * 32 // esize
    r_num = T.ceildiv(R, 2 * ROWS_PB)

    @T.prim_func
    def main(
        A: T.Tensor((R, W0), dtype),  # type: ignore (part [0, W0) output)
        B: T.Tensor((R, W1), dtype),  # type: ignore (part [W0, OW) output)
        Y: T.Tensor((R, OW), dtype),  # type: ignore (assembled output)
    ):
        with T.Kernel(r_num, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((ROWS_PB, W0P), dtype)
            b_ub = T.alloc_ub((ROWS_PB, W1P), dtype)
            r0 = (cid * 2 + vid) * ROWS_PB
            nr = T.min(R - r0, ROWS_PB)
            if nr > 0:
                T.copy(A[r0 : r0 + nr, :], a_ub[0:nr, 0:W0])
                T.copy(B[r0 : r0 + nr, :], b_ub[0:nr, 0:W1])
                T.barrier_all()
                T.copy(a_ub[0:nr, 0:W0], Y[r0 : r0 + nr, 0:W0])
                T.copy(b_ub[0:nr, 0:W1], Y[r0 : r0 + nr, W0:W0 + W1])

    return main


def _ow_concat(y_parts, nw, C, OH, OW, dtype, dev):
    R = nw * C * OH
    (st0, en0, y0) = y_parts[0]
    (st1, en1, y1) = y_parts[1]
    W0 = en0 - st0
    dstr = _DTYPE_MAP[dtype]
    key = ("owcat", R, OW, W0, dstr)
    if key not in _kernel_cache:
        _kernel_cache[key] = _ow_concat_kernel(R, OW, W0, dstr)
    kernel = _kernel_cache[key]
    y = kernel(y0, y1)
    return y.view(nw, C, OH, OW)


@tilelang.jit(out_idx=[3], pass_configs=PASS_CONFIGS)
def _ow_assemble_kernel(N, K, L, dtype="float"):
    esize = 2 if dtype == "float16" else 4
    LP = (L * esize + 31) // 32 * 32 // esize

    @T.prim_func
    def main(
        A: T.Tensor((K, L), dtype),  # type: ignore (narrow full-OW rows)
        B: T.Tensor((N - K, L), dtype),  # type: ignore (wide concat rows)
        MP: T.Tensor((N, 2), "int32"),  # type: ignore (flag, rank per row)
        Y: T.Tensor((N, L), dtype),  # type: ignore (assembled output)
    ):
        with T.Kernel(N, is_npu=True) as (r, vid):
            if vid == 0:
                m = T.alloc_ub((2,), "int32")
                ub = T.alloc_ub((1, LP), dtype)
                T.copy(MP[r, 0:2], m)
                T.barrier_all()
                if m[0] > 0:
                    T.copy(A[m[1], 0:L], ub[0, 0:L])
                else:
                    T.copy(B[m[1], 0:L], ub[0, 0:L])
                T.barrier_all()
                T.copy(ub[0:1, 0:L], Y[r : r + 1, 0:L])

    return main


def _roi_align_ow_split(
    x, boxes_cpu, B, C, H, W, N, OH, OW, spatial_scale, sampling_ratio,
    aligned, dtype, geom, depth,
):
    gw_n = geom[9]
    b32 = boxes_cpu.float()
    off = 0.5 if aligned else 0.0
    rw = (b32[:, 3] * spatial_scale - off) - (b32[:, 1] * spatial_scale - off)
    if not aligned:
        rw = rw.clamp(min=1.0)
    bw = rw / OW
    narrow = gw_n <= 1
    has_narrow = bool(narrow.any())
    has_wide = bool((~narrow).any())

    def _full_ow_call(boxes_sub):
        return roi_align(
            x, boxes_sub, outputHeight=OH, outputWidth=OW,
            spatial_scale=spatial_scale, sampling_ratio=sampling_ratio,
            aligned=aligned, _split_depth=depth)

    if has_narrow and not has_wide:
        return _full_ow_call(boxes_cpu[narrow].contiguous())

    boxes_w = boxes_cpu[~narrow]
    bw_w = bw[~narrow]
    x0_w = boxes_w[:, 1].float()
    nw = boxes_w.shape[0]
    h1 = OW // 2
    parts = ((0, h1), (h1, OW)) if h1 > 0 else ((0, OW),)
    y_parts = []
    for st, en in parts:
        if en <= st:
            continue
        part = torch.zeros((nw, 5), dtype=torch.float32)
        part[:, 0] = boxes_w[:, 0].float()
        part[:, 2] = boxes_w[:, 2].float()
        part[:, 4] = boxes_w[:, 4].float()
        part[:, 1] = x0_w + st * bw_w / spatial_scale
        part[:, 3] = x0_w + en * bw_w / spatial_scale
        y_part = roi_align(
            x, part.contiguous(), outputHeight=OH, outputWidth=en - st,
            spatial_scale=spatial_scale, sampling_ratio=sampling_ratio,
            aligned=aligned, _split_depth=depth)
        y_parts.append((st, en, y_part))

    if len(y_parts) == 1:
        y_w = y_parts[0][2]
    else:
        y_w = _ow_concat(y_parts, nw, C, OH, OW, dtype, x.device)

    if not has_narrow:
        return y_w

    y_n = _full_ow_call(boxes_cpu[narrow].contiguous())
    L = C * OH * OW
    k = int(narrow.sum())
    sel = torch.zeros(N, 2, dtype=torch.int32)
    m_pos = narrow.nonzero().flatten()
    w_pos = (~narrow).nonzero().flatten()
    sel[m_pos, 0] = 1
    sel[m_pos, 1] = torch.arange(k, dtype=torch.int32)
    sel[w_pos, 1] = torch.arange(N - k, dtype=torch.int32)
    key = ("owasm", N, k, L, _DTYPE_MAP[dtype])
    if key not in _kernel_cache:
        _kernel_cache[key] = _ow_assemble_kernel(
            N, k, L, _DTYPE_MAP[dtype])
    y = _kernel_cache[key](
        y_n.reshape(k, L), y_w.reshape(N - k, L),
        sel.to(x.device, non_blocking=False))
    return y.view(N, C, OH, OW)


def _roi_align_ow_split_dev(
    x, boxes, B, C, H, W, N, OH, OW, spatial_scale, sampling_ratio, aligned,
    dtype, boxes_d, geom, depth,
):
    gw_n = geom[9]
    dev = x.device
    b32 = boxes_d.float()
    off = 0.5 if aligned else 0.0
    rw = (b32[:, 3] * spatial_scale - off) - (b32[:, 1] * spatial_scale - off)
    if not aligned:
        rw = rw.clamp(min=1.0)
    bw = rw / OW
    narrow = gw_n <= 1
    has_narrow = bool(narrow.any().item())
    has_wide = bool((~narrow).any().item())

    def _full_ow_call(boxes_sub):
        return roi_align(
            x, boxes_sub, outputHeight=OH, outputWidth=OW,
            spatial_scale=spatial_scale, sampling_ratio=sampling_ratio,
            aligned=aligned, _split_depth=depth).to(dtype)

    if has_narrow and not has_wide:
        return _full_ow_call(boxes[narrow].contiguous())

    boxes_w = boxes[~narrow]
    bw_w = bw[~narrow]
    x0_w = boxes_w[:, 1].float()
    nw = boxes_w.shape[0]
    h1 = OW // 2
    parts = ((0, h1), (h1, OW)) if h1 > 0 else ((0, OW),)
    y_parts = []
    for st, en in parts:
        if en <= st:
            continue
        part = torch.zeros((nw, 5), dtype=torch.float32, device=dev)
        part[:, 0] = boxes_w[:, 0].float()
        part[:, 2] = boxes_w[:, 2].float()
        part[:, 4] = boxes_w[:, 4].float()
        part[:, 1] = x0_w + st * bw_w / spatial_scale
        part[:, 3] = x0_w + en * bw_w / spatial_scale
        y_part = roi_align(
            x, part.contiguous(), outputHeight=OH, outputWidth=en - st,
            spatial_scale=spatial_scale, sampling_ratio=sampling_ratio,
            aligned=aligned, _split_depth=depth)
        y_parts.append((st, en, y_part))

    if len(y_parts) == 1:
        y_w = y_parts[0][2]
    else:
        y_w = _ow_concat(y_parts, nw, C, OH, OW, dtype, dev)

    if not has_narrow:
        return y_w

    y_n = _full_ow_call(boxes[narrow].contiguous())
    L = C * OH * OW
    k = int(narrow.sum().item())
    nar_i64 = narrow.to(torch.int64)
    rank_n = torch.cumsum(nar_i64, 0) - 1
    rank_w = torch.cumsum(1 - nar_i64, 0) - 1
    sel = torch.stack(
        [nar_i64, torch.where(narrow, rank_n, rank_w)], dim=1
    ).to(torch.int32).contiguous()
    key = ("owasm", N, k, L, _DTYPE_MAP[dtype])
    if key not in _kernel_cache:
        _kernel_cache[key] = _ow_assemble_kernel(
            N, k, L, _DTYPE_MAP[dtype])
    y = _kernel_cache[key](y_n.reshape(k, L), y_w.reshape(N - k, L), sel)
    return y.view(N, C, OH, OW)

def _roi_align_host(
    x: torch.Tensor,
    boxes: torch.Tensor,
    outputHeight: int,
    outputWidth: int,
    spatial_scale: float,
    sampling_ratio: int = -1,
    aligned: bool = False,
    _split_depth: int = 0,
) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError(f"x must be a 4D [B, C, H, W] tensor, got {x.dim()}D")
    if boxes.dim() != 2 or boxes.shape[1] != 5:
        raise ValueError(f"boxes must be [N, 5], got {tuple(boxes.shape)}")
    if x.dtype != boxes.dtype and _split_depth == 0:
        raise ValueError(f"x and boxes dtypes must match: {x.dtype} vs {boxes.dtype}")
    if int(outputHeight) <= 0 or int(outputWidth) <= 0:
        raise ValueError(
            f"outputHeight/outputWidth must be positive, got {outputHeight}/{outputWidth}"
        )

    B, C, H, W = x.shape
    N = boxes.shape[0]
    OH = int(outputHeight)
    OW = int(outputWidth)
    dtype = x.dtype
    dtype_str = _DTYPE_MAP[dtype]
    is_fp16 = dtype == torch.float16
    esize = 2 if is_fp16 else 4
    dev = x.device

    if N == 0:
        return torch.empty((0, C, OH, OW), dtype=dtype, device=dev)

    boxes_cpu = boxes.detach().cpu().contiguous()
    bidx_cpu = boxes_cpu[:, 0].long()
    if bool((bidx_cpu < 0).any() or (bidx_cpu >= B).any()):
        raise ValueError(f"boxes[:, 0] (batch index) must be within [0, {B})")

    if sampling_ratio > 0:
        GH = int(sampling_ratio)
        GW = int(sampling_ratio)
    else:
        b64 = boxes_cpu.double()
        offset = 0.5 if aligned else 0.0
        rh = b64[:, 4] * spatial_scale - offset - (b64[:, 2] * spatial_scale - offset)
        rw = b64[:, 3] * spatial_scale - offset - (b64[:, 1] * spatial_scale - offset)
        if not aligned:
            rh = torch.clamp(rh, min=1.0)
            rw = torch.clamp(rw, min=1.0)
        GH = max(int(torch.ceil(rh / OH).max().item()), 1)
        GW = max(int(torch.ceil(rw / OW).max().item()), 1)

    TGH = OH * GH
    W_PAD = _align_elems(W, esize)
    OW_PAD = _align_elems(OW, esize)

    def _staging_params(ylo_int, yhi_int, y_valid):
        ylo_r = ylo_int.reshape(N, OH, GH)
        yhi_r = yhi_int.reshape(N, OH, GH)
        big = torch.full_like(ylo_r, 1 << 30)
        ylo_v = torch.where(y_valid, ylo_r, big)
        yhi_v = torch.where(y_valid, yhi_r, torch.full_like(yhi_r, -(1 << 30)))
        ylo_min = ylo_v.amin(dim=2)
        yhi_max = yhi_v.amax(dim=2)
        span = (yhi_max - ylo_min + 1).clamp(min=1)
        rows = min(int(span.max().item()), H)
        y0c = torch.minimum(ylo_min, torch.full_like(ylo_min, H - rows))
        return rows, y0c

    cache_key = (
        B, C, H, W, N, OH, OW, GH, GW,
        float(spatial_scale), int(sampling_ratio), bool(aligned),
        dtype, str(dev),
    )
    cached = _host_table_cache.get(cache_key)
    if cached is not None and torch.equal(cached[0], boxes_cpu):
        mode = cached[1]
        data = cached[2]
    else:
        mode = None
        data = None
        geom = None

        def _pack_us(us, C_BLK_US, est_us):
            lane = _build_us_lane_tables(
                us, N, OW, W_PAD, esize, C_BLK_US, OW_PAD
            )
            merge_k = _select_merge_k(
                est_us, C_BLK_US, OW, OH, esize, OW_PAD, dtype != torch.float32
            )
            if merge_k > 0:
                CP = _align_elems(C_BLK_US * OW, esize)
                KP = _align_elems(merge_k * OW, esize)
                one_d = merge_k == OH
                map_t = _build_offc_map(
                    C_BLK_US, OW, OH, merge_k, CP, KP, one_d, esize
                )
                MAP_LEN = map_t.numel()
            else:
                map_t = lane["offb"]
                MAP_LEN = C_BLK_US * OW_PAD
            bidx_t = bidx_cpu.to(torch.int32).contiguous()
            if isinstance(us["count"], torch.Tensor):
                cnt_t = (1.0 / us["count"].double()).to(
                    torch.float32).contiguous()
            else:
                cnt_t = torch.full(
                    (N,), 1.0 / us["count"], dtype=torch.float32
                ).contiguous()
            MB = lane["MASK_BYTES"]
            selmask_words = lane["selmask"].reshape(-1).view(
                torch.int32)
            tables = [
                us["rw_t"], us["rbyte_t"], us["wt_tk"],
                lane["offA_t"], lane["offB_t"], selmask_words,
                lane["offmap_d"], us["y0cw_t"], bidx_t, cnt_t,
                us["nreff_t"], map_t,
            ]
            packed, layout = _pack_tables(tables)
            packed_dev = packed.to(dev, non_blocking=False)
            mask_i32 = (N * MB) // 4
            base_views = _unpack_on_device(
                packed_dev,
                [layout[i] for i in range(len(tables)) if i != 5],
                u32_slots=(3, 4, 5, 10),
            )
            selmask_dev = packed_dev[
                layout[5][0] : layout[5][0] + mask_i32
            ].view(torch.uint8)[0 : N * MB].view(N, MB)
            it = iter(base_views)
            views = [
                next(it), next(it), next(it), next(it), next(it),
                selmask_dev, next(it), next(it), next(it), next(it),
                next(it), next(it),
            ]
            return dict(
                views=views, ROWS=us["ROWS"], NREFFMAX=us["NREFFMAX"],
                NCEFF=us["NCEFF"], K_U=us["K_U"], D3B=us["D3B"],
                TNR=us["TNR"], C_BLK=C_BLK_US, MASK_BYTES=MB,
                MERGE_K=merge_k, MAP_LEN=MAP_LEN,
            )

        if sampling_ratio <= 0 and _split_depth == 0:
            geom = _precompute_geometry(
                boxes_cpu, OH, OW, spatial_scale, sampling_ratio, aligned, H, W, GH, GW
            )
            ts = _build_twostep_tables(geom, N, OH, OW, H, W, esize, OW_PAD)
            C_BLK_TS, est_ts = _select_c_blk_ts(
                ts["OW"], W_PAD, ts["D3"], ts["NC_PAD"], ts["TNR"], N,
                esize, ts["ROWS"], OH, OW_PAD,
            )
            us = _build_us_tables(geom, N, OH, OW, H, W, esize)
            E_ts = us["mean_nr_span"] * ts["NC_PAD"] * ts["OW"]
            if us["E_us"] <= US_ELEM_RATIO_THRESH * E_ts:
                C_BLK_US, est_us = _select_c_blk_us(
                    W_PAD, us["K_U"], us["NCEFF"], us["D3B"], us["TNR"], N,
                    esize, us["ROWS"], OH, OW, OW_PAD, C=C,
                )
                if C_BLK_US is not None:
                    data = _pack_us(us, C_BLK_US, est_us)
                    mode = "us"
            if mode is None and C_BLK_TS is not None:
                merge_k = _select_merge_k(
                    est_ts, C_BLK_TS, OW, OH, esize, OW_PAD,
                    dtype != torch.float32,
                )
                if merge_k > 0:
                    CP = _align_elems(C_BLK_TS * OW, esize)
                    KP = _align_elems(merge_k * OW, esize)
                    one_d = merge_k == OH
                    map_t = _build_offc_map(
                        C_BLK_TS, OW, OH, merge_k, CP, KP, one_d, esize
                    )
                    MAP_LEN = map_t.numel()
                else:
                    jc = torch.clamp(
                        torch.arange(OW_PAD, dtype=torch.int64), max=OW - 1
                    )
                    map_t = (
                        (
                            torch.arange(C_BLK_TS, dtype=torch.int64).view(C_BLK_TS, 1) * OW
                            + jc.view(1, OW_PAD)
                        )
                        * esize
                    ).reshape(-1).to(torch.int32).contiguous()
                    MAP_LEN = C_BLK_TS * OW_PAD
                bidx_t = bidx_cpu.to(torch.int32).contiguous()
                if isinstance(ts["count"], torch.Tensor):
                    cnt_t = (1.0 / ts["count"].double()).to(torch.float32).contiguous()
                else:
                    cnt_t = torch.full(
                        (N,), 1.0 / ts["count"], dtype=torch.float32
                    ).contiguous()
                packed, layout = _pack_tables(
                    [ts["rw_t"], ts["rbyte_t"], ts["wt_t"], ts["cp_t"],
                     ts["y0cw_t"], bidx_t, cnt_t, ts["nr_t"], map_t]
                )
                packed_dev = packed.to(dev, non_blocking=False)
                shapes = [
                    (N, ts["TNR"]), (N, ts["TNR"]), (N, ts["D3"]),
                    (N, ts["D3"]), (N, OH), (N,), (N,), (N, OH),
                    (MAP_LEN,),
                ]
                views = [
                    v.view(s)
                    for v, s in zip(
                        _unpack_on_device(packed_dev, layout, u32_slots=(8,)),
                        shapes,
                    )
                ]
                mode = "ts"
                data = dict(
                    views=views, ROWS=ts["ROWS"], NRMAX=ts["NRMAX"],
                    NC_PAD=ts["NC_PAD"], D3=ts["D3"], TNR=ts["TNR"],
                    C_BLK=C_BLK_TS, MERGE_K=merge_k, MAP_LEN=MAP_LEN,
                )

        _ss_dyad = spatial_scale > 0 and math.frexp(float(spatial_scale))[0] == 0.5
        if (mode is None and sampling_ratio > 0 and is_fp16
                and _split_depth == 0
                and sampling_ratio <= 4 and _ss_dyad
                and OH * OW <= 1024):
            geom = _precompute_geometry(
                boxes_cpu, OH, OW, spatial_scale, sampling_ratio, aligned,
                H, W, GH, GW,
            )
            tsk = _build_tsk_tables(
                geom, N, OH, OW, H, W, esize, mean_fp64=False)
            E_pw = GH * 4 * GW * OW
            E_tsk = tsk["mean_nreff"] * tsk["NCEFF"] * OW
            on_tsk = E_tsk < TSK_ELEM_RATIO_THRESH * E_pw
            us = _build_us_tables(geom, N, OH, OW, H, W, esize)
            E_cur = E_tsk if on_tsk else E_pw
            if us["E_us"] <= US_ELEM_RATIO_THRESH * E_cur:
                C_BLK_US, est_us = _select_c_blk_us(
                    W_PAD, us["K_U"], us["NCEFF"], us["D3B"], us["TNR"], N,
                    esize, us["ROWS"], OH, OW, OW_PAD, C=C,
                )
                if C_BLK_US is not None:
                    data = _pack_us(us, C_BLK_US, est_us)
                    mode = "us"
            if mode is None and on_tsk:
                C_BLK_TSK, est_tsk = _select_c_blk_tsk(
                    W_PAD, tsk["D3K"], tsk["TNR"], N, esize, tsk["ROWS"],
                    OH, OW, OW_PAD, C=C,
                )
                if C_BLK_TSK is not None:
                    lane = _build_tsk_lane_tables(
                        tsk, N, OW, W_PAD, esize, C_BLK_TSK, OW_PAD
                    )
                    merge_k = _select_merge_k(
                        est_tsk, C_BLK_TSK, OW, OH, esize, OW_PAD,
                        dtype != torch.float32,
                    )
                    if merge_k > 0:
                        CP = _align_elems(C_BLK_TSK * OW, esize)
                        KP = _align_elems(merge_k * OW, esize)
                        one_d = merge_k == OH
                        map_t = _build_offc_map(
                            C_BLK_TSK, OW, OH, merge_k, CP, KP, one_d, esize
                        )
                        MAP_LEN = map_t.numel()
                    else:
                        map_t = lane["offb"]
                        MAP_LEN = C_BLK_TSK * OW_PAD
                    bidx_t = bidx_cpu.to(torch.int32).contiguous()
                    if isinstance(tsk["count"], torch.Tensor):
                        cnt_t = (1.0 / tsk["count"].double()).to(
                            torch.float32).contiguous()
                    else:
                        cnt_t = torch.full(
                            (N,), 1.0 / tsk["count"], dtype=torch.float32
                        ).contiguous()
                    MB = lane["MASK_BYTES"]
                    selmask_words = lane["selmask"].reshape(-1).view(
                        torch.int32)
                    tables = [
                        tsk["rw_t"], tsk["rbyte_t"], tsk["wt_tk"],
                        lane["off_t"], selmask_words, lane["offmap_d"],
                        tsk["y0cw_t"], bidx_t, cnt_t, tsk["nreff_t"],
                        map_t,
                    ]
                    packed, layout = _pack_tables(tables)
                    packed_dev = packed.to(dev, non_blocking=False)
                    mask_i32 = (N * MB) // 4
                    base_views = _unpack_on_device(
                        packed_dev,
                        [layout[i] for i in range(len(tables)) if i != 4],
                        u32_slots=(3, 4, 9),
                    )
                    selmask_dev = packed_dev[
                        layout[4][0] : layout[4][0] + mask_i32
                    ].view(torch.uint8)[0 : N * MB].view(N, MB)
                    it = iter(base_views)
                    views = [
                        next(it), next(it), next(it), next(it),
                        selmask_dev, next(it), next(it), next(it), next(it),
                        next(it), next(it),
                    ]
                    mode = "tsk"
                    data = dict(
                        views=views, ROWS=tsk["ROWS"], NREFFMAX=tsk["NREFFMAX"],
                        NCEFF=tsk["NCEFF"], D3K=tsk["D3K"], TNR=tsk["TNR"],
                        C_BLK=C_BLK_TSK, MASK_BYTES=MB,
                        MERGE_K=merge_k, MAP_LEN=MAP_LEN, geom=geom,
                    )

        if mode is None:
            if geom is None:
                geom = _precompute_geometry(
                    boxes_cpu, OH, OW, spatial_scale, sampling_ratio, aligned,
                    H, W, GH, GW,
                )
            (ylo, yhi, wylo, wyhi, xlo, xhi, wxlo, wxhi, gh_n, gw_n, count,
             y_valid, x_valid) = geom
            stage_every_it = False

            ROWS, y0c = _staging_params(ylo, yhi, y_valid.reshape(N, OH, GH))

            row_fold = False
            if GH * GW in (64, 256) and ROWS < 2 * GH:
                C_BLK, est_pw = _select_c_blk_rf(
                    GW, OW, W_PAD, TGH, N, esize, ROWS, OH, OW_PAD, C=C)
                if C_BLK is not None:
                    row_fold = True
            if not row_fold:
                C_BLK, est_pw = _select_c_blk(
                    GW, OW, W_PAD, TGH, N, esize, ROWS, OH, OW_PAD, C=C,
                    min_cand=8)
                if C_BLK is None:
                    stage_every_it = True
                    ROWS = 2
                    C_BLK, est_pw = _select_c_blk(
                        GW, OW, W_PAD, TGH, N, esize, ROWS, OH, OW_PAD, C=C,
                        min_cbow_bytes=32)
                    if C_BLK is None:
                        for cand in (8, 4, 2, 1):
                            if (cand * OW) % 8 != 0:
                                continue
                            e = _est_pw(
                                cand, GW, OW, W_PAD, TGH, N, esize, ROWS,
                                OH, OW_PAD)
                            if e <= _PW_BUDGET:
                                C_BLK = cand
                                est_pw = e
                                break
                        if C_BLK is None:
                            if sampling_ratio <= 0 and _split_depth < 3:
                                return _roi_align_ow_split(
                                    x, boxes_cpu, B, C, H, W, N, OH, OW,
                                    spatial_scale, sampling_ratio, aligned,
                                    dtype, geom, _split_depth + 1,
                                )
                            C_BLK = 8
                            est_pw = _est_pw(
                                8, GW, OW, W_PAD, TGH, N, esize, ROWS, OH,
                                OW_PAD)
                    y0c = torch.minimum(
                        ylo.reshape(N, OH, GH),
                        torch.full_like(ylo.reshape(N, OH, GH), H - ROWS),
                    )

            ylo_r = ylo.reshape(N, OH, GH)
            yhi_r = yhi.reshape(N, OH, GH)
            if stage_every_it:
                y0c_e = y0c
            else:
                y0c_e = y0c.unsqueeze(2).expand(N, OH, GH)
            ylo_c = torch.clamp(ylo_r, min=y0c_e, max=y0c_e + (ROWS - 1))
            yhi_c = torch.clamp(yhi_r, min=y0c_e, max=y0c_e + (ROWS - 1))
            if row_fold:
                yroff_lo_t = (ylo_c - y0c_e).reshape(N, TGH).to(
                    torch.int32).contiguous()
                yroff_hi_t = (yhi_c - y0c_e).reshape(N, TGH).to(
                    torch.int32).contiguous()
            else:
                yroff_lo_t = ((ylo_c - y0c_e) * W * esize).reshape(N, TGH).to(torch.int32).contiguous()
                yroff_hi_t = ((yhi_c - y0c_e) * W * esize).reshape(N, TGH).to(torch.int32).contiguous()
            wylo_t = wylo.reshape(N, TGH).to(torch.float32).contiguous()
            wyhi_t = wyhi.reshape(N, TGH).to(torch.float32).contiguous()

            wxlo_flat = wxlo.transpose(1, 2).reshape(N, GW * OW).to(torch.float32).contiguous()
            wxhi_flat = wxhi.transpose(1, 2).reshape(N, GW * OW).to(torch.float32).contiguous()
            rowbase_c = (
                torch.arange(C_BLK, dtype=torch.int64).view(1, 1, C_BLK, 1)
                * (ROWS * W_PAD * esize)
            )
            xolo_e = (xlo.transpose(1, 2) * esize).to(torch.int64)
            xohi_e = (xhi.transpose(1, 2) * esize).to(torch.int64)
            offlo_t = (
                (xolo_e.unsqueeze(2) + rowbase_c).reshape(N, GW * C_BLK * OW)
                .to(torch.int32).contiguous()
            )
            offhi_t = (
                (xohi_e.unsqueeze(2) + rowbase_c).reshape(N, GW * C_BLK * OW)
                .to(torch.int32).contiguous()
            )
            wxp_t = torch.cat([wxlo_flat, wxhi_flat], dim=1).contiguous()

            jj = torch.arange(OW, dtype=torch.int64)
            offmap = (
                (torch.arange(GW, dtype=torch.int64).view(GW, 1, 1) * OW + jj.view(1, 1, OW))
                .expand(GW, C_BLK, OW)
                .reshape(GW * C_BLK * OW)
                * 4
            ).to(torch.int32).contiguous()
            offmap_wxlo_t = offmap.contiguous()
            offmap_wxhi_t = (offmap + GW * OW * 4).contiguous()
            merge_k = _select_merge_k(
                est_pw, C_BLK, OW, OH, esize, OW_PAD, dtype != torch.float32,
                budget=_PW_BUDGET,
            )
            if merge_k > 0:
                cv_min = min(
                    cv for cv in (
                        min(C - (cb * 2 * C_BLK + v * C_BLK), C_BLK)
                        for cb in range((C + 2 * C_BLK - 1) // (2 * C_BLK))
                        for v in (0, 1)
                    ) if cv > 0
                )
                wr = cv_min * OW * (OH if merge_k > 1 else 1)
                if wr * esize < 32:
                    merge_k = 0
            if merge_k > 0:
                CP = _align_elems(C_BLK * OW, esize)
                KP = _align_elems(merge_k * OW, esize)
                one_d = merge_k == OH
                map_t = _build_offc_map(
                    C_BLK, OW, OH, merge_k, CP, KP, one_d, esize)
                MAP_LEN = map_t.numel()
            else:
                jc = torch.clamp(torch.arange(OW_PAD, dtype=torch.int64), max=OW - 1)
                map_t = (
                    (
                        torch.arange(C_BLK, dtype=torch.int64).view(C_BLK, 1) * OW
                        + jc.view(1, OW_PAD)
                    )
                    * esize
                ).to(torch.int32).contiguous()
                MAP_LEN = C_BLK * OW_PAD

            if stage_every_it:
                y0cw_t = (y0c * W).reshape(N, TGH).to(torch.int32).contiguous()
            else:
                y0cw_t = (
                    (y0c * W).unsqueeze(2).expand(N, OH, GH).reshape(N, TGH)
                    .to(torch.int32).contiguous()
                )

            bidx_t = bidx_cpu.to(torch.int32).contiguous()
            if isinstance(count, torch.Tensor):
                cnt_t = (1.0 / count.double()).to(torch.float32).contiguous()
            else:
                cnt_t = torch.full((N,), 1.0 / count, dtype=torch.float32).contiguous()

            precise_w = not is_fp16 and GH * GW <= 4
            if precise_w:
                b64 = boxes_cpu.double()
                off64 = 0.5 if aligned else 0.0
                rs_h64 = b64[:, 2] * spatial_scale - off64
                rs_w64 = b64[:, 1] * spatial_scale - off64
                rh64 = (b64[:, 4] * spatial_scale - off64) - rs_h64
                rw64 = (b64[:, 3] * spatial_scale - off64) - rs_w64
                if not aligned:
                    rh64 = torch.clamp(rh64, min=1.0)
                    rw64 = torch.clamp(rw64, min=1.0)
                bh64 = rh64 / OH
                bw64 = rw64 / OW
                ph64 = torch.arange(OH, dtype=torch.float64).view(1, OH, 1)
                iy64 = (torch.arange(GH, dtype=torch.float64) + 0.5).view(1, 1, GH)
                y64 = (
                    rs_h64.view(N, 1, 1) + ph64 * bh64.view(N, 1, 1)
                    + iy64 * (bh64 / GH).view(N, 1, 1)
                )
                pw64 = torch.arange(OW, dtype=torch.float64).view(1, OW, 1)
                ix64 = (torch.arange(GW, dtype=torch.float64) + 0.5).view(1, 1, GW)
                x64 = (
                    rs_w64.view(N, 1, 1) + pw64 * bw64.view(N, 1, 1)
                    + ix64 * (bw64 / GW).view(N, 1, 1)
                )
                ylo64, yhi64, wylo64, wyhi64 = _clamp_axis(y64, H, y_valid)
                xlo64, xhi64, wxlo64, wxhi64 = _clamp_axis(x64, W, x_valid)
                ycross = ((ylo != ylo64) | (yhi != yhi64)).reshape(N, TGH)
                xcross = ((xlo != xlo64) | (xhi != xhi64)).transpose(1, 2).reshape(
                    N, GW * OW)
                wylop_t = torch.where(
                    ycross, wylo_t,
                    wylo64.reshape(N, TGH).to(torch.float32)).contiguous()
                wyhip_t = torch.where(
                    ycross, wyhi_t,
                    wyhi64.reshape(N, TGH).to(torch.float32)).contiguous()
                wxlop_flat = torch.where(
                    xcross, wxlo_flat,
                    wxlo64.transpose(1, 2).reshape(N, GW * OW).to(torch.float32)
                ).contiguous()
                wxhip_flat = torch.where(
                    xcross, wxhi_flat,
                    wxhi64.transpose(1, 2).reshape(N, GW * OW).to(torch.float32)
                ).contiguous()
                wxpp_t = torch.cat([wxlop_flat, wxhip_flat], dim=1).contiguous()

            TI_LEN = _align_elems(1 + 3 * TGH, 4)
            TF_LEN = _align_elems(
                1 + 2 * TGH + 2 * GW * OW
                + (2 * TGH + 2 * GW * OW if precise_w else 0), 4)
            WX = 1 + 2 * TGH
            ti_raw = torch.cat(
                [bidx_t.view(N, 1), yroff_lo_t, yroff_hi_t, y0cw_t], dim=1
            )
            if precise_w:
                tf_raw = torch.cat(
                    [cnt_t.view(N, 1), wylo_t, wyhi_t, wxp_t,
                     wylop_t, wyhip_t, wxpp_t], dim=1
                )
            else:
                tf_raw = torch.cat(
                    [cnt_t.view(N, 1), wylo_t, wyhi_t, wxp_t], dim=1
                )
            if ti_raw.shape[1] < TI_LEN:
                ti_raw = torch.nn.functional.pad(
                    ti_raw, (0, TI_LEN - ti_raw.shape[1]))
            if tf_raw.shape[1] < TF_LEN:
                tf_raw = torch.nn.functional.pad(
                    tf_raw, (0, TF_LEN - tf_raw.shape[1]))
            ti_t = ti_raw.contiguous()
            tf_t = tf_raw.contiguous()
            offmap_wxlo_t = (offmap_wxlo_t.to(torch.int64)
                             + WX * 4).to(torch.int32).contiguous()
            offmap_wxhi_t = (offmap_wxhi_t.to(torch.int64)
                             + WX * 4).to(torch.int32).contiguous()
            if precise_w:
                PXB = 1 + 4 * TGH + 2 * GW * OW
                offmap_wxlop_t = (offmap.to(torch.int64)
                                  + PXB * 4).to(torch.int32).contiguous()
                offmap_wxhip_t = ((offmap + GW * OW * 4).to(torch.int64)
                                  + PXB * 4).to(torch.int32).contiguous()
            else:
                offmap_wxlop_t = offmap_wxlo_t
                offmap_wxhip_t = offmap_wxhi_t

            FUSED = GW * C_BLK * OW
            packed, layout = _pack_tables(
                [ti_t, tf_t, offlo_t, offhi_t, offmap_wxlo_t,
                 offmap_wxhi_t, offmap_wxlop_t, offmap_wxhip_t, map_t]
            )
            packed_dev = packed.to(dev, non_blocking=False)
            shapes = [
                (N, TI_LEN), (N, TF_LEN), (N, FUSED), (N, FUSED),
                (FUSED,), (FUSED,), (FUSED,), (FUSED,), (MAP_LEN,),
            ]
            views = [
                v.view(s)
                for v, s in zip(
                    _unpack_on_device(
                        packed_dev, layout, u32_slots=(2, 3, 4, 5, 6, 7, 8)),
                    shapes,
                )
            ]

            mode = "old"
            data = dict(
                views=views, ROWS=ROWS, stage_every_it=stage_every_it,
                C_BLK=C_BLK, MERGE_K=merge_k, MAP_LEN=MAP_LEN,
                row_fold=row_fold,
            )

        _host_table_cache[cache_key] = (boxes_cpu, mode, data)
        while len(_host_table_cache) > _TABLE_CACHE_MAX_ENTRIES:
            _host_table_cache.popitem(last=False)

    if not x.is_contiguous():
        x = x.contiguous()
    x2 = x.view(B, C, H * W)

    if mode == "us":
        ROWS = data["ROWS"]
        C_BLK = data["C_BLK"]
        MERGE_K = data["MERGE_K"]
        key = ("us", B, C, H, W, N, OH, OW, C_BLK, W_PAD, OW_PAD,
               data["K_U"], data["NCEFF"], data["D3B"], data["TNR"],
               data["NREFFMAX"], ROWS, data["MASK_BYTES"], MERGE_K, dtype_str)
        if key not in _kernel_cache:
            _kernel_cache[key] = _roi_align_us_kernel(
                B, C, H, W, N, OH, OW,
                C_BLK=C_BLK,
                W_PAD=W_PAD,
                OW_PAD=OW_PAD,
                K_U=data["K_U"],
                NCEFF=data["NCEFF"],
                D3B=data["D3B"],
                TNR=data["TNR"],
                NREFFMAX=data["NREFFMAX"],
                ROWS=ROWS,
                MASK_BYTES=data["MASK_BYTES"],
                HW=H * W,
                MERGE_K=MERGE_K,
                dtype=dtype_str,
            )
        kernel = _kernel_cache[key]
        (rw_t, rbyte_t, wt_t, offA_t, offB_t, selmask_t, offmap_d_t, y0cw_t,
         bidx_t, cnt_t, nreff_t, offb_t) = data["views"]
        y = kernel(x2, rw_t, rbyte_t, wt_t, offA_t, offB_t, selmask_t,
                   offmap_d_t, y0cw_t, bidx_t, cnt_t, nreff_t, offb_t)
        if MERGE_K > 0:
            y = y.view(N, C, OH, OW)
        return y

    if mode == "tsk":
        ROWS = data["ROWS"]
        C_BLK = data["C_BLK"]
        MERGE_K = data["MERGE_K"]
        key = ("tsk", B, C, H, W, N, OH, OW, C_BLK, W_PAD, OW_PAD,
               data["NCEFF"], data["D3K"], data["TNR"], data["NREFFMAX"],
               ROWS, data["MASK_BYTES"], MERGE_K, dtype_str)
        if key not in _kernel_cache:
            _kernel_cache[key] = _roi_align_tsk_kernel(
                B, C, H, W, N, OH, OW,
                C_BLK=C_BLK,
                W_PAD=W_PAD,
                OW_PAD=OW_PAD,
                NCEFF=data["NCEFF"],
                D3K=data["D3K"],
                TNR=data["TNR"],
                NREFFMAX=data["NREFFMAX"],
                ROWS=ROWS,
                MASK_BYTES=data["MASK_BYTES"],
                HW=H * W,
                MERGE_K=MERGE_K,
                dtype=dtype_str,
            )
        kernel = _kernel_cache[key]
        (rw_t, rbyte_t, wt_t, off_t, selmask_t, offmap_d_t, y0cw_t, bidx_t,
         cnt_t, nreff_t, offb_t) = data["views"]
        y = kernel(x2, rw_t, rbyte_t, wt_t, off_t, selmask_t, offmap_d_t,
                   y0cw_t, bidx_t, cnt_t, nreff_t, offb_t)
        if MERGE_K > 0:
            y = y.view(N, C, OH, OW)
        return y

    if mode == "ts":
        ROWS = data["ROWS"]
        C_BLK = data["C_BLK"]
        MERGE_K = data["MERGE_K"]
        key = ("ts", B, C, H, W, N, OH, OW, C_BLK, W_PAD, OW_PAD,
               data["NC_PAD"], data["D3"], data["TNR"], data["NRMAX"], ROWS,
               MERGE_K, dtype_str)
        if key not in _kernel_cache:
            _kernel_cache[key] = _roi_align_ts_kernel(
            B,
            C,
            H,
            W,
            N,
            OH,
            OW,
            C_BLK=C_BLK,
            W_PAD=W_PAD,
            OW_PAD=OW_PAD,
            NC_PAD=data["NC_PAD"],
            D3=data["D3"],
            TNR=data["TNR"],
            NRMAX=data["NRMAX"],
            ROWS=ROWS,
            HW=H * W,
            MERGE_K=MERGE_K,
            dtype=dtype_str,
            )
        kernel = _kernel_cache[key]
        (rw_t, rbyte_t, wt_t, cp_t, y0cw_t, bidx_t, cnt_t, nr_t,
         offb_t) = data["views"]
        y = kernel(x2, rw_t, rbyte_t, wt_t, cp_t, y0cw_t, bidx_t, cnt_t,
                   nr_t, offb_t)
        if MERGE_K > 0:
            y = y.view(N, C, OH, OW)
        return y

    ROWS = data["ROWS"]
    stage_every_it = data["stage_every_it"]
    C_BLK = data["C_BLK"]
    MERGE_K = data["MERGE_K"]
    row_fold = data.get("row_fold", False)
    (ti_t, tf_t, offlo_t, offhi_t, offmap_wxlo_t, offmap_wxhi_t,
     offmap_wxlop_t, offmap_wxhip_t, offb_t) = data["views"]
    key = (B, C, H, W, N, OH, OW, GH, GW, C_BLK, W_PAD, OW_PAD, TGH, ROWS,
           dtype_str, stage_every_it, MERGE_K, row_fold)
    if key not in _kernel_cache:
        _kernel_cache[key] = _roi_align_kernel(
            B, C, H, W, N, OH, OW, GH, GW,
            C_BLK=C_BLK, W_PAD=W_PAD, OW_PAD=OW_PAD, TGH=TGH, ROWS=ROWS,
            HW=H * W, MERGE_K=MERGE_K, dtype=dtype_str,
            stage_every_it=stage_every_it, row_fold=row_fold,
        )
    kernel = _kernel_cache[key]
    y = kernel(x2, ti_t, tf_t, offlo_t, offhi_t, offmap_wxlo_t,
               offmap_wxhi_t, offmap_wxlop_t, offmap_wxhip_t, offb_t)
    if MERGE_K > 0:
        y = y.view(N, C, OH, OW)
    return y
def _roi_align_dev(
    x: torch.Tensor,
    boxes: torch.Tensor,
    outputHeight: int,
    outputWidth: int,
    spatial_scale: float,
    sampling_ratio: int = -1,
    aligned: bool = False,
    _split_depth: int = 0,
) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError(f"x must be a 4D [B, C, H, W] tensor, got {x.dim()}D")
    if boxes.dim() != 2 or boxes.shape[1] != 5:
        raise ValueError(f"boxes must be [N, 5], got {tuple(boxes.shape)}")
    if x.dtype != boxes.dtype and _split_depth == 0:
        raise ValueError(f"x and boxes dtypes must match: {x.dtype} vs {boxes.dtype}")
    if int(outputHeight) <= 0 or int(outputWidth) <= 0:
        raise ValueError(
            f"outputHeight/outputWidth must be positive, got {outputHeight}/{outputWidth}"
        )

    B, C, H, W = x.shape
    N = boxes.shape[0]
    OH = int(outputHeight)
    OW = int(outputWidth)
    dtype = x.dtype
    dtype_str = _DTYPE_MAP[dtype]
    is_fp16 = dtype == torch.float16
    esize = 2 if is_fp16 else 4
    dev = x.device

    if N == 0:
        return torch.empty((0, C, OH, OW), dtype=dtype, device=dev)

    boxes_d = boxes.detach().contiguous()

    geom_key = (
        B, C, H, W, N, OH, OW,
        float(spatial_scale), int(sampling_ratio), bool(aligned),
        dtype, str(dev),
    )
    mode = None
    data = None
    geom = None
    boxes_ptr = boxes_d.data_ptr()
    l1 = _dev_ptr_cache.get(boxes_ptr)
    if l1 is not None and l1[0]() is not None:
        mode = l1[1]
        data = l1[2]
        GH = l1[3]
        GW = l1[4]
        TGH = OH * GH
        W_PAD = _align_elems(W, esize)
        OW_PAD = _align_elems(OW, esize)
    else:
        cached = _dev_table_cache.get(geom_key)
        if cached is not None and torch.equal(cached[5], boxes_d):
            mode = cached[1]
            data = cached[2]
            GH = cached[3]
            GW = cached[4]
            TGH = OH * GH
            W_PAD = _align_elems(W, esize)
            OW_PAD = _align_elems(OW, esize)
            _remember_ptr(boxes, boxes_ptr, mode, data, GH, GW)
        else:
            mode = None
            data = None
            geom = None

            bidx_l = boxes_d[:, 0].long()
            if bool((bidx_l < 0).any() | (bidx_l >= B).any()):
                raise ValueError(
                    f"boxes[:, 0] (batch index) must be within [0, {B})")

            if sampling_ratio > 0:
                GH = int(sampling_ratio)
                GW = int(sampling_ratio)
            else:
                b64 = _to_f64(boxes_d)
                offset = 0.5 if aligned else 0.0
                rh = b64[:, 4] * spatial_scale - offset - (b64[:, 2] * spatial_scale - offset)
                rw = b64[:, 3] * spatial_scale - offset - (b64[:, 1] * spatial_scale - offset)
                if not aligned:
                    rh = torch.clamp(rh, min=1.0)
                    rw = torch.clamp(rw, min=1.0)
                GH = max(int(torch.ceil(rh / OH).long().max().item()), 1)
                GW = max(int(torch.ceil(rw / OW).long().max().item()), 1)

            TGH = OH * GH
            W_PAD = _align_elems(W, esize)
            OW_PAD = _align_elems(OW, esize)

            def _staging_params(ylo_int, yhi_int, y_valid):
                ylo_r = ylo_int.reshape(N, OH, GH)
                yhi_r = yhi_int.reshape(N, OH, GH)
                big = torch.full_like(ylo_r, 1 << 30)
                ylo_v = torch.where(y_valid, ylo_r, big)
                yhi_v = torch.where(y_valid, yhi_r,
                                    torch.full_like(yhi_r, -(1 << 30)))
                ylo_min = ylo_v.amin(dim=2)
                yhi_max = yhi_v.amax(dim=2)
                span = (yhi_max - ylo_min + 1).clamp(min=1)
                rows = min(int(span.max().item()), H)
                y0c = torch.minimum(ylo_min, torch.full_like(ylo_min, H - rows))
                return rows, y0c

            def _pack_us(us, C_BLK_US, est_us):
                lane = _build_us_lane_tables(
                    us, N, OW, W_PAD, esize, C_BLK_US, OW_PAD
                )
                merge_k = _select_merge_k(
                    est_us, C_BLK_US, OW, OH, esize, OW_PAD, dtype != torch.float32
                )
                if merge_k > 0:
                    CP = _align_elems(C_BLK_US * OW, esize)
                    KP = _align_elems(merge_k * OW, esize)
                    one_d = merge_k == OH
                    map_t = _build_offc_map(
                        C_BLK_US, OW, OH, merge_k, CP, KP, one_d, esize, dev
                    )
                    MAP_LEN = map_t.numel()
                else:
                    map_t = lane["offb"]
                    MAP_LEN = C_BLK_US * OW_PAD
                bidx_t = bidx_l.to(torch.int32).contiguous()
                if isinstance(us["count"], torch.Tensor):
                    cnt_t = (1.0 / _to_f64(us["count"])).to(
                        torch.float32).contiguous()
                else:
                    cnt_t = torch.full(
                        (N,), 1.0 / us["count"], dtype=torch.float32,
                        device=dev,
                    ).contiguous()
                views = [
                    us["rw_t"], us["rbyte_t"], us["wt_tk"],
                    lane["offA_t"].view(torch.uint32),
                    lane["offB_t"].view(torch.uint32),
                    lane["selmask"],
                    lane["offmap_d"].view(torch.uint32),
                    us["y0cw_t"], bidx_t, cnt_t,
                    us["nreff_t"], map_t.view(torch.uint32),
                ]
                return dict(
                    views=views, ROWS=us["ROWS"], NREFFMAX=us["NREFFMAX"],
                    NCEFF=us["NCEFF"], K_U=us["K_U"], D3B=us["D3B"],
                    TNR=us["TNR"], C_BLK=C_BLK_US, MASK_BYTES=lane["MASK_BYTES"],
                    MERGE_K=merge_k, MAP_LEN=MAP_LEN,
                )

            if sampling_ratio <= 0 and _split_depth == 0:
                geom = _precompute_geometry(
                    boxes_d, OH, OW, spatial_scale, sampling_ratio, aligned, H, W, GH, GW
                )
                ts = _build_twostep_tables(geom, N, OH, OW, H, W, esize, OW_PAD)
                C_BLK_TS, est_ts = _select_c_blk_ts(
                    ts["OW"], W_PAD, ts["D3"], ts["NC_PAD"], ts["TNR"], N,
                    esize, ts["ROWS"], OH, OW_PAD,
                )
                us = _build_us_tables(geom, N, OH, OW, H, W, esize)
                E_ts = us["mean_nr_span"] * ts["NC_PAD"] * ts["OW"]
                if us["E_us"] <= US_ELEM_RATIO_THRESH * E_ts:
                    C_BLK_US, est_us = _select_c_blk_us(
                        W_PAD, us["K_U"], us["NCEFF"], us["D3B"], us["TNR"], N,
                        esize, us["ROWS"], OH, OW, OW_PAD, C=C,
                    )
                    if C_BLK_US is not None:
                        data = _pack_us(us, C_BLK_US, est_us)
                        mode = "us"
                if mode is None and C_BLK_TS is not None:
                    merge_k = _select_merge_k(
                        est_ts, C_BLK_TS, OW, OH, esize, OW_PAD,
                        dtype != torch.float32,
                    )
                    if merge_k > 0:
                        CP = _align_elems(C_BLK_TS * OW, esize)
                        KP = _align_elems(merge_k * OW, esize)
                        one_d = merge_k == OH
                        map_t = _build_offc_map(
                            C_BLK_TS, OW, OH, merge_k, CP, KP, one_d, esize, dev
                        )
                        MAP_LEN = map_t.numel()
                    else:
                        jc = torch.clamp(
                            torch.arange(OW_PAD, dtype=torch.int64, device=dev),
                            max=OW - 1
                        )
                        map_t = (
                            (
                                torch.arange(
                                    C_BLK_TS, dtype=torch.int64, device=dev
                                ).view(C_BLK_TS, 1) * OW
                                + jc.view(1, OW_PAD)
                            )
                            * esize
                        ).reshape(-1).to(torch.int32).contiguous()
                        MAP_LEN = C_BLK_TS * OW_PAD
                    bidx_t = bidx_l.to(torch.int32).contiguous()
                    if isinstance(ts["count"], torch.Tensor):
                        cnt_t = (1.0 / _to_f64(ts["count"])).to(
                            torch.float32).contiguous()
                    else:
                        cnt_t = torch.full(
                            (N,), 1.0 / ts["count"], dtype=torch.float32,
                            device=dev,
                        ).contiguous()
                    views = [
                        ts["rw_t"].view(N, ts["TNR"]),
                        ts["rbyte_t"].view(N, ts["TNR"]),
                        ts["wt_t"].view(N, ts["D3"]),
                        ts["cp_t"].view(N, ts["D3"]),
                        ts["y0cw_t"].view(N, OH),
                        bidx_t.view(N),
                        cnt_t.view(N),
                        ts["nr_t"].view(N, OH),
                        map_t.view(MAP_LEN).view(torch.uint32),
                    ]
                    mode = "ts"
                    data = dict(
                        views=views, ROWS=ts["ROWS"], NRMAX=ts["NRMAX"],
                        NC_PAD=ts["NC_PAD"], D3=ts["D3"], TNR=ts["TNR"],
                        C_BLK=C_BLK_TS, MERGE_K=merge_k, MAP_LEN=MAP_LEN,
                    )

            _ss_dyad = spatial_scale > 0 and math.frexp(float(spatial_scale))[0] == 0.5
            if (mode is None and sampling_ratio > 0 and is_fp16
                    and _split_depth == 0
                    and sampling_ratio <= 4 and _ss_dyad
                    and OH * OW <= 1024):
                geom = _precompute_geometry(
                    boxes_d, OH, OW, spatial_scale, sampling_ratio, aligned,
                    H, W, GH, GW,
                )
                tsk = _build_tsk_tables(geom, N, OH, OW, H, W, esize)
                E_pw = GH * 4 * GW * OW
                E_tsk = tsk["mean_nreff"] * tsk["NCEFF"] * OW
                on_tsk = E_tsk < TSK_ELEM_RATIO_THRESH * E_pw
                us = _build_us_tables(geom, N, OH, OW, H, W, esize)
                E_cur = E_tsk if on_tsk else E_pw
                if us["E_us"] <= US_ELEM_RATIO_THRESH * E_cur:
                    C_BLK_US, est_us = _select_c_blk_us(
                        W_PAD, us["K_U"], us["NCEFF"], us["D3B"], us["TNR"], N,
                        esize, us["ROWS"], OH, OW, OW_PAD, C=C,
                    )
                    if C_BLK_US is not None:
                        data = _pack_us(us, C_BLK_US, est_us)
                        mode = "us"
                if mode is None and on_tsk:
                    C_BLK_TSK, est_tsk = _select_c_blk_tsk(
                        W_PAD, tsk["D3K"], tsk["TNR"], N, esize, tsk["ROWS"],
                        OH, OW, OW_PAD, C=C,
                    )
                    if C_BLK_TSK is not None:
                        lane = _build_tsk_lane_tables(
                            tsk, N, OW, W_PAD, esize, C_BLK_TSK, OW_PAD
                        )
                        merge_k = _select_merge_k(
                            est_tsk, C_BLK_TSK, OW, OH, esize, OW_PAD,
                            dtype != torch.float32,
                        )
                        if merge_k > 0:
                            CP = _align_elems(C_BLK_TSK * OW, esize)
                            KP = _align_elems(merge_k * OW, esize)
                            one_d = merge_k == OH
                            map_t = _build_offc_map(
                                C_BLK_TSK, OW, OH, merge_k, CP, KP, one_d, esize,
                                dev
                            )
                            MAP_LEN = map_t.numel()
                        else:
                            map_t = lane["offb"]
                            MAP_LEN = C_BLK_TSK * OW_PAD
                        bidx_t = bidx_l.to(torch.int32).contiguous()
                        if isinstance(tsk["count"], torch.Tensor):
                            cnt_t = (1.0 / _to_f64(tsk["count"])).to(
                                torch.float32).contiguous()
                        else:
                            cnt_t = torch.full(
                                (N,), 1.0 / tsk["count"], dtype=torch.float32,
                                device=dev,
                            ).contiguous()
                        views = [
                            tsk["rw_t"], tsk["rbyte_t"], tsk["wt_tk"],
                            lane["off_t"].view(torch.uint32),
                            lane["selmask"],
                            lane["offmap_d"].view(torch.uint32),
                            tsk["y0cw_t"], bidx_t, cnt_t, tsk["nreff_t"],
                            map_t.view(torch.uint32),
                        ]
                        mode = "tsk"
                        data = dict(
                            views=views, ROWS=tsk["ROWS"], NREFFMAX=tsk["NREFFMAX"],
                            NCEFF=tsk["NCEFF"], D3K=tsk["D3K"], TNR=tsk["TNR"],
                            C_BLK=C_BLK_TSK, MASK_BYTES=lane["MASK_BYTES"],
                            MERGE_K=merge_k, MAP_LEN=MAP_LEN,
                        )

            if mode is None:
                if geom is None:
                    geom = _precompute_geometry(
                        boxes_d, OH, OW, spatial_scale, sampling_ratio, aligned,
                        H, W, GH, GW,
                    )
                (ylo, yhi, wylo, wyhi, xlo, xhi, wxlo, wxhi, gh_n, gw_n, count,
                 y_valid, x_valid) = geom
                stage_every_it = False

                ROWS, y0c = _staging_params(ylo, yhi, y_valid.reshape(N, OH, GH))

                row_fold = False
                if GH * GW in (64, 256) and ROWS < 2 * GH:
                    C_BLK, est_pw = _select_c_blk_rf(
                        GW, OW, W_PAD, TGH, N, esize, ROWS, OH, OW_PAD, C=C)
                    if C_BLK is not None:
                        row_fold = True
                if not row_fold:
                    C_BLK, est_pw = _select_c_blk(
                        GW, OW, W_PAD, TGH, N, esize, ROWS, OH, OW_PAD, C=C,
                        min_cand=8)
                    if C_BLK is None:
                        stage_every_it = True
                        ROWS = 2
                        C_BLK, est_pw = _select_c_blk(
                            GW, OW, W_PAD, TGH, N, esize, ROWS, OH, OW_PAD, C=C,
                            min_cbow_bytes=32)
                        if C_BLK is None:
                            for cand in (8, 4, 2, 1):
                                if (cand * OW) % 8 != 0:
                                    continue
                                e = _est_pw(
                                    cand, GW, OW, W_PAD, TGH, N, esize, ROWS,
                                    OH, OW_PAD)
                                if e <= _PW_BUDGET:
                                    C_BLK = cand
                                    est_pw = e
                                    break
                            if C_BLK is None:
                                if sampling_ratio <= 0 and _split_depth < 3:
                                    return _roi_align_ow_split_dev(
                                        x, boxes, B, C, H, W, N, OH, OW,
                                        spatial_scale, sampling_ratio, aligned,
                                        dtype, boxes_d, geom, _split_depth + 1,
                                    )
                                C_BLK = 8
                                est_pw = _est_pw(
                                    8, GW, OW, W_PAD, TGH, N, esize, ROWS, OH,
                                    OW_PAD)
                        y0c = torch.minimum(
                            ylo.reshape(N, OH, GH),
                            torch.full_like(ylo.reshape(N, OH, GH), H - ROWS),
                        )

                ylo_r = ylo.reshape(N, OH, GH)
                yhi_r = yhi.reshape(N, OH, GH)
                if stage_every_it:
                    y0c_e = y0c
                else:
                    y0c_e = y0c.unsqueeze(2).expand(N, OH, GH)
                ylo_c = torch.clamp(ylo_r, min=y0c_e, max=y0c_e + (ROWS - 1))
                yhi_c = torch.clamp(yhi_r, min=y0c_e, max=y0c_e + (ROWS - 1))
                if row_fold:
                    yroff_lo_t = (ylo_c - y0c_e).reshape(N, TGH).to(
                        torch.int32).contiguous()
                    yroff_hi_t = (yhi_c - y0c_e).reshape(N, TGH).to(
                        torch.int32).contiguous()
                else:
                    yroff_lo_t = ((ylo_c - y0c_e) * W * esize).reshape(N, TGH).to(torch.int32).contiguous()
                    yroff_hi_t = ((yhi_c - y0c_e) * W * esize).reshape(N, TGH).to(torch.int32).contiguous()
                wylo_t = wylo.reshape(N, TGH).to(torch.float32).contiguous()
                wyhi_t = wyhi.reshape(N, TGH).to(torch.float32).contiguous()

                wxlo_flat = wxlo.transpose(1, 2).reshape(N, GW * OW).to(torch.float32).contiguous()
                wxhi_flat = wxhi.transpose(1, 2).reshape(N, GW * OW).to(torch.float32).contiguous()
                rowbase_c = (
                    torch.arange(C_BLK, dtype=torch.int64, device=dev).view(
                        1, 1, C_BLK, 1)
                    * (ROWS * W_PAD * esize)
                )
                xolo_e = (xlo.transpose(1, 2) * esize).to(torch.int64)
                xohi_e = (xhi.transpose(1, 2) * esize).to(torch.int64)
                offlo_t = (
                    (xolo_e.unsqueeze(2) + rowbase_c).reshape(N, GW * C_BLK * OW)
                    .to(torch.int32).contiguous()
                )
                offhi_t = (
                    (xohi_e.unsqueeze(2) + rowbase_c).reshape(N, GW * C_BLK * OW)
                    .to(torch.int32).contiguous()
                )
                wxp_t = torch.cat([wxlo_flat, wxhi_flat], dim=1).contiguous()

                jj = torch.arange(OW, dtype=torch.int64, device=dev)
                offmap = (
                    (torch.arange(GW, dtype=torch.int64, device=dev).view(
                        GW, 1, 1) * OW + jj.view(1, 1, OW))
                    .expand(GW, C_BLK, OW)
                    .reshape(GW * C_BLK * OW)
                    * 4
                ).to(torch.int32).contiguous()
                offmap_wxlo_t = offmap.contiguous()
                offmap_wxhi_t = (offmap + GW * OW * 4).contiguous()
                merge_k = _select_merge_k(
                    est_pw, C_BLK, OW, OH, esize, OW_PAD, dtype != torch.float32,
                    budget=_PW_BUDGET,
                )
                if merge_k > 0:
                    cv_min = min(
                        cv for cv in (
                            min(C - (cb * 2 * C_BLK + v * C_BLK), C_BLK)
                            for cb in range((C + 2 * C_BLK - 1) // (2 * C_BLK))
                            for v in (0, 1)
                        ) if cv > 0
                    )
                    wr = cv_min * OW * (OH if merge_k > 1 else 1)
                    if wr * esize < 32:
                        merge_k = 0
                if merge_k > 0:
                    CP = _align_elems(C_BLK * OW, esize)
                    KP = _align_elems(merge_k * OW, esize)
                    one_d = merge_k == OH
                    map_t = _build_offc_map(
                        C_BLK, OW, OH, merge_k, CP, KP, one_d, esize, dev)
                    MAP_LEN = map_t.numel()
                else:
                    jc = torch.clamp(
                        torch.arange(OW_PAD, dtype=torch.int64, device=dev),
                        max=OW - 1)
                    map_t = (
                        (
                            torch.arange(C_BLK, dtype=torch.int64, device=dev).view(
                                C_BLK, 1) * OW
                            + jc.view(1, OW_PAD)
                        )
                        * esize
                    ).to(torch.int32).contiguous()
                    MAP_LEN = C_BLK * OW_PAD

                if stage_every_it:
                    y0cw_t = (y0c * W).reshape(N, TGH).to(torch.int32).contiguous()
                else:
                    y0cw_t = (
                        (y0c * W).unsqueeze(2).expand(N, OH, GH).reshape(N, TGH)
                        .to(torch.int32).contiguous()
                    )

                bidx_t = bidx_l.to(torch.int32).contiguous()
                if isinstance(count, torch.Tensor):
                    cnt_t = (1.0 / _to_f64(count)).to(torch.float32).contiguous()
                else:
                    cnt_t = torch.full(
                        (N,), 1.0 / count, dtype=torch.float32, device=dev
                    ).contiguous()

                precise_w = not is_fp16 and GH * GW <= 4
                if precise_w:
                    b64 = _to_f64(boxes_d)
                    assert b64.dtype == torch.float64
                    off64 = 0.5 if aligned else 0.0
                    rs_h64 = b64[:, 2] * spatial_scale - off64
                    rs_w64 = b64[:, 1] * spatial_scale - off64
                    rh64 = (b64[:, 4] * spatial_scale - off64) - rs_h64
                    rw64 = (b64[:, 3] * spatial_scale - off64) - rs_w64
                    if not aligned:
                        rh64 = torch.clamp(rh64, min=1.0)
                        rw64 = torch.clamp(rw64, min=1.0)
                    bh64 = rh64 / OH
                    bw64 = rw64 / OW
                    ph64 = torch.arange(
                        OH, dtype=torch.float64, device=dev).view(1, OH, 1)
                    iy64 = (torch.arange(
                        GH, dtype=torch.float64, device=dev) + 0.5).view(1, 1, GH)
                    y64 = (
                        rs_h64.view(N, 1, 1) + ph64 * bh64.view(N, 1, 1)
                        + iy64 * (bh64 / GH).view(N, 1, 1)
                    )
                    pw64 = torch.arange(
                        OW, dtype=torch.float64, device=dev).view(1, OW, 1)
                    ix64 = (torch.arange(
                        GW, dtype=torch.float64, device=dev) + 0.5).view(1, 1, GW)
                    x64 = (
                        rs_w64.view(N, 1, 1) + pw64 * bw64.view(N, 1, 1)
                        + ix64 * (bw64 / GW).view(N, 1, 1)
                    )
                    ylo64, yhi64, wylo64, wyhi64 = _clamp_axis(y64, H, y_valid)
                    xlo64, xhi64, wxlo64, wxhi64 = _clamp_axis(x64, W, x_valid)
                    ycross = ((ylo != ylo64) | (yhi != yhi64)).reshape(N, TGH)
                    xcross = ((xlo != xlo64) | (xhi != xhi64)).transpose(1, 2).reshape(
                        N, GW * OW)
                    wylop_t = torch.where(
                        ycross, wylo_t,
                        wylo64.reshape(N, TGH).to(torch.float32)).contiguous()
                    wyhip_t = torch.where(
                        ycross, wyhi_t,
                        wyhi64.reshape(N, TGH).to(torch.float32)).contiguous()
                    wxlop_flat = torch.where(
                        xcross, wxlo_flat,
                        wxlo64.transpose(1, 2).reshape(N, GW * OW).to(torch.float32)
                    ).contiguous()
                    wxhip_flat = torch.where(
                        xcross, wxhi_flat,
                        wxhi64.transpose(1, 2).reshape(N, GW * OW).to(torch.float32)
                    ).contiguous()
                    wxpp_t = torch.cat([wxlop_flat, wxhip_flat], dim=1).contiguous()

                TI_LEN = _align_elems(1 + 3 * TGH, 4)
                TF_LEN = _align_elems(
                    1 + 2 * TGH + 2 * GW * OW
                    + (2 * TGH + 2 * GW * OW if precise_w else 0), 4)
                WX = 1 + 2 * TGH
                ti_raw = torch.cat(
                    [bidx_t.view(N, 1), yroff_lo_t, yroff_hi_t, y0cw_t], dim=1
                )
                if precise_w:
                    tf_raw = torch.cat(
                        [cnt_t.view(N, 1), wylo_t, wyhi_t, wxp_t,
                         wylop_t, wyhip_t, wxpp_t], dim=1
                    )
                else:
                    tf_raw = torch.cat(
                        [cnt_t.view(N, 1), wylo_t, wyhi_t, wxp_t], dim=1
                    )
                if ti_raw.shape[1] < TI_LEN:
                    ti_raw = torch.nn.functional.pad(
                        ti_raw, (0, TI_LEN - ti_raw.shape[1]))
                if tf_raw.shape[1] < TF_LEN:
                    tf_raw = torch.nn.functional.pad(
                        tf_raw, (0, TF_LEN - tf_raw.shape[1]))
                ti_t = ti_raw.contiguous()
                tf_t = tf_raw.contiguous()
                offmap_wxlo_t = (offmap_wxlo_t.to(torch.int64)
                                 + WX * 4).to(torch.int32).contiguous()
                offmap_wxhi_t = (offmap_wxhi_t.to(torch.int64)
                                 + WX * 4).to(torch.int32).contiguous()
                if precise_w:
                    PXB = 1 + 4 * TGH + 2 * GW * OW
                    offmap_wxlop_t = (offmap.to(torch.int64)
                                      + PXB * 4).to(torch.int32).contiguous()
                    offmap_wxhip_t = ((offmap + GW * OW * 4).to(torch.int64)
                                      + PXB * 4).to(torch.int32).contiguous()
                else:
                    offmap_wxlop_t = offmap_wxlo_t
                    offmap_wxhip_t = offmap_wxhi_t

                FUSED = GW * C_BLK * OW
                views = [
                    ti_t.view(N, TI_LEN),
                    tf_t.view(N, TF_LEN),
                    offlo_t.view(N, FUSED).view(torch.uint32),
                    offhi_t.view(N, FUSED).view(torch.uint32),
                    offmap_wxlo_t.view(FUSED).view(torch.uint32),
                    offmap_wxhi_t.view(FUSED).view(torch.uint32),
                    offmap_wxlop_t.view(FUSED).view(torch.uint32),
                    offmap_wxhip_t.view(FUSED).view(torch.uint32),
                    map_t.view(MAP_LEN).view(torch.uint32),
                ]

                mode = "old"
                data = dict(
                    views=views, ROWS=ROWS, stage_every_it=stage_every_it,
                    C_BLK=C_BLK, MERGE_K=merge_k, MAP_LEN=MAP_LEN,
                    row_fold=row_fold,
                )

            _dev_table_cache[geom_key] = (weakref.ref(boxes), mode, data, GH, GW,
                                      boxes_d.clone())
            while len(_dev_table_cache) > _TABLE_CACHE_MAX_ENTRIES:
                _dev_table_cache.popitem(last=False)
            _remember_ptr(boxes, boxes_ptr, mode, data, GH, GW)

    if not x.is_contiguous():
        x = x.contiguous()
    x2 = x.view(B, C, H * W)

    if mode == "us":
        ROWS = data["ROWS"]
        C_BLK = data["C_BLK"]
        MERGE_K = data["MERGE_K"]
        key = ("us", B, C, H, W, N, OH, OW, C_BLK, W_PAD, OW_PAD,
               data["K_U"], data["NCEFF"], data["D3B"], data["TNR"],
               data["NREFFMAX"], ROWS, data["MASK_BYTES"], MERGE_K, dtype_str)
        if key not in _kernel_cache:
            _kernel_cache[key] = _roi_align_us_kernel(
                B, C, H, W, N, OH, OW,
                C_BLK=C_BLK,
                W_PAD=W_PAD,
                OW_PAD=OW_PAD,
                K_U=data["K_U"],
                NCEFF=data["NCEFF"],
                D3B=data["D3B"],
                TNR=data["TNR"],
                NREFFMAX=data["NREFFMAX"],
                ROWS=ROWS,
                MASK_BYTES=data["MASK_BYTES"],
                HW=H * W,
                MERGE_K=MERGE_K,
                dtype=dtype_str,
            )
        kernel = _kernel_cache[key]
        (rw_t, rbyte_t, wt_t, offA_t, offB_t, selmask_t, offmap_d_t, y0cw_t,
         bidx_t, cnt_t, nreff_t, offb_t) = data["views"]
        y = _grid_capped_launch(
            lambda n: _roi_align_us_kernel(
                B, C, H, W, n, OH, OW,
                C_BLK=C_BLK, W_PAD=W_PAD, OW_PAD=OW_PAD,
                K_U=data["K_U"], NCEFF=data["NCEFF"], D3B=data["D3B"],
                TNR=data["TNR"], NREFFMAX=data["NREFFMAX"], ROWS=ROWS,
                MASK_BYTES=data["MASK_BYTES"], HW=H * W, MERGE_K=MERGE_K,
                dtype=dtype_str),
            N, C, C_BLK, key,
            [x2, rw_t, rbyte_t, wt_t, offA_t, offB_t, selmask_t,
             offmap_d_t, y0cw_t, bidx_t, cnt_t, nreff_t, offb_t],
            [False, True, True, True, True, True, True, False, True, True,
             True, True, False])
        if y is not None:
            if MERGE_K > 0:
                y = y.view(N, C, OH, OW)
            return y
        y = kernel(x2, rw_t, rbyte_t, wt_t, offA_t, offB_t, selmask_t,
                   offmap_d_t, y0cw_t, bidx_t, cnt_t, nreff_t, offb_t)
        if MERGE_K > 0:
            y = y.view(N, C, OH, OW)
        return y

    if mode == "tsk":
        ROWS = data["ROWS"]
        C_BLK = data["C_BLK"]
        MERGE_K = data["MERGE_K"]
        key = ("tsk", B, C, H, W, N, OH, OW, C_BLK, W_PAD, OW_PAD,
               data["NCEFF"], data["D3K"], data["TNR"], data["NREFFMAX"],
               ROWS, data["MASK_BYTES"], MERGE_K, dtype_str)
        if key not in _kernel_cache:
            _kernel_cache[key] = _roi_align_tsk_kernel(
                B, C, H, W, N, OH, OW,
                C_BLK=C_BLK,
                W_PAD=W_PAD,
                OW_PAD=OW_PAD,
                NCEFF=data["NCEFF"],
                D3K=data["D3K"],
                TNR=data["TNR"],
                NREFFMAX=data["NREFFMAX"],
                ROWS=ROWS,
                MASK_BYTES=data["MASK_BYTES"],
                HW=H * W,
                MERGE_K=MERGE_K,
                dtype=dtype_str,
            )
        kernel = _kernel_cache[key]
        (rw_t, rbyte_t, wt_t, off_t, selmask_t, offmap_d_t, y0cw_t, bidx_t,
         cnt_t, nreff_t, offb_t) = data["views"]
        y = _grid_capped_launch(
            lambda n: _roi_align_tsk_kernel(
                B, C, H, W, n, OH, OW,
                C_BLK=C_BLK, W_PAD=W_PAD, OW_PAD=OW_PAD,
                NCEFF=data["NCEFF"], D3K=data["D3K"], TNR=data["TNR"],
                NREFFMAX=data["NREFFMAX"], ROWS=ROWS,
                MASK_BYTES=data["MASK_BYTES"], HW=H * W, MERGE_K=MERGE_K,
                dtype=dtype_str),
            N, C, C_BLK, key,
            [x2, rw_t, rbyte_t, wt_t, off_t, selmask_t, offmap_d_t,
             y0cw_t, bidx_t, cnt_t, nreff_t, offb_t],
            [False, True, True, True, True, True, False, True, True, True,
             True, False])
        if y is not None:
            if MERGE_K > 0:
                y = y.view(N, C, OH, OW)
            return y
        y = kernel(x2, rw_t, rbyte_t, wt_t, off_t, selmask_t, offmap_d_t,
                   y0cw_t, bidx_t, cnt_t, nreff_t, offb_t)
        if MERGE_K > 0:
            y = y.view(N, C, OH, OW)
        return y

    if mode == "ts":
        ROWS = data["ROWS"]
        C_BLK = data["C_BLK"]
        MERGE_K = data["MERGE_K"]
        key = ("ts", B, C, H, W, N, OH, OW, C_BLK, W_PAD, OW_PAD,
               data["NC_PAD"], data["D3"], data["TNR"], data["NRMAX"], ROWS,
               MERGE_K, dtype_str)
        if key not in _kernel_cache:
            _kernel_cache[key] = _roi_align_ts_kernel(
            B,
            C,
            H,
            W,
            N,
            OH,
            OW,
            C_BLK=C_BLK,
            W_PAD=W_PAD,
            OW_PAD=OW_PAD,
            NC_PAD=data["NC_PAD"],
            D3=data["D3"],
            TNR=data["TNR"],
            NRMAX=data["NRMAX"],
            ROWS=ROWS,
            HW=H * W,
            MERGE_K=MERGE_K,
            dtype=dtype_str,
            )
        kernel = _kernel_cache[key]
        (rw_t, rbyte_t, wt_t, cp_t, y0cw_t, bidx_t, cnt_t, nr_t,
         offb_t) = data["views"]
        y = _grid_capped_launch(
            lambda n: _roi_align_ts_kernel(
                B, C, H, W, n, OH, OW,
                C_BLK=C_BLK, W_PAD=W_PAD, OW_PAD=OW_PAD,
                NC_PAD=data["NC_PAD"], D3=data["D3"], TNR=data["TNR"],
                NRMAX=data["NRMAX"], ROWS=ROWS, HW=H * W, MERGE_K=MERGE_K,
                dtype=dtype_str),
            N, C, C_BLK, key,
            [x2, rw_t, rbyte_t, wt_t, cp_t, y0cw_t, bidx_t, cnt_t,
             nr_t, offb_t],
            [False, True, True, True, True, True, True, True, True, False])
        if y is not None:
            if MERGE_K > 0:
                y = y.view(N, C, OH, OW)
            return y
        y = kernel(x2, rw_t, rbyte_t, wt_t, cp_t, y0cw_t, bidx_t, cnt_t,
                   nr_t, offb_t)
        if MERGE_K > 0:
            y = y.view(N, C, OH, OW)
        return y

    ROWS = data["ROWS"]
    stage_every_it = data["stage_every_it"]
    C_BLK = data["C_BLK"]
    MERGE_K = data["MERGE_K"]
    row_fold = data.get("row_fold", False)
    (ti_t, tf_t, offlo_t, offhi_t, offmap_wxlo_t, offmap_wxhi_t,
     offmap_wxlop_t, offmap_wxhip_t, offb_t) = data["views"]
    key = (B, C, H, W, N, OH, OW, GH, GW, C_BLK, W_PAD, OW_PAD, TGH, ROWS,
           dtype_str, stage_every_it, MERGE_K, row_fold)
    if key not in _kernel_cache:
        _kernel_cache[key] = _roi_align_kernel(
            B, C, H, W, N, OH, OW, GH, GW,
            C_BLK=C_BLK, W_PAD=W_PAD, OW_PAD=OW_PAD, TGH=TGH, ROWS=ROWS,
            HW=H * W, MERGE_K=MERGE_K, dtype=dtype_str,
            stage_every_it=stage_every_it, row_fold=row_fold,
        )
    kernel = _kernel_cache[key]
    y = _grid_capped_launch(
        lambda n: _roi_align_kernel(
            B, C, H, W, n, OH, OW, GH, GW,
            C_BLK=C_BLK, W_PAD=W_PAD, OW_PAD=OW_PAD, TGH=TGH, ROWS=ROWS,
            HW=H * W, MERGE_K=MERGE_K, dtype=dtype_str,
            stage_every_it=stage_every_it, row_fold=row_fold),
        N, C, C_BLK, key,
        [x2, ti_t, tf_t, offlo_t, offhi_t, offmap_wxlo_t,
         offmap_wxhi_t, offmap_wxlop_t, offmap_wxhip_t, offb_t],
        [False, True, True, True, True, False, False, False, False, False])
    if y is not None:
        if MERGE_K > 0:
            y = y.view(N, C, OH, OW)
        return y
    y = kernel(x2, ti_t, tf_t, offlo_t, offhi_t, offmap_wxlo_t,
               offmap_wxhi_t, offmap_wxlop_t, offmap_wxhip_t, offb_t)
    if MERGE_K > 0:
        y = y.view(N, C, OH, OW)
    return y


def roi_align(
    x: torch.Tensor,
    boxes: torch.Tensor,
    outputHeight: int,
    outputWidth: int,
    spatial_scale: float,
    sampling_ratio: int = -1,
    aligned: bool = False,
    _split_depth: int = 0,
) -> torch.Tensor:
    global _SLOT_EAGER_BROKEN
    if (not _SLOT_EAGER_BROKEN
            and x.device.type == "npu"
            and boxes.device.type == "npu"):
        try:
            return _roi_align_dev(
                x, boxes, outputHeight, outputWidth, spatial_scale,
                sampling_ratio, aligned, _split_depth)
        except RuntimeError as exc:
            if not _is_aclnn_failure(exc):
                raise
            _mark_slot_broken(exc)
    return _roi_align_host(
        x, boxes, outputHeight, outputWidth, spatial_scale,
        sampling_ratio, aligned, _split_depth)
