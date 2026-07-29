# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import os

import tilelang
from tilelang import language as T

tilelang.disable_cache()
from tvm import tir


DEFAULT_BLOCK_I = 128


DEFAULT_CORE_NUM = 24


def build_sparse_flash_mla(
    *,
    batch: int,
    max_seq: int,
    total_tokens: int,
    ori_block_num: int,
    ori_block_size: int,
    ori_table_len: int,
    cmp_block_num: int,
    cmp_block_size: int,
    cmp_table_len: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    topk_cmp: int,
    cmp_ratio: int,
    scenario: int,
    ori_win_left: int,
    softmax_scale: float,
    dtype: str,
    core_num: int,
):
    if scenario not in (1, 2, 3):
        raise NotImplementedError(f"only SWA (1), HCA (2), CSA (3) are implemented; got scenario={scenario}")
    if n_kv_heads != 1 or n_heads != 64 or head_dim != 512:
        raise ValueError(f"kernel assumes N1=64, N2=1, D=512 (got N1={n_heads}, N2={n_kv_heads}, D={head_dim})")
    if ori_win_left + 1 > DEFAULT_BLOCK_I:
        raise ValueError(
            f"ori_win_left={ori_win_left} exceeds single-tile window (BI={DEFAULT_BLOCK_I}); the ori window is one tile for both SWA and HCA"
        )

    if scenario == 1:
        return _build_swa(
            batch=batch,
            max_seq=max_seq,
            total_tokens=total_tokens,
            ori_block_num=ori_block_num,
            ori_block_size=ori_block_size,
            ori_table_len=ori_table_len,
            cmp_block_num=cmp_block_num,
            cmp_block_size=cmp_block_size,
            cmp_table_len=cmp_table_len,
            n_heads=n_heads,
            head_dim=head_dim,
            ori_win_left=ori_win_left,
            softmax_scale=float(softmax_scale),
            dtype=dtype,
            core_num=core_num,
        )

    if scenario == 3:
        return _build_csa(
            batch=batch,
            max_seq=max_seq,
            total_tokens=total_tokens,
            ori_block_num=ori_block_num,
            ori_block_size=ori_block_size,
            ori_table_len=ori_table_len,
            cmp_block_num=cmp_block_num,
            cmp_block_size=cmp_block_size,
            cmp_table_len=cmp_table_len,
            n_heads=n_heads,
            head_dim=head_dim,
            topk_cmp=topk_cmp,
            ori_win_left=ori_win_left,
            cmp_ratio=cmp_ratio,
            softmax_scale=float(softmax_scale),
            dtype=dtype,
            core_num=core_num,
        )

    return _build_hca(
        batch=batch,
        max_seq=max_seq,
        total_tokens=total_tokens,
        ori_block_num=ori_block_num,
        ori_block_size=ori_block_size,
        ori_table_len=ori_table_len,
        cmp_block_num=cmp_block_num,
        cmp_block_size=cmp_block_size,
        cmp_table_len=cmp_table_len,
        n_heads=n_heads,
        head_dim=head_dim,
        ori_win_left=ori_win_left,
        cmp_ratio=cmp_ratio,
        softmax_scale=float(softmax_scale),
        dtype=dtype,
        core_num=core_num,
    )


def _build_hca(
    *,
    batch,
    max_seq,
    total_tokens,
    ori_block_num,
    ori_block_size,
    ori_table_len,
    cmp_block_num,
    cmp_block_size,
    cmp_table_len,
    n_heads,
    head_dim,
    ori_win_left,
    cmp_ratio,
    softmax_scale,
    dtype,
    core_num,
):
    N1 = n_heads
    N2 = 1
    D = head_dim
    D2 = D // 2
    G = N1 // N2
    BI = DEFAULT_BLOCK_I
    PV_NT = D // BI
    PV_NW = BI

    KL0 = 128
    VEC_NUM = 2
    G2 = G // VEC_NUM
    BLK = 8
    S2_BASE = 512
    MAX_COLBLK = S2_BASE // BI

    M_CHUNK = 16
    NMC = G2 // M_CHUNK

    accum_dtype = "float"
    idx_dtype = "int32"
    out_cast_mode = "CAST_RINT" if dtype == "bfloat16" else "CAST_ROUND"

    max_cmp_len = (max_seq + cmp_ratio - 1) // cmp_ratio
    MAX_CMP_TILES = (max_cmp_len + S2_BASE - 1) // S2_BASE
    MAX_TILES = 1 + MAX_CMP_TILES

    block_num = batch * max_seq
    n_iter = (block_num + core_num - 1) // core_num

    GLOOP = n_iter * MAX_TILES

    EV_QK = 0
    EV_P = 1
    EV_PV = 2

    DEBUG_SERIAL = False

    KV_EV0 = 2
    KV_EV1 = 3
    KV_EV2 = 6

    Q_EV = 1
    P_EV = 7

    L0AB_EV0 = 4
    L0AB_EV1 = 5

    IN_EV = 2
    ACC_EV = 4
    OUT_EV = 5
    LSE_EV = 0

    FENCE = 0

    q_shape = [total_tokens, N1, D]
    ori_kv_shape = [ori_block_num, ori_block_size, N2, D]
    cmp_kv_shape = [cmp_block_num, cmp_block_size, N2, D]
    cmp_idx_shape = [total_tokens, N2, DEFAULT_BLOCK_I]

    @tilelang.jit(out_idx=[10, 11], workspace_idx=[12, 13, 14, 15, 16], target="ascendc")
    def kernel():
        @T.prim_func
        def sparse_flash_mla_hca(
            Q: T.Tensor(q_shape, dtype),
            ori_kv: T.Tensor(ori_kv_shape, dtype),
            ori_block_table: T.Tensor([batch, ori_table_len], idx_dtype),
            cmp_kv: T.Tensor(cmp_kv_shape, dtype),
            cmp_block_table: T.Tensor([batch, cmp_table_len], idx_dtype),
            cmp_indices: T.Tensor(cmp_idx_shape, idx_dtype),
            q_prefix: T.Tensor([batch], idx_dtype),
            act_q_lens: T.Tensor([batch], idx_dtype),
            seqused_kv: T.Tensor([batch], idx_dtype),
            sinks: T.Tensor([N1], accum_dtype),
            Output: T.Tensor(q_shape, dtype),
            LSE: T.Tensor([total_tokens, N1], accum_dtype),
            workspace_s: T.Tensor([core_num, 2, G, S2_BASE], accum_dtype),
            workspace_p: T.Tensor([core_num, 2, G, S2_BASE], dtype),
            workspace_o: T.Tensor([core_num, 2, G, D], accum_dtype),
            workspace_acc: T.Tensor([core_num, 2, G, D], accum_dtype),
            workspace_qk: T.Tensor([core_num, G, BI], accum_dtype),
        ):
            with T.Kernel(core_num, is_npu=True) as (cid, vid):
                q_l1 = T.alloc_L1([2, G, D2], dtype)

                kv_ring = T.alloc_L1([3, BI, D2], dtype)
                p_l1 = T.alloc_L1([G, S2_BASE], dtype)
                cL0 = T.alloc_L0C([2, G, BI], accum_dtype)
                p_l0a = T.alloc_L0A([2, G, BI], dtype)
                v_l0b = T.alloc_L0B([2, BI, BI], dtype)

                kv_iter = T.alloc_var("int32", init=0)
                cl0_iter = T.alloc_var("int32", init=0)
                ab_iter = T.alloc_var("int32", init=0)

                in_ub = T.alloc_ub([2, M_CHUNK, S2_BASE], accum_dtype)
                softmax_cmp = T.alloc_ub([M_CHUNK, S2_BASE], accum_dtype)
                out_ub = T.alloc_ub([M_CHUNK, S2_BASE], dtype)
                acc_pre = T.alloc_ub([M_CHUNK, D], accum_dtype)
                sink_ub = T.alloc_ub([2, M_CHUNK, 1], accum_dtype)
                lse_ub = T.alloc_ub([M_CHUNK, 1], accum_dtype)
                ones_ub = T.alloc_ub([M_CHUNK, 1], accum_dtype)
                brcb_d = T.alloc_ub([M_CHUNK, BLK], accum_dtype)

                sumP = T.alloc_ub([M_CHUNK, 1], accum_dtype)

                part = T.alloc_ub([M_CHUNK, 1], accum_dtype)
                brcb_s = T.alloc_ub([M_CHUNK, BLK], accum_dtype)

                m_i = T.alloc_ub([2, NMC, M_CHUNK, 1], accum_dtype)
                denom = T.alloc_ub([2, NMC, M_CHUNK, 1], accum_dtype)
                expmax = T.alloc_ub([2, NMC, M_CHUNK, 1], accum_dtype)

                with T.Scope("C"):
                    T.set_flag("m", "mte1", L0AB_EV0)
                    T.set_flag("m", "mte1", L0AB_EV1)
                    T.set_flag("mte1", "mte2", KV_EV0)
                    T.set_flag("mte1", "mte2", KV_EV1)
                    T.set_flag("mte1", "mte2", KV_EV2)
                    T.set_flag("mte1", "mte2", Q_EV)
                    T.set_flag("mte1", "mte2", P_EV)
                    for g in T.serial(GLOOP + 1):
                        if g < GLOOP:
                            buf = g % 2
                            task = g // MAX_TILES
                            tile = g % MAX_TILES
                            pid = task * core_num + cid
                            if pid < block_num:
                                b = T.cast(pid // max_seq, "int32")
                                s = T.cast(pid % max_seq, "int32")
                                act_q = act_q_lens[b]
                                if s < act_q:
                                    act_kv = seqused_kv[b]
                                    s_global = act_kv - act_q + s
                                    act_cmp = (s_global + 1) // cmp_ratio
                                    cmp_tiles = (act_cmp + S2_BASE - 1) // S2_BASE
                                    s2lt = 1 + cmp_tiles
                                    if tile < s2lt:
                                        is_ori = tile == 0
                                        ori_left = T.max(s_global - ori_win_left, 0)
                                        win = s_global + 1 - ori_left
                                        rel = tile - 1

                                        tw = tir.Select(
                                            is_ori,
                                            win,
                                            T.min(S2_BASE, act_cmp - rel * S2_BASE),
                                        )
                                        s2base = tir.Select(is_ori, ori_left, rel * S2_BASE)
                                        tok = q_prefix[b] + s

                                        T.wait_flag("mte1", "mte2", Q_EV)
                                        T.copy(Q[tok, :, 0:D2], q_l1[0, :, :])
                                        T.copy(Q[tok, :, D2:D], q_l1[1, :, :])
                                        T.set_flag("mte2", "mte1", Q_EV)
                                        T.wait_flag("mte2", "mte1", Q_EV)
                                        if DEBUG_SERIAL:
                                            T.barrier_all()

                                        for cb in range(MAX_COLBLK):
                                            if cb * BI < tw:
                                                ncols = T.min(BI, tw - cb * BI)
                                                ncols_a = (ncols + 15) // 16 * 16
                                                cs = cl0_iter % 2

                                                s0 = kv_iter % 3
                                                s1 = (kv_iter + 1) % 3
                                                ev0 = tir.Select(s0 < 2, KV_EV0 + s0, KV_EV2)
                                                ev1 = tir.Select(s1 < 2, KV_EV0 + s1, KV_EV2)
                                                for h in range(2):
                                                    slot = (kv_iter + h) % 3
                                                    ev = tir.Select(slot < 2, KV_EV0 + slot, KV_EV2)
                                                    T.wait_flag("mte1", "mte2", ev)
                                                    if is_ori:
                                                        pa_start = s2base + cb * BI
                                                        pa_cur = T.alloc_var("int32", init=pa_start)
                                                        for _pg in range((BI + ori_block_size - 1) // ori_block_size + 1):
                                                            pa_done = pa_cur - pa_start
                                                            if pa_done < ncols:
                                                                pa_lg = pa_cur // ori_block_size
                                                                pa_ph = ori_block_table[b, pa_lg]
                                                                pa_rem = pa_cur % ori_block_size
                                                                pa_run = T.min(
                                                                    ori_block_size - pa_rem,
                                                                    ncols - pa_done,
                                                                )
                                                                T.copy(
                                                                    ori_kv[
                                                                        pa_ph,
                                                                        pa_rem : pa_rem + pa_run,
                                                                        0,
                                                                        h * D2 : h * D2 + D2,
                                                                    ],
                                                                    kv_ring[
                                                                        slot,
                                                                        pa_done : pa_done + pa_run,
                                                                        :,
                                                                    ],
                                                                )
                                                                pa_cur += pa_run
                                                    else:
                                                        pc_start = s2base + cb * BI
                                                        pc_cur = T.alloc_var("int32", init=pc_start)
                                                        for _pg in range((BI + cmp_block_size - 1) // cmp_block_size + 1):
                                                            pc_done = pc_cur - pc_start
                                                            if pc_done < ncols:
                                                                pc_lg = pc_cur // cmp_block_size
                                                                pc_ph = cmp_block_table[b, pc_lg]
                                                                pc_rem = pc_cur % cmp_block_size
                                                                pc_run = T.min(
                                                                    cmp_block_size - pc_rem,
                                                                    ncols - pc_done,
                                                                )
                                                                T.copy(
                                                                    cmp_kv[
                                                                        pc_ph,
                                                                        pc_rem : pc_rem + pc_run,
                                                                        0,
                                                                        h * D2 : h * D2 + D2,
                                                                    ],
                                                                    kv_ring[
                                                                        slot,
                                                                        pc_done : pc_done + pc_run,
                                                                        :,
                                                                    ],
                                                                )
                                                                pc_cur += pc_run
                                                    T.set_flag("mte2", "mte1", ev)
                                                if DEBUG_SERIAL:
                                                    T.barrier_all()

                                                T.wait_flag("mte2", "mte1", ev0)
                                                if DEBUG_SERIAL:
                                                    T.gemm_v0(
                                                        q_l1[0, :, :],
                                                        kv_ring[s0, :, :],
                                                        cL0[cs, :, :],
                                                        transpose_B=True,
                                                        init=True,
                                                        n_actual=ncols_a,
                                                    )
                                                else:
                                                    T.wait_flag("m", "mte1", L0AB_EV0)
                                                    T.copy(
                                                        q_l1[0, :, 0:KL0],
                                                        p_l0a[0, :, :],
                                                    )
                                                    T.copy(
                                                        kv_ring[s0, :, 0:KL0],
                                                        v_l0b[0, :, :],
                                                        transpose=True,
                                                        real_k=KL0,
                                                        real_n=ncols_a,
                                                    )
                                                    T.set_flag("mte1", "m", L0AB_EV0)
                                                    T.wait_flag("mte1", "m", L0AB_EV0)
                                                    T.mma(
                                                        p_l0a[0, :, :],
                                                        v_l0b[0, :, :],
                                                        cL0[cs, :, :],
                                                        init=True,
                                                        k_actual=KL0,
                                                        n_actual=ncols_a,
                                                        unit_flag=0b10,
                                                    )

                                                    if ncols_a < 40:
                                                        T.pipe_barrier("m")
                                                    T.set_flag("m", "mte1", L0AB_EV0)
                                                    T.wait_flag("m", "mte1", L0AB_EV1)
                                                    T.copy(
                                                        q_l1[0, :, KL0 : 2 * KL0],
                                                        p_l0a[1, :, :],
                                                    )
                                                    T.copy(
                                                        kv_ring[s0, :, KL0 : 2 * KL0],
                                                        v_l0b[1, :, :],
                                                        transpose=True,
                                                        real_k=KL0,
                                                        real_n=ncols_a,
                                                    )
                                                    T.set_flag("mte1", "m", L0AB_EV1)
                                                    T.wait_flag("mte1", "m", L0AB_EV1)
                                                    T.mma(
                                                        p_l0a[1, :, :],
                                                        v_l0b[1, :, :],
                                                        cL0[cs, :, :],
                                                        init=False,
                                                        k_actual=KL0,
                                                        n_actual=ncols_a,
                                                        unit_flag=0b10,
                                                    )
                                                    if ncols_a < 40:
                                                        T.pipe_barrier("m")
                                                    T.set_flag("m", "mte1", L0AB_EV1)
                                                T.set_flag("mte1", "mte2", ev0)
                                                T.wait_flag("mte2", "mte1", ev1)
                                                if DEBUG_SERIAL:
                                                    T.gemm_v0(
                                                        q_l1[1, :, :],
                                                        kv_ring[s1, :, :],
                                                        cL0[cs, :, :],
                                                        transpose_B=True,
                                                        init=False,
                                                        n_actual=ncols_a,
                                                    )
                                                else:
                                                    T.wait_flag("m", "mte1", L0AB_EV0)
                                                    T.copy(
                                                        q_l1[1, :, 0:KL0],
                                                        p_l0a[0, :, :],
                                                    )
                                                    T.copy(
                                                        kv_ring[s1, :, 0:KL0],
                                                        v_l0b[0, :, :],
                                                        transpose=True,
                                                        real_k=KL0,
                                                        real_n=ncols_a,
                                                    )
                                                    T.set_flag("mte1", "m", L0AB_EV0)
                                                    T.wait_flag("mte1", "m", L0AB_EV0)
                                                    T.mma(
                                                        p_l0a[0, :, :],
                                                        v_l0b[0, :, :],
                                                        cL0[cs, :, :],
                                                        init=False,
                                                        k_actual=KL0,
                                                        n_actual=ncols_a,
                                                        unit_flag=0b10,
                                                    )
                                                    if ncols_a < 40:
                                                        T.pipe_barrier("m")
                                                    T.set_flag("m", "mte1", L0AB_EV0)
                                                    T.wait_flag("m", "mte1", L0AB_EV1)
                                                    T.copy(
                                                        q_l1[1, :, KL0 : 2 * KL0],
                                                        p_l0a[1, :, :],
                                                    )
                                                    T.copy(
                                                        kv_ring[s1, :, KL0 : 2 * KL0],
                                                        v_l0b[1, :, :],
                                                        transpose=True,
                                                        real_k=KL0,
                                                        real_n=ncols_a,
                                                    )
                                                    T.set_flag("mte1", "m", L0AB_EV1)
                                                    T.wait_flag("mte1", "m", L0AB_EV1)
                                                    T.mma(
                                                        p_l0a[1, :, :],
                                                        v_l0b[1, :, :],
                                                        cL0[cs, :, :],
                                                        init=False,
                                                        k_actual=KL0,
                                                        n_actual=ncols_a,
                                                        unit_flag=0b11,
                                                    )
                                                    if ncols_a < 40:
                                                        T.pipe_barrier("m")
                                                    T.set_flag("m", "mte1", L0AB_EV1)
                                                T.set_flag("mte1", "mte2", ev1)
                                                kv_iter += 2
                                                if DEBUG_SERIAL:
                                                    T.barrier_all()

                                                if DEBUG_SERIAL:
                                                    T.copy(
                                                        cL0[cs, :, :],
                                                        workspace_s[
                                                            cid,
                                                            buf,
                                                            :,
                                                            cb * BI : (cb + 1) * BI,
                                                        ],
                                                    )
                                                    T.barrier_all()
                                                else:
                                                    T.copy(
                                                        cL0[cs, :, 0:ncols_a],
                                                        workspace_s[
                                                            cid,
                                                            buf,
                                                            :,
                                                            cb * BI : cb * BI + ncols_a,
                                                        ],
                                                        unit_flag=0b11,
                                                    )
                                                cl0_iter += 1

                                        T.set_flag("mte1", "mte2", Q_EV)

                            T.set_cross_flag("FIX", EV_QK)

                        if g >= 1:
                            T.wait_cross_flag(EV_P)
                            gm = g - 1
                            bufm = gm % 2
                            taskm = gm // MAX_TILES
                            tilem = gm % MAX_TILES
                            pidm = taskm * core_num + cid
                            if pidm < block_num:
                                bm = T.cast(pidm // max_seq, "int32")
                                sm = T.cast(pidm % max_seq, "int32")
                                if sm < act_q_lens[bm]:
                                    act_kvm = seqused_kv[bm]
                                    s_globalm = act_kvm - act_q_lens[bm] + sm
                                    act_cmpm = (s_globalm + 1) // cmp_ratio
                                    cmp_tilesm = (act_cmpm + S2_BASE - 1) // S2_BASE
                                    s2ltm = 1 + cmp_tilesm
                                    if tilem < s2ltm:
                                        is_orim = tilem == 0
                                        ori_leftm = T.max(s_globalm - ori_win_left, 0)
                                        winm = s_globalm + 1 - ori_leftm
                                        relm = tilem - 1

                                        twm = tir.Select(
                                            is_orim,
                                            winm,
                                            T.min(S2_BASE, act_cmpm - relm * S2_BASE),
                                        )
                                        s2basem = tir.Select(is_orim, ori_leftm, relm * S2_BASE)

                                        T.wait_flag("mte1", "mte2", P_EV)
                                        T.copy(
                                            workspace_p[cid, bufm, :, 0:S2_BASE],
                                            p_l1[:, :],
                                        )
                                        T.set_flag("mte2", "mte1", P_EV)
                                        T.wait_flag("mte2", "mte1", P_EV)
                                        if DEBUG_SERIAL:
                                            T.barrier_all()

                                        for nl in range(PV_NT):
                                            cs = cl0_iter % 2
                                            for ks in range(MAX_COLBLK):
                                                if ks * BI < twm:
                                                    krows = T.min(BI, twm - ks * BI)
                                                    is_first_ks = ks == 0
                                                    is_last_ks = (ks + 1) * BI >= twm
                                                    slot = kv_iter % 3
                                                    ev = tir.Select(slot < 2, KV_EV0 + slot, KV_EV2)
                                                    pp = ab_iter % 2

                                                    T.wait_flag("mte1", "mte2", ev)
                                                    if is_orim:
                                                        pv_start = s2basem + ks * BI
                                                        pv_cur = T.alloc_var("int32", init=pv_start)
                                                        for _pg in range((BI + ori_block_size - 1) // ori_block_size + 1):
                                                            pv_done = pv_cur - pv_start
                                                            if pv_done < krows:
                                                                pv_lg = pv_cur // ori_block_size
                                                                pv_ph = ori_block_table[bm, pv_lg]
                                                                pv_rem = pv_cur % ori_block_size
                                                                pv_run = T.min(
                                                                    ori_block_size - pv_rem,
                                                                    krows - pv_done,
                                                                )
                                                                T.copy(
                                                                    ori_kv[
                                                                        pv_ph,
                                                                        pv_rem : pv_rem + pv_run,
                                                                        0,
                                                                        nl * PV_NW : nl * PV_NW + PV_NW,
                                                                    ],
                                                                    kv_ring[
                                                                        slot,
                                                                        pv_done : pv_done + pv_run,
                                                                        0:PV_NW,
                                                                    ],
                                                                )
                                                                pv_cur += pv_run
                                                    else:
                                                        qv_start = s2basem + ks * BI
                                                        qv_cur = T.alloc_var("int32", init=qv_start)
                                                        for _pg in range((BI + cmp_block_size - 1) // cmp_block_size + 1):
                                                            qv_done = qv_cur - qv_start
                                                            if qv_done < krows:
                                                                qv_lg = qv_cur // cmp_block_size
                                                                qv_ph = cmp_block_table[bm, qv_lg]
                                                                qv_rem = qv_cur % cmp_block_size
                                                                qv_run = T.min(
                                                                    cmp_block_size - qv_rem,
                                                                    krows - qv_done,
                                                                )
                                                                T.copy(
                                                                    cmp_kv[
                                                                        qv_ph,
                                                                        qv_rem : qv_rem + qv_run,
                                                                        0,
                                                                        nl * PV_NW : nl * PV_NW + PV_NW,
                                                                    ],
                                                                    kv_ring[
                                                                        slot,
                                                                        qv_done : qv_done + qv_run,
                                                                        0:PV_NW,
                                                                    ],
                                                                )
                                                                qv_cur += qv_run
                                                    T.set_flag("mte2", "mte1", ev)
                                                    if DEBUG_SERIAL:
                                                        T.barrier_all()

                                                    T.wait_flag("m", "mte1", L0AB_EV0 + pp)
                                                    T.wait_flag("mte2", "mte1", ev)
                                                    T.copy(
                                                        p_l1[:, ks * BI : (ks + 1) * BI],
                                                        p_l0a[pp, :, :],
                                                        real_k=krows,
                                                    )
                                                    T.copy(
                                                        kv_ring[slot, :, 0:PV_NW],
                                                        v_l0b[pp, :, :],
                                                        real_k=krows,
                                                    )
                                                    T.set_flag("mte1", "m", L0AB_EV0 + pp)
                                                    T.wait_flag("mte1", "m", L0AB_EV0 + pp)

                                                    uf = 0 if DEBUG_SERIAL else tir.Select(is_last_ks, 0b11, 0b10)
                                                    T.mma(
                                                        p_l0a[pp, :, :],
                                                        v_l0b[pp, :, :],
                                                        cL0[cs, :, :],
                                                        init=is_first_ks,
                                                        k_actual=krows,
                                                        unit_flag=uf,
                                                    )
                                                    T.set_flag("m", "mte1", L0AB_EV0 + pp)
                                                    T.set_flag("mte1", "mte2", ev)
                                                    kv_iter += 1
                                                    ab_iter += 1
                                                    if DEBUG_SERIAL:
                                                        T.barrier_all()

                                            if DEBUG_SERIAL:
                                                T.copy(
                                                    cL0[cs, :, :],
                                                    workspace_o[
                                                        cid,
                                                        bufm,
                                                        :,
                                                        nl * PV_NW : (nl + 1) * PV_NW,
                                                    ],
                                                )
                                                T.barrier_all()
                                            else:
                                                T.copy(
                                                    cL0[cs, :, :],
                                                    workspace_o[
                                                        cid,
                                                        bufm,
                                                        :,
                                                        nl * PV_NW : (nl + 1) * PV_NW,
                                                    ],
                                                    unit_flag=0b11,
                                                )
                                            cl0_iter += 1

                                        T.set_flag("mte1", "mte2", P_EV)

                            T.set_cross_flag("FIX", EV_PV)

                    T.wait_flag("m", "mte1", L0AB_EV0)
                    T.wait_flag("m", "mte1", L0AB_EV1)
                    T.wait_flag("mte1", "mte2", KV_EV0)
                    T.wait_flag("mte1", "mte2", KV_EV1)
                    T.wait_flag("mte1", "mte2", KV_EV2)
                    T.wait_flag("mte1", "mte2", Q_EV)
                    T.wait_flag("mte1", "mte2", P_EV)

                with T.Scope("V"):
                    T.tile.fill(ones_ub, T.float32(1.0))

                    T.set_flag("v", "mte2", IN_EV)
                    T.set_flag("v", "mte2", IN_EV + 1)
                    T.set_flag("v", "mte2", ACC_EV)
                    T.set_flag("mte3", "v", OUT_EV)
                    T.set_flag("mte3", "v", LSE_EV)
                    for g in T.serial(1, GLOOP + 2):
                        if g < GLOOP + 1:
                            T.wait_cross_flag(EV_QK)
                            gs = g - 1
                            buf = gs % 2
                            prev = (gs - 1) % 2
                            task = gs // MAX_TILES
                            tile = gs % MAX_TILES
                            pid = task * core_num + cid
                            if pid < block_num:
                                b = T.cast(pid // max_seq, "int32")
                                s = T.cast(pid % max_seq, "int32")
                                act_q = act_q_lens[b]
                                if s < act_q:
                                    act_kv = seqused_kv[b]
                                    s_global = act_kv - act_q + s
                                    act_cmp = (s_global + 1) // cmp_ratio
                                    cmp_tiles = (act_cmp + S2_BASE - 1) // S2_BASE
                                    s2lt = 1 + cmp_tiles
                                    if tile < s2lt:
                                        is_first = tile == 0
                                        ori_left = T.max(s_global - ori_win_left, 0)
                                        win = s_global + 1 - ori_left
                                        rel = tile - 1

                                        tw = tir.Select(
                                            is_first,
                                            win,
                                            T.min(S2_BASE, act_cmp - rel * S2_BASE),
                                        )
                                        tw_a = (tw + 15) // 16 * 16

                                        tw_64 = (tw + 63) // 64 * 64

                                        is_narrow = tw <= BI
                                        for mc in range(NMC):
                                            r0 = vid * G2 + mc * M_CHUNK
                                            ps = mc & 1

                                            T.wait_flag("v", "mte2", IN_EV + ps)

                                            T.tile.fill(
                                                in_ub[ps, :, :],
                                                -T.infinity(accum_dtype),
                                            )
                                            T.set_flag("v", "mte2", IN_EV + ps)
                                            T.wait_flag("v", "mte2", IN_EV + ps)

                                            T.copy(
                                                workspace_s[
                                                    cid,
                                                    buf,
                                                    r0 : r0 + M_CHUNK,
                                                    0:tw,
                                                ],
                                                in_ub[ps, :, 0:tw],
                                            )
                                            T.copy(
                                                sinks[r0 : r0 + M_CHUNK],
                                                sink_ub[ps, :, :],
                                            )

                                            T.set_flag("mte2", "v", IN_EV + ps)
                                            T.wait_flag("mte2", "v", IN_EV + ps)
                                            T.tile.mul(
                                                in_ub[ps, :, :],
                                                in_ub[ps, :, :],
                                                softmax_scale,
                                            )

                                            T.pipe_barrier("v")

                                            if is_narrow:
                                                T.tile.fill(
                                                    m_i[buf, mc, :, :],
                                                    -T.infinity(accum_dtype),
                                                )
                                                for kc in range(BI // 64):
                                                    if kc * 64 < tw_64:
                                                        T.reduce_max(
                                                            in_ub[
                                                                ps,
                                                                :,
                                                                kc * 64 : kc * 64 + 64,
                                                            ],
                                                            part,
                                                            dim=-1,
                                                        )
                                                        T.tile.max(
                                                            m_i[buf, mc, :, :],
                                                            m_i[buf, mc, :, :],
                                                            part,
                                                        )
                                            else:
                                                T.reduce_max(
                                                    in_ub[ps, :, :],
                                                    m_i[buf, mc, :, :],
                                                    dim=-1,
                                                )
                                            if is_first:
                                                T.tile.max(
                                                    m_i[buf, mc, :, :],
                                                    sink_ub[ps, :, :],
                                                    m_i[buf, mc, :, :],
                                                )
                                                T.tile.sub(
                                                    expmax[buf, mc, :, :],
                                                    sink_ub[ps, :, :],
                                                    m_i[buf, mc, :, :],
                                                )
                                            else:
                                                T.tile.max(
                                                    m_i[buf, mc, :, :],
                                                    m_i[prev, mc, :, :],
                                                    m_i[buf, mc, :, :],
                                                )
                                                T.tile.sub(
                                                    expmax[buf, mc, :, :],
                                                    m_i[prev, mc, :, :],
                                                    m_i[buf, mc, :, :],
                                                )
                                            T.tile.exp(
                                                expmax[buf, mc, :, :],
                                                expmax[buf, mc, :, :],
                                            )

                                            if is_narrow:
                                                T.tile.brcb_experiment(
                                                    brcb_s,
                                                    m_i[buf, mc, :, :],
                                                    M_CHUNK // 8,
                                                    1,
                                                    8,
                                                )
                                                for kc in range(BI // 64):
                                                    if kc * 64 < tw_64:
                                                        T.tile.row_expand_sub_experiment(
                                                            in_ub[
                                                                ps,
                                                                :,
                                                                kc * 64 : kc * 64 + 64,
                                                            ],
                                                            in_ub[
                                                                ps,
                                                                :,
                                                                kc * 64 : kc * 64 + 64,
                                                            ],
                                                            brcb_s,
                                                        )
                                                for kc in range(BI // 64):
                                                    if kc * 64 < tw_64:
                                                        T.tile.exp_experiment(
                                                            in_ub[
                                                                ps,
                                                                :,
                                                                kc * 64 : kc * 64 + 64,
                                                            ],
                                                            in_ub[
                                                                ps,
                                                                :,
                                                                kc * 64 : kc * 64 + 64,
                                                            ],
                                                        )
                                            else:
                                                T.tile.broadcast(softmax_cmp, m_i[buf, mc, :, :])
                                                T.tile.sub(
                                                    in_ub[ps, :, :],
                                                    in_ub[ps, :, :],
                                                    softmax_cmp,
                                                )
                                                T.tile.exp(in_ub[ps, :, :], in_ub[ps, :, :])
                                            if is_first:
                                                T.tile.mul(
                                                    denom[buf, mc, :, :],
                                                    ones_ub,
                                                    expmax[buf, mc, :, :],
                                                )
                                            else:
                                                T.tile.mul(
                                                    denom[buf, mc, :, :],
                                                    denom[prev, mc, :, :],
                                                    expmax[buf, mc, :, :],
                                                )

                                            T.pipe_barrier("v")

                                            T.wait_flag("mte3", "v", OUT_EV)
                                            T.tile.cast(
                                                out_ub,
                                                in_ub[ps, :, :],
                                                "CAST_ROUND",
                                                M_CHUNK * S2_BASE,
                                            )

                                            T.pipe_barrier("v")

                                            if is_narrow:
                                                T.tile.fill(sumP, T.float32(0.0))
                                                for kc in range(BI // 64):
                                                    if kc * 64 < tw_64:
                                                        T.reduce_sum(
                                                            in_ub[
                                                                ps,
                                                                :,
                                                                kc * 64 : kc * 64 + 64,
                                                            ],
                                                            part,
                                                            dim=-1,
                                                        )
                                                        T.tile.add(sumP, sumP, part)
                                            else:
                                                T.reduce_sum(in_ub[ps, :, :], sumP, dim=-1)
                                            T.tile.add(
                                                denom[buf, mc, :, :],
                                                denom[buf, mc, :, :],
                                                sumP,
                                            )

                                            T.set_flag("v", "mte2", IN_EV + ps)
                                            T.set_flag("v", "mte3", OUT_EV)
                                            T.wait_flag("v", "mte3", OUT_EV)

                                            T.copy(
                                                out_ub[:, 0:tw_a],
                                                workspace_p[
                                                    cid,
                                                    buf,
                                                    r0 : r0 + M_CHUNK,
                                                    0:tw_a,
                                                ],
                                            )

                                            T.set_flag("mte3", "v", OUT_EV)
                            T.set_cross_flag("MTE3", EV_P)

                        if g >= 2:
                            T.wait_cross_flag(EV_PV)
                            go = g - 2
                            bufo = go % 2
                            prevo = (go - 1) % 2
                            task = go // MAX_TILES
                            tile = go % MAX_TILES
                            pid = task * core_num + cid
                            if pid < block_num:
                                b = T.cast(pid // max_seq, "int32")
                                s = T.cast(pid % max_seq, "int32")
                                act_q = act_q_lens[b]
                                if s < act_q:
                                    act_kv = seqused_kv[b]
                                    s_global = act_kv - act_q + s

                                    act_cmp = tir.Select(
                                        s_global < cmp_ratio - 1,
                                        0,
                                        (s_global + 1) // cmp_ratio,
                                    )
                                    cmp_tiles = (act_cmp + S2_BASE - 1) // S2_BASE
                                    s2lt = 1 + cmp_tiles
                                    if tile < s2lt:
                                        not_first = tile != 0
                                        is_last = tile == s2lt - 1
                                        tok = q_prefix[b] + s
                                        for mc in range(NMC):
                                            r0 = vid * G2 + mc * M_CHUNK
                                            po = mc & 1

                                            T.wait_flag("v", "mte2", IN_EV + po)
                                            T.copy(
                                                workspace_o[cid, bufo, r0 : r0 + M_CHUNK, :],
                                                in_ub[po, :, :],
                                            )

                                            T.set_flag("mte2", "v", IN_EV + po)
                                            T.wait_flag("mte2", "v", IN_EV + po)
                                            if not_first:
                                                T.set_flag("mte3", "mte2", FENCE)
                                                T.wait_flag("mte3", "mte2", FENCE)

                                                T.wait_flag("v", "mte2", ACC_EV)
                                                T.copy(
                                                    workspace_acc[cid, prevo, r0 : r0 + M_CHUNK, :],
                                                    acc_pre,
                                                )

                                                T.set_flag("mte2", "v", ACC_EV)
                                                T.wait_flag("mte2", "v", ACC_EV)

                                                T.tile.brcb_experiment(
                                                    brcb_d,
                                                    expmax[bufo, mc, :, :],
                                                    M_CHUNK // 8,
                                                    1,
                                                    8,
                                                )
                                                T.pipe_barrier("v")
                                                for dcol in T.serial(D // (8 * BLK)):
                                                    cb = dcol * (8 * BLK)
                                                    T.tile.row_expand_mul_experiment(
                                                        acc_pre[:, cb : cb + 8 * BLK],
                                                        acc_pre[:, cb : cb + 8 * BLK],
                                                        brcb_d,
                                                    )
                                                T.pipe_barrier("v")
                                                T.tile.add(
                                                    in_ub[po, :, :],
                                                    in_ub[po, :, :],
                                                    acc_pre,
                                                )
                                                T.pipe_barrier("v")

                                                T.set_flag("v", "mte2", ACC_EV)
                                            if is_last:
                                                T.tile.brcb_experiment(
                                                    brcb_d,
                                                    denom[bufo, mc, :, :],
                                                    M_CHUNK // 8,
                                                    1,
                                                    8,
                                                )
                                                T.pipe_barrier("v")
                                                for dcol in T.serial(D // (8 * BLK)):
                                                    cb = dcol * (8 * BLK)
                                                    T.tile.row_expand_div_experiment(
                                                        in_ub[po, :, cb : cb + 8 * BLK],
                                                        in_ub[po, :, cb : cb + 8 * BLK],
                                                        brcb_d,
                                                    )
                                                T.pipe_barrier("v")

                                                T.wait_flag("mte3", "v", OUT_EV)
                                                T.tile.cast(
                                                    out_ub,
                                                    in_ub[po, :, :],
                                                    out_cast_mode,
                                                    M_CHUNK * D,
                                                )

                                                T.set_flag("v", "mte2", IN_EV + po)
                                                T.set_flag("v", "mte3", OUT_EV)
                                                T.wait_flag("v", "mte3", OUT_EV)
                                                T.copy(
                                                    out_ub,
                                                    Output[tok, r0 : r0 + M_CHUNK, :],
                                                )

                                                T.set_flag("mte3", "v", OUT_EV)

                                                T.wait_flag("mte3", "v", LSE_EV)
                                                T.tile.ln(
                                                    lse_ub,
                                                    denom[bufo, mc, :, :],
                                                )
                                                T.tile.add(
                                                    lse_ub,
                                                    lse_ub,
                                                    m_i[bufo, mc, :, :],
                                                )
                                                T.set_flag("v", "mte3", LSE_EV)
                                                T.wait_flag("v", "mte3", LSE_EV)
                                                T.copy(lse_ub, LSE[tok, r0 : r0 + M_CHUNK])
                                                T.set_flag("mte3", "v", LSE_EV)
                                            else:
                                                T.set_flag("v", "mte3", IN_EV + po)
                                                T.wait_flag("v", "mte3", IN_EV + po)
                                                T.copy(
                                                    in_ub[po, :, :],
                                                    workspace_acc[cid, bufo, r0 : r0 + M_CHUNK, :],
                                                )
                                                T.set_flag("mte3", "v", IN_EV + po)
                                                T.wait_flag("mte3", "v", IN_EV + po)
                                                T.set_flag("v", "mte2", IN_EV + po)

                            T.set_cross_flag("MTE3", EV_PV)

                    T.wait_flag("v", "mte2", IN_EV)
                    T.wait_flag("v", "mte2", IN_EV + 1)
                    T.wait_flag("v", "mte2", ACC_EV)
                    T.wait_flag("mte3", "v", OUT_EV)
                    T.wait_flag("mte3", "v", LSE_EV)

        return sparse_flash_mla_hca

    func = kernel()
    if os.environ.get("SAS_DUMP_SRC"):
        try:
            src = func.get_kernel_source()
            with open("/tmp/hca_gen.cpp", "w") as fh:
                fh.write(src)
            print(f"[SAS_DUMP_SRC] wrote {len(src)} chars to /tmp/hca_gen.cpp")
        except Exception as exc:
            print(f"[SAS_DUMP_SRC] get_kernel_source failed: {exc!r}")
    return func


def _build_csa(
    *,
    batch,
    max_seq,
    total_tokens,
    ori_block_num,
    ori_block_size,
    ori_table_len,
    cmp_block_num,
    cmp_block_size,
    cmp_table_len,
    n_heads,
    head_dim,
    topk_cmp,
    ori_win_left,
    cmp_ratio,
    softmax_scale,
    dtype,
    core_num,
):
    N1 = n_heads
    N2 = 1
    D = head_dim
    D2 = D // 2
    G = N1 // N2
    BI = DEFAULT_BLOCK_I
    PV_NT = D // BI
    PV_NW = BI

    KL0 = 128
    VEC_NUM = 2
    G2 = G // VEC_NUM
    BLK = 8
    S2_BASE = 512
    MAX_COLBLK = S2_BASE // BI

    M_CHUNK = 16
    NMC = G2 // M_CHUNK

    MERGE_ROWS = 6

    accum_dtype = "float"
    idx_dtype = "int32"
    out_cast_mode = "CAST_RINT" if dtype == "bfloat16" else "CAST_ROUND"

    max_cmp_len = min((max_seq + cmp_ratio - 1) // cmp_ratio, topk_cmp)
    MAX_CMP_TILES = (max_cmp_len + S2_BASE - 1) // S2_BASE
    MAX_TILES = 1 + MAX_CMP_TILES

    block_num = batch * max_seq
    n_iter = (block_num + core_num - 1) // core_num

    GLOOP = n_iter * MAX_TILES

    EV_QK = 0
    EV_P = 1
    EV_PV = 2

    EV_V0 = 3

    EV_CREDIT = 4

    DEBUG_SERIAL = False

    KV_EV0 = 2
    KV_EV1 = 3
    KV_EV2 = 6

    Q_EV = 1
    P_EV = 7

    L0AB_EV0 = 4
    L0AB_EV1 = 5

    IN_EV = 2
    ACC_EV = 4
    OUT_EV = 5
    LSE_EV = 0

    FENCE = 0

    MRG_EV = 6

    KVMERGE_RING = 4

    q_shape = [total_tokens, N1, D]
    ori_kv_shape = [ori_block_num, ori_block_size, N2, D]
    cmp_kv_shape = [cmp_block_num, cmp_block_size, N2, D]

    cmp_idx_shape = [total_tokens, N2, topk_cmp]

    @tilelang.jit(out_idx=[10, 11], workspace_idx=[12, 13, 14, 15, 16, 17], target="ascendc")
    def kernel():
        @T.prim_func
        def sparse_flash_mla_csa(
            Q: T.Tensor(q_shape, dtype),
            ori_kv: T.Tensor(ori_kv_shape, dtype),
            ori_block_table: T.Tensor([batch, ori_table_len], idx_dtype),
            cmp_kv: T.Tensor(cmp_kv_shape, dtype),
            cmp_block_table: T.Tensor([batch, cmp_table_len], idx_dtype),
            cmp_indices: T.Tensor(cmp_idx_shape, idx_dtype),
            q_prefix: T.Tensor([batch], idx_dtype),
            act_q_lens: T.Tensor([batch], idx_dtype),
            seqused_kv: T.Tensor([batch], idx_dtype),
            sinks: T.Tensor([N1], accum_dtype),
            Output: T.Tensor(q_shape, dtype),
            LSE: T.Tensor([total_tokens, N1], accum_dtype),
            workspace_s: T.Tensor([core_num, 2, G, S2_BASE], accum_dtype),
            workspace_p: T.Tensor([core_num, 2, G, S2_BASE], dtype),
            workspace_o: T.Tensor([core_num, 2, G, D], accum_dtype),
            workspace_acc: T.Tensor([core_num, 2, G, D], accum_dtype),
            workspace_qk: T.Tensor([core_num, G, BI], accum_dtype),
            kvMergeGm: T.Tensor([core_num, KVMERGE_RING, S2_BASE, D], dtype),
        ):
            with T.Kernel(core_num, is_npu=True) as (cid, vid):
                q_l1 = T.alloc_L1([2, G, D2], dtype)

                kv_ring = T.alloc_L1([3, BI, D2], dtype)
                p_l1 = T.alloc_L1([G, S2_BASE], dtype)
                cL0 = T.alloc_L0C([2, G, BI], accum_dtype)
                p_l0a = T.alloc_L0A([2, G, BI], dtype)
                v_l0b = T.alloc_L0B([2, BI, BI], dtype)

                kv_iter = T.alloc_var("int32", init=0)
                cl0_iter = T.alloc_var("int32", init=0)
                ab_iter = T.alloc_var("int32", init=0)

                in_ub = T.alloc_ub([2, M_CHUNK, S2_BASE], accum_dtype)
                softmax_cmp = T.alloc_ub([M_CHUNK, S2_BASE], accum_dtype)
                out_ub = T.alloc_ub([M_CHUNK, S2_BASE], dtype)
                acc_pre = T.alloc_ub([M_CHUNK, D], accum_dtype)
                sink_ub = T.alloc_ub([2, M_CHUNK, 1], accum_dtype)
                lse_ub = T.alloc_ub([M_CHUNK, 1], accum_dtype)
                ones_ub = T.alloc_ub([M_CHUNK, 1], accum_dtype)
                brcb_d = T.alloc_ub([M_CHUNK, BLK], accum_dtype)

                sumP = T.alloc_ub([M_CHUNK, 1], accum_dtype)

                part = T.alloc_ub([M_CHUNK, 1], accum_dtype)
                brcb_s = T.alloc_ub([M_CHUNK, BLK], accum_dtype)

                merge_ub = T.alloc_ub([2, MERGE_ROWS, D], dtype)

                m_i = T.alloc_ub([2, NMC, M_CHUNK, 1], accum_dtype)
                denom = T.alloc_ub([2, NMC, M_CHUNK, 1], accum_dtype)
                expmax = T.alloc_ub([2, NMC, M_CHUNK, 1], accum_dtype)

                with T.Scope("C"):
                    T.set_flag("m", "mte1", L0AB_EV0)
                    T.set_flag("m", "mte1", L0AB_EV1)
                    T.set_flag("mte1", "mte2", KV_EV0)
                    T.set_flag("mte1", "mte2", KV_EV1)
                    T.set_flag("mte1", "mte2", KV_EV2)
                    T.set_flag("mte1", "mte2", Q_EV)
                    T.set_flag("mte1", "mte2", P_EV)

                    for _cr in range(KVMERGE_RING):
                        T.set_cross_flag("MTE2", EV_CREDIT)
                    for g in T.serial(GLOOP + 1):
                        if g < GLOOP:
                            buf = g % 2
                            task = g // MAX_TILES
                            tile = g % MAX_TILES
                            pid = task * core_num + cid
                            if pid < block_num:
                                b = T.cast(pid // max_seq, "int32")
                                s = T.cast(pid % max_seq, "int32")
                                act_q = act_q_lens[b]
                                if s < act_q:
                                    act_kv = seqused_kv[b]
                                    s_global = act_kv - act_q + s
                                    act_cmp = (s_global + 1) // cmp_ratio
                                    cmp_tiles = (T.min(act_cmp, topk_cmp) + S2_BASE - 1) // S2_BASE
                                    s2lt = 1 + cmp_tiles
                                    if tile < s2lt:
                                        is_ori = tile == 0
                                        ori_left = T.max(s_global - ori_win_left, 0)
                                        win = s_global + 1 - ori_left
                                        rel = tile - 1

                                        tw = tir.Select(
                                            is_ori,
                                            win,
                                            T.min(
                                                S2_BASE,
                                                T.min(act_cmp, topk_cmp) - rel * S2_BASE,
                                            ),
                                        )
                                        s2base = tir.Select(is_ori, ori_left, rel * S2_BASE)
                                        tok = q_prefix[b] + s

                                        T.wait_flag("mte1", "mte2", Q_EV)
                                        T.copy(Q[tok, :, 0:D2], q_l1[0, :, :])
                                        T.copy(Q[tok, :, D2:D], q_l1[1, :, :])
                                        T.set_flag("mte2", "mte1", Q_EV)
                                        T.wait_flag("mte2", "mte1", Q_EV)
                                        if DEBUG_SERIAL:
                                            T.barrier_all()

                                        if tile != 0:
                                            T.wait_cross_flag(EV_V0)
                                        for cb in range(MAX_COLBLK):
                                            if cb * BI < tw:
                                                ncols = T.min(BI, tw - cb * BI)
                                                ncols_a = (ncols + 15) // 16 * 16
                                                cs = cl0_iter % 2

                                                s0 = kv_iter % 3
                                                s1 = (kv_iter + 1) % 3
                                                ev0 = tir.Select(s0 < 2, KV_EV0 + s0, KV_EV2)
                                                ev1 = tir.Select(s1 < 2, KV_EV0 + s1, KV_EV2)
                                                for h in range(2):
                                                    slot = (kv_iter + h) % 3
                                                    ev = tir.Select(slot < 2, KV_EV0 + slot, KV_EV2)
                                                    T.wait_flag("mte1", "mte2", ev)
                                                    if is_ori:
                                                        pa_start = s2base + cb * BI
                                                        pa_cur = T.alloc_var("int32", init=pa_start)
                                                        for _pg in range((BI + ori_block_size - 1) // ori_block_size + 1):
                                                            pa_done = pa_cur - pa_start
                                                            if pa_done < ncols:
                                                                pa_lg = pa_cur // ori_block_size
                                                                pa_ph = ori_block_table[b, pa_lg]
                                                                pa_rem = pa_cur % ori_block_size
                                                                pa_run = T.min(
                                                                    ori_block_size - pa_rem,
                                                                    ncols - pa_done,
                                                                )
                                                                T.copy(
                                                                    ori_kv[
                                                                        pa_ph,
                                                                        pa_rem : pa_rem + pa_run,
                                                                        0,
                                                                        h * D2 : h * D2 + D2,
                                                                    ],
                                                                    kv_ring[
                                                                        slot,
                                                                        pa_done : pa_done + pa_run,
                                                                        :,
                                                                    ],
                                                                )
                                                                pa_cur += pa_run
                                                    else:
                                                        T.copy(
                                                            kvMergeGm[
                                                                cid,
                                                                task % KVMERGE_RING,
                                                                cb * BI : cb * BI + ncols,
                                                                h * D2 : h * D2 + D2,
                                                            ],
                                                            kv_ring[slot, 0:ncols, :],
                                                        )
                                                    T.set_flag("mte2", "mte1", ev)
                                                if DEBUG_SERIAL:
                                                    T.barrier_all()

                                                T.wait_flag("mte2", "mte1", ev0)
                                                if DEBUG_SERIAL:
                                                    T.gemm_v0(
                                                        q_l1[0, :, :],
                                                        kv_ring[s0, :, :],
                                                        cL0[cs, :, :],
                                                        transpose_B=True,
                                                        init=True,
                                                        n_actual=ncols_a,
                                                    )
                                                else:
                                                    T.wait_flag("m", "mte1", L0AB_EV0)
                                                    T.copy(
                                                        q_l1[0, :, 0:KL0],
                                                        p_l0a[0, :, :],
                                                    )
                                                    T.copy(
                                                        kv_ring[s0, :, 0:KL0],
                                                        v_l0b[0, :, :],
                                                        transpose=True,
                                                        real_k=KL0,
                                                        real_n=ncols_a,
                                                    )
                                                    T.set_flag("mte1", "m", L0AB_EV0)
                                                    T.wait_flag("mte1", "m", L0AB_EV0)
                                                    T.mma(
                                                        p_l0a[0, :, :],
                                                        v_l0b[0, :, :],
                                                        cL0[cs, :, :],
                                                        init=True,
                                                        k_actual=KL0,
                                                        n_actual=ncols_a,
                                                        unit_flag=0b10,
                                                    )

                                                    if ncols_a < 40:
                                                        T.pipe_barrier("m")
                                                    T.set_flag("m", "mte1", L0AB_EV0)
                                                    T.wait_flag("m", "mte1", L0AB_EV1)
                                                    T.copy(
                                                        q_l1[0, :, KL0 : 2 * KL0],
                                                        p_l0a[1, :, :],
                                                    )
                                                    T.copy(
                                                        kv_ring[s0, :, KL0 : 2 * KL0],
                                                        v_l0b[1, :, :],
                                                        transpose=True,
                                                        real_k=KL0,
                                                        real_n=ncols_a,
                                                    )
                                                    T.set_flag("mte1", "m", L0AB_EV1)
                                                    T.wait_flag("mte1", "m", L0AB_EV1)
                                                    T.mma(
                                                        p_l0a[1, :, :],
                                                        v_l0b[1, :, :],
                                                        cL0[cs, :, :],
                                                        init=False,
                                                        k_actual=KL0,
                                                        n_actual=ncols_a,
                                                        unit_flag=0b10,
                                                    )
                                                    if ncols_a < 40:
                                                        T.pipe_barrier("m")
                                                    T.set_flag("m", "mte1", L0AB_EV1)
                                                T.set_flag("mte1", "mte2", ev0)
                                                T.wait_flag("mte2", "mte1", ev1)
                                                if DEBUG_SERIAL:
                                                    T.gemm_v0(
                                                        q_l1[1, :, :],
                                                        kv_ring[s1, :, :],
                                                        cL0[cs, :, :],
                                                        transpose_B=True,
                                                        init=False,
                                                        n_actual=ncols_a,
                                                    )
                                                else:
                                                    T.wait_flag("m", "mte1", L0AB_EV0)
                                                    T.copy(
                                                        q_l1[1, :, 0:KL0],
                                                        p_l0a[0, :, :],
                                                    )
                                                    T.copy(
                                                        kv_ring[s1, :, 0:KL0],
                                                        v_l0b[0, :, :],
                                                        transpose=True,
                                                        real_k=KL0,
                                                        real_n=ncols_a,
                                                    )
                                                    T.set_flag("mte1", "m", L0AB_EV0)
                                                    T.wait_flag("mte1", "m", L0AB_EV0)
                                                    T.mma(
                                                        p_l0a[0, :, :],
                                                        v_l0b[0, :, :],
                                                        cL0[cs, :, :],
                                                        init=False,
                                                        k_actual=KL0,
                                                        n_actual=ncols_a,
                                                        unit_flag=0b10,
                                                    )
                                                    if ncols_a < 40:
                                                        T.pipe_barrier("m")
                                                    T.set_flag("m", "mte1", L0AB_EV0)
                                                    T.wait_flag("m", "mte1", L0AB_EV1)
                                                    T.copy(
                                                        q_l1[1, :, KL0 : 2 * KL0],
                                                        p_l0a[1, :, :],
                                                    )
                                                    T.copy(
                                                        kv_ring[s1, :, KL0 : 2 * KL0],
                                                        v_l0b[1, :, :],
                                                        transpose=True,
                                                        real_k=KL0,
                                                        real_n=ncols_a,
                                                    )
                                                    T.set_flag("mte1", "m", L0AB_EV1)
                                                    T.wait_flag("mte1", "m", L0AB_EV1)
                                                    T.mma(
                                                        p_l0a[1, :, :],
                                                        v_l0b[1, :, :],
                                                        cL0[cs, :, :],
                                                        init=False,
                                                        k_actual=KL0,
                                                        n_actual=ncols_a,
                                                        unit_flag=0b11,
                                                    )
                                                    if ncols_a < 40:
                                                        T.pipe_barrier("m")
                                                    T.set_flag("m", "mte1", L0AB_EV1)
                                                T.set_flag("mte1", "mte2", ev1)
                                                kv_iter += 2
                                                if DEBUG_SERIAL:
                                                    T.barrier_all()

                                                if DEBUG_SERIAL:
                                                    T.copy(
                                                        cL0[cs, :, :],
                                                        workspace_s[
                                                            cid,
                                                            buf,
                                                            :,
                                                            cb * BI : (cb + 1) * BI,
                                                        ],
                                                    )
                                                    T.barrier_all()
                                                else:
                                                    T.copy(
                                                        cL0[cs, :, 0:ncols_a],
                                                        workspace_s[
                                                            cid,
                                                            buf,
                                                            :,
                                                            cb * BI : cb * BI + ncols_a,
                                                        ],
                                                        unit_flag=0b11,
                                                    )
                                                cl0_iter += 1

                                        T.set_flag("mte1", "mte2", Q_EV)

                            T.set_cross_flag("FIX", EV_QK)

                        if g >= 1:
                            T.wait_cross_flag(EV_P)
                            gm = g - 1
                            bufm = gm % 2
                            taskm = gm // MAX_TILES
                            tilem = gm % MAX_TILES
                            pidm = taskm * core_num + cid
                            if pidm < block_num:
                                bm = T.cast(pidm // max_seq, "int32")
                                sm = T.cast(pidm % max_seq, "int32")
                                if sm < act_q_lens[bm]:
                                    act_kvm = seqused_kv[bm]
                                    s_globalm = act_kvm - act_q_lens[bm] + sm
                                    act_cmpm = (s_globalm + 1) // cmp_ratio
                                    cmp_tilesm = (T.min(act_cmpm, topk_cmp) + S2_BASE - 1) // S2_BASE
                                    s2ltm = 1 + cmp_tilesm
                                    if tilem < s2ltm:
                                        is_orim = tilem == 0
                                        ori_leftm = T.max(s_globalm - ori_win_left, 0)
                                        winm = s_globalm + 1 - ori_leftm
                                        relm = tilem - 1

                                        twm = tir.Select(
                                            is_orim,
                                            winm,
                                            T.min(
                                                S2_BASE,
                                                T.min(act_cmpm, topk_cmp) - relm * S2_BASE,
                                            ),
                                        )
                                        s2basem = tir.Select(is_orim, ori_leftm, relm * S2_BASE)

                                        T.wait_flag("mte1", "mte2", P_EV)
                                        T.copy(
                                            workspace_p[cid, bufm, :, 0:S2_BASE],
                                            p_l1[:, :],
                                        )
                                        T.set_flag("mte2", "mte1", P_EV)
                                        T.wait_flag("mte2", "mte1", P_EV)
                                        if DEBUG_SERIAL:
                                            T.barrier_all()

                                        for nl in range(PV_NT):
                                            cs = cl0_iter % 2
                                            for ks in range(MAX_COLBLK):
                                                if ks * BI < twm:
                                                    krows = T.min(BI, twm - ks * BI)
                                                    is_first_ks = ks == 0
                                                    is_last_ks = (ks + 1) * BI >= twm
                                                    slot = kv_iter % 3
                                                    ev = tir.Select(slot < 2, KV_EV0 + slot, KV_EV2)
                                                    pp = ab_iter % 2

                                                    T.wait_flag("mte1", "mte2", ev)
                                                    if is_orim:
                                                        pv_start = s2basem + ks * BI
                                                        pv_cur = T.alloc_var("int32", init=pv_start)
                                                        for _pg in range((BI + ori_block_size - 1) // ori_block_size + 1):
                                                            pv_done = pv_cur - pv_start
                                                            if pv_done < krows:
                                                                pv_lg = pv_cur // ori_block_size
                                                                pv_ph = ori_block_table[bm, pv_lg]
                                                                pv_rem = pv_cur % ori_block_size
                                                                pv_run = T.min(
                                                                    ori_block_size - pv_rem,
                                                                    krows - pv_done,
                                                                )
                                                                T.copy(
                                                                    ori_kv[
                                                                        pv_ph,
                                                                        pv_rem : pv_rem + pv_run,
                                                                        0,
                                                                        nl * PV_NW : nl * PV_NW + PV_NW,
                                                                    ],
                                                                    kv_ring[
                                                                        slot,
                                                                        pv_done : pv_done + pv_run,
                                                                        0:PV_NW,
                                                                    ],
                                                                )
                                                                pv_cur += pv_run
                                                    else:
                                                        T.copy(
                                                            kvMergeGm[
                                                                cid,
                                                                taskm % KVMERGE_RING,
                                                                ks * BI : ks * BI + krows,
                                                                nl * PV_NW : nl * PV_NW + PV_NW,
                                                            ],
                                                            kv_ring[slot, 0:krows, 0:PV_NW],
                                                        )
                                                    T.set_flag("mte2", "mte1", ev)
                                                    if DEBUG_SERIAL:
                                                        T.barrier_all()

                                                    T.wait_flag("m", "mte1", L0AB_EV0 + pp)
                                                    T.wait_flag("mte2", "mte1", ev)
                                                    T.copy(
                                                        p_l1[:, ks * BI : (ks + 1) * BI],
                                                        p_l0a[pp, :, :],
                                                        real_k=krows,
                                                    )
                                                    T.copy(
                                                        kv_ring[slot, :, 0:PV_NW],
                                                        v_l0b[pp, :, :],
                                                        real_k=krows,
                                                    )
                                                    T.set_flag("mte1", "m", L0AB_EV0 + pp)
                                                    T.wait_flag("mte1", "m", L0AB_EV0 + pp)

                                                    uf = 0 if DEBUG_SERIAL else tir.Select(is_last_ks, 0b11, 0b10)
                                                    T.mma(
                                                        p_l0a[pp, :, :],
                                                        v_l0b[pp, :, :],
                                                        cL0[cs, :, :],
                                                        init=is_first_ks,
                                                        k_actual=krows,
                                                        unit_flag=uf,
                                                    )
                                                    T.set_flag("m", "mte1", L0AB_EV0 + pp)
                                                    T.set_flag("mte1", "mte2", ev)
                                                    kv_iter += 1
                                                    ab_iter += 1
                                                    if DEBUG_SERIAL:
                                                        T.barrier_all()

                                            if DEBUG_SERIAL:
                                                T.copy(
                                                    cL0[cs, :, :],
                                                    workspace_o[
                                                        cid,
                                                        bufm,
                                                        :,
                                                        nl * PV_NW : (nl + 1) * PV_NW,
                                                    ],
                                                )
                                                T.barrier_all()
                                            else:
                                                T.copy(
                                                    cL0[cs, :, :],
                                                    workspace_o[
                                                        cid,
                                                        bufm,
                                                        :,
                                                        nl * PV_NW : (nl + 1) * PV_NW,
                                                    ],
                                                    unit_flag=0b11,
                                                )
                                            cl0_iter += 1

                                        T.set_flag("mte1", "mte2", P_EV)

                                        if tilem == s2ltm - 1:
                                            T.set_cross_flag("MTE2", EV_CREDIT)

                            T.set_cross_flag("FIX", EV_PV)

                    T.wait_flag("m", "mte1", L0AB_EV0)
                    T.wait_flag("m", "mte1", L0AB_EV1)
                    T.wait_flag("mte1", "mte2", KV_EV0)
                    T.wait_flag("mte1", "mte2", KV_EV1)
                    T.wait_flag("mte1", "mte2", KV_EV2)
                    T.wait_flag("mte1", "mte2", Q_EV)
                    T.wait_flag("mte1", "mte2", P_EV)

                with T.Scope("V"):
                    T.tile.fill(ones_ub, T.float32(1.0))

                    T.set_flag("v", "mte2", IN_EV)
                    T.set_flag("v", "mte2", IN_EV + 1)
                    T.set_flag("v", "mte2", ACC_EV)
                    T.set_flag("mte3", "v", OUT_EV)
                    T.set_flag("mte3", "v", LSE_EV)

                    V0_NMAX = min(S2_BASE, topk_cmp)
                    V0_NB = (V0_NMAX + MERGE_ROWS - 1) // MERGE_ROWS
                    for g in T.serial(1, GLOOP + 2):
                        if g < GLOOP:
                            v0task = g // MAX_TILES
                            v0tile = g % MAX_TILES
                            v0pid = v0task * core_num + cid
                            if v0pid < block_num:
                                v0b = T.cast(v0pid // max_seq, "int32")
                                v0s = T.cast(v0pid % max_seq, "int32")

                                v0tok = q_prefix[v0b] + v0s
                                if v0s < act_q_lens[v0b]:
                                    if v0tile == 0 or g == 1:
                                        T.wait_cross_flag(EV_CREDIT)
                                    v0skv = seqused_kv[v0b] - act_q_lens[v0b] + v0s
                                    v0acmp = (v0skv + 1) // cmp_ratio

                                    v0asparse = T.min(v0acmp, topk_cmp)
                                    v0ctiles = (v0asparse + S2_BASE - 1) // S2_BASE
                                    if v0tile != 0 and v0tile < 1 + v0ctiles:
                                        v0rel = v0tile - 1
                                        v0n = T.min(S2_BASE, v0asparse - v0rel * S2_BASE)

                                        v0half = (v0n + 1) // 2
                                        v0start = tir.Select(vid == 0, 0, v0half)
                                        v0lim = tir.Select(vid == 0, v0half, v0n)

                                        T.set_flag("mte3", "mte2", MRG_EV + 0)
                                        T.set_flag("mte3", "mte2", MRG_EV + 1)
                                        for jb in T.serial(V0_NB):
                                            pp = jb % 2
                                            if v0start + jb * MERGE_ROWS < v0lim:
                                                T.wait_flag("mte3", "mte2", MRG_EV + pp)
                                                for jj in range(MERGE_ROWS):
                                                    jtok = v0start + jb * MERGE_ROWS + jj
                                                    if jtok < v0lim:
                                                        r2 = cmp_indices[
                                                            v0tok,
                                                            0,
                                                            v0rel * S2_BASE + jtok,
                                                        ]
                                                        blkid = cmp_block_table[
                                                            v0b,
                                                            r2 // cmp_block_size,
                                                        ]

                                                        T.copy(
                                                            cmp_kv[
                                                                blkid,
                                                                r2 % cmp_block_size,
                                                                0,
                                                                :,
                                                            ],
                                                            merge_ub[pp, jj, :],
                                                        )

                                                T.set_flag("mte2", "mte3", MRG_EV + pp)
                                                T.wait_flag("mte2", "mte3", MRG_EV + pp)
                                                bcnt = T.min(
                                                    MERGE_ROWS,
                                                    v0lim - (v0start + jb * MERGE_ROWS),
                                                )
                                                T.copy(
                                                    merge_ub[pp, 0:bcnt, :],
                                                    kvMergeGm[
                                                        cid,
                                                        v0task % KVMERGE_RING,
                                                        v0start + jb * MERGE_ROWS : v0start + jb * MERGE_ROWS + bcnt,
                                                        :,
                                                    ],
                                                )

                                                T.set_flag("mte3", "mte2", MRG_EV + pp)

                                        T.wait_flag("mte3", "mte2", MRG_EV + 0)
                                        T.wait_flag("mte3", "mte2", MRG_EV + 1)

                                        T.set_cross_flag("MTE3", EV_V0)

                        if g < GLOOP + 1:
                            T.wait_cross_flag(EV_QK)
                            gs = g - 1
                            buf = gs % 2
                            prev = (gs - 1) % 2
                            task = gs // MAX_TILES
                            tile = gs % MAX_TILES
                            pid = task * core_num + cid
                            if pid < block_num:
                                b = T.cast(pid // max_seq, "int32")
                                s = T.cast(pid % max_seq, "int32")
                                act_q = act_q_lens[b]
                                if s < act_q:
                                    act_kv = seqused_kv[b]
                                    s_global = act_kv - act_q + s
                                    act_cmp = (s_global + 1) // cmp_ratio
                                    cmp_tiles = (T.min(act_cmp, topk_cmp) + S2_BASE - 1) // S2_BASE
                                    s2lt = 1 + cmp_tiles
                                    if tile < s2lt:
                                        is_first = tile == 0
                                        ori_left = T.max(s_global - ori_win_left, 0)
                                        win = s_global + 1 - ori_left
                                        rel = tile - 1

                                        tw = tir.Select(
                                            is_first,
                                            win,
                                            T.min(
                                                S2_BASE,
                                                T.min(act_cmp, topk_cmp) - rel * S2_BASE,
                                            ),
                                        )
                                        tw_a = (tw + 15) // 16 * 16

                                        tw_64 = (tw + 63) // 64 * 64

                                        is_narrow = tw <= BI
                                        for mc in range(NMC):
                                            r0 = vid * G2 + mc * M_CHUNK
                                            ps = mc & 1

                                            T.wait_flag("v", "mte2", IN_EV + ps)

                                            T.tile.fill(
                                                in_ub[ps, :, :],
                                                -T.infinity(accum_dtype),
                                            )
                                            T.set_flag("v", "mte2", IN_EV + ps)
                                            T.wait_flag("v", "mte2", IN_EV + ps)

                                            T.copy(
                                                workspace_s[
                                                    cid,
                                                    buf,
                                                    r0 : r0 + M_CHUNK,
                                                    0:tw,
                                                ],
                                                in_ub[ps, :, 0:tw],
                                            )
                                            T.copy(
                                                sinks[r0 : r0 + M_CHUNK],
                                                sink_ub[ps, :, :],
                                            )

                                            T.set_flag("mte2", "v", IN_EV + ps)
                                            T.wait_flag("mte2", "v", IN_EV + ps)
                                            T.tile.mul(
                                                in_ub[ps, :, :],
                                                in_ub[ps, :, :],
                                                softmax_scale,
                                            )

                                            T.pipe_barrier("v")

                                            if is_narrow:
                                                T.tile.fill(
                                                    m_i[buf, mc, :, :],
                                                    -T.infinity(accum_dtype),
                                                )
                                                for kc in range(BI // 64):
                                                    if kc * 64 < tw_64:
                                                        T.reduce_max(
                                                            in_ub[
                                                                ps,
                                                                :,
                                                                kc * 64 : kc * 64 + 64,
                                                            ],
                                                            part,
                                                            dim=-1,
                                                        )
                                                        T.tile.max(
                                                            m_i[buf, mc, :, :],
                                                            m_i[buf, mc, :, :],
                                                            part,
                                                        )
                                            else:
                                                T.reduce_max(
                                                    in_ub[ps, :, :],
                                                    m_i[buf, mc, :, :],
                                                    dim=-1,
                                                )
                                            if is_first:
                                                T.tile.max(
                                                    m_i[buf, mc, :, :],
                                                    sink_ub[ps, :, :],
                                                    m_i[buf, mc, :, :],
                                                )
                                                T.tile.sub(
                                                    expmax[buf, mc, :, :],
                                                    sink_ub[ps, :, :],
                                                    m_i[buf, mc, :, :],
                                                )
                                            else:
                                                T.tile.max(
                                                    m_i[buf, mc, :, :],
                                                    m_i[prev, mc, :, :],
                                                    m_i[buf, mc, :, :],
                                                )
                                                T.tile.sub(
                                                    expmax[buf, mc, :, :],
                                                    m_i[prev, mc, :, :],
                                                    m_i[buf, mc, :, :],
                                                )
                                            T.tile.exp(
                                                expmax[buf, mc, :, :],
                                                expmax[buf, mc, :, :],
                                            )

                                            if is_narrow:
                                                T.tile.brcb_experiment(
                                                    brcb_s,
                                                    m_i[buf, mc, :, :],
                                                    M_CHUNK // 8,
                                                    1,
                                                    8,
                                                )
                                                for kc in range(BI // 64):
                                                    if kc * 64 < tw_64:
                                                        T.tile.row_expand_sub_experiment(
                                                            in_ub[
                                                                ps,
                                                                :,
                                                                kc * 64 : kc * 64 + 64,
                                                            ],
                                                            in_ub[
                                                                ps,
                                                                :,
                                                                kc * 64 : kc * 64 + 64,
                                                            ],
                                                            brcb_s,
                                                        )
                                                for kc in range(BI // 64):
                                                    if kc * 64 < tw_64:
                                                        T.tile.exp_experiment(
                                                            in_ub[
                                                                ps,
                                                                :,
                                                                kc * 64 : kc * 64 + 64,
                                                            ],
                                                            in_ub[
                                                                ps,
                                                                :,
                                                                kc * 64 : kc * 64 + 64,
                                                            ],
                                                        )
                                            else:
                                                T.tile.broadcast(softmax_cmp, m_i[buf, mc, :, :])
                                                T.tile.sub(
                                                    in_ub[ps, :, :],
                                                    in_ub[ps, :, :],
                                                    softmax_cmp,
                                                )
                                                T.tile.exp(in_ub[ps, :, :], in_ub[ps, :, :])
                                            if is_first:
                                                T.tile.mul(
                                                    denom[buf, mc, :, :],
                                                    ones_ub,
                                                    expmax[buf, mc, :, :],
                                                )
                                            else:
                                                T.tile.mul(
                                                    denom[buf, mc, :, :],
                                                    denom[prev, mc, :, :],
                                                    expmax[buf, mc, :, :],
                                                )

                                            T.pipe_barrier("v")

                                            T.wait_flag("mte3", "v", OUT_EV)
                                            T.tile.cast(
                                                out_ub,
                                                in_ub[ps, :, :],
                                                "CAST_ROUND",
                                                M_CHUNK * S2_BASE,
                                            )

                                            T.pipe_barrier("v")

                                            if is_narrow:
                                                T.tile.fill(sumP, T.float32(0.0))
                                                for kc in range(BI // 64):
                                                    if kc * 64 < tw_64:
                                                        T.reduce_sum(
                                                            in_ub[
                                                                ps,
                                                                :,
                                                                kc * 64 : kc * 64 + 64,
                                                            ],
                                                            part,
                                                            dim=-1,
                                                        )
                                                        T.tile.add(sumP, sumP, part)
                                            else:
                                                T.reduce_sum(in_ub[ps, :, :], sumP, dim=-1)
                                            T.tile.add(
                                                denom[buf, mc, :, :],
                                                denom[buf, mc, :, :],
                                                sumP,
                                            )

                                            T.set_flag("v", "mte2", IN_EV + ps)
                                            T.set_flag("v", "mte3", OUT_EV)
                                            T.wait_flag("v", "mte3", OUT_EV)

                                            T.copy(
                                                out_ub[:, 0:tw_a],
                                                workspace_p[
                                                    cid,
                                                    buf,
                                                    r0 : r0 + M_CHUNK,
                                                    0:tw_a,
                                                ],
                                            )

                                            T.set_flag("mte3", "v", OUT_EV)
                            T.set_cross_flag("MTE3", EV_P)

                        if g >= 2:
                            T.wait_cross_flag(EV_PV)
                            go = g - 2
                            bufo = go % 2
                            prevo = (go - 1) % 2
                            task = go // MAX_TILES
                            tile = go % MAX_TILES
                            pid = task * core_num + cid
                            if pid < block_num:
                                b = T.cast(pid // max_seq, "int32")
                                s = T.cast(pid % max_seq, "int32")
                                act_q = act_q_lens[b]
                                if s < act_q:
                                    act_kv = seqused_kv[b]
                                    s_global = act_kv - act_q + s

                                    act_cmp = tir.Select(
                                        s_global < cmp_ratio - 1,
                                        0,
                                        (s_global + 1) // cmp_ratio,
                                    )
                                    cmp_tiles = (T.min(act_cmp, topk_cmp) + S2_BASE - 1) // S2_BASE
                                    s2lt = 1 + cmp_tiles
                                    if tile < s2lt:
                                        not_first = tile != 0
                                        is_last = tile == s2lt - 1
                                        tok = q_prefix[b] + s
                                        for mc in range(NMC):
                                            r0 = vid * G2 + mc * M_CHUNK
                                            po = mc & 1

                                            T.wait_flag("v", "mte2", IN_EV + po)
                                            T.copy(
                                                workspace_o[cid, bufo, r0 : r0 + M_CHUNK, :],
                                                in_ub[po, :, :],
                                            )

                                            T.set_flag("mte2", "v", IN_EV + po)
                                            T.wait_flag("mte2", "v", IN_EV + po)
                                            if not_first:
                                                T.set_flag("mte3", "mte2", FENCE)
                                                T.wait_flag("mte3", "mte2", FENCE)

                                                T.wait_flag("v", "mte2", ACC_EV)
                                                T.copy(
                                                    workspace_acc[cid, prevo, r0 : r0 + M_CHUNK, :],
                                                    acc_pre,
                                                )

                                                T.set_flag("mte2", "v", ACC_EV)
                                                T.wait_flag("mte2", "v", ACC_EV)

                                                T.tile.brcb_experiment(
                                                    brcb_d,
                                                    expmax[bufo, mc, :, :],
                                                    M_CHUNK // 8,
                                                    1,
                                                    8,
                                                )
                                                T.pipe_barrier("v")
                                                for dcol in T.serial(D // (8 * BLK)):
                                                    cb = dcol * (8 * BLK)
                                                    T.tile.row_expand_mul_experiment(
                                                        acc_pre[:, cb : cb + 8 * BLK],
                                                        acc_pre[:, cb : cb + 8 * BLK],
                                                        brcb_d,
                                                    )
                                                T.pipe_barrier("v")
                                                T.tile.add(
                                                    in_ub[po, :, :],
                                                    in_ub[po, :, :],
                                                    acc_pre,
                                                )
                                                T.pipe_barrier("v")

                                                T.set_flag("v", "mte2", ACC_EV)
                                            if is_last:
                                                T.tile.brcb_experiment(
                                                    brcb_d,
                                                    denom[bufo, mc, :, :],
                                                    M_CHUNK // 8,
                                                    1,
                                                    8,
                                                )
                                                T.pipe_barrier("v")
                                                for dcol in T.serial(D // (8 * BLK)):
                                                    cb = dcol * (8 * BLK)
                                                    T.tile.row_expand_div_experiment(
                                                        in_ub[po, :, cb : cb + 8 * BLK],
                                                        in_ub[po, :, cb : cb + 8 * BLK],
                                                        brcb_d,
                                                    )
                                                T.pipe_barrier("v")

                                                T.wait_flag("mte3", "v", OUT_EV)
                                                T.tile.cast(
                                                    out_ub,
                                                    in_ub[po, :, :],
                                                    out_cast_mode,
                                                    M_CHUNK * D,
                                                )

                                                T.set_flag("v", "mte2", IN_EV + po)
                                                T.set_flag("v", "mte3", OUT_EV)
                                                T.wait_flag("v", "mte3", OUT_EV)
                                                T.copy(
                                                    out_ub,
                                                    Output[tok, r0 : r0 + M_CHUNK, :],
                                                )

                                                T.set_flag("mte3", "v", OUT_EV)

                                                T.wait_flag("mte3", "v", LSE_EV)
                                                T.tile.ln(
                                                    lse_ub,
                                                    denom[bufo, mc, :, :],
                                                )
                                                T.tile.add(
                                                    lse_ub,
                                                    lse_ub,
                                                    m_i[bufo, mc, :, :],
                                                )
                                                T.set_flag("v", "mte3", LSE_EV)
                                                T.wait_flag("v", "mte3", LSE_EV)
                                                T.copy(lse_ub, LSE[tok, r0 : r0 + M_CHUNK])
                                                T.set_flag("mte3", "v", LSE_EV)
                                            else:
                                                T.set_flag("v", "mte3", IN_EV + po)
                                                T.wait_flag("v", "mte3", IN_EV + po)
                                                T.copy(
                                                    in_ub[po, :, :],
                                                    workspace_acc[cid, bufo, r0 : r0 + M_CHUNK, :],
                                                )
                                                T.set_flag("mte3", "v", IN_EV + po)
                                                T.wait_flag("mte3", "v", IN_EV + po)
                                                T.set_flag("v", "mte2", IN_EV + po)

                            T.set_cross_flag("MTE3", EV_PV)

                    T.wait_flag("v", "mte2", IN_EV)
                    T.wait_flag("v", "mte2", IN_EV + 1)
                    T.wait_flag("v", "mte2", ACC_EV)
                    T.wait_flag("mte3", "v", OUT_EV)
                    T.wait_flag("mte3", "v", LSE_EV)

                    for _cr in range(KVMERGE_RING):
                        T.wait_cross_flag(EV_CREDIT)

        return sparse_flash_mla_csa

    func = kernel()
    if os.environ.get("SAS_DUMP_SRC"):
        try:
            src = func.get_kernel_source()
            with open("/tmp/csa_gen.cpp", "w") as fh:
                fh.write(src)
            print(f"[SAS_DUMP_SRC] wrote {len(src)} chars to /tmp/csa_gen.cpp")
        except Exception as exc:
            print(f"[SAS_DUMP_SRC] get_kernel_source failed: {exc!r}")
    return func


def _build_swa(
    *,
    batch,
    max_seq,
    total_tokens,
    ori_block_num,
    ori_block_size,
    ori_table_len,
    cmp_block_num,
    cmp_block_size,
    cmp_table_len,
    n_heads,
    head_dim,
    ori_win_left,
    softmax_scale,
    dtype,
    core_num,
):
    N1 = n_heads
    N2 = 1
    D = head_dim

    D2 = D // 2
    G = N1 // N2
    BI = DEFAULT_BLOCK_I

    PV_NT = D // BI
    PV_NW = BI

    KL0 = 128
    VEC_NUM = 2
    G2 = G // VEC_NUM
    BLK = 8

    DEBUG_SERIAL = False

    accum_dtype = "float"
    idx_dtype = "int32"

    out_cast_mode = "CAST_RINT" if dtype == "bfloat16" else "CAST_ROUND"

    block_num = batch * max_seq
    n_iter = (block_num + core_num - 1) // core_num

    EV_QK = 0
    EV_P = 1
    EV_PV = 2

    KV_EV0 = 2
    KV_EV1 = 3
    KV_EV2 = 6

    Q_EV = 1
    P_EV = 7

    L0AB_EV0 = 4
    L0AB_EV1 = 5

    SV_S_EV = 2
    SV_P_EV = 3
    VO_O_EV = 4
    VO_OUT_EV = 5
    VO_LSE_EV = 6

    q_shape = [total_tokens, N1, D]
    ori_kv_shape = [ori_block_num, ori_block_size, N2, D]
    cmp_kv_shape = [cmp_block_num, cmp_block_size, N2, D]
    cmp_idx_shape = [total_tokens, N2, DEFAULT_BLOCK_I]

    @tilelang.jit(out_idx=[10, 11], workspace_idx=[12, 13, 14], target="ascendc")
    def kernel():
        @T.prim_func
        def sparse_flash_mla_swa(
            Q: T.Tensor(q_shape, dtype),
            ori_kv: T.Tensor(ori_kv_shape, dtype),
            ori_block_table: T.Tensor([batch, ori_table_len], idx_dtype),
            cmp_kv: T.Tensor(cmp_kv_shape, dtype),
            cmp_block_table: T.Tensor([batch, cmp_table_len], idx_dtype),
            cmp_indices: T.Tensor(cmp_idx_shape, idx_dtype),
            q_prefix: T.Tensor([batch], idx_dtype),
            act_q_lens: T.Tensor([batch], idx_dtype),
            seqused_kv: T.Tensor([batch], idx_dtype),
            sinks: T.Tensor([N1], accum_dtype),
            Output: T.Tensor(q_shape, dtype),
            LSE: T.Tensor([total_tokens, N1], accum_dtype),
            workspace_s: T.Tensor([core_num, 2, G, BI], accum_dtype),
            workspace_p: T.Tensor([core_num, 2, G, BI], dtype),
            workspace_o: T.Tensor([core_num, 2, G, D], accum_dtype),
        ):
            with T.Kernel(core_num, is_npu=True) as (cid, vid):
                q_l1_0 = T.alloc_L1([G, D2], dtype)
                q_l1_1 = T.alloc_L1([G, D2], dtype)

                kv_ring = T.alloc_L1([3, BI, D2], dtype)
                p_l1 = T.alloc_L1([G, BI], dtype)

                cL0 = T.alloc_L0C([2, G, BI], accum_dtype)

                cl0_iter = T.alloc_var("int32", init=0)

                p_l0a = T.alloc_L0A([2, G, BI], dtype)
                v_l0b = T.alloc_L0B([2, BI, BI], dtype)

                s_ub = T.alloc_ub([G2, BI], accum_dtype)
                p_half = T.alloc_ub([G2, BI], dtype)

                m_2d = T.alloc_ub([G2, BI], accum_dtype)
                sumP = T.alloc_ub([G2, 1], accum_dtype)
                o_ub = T.alloc_ub([G2, D], accum_dtype)
                o_half = T.alloc_ub([G2, D], dtype)
                sink_ub = T.alloc_ub([G2, 1], accum_dtype)
                lse_ub = T.alloc_ub([G2, 1], accum_dtype)

                ones_ub = T.alloc_ub([G2, 1], accum_dtype)
                expmax_ub = T.alloc_ub([G2, 1], accum_dtype)

                brcb_d = T.alloc_ub([G2, BLK], accum_dtype)

                m_i = T.alloc_ub([2, G2, 1], accum_dtype)
                denom = T.alloc_ub([2, G2, 1], accum_dtype)

                with T.Scope("C"):
                    T.set_flag("m", "mte1", L0AB_EV0)
                    T.set_flag("m", "mte1", L0AB_EV1)

                    T.set_flag("mte1", "mte2", KV_EV0)
                    T.set_flag("mte1", "mte2", KV_EV1)
                    T.set_flag("mte1", "mte2", KV_EV2)

                    T.set_flag("mte1", "mte2", Q_EV)
                    T.set_flag("mte1", "mte2", P_EV)
                    for j in T.serial(n_iter + 1):
                        if j < n_iter:
                            pid = j * core_num + cid
                            buf = j % 2
                            if pid < block_num:
                                b = T.cast(pid // max_seq, "int32")
                                s = T.cast(pid % max_seq, "int32")
                                act_q = act_q_lens[b]
                                if s < act_q:
                                    act_kv = seqused_kv[b]
                                    tok = q_prefix[b] + s
                                    s_global = act_kv - act_q + s
                                    ori_left = T.max(s_global - ori_win_left, 0)
                                    win = s_global + 1 - ori_left

                                    T.wait_flag("mte1", "mte2", Q_EV)
                                    T.copy(Q[tok, :, 0:D2], q_l1_0)
                                    T.copy(Q[tok, :, D2:D], q_l1_1)
                                    T.set_flag("mte2", "mte1", Q_EV)

                                    for h in range(2):
                                        T.wait_flag("mte1", "mte2", KV_EV0 + h)

                                        pa_start = ori_left
                                        pa_cur = T.alloc_var("int32", init=pa_start)
                                        for _pg in range((BI + ori_block_size - 1) // ori_block_size + 1):
                                            pa_done = pa_cur - pa_start
                                            if pa_done < win:
                                                pa_lg = pa_cur // ori_block_size
                                                pa_ph = ori_block_table[b, pa_lg]
                                                pa_rem = pa_cur % ori_block_size
                                                pa_run = T.min(
                                                    ori_block_size - pa_rem,
                                                    win - pa_done,
                                                )
                                                T.copy(
                                                    ori_kv[
                                                        pa_ph,
                                                        pa_rem : pa_rem + pa_run,
                                                        0,
                                                        h * D2 : h * D2 + D2,
                                                    ],
                                                    kv_ring[
                                                        h,
                                                        pa_done : pa_done + pa_run,
                                                        :,
                                                    ],
                                                )
                                                pa_cur += pa_run
                                        T.set_flag("mte2", "mte1", KV_EV0 + h)

                                    win_align = (win + 15) // 16 * 16
                                    T.wait_flag("mte2", "mte1", Q_EV)

                                    cs = cl0_iter % 2

                                    T.wait_flag("mte2", "mte1", KV_EV0)
                                    T.wait_flag("m", "mte1", L0AB_EV0)
                                    T.copy(q_l1_0[:, 0:KL0], p_l0a[0, :, :])
                                    T.copy(
                                        kv_ring[0, :, 0:KL0],
                                        v_l0b[0, :, :],
                                        transpose=True,
                                        real_k=KL0,
                                        real_n=win_align,
                                    )
                                    T.set_flag("mte1", "m", L0AB_EV0)
                                    T.wait_flag("mte1", "m", L0AB_EV0)
                                    T.mma(
                                        p_l0a[0, :, :],
                                        v_l0b[0, :, :],
                                        cL0[cs, :, :],
                                        init=True,
                                        k_actual=KL0,
                                        n_actual=win_align,
                                        unit_flag=0b10,
                                    )

                                    if win_align < 40:
                                        T.pipe_barrier("m")
                                    T.set_flag("m", "mte1", L0AB_EV0)
                                    T.wait_flag("m", "mte1", L0AB_EV1)
                                    T.copy(q_l1_0[:, KL0 : 2 * KL0], p_l0a[1, :, :])
                                    T.copy(
                                        kv_ring[0, :, KL0 : 2 * KL0],
                                        v_l0b[1, :, :],
                                        transpose=True,
                                        real_k=KL0,
                                        real_n=win_align,
                                    )
                                    T.set_flag("mte1", "m", L0AB_EV1)
                                    T.wait_flag("mte1", "m", L0AB_EV1)
                                    T.mma(
                                        p_l0a[1, :, :],
                                        v_l0b[1, :, :],
                                        cL0[cs, :, :],
                                        init=False,
                                        k_actual=KL0,
                                        n_actual=win_align,
                                        unit_flag=0b10,
                                    )
                                    if win_align < 40:
                                        T.pipe_barrier("m")
                                    T.set_flag("m", "mte1", L0AB_EV1)
                                    T.set_flag("mte1", "mte2", KV_EV0)

                                    T.wait_flag("mte2", "mte1", KV_EV1)
                                    T.wait_flag("m", "mte1", L0AB_EV0)
                                    T.copy(q_l1_1[:, 0:KL0], p_l0a[0, :, :])
                                    T.copy(
                                        kv_ring[1, :, 0:KL0],
                                        v_l0b[0, :, :],
                                        transpose=True,
                                        real_k=KL0,
                                        real_n=win_align,
                                    )
                                    T.set_flag("mte1", "m", L0AB_EV0)
                                    T.wait_flag("mte1", "m", L0AB_EV0)
                                    T.mma(
                                        p_l0a[0, :, :],
                                        v_l0b[0, :, :],
                                        cL0[cs, :, :],
                                        init=False,
                                        k_actual=KL0,
                                        n_actual=win_align,
                                        unit_flag=0b10,
                                    )
                                    if win_align < 40:
                                        T.pipe_barrier("m")
                                    T.set_flag("m", "mte1", L0AB_EV0)
                                    T.wait_flag("m", "mte1", L0AB_EV1)
                                    T.copy(q_l1_1[:, KL0 : 2 * KL0], p_l0a[1, :, :])
                                    T.copy(
                                        kv_ring[1, :, KL0 : 2 * KL0],
                                        v_l0b[1, :, :],
                                        transpose=True,
                                        real_k=KL0,
                                        real_n=win_align,
                                    )
                                    T.set_flag("mte1", "m", L0AB_EV1)
                                    T.wait_flag("mte1", "m", L0AB_EV1)
                                    T.mma(
                                        p_l0a[1, :, :],
                                        v_l0b[1, :, :],
                                        cL0[cs, :, :],
                                        init=False,
                                        k_actual=KL0,
                                        n_actual=win_align,
                                        unit_flag=0b11,
                                    )
                                    if win_align < 40:
                                        T.pipe_barrier("m")
                                    T.set_flag("m", "mte1", L0AB_EV1)
                                    T.set_flag("mte1", "mte2", KV_EV1)

                                    T.copy(
                                        cL0[cs, :, :],
                                        workspace_s[cid, buf, :, 0:win_align],
                                        unit_flag=0b11,
                                    )

                                    T.set_flag("mte1", "mte2", Q_EV)

                                    cl0_iter += 1
                            T.set_cross_flag("FIX", EV_QK)

                        if j >= 1:
                            T.wait_cross_flag(EV_P)

                            pidm = (j - 1) * core_num + cid
                            bufm = (j - 1) % 2
                            if pidm < block_num:
                                bm = T.cast(pidm // max_seq, "int32")
                                sm = T.cast(pidm % max_seq, "int32")
                                if sm < act_q_lens[bm]:
                                    act_kvm = seqused_kv[bm]
                                    s_globalm = act_kvm - act_q_lens[bm] + sm
                                    ori_leftm = T.max(s_globalm - ori_win_left, 0)
                                    winm = s_globalm + 1 - ori_leftm

                                    T.wait_flag("mte1", "mte2", P_EV)
                                    T.copy(workspace_p[cid, bufm, :, :], p_l1)
                                    T.set_flag("mte2", "mte1", P_EV)
                                    T.wait_flag("mte2", "mte1", P_EV)

                                    for nl in range(PV_NT):
                                        slot = (2 + nl) % 3
                                        pp = nl % 2
                                        cs = cl0_iter % 2

                                        ev = T.if_then_else(slot < 2, KV_EV0 + slot, KV_EV2)

                                        T.wait_flag("mte1", "mte2", ev)

                                        pv_start = ori_leftm
                                        pv_cur = T.alloc_var("int32", init=pv_start)
                                        for _pg in range((BI + ori_block_size - 1) // ori_block_size + 1):
                                            pv_done = pv_cur - pv_start
                                            if pv_done < winm:
                                                pv_lg = pv_cur // ori_block_size
                                                pv_ph = ori_block_table[bm, pv_lg]
                                                pv_rem = pv_cur % ori_block_size
                                                pv_run = T.min(
                                                    ori_block_size - pv_rem,
                                                    winm - pv_done,
                                                )
                                                T.copy(
                                                    ori_kv[
                                                        pv_ph,
                                                        pv_rem : pv_rem + pv_run,
                                                        0,
                                                        nl * PV_NW : nl * PV_NW + PV_NW,
                                                    ],
                                                    kv_ring[
                                                        slot,
                                                        pv_done : pv_done + pv_run,
                                                        0:PV_NW,
                                                    ],
                                                )
                                                pv_cur += pv_run
                                        T.set_flag("mte2", "mte1", ev)

                                        T.wait_flag("m", "mte1", L0AB_EV0 + pp)
                                        T.wait_flag("mte2", "mte1", ev)
                                        T.copy(p_l1[:, :], p_l0a[pp, :, :], real_k=winm)
                                        T.copy(
                                            kv_ring[slot, :, 0:PV_NW],
                                            v_l0b[pp, :, :],
                                            real_k=winm,
                                        )
                                        T.set_flag("mte1", "m", L0AB_EV0 + pp)
                                        T.wait_flag("mte1", "m", L0AB_EV0 + pp)
                                        T.mma(
                                            p_l0a[pp, :, :],
                                            v_l0b[pp, :, :],
                                            cL0[cs, :, :],
                                            init=True,
                                            k_actual=winm,
                                            unit_flag=0b11,
                                        )
                                        T.set_flag("m", "mte1", L0AB_EV0 + pp)

                                        T.set_flag("mte1", "mte2", ev)
                                        T.copy(
                                            cL0[cs, :, :],
                                            workspace_o[
                                                cid,
                                                bufm,
                                                :,
                                                nl * PV_NW : (nl + 1) * PV_NW,
                                            ],
                                            unit_flag=0b11,
                                        )

                                        cl0_iter += 1

                                    T.set_flag("mte1", "mte2", P_EV)

                                    if DEBUG_SERIAL:
                                        T.barrier_all()
                            T.set_cross_flag("FIX", EV_PV)

                    T.wait_flag("m", "mte1", L0AB_EV0)
                    T.wait_flag("m", "mte1", L0AB_EV1)

                    T.wait_flag("mte1", "mte2", KV_EV0)
                    T.wait_flag("mte1", "mte2", KV_EV1)
                    T.wait_flag("mte1", "mte2", KV_EV2)

                    T.wait_flag("mte1", "mte2", Q_EV)
                    T.wait_flag("mte1", "mte2", P_EV)

                with T.Scope("V"):
                    T.tile.fill(ones_ub, T.float32(1.0))

                    T.set_flag("v", "mte2", SV_S_EV)
                    T.set_flag("mte3", "v", SV_P_EV)

                    T.set_flag("v", "mte2", VO_O_EV)
                    T.set_flag("mte3", "v", VO_OUT_EV)
                    T.set_flag("mte3", "v", VO_LSE_EV)

                    for j in T.serial(1, n_iter + 2):
                        if j < n_iter + 1:
                            T.wait_cross_flag(EV_QK)
                            pid = (j - 1) * core_num + cid
                            buf = (j - 1) % 2
                            if pid < block_num:
                                b = T.cast(pid // max_seq, "int32")
                                s = T.cast(pid % max_seq, "int32")
                                act_q = act_q_lens[b]
                                if s < act_q:
                                    act_kv = seqused_kv[b]
                                    s_global = act_kv - act_q + s
                                    ori_left = T.max(s_global - ori_win_left, 0)

                                    winm = s_global + 1 - ori_left

                                    T.wait_flag("v", "mte2", SV_S_EV)

                                    T.tile.fill(s_ub, -T.infinity(accum_dtype))
                                    T.set_flag("v", "mte2", SV_S_EV)
                                    T.wait_flag("v", "mte2", SV_S_EV)
                                    T.copy(
                                        workspace_s[cid, buf, vid * G2 : (vid + 1) * G2, 0:winm],
                                        s_ub[:, 0:winm],
                                    )
                                    T.copy(sinks[vid * G2 : (vid + 1) * G2], sink_ub)

                                    T.set_flag("mte2", "v", SV_S_EV)
                                    T.wait_flag("mte2", "v", SV_S_EV)
                                    T.tile.mul(s_ub, s_ub, softmax_scale)

                                    T.pipe_barrier("v")

                                    T.reduce_max(s_ub, m_i[buf, :, :], dim=-1)

                                    T.tile.max(m_i[buf, :, :], sink_ub, m_i[buf, :, :])
                                    T.tile.broadcast(m_2d, m_i[buf, :, :])
                                    T.tile.sub(s_ub, s_ub, m_2d)
                                    T.tile.exp(s_ub, s_ub)

                                    T.tile.sub(expmax_ub, sink_ub, m_i[buf, :, :])
                                    T.tile.exp(expmax_ub, expmax_ub)

                                    T.pipe_barrier("v")

                                    T.wait_flag("mte3", "v", SV_P_EV)
                                    T.tile.cast(p_half, s_ub, "CAST_ROUND", G2 * BI)

                                    T.pipe_barrier("v")
                                    T.reduce_sum(s_ub, sumP, dim=-1)
                                    T.tile.add(denom[buf, :, :], expmax_ub, sumP)

                                    T.set_flag("v", "mte2", SV_S_EV)
                                    T.set_flag("v", "mte3", SV_P_EV)
                                    T.wait_flag("v", "mte3", SV_P_EV)
                                    T.copy(
                                        p_half,
                                        workspace_p[cid, buf, vid * G2 : (vid + 1) * G2, :],
                                    )

                                    T.set_flag("mte3", "v", SV_P_EV)
                            T.set_cross_flag("MTE3", EV_P)

                        if j >= 2:
                            T.wait_cross_flag(EV_PV)
                            pidm = (j - 2) * core_num + cid
                            bufm = (j - 2) % 2
                            if pidm < block_num:
                                bm = T.cast(pidm // max_seq, "int32")
                                sm = T.cast(pidm % max_seq, "int32")
                                if sm < act_q_lens[bm]:
                                    tokm = q_prefix[bm] + sm

                                    T.wait_flag("v", "mte2", VO_O_EV)
                                    T.copy(
                                        workspace_o[cid, bufm, vid * G2 : (vid + 1) * G2, :],
                                        o_ub,
                                    )

                                    T.set_flag("mte2", "v", VO_O_EV)
                                    T.wait_flag("mte2", "v", VO_O_EV)

                                    T.tile.brcb_experiment(brcb_d, denom[bufm, :, :], G2 // 8, 1, 8)
                                    T.pipe_barrier("v")
                                    for dcol in T.serial(D // (8 * BLK)):
                                        cb = dcol * (8 * BLK)
                                        T.tile.row_expand_div_experiment(
                                            o_ub[:, cb : cb + 8 * BLK],
                                            o_ub[:, cb : cb + 8 * BLK],
                                            brcb_d,
                                        )

                                    T.pipe_barrier("v")

                                    T.wait_flag("mte3", "v", VO_OUT_EV)
                                    T.tile.cast(o_half, o_ub, out_cast_mode, G2 * D)

                                    T.set_flag("v", "mte2", VO_O_EV)
                                    T.set_flag("v", "mte3", VO_OUT_EV)
                                    T.wait_flag("v", "mte3", VO_OUT_EV)
                                    T.copy(
                                        o_half,
                                        Output[tokm, vid * G2 : (vid + 1) * G2, :],
                                    )

                                    T.set_flag("mte3", "v", VO_OUT_EV)

                                    T.wait_flag("mte3", "v", VO_LSE_EV)
                                    T.tile.ln(lse_ub, denom[bufm, :, :])
                                    T.tile.add(lse_ub, lse_ub, m_i[bufm, :, :])
                                    T.set_flag("v", "mte3", VO_LSE_EV)
                                    T.wait_flag("v", "mte3", VO_LSE_EV)
                                    T.copy(lse_ub, LSE[tokm, vid * G2 : (vid + 1) * G2])
                                    T.set_flag("mte3", "v", VO_LSE_EV)

                    T.wait_flag("v", "mte2", SV_S_EV)
                    T.wait_flag("mte3", "v", SV_P_EV)
                    T.wait_flag("v", "mte2", VO_O_EV)
                    T.wait_flag("mte3", "v", VO_OUT_EV)
                    T.wait_flag("mte3", "v", VO_LSE_EV)

        return sparse_flash_mla_swa

    func = kernel()

    if os.environ.get("SAS_DUMP_SRC"):
        try:
            src = func.get_kernel_source()
            with open("/tmp/swa_gen.cpp", "w") as fh:
                fh.write(src)
            print(f"[SAS_DUMP_SRC] wrote {len(src)} chars to /tmp/swa_gen.cpp")
        except Exception as exc:
            print(f"[SAS_DUMP_SRC] get_kernel_source failed: {exc!r}")
    return func


if __name__ == "__main__":
    build_sparse_flash_mla(
        batch=1,
        max_seq=1024,
        total_tokens=1024,
        ori_block_num=10,
        ori_block_size=128,
        ori_table_len=8,
        cmp_block_num=1,
        cmp_block_size=1,
        cmp_table_len=1,
        n_heads=64,
        n_kv_heads=1,
        head_dim=512,
        topk_cmp=0,
        cmp_ratio=4,
        scenario=1,
        ori_win_left=127,
        softmax_scale=0.04419417,
        dtype="bfloat16",
        core_num=DEFAULT_CORE_NUM,
    )
    print("TEST PASSED!")
