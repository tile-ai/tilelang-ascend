import math
import torch
import tilelang as tl
import tilelang.language as T
import os

ALIGNMENT = 256
CONV_KERNEL_SIZE = 4


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


def engram_gate_conv_bwd_pass1(M, seq_len, d, dtype):

    accum_dtype = "float32"
    d_padded = _align_up(d, ALIGNMENT)
    KS = CONV_KERNEL_SIZE
    block_N = 32

    @tl.jit(
        target="npuir",
    )
    def _func1():
        @T.macro
        def _bwd_pass1(
            dY: T.Tensor((M, seq_len, d_padded), dtype),
            rms_w_v: T.Tensor((d_padded,), dtype),
            conv_w: T.Tensor((KS, d_padded), dtype),
            vhat: T.Tensor((M, seq_len, d_padded), dtype),
            rrms_v: T.Tensor((M, seq_len), accum_dtype),
            dconv_w: T.Tensor((KS, d_padded), accum_dtype),
            dvhat_buf: T.Tensor((M, seq_len, d_padded), accum_dtype),
        ):

            # ---------------------------------
            # Pass 1: SiLU bwd + dconv_w
            # grid flatten: M * seq_len
            # ---------------------------------
            with T.Kernel(T.ceildiv(seq_len, block_N) * M, is_npu=True) as (cid, _):
                bid = cid % M
                tile_id = cid // M
                tid_base = tile_id * block_N

                rms_w_v_cast = T.alloc_ub((d_padded,), dtype)
                rms_w_v_shared = T.alloc_ub((d_padded,), accum_dtype)
                T.copy(rms_w_v, rms_w_v_cast)
                T.vcast(rms_w_v_cast, rms_w_v_shared)

                dy_local_cast = T.alloc_ub((1, d_padded), dtype)
                dy_local = T.alloc_ub((1, d_padded), accum_dtype)
                raw_val_cast = T.alloc_ub((1, d_padded), dtype)
                raw_val = T.alloc_ub((1, d_padded), accum_dtype)
                conv_w_cast = T.alloc_ub((1, d_padded), dtype)
                conv_w_local = T.alloc_ub((1, d_padded), accum_dtype)
                src_rrms = T.alloc_ub((1, 1), accum_dtype)
                conv_out = T.alloc_ub((1, d_padded), accum_dtype)
                sig = T.alloc_ub((1, d_padded), accum_dtype)
                d_silu = T.alloc_ub((1, d_padded), accum_dtype)
                tmp = T.alloc_ub((1, d_padded), accum_dtype)

                for ti in T.serial(block_N):
                    tid = tid_base + ti

                    if tid < seq_len:
                        T.copy(dY[bid, tid, :], dy_local_cast)
                        T.vcast(dy_local_cast, dy_local)

                        T.clear(conv_out)

                        for p in T.serial(KS):
                            src_t = tid - (KS - 1) + p

                            T.clear(raw_val_cast)
                            T.clear(raw_val)
                            T.clear(src_rrms)

                            if src_t >= 0:
                                T.copy(vhat[bid, src_t, :], raw_val_cast)
                                T.copy(rrms_v[bid, src_t], src_rrms)

                            T.vcast(raw_val_cast, raw_val)

                            T.copy(conv_w[p : p + 1, 0:d_padded], conv_w_cast)
                            T.vcast(conv_w_cast, conv_w_local)

                            for j in T.Parallel(d_padded):
                                conv_out[0, j] += (
                                    conv_w_local[0, j]
                                    * raw_val[0, j]
                                    * src_rrms[0, 0]
                                    * rms_w_v_shared[j]
                                )

                        T.clear(sig)
                        T.clear(d_silu)
                        T.vsigmoid(conv_out, sig)

                        for j in T.Parallel(d_padded):
                            d_silu[0, j] = dy_local[0, j] * (
                                sig[0, j]
                                + conv_out[0, j] * sig[0, j] * (1.0 - sig[0, j])
                            )

                        for p in T.serial(KS):
                            src_t = tid - (KS - 1) + p

                            T.clear(raw_val_cast)
                            T.clear(raw_val)
                            T.clear(src_rrms)
                            T.clear(tmp)

                            if src_t >= 0:
                                T.copy(vhat[bid, src_t, :], raw_val_cast)
                                T.copy(
                                    rrms_v[bid : bid + 1, src_t : src_t + 1], src_rrms
                                )

                            T.vcast(raw_val_cast, raw_val)

                            for j in T.Parallel(d_padded):
                                tmp[0, j] = (
                                    d_silu[0, j]
                                    * raw_val[0, j]
                                    * src_rrms[0, 0]
                                    * rms_w_v_shared[j]
                                )
                            T.atomic_add(dconv_w[p, :], tmp)

                        T.copy(d_silu, dvhat_buf[bid, tid : tid + 1, :])

        @T.prim_func
        def pass1(
            dY: T.Tensor((M, seq_len, d_padded), dtype),
            rms_w_v: T.Tensor((d_padded,), dtype),
            conv_w: T.Tensor((KS, d_padded), dtype),
            vhat: T.Tensor((M, seq_len, d_padded), dtype),
            rrms_v: T.Tensor((M, seq_len), accum_dtype),
            dconv_w: T.Tensor((KS, d_padded), accum_dtype),
            dvhat_buf: T.Tensor((M, seq_len, d_padded), accum_dtype),
        ):
            _bwd_pass1(dY, rms_w_v, conv_w, vhat, rrms_v, dconv_w, dvhat_buf)

        return pass1

    return _func1


def engram_gate_conv_bwd_pass2(
    M,
    seq_len,
    d,
    dtype,
):
    accum_dtype = "float32"
    d_padded = _align_up(d, ALIGNMENT)
    KS = CONV_KERNEL_SIZE
    block_N = 16

    @tl.jit(
        target="npuir",
    )
    def _func2():
        @T.macro
        def _bwd_pass2(
            dY: T.Tensor((M, seq_len, d_padded), dtype),
            rms_w_v: T.Tensor((d_padded,), dtype),
            conv_w: T.Tensor((KS, d_padded), dtype),
            vhat: T.Tensor((M, seq_len, d_padded), dtype),
            rrms_v: T.Tensor((M, seq_len), accum_dtype),
            drms_w_v: T.Tensor((d_padded,), accum_dtype),
            dvhat_in: T.Tensor((M, seq_len, d_padded), accum_dtype),
            dvhat_buf: T.Tensor((M, seq_len, d_padded), accum_dtype),
        ):
            with T.Kernel(T.ceildiv(seq_len, block_N) * M, is_npu=True) as (cid, _):
                bid = cid % M
                tile_id = cid // M
                tid_base = tile_id * block_N

                rms_w_v_cast = T.alloc_ub((d_padded,), dtype)
                rms_w_v_shared = T.alloc_ub((d_padded,), accum_dtype)
                T.copy(rms_w_v, rms_w_v_cast)
                T.vcast(rms_w_v_cast, rms_w_v_shared)

                d_vnorm = T.alloc_ub((d_padded,), accum_dtype)
                d_si = T.alloc_ub((d_padded,), accum_dtype)
                conv_w_cast = T.alloc_ub((d_padded,), dtype)
                conv_w_shared = T.alloc_ub((d_padded,), accum_dtype)
                vhat_cast = T.alloc_ub((d_padded,), dtype)
                vhat_local = T.alloc_ub((d_padded,), accum_dtype)
                add_val = T.alloc_ub((d_padded,), accum_dtype)
                dot_2d = T.alloc_ub((d_padded,), accum_dtype)
                dot_sum = T.alloc_ub((1,), accum_dtype)
                dvhat_local = T.alloc_ub((d_padded,), accum_dtype)
                val_local = T.alloc_ub((d_padded,), accum_dtype)
                val_1 = T.alloc_ub((d_padded,), accum_dtype)
                dy_local_cast = T.alloc_ub((d_padded,), dtype)
                dy_local = T.alloc_ub((d_padded,), accum_dtype)

                for ti in T.serial(block_N):
                    tid = tid_base + ti

                    if tid < seq_len:
                        T.clear(d_vnorm)

                        for p in T.serial(KS):
                            dst_t = tid + (KS - 1) - p
                            T.clear(d_si)

                            if (dst_t >= 0) * (dst_t < seq_len):
                                T.copy(dvhat_in[bid, dst_t, :], d_si)

                            T.copy(conv_w[p, 0:d_padded], conv_w_cast)
                            T.vcast(conv_w_cast, conv_w_shared)

                            for j in T.Parallel(d_padded):
                                d_vnorm[j] += conv_w_shared[j] * d_si[j]

                        T.copy(vhat[bid, tid, :], vhat_cast)
                        T.vcast(vhat_cast, vhat_local)

                        rrms_v_val = rrms_v[bid, tid]

                        T.vmul(d_vnorm, vhat_local, add_val)
                        T.vmul(add_val, rrms_v_val, add_val)
                        T.atomic_add(drms_w_v, add_val)

                        for j in T.Parallel(d_padded):
                            dot_2d[j] = vhat_local[j] * rms_w_v_shared[j] * d_vnorm[j]
                        T.reduce_sum(dot_2d, dot_sum, dim=0)

                        tmp_val = rrms_v_val * rrms_v_val * rrms_v_val * dot_sum[0] / d

                        T.vmul(vhat_local, tmp_val, val_local)
                        T.vmul(rms_w_v_shared, d_vnorm, val_1)
                        T.vmul(val_1, rrms_v_val, val_1)

                        for j in T.Parallel(d_padded):
                            dvhat_local[j] = val_1[j] - val_local[j]

                        T.copy(dY[bid, tid, :], dy_local_cast)
                        T.vcast(dy_local_cast, dy_local)

                        for j in T.Parallel(d_padded):
                            dvhat_local[j] += dy_local[j]

                        T.copy(dvhat_local, dvhat_buf[bid, tid, :])

        @T.prim_func
        def pass2(
            dY: T.Tensor((M, seq_len, d_padded), dtype),
            rms_w_v: T.Tensor((d_padded,), dtype),
            conv_w: T.Tensor((KS, d_padded), dtype),
            vhat: T.Tensor((M, seq_len, d_padded), dtype),
            rrms_v: T.Tensor((M, seq_len), accum_dtype),
            drms_w_v: T.Tensor((d_padded,), accum_dtype),
            dvhat_in: T.Tensor((M, seq_len, d_padded), accum_dtype),
            dvhat_buf: T.Tensor((M, seq_len, d_padded), accum_dtype),
        ):
            _bwd_pass2(dY, rms_w_v, conv_w, vhat, rrms_v, drms_w_v, dvhat_in, dvhat_buf)

        return pass2

    return _func2


def engram_gate_conv_bwd_pass3(M, seq_len, d, dtype):
    accum_dtype = "float32"
    d_padded = _align_up(d, ALIGNMENT)
    block_N = 8

    @tl.jit(
        target="npuir",
    )
    def _func3():
        @T.macro
        def _bwd_pass3(
            H: T.Tensor((M, seq_len, d_padded), dtype),
            k: T.Tensor((M, seq_len, d_padded), dtype),
            v: T.Tensor((M, seq_len, d_padded), dtype),
            rms_w_h: T.Tensor((d_padded,), dtype),
            alpha: T.Tensor((M, seq_len), accum_dtype),
            rrms_h: T.Tensor((M, seq_len), accum_dtype),
            rrms_k: T.Tensor((M, seq_len), accum_dtype),
            dH: T.Tensor((M, seq_len, d_padded), dtype),
            dk: T.Tensor((M, seq_len, d_padded), dtype),
            dv: T.Tensor((M, seq_len, d_padded), dtype),
            drms_w_h: T.Tensor((d_padded,), accum_dtype),
            dvhat_buf: T.Tensor((M, seq_len, d_padded), accum_dtype),
        ):

            with T.Kernel(T.ceildiv(seq_len, block_N) * M, is_npu=True) as (cid, _):
                bid = cid % M
                tile_id = cid // M
                tid_base = tile_id * block_N

                rms_w_h_cast = T.alloc_ub((d_padded,), dtype)
                rms_w_h_local = T.alloc_ub((d_padded,), accum_dtype)
                T.copy(rms_w_h, rms_w_h_cast)
                T.vcast(rms_w_h_cast, rms_w_h_local)

                sqrt_d = T.alloc_ub((1, 1), accum_dtype)
                sqrt_d[0, 0] = d
                T.vsqrt(sqrt_d, sqrt_d)
                drms_w_h_local = T.alloc_ub((d_padded,), accum_dtype)
                T.clear(drms_w_h_local)

                dvhat_local = T.alloc_ub((d_padded,), accum_dtype)
                v_local_cast = T.alloc_ub((d_padded,), dtype)
                v_local = T.alloc_ub((d_padded,), accum_dtype)
                h_local_cast = T.alloc_ub((d_padded,), dtype)
                h_local = T.alloc_ub((d_padded,), accum_dtype)
                k_local_cast = T.alloc_ub((d_padded,), dtype)
                k_local = T.alloc_ub((d_padded,), accum_dtype)
                dalpha_2d = T.alloc_ub((d_padded,), accum_dtype)
                dv_local = T.alloc_ub((d_padded,), accum_dtype)
                dalpha_sum = T.alloc_ub((1,), accum_dtype)
                h_norm_local = T.alloc_ub((d_padded,), accum_dtype)
                k_norm_local = T.alloc_ub((d_padded,), accum_dtype)
                dot_h_2d = T.alloc_ub((d_padded,), accum_dtype)
                dot_h_sum = T.alloc_ub((1,), accum_dtype)
                dh_local = T.alloc_ub((d_padded,), accum_dtype)
                dot_k_2d = T.alloc_ub((d_padded,), accum_dtype)
                dot_k_sum = T.alloc_ub((1,), accum_dtype)
                dk_local = T.alloc_ub((d_padded,), accum_dtype)
                dh_val = T.alloc_ub((d_padded,), dtype)
                dk_val = T.alloc_ub((d_padded,), dtype)
                dv_val = T.alloc_ub((d_padded,), dtype)

                for ti in T.serial(block_N):
                    tid = tid_base + ti

                    if tid < seq_len:
                        T.copy(dvhat_buf[bid, tid, :], dvhat_local)
                        T.copy(v[bid, tid, :], v_local_cast)
                        T.vcast(v_local_cast, v_local)
                        T.copy(H[bid, tid, :], h_local_cast)
                        T.vcast(h_local_cast, h_local)
                        T.copy(k[bid, tid, :], k_local_cast)
                        T.vcast(k_local_cast, k_local)

                        alpha_val = alpha[bid, tid]
                        rrms_h_val = rrms_h[bid, tid]
                        rrms_k_val = rrms_k[bid, tid]

                        # ---------- dalpha / dv ----------
                        for j in T.Parallel(d_padded):
                            dalpha_2d[j] = dvhat_local[j] * v_local[j]
                        for j in T.Parallel(d_padded):
                            dv_local[j] = alpha_val * dvhat_local[j]
                        T.reduce_sum(dalpha_2d, dalpha_sum, dim=0)

                        ddot_val = (
                            dalpha_sum[0] * alpha_val * (1.0 - alpha_val) / sqrt_d[0, 0]
                        )

                        # ---------- h_norm / k_norm ----------
                        for j in T.Parallel(d_padded):
                            h_norm_local[j] = h_local[j] * rrms_h_val * rms_w_h_local[j]
                        for j in T.Parallel(d_padded):
                            k_norm_local[j] = k_local[j] * rrms_k_val * rms_w_h_local[j]

                        # ---------- dh ----------
                        for j in T.Parallel(d_padded):
                            dot_h_2d[j] = (
                                h_local[j]
                                * rms_w_h_local[j]
                                * ddot_val
                                * k_norm_local[j]
                            )
                        T.reduce_sum(dot_h_2d, dot_h_sum, dim=0)

                        tmp_h_val = (
                            rrms_h_val * rrms_h_val * rrms_h_val * dot_h_sum[0] / d
                        )
                        for j in T.Parallel(d_padded):
                            dh_local[j] = (
                                rrms_h_val
                                * rms_w_h_local[j]
                                * ddot_val
                                * k_norm_local[j]
                                - tmp_h_val * h_local[j]
                            )

                        # ---------- dk ----------
                        for j in T.Parallel(d_padded):
                            dot_k_2d[j] = (
                                k_local[j]
                                * rms_w_h_local[j]
                                * ddot_val
                                * h_norm_local[j]
                            )
                        T.reduce_sum(dot_k_2d, dot_k_sum, dim=0)

                        tmp_k_val = (
                            rrms_k_val * rrms_k_val * rrms_k_val * dot_k_sum[0] / d
                        )
                        for j in T.Parallel(d_padded):
                            dk_local[j] = (
                                rrms_k_val
                                * rms_w_h_local[j]
                                * ddot_val
                                * h_norm_local[j]
                                - tmp_k_val * k_local[j]
                            )

                        for j in T.Parallel(d_padded):
                            drms_w_h_local[j] += (
                                ddot_val * k_norm_local[j] * h_local[j] * rrms_h_val
                                + ddot_val * h_norm_local[j] * k_local[j] * rrms_k_val
                            )

                        T.vcast(dh_local, dh_val)
                        T.vcast(dk_local, dk_val)
                        T.vcast(dv_local, dv_val)
                        T.copy(dh_val, dH[bid, tid, :])
                        T.copy(dk_val, dk[bid, tid, :])
                        T.copy(dv_val, dv[bid, tid, :])

                T.atomic_add(drms_w_h, drms_w_h_local)

        @T.prim_func
        def pass3(
            H: T.Tensor((M, seq_len, d_padded), dtype),
            k: T.Tensor((M, seq_len, d_padded), dtype),
            v: T.Tensor((M, seq_len, d_padded), dtype),
            rms_w_h: T.Tensor((d_padded,), dtype),
            alpha: T.Tensor((M, seq_len), accum_dtype),
            rrms_h: T.Tensor((M, seq_len), accum_dtype),
            rrms_k: T.Tensor((M, seq_len), accum_dtype),
            dH: T.Tensor((M, seq_len, d_padded), dtype),
            dk: T.Tensor((M, seq_len, d_padded), dtype),
            dv: T.Tensor((M, seq_len, d_padded), dtype),
            drms_w_h: T.Tensor((d_padded,), accum_dtype),
            dvhat_buf: T.Tensor((M, seq_len, d_padded), accum_dtype),
        ):
            _bwd_pass3(
                H, k, v, rms_w_h, alpha, rrms_h, rrms_k, dH, dk, dv, drms_w_h, dvhat_buf
            )

        return pass3

    return _func3


def _engram_gate_conv_bwd_wrapped(
    M: int,
    seq_len: int,
    d: int,
    dtype_str: str,
    dY: torch.Tensor,
    H: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rms_w_h: torch.Tensor,
    rms_w_v: torch.Tensor,
    conv_w: torch.Tensor,
    vhat: torch.Tensor,
    alpha: torch.Tensor,
    rrms_h: torch.Tensor,
    rrms_k: torch.Tensor,
    rrms_v: torch.Tensor,
) -> list[torch.Tensor]:
    d_padded = _align_up(d, ALIGNMENT)
    KS = CONV_KERNEL_SIZE
    dH = torch.zeros((M, seq_len, d_padded), dtype=eval("torch." + dtype_str)).npu()
    dk = torch.zeros((M, seq_len, d_padded), dtype=eval("torch." + dtype_str)).npu()
    dv = torch.zeros((M, seq_len, d_padded), dtype=eval("torch." + dtype_str)).npu()

    drms_w_h = torch.zeros((d_padded,), dtype=torch.float32).npu()
    drms_w_v = torch.zeros((d_padded,), dtype=torch.float32).npu()
    dconv_w = torch.zeros((KS, d_padded), dtype=torch.float32).npu()
    dvhat_buf_1 = torch.zeros((M, seq_len, d_padded), dtype=torch.float32).npu()
    dvhat_buf = torch.zeros((M, seq_len, d_padded), dtype=torch.float32).npu()

    engram_gate_conv_bwd_pass1(M, seq_len, d, dtype_str)()(
        dY, rms_w_v, conv_w, vhat, rrms_v, dconv_w, dvhat_buf_1
    )

    engram_gate_conv_bwd_pass2(M, seq_len, d, dtype_str)()(
        dY, rms_w_v, conv_w, vhat, rrms_v, drms_w_v, dvhat_buf_1, dvhat_buf
    )

    engram_gate_conv_bwd_pass3(M, seq_len, d, dtype_str)()(
        H, k, v, rms_w_h, alpha, rrms_h, rrms_k, dH, dk, dv, drms_w_h, dvhat_buf
    )
    results = [dH, dk, dv, drms_w_h, drms_w_v, dconv_w, dvhat_buf]
    return results


def ref_engram_gate_conv_bwd(
    dY,
    H,
    k,
    v,
    rms_w_h,
    rms_w_v,
    conv_w,
    vhat,
    alpha,
    rrms_h,
    rrms_k,
    rrms_v,
    d,
):
    device = dY.device
    out_dtype = dY.dtype

    M, seq_len, d_padded = dY.shape
    KS = conv_w.shape[0]

    dY_f = dY.float()
    H_f = H.float()
    k_f = k.float()
    v_f = v.float()
    rms_w_h_f = rms_w_h.float()
    rms_w_v_f = rms_w_v.float()
    conv_w_f = conv_w.float()
    vhat_f = vhat.float()
    alpha_f = alpha.float()
    rrms_h_f = rrms_h.float()
    rrms_k_f = rrms_k.float()
    rrms_v_f = rrms_v.float()

    # pass1
    dconv_w = torch.zeros((KS, d_padded), dtype=torch.float32, device=device)
    dvhat_buf_pass1 = torch.zeros(
        (M, seq_len, d_padded), dtype=torch.float32, device=device
    )

    for tid in range(seq_len):
        for bid in range(M):
            dy_local = dY_f[bid, tid]
            conv_out = torch.zeros((d_padded,), dtype=torch.float32, device=device)

            for p in range(KS):
                src_t = tid - (KS - 1) + p
                if src_t >= 0:
                    raw_val = vhat_f[bid, src_t]
                    src_rr = rrms_v_f[bid, src_t]
                else:
                    raw_val = torch.zeros(
                        (d_padded,), dtype=torch.float32, device=device
                    )
                    src_rr = 0.0

                normed = raw_val * src_rr * rms_w_v_f
                conv_out += conv_w_f[p] * normed

            sig = torch.sigmoid(conv_out)
            d_silu = dy_local * (sig + conv_out * sig * (1.0 - sig))

            for p in range(KS):
                src_t = tid - (KS - 1) + p
                if src_t >= 0:
                    raw_val = vhat_f[bid, src_t]
                    src_rr = rrms_v_f[bid, src_t]
                else:
                    raw_val = torch.zeros(
                        (d_padded,), dtype=torch.float32, device=device
                    )
                    src_rr = 0.0

                normed = raw_val * src_rr * rms_w_v_f
                dconv_w[p] += d_silu * normed

            dvhat_buf_pass1[bid, tid] = d_silu

    # pass2
    drms_w_v = torch.zeros((d_padded,), dtype=torch.float32, device=device)
    dvhat_buf = torch.empty_like(dvhat_buf_pass1)

    for tid in range(seq_len):
        for bid in range(M):
            d_vnorm = torch.zeros((d_padded,), dtype=torch.float32, device=device)

            for p in range(KS):
                dst_t = tid + (KS - 1) - p
                if 0 <= dst_t < seq_len:
                    d_si = dvhat_buf_pass1[bid, dst_t]
                else:
                    d_si = torch.zeros((d_padded,), dtype=torch.float32, device=device)
                d_vnorm += conv_w_f[p] * d_si

            rr_v = rrms_v_f[bid, tid]
            vhat_local = vhat_f[bid, tid]

            drms_w_v += d_vnorm * vhat_local * rr_v

            dot_sum = torch.sum(vhat_local * rms_w_v_f * d_vnorm)

            dvhat_local = (
                rr_v * rms_w_v_f * d_vnorm
                - rr_v * rr_v * rr_v * vhat_local * dot_sum / float(d)
            )
            dvhat_local += dY_f[bid, tid]

            dvhat_buf[bid, tid] = dvhat_local

    # pass3
    dH = torch.zeros((M, seq_len, d_padded), dtype=torch.float32, device=device)
    dk = torch.zeros((M, seq_len, d_padded), dtype=torch.float32, device=device)
    dv = torch.zeros((M, seq_len, d_padded), dtype=torch.float32, device=device)
    drms_w_h = torch.zeros((d_padded,), dtype=torch.float32, device=device)

    sqrt_d = math.sqrt(float(d))

    for tid in range(seq_len):
        for bid in range(M):
            dvhat_local = dvhat_buf[bid, tid]
            v_local = v_f[bid, tid]
            h_local = H_f[bid, tid]
            k_local = k_f[bid, tid]

            alpha_val = alpha_f[bid, tid]
            rr_h = rrms_h_f[bid, tid]
            rr_k = rrms_k_f[bid, tid]

            dalpha = torch.sum(dvhat_local * v_local)
            dv_local = alpha_val * dvhat_local
            ddot = dalpha * alpha_val * (1.0 - alpha_val) / sqrt_d

            h_norm_local = h_local * rr_h * rms_w_h_f
            k_norm_local = k_local * rr_k * rms_w_h_f

            dot_h_sum = torch.sum(h_local * rms_w_h_f * ddot * k_norm_local)
            dh_local = (
                rr_h * rms_w_h_f * ddot * k_norm_local
                - rr_h * rr_h * rr_h * h_local * dot_h_sum / float(d)
            )
            drms_w_h += ddot * k_norm_local * h_local * rr_h

            dot_k_sum = torch.sum(k_local * rms_w_h_f * ddot * h_norm_local)
            dk_local = (
                rr_k * rms_w_h_f * ddot * h_norm_local
                - rr_k * rr_k * rr_k * k_local * dot_k_sum / float(d)
            )
            drms_w_h += ddot * h_norm_local * k_local * rr_k

            dH[bid, tid] = dh_local
            dk[bid, tid] = dk_local
            dv[bid, tid] = dv_local

    return (
        dH.to(out_dtype),
        dk.to(out_dtype),
        dv.to(out_dtype),
        drms_w_h,
        drms_w_v,
        dconv_w,
        dvhat_buf,
    )


def run_test():
    torch.manual_seed(0)

    # fixed config
    M = 1
    seq_len = 4096
    d = 512
    dtype_str = "float16"

    d_padded = _align_up(d, ALIGNMENT)
    KS = CONV_KERNEL_SIZE
    dev = "npu"

    # inputs
    dY = torch.randn((M, seq_len, d_padded), dtype=torch.float16, device=dev)
    H = torch.randn((M, seq_len, d_padded), dtype=torch.float16, device=dev)
    k = torch.randn((M, seq_len, d_padded), dtype=torch.float16, device=dev)
    v = torch.randn((M, seq_len, d_padded), dtype=torch.float16, device=dev)

    rms_w_h = torch.randn((d_padded,), dtype=torch.float16, device=dev)
    rms_w_v = torch.randn((d_padded,), dtype=torch.float16, device=dev)
    conv_w = torch.randn((KS, d_padded), dtype=torch.float16, device=dev)
    vhat = torch.randn((M, seq_len, d_padded), dtype=torch.float16, device=dev)

    alpha = torch.sigmoid(torch.randn((M, seq_len), dtype=torch.float32, device=dev))
    rrms_h = torch.rand((M, seq_len), dtype=torch.float32, device=dev) + 0.5
    rrms_k = torch.rand((M, seq_len), dtype=torch.float32, device=dev) + 0.5
    rrms_v = torch.rand((M, seq_len), dtype=torch.float32, device=dev) + 0.5

    # run kernel wrapper
    out = _engram_gate_conv_bwd_wrapped(
        M,
        seq_len,
        d,
        dtype_str,
        dY,
        H,
        k,
        v,
        rms_w_h,
        rms_w_v,
        conv_w,
        vhat,
        alpha,
        rrms_h,
        rrms_k,
        rrms_v,
    )

    dH, dk, dv, drms_w_h, drms_w_v, dconv_w, dvhat_buf = out
    print("kernel finished!")

    ref_out = ref_engram_gate_conv_bwd(
        dY=dY,
        H=H,
        k=k,
        v=v,
        rms_w_h=rms_w_h,
        rms_w_v=rms_w_v,
        conv_w=conv_w,
        vhat=vhat,
        alpha=alpha,
        rrms_h=rrms_h,
        rrms_k=rrms_k,
        rrms_v=rrms_v,
        d=d,
    )

    for out_tensor, ref_tensor in zip(out, ref_out, strict=True):
        torch.testing.assert_close(
            out_tensor.float().cpu(),
            ref_tensor.float().cpu(),
            rtol=1e-2,
            atol=1e-2,
        )
    print("All check passed!")


if __name__ == "__main__":
    os.environ["TILELANG_ASCEND_MODE"] = "Expert"
    run_test()
