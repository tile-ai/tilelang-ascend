import functools
import math
import torch
import tilelang as tl
import tilelang.language as T
import os

ALIGNMENT = 256
CONV_KERNEL_SIZE = 4


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


@functools.lru_cache(maxsize=32)
def engram_gate_conv_bwd_pass1(M, seq_len, d, dtype):
    accum_dtype = "float32"
    d_padded = _align_up(d, ALIGNMENT)
    KS = CONV_KERNEL_SIZE

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
            with T.Kernel(seq_len * M, is_npu=True) as (cid, _):
                bid = cid % M
                tid = cid // M

                dy_local = T.alloc_shared((d_padded,), accum_dtype)
                T.copy(dY[bid, tid, :], dy_local)
                conv_out = T.alloc_shared((d_padded,), accum_dtype)
                T.clear(conv_out)

                for p in T.serial(KS):
                    src_t = tid - (KS - 1) + p

                    raw_val = T.alloc_shared((d_padded,), accum_dtype)
                    T.clear(raw_val)
                    src_rrms = T.alloc_shared((1, 1), accum_dtype)
                    T.clear(src_rrms)

                    if src_t >= 0:
                        T.copy(vhat[bid, src_t, :], raw_val)
                        T.copy(rrms_v[bid : bid + 1, src_t : src_t + 1], src_rrms)
                    rms_w_v_shared = T.alloc_shared((d_padded,), accum_dtype)
                    T.copy(rms_w_v, rms_w_v_shared)
                    conv_w_shared = T.alloc_shared((1, d_padded), accum_dtype)
                    T.copy(conv_w[p : p + 1, 0:d_padded], conv_w_shared)
                    for j in T.Parallel(d_padded):
                        normed = raw_val[j] * src_rrms[0, 0] * rms_w_v_shared[j]
                        conv_out[j] += conv_w_shared[0, j] * normed
                sig = T.alloc_shared((d_padded,), accum_dtype)
                d_silu = T.alloc_shared((d_padded,), accum_dtype)
                T.vsigmoid(conv_out, sig)
                for j in T.Parallel(d_padded):
                    d_silu[j] = dy_local[j] * (
                        sig[j] + conv_out[j] * sig[j] * (1.0 - sig[j])
                    )

                for p in T.serial(KS):
                    src_t = tid - (KS - 1) + p
                    raw_val = T.alloc_shared((d_padded,), accum_dtype)
                    T.clear(raw_val)
                    src_rrms = T.alloc_shared((1, 1), accum_dtype)
                    T.clear(src_rrms)

                    if src_t >= 0:
                        T.copy(vhat[bid, src_t, :], raw_val)
                        T.copy(rrms_v[bid : bid + 1, src_t : src_t + 1], src_rrms)
                    rms_w_v_shared = T.alloc_shared((d_padded,), accum_dtype)
                    T.copy(rms_w_v, rms_w_v_shared)
                    tmp = T.alloc_shared((1, d_padded), accum_dtype)
                    for j in T.Parallel(d_padded):
                        normed = raw_val[j] * src_rrms[0, 0] * rms_w_v_shared[j]
                        tmp[0, j] = d_silu[j] * normed
                    T.atomic_add(dconv_w[p : p + 1, :], tmp)

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


@functools.lru_cache(maxsize=32)
def engram_gate_conv_bwd_pass2(
    M,
    seq_len,
    d,
    dtype,
):
    accum_dtype = "float32"
    d_padded = _align_up(d, ALIGNMENT)
    KS = CONV_KERNEL_SIZE

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
            dvhat_buf: T.Tensor((M, seq_len, d_padded), accum_dtype),
        ):
            with T.Kernel(seq_len * M, is_npu=True) as (cid, _):
                bid = cid % M
                tid = cid // M
                d_vnorm = T.alloc_shared((d_padded,), accum_dtype)
                T.clear(d_vnorm)
                for p in T.serial(KS):
                    dst_t = tid + (KS - 1) - p
                    d_si = T.alloc_shared((d_padded,), accum_dtype)
                    T.clear(d_si)
                    if (dst_t >= 0) * (dst_t < seq_len):
                        T.copy(dvhat_buf[bid, dst_t, :], d_si)
                    conv_w_shared = T.alloc_shared((d_padded,), accum_dtype)
                    T.copy(conv_w[p : p + 1, 0:d_padded], conv_w_shared)
                    for j in T.Parallel(d_padded):
                        d_vnorm[j] += conv_w_shared[j] * d_si[j]

                rrms_v_val = rrms_v[bid, tid]
                vhat_local = T.alloc_shared((d_padded,), accum_dtype)
                T.copy(vhat[bid, tid, :], vhat_local)

                add_val = T.alloc_shared((d_padded,), accum_dtype)
                T.vmul(d_vnorm, vhat_local, add_val)
                T.vmul(add_val, rrms_v_val, add_val)
                T.atomic_add(drms_w_v, add_val)

                dot_2d = T.alloc_shared((1, d_padded), accum_dtype)
                rms_w_v_shared = T.alloc_shared((d_padded,), accum_dtype)
                T.copy(rms_w_v, rms_w_v_shared)
                for j in T.Parallel(d_padded):
                    dot_2d[0, j] = vhat_local[j] * rms_w_v_shared[j] * d_vnorm[j]
                dot_sum = T.alloc_shared((1, 1), accum_dtype)
                T.reduce_sum(dot_2d, dot_sum, dim=1)

                dvhat_local = T.alloc_shared((d_padded,), accum_dtype)
                for j in T.serial(d_padded):
                    dvhat_local[j] = (
                        rrms_v_val * rms_w_v_shared[j] * d_vnorm[j]
                        - rrms_v_val
                        * rrms_v_val
                        * rrms_v_val
                        * vhat_local[j]
                        * dot_sum[0, 0]
                        / d
                    )
                dy_local = T.alloc_shared((d_padded,), accum_dtype)
                T.copy(dY[bid, tid, :], dy_local)
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
            dvhat_buf: T.Tensor((M, seq_len, d_padded), accum_dtype),
        ):
            _bwd_pass2(dY, rms_w_v, conv_w, vhat, rrms_v, drms_w_v, dvhat_buf)

        return pass2

    return _func2


@functools.lru_cache(maxsize=32)
def engram_gate_conv_bwd_pass3(M, seq_len, d, dtype):
    accum_dtype = "float32"
    d_padded = _align_up(d, ALIGNMENT)

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
            with T.Kernel(seq_len * M, is_npu=True) as (cid, _):
                bid = cid % M
                tid = cid // M
                dvhat_local = T.alloc_shared((1, d_padded), accum_dtype)
                T.copy(dvhat_buf[bid, tid, :], dvhat_local)
                v_local = T.alloc_shared((1, d_padded), accum_dtype)
                T.copy(v[bid, tid, :], v_local)
                h_local = T.alloc_shared((d_padded,), accum_dtype)
                T.copy(H[bid, tid, :], h_local)
                k_local = T.alloc_shared((d_padded,), accum_dtype)
                T.copy(k[bid, tid, :], k_local)

                alpha_val = alpha[bid, tid]
                rrms_h_val = rrms_h[bid, tid]
                rrms_k_val = rrms_k[bid, tid]

                dalpha_2d = T.alloc_shared((1, d_padded), accum_dtype)
                dv_local = T.alloc_shared((d_padded,), accum_dtype)
                for j in T.Parallel(d_padded):
                    dalpha_2d[0, j] = dvhat_local[0, j] * v_local[0, j]
                    dv_local[j] = alpha_val * dvhat_local[0, j]
                dalpha_sum = T.alloc_shared((1, 1), accum_dtype)
                T.reduce_sum(dalpha_2d, dalpha_sum, dim=1)

                sqrt_d = T.alloc_shared((1, 1), accum_dtype)
                sqrt_d[0, 0] = d
                T.vsqrt(sqrt_d, sqrt_d)

                ddot_val = (
                    dalpha_sum[0, 0] * alpha_val * (1.0 - alpha_val) / sqrt_d[0, 0]
                )

                h_norm_local = T.alloc_shared((d_padded,), accum_dtype)
                k_norm_local = T.alloc_shared((d_padded,), accum_dtype)
                rms_w_h_local = T.alloc_shared((d_padded,), accum_dtype)
                T.copy(rms_w_h, rms_w_h_local)
                for j in T.Parallel(d_padded):
                    h_norm_local[j] = h_local[j] * rrms_h_val * rms_w_h_local[j]
                    k_norm_local[j] = k_local[j] * rrms_k_val * rms_w_h_local[j]
                dot_h_2d = T.alloc_shared((1, d_padded), accum_dtype)
                for j in T.Parallel(d_padded):
                    dot_h_2d[0, j] = (
                        h_local[j] * rms_w_h_local[j] * ddot_val * k_norm_local[j]
                    )
                dot_h_sum = T.alloc_shared((1, 1), accum_dtype)
                T.reduce_sum(dot_h_2d, dot_h_sum, dim=1)

                dh_local = T.alloc_shared((d_padded,), accum_dtype)
                for j in T.serial(d_padded):
                    dh_local[j] = (
                        rrms_h_val * rms_w_h_local[j] * ddot_val * k_norm_local[j]
                        - rrms_h_val
                        * rrms_h_val
                        * rrms_h_val
                        * h_local[j]
                        * dot_h_sum[0, 0]
                        / d
                    )
                add_val = T.alloc_shared((d_padded,), accum_dtype)
                for j in T.Parallel(d_padded):
                    add_val[j] = ddot_val * k_norm_local[j] * h_local[j] * rrms_h_val
                T.atomic_add(drms_w_h, add_val)

                dot_k_2d = T.alloc_shared((1, d_padded), accum_dtype)
                for j in T.Parallel(d_padded):
                    dot_k_2d[0, j] = (
                        k_local[j] * rms_w_h_local[j] * ddot_val * h_norm_local[j]
                    )
                dot_k_sum = T.alloc_shared((1, 1), accum_dtype)
                T.reduce_sum(dot_k_2d, dot_k_sum, dim=1)

                dk_local = T.alloc_shared((d_padded,), accum_dtype)
                for j in T.serial(d_padded):
                    dk_local[j] = (
                        rrms_k_val * rms_w_h_local[j] * ddot_val * h_norm_local[j]
                        - rrms_k_val
                        * rrms_k_val
                        * rrms_k_val
                        * k_local[j]
                        * dot_k_sum[0, 0]
                        / d
                    )
                add_val_2 = T.alloc_shared((d_padded,), accum_dtype)
                for j in T.Parallel(d_padded):
                    add_val_2[j] = ddot_val * h_norm_local[j] * k_local[j] * rrms_k_val
                T.atomic_add(drms_w_h, add_val_2)

                T.copy(dh_local, dH[bid, tid, :])
                T.copy(dk_local, dk[bid, tid, :])
                T.copy(dv_local, dv[bid, tid, :])

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
    eps: float,
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
    dvhat_buf = torch.zeros((M, seq_len, d_padded), dtype=torch.float32).npu()

    engram_gate_conv_bwd_pass1(M, seq_len, d, dtype_str)()(
        dY, rms_w_v, conv_w, vhat, rrms_v, dconv_w, dvhat_buf
    )

    engram_gate_conv_bwd_pass2(M, seq_len, d, dtype_str)()(
        dY, rms_w_v, conv_w, vhat, rrms_v, drms_w_v, dvhat_buf
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
    M = 2
    seq_len = 8
    d = 64
    eps = 1e-6
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
        eps,
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
    os.environ["TILELANG_ASCEND_MODE"] = "Dev"
    run_test()
