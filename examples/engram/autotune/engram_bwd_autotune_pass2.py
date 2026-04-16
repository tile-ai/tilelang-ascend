import torch
import tilelang as tl
import tilelang.language as T
import os

ALIGNMENT = 256
CONV_KERNEL_SIZE = 4

tl.cache.clear_cache()


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


def get_config():
    return [
        {"block_N": 8},
        {"block_N": 16},
        {"block_N": 32},
        {"block_N": 64},
        {"block_N": 128},
        {"block_N": 256},
        {"block_N": 512},
        {"block_N": 1024},
        {"block_N": 4096},
    ]


def ref_prog(
    dY: torch.Tensor,  # (M, seq_len, d_padded)  fp16
    rms_w_v: torch.Tensor,  # (d_padded,)             fp16
    conv_w: torch.Tensor,  # (KS, d_padded)          fp16
    vhat: torch.Tensor,  # (M, seq_len, d_padded)  fp16
    rrms_v: torch.Tensor,  # (M, seq_len)            fp32
    drms_w_v: torch.Tensor,  # (d_padded,)             fp32
    dvhat_buf: torch.Tensor,  # (M, seq_len, d_padded)  fp32  来自 pass1 输出
):
    """
    Returns:
        drms_w_v : (d_padded,)            fp32
        dvhat_out: (M, seq_len, d_padded) fp32  覆盖写 dvhat_buf
    """
    d = 512
    device = dY.device
    M, seq_len, d_padded = dY.shape
    KS = conv_w.shape[0]

    dY_f = dY.float()
    rms_w_v_f = rms_w_v.float()
    conv_w_f = conv_w.float()
    vhat_f = vhat.float()
    rrms_v_f = rrms_v.float()
    dvhat_f = dvhat_buf.float().clone()

    drms_w_v = torch.zeros((d_padded,), dtype=torch.float32, device=device)
    dvhat_out = torch.zeros((M, seq_len, d_padded), dtype=torch.float32, device=device)

    for bid in range(M):
        for tid in range(seq_len):
            # ---------- d_vnorm ----------
            d_vnorm = torch.zeros((d_padded,), dtype=torch.float32, device=device)
            for p in range(KS):
                dst_t = tid + (KS - 1) - p
                if 0 <= dst_t < seq_len:
                    d_si = dvhat_f[bid, dst_t]
                else:
                    d_si = torch.zeros((d_padded,), dtype=torch.float32, device=device)
                d_vnorm += conv_w_f[p] * d_si

            vhat_local = vhat_f[bid, tid]
            rrms_v_val = rrms_v_f[bid, tid].item()

            # ---------- drms_w_v ----------
            drms_w_v += d_vnorm * vhat_local * rrms_v_val

            # ---------- dot_sum ----------
            dot_sum = (vhat_local * rms_w_v_f * d_vnorm).sum().item()

            # ---------- dvhat_local ----------
            tmp_val = rrms_v_val**3 * dot_sum / d
            val_local = vhat_local * tmp_val
            val_1 = rms_w_v_f * d_vnorm * rrms_v_val
            dvhat_local = val_1 - val_local

            # ---------- + dY ----------
            dvhat_local = dvhat_local + dY_f[bid, tid]

            dvhat_out[bid, tid] = dvhat_local

    return dvhat_out


def supply_prog(params):

    dev = "npu"
    M = 1
    seq_len = 4096
    d = 512
    d_padded = _align_up(d, ALIGNMENT)
    KS = CONV_KERNEL_SIZE

    torch.manual_seed(0)

    dY = torch.randn((M, seq_len, d_padded), dtype=torch.float16, device=dev)
    rms_w_v = torch.randn((d_padded,), dtype=torch.float16, device=dev)
    conv_w = torch.randn((KS, d_padded), dtype=torch.float16, device=dev)
    vhat = torch.randn((M, seq_len, d_padded), dtype=torch.float16, device=dev)
    rrms_v = torch.rand((M, seq_len), dtype=torch.float32, device=dev) + 0.5

    drms_w_v = torch.zeros((d_padded,), dtype=torch.float32, device=dev)

    dvhat_in = torch.randn((M, seq_len, d_padded), dtype=torch.float32, device=dev)

    return [dY, rms_w_v, conv_w, vhat, rrms_v, drms_w_v, dvhat_in]


@tl.autotune(
    configs=get_config(),
    ref_prog=ref_prog,
    supply_prog=supply_prog,
    atol=1e-2,
    rtol=1e-2,
)
@tl.jit(
    out_idx=[7],
    target="npuir",
)
def engram_gate_conv_bwd_pass2(M, seq_len, d, block_N, dtype):
    accum_dtype = "float32"
    d_padded = _align_up(d, ALIGNMENT)
    KS = CONV_KERNEL_SIZE

    @T.prim_func
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

    return _bwd_pass2


os.environ["TILELANG_ASCEND_MODE"] = "Expert"
func = engram_gate_conv_bwd_pass2(1, 4096, 512, dtype="float16")
print("Best Config:", func.get_tuner_result())
print("Test passed!")
