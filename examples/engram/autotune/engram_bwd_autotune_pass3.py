import math
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
    H: torch.Tensor,  # (M, seq_len, d_padded)  fp16
    k: torch.Tensor,  # (M, seq_len, d_padded)  fp16
    v: torch.Tensor,  # (M, seq_len, d_padded)  fp16
    rms_w_h: torch.Tensor,  # (d_padded,)             fp16
    alpha: torch.Tensor,  # (M, seq_len)            fp32
    rrms_h: torch.Tensor,  # (M, seq_len)            fp32
    rrms_k: torch.Tensor,  # (M, seq_len)            fp32
    drms_w_h: torch.Tensor,  # (d_padded,)            fp32
    dvhat_buf: torch.Tensor,  # (M, seq_len, d_padded)  fp32  来自 pass2 输出
):
    """
    Returns:
        dH        : (M, seq_len, d_padded)  fp16
        dk        : (M, seq_len, d_padded)  fp16
        dv        : (M, seq_len, d_padded)  fp16
        drms_w_h  : (d_padded,)             fp32
    """
    d = 512
    device = H.device
    M, seq_len, d_padded = H.shape

    H_f = H.float()
    k_f = k.float()
    v_f = v.float()
    rms_w_h_f = rms_w_h.float()
    alpha_f = alpha.float()
    rrms_h_f = rrms_h.float()
    rrms_k_f = rrms_k.float()
    dvhat_f = dvhat_buf.float()

    sqrt_d = math.sqrt(d)

    dH_out = torch.zeros((M, seq_len, d_padded), dtype=torch.float32, device=device)
    dk_out = torch.zeros((M, seq_len, d_padded), dtype=torch.float32, device=device)
    dv_out = torch.zeros((M, seq_len, d_padded), dtype=torch.float32, device=device)

    for bid in range(M):
        for tid in range(seq_len):
            dvhat_local = dvhat_f[bid, tid]
            v_local = v_f[bid, tid]
            h_local = H_f[bid, tid]
            k_local = k_f[bid, tid]

            alpha_val = alpha_f[bid, tid].item()
            rrms_h_val = rrms_h_f[bid, tid].item()
            rrms_k_val = rrms_k_f[bid, tid].item()

            # ---------- dalpha / dv ----------
            dalpha_sum = (dvhat_local * v_local).sum().item()
            dv_local = alpha_val * dvhat_local

            ddot_val = dalpha_sum * alpha_val * (1.0 - alpha_val) / sqrt_d

            # ---------- h_norm / k_norm ----------
            h_norm_local = h_local * rrms_h_val * rms_w_h_f
            k_norm_local = k_local * rrms_k_val * rms_w_h_f

            # ---------- dh ----------
            dot_h_sum = (h_local * rms_w_h_f * ddot_val * k_norm_local).sum().item()
            tmp_h_val = rrms_h_val**3 * dot_h_sum / d
            dh_local = (
                rrms_h_val * rms_w_h_f * ddot_val * k_norm_local - tmp_h_val * h_local
            )

            # ---------- drms_w_h 累加（h 部分）----------
            drms_w_h += ddot_val * k_norm_local * h_local * rrms_h_val

            # ---------- dk ----------
            dot_k_sum = (k_local * rms_w_h_f * ddot_val * h_norm_local).sum().item()
            tmp_k_val = rrms_k_val**3 * dot_k_sum / d
            dk_local = (
                rrms_k_val * rms_w_h_f * ddot_val * h_norm_local - tmp_k_val * k_local
            )

            # ---------- drms_w_h 累加（k 部分）----------
            drms_w_h += ddot_val * h_norm_local * k_local * rrms_k_val

            dH_out[bid, tid] = dh_local
            dk_out[bid, tid] = dk_local
            dv_out[bid, tid] = dv_local

    return [
        dH_out.half(),
        dk_out.half(),
        dv_out.half(),
    ]


def supply_prog(params):
    """
    返回 tensor 顺序需与 kernel 参数顺序一致：
      H, k, v, rms_w_h, alpha, rrms_h, rrms_k, drms_w_h, dvhat_buf
    out_idx=[9,10,11] 的 dH, dk, dv 由框架自动分配，不需要传入
    """
    dev = "npu"
    M = 1
    seq_len = 4096
    d = 512
    d_padded = _align_up(d, ALIGNMENT)

    torch.manual_seed(0)

    H = torch.randn((M, seq_len, d_padded), dtype=torch.float16, device=dev)
    k = torch.randn((M, seq_len, d_padded), dtype=torch.float16, device=dev)
    v = torch.randn((M, seq_len, d_padded), dtype=torch.float16, device=dev)
    rms_w_h = torch.randn((d_padded,), dtype=torch.float16, device=dev)
    alpha = torch.sigmoid(torch.randn((M, seq_len), dtype=torch.float32, device=dev))
    rrms_h = torch.rand((M, seq_len), dtype=torch.float32, device=dev) + 0.5
    rrms_k = torch.rand((M, seq_len), dtype=torch.float32, device=dev) + 0.5

    # atomic_add 累加目标，必须预清零
    drms_w_h = torch.zeros((d_padded,), dtype=torch.float32, device=dev)

    # 模拟 pass2 的输出作为 pass3 的只读输入
    dvhat_buf = torch.randn((M, seq_len, d_padded), dtype=torch.float32, device=dev)

    return [H, k, v, rms_w_h, alpha, rrms_h, rrms_k, drms_w_h, dvhat_buf]


@tl.autotune(
    configs=get_config(),
    ref_prog=ref_prog,
    supply_prog=supply_prog,
    atol=1e-2,
    rtol=1e-2,
)
@tl.jit(
    out_idx=[9, 10, 11],
    target="npuir",
)
def engram_gate_conv_bwd_pass3(M, seq_len, d, block_N, dtype):
    accum_dtype = "float32"
    d_padded = _align_up(d, ALIGNMENT)

    @T.prim_func
    def _bwd_pass3(
        H: T.Tensor((M, seq_len, d_padded), dtype),
        k: T.Tensor((M, seq_len, d_padded), dtype),
        v: T.Tensor((M, seq_len, d_padded), dtype),
        rms_w_h: T.Tensor((d_padded,), dtype),
        alpha: T.Tensor((M, seq_len), accum_dtype),
        rrms_h: T.Tensor((M, seq_len), accum_dtype),
        rrms_k: T.Tensor((M, seq_len), accum_dtype),
        drms_w_h: T.Tensor((d_padded,), accum_dtype),  # 外部预清零，atomic_add 累加
        dvhat_buf: T.Tensor((M, seq_len, d_padded), accum_dtype),  # pass2 输出，只读
        dH: T.Tensor((M, seq_len, d_padded), dtype),  # out_idx 自动分配
        dk: T.Tensor((M, seq_len, d_padded), dtype),  # out_idx 自动分配
        dv: T.Tensor((M, seq_len, d_padded), dtype),  # out_idx 自动分配
    ):
        with T.Kernel(T.ceildiv(seq_len, block_N) * M, is_npu=True) as (cid, _):
            bid = cid % M
            tile_id = cid // M
            tid_base = tile_id * block_N

            # ---- rms_w_h 对所有 tid 相同，tile 外 load 一次 ----
            rms_w_h_cast = T.alloc_ub((d_padded,), dtype)
            rms_w_h_local = T.alloc_ub((d_padded,), accum_dtype)
            T.copy(rms_w_h, rms_w_h_cast)
            T.vcast(rms_w_h_cast, rms_w_h_local)

            # ---- sqrt_d 只算一次 ----
            sqrt_d = T.alloc_ub((1, 1), accum_dtype)
            sqrt_d[0, 0] = d
            T.vsqrt(sqrt_d, sqrt_d)

            # ---- 复用缓冲区，tile 外分配 ----
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
            add_val = T.alloc_ub((d_padded,), accum_dtype)

            dot_k_2d = T.alloc_ub((d_padded,), accum_dtype)
            dot_k_sum = T.alloc_ub((1,), accum_dtype)
            dk_local = T.alloc_ub((d_padded,), accum_dtype)
            add_val_2 = T.alloc_ub((d_padded,), accum_dtype)

            dh_val = T.alloc_ub((d_padded,), dtype)
            dk_val = T.alloc_ub((d_padded,), dtype)
            dv_val = T.alloc_ub((d_padded,), dtype)

            for ti in T.serial(block_N):
                tid = tid_base + ti

                if tid < seq_len:
                    # ---------- load ----------
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
                            h_local[j] * rms_w_h_local[j] * ddot_val * k_norm_local[j]
                        )
                    T.reduce_sum(dot_h_2d, dot_h_sum, dim=0)

                    tmp_h_val = rrms_h_val * rrms_h_val * rrms_h_val * dot_h_sum[0] / d
                    for j in T.Parallel(d_padded):
                        dh_local[j] = (
                            rrms_h_val * rms_w_h_local[j] * ddot_val * k_norm_local[j]
                            - tmp_h_val * h_local[j]
                        )

                    # ---------- drms_w_h 累加（h 部分）----------
                    for j in T.Parallel(d_padded):
                        add_val[j] = (
                            ddot_val * k_norm_local[j] * h_local[j] * rrms_h_val
                        )
                    T.atomic_add(drms_w_h, add_val)

                    # ---------- dk ----------
                    for j in T.Parallel(d_padded):
                        dot_k_2d[j] = (
                            k_local[j] * rms_w_h_local[j] * ddot_val * h_norm_local[j]
                        )
                    T.reduce_sum(dot_k_2d, dot_k_sum, dim=0)

                    tmp_k_val = rrms_k_val * rrms_k_val * rrms_k_val * dot_k_sum[0] / d
                    for j in T.Parallel(d_padded):
                        dk_local[j] = (
                            rrms_k_val * rms_w_h_local[j] * ddot_val * h_norm_local[j]
                            - tmp_k_val * k_local[j]
                        )

                    # ---------- drms_w_h 累加（k 部分）----------
                    for j in T.Parallel(d_padded):
                        add_val_2[j] = (
                            ddot_val * h_norm_local[j] * k_local[j] * rrms_k_val
                        )
                    T.atomic_add(drms_w_h, add_val_2)

                    # ---------- cast & store ----------
                    T.vcast(dh_local, dh_val)
                    T.vcast(dk_local, dk_val)
                    T.vcast(dv_local, dv_val)
                    T.copy(dh_val, dH[bid, tid, :])
                    T.copy(dk_val, dk[bid, tid, :])
                    T.copy(dv_val, dv[bid, tid, :])

    return _bwd_pass3


os.environ["TILELANG_ASCEND_MODE"] = "Expert"
func = engram_gate_conv_bwd_pass3(1, 4096, 512, dtype="float16")
print("Best Config:", func.get_tuner_result())
print("Test passed!")
