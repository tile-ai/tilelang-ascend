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
    dY: torch.Tensor,
    rms_w_v: torch.Tensor,
    conv_w: torch.Tensor,
    vhat: torch.Tensor,
    rrms_v: torch.Tensor,
    dconv_w: torch.Tensor,
):
    device = dY.device
    M, seq_len, d_padded = dY.shape
    KS = conv_w.shape[0]

    dY_f = dY.float()
    rms_w_v_f = rms_w_v.float()
    conv_w_f = conv_w.float()
    vhat_f = vhat.float()
    rrms_v_f = rrms_v.float()

    dvhat_buf = torch.zeros((M, seq_len, d_padded), dtype=torch.float32, device=device)

    zero_vec = torch.zeros((d_padded,), dtype=torch.float32, device=device)

    for bid in range(M):
        for tid in range(seq_len):
            dy_local = dY_f[bid, tid]
            conv_out = torch.zeros((d_padded,), dtype=torch.float32, device=device)

            for p in range(KS):
                src_t = tid - (KS - 1) + p
                if src_t >= 0:
                    raw_val = vhat_f[bid, src_t]
                    src_rr = rrms_v_f[bid, src_t]
                else:
                    raw_val = zero_vec
                    src_rr = 0.0
                conv_out += conv_w_f[p] * raw_val * src_rr * rms_w_v_f

            sig = torch.sigmoid(conv_out)
            d_silu = dy_local * (sig + conv_out * sig * (1.0 - sig))

            for p in range(KS):
                src_t = tid - (KS - 1) + p
                if src_t >= 0:
                    raw_val = vhat_f[bid, src_t]
                    src_rr = rrms_v_f[bid, src_t]
                else:
                    raw_val = zero_vec
                    src_rr = 0.0
                dconv_w[p] += d_silu * raw_val * src_rr * rms_w_v_f

            dvhat_buf[bid, tid] = d_silu

    return dvhat_buf


def supply_prog(params):
    """
    Return concrete tensors for autotune profiling/checking.
    The returned tensor order must match the kernel input order.
    """
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
    dconv_w = torch.zeros((KS, d_padded), dtype=torch.float32, device=dev)

    return [dY, rms_w_v, conv_w, vhat, rrms_v, dconv_w]


@tl.autotune(
    configs=get_config(),
    ref_prog=ref_prog,
    supply_prog=supply_prog,
    atol=1e-2,
    rtol=1e-2,
)
@tl.jit(
    out_idx=[6],
    target="npuir",
)
def engram_gate_conv_bwd_pass1(M, seq_len, d, block_N, dtype):
    accum_dtype = "float32"
    d_padded = _align_up(d, ALIGNMENT)
    KS = CONV_KERNEL_SIZE

    @T.prim_func
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

            for ti in T.serial(block_N):
                tid = tid_base + ti

                if tid < seq_len:
                    # ---------- load dY ----------
                    dy_local_cast = T.alloc_ub((1, d_padded), dtype)
                    dy_local = T.alloc_ub((1, d_padded), accum_dtype)
                    T.copy(dY[bid, tid, :], dy_local_cast)
                    T.vcast(dy_local_cast, dy_local)

                    # ---------- shared buffers ----------
                    rms_w_v_cast = T.alloc_ub((d_padded,), dtype)
                    rms_w_v_shared = T.alloc_ub((d_padded,), accum_dtype)
                    T.copy(rms_w_v, rms_w_v_cast)
                    T.vcast(rms_w_v_cast, rms_w_v_shared)

                    raw_val_cast = T.alloc_ub((1, d_padded), dtype)
                    raw_val = T.alloc_ub((1, d_padded), accum_dtype)
                    conv_w_cast = T.alloc_ub((1, d_padded), dtype)
                    conv_w_local = T.alloc_ub((1, d_padded), accum_dtype)
                    src_rrms = T.alloc_ub((1, 1), accum_dtype)

                    # ---------- conv_out ----------
                    conv_out = T.alloc_ub((1, d_padded), accum_dtype)
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

                    # ---------- d_silu ----------
                    sig = T.alloc_ub((1, d_padded), accum_dtype)
                    d_silu = T.alloc_ub((1, d_padded), accum_dtype)
                    T.clear(sig)
                    T.clear(d_silu)

                    T.vsigmoid(conv_out, sig)

                    for j in T.Parallel(d_padded):
                        d_silu[0, j] = dy_local[0, j] * (
                            sig[0, j] + conv_out[0, j] * sig[0, j] * (1.0 - sig[0, j])
                        )

                    # ---------- dconv_w ----------
                    tmp = T.alloc_ub((1, d_padded), accum_dtype)

                    for p in T.serial(KS):
                        src_t = tid - (KS - 1) + p

                        T.clear(raw_val_cast)
                        T.clear(raw_val)
                        T.clear(src_rrms)
                        T.clear(tmp)

                        if src_t >= 0:
                            T.copy(vhat[bid, src_t, :], raw_val_cast)
                            T.copy(rrms_v[bid : bid + 1, src_t : src_t + 1], src_rrms)

                        T.vcast(raw_val_cast, raw_val)

                        for j in T.Parallel(d_padded):
                            tmp[0, j] = (
                                d_silu[0, j]
                                * raw_val[0, j]
                                * src_rrms[0, 0]
                                * rms_w_v_shared[j]
                            )
                        T.atomic_add(dconv_w[p, :], tmp)

                    # ---------- write dvhat_buf ----------
                    T.copy(d_silu, dvhat_buf[bid, tid : tid + 1, :])

    return _bwd_pass1


os.environ["TILELANG_ASCEND_MODE"] = "Expert"
func = engram_gate_conv_bwd_pass1(1, 4096, 512, dtype="float16")
print("Best Config:", func.get_tuner_result())
print("Test passed!")
