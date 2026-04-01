import functools
from typing import Optional
import torch
import math
import tilelang as tl
import tilelang.language as T
import os

__all__ = ["EngramGateConvFwdKernel"]

ALIGNMENT = 256
CONV_KERNEL_SIZE = 4


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


@functools.lru_cache(maxsize=32)
def _engram_gate_pass1_kernel(M, seq_len, d, eps, dtype):
    accum_dtype = "float32"
    d_padded = _align_up(d, ALIGNMENT)

    @tl.jit(
        out_idx=[4, 5, 6, 7, 8],
        target="npuir",
    )
    def _func1(threads):
        @T.macro
        def _gate_pass1(
            H: T.Tensor((M, seq_len, d_padded), dtype),
            k: T.Tensor((M, seq_len, d_padded), dtype),
            v: T.Tensor((M, seq_len, d_padded), dtype),
            rms_w_h: T.Tensor((d_padded,), dtype),
            vhat_buf: T.Tensor((M, seq_len, d_padded), dtype),
            alpha_buf: T.Tensor((M, seq_len), accum_dtype),
            rrms_h_buf: T.Tensor((M, seq_len), accum_dtype),
            rrms_k_buf: T.Tensor((M, seq_len), accum_dtype),
            rrms_v_buf: T.Tensor((M, seq_len), accum_dtype),
        ):
            with T.Kernel(seq_len * M, is_npu=True, threads=[threads]) as (cid, _):
                bx = cid // M
                by = cid % M
                h_2d = T.alloc_shared((1, d_padded), accum_dtype)
                h_2d_cast = T.alloc_shared((1, d_padded), dtype)

                k_2d = T.alloc_shared((1, d_padded), accum_dtype)
                k_2d_cast = T.alloc_shared((1, d_padded), dtype)

                v_1d = T.alloc_shared((d_padded,), accum_dtype)
                v_1d_cast = T.alloc_shared((d_padded,), dtype)

                hsq = T.alloc_shared((1, d_padded), accum_dtype)
                ksq = T.alloc_shared((1, d_padded), accum_dtype)
                hk_prod = T.alloc_shared((1, d_padded), accum_dtype)
                sumsq_h = T.alloc_shared((1, 1), accum_dtype)
                sumsq_k = T.alloc_shared((1, 1), accum_dtype)
                dot_hk = T.alloc_shared((1, 1), accum_dtype)

                bid = by
                tid = bx

                T.copy(H[bid, tid, 0:d_padded], h_2d_cast)
                T.vcast(h_2d_cast, h_2d)
                T.copy(k[bid, tid, 0:d_padded], k_2d_cast)
                T.vcast(k_2d_cast, k_2d)
                T.copy(v[bid, tid, 0:d_padded], v_1d_cast)
                T.vcast(v_1d_cast, v_1d)

                d_shared = T.alloc_shared((1, 1), accum_dtype)
                d_shared[0, 0] = d

                for j in T.Parallel(d_padded):
                    hsq[0, j] = h_2d[0, j] * h_2d[0, j]
                T.reduce_sum(hsq, sumsq_h, dim=1)

                for j in T.Parallel(d_padded):
                    ksq[0, j] = k_2d[0, j] * k_2d[0, j]
                T.reduce_sum(ksq, sumsq_k, dim=1)

                T.vdiv(sumsq_k, d_shared, sumsq_k)
                T.vadd(sumsq_k, eps, sumsq_k)
                rrms_k_val = T.alloc_shared((1, 1), accum_dtype)
                T.vrsqrt(sumsq_k, rrms_k_val)
                T.copy(rrms_k_val, rrms_k_buf[bid : bid + 1, tid : tid + 1])

                T.vdiv(sumsq_h, d_shared, sumsq_h)
                T.vadd(sumsq_h, eps, sumsq_h)
                rrms_h_val = T.alloc_shared((1, 1), accum_dtype)
                T.vrsqrt(sumsq_h, rrms_h_val)
                T.copy(rrms_h_val, rrms_h_buf[bid : bid + 1, tid : tid + 1])

                rms_w_h_cast = T.alloc_shared((d_padded,), dtype)
                rms_w_h_shared = T.alloc_shared((d_padded,), accum_dtype)
                rms_w_h_reshape = T.alloc_shared((1, d_padded), accum_dtype)
                T.copy(rms_w_h, rms_w_h_cast)
                T.vcast(rms_w_h_cast, rms_w_h_shared)
                T.reshape(rms_w_h_shared, rms_w_h_reshape)
                for j in T.Parallel(d_padded):
                    h_2d[0, j] = h_2d[0, j] * rms_w_h_reshape[0, j]
                for j in T.Parallel(d_padded):
                    k_2d[0, j] = k_2d[0, j] * rms_w_h_reshape[0, j]
                T.vmul(h_2d, rrms_h_val, h_2d)
                T.vmul(k_2d, rrms_k_val, k_2d)

                for j in T.Parallel(d_padded):
                    hk_prod[0, j] = h_2d[0, j] * k_2d[0, j]
                T.reduce_sum(hk_prod, dot_hk, dim=1)

                exp_val = T.alloc_shared((1, 1), accum_dtype)
                alpha_val = T.alloc_shared((1, 1), accum_dtype)

                T.vsqrt(d_shared, exp_val)
                T.vdiv(dot_hk, exp_val, exp_val)
                T.vsigmoid(exp_val, alpha_val)
                T.copy(alpha_val, alpha_buf[bid : bid + 1, tid : tid + 1])

                vhsq = T.alloc_shared((1, d_padded), accum_dtype)
                sumsq_v = T.alloc_shared((1, 1), accum_dtype)

                for j in T.Parallel(d_padded):
                    vhat_val = alpha_val[0, 0] * v_1d[j]
                    vhat_buf[bid, tid, j] = vhat_val
                    vhsq[0, j] = vhat_val * vhat_val
                T.reduce_sum(vhsq, sumsq_v, dim=1)

                T.vdiv(sumsq_v, d_shared, sumsq_v)
                T.vadd(sumsq_v, eps, sumsq_v)
                tmp_val = T.alloc_shared((1, 1), accum_dtype)
                T.vrsqrt(sumsq_v, tmp_val)
                T.copy(tmp_val, rrms_v_buf[bid : bid + 1, tid : tid + 1])

        @T.prim_func
        def pass1(
            H: T.Tensor((M, seq_len, d_padded), dtype),
            k: T.Tensor((M, seq_len, d_padded), dtype),
            v: T.Tensor((M, seq_len, d_padded), dtype),
            rms_w_h: T.Tensor((d_padded,), dtype),
            vhat_buf: T.Tensor((M, seq_len, d_padded), dtype),
            alpha_buf: T.Tensor((M, seq_len), accum_dtype),
            rrms_h_buf: T.Tensor((M, seq_len), accum_dtype),
            rrms_k_buf: T.Tensor((M, seq_len), accum_dtype),
            rrms_v_buf: T.Tensor((M, seq_len), accum_dtype),
        ):
            _gate_pass1(
                H,
                k,
                v,
                rms_w_h,
                vhat_buf,
                alpha_buf,
                rrms_h_buf,
                rrms_k_buf,
                rrms_v_buf,
            )

        return pass1

    return _func1


@functools.lru_cache(maxsize=32)
def _engram_gate_pass2_kernel(M, seq_len, d, dtype):
    accum_dtype = "float32"
    d_padded = _align_up(d, ALIGNMENT)

    @tl.jit(
        target="npuir",
    )
    def _func2(threads):
        @T.macro
        def _gate_pass2(
            rms_w_v: T.Tensor((d_padded,), dtype),
            conv_w: T.Tensor((CONV_KERNEL_SIZE, d_padded), dtype),
            vhat_buf: T.Tensor((M, seq_len, d_padded), dtype),
            rrms_v_buf: T.Tensor((M, seq_len), accum_dtype),
            Y: T.Tensor((M, seq_len, d_padded), dtype),
        ):
            with T.Kernel(seq_len * M, is_npu=True, threads=[threads]) as (cid, _):
                tid = cid // M
                bid = cid % M

                vhat_cur_cast = T.alloc_shared((1, d_padded), dtype)
                vhat_cur = T.alloc_shared((1, d_padded), accum_dtype)
                T.copy(vhat_buf[bid, tid, 0:d_padded], vhat_cur_cast)
                T.vcast(vhat_cur_cast, vhat_cur)

                conv_out = T.alloc_shared((1, d_padded), accum_dtype)
                T.clear(conv_out)

                for p in T.serial(CONV_KERNEL_SIZE):
                    src_t = tid - (CONV_KERNEL_SIZE - 1) + p
                    src_rrms = T.alloc_shared((1, 1), accum_dtype)
                    T.clear(src_rrms)

                    raw_val_cast = T.alloc_shared((1, d_padded), dtype)
                    raw_val = T.alloc_shared((1, d_padded), accum_dtype)
                    T.clear(raw_val)

                    if src_t >= 0:
                        T.copy(vhat_buf[bid, src_t, 0:d_padded], raw_val_cast)
                        T.copy(rrms_v_buf[bid : bid + 1, src_t : src_t + 1], src_rrms)
                    T.vcast(raw_val_cast, raw_val)
                    rms_w_v_cast = T.alloc_shared((d_padded,), dtype)
                    rms_w_v_shared = T.alloc_shared((d_padded,), accum_dtype)
                    T.copy(rms_w_v, rms_w_v_cast)
                    T.vcast(rms_w_v_cast, rms_w_v_shared)
                    conv_w_cast = T.alloc_shared((1, d_padded), dtype)
                    conv_w_shared = T.alloc_shared((1, d_padded), accum_dtype)
                    T.copy(conv_w[p : p + 1, 0:d_padded], conv_w_cast)
                    T.vcast(conv_w_cast, conv_w_shared)

                    for j in T.Parallel(d_padded):
                        normed = raw_val[0, j] * src_rrms[0, 0] * rms_w_v_shared[j]
                        conv_out[0, j] += conv_w_shared[0, j] * normed
                sig = T.alloc_shared((1, d_padded), accum_dtype)
                T.vsigmoid(conv_out, sig)
                for j in T.Parallel(d_padded):
                    Y[bid, tid, j] = conv_out[0, j] * sig[0, j] + vhat_cur[0, j]

        @T.prim_func
        def pass2(
            rms_w_v: T.Tensor((d_padded,), dtype),
            conv_w: T.Tensor((CONV_KERNEL_SIZE, d_padded), dtype),
            vhat_buf: T.Tensor((M, seq_len, d_padded), dtype),
            rrms_v_buf: T.Tensor((M, seq_len), accum_dtype),
            Y: T.Tensor((M, seq_len, d_padded), dtype),
        ):
            _gate_pass2(rms_w_v, conv_w, vhat_buf, rrms_v_buf, Y)

        return pass2

    return _func2


@torch.library.custom_op("top::engram_gate_conv_fwd", mutates_args=())
def _engram_gate_conv_fwd_wrapped(
    M: int,
    seq_len: int,
    d: int,
    eps: float,
    dtype_str: str,
    threads: int,
    H: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rms_w_h: torch.Tensor,
    rms_w_v: torch.Tensor,
    conv_w: torch.Tensor,
) -> list[torch.Tensor]:
    pass1_res = _engram_gate_pass1_kernel(M, seq_len, d, eps, dtype_str)(
        threads,
    )(H, k, v, rms_w_h)
    vhat_buf = pass1_res[0]
    alpha_buf = pass1_res[1]
    rrms_h_buf = pass1_res[2]
    rrms_k_buf = pass1_res[3]
    rrms_v_buf = pass1_res[4]
    d_padded = _align_up(d, ALIGNMENT)
    Y = torch.empty((M, seq_len, d_padded), dtype=eval("torch." + dtype_str)).npu()
    _engram_gate_pass2_kernel(M, seq_len, d, dtype_str)(
        threads,
    )(rms_w_v, conv_w, vhat_buf, rrms_v_buf, Y)
    results = [Y, vhat_buf, alpha_buf, rrms_h_buf, rrms_k_buf, rrms_v_buf]
    return results


@_engram_gate_conv_fwd_wrapped.register_fake
def _(M, seq_len, d, eps, dtype_str, threads, H, k, v, rms_w_h, rms_w_v, conv_w):
    d_padded = _align_up(d, ALIGNMENT)
    device = H.device
    dt = H.dtype
    return [
        torch.empty((M, seq_len, d_padded), dtype=dt, device=device),  # Y
        torch.empty((M, seq_len, d_padded), dtype=dt, device=device),  # vhat_buf
        torch.empty((M, seq_len), dtype=torch.float32, device=device),  # alpha
        torch.empty((M, seq_len), dtype=torch.float32, device=device),  # rrms_h
        torch.empty((M, seq_len), dtype=torch.float32, device=device),  # rrms_k
        torch.empty((M, seq_len), dtype=torch.float32, device=device),  # rrms_v
    ]


class EngramGateConvFwdKernel:
    """Engram GateConv forward kernel.

    Fuses RMSNorm-based scalar gating and depthwise causal Conv1D
    with SiLU activation. Projection (GEMM) is done externally.

    Inputs:  H (M,T,d), k (M,T,d), v (M,T,d), rms_w_h, rms_w_v, conv_w
    Outputs: Y (M,T,d), vhat (M,T,d), alpha (M,T), rrms_h/k/v (M,T)
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        M: int,
        seq_len: int,
        d: int,
        eps: float,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        super().__init__()
        self.M = M
        self.seq_len = seq_len
        self.d = d
        self.eps = eps
        self.dtype = dtype
        self.d_padded = _align_up(d, ALIGNMENT)
        # TODO
        # self.kernel = _engram_gate_conv_fwd_kernel(
        #     M, seq_len, d, eps, self.dtype_str,
        # )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"threads": 128}

    @property
    def autotune_configs(self) -> list[dict]:
        return [{"threads": t} for t in [128, 256, 512]]

    def forward(
        self,
        H: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        rms_w_h: torch.Tensor,
        rms_w_v: torch.Tensor,
        conv_w: torch.Tensor,
    ) -> list[torch.Tensor]:
        return _engram_gate_conv_fwd_wrapped(
            self.M,
            self.seq_len,
            self.d,
            self.eps,
            self.dtype_str,
            self.config["threads"],
            H,
            k,
            v,
            rms_w_h,
            rms_w_v,
            conv_w,
        )


def ref_engram_gate_conv_fwd(
    M: int,
    seq_len: int,
    d: int,
    eps: float,
    H: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rms_w_h: torch.Tensor,
    rms_w_v: torch.Tensor,
    conv_w: torch.Tensor,
):
    d_padded = _align_up(d, ALIGNMENT)
    device = H.device
    out_dtype = H.dtype

    def pad_last_dim(x: torch.Tensor, target: int) -> torch.Tensor:
        if x.shape[-1] == target:
            return x
        assert x.shape[-1] < target, f"last dim {x.shape[-1]} > target {target}"
        pad_shape = list(x.shape[:-1]) + [target - x.shape[-1]]
        pad = torch.zeros(*pad_shape, dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=-1)

    H = pad_last_dim(H, d_padded)
    k = pad_last_dim(k, d_padded)
    v = pad_last_dim(v, d_padded)
    rms_w_h = pad_last_dim(rms_w_h, d_padded)
    rms_w_v = pad_last_dim(rms_w_v, d_padded)
    conv_w = pad_last_dim(conv_w, d_padded)

    Hf = H.float()
    kf = k.float()
    vf = v.float()
    rms_w_hf = rms_w_h.float()
    rms_w_vf = rms_w_v.float()
    conv_wf = conv_w.float()

    # pass1
    hsq = (Hf * Hf).sum(dim=-1)  # [M, T]
    ksq = (kf * kf).sum(dim=-1)  # [M, T]

    rrms_h = torch.rsqrt(hsq / float(d) + eps)  # [M, T]
    rrms_k = torch.rsqrt(ksq / float(d) + eps)  # [M, T]

    h_norm = Hf * rms_w_hf.view(1, 1, -1) * rrms_h.unsqueeze(-1)
    k_norm = kf * rms_w_hf.view(1, 1, -1) * rrms_k.unsqueeze(-1)

    dot_hk = (h_norm * k_norm).sum(dim=-1)  # [M, T]
    alpha = torch.sigmoid(dot_hk / math.sqrt(float(d)))  # [M, T]

    vhat_buf = alpha.unsqueeze(-1) * vf  # [M, T, d_padded]
    vhsq = (vhat_buf * vhat_buf).sum(dim=-1)  # [M, T]
    rrms_v = torch.rsqrt(vhsq / float(d) + eps)  # [M, T]

    # pass2
    Y = torch.zeros((M, seq_len, d_padded), dtype=torch.float32, device=device)

    for bid in range(M):
        for tid in range(seq_len):
            conv_out = torch.zeros((d_padded,), dtype=torch.float32, device=device)

            for p in range(CONV_KERNEL_SIZE):
                src_t = tid - (CONV_KERNEL_SIZE - 1) + p
                if src_t >= 0:
                    raw_val = vhat_buf[bid, src_t, :]  # [d_padded]
                    normed = raw_val * rrms_v[bid, src_t] * rms_w_vf  # [d_padded]
                    conv_out += conv_wf[p, :] * normed

            sig = torch.sigmoid(conv_out)
            Y[bid, tid, :] = conv_out * sig + vhat_buf[bid, tid, :]

    return (
        Y.to(out_dtype),
        vhat_buf.to(out_dtype),
        alpha,
        rrms_h,
        rrms_k,
        rrms_v,
    )


def run_test(
    M=2,
    seq_len=16,
    d=128,
    eps=1e-6,
    dtype=torch.float16,
    threads=128,
    atol=1e-2,
    rtol=1e-2,
    seed=0,
):
    torch.manual_seed(seed)

    d_padded = _align_up(d, ALIGNMENT)
    dtype_str = str(dtype).split(".")[-1]

    device = "npu"

    H = torch.randn((M, seq_len, d_padded), dtype=dtype, device=device)
    k = torch.randn((M, seq_len, d_padded), dtype=dtype, device=device)
    v = torch.randn((M, seq_len, d_padded), dtype=dtype, device=device)
    rms_w_h = torch.randn((d_padded,), dtype=dtype, device=device)
    rms_w_v = torch.randn((d_padded,), dtype=dtype, device=device)
    conv_w = torch.randn((CONV_KERNEL_SIZE, d_padded), dtype=dtype, device=device)

    print("compile finished!")

    Y, vhat_buf, alpha_buf, rrms_h_buf, rrms_k_buf, rrms_v_buf = (
        _engram_gate_conv_fwd_wrapped(
            M,
            seq_len,
            d,
            eps,
            dtype_str,
            threads,
            H,
            k,
            v,
            rms_w_h,
            rms_w_v,
            conv_w,
        )
    )

    print("kernel finished!")

    (
        Y_ref,
        vhat_buf_ref,
        alpha_buf_ref,
        rrms_h_buf_ref,
        rrms_k_buf_ref,
        rrms_v_buf_ref,
    ) = ref_engram_gate_conv_fwd(
        M=M,
        seq_len=seq_len,
        d=d,
        eps=eps,
        H=H,
        k=k,
        v=v,
        rms_w_h=rms_w_h,
        rms_w_v=rms_w_v,
        conv_w=conv_w,
    )

    try:
        torch.testing.assert_close(Y.float(), Y_ref.float(), atol=atol, rtol=rtol)
        print("Y check passed!")
    except AssertionError as e:
        print("Y check failed!")
        print(e)

    try:
        torch.testing.assert_close(
            vhat_buf.float(), vhat_buf_ref.float(), atol=atol, rtol=rtol
        )
        print("vhat_buf check passed!")
    except AssertionError as e:
        print("vhat_buf check failed!")
        print(e)

    try:
        torch.testing.assert_close(
            alpha_buf.float(), alpha_buf_ref.float(), atol=atol, rtol=rtol
        )
        print("alpha_buf check passed!")
    except AssertionError as e:
        print("alpha_buf check failed!")
        print(e)

    try:
        torch.testing.assert_close(
            rrms_h_buf.float(), rrms_h_buf_ref.float(), atol=atol, rtol=rtol
        )
        print("rrms_h_buf check passed!")
    except AssertionError as e:
        print("rrms_h_buf check failed!")
        print(e)

    try:
        torch.testing.assert_close(
            rrms_k_buf.float(), rrms_k_buf_ref.float(), atol=atol, rtol=rtol
        )
        print("rrms_k_buf check passed!")
    except AssertionError as e:
        print("rrms_k_buf check failed!")
        print(e)

    try:
        torch.testing.assert_close(
            rrms_v_buf.float(), rrms_v_buf_ref.float(), atol=atol, rtol=rtol
        )
        print("rrms_v_buf check passed!")
    except AssertionError as e:
        print("rrms_v_buf check failed!")
        print(e)

    return {
        "inputs": {
            "H": H,
            "k": k,
            "v": v,
            "rms_w_h": rms_w_h,
            "rms_w_v": rms_w_v,
            "conv_w": conv_w,
        },
        "outputs": {
            "Y": Y,
            "vhat_buf": vhat_buf,
            "alpha_buf": alpha_buf,
            "rrms_h_buf": rrms_h_buf,
            "rrms_k_buf": rrms_k_buf,
            "rrms_v_buf": rrms_v_buf,
        },
        "refs": {
            "Y_ref": Y_ref,
            "vhat_buf_ref": vhat_buf_ref,
            "alpha_buf_ref": alpha_buf_ref,
            "rrms_h_buf_ref": rrms_h_buf_ref,
            "rrms_k_buf_ref": rrms_k_buf_ref,
            "rrms_v_buf_ref": rrms_v_buf_ref,
        },
    }


if __name__ == "__main__":
    os.environ["TILELANG_ASCEND_MODE"] = "Dev"
    run_test(
        M=2,
        seq_len=16,
        d=128,
        eps=1e-6,
        dtype=torch.float16,
        threads=128,
    )
