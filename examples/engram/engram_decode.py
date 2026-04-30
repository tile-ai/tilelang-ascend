import math
import torch
import tilelang as tl
import tilelang.language as T
import os
import functools

ALIGNMENT = 256


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


@functools.lru_cache(maxsize=32)
def _engram_decode_kernel(
    batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, eps, dtype
):
    d_padded = _align_up(d, ALIGNMENT)
    w = conv_kernel_size
    accum_dtype = "float32"

    @tl.jit(out_idx=[8, 9], target="npuir")
    def _func():

        @T.macro
        def _decode_fused(
            e_t: T.Tensor((batch, d_mem), dtype),
            h_t: T.Tensor((batch, d_padded), dtype),
            conv_state: T.Tensor((batch, max_conv_len, d_padded), dtype),
            W_K: T.Tensor((d_mem, d_padded), dtype),
            W_V: T.Tensor((d_mem, d_padded), dtype),
            rms_w_h: T.Tensor((d_padded,), dtype),
            rms_w_v: T.Tensor((d_padded,), dtype),
            conv_w: T.Tensor((w, d_padded), dtype),
            y_t: T.Tensor((batch, d_padded), dtype),
            new_conv_state: T.Tensor((batch, max_conv_len, d_padded), dtype),
        ):
            with T.Kernel(batch, is_npu=True) as (bid, _):
                e_local = T.alloc_shared((d_mem,), accum_dtype)
                T.copy(e_t[bid, :], e_local)

                k_local = T.alloc_shared((d_padded,), accum_dtype)
                v_local = T.alloc_shared((d_padded,), accum_dtype)
                T.clear(k_local)
                T.clear(v_local)

                for i in T.serial(d_mem):
                    W_K_local = T.alloc_shared((d_padded,), accum_dtype)
                    T.copy(W_K[i, :], W_K_local)

                    W_V_local = T.alloc_shared((d_padded,), accum_dtype)
                    T.copy(W_V[i, :], W_V_local)

                    for j in T.Parallel(d_padded):
                        k_local[j] += e_local[i] * W_K_local[j]
                        v_local[j] += e_local[i] * W_V_local[j]

                h_local = T.alloc_shared((d_padded,), accum_dtype)
                T.copy(h_t[bid, :], h_local)

                hsq_2d = T.alloc_shared((1, d_padded), accum_dtype)
                for j in T.Parallel(d_padded):
                    hsq_2d[0, j] = h_local[j] * h_local[j]

                sumsq_h = T.alloc_shared((1, 1), accum_dtype)
                T.reduce_sum(hsq_2d, sumsq_h, dim=1)

                sumsq_h[0, 0] = sumsq_h[0, 0] / d + eps
                T.vrsqrt(sumsq_h, sumsq_h)

                rms_w_h_local = T.alloc_shared((d_padded,), accum_dtype)
                T.copy(rms_w_h, rms_w_h_local)
                for j in T.Parallel(d_padded):
                    h_local[j] = h_local[j] * sumsq_h[0, 0] * rms_w_h_local[j]

                ksq_2d = T.alloc_shared((1, d_padded), accum_dtype)
                for j in T.Parallel(d_padded):
                    ksq_2d[0, j] = k_local[j] * k_local[j]
                sumsq_k = T.alloc_shared((1, 1), accum_dtype)
                T.reduce_sum(ksq_2d, sumsq_k, dim=1)
                T.vdiv(sumsq_k, d, sumsq_k)
                T.vadd(sumsq_k, eps, sumsq_k)
                T.vrsqrt(sumsq_k, sumsq_k)
                for j in T.Parallel(d_padded):
                    k_local[j] = k_local[j] * sumsq_k[0, 0] * rms_w_h_local[j]

                hk_2d = T.alloc_shared((1, d_padded), accum_dtype)
                for j in T.Parallel(d_padded):
                    hk_2d[0, j] = h_local[j] * k_local[j]
                dot_hk = T.alloc_shared((1, 1), accum_dtype)
                T.reduce_sum(hk_2d, dot_hk, dim=1)
                sqrt_d = T.alloc_shared((1, 1), accum_dtype)
                sqrt_d[0, 0] = d
                T.vsqrt(sqrt_d, sqrt_d)
                alpha = T.alloc_shared((1, 1), accum_dtype)
                alpha[0, 0] = dot_hk[0, 0] / sqrt_d[0, 0]
                T.vsigmoid(alpha, alpha)
                val = alpha[0, 0]
                vhat_local = T.alloc_shared((d_padded,), accum_dtype)
                for j in T.Parallel(d_padded):
                    vhat_local[j] = val * v_local[j]

                vsq_2d = T.alloc_shared((1, d_padded), accum_dtype)
                for j in T.Parallel(d_padded):
                    vsq_2d[0, j] = vhat_local[j] * vhat_local[j]
                sumsq_v = T.alloc_shared((1, 1), accum_dtype)
                T.reduce_sum(vsq_2d, sumsq_v, dim=1)
                T.vdiv(sumsq_v, d, sumsq_v)
                T.vadd(sumsq_v, eps, sumsq_v)
                T.vrsqrt(sumsq_v, sumsq_v)
                rms_w_v_local = T.alloc_shared((d_padded,), accum_dtype)
                T.copy(rms_w_v, rms_w_v_local)
                for j in T.Parallel(d_padded):
                    vhat_local[j] = vhat_local[j] * sumsq_v[0, 0] * rms_w_v_local[j]

                conv_out = T.alloc_shared((d_padded,), accum_dtype)
                T.clear(conv_out)

                for p in T.serial(w - 1):
                    conv_w_local = T.alloc_shared((d_padded,), accum_dtype)
                    T.copy(conv_w[p, :], conv_w_local)
                    conv_state_local = T.alloc_shared((d_padded,), accum_dtype)
                    T.copy(
                        conv_state[bid, max_conv_len - (w - 1 - p) * dilation, :],
                        conv_state_local,
                    )
                    for j in T.Parallel(d_padded):
                        conv_out[j] += conv_w_local[j] * conv_state_local[j]

                conv_w_l = T.alloc_shared((d_padded,), accum_dtype)
                T.copy(conv_w[w - 1, :], conv_w_l)
                for j in T.Parallel(d_padded):
                    conv_out[j] += conv_w_l[j] * vhat_local[j]

                # Unsupported memref to memref copy yet.
                # for s in T.serial(max_conv_len - 1):
                #     for j in T.Parallel(d_padded):
                #         new_conv_state[bid, s, j] = conv_state[bid, s + 1, j]
                for s in T.serial(max_conv_len - 1):
                    tmp = T.alloc_shared((d_padded,), dtype)
                    T.copy(conv_state[bid, s + 1, :], tmp)
                    T.copy(tmp, new_conv_state[bid, s, :])

                T.copy(vhat_local, new_conv_state[bid, max_conv_len - 1, :])

                sig = T.alloc_shared((d_padded,), accum_dtype)
                T.vsigmoid(conv_out, sig)
                alpha_val = alpha[0, 0]
                for j in T.Parallel(d_padded):
                    conv_out[j] = conv_out[j] * sig[j] + alpha_val * v_local[j]
                T.copy(conv_out, y_t[bid, :])

        @T.prim_func
        def main(
            e_t: T.Tensor((batch, d_mem), dtype),
            h_t: T.Tensor((batch, d_padded), dtype),
            conv_state: T.Tensor((batch, max_conv_len, d_padded), dtype),
            W_K: T.Tensor((d_mem, d_padded), dtype),
            W_V: T.Tensor((d_mem, d_padded), dtype),
            rms_w_h: T.Tensor((d_padded,), dtype),
            rms_w_v: T.Tensor((d_padded,), dtype),
            conv_w: T.Tensor((w, d_padded), dtype),
            y_t: T.Tensor((batch, d_padded), dtype),
            new_conv_state: T.Tensor((batch, max_conv_len, d_padded), dtype),
        ):
            _decode_fused(
                e_t,
                h_t,
                conv_state,
                W_K,
                W_V,
                rms_w_h,
                rms_w_v,
                conv_w,
                y_t,
                new_conv_state,
            )

        return main

    return _func


def _engram_decode_wrapped(
    batch: int,
    d_mem: int,
    d: int,
    max_conv_len: int,
    conv_kernel_size: int,
    dilation: int,
    eps: float,
    dtype_str: str,
    e_t: torch.Tensor,
    h_t: torch.Tensor,
    conv_state: torch.Tensor,
    W_K: torch.Tensor,
    W_V: torch.Tensor,
    rms_w_h: torch.Tensor,
    rms_w_v: torch.Tensor,
    conv_w: torch.Tensor,
) -> list[torch.Tensor]:
    results = _engram_decode_kernel(
        batch,
        d_mem,
        d,
        max_conv_len,
        conv_kernel_size,
        dilation,
        eps,
        dtype_str,
    )()(e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w)
    return list(results)


def ref_engram_decode(
    batch: int,
    d_mem: int,
    d: int,
    max_conv_len: int,
    conv_kernel_size: int,
    dilation: int,
    eps: float,
    e_t: torch.Tensor,
    h_t: torch.Tensor,
    conv_state: torch.Tensor,
    W_K: torch.Tensor,
    W_V: torch.Tensor,
    rms_w_h: torch.Tensor,
    rms_w_v: torch.Tensor,
    conv_w: torch.Tensor,
):
    d_padded = _align_up(d, ALIGNMENT)
    w = conv_kernel_size
    out_dtype = e_t.dtype
    device = e_t.device

    def pad_last_dim(x: torch.Tensor, target: int) -> torch.Tensor:
        if x.shape[-1] == target:
            return x
        assert x.shape[-1] < target
        pad_shape = list(x.shape[:-1]) + [target - x.shape[-1]]
        pad = torch.zeros(*pad_shape, dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=-1)

    h_t = pad_last_dim(h_t, d_padded)
    conv_state = pad_last_dim(conv_state, d_padded)
    W_K = pad_last_dim(W_K, d_padded)
    W_V = pad_last_dim(W_V, d_padded)
    rms_w_h = pad_last_dim(rms_w_h, d_padded)
    rms_w_v = pad_last_dim(rms_w_v, d_padded)
    conv_w = pad_last_dim(conv_w, d_padded)

    e_f = e_t.float()
    h_f = h_t.float()
    conv_state_f = conv_state.float()
    W_K_f = W_K.float()
    W_V_f = W_V.float()
    rms_w_h_f = rms_w_h.float()
    rms_w_v_f = rms_w_v.float()
    conv_w_f = conv_w.float()

    k_local = e_f @ W_K_f
    v_local = e_f @ W_V_f

    h_mean_sq = (h_f * h_f).sum(dim=-1, keepdim=True) / float(d)
    h_norm = h_f * torch.rsqrt(h_mean_sq + eps) * rms_w_h_f.unsqueeze(0)

    k_mean_sq = (k_local * k_local).sum(dim=-1, keepdim=True) / float(d)
    k_norm = k_local * torch.rsqrt(k_mean_sq + eps) * rms_w_h_f.unsqueeze(0)

    dot_hk = (h_norm * k_norm).sum(dim=-1, keepdim=True)
    alpha = torch.sigmoid(dot_hk / math.sqrt(float(d)))

    vhat = alpha * v_local
    v_mean_sq = (vhat * vhat).sum(dim=-1, keepdim=True) / float(d)
    vhat_norm = vhat * torch.rsqrt(v_mean_sq + eps) * rms_w_v_f.unsqueeze(0)

    conv_out = torch.zeros((batch, d_padded), dtype=torch.float32, device=device)
    for p in range(w - 1):
        idx = max_conv_len - (w - 1 - p) * dilation
        conv_out += conv_w_f[p].unsqueeze(0) * conv_state_f[:, idx, :]

    conv_out += conv_w_f[w - 1].unsqueeze(0) * vhat_norm

    new_conv_state = torch.empty_like(conv_state_f)
    new_conv_state[:, :-1, :] = conv_state_f[:, 1:, :]
    new_conv_state[:, -1, :] = vhat_norm

    sig = torch.sigmoid(conv_out)
    y = conv_out * sig + alpha * v_local

    return y.to(out_dtype), new_conv_state.to(out_dtype)


def run_test(
    batch=2,
    d_mem=32,
    d=64,
    max_conv_len=9,
    conv_kernel_size=4,
    dilation=2,
    eps=1e-6,
    dtype=torch.float16,
    atol=2e-2,
    rtol=2e-2,
    seed=0,
):
    torch.manual_seed(seed)

    assert max_conv_len >= dilation * (conv_kernel_size - 1), (
        f"max_conv_len must be >= dilation * (conv_kernel_size - 1), "
        f"got {max_conv_len} < {dilation * (conv_kernel_size - 1)}"
    )

    d_padded = _align_up(d, ALIGNMENT)
    device = "npu" if hasattr(torch, "npu") else "cpu"
    dtype_str = str(dtype).split(".")[-1]

    e_t = torch.randn((batch, d_mem), dtype=dtype, device=device)
    h_t = torch.randn((batch, d_padded), dtype=dtype, device=device)
    conv_state = torch.randn(
        (batch, max_conv_len, d_padded), dtype=dtype, device=device
    )
    W_K = torch.randn((d_mem, d_padded), dtype=dtype, device=device)
    W_V = torch.randn((d_mem, d_padded), dtype=dtype, device=device)
    rms_w_h = torch.randn((d_padded,), dtype=dtype, device=device)
    rms_w_v = torch.randn((d_padded,), dtype=dtype, device=device)
    conv_w = torch.randn((conv_kernel_size, d_padded), dtype=dtype, device=device)

    print("compile + run wrapper ...")
    y_t, new_conv_state = _engram_decode_wrapped(
        batch,
        d_mem,
        d,
        max_conv_len,
        conv_kernel_size,
        dilation,
        eps,
        dtype_str,
        e_t,
        h_t,
        conv_state,
        W_K,
        W_V,
        rms_w_h,
        rms_w_v,
        conv_w,
    )

    y_ref, new_conv_state_ref = ref_engram_decode(
        batch=batch,
        d_mem=d_mem,
        d=d,
        max_conv_len=max_conv_len,
        conv_kernel_size=conv_kernel_size,
        dilation=dilation,
        eps=eps,
        e_t=e_t,
        h_t=h_t,
        conv_state=conv_state,
        W_K=W_K,
        W_V=W_V,
        rms_w_h=rms_w_h,
        rms_w_v=rms_w_v,
        conv_w=conv_w,
    )

    torch.testing.assert_close(y_t.float(), y_ref.float(), atol=atol, rtol=rtol)

    torch.testing.assert_close(
        new_conv_state.float(),
        new_conv_state_ref.float(),
        atol=atol,
        rtol=rtol,
    )
    print("All check passed!")


if __name__ == "__main__":
    os.environ["TILELANG_ASCEND_MODE"] = "Dev"
    run_test(
        batch=2,
        d_mem=32,
        d=64,
        max_conv_len=9,
        conv_kernel_size=4,
        dilation=2,
        eps=1e-6,
        dtype=torch.float16,
    )
