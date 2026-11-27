import math
import os

import tilelang
import torch
from tilelang import language as T

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

VEC_NUM = 2
MAX_H_BLK = 4096
MAX_UB_BYTES = 192 * 1024


def _get_npu_num_sms(default=32):
    try:
        import torch_npu

        props = torch_npu.npu.get_device_properties(0)
        cube_core_num = props.cube_core_num
        return cube_core_num
    except Exception:
        return int(os.environ.get("ASCEND_NUM_SMS", default))


def _divisors_desc(n: int) -> list[int]:
    divs = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs, reverse=True)


def _compute_safe_h_blk(hidden: int, mhc: int, max_h_blk: int) -> int:
    for h_blk in _divisors_desc(hidden):
        if h_blk > max_h_blk:
            continue
        if h_blk % VEC_NUM != 0:
            continue
        fwd_ub = _estimate_fwd_ub_bytes(mhc, h_blk)
        bwd_ub = _estimate_bwd_ub_bytes(mhc, h_blk)
        if fwd_ub <= MAX_UB_BYTES and bwd_ub <= MAX_UB_BYTES:
            return h_blk
    raise ValueError(f"No safe h_blk for hidden={hidden}, mhc={mhc}")


def _compute_token_block_size(n: int, num_cores: int = 0) -> int:
    if num_cores <= 0:
        num_cores = _get_npu_num_sms()
    if n <= num_cores:
        return 1
    tbs = 1
    while tbs * 2 <= n // num_cores:
        tbs *= 2
    return max(1, tbs // 2)


def _estimate_fwd_ub_bytes(mhc: int, h_blk: int) -> int:
    sub_h = h_blk // VEC_NUM
    pad_sub_h = ((sub_h + 15) // 16) * 16
    pad_mhc = ((mhc + 7) // 8) * 8
    per_vec = 2 * pad_mhc * 4 + 2 * mhc * pad_sub_h * 2 + mhc * pad_sub_h * 4 + pad_sub_h * 4 + pad_sub_h * 2
    return per_vec * VEC_NUM


def _estimate_bwd_ub_bytes(mhc: int, h_blk: int) -> int:
    sub_mhc = mhc // VEC_NUM
    pad_h_blk = ((h_blk + 15) // 16) * 16
    per_vec = (
        sub_mhc * 4
        + sub_mhc * 4
        + pad_h_blk * 2
        + pad_h_blk * 4
        + sub_mhc * pad_h_blk * 4
        + sub_mhc * pad_h_blk * 2
        + sub_mhc * pad_h_blk * 4
        + sub_mhc * pad_h_blk * 4
        + sub_mhc * pad_h_blk * 2
    )
    return per_vec * VEC_NUM


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_pre_apply_mix_fwd(
    mhc_mult: int,
    hidden: int,
    n_thr: int = 128,
    h_blk: int = 1920,
    token_block_size: int = 32,
) -> tilelang.JITKernel:
    n = T.symbolic("n")
    dtype = "float32"
    dbtype = "bfloat16"
    h = hidden
    mhc = mhc_mult
    tbs = token_block_size

    assert hidden % h_blk == 0, f"hidden={hidden} not divisible by h_blk={h_blk}"
    assert h_blk % VEC_NUM == 0, f"h_blk={h_blk} not divisible by VEC_NUM={VEC_NUM}"
    assert mhc % VEC_NUM == 0, f"mhc={mhc} not divisible by VEC_NUM={VEC_NUM}"

    num_blocks_h = h // h_blk
    sub_h = h_blk // VEC_NUM
    pad_sub_h = T.ceildiv(sub_h, 16) * 16
    pad_mhc = T.ceildiv(mhc, 8) * 8

    @T.prim_func
    def _mhc_pre_apply_mix_fwd_kernel(
        x: T.Tensor[(n, mhc, h), dbtype],
        mix: T.Tensor[(n, mhc), dtype],
        o: T.Tensor[(n, h), dbtype],
    ) -> None:
        num_blocks_n = T.ceildiv(n, tbs)
        total_cores = num_blocks_n * num_blocks_h

        with T.Kernel(total_cores, is_npu=True) as (cid, vid):
            pid_n_block = cid // num_blocks_h
            pid_h_block = cid % num_blocks_h
            h_start = pid_h_block * h_blk + vid * sub_h

            mixl_db = T.alloc_ub((2, pad_mhc), dtype)
            x_bf16_db = T.alloc_ub((2, mhc, pad_sub_h), dbtype)
            x_f32_ub = T.alloc_ub((mhc, pad_sub_h), dtype)
            o_f32_ub = T.alloc_ub((pad_sub_h,), dtype)
            o_bf16_ub = T.alloc_ub((pad_sub_h,), dbtype)

            pid_n_first = pid_n_block * tbs
            if pid_n_first < n:
                T.copy(mix[pid_n_first, 0:mhc], mixl_db[0, 0:mhc])
                T.copy(x[pid_n_first, 0:mhc, h_start : h_start + sub_h], x_bf16_db[0, :, :])
                T.set_flag("MTE2", "V", 0)

            for i_token in T.serial(tbs):
                pid_n = pid_n_block * tbs + i_token
                buf_pid = i_token % 2

                if pid_n < n:
                    T.wait_flag("MTE2", "V", buf_pid)

                    if i_token + 1 < tbs:
                        next_pid_n = pid_n + 1
                        next_pid = (i_token + 1) % 2
                        if next_pid_n < n:
                            T.copy(mix[next_pid_n, 0:mhc], mixl_db[next_pid, 0:mhc])
                            T.copy(x[next_pid_n, 0:mhc, h_start : h_start + sub_h], x_bf16_db[next_pid, :, :])
                            T.set_flag("MTE2", "V", next_pid)

                    T.tile.fill(o_f32_ub, 0.0)
                    T.tile.cast(x_f32_ub, x_bf16_db[buf_pid, :, :], "CAST_NONE", mhc * pad_sub_h)

                    for i_mhc in T.serial(mhc):
                        T.tile.axpy(o_f32_ub, x_f32_ub[i_mhc, :], mixl_db[buf_pid, i_mhc])

                    T.tile.cast(o_bf16_ub, o_f32_ub, "CAST_ROUND", pad_sub_h)

                    T.set_flag("V", "MTE3", 0)
                    T.wait_flag("V", "MTE3", 0)
                    T.copy(o_bf16_ub[0:sub_h], o[pid_n, h_start : h_start + sub_h])

    return _mhc_pre_apply_mix_fwd_kernel


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_pre_apply_mix_bwd(
    mhc_mult: int,
    hidden: int,
    n_thr: int = 128,
    h_blk: int = 4096,
    token_block_size: int = 1,
) -> tilelang.JITKernel:
    n = T.symbolic("n")
    dtype = "float32"
    dbtype = "bfloat16"
    h = hidden
    mhc = mhc_mult
    tbs = token_block_size

    h_blk = math.gcd(h_blk, hidden)
    assert h_blk % VEC_NUM == 0, f"h_blk={h_blk} not divisible by VEC_NUM={VEC_NUM}"
    assert mhc % VEC_NUM == 0, f"mhc={mhc} not divisible by VEC_NUM={VEC_NUM}"

    sub_mhc = mhc // VEC_NUM
    pad_h_blk = T.ceildiv(h_blk, 16) * 16
    needs_pad = h_blk % 16 != 0
    num_h_iters = h // h_blk

    @T.prim_func
    def _mhc_pre_apply_mix_bwd_kernel(
        o_grad: T.Tensor[(n, h), dbtype],
        x: T.Tensor[(n, mhc, h), dbtype],
        mix: T.Tensor[(n, mhc), dtype],
        x_grad: T.Tensor[(n, mhc, h), dbtype],
        mix_grad: T.Tensor[(n, mhc), dtype],
    ) -> None:
        with T.Kernel(T.ceildiv(n, tbs), is_npu=True) as (cid, vid):
            mhc_start = vid * sub_mhc

            mixl_ub = T.alloc_ub((sub_mhc,), dtype)
            mgl_ub = T.alloc_ub((sub_mhc,), dtype)

            o_grad_bf16_ub = T.alloc_ub((pad_h_blk,), dbtype)
            o_grad_f32_ub = T.alloc_ub((pad_h_blk,), dtype)
            o_grad_2d_ub = T.alloc_ub((sub_mhc, pad_h_blk), dtype)

            x_bf16_ub = T.alloc_ub((sub_mhc, pad_h_blk), dbtype)
            x_f32_ub = T.alloc_ub((sub_mhc, pad_h_blk), dtype)

            x_grad_f32_ub = T.alloc_ub((sub_mhc, pad_h_blk), dtype)
            x_grad_bf16_ub = T.alloc_ub((sub_mhc, pad_h_blk), dbtype)

            for i_token in T.serial(tbs):
                pid_n = cid * tbs + i_token

                if pid_n < n:
                    T.tile.fill(mgl_ub, 0.0)
                    if needs_pad:
                        T.tile.fill(o_grad_bf16_ub, T.cast(0.0, dbtype))
                        T.tile.fill(x_bf16_ub, T.cast(0.0, dbtype))

                    if needs_pad:
                        T.set_flag("V", "MTE2", 2)
                        T.wait_flag("V", "MTE2", 2)

                    T.copy(mix[pid_n, mhc_start : mhc_start + sub_mhc], mixl_ub[0:sub_mhc])
                    T.set_flag("MTE2", "V", 0)
                    T.wait_flag("MTE2", "V", 0)

                    for i0_h in T.serial(num_h_iters):
                        h_offset = i0_h * h_blk

                        T.copy(o_grad[pid_n, h_offset : h_offset + h_blk], o_grad_bf16_ub[0:h_blk])
                        T.copy(x[pid_n, mhc_start : mhc_start + sub_mhc, h_offset : h_offset + h_blk], x_bf16_ub[0:sub_mhc, 0:h_blk])
                        T.set_flag("MTE2", "V", 1)
                        T.wait_flag("MTE2", "V", 1)

                        T.tile.cast(o_grad_f32_ub, o_grad_bf16_ub, "CAST_NONE", pad_h_blk)
                        T.tile.cast(x_f32_ub, x_bf16_ub, "CAST_NONE", sub_mhc * pad_h_blk)

                        T.tile.broadcast(o_grad_2d_ub, o_grad_f32_ub)
                        T.tile.mul(x_grad_f32_ub, x_f32_ub, o_grad_2d_ub)
                        T.reduce_sum(x_grad_f32_ub, mgl_ub, dim=1, clear=False)
                        T.tile.broadcast(x_f32_ub, mixl_ub)
                        T.tile.mul(x_grad_f32_ub, x_f32_ub, o_grad_2d_ub)

                        T.tile.cast(x_grad_bf16_ub, x_grad_f32_ub, "CAST_ROUND", sub_mhc * pad_h_blk)

                        T.set_flag("V", "MTE2", 3)
                        T.wait_flag("V", "MTE2", 3)

                        T.set_flag("V", "MTE3", 4)
                        T.wait_flag("V", "MTE3", 4)
                        T.copy(
                            x_grad_bf16_ub[0:sub_mhc, 0:h_blk], x_grad[pid_n, mhc_start : mhc_start + sub_mhc, h_offset : h_offset + h_blk]
                        )

                        T.set_flag("MTE3", "V", 4)
                        T.wait_flag("MTE3", "V", 4)

                    T.set_flag("V", "MTE2", 5)
                    T.wait_flag("V", "MTE2", 5)

                    T.set_flag("V", "MTE3", 6)
                    T.wait_flag("V", "MTE3", 6)
                    T.copy(mgl_ub[0:sub_mhc], mix_grad[pid_n, mhc_start : mhc_start + sub_mhc])

                    T.set_flag("MTE3", "V", 6)
                    T.wait_flag("MTE3", "V", 6)

    return _mhc_pre_apply_mix_bwd_kernel


def mhc_pre_apply_mix_fwd(
    x: torch.Tensor,
    mix: torch.Tensor,
    h_blk: int = 0,
    token_block_size: int = 0,
) -> torch.Tensor:
    num_seqs, num_tokens, mhc, hidden = x.shape
    n = num_seqs * num_tokens

    assert x.dtype == torch.bfloat16
    assert mix.dtype == torch.float32
    assert mix.shape == (num_seqs, num_tokens, mhc)
    assert mhc % VEC_NUM == 0, f"mhc={mhc} must be divisible by VEC_NUM={VEC_NUM}"

    x = x.contiguous()
    mix = mix.contiguous()

    out = torch.empty((num_seqs, num_tokens, hidden), dtype=torch.bfloat16, device=x.device)

    fwd_h_blk = h_blk if h_blk > 0 else _compute_safe_h_blk(hidden, mhc, MAX_H_BLK)
    tbs = token_block_size if token_block_size > 0 else _compute_token_block_size(n)

    kernel = _mhc_pre_apply_mix_fwd(mhc, hidden, h_blk=fwd_h_blk, token_block_size=tbs)
    kernel(x.flatten(0, 1), mix.flatten(0, 1), out.flatten(0, 1))
    return out


def mhc_pre_apply_mix_bwd(
    x: torch.Tensor,
    mix: torch.Tensor,
    o_grad: torch.Tensor,
    h_blk: int = 0,
    token_block_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_seqs, num_tokens, mhc, hidden = x.shape
    n = num_seqs * num_tokens

    assert x.dtype == torch.bfloat16
    assert mix.dtype == torch.float32
    assert o_grad.dtype == torch.bfloat16
    assert o_grad.shape == (num_seqs, num_tokens, hidden)
    assert mix.shape == (num_seqs, num_tokens, mhc)
    assert mhc % VEC_NUM == 0, f"mhc={mhc} must be divisible by VEC_NUM={VEC_NUM}"

    x = x.contiguous()
    mix = mix.contiguous()
    o_grad = o_grad.contiguous()

    x_grad = torch.empty((n, mhc, hidden), dtype=torch.bfloat16, device=x.device)
    mix_grad = torch.empty((n, mhc), dtype=torch.float32, device=x.device)

    bwd_h_blk = h_blk if h_blk > 0 else _compute_safe_h_blk(hidden, mhc, MAX_H_BLK)
    tbs = token_block_size if token_block_size > 0 else _compute_token_block_size(n)

    bwd_kernel = _mhc_pre_apply_mix_bwd(mhc, hidden, h_blk=bwd_h_blk, token_block_size=tbs)
    bwd_kernel(
        o_grad.flatten(0, 1),
        x.flatten(0, 1),
        mix.flatten(0, 1),
        x_grad,
        mix_grad,
    )

    return (
        x_grad.view_as(x),
        mix_grad.view_as(mix),
    )


def mhc_pre_apply_mix_fwd_ref(
    x: torch.Tensor,
    mix: torch.Tensor,
) -> torch.Tensor:
    return (mix.unsqueeze(-1).float() * x.float()).sum(dim=2).to(torch.bfloat16)


def mhc_pre_apply_mix_bwd_ref(
    x: torch.Tensor,
    mix: torch.Tensor,
    o_grad: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_seqs, num_tokens, mhc, hidden = x.shape
    n = num_seqs * num_tokens

    o_grad_f = o_grad.float().view(n, hidden)
    x_f = x.float().view(n, mhc, hidden)
    mix_f = mix.view(n, mhc)

    x_grad_ref = (mix_f.unsqueeze(-1) * o_grad_f.unsqueeze(1)).to(torch.bfloat16)
    mix_grad_ref = (o_grad_f.unsqueeze(1) * x_f).sum(dim=-1)

    return x_grad_ref.view_as(x), mix_grad_ref.view_as(mix)


def test_fwd():
    device = "npu"
    configs = [
        (16, 1280, 4),
        (8192, 7680, 4),
    ]

    for num_tokens, hidden, mhc in configs:
        torch.manual_seed(42)
        x = torch.randn((1, num_tokens, mhc, hidden), dtype=torch.bfloat16, device=device)
        mix = torch.randn((1, num_tokens, mhc), dtype=torch.float32, device=device)

        o_ref = mhc_pre_apply_mix_fwd_ref(x, mix)
        o_tl = mhc_pre_apply_mix_fwd(x, mix)

        torch.testing.assert_close(o_tl, o_ref, atol=4e-2, rtol=1e-2)
        print("Kernel Output Match!")


def test_bwd():
    device = "npu"
    configs = [
        (16, 1280, 4),
        (8192, 7680, 4),
    ]

    for num_tokens, hidden, mhc in configs:
        torch.manual_seed(42)
        x = torch.randn((1, num_tokens, mhc, hidden), dtype=torch.bfloat16, device=device)
        mix = torch.randn((1, num_tokens, mhc), dtype=torch.float32, device=device)
        o_grad = torch.randn((1, num_tokens, hidden), dtype=torch.bfloat16, device=device)

        x_grad_ref, mix_grad_ref = mhc_pre_apply_mix_bwd_ref(x, mix, o_grad)
        x_grad_tl, mix_grad_tl = mhc_pre_apply_mix_bwd(x, mix, o_grad)

        torch.testing.assert_close(x_grad_tl, x_grad_ref, atol=4e-2, rtol=1e-2)
        torch.testing.assert_close(mix_grad_tl, mix_grad_ref, atol=4e-2, rtol=1e-2)
        print("Kernel Output Match!")


if __name__ == "__main__":
    test_fwd()
    test_bwd()
