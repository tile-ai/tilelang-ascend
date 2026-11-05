import math

import tilelang
import torch
from tilelang import language as T

VEC_NUM = 2
BATCH_SIZE = 8

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_pre_big_fuse(
    hidden_size: int,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 16,
    mhc_mult: int = 4,
):
    num_tokens = T.symbolic("num_tokens")
    dtype = "float32"
    dbtype = "bfloat16"
    mhc_mult2 = mhc_mult * mhc_mult
    mhc_mult3 = mhc_mult * 2 + mhc_mult2
    hidden_block = math.gcd(512, hidden_size)
    while hidden_size // hidden_block % VEC_NUM != 0 and hidden_block > VEC_NUM:
        hidden_block = hidden_block // 2

    pad_mhc3 = ((mhc_mult3 + 7) // 8) * 8
    pad_mhc = ((mhc_mult + 7) // 8) * 8
    pad_comb = ((mhc_mult * mhc_mult + 7) // 8) * 8
    pad_h_blk = ((hidden_block + 15) // 16) * 16
    pad_batch = ((BATCH_SIZE + 7) // 8) * 8
    pad_n_splits = ((n_splits + 7) // 8) * 8

    total_h_blocks = hidden_size // hidden_block
    blocks_per_vid = total_h_blocks // VEC_NUM

    inv_mhc_h = 1.0 / (mhc_mult * hidden_size)

    @T.prim_func
    def mhc_pre_big_fuse(
        gemm_out_mul: T.Tensor[(n_splits, num_tokens, mhc_mult3), dtype],
        gemm_out_sqrsum: T.Tensor[(n_splits, num_tokens), dtype],
        mhc_scale: T.Tensor[(3,), dtype],
        mhc_base: T.Tensor[(mhc_mult3,), dtype],
        residual: T.Tensor[(num_tokens, mhc_mult, hidden_size), dbtype],
        post_mix: T.Tensor[(num_tokens, mhc_mult), dtype],
        comb_mix: T.Tensor[(num_tokens, mhc_mult * mhc_mult), dtype],
        layer_input: T.Tensor[(num_tokens, hidden_size), dbtype],
    ) -> None:
        with T.Kernel(num_tokens // BATCH_SIZE, is_npu=True) as (batch_pid, vid):
            mixes_ub = T.alloc_ub((pad_mhc3,), dtype)
            scale_ub = T.alloc_ub((8,), dtype)
            base_ub = T.alloc_ub((pad_mhc3,), dtype)

            sqrsum_block_ub = T.alloc_ub((pad_n_splits, pad_batch), dtype)
            sqrsum_row_ub = T.alloc_ub((pad_batch,), dtype)
            sqrsum_accum_ub = T.alloc_ub((8,), dtype)
            rsqrt_src_ub = T.alloc_ub((8,), dtype)
            rsqrt_dst_ub = T.alloc_ub((8,), dtype)
            nr_tmp1 = T.alloc_ub((8,), dtype)

            mul_split_ub = T.alloc_ub((2, pad_mhc3), dtype)

            sigmoid_src_ub = T.alloc_ub((pad_mhc,), dtype)
            sigmoid_dst_ub = T.alloc_ub((pad_mhc,), dtype)
            base_slice_ub = T.alloc_ub((pad_mhc,), dtype)
            post_mix_ub = T.alloc_ub((pad_mhc,), dtype)
            pre_mix_cache_ub = T.alloc_ub((pad_mhc,), dtype)

            cm_ub = T.alloc_ub((mhc_mult, pad_mhc), dtype)
            row_max_ub = T.alloc_ub((pad_mhc,), dtype)
            row_sum_ub = T.alloc_ub((pad_mhc,), dtype)
            col_sum_ub = T.alloc_ub((1, pad_mhc), dtype)
            row_div_ub = T.alloc_ub((mhc_mult, pad_mhc), dtype)
            col_broadcast_ub = T.alloc_ub((mhc_mult, pad_mhc), dtype)
            comb_flat_ub = T.alloc_ub((pad_comb,), dtype)

            xs_ub = T.alloc_ub((2, mhc_mult, pad_h_blk), dbtype)
            xl_2d_ub = T.alloc_ub((mhc_mult, pad_h_blk), dtype)
            ol_f32_ub = T.alloc_ub((pad_h_blk,), dtype)
            ol_bf16_ub = T.alloc_ub((pad_h_blk,), dbtype)

            # norm_fn_fwd: init DMA + sqrsum reduce
            with T.Scope("V"):
                T.copy(mhc_scale[0:3], scale_ub[0:3])
                T.copy(mhc_base[0:mhc_mult3], base_ub[0:mhc_mult3])
                T.copy(
                    gemm_out_sqrsum[0:n_splits, batch_pid * BATCH_SIZE : (batch_pid + 1) * BATCH_SIZE],
                    sqrsum_block_ub[0:n_splits, 0:BATCH_SIZE],
                )

            T.set_flag("MTE2", "V", 0)
            T.wait_flag("MTE2", "V", 0)

            with T.Scope("V"):
                T.reduce_sum(sqrsum_block_ub, sqrsum_row_ub, dim=0, real_shape=[n_splits, BATCH_SIZE])

            for token_offset in T.serial(BATCH_SIZE):
                pid = batch_pid * BATCH_SIZE + token_offset

                # norm_fn_fwd: fill + rsqrt (Newton)
                with T.Scope("V"):
                    T.tile.fill(mixes_ub, 0.0)
                    T.tile.fill(sqrsum_accum_ub, 0.0)
                    sqrsum_accum_ub[0] = sqrsum_row_ub[token_offset]
                    T.tile.mul(rsqrt_src_ub, sqrsum_accum_ub, inv_mhc_h)
                    T.tile.add(rsqrt_src_ub, rsqrt_src_ub, rms_eps)
                    T.tile.rsqrt(rsqrt_dst_ub, rsqrt_src_ub)

                    T.tile.mul(nr_tmp1, rsqrt_dst_ub, rsqrt_dst_ub)
                    T.tile.mul(nr_tmp1, rsqrt_src_ub, nr_tmp1)
                    T.tile.mul(nr_tmp1, nr_tmp1, -1.0)
                    T.tile.add(nr_tmp1, nr_tmp1, 3.0)
                    T.tile.mul(rsqrt_dst_ub, rsqrt_dst_ub, nr_tmp1)
                    T.tile.mul(rsqrt_dst_ub, rsqrt_dst_ub, 0.5)

                # norm_fn_fwd: mul accumulate (double buffer)
                T.copy(gemm_out_mul[0, pid, 0:mhc_mult3], mul_split_ub[0, :])
                T.set_flag("MTE2", "V", 0)

                for i_split in T.serial(n_splits):
                    buf_pid = i_split % 2

                    T.wait_flag("MTE2", "V", buf_pid)

                    if i_split + 1 < n_splits:
                        next_pid = (i_split + 1) % 2
                        T.copy(gemm_out_mul[i_split + 1, pid, 0:mhc_mult3], mul_split_ub[next_pid, :])
                        T.set_flag("MTE2", "V", next_pid)

                    T.tile.add(mixes_ub, mixes_ub, mul_split_ub[buf_pid, :])

                with T.Scope("V"):
                    T.tile.mul(mixes_ub, mixes_ub, rsqrt_dst_ub[0])
                    for i in T.serial(mhc_mult):
                        sigmoid_src_ub[i] = mixes_ub[mhc_mult + i]
                        base_slice_ub[i] = base_ub[mhc_mult + i]
                    T.tile.mul(sigmoid_src_ub, sigmoid_src_ub, scale_ub[1])
                    T.tile.add(sigmoid_src_ub, sigmoid_src_ub, base_slice_ub)
                    T.tile.sigmoid(sigmoid_dst_ub, sigmoid_src_ub)
                    T.tile.mul(post_mix_ub, sigmoid_dst_ub, mhc_post_mult_value)

                if vid == 0:
                    T.set_flag("V", "MTE3", 3)
                    T.wait_flag("V", "MTE3", 3)
                    T.copy(post_mix_ub[0:mhc_mult], post_mix[pid, 0:mhc_mult])

                with T.Scope("V"):
                    T.copy(mixes_ub[0:pad_mhc], sigmoid_src_ub[0:pad_mhc])
                    T.copy(base_ub[0:pad_mhc], base_slice_ub[0:pad_mhc])
                    T.tile.mul(sigmoid_src_ub, sigmoid_src_ub, scale_ub[0])
                    T.tile.add(sigmoid_src_ub, sigmoid_src_ub, base_slice_ub)
                    T.tile.sigmoid(sigmoid_dst_ub, sigmoid_src_ub)
                    T.tile.add(pre_mix_cache_ub, sigmoid_dst_ub, mhc_pre_eps)

                    T.tile.mul(mixes_ub, mixes_ub, scale_ub[2])
                    T.tile.add(mixes_ub, mixes_ub, base_ub)

                    T.tile.fill(cm_ub, -1e10)
                    for j in T.serial(mhc_mult):
                        for k in T.serial(mhc_mult):
                            cm_ub[j, k] = mixes_ub[j * mhc_mult + k + 2 * mhc_mult]

                with T.Scope("V"):
                    T.reduce_max(cm_ub, row_max_ub, dim=-1, real_shape=[mhc_mult, mhc_mult])
                    for i in T.serial(mhc_mult):
                        T.tile.fill(row_div_ub[i, :], row_max_ub[i])
                    T.tile.sub(cm_ub, cm_ub, row_div_ub)
                    T.tile.exp(cm_ub, cm_ub)

                    T.reduce_sum(cm_ub, row_sum_ub, dim=-1, real_shape=[mhc_mult, mhc_mult])
                    T.tile.add(row_sum_ub, row_sum_ub, mhc_sinkhorn_eps)
                    for i in T.serial(mhc_mult):
                        T.tile.fill(row_div_ub[i, :], row_sum_ub[i])
                    T.tile.div(cm_ub, cm_ub, row_div_ub)

                    T.reduce_sum(cm_ub, col_sum_ub, dim=0, real_shape=[mhc_mult, pad_mhc])
                    T.tile.add(col_sum_ub, col_sum_ub, mhc_sinkhorn_eps)
                    T.tile.broadcast(col_broadcast_ub, col_sum_ub)
                    T.tile.div(cm_ub, cm_ub, col_broadcast_ub)

                    for _ in T.serial(sinkhorn_repeat - 1):
                        T.reduce_sum(cm_ub, row_sum_ub, dim=-1, real_shape=[mhc_mult, mhc_mult])
                        T.tile.add(row_sum_ub, row_sum_ub, mhc_sinkhorn_eps)
                        for i in T.serial(mhc_mult):
                            T.tile.fill(row_div_ub[i, :], row_sum_ub[i])
                        T.tile.div(cm_ub, cm_ub, row_div_ub)

                        T.reduce_sum(cm_ub, col_sum_ub, dim=0, real_shape=[mhc_mult, pad_mhc])
                        T.tile.add(col_sum_ub, col_sum_ub, mhc_sinkhorn_eps)
                        T.tile.broadcast(col_broadcast_ub, col_sum_ub)
                        T.tile.div(cm_ub, cm_ub, col_broadcast_ub)

                    for j in T.serial(mhc_mult):
                        for k in T.serial(mhc_mult):
                            comb_flat_ub[j * mhc_mult + k] = cm_ub[j, k]

                # sinkhorn_fwd: comb_mix output
                if vid == 0:
                    T.set_flag("V", "MTE3", 4)
                    T.wait_flag("V", "MTE3", 4)
                    T.copy(comb_flat_ub[0 : mhc_mult * mhc_mult], comb_mix[pid, 0 : mhc_mult * mhc_mult])

                # pre_apply_mix_fwd: layer-input (double buffer)
                i0_h_first = vid * blocks_per_vid
                T.copy(residual[pid, 0:mhc_mult, i0_h_first * hidden_block : (i0_h_first + 1) * hidden_block], xs_ub[0, :, :])
                T.set_flag("MTE2", "V", 5)

                for i0_h_local in T.serial(blocks_per_vid):
                    i0_h = vid * blocks_per_vid + i0_h_local
                    buf_pid = i0_h_local % 2

                    T.wait_flag("MTE2", "V", 5 + buf_pid)

                    if i0_h_local + 1 < blocks_per_vid:
                        next_pid = (i0_h_local + 1) % 2
                        next_i0_h = vid * blocks_per_vid + i0_h_local + 1
                        T.copy(residual[pid, 0:mhc_mult, next_i0_h * hidden_block : (next_i0_h + 1) * hidden_block], xs_ub[next_pid, :, :])
                        T.set_flag("MTE2", "V", 5 + next_pid)

                    with T.Scope("V"):
                        T.tile.cast(xl_2d_ub, xs_ub[buf_pid, :, :], "CAST_NONE", mhc_mult * pad_h_blk)

                        T.tile.mul(ol_f32_ub, xl_2d_ub[0, :], pre_mix_cache_ub[0])
                        for i_mhc in T.serial(mhc_mult - 1):
                            T.tile.axpy(ol_f32_ub, xl_2d_ub[i_mhc + 1, :], pre_mix_cache_ub[i_mhc + 1])

                        T.tile.cast(ol_bf16_ub, ol_f32_ub, "CAST_ROUND", pad_h_blk)

                    T.set_flag("V", "MTE3", 6)
                    T.wait_flag("V", "MTE3", 6)
                    T.copy(ol_bf16_ub[0:hidden_block], layer_input[pid, i0_h * hidden_block : (i0_h + 1) * hidden_block])

    return mhc_pre_big_fuse


def _round_to_tf32(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    return (x.view(torch.int32) + 0x1000).view(torch.float32)


def generate_big_fuse_test_data(
    n1: int,
    mhc_mult: int,
    hidden_size: int,
    rms_eps: float = 1e-6,
    mhc_pre_eps: float = 1e-6,
    mhc_sinkhorn_eps: float = 1e-6,
    mhc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 10,
    n_splits: int = 16,
) -> dict[str, torch.Tensor | float]:
    n0 = 1
    mhc_mult2 = mhc_mult * mhc_mult
    mhc_mult3 = mhc_mult * 2 + mhc_mult2
    device = "npu"

    residual = (
        torch.randn((n0, n1, mhc_mult, hidden_size), dtype=torch.float, device=device)
        .mul(1 + torch.arange(mhc_mult, device=device).mul(0.01).view(1, 1, -1, 1))
        .bfloat16()
    )

    fn = (
        (
            torch.randn((mhc_mult3, mhc_mult, hidden_size), dtype=torch.float, device=device)
            * 1e-4
            * (1 + torch.arange(mhc_mult, device=device).mul(0.01).view(1, 1, -1, 1))
        )
        .flatten(1, 2)
        .contiguous()
    )

    mhc_scale = torch.randn((3,), dtype=torch.float, device=device) * 0.1
    mhc_base = torch.randn((mhc_mult3,), dtype=torch.float, device=device) * 0.1

    return {
        "residual": residual,
        "fn": fn,
        "mhc_scale": mhc_scale,
        "mhc_base": mhc_base,
        "rms_eps": rms_eps,
        "mhc_pre_eps": mhc_pre_eps,
        "mhc_sinkhorn_eps": mhc_sinkhorn_eps,
        "mhc_post_mult_value": mhc_post_mult_value,
        "sinkhorn_repeat": sinkhorn_repeat,
        "n_splits": n_splits,
    }


def big_fuse_reference(
    residual: torch.Tensor,
    fn: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mhc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    mhc_mult2 = mhc_mult * mhc_mult
    mhc_mult3 = mhc_mult * 2 + mhc_mult2
    outer_shape = residual.shape[:-2]
    mhc_hidden_size = mhc_mult * hidden_size

    residual_cpu = residual.float().cpu().reshape(-1, mhc_hidden_size)
    fn_2d = fn.cpu().contiguous().reshape(mhc_mult3, mhc_hidden_size)
    fn_rounded = _round_to_tf32(fn_2d)
    mhc_scale_cpu = mhc_scale.cpu()
    mhc_base_cpu = mhc_base.cpu()

    gemm_result = residual_cpu @ fn_rounded.transpose(0, 1)
    sqrsum = (residual_cpu**2).sum(dim=-1)

    inv_mhc_h = 1.0 / mhc_hidden_size
    norm_factor = torch.rsqrt(sqrsum * inv_mhc_h + rms_eps)
    mixes = gemm_result * norm_factor.unsqueeze(-1)

    mixes = mixes.reshape(*outer_shape, mhc_mult3)

    mhc_scale_expanded = torch.cat(
        [
            mhc_scale_cpu[0].unsqueeze(0).expand(mhc_mult),
            mhc_scale_cpu[1].unsqueeze(0).expand(mhc_mult),
            mhc_scale_cpu[2].unsqueeze(0).expand(mhc_mult * mhc_mult),
        ]
    )
    mixes_scaled = mixes * mhc_scale_expanded + mhc_base_cpu

    pre_mix = (mixes_scaled[..., :mhc_mult].sigmoid() + mhc_pre_eps).unsqueeze(-1)
    post_mix = (mixes_scaled[..., mhc_mult : 2 * mhc_mult].sigmoid() * mhc_post_mult_value).unsqueeze(-1)
    comb_mix_raw = mixes_scaled[..., 2 * mhc_mult :].reshape(*outer_shape, mhc_mult, mhc_mult)

    comb_mix_softmax = torch.softmax(comb_mix_raw, dim=-1) + mhc_sinkhorn_eps
    col_sum = comb_mix_softmax.sum(dim=-2) + mhc_sinkhorn_eps
    comb_mix = comb_mix_softmax / col_sum.unsqueeze(-2)

    for _ in range(sinkhorn_repeat - 1):
        row_sum = comb_mix.sum(dim=-1) + mhc_sinkhorn_eps
        comb_mix = comb_mix / row_sum.unsqueeze(-1)
        col_sum = comb_mix.sum(dim=-2) + mhc_sinkhorn_eps
        comb_mix = comb_mix / col_sum.unsqueeze(-2)

    residual_cpu_bf16 = residual.cpu().bfloat16()
    layer_input = (pre_mix * residual_cpu_bf16).sum(dim=-2, keepdim=False).bfloat16()

    return post_mix.npu(), comb_mix.npu(), layer_input.npu()


def test_correctness(
    n1: int,
    hidden_size: int,
    mhc_mult: int,
) -> None:
    assert n1 % BATCH_SIZE == 0, f"num_tokens={n1} must be divisible by BATCH_SIZE={BATCH_SIZE}"

    test_data = generate_big_fuse_test_data(
        n1=n1,
        mhc_mult=mhc_mult,
        hidden_size=hidden_size,
    )

    residual = test_data["residual"]
    fn = test_data["fn"]
    mhc_scale = test_data["mhc_scale"]
    mhc_base = test_data["mhc_base"]

    mhc_mult2 = mhc_mult * mhc_mult
    mhc_mult3 = mhc_mult * 2 + mhc_mult2
    hidden_size_val = residual.shape[-1]
    mhc_hidden_size = mhc_mult * hidden_size_val

    outer_shape = residual.shape[:-2]
    residual_flat = residual.view(-1, mhc_mult, hidden_size_val)
    num_tokens = residual_flat.shape[0]

    n_splits_actual = 1

    residual_cpu = residual.float().cpu().reshape(num_tokens, mhc_hidden_size)
    fn_2d = fn.cpu().contiguous().reshape(mhc_mult3, mhc_hidden_size)
    fn_rounded = _round_to_tf32(fn_2d)
    gemm_result = residual_cpu @ fn_rounded.transpose(0, 1)
    sqrsum = (residual_cpu**2).sum(dim=-1)

    gemm_out_mul = gemm_result.unsqueeze(0).npu()
    gemm_out_sqrsum = sqrsum.unsqueeze(0).npu()

    post_mix_out = torch.empty(num_tokens, mhc_mult, dtype=torch.float32, device=residual.device)
    comb_mix_out = torch.empty(num_tokens, mhc_mult2, dtype=torch.float32, device=residual.device)
    layer_input_out = torch.empty(num_tokens, hidden_size_val, dtype=torch.bfloat16, device=residual.device)

    kernel = _mhc_pre_big_fuse(
        hidden_size=hidden_size_val,
        rms_eps=test_data["rms_eps"],
        mhc_pre_eps=test_data["mhc_pre_eps"],
        mhc_sinkhorn_eps=test_data["mhc_sinkhorn_eps"],
        mhc_post_mult_value=test_data["mhc_post_mult_value"],
        sinkhorn_repeat=test_data["sinkhorn_repeat"],
        n_splits=n_splits_actual,
        mhc_mult=mhc_mult,
    )

    kernel(
        gemm_out_mul,
        gemm_out_sqrsum,
        mhc_scale,
        mhc_base,
        residual_flat,
        post_mix_out,
        comb_mix_out,
        layer_input_out,
    )

    post_mix_fused = post_mix_out.view(*outer_shape, mhc_mult, 1)
    comb_mix_fused = comb_mix_out.view(*outer_shape, mhc_mult, mhc_mult)
    layer_input_fused = layer_input_out.view(*outer_shape, hidden_size_val)

    post_mix_ref, comb_mix_ref, layer_input_ref = big_fuse_reference(
        residual,
        fn,
        mhc_scale,
        mhc_base,
        test_data["rms_eps"],
        test_data["mhc_pre_eps"],
        test_data["mhc_sinkhorn_eps"],
        test_data["mhc_post_mult_value"],
        test_data["sinkhorn_repeat"],
        n_splits_actual,
    )

    torch.testing.assert_close(post_mix_fused, post_mix_ref, atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(comb_mix_fused, comb_mix_ref, atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(layer_input_fused, layer_input_ref, atol=4e-2, rtol=1e-2)
    print("Kernel Output Match!")


if __name__ == "__main__":
    test_configs = [
        (512, 1280, 4),
        (512, 2560, 4),
        (512, 512, 4),
        (256, 1280, 4),
        (64, 1280, 4),
    ]

    for n1, hidden_size, mhc_mult in test_configs:
        test_correctness(n1=n1, hidden_size=hidden_size, mhc_mult=mhc_mult)
