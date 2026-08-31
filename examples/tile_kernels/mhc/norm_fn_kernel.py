import tilelang
import torch
from tilelang import language as T

VEC_NUM = 2

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

_EXPERT_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_fn_normw_merge_fwd(m: int, n: int, dtype: torch.dtype = "float32") -> tilelang.JITKernel:
    n_blk = 256
    pad_n_blk = 256
    sub_n_blk = pad_n_blk // VEC_NUM

    total_grid_blocks = T.ceildiv(n, n_blk)

    @T.prim_func
    def _mhc_fn_normw_merge_fwd_(
        fn: T.Tensor[(m, n), dtype],
        normw: T.Tensor[n, dtype],
        out_fn: T.Tensor[(m, n), dtype],
    ) -> None:
        with T.Kernel(total_grid_blocks, is_npu=True) as (cid, vid):
            pid_n = cid
            n_offset = pid_n * n_blk + vid * sub_n_blk
            cur_n_blk = T.min(sub_n_blk, n - n_offset)

            fn_ub = T.alloc_ub((sub_n_blk,), dtype)
            normw_ub = T.alloc_ub((sub_n_blk,), dtype)
            out_ub = T.alloc_ub((sub_n_blk,), dtype)

            if cur_n_blk > 0:
                T.copy(normw[n_offset : n_offset + cur_n_blk], normw_ub[0:cur_n_blk])

                for i_m in T.unroll(m):
                    T.copy(fn[i_m, n_offset : n_offset + cur_n_blk], fn_ub[0:cur_n_blk])
                    T.tile.mul(out_ub, fn_ub, normw_ub)
                    T.copy(out_ub[0:cur_n_blk], out_fn[i_m, n_offset : n_offset + cur_n_blk])

    return _mhc_fn_normw_merge_fwd_


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_fn_normw_merge_bwd(m: int, n: int, dtype: torch.dtype = "float32") -> tilelang.JITKernel:
    n_blk = 256
    pad_n_blk = 256
    sub_n_blk = pad_n_blk // VEC_NUM

    @T.prim_func
    def _mhc_fn_normw_merge_bwd_(
        fn: T.Tensor[(m, n), dtype],
        normw: T.Tensor[n, dtype],
        out_fn_grad: T.Tensor[(m, n), dtype],
        fn_grad: T.Tensor[(m, n), dtype],
        normw_grad: T.Tensor[n, dtype],
    ) -> None:
        with T.Kernel(T.ceildiv(n, n_blk), is_npu=True) as (cid_n, vid):
            pid_n = cid_n

            n_offset = pid_n * n_blk + vid * sub_n_blk
            cur_n_blk = T.min(sub_n_blk, n - n_offset)

            normw_ub = T.alloc_ub((sub_n_blk,), dtype)
            normw_grad_ub = T.alloc_ub((sub_n_blk,), dtype)
            fn_ub = T.alloc_ub((sub_n_blk,), dtype)
            out_fn_grad_ub = T.alloc_ub((sub_n_blk,), dtype)
            fn_grad_ub = T.alloc_ub((sub_n_blk,), dtype)
            mul_ub = T.alloc_ub((sub_n_blk,), dtype)

            if cur_n_blk > 0:
                with T.Scope("V"):
                    T.tile.fill(normw_ub, 0.0)
                    T.tile.fill(normw_grad_ub, 0.0)
                    T.copy(normw[n_offset : n_offset + cur_n_blk], normw_ub[0:cur_n_blk])

                for i_m in T.serial(m):
                    for k in T.serial(cur_n_blk):
                        out_fn_grad_ub[k] = out_fn_grad[i_m, n_offset + k]
                        fn_ub[k] = fn[i_m, n_offset + k]
                        fn_grad_ub[k] = fn_grad[i_m, n_offset + k]

                    with T.Scope("V"):
                        T.tile.mul(mul_ub, out_fn_grad_ub, normw_ub)
                        T.tile.add(fn_grad_ub, fn_grad_ub, mul_ub)

                    with T.Scope("V"):
                        T.tile.mul(mul_ub, out_fn_grad_ub, fn_ub)
                        T.tile.add(normw_grad_ub, normw_grad_ub, mul_ub)

                    with T.Scope("V"):
                        T.copy(fn_grad_ub[0:cur_n_blk], fn_grad[i_m, n_offset : n_offset + cur_n_blk])

                with T.Scope("V"):
                    T.copy(normw_grad_ub[0:cur_n_blk], normw_grad[n_offset : n_offset + cur_n_blk])

    return _mhc_fn_normw_merge_bwd_


@tilelang.jit(pass_configs=_EXPERT_PASS_CONFIGS)
def _mhc_pre_norm_fn_fwd_fused(
    mhc_mult3: int,
    n_rms_group: int,
    rms_group_size: int,
    rms_eps: float,
    token_block: int = 32,
    hidden_block: int = 256,
) -> tilelang.JITKernel:
    assert mhc_mult3 <= 32
    num_tokens = T.symbolic("num_tokens")
    dtype = "float32"
    dbtype = "bfloat16"
    assert rms_group_size % hidden_block == 0

    pad_mhc = T.ceildiv(mhc_mult3, 8) * 8
    pad_mhc3_cube = ((mhc_mult3 + 15) // 16) * 16
    total_grid_blocks = T.ceildiv(num_tokens, token_block) * n_rms_group
    y_blocks_num = n_rms_group

    @T.prim_func
    def _mhc_pre_norm_fn_fwd_fused_kernel(
        x: T.Tensor[(num_tokens, n_rms_group * rms_group_size), dbtype],
        fn: T.Tensor[(pad_mhc3_cube, n_rms_group * rms_group_size), dbtype],
        sqrsum: T.Tensor[(num_tokens, n_rms_group), dtype],
        out_mul: T.Tensor[(num_tokens, n_rms_group, mhc_mult3), dtype],
        out: T.Tensor[(num_tokens * n_rms_group, mhc_mult3), dtype],
        workspace_gemm: T.Tensor[(total_grid_blocks, token_block, pad_mhc3_cube), dtype],
    ) -> None:
        with T.Kernel(total_grid_blocks, is_npu=True) as (cid, vid):
            pid_x = cid // y_blocks_num
            pid_y = cid % y_blocks_num
            base_row = pid_x * token_block
            cur_token_block = T.min(token_block, num_tokens - base_row)

            x_L1 = T.alloc_L1((token_block, hidden_block), dbtype)
            fn_L1 = T.alloc_L1((pad_mhc3_cube, hidden_block), dbtype)
            out_L0C = T.alloc_L0C((token_block, pad_mhc3_cube), dtype)

            x_bf16_ub = T.alloc_ub((token_block, hidden_block), dbtype)
            x_f32_ub = T.alloc_ub((token_block, hidden_block), dtype)
            sqrsum_mul_ub = T.alloc_ub((token_block, hidden_block), dtype)
            sqrsum_part_ub = T.alloc_ub((token_block,), dtype)
            sqrsum_tmp = T.alloc_ub((token_block,), dtype)

            workspace_read_ub = T.alloc_ub((token_block, pad_mhc3_cube), dtype)
            out_mul_row_ub = T.alloc_ub((pad_mhc,), dtype)
            out_row_ub = T.alloc_ub((pad_mhc,), dtype)
            rsqrt_broadcast_ub = T.alloc_ub((pad_mhc,), dtype)

            sqrsum_accum_ub = T.alloc_ub((8,), dtype)
            rsqrt_src_ub = T.alloc_ub((8,), dtype)
            rsqrt_dst_ub = T.alloc_ub((8,), dtype)
            tmp_scalar_buf = T.alloc_ub((8,), dtype)
            nr_tmp1 = T.alloc_ub((8,), dtype)
            nr_tmp2 = T.alloc_ub((8,), dtype)
            inv_rms_ub = T.alloc_ub((8,), dtype)
            sqrsum_tmp_ub = T.alloc_ub((1,), dtype)

            with T.Scope("C"):
                for pz in T.serial(rms_group_size // hidden_block):
                    global_h_offset = pid_y * rms_group_size + pz * hidden_block
                    T.copy(
                        x[base_row : base_row + cur_token_block, global_h_offset : global_h_offset + hidden_block],
                        x_L1[0:cur_token_block, 0:hidden_block],
                    )
                    T.copy(fn[0:pad_mhc3_cube, global_h_offset : global_h_offset + hidden_block], fn_L1[0:pad_mhc3_cube, 0:hidden_block])

                    T.gemm_v0(x_L1, fn_L1, out_L0C, transpose_B=True, init=(pz == 0))

                T.copy(out_L0C, workspace_gemm[cid, :, :])
                T.set_cross_flag("FIX", 0)
                T.wait_cross_flag(1)

            with T.Scope("V"):
                T.wait_cross_flag(0)

                T.copy(workspace_gemm[cid, 0:cur_token_block, :], workspace_read_ub[0:cur_token_block, :])

                T.tile.fill(sqrsum_part_ub, 0.0)
                T.tile.fill(inv_rms_ub, 1.0 / T.cast(rms_group_size, "float32"))

                for pz in T.serial(rms_group_size // hidden_block):
                    global_h_offset = pid_y * rms_group_size + pz * hidden_block
                    T.copy(
                        x[base_row : base_row + cur_token_block, global_h_offset : global_h_offset + hidden_block],
                        x_bf16_ub[0:cur_token_block, 0:hidden_block],
                    )
                    T.tile.cast(x_f32_ub, x_bf16_ub, "CAST_NONE", token_block * hidden_block)
                    T.tile.mul(sqrsum_mul_ub, x_f32_ub, x_f32_ub)
                    T.reduce_sum(sqrsum_mul_ub, sqrsum_tmp, dim=-1, clear=True, real_shape=[token_block, hidden_block])
                    T.tile.add(sqrsum_part_ub, sqrsum_part_ub, sqrsum_tmp)

                for i_t in T.serial(cur_token_block):
                    token_idx = base_row + i_t
                    T.copy(
                        workspace_read_ub[i_t : i_t + 1, 0:mhc_mult3],
                        out_mul[token_idx : token_idx + 1, pid_y : pid_y + 1, 0:mhc_mult3],
                    )

                for i_t in T.serial(cur_token_block):
                    token_idx = base_row + i_t
                    sqrsum_tmp_ub[0] = sqrsum_part_ub[i_t]
                    T.copy(sqrsum_tmp_ub[0:1], sqrsum[token_idx : token_idx + 1, pid_y : pid_y + 1])

                for i_t in T.serial(cur_token_block):
                    token_idx = base_row + i_t

                    sqrsum_tmp_ub[0] = sqrsum_part_ub[i_t]
                    T.tile.fill(sqrsum_accum_ub, sqrsum_tmp_ub[0])
                    T.tile.mul(rsqrt_src_ub, sqrsum_accum_ub, inv_rms_ub)
                    T.tile.fill(tmp_scalar_buf, rms_eps)
                    T.tile.add(rsqrt_src_ub, rsqrt_src_ub, tmp_scalar_buf)
                    T.tile.rsqrt(rsqrt_dst_ub, rsqrt_src_ub)

                    T.tile.mul(nr_tmp1, rsqrt_dst_ub, rsqrt_dst_ub)
                    T.tile.mul(nr_tmp1, rsqrt_src_ub, nr_tmp1)
                    T.tile.fill(nr_tmp2, 3.0)
                    T.tile.sub(nr_tmp1, nr_tmp2, nr_tmp1)
                    T.tile.mul(rsqrt_dst_ub, rsqrt_dst_ub, nr_tmp1)
                    T.tile.fill(nr_tmp1, 0.5)
                    T.tile.mul(rsqrt_dst_ub, rsqrt_dst_ub, nr_tmp1)

                    T.copy(workspace_read_ub[i_t : i_t + 1, 0:mhc_mult3], out_mul_row_ub[0:mhc_mult3])
                    T.tile.fill(rsqrt_broadcast_ub, rsqrt_dst_ub[0])
                    T.tile.mul(out_row_ub, out_mul_row_ub, rsqrt_broadcast_ub)

                    global_row_idx = token_idx * n_rms_group + pid_y
                    T.copy(out_row_ub[0:mhc_mult3], out[global_row_idx : global_row_idx + 1, 0:mhc_mult3])

                T.set_cross_flag("V", 1)

    return _mhc_pre_norm_fn_fwd_fused_kernel


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_pre_norm_fn_bwd_fused(
    mhc_mult3: int,
    n_rms_group: int,
    rms_group_size: int,
    rms_eps: float,
    token_block: int = 64,
    hidden_block: int = 64,
) -> tilelang.JITKernel:
    assert mhc_mult3 <= 32
    num_tokens = T.symbolic("num_tokens")
    dtype = "float32"
    dbtype = "bfloat16"
    assert rms_group_size % hidden_block == 0

    assert hidden_block == 64
    assert token_block == 64
    assert token_block % VEC_NUM == 0

    sub_blk = token_block // VEC_NUM

    z_blocks_num = T.ceildiv(rms_group_size, hidden_block)
    total_grid_blocks = n_rms_group * z_blocks_num

    pad_mhc3 = ((mhc_mult3 + 7) // 8) * 8

    @T.prim_func
    def _mhc_pre_norm_fn_bwd_fused_kernel(
        out_grad: T.Tensor[(num_tokens, mhc_mult3), dtype],
        out_mul: T.Tensor[(num_tokens, n_rms_group, mhc_mult3), dtype],
        sqrsum: T.Tensor[(num_tokens, n_rms_group), dtype],
        x: T.Tensor[(num_tokens, n_rms_group * rms_group_size), dbtype],
        fn: T.Tensor[(mhc_mult3, n_rms_group * rms_group_size), dtype],
        x_grad: T.Tensor[(num_tokens, n_rms_group * rms_group_size), dbtype],
        fn_grad_partial: T.Tensor[(VEC_NUM, mhc_mult3, n_rms_group * rms_group_size), dtype],
    ) -> None:
        with T.Kernel(total_grid_blocks, is_npu=True) as (cid, vid):
            pid_y = cid // z_blocks_num
            pid_z = cid % z_blocks_num
            yz = pid_y * rms_group_size + pid_z * hidden_block

            local_fn_2d = T.alloc_ub((mhc_mult3, 64), dtype)
            fn_grad_local = T.alloc_ub((mhc_mult3, 64), dtype)
            xgm_partial_ub = T.alloc_ub((sub_blk, 64), dtype)

            x_bf16_ub = T.alloc_ub((sub_blk, 64), dbtype)
            x_f32_ub = T.alloc_ub((sub_blk, 64), dtype)
            out_grad_ub = T.alloc_ub((sub_blk, pad_mhc3), dtype)
            out_mul_ub = T.alloc_ub((sub_blk, pad_mhc3), dtype)
            sqrsum_ub = T.alloc_ub((sub_blk,), dtype)
            out_mul_grad_ub = T.alloc_ub((sub_blk, pad_mhc3), dtype)
            sqrsum_grad_ub = T.alloc_ub((sub_blk,), dtype)

            rsqrt_src_ub = T.alloc_ub((sub_blk,), dtype)
            rsqrt_dst_ub = T.alloc_ub((sub_blk,), dtype)
            nr_tmp1 = T.alloc_ub((sub_blk,), dtype)
            nr_tmp2 = T.alloc_ub((sub_blk,), dtype)
            inv_rms_ub = T.alloc_ub((sub_blk,), dtype)
            inv_neg_half_rms_ub = T.alloc_ub((sub_blk,), dtype)
            rms_cube_ub = T.alloc_ub((sub_blk,), dtype)

            mul_batch_ub = T.alloc_ub((sub_blk, pad_mhc3), dtype)
            sqrsum_reduce_ub = T.alloc_ub((sub_blk,), dtype)

            grad_bc_2d = T.alloc_ub((sub_blk, 64), dtype)
            fn_bc_2d = T.alloc_ub((sub_blk, 64), dtype)
            batch_ub = T.alloc_ub((sub_blk, 64), dtype)
            fn_grad_row_ub = T.alloc_ub((1, 64), dtype)

            x_grad_f32_ub = T.alloc_ub((sub_blk, 64), dtype)
            x_grad_bf16_ub = T.alloc_ub((sub_blk, 64), dbtype)

            T.tile.fill(inv_rms_ub, 1.0 / T.cast(rms_group_size, dtype))
            T.tile.fill(inv_neg_half_rms_ub, -0.5 / T.cast(rms_group_size, dtype))
            T.tile.fill(local_fn_2d, 0.0)
            for j_m3 in T.serial(mhc_mult3):
                T.copy(fn[j_m3, yz : yz + 64], local_fn_2d[j_m3, 0:64])

            T.tile.fill(fn_grad_local, 0.0)

            for px in T.serial(T.ceildiv(num_tokens, token_block)):
                token_base = px * token_block

                T.tile.fill(x_bf16_ub, T.cast(0.0, dbtype))
                T.tile.fill(x_f32_ub, 0.0)
                T.tile.fill(out_grad_ub, 0.0)
                T.tile.fill(out_mul_ub, 0.0)
                T.tile.fill(sqrsum_ub, 0.0)
                T.tile.fill(xgm_partial_ub, 0.0)

                for i_t in T.serial(sub_blk):
                    token_idx = token_base + vid * sub_blk + i_t
                    if token_idx < num_tokens:
                        T.copy(x[token_idx, yz : yz + 64], x_bf16_ub[i_t, 0:64])
                        T.copy(out_grad[token_idx, 0:mhc_mult3], out_grad_ub[i_t, 0:mhc_mult3])
                        T.copy(out_mul[token_idx, pid_y, 0:mhc_mult3], out_mul_ub[i_t, 0:mhc_mult3])
                        sqrsum_ub[i_t] = sqrsum[token_idx, pid_y]

                T.tile.cast(x_f32_ub, x_bf16_ub, "CAST_NONE", sub_blk * 64)

                T.tile.mul(rsqrt_src_ub, sqrsum_ub, inv_rms_ub)
                T.tile.fill(nr_tmp2, rms_eps)
                T.tile.add(rsqrt_src_ub, rsqrt_src_ub, nr_tmp2)
                T.tile.rsqrt(rsqrt_dst_ub, rsqrt_src_ub)

                T.tile.mul(nr_tmp1, rsqrt_dst_ub, rsqrt_dst_ub)
                T.tile.mul(nr_tmp1, rsqrt_src_ub, nr_tmp1)
                T.tile.fill(nr_tmp2, 3.0)
                T.tile.sub(nr_tmp1, nr_tmp2, nr_tmp1)
                T.tile.mul(rsqrt_dst_ub, rsqrt_dst_ub, nr_tmp1)
                T.tile.fill(nr_tmp1, 0.5)
                T.tile.mul(rsqrt_dst_ub, rsqrt_dst_ub, nr_tmp1)

                for i_t in T.serial(sub_blk):
                    T.tile.mul(out_mul_grad_ub[i_t, :], out_grad_ub[i_t, :], rsqrt_dst_ub[i_t])

                T.tile.mul(mul_batch_ub, out_grad_ub, out_mul_ub)
                T.reduce_sum(mul_batch_ub, sqrsum_reduce_ub, dim=-1, clear=True, real_shape=[sub_blk, pad_mhc3])

                T.tile.mul(rms_cube_ub, rsqrt_dst_ub, rsqrt_dst_ub)
                T.tile.mul(rms_cube_ub, rms_cube_ub, rsqrt_dst_ub)

                T.tile.mul(sqrsum_grad_ub, sqrsum_reduce_ub, rms_cube_ub)
                T.tile.mul(sqrsum_grad_ub, sqrsum_grad_ub, inv_neg_half_rms_ub)

                for j in T.serial(mhc_mult3):
                    for i_t in T.serial(sub_blk):
                        T.tile.fill(grad_bc_2d[i_t, :], out_mul_grad_ub[i_t, j])

                    T.tile.mul(batch_ub, x_f32_ub, grad_bc_2d)
                    T.reduce_sum(batch_ub, fn_grad_row_ub, dim=0, clear=True, real_shape=[sub_blk, 64])
                    T.tile.add(fn_grad_local[j, :], fn_grad_local[j, :], fn_grad_row_ub[0, :])

                    for i_t in T.serial(sub_blk):
                        T.copy(local_fn_2d[j, :], fn_bc_2d[i_t, :])

                    T.tile.mul(batch_ub, fn_bc_2d, grad_bc_2d)
                    T.tile.add(xgm_partial_ub, xgm_partial_ub, batch_ub)

                T.tile.fill(nr_tmp2, 2.0)
                T.tile.mul(sqrsum_grad_ub, sqrsum_grad_ub, nr_tmp2)

                for i_t in T.serial(sub_blk):
                    T.tile.mul(x_grad_f32_ub[i_t, :], x_f32_ub[i_t, :], sqrsum_grad_ub[i_t])

                T.tile.add(x_grad_f32_ub, xgm_partial_ub, x_grad_f32_ub)

                T.tile.cast(x_grad_bf16_ub, x_grad_f32_ub, "CAST_ROUND", sub_blk * 64)

                for i_t in T.serial(sub_blk):
                    token_idx = token_base + vid * sub_blk + i_t
                    if token_idx < num_tokens:
                        T.copy(x_grad_bf16_ub[i_t, 0:64], x_grad[token_idx, yz : yz + 64])

            for j in T.serial(mhc_mult3):
                T.copy(fn_grad_local[j, :], fn_grad_partial[vid, j, yz : yz + 64])

    return _mhc_pre_norm_fn_bwd_fused_kernel


def round_to_tf32(x: torch.Tensor) -> torch.Tensor:
    return (x.view(torch.int32) + 0x1000).view(torch.float32)


def mhc_pre_norm_fn_ref(
    residual: torch.Tensor, mhc_fn: torch.Tensor, mhc_norm_weight: torch.Tensor | None, mhc_norm_eps: float
) -> torch.Tensor:
    if mhc_norm_weight is not None:
        mhc_fn = mhc_fn * mhc_norm_weight
    residual = residual.flatten(2, 3).float()
    mhc_mult3 = mhc_fn.shape[0]
    rms_group_size = mhc_fn.shape[-1]
    mixes = torch.einsum("mbk,nbk->mbn", residual.view(-1, 1, rms_group_size), mhc_fn.view(mhc_mult3, 1, rms_group_size))
    sqrsum = residual.view(-1, 1, rms_group_size).square().sum(-1)
    mixes = (mixes * (sqrsum.unsqueeze(-1) / rms_group_size + mhc_norm_eps).rsqrt()).sum(-2)
    return mixes.view(*residual.shape[:2], -1)


def test_fn_normw_merge_fwd(m: int = 8, n: int = 1024):
    device = "npu"
    n_blk = 256
    n_padded = ((n + n_blk - 1) // n_blk) * n_blk

    torch.manual_seed(42)
    fn = torch.randn((m, n_padded), dtype=torch.float32, device=device)
    normw = torch.randn((n_padded,), dtype=torch.float32, device=device)
    out_fn = torch.zeros((m, n_padded), dtype=torch.float32, device=device)

    ref_out = fn * normw.unsqueeze(0)

    fwd_func = _mhc_fn_normw_merge_fwd(m, n_padded)
    fwd_func(fn, normw, out_fn)

    torch.testing.assert_close(out_fn.cpu(), ref_out.cpu(), atol=1e-2, rtol=1e-2)
    print("Kernel Output Match!")

    del fn, normw, out_fn, ref_out, fwd_func
    torch.npu.empty_cache()


def test_fn_normw_merge_bwd(m: int = 8, n: int = 1024):
    device = "npu"
    n_blk = 256
    n_padded = ((n + n_blk - 1) // n_blk) * n_blk

    torch.manual_seed(42)
    fn = torch.randn((m, n_padded), dtype=torch.float32, device=device)
    normw = torch.randn((n_padded,), dtype=torch.float32, device=device)
    out_fn_grad = torch.randn((m, n_padded), dtype=torch.float32, device=device)
    fn_grad = torch.zeros((m, n_padded), dtype=torch.float32, device=device)
    normw_grad = torch.zeros((n_padded,), dtype=torch.float32, device=device)

    ref_fn_grad = out_fn_grad * normw.unsqueeze(0)
    ref_normw_grad = (out_fn_grad * fn).sum(dim=0)

    bwd_func = _mhc_fn_normw_merge_bwd(m, n_padded)
    bwd_func(fn, normw, out_fn_grad, fn_grad, normw_grad)

    torch.testing.assert_close(fn_grad.cpu(), ref_fn_grad.cpu(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(normw_grad.cpu(), ref_normw_grad.cpu(), atol=1e-2, rtol=1e-2)
    print("Kernel Output Match!")

    del fn, normw, out_fn_grad, fn_grad, normw_grad, ref_fn_grad, ref_normw_grad, bwd_func
    torch.npu.empty_cache()


def test_fwd_fused(mhc_mult: int = 4, hidden_size: int = 320, num_tokens: int = 64, mhc_norm_eps: float = 1e-6):
    device = "npu"
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    mhc_hidden_size = mhc_mult * hidden_size
    n_rms_group = 1
    rms_group_size = mhc_hidden_size
    pad_mhc3_cube = ((mhc_mult3 + 15) // 16) * 16

    torch.manual_seed(42)
    x = torch.randn((num_tokens, mhc_mult, hidden_size), dtype=torch.float, device=device).bfloat16()
    fn_f32 = (torch.randn((mhc_mult3, mhc_mult, hidden_size), dtype=torch.float, device=device) * 1e-4).flatten(1, 2).contiguous()
    fn_f32 = round_to_tf32(fn_f32)
    fn_bf16 = fn_f32.to(torch.bfloat16)

    fn_bf16_padded = torch.zeros(pad_mhc3_cube, mhc_hidden_size, dtype=torch.bfloat16, device=device)
    fn_bf16_padded[:mhc_mult3, :] = fn_bf16

    token_block = 32
    total_grid_blocks = ((num_tokens + token_block - 1) // token_block) * n_rms_group
    pad_mhc3_cube = ((mhc_mult3 + 15) // 16) * 16

    sqrsum = torch.empty(num_tokens, n_rms_group, dtype=torch.float32, device=device)
    out_mul = torch.empty(num_tokens, n_rms_group, mhc_mult3, dtype=torch.float32, device=device)
    out = torch.empty(num_tokens * n_rms_group, mhc_mult3, dtype=torch.float32, device=device)
    workspace_gemm = torch.empty(total_grid_blocks, token_block, pad_mhc3_cube, dtype=torch.float32, device=device)

    fwd_fused_func = _mhc_pre_norm_fn_fwd_fused(mhc_mult3, n_rms_group, rms_group_size, mhc_norm_eps)
    fwd_fused_func(
        x.view(-1, mhc_hidden_size),
        fn_bf16_padded,
        sqrsum,
        out_mul,
        out,
        workspace_gemm,
    )

    x_ref = x.clone().detach().float().view(-1, 1, rms_group_size)
    mixes_ref = torch.einsum("nbk,mbk->nm", x_ref, fn_f32.view(mhc_mult3, 1, rms_group_size))
    sqrsum_ref = x_ref.square().sum(-1)
    out_ref = mixes_ref * (sqrsum_ref / rms_group_size + mhc_norm_eps).rsqrt()

    torch.testing.assert_close(out[:num_tokens].cpu(), out_ref.cpu(), atol=3e-3, rtol=3e-3)
    print("Kernel Output Match!")
    del x, fn_f32, fn_bf16, fn_bf16_padded, sqrsum, out_mul, out, workspace_gemm, fwd_fused_func
    torch.npu.empty_cache()


def test_bwd_fused(mhc_mult: int = 4, hidden_size: int = 320, num_tokens: int = 64, mhc_norm_eps: float = 1e-6):
    device = "npu"
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    mhc_hidden_size = mhc_mult * hidden_size
    n_rms_group = 1
    rms_group_size = mhc_hidden_size
    pad_mhc3_cube = ((mhc_mult3 + 15) // 16) * 16

    torch.manual_seed(42)
    x = torch.randn((num_tokens, mhc_mult, hidden_size), dtype=torch.float, device=device).bfloat16()
    fn_f32 = (torch.randn((mhc_mult3, mhc_mult, hidden_size), dtype=torch.float, device=device) * 1e-4).flatten(1, 2).contiguous()
    fn_f32 = round_to_tf32(fn_f32)
    fn_bf16 = fn_f32.to(torch.bfloat16)

    fn_bf16_padded = torch.zeros(pad_mhc3_cube, mhc_hidden_size, dtype=torch.bfloat16, device=device)
    fn_bf16_padded[:mhc_mult3, :] = fn_bf16

    token_block_fwd = 32
    total_grid_blocks_fwd = ((num_tokens + token_block_fwd - 1) // token_block_fwd) * n_rms_group

    sqrsum = torch.empty(num_tokens, n_rms_group, dtype=torch.float32, device=device)
    out_mul = torch.empty(num_tokens, n_rms_group, mhc_mult3, dtype=torch.float32, device=device)
    out_fwd = torch.empty(num_tokens * n_rms_group, mhc_mult3, dtype=torch.float32, device=device)
    workspace_gemm_fwd = torch.empty(total_grid_blocks_fwd, token_block_fwd, pad_mhc3_cube, dtype=torch.float32, device=device)

    fwd_fused_func = _mhc_pre_norm_fn_fwd_fused(mhc_mult3, n_rms_group, rms_group_size, mhc_norm_eps)
    fwd_fused_func(
        x.view(-1, mhc_hidden_size),
        fn_bf16_padded,
        sqrsum,
        out_mul,
        out_fwd,
        workspace_gemm_fwd,
    )

    out_grad = torch.randn(num_tokens, mhc_mult3, dtype=torch.float32, device=device)
    x_grad = torch.zeros(num_tokens, mhc_hidden_size, dtype=torch.bfloat16, device=device)
    fn_grad_partial = torch.zeros(VEC_NUM, mhc_mult3, mhc_hidden_size, dtype=torch.float32, device=device)

    bwd_fused_func = _mhc_pre_norm_fn_bwd_fused(mhc_mult3, n_rms_group, rms_group_size, mhc_norm_eps)
    bwd_fused_func(
        out_grad,
        out_mul,
        sqrsum,
        x.view(-1, mhc_hidden_size),
        fn_f32,
        x_grad.view(-1, mhc_hidden_size),
        fn_grad_partial,
    )

    fn_grad = fn_grad_partial.sum(dim=0)

    # ============ Reference computation ============
    x_ref = x.clone().detach().float().view(-1, 1, rms_group_size)
    fn_ref = fn_f32.view(mhc_mult3, 1, rms_group_size)
    mixes_ref = torch.einsum("nbk,mbk->nm", x_ref, fn_ref)
    sqrsum_ref = x_ref.square().sum(-1)
    rs_ref = (sqrsum_ref / rms_group_size + mhc_norm_eps).rsqrt()

    out_mul_ref = mixes_ref
    out_mul_grad_ref = out_grad * rs_ref

    fn_grad_ref = torch.einsum("nm,nk->mk", out_mul_grad_ref, x_ref.view(-1, rms_group_size))

    xgm_partial_ref = torch.einsum("nm,mk->nk", out_mul_grad_ref, fn_ref.view(mhc_mult3, rms_group_size))
    sqrsum_grad_ref = -0.5 * rs_ref.squeeze(-1) ** 3 / rms_group_size * (out_grad * out_mul_ref).sum(-1)
    x_grad_ref = xgm_partial_ref + 2 * sqrsum_grad_ref.unsqueeze(-1) * x_ref.view(-1, rms_group_size)

    torch.testing.assert_close(fn_grad[:mhc_mult3].cpu(), fn_grad_ref.cpu(), atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(x_grad.view(-1, rms_group_size).float().cpu(), x_grad_ref.cpu(), atol=3e-3, rtol=3e-3)
    print("Kernel Output Match!")
    del x, fn_f32, fn_bf16, fn_bf16_padded, sqrsum, out_mul, out_fwd, out_grad
    del x_grad, fn_grad, fn_grad_partial
    del fwd_fused_func, bwd_fused_func
    torch.npu.empty_cache()


def test_all():
    test_fn_normw_merge_fwd()
    test_fn_normw_merge_bwd()
    test_fwd_fused(4, 7168, 8192)
    test_bwd_fused(4, 7168, 8192)


if __name__ == "__main__":
    test_all()
