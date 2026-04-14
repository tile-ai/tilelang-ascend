import torch
import tilelang
import tilelang.language as T
import argparse

"""Take the RMS Norm operator and RoPE operator as examples to demonstrate the aclGraph mode."""

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}

device = torch.device("npu")

# ======================== RMS Norm Kernel ========================
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def rms_norm_kernel(M, head_dim, block_M, eps, dtype="float16"):

    VEC_NUM = 2
    m_num = M // block_M
    row_per_vec = block_M // VEC_NUM

    ACC_DTYPE = "float32"
    TMP_DTYPE = "uint8"

    @T.prim_func
    def main_rms(
        x: T.Tensor((M, head_dim), dtype),  # type: ignore
        out: T.Tensor((M, head_dim), dtype),  # type: ignore
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            row_x = cid * block_M + vid * row_per_vec

            x_ub = T.alloc_shared([row_per_vec, head_dim], dtype)
            x_ub_fp32 = T.alloc_shared([row_per_vec, head_dim], ACC_DTYPE)

            sum_square_ub = T.alloc_shared([row_per_vec, head_dim], ACC_DTYPE)
            rms_ub = T.alloc_shared([row_per_vec], ACC_DTYPE)

            ## Accumulation
            T.copy(x[row_x : row_x + row_per_vec, :], x_ub)
            T.copy(x_ub, x_ub_fp32)  # fp16 --> fp32
            T.tile.mul(sum_square_ub, x_ub_fp32, x_ub_fp32)

            ## Reduce
            T.reduce_sum(sum_square_ub, rms_ub, dim=-1)

            ## Compute mean and variance
            T.tile.div(rms_ub, rms_ub, head_dim)
            T.tile.add(rms_ub, rms_ub, eps)
            T.tile.sqrt(rms_ub, rms_ub)

            ## Normalize
            for i in T.serial(0, row_per_vec):
                T.tile.div(x_ub_fp32[i, :], x_ub_fp32[i, :], rms_ub[i])
            T.copy(x_ub_fp32, x_ub)  # fp32 --> fp16
            T.copy(x_ub, out[row_x : row_x + row_per_vec, :])

    return main_rms


def tilelang_rms_norm(q, variance_epsilon):
    batch_size, head_num, hidden_size = q.shape
    total_batch = batch_size * head_num
    q = q.view(total_batch, hidden_size)

    block_M = 32

    func = rms_norm_kernel(total_batch, hidden_size, block_M, variance_epsilon)
    q_out = func(q)

    return q_out.view(batch_size, head_num, hidden_size)


# ======================== RoPE Kernel ========================
@tilelang.jit(pass_configs=pass_configs)
def rope_kernel_in_place(
    M, block_M, batch_size, hidden_size, rope_dim, head_num, dtype="float16"
):
    VEC_NUM = 2
    m_num = M // block_M

    dim_start = hidden_size - rope_dim

    row_per_vec = block_M // VEC_NUM

    ACC_DTYPE = "float32"
    MASK_DTYPE = "uint32"
    TMP_DTYPE = "uint8"

    @T.prim_func
    def main_rope(
        x: T.Tensor([M, hidden_size], dtype),  # type: ignore
        sin: T.Tensor([batch_size, rope_dim], dtype),  # type: ignore
        cos: T.Tensor([batch_size, rope_dim], dtype),  # type: ignore
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            row_x = cid * block_M + vid * row_per_vec
            row_sin_cos = row_x // head_num
            # 1. copy x: gm -> ub
            x_half_ub = T.alloc_shared([row_per_vec, rope_dim], dtype)
            x_ub = T.alloc_shared([row_per_vec, rope_dim], ACC_DTYPE)

            for i in T.serial(0, row_per_vec):
                T.copy(x[row_x + i, dim_start:], x_half_ub[i, :])
            T.copy(x_half_ub, x_ub)  # cast float16 -> float32

            # 2. copy sin/cos: gm -> ub
            sin_ub = T.alloc_shared([1, rope_dim], ACC_DTYPE)
            sin_half_ub = T.alloc_shared([1, rope_dim], dtype)
            cos_ub = T.alloc_shared([1, rope_dim], ACC_DTYPE)
            cos_half_ub = T.alloc_shared([1, rope_dim], dtype)
            T.copy(sin[row_sin_cos, :], sin_half_ub[0, :])
            T.copy(sin_half_ub, sin_ub)
            T.copy(cos[row_sin_cos, :], cos_half_ub[0, :])
            T.copy(cos_half_ub, cos_ub)

            # 3. set mask: gm -> ub
            mask_ub_i16 = T.alloc_shared([row_per_vec, rope_dim], "int16")
            mask_ub_f32 = T.alloc_shared([row_per_vec, rope_dim], "float32")
            mask_ub_i32 = T.alloc_shared([row_per_vec, rope_dim], "int32")
            mask_ub = T.alloc_shared([row_per_vec, rope_dim], MASK_DTYPE)
            idx_ub = T.alloc_shared([row_per_vec, rope_dim], "int32")
            tmp_ub_i16 = T.alloc_shared([row_per_vec, rope_dim], "int16")
            ones_mask_ub = T.alloc_shared([row_per_vec, rope_dim], "int16")
            T.tile.createvecindex(idx_ub, 0)
            T.copy(idx_ub, tmp_ub_i16)
            T.tile.fill(ones_mask_ub, 1)
            T.tile.bitwise_xor(mask_ub_i16, tmp_ub_i16, ones_mask_ub)
            T.copy(mask_ub_i16, mask_ub_f32)
            T.copy(mask_ub_f32, mask_ub_i32)
            T.tile.mul(mask_ub_i32, mask_ub_i32, 4)
            T.reinterpretcast(mask_ub, mask_ub_i32, "uint32_t")

            sin_mask_ub = T.alloc_ub(rope_dim, ACC_DTYPE)
            T.tile.fill(sin_mask_ub, -1.0)
            for i in T.serial(0, rope_dim // 2):
                sin_mask_ub[2 * i + 1] = 1.0
            T.tile.mul(sin_ub[0, :], sin_ub[0, :], sin_mask_ub)

            # 4. broadcast sin/cos: [1, rope_dim] -> [row_per_vec, rope_dim]
            sin_block_ub = T.alloc_shared([row_per_vec, rope_dim], ACC_DTYPE)
            T.tile.broadcast(sin_block_ub, sin_ub)
            cos_block_ub = T.alloc_shared([row_per_vec, rope_dim], ACC_DTYPE)
            T.tile.broadcast(cos_block_ub, cos_ub)

            # 5. rotate x
            x_rotate_ub = T.alloc_shared([row_per_vec, rope_dim], ACC_DTYPE)
            T.tile.gather(x_rotate_ub, x_ub, mask_ub, 0)

            # 6. x * cos - x_rotate * sin
            out_ub = T.alloc_shared([row_per_vec, rope_dim], ACC_DTYPE)
            T.tile.mul(x_ub, x_ub, cos_block_ub)
            T.tile.mul(x_rotate_ub, x_rotate_ub, sin_block_ub)
            T.tile.add(out_ub, x_ub, x_rotate_ub)

            # 7. copy out
            T.copy(out_ub, x_half_ub)  # cast float32 -> float16
            for i in T.serial(0, row_per_vec):
                T.copy(x_half_ub[i, :], x[row_x + i, dim_start:])

    return main_rope


def tilelang_apply_rope(x, sin, cos):
    rope_dim = sin.shape[-1]
    org_shape = x.shape
    if x.dim() == 2:
        batch_size, hidden_size = x.shape
        head_num = 1
    elif x.dim() == 3:
        batch_size, head_num, hidden_size = x.shape
        x = x.view(-1, hidden_size)
    else:
        raise NotImplementedError(f"x_shape={x.shape} not supported")

    total_rows = batch_size * head_num
    block_M = 32

    x = x.to(device)
    sin = sin.to(device)
    cos = cos.to(device)

    kernel = rope_kernel_in_place(
        total_rows, block_M, batch_size, hidden_size, rope_dim, head_num
    )
    kernel(x, sin, cos)

    return x.view(org_shape)

# ======================== Reference Implementations ========================
def rms_norm_reference(q, variance_epsilon):
    batch_size, head_num, hidden_size = q.shape
    total_batch = batch_size * head_num
    q = q.view(total_batch, hidden_size)

    q_fp32 = q.float()
    sum_squares = torch.sum(q_fp32 * q_fp32, dim=-1, keepdim=True)
    mean_square = sum_squares / float(hidden_size)
    rstd = torch.sqrt(mean_square + variance_epsilon)

    q_fp32_reload = q.float()
    q_normalized_fp32 = q_fp32_reload / rstd

    return q_normalized_fp32.half().view(batch_size, head_num, hidden_size)


def torch_rope_ref(x, sin, cos):
    if cos.dim() == 2:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
    x0 = x_reshaped[..., 0]
    x1 = x_reshaped[..., 1]

    x_rotated = torch.stack([-x1, x0], dim=-1).flatten(-2)

    out = x * cos + x_rotated * sin
    return out.to(torch.float16)


# ======================== Main ========================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--shape",
        type=int,
        nargs=4,
        metavar=("BS", "H", "HS", "RD"),
        default=[16, 64, 512, 256],
        help="batch_size head_num hidden_size rope_dim",
    )
    p.add_argument(
        "--eps",
        type=float,
        default=1e-6,
        help="variance epsilon for RMS norm",
    )
    return p.parse_args()


if __name__ == "__main__":
    torch.manual_seed(42)
    tilelang.disable_cache()

    args = parse_args()
    batch_size, head_num, hidden_size, rope_dim = args.shape
    variance_epsilon = args.eps

    torch_dtype = torch.float16

    q = torch.randn(
        (batch_size, head_num, hidden_size), device=device, dtype=torch_dtype
    )
    sin = torch.randn((batch_size, rope_dim), device=device, dtype=torch_dtype)
    cos = torch.randn((batch_size, rope_dim), device=device, dtype=torch_dtype)

    # ---- Reference: RMS Norm -> RoPE ----
    q_ref = rms_norm_reference(q.clone(), variance_epsilon)
    dim_start = hidden_size - rope_dim
    q_part = q_ref[..., dim_start:]
    q_part_out = torch_rope_ref(
        q_part.to(torch.float32), sin.to(torch.float32), cos.to(torch.float32)
    )
    q_ref[..., dim_start:] = q_part_out

    # ---- aclgraph: capture begin ----
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        q = tilelang_rms_norm(q, variance_epsilon)
        q = tilelang_apply_rope(q, sin, cos)

    # aclgraph: execute
    g.replay()

    torch.testing.assert_close(q, q_ref, rtol=1e-2, atol=1e-2)
    print("Kernel Output Match!")