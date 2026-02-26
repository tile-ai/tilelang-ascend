import tilelang
from tilelang import language as T
import torch
import argparse

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def rms_rope_fused(
    M, block_M, batch_size, head_dim, rope_dim, head_num, eps, dtype="float16"
):

    ACC_DTYPE = "float"
    MASK_DTYPE = "uint32"
    TMP_DTYPE = "uint8"

    VEC_NUM = 2
    m_num = M // block_M
    row_per_vec = block_M // VEC_NUM
    dim_start = head_dim - rope_dim

    @T.prim_func
    def main(
        x: T.Tensor((M, head_dim), dtype),  # type: ignore
        sin: T.Tensor([batch_size, rope_dim], dtype),  # type: ignore
        cos: T.Tensor([batch_size, rope_dim], dtype),  # type: ignore
        out: T.Tensor((M, head_dim), dtype),  # type: ignore
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):

            row_x = cid * block_M + vid * row_per_vec
            row_sin_cos = row_x // head_num

            x_ub = T.alloc_shared([row_per_vec, head_dim], dtype)
            x_ub_fp32 = T.alloc_shared([row_per_vec, head_dim], ACC_DTYPE)

            sum_square_ub = T.alloc_shared([row_per_vec, head_dim], ACC_DTYPE)
            rms_ub = T.alloc_shared([row_per_vec], ACC_DTYPE)
            tmp_ub = T.alloc_shared(row_per_vec * head_dim, TMP_DTYPE)

            rope_ub = T.alloc_shared([row_per_vec, rope_dim], ACC_DTYPE)
            sin_ub = T.alloc_shared([1, rope_dim], ACC_DTYPE)
            sin_block_ub = T.alloc_shared([row_per_vec, rope_dim], ACC_DTYPE)
            cos_ub = T.alloc_shared([1, rope_dim], ACC_DTYPE)
            cos_block_ub = T.alloc_shared([row_per_vec, rope_dim], ACC_DTYPE)
            sin_cos_half_ub = T.alloc_shared([1, rope_dim], dtype)

            tmp_ub2 = T.alloc_shared(row_per_vec * rope_dim, TMP_DTYPE)

            rope_rotate_ub = T.alloc_shared([row_per_vec, rope_dim], ACC_DTYPE)

            # 1.RMS-norm
            ## Accumulation
            T.copy(x[row_x : row_x + row_per_vec, :], x_ub)
            T.copy(x_ub, x_ub_fp32)  # fp16 --> fp32
            T.tile.mul(sum_square_ub, x_ub_fp32, x_ub_fp32)

            ## Reduce
            T.reduce_sum(sum_square_ub, rms_ub, tmp_ub, dim=-1)

            ## Compute mean and variance
            T.tile.div(rms_ub, rms_ub, head_dim)
            T.tile.add(rms_ub, rms_ub, eps)
            T.tile.sqrt(rms_ub, rms_ub)

            ## Normalize
            for i in T.serial(0, row_per_vec):
                T.tile.div(x_ub_fp32[i, :], x_ub_fp32[i, :], rms_ub[i])
                T.copy(x_ub_fp32[i, dim_start:], rope_ub[i, :])

            # 2. rope
            ## copy sin/cos: gm -> ub
            T.copy(sin[row_sin_cos, :], sin_cos_half_ub[0, :])
            T.copy(sin_cos_half_ub, sin_ub)
            T.copy(cos[row_sin_cos, :], sin_cos_half_ub[0, :])
            T.copy(sin_cos_half_ub, cos_ub)

            ## set mask: gm -> ub
            mask_ub_i16 = T.alloc_shared([row_per_vec, rope_dim], "int16")
            mask_ub_f32 = T.alloc_shared([row_per_vec, rope_dim], "float32")
            mask_ub_i32 = T.alloc_shared([row_per_vec, rope_dim], "int32")
            mask_ub = T.alloc_shared([row_per_vec, rope_dim], MASK_DTYPE)
            idx_ub = T.alloc_shared([row_per_vec, rope_dim], "int32")
            tmp_ub_i16 = T.alloc_shared([row_per_vec, rope_dim], "int16")
            ones_mask_ub = T.alloc_shared([row_per_vec, rope_dim], "int16")
            xor_tmp_ub = T.alloc_shared([row_per_vec, rope_dim], "int16")
            T.tile.createvecindex(idx_ub, 0)
            T.copy(idx_ub, tmp_ub_i16)
            T.tile.fill(ones_mask_ub, 1)
            T.tile.bitwise_xor(mask_ub_i16, tmp_ub_i16, ones_mask_ub, xor_tmp_ub)
            T.copy(mask_ub_i16, mask_ub_f32)
            T.copy(mask_ub_f32, mask_ub_i32)
            T.tile.mul(mask_ub_i32, mask_ub_i32, 4)
            T.reinterpretcast(mask_ub, mask_ub_i32, "uint32_t")
                    
            sin_mask_ub = T.alloc_ub(rope_dim, ACC_DTYPE)
            T.tile.fill(sin_mask_ub, -1.0)
            for i in T.serial(0, rope_dim // 2):
                sin_mask_ub[2 * i + 1] = 1.0
            T.tile.mul(sin_ub[0, :], sin_ub[0, :], sin_mask_ub)

            ## broadcast sin/cos: [1, rope_dim] -> [row_per_vec, rope_dim]
            T.tile.broadcast(sin_block_ub, sin_ub, tmp_ub2)
            T.tile.broadcast(cos_block_ub, cos_ub, tmp_ub2)

            ## rotate x
            T.tile.gather(rope_rotate_ub, rope_ub, mask_ub, 0)

            ## x * cos - x_rotate * sin
            T.tile.mul(rope_ub, rope_ub, cos_block_ub)
            T.tile.mul(rope_rotate_ub, rope_rotate_ub, sin_block_ub)
            T.tile.add(rope_ub, rope_ub, rope_rotate_ub)

            ## copy out
            for i in T.serial(0, row_per_vec):
                T.copy(rope_ub[i, :], x_ub_fp32[i, dim_start:])
            T.copy(x_ub_fp32, x_ub)  # cast float32 -> float16
            T.copy(x_ub, out[row_x : row_x + row_per_vec, :])

    return main


def tilelang_rms_rope_fused(q, sin, cos, eps):
    rope_dim = sin.shape[-1]
    org_shape = q.shape
    if q.dim() == 2:
        batch_size, head_dim = q.shape
        head_num = 1
    elif q.dim() == 3:
        batch_size, head_num, head_dim = q.shape
        q = q.view(-1, head_dim)
    else:
        raise NotImplementedError(f"q_shape={q.shape} not supported")

    total_batch = batch_size * head_num
    block_M = 16

    q = q.to(device)
    sin = sin.to(device)
    cos = cos.to(device)

    kernel = rms_rope_fused(
        total_batch, block_M, batch_size, head_dim, rope_dim, head_num, eps
    )
    q_out = kernel(q, sin, cos)

    return q_out.view(org_shape)


def rope_reference(q, cos, sin):
    # q: [batch, head, dim]
    # cos, sin: [batch, dim] (broadcast over head)

    # [batch, 1, dim] -> [batch, head, dim]
    if cos.dim() == 2:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    # Interleaved logic: [-x1, x0, -x3, x2]
    q_reshaped = q.reshape(*q.shape[:-1], -1, 2)
    q0 = q_reshaped[..., 0]
    q1 = q_reshaped[..., 1]

    q_rotated = torch.stack([-q1, q0], dim=-1).flatten(-2)

    out = q * cos + q_rotated * sin

    return out.to(torch.float16)


def rms_norm_reference(q, head_dim, eps):
    q_fp32 = q.float()
    sum_squares = torch.sum(q_fp32 * q_fp32, dim=-1, keepdim=True)
    mean_square = sum_squares / float(head_dim)
    rstd = torch.sqrt(mean_square + eps)
    q_fp32_reload = q.float()
    q_normalized_fp32 = q_fp32_reload / rstd

    return q_normalized_fp32.half()


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
    return p.parse_args()


if __name__ == "__main__":
    torch.manual_seed(42)
    tilelang.disable_cache()

    args = parse_args()
    batch_size, head_num, head_dim, rope_dim = args.shape
    eps = 1e-6

    device = "npu"
    torch_dtype = torch.float16

    q = torch.randn((batch_size, head_num, head_dim), device=device, dtype=torch_dtype)
    sin = torch.randn((batch_size, rope_dim), device=device, dtype=torch_dtype)
    cos = torch.randn((batch_size, rope_dim), device=device, dtype=torch_dtype)

    # 1. Run PyTorch Reference
    dim_start = head_dim - rope_dim
    q_ref = q.clone()
    q_ref = rms_norm_reference(q, head_dim, eps)

    q_part = q_ref[..., dim_start:]
    q_part_out = rope_reference(
        q_part.to(torch.float32), cos.to(torch.float32), sin.to(torch.float32)
    )
    q_ref[..., dim_start:] = q_part_out

    # 2. Run TileLang Kernel
    q_tl = q.clone()
    q_out = tilelang_rms_rope_fused(q_tl, sin, cos, eps)

    torch.testing.assert_close(q_out, q_ref, rtol=1e-2, atol=1e-2)
    print("Kernel Output Match!")
