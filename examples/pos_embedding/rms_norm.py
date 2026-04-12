import tilelang
from tilelang import language as T
import torch

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def rms_norm(M, head_dim, block_M, eps, dtype="float16"):

    VEC_NUM = 2
    m_num = M // block_M
    row_per_vec = block_M // VEC_NUM

    ACC_DTYPE = "float32"
    TMP_DTYPE = "uint8"

    @T.prim_func
    def main(
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

    return main


def tilelang_q_rms(
    q,  # bs, 64, 512
    variance_epsilon,
):
    bs, head_num, dim = q.shape
    total_batch = bs * head_num
    q = q.view(total_batch, dim)

    block_M = 16

    func = rms_norm(total_batch, dim, block_M, variance_epsilon)
    q_out = func(q)

    return q_out.view(bs, head_num, dim)


def rms_norm_reference(q, variance_epsilon):

    bs, head_num, dim = q.shape
    total_batch = bs * head_num
    q = q.view(total_batch, dim)

    q_fp32 = q.float()

    sum_squares = torch.sum(q_fp32 * q_fp32, dim=-1, keepdim=True)

    mean_square = sum_squares / float(dim)

    rstd = torch.sqrt(mean_square + variance_epsilon)

    q_fp32_reload = q.float()

    q_normalized_fp32 = q_fp32_reload / rstd

    return q_normalized_fp32.half().view(bs, head_num, dim)


if __name__ == "__main__":
    torch.manual_seed(0)
    variance_epsilon = 1e-6

    q = torch.randn(16, 64, 512, dtype=torch.float16).npu()  # bs, head_num, dim

    q_out = tilelang_q_rms(q, variance_epsilon)
    ref_q = rms_norm_reference(q, variance_epsilon)

    torch.testing.assert_close(q_out.cpu(), ref_q.cpu(), rtol=1e-2, atol=1e-2)
    print("Kernel Output Match!")
