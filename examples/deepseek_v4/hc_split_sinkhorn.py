# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import torch
import logging

import tilelang
from tilelang import language as T


torch.set_default_device("npu")
torch.manual_seed(42)

logging.basicConfig(level=logging.INFO)

tilelang.disable_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


# kernel
@tilelang.jit(out_idx=[4, 5, 6], workspace_idx=[3], pass_configs=pass_configs)
def hc_split_sinkhorn(hc, sinkhorn_iters, eps):
    n = T.symbolic("n")
    mix_hc = (2 + hc) * hc
    dtype = "float"

    block_M = 2
    VEC_NUM = 2

    m_num = tilelang.cdiv(n, block_M)

    hc_pad = hc
    if hc * 4 % 32 != 0:
        hc_pad = tilelang.cdiv(hc * 4, 32) * 32 // 4

    @T.prim_func
    def main(
        mixes: T.Tensor([n, mix_hc], dtype),
        hc_scale: T.Tensor([3], dtype),
        hc_base: T.Tensor([mix_hc], dtype),
        workspace: T.Tensor([n, mix_hc], dtype),
        pre: T.Tensor([n, hc], dtype),
        post: T.Tensor([n, hc], dtype),
        comb: T.Tensor([n, hc, hc], dtype),
    ):

        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            mixes_shared = T.alloc_shared(mix_hc, dtype)
            hc_base_shared = T.alloc_shared(mix_hc, dtype)
            hc_scale_shared = T.alloc_ub(mix_hc, dtype)

            comb_shared = T.alloc_shared((hc, hc_pad), dtype)
            pre_shared = T.alloc_shared(hc_pad, dtype)
            post_shared = T.alloc_shared(hc_pad, dtype)

            tmp_shared = T.alloc_shared(hc_pad, dtype)

            row_sum = T.alloc_shared(hc_pad, dtype)
            col_sum = T.alloc_shared((1, hc_pad), dtype)
            row_max = T.alloc_shared(hc_pad, dtype)

            col_broadcast = T.alloc_shared((hc, hc_pad), dtype)
            row_div = T.alloc_shared((hc, hc_pad), dtype)

            if cid * block_M + vid * block_M // VEC_NUM < n:
                alpha_0 = hc_scale[0]
                alpha_1 = hc_scale[1]
                alpha_2 = hc_scale[2]

                for i in T.serial(hc):
                    hc_scale_shared[i] = alpha_0
                for i in T.serial(hc):
                    hc_scale_shared[hc + i] = alpha_1
                for i in T.serial(hc * hc):
                    hc_scale_shared[2 * hc + i] = alpha_2
                T.copy(hc_base, hc_base_shared)
                T.copy(mixes[cid * block_M + vid * block_M // VEC_NUM, :], mixes_shared)

                T.tile.mul(mixes_shared, mixes_shared, hc_scale_shared)
                T.tile.add(mixes_shared, mixes_shared, hc_base_shared)
                T.copy(mixes_shared, workspace[cid * block_M + vid * block_M // VEC_NUM, :])

                # pre
                T.copy(workspace[cid * block_M + vid * block_M // VEC_NUM, :hc], tmp_shared)
                T.tile.sigmoid(pre_shared, tmp_shared)
                T.tile.add(pre_shared, pre_shared, eps)
                T.copy(pre_shared[:hc], pre[cid * block_M + vid * block_M // VEC_NUM, :hc])

                # post
                T.copy(workspace[cid * block_M + vid * block_M // VEC_NUM, hc : hc + hc_pad], tmp_shared)
                T.tile.sigmoid(post_shared, tmp_shared)
                T.tile.mul(post_shared, post_shared, 2.0)
                T.copy(post_shared[:hc], post[cid * block_M + vid * block_M // VEC_NUM, :hc])

                # comb
                for i in T.serial(hc):
                    start = 2 * hc + i * hc
                    end = 2 * hc + i * hc + hc
                    T.copy(workspace[cid * block_M + vid * block_M // VEC_NUM, start:end], tmp_shared)
                    T.copy(tmp_shared, comb_shared[i, :])

                # comb = comb.softmax(-1) + eps
                T.reduce_max(comb_shared, row_max, dim=-1, real_shape=[hc, hc])
                for i in T.serial(hc):
                    T.tile.fill(row_div[i, :], row_max[i])
                T.tile.sub(comb_shared, comb_shared, row_div)
                T.tile.exp(comb_shared, comb_shared)
                T.reduce_sum(comb_shared, row_sum, dim=-1, real_shape=[hc, hc])
                for i in T.serial(hc):
                    T.tile.fill(row_div[i, :], row_sum[i])
                T.tile.div(comb_shared, comb_shared, row_div)
                T.tile.add(comb_shared, comb_shared, eps)

                # comb = comb / (comb.sum(-2) + eps)
                T.reduce_sum(comb_shared, col_sum, dim=0, real_shape=[hc, hc_pad])
                T.tile.add(col_sum, col_sum, eps)
                T.tile.broadcast(col_broadcast, col_sum)
                T.tile.div(comb_shared, comb_shared, col_broadcast)

                for _ in T.serial(sinkhorn_iters - 1):
                    # comb = comb / (comb.sum(-1) + eps)
                    T.reduce_sum(comb_shared, row_sum, dim=-1, real_shape=[hc, hc])
                    T.tile.add(row_sum, row_sum, eps)
                    for i in T.serial(hc):
                        T.tile.fill(row_div[i, :], row_sum[i])
                    T.tile.div(comb_shared, comb_shared, row_div)
                    # comb = comb / (comb.sum(-2) + eps)
                    T.reduce_sum(comb_shared, col_sum, dim=0, real_shape=[hc, hc_pad])
                    T.tile.add(col_sum, col_sum, eps)
                    T.tile.broadcast(col_broadcast, col_sum)
                    T.tile.div(comb_shared, comb_shared, col_broadcast)

                for i in T.serial(hc):
                    T.copy(comb_shared[i, :hc], comb[cid * block_M + vid * block_M // VEC_NUM, i, :])

    return main


# golden
def hc_split_sinkhorn_ref(mixes, hc_scale, hc_base, hc, sinkhorn_iters, eps):
    n = mixes.shape[0]
    dtype = torch.float32
    pre_ref = torch.empty((n, hc), dtype=dtype)
    post_ref = torch.empty((n, hc), dtype=dtype)
    comb_ref = torch.empty((n, hc, hc), dtype=dtype)

    for i in range(n):
        for j in range(hc):
            pre_ref[i, j] = torch.sigmoid(mixes[i, j] * hc_scale[0] + hc_base[j]) + eps
            post_ref[i, j] = 2 * torch.sigmoid(mixes[i, j + hc] * hc_scale[1] + hc_base[j + hc])
            for k in range(hc):
                comb_ref[i, j, k] = mixes[i, j * hc + k + hc * 2] * hc_scale[2] + hc_base[j * hc + k + hc * 2]

        # comb = comb.softmax(-1) + eps
        row_max, row_max_indices = torch.max(comb_ref[i, :, :], dim=1)
        for j in range(hc):
            comb_ref[i, j, :] = torch.exp(comb_ref[i, j, :] - row_max[j])
        row_sum = torch.sum(comb_ref[i, :, :], dim=1)
        for j in range(hc):
            comb_ref[i, j, :] = comb_ref[i, j, :] / row_sum[j] + eps
        # print(f"i: {i}, comb_ref: {comb_ref[i, :, :]}")

        # comb = comb / (comb.sum(-2) + eps)
        col_sum = torch.sum(comb_ref[i, :, :], dim=0)
        # print(f"i: {i}, col_sum: {col_sum}")
        for k in range(hc):
            comb_ref[i, :, k] = comb_ref[i, :, k] / (col_sum[k] + eps)

        for _ in range(sinkhorn_iters - 1):
            # comb = comb / (comb.sum(-1) + eps)
            row_sum = torch.sum(comb_ref[i, :, :], dim=1)
            for j in range(hc):
                comb_ref[i, j, :] = comb_ref[i, j, :] / (row_sum[j] + eps)
            # comb = comb / (comb.sum(-2) + eps)
            col_sum = torch.sum(comb_ref[i, :, :], dim=0)
            for k in range(hc):
                comb_ref[i, :, k] = comb_ref[i, :, k] / (col_sum[k] + eps)

    return pre_ref, post_ref, comb_ref


def test():
    # Input data dtype and shape
    dtype = torch.float32
    B, S, hc_mult = 1, 5, 4
    mix_hc = (2 + hc_mult) * hc_mult
    N = B * S

    mixes = torch.rand((N, mix_hc), dtype=dtype)
    hc_scale = torch.rand(3, dtype=dtype)
    hc_base = torch.rand(mix_hc, dtype=dtype)

    pre = torch.empty((N, hc_mult), dtype=dtype).npu()
    post = torch.empty((N, hc_mult), dtype=dtype).npu()
    comb = torch.empty((N, hc_mult, hc_mult), dtype=dtype).npu()
    torch.npu.synchronize()

    func = hc_split_sinkhorn(hc=hc_mult, sinkhorn_iters=20, eps=1e-6)

    logging.info("init successful!")

    pre, post, comb = func(mixes, hc_scale, hc_base)

    pre_ref, post_ref, comb_ref = hc_split_sinkhorn_ref(mixes, hc_scale, hc_base, hc_mult, 20, 1e-6)
    torch.npu.synchronize()

    torch.testing.assert_close(pre_ref, pre, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(post_ref, post, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(comb_ref, comb, rtol=1e-2, atol=1e-2)

    logging.info("Kernel Output Match!")


if __name__ == "__main__":
    test()
