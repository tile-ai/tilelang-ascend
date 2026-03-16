import pytest
import torch

import tilelang
import tilelang.language as T


pytestmark = [
    pytest.mark.op("parallel_brc"),
    pytest.mark.mode("Developer"),
]

ACCUM_DTYPE = "float32"

CASES = [
    (1, 32, 16, 32, 16, 0.125),
]


def kernel_parallel_delta_broadcast(B, S, H, BS, block_H, sm_scale):
    h_tiles = T.ceildiv(H, block_H)

    @T.prim_func
    def parallel_delta_broadcast(
        acc_p: T.Tensor((B, S, H, BS), ACCUM_DTYPE),
        acc_dp: T.Tensor((B, S, H, BS), ACCUM_DTYPE),
        Delta: T.Tensor((B, S, H), ACCUM_DTYPE),
        out: T.Tensor((B, S, H, BS), ACCUM_DTYPE),
    ):
        with T.Kernel(B * S * h_tiles, is_npu=True) as (cid, _):
            bz_batch = cid // (S * h_tiles)
            rem = cid % (S * h_tiles)
            s_i = rem // h_tiles
            h_tile = rem % h_tiles

            h_start = h_tile * block_H
            h_tail = T.min(block_H, H - h_start)

            acc_p_buf = T.alloc_shared((block_H, BS), ACCUM_DTYPE)
            acc_dp_buf = T.alloc_shared((block_H, BS), ACCUM_DTYPE)
            out_buf = T.alloc_shared((block_H, BS), ACCUM_DTYPE)

            T.copy(
                acc_p[bz_batch, s_i, h_start : h_start + h_tail, :],
                acc_p_buf[:h_tail, :],
            )
            T.copy(
                acc_dp[bz_batch, s_i, h_start : h_start + h_tail, :],
                acc_dp_buf[:h_tail, :],
            )

            for h_i, bi_i in T.Parallel(block_H, BS):
                out_buf[h_i, bi_i] = (
                    acc_p_buf[h_i, bi_i]
                    * (acc_dp_buf[h_i, bi_i] - Delta[bz_batch, s_i, h_start + h_i])
                    * T.float32(sm_scale)
                )

            T.copy(
                out_buf[:h_tail, :],
                out[bz_batch, s_i, h_start : h_start + h_tail, :],
            )

    return parallel_delta_broadcast


@pytest.mark.parametrize("B, S, H, BS, block_H, sm_scale", CASES)
def test_parallel_delta_broadcast(B, S, H, BS, block_H, sm_scale):
    func = kernel_parallel_delta_broadcast(B, S, H, BS, block_H, sm_scale)
    kernel = tilelang.compile(func, target="npuir")

    acc_p = torch.randn((B, S, H, BS), dtype=torch.float32, device="npu")
    acc_dp = torch.randn((B, S, H, BS), dtype=torch.float32, device="npu")
    Delta = torch.randn((B, S, H), dtype=torch.float32, device="npu")
    out = torch.zeros((B, S, H, BS), dtype=torch.float32, device="npu")

    ref_output = acc_p * (acc_dp - Delta.unsqueeze(-1)) * sm_scale

    kernel(acc_p, acc_dp, Delta, out)

    torch.testing.assert_close(out, ref_output, rtol=1e-3, atol=1e-3)
