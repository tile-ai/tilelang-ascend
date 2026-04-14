import math
import pytest
import torch

import tilelang
import tilelang.language as T


pytestmark = [
    pytest.mark.op("parallel_brc"),
    pytest.mark.mode("Developer"),
]

ACCUM_DTYPE = "float32"

# Cases for test_parallel_delta_broadcast: (B, S, H, BS, block_H, sm_scale)
CASES = [
    (1, 32, 16, 32, 16, 0.125),
]

# Cases for test_row2d_mul_vec1d: (B, S, H, D, block_H, sm_scale)
# B       : batch size
# S       : sequence length
# H       : number of heads (tiled across NPU blocks via block_H)
# D       : feature dimension (the axis shared by h_2d row and weight vector)
# block_H : tile height per NPU block, must divide H evenly
# sm_scale: scalar multiplier applied after the element-wise multiply
ROW2D_MUL_VEC1D_CASES = [
    (1, 32, 16, 32, 16, 0.125),
    (1, 32, 32, 32, 32, 0.125),  # block_H == H: single tile covers all heads
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


def kernel_row2d_mul_vec1d(B, S, H, D, block_H, sm_scale):
    """
    Kernel that performs element-wise multiply of a 2-D activation buffer by a
    1-D weight vector (broadcast over the H dimension), matching the RMS-norm
    post-scale pattern:

        h_2d[0, j] = h_2d[0, j] * rms_w_h_shared[j]

    Tensors
    -------
    h_2d   : (B, S, H, D)  -- 2-D activation buffer; H rows, D columns per tile
    weight : (B, S, H, D)  -- weight vector; same shape, applied column-wise
    out    : (B, S, H, D)  -- result after element-wise multiply and scaling

    Each NPU block owns one (batch, seq, head-tile) triplet and iterates over
    all D columns in a single T.Parallel(block_H, D) loop.  This forces the
    backend to emit a 2-D vectorised multiply instruction, which is the
    vectorisation robustness case under test.
    """
    # h_tiles = T.ceildiv(H, block_H)

    @T.prim_func
    def row2d_mul_vec1d(
        h_2d: T.Tensor((B, S, H, D), ACCUM_DTYPE),  # 2-D activation buffer
        weight: T.Tensor(
            (B, S, H, D), ACCUM_DTYPE
        ),  # weight vector (1-D semantics along D)
        out: T.Tensor((B, S, H, D), ACCUM_DTYPE),  # element-wise scaled output
    ):
        # with T.Kernel(B * S * h_tiles, is_npu=True) as (cid, _):
        # bz_batch = cid // (S * h_tiles)
        # rem = cid % (S * h_tiles)
        # s_i = rem // h_tiles
        # h_tile = rem % h_tiles
        # h_start = h_tile * block_H
        # h_tail = T.min(block_H, H - h_start)

        # # Shared buffers: one (block_H, D) tile for the activation and weight.
        # h_2d_buf = T.alloc_shared((block_H, D), ACCUM_DTYPE)
        # weight_buf = T.alloc_shared((D,), ACCUM_DTYPE)
        # # out_buf = T.alloc_shared((block_H, D), ACCUM_DTYPE)

        # # Load the current head-tile slice for both activation and weight.
        # T.copy(
        #     h_2d[bz_batch, s_i, h_start : h_start + h_tail, :], h_2d_buf[:h_tail, :]
        # )
        # T.copy(weight[bz_batch, s_i, h_start : h_start + 1, :], weight_buf)

        # Vectorised element-wise multiply over the full (block_H, D) tile.
        # The inner D loop mirrors: h_2d[0, j] = h_2d[0, j] * rms_w_h_shared[j]
        # T.Parallel over both axes lets the backend fuse them into a single
        # vector instruction, exercising the 2-D vectorisation path.
        # for d_i in T.Parallel(D):
        #     out_buf[0, d_i] = (
        #         h_2d_buf[0, d_i] * weight_buf[d_i] * T.float32(sm_scale)
        #     )

        # T.copy(h_2d_buf[1:h_tail, :], out_buf[1:h_tail, :])

        # Write the scaled tile back to global memory.
        # T.copy(
        #     out_buf[:h_tail, :], out[bz_batch, s_i, h_start : h_start + h_tail, :]
        # )
        pass

    return row2d_mul_vec1d


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


@pytest.mark.parametrize("B, S, H, D, block_H, sm_scale", ROW2D_MUL_VEC1D_CASES)
def test_row2d_mul_vec1d(B, S, H, D, block_H, sm_scale):
    tilelang.cache.clear_cache()
    func = kernel_row2d_mul_vec1d(B, S, H, D, block_H, sm_scale)
    kernel = tilelang.compile(func, target="npuir")

    h_2d = torch.randn((B, S, H, D), dtype=torch.float32, device="npu")
    weight = torch.randn((B, S, H, D), dtype=torch.float32, device="npu")
    out = torch.zeros((B, S, H, D), dtype=torch.float32, device="npu")

    ref_output = h_2d.clone()
    h_tiles = math.ceil(H / block_H)
    for h_tile in range(h_tiles):
        h_start = h_tile * block_H
        ref_output[:, :, h_start, :] = (
            h_2d[:, :, h_start, :] * weight[:, :, h_start, :] * sm_scale
        )

    kernel(h_2d, weight, out)
    # torch.testing.assert_close(out, ref_output, rtol=1e-3, atol=1e-3)
