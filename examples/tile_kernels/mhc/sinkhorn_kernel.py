import math
import tilelang
import torch
from tilelang import language as T

_FWD_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

_BWD_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

VEC_NUM = 2

_LARGE_NEG = -1e30


def _compute_sums_cols(hidden_size: int, repeat: int) -> int:
    pad_h = ((hidden_size + 7) // 8) * 8
    return pad_h + (repeat - 1) * (hidden_size + 1) * pad_h


@tilelang.jit(pass_configs=_FWD_PASS_CONFIGS)
def _mhc_sinkhorn_fwd(
    hidden_size: int,
    token_block_size: int,
    repeat: int,
    eps: float,
) -> tilelang.JITKernel:
    assert hidden_size == 4
    assert repeat == 10
    num_tokens = T.symbolic("num_tokens")
    dtype = "float32"
    pad_h = ((hidden_size + 7) // 8) * 8
    sub_block_tokens = token_block_size // VEC_NUM
    sums_cols = _compute_sums_cols(hidden_size, repeat)

    @T.prim_func
    def mhc_sinkhorn_kernel(
        row0_in: T.Tensor[(num_tokens, pad_h), dtype],
        row1_in: T.Tensor[(num_tokens, pad_h), dtype],
        row2_in: T.Tensor[(num_tokens, pad_h), dtype],
        row3_in: T.Tensor[(num_tokens, pad_h), dtype],
        row0_out: T.Tensor[(num_tokens, hidden_size), dtype],
        row1_out: T.Tensor[(num_tokens, hidden_size), dtype],
        row2_out: T.Tensor[(num_tokens, hidden_size), dtype],
        row3_out: T.Tensor[(num_tokens, hidden_size), dtype],
        sums_out: T.Tensor[(num_tokens, sums_cols), dtype],
    ) -> None:
        with T.Kernel(T.ceildiv(num_tokens, token_block_size), is_npu=True) as (cid, vid):
            ub_0 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            ub_1 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            ub_2 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            ub_3 = T.alloc_ub((sub_block_tokens, pad_h), dtype)

            row_max_vec = T.alloc_ub((sub_block_tokens,), dtype)
            row_max_ub = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            row_sum_vec = T.alloc_ub((sub_block_tokens,), dtype)
            row_sum_ub = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            col_sum = T.alloc_ub((sub_block_tokens, pad_h), dtype)

            cs0 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            cs1 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            cs2 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            cs3 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            cs4 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            cs5 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            cs6 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            cs7 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            cs8 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            cs9 = T.alloc_ub((sub_block_tokens, pad_h), dtype)

            rs_0_0 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            rs_0_1 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            rs_0_2 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            rs_0_3 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            rs_1_0 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            rs_1_1 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            rs_1_2 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            rs_1_3 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            rs_2_0 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            rs_2_1 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            rs_2_2 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            rs_2_3 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            rs_3_0 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            rs_3_1 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            rs_3_2 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            rs_3_3 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            rs_4_0 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            rs_4_1 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            rs_4_2 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            rs_4_3 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            rs_5_0 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            rs_5_1 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            rs_5_2 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            rs_5_3 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            rs_6_0 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            rs_6_1 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            rs_6_2 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            rs_6_3 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            rs_7_0 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            rs_7_1 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            rs_7_2 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            rs_7_3 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            rs_8_0 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            rs_8_1 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            rs_8_2 = T.alloc_ub((sub_block_tokens, pad_h), dtype)
            rs_8_3 = T.alloc_ub((sub_block_tokens, pad_h), dtype)

            row_start = cid * token_block_size + vid * sub_block_tokens

            with T.Scope("V"):
                T.set_flag("mte3", "mte2", 0)
                T.wait_flag("mte3", "mte2", 0)

                T.copy(row0_in[row_start : row_start + sub_block_tokens, :], ub_0)
                T.copy(row1_in[row_start : row_start + sub_block_tokens, :], ub_1)
                T.copy(row2_in[row_start : row_start + sub_block_tokens, :], ub_2)
                T.copy(row3_in[row_start : row_start + sub_block_tokens, :], ub_3)

                T.set_flag("mte2", "v", 0)
                T.wait_flag("mte2", "v", 0)

                T.reduce_max(ub_0, row_max_vec, dim=-1)
                T.tile.broadcast(row_max_ub, row_max_vec)
                T.tile.sub(ub_0, ub_0, row_max_ub)
                T.reduce_max(ub_1, row_max_vec, dim=-1)
                T.tile.broadcast(row_max_ub, row_max_vec)
                T.tile.sub(ub_1, ub_1, row_max_ub)
                T.reduce_max(ub_2, row_max_vec, dim=-1)
                T.tile.broadcast(row_max_ub, row_max_vec)
                T.tile.sub(ub_2, ub_2, row_max_ub)
                T.reduce_max(ub_3, row_max_vec, dim=-1)
                T.tile.broadcast(row_max_ub, row_max_vec)
                T.tile.sub(ub_3, ub_3, row_max_ub)

                T.tile.exp(ub_0, ub_0)
                T.tile.exp(ub_1, ub_1)
                T.tile.exp(ub_2, ub_2)
                T.tile.exp(ub_3, ub_3)

                T.reduce_sum(ub_0, row_sum_vec, dim=-1)
                T.tile.add(row_sum_vec, row_sum_vec, eps)
                T.tile.broadcast(row_sum_ub, row_sum_vec)
                T.tile.div(ub_0, ub_0, row_sum_ub)
                T.reduce_sum(ub_1, row_sum_vec, dim=-1)
                T.tile.add(row_sum_vec, row_sum_vec, eps)
                T.tile.broadcast(row_sum_ub, row_sum_vec)
                T.tile.div(ub_1, ub_1, row_sum_ub)
                T.reduce_sum(ub_2, row_sum_vec, dim=-1)
                T.tile.add(row_sum_vec, row_sum_vec, eps)
                T.tile.broadcast(row_sum_ub, row_sum_vec)
                T.tile.div(ub_2, ub_2, row_sum_ub)
                T.reduce_sum(ub_3, row_sum_vec, dim=-1)
                T.tile.add(row_sum_vec, row_sum_vec, eps)
                T.tile.broadcast(row_sum_ub, row_sum_vec)
                T.tile.div(ub_3, ub_3, row_sum_ub)

                T.tile.add(col_sum, ub_0, ub_1)
                T.tile.add(col_sum, col_sum, ub_2)
                T.tile.add(col_sum, col_sum, ub_3)
                T.tile.fill(row_sum_ub, eps)
                T.tile.add(col_sum, col_sum, row_sum_ub)
                T.tile.div(ub_0, ub_0, col_sum)
                T.tile.div(ub_1, ub_1, col_sum)
                T.tile.div(ub_2, ub_2, col_sum)
                T.tile.div(ub_3, ub_3, col_sum)

                T.tile.fill(cs0, 0.0)
                T.tile.add(cs0, cs0, col_sum)

                T.reduce_sum(ub_0, row_sum_vec, dim=-1)
                T.tile.add(row_sum_vec, row_sum_vec, eps)
                T.tile.broadcast(row_sum_ub, row_sum_vec)
                T.tile.fill(rs_0_0, 0.0)
                T.tile.add(rs_0_0, rs_0_0, row_sum_ub)
                T.tile.div(ub_0, ub_0, row_sum_ub)
                T.reduce_sum(ub_1, row_sum_vec, dim=-1)
                T.tile.add(row_sum_vec, row_sum_vec, eps)
                T.tile.broadcast(row_sum_ub, row_sum_vec)
                T.tile.fill(rs_0_1, 0.0)
                T.tile.add(rs_0_1, rs_0_1, row_sum_ub)
                T.tile.div(ub_1, ub_1, row_sum_ub)
                T.reduce_sum(ub_2, row_sum_vec, dim=-1)
                T.tile.add(row_sum_vec, row_sum_vec, eps)
                T.tile.broadcast(row_sum_ub, row_sum_vec)
                T.tile.fill(rs_0_2, 0.0)
                T.tile.add(rs_0_2, rs_0_2, row_sum_ub)
                T.tile.div(ub_2, ub_2, row_sum_ub)
                T.reduce_sum(ub_3, row_sum_vec, dim=-1)
                T.tile.add(row_sum_vec, row_sum_vec, eps)
                T.tile.broadcast(row_sum_ub, row_sum_vec)
                T.tile.fill(rs_0_3, 0.0)
                T.tile.add(rs_0_3, rs_0_3, row_sum_ub)
                T.tile.div(ub_3, ub_3, row_sum_ub)
                T.tile.add(col_sum, ub_0, ub_1)
                T.tile.add(col_sum, col_sum, ub_2)
                T.tile.add(col_sum, col_sum, ub_3)
                T.tile.fill(row_sum_ub, eps)
                T.tile.add(col_sum, col_sum, row_sum_ub)
                T.tile.div(ub_0, ub_0, col_sum)
                T.tile.div(ub_1, ub_1, col_sum)
                T.tile.div(ub_2, ub_2, col_sum)
                T.tile.div(ub_3, ub_3, col_sum)
                T.tile.fill(cs1, 0.0)
                T.tile.add(cs1, cs1, col_sum)

                T.reduce_sum(ub_0, row_sum_vec, dim=-1)
                T.tile.add(row_sum_vec, row_sum_vec, eps)
                T.tile.broadcast(row_sum_ub, row_sum_vec)
                T.tile.fill(rs_1_0, 0.0)
                T.tile.add(rs_1_0, rs_1_0, row_sum_ub)
                T.tile.div(ub_0, ub_0, row_sum_ub)
                T.reduce_sum(ub_1, row_sum_vec, dim=-1)
                T.tile.add(row_sum_vec, row_sum_vec, eps)
                T.tile.broadcast(row_sum_ub, row_sum_vec)
                T.tile.fill(rs_1_1, 0.0)
                T.tile.add(rs_1_1, rs_1_1, row_sum_ub)
                T.tile.div(ub_1, ub_1, row_sum_ub)
                T.reduce_sum(ub_2, row_sum_vec, dim=-1)
                T.tile.add(row_sum_vec, row_sum_vec, eps)
                T.tile.broadcast(row_sum_ub, row_sum_vec)
                T.tile.fill(rs_1_2, 0.0)
                T.tile.add(rs_1_2, rs_1_2, row_sum_ub)
                T.tile.div(ub_2, ub_2, row_sum_ub)
                T.reduce_sum(ub_3, row_sum_vec, dim=-1)
                T.tile.add(row_sum_vec, row_sum_vec, eps)
                T.tile.broadcast(row_sum_ub, row_sum_vec)
                T.tile.fill(rs_1_3, 0.0)
                T.tile.add(rs_1_3, rs_1_3, row_sum_ub)
                T.tile.div(ub_3, ub_3, row_sum_ub)
                T.tile.add(col_sum, ub_0, ub_1)
                T.tile.add(col_sum, col_sum, ub_2)
                T.tile.add(col_sum, col_sum, ub_3)
                T.tile.fill(row_sum_ub, eps)
                T.tile.add(col_sum, col_sum, row_sum_ub)
                T.tile.div(ub_0, ub_0, col_sum)
                T.tile.div(ub_1, ub_1, col_sum)
                T.tile.div(ub_2, ub_2, col_sum)
                T.tile.div(ub_3, ub_3, col_sum)
                T.tile.fill(cs2, 0.0)
                T.tile.add(cs2, cs2, col_sum)

                T.reduce_sum(ub_0, row_sum_vec, dim=-1)
                T.tile.add(row_sum_vec, row_sum_vec, eps)
                T.tile.broadcast(row_sum_ub, row_sum_vec)
                T.tile.fill(rs_2_0, 0.0)
                T.tile.add(rs_2_0, rs_2_0, row_sum_ub)
                T.tile.div(ub_0, ub_0, row_sum_ub)
                T.reduce_sum(ub_1, row_sum_vec, dim=-1)
                T.tile.add(row_sum_vec, row_sum_vec, eps)
                T.tile.broadcast(row_sum_ub, row_sum_vec)
                T.tile.fill(rs_2_1, 0.0)
                T.tile.add(rs_2_1, rs_2_1, row_sum_ub)
                T.tile.div(ub_1, ub_1, row_sum_ub)
                T.reduce_sum(ub_2, row_sum_vec, dim=-1)
                T.tile.add(row_sum_vec, row_sum_vec, eps)
                T.tile.broadcast(row_sum_ub, row_sum_vec)
                T.tile.fill(rs_2_2, 0.0)
                T.tile.add(rs_2_2, rs_2_2, row_sum_ub)
                T.tile.div(ub_2, ub_2, row_sum_ub)
                T.reduce_sum(ub_3, row_sum_vec, dim=-1)
                T.tile.add(row_sum_vec, row_sum_vec, eps)
                T.tile.broadcast(row_sum_ub, row_sum_vec)
                T.tile.fill(rs_2_3, 0.0)
                T.tile.add(rs_2_3, rs_2_3, row_sum_ub)
                T.tile.div(ub_3, ub_3, row_sum_ub)
                T.tile.add(col_sum, ub_0, ub_1)
                T.tile.add(col_sum, col_sum, ub_2)
                T.tile.add(col_sum, col_sum, ub_3)
                T.tile.fill(row_sum_ub, eps)
                T.tile.add(col_sum, col_sum, row_sum_ub)
                T.tile.div(ub_0, ub_0, col_sum)
                T.tile.div(ub_1, ub_1, col_sum)
                T.tile.div(ub_2, ub_2, col_sum)
                T.tile.div(ub_3, ub_3, col_sum)
                T.tile.fill(cs3, 0.0)
                T.tile.add(cs3, cs3, col_sum)

                T.reduce_sum(ub_0, row_sum_vec, dim=-1)
                T.tile.add(row_sum_vec, row_sum_vec, eps)
                T.tile.broadcast(row_sum_ub, row_sum_vec)
                T.tile.fill(rs_3_0, 0.0)
                T.tile.add(rs_3_0, rs_3_0, row_sum_ub)
                T.tile.div(ub_0, ub_0, row_sum_ub)
                T.reduce_sum(ub_1, row_sum_vec, dim=-1)
                T.tile.add(row_sum_vec, row_sum_vec, eps)
                T.tile.broadcast(row_sum_ub, row_sum_vec)
                T.tile.fill(rs_3_1, 0.0)
                T.tile.add(rs_3_1, rs_3_1, row_sum_ub)
                T.tile.div(ub_1, ub_1, row_sum_ub)
                T.reduce_sum(ub_2, row_sum_vec, dim=-1)
                T.tile.add(row_sum_vec, row_sum_vec, eps)
                T.tile.broadcast(row_sum_ub, row_sum_vec)
                T.tile.fill(rs_3_2, 0.0)
                T.tile.add(rs_3_2, rs_3_2, row_sum_ub)
                T.tile.div(ub_2, ub_2, row_sum_ub)
                T.reduce_sum(ub_3, row_sum_vec, dim=-1)
                T.tile.add(row_sum_vec, row_sum_vec, eps)
                T.tile.broadcast(row_sum_ub, row_sum_vec)
                T.tile.fill(rs_3_3, 0.0)
                T.tile.add(rs_3_3, rs_3_3, row_sum_ub)
                T.tile.div(ub_3, ub_3, row_sum_ub)
                T.tile.add(col_sum, ub_0, ub_1)
                T.tile.add(col_sum, col_sum, ub_2)
                T.tile.add(col_sum, col_sum, ub_3)
                T.tile.fill(row_sum_ub, eps)
                T.tile.add(col_sum, col_sum, row_sum_ub)
                T.tile.div(ub_0, ub_0, col_sum)
                T.tile.div(ub_1, ub_1, col_sum)
                T.tile.div(ub_2, ub_2, col_sum)
                T.tile.div(ub_3, ub_3, col_sum)
                T.tile.fill(cs4, 0.0)
                T.tile.add(cs4, cs4, col_sum)

                T.reduce_sum(ub_0, row_sum_vec, dim=-1)
                T.tile.add(row_sum_vec, row_sum_vec, eps)
                T.tile.broadcast(row_sum_ub, row_sum_vec)
                T.tile.fill(rs_4_0, 0.0)
                T.tile.add(rs_4_0, rs_4_0, row_sum_ub)
                T.tile.div(ub_0, ub_0, row_sum_ub)
                T.reduce_sum(ub_1, row_sum_vec, dim=-1)
                T.tile.add(row_sum_vec, row_sum_vec, eps)
                T.tile.broadcast(row_sum_ub, row_sum_vec)
                T.tile.fill(rs_4_1, 0.0)
                T.tile.add(rs_4_1, rs_4_1, row_sum_ub)
                T.tile.div(ub_1, ub_1, row_sum_ub)
                T.reduce_sum(ub_2, row_sum_vec, dim=-1)
                T.tile.add(row_sum_vec, row_sum_vec, eps)
                T.tile.broadcast(row_sum_ub, row_sum_vec)
                T.tile.fill(rs_4_2, 0.0)
                T.tile.add(rs_4_2, rs_4_2, row_sum_ub)
                T.tile.div(ub_2, ub_2, row_sum_ub)
                T.reduce_sum(ub_3, row_sum_vec, dim=-1)
                T.tile.add(row_sum_vec, row_sum_vec, eps)
                T.tile.broadcast(row_sum_ub, row_sum_vec)
                T.tile.fill(rs_4_3, 0.0)
                T.tile.add(rs_4_3, rs_4_3, row_sum_ub)
                T.tile.div(ub_3, ub_3, row_sum_ub)
                T.tile.add(col_sum, ub_0, ub_1)
                T.tile.add(col_sum, col_sum, ub_2)
                T.tile.add(col_sum, col_sum, ub_3)
                T.tile.fill(row_sum_ub, eps)
                T.tile.add(col_sum, col_sum, row_sum_ub)
                T.tile.div(ub_0, ub_0, col_sum)
                T.tile.div(ub_1, ub_1, col_sum)
                T.tile.div(ub_2, ub_2, col_sum)
                T.tile.div(ub_3, ub_3, col_sum)
                T.tile.fill(cs5, 0.0)
                T.tile.add(cs5, cs5, col_sum)

                T.reduce_sum(ub_0, row_sum_vec, dim=-1)
                T.tile.add(row_sum_vec, row_sum_vec, eps)
                T.tile.broadcast(row_sum_ub, row_sum_vec)
                T.tile.fill(rs_5_0, 0.0)
                T.tile.add(rs_5_0, rs_5_0, row_sum_ub)
                T.tile.div(ub_0, ub_0, row_sum_ub)
                T.reduce_sum(ub_1, row_sum_vec, dim=-1)
                T.tile.add(row_sum_vec, row_sum_vec, eps)
                T.tile.broadcast(row_sum_ub, row_sum_vec)
                T.tile.fill(rs_5_1, 0.0)
                T.tile.add(rs_5_1, rs_5_1, row_sum_ub)
                T.tile.div(ub_1, ub_1, row_sum_ub)
                T.reduce_sum(ub_2, row_sum_vec, dim=-1)
                T.tile.add(row_sum_vec, row_sum_vec, eps)
                T.tile.broadcast(row_sum_ub, row_sum_vec)
                T.tile.fill(rs_5_2, 0.0)
                T.tile.add(rs_5_2, rs_5_2, row_sum_ub)
                T.tile.div(ub_2, ub_2, row_sum_ub)
                T.reduce_sum(ub_3, row_sum_vec, dim=-1)
                T.tile.add(row_sum_vec, row_sum_vec, eps)
                T.tile.broadcast(row_sum_ub, row_sum_vec)
                T.tile.fill(rs_5_3, 0.0)
                T.tile.add(rs_5_3, rs_5_3, row_sum_ub)
                T.tile.div(ub_3, ub_3, row_sum_ub)
                T.tile.add(col_sum, ub_0, ub_1)
                T.tile.add(col_sum, col_sum, ub_2)
                T.tile.add(col_sum, col_sum, ub_3)
                T.tile.fill(row_sum_ub, eps)
                T.tile.add(col_sum, col_sum, row_sum_ub)
                T.tile.div(ub_0, ub_0, col_sum)
                T.tile.div(ub_1, ub_1, col_sum)
                T.tile.div(ub_2, ub_2, col_sum)
                T.tile.div(ub_3, ub_3, col_sum)
                T.tile.fill(cs6, 0.0)
                T.tile.add(cs6, cs6, col_sum)

                T.reduce_sum(ub_0, row_sum_vec, dim=-1)
                T.tile.add(row_sum_vec, row_sum_vec, eps)
                T.tile.broadcast(row_sum_ub, row_sum_vec)
                T.tile.fill(rs_6_0, 0.0)
                T.tile.add(rs_6_0, rs_6_0, row_sum_ub)
                T.tile.div(ub_0, ub_0, row_sum_ub)
                T.reduce_sum(ub_1, row_sum_vec, dim=-1)
                T.tile.add(row_sum_vec, row_sum_vec, eps)
                T.tile.broadcast(row_sum_ub, row_sum_vec)
                T.tile.fill(rs_6_1, 0.0)
                T.tile.add(rs_6_1, rs_6_1, row_sum_ub)
                T.tile.div(ub_1, ub_1, row_sum_ub)
                T.reduce_sum(ub_2, row_sum_vec, dim=-1)
                T.tile.add(row_sum_vec, row_sum_vec, eps)
                T.tile.broadcast(row_sum_ub, row_sum_vec)
                T.tile.fill(rs_6_2, 0.0)
                T.tile.add(rs_6_2, rs_6_2, row_sum_ub)
                T.tile.div(ub_2, ub_2, row_sum_ub)
                T.reduce_sum(ub_3, row_sum_vec, dim=-1)
                T.tile.add(row_sum_vec, row_sum_vec, eps)
                T.tile.broadcast(row_sum_ub, row_sum_vec)
                T.tile.fill(rs_6_3, 0.0)
                T.tile.add(rs_6_3, rs_6_3, row_sum_ub)
                T.tile.div(ub_3, ub_3, row_sum_ub)
                T.tile.add(col_sum, ub_0, ub_1)
                T.tile.add(col_sum, col_sum, ub_2)
                T.tile.add(col_sum, col_sum, ub_3)
                T.tile.fill(row_sum_ub, eps)
                T.tile.add(col_sum, col_sum, row_sum_ub)
                T.tile.div(ub_0, ub_0, col_sum)
                T.tile.div(ub_1, ub_1, col_sum)
                T.tile.div(ub_2, ub_2, col_sum)
                T.tile.div(ub_3, ub_3, col_sum)
                T.tile.fill(cs7, 0.0)
                T.tile.add(cs7, cs7, col_sum)

                T.reduce_sum(ub_0, row_sum_vec, dim=-1)
                T.tile.add(row_sum_vec, row_sum_vec, eps)
                T.tile.broadcast(row_sum_ub, row_sum_vec)
                T.tile.fill(rs_7_0, 0.0)
                T.tile.add(rs_7_0, rs_7_0, row_sum_ub)
                T.tile.div(ub_0, ub_0, row_sum_ub)
                T.reduce_sum(ub_1, row_sum_vec, dim=-1)
                T.tile.add(row_sum_vec, row_sum_vec, eps)
                T.tile.broadcast(row_sum_ub, row_sum_vec)
                T.tile.fill(rs_7_1, 0.0)
                T.tile.add(rs_7_1, rs_7_1, row_sum_ub)
                T.tile.div(ub_1, ub_1, row_sum_ub)
                T.reduce_sum(ub_2, row_sum_vec, dim=-1)
                T.tile.add(row_sum_vec, row_sum_vec, eps)
                T.tile.broadcast(row_sum_ub, row_sum_vec)
                T.tile.fill(rs_7_2, 0.0)
                T.tile.add(rs_7_2, rs_7_2, row_sum_ub)
                T.tile.div(ub_2, ub_2, row_sum_ub)
                T.reduce_sum(ub_3, row_sum_vec, dim=-1)
                T.tile.add(row_sum_vec, row_sum_vec, eps)
                T.tile.broadcast(row_sum_ub, row_sum_vec)
                T.tile.fill(rs_7_3, 0.0)
                T.tile.add(rs_7_3, rs_7_3, row_sum_ub)
                T.tile.div(ub_3, ub_3, row_sum_ub)
                T.tile.add(col_sum, ub_0, ub_1)
                T.tile.add(col_sum, col_sum, ub_2)
                T.tile.add(col_sum, col_sum, ub_3)
                T.tile.fill(row_sum_ub, eps)
                T.tile.add(col_sum, col_sum, row_sum_ub)
                T.tile.div(ub_0, ub_0, col_sum)
                T.tile.div(ub_1, ub_1, col_sum)
                T.tile.div(ub_2, ub_2, col_sum)
                T.tile.div(ub_3, ub_3, col_sum)
                T.tile.fill(cs8, 0.0)
                T.tile.add(cs8, cs8, col_sum)

                T.reduce_sum(ub_0, row_sum_vec, dim=-1)
                T.tile.add(row_sum_vec, row_sum_vec, eps)
                T.tile.broadcast(row_sum_ub, row_sum_vec)
                T.tile.fill(rs_8_0, 0.0)
                T.tile.add(rs_8_0, rs_8_0, row_sum_ub)
                T.tile.div(ub_0, ub_0, row_sum_ub)
                T.reduce_sum(ub_1, row_sum_vec, dim=-1)
                T.tile.add(row_sum_vec, row_sum_vec, eps)
                T.tile.broadcast(row_sum_ub, row_sum_vec)
                T.tile.fill(rs_8_1, 0.0)
                T.tile.add(rs_8_1, rs_8_1, row_sum_ub)
                T.tile.div(ub_1, ub_1, row_sum_ub)
                T.reduce_sum(ub_2, row_sum_vec, dim=-1)
                T.tile.add(row_sum_vec, row_sum_vec, eps)
                T.tile.broadcast(row_sum_ub, row_sum_vec)
                T.tile.fill(rs_8_2, 0.0)
                T.tile.add(rs_8_2, rs_8_2, row_sum_ub)
                T.tile.div(ub_2, ub_2, row_sum_ub)
                T.reduce_sum(ub_3, row_sum_vec, dim=-1)
                T.tile.add(row_sum_vec, row_sum_vec, eps)
                T.tile.broadcast(row_sum_ub, row_sum_vec)
                T.tile.fill(rs_8_3, 0.0)
                T.tile.add(rs_8_3, rs_8_3, row_sum_ub)
                T.tile.div(ub_3, ub_3, row_sum_ub)
                T.tile.add(col_sum, ub_0, ub_1)
                T.tile.add(col_sum, col_sum, ub_2)
                T.tile.add(col_sum, col_sum, ub_3)
                T.tile.fill(row_sum_ub, eps)
                T.tile.add(col_sum, col_sum, row_sum_ub)
                T.tile.div(ub_0, ub_0, col_sum)
                T.tile.div(ub_1, ub_1, col_sum)
                T.tile.div(ub_2, ub_2, col_sum)
                T.tile.div(ub_3, ub_3, col_sum)
                T.tile.fill(cs9, 0.0)
                T.tile.add(cs9, cs9, col_sum)

                T.set_flag("v", "mte3", 0)
                T.wait_flag("v", "mte3", 0)

                T.copy(ub_0[:, 0:hidden_size], row0_out[row_start : row_start + sub_block_tokens, :])
                T.copy(ub_1[:, 0:hidden_size], row1_out[row_start : row_start + sub_block_tokens, :])
                T.copy(ub_2[:, 0:hidden_size], row2_out[row_start : row_start + sub_block_tokens, :])
                T.copy(ub_3[:, 0:hidden_size], row3_out[row_start : row_start + sub_block_tokens, :])

                T.copy(cs0, sums_out[row_start : row_start + sub_block_tokens, 0:8])
                T.copy(rs_0_0, sums_out[row_start : row_start + sub_block_tokens, 8:16])
                T.copy(rs_0_1, sums_out[row_start : row_start + sub_block_tokens, 16:24])
                T.copy(rs_0_2, sums_out[row_start : row_start + sub_block_tokens, 24:32])
                T.copy(rs_0_3, sums_out[row_start : row_start + sub_block_tokens, 32:40])
                T.copy(cs1, sums_out[row_start : row_start + sub_block_tokens, 40:48])
                T.copy(rs_1_0, sums_out[row_start : row_start + sub_block_tokens, 48:56])
                T.copy(rs_1_1, sums_out[row_start : row_start + sub_block_tokens, 56:64])
                T.copy(rs_1_2, sums_out[row_start : row_start + sub_block_tokens, 64:72])
                T.copy(rs_1_3, sums_out[row_start : row_start + sub_block_tokens, 72:80])
                T.copy(cs2, sums_out[row_start : row_start + sub_block_tokens, 80:88])
                T.copy(rs_2_0, sums_out[row_start : row_start + sub_block_tokens, 88:96])
                T.copy(rs_2_1, sums_out[row_start : row_start + sub_block_tokens, 96:104])
                T.copy(rs_2_2, sums_out[row_start : row_start + sub_block_tokens, 104:112])
                T.copy(rs_2_3, sums_out[row_start : row_start + sub_block_tokens, 112:120])
                T.copy(cs3, sums_out[row_start : row_start + sub_block_tokens, 120:128])
                T.copy(rs_3_0, sums_out[row_start : row_start + sub_block_tokens, 128:136])
                T.copy(rs_3_1, sums_out[row_start : row_start + sub_block_tokens, 136:144])
                T.copy(rs_3_2, sums_out[row_start : row_start + sub_block_tokens, 144:152])
                T.copy(rs_3_3, sums_out[row_start : row_start + sub_block_tokens, 152:160])
                T.copy(cs4, sums_out[row_start : row_start + sub_block_tokens, 160:168])
                T.copy(rs_4_0, sums_out[row_start : row_start + sub_block_tokens, 168:176])
                T.copy(rs_4_1, sums_out[row_start : row_start + sub_block_tokens, 176:184])
                T.copy(rs_4_2, sums_out[row_start : row_start + sub_block_tokens, 184:192])
                T.copy(rs_4_3, sums_out[row_start : row_start + sub_block_tokens, 192:200])
                T.copy(cs5, sums_out[row_start : row_start + sub_block_tokens, 200:208])
                T.copy(rs_5_0, sums_out[row_start : row_start + sub_block_tokens, 208:216])
                T.copy(rs_5_1, sums_out[row_start : row_start + sub_block_tokens, 216:224])
                T.copy(rs_5_2, sums_out[row_start : row_start + sub_block_tokens, 224:232])
                T.copy(rs_5_3, sums_out[row_start : row_start + sub_block_tokens, 232:240])
                T.copy(cs6, sums_out[row_start : row_start + sub_block_tokens, 240:248])
                T.copy(rs_6_0, sums_out[row_start : row_start + sub_block_tokens, 248:256])
                T.copy(rs_6_1, sums_out[row_start : row_start + sub_block_tokens, 256:264])
                T.copy(rs_6_2, sums_out[row_start : row_start + sub_block_tokens, 264:272])
                T.copy(rs_6_3, sums_out[row_start : row_start + sub_block_tokens, 272:280])
                T.copy(cs7, sums_out[row_start : row_start + sub_block_tokens, 280:288])
                T.copy(rs_7_0, sums_out[row_start : row_start + sub_block_tokens, 288:296])
                T.copy(rs_7_1, sums_out[row_start : row_start + sub_block_tokens, 296:304])
                T.copy(rs_7_2, sums_out[row_start : row_start + sub_block_tokens, 304:312])
                T.copy(rs_7_3, sums_out[row_start : row_start + sub_block_tokens, 312:320])
                T.copy(cs8, sums_out[row_start : row_start + sub_block_tokens, 320:328])
                T.copy(rs_8_0, sums_out[row_start : row_start + sub_block_tokens, 328:336])
                T.copy(rs_8_1, sums_out[row_start : row_start + sub_block_tokens, 336:344])
                T.copy(rs_8_2, sums_out[row_start : row_start + sub_block_tokens, 344:352])
                T.copy(rs_8_3, sums_out[row_start : row_start + sub_block_tokens, 352:360])
                T.copy(cs9, sums_out[row_start : row_start + sub_block_tokens, 360:368])

                T.set_flag("mte3", "mte2", 0)
                T.wait_flag("mte3", "mte2", 0)

    return mhc_sinkhorn_kernel


@tilelang.jit(pass_configs=_BWD_PASS_CONFIGS)
def _mhc_sinkhorn_bwd(
    hidden_size: int,
    token_block_size: int,
    repeat: int,
    eps: float,
) -> tilelang.JITKernel:
    assert hidden_size == 4
    num_tokens = T.symbolic("num_tokens")
    dtype = "float32"
    pad_h = ((hidden_size + 7) // 8) * 8
    blk = token_block_size
    sums_cols = _compute_sums_cols(hidden_size, repeat)
    step_stride = (hidden_size + 1) * pad_h

    @T.prim_func
    def mhc_sinkhorn_backward_kernel(
        grad_row0_in: T.Tensor[(num_tokens, pad_h), dtype],
        grad_row1_in: T.Tensor[(num_tokens, pad_h), dtype],
        grad_row2_in: T.Tensor[(num_tokens, pad_h), dtype],
        grad_row3_in: T.Tensor[(num_tokens, pad_h), dtype],
        y_row0_in: T.Tensor[(num_tokens, pad_h), dtype],
        y_row1_in: T.Tensor[(num_tokens, pad_h), dtype],
        y_row2_in: T.Tensor[(num_tokens, pad_h), dtype],
        y_row3_in: T.Tensor[(num_tokens, pad_h), dtype],
        sums_in: T.Tensor[(num_tokens, sums_cols), dtype],
        pad_mask: T.Tensor[(pad_h,), dtype],
        grad_row0_out: T.Tensor[(num_tokens, hidden_size), dtype],
        grad_row1_out: T.Tensor[(num_tokens, hidden_size), dtype],
        grad_row2_out: T.Tensor[(num_tokens, hidden_size), dtype],
        grad_row3_out: T.Tensor[(num_tokens, hidden_size), dtype],
    ) -> None:
        with T.Kernel(T.ceildiv(num_tokens, blk * VEC_NUM), is_npu=True) as (cid, vid):
            Y_0 = T.alloc_ub((blk, pad_h), dtype)
            Y_1 = T.alloc_ub((blk, pad_h), dtype)
            Y_2 = T.alloc_ub((blk, pad_h), dtype)
            Y_3 = T.alloc_ub((blk, pad_h), dtype)

            g_0 = T.alloc_ub((blk, pad_h), dtype)
            g_1 = T.alloc_ub((blk, pad_h), dtype)
            g_2 = T.alloc_ub((blk, pad_h), dtype)
            g_3 = T.alloc_ub((blk, pad_h), dtype)

            csum_ub = T.alloc_ub((blk, pad_h), dtype)
            csum_gy = T.alloc_ub((blk, pad_h), dtype)
            rs_0 = T.alloc_ub((blk, pad_h), dtype)
            rs_1 = T.alloc_ub((blk, pad_h), dtype)
            rs_2 = T.alloc_ub((blk, pad_h), dtype)
            rs_3 = T.alloc_ub((blk, pad_h), dtype)

            rdiv = T.alloc_ub((blk, pad_h), dtype)
            rh_v = T.alloc_ub((blk,), dtype)
            rh_b = T.alloc_ub((blk, pad_h), dtype)

            mask_1d = T.alloc_ub(pad_h, dtype)

            row_start = cid * blk * VEC_NUM + vid * blk

            T.set_flag("mte3", "mte2", 0)
            T.wait_flag("mte3", "mte2", 0)

            T.copy(pad_mask, mask_1d)
            T.copy(grad_row0_in[row_start : row_start + blk, :], g_0)
            T.copy(grad_row1_in[row_start : row_start + blk, :], g_1)
            T.copy(grad_row2_in[row_start : row_start + blk, :], g_2)
            T.copy(grad_row3_in[row_start : row_start + blk, :], g_3)
            T.copy(y_row0_in[row_start : row_start + blk, :], Y_0)
            T.copy(y_row1_in[row_start : row_start + blk, :], Y_1)
            T.copy(y_row2_in[row_start : row_start + blk, :], Y_2)
            T.copy(y_row3_in[row_start : row_start + blk, :], Y_3)

            T.set_flag("mte2", "v", 0)
            T.wait_flag("mte2", "v", 0)

            for inv_step in T.serial(repeat):
                step_idx = repeat - 1 - inv_step

                csum_offset = step_idx * step_stride
                T.copy(sums_in[row_start : row_start + blk, csum_offset : csum_offset + pad_h], csum_ub)

                if step_idx > 0:
                    rs_base = step_idx * step_stride - hidden_size * pad_h
                    T.copy(sums_in[row_start : row_start + blk, rs_base : rs_base + pad_h], rs_0)
                    T.copy(sums_in[row_start : row_start + blk, rs_base + pad_h : rs_base + 2 * pad_h], rs_1)
                    T.copy(sums_in[row_start : row_start + blk, rs_base + 2 * pad_h : rs_base + 3 * pad_h], rs_2)
                    T.copy(sums_in[row_start : row_start + blk, rs_base + 3 * pad_h : rs_base + 4 * pad_h], rs_3)

                T.set_flag("mte2", "v", 0)
                T.wait_flag("mte2", "v", 0)

                T.tile.mul(csum_gy, g_0, Y_0)
                T.tile.mul(rdiv, g_1, Y_1)
                T.tile.add(csum_gy, csum_gy, rdiv)
                T.tile.mul(rdiv, g_2, Y_2)
                T.tile.add(csum_gy, csum_gy, rdiv)
                T.tile.mul(rdiv, g_3, Y_3)
                T.tile.add(csum_gy, csum_gy, rdiv)

                T.tile.broadcast(rdiv, mask_1d)
                T.tile.div(rdiv, rdiv, csum_ub)
                T.tile.sub(g_0, g_0, csum_gy)
                T.tile.mul(g_0, g_0, rdiv)
                T.tile.sub(g_1, g_1, csum_gy)
                T.tile.mul(g_1, g_1, rdiv)
                T.tile.sub(g_2, g_2, csum_gy)
                T.tile.mul(g_2, g_2, rdiv)
                T.tile.sub(g_3, g_3, csum_gy)
                T.tile.mul(g_3, g_3, rdiv)

                T.tile.mul(Y_0, Y_0, csum_ub)
                T.tile.mul(Y_1, Y_1, csum_ub)
                T.tile.mul(Y_2, Y_2, csum_ub)
                T.tile.mul(Y_3, Y_3, csum_ub)

                if step_idx > 0:
                    T.tile.mul(rdiv, g_0, Y_0)
                    T.reduce_sum(rdiv, rh_v, dim=-1)
                    T.tile.broadcast(rh_b, rh_v)
                    T.tile.sub(g_0, g_0, rh_b)
                    T.tile.div(g_0, g_0, rs_0)
                    T.tile.broadcast(rdiv, mask_1d)
                    T.tile.mul(g_0, g_0, rdiv)

                    T.tile.mul(rdiv, g_1, Y_1)
                    T.reduce_sum(rdiv, rh_v, dim=-1)
                    T.tile.broadcast(rh_b, rh_v)
                    T.tile.sub(g_1, g_1, rh_b)
                    T.tile.div(g_1, g_1, rs_1)
                    T.tile.broadcast(rdiv, mask_1d)
                    T.tile.mul(g_1, g_1, rdiv)

                    T.tile.mul(rdiv, g_2, Y_2)
                    T.reduce_sum(rdiv, rh_v, dim=-1)
                    T.tile.broadcast(rh_b, rh_v)
                    T.tile.sub(g_2, g_2, rh_b)
                    T.tile.div(g_2, g_2, rs_2)
                    T.tile.broadcast(rdiv, mask_1d)
                    T.tile.mul(g_2, g_2, rdiv)

                    T.tile.mul(rdiv, g_3, Y_3)
                    T.reduce_sum(rdiv, rh_v, dim=-1)
                    T.tile.broadcast(rh_b, rh_v)
                    T.tile.sub(g_3, g_3, rh_b)
                    T.tile.div(g_3, g_3, rs_3)
                    T.tile.broadcast(rdiv, mask_1d)
                    T.tile.mul(g_3, g_3, rdiv)

                    T.tile.mul(Y_0, Y_0, rs_0)
                    T.tile.mul(Y_1, Y_1, rs_1)
                    T.tile.mul(Y_2, Y_2, rs_2)
                    T.tile.mul(Y_3, Y_3, rs_3)

                T.set_flag("v", "mte2", 1)
                T.wait_flag("v", "mte2", 1)

            T.tile.mul(g_0, g_0, Y_0)
            T.reduce_sum(g_0, rh_v, dim=-1)
            T.tile.broadcast(rh_b, rh_v)
            T.tile.mul(rdiv, Y_0, rh_b)
            T.tile.sub(g_0, g_0, rdiv)

            T.tile.mul(g_1, g_1, Y_1)
            T.reduce_sum(g_1, rh_v, dim=-1)
            T.tile.broadcast(rh_b, rh_v)
            T.tile.mul(rdiv, Y_1, rh_b)
            T.tile.sub(g_1, g_1, rdiv)

            T.tile.mul(g_2, g_2, Y_2)
            T.reduce_sum(g_2, rh_v, dim=-1)
            T.tile.broadcast(rh_b, rh_v)
            T.tile.mul(rdiv, Y_2, rh_b)
            T.tile.sub(g_2, g_2, rdiv)

            T.tile.mul(g_3, g_3, Y_3)
            T.reduce_sum(g_3, rh_v, dim=-1)
            T.tile.broadcast(rh_b, rh_v)
            T.tile.mul(rdiv, Y_3, rh_b)
            T.tile.sub(g_3, g_3, rdiv)

            T.set_flag("v", "mte3", 0)
            T.wait_flag("v", "mte3", 0)

            T.copy(g_0[:, 0:hidden_size], grad_row0_out[row_start : row_start + blk, :])
            T.copy(g_1[:, 0:hidden_size], grad_row1_out[row_start : row_start + blk, :])
            T.copy(g_2[:, 0:hidden_size], grad_row2_out[row_start : row_start + blk, :])
            T.copy(g_3[:, 0:hidden_size], grad_row3_out[row_start : row_start + blk, :])

            T.set_flag("mte3", "mte2", 0)
            T.wait_flag("mte3", "mte2", 0)

    return mhc_sinkhorn_backward_kernel


def sinkhorn_normalize_ref(x: torch.Tensor, repeat: int = 10, eps: float = 1e-6) -> torch.Tensor:
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


def _prepare_fwd_inputs(x: torch.Tensor, pad_h: int, token_block_size: int):
    n, h, w = x.shape
    padded_n = math.ceil(n / token_block_size) * token_block_size
    row_inputs = []
    for i in range(h):
        row_i_padded = torch.full((padded_n, pad_h), _LARGE_NEG, dtype=x.dtype, device=x.device)
        row_i_padded[:n, :w] = x[:, i, :]
        row_inputs.append(row_i_padded)
    row_outputs = []
    for _i in range(h):
        row_outputs.append(torch.empty((padded_n, h), dtype=x.dtype, device=x.device))
    return row_inputs, row_outputs, padded_n


def _prepare_fwd_sums_output(padded_n: int, sums_cols: int, dtype, device):
    return torch.ones((padded_n, sums_cols), dtype=dtype, device=device)


def _extract_fwd_output(row_outputs, n: int, hidden_size: int) -> torch.Tensor:
    out = torch.stack([row_outputs[i][:n, :] for i in range(hidden_size)], dim=1)
    return out


def _extract_fwd_sums(sums_out, n: int) -> torch.Tensor:
    return sums_out[:n, :].clone()


def _prepare_bwd_inputs(
    grad_output: torch.Tensor, y_3d: torch.Tensor, sums_n: torch.Tensor, pad_h: int, token_block_size: int, hidden_size: int, sums_cols: int
):
    n = grad_output.shape[0]
    padded_n = math.ceil(n / (token_block_size * VEC_NUM)) * (token_block_size * VEC_NUM)

    grad_row_inputs = []
    for i in range(hidden_size):
        row_i_padded = torch.zeros((padded_n, pad_h), dtype=grad_output.dtype, device=grad_output.device)
        row_i_padded[:n, :hidden_size] = grad_output[:, i, :]
        grad_row_inputs.append(row_i_padded)

    y_row_inputs = []
    for i in range(hidden_size):
        row_i_padded = torch.zeros((padded_n, pad_h), dtype=y_3d.dtype, device=y_3d.device)
        row_i_padded[:n, :hidden_size] = y_3d[:, i, :]
        y_row_inputs.append(row_i_padded)

    sums_padded = torch.ones((padded_n, sums_cols), dtype=sums_n.dtype, device=sums_n.device)
    sums_padded[:n, :] = sums_n

    grad_row_outputs = []
    for _i in range(hidden_size):
        grad_row_outputs.append(torch.empty((padded_n, hidden_size), dtype=y_3d.dtype, device=y_3d.device))

    return grad_row_inputs, y_row_inputs, sums_padded, grad_row_outputs, padded_n


def _extract_bwd_output(grad_row_outputs, n: int, hidden_size: int) -> torch.Tensor:
    out = torch.stack([grad_row_outputs[i][:n, :] for i in range(hidden_size)], dim=1)
    return out


def _create_pad_mask(pad_h: int, hidden_size: int, dtype, device):
    mask = torch.zeros(pad_h, dtype=dtype, device=device)
    mask[:hidden_size] = 1.0
    return mask


def test_fwd():
    n = 8192
    hidden_size = 4
    repeat = 10
    eps = 1e-6
    token_block_size = 128
    pad_h = ((hidden_size + 7) // 8) * 8
    sums_cols = _compute_sums_cols(hidden_size, repeat)

    device = "npu"

    torch.manual_seed(42)
    x = torch.randn((n, hidden_size, hidden_size), dtype=torch.float32, device=device)

    out_ref = sinkhorn_normalize_ref(x, repeat, eps)

    row_inputs, row_outputs, padded_n = _prepare_fwd_inputs(x, pad_h, token_block_size)
    sums_out = _prepare_fwd_sums_output(padded_n, sums_cols, torch.float32, device)

    fwd_func = _mhc_sinkhorn_fwd(hidden_size, token_block_size, repeat, eps)
    fwd_func(
        row_inputs[0],
        row_inputs[1],
        row_inputs[2],
        row_inputs[3],
        row_outputs[0],
        row_outputs[1],
        row_outputs[2],
        row_outputs[3],
        sums_out,
    )

    out_tl = _extract_fwd_output(row_outputs, n, hidden_size)

    torch.testing.assert_close(out_tl, out_ref, atol=1e-4, rtol=1e-4)
    print("Kernel Output Match!")


def test_bwd():
    n = 8192
    hidden_size = 4
    repeat = 10
    eps = 1e-6
    token_block_size_fwd = 128
    token_block_size_bwd = 64
    pad_h = ((hidden_size + 7) // 8) * 8
    sums_cols = _compute_sums_cols(hidden_size, repeat)

    device = "npu"

    torch.manual_seed(42)
    x_ref = torch.randn((n, hidden_size, hidden_size), dtype=torch.float32, device=device, requires_grad=True)
    grad_output = torch.randn((n, hidden_size, hidden_size), dtype=torch.float32, device=device)

    out_ref = sinkhorn_normalize_ref(x_ref, repeat, eps)
    torch.autograd.backward(out_ref, grad_output)
    ref_grad = x_ref.grad

    x_tl = x_ref.detach().clone()

    row_inputs, row_outputs, padded_n_fwd = _prepare_fwd_inputs(x_tl, pad_h, token_block_size_fwd)
    sums_out_fwd = _prepare_fwd_sums_output(padded_n_fwd, sums_cols, torch.float32, device)

    fwd_func = _mhc_sinkhorn_fwd(hidden_size, token_block_size_fwd, repeat, eps)
    fwd_func(
        row_inputs[0],
        row_inputs[1],
        row_inputs[2],
        row_inputs[3],
        row_outputs[0],
        row_outputs[1],
        row_outputs[2],
        row_outputs[3],
        sums_out_fwd,
    )

    y_3d = _extract_fwd_output(row_outputs, n, hidden_size)
    sums_n = _extract_fwd_sums(sums_out_fwd, n)

    grad_row_inputs, y_row_inputs, sums_padded, grad_row_outputs, padded_n_bwd = _prepare_bwd_inputs(
        grad_output, y_3d, sums_n, pad_h, token_block_size_bwd, hidden_size, sums_cols
    )
    pad_mask = _create_pad_mask(pad_h, hidden_size, torch.float32, device)

    bwd_func = _mhc_sinkhorn_bwd(hidden_size, token_block_size_bwd, repeat, eps)
    bwd_func(
        grad_row_inputs[0],
        grad_row_inputs[1],
        grad_row_inputs[2],
        grad_row_inputs[3],
        y_row_inputs[0],
        y_row_inputs[1],
        y_row_inputs[2],
        y_row_inputs[3],
        sums_padded,
        pad_mask,
        grad_row_outputs[0],
        grad_row_outputs[1],
        grad_row_outputs[2],
        grad_row_outputs[3],
    )

    grad_input_tl = _extract_bwd_output(grad_row_outputs, n, hidden_size)

    torch.testing.assert_close(grad_input_tl, ref_grad, atol=1e-4, rtol=1e-4)
    print("Kernel Output Match!")


if __name__ == "__main__":
    test_fwd()
    test_bwd()
