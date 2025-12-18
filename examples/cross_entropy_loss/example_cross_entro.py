
from typing import Literal
import argparse

import tilelang as tl
import tilelang.language as T
from tilelang import DataType

import torch
from torch import nn

@tl.jit(
    out_idx=[-2, -1],
    pass_configs={
        tl.PassConfigKey.TIR_MERGE_STATIC_SMEM: True,
        tl.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    }
)
def cross_entropy(
    N:int, C:int, block_N:int, block_C:int,
    x_dtype: Literal["float16", "bfloat16", "float32"] = "float16",
    y_dtype: Literal["int32", "int64"] = "int32"
):

    VEC_NUM = 2
    CAL_DTYPE = "float32"
    TEMP_DTYPE = "uint8"
    CAST_MODE_LOW2HIGH = "CAST_NONE"
    CAST_MODE_HIGH2LOW = "CAST_RINT"

    n_num = T.ceildiv(N, block_N)
    c_num = T.ceildiv(C, block_C)
    block_N_2 = T.ceildiv(block_N, VEC_NUM)
    n_2_num = T.ceildiv(N, block_N_2)

    def bytes_of(dtype: str) -> int:
        return DataType(dtype).bits // 8

    not_same_dtype = x_dtype != CAL_DTYPE

    def cast_or_copy(dst, src, mode, count):
        if not_same_dtype:
            return T.tile.cast_tl(dst, src, mode, count)
        else:
            return T.copy(src, dst)

    @T.prim_func
    def main(
        x: T.Tensor([N, C], x_dtype),         # type: ignore
        y: T.Tensor([N], y_dtype),            # type: ignore
        loss: T.Tensor([N], x_dtype),         # type: ignore
        log_prob: T.Tensor([N, C], x_dtype),  # type: ignore
    ):
        with T.Kernel(n_num, is_npu=True) as (cid, vid):
            bn = (cid * VEC_NUM + vid) % n_2_num
            
            x_ub = T.alloc_ub([block_N_2, block_C], x_dtype)
            x_32 = T.alloc_ub([block_N_2, block_C], CAL_DTYPE)
            tile_max = T.alloc_ub([block_N_2], CAL_DTYPE)
            prev_max = T.alloc_ub([block_N_2], CAL_DTYPE)  # x_max
            tile_sum = T.alloc_ub([block_N_2], CAL_DTYPE)
            prev_sum = T.alloc_ub([block_N_2], CAL_DTYPE)  # sum e^{x_c - x_max}
            temp_exp = T.alloc_ub([block_N_2], CAL_DTYPE)
            temp_reduce = T.alloc_ub([3 * bytes_of(CAL_DTYPE) * block_N_2 * block_C], TEMP_DTYPE)

            y_ub = T.alloc_ub([block_N_2], y_dtype)
            l_n_32 = T.alloc_ub([block_N_2], CAL_DTYPE)
            l_n = T.alloc_ub([block_N_2], x_dtype)

            with T.Scope("V"):
                T.tile.fill(prev_max, -T.infinity(CAL_DTYPE))
                T.tile.fill(prev_sum, 0.0)

                # Find sum e^{x_c - x_max}
                for bc in T.serial(c_num):
                    T.copy(x[bn * block_N_2, bc * block_C], x_ub)
                    cast_or_copy(x_32, x_ub, CAST_MODE_LOW2HIGH, block_N_2 * block_C)

                    T.tile.reduce_max(tile_max, x_32, temp_reduce, dim=-1)
                    T.tile.max(tile_max, prev_max, tile_max)
                    T.tile.sub(temp_exp, prev_max, tile_max)
                    T.tile.exp(temp_exp, temp_exp)
                    T.tile.mul(temp_exp, prev_sum, temp_exp)
                    for n_idx in T.serial(block_N_2):
                        T.tile.sub(x_32[n_idx, :], x_32[n_idx, :], tile_max[n_idx])
                    T.tile.exp(x_32, x_32)
                    T.tile.reduce_sum(tile_sum, x_32, temp_reduce, dim=-1)
                    T.tile.add(prev_sum, tile_sum, temp_exp)
                    T.copy(tile_max, prev_max)
                
                T.copy(y[bn * block_N_2], y_ub)
                T.tile.ln(prev_sum, prev_sum)  # log(sum e^{x_c - x_max})
                # log(e^{x_c - x_max} / sum e^{x_c - x_max}) = x_c - x_max - log(sum e^{x_c - x_max})
                for bc in T.serial(c_num):
                    T.copy(x[bn * block_N_2, bc * block_C], x_ub)
                    cast_or_copy(x_32, x_ub, CAST_MODE_LOW2HIGH, block_N_2 * block_C)

                    for n_idx in T.serial(block_N_2):
                        T.tile.sub(x_32[n_idx, :], x_32[n_idx, :], prev_max[n_idx] + prev_sum[n_idx])  # x_c - (x_max + log(sum e^{x_c - x_max}))
                    
                    cast_or_copy(x_ub, x_32, CAST_MODE_HIGH2LOW, block_N_2 * block_C)
                    T.copy(x_ub, log_prob[bn * block_N_2, bc * block_C])
                    
                    for n_idx in T.serial(block_N_2):
                        if 0 <= y_ub[n_idx] and y_ub[n_idx] < block_C:
                            l_n_32[n_idx] = -x_32[n_idx, y_ub[n_idx]]  # -(x_c - x_max - log(sum e^{x_c - x_max}))
                    T.tile.sub(y_ub, y_ub, block_C)

                cast_or_copy(l_n, l_n_32, CAST_MODE_HIGH2LOW, block_N_2)
                T.copy(l_n, loss[bn * block_N_2])

    return main
        
def ref_program(x, y):
    cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
    loss = cross_entropy_loss(x, y)
    log_prob = nn.functional.log_softmax(x, dim=-1)
    return loss, log_prob

def ref_program_npu(x, y):
    import torch_npu
    loss, log_prob, _, _ = torch_npu.npu_cross_entropy_loss(x, y, reduction="none")
    return loss, log_prob


def check_case(N:int, C:int, block_N: int = 128, block_C: int = 128, x_dtype="float16", y_dtype="int32"):
    torch_dtype_map = {
        "float16": torch.half, "float32": torch.float32, "float": torch.float32, "bfloat16": torch.bfloat16,
        "int32": torch.int32, "int64": torch.int64
    }
    x = torch.randn(N, C).to(torch_dtype_map[x_dtype]).npu()
    y = torch.randint(0, C, (N,)).to(torch_dtype_map[y_dtype]).npu()

    kernel = cross_entropy(N, C, block_N, block_C, x_dtype=x_dtype, y_dtype=y_dtype)

    loss, log_prob = kernel(x, y)
    ref_loss, ref_log_prob = ref_program(x, y)

    torch.testing.assert_close(loss, ref_loss, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(log_prob, ref_log_prob, rtol=1e-2, atol=1e-2)


def main(custom_args=None):
    parser = argparse.ArgumentParser(description="cross_entropy_loss Example")
    parser.add_argument("--n", type=int, default=1024, help="Matrix dimension N")
    parser.add_argument("--c", type=int, default=1024, help="Matrix dimension C")
    args, remains = parser.parse_known_args(custom_args)
    if remains:
        print(f"[{parser.description}]", "Unknown args:", remains)
    N, C = args.n, args.c

    torch.manual_seed(0)

    check_case(N, C, 128, 128)
    check_case(N, C, 128, 128, x_dtype="float32")
    check_case(N, C, 128, 128, x_dtype="bfloat16")
    check_case(64, 64, 16, 16)

    print("cross_entropy_loss example passed!")
    print("Kernel Output Match!")

if __name__ == "__main__":
    main()