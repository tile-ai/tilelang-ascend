import argparse

import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Random Number Generation Kernel")
parser.add_argument("--m", type=int, default=1024, help="Number of elements")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--block_size", type=int, default=128, help="Block size per vid")
parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
args = parser.parse_args()

M = args.m
SEED = args.seed
BLOCK_SIZE = args.block_size

LCG_A = 1103515245
LCG_C = 12345

VEC_NUM = 2
TOTAL_BLOCK_SIZE = BLOCK_SIZE * VEC_NUM

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def random_1d(M, block_size, seed, lcg_a, lcg_c):
    M_aligned = (M + TOTAL_BLOCK_SIZE - 1) // TOTAL_BLOCK_SIZE * TOTAL_BLOCK_SIZE
    num_blocks = M_aligned // TOTAL_BLOCK_SIZE

    @T.prim_func
    def main(
        output: T.Tensor((M_aligned,), "int32"),
    ):
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            idx_ub = T.alloc_ub((block_size,), "int32")
            state_ub = T.alloc_ub((block_size,), "int32")
            temp_ub = T.alloc_ub((block_size,), "int32")
            a_ub = T.alloc_ub((block_size,), "int32")
            c_ub = T.alloc_ub((block_size,), "int32")
            seed_ub = T.alloc_ub((block_size,), "int32")

            T.tile.fill(a_ub, lcg_a)
            T.tile.fill(c_ub, lcg_c)
            T.tile.fill(seed_ub, seed)

            global_base = cid * TOTAL_BLOCK_SIZE + vid * block_size
            T.tile.arith_progression(idx_ub, global_base, 1, block_size)

            T.tile.add(state_ub, idx_ub, seed_ub)

            for _ in range(3):
                T.tile.mul(temp_ub, state_ub, a_ub)
                T.tile.add(state_ub, temp_ub, c_ub)

            base_offset = cid * TOTAL_BLOCK_SIZE + vid * block_size
            T.copy(state_ub, output[base_offset : base_offset + block_size])

    return main


def reference_random_1d(M, seed):
    import numpy as np
    import ctypes

    result = np.zeros(M, dtype=np.int32)
    for i in range(M):
        state = seed + i
        state = ctypes.c_int32(state * LCG_A + LCG_C).value
        state = ctypes.c_int32(state * LCG_A + LCG_C).value
        state = ctypes.c_int32(state * LCG_A + LCG_C).value
        result[i] = state
    return torch.from_numpy(result)


if __name__ == "__main__":
    func = random_1d(M, BLOCK_SIZE, SEED, LCG_A, LCG_C)

    print("init successful!")

    output = func()

    output_truncated = output[:M]

    ref_output = reference_random_1d(M, SEED)
    torch.testing.assert_close(output_truncated.cpu(), ref_output, rtol=0, atol=0)
    print("Random Kernel Output Match!")
