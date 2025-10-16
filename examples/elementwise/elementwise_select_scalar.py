import argparse

import tilelang
import tilelang.language as T
import torch
import numpy as np

tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--m", type=int, default=256, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=256, help="Matrix N dimension")
args = parser.parse_args()

M = args.m
N = args.n

def select_by_mask_bits(A, b, M):
    """
    select value by mask bits of M from A or B to assign to C
    
    Args:
        A: torch.Tensor, shape (128, 128), float32
        B: torch.Tensor, shape (128, 128), float32
        M: torch.Tensor, shape (128, 16), uint8, 每个元素存储8个bit位
    
    Returns:
        C: torch.Tensor, shape (128, 128), float32
    """
    assert A.dtype == torch.float32, "A must be float32"
    assert M.dtype == torch.uint8, "M must be uint8"
    
    C = torch.zeros_like(A).npu()
    M_cpu = M.cpu()
    
    for i in range(M_cpu.size(0)):
        for j in range(M_cpu.size(1)):
            byte_val = M_cpu[i, j]

            start_col = j * 8
            end_col = start_col + 8
            
            for bit_pos in range(8):
                col = start_col + bit_pos
                if (byte_val >> bit_pos) & 1:
                    C[i, col] = A[i, col]
                else:
                    C[i, col] = b
    
    return C

@tilelang.jit(out_idx=[-1])
def vec_select(M, N, block_M, block_N, mode, b_scalar, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            MASK: T.Tensor((M, N // 8), "uint8"),
            C: T.Tensor((M, N), dtype),
            
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            selmask_ub = T.alloc_ub((block_M // VEC_NUM, block_N // 8), "uint8")
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                T.copy(MASK[bx * block_M + vid * block_M // VEC_NUM, by * block_N // 8], selmask_ub)

                T.barrier_all()
                T.select(c_ub, selmask_ub, a_ub, b_scalar, mode)
                T.barrier_all()

                T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main

torch.npu.config.allow_internal_format = True
#torch.set_printoptions(threshold=np.inf)

b_scalar = 1.0
func = vec_select(M, N, 64, 128, "VSEL_TENSOR_SCALAR_MODE", b_scalar)

a = torch.zeros(M, N).npu()
m = torch.full((M, N // 8), 0xF, dtype=torch.uint8).npu()

torch.npu.synchronize()
print("init successful!")

c = func(a, m)

ref_c = select_by_mask_bits(a, b_scalar, m)
print("--------a--------")
print(a)
print("--------b--------")
print(b_scalar)
print("--------m--------")
print(m)
print("--------c--------")
print(c)
print("--------ref_c--------")
print(ref_c)

torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")


