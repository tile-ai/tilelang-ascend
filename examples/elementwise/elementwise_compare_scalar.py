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

def compare_with_scalar_and_set_bits(A, b, C):
    """
    compare A's element to b, and set C's element according to comparison result 
    Args:
        A: torch.Tensor, shape (128, 128), float32
        b: float, scalar value
        C: torch.Tensor, shape (128, 16), uint8
    
    Returns:
        C: torch.Tensor, shape (128, 16), uint8
    """
    assert A.dtype == torch.float32, "A must be float32"
    assert C.dtype == torch.uint8, "C must be uint8"
    
    # set mask position to True or False(position set to True when A < b, else False)
    mask = A < b  # shape: (128, 128)
    
    C_result = torch.zeros(C.size(0), C.size(1), dtype=torch.uint8, device=A.device)
    
    for i in range(C.size(0)):
        for j in range(C.size(1)):
            start_bit = j * 8
            end_bit = start_bit + 8
            
            bits = mask[i, start_bit:end_bit]  # shape: (8,)
            
            byte_value = 0
            for k in range(8):
                if bits[k]:
                    byte_value |= (1 << k)
            
            C_result[i, j] = byte_value
    
    return C_result

@tilelang.jit(out_idx=[-1])
def vec_compare(M, N, block_M, block_N, mode, b_scalar, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N // 8), "uint8")
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N // 8), "uint8")
            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

                T.barrier_all()
                T.compare(c_ub, a_ub, b_scalar, mode)
                T.barrier_all()

                T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N // 8])

    return main

torch.npu.config.allow_internal_format = True
#torch.set_printoptions(threshold=np.inf)

b_scalar = 1.0
func = vec_compare(M, N, 128, 256, "LT", b_scalar)

a = torch.zeros(M, N).npu()


torch.npu.synchronize()
print("init successful!")

c = func(a)

ref_c = torch.zeros(M, N // 8, dtype=torch.uint8).npu()
ref_c = compare_with_scalar_and_set_bits(a, b_scalar, ref_c)

print("--------a--------")
print(a)
print("--------b--------")
print(b_scalar)
print("--------c--------")
print(c)
print("--------ref_c--------")
print(ref_c)

torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")


