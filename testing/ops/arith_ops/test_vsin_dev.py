# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import torch
import tilelang
import tilelang.language as T
import os

import pytest
import testcommon as tc

M = 16
N = 16
BLOCK_M = 16
BLOCK_N = 16
DTYPE_CASES = ["float16", "float32"]

def generate_tensor_new(shape, dtype, data_range):
    return torch.empty(shape, dtype = dtype).uniform_(data_range[0], data_range[1])

def vec_sin(M, N, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N
    BLOCK_SIZE = 8

    @T.prim_func
    def vecSinDev(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            A_VEC = T.alloc_shared((block_M, block_N), dtype)
            B_VEC = T.alloc_shared((block_M, block_N), dtype)

            for i in T.serial(T.ceildiv(m_num * n_num, BLOCK_SIZE)):
                block_id = i * BLOCK_SIZE + cid
                if block_id < m_num * n_num:
                    block_id_m = block_id // n_num
                    block_id_n = block_id % n_num
                    bx = block_id_m * block_M
                    by = block_id_n * block_N

                    T.copy(A[bx, by], A_VEC)
                    T.npuir_vsin(A_VEC, B_VEC)
                    T.copy(B_VEC, B[bx, by])
    return vecSinDev

@pytest.mark.op("vec_sin_dev")
@pytest.mark.mode("Developer")
@pytest.mark.parametrize("dtype", DTYPE_CASES)
def test_vec_sin(dtype):
    datatype = tc.resolve_dtype(dtype)
    func = vec_sin(M, N, BLOCK_M, BLOCK_N, dtype)
    compiled_kernel = tilelang.compile(func, target="npuir")

    A = generate_tensor_new(
        shape = (M, N),
        dtype = datatype,
        data_range = (-1.0, 1.0),
    ).npu()
    B = torch.zeros((M, N), dtype = datatype).npu()

    compiled_kernel(A, B)

    A_cpu = A.cpu()
    B_cpu = B.cpu()
    ref_cpu = torch.sin(A_cpu)
    
    tc.assert_close(B_cpu, ref_cpu, rtol=1e-3, atol=1e-3)

