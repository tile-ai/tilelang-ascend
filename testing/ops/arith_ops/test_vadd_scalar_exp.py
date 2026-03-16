import sys
import os
import argparse
import torch

import tilelang
import tilelang.language as T

import pytest
import testcommon as tc

pytestmark = [pytest.mark.mode("Expert")]
DATATYPE_CASES = ["float16", "float32"]

def vec_add(M, N, block_M, block_N, dtype):
    m_num = M // block_M
    n_num = N // block_N
    BLOCK_SIZE = 20

    @T.prim_func
    def vecAdd2dScalarInput(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            B_VEC = T.alloc_ub((block_N), dtype)
            C_VEC = T.alloc_ub((block_M, block_N), dtype)
            for i in T.serial(T.ceildiv(m_num*n_num, BLOCK_SIZE)):
                block_id = i * BLOCK_SIZE + cid
                if block_id < m_num * n_num:
                    block_id_m = block_id // n_num
                    block_id_n = block_id % n_num
                    bx = block_id_m * block_M
                    by = block_id_n * block_N
                    T.copy(A[bx, by], A_VEC)
                    T.copy(B[:block_N], B_VEC)
                    T.npuir_add(A_VEC, B_VEC[0], C_VEC)
                    T.copy(C_VEC, C[bx, by])

    return vecAdd2dScalarInput

def vec_add_2(M, N, block_M, block_N, dtype):
    m_num = M // block_M
    n_num = N // block_N
    BLOCK_SIZE = 20

    @T.prim_func
    def vecAdd2dScalarTensor(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            B_VEC = T.alloc_ub((block_M, block_N), dtype)
            C_VEC = T.alloc_ub((block_M, block_N), dtype)
            for i in T.serial(T.ceildiv(m_num*n_num, BLOCK_SIZE)):
                block_id = i * BLOCK_SIZE + cid
                if block_id < m_num * n_num:
                    block_id_m = block_id // n_num
                    block_id_n = block_id % n_num
                    bx = block_id_m * block_M
                    by = block_id_n * block_N
                    T.copy(A[bx, by], A_VEC)
                    T.copy(B[0, 0], B_VEC)
                    T.npuir_add(A_VEC, B_VEC[0, 0], C_VEC)
                    T.copy(C_VEC, C[bx, by])

    return vecAdd2dScalarTensor

def vec_add_3(M, N, block_M, dtype):
    m_num = M // block_M
    n_num = 1
    BLOCK_SIZE = 20

    @T.prim_func
    def vecAdd2dScalarTensorRev(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num*n_num, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((block_M, N), dtype)
            B_VEC = T.alloc_ub((block_M, N), dtype)
            C_VEC = T.alloc_ub((block_M, N), dtype)
            bx = cid * block_M
            T.copy(A[bx, 0], A_VEC)
            T.copy(B[bx, 0], B_VEC)
            j = 0 # to avoid 32byte alignment error
            for i in T.serial(block_M):
                T.npuir_add(A_VEC[i, j], B_VEC[i, j], C_VEC[i, j])
            T.copy(C_VEC[:, 0], C[bx, 0])

    return vecAdd2dScalarTensorRev

# case 1: VEC_A + VEC_B[0]
@pytest.mark.op("vadd_scalar_1_exp")
@pytest.mark.parametrize("dtype", DATATYPE_CASES)
def test_vadd_scalar_1(dtype):
    datatype = tc.resolve_dtype(dtype)
    M, N = 128, 256
    func = vec_add(M, N, 32, 32, dtype)
    compiled_kernel = tilelang.compile(func, target='npuir')
    a = torch.randn(M, N, dtype=datatype).npu()
    b = torch.randn(N, dtype=datatype).npu()
    c = torch.randn(M, N, dtype=datatype).npu()
    ref_output = a + b[0]
    compiled_kernel(a, b, c)
    tc.assert_close(c, ref_output, rtol=1e-3, atol=1e-3)

# case 2: VEC_A + VEC_B[0, 0]
@pytest.mark.op("vadd_scalar_2_exp")
@pytest.mark.parametrize("dtype", DATATYPE_CASES)
def test_vadd_scalar_2(dtype):
    datatype = tc.resolve_dtype(dtype)
    M, N = 128, 256
    func = vec_add_2(M, N, 32, 32, dtype)
    compiled_kernel = tilelang.compile(func, target='npuir')
    a = torch.randn(M, N, dtype=datatype).npu()
    b = torch.randn(M, N, dtype=datatype).npu()
    c = torch.randn(M, N, dtype=datatype).npu()
    ref_output = a + b[0, 0]
    compiled_kernel(a, b, c)
    tc.assert_close(c, ref_output, rtol=1e-3, atol=1e-3)

# case 3: VEC_A[i, j] + VEC_B[i, j]
@pytest.mark.op("vadd_scalar_3_exp")
@pytest.mark.parametrize("dtype", DATATYPE_CASES)
def test_vadd_scalar_3(dtype):
    datatype = tc.resolve_dtype(dtype)
    M, N = 128, 32
    func = vec_add_3(M, N, 32, dtype)
    compiled_kernel = tilelang.compile(func, target='npuir')
    a = torch.randn(M, N, dtype=datatype).npu()
    b = torch.randn(M, N, dtype=datatype).npu()
    c = torch.randn(M, N, dtype=datatype).npu()
    ref_output = c.clone()
    ref_output[:, 0:1] = a[:, 0:1] + b[:, 0:1]
    compiled_kernel(a, b, c)
    tc.assert_close(c, ref_output, rtol=1e-3, atol=1e-3)
