import os
import random

import pytest

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T
import tilelang.language.reduce_ascend as reduce_ascend_lang

tir = tilelang.tvm.tir

"""
This is an element-wise pytest automation test suite.
All test cases are written and executed at the Developer Level.
"""

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    """Clear tilelang cache before tests"""
    tilelang.cache.clear_cache()
    yield


@pytest.fixture
def setup_random_seed():
    """Set random seed for reproducibility"""
    torch.manual_seed(0)
    yield


def assert_close_npu(actual, expected, dtype, rtol=1e-2, atol=1e-2, **kwargs):
    """Helper function to handle uint16/uint32 dtype for torch.testing.assert_close

    torch-npu doesn't support isclose operations for uint16/uint32 dtype,
    so we convert to int16/int32 for comparison when needed.
    """
    if dtype == "uint16":
        torch.testing.assert_close(actual.to(torch.int16), expected.to(torch.int16), rtol=rtol, atol=atol, **kwargs)
    elif dtype == "uint32":
        torch.testing.assert_close(actual.to(torch.int32), expected.to(torch.int32), rtol=rtol, atol=atol, **kwargs)
    else:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, **kwargs)


def _get_reduce_fn(op: str):
    reduce_fns = {
        "sum": T.reduce_sum,
        "max": T.reduce_max,
        "min": T.reduce_min,
    }
    return reduce_fns[op]


def vec_abs(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

            T.tile.abs(b_ub, a_ub)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_abs(M, N, block_M, block_N, dtype, target):
    func = vec_abs(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N, dtype=torch.float32 if dtype == "float" else torch.float16).npu()

    torch.npu.synchronize()
    b = func(a)

    ref_b = torch.abs(a)
    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["float", "float16"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_abs(dtype, target, shape):
    M, N = shape
    run_test_abs(M, N, 128, 256, dtype, target=target)


def vec_add_auto_copy(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

            for i, j in T.Parallel(block_M // VEC_NUM, block_N):
                C[bx * block_M + vid * block_M // VEC_NUM + i, by * block_N + j] = a_ub[i, j] + b_ub[i, j]

    return main


def run_test_add_auto_copy(M, N, block_M, block_N, dtype, target):
    func = vec_add_auto_copy(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    dtype_map = {
        "float": torch.float32,
        "float16": torch.float16,
        "int16": torch.int16,
        "int32": torch.int32,
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)

    if dtype in ["int16", "int32"]:
        a = torch.randint(0, 100, (M, N), dtype=torch_dtype).npu()
        b = torch.randint(0, 100, (M, N), dtype=torch_dtype).npu()
    else:
        a = torch.randn(M, N, dtype=torch_dtype).npu()
        b = torch.randn(M, N, dtype=torch_dtype).npu()

    torch.npu.synchronize()

    c = func(a, b)
    ref_c = a + b
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


add_dtype_target_params = [
    ("float", "ascendc"),
    ("float16", "ascendc"),
    ("float", "pto"),
    ("float16", "pto"),
    ("int16", "pto"),
    ("int32", "pto"),
]


@pytest.mark.parametrize("dtype,target", add_dtype_target_params)
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_add_auto_copy(dtype, target, shape):
    M, N = shape
    run_test_add_auto_copy(M, N, 128, 128, dtype, target=target)


def vec_add_developer(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

            for i, j in T.Parallel(block_M // VEC_NUM, block_N):
                c_ub[i, j] = a_ub[i, j] + b_ub[i, j]

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_add_developer(M, N, block_M, block_N, dtype, target):
    func = vec_add_developer(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    dtype_map = {
        "float": torch.float32,
        "float16": torch.float16,
        "int16": torch.int16,
        "int32": torch.int32,
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)

    if dtype in ["int16", "int32"]:
        a = torch.randint(0, 100, (M, N), dtype=torch_dtype).npu()
        b = torch.randint(0, 100, (M, N), dtype=torch_dtype).npu()
    else:
        a = torch.randn(M, N, dtype=torch_dtype).npu()
        b = torch.randn(M, N, dtype=torch_dtype).npu()
    torch.npu.synchronize()

    c = func(a, b)
    ref_c = a + b
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype,target", add_dtype_target_params)
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_add_developer(dtype, target, shape):
    M, N = shape
    run_test_add_developer(M, N, 128, 128, dtype, target=target)


def adds(M, N, block_M, block_N, scalar, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M, block_N), dtype)
            b_ub = T.alloc_ub((block_M, block_N), dtype)
            T.copy(A[bx * block_M, by * block_N], a_ub)
            T.tile.add(b_ub, a_ub, scalar)
            T.copy(b_ub, B[bx * block_M, by * block_N])

    return main


def run_test_adds(M, N, block_M, block_N, scalar, dtype, target):
    func = adds(M, N, block_M, block_N, scalar, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    dtype_map = {
        "float": torch.float32,
        "float16": torch.float16,
        "int16": torch.int16,
        "int32": torch.int32,
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)

    if dtype in ["int16", "int32"]:
        a = torch.randint(0, 100, (M, N), dtype=torch_dtype).npu()
        ref_b = a + int(scalar)
    else:
        a = torch.randn(M, N, dtype=torch_dtype).npu()
        ref_b = a + scalar

    b = func(a)
    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["float", "float16", "int16", "int32"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_adds(dtype, target, shape):
    M, N = shape
    scalar = 2.0 if dtype in ["float", "float16"] else 2
    run_test_adds(M, N, 64, 32, scalar, dtype, target=target)


def bitwise_and(M, N, block_M, block_N, dtype="int16"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)
            T.tile.bitwise_and(c_ub, a_ub, b_ub)
            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_bitwise_and(M, N, block_M, block_N, dtype, target):
    func = bitwise_and(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    torch_dtype = torch.int16 if dtype == "int16" else torch.uint16
    a = torch.randint(0, 10, (M, N), dtype=torch_dtype).npu()
    b = torch.randint(0, 10, (M, N), dtype=torch_dtype).npu()

    torch.npu.synchronize()

    c = func(a, b)
    ref_c = a & b
    assert_close_npu(c, ref_c, dtype, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["int16", "uint16"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_bitwise_and(dtype, target, shape):
    M, N = shape
    run_test_bitwise_and(M, N, 128, 256, dtype, target=target)


def axpy(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            T.tile.fill(b_ub, 0)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

            T.tile.axpy(b_ub, a_ub, 2.0)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_axpy(M, N, block_M, block_N, dtype, target):
    func = axpy(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N, dtype=torch.float32 if dtype == "float" else torch.float16).npu()

    torch.npu.synchronize()

    b = func(a)

    ref_b = a * 2.0
    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["float", "float16"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_axpy(dtype, target, shape):
    M, N = shape
    run_test_axpy(M, N, 128, 256, dtype, target)


def axpy_slice(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            T.tile.fill(b_ub, 0)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            for i in range(block_M // VEC_NUM):
                T.tile.axpy(b_ub[i, :], a_ub[i, :], 2.0)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_axpy_slice(M, N, block_M, block_N, target):
    func = axpy_slice(M, N, block_M, block_N)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N).npu()

    torch.npu.synchronize()

    b = func(a)

    ref_b = a * 2.0
    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_axpy_slice(target, shape):
    M, N = shape
    run_test_axpy_slice(M, N, 128, 256, target)


def bilinear_interpolation(mask, h_repeat, repeat_mode, dst_blk_stride, v_r_offset, v_repeat, src0, src0offset_int, src0offset, src1):
    m_num = 1
    n_num = 1

    VEC_NUM = 1

    @T.prim_func
    def main(
        src0: T.Tensor((src0.shape[0], src0.shape[1]), "float16"),  # type: ignore
        src0_offset: T.Tensor((src0offset.shape[0], src0offset.shape[1]), "uint32"),  # type: ignore
        src1: T.Tensor((src1.shape[0], src1.shape[1]), "float16"),  # type: ignore
        dst: T.Tensor((src0.shape[0], src0.shape[1] // 2), "float16"),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            src0_ub = T.alloc_ub((src0.shape[0] // VEC_NUM, src0.shape[1]), "float16")
            src0_offset_ub = T.alloc_ub((src0offset.shape[0] // VEC_NUM, src0offset.shape[1]), "uint32")
            src1_ub = T.alloc_ub((src1.shape[0] // VEC_NUM, src1.shape[1]), "float16")
            dst_ub = T.alloc_ub((src0.shape[0] // VEC_NUM, src0.shape[1] // 2), "float16")

            T.copy(src0[0, 0], src0_ub)
            T.copy(src0_offset[0, 0], src0_offset_ub)
            T.copy(src1[0, 0], src1_ub)

            T.tile.bilinear_interpolation(
                dst_ub,
                src0_ub,
                src0_offset_ub,
                src1_ub,
                mask,
                h_repeat,
                repeat_mode,
                dst_blk_stride,
                v_r_offset,
                v_repeat,
            )

            T.copy(dst_ub, dst[0, 0])

    return main


def fun_ref(a, b, c, hRepeat, vRepeat, repeatMode, vROffset):
    a = a.flatten()
    b = b.flatten()
    c = c.flatten()
    re = []

    if repeatMode:
        for k in range(vRepeat):
            s = torch.zeros(128, dtype=torch.float16).npu()  # 初始化累加器
            r = torch.zeros(128 * hRepeat, dtype=torch.float16).npu()
            for i in range(hRepeat):
                for j in range(8):
                    idx = b[k * 8 * hRepeat + i * 8 + j].to(torch.int64) // 32
                    r[i * 128 + j * 16 : i * 128 + (j + 1) * 16] = a[idx * 16 : (idx + 1) * 16] * c[k * 8 * hRepeat + i * 8 + j]
                s += r[i * 128 : (i + 1) * 128]
            re.append(s)
    else:
        for k in range(vRepeat):
            s = torch.zeros(128, dtype=torch.float16).npu()
            r = torch.zeros(128 * hRepeat, dtype=torch.float16).npu()
            for i in range(hRepeat):
                for j in range(8):
                    idx = b[k * 8 * hRepeat + i * 8 + j].to(torch.int64) // 32
                    r[i * 128 + j * 16 : i * 128 + (j + 1) * 16] = a[idx * 16 : (idx + 1) * 16] * c[k * hRepeat + i]
                s += r[i * 128 : (i + 1) * 128]
            re.append(s)
    return torch.cat(re, dim=0).flatten()


def run_test_bilinear_interpolation(target):
    src0 = torch.arange(1, 513, dtype=torch.float16).reshape(1, -1).npu()
    src0offset_int = torch.arange(0, 1024, 32, dtype=torch.int64).reshape(1, -1).npu()
    src0offset = src0offset_int.to(dtype=torch.uint32)
    src1 = torch.arange(2, 18, dtype=torch.float16).reshape(1, -1).npu()
    hRepeat = 2
    mask1 = 128
    repeatMode = False
    dstBlkStride = 1
    vROffset = 128
    vRepeat = 2
    mask0 = 0
    func = bilinear_interpolation(mask1, hRepeat, repeatMode, dstBlkStride, vROffset, vRepeat, src0, src0offset_int, src0offset, src1)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    torch.npu.synchronize()

    c = func(src0, src0offset, src1)

    if vROffset > 128:
        outsize = vRepeat * vROffset
    else:
        outsize = vRepeat * 128

    out = fun_ref(src0, src0offset, src1, hRepeat, vRepeat, repeatMode, vROffset)

    out_real = torch.zeros(outsize, dtype=torch.float16).npu()
    if mask0 == 0:
        for i in range(vRepeat):
            n = mask1 // 16
            l = mask1 % 16
            for j in range(n):
                out_real[i * vROffset + j * 16 : i * vROffset + (j + 1) * 16] = out[i * 128 + j * 16 : i * 128 + (j + 1) * 16]
            out_real[i * vROffset + n * 16 : i * vROffset + n * 16 + l] = out[i * 128 + n * 16 : i * 128 + n * 16 + l]

    ref_c = out_real[: vRepeat * 128].unsqueeze(0)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc"])
def test_bilinear_interpolation(target):
    run_test_bilinear_interpolation(target)


def bitwise_lshift(M, N, block_M, block_N, scalarvalue, dtype="int32"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.tile.bitwise_lshift(b_ub, a_ub, scalarvalue)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_bitwise_lshift(M, N, block_M, block_N, scalarvalue, dtype, target):
    func = bitwise_lshift(M, N, block_M, block_N, scalarvalue, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    torch_dtype_map = {
        "int16": torch.int16,
        "int32": torch.int32,
        "uint16": torch.uint16,
        "uint32": torch.uint32,
    }
    torch_dtype = torch_dtype_map[dtype]
    a = torch.randint(low=1, high=101, size=(M, N), dtype=torch_dtype).npu()

    torch.npu.synchronize()

    b = func(a)

    # Compute reference on CPU to avoid NPU dtype limitations
    a_cpu = a.cpu()
    ref_b = (pow(2, scalarvalue) * a_cpu).npu()

    assert_close_npu(b, ref_b, dtype, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["int16", "int32", "uint16", "uint32"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_bitwise_lshift(dtype, target, shape):
    M, N = shape
    max_shift = 16 if dtype in ["int16", "uint16"] else 32
    scalarvalue = random.randint(1, max_shift)
    run_test_bitwise_lshift(M, N, 128, 256, scalarvalue=scalarvalue, dtype=dtype, target=target)


def bitwise_lshift_slice(M, N, block_M, block_N, scalarvalue, dtype="int32"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            for i in range(block_M // VEC_NUM):
                T.tile.bitwise_lshift(b_ub[i, :], a_ub[i, :], scalarvalue)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_bitwise_lshift_slice(M, N, block_M, block_N, scalarvalue, dtype, target):
    func = bitwise_lshift_slice(M, N, block_M, block_N, scalarvalue, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    torch_dtype_map = {
        "int16": torch.int16,
        "int32": torch.int32,
        "uint16": torch.uint16,
        "uint32": torch.uint32,
    }
    torch_dtype = torch_dtype_map[dtype]
    a = torch.randint(low=1, high=101, size=(M, N), dtype=torch_dtype).npu()

    torch.npu.synchronize()

    b = func(a)

    # Compute reference on CPU to avoid NPU dtype limitations
    a_cpu = a.cpu()
    ref_b = (pow(2, scalarvalue) * a_cpu).npu()

    assert_close_npu(b, ref_b, dtype, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["int16", "int32", "uint16", "uint32"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_bitwise_lshift_slice(dtype, target, shape):
    M, N = shape
    max_shift = 16 if dtype in ["int16", "uint16"] else 32
    scalarvalue = random.randint(1, max_shift)
    run_test_bitwise_lshift_slice(M, N, 128, 256, scalarvalue=scalarvalue, dtype=dtype, target=target)


def bitwise_not(M, N, block_M, block_N, dtype="int16"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.tile.bitwise_not(b_ub, a_ub)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_bitwise_not(M, N, block_M, block_N, dtype, target):
    func = bitwise_not(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    torch_dtype = torch.int16 if dtype == "int16" else torch.uint16
    a = torch.randint(0, 10, (M, N), dtype=torch_dtype).npu()

    torch.npu.synchronize()

    b = func(a)

    a_cpu = a.cpu()
    if dtype == "uint16":
        ref_b = (~a_cpu.to(torch.int32)).to(torch.uint16).npu()
    else:
        ref_b = (~a_cpu).npu()

    assert_close_npu(b, ref_b, dtype, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["int16", "uint16"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_bitwise_not(dtype, target, shape):
    M, N = shape
    run_test_bitwise_not(M, N, 128, 256, dtype, target=target)


def bitwise_rshift(M, N, block_M, block_N, scalarvalue, dtype="int32"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

            T.tile.bitwise_rshift(b_ub, a_ub, scalarvalue)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_bitwise_rshift(M, N, block_M, block_N, scalarvalue, dtype, target):
    func = bitwise_rshift(M, N, block_M, block_N, scalarvalue, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    torch_dtype_map = {
        "int16": torch.int16,
        "int32": torch.int32,
        "uint16": torch.uint16,
        "uint32": torch.uint32,
    }
    torch_dtype = torch_dtype_map[dtype]
    a = torch.randint(low=1, high=101, size=(M, N), dtype=torch_dtype).npu()

    torch.npu.synchronize()

    b = func(a)

    a_cpu = a.cpu()
    if dtype == "uint16":
        ref_b = (a_cpu.to(torch.int32) >> scalarvalue).to(torch.uint16).npu()
    elif dtype == "uint32":
        ref_b = (a_cpu.to(torch.int64) >> scalarvalue).to(torch.uint32).npu()
    else:
        ref_b = (a_cpu >> scalarvalue).npu()

    assert_close_npu(b, ref_b, dtype, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["int16", "int32", "uint16", "uint32"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_bitwise_rshift(dtype, target, shape):
    M, N = shape
    max_shift = 16 if dtype in ["int16", "uint16"] else 32
    scalarvalue = random.randint(1, max_shift)
    run_test_bitwise_rshift(M, N, 128, 256, scalarvalue=scalarvalue, dtype=dtype, target=target)


def bitwise_rshift_slice(M, N, block_M, block_N, scalarvalue, dtype="int32"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            for i in range(block_M // VEC_NUM):
                T.tile.bitwise_rshift(b_ub[i, :], a_ub[i, :], scalarvalue)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_bitwise_rshift_slice(M, N, block_M, block_N, scalarvalue, dtype, target):
    func = bitwise_rshift_slice(M, N, block_M, block_N, scalarvalue, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    torch_dtype_map = {
        "int16": torch.int16,
        "int32": torch.int32,
        "uint16": torch.uint16,
        "uint32": torch.uint32,
    }
    torch_dtype = torch_dtype_map[dtype]
    a = torch.randint(low=1, high=101, size=(M, N), dtype=torch_dtype).npu()

    torch.npu.synchronize()

    b = func(a)

    a_cpu = a.cpu()
    if dtype == "uint16":
        ref_b = (a_cpu.to(torch.int32) >> scalarvalue).to(torch.uint16).npu()
    elif dtype == "uint32":
        ref_b = (a_cpu.to(torch.int64) >> scalarvalue).to(torch.uint32).npu()
    else:
        ref_b = (a_cpu >> scalarvalue).npu()

    assert_close_npu(b, ref_b, dtype, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["int16", "int32", "uint16", "uint32"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_bitwise_rshift_slice(dtype, target, shape):
    M, N = shape
    max_shift = 16 if dtype in ["int16", "uint16"] else 32
    scalarvalue = random.randint(1, max_shift)
    run_test_bitwise_rshift_slice(M, N, 128, 256, scalarvalue=scalarvalue, dtype=dtype, target=target)


def bitwise_xor(M, N, block_M, block_N, dtype="int16"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

            T.tile.bitwise_xor(c_ub, a_ub, b_ub)

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_bitwise_xor(M, N, block_M, block_N, dtype, target):
    func = bitwise_xor(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    torch_dtype = torch.int16 if dtype == "int16" else torch.uint16
    a = torch.randint(0, 10, (M, N), dtype=torch_dtype).npu()
    b = torch.randint(0, 10, (M, N), dtype=torch_dtype).npu()

    torch.npu.synchronize()

    c = func(a, b)

    a_cpu = a.cpu()
    b_cpu = b.cpu()
    if dtype == "uint16":
        ref_c = torch.bitwise_xor(a_cpu.to(torch.int32), b_cpu.to(torch.int32)).to(torch.uint16).npu()
    else:
        ref_c = torch.bitwise_xor(a_cpu, b_cpu).npu()
    assert_close_npu(c, ref_c, dtype, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["int16", "uint16"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_bitwise_xor(dtype, target, shape):
    M, N = shape
    run_test_bitwise_xor(M, N, 128, 256, dtype, target)


def bitwise_xor_slice(M, N, block_M, block_N, dtype="int16"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)
            for i in range(block_M // VEC_NUM):
                T.tile.bitwise_xor(c_ub[i, :], a_ub[i, :], b_ub[i, :])

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_bitwise_xor_slice(M, N, block_M, block_N, dtype, target):
    func = bitwise_xor_slice(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    torch_dtype = torch.int16 if dtype == "int16" else torch.uint16
    a = torch.randint(0, 10, (M, N), dtype=torch_dtype).npu()
    b = torch.randint(0, 10, (M, N), dtype=torch_dtype).npu()

    torch.npu.synchronize()

    c = func(a, b)

    a_cpu = a.cpu()
    b_cpu = b.cpu()
    if dtype == "uint16":
        ref_c = torch.bitwise_xor(a_cpu.to(torch.int32), b_cpu.to(torch.int32)).to(torch.uint16).npu()
    else:
        ref_c = torch.bitwise_xor(a_cpu, b_cpu).npu()
    assert_close_npu(c, ref_c, dtype, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["int16", "uint16"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_bitwise_xor_slice(dtype, target, shape):
    M, N = shape
    run_test_bitwise_xor_slice(M, N, 128, 256, dtype, target)


def block_reduce_max(M, N, block_M, block_N, repeat, mask, dstRepStride, srcBlkStride, srcRepStride, dataBlockNum, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N // dataBlockNum), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N // dataBlockNum), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.tile.block_reduce_max(b_ub, a_ub, repeat, mask, dstRepStride, srcBlkStride, srcRepStride)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N // dataBlockNum])

    return main


def run_test_block_reduce_max(M, N, block_M, block_N, repeat, mask, dstRepStride, srcBlkStride, srcRepStride, dataBlockNum, dtype, target):
    func = block_reduce_max(M, N, block_M, block_N, repeat, mask, dstRepStride, srcBlkStride, srcRepStride, dataBlockNum, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    torch_dtype = torch.float32 if dtype == "float" else torch.float16
    a = torch.randn(M, N, dtype=torch_dtype).npu()

    torch.npu.synchronize()

    b = func(a)

    num_groups = M * N // dataBlockNum
    ref_b = torch.zeros((1, num_groups)).to(torch_dtype)
    a_flag = a.reshape(-1)
    for i in range(num_groups):
        start = i * dataBlockNum
        end = start + dataBlockNum
        group = a_flag[start:end]
        max_val = torch.max(group).item()
        ref_b[0, i] = max_val
    ref_b = ref_b.reshape(M, N // dataBlockNum)
    ref_b = ref_b.npu().to(dtype=torch_dtype)
    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["float16"])
@pytest.mark.parametrize("target", ["ascendc"])
def test_block_reduce_max(dtype, target):
    M = 2
    N = 512
    block_M = 2
    block_N = 128
    dataBlockNum = 16
    mask = 128
    repeat = 1
    dstRepStride = 1
    srcBlkStride = 1
    srcRepStride = 8
    run_test_block_reduce_max(M, N, block_M, block_N, repeat, mask, dstRepStride, srcBlkStride, srcRepStride, dataBlockNum, dtype, target)


def block_reduce_min(M, N, block_M, block_N, repeat, mask, dstRepStride, srcBlkStride, srcRepStride, dataBlockNum, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N // dataBlockNum), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N // dataBlockNum), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

            T.tile.block_reduce_min(b_ub, a_ub, repeat, mask, dstRepStride, srcBlkStride, srcRepStride)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N // dataBlockNum])

    return main


def run_test_block_reduce_min(M, N, block_M, block_N, repeat, mask, dstRepStride, srcBlkStride, srcRepStride, dataBlockNum, dtype, target):
    func = block_reduce_min(M, N, block_M, block_N, repeat, mask, dstRepStride, srcBlkStride, srcRepStride, dataBlockNum, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    torch_dtype = torch.float32 if dtype == "float" else torch.float16
    a = torch.randn(M, N, dtype=torch_dtype).npu()

    torch.npu.synchronize()

    b = func(a)

    num_groups = M * N // dataBlockNum
    ref_b = torch.zeros((1, num_groups)).to(torch_dtype)
    a_flag = a.reshape(-1)
    for i in range(num_groups):
        start = i * dataBlockNum
        end = start + dataBlockNum
        group = a_flag[start:end]
        min_val = torch.min(group).item()
        ref_b[0, i] = min_val
    ref_b = ref_b.reshape(M, N // dataBlockNum)
    ref_b = ref_b.npu().to(dtype=torch_dtype)

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["float16"])
@pytest.mark.parametrize("target", ["ascendc"])
def test_block_reduce_min(dtype, target):
    M = 2
    N = 512
    block_M = 2
    block_N = 128
    dataBlockNum = 16
    mask = 128
    repeat = 1
    dstRepStride = 1
    srcBlkStride = 1
    srcRepStride = 8
    run_test_block_reduce_min(M, N, block_M, block_N, repeat, mask, dstRepStride, srcBlkStride, srcRepStride, dataBlockNum, dtype, target)


def block_reduce_sum(M, N, block_M, block_N, repeat, mask, dstRepStride, srcBlkStride, srcRepStride, dataBlockNum, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N // dataBlockNum), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N // dataBlockNum), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

            T.tile.block_reduce_sum(b_ub, a_ub, repeat, mask, dstRepStride, srcBlkStride, srcRepStride)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N // dataBlockNum])

    return main


def run_test_block_reduce_sum(M, N, block_M, block_N, repeat, mask, dstRepStride, srcBlkStride, srcRepStride, dataBlockNum, dtype, target):
    func = block_reduce_sum(M, N, block_M, block_N, repeat, mask, dstRepStride, srcBlkStride, srcRepStride, dataBlockNum, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    torch_dtype = torch.float32 if dtype == "float" else torch.float16
    a = torch.randn(M, N, dtype=torch_dtype).npu()

    torch.npu.synchronize()

    b = func(a)

    num_groups = M * N // dataBlockNum
    ref_b = torch.zeros((1, num_groups)).to(torch_dtype)
    a_flag = a.reshape(-1)
    for i in range(num_groups):
        start = i * dataBlockNum
        end = start + dataBlockNum
        group = a_flag[start:end]
        sum_val = torch.sum(group).item()
        ref_b[0, i] = sum_val
    ref_b = ref_b.reshape(M, N // dataBlockNum)
    ref_b = ref_b.npu().to(dtype=torch_dtype)

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["float16"])
@pytest.mark.parametrize("target", ["ascendc"])
def test_block_reduce_sum(dtype, target):
    M = 2
    N = 512
    block_M = 2
    block_N = 128
    dataBlockNum = 16
    mask = 128
    repeat = 1
    dstRepStride = 1
    srcBlkStride = 1
    srcRepStride = 8
    run_test_block_reduce_sum(M, N, block_M, block_N, repeat, mask, dstRepStride, srcBlkStride, srcRepStride, dataBlockNum, dtype, target)


def cast(M, N, block_M, block_N, mode, count):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 1

    @T.prim_func
    def main(
        A: T.Tensor((M, N), "float"),  # type: ignore
        B: T.Tensor((M, N), "float"),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), "float")
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), "float")

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

            T.tile.cast(b_ub, a_ub, mode, count)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_cast(M, N, block_M, block_N, mode, count, target):
    func = cast(M, N, block_M, block_N, mode, count)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    # without setdeqscale
    a = torch.full((M, N), 0.5, dtype=torch.float).npu()

    torch.npu.synchronize()

    b = func(a)

    ref_b = torch.full((M, N), 0.0, dtype=torch.float).npu()

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(64, 64)])
def test_cast(dtype, target, shape):
    M, N = shape
    run_test_cast(M, N, 16, 16, "CAST_RINT", 4096, target)


def cast_slice(M, N, block_M, block_N, mode, count):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 1

    @T.prim_func
    def main(
        A: T.Tensor((M, N), "float"),  # type: ignore
        B: T.Tensor((M, N), "float"),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), "float")
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), "float")

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            for i in range(block_M // VEC_NUM):
                T.tile.cast(b_ub[i, :], a_ub[i, :], mode, count)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_cast_slice(M, N, block_M, block_N, mode, count, target):
    func = cast_slice(M, N, block_M, block_N, mode, count)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    # without setdeqscale
    a = torch.full((M, N), 0.5, dtype=torch.float).npu()

    torch.npu.synchronize()

    b = func(a)

    ref_b = torch.full((M, N), 0.0, dtype=torch.float).npu()

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(64, 64)])
def test_cast_slice(dtype, target, shape):
    M, N = shape
    run_test_cast_slice(M, N, 16, 16, "CAST_RINT", 4096, target)


def cast_scale(M, N, block_M, block_N, mode, count, scale):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 1

    @T.prim_func
    def main(
        A: T.Tensor((M, N), "int32"),  # type: ignore
        B: T.Tensor((M, N), "float16"),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), "int32")
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), "float16")

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

            T.set_deq_scale(scale)

            T.tile.cast(b_ub, a_ub, mode, count)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_cast_scale(M, N, block_M, block_N, mode, count, scale, target):
    func = cast_scale(M, N, block_M, block_N, mode, count, scale)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    # should setdeqscale
    a = torch.full((M, N), 1, dtype=torch.int32).npu()

    torch.npu.synchronize()

    b = func(a)

    ref_b = torch.full((M, N), 1.0, dtype=torch.float16).npu()

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["int32", "float16"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(64, 64)])
def test_cast_scale(dtype, target, shape):
    M, N = shape
    run_test_cast_scale(M, N, 16, 16, "CAST_RINT", 4096, 1.0, target)


def clamp(M, N, block_M, block_N, max_val, min_val, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N
    num_blocks = m_num * n_num

    VEC_NUM = 2

    @T.prim_func
    def main(
        input: T.Tensor((M, N), dtype),  # type: ignore
        output: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            block_size = block_M * block_N // VEC_NUM
            in_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            out_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(input[bx * block_M + vid * block_M // VEC_NUM, by * block_N], in_ub)

            T.tile.clamp(out_ub, in_ub, min_val, max_val, block_size)

            T.copy(out_ub, output[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_clamp(M, N, max_val, min_val, thresh, dtype, target, block_M=64, block_N=64):
    if min_val > max_val:
        max_val, min_val = min_val, max_val

    func = clamp(M, N, block_M, block_N, max_val, min_val, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = (torch.rand(M, N) - 0.5) * 2 * thresh
    a = a.half().npu() if dtype == "float16" else a.float().npu()

    b = func(a)
    ref_b = torch.clamp(a, min_val, max_val)

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["float", "float16"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_clamp(dtype, target):
    M, N = 1024, 1024
    thresh = 10000
    max_val = random.uniform(-thresh, thresh)
    min_val = random.uniform(-thresh, thresh)
    run_test_clamp(M, N, max_val, min_val, thresh, dtype, target, block_M=64, block_N=64)


def clamp_slice(M, N, block_M, block_N, max_val, min_val, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N
    num_blocks = m_num * n_num

    VEC_NUM = 2

    @T.prim_func
    def main(
        input: T.Tensor((M, N), dtype),  # type: ignore
        output: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            block_size = block_M * block_N // VEC_NUM
            in_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            out_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            T.copy(input[bx * block_M + vid * block_M // VEC_NUM, by * block_N], in_ub)
            for i in range(block_M // VEC_NUM):
                T.tile.clamp(out_ub[i, :], in_ub[i, :], min_val, max_val, block_size)

            T.copy(out_ub, output[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_clamp_slice(M, N, max_val, min_val, thresh, dtype, target, block_M=64, block_N=64):
    if min_val > max_val:
        max_val, min_val = min_val, max_val

    func = clamp_slice(M, N, block_M, block_N, max_val, min_val, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = (torch.rand(M, N) - 0.5) * 2 * thresh
    a = a.half().npu() if dtype == "float16" else a.float().npu()

    b = func(a)
    ref_b = torch.clamp(a, min_val, max_val)

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["float", "float16"])
@pytest.mark.parametrize("target", ["ascendc"])
def test_clamp_slice(dtype, target):
    M, N = 1024, 1024
    thresh = 10000
    max_val = random.uniform(-thresh, thresh)
    min_val = random.uniform(-thresh, thresh)
    run_test_clamp_slice(M, N, max_val, min_val, thresh, dtype, target, block_M=64, block_N=64)


def compare(M, N, block_M, block_N, mode, dtype="float", out_dtype="uint8"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
        C: T.Tensor((M, N // 8), out_dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N // 8), out_dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

            T.tile.compare(c_ub, a_ub, b_ub, mode)

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N // 8])

    return main


def compare_and_set_bits(A, B, C, out_dtype="uint8"):
    """
    compare A to B, and set C's element according to comparison result
    Args:
        A: torch.Tensor, shape (128, 128), float32, float16 or int32
        B: torch.Tensor, shape (128, 128), float32, float16 or int32
        C: torch.Tensor, shape (128, 16), int8 or uint8
        out_dtype: str, output dtype ("int8" or "uint8")

    Returns:
        C: torch.Tensor, shape (128, 16), int8 or uint8
    """
    assert A.dtype in [torch.float32, torch.float16, torch.int32], "A must be float32, float16 or int32"
    assert B.dtype in [torch.float32, torch.float16, torch.int32], "B must be float32, float16 or int32"
    assert out_dtype in ["int8", "uint8"], "out_dtype must be int8 or uint8"

    mask = A < B

    torch_out_dtype = torch.int8 if out_dtype == "int8" else torch.uint8
    C_result = torch.zeros(C.size(0), C.size(1), dtype=torch_out_dtype, device=A.device)

    for i in range(C.size(0)):
        for j in range(C.size(1)):
            start_bit = j * 8
            end_bit = start_bit + 8

            bits = mask[i, start_bit:end_bit]

            byte_value = 0
            for k in range(8):
                if bits[k]:
                    byte_value |= 1 << k

            C_result[i, j] = byte_value

    return C_result


def run_test_compare(M, N, block_M, block_N, mode, dtype, out_dtype, target):
    torch.npu.config.allow_internal_format = True

    func = compare(M, N, block_M, block_N, mode, dtype, out_dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    dtype_map = {
        "float": torch.float32,
        "float16": torch.float16,
        "int32": torch.int32,
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)

    if dtype == "int32":
        a = torch.zeros(M, N, dtype=torch_dtype).npu()
        b = torch.ones(M, N, dtype=torch_dtype).npu()
    else:
        a = torch.zeros(M, N, dtype=torch_dtype).npu()
        b = torch.ones(M, N, dtype=torch_dtype).npu()

    torch.npu.synchronize()

    c = func(a, b)

    torch_out_dtype = torch.int8 if out_dtype == "int8" else torch.uint8
    ref_c = torch.zeros(M, N // 8, dtype=torch_out_dtype).npu()
    ref_c = compare_and_set_bits(a, b, ref_c, out_dtype)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


compare_dtype_target_params = [
    ("float", "ascendc"),
    ("float16", "ascendc"),
    ("float", "pto"),
    ("float16", "pto"),
    # ("int32", "pto"),
]


@pytest.mark.parametrize("out_dtype", ["int8", "uint8"])
@pytest.mark.parametrize("dtype,target", compare_dtype_target_params)
@pytest.mark.parametrize("shape", [(256, 256)])
def test_compare(out_dtype, dtype, target, shape):
    M, N = shape
    run_test_compare(M, N, 128, 256, "LT", dtype, out_dtype, target)


def compare_slice(M, N, block_M, block_N, mode, dtype="float", out_dtype="uint8"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
        C: T.Tensor((M, N // 8), out_dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N // 8), out_dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)
            for i in range(block_M // VEC_NUM):
                T.tile.compare(c_ub[i, :], a_ub[i, :], b_ub[i, :], mode)

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N // 8])

    return main


def run_test_compare_slice(M, N, block_M, block_N, mode, dtype, out_dtype, target):
    torch.npu.config.allow_internal_format = True

    func = compare_slice(M, N, block_M, block_N, mode, dtype, out_dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    torch_dtype = torch.float32 if dtype == "float" else torch.float16
    a = torch.zeros(M, N, dtype=torch_dtype).npu()
    b = torch.ones(M, N, dtype=torch_dtype).npu()

    torch.npu.synchronize()

    c = func(a, b)

    torch_out_dtype = torch.int8 if out_dtype == "int8" else torch.uint8
    ref_c = torch.zeros(M, N // 8, dtype=torch_out_dtype).npu()
    ref_c = compare_and_set_bits(a, b, ref_c, out_dtype)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("out_dtype", ["int8", "uint8"])
@pytest.mark.parametrize("dtype", ["float", "float16"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(256, 256)])
def test_compare_slice(out_dtype, dtype, target, shape):
    M, N = shape
    run_test_compare_slice(M, N, 128, 256, "LT", dtype, out_dtype, target)


def compare_scalar(M, N, block_M, block_N, mode, b_scalar, dtype="float", out_dtype="uint8"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), C: T.Tensor((M, N // 8), out_dtype)):  # type: ignore
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N // 8), out_dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

            T.tile.compare(c_ub, a_ub, b_scalar, mode)

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N // 8])

    return main


def compare_with_scalar_and_set_bits(A, b, C, out_dtype="uint8"):
    """
    compare A's element to b, and set C's element according to comparison result
    Args:
        A: torch.Tensor, shape (128, 128), float32 or float16
        b: scalar value
        C: torch.Tensor, shape (128, 16), int8 or uint8
        out_dtype: str, output dtype ("int8" or "uint8")

    Returns:
        C: torch.Tensor, shape (128, 16), int8 or uint8
    """
    assert A.dtype in [torch.float32, torch.float16], "A must be float32 or float16"
    assert out_dtype in ["int8", "uint8"], "out_dtype must be int8 or uint8"

    mask = b > A

    torch_out_dtype = torch.int8 if out_dtype == "int8" else torch.uint8
    C_result = torch.zeros(C.size(0), C.size(1), dtype=torch_out_dtype, device=A.device)

    for i in range(C.size(0)):
        for j in range(C.size(1)):
            start_bit = j * 8
            end_bit = start_bit + 8

            bits = mask[i, start_bit:end_bit]

            byte_value = 0
            for k in range(8):
                if bits[k]:
                    byte_value |= 1 << k

            C_result[i, j] = byte_value

    return C_result


def run_test_compare_scalar(M, N, block_M, block_N, mode, b_scalar, dtype, out_dtype, target):
    torch.npu.config.allow_internal_format = True

    func = compare_scalar(M, N, block_M, block_N, mode, b_scalar, dtype, out_dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    torch_dtype = torch.float32 if dtype == "float" else torch.float16
    a = torch.zeros(M, N, dtype=torch_dtype).npu()

    torch.npu.synchronize()

    c = func(a)

    torch_out_dtype = torch.int8 if out_dtype == "int8" else torch.uint8
    ref_c = torch.zeros(M, N // 8, dtype=torch_out_dtype).npu()
    ref_c = compare_with_scalar_and_set_bits(a, b_scalar, ref_c, out_dtype)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("out_dtype", ["int8", "uint8"])
@pytest.mark.parametrize("dtype", ["float", "float16"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(256, 256)])
def test_compare_scalar(out_dtype, dtype, target, shape):
    M, N = shape
    block_M = 128
    block_N = 256
    mode = "LT"
    b_scalar = 1.0
    run_test_compare_scalar(M, N, block_M, block_N, mode, b_scalar, dtype, out_dtype, target)


def compare_scalar_slice(M, N, block_M, block_N, mode, b_scalar, dtype="float", out_dtype="uint8"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        C: T.Tensor((M, N // 8), out_dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N // 8), out_dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            for i in range(block_M // VEC_NUM):
                T.tile.compare(c_ub[i, :], a_ub[i, :], b_scalar, mode)

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N // 8])

    return main


def run_test_compare_scalar_slice(M, N, block_M, block_N, mode, b_scalar, dtype, out_dtype, target):
    torch.npu.config.allow_internal_format = True

    func = compare_scalar_slice(M, N, block_M, block_N, mode, b_scalar, dtype, out_dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    torch_dtype = torch.float32 if dtype == "float" else torch.float16
    a = torch.zeros(M, N, dtype=torch_dtype).npu()

    torch.npu.synchronize()

    c = func(a)

    torch_out_dtype = torch.int8 if out_dtype == "int8" else torch.uint8
    ref_c = torch.zeros(M, N // 8, dtype=torch_out_dtype).npu()
    ref_c = compare_with_scalar_and_set_bits(a, b_scalar, ref_c, out_dtype)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("out_dtype", ["int8", "uint8"])
@pytest.mark.parametrize("dtype", ["float", "float16"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(256, 256)])
def test_compare_scalar_slice(out_dtype, dtype, target, shape):
    M, N = shape
    block_M = 128
    block_N = 256
    mode = "LT"
    b_scalar = 1.0
    run_test_compare_scalar_slice(M, N, block_M, block_N, mode, b_scalar, dtype, out_dtype, target)


def cos(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(
        A: T.Tensor([M, N], dtype),  # type: ignore
        B: T.Tensor([M, N], dtype),  # type: ignore
    ):
        T.func_attr({"enable_auto_sync": True})
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            a = T.alloc_ub([sub_block_M, block_N], dtype)
            b = T.alloc_ub([sub_block_M, block_N], dtype)

            T.copy(
                A[bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M, by * block_N : (by + 1) * block_N], a
            )  # Load input
            T.tile.cos(b, a)  # Compute cos
            T.copy(
                b, B[bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M, by * block_N : (by + 1) * block_N]
            )  # Store output

    return main


def run_test_cos(dtype, target):
    test_configs = [
        (1024, 1024, 128, 128),
    ]

    for M, N, block_M, block_N in test_configs:
        func = cos(M, N, block_M, block_N, dtype)
        func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)
        a = torch.randn(M, N, dtype=torch.float32 if dtype == "float" else torch.float16).npu()
        b = func(a)
        ref_b = torch.cos(a)
        torch.testing.assert_close(b, ref_b, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", ["float", "float16"])
@pytest.mark.parametrize("target", ["ascendc"])
def test_cos(dtype, target):
    run_test_cos(dtype, target)


def cos_slice(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(
        A: T.Tensor([M, N], dtype),  # type: ignore
        B: T.Tensor([M, N], dtype),  # type: ignore
    ):
        T.func_attr({"enable_auto_sync": True})
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            a = T.alloc_ub([sub_block_M, block_N], dtype)
            b = T.alloc_ub([sub_block_M, block_N], dtype)

            T.copy(
                A[bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M, by * block_N : (by + 1) * block_N], a
            )  # Load input
            for i in range(sub_block_M):
                T.tile.cos(b[i, :], a[i, :])  # Compute cos
            T.copy(
                b, B[bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M, by * block_N : (by + 1) * block_N]
            )  # Store output

    return main


def run_test_cos_slice(dtype, target):
    test_configs = [
        (1024, 1024, 128, 128),
    ]

    for M, N, block_M, block_N in test_configs:
        func = cos_slice(M, N, block_M, block_N, dtype)
        func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)
        a = torch.randn(M, N, dtype=torch.float32 if dtype == "float" else torch.float16).npu()
        b = func(a)
        ref_b = torch.cos(a)
        torch.testing.assert_close(b, ref_b, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", ["float", "float16"])
@pytest.mark.parametrize("target", ["ascendc"])
def test_cos_slice(dtype, target):
    run_test_cos_slice(dtype, target)


def createvecindex(M, N, block_M, block_N, firstValue, dtype="int32"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 1

    @T.prim_func
    def main(
        C: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.tile.createvecindex(c_ub, firstValue)

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_createvecindex(M, N, block_M, block_N, firstValue, dtype, target):
    func = createvecindex(M, N, block_M, block_N, firstValue, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    torch.npu.synchronize()

    c = func()

    dtype_map = {
        "int16": torch.int16,
        "int32": torch.int32,
        "uint16": torch.uint16,
        "uint32": torch.uint32,
        "float16": torch.float16,
        "float": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.int32)

    if dtype in ["uint16", "uint32"]:
        ref_c = torch.arange(start=firstValue, end=firstValue + block_N, dtype=torch.int32).to(torch_dtype).reshape(M, N)
    else:
        ref_c = torch.arange(start=firstValue, end=firstValue + block_N, dtype=torch_dtype).reshape(M, N)
    ref_c = ref_c.npu()

    if dtype in ["uint16", "uint32"]:
        torch.testing.assert_close(c.cpu(), ref_c.cpu(), rtol=1e-2, atol=1e-2)
    else:
        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["int16", "int32", "float16", "float"])
@pytest.mark.parametrize("target", ["ascendc"])
def test_createvecindex_ascendc(dtype, target):
    M = 1
    N = 1024
    block_M = 1
    block_N = 1024
    firstValue = 0
    run_test_createvecindex(M, N, block_M, block_N, firstValue, dtype, target)


@pytest.mark.parametrize("dtype", ["int32", "uint32", "int16", "uint16"])
@pytest.mark.parametrize("target", ["pto"])
def test_createvecindex_pto(dtype, target):
    M = 1
    N = 1024
    block_M = 1
    block_N = 1024
    firstValue = 0
    run_test_createvecindex(M, N, block_M, block_N, firstValue, dtype, target)


def vec_div(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

            T.tile.div(c_ub, a_ub, b_ub)

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_vec_div(M, N, block_M, block_N, dtype, target):
    func = vec_div(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N, dtype=torch.float32 if dtype == "float" else torch.float16).npu()
    b = torch.randn(M, N, dtype=torch.float32 if dtype == "float" else torch.float16).npu()
    b = torch.where(b >= 0, torch.clamp(b, min=0.5), torch.clamp(b, max=-0.5))

    torch.npu.synchronize()

    c = func(a, b)

    ref_c = a / b

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["float", "float16"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_vec_div(dtype, target, shape):
    M, N = shape
    run_test_vec_div(M, N, 64, 128, dtype, target=target)


def exp(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

            T.tile.exp(b_ub, a_ub)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_exp(M, N, block_M, block_N, dtype, target):
    func = exp(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N, dtype=torch.float32 if dtype == "float" else torch.float16).npu()

    torch.npu.synchronize()

    b = func(a)

    ref_b = torch.exp(a)

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["float", "float16"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_exp(dtype, target, shape):
    M, N = shape
    run_test_exp(M, N, 128, 256, dtype, target=target)


def fill(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype)):  # type: ignore
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((block_M, block_N), dtype)

            T.tile.fill(a_ub, 10.0)
            T.copy(a_ub, A[bx * block_M, by * block_N])

    return main


def run_test_fill(M, N, block_M, block_N, dtype, target):
    func = fill(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    torch.npu.synchronize()

    b = func()

    ref_b = torch.full((M, N), 10.0, dtype=torch.float32 if dtype == "float" else torch.float16).npu()

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["float", "float16"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_fill(dtype, target, shape):
    M, N = shape
    run_test_fill(M, N, 64, 32, dtype, target=target)


def clear(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype)):  # type: ignore
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((block_M, block_N), dtype)

            T.tile.fill(a_ub, 10.0)
            T.tile.clear(a_ub)
            T.copy(a_ub, A[bx * block_M, by * block_N])

    return main


def run_test_clear(M, N, block_M, block_N, dtype, target):
    func = clear(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    torch.npu.synchronize()

    b = func()

    ref_b = torch.zeros((M, N), dtype=torch.float32 if dtype == "float" else torch.float16).npu()

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["float", "float16"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_clear(dtype, target, shape):
    M, N = shape
    run_test_clear(M, N, 64, 32, dtype, target=target)


def gather(M, N, block_M, block_N, dtype="int32"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), "uint32"),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), "uint32")
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

            T.tile.gather(c_ub, a_ub, b_ub, 0)

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def generate_golden_gather(a, b):
    result = torch.zeros(a.size(0), a.size(1), dtype=a.dtype)
    element_size = a.element_size()
    for i in range(a.size(0)):
        tmp_result = torch.zeros(1, a.size(1), dtype=a.dtype)
        for j in range(a.size(1)):
            index = b[i, j].to(torch.int32) // element_size
            index = index.long()
            tmp_result[0, j] = a[i, index]
        result[i:] = tmp_result
    return result


def run_test_gather(M, N, block_M, block_N, dtype, target):
    func = gather(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    dtype_map = {
        "int16": torch.int16,
        "int32": torch.int32,
        "uint16": torch.uint16,
        "uint32": torch.uint32,
        "float": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    element_sizes = {
        "int16": 2,
        "int32": 4,
        "uint16": 2,
        "uint32": 4,
        "float": 4,
        "float16": 2,
        "bfloat16": 2,
    }
    torch_dtype = dtype_map.get(dtype, torch.int32)
    element_size = element_sizes.get(dtype, 4)

    if dtype in ["uint16", "uint32"]:
        a = torch.arange(N, dtype=torch.int32).to(torch_dtype).unsqueeze(0).expand(M, -1).npu()
    else:
        a = torch.arange(N, dtype=torch_dtype).unsqueeze(0).expand(M, -1).npu()
    all_multiples = torch.arange(0, element_size * N, element_size)
    random_indices = torch.randperm(len(all_multiples))[:N]
    random_multiples = all_multiples[random_indices].to(torch.uint32)
    tmp_tensor = random_multiples.reshape(1, N)
    tensor_cpu = tmp_tensor.repeat(M, 1)
    b = tensor_cpu.npu()

    torch.npu.synchronize()

    c = func(a, b)

    ref_c = generate_golden_gather(a, b).npu()

    if dtype in ["uint16", "uint32"]:
        torch.testing.assert_close(c.cpu(), ref_c.cpu(), rtol=1e-2, atol=1e-2)
    else:
        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["int16", "int32", "uint16", "uint32", "float", "float16", "bfloat16"])
@pytest.mark.parametrize("target", ["ascendc"])
@pytest.mark.parametrize("shape", [(128, 1024)])
def test_gather(dtype, target, shape):
    M, N = shape
    block_M = 16
    block_N = N
    run_test_gather(M, N, block_M, block_N, dtype, target)


def gather(M, N, block_M, block_N, dtype="int32"):  # noqa: F811
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), "uint32"),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), "uint32")
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

            T.tile.gather(c_ub, a_ub, b_ub, 0)

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def gather_slice(M, N, block_M, block_N, dtype="int32"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), "uint32"),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), "uint32")
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)
            for i in range(block_M // VEC_NUM):
                T.tile.gather(c_ub[i, :], a_ub[i, :], b_ub[i, :], 0)

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_gather_slice(M, N, block_M, block_N, dtype, target):
    func = gather_slice(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    dtype_map = {
        "int16": torch.int16,
        "int32": torch.int32,
        "uint16": torch.uint16,
        "uint32": torch.uint32,
        "float": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    element_sizes = {
        "int16": 2,
        "int32": 4,
        "uint16": 2,
        "uint32": 4,
        "float": 4,
        "float16": 2,
        "bfloat16": 2,
    }
    torch_dtype = dtype_map.get(dtype, torch.int32)
    element_size = element_sizes.get(dtype, 4)

    if dtype in ["uint16", "uint32"]:
        a = torch.arange(N, dtype=torch.int32).to(torch_dtype).unsqueeze(0).expand(M, -1).npu()
    else:
        a = torch.arange(N, dtype=torch_dtype).unsqueeze(0).expand(M, -1).npu()
    all_multiples = torch.arange(0, element_size * N, element_size)
    random_indices = torch.randperm(len(all_multiples))[:N]
    random_multiples = all_multiples[random_indices].to(torch.uint32)
    tmp_tensor = random_multiples.reshape(1, N)
    tensor_cpu = tmp_tensor.repeat(M, 1)
    b = tensor_cpu.npu()

    torch.npu.synchronize()

    c = func(a, b)

    ref_c = generate_golden_gather(a, b).npu()

    if dtype in ["uint16", "uint32"]:
        torch.testing.assert_close(c.cpu(), ref_c.cpu(), rtol=1e-2, atol=1e-2)
    else:
        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["int16", "int32", "uint16", "uint32", "float", "float16", "bfloat16"])
@pytest.mark.parametrize("target", ["ascendc"])
@pytest.mark.parametrize("shape", [(128, 1024)])
def test_gather_slice(dtype, target, shape):
    M, N = shape
    block_M = 16
    block_N = N
    run_test_gather_slice(M, N, block_M, block_N, dtype, target)


def gatherb(M, N, block_M, block_N, b_len, repeat_time, dtype="uint16"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, b_len), "uint32"),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, b_len), "uint32")
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * b_len], b_ub)

            T.tile.gatherb(c_ub, a_ub, b_ub, repeat_time, 1, 8)

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def generate_golden_gatherb(a, b, N, max_index, element_size):
    result = torch.zeros(1, N, dtype=a.dtype)
    num_elements_per_gather = 32 // element_size
    for i in range(max_index):
        start = b[0, i].to(torch.int32) // element_size
        for j in range(num_elements_per_gather):
            result[0, i * num_elements_per_gather + j] = start + j
    return result


def run_test_gatherb(M, N, block_M, block_N, b_len, repeat_time, dtype, target):
    func = gatherb(M, N, block_M, block_N, b_len, repeat_time, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    dtype_map = {
        "uint16": torch.uint16,
        "uint32": torch.uint32,
    }
    element_sizes = {
        "uint16": 2,
        "uint32": 4,
    }
    torch_dtype = dtype_map.get(dtype, torch.uint16)
    element_size = element_sizes.get(dtype, 2)

    a = torch.arange(N, dtype=torch.int32).to(torch_dtype).unsqueeze(0).expand(M, -1).npu()
    tmp_tensor = torch.zeros(1, b_len, dtype=torch.uint32)
    for i in range(b_len):
        tmp_tensor[0, b_len - 1 - i] = i * 32
    b = tmp_tensor.expand(M, -1).npu()

    torch.npu.synchronize()

    c = func(a, b)

    ref_c = generate_golden_gatherb(a, b, N, b_len, element_size).expand(M, -1).npu()

    torch.testing.assert_close(c.cpu(), ref_c.cpu(), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["uint16", "uint32"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(64, 1024)])
def test_gatherb(dtype, target, shape):
    M, N = shape
    block_M = 2
    block_N = N
    element_sizes = {
        "uint16": 2,
        "uint32": 4,
    }
    element_size = element_sizes.get(dtype, 2)
    b_len = (N * element_size - 1) // 32 + 1
    repeat_time = N * element_size // (32 * 8)
    run_test_gatherb(M, N, block_M, block_N, b_len, repeat_time, dtype, target)


def leaky_relu(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

            T.tile.leaky_relu(b_ub, a_ub, 2.0)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_leaky_relu(M, N, block_M, block_N, dtype, target):
    func = leaky_relu(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N, dtype=torch.float32 if dtype == "float" else torch.float16).npu()

    torch.npu.synchronize()

    b = func(a)

    leaky_relu_golden = nn.LeakyReLU(negative_slope=2.0)
    ref_b = leaky_relu_golden(a)

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["float", "float16"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_leaky_relu(dtype, target, shape):
    M, N = shape
    block_M = 128
    block_N = 256
    run_test_leaky_relu(M, N, block_M, block_N, dtype, target)


def leaky_relu_slice(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            for i in range(block_M // VEC_NUM):
                T.tile.leaky_relu(b_ub[i, :], a_ub[i, :], 2.0)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_leaky_relu_slice(M, N, block_M, block_N, dtype, target):
    func = leaky_relu_slice(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N, dtype=torch.float32 if dtype == "float" else torch.float16).npu()

    torch.npu.synchronize()

    b = func(a)

    leaky_relu_golden = nn.LeakyReLU(negative_slope=2.0)
    ref_b = leaky_relu_golden(a)

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["float", "float16"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_leaky_relu_slice(dtype, target, shape):
    M, N = shape
    block_M = 128
    block_N = 256
    run_test_leaky_relu_slice(M, N, block_M, block_N, dtype, target)


def ln(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

            T.tile.ln(b_ub, a_ub)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_ln(M, N, block_M, block_N, dtype, target):
    func = ln(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = abs(torch.randn(M, N, dtype=torch.float32 if dtype == "float" else torch.float16).npu())

    torch.npu.synchronize()

    b = func(a)

    ref_b = torch.log(a)

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["float", "float16"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_ln(dtype, target, shape):
    M, N = shape
    block_M = 128
    block_N = 256
    run_test_ln(M, N, block_M, block_N, dtype, target)


def gathermask_fixed_mode(M, N, block_M, block_N, dtype="int32"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

            T.tile.gather_mask(b_ub, a_ub, "P0101")

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_gathermask(M, N, block_M, block_N, dtype, target):
    func = gathermask_fixed_mode(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    dtype_map = {
        "int16": torch.int16,
        "int32": torch.int32,
        "uint16": torch.uint16,
        "uint32": torch.uint32,
        "float": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype, torch.int32)

    if dtype in ["uint16", "uint32"]:
        a = torch.arange(1, M * N + 1, dtype=torch.int32).to(torch_dtype).reshape(M, N).npu()
    else:
        a = torch.arange(1, M * N + 1, dtype=torch_dtype).reshape(M, N).npu()

    torch.npu.synchronize()

    b = func(a)
    b = b[:, 0 : N // 2]

    ref_b = a[:, ::2].reshape(M, N // 2)

    if dtype in ["uint16", "uint32"]:
        torch.testing.assert_close(b.cpu(), ref_b.cpu(), rtol=1e-2, atol=1e-2)
    else:
        torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["int16", "int32", "uint16", "uint32", "float", "float16", "bfloat16"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(4, 256)])
def test_gathermask(dtype, target, shape):
    M, N = shape
    run_test_gathermask(M, N, 2, 256, dtype, target=target)


def gathermask_custom_mode(M, N, block_M, block_N, dtype="int32"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        idx: T.Tensor((1, 8), "uint32"),  # type: ignore
        B: T.Tensor((M, 8), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            idx_ub = T.alloc_shared((1, 8), "uint32")
            b_ub = T.alloc_shared((block_M // VEC_NUM, 8), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(idx, idx_ub)

            T.tile.gather_mask(b_ub, a_ub, idx_ub)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * 8])

    return main


def run_test_gathermask_custom_mode(M, N, block_M, block_N, target):
    func = gathermask_custom_mode(M, N, block_M, block_N)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.arange(1, M * N + 1, dtype=torch.int32).reshape(M, N).npu()
    idx = torch.tensor([[1, 2, 2, 5, 4, 6, 7, 8]], dtype=torch.uint32).npu()
    torch.npu.synchronize()

    b = func(a, idx)

    idx_cpu = idx.cpu().expand(M, -1).long()
    a_cpu = a.cpu()
    ref_b = a_cpu[torch.arange(M).unsqueeze(1), idx_cpu].npu()

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(4, 256)])
def test_gathermask_custom_mode(target, shape):
    M, N = shape
    run_test_gathermask_custom_mode(M, N, 2, 256, target=target)


def vec_max(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

            T.tile.max(c_ub, a_ub, b_ub)

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_vec_max(M, N, block_M, block_N, dtype, target):
    func = vec_max(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    dtype_map = {
        "float": torch.float32,
        "float16": torch.float16,
        "int16": torch.int16,
        "int32": torch.int32,
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)

    if dtype in ["int16", "int32"]:
        a = torch.randint(-100, 100, (M, N), dtype=torch_dtype).npu()
        b = torch.randint(-100, 100, (M, N), dtype=torch_dtype).npu()
    else:
        a = torch.randn(M, N, dtype=torch_dtype).npu()
        b = torch.randn(M, N, dtype=torch_dtype).npu()

    torch.npu.synchronize()

    c = func(a, b)

    ref_c = torch.max(a, b)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["float", "float16", "int16", "int32"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_vec_max(dtype, target, shape):
    M, N = shape
    run_test_vec_max(M, N, 64, 128, dtype, target=target)


def vec_maxs(M, N, block_M, block_N, scalar, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M, block_N), dtype)
            b_ub = T.alloc_ub((block_M, block_N), dtype)

            T.copy(A[bx * block_M, by * block_N], a_ub)
            T.tile.max(b_ub, a_ub, scalar)
            T.copy(b_ub, B[bx * block_M, by * block_N])

    return main


def run_test_vec_maxs(M, N, block_M, block_N, scalar, dtype, target):
    func = vec_maxs(M, N, block_M, block_N, scalar, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    dtype_map = {
        "float": torch.float32,
        "float16": torch.float16,
        "int16": torch.int16,
        "int32": torch.int32,
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)

    if dtype in ["int16", "int32"]:
        a = torch.randint(-100, 100, (M, N), dtype=torch_dtype).npu()
    else:
        a = torch.randn(M, N, dtype=torch_dtype).npu()
        a = a * 50

    torch.npu.synchronize()

    b = func(a)

    ref_b = torch.clamp_min(a, scalar)

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["float", "float16", "int16", "int32"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_vec_maxs(dtype, target, shape):
    M, N = shape
    scalar = 2.0 if dtype in ["float", "float16"] else 2
    run_test_vec_maxs(M, N, 64, 32, scalar=scalar, dtype=dtype, target=target)


def vec_min(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

            T.tile.min(c_ub, a_ub, b_ub)

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_vec_min(M, N, block_M, block_N, dtype, target):
    func = vec_min(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    dtype_map = {
        "float": torch.float32,
        "float16": torch.float16,
        "int16": torch.int16,
        "int32": torch.int32,
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)

    if dtype in ["int16", "int32"]:
        a = torch.randint(-100, 100, (M, N), dtype=torch_dtype).npu()
        b = torch.randint(-100, 100, (M, N), dtype=torch_dtype).npu()
    else:
        a = torch.randn(M, N, dtype=torch_dtype).npu()
        b = torch.randn(M, N, dtype=torch_dtype).npu()

    torch.npu.synchronize()

    c = func(a, b)

    ref_c = torch.min(a, b)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["float", "float16", "int16", "int32"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_vec_min(dtype, target, shape):
    M, N = shape
    run_test_vec_min(M, N, 64, 128, dtype, target=target)


def vec_mins(M, N, block_M, block_N, scalar, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M, block_N), dtype)
            b_ub = T.alloc_ub((block_M, block_N), dtype)

            T.copy(A[bx * block_M, by * block_N], a_ub)
            T.tile.min(b_ub, a_ub, scalar)
            T.copy(b_ub, B[bx * block_M, by * block_N])

    return main


def run_test_vec_mins(M, N, block_M, block_N, scalar, dtype, target):
    func = vec_mins(M, N, block_M, block_N, scalar, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    dtype_map = {
        "float": torch.float32,
        "float16": torch.float16,
        "int16": torch.int16,
        "int32": torch.int32,
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)

    if dtype in ["int16", "int32"]:
        a = torch.randint(-100, 100, (M, N), dtype=torch_dtype).npu()
    else:
        a = torch.randn(M, N, dtype=torch_dtype).npu()
        a = a * 50

    torch.npu.synchronize()

    b = func(a)

    ref_b = torch.clamp_max(a, scalar)

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["float", "float16", "int16", "int32"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_vec_mins(dtype, target, shape):
    M, N = shape
    scalar = 2.0 if dtype in ["float", "float16"] else 2
    run_test_vec_mins(M, N, 64, 32, scalar=scalar, dtype=dtype, target=target)


def vec_mul(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

            T.tile.mul(c_ub, a_ub, b_ub)

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_vec_mul(M, N, block_M, block_N, dtype, target):
    func = vec_mul(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    dtype_map = {
        "float": torch.float32,
        "float16": torch.float16,
        "int16": torch.int16,
        "int32": torch.int32,
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)

    if dtype in ["int16", "int32"]:
        a = torch.randint(-100, 100, (M, N), dtype=torch_dtype).npu()
        b = torch.randint(-100, 100, (M, N), dtype=torch_dtype).npu()
    else:
        a = torch.randn(M, N, dtype=torch_dtype).npu()
        b = torch.randn(M, N, dtype=torch_dtype).npu()

    torch.npu.synchronize()

    c = func(a, b)

    ref_c = a * b

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["float", "float16", "int16", "int32"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_vec_mul(dtype, target, shape):
    M, N = shape
    run_test_vec_mul(M, N, 64, 128, dtype, target=target)


def vec_muls(M, N, block_M, block_N, scalar, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M, block_N), dtype)
            b_ub = T.alloc_ub((block_M, block_N), dtype)
            T.copy(A[bx * block_M, by * block_N], a_ub)
            T.tile.mul(b_ub, a_ub, scalar)
            T.copy(b_ub, B[bx * block_M, by * block_N])

    return main


def run_test_vec_muls(M, N, block_M, block_N, scalar, dtype, target):
    func = vec_muls(M, N, block_M, block_N, scalar, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    dtype_map = {
        "float": torch.float32,
        "float16": torch.float16,
        "int16": torch.int16,
        "int32": torch.int32,
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)

    if dtype in ["int16", "int32"]:
        a = torch.randint(-100, 100, (M, N), dtype=torch_dtype).npu()
    else:
        a = torch.randn(M, N, dtype=torch_dtype).npu()

    torch.npu.synchronize()

    b = func(a)

    ref_b = a * scalar

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["float", "float16", "int16", "int32"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_vec_muls(dtype, target, shape):
    M, N = shape
    scalar = 2.0 if dtype in ["float", "float16"] else 2
    run_test_vec_muls(M, N, 64, 32, scalar=scalar, dtype=dtype, target=target)


def bitwise_or(M, N, block_M, block_N, dtype="int16"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

            T.tile.bitwise_or(c_ub, a_ub, b_ub)

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_bitwise_or(M, N, block_M, block_N, dtype, target):
    func = bitwise_or(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    torch_dtype = torch.int16 if dtype == "int16" else torch.uint16
    a = torch.randint(0, 10, (M, N), dtype=torch_dtype).npu()
    b = torch.randint(0, 10, (M, N), dtype=torch_dtype).npu()

    torch.npu.synchronize()

    c = func(a, b)

    ref_c = a | b

    assert_close_npu(c, ref_c, dtype, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["int16", "uint16"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_bitwise_or(dtype, target, shape):
    M, N = shape
    run_test_bitwise_or(M, N, 128, 256, dtype, target=target)


def vec_pow(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

            T.tile.pow(c_ub, a_ub, b_ub)

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_pow(M, N, block_M, block_N, dtype, target):
    func = vec_pow(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    dtype_map = {
        "float": torch.float32,
        "float16": torch.float16,
        "int32": torch.int32,
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)

    if dtype == "int32":
        a = torch.randint(1, 10, (M, N), dtype=torch_dtype).npu()
        b = torch.randint(0, 5, (M, N), dtype=torch_dtype).npu()
    else:
        a = torch.rand(M, N, dtype=torch_dtype).npu() + 0.5
        b = torch.rand(M, N, dtype=torch_dtype).npu()

    torch.npu.synchronize()

    c = func(a, b)

    ref_c = torch.pow(a, b)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


pow_dtype_target_params = [
    ("float", "ascendc"),
    ("float16", "ascendc"),
    ("float", "pto"),
    ("float16", "pto"),
    ("int32", "ascendc"),
]


@pytest.mark.parametrize("dtype,target", pow_dtype_target_params)
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_pow(dtype, target, shape):
    M, N = shape
    run_test_pow(M, N, 128, 128, dtype, target=target)


def vec_pow_slice(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)
            for i in range(block_M // VEC_NUM):
                T.tile.pow(c_ub[i, :], a_ub[i, :], b_ub[i, :])

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_pow_slice(M, N, block_M, block_N, dtype, target):
    func = vec_pow_slice(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    dtype_map = {
        "float": torch.float32,
        "float16": torch.float16,
        "int32": torch.int32,
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)

    if dtype == "int32":
        a = torch.randint(1, 10, (M, N), dtype=torch_dtype).npu()
        b = torch.randint(0, 5, (M, N), dtype=torch_dtype).npu()
    else:
        a = torch.rand(M, N, dtype=torch_dtype).npu() + 0.5
        b = torch.rand(M, N, dtype=torch_dtype).npu()

    torch.npu.synchronize()

    c = func(a, b)

    ref_c = torch.pow(a, b)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype,target", pow_dtype_target_params)
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_pow_slice(dtype, target, shape):
    M, N = shape
    run_test_pow_slice(M, N, 128, 128, dtype, target=target)


def reciprocal(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

            T.tile.reciprocal(b_ub, a_ub)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_reciprocal(M, N, block_M, block_N, dtype, target):
    func = reciprocal(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.rand(M, N, dtype=torch.float32 if dtype == "float" else torch.float16).npu() * 0.9 + 0.1

    torch.npu.synchronize()

    b = func(a)

    ref_b = 1 / a

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["float", "float16"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_reciprocal(dtype, target, shape):
    M, N = shape
    run_test_reciprocal(M, N, 128, 128, dtype, target=target)


def relu(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

            T.tile.relu(b_ub, a_ub)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_relu(M, N, block_M, block_N, dtype, target):
    func = relu(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    dtype_map = {
        "float": torch.float32,
        "float16": torch.float16,
        "float32": torch.float32,
        "int32": torch.int32,
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)

    if dtype == "int32":
        a = torch.randint(-100, 100, (M, N), dtype=torch_dtype).npu()
    else:
        a = torch.randn(M, N, dtype=torch_dtype).npu()

    torch.npu.synchronize()

    b = func(a)

    ref_b = torch.relu(a)

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["float16", "float32", "int32"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_relu(dtype, target, shape):
    M, N = shape
    run_test_relu(M, N, 128, 256, dtype, target=target)


def rsqrt(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

            T.tile.rsqrt(b_ub, a_ub)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_rsqrt(M, N, block_M, block_N, dtype, target):
    func = rsqrt(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N, dtype=torch.float32 if dtype == "float" else torch.float16).npu()

    torch.npu.synchronize()

    b = func(a)

    ref_b = torch.rsqrt(a)

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2, equal_nan=True)


@pytest.mark.parametrize("dtype", ["float", "float16"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_rsqrt(dtype, target, shape):
    M, N = shape
    run_test_rsqrt(M, N, 128, 256, dtype, target=target)


def vec_select(M, N, block_M, block_N, mode, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
        MASK: T.Tensor((M, N // 8), "uint8"),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            selmask_ub = T.alloc_ub((block_M // VEC_NUM, block_N // 8), "uint8")
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)
            T.copy(MASK[bx * block_M + vid * block_M // VEC_NUM, by * block_N // 8], selmask_ub)

            T.tile.select(c_ub, selmask_ub, a_ub, b_ub, mode)

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def select_by_mask_bits_select(A, B, M):
    """
    select element from A or B by bits in M, then set it to C

    Args:
        A: torch.Tensor, shape (128, 128), float32 or float16
        B: torch.Tensor, shape (128, 128), float32 or float16
        M: torch.Tensor, shape (128, 16), uint8,

    Returns:
        C: torch.Tensor, shape (128, 128), same dtype as A
    """
    assert A.dtype == B.dtype, "A and B must have the same dtype"
    assert M.dtype == torch.uint8, "M must be uint8"

    C = torch.zeros_like(A).npu()
    M_cpu = M.cpu()

    for i in range(M_cpu.size(0)):
        for j in range(M_cpu.size(1)):
            byte_val = M_cpu[i, j]

            start_col = j * 8

            for bit_pos in range(8):
                col = start_col + bit_pos
                if (byte_val >> bit_pos) & 1:
                    C[i, col] = A[i, col]
                else:
                    C[i, col] = B[i, col]

    return C


def run_test_vec_select(M, N, block_M, block_N, mode, dtype, target):
    torch.npu.config.allow_internal_format = True

    func = vec_select(M, N, block_M, block_N, mode, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)
    a = torch.zeros(M, N, dtype=torch.float32 if dtype == "float" else torch.float16).npu()
    b = torch.ones(M, N, dtype=torch.float32 if dtype == "float" else torch.float16).npu()
    m = torch.full((M, N // 8), 0xF, dtype=torch.uint8).npu()

    torch.npu.synchronize()

    c = func(a, b, m)

    ref_c = select_by_mask_bits_select(a, b, m)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["float", "float16"])
@pytest.mark.parametrize("target", ["ascendc"])
@pytest.mark.parametrize("shape", [(256, 256)])
def test_vec_select(dtype, target, shape):
    M, N = shape
    block_M = 64
    block_N = 128
    mode = "VSEL_CMPMASK_SPR"
    run_test_vec_select(M, N, block_M, block_N, mode, dtype, target)


def vec_select_scalar(M, N, block_M, block_N, mode, b_scalar, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        MASK: T.Tensor((M, N // 8), "uint8"),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            selmask_ub = T.alloc_ub((block_M // VEC_NUM, block_N // 8), "uint8")
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(MASK[bx * block_M + vid * block_M // VEC_NUM, by * block_N // 8], selmask_ub)

            T.tile.select(c_ub, selmask_ub, a_ub, b_scalar, mode)

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def select_by_mask_bits_select_scalar(A, b, M):
    """
    select value by mask bits of M from A or B to assign to C

    Args:
        A: torch.Tensor, shape (128, 128), float32 or float16
        B: scalar value
        M: torch.Tensor, shape (128, 16), uint8, 每个元素存储8个bit位

    Returns:
        C: torch.Tensor, shape (128, 128), same dtype as A
    """
    assert M.dtype == torch.uint8, "M must be uint8"

    C = torch.zeros_like(A).npu()
    M_cpu = M.cpu()

    for i in range(M_cpu.size(0)):
        for j in range(M_cpu.size(1)):
            byte_val = M_cpu[i, j]

            start_col = j * 8

            for bit_pos in range(8):
                col = start_col + bit_pos
                if (byte_val >> bit_pos) & 1:
                    C[i, col] = A[i, col]
                else:
                    C[i, col] = b

    return C


def run_test_vec_select_scalar(M, N, block_M, block_N, mode, b_scalar, dtype, target):
    torch.npu.config.allow_internal_format = True

    func = vec_select_scalar(M, N, block_M, block_N, mode, b_scalar, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.zeros(M, N, dtype=torch.float32 if dtype == "float" else torch.float16).npu()
    m = torch.full((M, N // 8), 0xF, dtype=torch.uint8).npu()

    torch.npu.synchronize()

    c = func(a, m)

    ref_c = select_by_mask_bits_select_scalar(a, b_scalar, m)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["float", "float16"])
@pytest.mark.parametrize("target", ["ascendc"])
@pytest.mark.parametrize("shape", [(256, 256)])
def test_vec_select_scalar(dtype, target, shape):
    M, N = shape
    block_M = 64
    block_N = 128
    mode = "VSEL_TENSOR_SCALAR_MODE"
    b_scalar = 1.0
    run_test_vec_select_scalar(M, N, block_M, block_N, mode, b_scalar, dtype, target)


def sin(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(
        A: T.Tensor([M, N], dtype),  # type: ignore
        B: T.Tensor([M, N], dtype),  # type: ignore
    ):
        T.func_attr({"enable_auto_sync": True})
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            a = T.alloc_ub([sub_block_M, block_N], dtype)
            b = T.alloc_ub([sub_block_M, block_N], dtype)
            T.copy(
                A[bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M, by * block_N : (by + 1) * block_N], a
            )  # Load input
            T.tile.sin(b, a)  # Compute sin
            T.copy(
                b, B[bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M, by * block_N : (by + 1) * block_N]
            )  # Store output

    return main


def run_test_sin(dtype, target):
    test_configs = [
        (1024, 1024, 128, 128),
    ]

    for M, N, block_M, block_N in test_configs:
        func = sin(M, N, block_M, block_N, dtype)
        func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)
        a = torch.randn(M, N, dtype=torch.float32 if dtype == "float" else torch.float16).npu()
        b = func(a)
        ref_b = torch.sin(a)
        torch.testing.assert_close(b, ref_b, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", ["float", "float16"])
@pytest.mark.parametrize("target", ["ascendc"])
def test_sin(dtype, target):
    run_test_sin(dtype, target)


def sin_slice(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(
        A: T.Tensor([M, N], dtype),  # type: ignore
        B: T.Tensor([M, N], dtype),  # type: ignore
    ):
        T.func_attr({"enable_auto_sync": True})
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            a = T.alloc_ub([sub_block_M, block_N], dtype)
            b = T.alloc_ub([sub_block_M, block_N], dtype)
            T.copy(
                A[bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M, by * block_N : (by + 1) * block_N], a
            )  # Load input
            for i in range(sub_block_M):
                T.tile.sin(b[i, :], a[i, :])  # Compute sin
            T.copy(
                b, B[bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M, by * block_N : (by + 1) * block_N]
            )  # Store output

    return main


def run_test_sin_slice(dtype, target):
    test_configs = [
        (1024, 1024, 128, 128),
    ]

    for M, N, block_M, block_N in test_configs:
        func = sin_slice(M, N, block_M, block_N, dtype)
        func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)
        a = torch.randn(M, N, dtype=torch.float32 if dtype == "float" else torch.float16).npu()
        b = func(a)
        ref_b = torch.sin(a)
        torch.testing.assert_close(b, ref_b, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", ["float", "float16"])
@pytest.mark.parametrize("target", ["ascendc"])
def test_sin_slice(dtype, target):
    run_test_sin_slice(dtype, target)


def sort(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 1

    ub_N = ((block_N + 31) // 32) * 32

    @T.prim_func
    def main(
        a: T.Tensor((M, N), dtype),
        b: T.Tensor((M, 2 * N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            src_ub = T.alloc_ub((block_M // VEC_NUM, ub_N), dtype)
            dst_ub = T.alloc_ub((block_M // VEC_NUM, ub_N * 2), dtype)

            T.copy(a[bx * block_M + vid * block_M // VEC_NUM, by * ub_N], src_ub)

            T.tile.sort(dst_ub, src_ub, block_N)

            T.copy(dst_ub, b[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_sort(M, N, block_M, block_N, dtype, target):
    func = sort(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    torch_dtype = torch.float if dtype == "float" else torch.float16
    a = torch.arange(0, M * N, dtype=torch_dtype).reshape(M, N).npu()
    b = func(a)

    b_cpu = b.cpu().float().reshape(-1)
    a_cpu = a.cpu().float().reshape(-1)

    out_values = b_cpu[0::2][:N]
    out_indices = b_cpu[1::2][:N]

    ref_vals, ref_index = torch.sort(a_cpu, descending=True)

    torch.testing.assert_close(out_values, ref_vals, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(out_indices, ref_index.float(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dtype", ["float16", "float"])
@pytest.mark.parametrize("target", ["ascendc"])
@pytest.mark.parametrize("shape", [(1, 131)])
def test_sort(dtype, target, shape):
    M, N = shape
    block_M = 1
    block_N = 131
    run_test_sort(M, N, block_M, block_N, dtype, target)


def sort32(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    out_size_multiplier = 4 if dtype == "float16" else 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), "uint32"),  # type: ignore
        C: T.Tensor((M, out_size_multiplier * N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M, block_N // VEC_NUM), dtype)
            b_ub = T.alloc_ub((block_M, block_N // VEC_NUM), "uint32")
            c_ub = T.alloc_ub((block_M, out_size_multiplier * block_N // VEC_NUM), dtype)

            T.copy(A[bx * block_M, by * block_N + vid * block_N // VEC_NUM], a_ub)
            T.copy(B[bx * block_M, by * block_N + vid * block_N // VEC_NUM], b_ub)

            T.tile.sort32(c_ub, a_ub, b_ub)

            T.copy(c_ub, C[bx * block_M, out_size_multiplier * (by * block_N + vid * block_N // VEC_NUM)])

    return main


def run_test_sort32(M, N, block_M, block_N, dtype, target):
    func = sort32(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    dtype_map = {
        "float": torch.float32,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)

    out_size_multiplier = 4 if dtype == "float16" else 2

    if dtype == "float16":
        a = torch.randint(low=1, high=101, size=(M, N), dtype=torch.int32).to(torch.float16).npu()
    else:
        a = torch.randint(low=1, high=101, size=(M, N), dtype=torch_dtype).npu()
    b = torch.zeros((M, N), dtype=torch.uint32).npu()

    torch.npu.synchronize()

    c = func(a, b)
    group_size = 32
    total_elements = M * N
    b_ref = torch.zeros((M, N), dtype=torch.int32, device="npu")
    src0_flat = a.flatten()
    src1_flat = b_ref.flatten()

    total_groups = (total_elements + group_size - 1) // group_size
    for i in range(total_groups):
        start = i * group_size
        end = min((i + 1) * group_size, total_elements)

        group_src0 = src0_flat[start:end]
        group_src1 = src1_flat[start:end]
        sorted_indices = torch.argsort(group_src0, descending=True)

        src0_flat[start:end] = group_src0[sorted_indices]
        src1_flat[start:end] = group_src1[sorted_indices]

    sorted_src0 = src0_flat.reshape(M, N)
    sorted_src1 = src1_flat.reshape(M, N)

    ref_c = torch.empty((M, out_size_multiplier * N), dtype=torch_dtype)
    if dtype == "float16":
        sorted_values = sorted_src0.to(torch_dtype)
        sorted_indices_bytes = sorted_src1.view(torch.int16).reshape(M, N * 2)
        ref_c[:, 0::4] = sorted_values
        ref_c[:, 1::4] = torch.zeros_like(sorted_values)
        ref_c[:, 2::4] = sorted_indices_bytes[:, 0::2]
        ref_c[:, 3::4] = sorted_indices_bytes[:, 1::2]
    else:
        ref_c[:, ::2] = sorted_src0.to(torch_dtype)
        ref_c[:, 1::2] = sorted_src1.to(torch_dtype).view(torch_dtype)

    torch.testing.assert_close(c, ref_c.npu(), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_sort32(dtype, target, shape):
    M, N = shape
    block_M = 64
    block_N = 128
    run_test_sort32(M, N, block_M, block_N, dtype, target)


def topk(M, N, K, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 1

    ub_N = ((block_N + 31) // 32) * 32

    @T.prim_func
    def main(
        a: T.Tensor((M, N), dtype),
        b: T.Tensor((M, 2 * K), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            src_ub = T.alloc_ub((block_M // VEC_NUM, ub_N), dtype)
            dst_ub = T.alloc_ub((block_M // VEC_NUM, K * 2), dtype)

            T.copy(a[bx * block_M + vid * block_M // VEC_NUM, by * ub_N], src_ub)

            T.tile.topk(dst_ub, src_ub, K, block_N)

            T.copy(dst_ub, b[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_topk(M, N, K, block_M, block_N, dtype, target):
    func = topk(M, N, K, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    torch_dtype = torch.float if dtype == "float" else torch.float16
    a = torch.arange(0, M * N, dtype=torch_dtype).reshape(M, N).npu()
    b = func(a)

    b_cpu = b.cpu().float().reshape(-1)
    a_cpu = a.cpu().float().reshape(-1)

    out_values = b_cpu[0::2][:K]
    out_indices = b_cpu[1::2][:K]

    topk_vals, topk_index = torch.sort(a_cpu, descending=True)
    ref_values = topk_vals[:K]
    ref_indices = topk_index[:K].float()

    torch.testing.assert_close(out_values, ref_values, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(out_indices, ref_indices.float(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dtype", ["float16", "float"])
@pytest.mark.parametrize("target", ["ascendc"])
@pytest.mark.parametrize("shape", [(1, 51)])
def test_topk(dtype, target, shape):
    M, N = shape
    block_M = 1
    block_N = 51
    K = 10
    run_test_topk(M, N, K, block_M, block_N, dtype, target)


def sqrt(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

            T.tile.sqrt(b_ub, a_ub)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_sqrt(M, N, block_M, block_N, dtype, target):
    func = sqrt(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.rand(M, N, dtype=torch.float32 if dtype == "float" else torch.float16).npu()

    torch.npu.synchronize()

    b = func(a)

    ref_b = torch.sqrt(a)

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["float", "float16"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_sqrt(dtype, target, shape):
    M, N = shape
    run_test_sqrt(M, N, 128, 256, dtype, target=target)


def vec_sub(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

            T.tile.sub(c_ub, a_ub, b_ub)

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_vec_sub(M, N, block_M, block_N, dtype, target):
    func = vec_sub(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    dtype_map = {
        "float": torch.float32,
        "float16": torch.float16,
        "int16": torch.int16,
        "int32": torch.int32,
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)

    if dtype in ["int16", "int32"]:
        a = torch.randint(-100, 100, (M, N), dtype=torch_dtype).npu()
        b = torch.randint(-100, 100, (M, N), dtype=torch_dtype).npu()
    else:
        a = torch.randn(M, N, dtype=torch_dtype).npu()
        b = torch.randn(M, N, dtype=torch_dtype).npu()

    torch.npu.synchronize()

    c = func(a, b)

    ref_c = a - b

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["float", "float16", "int16", "int32"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_vec_sub(dtype, target, shape):
    M, N = shape
    run_test_vec_sub(M, N, 64, 128, dtype, target=target)


def vec_subs(M, N, block_M, block_N, scalar, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor[(M, N), dtype],
        B: T.Tensor[(M, N), dtype],
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

            T.tile.sub(b_ub, a_ub, scalar)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_vec_subs(M, N, block_M, block_N, scalar, dtype, target):
    func = vec_subs(M, N, block_M, block_N, scalar, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    dtype_map = {
        "float": torch.float32,
        "float16": torch.float16,
        "int16": torch.int16,
        "int32": torch.int32,
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)

    if dtype in ["int16", "int32"]:
        a = torch.randint(-100, 100, (M, N), dtype=torch_dtype).npu()
    else:
        a = torch.randn(M, N, dtype=torch_dtype).npu()

    torch.npu.synchronize()

    b = func(a)

    ref_b = a - scalar

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["float", "float16", "int16", "int32"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_vec_subs(dtype, target, shape):
    M, N = shape
    scalar = 3.0 if dtype in ["float", "float16"] else 3
    run_test_vec_subs(M, N, 128, 256, scalar, dtype, target=target)


def transpose(M, N, block_M, block_N, dtype="int16"):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((M, N), dtype)
            b_ub = T.alloc_ub((M, N), dtype)

            T.copy(A, a_ub)

            T.tile.transpose(b_ub, a_ub)

            T.copy(b_ub, B)

    return main


def run_test_transpose(M, N, block_M, block_N, dtype, target):
    func = transpose(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    dtype_map = {
        "float": torch.float32,
        "float16": torch.float16,
        "int16": torch.int16,
        "int32": torch.int32,
        "uint16": torch.uint16,
        "uint32": torch.uint32,
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)

    if dtype in ["int16", "int32", "uint16", "uint32"]:
        a = torch.randint(
            -100 if dtype in ["int16", "int32"] else 0, 100 if dtype in ["int16", "int32"] else 200, (M, N), dtype=torch_dtype
        ).npu()
    else:
        a = torch.randn(M, N, dtype=torch_dtype).npu()

    torch.npu.synchronize()

    b = func(a)

    ref_b = a.T

    assert_close_npu(b, ref_b, dtype, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["int16", "uint16", "float16", "int32", "uint32", "float"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(16, 16)])
def test_transpose(dtype, target, shape):
    M, N = shape
    run_test_transpose(M, N, 16, 16, dtype, target)


def wholereducemax(M, N, block_M, block_N, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, 2 * N // mask), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, 2 * block_N // mask), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.tile.wholereducemax(b_ub, a_ub, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride)
            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * 2 * block_N // mask])

    return main


def run_test_wholereducemax(M, N, block_M, block_N, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride, target):
    func = wholereducemax(M, N, block_M, block_N, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N, dtype=torch.float16).npu()

    torch.npu.synchronize()

    b = func(a)

    num_groups = M * N // mask
    ref_b = torch.zeros((1, 2 * num_groups)).to(torch.float16)
    a_flag = a.reshape(-1)
    for i in range(num_groups):
        start = i * mask
        end = start + mask
        group = a_flag[start:end]
        max_val = torch.max(group).item()
        max_idx_in_group = torch.argmax(group).item()
        result = torch.tensor([max_idx_in_group], dtype=torch.uint16).view(torch.float16).float().item()
        ref_b[0, 2 * i] = max_val
        ref_b[0, 2 * i + 1] = result
    ref_b = ref_b.reshape(M, 2 * N // mask)
    ref_b = ref_b.npu().to(dtype=torch.float16)

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc"])
def test_wholereducemax(target):
    M = 2
    N = 512
    block_M = 2
    block_N = 128
    mask = 64
    repeatTimes = 2
    dstRepStride = 1
    srcBlkStride = 1
    srcRepStride = 4
    run_test_wholereducemax(M, N, block_M, block_N, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride, target)


def wholereducemin(M, N, block_M, block_N, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, 2 * N // mask), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, 2 * block_N // mask), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.tile.wholereducemin(b_ub, a_ub, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride)
            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * 2 * block_N // mask])

    return main


def run_test_wholereducemin(M, N, block_M, block_N, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride, target):
    func = wholereducemin(M, N, block_M, block_N, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N, dtype=torch.float16).npu()

    torch.npu.synchronize()

    b = func(a)

    num_groups = M * N // mask
    ref_b = torch.zeros((1, 2 * num_groups)).to(torch.float16)
    a_flag = a.reshape(-1)
    for i in range(num_groups):
        start = i * mask
        end = start + mask
        group = a_flag[start:end]
        min_val = torch.min(group).item()
        min_idx_in_group = torch.argmin(group).item()
        result = torch.tensor([min_idx_in_group], dtype=torch.uint16).view(torch.float16).float().item()
        ref_b[0, 2 * i] = min_val
        ref_b[0, 2 * i + 1] = result
    ref_b = ref_b.reshape(M, 2 * N // mask)
    ref_b = ref_b.npu().to(dtype=torch.float16)

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc"])
def test_wholereducemin(target):
    M = 2
    N = 512
    block_M = 2
    block_N = 128
    mask = 64
    repeatTimes = 2
    dstRepStride = 1
    srcBlkStride = 1
    srcRepStride = 4
    run_test_wholereducemin(M, N, block_M, block_N, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride, target)


def wholereducesum(M, N, block_M, block_N, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N // mask), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N // mask), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.tile.wholereducesum(b_ub, a_ub, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride)
            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N // mask])

    return main


def run_test_wholereducesum(M, N, block_M, block_N, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride, target):
    func = wholereducesum(M, N, block_M, block_N, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N, dtype=torch.float16).npu()

    torch.npu.synchronize()

    b = func(a)

    num_groups = M * N // mask
    ref_b = torch.zeros((1, num_groups))
    a_flag = a.reshape(-1)
    for i in range(num_groups):
        start = i * mask
        end = start + mask
        group = a_flag[start:end]
        sum_val = torch.sum(group).item()
        ref_b[0, i] = sum_val
    ref_b = ref_b.reshape(M, N // mask)
    ref_b = ref_b.npu().to(dtype=torch.float16)

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc"])
def test_wholereducesum(target):
    M = 2
    N = 512
    block_M = 2
    block_N = 128
    mask = 64
    repeatTimes = 2
    dstRepStride = 1
    srcBlkStride = 1
    srcRepStride = 4
    run_test_wholereducesum(M, N, block_M, block_N, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride, target)


def generate_arithmetic_progression(N, block_size, dtype="int32"):
    num_blocks = N // block_size

    @T.prim_func
    def main(
        output: T.Tensor((N,), dtype),  # type: ignore
    ):
        with T.Kernel(num_blocks, is_npu=True) as (cid, _):
            start_idx = cid * block_size

            seq_ub = T.alloc_shared((block_size,), dtype)

            T.tile.arith_progression(seq_ub, start_idx, 1, block_size)

            T.copy(seq_ub, output[start_idx])

    return main


def run_test_generate_arithmetic_progression(N, block_size, target):
    func = generate_arithmetic_progression(N, block_size)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    output = torch.zeros(N, dtype=torch.int32).npu()

    torch.npu.synchronize()

    result = func(output)

    ref_result = torch.arange(0, N, dtype=torch.int32).npu()

    torch.testing.assert_close(result, ref_result, rtol=0, atol=0)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [1024])
def test_generate_arithmetic_progression(target, shape):
    N = shape
    block_size = 64
    run_test_generate_arithmetic_progression(N, block_size, target)


def reduce_sum(M, N, block_M, block_N, dim, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

            T.reduce_sum(a_ub, b_ub, dim, [block_M // VEC_NUM, block_N])
            T.barrier_all()

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM])

    return main


def run_test_reduce_sum(M, N, block_M, block_N, dim, dtype, target):
    func = reduce_sum(M, N, block_M, block_N, dim, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N, dtype=torch.float32 if dtype == "float" else torch.float16).npu()
    b = torch.zeros(M, dtype=torch.float32).npu()
    torch.npu.synchronize()

    b = func(a)

    if dim == -1:
        ref_b = torch.sum(a, dim=1)
    else:
        ref_b = torch.sum(a, dim=0)
    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dim", [-1])
@pytest.mark.parametrize("dtype", ["float", "float16"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_reduce_sum(dim, dtype, target):
    if dtype == "float16":
        pytest.xfail(reason="float16 reduction sum may overflow")
    M, N = 1024, 64
    run_test_reduce_sum(M, N, 64, 64, dim, dtype, target)


def reduce_max(M, N, block_M, block_N, dim, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M, block_N), dtype)
            b_ub = T.alloc_ub((block_M), dtype)

            T.copy(A[bx * block_M, by * block_N], a_ub)

            T.reduce_max(a_ub, b_ub, dim, [block_M, block_N])

            T.copy(b_ub, B[bx * block_M])

    return main


def run_test_reduce_max(M, N, block_M, block_N, dim, dtype, target):
    func = reduce_max(M, N, block_M, block_N, dim, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N, dtype=torch.float32 if dtype == "float" else torch.float16).npu()

    torch.npu.synchronize()

    b = func(a)

    if dim == -1:
        ref_b = torch.max(a, dim=1)[0]
    else:
        ref_b = torch.max(a, dim=0)[0]
    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dim", [-1])
@pytest.mark.parametrize("dtype", ["float", "float16"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_reduce_max(dim, dtype, target):
    M, N = 1024, 64
    run_test_reduce_max(M, N, 64, 64, dim, dtype, target)


def reduce_min(M, N, block_M, block_N, dim, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M, block_N), dtype)
            b_ub = T.alloc_ub((block_M), dtype)

            T.copy(A[bx * block_M, by * block_N], a_ub)

            T.reduce_min(a_ub, b_ub, dim, [block_M, block_N])

            T.copy(b_ub, B[bx * block_M])

    return main


def run_test_reduce_min(M, N, block_M, block_N, dim, dtype, target):
    func = reduce_min(M, N, block_M, block_N, dim, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N, dtype=torch.float32 if dtype == "float" else torch.float16).npu()

    torch.npu.synchronize()

    b = func(a)

    if dim == -1:
        ref_b = torch.min(a, dim=1)[0]
    else:
        ref_b = torch.min(a, dim=0)[0]
    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dim", [-1])
@pytest.mark.parametrize("dtype", ["float", "float16"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_reduce_min(dim, dtype, target):
    M, N = 1024, 64
    run_test_reduce_min(M, N, 64, 64, dim, dtype, target)


def reduce_runtime_semantics_kernel(M, N, op, dim, clear=True, init_value=0.0, dtype="float"):
    reduce_fn = _get_reduce_fn(op)
    output_size = M if dim == -1 else N

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((output_size,), dtype),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (_, vid):
            a_ub = T.alloc_ub((M, N), dtype)
            b_ub = T.alloc_ub((output_size,), dtype)

            if vid == 0:
                T.copy(A, a_ub)
                if not clear:
                    T.tile.fill(b_ub, init_value)
                reduce_fn(a_ub, b_ub, dim=dim, clear=clear)
                T.copy(b_ub, B)

    return main


def reduce_runtime_reference(a, op, dim, clear=True, init_value=0.0):
    reduce_dim = 1 if dim == -1 else 0
    if op == "sum":
        reduced = torch.sum(a, dim=reduce_dim)
    elif op == "max":
        reduced = torch.max(a, dim=reduce_dim).values
    else:
        reduced = torch.min(a, dim=reduce_dim).values

    if clear:
        return reduced

    init = torch.full_like(reduced, init_value)
    if op == "sum":
        return reduced + init
    if op == "max":
        return torch.maximum(reduced, init)
    return torch.minimum(reduced, init)


def run_test_reduce_runtime_semantics(op, dim, target, clear=True, init_value=0.0):
    M, N = 64, 64
    func = reduce_runtime_semantics_kernel(M, N, op, dim, clear=clear, init_value=init_value, dtype="float")
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N, dtype=torch.float32).npu()
    torch.npu.synchronize()
    b = func(a)

    ref_b = reduce_runtime_reference(a, op, dim, clear=clear, init_value=init_value)
    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("op", ["sum", "max", "min"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_reduce_dim0_runtime_smoke(op, target):
    run_test_reduce_runtime_semantics(op, dim=0, target=target, clear=True)


@pytest.mark.parametrize(
    ("op", "init_value"),
    [
        pytest.param("sum", 1.25, id="sum"),
        pytest.param("max", -0.5, id="max"),
        pytest.param("min", 0.5, id="min"),
    ],
)
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_reduce_clear_false_runtime_merge(op, init_value, target):
    run_test_reduce_runtime_semantics(op, dim=-1, target=target, clear=False, init_value=init_value)


@pytest.mark.parametrize("op", ["sum", "max", "min"])
def test_reduce_api_compat_positional_and_keyword(monkeypatch, op):
    captured = []

    def fake_reduce_with_clear(buffer, out, reduce_type, dim, clear, real_shape):
        captured.append((reduce_type, dim, clear, real_shape))
        return "ok"

    monkeypatch.setattr(reduce_ascend_lang, "_reduce_with_clear", fake_reduce_with_clear)

    reduce_fn = _get_reduce_fn(op)
    input_buffer = tir.decl_buffer((4, 8), "float32")
    output_buffer = tir.decl_buffer((4,), "float32")

    assert reduce_fn(input_buffer, output_buffer, dim=-1) == "ok"
    assert reduce_fn(input_buffer, output_buffer, dim=-1, clear=False) == "ok"
    assert reduce_fn(input_buffer, output_buffer, -1, False) == "ok"
    assert reduce_fn(input_buffer, output_buffer, dim=0, real_shape=[4, 8]) == "ok"
    assert reduce_fn(input_buffer, output_buffer, 0, [4, 8]) == "ok"
    assert reduce_fn(input_buffer, output_buffer, 0, [4, 8], False) == "ok"
    assert reduce_fn(input_buffer, output_buffer, 0, False, [4, 8]) == "ok"

    expected_reduce_type = f"reduce_{op}"
    assert captured == [
        (expected_reduce_type, -1, True, None),
        (expected_reduce_type, -1, False, None),
        (expected_reduce_type, -1, False, None),
        (expected_reduce_type, 0, True, [4, 8]),
        (expected_reduce_type, 0, True, [4, 8]),
        (expected_reduce_type, 0, False, [4, 8]),
        (expected_reduce_type, 0, False, [4, 8]),
    ]


@pytest.mark.parametrize("op", ["sum", "max", "min"])
@pytest.mark.parametrize(("dim", "expected_dim"), [(1, -1), (-2, 0)])
def test_reduce_axis_legalization(monkeypatch, op, dim, expected_dim):
    captured = []

    def fake_reduce_with_clear(buffer, out, reduce_type, dim, clear, real_shape):
        captured.append((reduce_type, dim, clear, real_shape))
        return "ok"

    monkeypatch.setattr(reduce_ascend_lang, "_reduce_with_clear", fake_reduce_with_clear)

    reduce_fn = _get_reduce_fn(op)
    input_buffer = tir.decl_buffer((4, 8), "float32")
    output_buffer = tir.decl_buffer((4,), "float32")

    assert reduce_fn(input_buffer, output_buffer, dim=dim) == "ok"
    assert captured == [(f"reduce_{op}", expected_dim, True, None)]


@pytest.mark.parametrize("op", ["sum", "max", "min"])
@pytest.mark.parametrize("axis", [2, -3], ids=lambda axis: f"axis{axis}")
def test_reduce_invalid_axis_raises_value_error(op, axis):
    reduce_fn = _get_reduce_fn(op)
    input_buffer = tir.decl_buffer((4, 8), "float32")
    output_buffer = tir.decl_buffer((8,), "float32")
    with pytest.raises(ValueError):
        reduce_fn(input_buffer, output_buffer, dim=axis)


@pytest.mark.parametrize(
    ("input_shape", "real_shape", "dim", "out_shape"),
    [
        pytest.param((4, 8), (4, 4), -1, (8,), id="row-slice-flat-physical-layout"),
        pytest.param((4, 8), (4, 4), -1, (1, 8), id="row-slice-keepdim-physical-layout"),
        pytest.param((5, 8), (3, 4), 0, (1, 8), id="col-slice-keepdim-physical-layout"),
    ],
)
def test_reduce_slice_buffer_physical_output_shape_is_accepted(input_shape, real_shape, dim, out_shape):
    input_buffer = tir.decl_buffer(input_shape, "float32")
    output_buffer = tir.decl_buffer(out_shape, "float32")
    result = T.reduce_sum(input_buffer, output_buffer, dim=dim, real_shape=list(real_shape))
    assert isinstance(result, tir.Call)
    assert result.op.same_as(tir.op.Op.get("tl.ascend_reduce"))


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    elementwise_test_path = os.path.join(current_dir, "test_tilelang_ascend_language_elementwise.py")
    pytest.main(["--forked", elementwise_test_path])
