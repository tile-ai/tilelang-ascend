import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor


pytestmark = [pytest.mark.mode("Developer")]

M = 4
N = 1024
NUM = 4


@tilelang.jit(target="npuir")
def vec_for_add(dtype="float16"):
    block_size = 1

    @T.prim_func
    def vecForAdd(
        Input: T.Tensor((M, N), dtype),
        Output: T.Tensor((1, N), dtype),
    ):
        with T.Kernel(block_size, is_npu=True) as (cid, _):
            Add_result = T.alloc_shared([1, N], dtype=dtype)
            src = T.alloc_shared([1, N], dtype=dtype)

            value_zero = 0
            T.npuir_brc(value_zero, Add_result)
            for i in T.Pipelined(M):
                T.copy(Input[i, 0], src)
                T.npuir_add(src, Add_result, Add_result)

            T.copy(Add_result, Output)

    return vecForAdd


@tilelang.jit(target="npuir")
def vec_for_sub(dtype="float16"):
    block_size = 1

    @T.prim_func
    def vecForSub(
        Input: T.Tensor((M, N), dtype),
        Output: T.Tensor((1, N), dtype),
    ):
        with T.Kernel(block_size, is_npu=True) as (cid, _):
            Sub_result = T.alloc_shared([1, N], dtype=dtype)
            src = T.alloc_shared([1, N], dtype=dtype)

            value_zero = 0
            T.npuir_brc(value_zero, Sub_result)
            for i in T.Pipelined(M):
                T.copy(Input[i, 0], src)
                T.npuir_sub(Sub_result, src, Sub_result)

            T.copy(Sub_result, Output)

    return vecForSub


@tilelang.jit(target="npuir")
def vec_for_mul(dtype="float16"):
    block_size = 1

    @T.prim_func
    def vecForMul(
        Input: T.Tensor((M, N), dtype),
        Output: T.Tensor((1, N), dtype),
    ):
        with T.Kernel(block_size, is_npu=True) as (cid, _):
            Mul_result = T.alloc_shared([1, N], dtype=dtype)
            src = T.alloc_shared([1, N], dtype=dtype)

            value_zero = 1
            T.npuir_brc(value_zero, Mul_result)
            for i in T.Pipelined(M):
                T.copy(Input[i, 0], src)
                T.npuir_mul(Mul_result, src, Mul_result)

            T.copy(Mul_result, Output)

    return vecForMul


@tilelang.jit(target="npuir")
def vec_for_div(dtype="float16"):
    block_size = 1

    @T.prim_func
    def vecForDiv(
        Input: T.Tensor((M, N), dtype),
        Output: T.Tensor((1, N), dtype),
    ):
        with T.Kernel(block_size, is_npu=True) as (cid, _):
            Div_result = T.alloc_shared([1, N], dtype=dtype)
            src = T.alloc_shared([1, N], dtype=dtype)

            value_zero = 1
            T.npuir_brc(value_zero, Div_result)
            for i in T.Pipelined(M):
                T.copy(Input[i, 0], src)
                T.npuir_div(Div_result, src, Div_result)

            T.copy(Div_result, Output)

    return vecForDiv


@tilelang.jit(target="npuir")
def vec_for_exp(dtype="float16"):
    block_size = 1

    @T.prim_func
    def vecForExp(
        Input: T.Tensor((1, N), dtype),
        Output: T.Tensor((1, N), dtype),
    ):
        with T.Kernel(block_size, is_npu=True) as (cid, _):
            src = T.alloc_shared([1, N], dtype=dtype)
            T.copy(Input, src)
            for i in T.Pipelined(4):
                T.npuir_exp(src, src)
            T.copy(src, Output)

    return vecForExp


@tilelang.jit(target="npuir")
def vec_for_ln(dtype="float16"):
    block_size = 1

    @T.prim_func
    def vecForLn(
        Input: T.Tensor((1, N), dtype),
        Output: T.Tensor((1, N), dtype),
    ):
        with T.Kernel(block_size, is_npu=True) as (cid, _):
            src = T.alloc_shared([1, N], dtype=dtype)
            T.copy(Input, src)
            for i in T.Pipelined(2):
                T.npuir_ln(src, src)
            T.copy(src, Output)

    return vecForLn


@tilelang.jit(target="npuir")
def vec_for_sqrt(dtype="float16"):
    block_size = 1

    @T.prim_func
    def vecForSqrt(
        Input: T.Tensor((1, N), dtype),
        Output: T.Tensor((1, N), dtype),
    ):
        with T.Kernel(block_size, is_npu=True) as (cid, _):
            src = T.alloc_shared([1, N], dtype=dtype)
            T.copy(Input, src)
            for i in T.Pipelined(4):
                T.npuir_sqrt(src, src)
            T.copy(src, Output)

    return vecForSqrt


@tilelang.jit(target="npuir")
def vec_for_rsqrt(dtype="float16"):
    block_size = 1

    @T.prim_func
    def vecForRsqrt(
        Input: T.Tensor((1, N), dtype),
        Output: T.Tensor((1, N), dtype),
    ):
        with T.Kernel(block_size, is_npu=True) as (cid, _):
            src = T.alloc_shared([1, N], dtype=dtype)
            T.copy(Input, src)
            for i in T.Pipelined(4):
                T.npuir_rsqrt(src, src)
            T.copy(src, Output)

    return vecForRsqrt


@tilelang.jit(target="npuir")
def vec_for_rec(dtype="float16"):
    block_size = 1

    @T.prim_func
    def vecForRec(
        Input: T.Tensor((1, N), dtype),
        Output: T.Tensor((1, N), dtype),
    ):
        with T.Kernel(block_size, is_npu=True) as (cid, _):
            src = T.alloc_shared([1, N], dtype=dtype)
            T.copy(Input, src)
            for i in T.Pipelined(5):
                T.npuir_rec(src, src)
            T.copy(src, Output)

    return vecForRec


def npuir_not_float16(x):
    x_int16 = x.view(torch.int16)
    x_int16_not = torch.bitwise_not(x_int16)
    return x_int16_not.view(torch.float16)


@tilelang.jit(target="npuir")
def vec_for_not(dtype="float16"):
    block_size = 1

    @T.prim_func
    def vecForNot(
        Input: T.Tensor((1, N), dtype),
        Output: T.Tensor((1, N), dtype),
    ):
        with T.Kernel(block_size, is_npu=True) as (cid, _):
            src = T.alloc_shared([1, N], dtype=dtype)
            T.copy(Input, src)
            for i in T.Pipelined(5):
                T.npuir_not(src, src)
            T.copy(src, Output)

    return vecForNot


@tilelang.jit(target="npuir")
def vec_for_reduce(dtype="float32"):
    block_size = 1

    @T.prim_func
    def vecForReduce(
        Input: T.Tensor((NUM, M, N), dtype),
        Output: T.Tensor((M, 1), dtype),
    ):
        with T.Kernel(block_size, is_npu=True) as (cid, _):
            reduce_result = T.alloc_shared([M, 1], dtype=dtype)
            src = T.alloc_shared([M, N], dtype=dtype)

            value_zero = 0
            T.npuir_brc(value_zero, reduce_result)
            for i in T.Pipelined(NUM):
                T.copy(Input[i, :, :], src)
                T.npuir_reduce(src, reduce_result, 1, "sum", clear=False)

            T.copy(reduce_result, Output)

    return vecForReduce


@tilelang.jit(target="npuir")
def vec_for_abs(dtype="float16"):
    block_size = 1

    @T.prim_func
    def vecForAbs(
        Input: T.Tensor((1, N), dtype),
        Output: T.Tensor((1, N), dtype),
    ):
        with T.Kernel(block_size, is_npu=True) as (cid, _):
            src = T.alloc_shared([1, N], dtype=dtype)
            T.copy(Input, src)
            for i in T.Pipelined(5):
                T.npuir_abs(src, src)
            T.copy(src, Output)

    return vecForAbs


@tilelang.jit(target="npuir")
def vec_for_relu(dtype="float16"):
    block_size = 1

    @T.prim_func
    def vecForRelu(
        Input: T.Tensor((1, N), dtype),
        Output: T.Tensor((1, N), dtype),
    ):
        with T.Kernel(block_size, is_npu=True) as (cid, _):
            src = T.alloc_shared([1, N], dtype=dtype)
            T.copy(Input, src)
            for i in T.Pipelined(5):
                T.npuir_relu(src, src)
            T.copy(src, Output)

    return vecForRelu


@pytest.mark.op("add")
@pytest.mark.parametrize("dtype", ["float16"])
def test_for_add(dtype):
    input_tensor = gen_tensor((M, N), dtype, kind="randn")
    output_tensor = gen_tensor((1, N), dtype, kind="randn")
    vecForAdd = vec_for_add(dtype)
    vecForAdd(input_tensor, output_tensor)
    ref_output = torch.sum(input_tensor, dim=0, keepdim=True)
    assert_close(output_tensor.cpu(), ref_output.cpu(), dtype=dtype, rtol=1e-2, atol=1e-2)


@pytest.mark.op("sub")
@pytest.mark.parametrize("dtype", ["float16"])
def test_for_sub(dtype):
    input_tensor = gen_tensor((M, N), dtype, kind="randn")
    output_tensor = gen_tensor((1, N), dtype, kind="randn")
    vecForSub = vec_for_sub(dtype)
    vecForSub(input_tensor, output_tensor)
    ref_output = -torch.sum(input_tensor, dim=0, keepdim=True)
    assert_close(output_tensor.cpu(), ref_output.cpu(), dtype=dtype, rtol=1e-2, atol=1e-2)


@pytest.mark.op("mul")
@pytest.mark.parametrize("dtype", ["float16"])
def test_for_mul(dtype):
    input_tensor = gen_tensor((M, N), dtype, kind="randn")
    output_tensor = gen_tensor((1, N), dtype, kind="randn")
    vecForMul = vec_for_mul(dtype)
    vecForMul(input_tensor, output_tensor)
    ref_output = torch.prod(input_tensor, dim=0, keepdim=True)
    assert_close(output_tensor.cpu(), ref_output.cpu(), dtype=dtype, rtol=1e-2, atol=1e-2)


@pytest.mark.op("div")
@pytest.mark.parametrize("dtype", ["float16"])
def test_for_div(dtype):
    input_tensor = gen_tensor((M, N), dtype, kind="rand") * 1.0 + 1.0
    output_tensor = gen_tensor((1, N), dtype, kind="randn")
    vecForDiv = vec_for_div(dtype)
    vecForDiv(input_tensor, output_tensor)
    ref_output = 1 / torch.prod(input_tensor, dim=0, keepdim=True)
    assert_close(output_tensor.cpu(), ref_output.cpu(), dtype=dtype, rtol=1e-2, atol=1e-2)


@pytest.mark.op("exp")
@pytest.mark.parametrize("dtype", ["float16"])
def test_for_exp(dtype):
    input_tensor = gen_tensor((1, N), dtype, kind="randn")
    output_tensor = gen_tensor((1, N), dtype, kind="randn")
    vecForExp = vec_for_exp(dtype)
    vecForExp(input_tensor, output_tensor)
    ref_output = input_tensor.clone()
    for _ in range(4):
        ref_output.exp_()
    assert_close(output_tensor.cpu(), ref_output.cpu(), dtype=dtype, rtol=1e-2, atol=1e-2)


@pytest.mark.op("ln")
@pytest.mark.parametrize("dtype", ["float16"])
def test_for_ln(dtype):
    base_tensor = gen_tensor((1, N), dtype, kind="randn") + 3
    base_tensor = torch.clamp(base_tensor, min=1.0)
    input_tensor = torch.exp(base_tensor).npu()
    output_tensor = gen_tensor((1, N), dtype, kind="randn")
    vecForLn = vec_for_ln(dtype)
    vecForLn(input_tensor, output_tensor)
    ref_output = input_tensor.clone()
    for _ in range(2):
        ref_output.log_()
    assert_close(output_tensor.cpu(), ref_output.cpu(), dtype=dtype, rtol=1e-2, atol=1e-2)


@pytest.mark.op("sqrt")
@pytest.mark.parametrize("dtype", ["float16"])
def test_for_sqrt(dtype):
    input_tensor = torch.exp(gen_tensor((1, N), dtype, kind="randn")).npu()
    output_tensor = gen_tensor((1, N), dtype, kind="randn")
    vecForSqrt = vec_for_sqrt(dtype)
    vecForSqrt(input_tensor, output_tensor)
    ref_output = input_tensor.clone()
    for _ in range(4):
        ref_output.sqrt_()
    assert_close(output_tensor.cpu(), ref_output.cpu(), dtype=dtype, rtol=1e-2, atol=1e-2)


@pytest.mark.op("rsqrt")
@pytest.mark.parametrize("dtype", ["float16"])
def test_for_rsqrt(dtype):
    input_tensor = torch.exp(gen_tensor((1, N), dtype, kind="randn")).npu()
    output_tensor = gen_tensor((1, N), dtype, kind="randn")
    vecForRsqrt = vec_for_rsqrt(dtype)
    vecForRsqrt(input_tensor, output_tensor)
    ref_output = input_tensor.clone()
    for _ in range(4):
        ref_output.rsqrt_()
    assert_close(output_tensor.cpu(), ref_output.cpu(), dtype=dtype, rtol=1e-2, atol=1e-2)


@pytest.mark.op("rec")
@pytest.mark.parametrize("dtype", ["float16"])
def test_for_rec(dtype):
    input_tensor = gen_tensor((1, N), dtype, kind="randn")
    output_tensor = gen_tensor((1, N), dtype, kind="randn")
    vecForRec = vec_for_rec(dtype)
    vecForRec(input_tensor, output_tensor)
    ref_output = input_tensor.clone()
    for _ in range(5):
        ref_output.reciprocal_()
    assert_close(output_tensor.cpu(), ref_output.cpu(), dtype=dtype, rtol=1e-2, atol=1e-2)


@pytest.mark.op("not")
@pytest.mark.parametrize("dtype", ["float16"])
def test_for_not(dtype):
    input_tensor = gen_tensor((1, N), dtype, kind="randn")
    output_tensor = gen_tensor((1, N), dtype, kind="randn")
    vecForNot = vec_for_not(dtype)
    vecForNot(input_tensor, output_tensor)
    ref_output = input_tensor.clone()
    for _ in range(5):
        ref_output = npuir_not_float16(ref_output)
    assert_close(output_tensor.cpu(), ref_output.cpu(), dtype=dtype, rtol=1e-2, atol=1e-2)


@pytest.mark.op("reduce")
@pytest.mark.parametrize("dtype", ["float32"])
def test_for_reduce(dtype):
    input_tensor = gen_tensor((NUM, M, N), dtype, kind="randn")
    output_tensor = gen_tensor((M, 1), dtype, kind="randn")
    vecForReduce = vec_for_reduce(dtype)
    vecForReduce(input_tensor, output_tensor)
    ref_output = torch.sum(input_tensor, dim=2, keepdim=True).sum(dim=0)
    assert_close(output_tensor.cpu(), ref_output.cpu(), dtype=dtype, rtol=1e-2, atol=1e-2)


@pytest.mark.op("abs")
@pytest.mark.parametrize("dtype", ["float16"])
def test_for_abs(dtype):
    input_tensor = gen_tensor((1, N), dtype, kind="randn")
    output_tensor = gen_tensor((1, N), dtype, kind="randn")
    vecForAbs = vec_for_abs(dtype)
    vecForAbs(input_tensor, output_tensor)
    ref_output = input_tensor.clone()
    for _ in range(5):
        ref_output.abs_()
    assert_close(output_tensor.cpu(), ref_output.cpu(), dtype=dtype, rtol=1e-2, atol=1e-2)


@pytest.mark.op("relu")
@pytest.mark.parametrize("dtype", ["float16"])
def test_for_relu(dtype):
    input_tensor = gen_tensor((1, N), dtype, kind="randn")
    output_tensor = gen_tensor((1, N), dtype, kind="randn")
    vecForRelu = vec_for_relu(dtype)
    vecForRelu(input_tensor, output_tensor)
    ref_output = input_tensor.clone()
    for _ in range(5):
        ref_output.relu_()
    assert_close(output_tensor.cpu(), ref_output.cpu(), dtype=dtype, rtol=1e-2, atol=1e-2)
