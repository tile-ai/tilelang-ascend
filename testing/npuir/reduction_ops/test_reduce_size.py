# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import torch
import os
import tilelang
import tilelang.language as T

torch.npu.set_device(0)
tilelang.cache.clear_cache()

@tilelang.jit(target="npuir")
def slice_reduce(block_M, block_N, dtype = "float16"):
    M = T.symbolic("M")
    N = T.symbolic("N")
    BLOCK_SIZE = 1
    @T.prim_func
    def reduce(
        Input: T.Tensor((M, N), dtype),
        Output: T.Tensor((1, N), dtype)
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            src = T.alloc_shared([block_M, block_N], dtype=dtype)
            dst = T.alloc_shared([1, block_N], dtype=dtype)

            for i in T.serial(T.ceildiv(N, block_N)):
                offset_n = i * block_N
                remain_n = T.min(block_N, N - offset_n)
                value_zero = 0
                T.npuir_brc(value_zero, dst)
                for j in T.serial(T.ceildiv(M, block_M)):
                    offset_m = j * block_M
                    remain_m = T.min(block_M, M - offset_m)
                    T.copy(Input[offset_m : offset_m + remain_m, offset_n : offset_n + remain_n], src)
                    T.npuir_reduce(src, dst, dims=0, reduce_mode="sum", size=[remain_m, remain_n], clear=False)
                T.copy(dst, Output[0 : 1, offset_n : offset_n + remain_n])

    return reduce

def test_slice_add():
    kernel = slice_reduce(32, 32)

    # case 1
    M, N = 17, 256
    input = torch.randn([M, N], dtype=torch.float16).npu()
    output = torch.randn([1, N], dtype=torch.float16).npu()
    kernel(input, output)
    ref_output = torch.sum(input, dim=0, keepdim=True)

    print("output")
    print(output)
    print("ref_output")
    print(ref_output)
    torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=1e-2)

    # case 2
    M, N = 39, 466
    input = torch.randn([M, N], dtype=torch.float16).npu()
    output = torch.randn([1, N], dtype=torch.float16).npu()
    kernel(input, output)
    ref_output = torch.sum(input, dim=0, keepdim=True)

    print("output")
    print(output)
    print("ref_output")
    print(ref_output)
    torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=1e-2)

    # case 3
    M, N = 77, 283
    input = torch.randn([M, N], dtype=torch.float16).npu()
    output = torch.randn([1, N], dtype=torch.float16).npu()
    kernel(input, output)
    ref_output = torch.sum(input, dim=0, keepdim=True)

    print("output")
    print(output)
    print("ref_output")
    print(ref_output)
    torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")

if __name__ == "__main__":
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'
    test_slice_add()