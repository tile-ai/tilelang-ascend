import os
import argparse
import torch
import tilelang
import tilelang.language as T

torch.npu.set_device(0)
tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--M", type=int, default=256, help="")
parser.add_argument("--N", type=int, default=256, help="")
parser.add_argument("--block_M", type=int, default=32, help="")
parser.add_argument("--block_N", type=int, default=32, help="")

dtype = "float16"

def arange_demo_dev(M, N, block_M, block_N):
    BLOCK_SIZE = 1
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            a = T.alloc_shared((block_M, block_N), dtype)
            for i in T.serial(T.ceildiv(m_num*n_num, BLOCK_SIZE)):
                block_id = i * BLOCK_SIZE + cid
                if block_id < m_num * n_num:
                    block_id_m = block_id // n_num
                    block_id_n = block_id % n_num
                    bx = block_id_m * block_M
                    by = block_id_n * block_N
                    T.npuir_arange(a, [bx,by], bx)
                    T.copy(a, A[bx : bx + block_M, by : by + block_N])

    return main

def arange_demo_exp(M, N, block_M, block_N):
    BLOCK_SIZE = 1
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            a = T.alloc_ub((block_M, block_N), dtype)
            for i in T.serial(T.ceildiv(m_num*n_num, BLOCK_SIZE)):
                block_id = i * BLOCK_SIZE + cid
                if block_id < m_num * n_num:
                    block_id_m = block_id // n_num
                    block_id_n = block_id % n_num
                    bx = block_id_m * block_M
                    by = block_id_n * block_N
                    T.npuir_arange(a, [bx,by], bx)
                    T.copy(a, A[bx : bx + block_M, by : by + block_N])

    return main

def generate_tensor(shape, dtype, clear=False):
    """generate tensor"""
    if clear:
        return torch.zeros(shape, dtype=eval("torch." + dtype))
    if dtype in ("float32", "float16", "bfloat16"):
        return torch.randn(size=shape, dtype=eval("torch." + dtype))
    if dtype in ("int32", "int64", "int16"):
        return torch.randint(low=0, high=2000, size=shape, dtype=eval("torch." + dtype))
    if dtype == "int8":
        return torch.randint(low=0, high=127, size=shape, dtype=eval("torch." + dtype))
    if dtype == "bool":
        return torch.randint(low=0, high=2, size=shape).bool()
    raise ValueError('Invalid parameter "dtype" is found : {}'.format(dtype))

def tile_arange(A, M, N, block_M, block_N):
    M, N = A.shape
    out = torch.empty_like(A)

    for by in range(0, M, block_M):
        for bx in range(0, N, block_N):
            tile = A[by:by+block_M, bx:bx+block_N]

            stride_y = by
            stride_x = bx
            offset = by

            m = torch.arange(block_M).view(block_M, 1)
            n = torch.arange(block_N).view(1, block_N)

            block = offset + m * stride_y + n * stride_x

            out[by:by+block_M, bx:bx+block_N] = block

    return out

def main(main_args):
    M = main_args.M
    N = main_args.N
    block_M = main_args.block_M
    block_N = main_args.block_N
    if os.environ['TILELANG_ASCEND_MODE'] == 'Dev':
        func = arange_demo_dev(M, N, block_M, block_N)
    else:
        func = arange_demo_exp(M, N, block_M, block_N)
    kernel = tilelang.compile(func, target="npuir")

    shape1 = (M, N)
    A = generate_tensor(shape1, dtype).npu()

    kernel(A)
    print("===A===")
    print(A)

    res = tile_arange(A, M, N, block_M, block_N)
    print("===res===")
    print(res)

    torch.testing.assert_close(
        A, res, rtol=1e-3, atol=1e-3
    )
    print("\033[92mArange demo passed!\033[0m")

if __name__ == "__main__":
    args = parser.parse_args()
    os.environ["TILELANG_ASCEND_MODE"] = "dev"
    main(args)
    os.environ['TILELANG_ASCEND_MODE'] = 'Expert'
    main(args)