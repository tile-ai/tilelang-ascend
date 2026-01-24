# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os
import argparse
import torch

import filecmp
import tilelang
import tilelang.language as T

tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--M", type=int, default=4, help="")
parser.add_argument("--N", type=int, default=4, help="")
parser.add_argument("--a", type=int, default=1, help="")
parser.add_argument("--b", type=int, default=16, help="")


def vec_insert(M, N, a, b):
    dtype = "float32"

    @T.prim_func
    def insert(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((a, b), dtype),
    ):
        with T.Kernel(M, is_npu=True) as (cid, _):
            A_ub = T.alloc_shared((4, 4), dtype)
            B_ub = T.alloc_shared((a, b), dtype)
            T.copy(B, B_ub)

            for i in T.serial(4):
                for j in T.serial(4):
                    A_ub[i, j] = B_ub[0, i * 4 + j]

            T.copy(A_ub, A)

    return insert


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


def test_tensor_insert():
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'
    main_args = parser.parse_args([])
    func = vec_insert(
        main_args.M,
        main_args.N,
        main_args.a,
        main_args.b,

    )
    kernel = tilelang.engine.lower(func, target='npuir')
    # print(kernel)

    curr_name = os.path.splitext(os.path.basename(__file__))[0][5:] + ".mlir"
    # Export to .mlir file
    output_file = './output/' + curr_name
    with open(output_file, 'w') as f:
        f.write(kernel)
    
    ref_file = "./mlir_files/" + curr_name
    # filecmp.cmp returns True if files are identical, False otherwise
    are_identical = filecmp.cmp(output_file, ref_file , shallow=False)
    # assertion for pytest
    assert are_identical, f"'{output_file}' and '{ref_file}' are not identical"


if __name__ == "__main__":
    run_test_insert()