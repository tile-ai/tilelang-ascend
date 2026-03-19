import argparse
import pytest
import torch
import tilelang

from testcommon import npuir_compile_to_bin

pytestmark = [pytest.mark.mode("Developer")]
import tilelang.language as T

tilelang.cache.clear_cache()


def sigmoid_kernel(M, N, dtype):
    BLOCK_SIZE = 1

    @T.prim_func
    def main(src: T.Tensor((M, N), dtype), dst: T.Tensor((M, N), dtype)):

        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            src_ub = T.alloc_ub((M, N), dtype)
            dst_ub = T.alloc_ub((M, N), dtype)

            T.copy(src, src_ub)
            T.npuir_sigmoid(src_ub, dst_ub)
            T.copy(dst_ub, dst)

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


def test_sigmoid():
    parser = argparse.ArgumentParser(description="NPU Sigmoid Kernel Test (Fixed)")
    parser.add_argument("--M", type=int, default=4, help="Size of dimension M")
    parser.add_argument("--N", type=int, default=4, help="Size of dimension N")
    parser.add_argument("--dtype", type=str, default="float16", help="Data type")

    main_args = parser.parse_args([])
    func = sigmoid_kernel(main_args.M, main_args.N, main_args.dtype)

    kernel = tilelang.engine.lower(func, target="npuir")
    result = npuir_compile_to_bin(kernel)
    assert result is not None and len(result) > 0, (
        "npuir compile failed or returned empty"
    )


if __name__ == "__main__":
    test_sigmoid()
