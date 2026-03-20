import pytest

import torch

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}
@pytest.fixture(scope="session")
def clear_cache():
    """Clear tilelang cache before tests"""
    tilelang.cache.clear_cache()
    yield

@pytest.fixture
def setup_random_seed():
    """Set random seed for reproducibility"""
    torch.manual_seed(0)
    yield


def alloc_var(N, block_N, dtype="int32"):
    VEC_NUM = 2
    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
    ):
        with T.Kernel(N // block_N, is_npu=True) as (cid, vid):
            a_ub = T.alloc_shared(block_N // VEC_NUM, dtype)

            flag = T.alloc_var("bool", init=False)
            a = T.alloc_var(dtype, init=1)
            b = T.alloc_var(dtype, init=a)

            T.tile.fill(a_ub, 0.0)
            a_ub[0] = b
            flag = True
            if flag:
                a = 2
                a_ub[1] = a
            else:
                a_ub[1] = a

            flag = False
            if flag:
                a_ub[2] = a
            else:
                a += 1
                a_ub[2] = a
            T.copy(a_ub, A[cid * block_N + vid * block_N // VEC_NUM])
    return main

def run_alloc_var(N, block_N, target):
    func = alloc_var(N, block_N)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)
    code = func.get_kernel_source()

    torch.manual_seed(0)
    torch.npu.synchronize()

    print("init successful!")
    a = func()
    torch.set_printoptions(threshold=torch.inf)
    print(f"b:{a}")
    # print(code)
    if "flag =" in code and "a =" in code and "b =" in code:
        print("Kernel Output Match!")
    else:
        print("T.alloc_var failed")


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_alloc_var(target):
    run_alloc_var(32, 16, target=target)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-n", "8"])
