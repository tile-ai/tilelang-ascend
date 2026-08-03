import tilelang
import tilelang.language as T


def scalar_sqrt_rsqrt(length=8):

    @T.prim_func
    def main(
            input_tensor: T.Tensor((length,), "float32"),
            sqrt_output: T.Tensor((length,), "float32"),
            rsqrt_output: T.Tensor((length,), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (_cid, _vid):
            input_ub = T.alloc_ub((length,), "float32")
            sqrt_ub = T.alloc_ub((length,), "float32")
            rsqrt_ub = T.alloc_ub((length,), "float32")
            T.copy(input_tensor, input_ub)
            for i in T.serial(length):
                value = input_ub[i]
                sqrt_ub[i] = T.sqrt(value)
                rsqrt_ub[i] = T.rsqrt(value + 1.0)
            T.copy(sqrt_ub, sqrt_output)
            T.copy(rsqrt_ub, rsqrt_output)

    return main


def test_scalar_sqrt_rsqrt_codegen():
    artifact = tilelang.lower(scalar_sqrt_rsqrt(), target="ascendc")
    source = artifact.kernel_source
    assert source.count("sqrt(") >= 2
    assert "1.0f / sqrt(" in source
