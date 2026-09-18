"""T.tile.leaky_relu test suite.

Registered against the shared unary-op framework in testing/python/base/
with a custom kernel factory (the API takes a scalar slope argument,
so the generic 2-buffer kernel does not apply).
Developer mode via DEFAULT_PASS_CONFIGS (auto sync + memory planning).

API: T.tile.leaky_relu(dst, src0, scalar_value)
     dst = src0 if src0 >= 0 else src0 * scalar_value
"""

import torch.nn.functional as F

import tilelang.language as T

from base import UnaryOpSpec, register_unary_op_tests

ALPHA = 0.1


def make_leaky_relu_kernel(M, N, block_M, block_N, dtype="float"):
    """Factory: leaky_relu(src, alpha) -> dst."""
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
            T.tile.leaky_relu(b_ub, a_ub, ALPHA)
            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def make_leaky_relu_1d_kernel(N, dtype="float"):
    """Factory: leaky_relu(src, alpha) -> dst on 1D buffers."""

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),  # type: ignore
        B: T.Tensor((N,), dtype),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((N,), dtype)
            b_ub = T.alloc_ub((N,), dtype)
            T.copy(A, a_ub)
            T.tile.leaky_relu(b_ub, a_ub, ALPHA)
            T.copy(b_ub, B)

    return main


def make_leaky_relu_slice_kernel(M, N, dtype="float"):
    """Factory: leaky_relu on the top half of a 2D buffer (BufferRegion)."""

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((M, N), dtype)
            b_ub = T.alloc_ub((M, N), dtype)
            T.copy(A, a_ub)
            T.tile.leaky_relu(b_ub[0 : M // 2, :], a_ub[0 : M // 2, :], ALPHA)
            T.copy(b_ub, B)

    return main


def make_leaky_relu_inplace_kernel(M, N, dtype="float"):
    """Factory: leaky_relu in-place on a single UB buffer (dst aliases src)."""

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((M, N), dtype)
            T.copy(A, a_ub)
            T.tile.leaky_relu(a_ub, a_ub, ALPHA)
            T.copy(a_ub, B)

    return main


leaky_relu_spec = UnaryOpSpec(
    name="leaky_relu",
    tile_op=T.tile.leaky_relu,
    golden=lambda a: F.leaky_relu(a, negative_slope=ALPHA),
    supported_dtypes=["float16", "float32"],
    low_priority_dtypes=["float32"],
    kernel_tensor=make_leaky_relu_kernel,
    kernel_1d=make_leaky_relu_1d_kernel,
    kernel_slice=make_leaky_relu_slice_kernel,
    kernel_inplace=make_leaky_relu_inplace_kernel,
)

classes = register_unary_op_tests(leaky_relu_spec)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run T.tile.leaky_relu tests.")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--target", default="ascendc", choices=["ascendc", "pto"])
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=1024)
    args = parser.parse_args()

    from base.unary_op import run_unary_op

    run_unary_op(
        leaky_relu_spec.kernel_tensor,
        args.M,
        args.N,
        128,
        128,
        args.dtype,
        args.target,
        golden_fn=leaky_relu_spec.golden,
    )
