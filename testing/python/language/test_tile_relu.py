"""T.tile.relu test suite.

Registered against the shared unary-op framework in testing/python/base/.
Developer mode via DEFAULT_PASS_CONFIGS (auto sync + memory planning).

dtype support: AscendC::Relu accepts only half/float (verified on A2, both
targets). int16/int32 compile with bisheng but fail with a __ubuf__ half
type error inside the CANN unary intrinsic, so they are not declared.
"""

import torch

import tilelang.language as T

from base import UnaryOpSpec, register_unary_op_tests

relu_spec = UnaryOpSpec(
    name="relu",
    tile_op=T.tile.relu,
    golden=torch.relu,
    supported_dtypes=["float16", "float32"],
    low_priority_dtypes=["float32"],
)

classes = register_unary_op_tests(relu_spec)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run T.tile.relu tests.")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--target", default="ascendc", choices=["ascendc", "pto"])
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=1024)
    args = parser.parse_args()

    from base.unary_op import run_unary_op

    run_unary_op(
        relu_spec.kernel_tensor,
        args.M,
        args.N,
        128,
        128,
        args.dtype,
        args.target,
        golden_fn=torch.relu,
    )
