"""T.tile.silu test suite.

Registered against the shared unary-op framework in testing/python/base/.
Developer mode via DEFAULT_PASS_CONFIGS (auto sync + memory planning).
"""

import torch.nn.functional as F

import tilelang.language as T

from base import UnaryOpSpec, register_unary_op_tests

silu_spec = UnaryOpSpec(
    name="silu",
    tile_op=T.tile.silu,
    golden=F.silu,
    supported_dtypes=["float16", "float32"],
    low_priority_dtypes=["float32"],
)
silu_spec.inplace_xfail_targets = ["ascendc", "pto"]

classes = register_unary_op_tests(silu_spec)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run T.tile.silu tests.")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--target", default="ascendc", choices=["ascendc", "pto"])
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=1024)
    args = parser.parse_args()

    from base.unary_op import run_unary_op

    run_unary_op(
        silu_spec.kernel_tensor,
        args.M,
        args.N,
        128,
        128,
        args.dtype,
        args.target,
        golden_fn=F.silu,
    )
