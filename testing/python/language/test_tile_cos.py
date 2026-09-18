"""T.tile.cos test suite.

Registered against the shared unary-op framework in testing/python/base/.
Developer mode via DEFAULT_PASS_CONFIGS (auto sync + memory planning).

Notes:
- PTO backend does not support cos (compile-time error), so all pto params
  are skipped.
- cos(inf) = nan (!= inf), so the inf-input special-value check is skipped.
"""

import torch

import tilelang.language as T

from base import UnaryOpSpec, register_unary_op_tests

cos_spec = UnaryOpSpec(
    name="cos",
    tile_op=T.tile.cos,
    golden=torch.cos,
    supported_dtypes=["float16", "float32"],
    low_priority_dtypes=["float32"],
    unsupported_targets=["pto"],
)
cos_spec.skip_inf_special = True

classes = register_unary_op_tests(cos_spec)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run T.tile.cos tests.")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--target", default="ascendc", choices=["ascendc", "pto"])
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=1024)
    args = parser.parse_args()

    from base.unary_op import run_unary_op

    run_unary_op(
        cos_spec.kernel_tensor,
        args.M,
        args.N,
        128,
        128,
        args.dtype,
        args.target,
        golden_fn=torch.cos,
    )
