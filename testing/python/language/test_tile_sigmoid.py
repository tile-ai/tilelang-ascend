"""T.tile.sigmoid test suite.

Registered against the shared unary-op framework in testing/python/base/.
Developer mode via DEFAULT_PASS_CONFIGS (auto sync + memory planning).

Known backend behavior (verified on A2, both targets):
- sigmoid(inf) = 1, sigmoid(-inf) = 0 by IEEE semantics (not inf), so the
  generic framework's inf-input check (asserts inf output) is skipped via
  ``skip_inf_special``.
"""

import torch

import tilelang.language as T

from base import UnaryOpSpec, register_unary_op_tests

sigmoid_spec = UnaryOpSpec(
    name="sigmoid",
    tile_op=T.tile.sigmoid,
    golden=torch.sigmoid,
    supported_dtypes=["float16", "float32"],
    low_priority_dtypes=["float32"],
)
sigmoid_spec.skip_inf_special = True
sigmoid_spec.inplace_xfail_targets = ["pto"]

classes = register_unary_op_tests(sigmoid_spec)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run T.tile.sigmoid tests.")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--target", default="ascendc", choices=["ascendc", "pto"])
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=1024)
    args = parser.parse_args()

    from base.unary_op import run_unary_op

    run_unary_op(
        sigmoid_spec.kernel_tensor,
        args.M,
        args.N,
        128,
        128,
        args.dtype,
        args.target,
        golden_fn=torch.sigmoid,
    )
