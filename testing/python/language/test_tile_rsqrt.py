"""T.tile.rsqrt test suite.

Registered against the shared unary-op framework in testing/python/base/.
Developer mode via DEFAULT_PASS_CONFIGS (auto sync + memory planning).

Known backend behavior (verified on A2, both targets):
- Rsqrt is a hardware approximation: max relative error ~3e-3 on both
  ascendc and pto (probe: max_rel=2.9e-3), so the suite uses rtol/atol 5e-3.
- rsqrt(0) = inf, rsqrt(inf) = 0 by IEEE semantics; the generic framework's
  inf-input check (asserts inf output) is skipped via ``skip_inf_special``.
"""

import torch

import tilelang.language as T

from base import UnaryOpSpec, register_unary_op_tests

rsqrt_spec = UnaryOpSpec(
    name="rsqrt",
    tile_op=T.tile.rsqrt,
    golden=torch.rsqrt,
    supported_dtypes=["float16", "float32"],
    low_priority_dtypes=["float32"],
    boundary_dtypes=["float16", "float32"],
    tol={"rtol": 5e-3, "atol": 5e-3},
)
rsqrt_spec.skip_inf_special = True

classes = register_unary_op_tests(rsqrt_spec)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run T.tile.rsqrt tests.")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--target", default="ascendc", choices=["ascendc", "pto"])
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=1024)
    args = parser.parse_args()

    from base.unary_op import run_unary_op

    run_unary_op(
        rsqrt_spec.kernel_tensor,
        args.M,
        args.N,
        128,
        128,
        args.dtype,
        args.target,
        golden_fn=torch.rsqrt,
        tol=rsqrt_spec.tol,
    )
