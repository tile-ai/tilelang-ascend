"""T.tile.reciprocal test suite.

Registered against the shared unary-op framework in testing/python/base/.
Developer mode via DEFAULT_PASS_CONFIGS (auto sync + memory planning).

Known backend differences (verified on A2, both targets):
- AscendC Reciprocal is a hardware approximation: max relative error ~3e-3
  (probe: max_rel=2.9e-3 on float16/float32), so the suite uses rtol/atol 5e-3.
- PTO Reciprocal is implemented via TDIVS(1, src) and is exact, but does not
  support in-place (dst == src0) operation: the result is wrong (all ones).
  The in-place case is therefore expected-fail on pto.

The generic framework's inf-input check asserts an inf output, but
reciprocal(inf) = 0 by IEEE semantics; reciprocal spec sets
``skip_inf_special`` so that case is skipped while the nan-input case runs.
"""

import pytest
import torch

import tilelang
import tilelang.language as T

from base import DEFAULT_PASS_CONFIGS
from base import UnaryOpSpec, register_unary_op_tests
from base.common import assert_close_npu

reciprocal_spec = UnaryOpSpec(
    name="reciprocal",
    tile_op=T.tile.reciprocal,
    golden=torch.reciprocal,
    supported_dtypes=["float16", "float32"],
    low_priority_dtypes=["float32"],
    boundary_dtypes=["float16", "float32"],
    tol={"rtol": 5e-3, "atol": 5e-3},
)
reciprocal_spec.skip_inf_special = True

classes = register_unary_op_tests(reciprocal_spec)


@pytest.mark.usefixtures("disable_tilelang_cache", "random_seed")
class TestTileReciprocalInplace:
    @pytest.mark.l2
    @pytest.mark.low_priority
    @pytest.mark.parametrize(
        "target", ["ascendc", pytest.param("pto", marks=pytest.mark.xfail(strict=False, reason="pto Reciprocal in-place result wrong"))]
    )
    def test_inplace(self, target):
        """dst aliases src0 on a single UB buffer (in-place)."""

        @T.prim_func
        def inplace_kernel(
            A: T.Tensor((64, 64), "float16"),
            B: T.Tensor((64, 64), "float16"),
        ):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                a_ub = T.alloc_ub((64, 64), "float16")
                T.copy(A, a_ub)
                T.tile.reciprocal(a_ub, a_ub)
                T.copy(a_ub, B)

        compiled = tilelang.compile(inplace_kernel, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target)
        a = torch.randn(64, 64, dtype=torch.float16, device="npu").abs() + 0.5
        b = compiled(a)
        torch.npu.synchronize()
        assert_close_npu(b, torch.reciprocal(a), "float16", **(reciprocal_spec.tol or {}))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run T.tile.reciprocal tests.")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--target", default="ascendc", choices=["ascendc", "pto"])
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=1024)
    args = parser.parse_args()

    from base.unary_op import run_unary_op

    run_unary_op(
        reciprocal_spec.kernel_tensor,
        args.M,
        args.N,
        128,
        128,
        args.dtype,
        args.target,
        golden_fn=torch.reciprocal,
        tol=reciprocal_spec.tol,
    )
