#!/usr/bin/env python3
"""Run S1's existing numerical regressions on one verified A2/A3 device."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

from _common import base_report, finish
from _npu_runner import run_cases, run_worker


CASES = (
    "vector_abs",
    "gemm_c128",
    "cv_workspace",
    "persistent_full",
    "persistent_partial",
    "persistent_multi",
    "l1_tail_stages2",
    "l1_tail_parallel",
    "alias_trip0",
    "alias_trip1",
    "alias_trip3",
    "copy_elem_offset",
    "atomic_elem_offset",
    "tile_fill_elem_offset",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--platform", required=True, choices=("a2", "a3"))
    result.add_argument("--device", type=int, default=0, help="Logical NPU device index")
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--repeat", type=int, default=100)
    result.add_argument("--timeout", type=float, default=120, help="Seconds per isolated case")
    result.add_argument("--output", required=True, help="JSON result; artifacts are saved beside it")
    result.add_argument("--_case", choices=("preflight", *CASES), help=argparse.SUPPRESS)
    return result


def run_case(name: str, args, torch) -> None:
    """Reuse existing kernels and their golden assertions, including every launch."""
    if name == "vector_abs":
        from test_tilelang_ascend_language_elementwise import run_test_abs

        run_test_abs(256, 256, 128, 256, "float", "ascendc", repeat=args.repeat)
    elif name == "gemm_c128":
        from test_tilelang_ascend_language_gemm_v0 import _run_precision_case

        _run_precision_case("ascendc", 128, "C", repeat=args.repeat)
    elif name == "cv_workspace":
        from test_tilelang_ascend_language_copy_cv import _ub_to_l1_case, _ub_to_l1_kernel

        _ub_to_l1_case(_ub_to_l1_kernel, target="ascendc", repeat=args.repeat)
    elif name.startswith("persistent_"):
        from test_tilelang_ascend_language_l1_to_l0 import run_test_persistent_gemm

        tiles, cores = {
            "persistent_full": (8, 8),
            "persistent_partial": (4, 8),
            "persistent_multi": (9, 4),
        }[name]
        run_test_persistent_gemm(
            tiles * 128,
            128,
            128,
            128,
            128,
            64,
            cores,
            "float16",
            "float",
            "ascendc",
            repeat=args.repeat,
            seed=args.seed,
            launch_core_num=cores,
        )
    elif name == "l1_tail_stages2":
        from test_tilelang_ascend_language_gemm_v0_k_tail import test_gemm_v0_ktail_pipeline

        test_gemm_v0_ktail_pipeline(200, 2, repeat=args.repeat, seed=args.seed)
    elif name == "l1_tail_parallel":
        from test_tilelang_ascend_language_gemm_v0_k_tail import (
            test_gemm_v0_ktail_pipeline_parallel,
        )

        test_gemm_v0_ktail_pipeline_parallel(200, repeat=args.repeat, seed=args.seed)
    elif name.startswith("alias_trip"):
        from test_ascend_sync_insert_zero_trip import (
            _optional_loop_kernel,
            test_optional_loop_preserves_the_previous_alias_value,
        )

        kernel = _optional_loop_kernel(args.platform.upper())
        test_optional_loop_preserves_the_previous_alias_value(kernel, int(name[-1]), repeat=args.repeat)
    elif name == "copy_elem_offset":
        import tilelang
        from ascend_repeat import repeat_launch
        from test_ascend_migration_baseline import make_copy_offset_kernel

        program = make_copy_offset_kernel(src_start=8, dst_start=8)
        kernel = tilelang.compile(
            program,
            out_idx=[],
            target="ascendc",
            platform=args.platform.upper(),
            pass_configs={
                tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
                tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
            },
        )
        source = torch.arange(128, dtype=torch.float32, device="npu")
        output = torch.empty_like(source)
        expected = torch.full((128,), float("nan"), dtype=torch.float32)
        expected[24:88] = source.cpu()[40:104]
        repeat_launch(
            kernel,
            (source, output),
            (output,),
            lambda: torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0, equal_nan=True),
            args.repeat,
        )
    elif name == "atomic_elem_offset":
        import tilelang
        from ascend_repeat import repeat_launch
        from test_ascend_migration_baseline import make_atomic_offset_kernel

        kernel = tilelang.compile(
            make_atomic_offset_kernel(),
            out_idx=[],
            target="ascendc",
            platform=args.platform.upper(),
            pass_configs={
                tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
                tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
            },
        )
        source = torch.arange(128, dtype=torch.float32, device="npu")
        output = torch.empty_like(source)
        expected = torch.full((128,), -99.0, dtype=torch.float32)
        expected[16:80] = 7.0 + 2 * source.cpu()[16:80]

        def reset_atomic_output():
            output.fill_(-99.0)
            output[16:80].fill_(7.0)

        repeat_launch(
            kernel,
            (source, output),
            (output,),
            lambda: torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0),
            args.repeat,
            reset=reset_atomic_output,
        )
    elif name == "tile_fill_elem_offset":
        import tilelang
        from ascend_repeat import repeat_launch
        from test_ascend_migration_baseline import make_tile_fill_offset_kernel

        kernel = tilelang.compile(
            make_tile_fill_offset_kernel(region_start=8),
            out_idx=[],
            target="ascendc",
            platform=args.platform.upper(),
            pass_configs={
                tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
                tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
            },
        )
        output = torch.empty((128,), dtype=torch.float32, device="npu")
        expected = torch.full((128,), -99.0, dtype=torch.float32)
        expected[24:88] = 1.0
        repeat_launch(
            kernel,
            (output,),
            (output,),
            lambda: torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0),
            args.repeat,
        )


def worker(args) -> int:
    sentinel = None
    if args._case == "atomic_elem_offset":
        sentinel = "-99 outside atomic destination; destination reset to 7 each launch"
    return run_worker(args, run_case, sentinel_description=sentinel)


def main() -> int:
    args = parser().parse_args()
    report = base_report("s1_run_baseline_npu", args)
    if sys.flags.optimize:
        return finish(args, report, "BLOCKED", "Python optimization disables mandatory assertions")
    if args.device < 0 or args.seed < 0 or args.repeat < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        return finish(
            args,
            report,
            "BLOCKED",
            "device/seed must be nonnegative; repeat/timeout must be positive",
        )
    if args._case:
        return worker(args)
    return run_cases(args, report, CASES, Path(__file__).resolve())


if __name__ == "__main__":
    raise SystemExit(main())
