#!/usr/bin/env python3
"""Validate serial Persistent domains on real A3 hardware; wave pipelining is S5."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys

from _common import ROOT, base_report, finish
from _npu_runner import run_cases, run_worker


STATIC_CASES = {
    "persistent_full": (2, 4, 8, 2),
    "persistent_partial": (2, 2, 8, 2),
    "persistent_multi": (3, 3, 4, 8),
    "group_tail_partial": (1, 5, 8, 2),
    "group_tail_multi": (3, 10, 8, 4),
    "group_tail_one_core": (1, 5, 1, 2),
}
CASES = (*STATIC_CASES, "persistent_symbolic", "persistent_single_stage", "stage_zero_vector")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--platform", choices=("a3",), required=True)
    result.add_argument("--device", type=int, default=0)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--repeat", type=int, default=100)
    result.add_argument("--timeout", type=float, default=120)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument(
        "--build-manifest",
        type=Path,
        default=Path(
            os.environ.get(
                "TL_ASCEND_MIGRATION_BUILD_MANIFEST",
                str(ROOT / "build/ascend_migration_s3_native_build.json"),
            )
        ),
    )
    result.add_argument("--_case", choices=("preflight", *CASES), help=argparse.SUPPRESS)
    return result


def parameters(name: str) -> dict:
    if name in STATIC_CASES or name == "persistent_single_stage":
        rows, cols, cores, group = STATIC_CASES.get(name, STATIC_CASES["persistent_full"])
        return {
            "domain": [rows, cols],
            "cores": cores,
            "group_size": group,
            "num_stages": 1 if name == "persistent_single_stage" else 0,
            "tile_shape": [128, 128],
            "K": 128,
        }
    if name == "persistent_symbolic":
        return {
            "domain": ["ceildiv(M, 128)", 10],
            "cores": 8,
            "group_size": 4,
            "num_stages": 0,
            "tile_shape": [128, 128],
            "K": 128,
            "positive_bindings": [{"M": 128}, {"M": 384}],
        }
    if name == "stage_zero_vector":
        return {"operation": "A+B", "numel": 64, "task_stage": 0, "owner": "V"}
    return {}


def run_case(name: str, args, torch) -> dict:
    from test_tilelang_ascend_language_l1_to_l0 import (
        CUBE_PASS_CONFIGS,
        check_persistent_gemm,
        persistent_gemm,
        run_test_persistent_gemm,
    )

    if name in STATIC_CASES or name == "persistent_single_stage":
        rows, cols, cores, group = STATIC_CASES.get(name, STATIC_CASES["persistent_full"])
        run_test_persistent_gemm(
            rows * 128,
            cols * 128,
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
            group_size=group,
            num_stages=1 if name == "persistent_single_stage" else 0,
        )
    elif name == "persistent_symbolic":
        import tilelang
        from tilelang import language as T

        m = T.symbolic("s3_m", "int32")
        program = persistent_gemm(
            m,
            1280,
            128,
            128,
            128,
            64,
            8,
            "float16",
            "float",
            launch_core_num=8,
            group_size=4,
            num_stages=0,
        )
        kernel = tilelang.compile(
            program,
            out_idx=[],
            target="ascendc",
            platform="A3",
            pass_configs=CUBE_PASS_CONFIGS,
        )
        # Reuse one dynamically shaped executable across two positive bindings.
        # Each binding keeps the existing golden and output poisoning per launch.
        for actual_m in (128, 384):
            check_persistent_gemm(
                kernel,
                actual_m,
                1280,
                128,
                "float16",
                repeat=args.repeat,
                seed=args.seed,
            )
    elif name == "stage_zero_vector":
        import tilelang
        from ascend_repeat import repeat_launch

        sys.path.insert(0, str(ROOT / "testing/python/ascend_migration"))
        from test_s3_schedule_hints import make_task_stage_vector_kernel

        kernel = tilelang.compile(
            make_task_stage_vector_kernel(marked=True),
            out_idx=[],
            target="ascendc",
            platform="A3",
            pass_configs={
                **CUBE_PASS_CONFIGS,
                tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC_VS: False,
            },
        )
        a = torch.randn((64,), dtype=torch.float32, device="npu")
        b = torch.randn_like(a)
        output = torch.empty_like(a)
        expected = a + b
        repeat_launch(
            kernel,
            (a, b, output),
            (output,),
            lambda: torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5),
            args.repeat,
        )
    else:
        raise ValueError(f"Unknown S3 NPU case: {name}")
    return parameters(name)


def main() -> int:
    args = parser().parse_args()
    report = base_report("s3_run_loop_npu", args, step="S3")
    if sys.flags.optimize:
        return finish(args, report, "BLOCKED", "Python optimization disables mandatory assertions")
    if args.device < 0 or args.seed < 0 or args.repeat < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        return finish(args, report, "BLOCKED", "device/seed must be nonnegative; repeat/timeout positive")
    if args._case:
        return run_worker(
            args,
            run_case,
            launch_multiplier=2 if args._case == "persistent_symbolic" else 1,
            case_parameters=parameters(args._case),
        )
    report["verification_scope"] = (
        "Serial Persistent domains and positive bindings, plus the explicitly allowed "
        "guardless single-full-wave num_stages=1 path and a consumed Stage(0)/Task Vector call. "
        "This does not validate cross-wave pipelining or S5 multi-version semantics."
    )
    return run_cases(args, report, CASES, Path(__file__).resolve())


if __name__ == "__main__":
    raise SystemExit(main())
