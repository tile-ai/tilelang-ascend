#!/usr/bin/env python3
"""Measure compile, host dispatch and NPU Event span after checked warmups on A3."""

from pathlib import Path

from _common import base_report
from _npu_runner import run_cases, run_worker
from _s8_npu import BENCHMARK_CASES, parser, preflight, validate


def main():
    args = parser(__doc__, benchmark=True).parse_args()
    report = base_report("s8_benchmark_npu", args, step="S8")
    invalid = validate(args, report)
    if invalid is not None:
        return invalid
    if args._case == "preflight":
        return run_worker(args, lambda *args: None, preflight_check=preflight)
    from _s8_benchmark import compare_performance, run_measured_worker

    if args._case:
        return run_measured_worker(args)
    report["verification_scope"] = (
        "Same-input/shape/seed/flags timing with three checked warmups before each timed call. "
        "Compile wall/CPU/Bisheng, host call wall and NPU Event span are separate measurements; "
        "abstract issue-step scheduling costs are not measured times or throughput predictions."
    )
    return run_cases(args, report, BENCHMARK_CASES, Path(__file__).resolve(), final_check=compare_performance)


if __name__ == "__main__":
    raise SystemExit(main())
