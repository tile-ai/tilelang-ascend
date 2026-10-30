#!/usr/bin/env python3
"""Check S8 GEMM, row reduction, mixed C/V and separate old Expert workloads on A3."""

from pathlib import Path

from _common import base_report
from _npu_runner import run_cases, run_worker
from _s8_npu import PARAMETERS, WORKLOAD_CASES, compare_pairs, parser, preflight, run_case, validate


def main():
    args = parser(__doc__).parse_args()
    report = base_report("s8_run_workloads_npu", args, step="S8")
    invalid = validate(args, report)
    if invalid is not None:
        return invalid
    if args._case:
        live = dict(PARAMETERS.get(args._case, {}))

        def execute(name, worker_args, torch):
            return run_case(name, worker_args, torch, metadata=live)

        return run_worker(args, execute, case_parameters=live, preflight_check=preflight)
    report["verification_scope"] = (
        "New finite-domain auto scheduler with genuine UB/L1 versions and sole synchronization "
        "ownership; separate legacy references and old Expert regressions. Caller GM workspace "
        "remains N1, with every output/canary and final workspace checked on each launch."
    )
    return run_cases(args, report, WORKLOAD_CASES, Path(__file__).resolve(), final_check=compare_pairs)


if __name__ == "__main__":
    raise SystemExit(main())
