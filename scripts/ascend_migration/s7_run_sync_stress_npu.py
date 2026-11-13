#!/usr/bin/env python3
"""Validate S7 local events and real mode2 workspace handoff on A3."""

from pathlib import Path

from _common import base_report
from _npu_runner import run_cases, run_worker
from _s7_npu import (
    PARAMETERS,
    compare_pairs,
    configure_submission_environment,
    parser,
    preflight,
    run_case,
    validate,
)


def main():
    args = parser().parse_args()
    report = base_report("s7_run_sync_stress_npu", args, step="S7")
    invalid = validate(args, report)
    if invalid is not None:
        return invalid
    if args._case:
        spec = PARAMETERS.get(args._case, {})
        args.submission_environment = configure_submission_environment(spec)
        live_parameters = dict(spec)

        def execute(name, worker_args, torch):
            return run_case(name, worker_args, torch, metadata=live_parameters)

        return run_worker(
            args,
            execute,
            preflight_check=preflight,
            case_parameters=live_parameters,
            launch_multiplier=spec.get("streams", 1) * spec.get("batches", 1),
            sentinel_description="NaN output/canaries; finite changing GM workspace prefix",
        )
    report["verification_scope"] = (
        "Fixed constraint Vector schedule with legacy/a3 local generators; fixed explicit-C/V "
        "single-slot caller GM workspace with manual legacy or new a3 READY/RELEASE. Same "
        "compiled kernel per case; independent real streams hold target batches behind separate "
        "ExternalEvent start gates until both completion markers are observed pending. "
        "This establishes queued submissions, not hardware execution overlap. "
        "No mode0 global barrier, performance measurement or S8 mixed scheduling."
    )
    return run_cases(args, report, tuple(PARAMETERS), Path(__file__).resolve(), final_check=compare_pairs)


if __name__ == "__main__":
    raise SystemExit(main())
