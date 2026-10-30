#!/usr/bin/env python3
"""Validate S6 initialized scratch, alias views and reuse on one real A3 device."""

from pathlib import Path

from _common import base_report
from _npu_runner import run_cases, run_worker
from _s6_npu import PARAMETERS, compare_pairs, parser, preflight, run_case, validate


def main():
    args = parser().parse_args()
    report = base_report("s6_run_reuse_npu", args, step="S6")
    invalid = validate(args, report)
    if invalid is not None:
        return invalid
    if args._case:
        return run_worker(
            args,
            run_case,
            preflight_check=preflight,
            case_parameters=PARAMETERS.get(args._case, {}),
        )
    report["verification_scope"] = (
        "Paired legacy/constraint initialized scratch, alias and actual multi-version reuse; "
        "legacy serial baselines are explicit where its positive-stage planner rejects writes. "
        "Every launch changes finite scratch/input/tile labels and checks golden plus canaries. "
        "No profiling."
    )
    return run_cases(args, report, tuple(PARAMETERS), Path(__file__).resolve(), final_check=compare_pairs)


if __name__ == "__main__":
    raise SystemExit(main())
