#!/usr/bin/env python3
"""Validate ordinary A3 Vector pipelines, or calibrate a legacy effective-cost model."""

from pathlib import Path

from _common import base_report, finish
from _npu_runner import run_cases, run_worker
from _s5_device_cases import compare_pairs, run_case as vector_case
from _s5_npu import check_identity, parser, validate_arguments


FIXTURES = {
    "zero": {"trips": 0, "num_stages": 3},
    "one": {"trips": 1, "num_stages": 3},
    "short": {"trips": 2, "num_stages": 3},
    "steady1": {"trips": 8, "num_stages": 1, "required_actual_versions": 1},
    "steady2": {"trips": 8, "num_stages": 2, "required_actual_versions": 2},
    "steady3": {"trips": 8, "num_stages": 3, "required_actual_versions": 3},
    "tail_f16": {"trips": 2, "num_stages": 2, "tail": 32, "dtype": "float16"},
    "tile32_f16": {"trips": 8, "num_stages": 2, "tile": 32, "dtype": "float16"},
}
PARAMETERS = {
    f"{name}_{strategy}": {"dtype": "float32", "tile": 64, "tail": 0, **spec, "strategy": strategy}
    for name, spec in FIXTURES.items()
    for strategy in ("legacy", "constraint")
}
PARAMETERS["zero_legacy"].update(num_stages=0, legacy_zero_baseline=True, role="serial_zero_baseline_with_live_anchor")
PARAMETERS["steady3_constraint"].update(manual_stages=(0, 0, 3, 3), enable_offset=True)
CASES = (*PARAMETERS, "l1_tail_stages2", "l1_tail_parallel")


def run_case(name, args, torch):
    if name.startswith("l1_"):
        from s1_run_baseline_npu import run_case as legacy_case

        legacy_case(name, args, torch)
        return {"role": "legacy_L1_compatibility", "case": name, "k_tail": 200}
    return vector_case(PARAMETERS[name], args, torch)


def final_check(report, args=None):
    compare_pairs(report, PARAMETERS)
    if args is not None:
        from _s5_performance import collect_performance

        collect_performance(report, args, Path(__file__).resolve(), ("steady2", "tail_f16"))


def main():
    args = parser(__doc__, CASES, calibration=True).parse_args()
    report = base_report("s5_run_pipeline_npu", args, step="S5")
    error = validate_arguments(args, report)
    if error is not None:
        return error
    if args.calibrate_profile is not None:
        from _s5_calibration_matrix import run_calibration

        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        return run_calibration(args, report)
    if args._case:
        if args._measure:
            if args._case not in PARAMETERS:
                return finish(args, report, "BLOCKED", "Timing is limited to the Vector fixtures")
            from _s5_performance import run_measured_worker

            return run_measured_worker(args, PARAMETERS[args._case])
        return run_worker(
            args,
            run_case,
            case_parameters=PARAMETERS.get(args._case, {}),
            preflight_check=check_identity,
        )
    report["verification_scope"] = (
        "Same-input legacy/constraint ordinary Vector pipelines; 0/1/short/steady trips, "
        "actual physical versions and aligned tail; two separate legacy L1 regressions."
    )
    return run_cases(
        args,
        report,
        CASES,
        Path(__file__).resolve(),
        final_check=lambda result: final_check(result, args),
    )


if __name__ == "__main__":
    raise SystemExit(main())
