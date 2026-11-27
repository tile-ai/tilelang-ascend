#!/usr/bin/env python3
"""Validate A3 Persistent wave pipelines with final-IR coverage and numerical canaries."""

from pathlib import Path

from _common import base_report
from _npu_runner import run_cases, run_worker
from _s5_device_cases import compare_pairs, run_case as vector_case
from _s5_npu import check_identity, parser, validate_arguments


FIXTURES = {
    "single": {"domain": (1, 1), "cores": 1, "group_size": 1, "num_stages": 2},
    "partial_f16": {
        "domain": (1, 3),
        "cores": 8,
        "group_size": 2,
        "num_stages": 3,
        "dtype": "float16",
        "tile": 32,
    },
    "group_tail": {"domain": (3, 10), "cores": 8, "group_size": 4, "num_stages": 2},
    "multi_tail": {"domain": (9, 1), "cores": 4, "group_size": 1, "num_stages": 3},
}
PARAMETERS = {
    f"{name}_{strategy}": {
        "dtype": "float32",
        "tile": 64,
        **spec,
        "strategy": strategy,
        "num_stages": 0 if strategy == "legacy" else spec["num_stages"],
        "role": "serial_baseline" if strategy == "legacy" else "constraint_wave_pipeline",
    }
    for name, spec in FIXTURES.items()
    for strategy in ("legacy", "constraint")
}
CASES = tuple(PARAMETERS)


def run_case(name, args, torch):
    return vector_case(PARAMETERS[name], args, torch)


def final_check(report, args=None):
    compare_pairs(report, PARAMETERS)
    if args is not None:
        from _s5_performance import collect_performance

        collect_performance(report, args, Path(__file__).resolve(), ("group_tail", "partial_f16"))


def main():
    args = parser(__doc__, CASES).parse_args()
    report = base_report("s5_run_persistent_pipeline_npu", args, step="S5")
    error = validate_arguments(args, report)
    if error is not None:
        return error
    if args._case:
        if args._measure:
            from _s5_performance import run_measured_worker

            return run_measured_worker(args, PARAMETERS[args._case])
        return run_worker(
            args,
            run_case,
            case_parameters=PARAMETERS.get(args._case, {}),
            preflight_check=check_identity,
        )
    report["verification_scope"] = (
        "Same-input legacy/constraint Persistent wave pipelines; finite inputs cover padded "
        "and overshoot slots; NaN output canaries and final-IR exact-once GM-store trace."
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
