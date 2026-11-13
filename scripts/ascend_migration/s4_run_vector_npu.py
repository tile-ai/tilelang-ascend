#!/usr/bin/env python3
"""Compare legacy/constraint Vector add on real A3, including aligned short tails."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import sys

from _common import ROOT, base_report, finish
from _npu_runner import run_cases, run_worker


PARAMETERS = {
    f"{shape}_{dtype}_{strategy}": (total, dtype, strategy)
    for shape, total in (("full", 192), ("tail", 160))
    for dtype in ("float16", "float32")
    for strategy in ("legacy", "constraint")
}
CASES = tuple(PARAMETERS)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.set_defaults(require_solver=True)
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
                str(ROOT / "build/ascend_migration_s4_native_build.json"),
            )
        ),
    )
    result.add_argument("--_case", choices=("preflight", *CASES), help=argparse.SUPPRESS)
    return result


def tensor_digest(tensor, torch) -> str:
    host = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(f"{host.dtype}:{tuple(host.shape)}:".encode())
    digest.update(bytes(host.view(torch.uint8).tolist()))
    return digest.hexdigest()


def parameters(name):
    if name == "preflight":
        return {}
    total, dtype, strategy = PARAMETERS[name]
    return {
        "total": total,
        "dtype": dtype,
        "strategy": strategy,
        "tile_elements": 64,
        "main_trips": total // 64,
        "tail_elements": total % 64,
        "num_stages": 1,
    }


def run_case(name, args, torch) -> dict:
    import tilelang
    from ascend_repeat import repeat_launch

    sys.path.insert(0, str(ROOT / "testing/python/ascend_migration"))
    from test_s4_kernels import make_vector_kernel, vector_pass_configs

    total, dtype, strategy = PARAMETERS[name]
    kernel = tilelang.compile(
        make_vector_kernel(total, dtype),
        out_idx=[],
        target="ascendc",
        platform="A3",
        pass_configs=vector_pass_configs(strategy),
    )
    torch_dtype = getattr(torch, dtype)
    # CPU generation makes independent strategy workers use identical input bits.
    generator = torch.Generator().manual_seed(args.seed)
    a_host = torch.randn(total, dtype=torch_dtype, generator=generator)
    b_host = torch.randn(total, dtype=torch_dtype, generator=generator)
    a, b = a_host.npu(), b_host.npu()
    output = torch.empty_like(a)
    expected = a_host + b_host
    tolerance = 1e-2 if dtype == "float16" else 1e-5
    repeat_launch(
        kernel,
        (a, b, output),
        (output,),
        lambda: torch.testing.assert_close(output.cpu(), expected, rtol=tolerance, atol=tolerance),
        args.repeat,
    )
    return {
        **parameters(name),
        "input_sha256": {"A": tensor_digest(a_host, torch), "B": tensor_digest(b_host, torch)},
        "output_sha256": tensor_digest(output, torch),
        "golden_sha256": tensor_digest(expected, torch),
    }


def compare_outputs(report):
    """Require actual result equality between independent strategy workers."""
    evidence = {case["name"]: case["evidence"] for case in report["cases"]}
    comparisons = []
    report["strategy_comparisons"] = comparisons
    for shape in ("full", "tail"):
        for dtype in ("float16", "float32"):
            names = [f"{shape}_{dtype}_{strategy}" for strategy in ("legacy", "constraint")]
            left, right = [evidence[name]["case_parameters"] for name in names]
            for name in names:
                observed = evidence[name]
                for key, value in parameters(name).items():
                    if observed["case_parameters"][key] != value:
                        raise AssertionError(f"Wrong actual case parameters for {name}: {key}")
                kernels = observed["kernels"]
                expected_strategy = PARAMETERS[name][2]
                if len(kernels) != 1 or kernels[0]["resolved_pass_configs"].get("tl.ascend_pipeline_planner") != expected_strategy:
                    raise AssertionError(f"Wrong resolved planner strategy for {name}")
            if left["input_sha256"] != right["input_sha256"]:
                raise AssertionError(f"Different actual inputs for {names}")
            if left["output_sha256"] != right["output_sha256"]:
                raise AssertionError(f"Different actual outputs for {names}")
            comparisons.append(
                {
                    "cases": names,
                    "status": "PASS",
                    "input_sha256": left["input_sha256"],
                    "output_sha256": left["output_sha256"],
                }
            )


def main() -> int:
    args = parser().parse_args()
    report = base_report("s4_run_vector_npu", args, step="S4")
    if sys.flags.optimize:
        return finish(args, report, "BLOCKED", "Python optimization disables mandatory assertions")
    if args.device < 0 or args.seed < 0 or args.repeat < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        return finish(args, report, "BLOCKED", "device/seed must be nonnegative; repeat/timeout positive")
    if args._case:
        return run_worker(args, run_case, case_parameters=parameters(args._case))
    report["verification_scope"] = (
        "Static single-stage/single-version Vector copy/add/copy with complete aligned DMA "
        "tiles. The short tail is a separate tile; no dynamic padding or S5 pipelining."
    )
    return run_cases(args, report, CASES, Path(__file__).resolve(), final_check=compare_outputs)


if __name__ == "__main__":
    raise SystemExit(main())
