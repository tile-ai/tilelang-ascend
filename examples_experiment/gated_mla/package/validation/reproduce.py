"""Portable r17 direct/simulator reproduction using the caller's current runtime."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

from bounded_process import run_owned_process


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package-root", type=Path, required=True, help="Directory containing gmla/")
    p.add_argument("--output", type=Path, required=True, help="New evidence directory")
    p.add_argument("--case", choices=("decode", "prefill", "all"), default="prefill")
    p.add_argument("--target", choices=("pto", "ascendc", "all"), default="pto")
    p.add_argument("--dtype", choices=("float16", "bfloat16", "all"), default="float16")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--soc-version", default="Ascend910B3")
    p.add_argument("--direct-only", action="store_true")
    args = p.parse_args()
    args.package_root = args.package_root.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    tools = Path(__file__).resolve().parent
    cases = ("decode", "prefill") if args.case == "all" else (args.case,)
    targets = ("pto", "ascendc") if args.target == "all" else (args.target,)
    dtypes = ("float16", "bfloat16") if args.dtype == "all" else (args.dtype,)
    report = {
        "status": "running",
        "source_revision": "r1.7 public route",
        "steps": [],
        "python": sys.executable,
        "package_root": str(args.package_root),
        "sdk_timeout_minutes": 15,
        "outer_simulator_timeout_seconds": 900,
        "runtime_policy": "prepare captures this host's own source/native/cache hashes; no author-specific binary hashes required",
        "tool_sources": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in tools.glob("*.py")},
    }

    def save(**extra):
        report.update(extra)
        (args.output / "reproduction.json").write_text(json.dumps(report, indent=2) + "\n")

    env = dict(os.environ, TL_PLATFORM="A2", TL_RUN_MODE="npu", TILELANG_CLEAR_CACHE="0")
    env.pop("ASCEND_RT_VISIBLE_DEVICES", None)

    def run(label, command, timeout):
        save(current=label, command=[str(x) for x in command])
        started = time.monotonic()
        with (args.output / (label + ".log")).open("w") as log:
            code = run_owned_process(
                [str(x) for x in command],
                cwd=args.output,
                env=env,
                stream=log,
                timeout=timeout,
                process_record=args.output / (label + "_processes.json"),
                started=lambda pid: save(child_pid=pid),
            )
        report["steps"].append({"step": label, "exit_code": code, "wall_seconds": time.monotonic() - started})
        save(child_pid=None)
        if code:
            raise RuntimeError(f"{label} failed; inspect its preserved log before any retry")

    try:
        for target in targets:
            for case in cases:
                for dtype in dtypes:
                    key = f"{case}_{target}_{dtype}"
                    out = args.output / key
                    common = [
                        "--package-root",
                        args.package_root,
                        "--case",
                        case,
                        "--target",
                        target,
                        "--dtype",
                        dtype,
                        "--device",
                        args.device,
                        "--cache-dir",
                        out / "cache",
                    ]
                    run(
                        key + "_prepare",
                        [sys.executable, tools / "release_entry.py", "--mode", "prepare", *common, "--output", out / "prepared"],
                        600,
                    )
                    manifest = json.loads((out / "prepared/entry.json").read_text())
                    call = [sys.executable, tools / "release_entry.py", "--mode", "run", *common, "--prepared-dir", out / "prepared"]
                    run(key + "_direct", [*call, "--output", out / "smoke"], 300)
                    if args.direct_only:
                        continue
                    run(
                        key + "_simulator",
                        [
                            "msprof",
                            "op",
                            "simulator",
                            "--kernel-name=main*kernel*",  # Literal argv; no shell glob expansion.
                            "--launch-count=" + str(manifest["expected_physical_launches"]),
                            "--soc-version=" + args.soc_version,
                            "--timeout=15",
                            "--output=" + str(out / "simulator"),
                            *call,
                            "--output",
                            out / "entry",
                        ],
                        900,
                    )
                    run(
                        key + "_audit",
                        [
                            sys.executable,
                            tools / "audit_simulator.py",
                            "--entry-dir",
                            out / "entry",
                            "--log",
                            args.output / (key + "_simulator.log"),
                            "--profile-dir",
                            out / "simulator",
                            "--output",
                            out / "simulator_audit.json",
                        ],
                        60,
                    )
        save(status="passed", current=None, child_pid=None)
    except BaseException as error:
        save(status="failed", error=repr(error))
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
