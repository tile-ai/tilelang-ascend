"""Shared native/device identity, cold compilation and isolated NPU execution.

Runtime imports happen only inside explicit workers; importing this helper is
standard-library-only. Step-specific scripts supply cases and golden checks.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

from _common import ROOT, finish, run_process


def digest(path: str | Path) -> str:
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def toolchain_identity() -> dict:
    selected = shutil.which("bisheng")
    if selected is None:
        raise RuntimeError("bisheng is not available on PATH")
    resolved = str(Path(selected).resolve())
    version = subprocess.check_output([selected, "--version"], text=True, timeout=10)
    resource = subprocess.check_output([selected, "-print-resource-dir"], text=True, timeout=10).strip()
    ascend_home = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_HOME")
    if not ascend_home:
        raise RuntimeError("ASCEND_HOME_PATH or ASCEND_HOME must identify CANN")
    version_file = Path(ascend_home) / "opp/version.info"
    return {
        "selected_path": selected,
        "realpath": resolved,
        "sha256": digest(resolved),
        "version": version,
        "resource_directory": resource,
        "cann_home": str(Path(ascend_home).resolve()),
        "cann_version_info": version_file.read_text(),
    }


def prepare_worker(args, evidence: dict) -> tuple:
    """Import the device runtime only inside an isolated worker, after parsing."""
    from _native_build import verify_manifest

    evidence["run_mode"] = os.environ.get("TL_RUN_MODE", "npu")
    if evidence["run_mode"] != "npu":
        raise RuntimeError("Real-device acceptance requires TL_RUN_MODE=npu")
    manifest = getattr(args, "build_manifest", None) or os.environ.get(
        "TL_ASCEND_MIGRATION_BUILD_MANIFEST",
        str(ROOT / "build/ascend_migration_native_build.json"),
    )
    evidence["native_build"] = verify_manifest(manifest)
    if getattr(args, "require_solver", False):
        import z3

        evidence["solver"] = {"version": z3.get_version_string()}
    libraries = {Path(item["path"]).name: Path(item["path"]) for item in evidence["native_build"]["libraries"]}
    os.environ["TVM_LIBRARY_PATH"] = str(libraries["libtvm.so"].parent)
    os.environ["TILELANG_LIBRARY_PATH"] = str(libraries["libtilelang_module.so"].parent)
    tvm_python = Path(evidence["native_build"]["tvm_source_root"]) / "python"
    os.environ["TVM_IMPORT_PYTHON_PATH"] = str(tvm_python)
    os.environ["PYTHONPATH"] = str(tvm_python) + os.pathsep + os.environ.get("PYTHONPATH", "")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "testing/python/language"))
    sys.path.insert(0, str(tvm_python))
    os.environ["ACL_OP_INIT_MODE"] = "1"
    import acl
    import torch
    import torch_npu  # noqa: F401
    import tilelang
    from tilelang import tvm

    if Path(tilelang.__file__).resolve() != (ROOT / "tilelang/__init__.py").resolve():
        raise RuntimeError(f"TileLang Python source is outside the repository: {tilelang.__file__}")
    if Path(tvm.__file__).resolve() != (tvm_python / "tvm/__init__.py").resolve():
        raise RuntimeError(f"TVM Python source does not match the verified build: {tvm.__file__}")
    evidence["loaded_tvm_python"] = tvm.__file__
    loaded = {
        str(Path(tilelang._LIB_PATH).resolve()),
        str(Path(tvm._ffi.base._LIB._name).resolve()),
    }
    expected = {item["path"] for item in evidence["native_build"]["libraries"]}
    if not loaded.issubset(expected):
        raise RuntimeError(f"Loaded native libraries do not match the verified build: {loaded}")
    evidence["loaded_libraries"] = sorted(loaded)
    evidence["versions"] = {
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "tvm": tvm.__version__,
        "tvm_git": tvm.support.libinfo().get("GIT_COMMIT_HASH"),
    }
    if not torch.npu.is_available():
        raise RuntimeError("The NPU runtime reports no available devices")
    count = torch.npu.device_count()
    if args.device >= count:
        raise RuntimeError(f"Logical device {args.device} is unavailable; device_count={count}")
    torch.npu.set_device(args.device)
    name = acl.get_soc_name()
    upper = name.upper()
    actual = "a2" if "910B" in upper else "a3" if "910_93" in upper or "910C" in upper else None
    evidence["actual_device"] = {
        "soc": name,
        "platform": actual,
        "logical_device": args.device,
        "name": torch.npu.get_device_name(args.device),
        "device_count": count,
    }
    if actual != args.platform:
        raise RuntimeError(f"Requested {args.platform}, but actual SoC {name!r} identifies {actual!r}")
    evidence["inherited_platform_override"] = os.environ.get("TL_PLATFORM")
    os.environ["TL_PLATFORM"] = actual.upper()
    tilelang.disable_cache()
    torch.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)
    evidence["toolchain"] = toolchain_identity()
    from tilelang.jit.adapter.cython import adapter

    evidence["cache"] = {
        "kernel_memory_and_disk": "disabled via tilelang.disable_cache()",
        "kernel_directory": os.environ["TILELANG_CACHE_DIR"],
        "jit_object": "fresh process and directly compiled kernels",
        "autotuner": "not used",
        "cython_host_wrapper": {
            "directory": str(adapter.get_cache_dir()),
            "source_sha256": digest(adapter.cython_wrapper_path),
            "library_sha256": digest(adapter.library_path),
            "mode": "independent host wrapper cache; not NPU kernel compilation evidence",
        },
    }
    return tilelang, torch


def run_worker(
    args,
    case_callback,
    *,
    launch_multiplier=1,
    sentinel_description=None,
    case_parameters=None,
    preflight_check=None,
) -> int:
    evidence = {
        "case": args._case,
        "status": "BLOCKED",
        "compilations": [],
        "kernels": [],
        "launches": {},
    }
    if case_parameters is not None:
        evidence["case_parameters"] = case_parameters
    evidence["launches_per_repeat"] = launch_multiplier
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(evidence, stream, indent=2, default=str)
            stream.write("\n")
        temporary.replace(output)

    save()
    try:
        tilelang, torch = prepare_worker(args, evidence)
        if preflight_check is not None:
            preflight_check(args, evidence, tilelang)
        save()
    except Exception as exc:
        evidence.update(error=str(exc), traceback=traceback.format_exc())
        save()
        return 2
    if args._case == "preflight":
        evidence["status"] = "PASS"
        save()
        return 0

    from tilelang.jit.adapter import libgen
    from tilelang.jit.kernel import JITKernel

    real_run = subprocess.run
    real_compile = tilelang.compile
    real_call = JITKernel.__call__

    def observed_run(command, *pargs, **kwargs):
        is_bisheng = bool(command) and Path(command[0]).name == "bisheng"
        record = None
        if is_bisheng:
            record = {"argv": list(map(str, command)), "status": "RUNNING"}
            record["compiler_realpath"] = str(Path(shutil.which(command[0])).resolve())
            record["source_sha256"] = [
                {"path": str(part), "sha256": digest(part)} for part in command if str(part).endswith(".cpp") and Path(part).is_file()
            ]
            evidence["compilations"].append(record)
            save()
        started = time.monotonic()
        try:
            result = real_run(command, *pargs, **kwargs)
            if record is not None:
                record.update(
                    returncode=result.returncode,
                    status="PASS" if result.returncode == 0 else "FAIL",
                )
                if result.returncode == 0 and "-o" in command:
                    artifact = command[command.index("-o") + 1]
                    record["artifact"] = {"path": artifact, "sha256": digest(artifact)}
            return result
        except Exception as exc:
            if record is not None:
                record.update(status="FAIL", error=str(exc))
            raise
        finally:
            if record is not None:
                record["elapsed_seconds"] = time.monotonic() - started
                save()

    def observed_compile(program, *pargs, **kwargs):
        kwargs["platform"] = args.platform.upper()
        record = {
            "target": kwargs.get("target", "auto"),
            "platform": args.platform.upper(),
            "pass_configs": {str(k): v for k, v in (kwargs.get("pass_configs") or {}).items()},
        }
        evidence["kernels"].append(record)
        kernel = real_compile(program, *pargs, **kwargs)
        record["resolved_flags"] = kernel.compile_flags
        record["resolved_pass_configs"] = dict(kernel.pass_configs)
        # These audited legacy controls have C++ defaults, rather than being
        # materialized by Python's process_default_pass_config().
        controls = {
            "tl.ascend_auto_cv_combine": True,
            "tl.ascend_auto_sync": False,
            "tl.ascend_auto_cross_core_sync": False,
            "tl.ascend_memory_planning": False,
            "tl.ascend_auto_sync_vs": False,
        }
        record["effective_legacy_controls"] = {key: kernel.pass_configs.get(key, default) for key, default in controls.items()}
        record["kernel_id"] = str(id(kernel))
        evidence["launches"][str(id(kernel))] = {"attempted": 0, "completed": 0}
        save()
        return kernel

    def observed_call(kernel, *pargs, **kwargs):
        counts = evidence["launches"][str(id(kernel))]
        counts["attempted"] += 1
        result = real_call(kernel, *pargs, **kwargs)
        counts["completed"] += 1
        return result

    libgen.subprocess.run = observed_run
    tilelang.compile = observed_compile
    JITKernel.__call__ = observed_call
    try:
        parameters = case_callback(args._case, args, torch)
        if parameters is not None:
            evidence["case_parameters"] = parameters
        expected_launches = args.repeat * launch_multiplier
        if len(evidence["compilations"]) != 1:
            raise AssertionError("Each case must invoke real Bisheng exactly once")
        if len(evidence["launches"]) != 1 or any(count["completed"] != expected_launches for count in evidence["launches"].values()):
            raise AssertionError(f"Each case must launch the same compiled kernel {expected_launches} times")
        evidence["status"] = "PASS"
        evidence["golden_checks"] = expected_launches
        evidence["output_sentinel"] = sentinel_description or ("NaN for floating point, -99 for integer, reset each launch")
        save()
        return 0
    except Exception as exc:
        evidence.update(status="FAIL", error=str(exc), traceback=traceback.format_exc())
        save()
        return 1
    finally:
        libgen.subprocess.run = real_run
        tilelang.compile = real_compile
        JITKernel.__call__ = real_call


def run_cases(args, report, cases, entry_path, *, final_check=None) -> int:
    # Keep previous failures and their logs intact when rerunning an output path.
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    artifacts = Path(tempfile.mkdtemp(prefix=destination.stem + "-artifacts-", dir=destination.parent))
    report["artifacts"] = str(artifacts)
    report["required_cases"] = list(cases)
    report["cases"] = []
    report["counts"]["total"] = len(cases)
    report["timeout_policy"] = "separate process group per case; stop the group on first failure or timeout"
    for name in ("preflight", *cases):
        case_output = artifacts / f"{name}.json"
        case_output.unlink(missing_ok=True)
        env = dict(os.environ)
        env["TILELANG_CACHE_DIR"] = tempfile.mkdtemp(prefix=f"{name}-cache-", dir=artifacts)
        env.pop("TILELANG_CLEAR_CACHE", None)
        command = [
            sys.executable,
            str(Path(entry_path).resolve()),
            "--platform",
            args.platform,
            "--device",
            str(args.device),
            "--seed",
            str(args.seed),
            "--repeat",
            str(args.repeat),
            "--timeout",
            str(args.timeout),
            "--output",
            str(case_output),
            "--_case",
            name,
        ]
        if getattr(args, "build_manifest", None) is not None:
            command.extend(["--build-manifest", str(args.build_manifest)])
        if getattr(args, "cost_profile", None) is not None:
            command.extend(["--cost-profile", str(args.cost_profile)])
        try:
            result = run_process(command, artifacts, name, env=env, timeout=args.timeout)
        except OSError as exc:
            result = {"command": command, "exit_code": 2, "status": "BLOCKED", "error": str(exc)}
        if case_output.is_file():
            try:
                result["evidence"] = json.loads(case_output.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                result["evidence_error"] = str(exc)
        evidence = result.get("evidence", {})
        if name == "preflight":
            report["preflight"] = result
        else:
            report["cases"].append({"name": name, **result})
        if result["exit_code"] or evidence.get("status") != "PASS":
            if name != "preflight":
                report["counts"]["failed"] += 1
            report["not_run"] = [item for item in cases if item not in {case["name"] for case in report["cases"]}]
            status = "TIMEOUT" if result["exit_code"] == 124 else "BLOCKED" if result["exit_code"] == 2 else "FAIL"
            error = (
                evidence.get("error") or result.get("error") or (f"{name} did not pass; see {result.get('log', 'process launch error')}")
            )
            if status == "TIMEOUT":
                error = f"{name} exceeded {args.timeout}s; its entire process group was stopped"
            return finish(
                args,
                report,
                status,
                error,
            )
        if name != "preflight":
            report["counts"]["passed"] += 1
    if final_check is not None:
        try:
            final_check(report)
        except (AssertionError, KeyError, TypeError, ValueError) as error:
            return finish(args, report, "FAIL", f"Final case comparison: {error}")
    return finish(args, report, "PASS")
