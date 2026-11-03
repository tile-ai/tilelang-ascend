#!/usr/bin/env python3
# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Isolated execution and instrumentation for one Ascend example entry."""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import runpy
import shlex
import sys
import traceback
from typing import Any


REPO = Path(__file__).resolve().parents[2]
SYNC_CONFIG = {
    "tl.ascend_sync_backend": "upstream",
    "tl.ascend_auto_sync": True,
    "tl.ascend_auto_sync_vs": True,
}


def entry_key(path: str) -> str:
    """Keep log names readable while distinguishing otherwise colliding paths."""
    readable = path.replace("/", "__").removesuffix(".py")
    readable = "".join(c if c.isalnum() or c in "_.-" else "_" for c in readable)
    suffix = hashlib.sha256(path.encode("utf-8")).hexdigest()[:12]
    return f"{readable[:180]}-{suffix}"


def atomic_json(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _prepare_environment(cache_path: Path | None = None) -> None:
    os.environ["ACL_OP_INIT_MODE"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["TILELANG_AUTO_TUNING_CPU_COUNTS"] = "2"
    os.environ["TILELANG_AUTO_TUNING_MAX_CPU_COUNT"] = "2"
    if cache_path is not None:
        os.environ["TILELANG_CACHE_DIR"] = str(cache_path)
    sys.path.insert(0, str(REPO))


def _load_runtime() -> tuple[Any, Any, Any]:
    tilelang = importlib.import_module("tilelang")
    torch = importlib.import_module("torch")
    torch_npu = importlib.import_module("torch_npu")
    return tilelang, torch, torch_npu


def runtime_details(tilelang: Any, torch: Any, torch_npu: Any) -> dict[str, Any]:
    tvm = tilelang.tvm
    abi_function = tvm.get_global_func("tl.ascend_sync.EngineCacheABI", allow_missing=True)
    z3_function = tvm.get_global_func(
        "tl.ascend_sync.Z3ProofServiceAvailable",
        allow_missing=True,
    )
    libraries = []
    for value in (
        getattr(tilelang, "_LIB_PATH", None),
        getattr(getattr(tilelang, "_LIB", None), "_name", None),
    ):
        if isinstance(value, str) and Path(value).is_file():
            actual = str(Path(value).resolve())
            if actual not in libraries:
                libraries.append(actual)
    return {
        "abi": str(abi_function()) if abi_function is not None else None,
        "native_abi": str(abi_function()) if abi_function is not None else None,
        "z3_available": bool(z3_function()) if z3_function is not None else False,
        "device_count": int(torch.npu.device_count()),
        "tilelang_file": str(Path(tilelang.__file__).resolve()),
        "tilelang_version": str(getattr(tilelang, "__version__", "unknown")),
        "torch_version": str(torch.__version__),
        "torch_npu_version": str(getattr(torch_npu, "__version__", "unknown")),
        "native_libraries": libraries,
        "python": sys.executable,
    }


def preflight() -> int:
    result: dict[str, Any] = {"success": False, "error": None}
    try:
        _prepare_environment()
        with contextlib.redirect_stdout(sys.stderr):
            result.update(runtime_details(*_load_runtime()))
        if not result["abi"]:
            raise RuntimeError("The loaded TileLang library has no Ascend sync ABI function")
        if not result["z3_available"]:
            raise RuntimeError("The loaded TileLang library has no Z3 proof service")
        if not result["native_libraries"]:
            raise RuntimeError("Cannot identify the actual loaded TileLang native library")
        result["success"] = True
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc(file=sys.stderr)
    print(json.dumps(result, default=str), flush=True)
    return 0 if result["success"] else 2


def _canonical_configs(configs: Any) -> dict[str, Any]:
    return {str(getattr(key, "value", key)): value for key, value in configs.items()}


def _flags(flags: Any) -> list[str]:
    if isinstance(flags, str):
        return shlex.split(flags)
    return list(flags or [])


def _instrument(
    tilelang: Any,
    torch: Any,
    metadata: dict[str, Any],
    run_dir: Path,
    key: str,
    device: int,
) -> Any:
    token = object()
    jit_module = importlib.import_module("tilelang.jit")
    original_compile = jit_module.compile
    compile_signature = inspect.signature(original_compile)

    def require_runtime() -> None:
        if int(torch.npu.current_device()) != device:
            raise RuntimeError("The example changed the selected NPU device")
        if tilelang.is_cache_enabled():
            raise RuntimeError("TileLang cache was enabled during the example run")

    @functools.wraps(original_compile)
    def compile_upstream(*pos: Any, **kw: Any) -> Any:
        metadata["compile_attempts"] += 1
        info = {
            "index": metadata["compile_attempts"],
            "error": None,
        }
        metadata["kernels"].append(info)
        try:
            bound = compile_signature.bind_partial(*pos, **kw)
            configs = dict(bound.arguments.get("pass_configs") or {})
            configs.update(SYNC_CONFIG)
            bound.arguments["pass_configs"] = configs
            bound.arguments["compile_flags"] = [
                *_flags(bound.arguments.get("compile_flags")),
                "--cce-auto-sync=off",
            ]
            info.update(
                target=str(bound.arguments.get("target", "auto")),
                platform=str(bound.arguments.get("platform", "auto")),
                pass_configs=_canonical_configs(configs),
                compile_flags=bound.arguments["compile_flags"],
            )
            print("PR1965_COMPILE", json.dumps(info, default=str), flush=True)
            require_runtime()
            kernel = original_compile(*bound.args, **bound.kwargs)
            source = run_dir / "kernels" / f"{key}-{info['index']:03d}.cc"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(kernel.get_kernel_source(), encoding="utf-8")
            info["source"] = str(source)
            info["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
            kernel.adapter._ascend_examples_token = token
            kernel.adapter._pr1965_configured = True
            metadata["compiled"] += 1
            return kernel
        except BaseException as error:
            info["error"] = f"{type(error).__name__}: {error}"
            raise

    jit_module.compile = compile_upstream
    tilelang.compile = compile_upstream

    def launch(original: Any, check: Any, kind: str) -> Any:
        @functools.wraps(original)
        def checked(self: Any, *pos: Any, **kw: Any) -> Any:
            metadata["launch_attempts"] += 1
            try:
                require_runtime()
                check(self)
                result = original(self, *pos, **kw)
            except BaseException as error:
                metadata["launch_failures"].append(
                    {"kind": kind, "error": f"{type(error).__name__}: {error}"},
                )
                raise
            metadata["launches"] += 1
            return result

        return checked

    def check_kernel(kernel: Any) -> None:
        check_adapter(kernel.adapter)
        configs = _canonical_configs(kernel.pass_configs)
        for name, expected in SYNC_CONFIG.items():
            if configs.get(name) != expected:
                raise RuntimeError(f"Unconfigured kernel launch: {name}")
        flags = _flags(kernel.compile_flags)
        sync_flags = [flag for flag in flags if flag.startswith("--cce-auto-sync=")]
        if not sync_flags or sync_flags[-1] != "--cce-auto-sync=off":
            raise RuntimeError("Bisheng automatic synchronization is enabled")

    def check_adapter(adapter: Any) -> None:
        if getattr(adapter, "_ascend_examples_token", None) is not token:
            raise RuntimeError("Adapter launch bypassed the configured compilation hook")

    tilelang.JITKernel.__call__ = launch(tilelang.JITKernel.__call__, check_kernel, "kernel")
    base = importlib.import_module("tilelang.jit.adapter.base").BaseKernelAdapter
    base.__call__ = launch(base.__call__, check_adapter, "adapter")

    def assertion(original: Any, name: str) -> Any:
        @functools.wraps(original)
        def checked(*pos: Any, **kw: Any) -> Any:
            try:
                result = original(*pos, **kw)
            except BaseException as error:
                metadata["assertion_failures"].append(
                    {"method": name, "error": f"{type(error).__name__}: {error}"},
                )
                raise
            metadata["assertions"] += 1
            return result

        return checked

    for name in ("assert_allclose", "manual_assert_close"):
        original = getattr(tilelang.Profiler, name)
        setattr(tilelang.Profiler, name, assertion(original, f"Profiler.{name}"))
    torch.testing.assert_close = assertion(torch.testing.assert_close, "torch.assert_close")
    numpy = importlib.import_module("numpy")
    for name in (
        "assert_allclose",
        "assert_equal",
        "assert_array_equal",
        "assert_almost_equal",
        "assert_array_almost_equal",
        "assert_array_less",
    ):
        original = getattr(numpy.testing, name)
        setattr(numpy.testing, name, assertion(original, f"numpy.testing.{name}"))

    pytest = importlib.import_module("pytest")
    original_main = pytest.main
    main_signature = inspect.signature(original_main)

    @functools.wraps(original_main)
    def pytest_main(*pos: Any, **kw: Any) -> Any:
        bound = main_signature.bind_partial(*pos, **kw)
        outcome = {
            "args": bound.arguments.get("args"),
            "exitstatus": None,
            "failed_reports": [],
            "collection_reports": [],
            "collected": 0,
            "tests_collected": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
        }
        metadata["pytest_sessions"].append(outcome)

        class Recorder:
            def pytest_collection_finish(self, session: Any) -> None:
                outcome["collected"] = len(session.items)
                outcome["tests_collected"] = len(session.items)

            def pytest_collectreport(self, report: Any) -> None:
                outcome["collection_reports"].append(
                    {"nodeid": report.nodeid, "outcome": report.outcome},
                )
                if report.failed:
                    outcome["failed_reports"].append({"nodeid": report.nodeid, "when": "collect"})

            def pytest_runtest_logreport(self, report: Any) -> None:
                if report.failed:
                    outcome["failed"] += 1
                    outcome["failed_reports"].append(
                        {"nodeid": report.nodeid, "when": report.when},
                    )
                elif report.skipped:
                    outcome["skipped"] += 1
                elif report.when == "call" and report.passed:
                    outcome["passed"] += 1

            def pytest_sessionfinish(self, session: Any, exitstatus: int) -> None:
                outcome["exitstatus"] = int(exitstatus)
                outcome["collected"] = int(session.testscollected)
                outcome["tests_collected"] = int(session.testscollected)

        bound.arguments["plugins"] = [*(bound.arguments.get("plugins") or []), Recorder()]
        try:
            code = original_main(*bound.args, **bound.kwargs)
        except BaseException as error:
            outcome["error"] = f"{type(error).__name__}: {error}"
            raise
        outcome["exitstatus"] = int(code)
        return code

    pytest.main = pytest_main
    return pytest


def child_main(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    metadata: dict[str, Any] = {
        "path": None,
        "mode": None,
        "variant": "upstream",
        "instrumentation_version": 6,
        "actual_device": args.device,
        "npu_device": None,
        "native_abi": None,
        "cache_disabled": False,
        "original_args": [],
        "pytest_markers": None,
        "compile_attempts": 0,
        "compiled": 0,
        "launch_attempts": 0,
        "launches": 0,
        "launch_failures": [],
        "assertions": 0,
        "assertion_failures": [],
        "kernels": [],
        "pytest_sessions": [],
        "error": None,
    }
    key = f"invalid-entry-{args.index}"
    torch = None
    code = 1
    try:
        entries = json.loads(Path(args.entry_file).read_text(encoding="utf-8"))
        if isinstance(entries, dict):
            entries = entries["entries"]
        entry = entries[args.index]
        key = entry_key(entry["path"])
        metadata.update(
            path=entry["path"],
            mode=entry["mode"],
            original_device=entry.get("device"),
            original_args=list(entry.get("args", [])),
            pytest_markers=entry.get("pytest_markers", "not bench"),
        )
        _prepare_environment(run_dir / "cache" / key)
        tilelang, torch, torch_npu = _load_runtime()
        details = runtime_details(tilelang, torch, torch_npu)
        metadata["native_abi"] = details["abi"]
        metadata["native_libraries"] = details["native_libraries"]
        print("PR1965_ENGINE_ABI", details["abi"], flush=True)
        if details["abi"] != args.expected_abi:
            raise RuntimeError("Loaded Ascend synchronization ABI differs from preflight")
        if not details["z3_available"]:
            raise RuntimeError("Ascend Z3 proof service is unavailable")
        torch.npu.set_device(args.device)
        metadata["npu_device"] = int(torch.npu.current_device())
        tilelang.disable_cache()
        metadata["cache_disabled"] = not tilelang.is_cache_enabled()
        if not metadata["cache_disabled"]:
            raise RuntimeError("Cannot disable TileLang cache")
        pytest = _instrument(tilelang, torch, metadata, run_dir, key, args.device)
        path = REPO / entry["path"]
        sys.path.insert(0, str(path.parent))
        if entry["mode"] == "pytest":
            code = int(
                pytest.main(
                    [
                        str(path),
                        *metadata["original_args"],
                        "-q",
                        "-m",
                        metadata["pytest_markers"],
                        "-p",
                        "no:cacheprovider",
                        "--tb=short",
                    ],
                ),
            )
        elif entry["mode"] == "script":
            sys.argv = [str(path), *metadata["original_args"]]
            runpy.run_path(str(path), run_name="__main__")
            code = 0
        else:
            raise ValueError("The worker only executes script or pytest entries")
    except SystemExit as error:
        code = error.code if isinstance(error.code, int) else int(error.code is not None)
        if code:
            metadata["error"] = f"SystemExit: {error.code}"
    except BaseException as error:
        code = 1
        metadata["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        if code == 0 and torch is not None:
            try:
                metadata["npu_device"] = int(torch.npu.current_device())
                metadata["cache_disabled"] = not tilelang.is_cache_enabled()
                if metadata["npu_device"] != args.device:
                    raise RuntimeError("The example changed the selected NPU device")
                if not metadata["cache_disabled"]:
                    raise RuntimeError("TileLang cache was enabled during the example run")
                if torch.npu.is_initialized():
                    torch.npu.synchronize()
            except BaseException as error:
                code = 1
                metadata["error"] = f"{type(error).__name__}: {error}"
                traceback.print_exc()
        metadata["returncode"] = code
        atomic_json(run_dir / "logs" / f"{key}.json", metadata)
        print("PR1965_RESULT", json.dumps(metadata, default=str), flush=True)
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--entry-file", type=Path)
    parser.add_argument("--index", type=int)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--device", type=int)
    parser.add_argument("--expected-abi")
    args = parser.parse_args()
    if args.preflight:
        return preflight()
    for name in ("entry_file", "index", "run_dir", "device", "expected_abi"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required without --preflight")
    return child_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
