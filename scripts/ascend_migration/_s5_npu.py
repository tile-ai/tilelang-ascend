"""Shared S5 device CLI, calibrated-identity checks and final physical IR evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import shutil
import sys

from _common import ROOT, finish
from _npu_runner import digest


def load_cost_module():
    name = "ascend_migration_s5_cost"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "tilelang/transform/ascend_pipeline_cost.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def parser(description, cases, *, calibration=False):
    result = argparse.ArgumentParser(description=description)
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
                str(ROOT / "build/ascend_migration_s5_native_build.json"),
            )
        ),
    )
    result.add_argument(
        "--cost-profile",
        type=Path,
        default=Path(
            os.environ.get(
                "TL_ASCEND_PIPELINE_COST_PROFILE",
                str(ROOT / "build/ascend_migration_s5_cost_profile.json"),
            )
        ),
    )
    result.add_argument("--_case", choices=("preflight", *cases), help=argparse.SUPPRESS)
    result.add_argument("--_measure", action="store_true", help=argparse.SUPPRESS)
    if calibration:
        result.add_argument(
            "--calibrate-profile",
            type=Path,
            help="Collect the complete legacy calibration matrix and write a canonical model",
        )
        result.add_argument(
            "--calibration-source-root",
            type=Path,
            help="Optional independent source snapshot matched by --build-manifest",
        )
        result.add_argument("--calibration-timeout", type=float, default=180)
    return result


def validate_arguments(args, report):
    if sys.flags.optimize:
        return finish(args, report, "BLOCKED", "Python optimization disables mandatory assertions")
    if args.device < 0 or args.seed < 0 or args.repeat < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        return finish(args, report, "BLOCKED", "device/seed must be nonnegative; repeat/timeout positive")
    if args._measure and args._case in (None, "preflight"):
        return finish(args, report, "BLOCKED", "Timing requires a concrete internal case worker")
    if getattr(args, "calibrate_profile", None) is not None:
        if not math.isfinite(args.calibration_timeout) or args.calibration_timeout <= 0:
            return finish(args, report, "BLOCKED", "calibration timeout must be finite and positive")
        if args._case or args._measure:
            return finish(args, report, "BLOCKED", "Calibration cannot be combined with a worker")
        if args.calibrate_profile.resolve() == args.output.resolve():
            return finish(args, report, "BLOCKED", "Model and report require different output paths")
        return None
    try:
        args.cost_model = load_cost_module().load_cost_model(args.cost_profile)
    except (OSError, ValueError, TypeError, KeyError) as error:
        return finish(args, report, "BLOCKED", f"Calibrated A3 cost profile: {error}")
    report["cost_profile"] = {
        "path": str(args.cost_profile.resolve()),
        "profile_sha256": args.cost_model["profile_sha256"],
        "file_sha256": digest(args.cost_profile),
    }
    if not args._case and shutil.which("msprof") is None:
        return finish(args, report, "BLOCKED", "msprof op is required for performance acceptance")
    return None


def pass_configs(args, strategy="constraint"):
    return {
        "tl.ascend_pipeline_planner": strategy,
        "tl.ascend_pipeline_cost_model": args.cost_model,
        "tl.ascend_auto_sync": True,
        "tl.ascend_memory_planning": True,
        "tl.ascend_auto_cross_core_sync": False,
        "tl.ascend_auto_sync_vs": False,
    }


def check_identity(args, evidence, tilelang):
    """Compare actual worker hardware/compiler/ordered flags before any compilation."""
    from tilelang.jit.adapter.libgen import resolve_compile_flags
    from tilelang.transform.pass_config import process_default_pass_config

    toolchain = evidence["toolchain"]
    version = next(line.removeprefix("Version=") for line in toolchain["cann_version_info"].splitlines() if line.startswith("Version="))
    flags = resolve_compile_flags("ascendc", process_default_pass_config("ascendc", pass_configs(args)), None)
    expected = {
        "platform": "A3",
        "soc": evidence["actual_device"]["soc"],
        "cann_version": version,
        "bisheng_sha256": toolchain["sha256"],
        "flags": flags,
    }
    load_cost_module().validate_cost_model(args.cost_model, expected_identity=expected)
    evidence["cost_profile"] = {
        "path": str(args.cost_profile.resolve()),
        "profile_sha256": args.cost_model["profile_sha256"],
        "file_sha256": digest(args.cost_profile),
        "matched_identity": expected,
    }


def physical_evidence(kernel):
    """Record actual injector remaps and final allocations, never infer from requested stages."""
    from tilelang import tvm

    tir = tvm.tir
    analyzer = tvm.arith.Analyzer()

    def number(value):
        simplified = analyzer.simplify(value)
        if not isinstance(simplified, tir.IntImm):
            raise AssertionError(f"Unresolved final physical size: {value}")
        return int(simplified)

    def plain(value):
        if isinstance(value, tir.IntImm):
            return int(value)
        if isinstance(value, tir.Var):
            return {"var": value.name, "dtype": str(value.dtype)}
        if isinstance(value, (tvm.ir.Map, dict)):
            return {str(key): plain(item) for key, item in value.items()}
        if isinstance(value, (tvm.ir.Array, list, tuple)):
            return [plain(item) for item in value]
        if isinstance(value, (str, int, bool)):
            return value
        return str(value)

    functions = []
    for symbol, function in kernel.artifact.device_mod.functions.items():
        if not isinstance(function, tir.PrimFunc):
            continue
        attributes = function.attrs or {}
        record = {
            "function": symbol.name_hint,
            "buffer_versions": plain(attributes.get("buffer_versions", {})),
            "buffer_remap": plain(attributes.get("tl.ascend_pipeline_buffer_remap", [])),
            "address_map": plain(attributes.get("address_map", {})),
            "size_map": plain(attributes.get("size_map", {})),
            "allocations": [],
        }

        def visit(node, record=record):
            if isinstance(node, tir.Allocate):
                shape = [number(extent) for extent in node.extents]
                datatype = tvm.DataType(node.dtype)
                size = datatype.bits * datatype.lanes // 8
                for extent in shape:
                    size *= extent
                record["allocations"].append(
                    {
                        "var": node.buffer_var.name,
                        "dtype": str(node.dtype),
                        "shape": shape,
                        "bytes": size,
                    }
                )

        tir.stmt_functor.post_order_visit(function.body, visit)
        functions.append(record)
    if not functions:
        raise AssertionError("Compiled artifact has no device PrimFunc evidence")
    return functions


def tensor_digest(tensor, torch):
    host = tensor.detach().cpu().contiguous()
    result = hashlib.sha256(f"{host.dtype}:{tuple(host.shape)}:".encode())
    result.update(bytes(host.view(torch.uint8).flatten().tolist()))
    return result.hexdigest()


def exact_output_trace(kernel, expected_indices):
    """Interpret final emitted GM stores, including injected guards and launch axes."""
    from tilelang import tvm

    tir = tvm.tir
    analyzer = tvm.arith.Analyzer()
    stores = Counter()

    def integer(value, environment):
        concrete = analyzer.simplify(tir.stmt_functor.substitute(value, environment))
        if not isinstance(concrete, tir.IntImm):
            raise AssertionError(f"Unresolved final output trace expression: {concrete}")
        return int(concrete)

    class BreakLoop(Exception):
        pass

    def execute(statement, environment):
        if isinstance(statement, tir.SeqStmt):
            for child in statement.seq:
                execute(child, environment)
        elif isinstance(statement, tir.For):
            start = integer(statement.min, environment)
            extent = integer(statement.extent, environment)
            if not 0 <= extent <= 65536:
                raise AssertionError("Final trace loop is outside the bounded test domain")
            for value in range(start, start + extent):
                nested = dict(environment)
                nested[statement.loop_var] = tir.IntImm(statement.loop_var.dtype, value)
                try:
                    execute(statement.body, nested)
                except BreakLoop:
                    break
        elif isinstance(statement, tir.AttrStmt) and statement.attr_key == "thread_extent":
            if statement.node.thread_tag not in ("blockIdx.x", "threadIdx.x"):
                raise AssertionError(f"Unexpected final launch axis: {statement.node.thread_tag}")
            extent = integer(statement.value, environment)
            if not 1 <= extent <= 128:
                raise AssertionError("Final trace launch extent is outside the test domain")
            for value in range(extent):
                nested = dict(environment)
                nested[statement.node.var] = tir.IntImm(statement.node.var.dtype, value)
                execute(statement.body, nested)
        elif isinstance(statement, (tir.AttrStmt, tir.Allocate, tir.DeclBuffer)):
            execute(statement.body, environment)
        elif isinstance(statement, tir.LetStmt):
            nested = dict(environment)
            nested[statement.var] = tir.IntImm(statement.var.dtype, integer(statement.value, environment))
            execute(statement.body, nested)
        elif isinstance(statement, tir.IfThenElse):
            selected = statement.then_case if integer(statement.condition, environment) else statement.else_case
            if selected is not None:
                execute(selected, environment)
        elif isinstance(statement, tir.AssertStmt):
            if not integer(statement.condition, environment):
                raise AssertionError("A final artifact assertion rejects the test binding")
            execute(statement.body, environment)
        elif isinstance(statement, tir.Evaluate):
            call = statement.value
            if not isinstance(call, tir.Call):
                if integer(call, environment) != 0:
                    raise AssertionError("Unexpected nonzero final evaluation")
                return
            if getattr(call.op, "name", None) == "tl.loop_break":
                raise BreakLoop
            if getattr(call.op, "name", None) == "tir.call_extern" and call.args[0].value.startswith("tl::ascend::copy_ub_to_gm<"):
                pointer = call.args[2]
                if getattr(pointer.op, "name", None) != "tir.tvm_access_ptr" or pointer.args[1].name != "O":
                    raise AssertionError("Unexpected final GM destination")
                offset = integer(pointer.args[2], environment)
                count = integer(pointer.args[3], environment)
                if count not in (32, 64):
                    raise AssertionError("Final trace requires a complete aligned32/64 tile")
                stores.update(range(offset, offset + count))
        else:
            raise AssertionError(f"Unsupported final output statement: {type(statement).__name__}")

    for function in kernel.artifact.device_mod.functions.values():
        if isinstance(function, tir.PrimFunc):
            execute(function.body, {})
    expected = Counter({index: 1 for index in expected_indices})
    if stores != expected:
        raise AssertionError(
            {
                "missing": list((expected - stores).items())[:8],
                "duplicate_or_invalid": list((stores - expected).items())[:8],
            }
        )
    return {
        "status": "PASS",
        "method": "interpret final device IR stores, loops and predicates",
        "element_writes": sum(stores.values()),
        "unique_elements": len(stores),
        "maximum_multiplicity": max(stores.values(), default=0),
        "indices_sha256": hashlib.sha256(",".join(map(str, sorted(stores))).encode()).hexdigest(),
    }
