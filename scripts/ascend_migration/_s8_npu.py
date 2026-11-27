"""S8 workload specifications and identity gates; stdlib-only until worker entry."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import sys

from _common import ROOT, finish
from _native_build import verify_manifest
from _npu_runner import digest

COMPILE_FLAGS = ["--cce-auto-sync=off", "-O3"]
BENCHMARK_WARMUPS = 3
FIXTURES = {
    "gemm_single": {"kind": "gemm", "trips": 1, "cores": 1, "m": 64, "n": 64, "k": 32},
    "gemm_batch3": {
        "kind": "gemm",
        "trips": 3,
        "cores": 4,
        "m": 128,
        "n": 64,
        "k": 32,
        "require_n2_pools": ("shared.l1",),
        "require_reorder": True,
    },
    "reduce_single": {"kind": "reduce", "trips": 1, "cores": 1},
    "reduce_batch3": {
        "kind": "reduce",
        "trips": 3,
        "cores": 4,
        "require_n2_pools": ("shared.ub",),
        "require_reorder": True,
    },
    "mixed_add": {"kind": "mixed", "operation": "add", "trips": 3, "cores": 1},
    "mixed_reduce": {"kind": "mixed", "operation": "reduce", "trips": 3, "cores": 4},
}
PARAMETERS = {f"{name}_{strategy}": {**spec, "strategy": strategy} for name, spec in FIXTURES.items() for strategy in ("legacy", "auto")}
EXPERT_CASES = ("expert_gemm_c128", "expert_l1_tail_stages2")
WORKLOAD_CASES = (*PARAMETERS, *EXPERT_CASES)
BENCHMARK_BASES = ("gemm_batch3", "reduce_batch3", "mixed_reduce")
BENCHMARK_CASES = tuple(f"{name}_{strategy}" for name in BENCHMARK_BASES for strategy in ("legacy", "auto"))


def parser(description, *, benchmark=False):
    result = argparse.ArgumentParser(description=description)
    result.set_defaults(require_solver=True)
    result.add_argument("--platform", choices=("a3",), required=True)
    result.add_argument("--device", type=int, default=0)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument(
        "--repeat",
        type=int,
        default=20 if benchmark else 100,
        help=(
            "Timed rounds; each has three checked warmups and one timed call"
            if benchmark
            else "Checked launches of each single compiled kernel"
        ),
    )
    result.add_argument("--timeout", type=float, default=120)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument(
        "--build-manifest",
        type=Path,
        default=Path(
            os.environ.get(
                "TL_ASCEND_MIGRATION_BUILD_MANIFEST",
                ROOT / "build/ascend_migration_s8_native_build.json",
            )
        ),
    )
    cases = BENCHMARK_CASES if benchmark else WORKLOAD_CASES
    result.add_argument("--_case", choices=("preflight", *cases), help=argparse.SUPPRESS)
    return result


def validate(args, report):
    if sys.flags.optimize:
        return finish(args, report, "BLOCKED", "Python optimization disables mandatory assertions")
    if args.device < 0 or args.seed < 0 or args.repeat < 1:
        return finish(args, report, "BLOCKED", "device/seed nonnegative and repeat positive required")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        return finish(args, report, "BLOCKED", "--timeout must be finite and positive")
    try:
        report["native_build"] = verify_manifest(args.build_manifest)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return finish(args, report, "BLOCKED", f"Native build identity: {error}")
    report["scheduling_model"] = {
        "name": "abstract_issue_step_v1",
        "purpose": "correctness scheduling only; not measured nanoseconds or throughput",
        "uses_s5_nanosecond_profile": False,
    }
    return None


def preflight(args, evidence, tilelang):
    names = (
        "AscendPrepareAutoSchedule",
        "AscendAutoSchedule",
        "AscendMaterializeSchedule",
        "AscendInsertScheduledSync",
        "AscendVerifyAutoSchedule",
    )
    missing = [name for name in names if tilelang.tvm.get_global_func("tl.transform." + name, allow_missing=True) is None]
    if missing:
        raise RuntimeError(f"Verified native library lacks the S8 backend: {missing}")
    evidence["s8_backend"] = {"registered_passes": list(names), "model": "abstract_issue_step_v1"}


def pass_configs(strategy):
    return {
        "tl.ascend_pipeline_planner": strategy,
        "tl.ascend_sync_strategy": "a3" if strategy == "auto" else "legacy",
        "tl.ascend_auto_sync": True,
        "tl.ascend_memory_planning": True,
        "tl.ascend_auto_cv_combine": True,
        "tl.ascend_auto_cross_core_sync": False,
        "tl.ascend_auto_sync_vs": False,
    }


def program(spec):
    sys.path.insert(0, str(ROOT / "testing/python/ascend_migration"))
    from test_s8_kernels import make_gemm_kernel, make_mixed_kernel, make_reduce_kernel

    common = {key: spec[key] for key in ("trips", "cores")}
    if spec["kind"] == "gemm":
        return make_gemm_kernel(**common, m=spec["m"], n=spec["n"], k=spec["k"], k_l0=spec["k"])
    if spec["kind"] == "reduce":
        return make_reduce_kernel(**common)
    return make_mixed_kernel(**common, operation=spec["operation"], manual_legacy=spec["strategy"] == "legacy")


def build_kernel(tilelang, spec):
    return tilelang.compile(
        program(spec),
        out_idx=[],
        target="ascendc",
        platform="A3",
        pass_configs=pass_configs(spec["strategy"]),
        compile_flags=COMPILE_FLAGS,
    )


def run_case(name, args, torch, *, metadata=None):
    if name in EXPERT_CASES:
        from s1_run_baseline_npu import run_case as legacy_case

        legacy_case("gemm_c128" if name == "expert_gemm_c128" else "l1_tail_stages2", args, torch)
        return {"case": name, "role": "separate existing Expert legacy compatibility"}
    import tilelang

    spec = PARAMETERS[name]
    kernel = build_kernel(tilelang, spec)
    if metadata is None:
        metadata = {}
    metadata.update({**spec, **describe_kernel(kernel, args, spec)})
    state = CaseData(spec, args, torch)
    metadata.update(reference_role=reference_role(spec), rounds=[], output_sentinel=state.sentinel)
    for index in range(args.repeat):
        state.prepare(index)
        kernel(*state.arguments)
        torch.npu.current_stream().synchronize()
        metadata["rounds"].append({"index": index, **state.check()})
    return metadata


def reference_role(spec):
    if spec["strategy"] == "auto":
        return "new complete finite-domain scheduler/materializer/synchronization"
    if spec["kind"] == "mixed":
        return "legacy compute with explicit balanced manual READY/FREE; directives differ"
    return "existing legacy planner/synchronization with the same computation"


def compare_pairs(report):
    results = {row["name"]: row["evidence"] for row in report["cases"]}
    comparisons = report["strategy_comparisons"] = []
    for name in FIXTURES:
        left, right = [results[f"{name}_{strategy}"] for strategy in ("legacy", "auto")]
        if len(left["case_parameters"]["rounds"]) != report["repeat"]:
            raise AssertionError("Missing per-launch golden records")
        if left["case_parameters"]["rounds"] != right["case_parameters"]["rounds"]:
            raise AssertionError("Paired input/output/golden/workspace hash records differ")
        if left["kernels"][0]["resolved_flags"] != right["kernels"][0]["resolved_flags"]:
            raise AssertionError("Paired effective ordered compiler flags differ")
        if compiler_options(left["compilations"][0]) != compiler_options(right["compilations"][0]):
            raise AssertionError("Paired actual ordered Bisheng options differ")
        comparisons.append({"fixture": name, "status": "PASS", "launch_records": report["repeat"]})


def compiler_options(record):
    """Ignore only per-worker compiler input/output paths, retaining ordered options."""
    argv = record["argv"]
    if record["status"] != "PASS" or not argv or Path(argv[0]).name != "bisheng":
        raise AssertionError("A real successful Bisheng invocation is required")
    options, inputs, outputs = [], 0, 0
    index = 1
    while index < len(argv):
        option = argv[index]
        if option == "-o":
            if index + 1 >= len(argv):
                raise AssertionError("Missing actual Bisheng output path")
            outputs += 1
            index += 2
            continue
        if option.endswith(".cpp"):
            inputs += 1
        else:
            options.append(option)
        index += 1
    if inputs != 1 or outputs != 1:
        raise AssertionError("The actual invocation must have one source and output")
    return options


def tensor_digest(tensor, torch):
    host = tensor.detach().cpu().contiguous()
    value = hashlib.sha256(f"{host.dtype}:{tuple(host.shape)}:".encode())
    value.update(host.view(torch.uint8).numpy().tobytes())
    return value.hexdigest()


class CaseData:
    """Keep one allocation set; refresh finite inputs and all output sentinels."""

    sentinel = "NaN output/canary; mixed workspace has finite random prefix and NaN trailing rows"

    def __init__(self, spec, args, torch):
        self.spec, self.torch = spec, torch
        self.generator = torch.Generator().manual_seed(args.seed)
        self.hosts = {}
        cores, trips, kind = spec["cores"], spec["trips"], spec["kind"]
        count = max(trips, 1)
        if kind == "gemm":
            m, n, k = spec["m"], spec["n"], spec["k"]
            inputs = {
                "A": ((cores, count, m, k), "float16"),
                "B": ((cores, count, k, n), "float16"),
            }
            self.output_shape = (cores, trips + 1, m, n)
        elif kind == "reduce":
            inputs = {"A": ((cores, count, 16, 64), "float32")}
            self.output_shape = (cores, trips + 1, 16)
        else:
            inputs = {
                "A": ((cores, count, 128, 128), "float16"),
                "B": ((cores, count, 128, 128), "float16"),
            }
            if spec["operation"] == "add":
                inputs["D"] = ((cores, count, 128, 128), "float32")
                self.output_shape = (cores, trips + 1, 128, 128)
            else:
                self.output_shape = (cores, trips + 1, 128)
            inputs["Workspace"] = ((cores, 144, 128), "float32")
        self.input_layout = inputs
        self.tensors = {key: torch.empty(shape, dtype=getattr(torch, dtype), device="npu") for key, (shape, dtype) in inputs.items()}
        self.tensors["O"] = torch.empty(self.output_shape, dtype=torch.float32, device="npu")
        self.arguments = tuple(self.tensors.values())
        self.buffer_pointers = {key: tensor.data_ptr() for key, tensor in self.tensors.items()}
        if len(set(self.buffer_pointers.values())) != len(self.buffer_pointers):
            raise AssertionError("The noalias caller contract requires independently allocated GM")
        self.canary_elements = cores * math.prod(self.output_shape[2:])
        if kind == "mixed":
            self.canary_elements += cores * 16 * 128

    def prepare(self, index):
        torch = self.torch
        for name, (shape, dtype) in self.input_layout.items():
            if name == "Workspace":
                continue
            values = torch.randint(-2, 3, shape, generator=self.generator).to(torch.float32)
            labels = (torch.arange(shape[0]).reshape(shape[0], *([1] * (len(shape) - 1))) + index % 5) % 3
            values = (values - labels if name == "B" else values + labels) / 32
            self.hosts[name] = values.to(getattr(torch, dtype))
        trips = self.spec["trips"]
        self.expected = torch.full(self.output_shape, float("nan"), dtype=torch.float32)
        if self.spec["kind"] == "reduce":
            self.expected[:, :trips] = self.hosts["A"][:, :trips].sum(dim=-1)
        else:
            product = self.hosts["A"].float() @ self.hosts["B"].float()
            if self.spec["kind"] == "gemm":
                self.expected[:, :trips] = product[:, :trips]
            elif self.spec["operation"] == "add":
                self.expected[:, :trips] = product[:, :trips] + self.hosts["D"][:, :trips]
            else:
                self.expected[:, :trips] = product[:, :trips].sum(dim=-1)
            if self.spec["kind"] == "mixed":
                shape = self.input_layout["Workspace"][0]
                initial = torch.full(shape, float("nan"), dtype=torch.float32)
                initial[:, :128] = torch.randint(-8, 9, (shape[0], 128, 128), generator=self.generator).float() / 16
                self.hosts["Workspace"] = initial
                self.workspace_expected = initial.clone()
                if trips:
                    self.workspace_expected[:, :128] = product[:, trips - 1]
        self.input_hashes = {name: tensor_digest(value, torch) for name, value in self.hosts.items()}
        self.golden_hash = tensor_digest(self.expected, torch)
        self.restore()

    def restore(self):
        for name, value in self.hosts.items():
            self.tensors[name].copy_(value)
        self.tensors["O"].fill_(float("nan"))
        self.torch.npu.current_stream().synchronize()

    def check(self):
        torch = self.torch
        actual = self.tensors["O"].cpu()
        torch.testing.assert_close(actual, self.expected, rtol=1e-4, atol=1e-4, equal_nan=True)
        result = {
            "golden_status": "PASS",
            "input_sha256": dict(self.input_hashes),
            "golden_sha256": self.golden_hash,
            "output_sha256": tensor_digest(actual, torch),
            "canary_elements": self.canary_elements,
        }
        if self.spec["kind"] == "mixed":
            workspace = self.tensors["Workspace"].cpu()
            torch.testing.assert_close(workspace, self.workspace_expected, rtol=1e-4, atol=1e-4, equal_nan=True)
            result["workspace_output_sha256"] = tensor_digest(workspace, torch)
            result["workspace_golden_sha256"] = tensor_digest(self.workspace_expected, torch)
        return result


def inspect_function(function):
    """Cross-check typed remap facts against actual Allocate/access_ptr and MP maps."""
    from tilelang import tvm
    from tilelang.transform.ascend_schedule_solver_ffi import _plain
    from tilelang.transform.ascend_z3_scheduler import verify_schedule

    attrs = function.attrs or {}
    plan = attrs.get("tl.ascend_auto_schedule")
    if plan is None:
        raise AssertionError("Auto code must carry the actual complete S8 schedule")
    if int(plan["schema_version"]) != 1 or str(plan["model"]) != "abstract_issue_step_v1":
        raise AssertionError("Wrong actual auto schema or scheduling cost unit")
    if not int(plan["materialized"]):
        raise AssertionError("A solver result without new materialization is not auto execution")
    problem, solution = _plain(plan["problem"]), _plain(plan["solution"])
    tasks = list(plan["source_analysis"]["tasks"])
    if tasks:
        verified = verify_schedule(problem, solution)
        if not verified["valid"]:
            raise AssertionError(f"Independent integer verifier rejected the auto result: {verified}")
    else:
        if (
            any(problem[key] for key in ("latencies", "iis", "resource_flags", "data_deps"))
            or solution["status"] != "SAT"
            or any(solution[key] for key in ("start_times", "order", "buffer_versions"))
            or solution["ii"] != 0
        ):
            raise AssertionError("An empty source domain cannot contain numeric/emitted work")
        verified = {
            "valid": True,
            "empty_source_domain": True,
            "numeric_verifier_called": False,
            "method": "Explicit empty source/problem/result oracle; S2 numeric tasks require n>=1",
        }
    order = solution["order"]
    if sorted(order) != list(range(len(tasks))):
        raise AssertionError("Execution order must cover every original finite task exactly once")
    groups = {}
    for task in tasks:
        groups.setdefault((int(task["group"]), int(task["actor"])), []).append(int(task["id"]))
    analyzer = tvm.arith.Analyzer()

    def integer(value):
        simplified = analyzer.simplify(value)
        if not isinstance(simplified, tvm.tir.IntImm):
            raise AssertionError(f"Finite emitted geometry remains symbolic: {simplified}")
        return int(simplified)

    allocations, pointers, emitted = [], [], []

    def emitted_statements(node):
        # Walk statement occurrences, including shared nodes twice when emitted twice.
        if isinstance(node, tvm.tir.AttrStmt) and node.attr_key == "tl.ascend_auto_schedule_task_id":
            if int(node.value) != 1:
                raise AssertionError("Wrong actual finite task-instance marker")
            emitted.append(int(node.node))
        if isinstance(node, tvm.tir.SeqStmt):
            for statement in node.seq:
                emitted_statements(statement)
        elif isinstance(node, tvm.tir.IfThenElse):
            emitted_statements(node.then_case)
            if node.else_case is not None:
                emitted_statements(node.else_case)
        elif isinstance(node, tvm.tir.BlockRealize):
            emitted_statements(node.block)
        elif isinstance(node, tvm.tir.Block):
            if node.init is not None:
                emitted_statements(node.init)
            emitted_statements(node.body)
        elif hasattr(node, "body"):
            emitted_statements(node.body)

    def visit(node):
        if isinstance(node, tvm.tir.Allocate):
            dtype = tvm.DataType(node.dtype)
            capacity = math.prod(integer(extent) for extent in node.extents) * dtype.bits * dtype.lanes // 8
            allocations.append((node.buffer_var, capacity, str(node.dtype)))
        elif isinstance(node, tvm.tir.Call) and str(getattr(node.op, "name", "")) == "tir.tvm_access_ptr":
            dtype = tvm.DataType(node.args[0].dtype)
            width = dtype.bits * dtype.lanes // 8
            pointers.append((node.args[1], node.args[2], node.args[3], width))

    emitted_statements(function.body)
    tvm.tir.stmt_functor.post_order_visit(function.body, visit)
    if sorted(emitted) != list(range(len(tasks))):
        raise AssertionError("Actual finite emission loses or duplicates a source task")
    for members in groups.values():
        if [task for task in emitted if task in members] != [task for task in order if task in members]:
            raise AssertionError("Actual actor execution differs from the solved permutation")
    reordered = any([task for task in emitted if task in members] != members for members in groups.values())

    def matching(items, variable):
        return [item for item in items if item[0].same_as(variable)]

    size_map = list(attrs.get("size_map", {}).items())
    address_map = list(attrs.get("address_map", {}).items())
    selected = list(plan["selected_versions"].items())
    records = []
    for row in attrs.get("tl.ascend_auto_schedule_buffer_remap", []):
        logical, physical = row["logical_data"], row["physical_data"]
        versions, pitch, span, total = (int(row[key]) for key in ("versions", "pitch", "span", "total"))
        if versions not in (1, 2) or not 0 < span <= pitch or total != versions * pitch:
            raise AssertionError("Invalid emitted byte-pitch/version geometry")
        choice = matching(selected, logical)
        if choice and int(choice[0][1]) != versions:
            raise AssertionError("Materialized versions differ from the selected candidate")
        allocated = matching(allocations, physical)
        sizes, addresses = matching(size_map, physical), matching(address_map, physical)
        accessed = [(item[0], integer(item[1]) * item[3], integer(item[2]) * item[3]) for item in matching(pointers, physical)]
        if allocated and any(item[1] < total for item in allocated):
            raise AssertionError("Actual allocation is smaller than the emitted versions")
        if sizes and int(sizes[0][1]) < total:
            raise AssertionError("Actual memory-planning size is smaller than all physical slots")
        if any(offset < 0 or offset + extent > total for _, offset, extent in accessed):
            raise AssertionError("Actual typed pointer exceeds its materialized owner")
        slots = [{"task": int(item["task"]), "slot": int(item["slot"])} for item in row["slots"]]
        if any(not 0 <= item["slot"] < versions for item in slots):
            raise AssertionError("A source task selects a nonexistent physical version")
        if versions == 2 and (
            not choice
            or not allocated
            or not sizes
            or not accessed
            or {item["slot"] for item in slots} != {0, 1}
            or not any(offset >= pitch for _, offset, _ in accessed)
        ):
            raise AssertionError("N2 needs real allocation/MP storage and accesses to both slots")
        scope = str(getattr(physical.type_annotation, "storage_scope", "")).lower()
        records.append(
            {
                "logical_data": str(logical),
                "physical_data": str(physical),
                "pool": scope,
                "versions": versions,
                "pitch": pitch,
                "span": span,
                "total": total,
                "slots": slots,
                "actual_allocation_bytes": [item[1] for item in allocated],
                "actual_size_map": [int(item[1]) for item in sizes],
                "actual_address_map": [int(item[1]) for item in addresses],
                "actual_typed_accesses": [{"offset_bytes": item[1], "extent_bytes": item[2]} for item in accessed],
            }
        )
    return {
        "model": str(plan["model"]),
        "task_count": len(tasks),
        "problem": problem,
        "solution": solution,
        "actual_emitted_task_ids": emitted,
        "integer_verification": verified,
        "within_actor_reordered": reordered,
        "buffer_remap": records,
    }


def describe_kernel(kernel, args, spec):
    from tilelang import tvm

    compiled_input = Path(kernel.adapter.srcpath)
    source = compiled_input.read_text()
    if source != kernel.get_kernel_source():
        raise AssertionError("The preserved source must be the real Bisheng input")
    destination = args.output.parent / f"{args.output.stem}-compiled.cpp"
    destination.write_text(source)
    functions = []
    for symbol, function in kernel.artifact.device_mod.functions.items():
        if not isinstance(function, tvm.tir.PrimFunc):
            continue
        attrs = function.attrs or {}
        captured = {str(key): attrs[key] for key in attrs.keys() if str(key).startswith("tl.ascend_auto_schedule")}
        record = {"function": symbol.name_hint, "attribute_names": sorted(captured)}
        device_ir = args.output.parent / f"{args.output.stem}-{symbol.name_hint}-device-ir.json"
        device_ir.write_text(tvm.ir.save_json(function) + "\n")
        record["actual_device_ir"] = {"path": str(device_ir), "sha256": digest(device_ir)}
        if spec["strategy"] == "auto":
            record.update(inspect_function(function))
            path = args.output.parent / f"{args.output.stem}-{symbol.name_hint}-s8.json"
            path.write_text(tvm.ir.save_json(tvm.runtime.convert(captured)) + "\n")
            record["contract"] = {"path": str(path), "sha256": digest(path)}
        elif captured:
            raise AssertionError("Legacy references must not claim the auto ownership contract")
        functions.append(record)
    if not functions:
        raise AssertionError("No actual compiled device PrimFunc was retained")
    if spec["strategy"] == "auto":
        pools = {row["pool"] for function in functions for row in function["buffer_remap"] if row["versions"] == 2}
        if not set(spec.get("require_n2_pools", ())).issubset(pools):
            raise AssertionError(f"Required actual N2 pool was not materialized: observed {pools}")
        if spec.get("require_reorder") and not any(fn["within_actor_reordered"] for fn in functions):
            raise AssertionError("An actor-local non-source execution permutation is required")
    return {
        "actual_functions": functions,
        "requested_compile_flags": list(COMPILE_FLAGS),
        "resolved_compile_flags": list(kernel.compile_flags),
        "compiled_source": {"path": str(destination), "sha256": digest(destination)},
    }
