"""S6 device CLI and changing-input evidence; importing this file needs only stdlib."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys

from _common import ROOT, finish
from _npu_runner import digest
from _s5_npu import check_identity, load_cost_module, physical_evidence, tensor_digest


FIXTURES = {
    "readonly_f32": {"kind": "readonly", "trips": 5, "dtype": "float32", "tile": 64},
    "readonly_f16": {"kind": "readonly", "trips": 5, "dtype": "float16", "tile": 32},
    "carry_zero": {"kind": "carry", "trips": 0, "dtype": "float32", "tile": 64},
    "carry_one": {"kind": "carry", "trips": 1, "dtype": "float32", "tile": 64},
    "carry_many": {"kind": "carry", "trips": 5, "dtype": "float32", "tile": 64},
    "alias_views": {"kind": "views", "trips": 5, "dtype": "float32", "tile": 64},
    "dtype_views": {"kind": "dtype_views", "trips": 5, "dtype": "float32", "tile": 32},
    "dtype_private": {
        "kind": "dtype_private",
        "trips": 5,
        "dtype": "float32",
        "tile": 32,
        "tiles": 1,
    },
    "consecutive": {"kind": "consecutive", "trips": 3, "dtype": "float32", "tile": 64},
    "nested": {"kind": "nested", "trips": 3, "dtype": "float32", "tile": 64},
}
PARAMETERS = {
    f"{name}_{strategy}": {"tiles": 3, **spec, "strategy": strategy}
    for name, spec in FIXTURES.items()
    for strategy in ("legacy", "constraint")
}


def parser():
    result = argparse.ArgumentParser(description="S6 A3 changing-scratch reuse validation")
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
                ROOT / "build/ascend_migration_s6_native_build.json",
            )
        ),
    )
    result.add_argument(
        "--cost-profile",
        type=Path,
        default=Path(
            os.environ.get(
                "TL_ASCEND_PIPELINE_COST_PROFILE",
                ROOT / "build/ascend_migration_s5_cost_profile.json",
            )
        ),
    )
    result.add_argument("--_case", choices=("preflight", *PARAMETERS), help=argparse.SUPPRESS)
    return result


def validate(args, report):
    if sys.flags.optimize:
        return finish(args, report, "BLOCKED", "Python optimization disables mandatory assertions")
    if args.device < 0 or args.seed < 0 or args.repeat < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        return finish(args, report, "BLOCKED", "device/seed nonnegative and repeat/timeout positive required")
    try:
        args.cost_model = load_cost_module().load_cost_model(args.cost_profile)
    except (OSError, ValueError, TypeError, KeyError) as error:
        return finish(args, report, "BLOCKED", f"Calibrated A3 cost profile: {error}")
    report["cost_profile"] = {
        "path": str(args.cost_profile.resolve()),
        "file_sha256": digest(args.cost_profile),
        "profile_sha256": args.cost_model["profile_sha256"],
    }
    return None


def pass_configs(args, strategy):
    return {
        "tl.ascend_pipeline_planner": strategy,
        "tl.ascend_pipeline_cost_model": args.cost_model,
        "tl.ascend_auto_sync": True,
        "tl.ascend_memory_planning": True,
        "tl.ascend_auto_cross_core_sync": False,
        "tl.ascend_auto_sync_vs": False,
    }


def preflight(args, evidence, tilelang):
    check_identity(args, evidence, tilelang)


def requested_stages(spec):
    if spec["kind"] == "dtype_private":
        return 3 if spec["strategy"] == "constraint" else 0
    if spec["strategy"] == "legacy" and spec["kind"] in ("readonly", "views", "dtype_views"):
        return 0
    return 1


def program(spec):
    sys.path.insert(0, str(ROOT / "testing/python/ascend_migration"))
    from test_s6_kernels import make_dtype_version_kernel, make_reuse_kernel

    if spec["kind"] == "dtype_private":
        return make_dtype_version_kernel(
            trips=spec["trips"],
            num_stages=requested_stages(spec),
            manual_stages=spec["strategy"] == "constraint",
        )
    return make_reuse_kernel(
        **{key: value for key, value in spec.items() if key != "strategy"},
        num_stages=requested_stages(spec),
    )


def contract_evidence(kernel, args, strategy):
    """Preserve the actual schema3 TVM contract without guessing catalog fields."""
    from tilelang import tvm

    result = []
    for symbol, function in kernel.artifact.device_mod.functions.items():
        if not isinstance(function, tvm.tir.PrimFunc):
            continue
        manifest = (function.attrs or {}).get("tl.ascend_constraint_plan")
        if strategy == "legacy":
            result.append({"function": symbol.name_hint, "constraint_contract": False})
            continue
        if manifest is None or int(manifest["schema_version"]) != 3:
            raise AssertionError("S6 constraint compilation requires an actual schema3 manifest")
        contract = manifest["lifetime_contract"]
        if not contract:
            raise AssertionError("An empty lifetime contract cannot establish S6 readiness")
        snapshot = args.output.parent / f"{args.output.stem}-{symbol.name_hint}-lifetime.json"
        snapshot.write_text(tvm.ir.save_json(contract) + "\n")
        result.append(
            {
                "function": symbol.name_hint,
                "manifest_schema_version": 3,
                "lifetime_contract": {"path": str(snapshot), "sha256": digest(snapshot)},
                "contract_fields": sorted(str(key) for key in contract),
            }
        )
    if not result:
        raise AssertionError("No compiled device functions contain lifetime evidence")
    return result


def mixed_owner_evidence(kernel):
    """Check the private mixed-dtype owner by Var identity before any device launch."""
    from tilelang import tvm

    tir = tvm.tir
    analyzer = tvm.arith.Analyzer()

    def number(value):
        value = analyzer.simplify(value)
        if not isinstance(value, tir.IntImm):
            raise AssertionError(f"Unresolved mixed-owner byte geometry: {value}")
        return int(value)

    def pointers(statement):
        result = []

        def visit(node):
            if isinstance(node, tir.Call) and node.op.same_as(tvm.ir.Op.get("tir.tvm_access_ptr")):
                result.append(node)

        tir.stmt_functor.post_order_visit(statement, visit)
        return result

    records = []
    for symbol, function in kernel.artifact.device_mod.functions.items():
        if not isinstance(function, tir.PrimFunc):
            continue
        lifetime = function.attrs["tl.ascend_constraint_plan"]["lifetime_contract"]
        half_roots = []
        for task in lifetime["source_graph"]["tasks"]:
            for pointer in pointers(task["statement"]):
                if str(pointer.args[0].dtype) == "float16" and not any(pointer.args[1].same_as(value) for value in half_roots):
                    half_roots.append(pointer.args[1])
        if len(half_roots) != 1:
            raise AssertionError("The private half view must resolve to one canonical owner")
        owners = [
            row
            for row in lifetime["storage_catalog"]["storages"]
            if bool(row["physical_allocation"]) and row["data"].same_as(half_roots[0]) and number(row["capacity_bytes"]) == 128
        ]
        if len(owners) != 1 or str(owners[0]["allocation"].dtype) != "float32":
            raise AssertionError("The half view must share the actual 128-byte float32 allocation")
        owner = owners[0]
        provenance = function.attrs.get("tl.ascend_storage_source_data", {})

        def original(variable, provenance=provenance):
            for _ in range(len(provenance) + 1):
                if variable not in provenance or provenance[variable].same_as(variable):
                    return variable
                variable = provenance[variable]
            raise AssertionError("Cyclic mixed-owner provenance")

        remaps = [
            row
            for row in function.attrs["tl.ascend_pipeline_buffer_remap"]
            if row["logical_data"].same_as(owner["data"]) or original(row["logical_data"]).same_as(owner["source_data"])
        ]
        if len(remaps) != 1:
            raise AssertionError("The actual mixed owner requires exactly one materializer remap")
        remap = remaps[0]
        versions = number(remap["versions"])
        pitch = number(remap["version_stride_bytes"])
        total = number(remap["total_bytes"])
        if not 2 <= versions <= 3 or pitch != 128 or total != versions * pitch:
            raise AssertionError("Private mixed-dtype storage must use actual multiple 128-byte slots")
        if number(remap["version_span_bytes"]) != pitch:
            raise AssertionError("The mixed owner's slot span and byte pitch differ")
        datatypes = set()
        for pointer in pointers(function.body):
            if not pointer.args[1].same_as(remap["physical_data"]):
                continue
            dtype = str(pointer.args[0].dtype)
            datatypes.add(dtype)
            width = tvm.DataType(dtype).bits // 8
            if number(pointer.args[3]) * width != 128 or not analyzer.can_prove(tir.floormod(pointer.args[2] * width, pitch) == 0):
                raise AssertionError("Typed accesses disagree on the actual mixed-owner byte pitch")
        allocations = []

        def visit(node, remap=remap, allocations=allocations):
            if isinstance(node, tir.Allocate) and node.buffer_var.same_as(remap["physical_data"]):
                size = tvm.DataType(node.dtype).bits // 8
                for extent in node.extents:
                    size *= number(extent)
                allocations.append(size)

        tir.stmt_functor.post_order_visit(function.body, visit)
        if datatypes != {"float16", "float32"} or allocations != [total]:
            raise AssertionError("Both dtype views must use the same real final allocation")
        records.append(
            {
                "function": symbol.name_hint,
                "storage_id": number(owner["id"]),
                "owner_dtype": str(owner["allocation"].dtype),
                "consumer_dtypes": sorted(datatypes),
                "source_capacity_bytes": number(owner["capacity_bytes"]),
                "actual_versions": versions,
                "version_stride_bytes": pitch,
                "allocation_bytes": total,
                "identity_and_byte_geometry_checked": True,
            }
        )
    if not records:
        raise AssertionError("No final mixed-owner evidence was produced")
    return records


def run_case(name, args, torch):
    import tilelang
    from ascend_repeat import repeat_launch
    from _s5_npu import exact_output_trace

    spec = PARAMETERS[name]
    kernel = tilelang.compile(
        program(spec),
        out_idx=[],
        target="ascendc",
        platform="A3",
        pass_configs=pass_configs(args, spec["strategy"]),
    )
    steps = spec["trips"] * (2 if spec["kind"] in ("consecutive", "nested") else 1)
    tile, tiles = spec["tile"], spec["tiles"]
    private_dtype = spec["kind"] == "dtype_private"
    dtype_view = spec["kind"] in ("dtype_views", "dtype_private")
    input_elements = 32 if spec["kind"] == "views" else tile
    dtype = getattr(torch, spec["dtype"])
    a_shape = (steps, tile) if private_dtype else (tiles, max(steps, 1), input_elements)
    scratch_shape = (tile,) if private_dtype else (tiles, tile)
    output_shape = (steps + 2, tile) if private_dtype else (tiles, steps + 3, tile)
    a = torch.empty(a_shape, dtype=dtype, device="npu")
    scratch = torch.empty(scratch_shape, dtype=dtype, device="npu")
    output = torch.empty(output_shape, dtype=dtype, device="npu")
    indices = (
        range((steps + 1) * tile)
        if private_dtype
        else [(owner * (steps + 3) + epoch) * tile + lane for owner in range(tiles) for epoch in range(steps + 2) for lane in range(tile)]
    )
    metadata = {
        **spec,
        "requested_directives": {
            "num_stages": requested_stages(spec),
            "manual_stages": [0, 3, 3] if private_dtype and spec["strategy"] == "constraint" else [],
            "enable_offset": private_dtype and spec["strategy"] == "constraint",
        },
        "physical_ir": physical_evidence(kernel),
        "owner_epoch_evidence": contract_evidence(kernel, args, spec["strategy"]),
        "exact_output_domain": exact_output_trace(kernel, indices),
        "rounds": [],
    }
    metadata["comparison_role"] = (
        "serial N1 legacy baseline; legacy positive-stage overlapping writes unsupported"
        if spec["strategy"] == "legacy" and requested_stages(spec) == 0
        else "legacy positive-stage baseline"
        if spec["strategy"] == "legacy"
        else "constraint readiness/reuse candidate"
    )
    if private_dtype and spec["strategy"] == "constraint":
        metadata["mixed_dtype_owner"] = mixed_owner_evidence(kernel)
    generator = torch.Generator().manual_seed(args.seed)
    state = {}

    def reset():
        index = len(metadata["rounds"])
        if dtype_view:
            a_half = torch.randint(0, 32, (*a.shape[:-1], 64), generator=generator).to(torch.float16)
            scratch_half = torch.randint(0, 32, (*scratch.shape[:-1], 64), generator=generator).to(torch.float16)
            a_half = a_half / 128 + 0.25 + (index + 1) / 8192
            scratch_half = scratch_half / 128 + 0.25 + (index + 1) / 4096
            if not private_dtype:
                tags = torch.arange(tiles, dtype=torch.float16)
                a_half += tags.reshape(tiles, 1, 1) / 256
                scratch_half += tags.reshape(tiles, 1) / 128
            a_host = a_half.contiguous().view(torch.float32)
            scratch_host = scratch_half.contiguous().view(torch.float32)
            doubled = (a_half + a_half).contiguous().view(torch.float32)
            if not (torch.isfinite(a_host).all() and torch.isfinite(scratch_host).all() and torch.isfinite(doubled).all()):
                raise AssertionError("Packed half pairs must remain finite in both dtype views")
            expected = torch.full(output.shape, float("nan"), dtype=dtype)
            if private_dtype:
                expected[0, :] = scratch_host
                expected[1 : steps + 1, :] = doubled
            else:
                expected[:, 0, :] = scratch_host
                expected[:, 1 : steps + 1, :] = doubled[:, :steps, :]
                expected[:, steps + 1, :] = doubled[:, steps - 1, :] if steps else scratch_host
            a.copy_(a_host)
            scratch.copy_(scratch_host)
            state.update(
                expected=expected,
                input_sha256={
                    "A": tensor_digest(a_host, torch),
                    "Scratch": tensor_digest(scratch_host, torch),
                },
            )
            return
        a_host = torch.randint(-16, 17, a.shape, generator=generator).to(dtype) / 1024
        scratch_host = torch.randint(-16, 17, scratch.shape, generator=generator).to(dtype) / 128
        # Independent tile and launch tags make stale data numerically visible.
        tags = torch.arange(tiles, dtype=dtype).reshape(tiles, 1)
        scratch_host += tags / 16 + (index + 1) / 1024
        a_host += tags.reshape(tiles, 1, 1) / 128 + (index + 1) / 4096
        a.copy_(a_host)
        scratch.copy_(scratch_host)
        expected = torch.full(output.shape, float("nan"), dtype=dtype)
        value = scratch_host.clone() if spec["kind"] == "views" else scratch_host + scratch_host
        expected[:, 0, :] = value
        for step in range(steps):
            if spec["kind"] == "views":
                value[:, 16:48] = a_host[:, step, :] + a_host[:, step, :]
                current = value
            elif spec["kind"] == "readonly":
                current = a_host[:, step, :] + value
            else:
                value = a_host[:, step, :] + value
                current = value
            expected[:, step + 1, :] = current
        expected[:, steps + 1, :] = value
        state.update(
            expected=expected,
            input_sha256={
                "A": tensor_digest(a_host, torch),
                "Scratch": tensor_digest(scratch_host, torch),
            },
        )

    def check():
        tolerance = 0 if dtype_view else 1e-2 if spec["dtype"] == "float16" else 1e-5
        torch.testing.assert_close(output.cpu(), state["expected"], rtol=tolerance, atol=tolerance, equal_nan=True)
        metadata["rounds"].append(
            {
                "index": len(metadata["rounds"]),
                "golden_status": "PASS",
                "input_sha256": state["input_sha256"],
                "output_sha256": tensor_digest(output, torch),
                "golden_sha256": tensor_digest(state["expected"], torch),
            }
        )

    repeat_launch(kernel, (a, scratch, output), (output,), check, args.repeat, reset=reset)
    if len(metadata["rounds"]) != args.repeat:
        raise AssertionError("Every actual launch requires its own golden and hash record")
    metadata["output_sentinel"] = "NaN every launch; complete finite scratch initialization and changing tile/launch tags"
    return metadata


def compare_pairs(report):
    results = {item["name"]: item["evidence"] for item in report["cases"]}
    comparisons = report["strategy_comparisons"] = []
    for fixture in FIXTURES:
        names = [fixture + "_" + strategy for strategy in ("legacy", "constraint")]
        left, right = [results[name]["case_parameters"] for name in names]
        if len(left["rounds"]) != report["repeat"] or len(right["rounds"]) != report["repeat"]:
            raise AssertionError("Incomplete per-launch strategy evidence")
        for index, (a, b) in enumerate(zip(left["rounds"], right["rounds"])):
            if a != b or a["golden_status"] != "PASS" or a["index"] != index:
                raise AssertionError(f"Paired inputs/output/golden differ: {names}/{a['index']}")
        for key in ("A", "Scratch"):
            if len({r["input_sha256"][key] for r in left["rounds"]}) != report["repeat"]:
                raise AssertionError(f"{key} must actually change between launches")
        for name in names:
            evidence = results[name]
            for key, value in PARAMETERS[name].items():
                if evidence["case_parameters"][key] != value:
                    raise AssertionError(f"Wrong actual reuse fixture parameters: {name}/{key}")
            if evidence["kernels"][0]["resolved_pass_configs"].get("tl.ascend_pipeline_planner") != PARAMETERS[name]["strategy"]:
                raise AssertionError("Wrong actual strategy in paired reuse validation")
        if results[names[0]]["kernels"][0]["resolved_flags"] != results[names[1]]["kernels"][0]["resolved_flags"]:
            raise AssertionError("Paired reuse strategies use different compiler flags")
        comparisons.append({"cases": names, "status": "PASS", "compared_launches": report["repeat"]})
