"""Shared S5 Vector fixtures, golden assertions and final-IR evidence."""

from __future__ import annotations

import sys

from _common import ROOT
from _s5_npu import exact_output_trace, pass_configs, physical_evidence, tensor_digest


def build_program(spec):
    sys.path.insert(0, str(ROOT / "testing/python/ascend_migration"))
    from test_s5_kernels import make_persistent_pipeline_kernel, make_pipeline_kernel

    keys = (
        "trips",
        "num_stages",
        "dtype",
        "tile",
        "tail",
        "manual_stages",
        "enable_offset",
        "legacy_zero_baseline",
    )
    factory = make_pipeline_kernel
    if "domain" in spec:
        factory = make_persistent_pipeline_kernel
        keys = ("domain", "cores", "group_size", "num_stages", "dtype", "tile")
    return factory(**{key: spec[key] for key in keys if key in spec})


def data_and_golden(spec, args, torch):
    """All input storage, including invalid Persistent slots, contains finite values."""
    from test_s5_kernels import persistent_layout

    dtype, tile = spec["dtype"], spec["tile"]
    layout = None
    if "domain" in spec:
        layout = persistent_layout(spec["domain"], spec["cores"], spec["group_size"])
        shape = (layout["rows"], layout["physical_columns"], tile)
    else:
        shape = ((spec["trips"] + 2) * tile + spec.get("tail", 0),)
    generator = torch.Generator().manual_seed(args.seed)
    a_host = torch.randn(shape, dtype=getattr(torch, dtype), generator=generator)
    b_host = torch.randn(shape, dtype=getattr(torch, dtype), generator=generator)
    if not torch.isfinite(a_host).all() or not torch.isfinite(b_host).all():
        raise AssertionError("The full input/canary domain must contain finite values")
    expected = torch.full_like(a_host, float("nan"))
    if layout is not None:
        rows, columns = spec["domain"]
        expected[:, :columns, :] = a_host[:, :columns, :] + b_host[:, :columns, :]
        indices = [
            (row * layout["physical_columns"] + column) * tile + element
            for row in range(rows)
            for column in range(columns)
            for element in range(tile)
        ]
    else:
        expected[:tile] = 7
        stop = (spec["trips"] + 1) * tile + spec.get("tail", 0)
        expected[tile:stop] = a_host[tile:stop] + b_host[tile:stop]
        indices = range(stop)
    a, b = a_host.npu(), b_host.npu()
    output = torch.empty_like(a)

    def check():
        tolerance = 1e-2 if dtype == "float16" else 1e-5
        torch.testing.assert_close(output.cpu(), expected, rtol=tolerance, atol=tolerance, equal_nan=True)

    metadata = {
        **spec,
        "input_sha256": {"A": tensor_digest(a_host, torch), "B": tensor_digest(b_host, torch)},
        "golden_sha256": tensor_digest(expected, torch),
        "output_sentinel": "NaN before every correctness launch; finite padded inputs",
    }
    if layout is not None:
        metadata["physical_layout"] = layout
    return (a, b, output), (output,), check, metadata, indices


def describe_kernel(kernel, spec, indices):
    physical = physical_evidence(kernel)
    actual = sorted({remap["versions"] for function in physical for remap in function["buffer_remap"]})
    required = spec.get("required_actual_versions")
    if spec["strategy"] == "constraint" and required is not None and required not in actual:
        raise AssertionError(f"Required actual {required} versions, observed {actual}")
    return {
        "physical_ir": physical,
        "actual_versions": actual,
        "exact_output_domain": exact_output_trace(kernel, indices),
    }


def run_case(spec, args, torch):
    import tilelang
    from ascend_repeat import repeat_launch

    program = build_program(spec)
    kernel = tilelang.compile(
        program,
        out_idx=[],
        target="ascendc",
        platform="A3",
        pass_configs=pass_configs(args, spec["strategy"]),
    )
    arguments, outputs, check, metadata, indices = data_and_golden(spec, args, torch)
    metadata.update(describe_kernel(kernel, spec, indices))
    repeat_launch(kernel, arguments, outputs, check, args.repeat)
    metadata["output_sha256"] = tensor_digest(outputs[0], torch)
    return metadata


def compare_pairs(report, parameters):
    observed = {case["name"]: case["evidence"] for case in report["cases"]}
    comparisons = []
    report["strategy_comparisons"] = comparisons
    bases = sorted({name.rsplit("_", 1)[0] for name in parameters if not name.startswith("l1_")})
    for base in bases:
        names = [base + "_" + strategy for strategy in ("legacy", "constraint")]
        values = [observed[name]["case_parameters"] for name in names]
        for name, value in zip(names, values):
            for key, expected in parameters[name].items():
                # JSON round-tripping changes tuple domain/manual stages to list.
                if isinstance(expected, tuple):
                    expected = list(expected)
                if value[key] != expected:
                    raise AssertionError(f"Wrong actual fixture parameters: {name}/{key}")
            kernels = observed[name]["kernels"]
            if len(kernels) != 1 or kernels[0]["resolved_pass_configs"].get("tl.ascend_pipeline_planner") != parameters[name]["strategy"]:
                raise AssertionError(f"Wrong resolved planner strategy: {name}")
        for key in ("input_sha256", "output_sha256", "golden_sha256"):
            if values[0][key] != values[1][key]:
                raise AssertionError(f"Paired strategies differ: {names}/{key}")
        comparisons.append(
            {
                "cases": names,
                "status": "PASS",
                "input_sha256": values[0]["input_sha256"],
                "output_sha256": values[0]["output_sha256"],
            }
        )
