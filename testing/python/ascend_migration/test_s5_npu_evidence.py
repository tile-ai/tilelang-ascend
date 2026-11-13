"""Check the device report extractor against real lowered IR, using synthetic costs."""

from pathlib import Path
from types import SimpleNamespace

import tilelang
from tilelang import tvm
import pytest

from test_s5_kernels import make_pipeline_kernel
from test_s5_versions_ir import configs


def test_physical_report_uses_real_remaps_and_final_allocations(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts/ascend_migration"))
    from _s5_npu import physical_evidence

    with tvm.transform.PassContext(config=configs()):
        artifact = tilelang.lower(make_pipeline_kernel(), target="ascendc", platform="A3")
    report = physical_evidence(SimpleNamespace(artifact=artifact))
    assert report and any(function["buffer_remap"] for function in report)
    for function in report:
        assert function["allocations"]
        assert function["address_map"] and function["size_map"]
        for remap in function["buffer_remap"]:
            assert remap["total_bytes"] == remap["versions"] * remap["version_stride_bytes"]
            assert remap["version_span_bytes"] <= remap["version_stride_bytes"]


@pytest.mark.parametrize("strategy", ["legacy", "constraint"])
@pytest.mark.parametrize("trip", [0, 1, 2, 8])
def test_final_pipeline_store_trace_covers_only_real_output(monkeypatch, strategy, trip):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts/ascend_migration"))
    from _s5_npu import exact_output_trace

    configuration = configs()
    configuration["tl.ascend_pipeline_planner"] = strategy
    with tvm.transform.PassContext(config=configuration):
        artifact = tilelang.lower(
            make_pipeline_kernel(
                trips=trip,
                num_stages=0 if trip == 0 and strategy == "legacy" else 3,
                legacy_zero_baseline=trip == 0 and strategy == "legacy",
            ),
            target="ascendc",
            platform="A3",
        )
    result = exact_output_trace(SimpleNamespace(artifact=artifact), range((trip + 1) * 64))
    assert result["maximum_multiplicity"] == 1


@pytest.mark.parametrize("strategy", ["legacy", "constraint"])
@pytest.mark.parametrize("domain,cores,group", [((1, 1), 1, 1), ((1, 3), 8, 2), ((3, 10), 8, 4), ((9, 1), 4, 1)])
def test_final_persistent_store_trace_interprets_guarded_waves(monkeypatch, strategy, domain, cores, group):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts/ascend_migration"))
    from _s5_npu import exact_output_trace
    from test_s5_kernels import make_persistent_pipeline_kernel, persistent_layout

    layout = persistent_layout(domain, cores, group)
    configuration = configs()
    configuration["tl.ascend_pipeline_planner"] = strategy
    with tvm.transform.PassContext(config=configuration):
        artifact = tilelang.lower(
            make_persistent_pipeline_kernel(domain, cores, group, 0 if strategy == "legacy" else 2),
            target="ascendc",
            platform="A3",
        )
    expected = [
        (row * layout["physical_columns"] + column) * 64 + element
        for row in range(domain[0])
        for column in range(domain[1])
        for element in range(64)
    ]
    result = exact_output_trace(SimpleNamespace(artifact=artifact), expected)
    assert result["maximum_multiplicity"] == 1


def test_final_trace_rejects_a_damaged_persistent_guard_without_device_launch(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts/ascend_migration"))
    from _s5_npu import exact_output_trace
    from test_s5_kernels import make_persistent_pipeline_kernel, persistent_layout

    layout = persistent_layout((3, 10), 8, 4)
    with tvm.transform.PassContext(config=configs()):
        artifact = tilelang.lower(make_persistent_pipeline_kernel(), target="ascendc", platform="A3")
    damaged = tvm.IRModule()
    for symbol, function in artifact.device_mod.functions.items():
        body = tvm.tir.stmt_functor.ir_transform(function.body, None, lambda node: node.then_case, ["tir.IfThenElse"])
        damaged[symbol] = function.with_body(body)
    indices = [
        (row * layout["physical_columns"] + column) * 64 + element for row in range(3) for column in range(10) for element in range(64)
    ]
    with pytest.raises(AssertionError):
        exact_output_trace(SimpleNamespace(artifact=SimpleNamespace(device_mod=damaged)), indices)
