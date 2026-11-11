"""CPU checks for source integrity and measurement-input immutability."""

import hashlib
from types import SimpleNamespace

import pytest
import torch

from ._perf_tools import load_perf_tool


_perf_tool = load_perf_tool("paired_benchmark")
assert_same_bytes = _perf_tool.assert_same_bytes
baseline_identity = _perf_tool.baseline_identity
finalize = _perf_tool.finalize
make_readonly = _perf_tool.make_readonly


def test_baseline_source_pin_rejects_modified_adapter(tmp_path):
    root = tmp_path / "benchmark"
    root.mkdir()
    adapter = root / "adapter_decode.py"
    adapter.write_text("pinned baseline\n")
    checksum = hashlib.sha256(adapter.read_bytes()).hexdigest()
    (tmp_path / "FILELIST.sha256").write_text(checksum + "  benchmark/adapter_decode.py\n")
    assert baseline_identity(root, False, False)["sources"]["adapter_decode.py"] == checksum
    adapter.write_text("changed baseline\n")
    with pytest.raises(RuntimeError, match="baseline source changed"):
        baseline_identity(root, False, False)


def test_baseline_source_must_be_registered(tmp_path):
    root = tmp_path / "benchmark"
    root.mkdir()
    (root / "adapter_decode.py").write_text("baseline\n")
    (tmp_path / "FILELIST.sha256").write_text("")
    with pytest.raises(RuntimeError, match="unregistered"):
        baseline_identity(root, False, False)


def test_source_mismatch_overrides_successful_return():
    report, saved = {"status": "cpu_reference_passed"}, []

    def run():
        try:
            return 0
        finally:
            finalize(report, lambda: saved.append(dict(report)), lambda: {"source": "changed"}, {"source": "pinned"})

    with pytest.raises(RuntimeError, match="sources changed"):
        run()
    assert saved[-1]["status"] == "failed"


def test_source_read_failure_is_fatal_and_saved():
    report, saved = {"status": "balanced_benchmark_completed"}, []

    def missing():
        raise FileNotFoundError("source removed")

    with pytest.raises(RuntimeError, match="could not be rechecked"):
        finalize(report, lambda: saved.append(dict(report)), missing, {})
    assert saved[-1]["status"] == "failed"


def test_matching_sources_keep_original_status():
    report, saved = {"status": "balanced_benchmark_completed"}, []
    finalize(report, lambda: saved.append(dict(report)), lambda: {"x": "same"}, {"x": "same"})
    assert saved == [{"status": "balanced_benchmark_completed"}]


def test_signed_zero_mutation_is_not_hidden_by_numeric_equality():
    a, b = torch.tensor([0.0]), torch.tensor([-0.0])
    assert torch.equal(a, b)
    with pytest.raises(AssertionError, match="payload changed"):
        assert_same_bytes(torch, a, b, "input")


def decode_check():
    cpu = {"w_v": torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)}
    device = {k: v.clone() for k, v in cpu.items()}
    meta = torch.tensor([[1, 4096, 16, 0, 0, 0, 0, 0]], dtype=torch.int32)
    table = torch.arange(32, dtype=torch.int32).view(1, 32)
    source = SimpleNamespace(constants={"w_uv": cpu["w_v"].transpose(1, 2).contiguous().clone(), "block_table": table.clone()})
    return make_readonly(torch, device, cpu, meta, meta.clone(), source, False, table), device, table, source


def test_historical_table_mutation_is_detected():
    check, _, table, _ = decode_check()
    check()
    table[0, 4] = -1
    with pytest.raises(AssertionError, match="historical_block_table"):
        check()


def test_source_table_mutation_is_detected():
    check, _, _, source = decode_check()
    check()
    source.constants["block_table"][0, 4] = -1
    with pytest.raises(AssertionError, match="source_block_table"):
        check()


def test_input_mutation_is_detected():
    check, device, _, _ = decode_check()
    check()
    device["w_v"][0, 0, 0] = 2
    with pytest.raises(AssertionError, match="w_v"):
        check()


def test_historical_prefill_mask_is_not_confused_with_source_mask():
    cpu = {"w_k": torch.zeros((96, 128, 512), dtype=torch.float16), "w_v": torch.zeros((96, 128, 512), dtype=torch.float16)}
    device = {k: v.clone() for k, v in cpu.items()}
    meta = torch.tensor([[128, 128, 0, 0, 0, 0, 0, 0]], dtype=torch.int32)
    mask = torch.triu(torch.ones(2048, 2048, dtype=torch.bool), diagonal=1)
    table = torch.zeros((1, 1), dtype=torch.int32)
    source = SimpleNamespace(
        constants={"packed_w_kv": torch.cat((cpu["w_k"], cpu["w_v"]), dim=1).view(96 * 256, 512), "causal_mask": mask.to(torch.int8)}
    )
    check = make_readonly(torch, device, cpu, meta, meta.clone(), source, True, table, mask)
    check()
    mask[0, 1] = False
    with pytest.raises(AssertionError, match="historical_causal_mask"):
        check()
