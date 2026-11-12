"""CPU checks for prepared boundaries and common-input component semantics."""

import importlib.util
import math
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from ._perf_tools import load_perf_tool


helper = load_perf_tool("aligned_execution")
PACKAGE = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def injected_candidate(monkeypatch):
    calls = {"validation": [], "selection": [], "execute": [], "core": [], "fia": []}
    package = ModuleType("gmla")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "gmla", package)
    modules = {}
    for name in ("_contracts", "_runtime", "_selection"):
        module = ModuleType("gmla." + name)
        monkeypatch.setitem(sys.modules, module.__name__, module)
        modules[name] = module
    spec = importlib.util.spec_from_file_location("_aligned_test_metadata", PACKAGE / "gmla/metadata.py")
    metadata_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metadata_module)
    monkeypatch.setitem(sys.modules, "gmla.metadata", metadata_module)

    def profile(batch, tq, heads, capacity, scale):
        return 128 if (batch, tq, heads, capacity) == (1, 128, 96, 128) and scale == 1 / math.sqrt(192) else 64

    def validate(*tensors, scale=None, block_n_selector=None):
        calls["validation"].append((tensors, scale, block_n_selector))
        q, _, cache, _, _, _, _, meta = tensors
        if q.ndim != 4 or q.shape[-1] != 128:
            raise ValueError("invalid public query")
        if tuple(meta.shape) != (q.shape[0], 8):
            raise ValueError("invalid public metadata")
        if scale is not None and scale <= 0:
            raise ValueError("invalid public scale")
        return q.shape[0], q.shape[1], q.shape[2], cache.shape[1], scale or 1 / math.sqrt(192)

    class FakeKernel:
        target = "pto"

        def get_kernel_source(self):
            return "actual selected fake kernel source"

    def uniform_attention(cache, lengths, starts, queries, heads):
        rows = []
        for query in range(queries):
            visible = min(lengths, starts + query + 1)
            rows.append(cache[0, :visible].double().mean(0).expand(heads, 512))
        return torch.stack(rows).unsqueeze(0)

    def make_bundle(block_n, target):
        def execute(*values, **kwargs):
            calls["execute"].append((block_n, values, kwargs))
            return values[0] + block_n

        def core(ql, qa, cache, auxiliary, metadata, **kwargs):
            calls["core"].append((block_n, ql, qa, kwargs))
            assert kwargs["ql_residual"] is None
            row = metadata[0].tolist()
            tq, heads = (1, ql.shape[1]) if ql.ndim == 3 else ql.shape[1:3]
            output = uniform_attention(cache, row[1], row[2], tq, heads).float()
            if ql.ndim == 3:
                output = output[:, 0]
            return output, {"attention": FakeKernel()}

        return SimpleNamespace(
            block_n=block_n,
            block_m=32,
            max_blocks=20,
            target=target,
            _forward_validated=execute,
            decode_module=SimpleNamespace(cube_decode=core),
            prefill_module=SimpleNamespace(cube_prefill=core),
        )

    bundles = {}

    def select(target, block_n):
        calls["selection"].append((target, block_n))
        return bundles.setdefault((target, block_n), make_bundle(block_n, target))

    modules["_contracts"].public_prefill_block_n = profile
    modules["_runtime"].validate_forward_inputs = validate
    modules["_selection"].selected_bundle = lambda target: select(target, 64)
    modules["_selection"].selected_prefill128_bundle = lambda target: select(target, 128)

    def fia(query, cache, value, **kwargs):
        calls["fia"].append((query, cache, value, kwargs))
        assert cache is value
        assert kwargs["block_size"] == 128
        assert kwargs["num_key_value_heads"] == 1
        flat = cache.view(1, -1, 512)
        heads = kwargs["num_query_heads"]
        if kwargs["input_layout"] == "BNSD_NBSD":
            assert heads == 128
            assert torch.count_nonzero(query[:, 96:]) == 0
            assert torch.count_nonzero(kwargs["query_rope"][:, 96:]) == 0
            assert kwargs["actual_seq_qlen"] is None
            visible = kwargs["actual_seq_kvlen"][0]
            output = uniform_attention(flat, visible, visible - 1, 1, heads)
            return output.permute(2, 0, 1, 3).to(query.dtype).contiguous(), None
        assert kwargs["input_layout"] == "TND"
        assert kwargs["actual_seq_qlen"] == [128]
        assert kwargs["actual_seq_kvlen"] == [128]
        mask = kwargs["atten_mask"]
        assert mask.dtype == torch.bool and tuple(mask.shape) == (2048, 2048)
        assert not mask[4, 4] and mask[4, 5] and not mask[5, 4]
        return uniform_attention(flat, 128, 0, 128, heads)[0].to(query.dtype), None

    return calls, metadata_module, SimpleNamespace(npu_fused_infer_attention_score_v2=fia)


def original_inputs(tq=1, tk=4096, dtype=torch.float16, heads=96):
    inputs = {
        "q_content": torch.zeros((1, tq, heads, 128), dtype=dtype),
        "q_aux": torch.zeros((1, tq, heads, 64), dtype=dtype),
        "latent_kv": torch.zeros((1, tk, 512), dtype=dtype),
        "key_aux": torch.zeros((1, tk, 64), dtype=dtype),
        "w_k": torch.zeros((heads, 128, 512), dtype=dtype),
        "w_v": torch.zeros((heads, 128, 512), dtype=dtype),
        "gate_logits": torch.zeros((1, tq, heads, 128), dtype=dtype),
    }
    inputs["latent_kv"][0, :, 0] = torch.arange(tk, dtype=torch.float32) / 16
    return inputs


def cpu_metadata(module, tq, tk, visible=None):
    return module.prepare_metadata([[tq, tk, 0 if tq > 1 else (visible or tk) - 1]], batch=1, query_capacity=tq, cache_capacity=tk)


def test_import_is_lazy_in_a_fresh_process():
    code = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location('isolated_aligned_execution', sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert 'torch' not in sys.modules
assert not any(name == 'gmla' or name.startswith('gmla.') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code, str(PACKAGE / "benchmark/aligned_execution.py")], check=True, timeout=30)


@pytest.mark.parametrize("tq,tk,block_n", [(1, 4096, 64), (128, 128, 128), (129, 129, 64)])
def test_prepared_reproduces_public_profile_and_reads_current_payload(injected_candidate, tq, tk, block_n):
    calls, meta_module, _ = injected_candidate
    data = original_inputs(tq, tk)
    meta = cpu_metadata(meta_module, tq, tk)
    runtime = SimpleNamespace(target="ascendc")
    prepared = helper.prepare_candidate(runtime, data, meta)
    assert len(calls["validation"]) == 1
    assert prepared.describe()["selected_block_n"] == block_n
    assert prepared.describe()["target"] == "ascendc"
    torch.testing.assert_close(prepared(), data["q_content"] + block_n)
    data["q_content"].fill_(7)
    torch.testing.assert_close(prepared(), torch.full_like(data["q_content"], 7 + block_n))
    assert len(calls["validation"]) == 1
    assert len(calls["selection"]) == 1
    assert len(calls["execute"]) == 2
    assert calls["validation"][0][2] is not None
    assert calls["execute"][0][1][-2] is meta


def test_prepared_rejects_bad_inputs_before_selection_or_execution(injected_candidate):
    calls, meta_module, _ = injected_candidate
    data = original_inputs()
    meta = cpu_metadata(meta_module, 1, 4096)
    with pytest.raises(ValueError, match="public scale"):
        helper.prepare_candidate(SimpleNamespace(target="pto"), data, meta, scale=-1)
    assert not calls["selection"] and not calls["execute"]
    with pytest.raises(ValueError, match="exact original seven"):
        helper.prepare_candidate(SimpleNamespace(target="pto"), {"q_content": data["q_content"]}, meta)


def test_prepared_binds_original_tensors_not_a_replacement_dictionary_entry(injected_candidate):
    _, meta_module, _ = injected_candidate
    data = original_inputs()
    meta = cpu_metadata(meta_module, 1, 4096)
    prepared = helper.prepare_candidate(SimpleNamespace(target="pto"), data, meta)
    original = data["q_content"]
    data["q_content"] = torch.ones_like(original)
    assert prepared.tensors[0] is original
    torch.testing.assert_close(prepared(), original + 64)


@pytest.mark.parametrize(
    "tq,tk,visible,dtype", [(1, 4096, 17, torch.float16), (1, 4096, 4096, torch.bfloat16), (128, 128, None, torch.float16)]
)
def test_core_uses_shared_operands_effective_prefix_and_causal_prefill(injected_candidate, tq, tk, visible, dtype):
    calls, meta_module, api = injected_candidate
    data = original_inputs(tq, tk, dtype)
    meta = cpu_metadata(meta_module, tq, tk, visible)
    shared = helper.prepare_shared_core(torch, SimpleNamespace(target="pto"), data, meta, data, meta, api=api)
    assert set(shared.methods) == {"candidate", "official_fia_core"}
    for name, function in shared.methods.items():
        output = function()
        assert tuple(output.shape) == (1, tq, 96, 512)
        assert output.dtype == (torch.float32 if name == "candidate" else dtype)
        torch.testing.assert_close(output, shared.golden.to(output.dtype), rtol=0, atol=0)
    expected = (torch.arange(tq).double() / 32) if tq > 1 else torch.tensor([(visible - 1) / 32]).double()
    torch.testing.assert_close(shared.golden[0, :, 0, 0], expected, rtol=0, atol=1e-12)
    shared.readonly()
    assert calls["core"][-1][3]["ql_residual"] is None
    report = shared.describe()
    assert report["production_BF16_HiLo"] is False
    assert report["candidate_matches_production_query_residual_policy"] is (dtype != torch.bfloat16)
    assert report["FIA_logical_heads"] == 96
    assert report["FIA_physical_heads"] == (128 if tq == 1 else 96)
    assert report["selected_block_n"] == (128 if tq > 1 else 64)
    sources = shared.kernel_sources()
    assert sources["attention"]["source"] == "actual selected fake kernel source"
    assert len(sources["attention"]["sha256"]) == 64


def test_core_does_not_cache_outputs_and_readonly_detects_payload_change(injected_candidate):
    _, meta_module, api = injected_candidate
    data = original_inputs()
    meta = cpu_metadata(meta_module, 1, 4096, 17)
    shared = helper.prepare_shared_core(torch, SimpleNamespace(target="pto"), data, meta, data, meta, api=api)
    before = {name: fn().clone() for name, fn in shared.methods.items()}
    data["latent_kv"][0, 0, 0] += 17
    for name, fn in shared.methods.items():
        after = fn()
        torch.testing.assert_close(after[..., 0] - before[name][..., 0], torch.ones_like(after[..., 0]), rtol=0, atol=0)
    with pytest.raises(AssertionError, match="latent_kv"):
        shared.readonly()


def test_core_readonly_covers_prepared_fia_padding_and_metadata(injected_candidate):
    _, meta_module, api = injected_candidate
    data = original_inputs()
    meta = cpu_metadata(meta_module, 1, 4096, 17)
    shared = helper.prepare_shared_core(torch, SimpleNamespace(target="pto"), data, meta, data, meta, api=api)
    shared.prepared_inputs["fia_query"][0, 127, 0, 0] = -0.0
    with pytest.raises(AssertionError, match="core_fia_query"):
        shared.readonly()
    shared.prepared_inputs["fia_query"].zero_()
    meta[0, 7] = 1
    with pytest.raises(AssertionError, match="metadata"):
        shared.readonly()


def test_core_rejects_invalid_metadata_before_preparing_device_data(injected_candidate):
    _, meta_module, api = injected_candidate
    data = original_inputs()
    meta = cpu_metadata(meta_module, 1, 4096, 17)
    meta[0, 5] = 1
    with pytest.raises(ValueError, match="reserved"):
        helper.prepare_shared_core(torch, SimpleNamespace(target="pto"), data, meta, data, meta, api=api)
    meta[0, 5], meta[0, 2] = 0, 18
    with pytest.raises(NotImplementedError, match="exactly the full or17"):
        helper.prepare_shared_core(torch, SimpleNamespace(target="pto"), data, meta, data, meta, api=api)


def test_core_rejects_an_unexpected_fia_result_without_fallback(injected_candidate):
    _, meta_module, _ = injected_candidate
    data = original_inputs()
    meta = cpu_metadata(meta_module, 1, 4096, 17)
    api = SimpleNamespace(npu_fused_infer_attention_score_v2=lambda *args, **kwargs: (torch.empty((1, 96, 1, 512)), None))
    shared = helper.prepare_shared_core(torch, SimpleNamespace(target="pto"), data, meta, data, meta, api=api)
    with pytest.raises(AssertionError, match="output shape/dtype changed"):
        shared.methods["official_fia_core"]()


def test_cpu_reference_rounds_query_then_computes_masked_attention(injected_candidate):
    _, meta_module, _ = injected_candidate
    data = original_inputs(2, 3, heads=1)
    data["q_content"][0, :, 0, :2] = torch.tensor([[1, 0.25], [0.5, 1]], dtype=torch.float16)
    data["w_k"][0, :2, 0] = torch.tensor([0.3333, 0.2], dtype=torch.float16)
    meta = meta_module.prepare_metadata([[2, 3, 0]], batch=1, query_capacity=2, cache_capacity=3)
    ql, golden = helper.shared_core_reference(torch, data, meta, scale=1.0)
    expected_ql = (data["q_content"][0, :, 0, :2].double() @ data["w_k"][0, :2, 0].double()).half()
    torch.testing.assert_close(ql[0, :, 0, 0], expected_ql, rtol=0, atol=0)
    assert golden[0, 0, 0, 0] == 0
    score = float(expected_ql[1]) / 16
    expected = (math.exp(score) / (1 + math.exp(score))) / 16
    assert float(golden[0, 1, 0, 0]) == pytest.approx(expected, rel=1e-14)
    assert ql.dtype == torch.float16 and golden.dtype == torch.float64
    assert not getattr(getattr(torch, "npu", None), "_initialized", False)
