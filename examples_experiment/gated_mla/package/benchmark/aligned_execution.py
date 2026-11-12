"""Explicit prepared boundaries for Gated MLA benchmark comparisons.

These helpers do not change the public API or production kernels. Importing this
module loads neither Torch nor the candidate package. Callers supply the real
Torch module and FIA API when preparing a shared-input component experiment.
"""

from dataclasses import dataclass
import hashlib
import math


NAMES = ("q_content", "q_aux", "latent_kv", "key_aux", "w_k", "w_v", "gate_logits")


def _ordered_inputs(inputs):
    if set(inputs) != set(NAMES):
        raise ValueError("exact original seven inputs are required")
    return tuple(inputs[name] for name in NAMES)


@dataclass(frozen=True)
class PreparedCandidate:
    """A fixed-shape, fixed-storage benchmark binding, validated at preparation."""

    bundle: object
    tensors: tuple
    contract: tuple

    def __call__(self):
        return self.bundle._forward_validated(*self.tensors, self.contract)

    def describe(self):
        return {
            "name": "candidate_prevalidated_full_chain",
            "scope": "original seven-input computation before W_O",
            "selected_block_n": self.bundle.block_n,
            "target": self.bundle.target,
            "public_api": False,
            "preparation_excluded": [
                "the unchanged public input preflight, performed once",
                "selection of the unchanged public profile bundle",
            ],
            "per_call_included": [
                "dynamic query projection, including production BF16 HiLo",
                "existing internal shape/layout/view guards and dispatch",
                "all attention launches, decode split merge and required tails",
                "dynamic value recovery and gate",
                "native per-call output/workspace allocation and kernel launch",
            ],
            "binding_contract": (
                "fixed shapes, dtype, device, storage and metadata; payloads are read on each call; "
                "the benchmark owner must keep the validated contract and check read-only inputs"
            ),
        }


def prepare_candidate(runtime, inputs, metadata, *, scale=None):
    """Hoist only public preflight and bundle selection; preserve execution code."""
    from gmla._contracts import public_prefill_block_n
    from gmla._runtime import validate_forward_inputs
    from gmla._selection import selected_bundle, selected_prefill128_bundle

    tensors = _ordered_inputs(inputs) + (metadata,)
    contract = validate_forward_inputs(*tensors, scale=scale, block_n_selector=public_prefill_block_n)
    bundle = selected_prefill128_bundle(runtime.target) if public_prefill_block_n(*contract) == 128 else selected_bundle(runtime.target)
    return PreparedCandidate(bundle, tensors, contract)


def _assert_bytes(torch, actual, expected, name):
    if tuple(actual.shape) != tuple(expected.shape) or actual.dtype != expected.dtype:
        raise AssertionError("readonly shape/dtype changed: " + name)
    raw = actual.detach().cpu().contiguous().view(torch.uint8)
    reference = expected.detach().cpu().contiguous().view(torch.uint8)
    if not torch.equal(raw, reference):
        raise AssertionError("readonly payload changed: " + name)


def _shared_golden(torch, ql_cpu, cpu_inputs, rows, scale):
    """Independent CPU64 latent readout for the actual common LP operands."""
    ql = ql_cpu.double()
    qa = cpu_inputs["q_aux"].double()
    cache = cpu_inputs["latent_kv"].double()
    auxiliary = cpu_inputs["key_aux"].double()
    batch, tq, heads, _ = ql.shape
    result = torch.zeros((batch, tq, heads, 512), dtype=torch.float64, device="cpu")
    for b, row in enumerate(rows):
        query_length, cache_length, start = row[:3]
        positions = torch.arange(query_length, device="cpu") + start
        visible = torch.arange(cache_length, device="cpu")[None, :] <= positions[:, None]
        scores = (
            torch.einsum("thr,kr->htk", ql[b, :query_length], cache[b, :cache_length])
            + torch.einsum("the,ke->htk", qa[b, :query_length], auxiliary[b, :cache_length])
        ) * scale
        probabilities = torch.softmax(scores.masked_fill(~visible[None], -torch.inf), dim=-1)
        result[b, :query_length] = torch.einsum("htk,kr->thr", probabilities, cache[b, :cache_length])
    return result


def shared_core_reference(torch, cpu_inputs, cpu_metadata, *, scale):
    """Return common rounded QL and its CPU64 latent reference without an NPU."""
    from gmla.metadata import validate_runtime_metadata_cpu

    _ordered_inputs(cpu_inputs)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    if any(value.device.type != "cpu" for value in cpu_inputs.values()):
        raise ValueError("shared-core reference requires CPU-owned inputs")
    q = cpu_inputs["q_content"]
    if q.ndim != 4 or q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("shared-core reference requires rank-4 FP16/BF16 query")
    batch, tq, heads, width = map(int, q.shape)
    if width != 128 or tuple(cpu_inputs["w_k"].shape) != (heads, 128, 512):
        raise ValueError("shared-core query/weight widths must be128/512")
    tk = int(cpu_inputs["latent_kv"].shape[1])
    validate_runtime_metadata_cpu(cpu_metadata, B=batch, Tq=tq, Tk=tk)
    ql_cpu = torch.einsum("bthd,hdr->bthr", q.double(), cpu_inputs["w_k"].double()).to(q.dtype).contiguous()
    return ql_cpu, _shared_golden(torch, ql_cpu, cpu_inputs, cpu_metadata.tolist(), float(scale))


@dataclass
class SharedCore:
    methods: dict
    golden: object
    prepared_inputs: dict
    _readonly: object
    _describe: dict
    _candidate_with_kernels: object

    def readonly(self):
        self._readonly()

    def describe(self):
        return dict(self._describe)

    def kernel_sources(self):
        """Execute an untimed audit call and read its actual compiled sources."""
        _, kernels = self._candidate_with_kernels()
        report = {}
        for name, kernel in kernels.items():
            source = kernel.get_kernel_source()
            report[name] = {
                "target": str(kernel.target),
                "source": source,
                "sha256": hashlib.sha256(source.encode()).hexdigest(),
            }
        return report


def prepare_shared_core(torch, runtime, device, metadata, cpu_inputs, cpu_metadata, *, api):
    """Prepare a shared-LP latent-attention component experiment.

    BF16 uses the existing no-query-residual core specialization. It is explicitly
    not the production BF16 HiLo path and cannot replace its full-chain evidence.
    The official side is an absorbed FIA-v2 component, not the source-prefill
    expanded K/V computation. All projection, packing and H2D happen here.
    """
    from gmla._contracts import public_prefill_block_n
    from gmla._selection import selected_bundle, selected_prefill128_bundle
    from gmla.metadata import validate_runtime_metadata_cpu

    _ordered_inputs(device)
    _ordered_inputs(cpu_inputs)
    q = cpu_inputs["q_content"]
    if not isinstance(q, torch.Tensor) or q.ndim != 4:
        raise ValueError("CPU q_content must be a rank-4 Tensor")
    if any(value.device.type != "cpu" for value in cpu_inputs.values()):
        raise ValueError("shared-core projection requires original CPU-owned inputs")
    batch, tq, heads, width = map(int, q.shape)
    tk = int(cpu_inputs["latent_kv"].shape[1])
    if (batch, heads, width) != (1, 96, 128) or (tq, tk) not in ((1, 4096), (128, 128)):
        raise NotImplementedError("shared core is scoped to B1/H96 decode4096/decode17/prefill128")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("shared core requires FP16 or BF16 operands")
    validate_runtime_metadata_cpu(cpu_metadata, B=batch, Tq=tq, Tk=tk)
    rows = cpu_metadata.tolist()
    if tq == 1:
        if rows[0][:3] not in ([1, 4096, 4095], [1, 4096, 16]):
            raise NotImplementedError("decode metadata must select exactly the full or17-token prefix")
    elif rows[0][:3] != [128, 128, 0]:
        raise NotImplementedError("prefill metadata must describe full causal prefill128")
    shapes = {
        "q_content": (batch, tq, heads, 128),
        "q_aux": (batch, tq, heads, 64),
        "latent_kv": (batch, tk, 512),
        "key_aux": (batch, tk, 64),
        "w_k": (heads, 128, 512),
        "w_v": (heads, 128, 512),
        "gate_logits": (batch, tq, heads, 128),
    }
    destination = device["q_content"].device
    for name, shape in shapes.items():
        cpu, value = cpu_inputs[name], device[name]
        if tuple(cpu.shape) != shape or cpu.dtype != q.dtype or not cpu.is_contiguous():
            raise ValueError("CPU shape/dtype/contiguity mismatch: " + name)
        if value.device != destination or not value.is_contiguous():
            raise ValueError("device/contiguity mismatch: " + name)
        _assert_bytes(torch, value, cpu, name)
    _assert_bytes(torch, metadata, cpu_metadata, "metadata")
    if metadata.device != destination:
        raise ValueError("metadata must share the input device")
    if not callable(getattr(api, "npu_fused_infer_attention_score_v2", None)):
        raise RuntimeError("real FIA v2 API is required; no fallback")

    scale = 1 / (192**0.5)
    block_n = public_prefill_block_n(batch, tq, heads, tk, scale)
    bundle = selected_prefill128_bundle(runtime.target) if block_n == 128 else selected_bundle(runtime.target)
    ql_cpu, golden = shared_core_reference(torch, cpu_inputs, cpu_metadata, scale=scale)
    ql = ql_cpu.to(destination)
    qa, cache, auxiliary = (device[name] for name in ("q_aux", "latent_kv", "key_aux"))
    if tq == 1:
        ql_native = ql.view(batch, heads, 512)
        qa_native = qa.view(batch, heads, 64)
        padded_heads = 1 << (heads - 1).bit_length()
        fia_query = torch.zeros((batch, padded_heads, 1, 512), dtype=q.dtype, device=destination)
        fia_aux = torch.zeros((batch, padded_heads, 1, 64), dtype=q.dtype, device=destination)
        fia_query[:, :heads].copy_(ql.view(batch, heads, 1, 512))
        fia_aux[:, :heads].copy_(qa.view(batch, heads, 1, 64))
        visible = [min(rows[0][1], rows[0][2] + 1)]
        mask = None
        common = dict(input_layout="BNSD_NBSD", sparse_mode=0, actual_seq_qlen=None, actual_seq_kvlen=visible)
    else:
        ql_native, qa_native = ql, qa
        padded_heads = heads
        fia_query = ql.view(tq, heads, 512)
        fia_aux = qa.view(tq, heads, 64)
        mask = torch.triu(torch.ones(2048, 2048, dtype=torch.bool, device="cpu"), diagonal=1).to(destination)
        common = dict(
            input_layout="TND",
            sparse_mode=3,
            actual_seq_qlen=[tq],
            actual_seq_kvlen=[tk],
            return_softmax_lse=False,
        )
    table = torch.arange(tk // 128, dtype=torch.int32, device="cpu").view(1, -1).to(destination)
    fia_cache = cache.view(tk // 128, 1, 128, 512)
    fia_cache_aux = auxiliary.view(tk // 128, 1, 128, 64)

    def candidate_with_kernels():
        kwargs = dict(block_m=bundle.block_m, block_n=bundle.block_n, target=runtime.target, ql_residual=None, return_kernels=True)
        if tq == 1:
            output, kernels = bundle.decode_module.cube_decode(ql_native, qa_native, cache, auxiliary, metadata, **kwargs)
        else:
            output, kernels = bundle.prefill_module.cube_prefill(
                ql_native, qa_native, cache, auxiliary, metadata, scale=scale, max_blocks=bundle.max_blocks, **kwargs
            )
        if output.dtype != torch.float32 or tuple(output.shape) != tuple(ql_native.shape):
            raise AssertionError("candidate shared-core output shape/dtype changed")
        return output.view(batch, tq, heads, 512), kernels

    def candidate():
        return candidate_with_kernels()[0]

    def official():
        result, _ = api.npu_fused_infer_attention_score_v2(
            fia_query,
            fia_cache,
            fia_cache,
            query_rope=fia_aux,
            key_rope=fia_cache_aux,
            block_table=table,
            block_size=128,
            atten_mask=mask,
            num_query_heads=padded_heads,
            num_key_value_heads=1,
            softmax_scale=scale,
            **common,
        )
        expected_shape = (padded_heads, batch, 1, 512) if tq == 1 else (tq, heads, 512)
        if result.dtype != q.dtype or tuple(result.shape) != expected_shape:
            raise AssertionError("official absorbed-core output shape/dtype changed; no fallback")
        return (result[:heads] if tq == 1 else result).view(batch, tq, heads, 512)

    prepared = {
        "ql": ql,
        "qa": qa,
        "cache": cache,
        "auxiliary_cache": auxiliary,
        "fia_query": fia_query,
        "fia_query_aux": fia_aux,
        "fia_cache": fia_cache,
        "fia_cache_aux": fia_cache_aux,
        "block_table": table,
    }
    if mask is not None:
        prepared["causal_mask"] = mask
    expected_original = {name: value.detach().cpu().clone() for name, value in cpu_inputs.items()}
    expected_metadata = cpu_metadata.detach().cpu().clone()
    expected_prepared = {name: value.detach().cpu().clone() for name, value in prepared.items()}

    def readonly():
        for name, value in device.items():
            _assert_bytes(torch, value, expected_original[name], name)
        _assert_bytes(torch, metadata, expected_metadata, "metadata")
        for name, value in prepared.items():
            _assert_bytes(torch, value, expected_prepared[name], "core_" + name)

    bf16 = q.dtype == torch.bfloat16
    description = {
        "name": "shared_LP_absorbed_attention_component",
        "scope": "QL/QA/C/E to latent512; query projection/value recovery/gate excluded on both sides",
        "selected_block_n": bundle.block_n,
        "target": runtime.target,
        "common_dtype": str(q.dtype),
        "production_BF16_HiLo": False,
        "candidate_matches_production_query_residual_policy": not bf16,
        "BF16_note": "component-only no-query-residual specialization; not production BF16 performance" if bf16 else "not applicable",
        "golden": "CPU64 attention using the actual common rounded-LP QL and original LP QA/C/E",
        "FIA_layout": common["input_layout"],
        "FIA_logical_heads": heads,
        "FIA_physical_heads": padded_heads,
        "FIA_source": "frozen historical absorbed FIA-v2 component; not expanded source-prefill",
        "visible_lengths": [min(rows[0][1], rows[0][2] + tq)] if tq == 1 else None,
        "metadata_rows": rows,
        "scale": scale,
        "candidate_output_dtype": "torch.float32",
        "official_output_dtype": str(q.dtype),
        "preparation_excluded": [
            "independent CPU64 Q@WK rounded once to common LP and copied to the device",
            "CPU-owned metadata validation and H2D",
            "zero-copy PA128 cache and query-layout views, identity page table H2D",
            "decode H96-to-H128 zero-head query/QA padding, preserving the source adapter layout",
            "static2048 causal mask construction/H2D for prefill",
        ],
        "per_call_included": [
            "candidate attention plus decode split merge, native output/workspace allocations and existing dispatch",
            "official absorbed FIA-v2 call and its native output allocation",
            "necessary output-shape/dtype checks and storage-preserving output view",
        ],
        "binding_contract": "fixed shared core operands and metadata; readonly checks occur outside timing",
    }
    return SharedCore(
        {"candidate": candidate, "official_fia_core": official}, golden, prepared, readonly, description, candidate_with_kernels
    )
