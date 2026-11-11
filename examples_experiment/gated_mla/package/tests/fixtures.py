"""Exact original CPU fixture/reference helpers; no device work on import."""

import math
import torch
from gmla.metadata import metadata_for_reference, pack_metadata


def make_inputs(spec, dtype, seed, gate_case):
    b, h, tq, tk = (spec[key] for key in ("B", "H", "Tq", "Tk"))
    rng = torch.Generator(device="cpu").manual_seed(seed)

    def normal(shape, scale=1):
        return (torch.randn(shape, generator=rng, dtype=torch.float64) * scale).to(dtype)

    inputs = {
        "q_content": normal((b, tq, h, 128)),
        "q_aux": normal((b, tq, h, 64)),
        "latent_kv": normal((b, tk, 512)),
        "key_aux": normal((b, tk, 64)),
        "w_k": normal((h, 128, 512), 1 / math.sqrt(512)),
        "w_v": normal((h, 128, 512), 1 / math.sqrt(512)),
        "gate_logits": torch.zeros((b, tq, h, 128), dtype=dtype),
    }
    if gate_case == "head_specific":
        head = torch.arange(h, dtype=torch.float64).view(1, 1, h, 1)
        token = torch.arange(tq, dtype=torch.float64).view(1, tq, 1, 1)
        batch = torch.arange(b, dtype=torch.float64).view(b, 1, 1, 1)
        channel = torch.linspace(-1, 1, 128, dtype=torch.float64).view(1, 1, 1, 128)
        inputs["gate_logits"] = (((head % 9) - 4) * 0.5 + channel * 0.125 + token / 32 + batch / 4).to(dtype)
        inputs["gate_logits"][:, :, 0, :16] = -20
        inputs["gate_logits"][:, :, -1, 96:] = 20
    elif gate_case != "zero":
        raise ValueError(gate_case)
    return inputs


def references(inputs, metadata, golden, scale):
    kwargs = metadata_for_reference(metadata)
    # The established original seven-input reference functions are unmodified.
    expanded = golden.expanded_reference(**inputs, **kwargs, scale=scale)
    absorbed = golden.absorbed_reference(**inputs, **kwargs, scale=scale)
    torch.testing.assert_close(expanded, absorbed, rtol=2e-12, atol=2e-12)
    ql = torch.einsum("bthd,hdr->bthr", inputs["q_content"].double(), inputs["w_k"].double())
    z = torch.zeros_like(ql)
    for b, (qlen, kvlen, start, *_) in enumerate(metadata.tolist()):
        # Bounded per-query FP64 reference; no B*H*Tq*Tk score tensor.
        for token in range(qlen):
            visible = min(kvlen, start + token + 1)
            c, e = inputs["latent_kv"][b, :visible].double(), inputs["key_aux"][b, :visible].double()
            scores = (ql[b, token] @ c.T + inputs["q_aux"][b, token].double() @ e.T) * scale
            z[b, token] = torch.softmax(scores, dim=-1) @ c
    return {"expanded": expanded, "absorbed": absorbed, "q_latent": ql, "latent_out": z}


def token_inputs(cpu, token):
    result = dict(cpu)
    for name in ("q_content", "q_aux", "gate_logits"):
        result[name] = cpu[name][:, token : token + 1].contiguous()
    return result


def token_metadata(cpu_metadata, token, spec):
    rows = []
    for qlen, kvlen, start, *_ in cpu_metadata.tolist():
        active = int(token < qlen)
        position = start + token if active else min(start + token, kvlen)
        rows.append([active, kvlen, position])
    return pack_metadata(rows, batch=spec["B"], query_capacity=1, capacity=spec["Tk"])


def verify_read_only(device, cpu, device_metadata, cpu_metadata):
    for name, value in device.items():
        torch.testing.assert_close(value.cpu(), cpu[name], rtol=0, atol=0)
    torch.testing.assert_close(device_metadata.cpu(), cpu_metadata, rtol=0, atol=0)
