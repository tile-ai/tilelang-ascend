"""Frozen original protocol definitions; see protocol_manifest.json."""

import math
import torch
from gmla import metadata_for_reference

CASES = {
    "small_tails": {"B": 2, "H": 3, "Tk": 65, "metadata": [[[1, 65, 16], [0, 0, 0]], [[1, 65, 64], [1, 32, 0]]]},
    "model_4k": {"B": 1, "H": 96, "Tk": 4096, "metadata": [[[1, 4096, 4095]], [[1, 4096, 16]]]},
    "batch_tail": {
        "B": 4,
        "H": 96,
        "Tk": 1025,
        "metadata": [[[1, 1025, 0], [1, 1025, 16], [1, 1025, 1024], [0, 0, 0]], [[1, 17, 16], [0, 0, 0], [1, 1025, 1023], [1, 513, 512]]],
    },
}


def make_inputs(spec, seed, dtype, gate_case):
    b, h, tk = spec["B"], spec["H"], spec["Tk"]
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def normal(shape, scale=1):
        return (torch.randn(shape, generator=generator, dtype=torch.float64) * scale).to(dtype)

    values = {
        "q_content": normal((b, 1, h, 128)),
        "q_aux": normal((b, 1, h, 64)),
        "latent_kv": normal((b, tk, 512)),
        "key_aux": normal((b, tk, 64)),
        "w_k": normal((h, 128, 512), 1 / math.sqrt(512)),
        "w_v": normal((h, 128, 512), 1 / math.sqrt(512)),
        "gate_logits": torch.zeros((b, 1, h, 128), dtype=dtype),
    }
    if gate_case == "head_specific":
        heads = torch.arange(h, dtype=torch.float64).reshape(1, 1, h, 1)
        batch = torch.arange(b, dtype=torch.float64).reshape(b, 1, 1, 1)
        channel = torch.linspace(-1, 1, 128, dtype=torch.float64).reshape(1, 1, 1, 128)
        values["gate_logits"] = (((heads % 9) - 4) * 0.5 + channel * 0.125 + batch * 0.25).to(dtype)
        values["gate_logits"][:, :, 0, :16] = -20
        values["gate_logits"][:, :, -1, 96:] = 20
    elif gate_case != "zero":
        raise ValueError(gate_case)
    return values


def goldens(inputs, metadata, golden):
    kwargs = metadata_for_reference(metadata)
    expanded = golden.expanded_reference(**inputs, **kwargs)
    absorbed = golden.absorbed_reference(**inputs, **kwargs)
    torch.testing.assert_close(expanded, absorbed, rtol=2e-12, atol=2e-12)
    ql = torch.einsum("bthd,hdr->bthr", inputs["q_content"].double(), inputs["w_k"].double())
    latent = torch.zeros_like(ql)
    for b, (active, kvlen, start, *_) in enumerate(metadata.tolist()):
        if active:
            visible = min(kvlen, start + 1)
            cache = inputs["latent_kv"][b, :visible].double()
            aux = inputs["key_aux"][b, :visible].double()
            scores = (ql[b, 0] @ cache.T + inputs["q_aux"][b, 0].double() @ aux.T) / math.sqrt(192)
            latent[b, 0] = torch.softmax(scores, dim=-1) @ cache
    return {"expanded": expanded, "absorbed": absorbed, "q_latent": ql, "latent_out": latent}
