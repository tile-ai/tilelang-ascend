"""Exact original CPU fixture/reference helpers; no device work on import."""

import math
import torch


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
