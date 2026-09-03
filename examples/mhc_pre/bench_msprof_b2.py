"""msprof script for B2 Sinkhorn ONLY (data prepared by PyTorch, no other kernel runs).

Usage:
  msprof op --kernel-name="main_kernel" --output=/tmp/opencode/msprof_b2 python3 examples/mhc_pre/bench_msprof_b2.py
"""

import tilelang
import torch

from examples.mhc_pre.example_mhc_pre import (
    generate_full_test_data,
    mhc_pre_ref,
    _get_kernel,
)

tilelang.disable_cache()

n, h, hc = 4096, 7168, 4
data = generate_full_test_data(n, h, hc)

# Use PyTorch ref to prepare B2 input (mixes), avoid running A1/A2/B1 kernels
_, _, _ = mhc_pre_ref(**data)  # warm up torch

# Compute mixes via PyTorch (same as mhc_pre_ref up to sinkhorn step)
residual = data["residual"]
fn = data["fn"]
hc_mult = hc
hidden = h
hc_mult3 = hc_mult * (2 + hc_mult)

residual_flat = residual.view(n, hc_mult * hidden).float()
fn_bf16 = fn.bfloat16()
out = residual_flat @ fn_bf16.float().T
sqrsum = residual_flat.square().sum(-1)
rms = (sqrsum / (hc_mult * hidden) + data["rms_eps"]).rsqrt()
mixes = out * rms.unsqueeze(-1)

hc_scale_exp = torch.cat(
    [
        data["hc_scale"][0].expand(hc_mult),
        data["hc_scale"][1].expand(hc_mult),
        data["hc_scale"][2].expand(hc_mult * hc_mult),
    ]
)
mixes = mixes * hc_scale_exp + data["hc_base"]

mixes = mixes.npu()

# B2 Sinkhorn kernel
sinkhorn_kernel = _get_kernel(
    "sinkhorn", hc, data["sinkhorn_repeat"], data["hc_pre_eps"], data["hc_sinkhorn_eps"], data["hc_post_mult_value"]
)

print("init successful!")

pre_mix, post_mix, comb_mix = sinkhorn_kernel(mixes, data["hc_scale"], data["hc_base"])

for _ in range(10):
    pre_mix, post_mix, comb_mix = sinkhorn_kernel(mixes, data["hc_scale"], data["hc_base"])

print("done!")
