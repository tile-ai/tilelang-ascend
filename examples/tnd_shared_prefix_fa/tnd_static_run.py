"""Standalone kernel run for msprof profiling (no assertion, no test)."""

import math
import os
import sys

import tilelang
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tnd_shared_prefix_fa_developer import tnd_shared_prefix_fa_developer, build_block_metadata

tilelang.disable_cache()
torch.manual_seed(0)

batch = 10
q_head = 14
kv_head = 2
head_dim = 64
shared_prefix_len = 24
private_q_lens = [150] * 10
block_M = 128
block_N = 64

total_q = shared_prefix_len + sum(private_q_lens)
total_private_kv = sum(private_q_lens)
max_private_kv_len = max(private_q_lens)
total_q_blocks = math.ceil(shared_prefix_len / block_M) + sum(math.ceil(l / block_M) for l in private_q_lens)
sm_scale = 1.0 / (head_dim**0.5)

Q = torch.randn(total_q, q_head, head_dim, dtype=torch.float16).npu()
KS = torch.randn(shared_prefix_len, kv_head, head_dim, dtype=torch.float16).npu()
VS = torch.randn(shared_prefix_len, kv_head, head_dim, dtype=torch.float16).npu()
KP = torch.randn(total_private_kv, kv_head, head_dim, dtype=torch.float16).npu()
VP = torch.randn(total_private_kv, kv_head, head_dim, dtype=torch.float16).npu()
bm = build_block_metadata(shared_prefix_len, private_q_lens, block_M, "npu")

kernel = tnd_shared_prefix_fa_developer(
    q_head=q_head,
    kv_head=kv_head,
    head_dim=head_dim,
    shared_prefix_len=shared_prefix_len,
    max_private_kv_len=max_private_kv_len,
    total_q=total_q,
    total_private_kv=total_private_kv,
    total_q_blocks=total_q_blocks,
    block_M=block_M,
    block_N=block_N,
    sm_scale=sm_scale,
)

torch.npu.synchronize()
output = kernel(Q, KS, VS, KP, VP, bm)
torch.npu.synchronize()
print("msprof run done")
