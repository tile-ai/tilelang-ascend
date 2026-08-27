"""Run msprof profiling on tnd_shared_prefix_fa and print kernel-level summary.

Usage:
    python run_msprof.py

Requires: msprof in PATH (source set_env.sh)
"""

import math
import os
import subprocess
import sys
import csv

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

from tilelang.profiler import do_bench

latency = do_bench(lambda: kernel(Q, KS, VS, KP, VP, bm))
print(f"do_bench latency: {latency:.4f} ms")

static_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tnd_static_run.py")
prof_dir = "/tmp/msprof_tnd"
os.makedirs(prof_dir, exist_ok=True)
app_cmd = f'"{sys.executable} {static_script}"'
cmd = f"msprof --output={prof_dir} --application={app_cmd}"
print(f"Running: {cmd}")
result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
if result.returncode != 0:
    print(f"msprof failed: {result.stderr[-500:]}")
    sys.exit(1)

prof_subdirs = [d for d in os.listdir(prof_dir) if d.startswith("PROF_")]
if not prof_subdirs:
    print("No PROF_ directory found")
    sys.exit(1)

prof_path = os.path.join(prof_dir, prof_subdirs[0])

op_summary = None
for root, _dirs, files in os.walk(prof_path):
    for f in files:
        if f.startswith("op_summary") and f.endswith(".csv"):
            op_summary = os.path.join(root, f)
            break

if not op_summary:
    print("No op_summary CSV found")
    sys.exit(1)

print("\n=== msprof Kernel Summary ===")
with open(op_summary) as f:
    reader = csv.DictReader(f)
    for row in reader:
        duration = float(row["Task Duration(us)"])
        aicore = float(row["aicore_time(us)"])
        aiv = float(row["aiv_time(us)"])
        vec = float(row["aiv_vec_time(us)"])
        mte2 = float(row["aiv_mte2_time(us)"])
        mte3 = float(row["aiv_mte3_time(us)"])
        cube_util = row.get("cube_utilization(%)", "N/A")
        blocks = row.get("Block Num", "N/A")
        print(f"  Total:       {duration:.1f} us ({duration / 1000:.3f} ms)")
        print(f"  AICore:      {aicore:.1f} us")
        print(f"  AIV:         {aiv:.1f} us")
        print(f"    Vec:       {vec:.1f} us")
        print(f"    MTE2:      {mte2:.1f} us")
        print(f"    MTE3:      {mte3:.1f} us")
        print(f"  Blocks:      {blocks}")
        print(f"  Cube util:   {cube_util}%")
