"""msprof benchmark script for mhc_pre pipeline.

Usage:
  msprof op --kernel-name="main_kernel" --output=/tmp/opencode/msprof_pre python3 examples/mhc_pre/bench_msprof.py
"""

import tilelang
import torch

from examples.mhc_pre.example_mhc_pre import (
    generate_full_test_data,
    mhc_pre,
    prepare_fn,
)

tilelang.disable_cache()

n, h, hc = 4096, 7168, 4

data = generate_full_test_data(n, h, hc)
fn = data["fn"]
fn_packed = prepare_fn(fn, hc)

print("init successful!")

post_mix, comb_mix, layer_input = mhc_pre(**data, fn_packed=fn_packed)

for _ in range(10):
    post_mix, comb_mix, layer_input = mhc_pre(**data, fn_packed=fn_packed)

print("done!")
