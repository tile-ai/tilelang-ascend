"""RoPE performance driver (TileLang side) for msprof.

Runs the TileLang RoPE kernel ``--repeats`` times on a single shape.
No timing, no correctness check — timing is done by the external ``msprof``
wrapper invoked from ``bench.sh``.

The AscendC baseline (aclnnRotaryPositionEmbedding) is a separate C++ driver
(``perf_rope_ascendc.cpp``); bench.sh dispatches to whichever side.

Usage (normally called by bench.sh, not directly)::

    python perf_rope.py --shape 4 64 128 128 --layout half
    python perf_rope.py --shape 4 4 64 128 128 --layout half --dtype bfloat16
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rope_half_interleaved import tilelang_rope  # noqa: E402

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _make_inputs(shape, dtype_str, kind, device):
    """Generate x, sin, cos on NPU (raw sin/cos, full rope: rope_dim == hidden_size)."""
    torch_dtype = _DTYPE_MAP[dtype_str]
    if kind == "tnd":
        bs, h, hs, rd = shape
        assert rd == hs, f"full rope only: rope_dim({rd}) must == hidden_size({hs})"
        x = torch.randn(bs, h, hs, dtype=torch_dtype, device=device)
        sin = torch.randn(bs, 1, hs, dtype=torch_dtype, device=device)
        cos = torch.randn(bs, 1, hs, dtype=torch_dtype, device=device)
    elif kind == "bsnd":
        b, s, h, hs, rd = shape
        assert rd == hs, f"full rope only: rope_dim({rd}) must == hidden_size({hs})"
        x = torch.randn(b, s, h, hs, dtype=torch_dtype, device=device)
        sin = torch.randn(1, s, 1, hs, dtype=torch_dtype, device=device)
        cos = torch.randn(1, s, 1, hs, dtype=torch_dtype, device=device)
    else:
        raise ValueError(f"Unknown kind: {kind}")
    return x, sin, cos


def run_kernel(shape, layout, dtype_str, kind, repeats, device="npu"):
    """Run TileLang RoPE ``repeats`` times. No timing — msprof wraps this."""
    torch.manual_seed(0)
    x, sin, cos = _make_inputs(shape, dtype_str, kind, device)
    x_work = x.clone()
    for _ in range(repeats):
        tilelang_rope(x_work, sin, cos, layout)
    torch.npu.synchronize()


def main():
    parser = argparse.ArgumentParser(description="RoPE TileLang perf driver (for msprof wrapping)")
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        required=True,
        help="4 ints (TND: BS H HS RD) or 5 ints (BSND: B S H HS RD). Full rope: RD must == HS.",
    )
    parser.add_argument("--layout", default="half", choices=["half", "interleaved"], help="RoPE layout")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"], help="Data type")
    parser.add_argument("--repeats", type=int, default=6, help="Kernel launches (1 warm-up + N-1 measured by msprof)")
    args = parser.parse_args()

    if len(args.shape) == 4:
        kind = "tnd"
    elif len(args.shape) == 5:
        kind = "bsnd"
    else:
        print(f"Error: --shape needs 4 (TND) or 5 (BSND) values, got {len(args.shape)}")
        sys.exit(1)

    run_kernel(args.shape, args.layout, args.dtype, kind, args.repeats)
    print("Test Passed!")


if __name__ == "__main__":
    main()
