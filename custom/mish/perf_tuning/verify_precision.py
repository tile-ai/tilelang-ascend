"""Quick precision verification for dynamic tiling on 20 cann-bench cases.

Verifies that smart-flatten + dynamic block_M/block_N selection maintains
precision across all cases (including non-aligned block_N like 67, 101).
"""

import sys
import os
from pathlib import Path

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
OP_DIR = HERE.parent
sys.path.insert(0, str(OP_DIR))

import torch
import torch_npu  # noqa: F401
from mish import mish_forward


CASES = [
    (1, [1024, 1024], "float16", (-1, 1)),
    (2, [2048, 2048], "float32", (-2, 2)),
    (3, [4096, 4096], "bfloat16", (-3, 3)),
    (4, [8192, 8192], "float16", (-10, 10)),
    (5, [8192, 8192], "float32", (-100, 100)),
    (6, [1023, 1023], "bfloat16", (-0.1, 0.1)),
    (7, [1009, 1021], "float16", (-1, 2)),
    (8, [1537, 769], "float32", (-5, 10)),
    (9, [363, 367, 373], "bfloat16", (-50, 100)),
    (10, [2049, 513], "float16", (-65504, 65504)),
    (11, [3, 7, 13, 4001], "float32", (-88, 88)),
    (12, [1000003], "bfloat16", (-1, 1)),
    (13, [11, 13, 17, 67, 67], "float32", (-1, 1)),
    (14, [3, 7, 11, 13, 1009], "float16", (-1, 1)),
    (15, [512, 2049], "float32", (-0.5, 0.5)),
    (16, [255, 8193], "bfloat16", (-1, 3)),
    (17, [4097, 511], "float16", (-1000, 1000)),
    (18, [2, 511, 2049], "float32", (-0.2, 0.2)),
    (19, [4, 255, 2049], "bfloat16", (-3, 6)),
    (20, [2, 3, 17, 1024, 101], "float32", (-20, 40)),
]

_DTYPE = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
_TOL = {"float16": (6.1e-5, 9.77e-4, 1e2), "bfloat16": (4.88e-4, 7.81e-3, 1e3), "float32": (7.63e-6, 1.22e-4, 1e0)}


def check(actual, golden, dtype_str):
    atol, rtol, max_limit = _TOL[dtype_str]
    a, g = actual.float(), golden.float()
    # Check NaN/Inf masks
    masks_ok = (
        torch.equal(torch.isnan(a), torch.isnan(g))
        and torch.equal(torch.isposinf(a), torch.isposinf(g))
        and torch.equal(torch.isneginf(a), torch.isneginf(g))
    )
    if not masks_ok:
        return False, 0.0, float("inf")
    finite = torch.isfinite(a) & torch.isfinite(g)
    if finite.sum() == 0:
        return True, 1.0, 0.0
    err = (a[finite] - g[finite]).abs()
    ratio = (err <= (atol + rtol * g[finite].abs())).float().mean().item()
    return ratio >= 0.99, ratio, err.max().item()


def main():
    torch.npu.set_device(0)
    torch.manual_seed(0)
    ok_count = 0
    for cid, shape, dt, vr in CASES:
        torch_dt = _DTYPE[dt]
        gen = torch.Generator().manual_seed(42 + cid)
        x = torch.rand(shape, generator=gen, dtype=torch.float32) * (vr[1] - vr[0]) + vr[0]
        x = x.to(torch_dt).npu()
        try:
            y = mish_forward(x)
            torch.npu.synchronize()
            ref = torch.nn.functional.mish(x)
            passed, ratio, max_err = check(y.cpu(), ref.cpu(), dt)
            tag = "PASS" if passed else "FAIL"
            if passed:
                ok_count += 1
            print(f"[{tag}] case {cid:>2} shape={str(shape):>24} {dt:>9} ratio={ratio:.4f} max_err={max_err:.3e}")
        except Exception as e:
            print(f"[FAIL] case {cid:>2} shape={str(shape):>24} {dt:>9}: {type(e).__name__}: {e}")
    print(f"\n{ok_count}/{len(CASES)} cases passed")


if __name__ == "__main__":
    main()
