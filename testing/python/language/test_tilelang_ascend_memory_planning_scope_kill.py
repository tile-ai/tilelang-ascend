"""Regression test: memory planning must not reuse a buffer that is defined
before an inner loop and read inside it, when a Let-bound scalar read
(e.g. ``wv = wF[3]``) precedes the loop.

Root cause (ISSUE-R7-1, depthwise_conv2d case20): ReorderKillPoints moves the
KILL of such a buffer onto the enclosing LetStmt — semantically the *end* of
the let scope — but FindEventIndex resolved the LetStmt to its *opening*
linear index (scope open/close entries share the same stmt object pointer).
The resulting inverted live interval [gen > kill] made the linear-scan
allocator treat the buffer as dead; once total UB demand exceeded the 192KB
limit, a loop-internal buffer (bigB) was placed on top of slabA/wF, silently
corrupting data.

This test compiles a minimal kernel with the same skeleton and asserts
  1. no overlap between the outer-read buffer (slabA) / scalar-read weight
     buffer (wF) and the loop-internal buffer (bigB);
  2. NPU numerical correctness (slabA values read inside the loop stay
     intact across iterations).
"""

import re

import torch

import tilelang
import tilelang.language as T

PASS_AUTO = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

NUMA = 30000  # fp32 elements (120KB slab, read inside inner loop)
NUMB = 9216  # fp32 elements (36KB loop-internal pool)
NUMC = 9216  # fp32 elements (36KB, genuinely dead before the inner loop —
# the legitimate reuse donor once liveness is correct)
NG = 16


def _offsets(src: str) -> dict[str, tuple[int, int]]:
    out = {}
    for line in src.split("\n"):
        m = re.search(r"auto\s+(\w+)\s*=.*GetWithOffset<[^>]+>\(\s*(\d+)\s*,\s*(\d+)\s*\)", line)
        if m:
            out[m.group(1)] = (int(m.group(3)), int(m.group(2)))
    return out


def _overlap(a, b):
    return not (a[0] + a[1] * 4 <= b[0] or b[0] + b[1] * 4 <= a[0])


def _build():
    @tilelang.jit(out_idx=[2], pass_configs=PASS_AUTO, target="ascendc")
    def kernel():
        @T.prim_func
        def main(x: T.Tensor((NUMA,), "float32"), w: T.Tensor((8,), "float32"), y: T.Tensor((NG, 8), "float32")):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                slabA = T.alloc_ub((NUMA,), "float32")
                wT = T.alloc_ub((8,), "float32")
                wF = T.alloc_ub((8,), "float32")
                bigB = T.alloc_ub((NUMB,), "float32")
                bigC = T.alloc_ub((NUMC,), "float32")
                tmp = T.alloc_ub((1024,), "float32")
                with T.Scope("V"):
                    for _t in T.serial(2):
                        T.copy(w[0:8], wT[0:8])
                        T.set_flag("mte2", "v", 0)
                        T.wait_flag("mte2", "v", 0)
                        T.tile.add(wF, wT, T.float32(0.0))
                        wv = wF[3]  # Let-bound scalar read before inner loop
                        T.tile.clear(slabA)
                        T.set_flag("v", "mte2", 0)
                        T.wait_flag("v", "mte2", 0)
                        T.copy(x[0:NUMA], slabA[0:NUMA])
                        T.set_flag("mte2", "v", 0)
                        T.wait_flag("mte2", "v", 0)
                        # bigC: written and consumed BEFORE the inner loop —
                        # legitimately reusable by bigB once liveness is right.
                        T.tile.fill(bigC[0:NUMC], T.float32(3.0))
                        T.tile.add(tmp, bigC[0:1024], T.float32(0.0))
                        T.tile.fill(tmp[0:1024], T.float32(0.0))
                        for og in T.serial(NG):
                            T.tile.fill(tmp[0:1024], T.float32(0.0))
                            # read slabA and scalar wv inside the loop ...
                            T.tile.axpy(tmp, slabA[og * 1024 : og * 1024 + 1024], wv)
                            # ... then write bigB after slabA's last read ...
                            T.tile.fill(bigB[0:NUMB], T.float32(1.0))
                            # ... and consume bigB.
                            T.tile.add(tmp, tmp, bigB[0:1024])
                            T.pipe_barrier("V")
                            T.set_flag("v", "mte3", 1)
                            T.wait_flag("v", "mte3", 1)
                            T.copy(tmp[0:8], y[og, 0:8])
                            T.set_flag("mte3", "v", 1)
                            T.wait_flag("mte3", "v", 1)

        return main

    return kernel()


def test_scope_kill_no_overlap():
    kernel = _build()
    off = _offsets(kernel.get_kernel_source())
    assert "slabA" in off and "bigB" in off and "wF" in off
    assert not _overlap(off["slabA"], off["bigB"]), f"bigB {off['bigB']} overlaps slabA {off['slabA']} (scope-kill opening-index bug)"
    assert not _overlap(off["wF"], off["bigB"]), f"bigB {off['bigB']} overlaps wF {off['wF']}"


def test_scope_kill_npu_correct():
    kernel = _build()
    x = torch.arange(NUMA, dtype=torch.float32).npu() * 0.25
    w = torch.full((8,), 2.0, dtype=torch.float32).npu()
    y = kernel(x, w).cpu()
    xx = x.cpu()
    for og in range(NG):
        exp = xx[og * 1024 : og * 1024 + 8] * 2.0 + 1.0
        assert torch.allclose(y[og], exp), f"row {og}: got {y[og].tolist()} exp {exp.tolist()} (slabA corrupted by overlapping bigB)"


if __name__ == "__main__":
    test_scope_kill_no_overlap()
    test_scope_kill_npu_correct()
    print("PASS")
