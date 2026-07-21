"""Kernel-scoped Bisheng compile flags (issue #1386).

This example demonstrates two synchronization controls that serve different
purposes:

* ``TL_ASCEND_AUTO_SYNC`` makes TileLang insert the synchronization required by
  a kernel's data dependencies.
* ``compile_flags`` configures Bisheng for one compiled kernel. In particular,
  ``--cce-auto-sync=off`` does not replace TileLang's synchronization pass.

Both kernels enable TileLang's synchronization pass and compute a dependent
Vector-operation chain correctly. The first uses explicit Bisheng flags; the
second uses defaults and verifies that the first kernel's flags did not leak.

Historically Bisheng options were driven by the process-wide
``TL_CCE_AUTO_SYNC`` / ``TL_CCE_OPT_LEVEL`` environment variables, which leaked
across kernels compiled in the same process. ``compile_flags`` makes them
kernel-scoped.

Run: python compile_flags_example.py
"""

import argparse

import torch

import tilelang
import tilelang.language as T

tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="compile_flags example")
parser.add_argument("--m", type=int, default=256, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=256, help="Matrix N dimension")
args = parser.parse_args()

M, N = args.m, args.n
VEC_NUM = 2


def sigmoid(
    M: int,
    N: int,
    block_M: int,
    block_N: int,
    compile_flags: list[str] | None,
    dtype: str = "float",
):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    pass_configs = {
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    }

    @tilelang.jit(out_idx=[1], pass_configs=pass_configs, compile_flags=compile_flags)
    def build():
        @T.prim_func
        def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
            with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
                bx = cid // n_num
                by = cid % n_num
                row_offset = bx * block_M + vid * block_M // VEC_NUM

                a_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
                b_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
                zero_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)

                # This load -> dependent Vector chain -> store has real pipeline
                # hazards. There are deliberately no manual barriers here.
                T.copy(A[row_offset, by * block_N], a_ub)
                T.tile.fill(zero_ub, 0.0)
                T.tile.sub(a_ub, zero_ub, a_ub)
                T.tile.exp(a_ub, a_ub)
                T.tile.add(a_ub, a_ub, 1.0)
                T.tile.reciprocal(b_ub, a_ub)
                T.copy(b_ub, B[row_offset, by * block_N])

        return main

    return build()


torch.manual_seed(0)
a = torch.randn(M, N).npu()
ref = torch.sigmoid(a)

custom_flags = ["--cce-auto-sync=off", "-O3"]
kernel_with_custom_flags = sigmoid(M, N, 64, 64, compile_flags=custom_flags)

# Compiled after the custom-flag kernel: its default flags must remain independent.
kernel_with_defaults = sigmoid(M, N, 64, 64, compile_flags=None)

print("Compilation successful!")

out_with_custom_flags = kernel_with_custom_flags(a)
out_with_defaults = kernel_with_defaults(a)

torch.testing.assert_close(out_with_custom_flags, ref, rtol=1e-2, atol=1e-2)
torch.testing.assert_close(out_with_defaults, ref, rtol=1e-2, atol=1e-2)

print("Custom compile flags: output matches PyTorch.")
print("Default compile flags: output matches PyTorch.")
print("Default compile flags remained kernel-scoped.")
print("Test Passed!")
