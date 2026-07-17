"""Kernel-scoped Bisheng compile flags (issue #1386).

Demonstrates the ``compile_flags`` argument of ``@tilelang.jit`` and shows that
flags are resolved per kernel: passing ``--cce-auto-sync=off`` / ``-O3`` to one
kernel does not change how a later kernel with default flags is compiled.

Historically these were driven by the process-wide ``TL_CCE_AUTO_SYNC`` /
``TL_CCE_OPT_LEVEL`` environment variables, which leaked across kernels compiled
in the same process. ``compile_flags`` makes them kernel-scoped.

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


def vec_add(M, N, block_M, block_N, compile_flags, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @tilelang.jit(out_idx=[-1], compile_flags=compile_flags)
    def build():
        @T.prim_func
        def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
                bx = cid // n_num
                by = cid % n_num
                a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
                b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
                c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
                # Explicit barriers -> correct regardless of --cce-auto-sync.
                with T.Scope("V"):
                    T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                    T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)
                    T.barrier_all()
                    T.tile.add(c_ub, a_ub, b_ub)
                    T.barrier_all()
                    T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

        return main

    return build()


torch.manual_seed(0)
a = torch.randn(M, N).npu()
b = torch.randn(M, N).npu()
ref = a + b

# Kernel A: per-kernel flags -> bisheng gets --cce-auto-sync=off and -O3.
kernel_a = vec_add(M, N, 128, 256, compile_flags=["--cce-auto-sync=off", "-O3"])
# Kernel B: default flags (no compile_flags) -> bisheng gets -O2, auto-sync on.
# If Kernel A's flags leaked, Kernel B would be miscompiled.
kernel_b = vec_add(M, N, 128, 256, compile_flags=None)

print("init successful!")

out_a = kernel_a(a, b)
out_b = kernel_b(a, b)

torch.testing.assert_close(out_a, ref, rtol=1e-2, atol=1e-2)
torch.testing.assert_close(out_b, ref, rtol=1e-2, atol=1e-2)

print("Kernel Output Match!")
