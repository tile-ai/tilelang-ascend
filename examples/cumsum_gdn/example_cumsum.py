import argparse

import tilelang
import tilelang.language as tl
import torch

tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="Chunk Local Cumsum Scalar on NPU")
parser.add_argument("--b", type=int, default=2, help="Batch size")
parser.add_argument("--h", type=int, default=32, help="Head number")
parser.add_argument("--l", type=int, default=256, help="Sequence length")
parser.add_argument("--c", type=int, default=32, help="Chunk size")
parser.add_argument("--reverse", type=bool, default=False, help="Reverse cumsum")
parser.add_argument("--head-first", type=bool, default=True, help="Head first layout")
parser.add_argument("--use-fragment", type=bool, default=False, help="Use fragment buffer (P2)")
args = parser.parse_args()

B = args.b
H = args.h
L = args.l
C = args.c
reverse = args.reverse
head_first = args.head_first
use_fragment = args.use_fragment

pass_configs = {tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True}


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def cumsum_ker(B, H, L, C, reverse=False, head_first=True, use_fragment=False, accum_dtype="float"):
    chunk_num = tl.ceildiv(L, C)
    VEC_NUM = 2

    shape = (B, H, L) if head_first else (B, L, H)

    @tl.prim_func
    def main(
        G: tl.Tensor(shape, accum_dtype),
        S: tl.Tensor(shape, accum_dtype),
    ):
        with tl.Kernel(B * (H // VEC_NUM) * chunk_num, is_npu=True) as (cid, vid):
            bx = cid % chunk_num
            by = (cid // chunk_num) % (H // VEC_NUM) * 2 + vid
            bz = (cid // chunk_num) // (H // VEC_NUM)

            g_ub = tl.alloc_ub([C], accum_dtype)
            s_ub = tl.alloc_ub([C], accum_dtype)
            total_ub = tl.alloc_ub([1], accum_dtype)
            fragment_ub = tl.alloc_ub([C], accum_dtype)

            with tl.Scope("V"):
                tl.tile.fill(s_ub, 0.0)

                if head_first:
                    tl.copy(G[bz, by, bx * C], g_ub)
                else:
                    tl.copy(G[bz, bx * C, by], g_ub)

                if use_fragment:
                    tl.copy(g_ub, fragment_ub)
                    tl.tile.fill(fragment_ub, 0.0)

                    for i in range(C):
                        if i > 0:
                            fragment_ub[i] = fragment_ub[i - 1]
                        fragment_ub[i] = fragment_ub[i] + g_ub[i]

                    if reverse:
                        tl.tile.fill(total_ub, 0.0)
                        for i in range(C):
                            total_ub[0] = total_ub[0] + g_ub[i]
                        for i in range(C):
                            fragment_ub[i] = total_ub[0] - fragment_ub[i] + g_ub[i]

                    tl.copy(fragment_ub, s_ub)
                else:
                    for i in range(C):
                        if i > 0:
                            s_ub[i] = s_ub[i - 1]
                        s_ub[i] = s_ub[i] + g_ub[i]

                    if reverse:
                        tl.tile.fill(total_ub, 0.0)
                        for i in range(C):
                            total_ub[0] = total_ub[0] + g_ub[i]
                        for i in range(C):
                            s_ub[i] = total_ub[0] - s_ub[i] + g_ub[i]

                if head_first:
                    tl.copy(s_ub, S[bz, by, bx * C])
                else:
                    tl.copy(s_ub, S[bz, bx * C, by])

    return main


def chunk_cumsum(g, C, reverse=False, head_first=True, use_fragment=False):
    if head_first:
        B, H, L = g.shape
    else:
        B, L, H = g.shape
    ker = cumsum_ker(B, H, L, C, reverse=reverse, head_first=head_first, use_fragment=use_fragment)
    g_sum = ker(g)
    return g_sum


def ref_chunk_cumsum(g, C, reverse=False, head_first=True):
    if head_first:
        B, H, L = g.shape
        chunk_num = (L + C - 1) // C
        g = g.view(B, H, chunk_num, C)
        if reverse:
            g_sum = torch.flip(torch.cumsum(torch.flip(g, dims=[3]), dim=3), dims=[3])
        else:
            g_sum = torch.cumsum(g, dim=-1)
        g_sum = g_sum.view(B, H, L)
    else:
        B, L, H = g.shape
        chunk_num = (L + C - 1) // C
        g = g.view(B, chunk_num, C, H)
        if reverse:
            g_sum = torch.flip(torch.cumsum(torch.flip(g, dims=[2]), dim=2), dims=[2])
        else:
            g_sum = torch.cumsum(g, dim=2)
        g_sum = g_sum.view(B, L, H)
    return g_sum


if __name__ == "__main__":
    tilelang.cache.clear_cache()
    torch.manual_seed(0)

    print("=== Testing chunk_cumsum (cumsum_gdn) - P0: reverse ===")

    test_configs = [
        (2, 32, 256, 32, False, True),
        (2, 32, 256, 32, True, True),
        (1, 16, 128, 64, False, True),
        (1, 16, 128, 64, True, True),
        (4, 8, 512, 64, False, True),
        (4, 8, 512, 64, True, True),
    ]

    for B, H, L, C, reverse, head_first in test_configs:
        shape = (B, H, L) if head_first else (B, L, H)
        print(f"Testing B={B}, H={H}, L={L}, C={C}, reverse={reverse}, head_first={head_first}")
        g = torch.randn(shape).npu().to(torch.float)
        g_sum = chunk_cumsum(g, C, reverse=reverse, head_first=head_first)
        ref_g_sum = ref_chunk_cumsum(g, C, reverse=reverse, head_first=head_first)
        torch.testing.assert_close(g_sum.cpu(), ref_g_sum.cpu(), rtol=1e-5, atol=1e-5)
        print("  Passed!")

    print("\n=== Testing chunk_cumsum with use_fragment (P2) ===")

    test_configs_fragment = [
        (2, 32, 256, 32, False, True, False),
        (2, 32, 256, 32, False, True, True),
        (2, 32, 256, 32, True, True, False),
        (2, 32, 256, 32, True, True, True),
        (1, 16, 128, 64, False, True, True),
        (1, 16, 128, 64, True, True, True),
    ]

    for B, H, L, C, reverse, head_first, use_fragment in test_configs_fragment:
        shape = (B, H, L) if head_first else (B, L, H)
        print(f"Testing B={B}, H={H}, L={L}, C={C}, reverse={reverse}, use_fragment={use_fragment}")
        g = torch.randn(shape).npu().to(torch.float)
        g_sum = chunk_cumsum(g, C, reverse=reverse, head_first=head_first, use_fragment=use_fragment)
        ref_g_sum = ref_chunk_cumsum(g, C, reverse=reverse, head_first=head_first)
        torch.testing.assert_close(g_sum.cpu(), ref_g_sum.cpu(), rtol=1e-5, atol=1e-5)
        print("  Passed!")

    print("\nKernel Output Match!")
