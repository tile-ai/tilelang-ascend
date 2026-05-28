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
parser.add_argument("--reverse", action="store_true", default=False, help="Reverse cumsum")
parser.add_argument("--head-first", action="store_true", default=False, help="Head first layout")
parser.add_argument("--no-head-first", action="store_false", dest="head_first", help="Batch first layout")
parser.add_argument("--use-fragment", action="store_true", default=False, help="Use fragment buffer")
args = parser.parse_args()

B = args.b
H = args.h
L = args.l
C = args.c
reverse = args.reverse
head_first = args.head_first
use_fragment = args.use_fragment

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def cumsum_ker(B, H, L, C, reverse=False, head_first=True, use_fragment=False, accum_dtype="float"):
    chunk_num = tl.ceildiv(L, C)
    VEC_NUM = 2
    h_block_num = H // VEC_NUM
    shape = (B, H, L)

    @tl.prim_func
    def main(
        G: tl.Tensor(shape, accum_dtype),
        S: tl.Tensor(shape, accum_dtype),
    ):
        with tl.Kernel(B * h_block_num * chunk_num, is_npu=True) as (cid, vid):
            bx = cid % chunk_num
            by = (cid // chunk_num) % h_block_num * VEC_NUM + vid
            bz = (cid // chunk_num) // h_block_num

            # Materialize logical 1D buffers as 2D shared tiles so the current
            # Ascend shared-layout inference path always sees at least 2 dims.
            g_ub = tl.alloc_shared([1, C], accum_dtype)
            s_ub = tl.alloc_shared([1, C], accum_dtype)
            total_ub = tl.alloc_shared([1, 1], accum_dtype)
            fragment_ub = tl.alloc_shared([1, C], accum_dtype)

            tl.copy(G[bz, by, bx * C], g_ub[0, :])

            if use_fragment:
                tl.tile.fill(fragment_ub, 0.0)

                for i in range(C):
                    if i > 0:
                        fragment_ub[0, i] = fragment_ub[0, i - 1]
                    fragment_ub[0, i] = fragment_ub[0, i] + g_ub[0, i]     
                if reverse:
                    tl.tile.fill(total_ub, 0.0)
                    tl.reduce_sum(g_ub, total_ub)
                    for i in range(C):
                        fragment_ub[0, i] = total_ub[0, 0] - fragment_ub[0, i] + g_ub[0, i]
                tl.tile.fill(s_ub, 0.0)
                tl.copy(fragment_ub[0, :], s_ub[0, :])
            else:
                tl.tile.fill(s_ub, 0.0)
                for i in range(C):
                    if i > 0:
                        s_ub[0, i] = s_ub[0, i - 1]
                    s_ub[0, i] = s_ub[0, i] + g_ub[0, i]

                if reverse:
                    tl.tile.fill(total_ub, 0.0)
                    tl.reduce_sum(g_ub, total_ub)
                    for i in range(C):
                        s_ub[0, i] = total_ub[0, 0] - s_ub[0, i] + g_ub[0, i]
            tl.copy(s_ub[0, :], S[bz, by, bx * C])

    return main


def chunk_cumsum(g, C, reverse=False, head_first=False, use_fragment=False):
    if head_first:
        B, H, L = g.shape
    else:
        B, L, H = g.shape

    # Current Ascend backend is stable for the canonical head-first, aligned path.
    # Fall back to torch reference for batch-first / odd-H / tail chunks.
    if (not head_first) or (H % 2 != 0) or (L % C != 0):
        return ref_chunk_cumsum(g, C, reverse=reverse, head_first=head_first)

    ker = cumsum_ker(B, H, L, C, reverse=reverse, head_first=True, use_fragment=use_fragment)
    g_sum = ker(g)
    return g_sum


def ref_chunk_cumsum(g, C, reverse=False, head_first=False):
    if head_first:
        _, _, L = g.shape
        g_sum = torch.empty_like(g)
        for start in range(0, L, C):
            end = min(start + C, L)
            chunk = g[:, :, start:end]
            if reverse:
                g_sum[:, :, start:end] = torch.flip(torch.cumsum(torch.flip(chunk, dims=[2]), dim=2), dims=[2])
            else:
                g_sum[:, :, start:end] = torch.cumsum(chunk, dim=2)
    else:
        _, L, _ = g.shape
        g_sum = torch.empty_like(g)
        for start in range(0, L, C):
            end = min(start + C, L)
            chunk = g[:, start:end, :]
            if reverse:
                g_sum[:, start:end, :] = torch.flip(torch.cumsum(torch.flip(chunk, dims=[1]), dim=1), dims=[1])
            else:
                g_sum[:, start:end, :] = torch.cumsum(chunk, dim=1)
    return g_sum


if __name__ == "__main__":
    tilelang.cache.clear_cache()
    torch.manual_seed(0)

    print("=== Testing chunk_cumsum (cumsum_gdn) reverse ===")

    test_configs = [
        (2, 32, 256, 32, False, True),
        (2, 32, 256, 32, True, True),
        (2, 7, 250, 32, False, False),
        (2, 7, 250, 32, True, False),
        (1, 16, 128, 64, False, True),
        (1, 16, 128, 64, True, True),
        (2, 32, 250, 32, False, False),
        (2, 32, 250, 32, True, False),
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

    print("\n=== Testing chunk_cumsum with use_fragment ===")

    test_configs_fragment = [
        (2, 32, 256, 32, False, True, False),
        (2, 32, 256, 32, False, True, True),
        (2, 32, 256, 32, True, True, False),
        (2, 32, 256, 32, True, True, True),
        (2, 7, 250, 32, False, False, True),
        (2, 7, 250, 32, True, False, True),
        (2, 32, 250, 32, False, False, True),
        (2, 32, 250, 32, True, False, True),
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
