# Copyright (c) Huawei Technologies Co., Ltd. 2026.
import os
import math
import torch
import torch_npu
import tilelang
import tilelang.language as T
import torch.nn.functional as F
from typing import Tuple, Optional

os.environ['TILELANG_ASCEND_MODE'] = 'Developer'
tilelang.set_log_level("WARNING")

T.float16 = "float16"
T.float32 = "float32"

@tilelang.jit(target="npuir")
def mhc_post_tilelang(hc: int, hidden: int, n_thr: int = 128, h_blk: int = 1024):
    # rename for shorter code
    n = T.symbolic("num_tokens")
    h = hidden

    h_blk = math.gcd(hidden, h_blk)

    @T.prim_func
    def mhc_post_tilelang_kernel_(
        a: T.Tensor((n, hc, hc), T.float32),
        b: T.Tensor((n, hc, h), T.float16),
        c: T.Tensor((n, hc), T.float32),
        d: T.Tensor((n, h), T.float16),
        x: T.Tensor((n, hc, h), T.float16),
    ):
        with T.Kernel(n, threads=n_thr) as i_n:
            x_shared = T.alloc_shared((hc, h_blk), T.float16)
            b_shared = T.alloc_shared((hc, h_blk), T.float16)
            d_shared = T.alloc_shared(h_blk, T.float16)

            x_local = T.alloc_shared((hc, h_blk), T.float32)
            b_local = T.alloc_shared((hc, h_blk), T.float32)
            d_local = T.alloc_shared(h_blk, T.float32)

            a_local = T.alloc_shared((hc, hc), T.float32)
            c_local = T.alloc_shared(hc, T.float32)
            T.copy(a[i_n, 0, 0], a_local)
            T.copy(c[i_n, 0], c_local)

            for i0_h in T.Pipelined(T.ceildiv(h, h_blk), num_stages=2):
                T.copy(b[i_n, 0, i0_h * h_blk], b_shared)
                T.copy(d[i_n, i0_h * h_blk], d_shared)

                T.copy(b_shared, b_local)
                T.copy(d_shared, d_local)
                for i_hco, i1_h in T.Parallel(hc, h_blk):
                    x_local[i_hco, i1_h] = c_local[i_hco] * d_local[i1_h]
                    for i_hci in T.serial(hc):
                        x_local[i_hco, i1_h] += a_local[i_hci, i_hco] * b_local[i_hci, i1_h]
                T.copy(x_local, x_shared)

                T.copy(x_shared, x[i_n, 0, i0_h * h_blk])

    return  mhc_post_tilelang_kernel_

def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    compiled_kernel = mhc_post_tilelang(residual.shape[-2], residual.shape[-1])
    out = torch.empty_like(residual)
    compiled_kernel(comb_res_mix, residual, post_layer_mix.squeeze(-1), x, out)
    return out


def mhc_post_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    term2 = torch.bmm(comb_res_mix.mT, residual.float())
    return (x.float().unsqueeze(-2) * post_layer_mix + term2).half()


def generate_test_data(
    n: int,
    h: int,
    hc_mult: int,
) -> dict[str, torch.Tensor]:
    """Generate test data for post operator."""
    torch.random.manual_seed(42)

    x = torch.randn((n, h), dtype=torch.float16).npu()
    residual = torch.randn((n, hc_mult, h), dtype=torch.float16).npu()
    post_layer_mix = torch.randn((n, hc_mult, 1), dtype=torch.float32).npu()
    comb_res_mix = torch.randn((n, hc_mult, hc_mult), dtype=torch.float32).npu()

    return {
        "x": x,
        "residual": residual,
        "post_layer_mix": post_layer_mix,
        "comb_res_mix": comb_res_mix,
    }


def test(n: int, h: int) -> None:
    print(f"Testing mhc_post with {n=} {h=}")
    test_data = generate_test_data(n=n, h=h, hc_mult=4)
    out_tl = mhc_post(**test_data)
    out_ref = mhc_post_ref(**test_data)
    torch.testing.assert_close(out_tl, out_ref)
    print("\033[92m out check passed!\033[0m")

def main():
    for n in [4096]:
        for h in [1280, 2560, 7168]:
            test(n=n, h=h)

if __name__ == "__main__":
    main()