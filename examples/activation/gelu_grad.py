import tilelang
import tilelang.language as T
import torch
import torch_npu

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def gelu_grad(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    VEC_NUM = 2

    BETAN = -1.595769121605730711759
    AN = -0.0713548162726002527220
    A3 = 0.2140644488178007
    BETA = 1.595769121605730711759

    @T.prim_func
    def main(
            dy: T.Tensor((M, N), dtype),
            x: T.Tensor((M, N), dtype),
            grad_input: T.Tensor((M, N), dtype)
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            cmp_ub = T.alloc_ub((block_M // VEC_NUM, block_N), "uint8")
            x_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            xsqr_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            px_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            res0_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            div_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            with T.Scope("V"):
                T.copy(x[bx * block_M + vid * block_M // VEC_NUM, by * block_N], x_ub)
                T.tile.mul(xsqr_ub, x_ub, x_ub)

                T.tile.fill(px_ub, BETAN)
                T.tile.axpy(px_ub, xsqr_ub, AN)

                T.tile.mul(px_ub, px_ub, x_ub)

                T.tile.exp(px_ub, px_ub)

                T.tile.fill(res0_ub, BETA)
                T.tile.axpy(res0_ub, xsqr_ub, A3)

                T.tile.mul(res0_ub, res0_ub, x_ub)

                T.tile.add(xsqr_ub, px_ub, 1.0)

                T.tile.fill(x_ub, 1.0) # reuse x_ub
                T.tile.div(div_ub, x_ub, xsqr_ub)

                T.tile.mul(x_ub, px_ub, div_ub) # reuse x_ub as resp_ub
                T.tile.mul(x_ub, x_ub, res0_ub)
                T.tile.mul(x_ub, x_ub, div_ub)

                T.tile.compare(cmp_ub, x_ub, x_ub, "EQ")

                T.tile.fill(px_ub, 0.0) # reuse px_ub as zero_ub
                T.tile.select(xsqr_ub, cmp_ub, x_ub, px_ub, "VSEL_CMPMASK_SPR")

                T.tile.add(x_ub, xsqr_ub, div_ub)
                T.copy(dy[bx * block_M + vid * block_M // VEC_NUM, by * block_N], res0_ub)

                T.tile.mul(div_ub, res0_ub, x_ub)  # reuse div_ub as grad_input_ub
                T.copy(div_ub, grad_input[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


torch.manual_seed(0)

test_configs = [
    (256, 256, 64, 64),
    (512, 512, 64, 256),
    (21293, 1024, 128, 128),
    (21293, 256, 128, 128),
    (6368, 3424, 64, 256),
    (3184, 1712, 64, 256),
]

for M, N, block_M, block_N in test_configs:
    print(f"Testing gelu_grad with M={M}, N={N}, block_M={block_M}, block_N={block_N}")
    func = gelu_grad(M, N, block_M, block_N, dtype="float")
    print("Init successful!")

    dy = torch.randn(M, N, dtype=torch.float).npu()
    x = torch.randn(M, N, dtype=torch.float).npu()

    grad_input = func(dy, x)

    ref_grad_input = torch_npu.npu_gelu_backward(dy, x, approximate="none")

    torch.testing.assert_close(grad_input.cpu(), ref_grad_input.cpu(), rtol=1e-3, atol=1e-2)
    print("Test passed!")

print("Kernel Output Match!")
