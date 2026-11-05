import argparse

import tilelang
import tilelang.language as T
import torch

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

CAST_MODE_LOW2HIGH = "CAST_NONE"
CAST_MODE_HIGH2LOW = "CAST_RINT"

VEC_NUM = 2


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def masked_scale(M, N, block_M, block_N, scale, dtype="float16", mask_dtype="int8"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    sub_block_M = block_M // VEC_NUM

    use_fp32_x = dtype in ("float16", "bfloat16")
    use_fp32_mask = mask_dtype in ("float16", "bfloat16")
    use_int_mask = mask_dtype in ("int8", "uint8")
    use_simple_path = (not use_fp32_x) and (not use_fp32_mask) and (not use_int_mask)
    use_native_int_path = use_int_mask and (dtype in ("float16", "float"))
    cal_dtype = "float32"

    @T.prim_func
    def main(
        X: T.Tensor((M, N), dtype),  # type: ignore
        Mask: T.Tensor((M, N), mask_dtype),  # type: ignore
        Y: T.Tensor((M, N), dtype),  # type: ignore
    ):
        T.func_attr({"enable_auto_sync": True})
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            row_start = bx * block_M + vid * sub_block_M
            col_start = by * block_N

            if use_simple_path:
                x_ub = T.alloc_ub((sub_block_M, block_N), dtype)
                mask_ub = T.alloc_ub((sub_block_M, block_N), mask_dtype)

                T.copy(X[row_start, col_start], x_ub, pad_value=0.0)
                T.copy(Mask[row_start, col_start], mask_ub, pad_value=0.0)

                T.tile.mul(x_ub, x_ub, mask_ub)
                if scale != 1.0:
                    T.tile.mul(x_ub, x_ub, scale)

                T.copy(x_ub, Y[row_start, col_start])
            elif use_native_int_path:
                x_ub = T.alloc_ub((sub_block_M, block_N), dtype)
                mask_load = T.alloc_ub((sub_block_M, block_N), mask_dtype)

                T.copy(X[row_start, col_start], x_ub, pad_value=0.0)
                T.copy(Mask[row_start, col_start], mask_load, pad_value=0.0)

                if dtype == "float16":
                    mask_cast = T.alloc_ub((sub_block_M, block_N), dtype)
                    T.tile.cast(mask_cast, mask_load, CAST_MODE_LOW2HIGH, sub_block_M * block_N)
                    T.tile.mul(x_ub, x_ub, mask_cast)
                else:
                    mask_f16 = T.alloc_ub((sub_block_M, block_N), "float16")
                    mask_cast = T.alloc_ub((sub_block_M, block_N), dtype)
                    T.tile.cast(mask_f16, mask_load, CAST_MODE_LOW2HIGH, sub_block_M * block_N)
                    T.tile.cast(mask_cast, mask_f16, CAST_MODE_LOW2HIGH, sub_block_M * block_N)
                    T.tile.mul(x_ub, x_ub, mask_cast)

                if scale != 1.0:
                    T.tile.mul(x_ub, x_ub, scale)

                T.copy(x_ub, Y[row_start, col_start])
            else:
                x_load = T.alloc_ub((sub_block_M, block_N), dtype)
                mask_load = T.alloc_ub((sub_block_M, block_N), mask_dtype)

                T.copy(X[row_start, col_start], x_load, pad_value=0.0)
                T.copy(Mask[row_start, col_start], mask_load, pad_value=0.0)

                if use_fp32_x:
                    x_cal = T.alloc_ub((sub_block_M, block_N), cal_dtype)
                    T.tile.cast(x_cal, x_load, CAST_MODE_LOW2HIGH, sub_block_M * block_N)

                    if use_int_mask:
                        mask_cal = T.alloc_ub((sub_block_M, block_N), cal_dtype)
                        mask_f16 = T.alloc_ub((sub_block_M, block_N), "float16")
                        T.tile.cast(mask_f16, mask_load, CAST_MODE_LOW2HIGH, sub_block_M * block_N)
                        T.tile.cast(mask_cal, mask_f16, CAST_MODE_LOW2HIGH, sub_block_M * block_N)
                        T.tile.mul(x_cal, x_cal, mask_cal)
                    elif use_fp32_mask:
                        mask_cal = T.alloc_ub((sub_block_M, block_N), cal_dtype)
                        T.tile.cast(mask_cal, mask_load, CAST_MODE_LOW2HIGH, sub_block_M * block_N)
                        T.tile.mul(x_cal, x_cal, mask_cal)
                    else:
                        T.tile.mul(x_cal, x_cal, mask_load)

                    if scale != 1.0:
                        T.tile.mul(x_cal, x_cal, scale)

                    y_store = T.alloc_ub((sub_block_M, block_N), dtype)
                    T.tile.cast(y_store, x_cal, CAST_MODE_HIGH2LOW, sub_block_M * block_N)
                    T.copy(y_store, Y[row_start, col_start])
                else:
                    if use_int_mask:
                        mask_cal = T.alloc_ub((sub_block_M, block_N), cal_dtype)
                        mask_f16 = T.alloc_ub((sub_block_M, block_N), "float16")
                        T.tile.cast(mask_f16, mask_load, CAST_MODE_LOW2HIGH, sub_block_M * block_N)
                        T.tile.cast(mask_cal, mask_f16, CAST_MODE_LOW2HIGH, sub_block_M * block_N)
                        T.tile.mul(x_load, x_load, mask_cal)
                    elif use_fp32_mask:
                        mask_cal = T.alloc_ub((sub_block_M, block_N), cal_dtype)
                        T.tile.cast(mask_cal, mask_load, CAST_MODE_LOW2HIGH, sub_block_M * block_N)
                        T.tile.mul(x_load, x_load, mask_cal)
                    else:
                        T.tile.mul(x_load, x_load, mask_load)

                    if scale != 1.0:
                        T.tile.mul(x_load, x_load, scale)

                    T.copy(x_load, Y[row_start, col_start])

    return main


def main():
    parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
    parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
    parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
    args = parser.parse_args()

    M = args.m
    N = args.n

    torch.manual_seed(0)

    # (x_dtype, mask_dtype, scale, block_M, block_N)
    test_configs = [
        (torch.float16, torch.int8, 1.0, 128, 128),
        (torch.float16, torch.uint8, 2.0, 128, 128),
        (torch.float32, torch.int8, 0.5, 128, 128),
        (torch.float32, torch.float32, 1.0, 128, 128),
        (torch.bfloat16, torch.float16, 1.0, 128, 128),
        (torch.bfloat16, torch.int8, 1.0, 128, 128),
    ]

    dtype_map = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float",
        torch.int8: "int8",
        torch.uint8: "uint8",
    }

    for x_dtype, mask_dtype, scale, block_M, block_N in test_configs:
        x_tl = dtype_map[x_dtype]
        mask_tl = dtype_map[mask_dtype]
        print(f"Testing MaskedScale with M={M}, N={N}, x={x_tl}, mask={mask_tl}, scale={scale}")

        func = masked_scale(M, N, block_M, block_N, scale, dtype=x_tl, mask_dtype=mask_tl)
        print("Init successful!")

        x = torch.randn(M, N, dtype=torch.float32).npu()
        if x_dtype in (torch.float16, torch.bfloat16):
            x = x.to(x_dtype)
        mask = torch.randint(0, 2, (M, N), dtype=mask_dtype).npu()

        y = func(x, mask)
        ref_y = (x * mask * scale).to(x_dtype)

        torch.testing.assert_close(y.cpu(), ref_y.cpu(), rtol=1e-2, atol=1e-2)
        print("Test passed!")

    print("Kernel Output Match!")


if __name__ == "__main__":
    main()
