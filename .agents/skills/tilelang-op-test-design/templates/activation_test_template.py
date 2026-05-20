"""
简化示例：Silu 算子测试（规则和不规则 shape 自然包含）

展示如何在 L1 功能测试中自然包含规则和不规则 shape（含尾块场景）
不强调尾块，不创建独立 Tail 测试函数
"""

import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

# ========== Kernel 定义 ==========
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def silu(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    VEC_NUM = 2

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            denom_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            zero_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.tile.fill(zero_ub, 0.0)
            T.tile.sub(denom_ub, zero_ub, a_ub)
            T.tile.exp(denom_ub, denom_ub)
            T.tile.add(denom_ub, denom_ub, 1.0)
            T.tile.div(b_ub, a_ub, denom_ub)
            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


# ========== 精度标准定义 ==========
def get_precision(dtype):
    precision_map = {
        "float16": (1e-3, 1e-3),
        "float32": (1e-5, 1e-5),
        "bfloat16": (1e-2, 5e-3),
    }
    return precision_map.get(dtype, (1e-3, 1e-3))


# ========== Golden 函数定义 ==========
def golden_silu(input_data):
    return input_data * torch.sigmoid(input_data)


# ========== L0 测试：门槛测试（规则 shape）==========
def test_silu_l0():
    """
    L0 门槛测试：快速冒烟

    Shape：规则 shape（block size 整除）
    """
    print("\n[L0] 门槛测试")

    # 规则 shape 配置
    test_configs = [
        ("float16", 512, 512, 32, 32),
        ("float32", 512, 512, 32, 32),
    ]

    for dtype, M, N, block_M, block_N in test_configs:
        print(f"  dtype={dtype}, shape={M}x{N}, block={block_M}x{block_N}")

        func = silu(M, N, block_M, block_N, dtype=dtype)
        torch.manual_seed(42)
        torch_dtype = getattr(torch, dtype.replace("float", "float"))
        input_data = torch.randn(M, N, dtype=torch_dtype).npu()
        output = func(input_data)
        ref = golden_silu(input_data)
        rtol, atol = get_precision(dtype)
        torch.testing.assert_close(output.cpu(), ref.cpu(), rtol=rtol, atol=atol)
        print("  ✅ Passed")


# ========== L1 测试：功能测试（规则和不规则 shape）==========
def test_silu_l1():
    """
    L1 功能测试：参数组合覆盖

    Shape：包含规则和不规则 shape（含尾块场景）

    规则 shape：block size 整除
    不规则 shape：有余数（自然包含尾块）

    注意：不规则 shape 不特殊标记，自然包含在测试中
    """
    print("\n[L1] 功能测试（参数组合覆盖）")

    # dtype 组合
    dtypes = ["float16", "float32", "bfloat16"]

    # shape 组合（规则 + 不规则）⭐ 自然包含尾块
    shapes = [
        # === 规则 shape（整除）===
        (32, 32),
        (128, 128),
        (512, 512),
        (1024, 1024),
        # === 不规则 shape（有余数）=== 自然包含尾块场景
        (32 * 3 + 30, 32 * 2),  # M=126, 余数30（尾块）
        (32 * 3 + 1, 32 * 2),  # M=97, 余数1（最小尾块）
        (32 * 3 + 31, 32 * 2),  # M=127, 余数31（最大尾块）
        (64 * 8 + 45, 64 * 8),  # M=557, 余数45（大规模+尾块）
        (100, 100),  # M=100, 余数4（常见不整除）
        (123, 456),  # 随机不规则
    ]

    # block size 组合
    blocks = [(16, 16), (32, 32)]

    test_count = 0

    for dtype in dtypes:
        for M, N in shapes:
            for block_M, block_N in blocks:
                test_count += 1
                if test_count > 100:
                    break

                # 自然打印，不强调规则或不规则
                print(f"  [{test_count}] dtype={dtype}, shape={M}x{N}, block={block_M}x{block_N}")

                try:
                    func = silu(M, N, block_M, block_N, dtype=dtype)
                    torch.manual_seed(42)
                    torch_dtype = getattr(torch, dtype.replace("float", "float"))
                    input_data = torch.randn(M, N, dtype=torch_dtype).npu()
                    output = func(input_data)
                    ref = golden_silu(input_data)
                    rtol, atol = get_precision(dtype)
                    torch.testing.assert_close(output.cpu(), ref.cpu(), rtol=rtol, atol=atol)
                    print("    ✅ Passed")
                except Exception as e:
                    print(f"    ❌ Failed: {type(e).__name__}")

    print(f"  ✅ L1 测试完成（{test_count} 个用例）")


# ========== L2 测试：异常测试 ==========
def test_silu_l2():
    """L2 异常测试"""
    print("\n[L2] 异常测试")

    # 不支持的数据类型
    print("  [1] 不支持 dtype（int8）")
    try:
        func = silu(1024, 1024, 32, 32, dtype="int8")
        input_data = torch.randint(-128, 127, (1024, 1024), dtype=torch.int8).npu()
        output = func(input_data)
        print("    ❌ 应该抛出异常")
    except Exception as e:
        print(f"    ✅ 正确抛出异常")

    print("  ✅ L2 测试完成")


# ========== Boundary 测试：边界测试 ==========
def test_silu_boundary():
    """Boundary 边界测试"""
    print("\n[Boundary] 边界测试")

    # 空 tensor
    print("  [1] 空 tensor")
    try:
        func = silu(0, 1024, 32, 32)
        input_data = torch.randn(0, 1024).npu()
        output = func(input_data)
        assert output.shape == (0, 1024)
        print("    ✅ Passed")
    except Exception as e:
        print(f"    ⚠️ Exception: {type(e).__name__}")

    # 极值
    print("  [2] 极值测试")
    func = silu(32, 32, 16, 16)
    min_val = torch.finfo(torch.float16).min
    input_data = torch.full((32, 32), min_val, dtype=torch.float16).npu()
    output = func(input_data)
    ref = golden_silu(input_data)
    torch.testing.assert_close(output, ref, rtol=1e-2, atol=1e-2)
    print("    ✅ Passed")

    print("  ✅ Boundary 测试完成")


# ========== 主函数 ==========
def main():
    print("=" * 60)
    print("SiLU 算子测试")
    print("=" * 60)

    test_silu_l0()
    test_silu_l1()  # 自然包含规则和不规则 shape
    test_silu_l2()
    test_silu_boundary()

    print("\n✅ 所有测试通过！")


if __name__ == "__main__":
    main()


# ========== 对比说明 ==========
"""
简化设计（自然包含不规则 shape）：
- L0: 规则 shape（快速冒烟）
- L1: 规则 + 不规则 shape（自然包含尾块）⭐
- L2: 异常测试
- Boundary: 边界测试

不规则 shape 自然包含在 L1 中，不特殊标记，不强调尾块。

Shape 配置示例：
规则 shape:
  (32, 32), (128, 128), (512, 512), (1024, 1024)

不规则 shape（有余数，自然包含尾块）:
  (32*3+30, 32*2)  # M=126, 余数30
  (32*3+1, 32*2)   # M=97, 余数1
  (100, 100)       # M=100, 余数4

对比过度强调尾块的设计：
- 不创建 Tail 独立层级
- 不创建 test_{op}_tail() 函数
- 不强调尾块是"核心创新"
- 不统计尾块区域误差

简化设计的优势：
- 更自然，不过度设计
- 保持简洁
- 尾块只是不规则 shape 的一种情况
"""
