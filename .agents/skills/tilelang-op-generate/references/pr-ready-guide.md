# 算子上库前收尾工作指南

本文档描述算子代码生成后，在提交 PR 上库前必须完成的收尾工作。

---

## 1. Golden 实现的重要性 ⭐⭐⭐

### 1.1 为什么 Golden 实现至关重要

Golden 实现是证明算子迁移正确性的**唯一证据**。如果 golden 实现不一致，即使测试通过也无法证明迁移正确。

### 1.2 Golden 实现来源优先级

| 优先级 | 来源 | 适用场景 |
|--------|------|----------|
| **最高** | 原算子的 golden 实现 | 迁移已有算子（必须使用） |
| **次高** | PyTorch 标准实现 | 新算子开发 |
| **最低** | 手写实现 | 无标准实现时 |

### 1.3 迁移算子时必须检查

**必须回答的问题**：如何证明我的实现与原算子一致？

| 检查项 | 方法 |
|--------|------|
| golden 函数是否一致 | 对比原算子代码，确保使用相同的 golden 实现 |
| 输出形状是否一致 | 检查原算子输出 shape，可能需要 transpose |
| 数据类型是否一致 | 确保输入输出 dtype 与原算子匹配 |

### 1.4 输出形状匹配示例

原算子可能输出 `(N, M)`，而你的 kernel 输出 `(M, N)`：

```python
# 原算子输出 (N, M)
def ref_program(A, qB):
    B = torch_convert(qB)
    C = torch.matmul(A.to(torch.float), B.T.to(torch.float))
    return C.transpose(0, 1)  # (M, N) → (N, M)

# 你的 kernel 输出 (M, N)
# 测试时需要 transpose 匹配
result = kernel(A_int8, qB, workspace, C_output)  # (M, N)
expected = ref_program(A_int8, qB)  # (N, M)

torch.testing.assert_close(result.cpu().transpose(0, 1), expected)
```

---

## 2. 参数处理灵活性

### 2.1 推荐模式

支持用户自定义参数 + 默认测试：

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=0)
    parser.add_argument("--N", type=int, default=0)
    parser.add_argument("--K", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(42)

    # 用户自定义参数时单测
    if args.M > 0 and args.N > 0 and args.K > 0:
        test(args.M, args.N, args.K)
    else:
        # 默认：多组测试
        test(64, 64, 256)
        test(128, 128, 512)
        test(256, 256, 1024)

    print("Test Passed!")
```

### 2.2 测试配置建议

| 场景 | 测试配置 |
|------|----------|
| 简单算子 | 2 组：小规模 + 典型规模 |
| 复杂算子 | 3 组：小规模 + 典型规模 + 大规模 |
| GEMM 类 | 3 组：K=256, 512, 1024 |

---

## 3. 测试输出规范

### 3.0 缓存禁用 ⭐

测试算子时，必须禁用缓存以确保每次重新编译：

```python
def main():
    tilelang.disable_cache()  # Disable cache for testing correctness
    torch.manual_seed(...)
    
    # ... test code ...
```

**⚠️ 重要规则**：

| 规则 | 说明 |
|------|------|
| ✅ 推荐 | `tilelang.disable_cache()` 放在 `__main__` 下方或 `main()` 内部，只在测试时禁用 |
| ❌ 禁止 | `tilelang.disable_cache()` 放在文件开头（全局声明），影响其他人 import kernel |
| ❌ 禁止 | `tilelang.cache.clear_cache()`，会清理全部编译缓存，影响其它算子 |

**原因**：
- 全局声明会影响其他人 import 这个算子的 kernel 函数
- `clear_cache()` 会清理全部编译缓存，影响其它算子的缓存
- 只在测试时禁用缓存，可以验证算子实现正确性

### 3.1 每组测试通过的提示

每组测试通过时应输出提示，避免让人觉得卡住：

```python
def test(M, N, K):
    kernel = op_name(M, N, K)
    
    # ... 执行 kernel ...
    
    torch.testing.assert_close(result.cpu(), expected, rtol=1e-2, atol=1e-2)
    print(f"Test passed: M={M}, N={N}, K={K}")  # ✓ 关键输出
```

### 3.2 最终输出格式

最后一行必须输出 `"Test Passed!"` 或 `"Kernel Output Match!"`，以符合 bench_test.sh 的判定逻辑：

```bash
# bench_test.sh 判定条件
if [[ "$output" =~ [Kk][Ee][Rr][Nn][Ee][Ll][[:space:]][Oo][Uu][Tt][Pp][Uu][Tt][[:space:]][Mm][Aa][Tt][Cc][Hh] ]] || \
   [[ "$output" =~ [Tt][Ee][Ss][Tt][[:space:]][Pp][Aa][Ss][Ss][Ee][Dd][!] ]]; then
    echo "[PASSED]"
fi
```

### 3.3 完整输出示例

```
Test passed: M=64, N=64, K=256
Test passed: M=128, N=128, K=512
Test passed: M=256, N=256, K=1024
Test Passed!
```

---

## 4. 代码风格规范

### 4.1 注释规范

| 规则 | 说明 |
|------|------|
| 中文注释转英文 | 所有注释使用英文 |
| 移除调试注释 | 上库前删除临时调试代码和注释 |
| 保留关键注释 | 保留算法说明、参数说明等关键注释 |

**示例**：

```python
# ❌ 中文注释（上库前修改）
# Vector 核部分 - 每个 V 核处理 block_N_2 行

# ✓ 英文注释
# Vector core: each V core processes block_N_2 rows
```

### 4.2 `# type: ignore` 使用

`T.Tensor` 参数定义会导致 Pylance 报错，需要添加 `# type: ignore`：

```python
@T.prim_func
def main(
    A: T.Tensor((M, K), "int8"),               # type: ignore
    B_packed: T.Tensor((N, K_half), "uint8"),  # type: ignore
    workspace: T.Tensor((N, K), "int8"),       # type: ignore
    C: T.Tensor((M, N), "int32"),              # type: ignore
):
    ...
```

**原因**：TileLang DSL 的 `T.Tensor` 是特殊类型定义，Pylance 无法正确识别。

### 4.3 `torch.set_default_device` 使用

**不推荐**：全局设置默认设备

```python
# ❌ 不推荐
torch.set_default_device("npu")
```

**推荐**：按需指定设备

```python
# ✓ 推荐
A_int8 = torch.randint(-8, 8, (M, K), dtype=torch.int8).npu()
result = kernel(A_int8, qB, workspace, C_output)
torch.npu.synchronize()
expected = ref_program(A_int8.cpu(), qB.cpu())  # golden 在 CPU 上计算
```

**原因**：
- 全局设置可能影响其他代码
- 按需指定更清晰、更可控

### 4.4 移除 try-catch

上库前移除测试代码中的 try-catch，fail fast 更利于问题暴露：

```python
# ❌ 不推荐（上库前修改）
try:
    result = kernel(A_int8, qB, workspace, C_output)
    torch.testing.assert_close(result.cpu(), expected)
except Exception as e:
    traceback.print_exc()
    logging.error(f"✗ Error: {e}")
    return False

# ✓ 推荐
result = kernel(A_int8, qB, workspace, C_output)
torch.npu.synchronize()
expected = ref_program(A_int8.cpu(), qB.cpu())
torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
```

---

## 5. 代码格式检查

### 5.1 检查工具

使用 `ruff` 进行 Python 代码检查：

```bash
# Lint 检查
ruff check examples/{op}/example_{op}.py

# Format 检查
ruff format --check examples/{op}/example_{op}.py
```

详细请参考技能：[tilelang-review-skill](../../tilelang-custom-skill/tilelang-review-skill/SKILL.md)

### 5.2 自动修复

```bash
# 自动修复 lint 问题
ruff check --fix examples/{op}/example_{op}.py

# 自动格式化
ruff format examples/{op}/example_{op}.py
```

### 5.3 常见问题修复

| 问题 | 修复方法 |
|------|----------|
| 未使用的变量 | 删除变量或添加 `_` 前缀 |
| 行宽超限 | 手动调整或 `ruff format` |
| 导入未使用 | 删除导入 |

---

## 6. 检查清单索引

本文档详细说明了上库前各项检查的要求和示例，完整检查清单请参考：

→ **[SKILL.md §8 Checklist](../SKILL.md#8-checklist)** - 唯一的完整检查清单（22项）

### 检查项与文档章节对应

| 检查项 | SKILL.md 编号 | 本文档章节 |
|--------|--------------|-----------|
| **Golden 实现一致** | #9 | §1 |
| **输出形状匹配** | #10 | §1.4 |
| **tilelang.disable_cache()** | #11 | §3.0 |
| **注释转英文** | #12 | §4.1 |
| **`# type: ignore`** | #13 | §4.2 |
| **移除 try-catch** | #14 | §4.4 |
| **每组测试提示** | #15 | §3.1 |
| **最终输出格式** | #16 | §3.2 |
| **参数处理灵活** | #17 | §2 |
| **代码格式检查** | #18 | §5 |

**使用方法**：
1. 先阅读本文档 §1-5 理解各项检查的详细要求
2. 按照 SKILL.md §8 Checklist逐项检查

---

## 7. 参考示例

完整的上库算子示例：

- `examples/dequantize_gemm/example_dequant_gemm_w4a8.py` - W4A8 GEMM CV 融合
- `examples/quant_batch_matmul/example_quant_batch_matmul.py` - 量化 Batch Matmul
- `examples/flash_attention/` - Flash Attention 系列

---

## 8. 常见错误排查

### 8.1 Golden 不一致导致测试失败

**症状**：测试失败，精度误差很大

**排查**：
1. 检查是否使用原算子的 golden 实现
2. 检查输出形状是否需要 transpose
3. 检查数据类型是否一致

### 8.2 bench_test.sh 未通过

**症状**：脚本判定失败

**排查**：
1. 检查最后一行输出是否为 `"Test Passed!"` 或 `"Kernel Output Match!"`
2. 检查是否有异常退出
3. 检查测试是否全部通过

### 8.3 Pylance 报错

**症状**：VSCode 显示类型错误

**排查**：
1. 添加 `# type: ignore` 到 T.Tensor 参数
2. 确保其他代码符合类型注解规范