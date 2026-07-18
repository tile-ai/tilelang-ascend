---
name: tilelang-example-merge
description: >-
  将算子的 kernel 文件和测试文件合并为单文件 example_{op}.py，用于上库提交 PR。
  合并后的文件包含完整 kernel 实现 + 1 个代表性 L0 用例 + 1 个代表性 L1 用例，
  全部内联，不依赖 import 兄弟模块。当用户提到上库、提交 PR、合并算子文件、
  生成 example 文件、单文件提交、准备上库、repo submission、提 PR 前合并文件、
  或需要把 kernel 和 test 合成一个文件时必须使用本 skill。即使用户没有明确说
  "merge"，只要意图是将算子代码整理成仓库可接收的单文件示例，也应触发。
---

# TileLang Example Merge

## 概述

将算子开发阶段的双文件结构（`{op}.py` 纯 kernel + `test_{op}.py` 分层测试套件）
合并为仓库上库用的单文件 `example_{op}.py`。

**为什么要合并**：算子开发时 kernel 和测试分离便于迭代，但仓库上库只接收单文件
示例（参考 `examples/normalization/layer_norm.py`、`examples/developer_mode/gelu_mul_developer.py`
的惯例）。单文件示例自包含、可直接 `python example_{op}.py` 运行验证。

**合并策略**：kernel 完整保留 + 从测试套件中自动选取 1 个 L0 代表性用例 + 1 个 L1
代表性用例，精简辅助函数，使用 `torch.testing.assert_close` 做精度检查。

## 触发条件

- 用户提到"上库"、"提交 PR"、"合并文件"、"生成 example"、"单文件提交"
- 用户要把算子代码整理成仓库可接收的单文件示例
- 用户提到 "example_softmax.py"、"example_layer_norm.py" 等命名模式

## 输入

| 参数 | 说明 | 示例 |
|------|------|------|
| 算子名 | 算子目录名和文件名前缀 | `softmax` |

输入文件（隐式从算子名推导）：
- `examples/{op}/{op}.py` — 纯 kernel 文件
- `examples/{op}/test_{op}.py` — 分层测试文件

输出文件：
- `examples/{op}/example_{op}.py` — 合并后的单文件示例

## 工作流程

### 第一步：读取源文件

1. 确认算子名（用户指定或从对话上下文提取）
2. 读取 `examples/{op}/{op}.py`，提取完整 kernel 代码
   - 包括模块级常量（`pass_configs`、`CAST_MODE_*` 等）
   - 包括 `@tilelang.jit` 装饰的函数及其内部的 `@T.prim_func`
   - **不要**包含 `if __name__ == "__main__"` 块（如果有的话）
3. 读取 `examples/{op}/test_{op}.py`，理解测试结构
   - 识别 L0 测试用例（通常在 `test_{op}_l0()` 函数或 `test_configs` 列表中）
   - 识别 L1 测试用例（通常在 `test_{op}_l1()` 函数或 `L1_CASES` 列表中）
   - 提取 golden 参考实现函数
   - 提取 `get_precision` 函数及 dtype→阈值映射表（用于第四步查表填占位符）

### 第二步：选取代表性用例

#### L0 代表性用例选取

L0 是门槛测试（规则 shape，block 整除），选取最具代表性的一个：

1. **优先**：名称含 "typical" 或 "standard" 的用例（如 `l0_typical`）
2. **次选**：shape 最大的用例（最大 N 或最大 B×N，最能代表真实工作负载）
3. **兜底**：第一个 L0 用例

#### L1 代表性用例选取

L1 是功能测试（含不规则 shape、数值范围覆盖），选取最标准的规则 shape 用例：

1. **优先**：带 `D-SHAPE-ALIGNED` tag 的用例（规则 shape，无尾块）
2. **次选**：第一个 shape 为规则对齐的用例（B % block_M == 0 且 N % block_N == 0）
3. **兜底**：第一个 L1 用例

> 选取时注意避开极端边界用例（如 B=1、N=1、超大数值范围），这些适合分层测试
> 但不适合作为上库示例的代表用例。

### 第三步：提取 golden 参考实现

从 `test_{op}.py` 中提取 golden 函数（通常名为 `golden_{op}` 或直接内联在测试中），
**作为独立函数复制到 `example_{op}.py` 的 kernel 之后、`if __name__` 块之前**。

提取规则：
1. **保留独立函数**：不要内联到测试循环里。golden 函数放循环外，循环内调用
   `ref = golden(x)`。这样 golden 逻辑只写一遍，多个用例复用，与 `test_{op}.py`
   结构一致。
2. **原样复制函数体**：保留数学逻辑，去掉冗长 docstring（一行注释说明即可）。
3. **不要重新实现**：如果原 golden 调用了 PyTorch 内置函数（如 `F.softmax`、
   `torch.layer_norm`），直接用该调用，不要手写等价实现，避免引入新 bug。
4. **函数签名对齐**：golden 函数的输入参数应与测试循环中传入的张量一致（通常是
   `def golden(x): return ...`）。

例如 `test_softmax.py` 的 golden 是：
```python
def golden_softmax(x):
    return torch.nn.functional.softmax(x.float(), dim=-1).to(x.dtype)
```
复制到 `example_softmax.py` 后保留为独立函数，循环内 `ref = golden_softmax(x)`。

#### 精度阈值提取

精度阈值**必须根据 test_configs 中选中用例的 dtype 动态确定**，不能硬编码某个
dtype 的阈值。提取步骤：

1. **确定选中 dtype**：读取选中的 L0/L1 用例的 dtype 字段。
2. **查 `test_{op}.py` 的 `get_precision` 表**：找到该 dtype 对应的
   `(atol, rtol, max_abs_limit, required_ratio)` 四元组。
3. **填入模板占位符**：将四个数值替换模板中的 `{atol}`、`{rtol}`、`{max_abs_limit}`、
   `{required_ratio}`。

各 dtype 的标准阈值参考（源自 `test_{op}.py` 的 `get_precision`，与
`tilelang-op-test-design/references/precision-standard.md` 一致）：

| dtype | atol | rtol | max_abs_limit | required_ratio |
|-------|------|------|---------------|----------------|
| float16 | 2**-14 | 2**-9 | 1e-1 | 0.99 |
| bfloat16 | 2**-10 | 2**-6 | 1e0 | 0.99 |
| float32 / "float" | 2**-16 | 2**-10 | 1e-2 | 0.99 |
| hifloat32 | 2**-16 | 2**-10 | 1e-2 | 0.99 |
| float8_e4m3 | 2**-4 | 2**-2 | 1e0 | 0.99 |
| float8_e5m2 | 2**-3 | 2**-1 | 1e-1 | 0.99 |
| int8/int16/int32/int64/uint8 | 0.0 | 0.0 | 0.0 | 1.0 |

> 注意：不同算子的 `test_{op}.py` 可能只覆盖表中部分 dtype。以目标算子 test 文件
> 中实际存在的为准，不要套用上表缺失的 dtype。

**多 dtype 处理**：如果选中用例存在多个不同 dtype（少见），不能在循环外写死一组
阈值。需在循环内按 dtype 分支查表，例如：

```python
for B, N, block_M, block_N, dtype, level in test_configs:
    ...
    # Precision thresholds by dtype (from test_{op}.py get_precision)
    if dtype == "float16":
        atol, rtol, max_abs_limit, required_ratio = 2**-14, 2**-9, 1e-1, 0.99
    elif dtype == "float32" or dtype == "float":
        atol, rtol, max_abs_limit, required_ratio = 2**-16, 2**-10, 1e-2, 0.99
    # ... 只列选中用例实际涉及的 dtype
    ratio = (abs_err <= (atol + rtol * ref_cpu[m].abs())).float().mean().item()
    max_abs = abs_err.max().item()
    assert ratio >= required_ratio and max_abs <= max_abs_limit, ...
```

**单 dtype（常见）**：如果选中用例 dtype 相同，直接在循环外写死该 dtype 的四个数值
字面量（不分支、不封装函数），保持代码精简。

### 第四步：生成合并文件

按以下模板生成 `example_{op}.py`（参考 `examples/normalization/layer_norm.py` 和
`examples/developer_mode/gelu_mul_developer.py` 的仓库惯例）：

```python
import tilelang
from tilelang import language as T
import torch

tilelang.cache.clear_cache()

# ========== Operator Implementation ==========
# （从 {op}.py 复制的 pass_configs、常量、@tilelang.jit 函数，原样保留）

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    # ... 其他配置
}

@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def {op}(...):
    """{算子简述}"""
    # ... kernel 完整实现 ...
    return main


# ========== Golden reference ==========
def golden_{op}(x):
    """{一句话说明}"""
    return {test_{op}.py 中的 golden 函数体，原样复制}


# ========== Tests ==========
if __name__ == "__main__":
torch.manual_seed(0)

# Representative configs: 1 L0 + 1 L1
test_configs = [
    # (B, N, block_M, block_N, dtype, level)
    (..., ..., ..., ..., "...", "L0"),  # {L0 选中用例描述}
    (..., ..., ..., ..., "...", "L1"),  # {L1 选中用例描述}
]

for B, N, block_M, block_N, dtype, level in test_configs:
    print(f"Testing {op} {level} with B={B}, N={N}, block=({block_M},{block_N}), dtype={dtype}")
    func = {op}(B, N, block_M, block_N, dtype=dtype)
    print("Init successful!")
    torch_dtype = getattr(torch, dtype) if dtype != "float" else torch.float32
    x = torch.randn(B, N, dtype=torch_dtype).npu()
    y = func(x)
    ref = golden_{op}(x)
    # Precision check ({dtype} mixed tolerance, inlined — thresholds from test_{op}.py)
    y_cpu, ref_cpu = y.detach().cpu().float(), ref.detach().cpu().float()
    m = torch.isfinite(ref_cpu)
    abs_err = (y_cpu[m] - ref_cpu[m]).abs()
    ratio = (abs_err <= ({atol} + {rtol} * ref_cpu[m].abs())).float().mean().item()
    max_abs = abs_err.max().item()
    assert ratio >= {required_ratio} and max_abs <= {max_abs_limit}, f"precision fail: ratio={ratio:.4f} max_abs={max_abs:.3e}"
    print(f"Test pass! matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")

print("Kernel Output Match!")
```

#### 关键格式约定（必须遵循仓库惯例）

1. **精度检查**：内联混合容差检查，**不要**用 `torch.testing.assert_close`，也**不要**
   封装成函数。直接在测试用例中按选中 dtype 的阈值内联计算。阈值**根据 test_configs
   中选中用例的 dtype 从 `test_{op}.py` 的 `get_precision` 表动态查取**，填入模板的
   `{atol}`/`{rtol}`/`{max_abs_limit}`/`{required_ratio}` 占位符——**禁止硬编码某个
   固定 dtype 的阈值**。双门控：逐元素 `|actual-golden| <= atol + rtol*|golden|`，
   整体 `matched_ratio >= required_ratio` 且 `max_abs_error <= max_abs_limit`。查表
   规则和多 dtype 处理见上方"精度阈值提取"小节。
2. **缓存清理**：文件顶部用 `tilelang.cache.clear_cache()`（不是 `tilelang.disable_cache()`），
   与仓库现有示例一致。
3. **输入数据**：用 `torch.randn(...).npu()` 生成随机输入。如果原测试用了特定数值范围
   （如 `uniform_(-1000, 1000)`），L1 代表用例可以保留该范围，但 L0 用标准 `randn`。
4. **打印格式**：`print(f"Testing {op} ... with ...")` → `print("Init successful!")` →
   `print("Test pass!")` → 末尾 `print("Kernel Output Match!")`，与仓库现有示例一致。
5. **无 import 兄弟模块**：`example_{op}.py` 中**禁止**出现 `from {op} import {op}`
   或 `sys.path.insert` 等导入语句。kernel 代码直接内联。
6. **无分层测试框架**：不要保留 `--level` 参数分发、`COVERAGE_CATEGORY`、
   `L1_CASES` 列表、`check_precision` 等分层测试基础设施。上库示例用单个
   `test_configs` 列表 + for 循环顺序执行两个代表性用例（1 个 L0 + 1 个 L1），
   循环体内复用同一套 kernel 编译/运行/golden/精度检查逻辑，避免代码重复。
   这与仓库现有示例（`layer_norm.py`、`gelu_mul_developer.py` 的 `test_configs`
   循环）一致。
7. **使用 `if __name__ == "__main__"` 守卫**：测试代码放在 `if __name__ == "__main__":`
   块中，`python example_{op}.py` 直接运行时会执行两个代表性用例并打印 PASS/FAIL。
   这比仓库现有示例（`layer_norm.py` 等用模块级测试）更显式，且 `import` 时不自动执行。

### 第五步：验证

生成文件后，运行验证：

```bash
source set_env.sh
python examples/{op}/example_{op}.py
```

确认输出包含 "Kernel Output Match!"。如果失败，检查：
- kernel 代码是否完整复制（漏了常量或辅助函数）
- golden 实现是否正确
- shape/dtype 是否与原测试一致
- 是否有遗留的 import 语句

## 输出文件结构

生成的 `example_{op}.py` 分四段：

```
1. imports + tilelang.cache.clear_cache()
2. kernel 实现（pass_configs + @tilelang.jit 函数）   ← 从 {op}.py 复制
3. golden 参考实现（独立函数）                          ← 从 test_{op}.py 复制
4. if __name__ == "__main__": 测试代码（test_configs 循环 + 末尾打印）  ← 精简
```

目标行数：通常 80-150 行（kernel 行数 + 每个测试约 10-15 行）。

## 注意事项

- **不要修改 kernel 逻辑**：kernel 代码从 `{op}.py` 原样复制，不做任何改动。如果
  kernel 依赖模块级辅助函数（如 `cast_or_copy`），一并复制。
- **保留必要的模块级常量**：`pass_configs`、`CAST_MODE_*`、`VEC_NUM` 等被 kernel
  使用的常量必须保留。未被选中队列测试使用的常量（如 `COVERAGE_MANIFEST`）丢弃。
- **golden 简化但不失真**：golden 函数保留正确的数学逻辑，但可以去掉冗长的 docstring。
  如果原 golden 调用了 PyTorch 内置函数（如 `F.softmax`、`torch.layer_norm`），直接用
  该调用，不要重新实现。
- **dtype 处理**：如果 kernel 的 dtype 参数用 `"float"` 表示 float32，测试中需用
  `getattr(torch, dtype) if dtype != "float" else torch.float32` 转换，与仓库惯例一致。
- **原文件保留**：生成 `example_{op}.py` 后，原 `{op}.py` 和 `test_{op}.py` 不删除，
  它们仍用于开发阶段的分层测试。`example_{op}.py` 是上库用的精简单文件。

## 常见问题

### 测试文件结构不是标准 L0/L1 分层怎么办？

有些算子的测试文件可能用不同的结构（如单个 `test_configs` 列表无 L0/L1 区分）。
此时：
- 将第一个规则 shape 用例作为 "L0 representative"
- 将第二个规则 shape 用例（或稍大 shape 的用例）作为 "L1 representative"
- 在注释中标注 "representative" 而非 "L0"/"L1"

### kernel 文件有 `if __name__ == "__main__"` 块怎么办？

纯 kernel 文件（`{op}.py`）通常没有 `__main__` 块。如果有，只复制 kernel 部分
（imports + pass_configs + @tilelang.jit 函数），丢弃 `__main__` 块。

### 算子有多个 kernel 函数怎么办？

如果 `{op}.py` 包含多个 `@tilelang.jit` 函数，全部保留（它们可能互相调用或用于
不同配置）。测试代码中调用主 kernel。

### 生成的文件跑不过怎么办？

最常见原因：
1. **漏复制常量/辅助函数**：检查 kernel 是否引用了未复制的模块级符号
2. **golden 与 kernel dtype 路径不一致**：确保 golden 的 dtype 转换与 kernel 对齐
3. **shape 不匹配**：确保测试 shape 与 kernel 的 jit 参数一致
4. **缺少 `torch.npu.synchronize()`**：如果原测试有同步调用，保留它

## 参考示例

仓库中现有的单文件示例（合并后的目标格式参考）：
- `examples/normalization/layer_norm.py` — kernel + test_configs 循环 + assert_close
- `examples/developer_mode/gelu_mul_developer.py` — kernel + test_configs 循环 + assert_close
- `examples/normalization/rms_norm.py` — 同上模式

这些文件的共同特征：单文件、模块级测试、`tilelang.cache.clear_cache()`、末尾
`print("Kernel Output Match!")`。

**与本 skill 的差异**：仓库现有示例用 `torch.testing.assert_close(rtol=1e-2, atol=1e-2)`
做精度检查，但本 skill 按用户要求采用 `test_{op}.py` 的混合容差标准（内联，按 dtype
写死阈值），比 `assert_close` 的单一 rtol/atol 更贴合算子精度分级要求。其余格式
（单文件、clear_cache、末尾打印）保持与仓库惯例一致。
