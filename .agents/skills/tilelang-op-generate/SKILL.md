---
name: tilelang-op-generate
description: "基于设计文档生成 TileLang-Ascend 算子实现代码与测试。从 design.md 中提取关键信息，结合 examples/ 中的参考实现生成可运行代码。触发：实现算子、写 kernel、生成代码、算子编码、根据设计文档实现。"
---

# TileLang-Ascend 算子代码生成

基于设计文档（`design.md`）和已有示例，生成可运行的算子实现与测试。

---

## 1. 从 design.md 中提取的信息（只取这些）

design.md 可能很长，**只提取以下字段，忽略其余内容**：

| 提取字段 | 所在章节 | 用途 |
|---------|---------|------|
| 数学公式 | §1 概述 | 理解计算逻辑 |
| 算法步骤分解 | §1 算法描述 | 确定计算顺序 |
| API 映射表 | §3 API 映射设计 | **核心**：每步用哪个 TileLang API |
| 伪代码 | §3 计算伪代码 | **核心**：代码骨架 |
| 输入输出 shape 和 dtype | §4 数据规格 | 函数签名和测试数据 |
| block 大小 | §5 Tiling 策略 | 分块参数 |
| pass_configs | §7 同步策略 | JIT 配置 |
| Golden 函数 | §8 验证方案 | 测试对比基准 |
| 测试用例表 | §8 验证方案 | 测试配置 |
| 精度标准 | §8 验证方案 | atol / rtol |

**明确忽略的内容**（这些容易误导）：
- 模式选型的分析推理过程
- 内存预算的计算过程和多轮优化迭代
- 风险点与注意事项（过于笼统）
- 交付清单（仅是文件列表）
- 任何标注为"待确认"的内容

---

## 2. 参考来源（优先级高于 design.md 伪代码）

**当 design.md 伪代码与 examples/ 中同类实现有冲突时，以 examples/ 为准。**

### 2.1 API 用法和模式选择

- **API 用法**：查阅 [tilelang-api-best-practices SKILL.md](../tilelang-custom-skill/tilelang-api-best-practices/SKILL.md) 及其 references 目录
- **编程模式和 pass_configs**：查阅 [tilelang-expert-to-developer SKILL.md](../tilelang-custom-skill/tilelang-expert-to-developer/SKILL.md) 及其 references 目录

### 2.2 同类算子示例

生成代码前，必须查阅 `examples/` 中的同类算子：

| 算子类型 | 参考示例 |
|---------|---------|
| 逐元素运算（add/mul/sigmoid/relu） | `examples/elementwise/`、`examples/activation/` |
| 归约运算（reduce_sum/max/min） | `examples/reduce/` |
| 归一化（softmax/layernorm/rmsnorm） | `examples/softmax/`、`examples/normalization/` |
| GEMM | `examples/gemm/`、`examples/developer_mode/gemm_developer.py` |
| 融合算子 | `examples/flash_attention/`、`examples/pipeline/`、`examples/developer_mode/matmul_add_developer.py` |
| Developer 模式 | `examples/developer_mode/` |

查阅示例时关注：
1. **Kernel 结构**：`T.Kernel` 参数、`cid`/`vid` 用法
2. **Buffer 分配方式**：shape 和 dtype
3. **pass_configs 配置**：该类算子实际使用哪些开关
4. **数据搬运**：`T.copy` 的索引写法
5. **workspace 配置**（融合算子）：workspace_idx、数量、shape

---

## 3. 代码生成流程

### 步骤 1：读取设计文档

读取 `design.md`，按 §1 的表格提取字段。

### 步骤 2：查找参考示例

在 `examples/` 中找到最相似的算子实现，**完整阅读其代码并记录技术决策**：

**必须记录的技术决策**（从参考实现中提取）：

| 决策项 | 示例值 | 说明 |
|--------|--------|------|
| 内存层级 API | `alloc_L1/L0C/ub`（显式）或 `alloc_shared/fragment`（自动） | 决定内存分配方式 |
| 同步策略 | 手动 `barrier_all/set_flag` 或自动同步 | 决定同步代码 |
| pass_configs | `AUTO_SYNC: True`，融合算子需 `AUTO_CV_COMBINE: True + AUTO_CV_SYNC: True` | 决定 JIT 配置 |
| 核分离方式 | `T.Scope("C"/"V")` 或无显式分离 | 决定核间协作方式 |
| workspace 配置（融合算子） | `{数量: 3, shape: [block_num, block_M, block_N], idx: [4,5,6]}` | 决定 workspace 参数 |

**对比差异分析**（如有 design.md）：

| 项目 | design.md 方案 | 参考实现方案 | 选择理由 |
|------|---------------|-------------|---------|
| 内存层级 API | | | |
| 同步策略 | | | |
| pass_configs | | | |
| workspace 配置 ⭐ | | | |

**冲突处理**：当 design.md 与参考实现冲突时：
- **优先参考实现**：参考实现已验证通过，可信度高
- **记录差异**：在代码注释中说明为何偏离 design.md
- **询问用户**：重大差异需确认

### 步骤 3：生成实现代码

基于 design.md 的 API 映射 + 参考示例的代码风格，生成 `example_{op}.py`。

文件结构：
```python
import tilelang
from tilelang import DataType, language as T
import torch

tilelang.cache.clear_cache()

# ========== 算子实现 ==========
@tilelang.jit(out_idx=[...], pass_configs={...})
def op_name(M, N, block_M, block_N, dtype="float"):
    # 分块计算
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2

    @T.prim_func
    def main(Input: T.Tensor((M, N), dtype), Output: T.Tensor((M, N), dtype)):
        with T.Kernel(..., is_npu=True) as (cid, vid):
            # buffer 分配
            # 数据搬入
            # 计算
            # 数据搬出
            pass

    return main

# ========== 测试 ==========
if __name__ == "__main__":
    torch.manual_seed(0)
    test_configs = [...]  # 来自 design.md §8

    for config in test_configs:
        # 1. 创建 kernel
        # 2. 生成输入数据
        # 3. 执行 kernel
        # 4. golden 对比
        # 5. 精度检查
        pass

    print("All tests passed!")
```

**融合算子注意事项**：
- 函数签名需包含 workspace 参数，`workspace_idx` 指定索引位置
- Cube 核输出通过 `T.copy` 写入 workspace，Vector 核从 workspace 读取

### 步骤 4：运行验证

```bash
python examples/{op}/example_{op}.py
```

如果报错，按以下顺序排查：
1. **编译错误** → 检查 buffer 大小、API 参数、对齐
2. **运行错误** → 检查索引越界、同步缺失
3. **精度错误** → 检查计算公式、数据类型、容差设置

### 步骤 5：校验原有实现正确性

**生成代码前必须先用默认参数跑通原有实现**，确认 baseline 正确后再扩展新功能/测试。

```bash
python examples/{op}/example_{op}.py  # 确认默认参数通过
```

### 步骤 6：设计测试用例的覆盖原则

测试用例必须覆盖以下 4 类场景：

| 类别 | 场景 | 说明 |
|------|------|------|
| 完美对齐 | M/N/K 均为 block 大小整数倍 | 验证零 padding 路径 |
| 单维 padding | 仅 M 或 N 或 K 不足 block 大小时 | 验证单边 padding+裁剪 |
| 全维 padding | M/N/K 同时需要 padding | 验证组合 padding |
| 多 block | 维度数倍于 block 大小 | 验证多 block 并行正确性 |

### 步骤 7：函数解耦全局变量

为实现多场景顺序测试，算子函数应**从 tensor shape 自推导所有维度参数**，而非依赖模块级全局变量：

```python
# ✅ 推荐：从 tensor 自推导
def conv_im2col_gemm(input_tensor, kernel, stride=1, padding=0):
    B, C, H, W = input_tensor.shape
    OC, C_k, KH, KW = kernel.shape

# ❌ 避免：依赖全局变量
def conv_im2col_gemm(...):
    C = globals()['C']  # 多测试场景会互相污染
```

---

## 4. 关键编码规范

### GEMM 算子：非整除维度处理

GEMM kernel 内部使用 `M // block_M` 和 `N // block_N`，要求 M、N 为 block 大小整数倍。非整除时需在调用的 Python 层 zero-padding 后裁剪：

```python
# padding
M_pad = ((M + block_M - 1) // block_M) * block_M
N_pad = ((N + block_N - 1) // block_N) * block_N
K_pad = ((K + block_K - 1) // block_K) * block_K

if M_pad > M or K_pad > K:
    kernel_padded = torch.zeros(M_pad, K_pad, ...)
    kernel_padded[:M, :K] = kernel_flat

# GEMM 后裁剪
output = output[:M, :N]
```

**关键约束**: 不 padding 时 `M // block_M = 0`（当 M < block_M）会导致零 block 启动（输出全零）或除零编译崩溃。

### Autotune 算子: supply_prog 与 get_configs 接口约定

- **`supply_prog(params)`**: `params` 仅含输入 tensor 描述符（不含输出 param）。从 `params[0].shape` / `params[1].shape` 提取维度，不可访问 `params[2]`。
- **`get_configs` 作为 callable**: autotuner 调用形式为 `get_configs(key_args_tuple, key_kwargs_tuple)`，须签名为 `get_configs(key_args, _key_kwargs=None)`，从 `key_args` 提取 M/N/K。
- **config 过滤**: 必须在 `get_configs` 中过滤 `block > dimension` 的无效组合（避免除零编译错误），及 `block_M * block_N * sizeof(accum) > L0C_capacity` 的组合（避免 L0C 溢出 segfault）。

### Buffer 分配

```python
# VEC_NUM = 2，每个 vector 核处理 block_M // VEC_NUM 行
a_ub = T.alloc_ub([block_M // VEC_NUM, block_N], dtype)
```

### 数据搬运索引

```python
# 标准索引模式
row_start = bx * block_M + vid * block_M // VEC_NUM
T.copy(A[row_start, by * block_N], a_ub)
T.copy(a_ub, B[row_start, by * block_N])
```

### 同步

```python
# Expert 模式：手��同步
with T.Scope("V"):
    T.copy(A[...], a_ub)
    T.barrier_all()
    T.tile.exp(a_ub, a_ub)
    T.barrier_all()
    T.copy(a_ub, B[...])

# Developer 模式 + 自动同步：无需手动 barrier
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
```

### 广播

```python
# 归约结果 [M, 1] 广播到 [M, N]
max_ub = T.alloc_ub([block_M // VEC_NUM, 1], dtype)
max_2d_ub = T.alloc_ub([block_M // VEC_NUM, block_N], dtype)
T.tile.broadcast(max_2d_ub, max_ub)
```

### 测试模板

```python
# golden 对比
ref_output = torch.nn.functional.softmax(input_data, dim=-1)  # 或手写 golden
torch.testing.assert_close(output.cpu(), ref_output.cpu(), rtol=rtol, atol=atol)
```

---

## 5. Checklist

生成代码后逐项检查：

### 基础检查

| # | 检查项 |
|---|--------|
| 1 | `out_idx` 与函数签名中的输出参数位置一致 |
| 2 | `block_M // VEC_NUM` 在 buffer 分配和索引中一致使用 |
| 3 | 所有 `T.alloc_ub` 的 shape 乘积不超 UB 容量 |
| 4 | Expert 模式有 `T.Scope("V")` 和 `T.barrier_all()` |
| 5 | Developer 模式有对应的 `pass_configs` |
| 6 | 测试包含至少 2 个配置（小规模 + 典型规模） |
| 7 | golden 函数使用 PyTorch 标准实现 |

### 融合算子检查

| # | 检查项 | 说明 |
|---|--------|------|
| 8 | **workspace_idx 与函数签名一致** | workspace 参数位置正确 |
| 9 | **AUTO_CV_COMBINE / AUTO_CV_SYNC 配置** | Developer 模式需开启 |
| 10 | **Cube → workspace → Vector 数据流正确** | T.copy 搬运路径完整 |
| 11 | **核分离方式与 pass_configs 匹配** | Developer 模式无需显式 T.Scope |

### 融合算子常见错误排查

| 错误类型 | 排查方向 |
|---------|---------|
| workspace 未正确搬运 | 检查 Cube 输出 T.copy 和 Vector 输入 T.copy 的索引 |
| 核间同步缺失 | 检查 AUTO_CV_SYNC 是否开启，或手动同步是否正确 |
| workspace shape 不匹配 | 检查 block_num 计算是否正确 |
| 核分离方式错误 | Developer + 自动同步模式应无显式 T.Scope("C"/"V") |
| 精度误差超过 1% | 优先检查内存层级 API 选择和 pass_configs 配置 |

---

## 6. Skill 反馈采集（强制，算子调试通过后执行）

本节是 **skill 自适应更新机制**的采集端。每次算子开发流程跑完后，必须把"哪些 skill 没讲清楚 / 被现实打脸 / 凭经验补的内容"写到 `.agents/skill-journal/`，由 `/tilelang-skill-review` 后续聚合评审。

**注意**：本节覆盖**整个开发链路**用到的所有 skill，不只是 op-design / op-generate。

### 6.1 触发时机

满足以下任一条件后立即执行：
- 算子代码已生成且至少跑通过一次（即使精度不达标但能编译）
- 用户明确表示"本次开发结束"或"暂时到这"
- 调试中卡了很久（即使没跑通也要把过程中的发现写下来，type 标 `unclear_workflow`）

### 6.2 步骤 1：枚举本次查阅过的所有 skill

回顾整个开发会话，列出**实际打开 / 引用 / 跳转过**的所有 skill 路径（相对 `.agents/skills/`），不只是 op-design 和 op-generate。常见包含：

| skill | 何时会被查阅 |
|-------|-------------|
| `tilelang-op-design` | 设计阶段全程 |
| `tilelang-op-generate` | 生成阶段全程（即本 skill 自身）|
| `tilelang-custom-skill/tilelang-api-best-practices` | 查 API 用法 / 参数 |
| `tilelang-custom-skill/tilelang-expert-to-developer` | 决定模式 / pass_configs |
| `tilelang-custom-skill/tilelang-debug-helper` | 调试报错 |
| `tilelang-custom-skill/tilelang-error-fixer` | 修编译/运行错误 |
| `tilelang-ascend-tile-api` | 查 T.tile.* 系列 |
| 其它 | 任何被 grep / read 过的 SKILL.md |

**规则**：宁可多列，不可漏列。漏列会导致那个 skill 的反馈永远收不上来。

### 6.3 步骤 2：针对每个 skill 反思（逐个过）

对**每一个**在步骤 1 列出的 skill，按以下四问逐项检查：

1. 该 skill 讲清楚的事项里，**有哪些被现实打脸**？（如说"支持 X"实际不支持）
2. 我**凭经验补了**它没讲的什么内容？（如自己加了个对齐处理）
3. 它的**示例 / API 描述是否过时**？（如示例 shape 写错、API 签名变了）
4. 它的**工作流步骤是否漏了关键检查**？（如没说"先 grep examples/"）

每个 yes 的发现 = 一条 entry。**没有发现也要记录**（写空 entries），便于统计 skill 的"完美命中率"。

### 6.4 步骤 3：写 journal 文件

按 `.agents/skill-journal/README.md` 的 schema，写到：

```
.agents/skill-journal/{op}-{YYYYMMDD-HHMMSS}.md
```

frontmatter 的 `skills_consulted` 字段必须包含步骤 1 的完整列表。

每条 entry 包含 `target_skill / target_section / type / severity / status:pending / observation / evidence / proposed_change`，字段含义见 README。

**禁止**：
- ❌ 把 `target_skill` 全部填成 op-generate（懒得分类的常见错误）
- ❌ 在 journal 里直接写完整修订后的 SKILL.md 段落（review skill 在 apply 阶段才生成具体修改文本）
- ❌ 漏写 evidence（无证据的提案会被 review 阶段直接拒）

### 6.5 自检

写完 journal 后逐项检查：

| # | 检查项 | 必须通过 |
|---|--------|---------|
| 1 | `skills_consulted` 包含本次查阅的所有 skill | ✅ |
| 2 | 至少 50% 的 `skills_consulted` 在 entries 中至少出现一次（避免只反思 op-generate 自己）| ✅ |
| 3 | 每条 entry 的 `evidence` 都有具体报错/代码/文件引用 | ✅ |
| 4 | 没有重复 entry（同 `target_skill + target_section + type` 只出现一次） | ✅ |
| 5 | `severity=high` 的 entry 都附带了具体踩坑过程 | ⭕ |

### 6.6 完成报告

写完 journal 后输出：

```
## Skill 反馈采集报告

- Journal 文件: .agents/skill-journal/{op}-{timestamp}.md
- 查阅的 skill 数量: N
- 写入 entries 数量: M
- 按 skill 分布:
  - tilelang-op-design: 3
  - tilelang-custom-skill/tilelang-api-best-practices: 2
  - ...
- 提示: 运行 /tilelang-skill-review 进入评审流程
```
