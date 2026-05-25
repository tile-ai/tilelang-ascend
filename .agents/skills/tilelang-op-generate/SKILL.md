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
    tilelang.disable_cache()  # 在 __main__ 中禁用编译缓存
    torch.manual_seed(...)
    test_configs = [...]  # 来自 design.md §8

    for config in test_configs:
        # 1. 创建 kernel
        # 2. 生成输入数据
        # 3. 执行 kernel
        # 4. golden 对比
        # 5. 精度检查
        pass

    print("Test Passed!")
```

**融合算子注意事项**：
- 函数签名需包含 workspace 参数，`workspace_idx` 指定索引位置
- Cube 核输出通过 `T.copy` 写入 workspace，Vector 核从 workspace 读取

### 步骤 4：运行验证

```bash
python examples/{op}/example_{op}.py
```

如果报错，查阅 [troubleshooting.md](references/troubleshooting.md) 进行排查：

| 错误类型 | 排查方向 | 详细参考 |
|---------|---------|---------|
| 编译错误 | buffer 大小、API 参数、对齐 | troubleshooting.md §编译时错误 |
| 运行错误 | 索引越界、同步缺失 | troubleshooting.md §运行时错误 |
| 精度错误 | Golden 实现、输出形状 | troubleshooting.md §精度问题 |

> **遇到具体错误信息时**，先查 [references/troubleshooting.md](references/troubleshooting.md) ——本 skill 配套的疑难解答手册，覆盖编译错误（UB 内存不足 / threads / 动态循环边界）、运行错误（index OOB / valid_shape）、精度错误（dtype / atol 阈值）等常见场景的具体解决方案。

### 步骤 8：上库前检查清单

运行通过后，必须按 §8 Checklist 检查所有项目。**最容易踩坑的 4 项重点提醒**：

| 关键项 | 说明 | §8 编号 |
|--------|------|---------|
| **Golden 实现一致** | 迁移算子必须使用原算子的 golden 实现 | #9 |
| **tilelang.disable_cache()** | 放在 `__main__` 下方或 `main()` 内部 | #11 |
| **最后一行输出** | `"Test Passed!"` 或 `"Kernel Output Match!"` | #16 |
| **代码格式** | `ruff check` + `ruff format --check` 通过 | #18 |

完整 22 项检查清单见下文 §8。

## 4. 关键编码规范

### Buffer 分配

```python
# VEC_NUM = 2，每个 vector 核处理 block_M // VEC_NUM 行
a_ub = T.alloc_ub([block_M // VEC_NUM, block_N], dtype)
```

Developer 模式下：
```python
# Vector 核 buffer（编译器映射到 UB）
packed_ub = T.alloc_shared([block_M // VEC_NUM, block_N], dtype)

# Cube 核 buffer（编译器映射到 L1/L0）
A_L1 = T.alloc_shared([block_M, block_K], dtype)
B_L1 = T.alloc_shared([block_N, block_K], dtype)
C_L0 = T.alloc_fragment([block_M, block_N], accum_dtype)
```

### 数据搬运索引

```python
# 标准索引模式（纯 Vector 算子）
row_start = bx * block_M + vid * block_M // VEC_NUM
T.copy(A[row_start, by * block_N], a_ub)
T.copy(a_ub, B[row_start, by * block_N])
```

**⚠️ CV 融合场景（workspace 索引一致性）**：
```python
VEC_NUM = 2
block_N_2 = block_N // VEC_NUM

for row in T.serial(block_N_2):
    actual_row = bn * block_N + vid * block_N_2 + row  # 关键索引
    
    # 读数据和写 workspace 都必须用 actual_row
    T.copy(B_packed[actual_row, chunk_offset], packed_ub)  # ✓
    # ... 处理 ...
    T.copy(output_ub, workspace[actual_row, chunk_offset * 2])  # ✓（必须一致）

# Cube 核读取完整 block_N（不涉及 vid）
T.copy(workspace[bn * block_N, k_offset], B_L1)  # 完整 block_N
```

**易错点**：workspace 写入时忘记使用 `actual_row`，导致数据错乱。

### 同步

```python
# Expert 模式：手动同步
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

## 5. V 核并行化编码规范

Ascend NPU C:V = 1:2，默认两个 V 核执行相同工作。正确使用 `vid` 可让两个 V 核分担任务。

### 按行切分

```python
VEC_NUM = 2
block_M_2 = block_M // VEC_NUM

with T.Kernel(grid_size, is_npu=True) as (cid, vid):
    row_start = cid * block_M + vid * block_M_2
    
    # Buffer 分配：只需分配 V 核负责的行数
    data_ub = T.alloc_shared((block_M_2, block_N), dtype)
    
    # 读入数据
    T.copy(A[row_start, by * block_N], data_ub)
    
    # 计算
    ...
    
    # 写出数据（索引必须与读一致）
    T.copy(data_ub, B[row_start, by * block_N])
```

### 中间 buffer 索引一致性

当 V 核读写中间 buffer（workspace、临时 buffer）时，必须保持索引一致：

```python
# 错误：读写索引不一致
for row in T.serial(block_N_2):
    actual_row = bn * block_N + vid * block_N_2 + row
    T.copy(src[actual_row, ...], temp_ub)
    T.copy(temp_ub, dst[bn * block_N + row, ...])  # ❌ 索引不一致

# 正确：读写索引一致
for row in T.serial(block_N_2):
    actual_row = bn * block_N + vid * block_N_2 + row
    T.copy(src[actual_row, ...], temp_ub)
    T.copy(temp_ub, dst[actual_row, ...])  # ✓ 索引一致
```

### 模式三：CV 融合中的 V 核并行化

CV 融合算子中，V 核负责预处理，Cube 核负责 GEMM：

```python
VEC_NUM = 2
block_N_2 = block_N // VEC_NUM

# Vector 核部分：使用 vid 分配任务
for row in T.serial(block_N_2):
    actual_row = bn * block_N + vid * block_N_2 + row
    T.copy(B_packed[actual_row, ...], ...)
    T.copy(..., workspace[actual_row, ...])

# Cube 核部分：读取完整 block_N（不涉及 vid）
T.copy(workspace[bn * block_N, ...], B_L1)
T.gemm_v0(A_L1, B_L1, C_L0, ...)
```

---

## 6. GEMM 编码规范

### gemm_v0 初始化

第一次调用必须清零 C_L0：

```python
for k_chunk in T.serial(k_num):
    T.gemm_v0(A_L1, B_L1, C_L0, transpose_B=True, init=(k_chunk == 0))
```

### NPU 分形限制

GEMM 的 block size 必须满足 L0A/L0B/L0C 分形限制（详见 [api-compute.md](../tilelang-custom-skill/tilelang-api-best-practices/references/api-compute.md)）：

- int8 GEMM：`block_M ≥ 16`, `block_N ≥ 16`, `block_K ≥ 32`
- float16 GEMM：`block_M ≥ 16`, `block_N ≥ 16`, `block_K ≥ 16`

---

## 7. CV 融合 pass_configs

CV 融合算子必须开启全部 4 个 pass_configs：

```python
PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,  # 自动分离 Cube/Vector
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
```

---

## 8. Checklist

生成代码后逐项检查：

### 功能验证

| # | 检查项 |
|---|--------|
| 1 | `out_idx` 与函数签名中的输出参数位置一致 |
| 2 | V 核并行化：`block_M // VEC_NUM` 在 buffer 分配和索引中一致使用（详见 §5） |
| 3 | 所有 `T.alloc_ub` 的 shape 乘积不超 UB 容量 |
| 4 | Expert 模式有 `T.Scope("V")` 和 `T.barrier_all()` |
| 5 | Developer 模式有对应的 `pass_configs` |
| 6 | 测试包含至少 2 个配置（小规模 + 典型规模） |
| 7 | 含 GEMM：`gemm_v0` 第一次调用有 `init=True`（详见 §6） |
| 8 | 含 GEMM：block size 满足分形限制（详见 §6） |

### Golden 与精度验证

| # | 检查项 | 说明 |
|---|--------|------|
| 9 | **Golden 实现一致** | 迁移算子必须使用原算子的 golden 实现 |
| 10 | **输出形状匹配** | 检查是否需要 transpose 来匹配原算子输出 shape |

### 上库前收尾检查

| # | 检查项 | 方法 |
|---|--------|------|
| 11 | **tilelang.disable_cache()** | 放在 `__main__` 下方或 `main()` 内部，避免编译缓存影响测试。**禁止**放在文件顶部全局调用、或用 `cache.clear_cache()`（会影响其他算子） |
| 12 | **注释转英文** | 人工检查所有注释，移除调试期临时中文注释 |
| 13 | **`# type: ignore`** | `T.Tensor` 参数定义后追加，避免 Pylance 报错 |
| 14 | **移除 try-catch** | 测试代码中不应有异常捕获，fail fast 更利于暴露问题 |
| 15 | **每组测试提示** | 每个用例打印 `print(f"Test passed: M={M}, N={N}, K={K}")`，避免看似卡住 |
| 16 | **最终输出格式** | 最后一行 `print("Test Passed!")` 或 `print("Kernel Output Match!")`，bench_test.sh 据此判定 |
| 17 | **参数处理灵活** | `argparse` 接收自定义参数 + 不传参数时跑默认多组测试 |
| 18 | **代码格式检查** | `ruff check examples/{op}/example_{op}.py` + `ruff format --check examples/{op}/example_{op}.py` 通过 |

### 融合算子专项检查

| # | 检查项 | 说明 |
|---|--------|------|
| 19 | **workspace_idx 与函数签名一致** | workspace 参数位置正确 |
| 20 | **AUTO_CV_COMBINE / AUTO_CV_SYNC 配置** | Developer 模式需开启 |
| 21 | **Cube → workspace → Vector 数据流正确** | T.copy 搬运路径完整 |
| 22 | **核分离方式与 pass_configs 匹配** | Developer 模式无需显式 T.Scope |

### 融合算子常见错误排查

| 错误类型 | 排查方向 |
|---------|---------|
| workspace 未正确搬运 | 检查 Cube 输出 T.copy 和 Vector 输入 T.copy 的索引 |
| 核间同步缺失 | 检查 AUTO_CV_SYNC 是否开启，或手动同步是否正确 |
| workspace shape 不匹配 | 检查 block_num 计算是否正确 |
| 核分离方式错误 | Developer + 自动同步模式应无显式 T.Scope("C"/"V") |
| 精度误差超过 1% | 优先检查内存层级 API 选择和 pass_configs 配置 |

---

## 6. Skill 反馈采集

> **本节的触发权归属取决于调用模式：**
>
> | 调用模式 | 由谁负责 skill 反馈采集 |
> |---------|----------------------|
> | 通过 `tilelang-op-orchestrator` 编排（推荐） | **由 orchestrator 在流程结束（SUCCESS / BLOCKED_*）时统一执行**，本 skill 不主动触发。详见 [.opencode/agents/tilelang-op-orchestrator.md §流程结束反思采集](../../../.opencode/agents/tilelang-op-orchestrator.md) |
> | 单独调用本 skill（`/tilelang-op-generate`，跳过编排） | **由调用者在算子调试通过后手动触发**，按下文 §6.1-6.6 流程执行 |
>
> 为什么分开：orchestrator 模式下本 skill 在 Subagent 隔离上下文中被多次调度，单次调度结束 ≠ "全流程结束"。让本 skill 自己触发反思会导致 ① 每次调用都做一次 → 浪费；② 看不到其他 Subagent 用过什么 skill → 反思不全。因此 orchestrator 模式下交给 orchestrator 在全流程视野下统一采集。
>
> **若你（developer subagent）在 orchestrator 模式下被调度本 skill，直接跳过本节即可**——orchestrator 会在最终阶段做反思。但你**仍然应该在 `debug_log.md` 里如实记录本次调度的 changes / error_summary / next_hint**，这是 orchestrator 反思的核心数据源。

本节（以下 §6.1-6.6）是 **skill 自适应更新机制**的采集端，**仅在单独调用模式下适用**。每次算子开发流程跑完后，必须把"哪些 skill 没讲清楚 / 被现实打脸 / 凭经验补的内容"写到 `.agents/skill-journal/`，由 `/tilelang-skill-review` 后续聚合评审。

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

每条 entry 包含 `target_skill / target_artifact / target_section / type / severity / status:pending / observation / evidence / proposed_change`，字段含义见 [skill-journal README](../skill-journal/README.md)。

**关于 `target_artifact`**（决定 apply 时改哪个文件）：
- 默认值 `skill`：反馈针对规则/流程/决策树/代码示例，改 SKILL.md
- 取值 `troubleshooting`：反馈是"具体错误 → 具体解决方案"，改 `references/troubleshooting.md`
- 同时满足三条才选 `troubleshooting`：① observation 含具体错误信息/错误码；② proposed_change 形如"症状 X → 改动 Y"；③ 改动可独立成"症状-原因-解决"条目，不依赖 SKILL.md 上下文

**禁止**：
- ❌ 把 `target_skill` 全部填成 op-generate（懒得分类的常见错误）
- ❌ 把所有 entry 全填 `target_artifact: skill`（懒得分流的常见错误）；带具体错误码的 entry 几乎都应该是 `troubleshooting`
- ❌ 在 journal 里直接写完整修订后的 SKILL.md 段落（review skill 在 apply 阶段才生成具体修改文本）
- ❌ 漏写 evidence（无证据的提案会被 review 阶段直接拒）

### 6.5 自检

写完 journal 后逐项检查：

| # | 检查项 | 必须通过 |
|---|--------|---------|
| 1 | `skills_consulted` 包含本次查阅的所有 skill | ✅ |
| 2 | 至少 50% 的 `skills_consulted` 在 entries 中至少出现一次（避免只反思 op-generate 自己）| ✅ |
| 3 | 每条 entry 的 `evidence` 都有具体报错/代码/文件引用 | ✅ |
| 4 | 没有重复 entry（同 `target_skill + target_artifact + target_section + type` 只出现一次） | ✅ |
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
