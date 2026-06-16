# {算子名称} 算子设计文档

## 1. 概述

### 1.1 算子名称

{算子名称}

### 1.2 功能描述

{一句话描述算子功能}

### 1.3 数学公式

$$
{数学公式}
$$

### 1.4 算法描述

{对于多步算子，描述计算步骤的分解逻辑。单步算子可省略。}

### 1.5 数据流图

```
输入张量 → [计算步骤1] → [计算步骤2] → ... → 输出张量
```

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: {Developer / Expert / 混合}

### 2.2 选型理由

{基于算子特征（计算类型、是否含 matmul、是否含归约、是否需要流水线）的分析}

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | {如: T.alloc_ub 显式指定 UB} |
| 计算方式 | {如: T.Parallel + 运算符} |
| 作用域 | {如: 编译器自动分离 / 显式 T.Scope} |
| 同步方式 | {如: 自动同步 / 手动 T.barrier_all} |

---

## 3. API 映射设计

### 3.1 公式拆解

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | {子表达式} | {说明} |
| 2 | {子表达式} | {说明} |
| ... | ... | ... |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 1 | {子表达式} | {如: T.tile.exp(dst, src)} | {参数说明} | {Developer/Expert} |
| 2 | {子表达式} | {如: T.reduce_sum(buf, out, dim=-1)} | {参数说明} | {Developer/Expert} |
| ... | ... | ... | ... | ... |

### 3.3 计算伪代码

```python
# 基于 TileLang API 的计算流程伪代码
with T.Kernel(block_num, is_npu=True) as (cid, vid):
    # 1. 分配 buffer
    {buffer 分配代码}

    # 2. 数据搬入
    {T.copy 搬入代码}

    # 3. 计算
    {核心计算代码}

    # 4. 数据搬出
    {T.copy 搬出代码}
```

### 3.4 API 可行性确认

{列出使用的 API 及其来源确认（速查表 / 示例 / 源码），标注是否经过验证}

---

## 3.5 技术约束确认

### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | {Yes/No} | {block_metadata 方案 / 不涉及} |
| threads 参数限制（仅 1 或 2） | {Yes/No} | {threads=2 或移除 / 不涉及} |
| 动态循环边界不支持 | {Yes/No} | {静态边界 + if 条件判断 / 不涉及} |
| 流水线不支持动态边界 | {Yes/No} | {改用 T.serial / 不涉及} |

### 3.5.2 参考实现差异说明

**重要**：如果用户提供了外部参考实现（GPU 版），必须列出差异：

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|----------|
| Kernel 维度 | {三维 T.Kernel(m, n, batch)} | {一维 + block_metadata} | {参考 examples/grouped_gemm/} |
| 循环边界 | {动态 T.Pipelined(batch_sizes[bz])} | {静态 + if k < k_iters} | {预计算 max_iters} |
| GEMM API | {T.gemm} | {T.gemm_v0} | {查阅 api-compute.md} |
| 内存分配 | {T.alloc_shared 自动映射} | {T.alloc_L1 显式层级} | {Expert 模式} |
| threads | {threads=128} | {threads=2 或移除} | {NPU 限制} |

### 3.5.3 本项目同类实现参考

**必须列出**：本项目 examples/ 中最相似的实现

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| {examples/xxx/example_xxx.py} | {高度相似} | {Kernel 结构、API 用法、同步方式} |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| {A} | {(M, N)} | {float16} | {输入矩阵} |
| ... | ... | ... | ... |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| {C} | {(M, N)} | {float16} | {输出矩阵} |
| ... | ... | ... | ... |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| {a_ub} | {(block_M, block_N)} | {float16} | {UB} | {输入 tile 缓冲} |
| {tmp} | {(1, block_N)} | {float32} | {UB} | {归约临时缓冲} |
| ... | ... | ... | ... | ... |

### 4.4 内存搬运路径

```
{完整的数据搬运路径图}

示例（纯 Vector）:
GM[A] --T.copy--> UB[a_ub] --计算--> UB[c_ub] --T.copy--> GM[C]

示例（Cube + Vector）:
GM[A] --T.copy--> L1[a_l1] --T.copy--> L0A[a_l0a]
GM[B] --T.copy--> L1[b_l1] --T.copy--> L0B[b_l0b]
L0A + L0B --T.gemm--> L0C[c_l0c] --T.copy--> UB[c_ub]
UB[c_ub] --后处理--> UB[c_ub] --T.copy--> GM[C]
```

### 4.5 UB 内存预算

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| {a_ub} | {(128, 128)} | {float16} | {32768} |
| ... | ... | ... | ... |
| **总计** | | | {总字节数} / {目标平台 UB 容量，例如 196608 (192KB, A2/A3设备)} |

### 4.6 动态轴定义

{如无动态轴，写"无"。如有：}

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| {K} | {T.dyn['K']} | {1 ~ 64K} |

### 4.7 JIT 配置

```python
@tilelang.jit(
    out_idx=[{输出索引}],
    pass_configs={
        {pass 配置项}
    },
)
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: {纯 Vector / 纯 Cube / 混合}

**判定依据**: {如: 算子仅包含 element-wise 运算，无 matmul，判定为纯 Vector}

### 5.2 Block 划分

```python
block_M = {值}  # {选择理由}
block_N = {值}  # {选择理由}
block_num = (M // block_M) * (N // block_N)
```

### 5.3 约束分析

- **对齐约束**: {如: block_N=128, fp16 尾轴 128 > 16 ✓}
- **UB 容量**: {如: 总 buffer = 64KB < 当前目标平台 UB 容量（例如 A2/A3 为 192KB） ✓}
- **L0 容量**: {如: 无 Cube 计算，不适用}

### 5.4 注意事项

{非整除情况的处理策略、边界块的特殊逻辑等}

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| {M 方向} | {block 级并行} | {T.Kernel} | {每个 block 处理一个 M 分块} |
| {K 方向} | {迭代} | {T.serial(K // block_K)} | {K 维分块迭代累加} |
| {元素级} | {向量化} | {T.Parallel(block_M, block_N)} | {block 内逐元素并行} |

### 6.2 循环伪代码

```python
# Block 级并行（隐式，由 T.Kernel 管理）
with T.Kernel(block_num, is_npu=True) as (cid, vid):
    {block 内循环结构}
```

### 6.3 流水线优化

{是否使用 T.Pipelined？如使用，说明 num_stages 设计和 buffer 管理策略}

### 6.4 尾块处理

{当输入 shape 不能被 block size 整除时的处理策略}

---

## 7. 同步策略

### 7.1 同步模式

**模式**: {自动同步 / 手动同步 / 混合}

### 7.2 同步点说明

{手动同步时，列出每个同步点及理由：}

| 位置 | 同步 API | 理由 |
|------|----------|------|
| {搬入后} | {T.barrier_all()} | {等待 DMA 搬运完成} |
| ... | ... | ... |

### 7.3 pass_configs 配置

```python
pass_configs = {
    {与同步相关的 pass 配置}
}
```

---

## 8. 融合算子设计（如有）

{仅融合算子需要填写本章节。融合算子指包含 GEMM + element-wise 后处理的算子。}

### 8.1 融合算子判定

**判定结果**: {是 / 否}

**判定依据**: {如: 算子包含 GEMM 计算，输出需 element-wise 后处理，判定为融合算子}

### 8.2 CV 交互设计（按编程模式）

**Developer 模式（推荐，默认消除 workspace/vid）**：不产出 workspace 规格，记录以下即可——
- `T.Kernel(block_num, threads=2, is_npu=True) as (cid)`（单轴 + `threads=2`）
- 装饰器无 `workspace_idx`，签名无 `workspace_*` 参数
- Cube↔Vector 改片上 `alloc_shared/alloc_fragment` 直连，中转/同步交给四个 pass
- 模板见 [tilelang-expert-to-developer mode-examples.md §6](../../tilelang-custom-skill/tilelang-expert-to-developer/references/mode-examples.md#6-cv-融合--推荐写法消除-workspace--vidthreads2)

**Expert / 混合 / Developer 复杂场景回退**：填写 workspace 表——

| workspace | Shape | dtype | 用途 |
|-----------|-------|-------|------|
| workspace_1 | {[block_num, block_M, block_N]} | {accum_dtype} | {用途} |
| workspace_2 | {...} | {...} | {用途} |

**workspace_idx**: {[4, 5, 6]}  # 根据函数签名参数位置确定（仅回退写法）

### 8.3 Cube 核计算流程

```python
# Developer（推荐）：Cube 输出直连片上 buffer，无 workspace
T.copy({输入}, {buffer})
T.gemm_v0({a}, {b}, {c}, transpose_B={True/False})
T.copy({c}, {vector_side_buffer})       # L0C → alloc_shared 直连

# 回退（Expert/混合）：经 workspace 中转
# T.copy({c}, workspace_1[cid, :, :])
```

### 8.4 Vector 核计算流程

```python
# Developer（推荐）：从片上 buffer 直读，无 workspace、无 vid 偏移
{element-wise 计算}
T.copy({output}, Output[...])           # 输出

# 回退（Expert/混合）：从 workspace 读取
# T.copy(workspace_1[cid, ...], {buffer})
```

### 8.5 pass_configs 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: {True/False},  # 自动 CV 分离
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: {True/False},     # 自动核间同步
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: {True/False},        # 自动同步
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: {True/False},  # 内存规划
}
```

### 8.6 注意事项

- {如: 核间流水线与核内流水线不能同时开启}
- {Developer 模式默认消除 workspace/vid：`threads=2` 是消 vid 前提，消 vid 是消 workspace 前提}
- {如: workspace_idx 与函数签名参数位置必须一致（仅 Expert/混合或回退写法）}

---

## 9. 验证方案

### 9.1 Golden 函数

```python
def golden_{算子名}({参数}):
    """基于 PyTorch/NumPy 的参考实现"""
    {参考实现代码}
```

### 9.2 L0 门槛测试计划

> 设计阶段**只给出 L0 门槛用例**（规则 shape，block 整除），供 Stage 2 快速精度收敛。
> L1（功能，含不规则/尾块 shape）/ L2（异常输入）/ Boundary（特殊值）的**完整分层套件由 `tilelang-op-test-design` 生成**——Stage 1（场景 A）产出本节 L0 计划，Stage 2（场景 B）在 L0 通过后读真实实现扩展 L1/L2/Boundary。**不在此手工枚举 L1/L2/Boundary。**

| 用例名 | 级别 | Shape | dtype | 说明 |
|--------|------|-------|-------|------|
| {l0_basic} | L0 | {(32, 32)} | {float16} | {最小功能验证，规则 shape（block 整除）} |
| {l0_typical} | L0 | {(128, 128)} | {float16} | {典型规则配置} |

### 9.3 精度标准

> L0 用 baseline 标准即可；L1/L2/Boundary 扩展时由 `tilelang-op-test-design` 按算子类别（GEMM / Softmax / Normalization / Activation / Reduction / Fusion）套用更细的精度标准。

| dtype | atol | rtol |
|-------|------|------|
| float16 | 1e-2 | 1e-2 |
| float32 | 1e-4 | 1e-4 |

---

## 10. 风险点与注意事项

### 10.1 已知约束

{列出本算子在 TileLang-Ascend 上的已知限制}

### 10.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| {UB 溢出} | {block 过大} | {编译失败} | {减小 block size} |
| ... | ... | ... | ... |

### 10.3 特殊场景处理

{如: 非整除分块、极小 shape、混合精度等}

---

## 11. 交付清单

### 11.1 目录结构

```
examples/{算子名}/
├── example_{算子名}.py     # 算子实现 + 简单测试
├── design.md               # 本设计文档
└── README.md               # 使用说明（可选）
```

### 11.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `design.md` | {已完成} | 设计文档 |
| `example_{算子名}.py` | {待实现} | 算子实现 |
| `test_{算子名}.py` | {待实现} | 测试文件（可选，放入 testing/） |

### 11.3 命名规范

- 目录名: `{算子名}`（snake_case）
- 实现文件: `example_{算子名}.py`
- 测试文件: `test_{算子名}.py`

### 11.4 实现顺序

1. ✅ 设计文档（design.md）+ L0 门槛测试计划（本文件 §9.2）
2. ⬜ Golden 函数（验证基准）
3. ⬜ 算子实现（example_{算子名}.py）+ 内嵌 L0 用例
4. ⬜ L0 门槛测试通过（精度收敛）
5. ⬜ 扩展分层套件（L1 功能 / L2 异常 / Boundary 特殊值，由 `tilelang-op-test-design` 场景 B 生成）
6. ⬜ 全量套件运行（L0/L1 须通过；L2/Boundary 失败仅记录不阻塞）
